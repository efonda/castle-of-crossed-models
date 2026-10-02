"""Judge orchestrator.

Sends anonymised, order-randomised outputs to blind LLM judges under one fixed
scoring prompt, parses the structured JSON, and verifies every evidence span
appears verbatim in the prose. Reuses the `Provider` protocol — a judge is just
a model call that returns JSON instead of a paragraph.

Load-bearing invariants (brief §10):
- Blinding: judges never see model identity. The blind set strips everything but
  a `blind_id` and the prose; the blind_id -> generation_id map stays here.
- One fixed scoring prompt for all judges (varying it would confound the
  agreement analysis).
- Evidence spans are verified deterministically; an unverified span is flagged,
  never trusted (judges fabricate quotes).
- The raw judge response is written to disk verbatim BEFORE parsing, so malformed
  replies survive for the failure-mode catalogue.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .models import (
    Condition,
    DimensionEvidence,
    DimensionScores,
    Generation,
    JudgeScore,
)
from .providers import GenerationRequest, Provider, QuotaError
from .storage import JUDGE_SCORES, RAW_JUDGE_RESPONSES, Store
from .validator import evidence_span_verified

DIMENSIONS = ("distinctness", "bridge", "tonal", "originality", "combinatorial")

# JSON schema for forced structured output (judge calls). FLAT, not nested: a
# `<dim>_score` integer and a `<dim>_evidence` string per dimension. A nested
# dict-of-dicts schema ({dim: {score, evidence}}) caused claude-opus-4-8 to mis-fill the
# tool arguments ~71% of the time (parameter-XML leaking into the values); flat scalar
# fields are filled far more reliably across models. `_parse_scores` still accepts the old
# nested shape for back-compat with already-stored data.
#
# `scale` is the max integer score (5 = canonical 0-5 rubric). A scale-10 pass is a
# ceiling-artifact / resolution check: scores are NORMALIZED back to the 0-5 axis at parse
# time (see `_parse_scores`), so they pool with 0-5 data; the verbatim 0-N reply is kept in
# raw_judge_responses.jsonl. Run scale-10 judges under a DISTINCT judge_model label.
# `execution` adds a SEPARATE holistic dimension at a fixed 0-EXECUTION_MAX scale (NOT the
# rubric `scale`): one overall execution-quality judgment with headroom the saturating five
# dims lack. It is parsed raw (never normalized to 0-5) and stored on its own JudgeScore field.
EXECUTION_MAX = 100


def build_judge_schema(scale: int = 5, execution: bool = False) -> dict:
    props = {
        **{f"{d}_score": {"type": "integer", "minimum": 0, "maximum": scale} for d in DIMENSIONS},
        **{f"{d}_evidence": {"type": "string"} for d in DIMENSIONS},
    }
    required = [f"{d}_score" for d in DIMENSIONS] + [f"{d}_evidence" for d in DIMENSIONS]
    if execution:
        props["execution_score"] = {"type": "integer", "minimum": 0, "maximum": EXECUTION_MAX}
        props["execution_evidence"] = {"type": "string"}
        required += ["execution_score", "execution_evidence"]
    return {"type": "object", "properties": props, "required": required}


JUDGE_JSON_SCHEMA = build_judge_schema(5)  # canonical 0-5 schema (back-compat alias)

# One fixed scoring prompt for all judges (within a scale). Anchors are abstract
# level-descriptions; the combinatorial block and the evidence-span instruction are
# reproduced from the brief. {BLIND_ID} and {OUTPUT} are filled per output. Do not vary
# per judge. The __MAX__/__TOP__/__MID__/__LOW__ tokens are the scale anchors, filled by
# `build_scoring_prompt` — scale=5 reproduces the canonical 0-5 prompt verbatim.
_SCORING_PROMPT_TEMPLATE = """\
You are scoring a short paragraph for a study of narrative craft. The paragraph is
meant to cross two separate stories through one shared object, where the object
means something different in each story, joined by a grounded physical, causal, or
temporal link.

Score each dimension from 0 to __MAX__ (integers only).

1. DISTINCTNESS — does the object mean clearly different things in the two stories,
   without changing what it physically is?
   __TOP__ = two sharply different meanings, same physical object.
   __MID__ = different but overlapping or related meanings.
   __LOW__ = essentially the same meaning, or the object physically changes.

2. BRIDGE COHERENCE — is the connection grounded (physical / causal / temporal),
   not coincidence, dream logic, telepathy, or symbolism alone?
   __TOP__ = a concrete grounded link the reader can trace.
   __MID__ = a link exists but leans partly on coincidence or symbolism.
   __LOW__ = the connection is only symbolic, coincidental, or dreamlike.

3. TONAL ECONOMY — is the prose concrete and controlled, free of melodrama,
   abstract moralising, and generic fantasy filler?
   __TOP__ = concrete and controlled, no filler.
   __MID__ = mostly controlled, with some abstraction or cliché.
   __LOW__ = melodramatic, abstract, or filler-laden.

4. INTERPRETIVE ORIGINALITY — does it avoid cliché and stock symbolic meanings;
   is the object freshly construed?
   __TOP__ = a fresh, non-obvious construal.
   __MID__ = competent but largely expected.
   __LOW__ = stock or cliché symbolism.

5. COMBINATORIAL FIDELITY (exploratory)
Does the output feel like a true crossing of two independent narrative paths
through one shared object, rather than a simple plot transition? High scores
show compression, structural elegance, and dual legibility. Do NOT score
imitation of Calvino's prose style.

For each score, include a short evidence span copied exactly from the output.
The evidence span must be present verbatim in the prose text.

Output ONLY a JSON object in exactly this shape (flat keys), with no other text:
{{
  "distinctness_score": <0-__MAX__>,  "distinctness_evidence": "<verbatim span>",
  "bridge_score": <0-__MAX__>,        "bridge_evidence": "<verbatim span>",
  "tonal_score": <0-__MAX__>,         "tonal_evidence": "<verbatim span>",
  "originality_score": <0-__MAX__>,   "originality_evidence": "<verbatim span>",
  "combinatorial_score": <0-__MAX__>, "combinatorial_evidence": "<verbatim span>"
}}

Output to score (id {BLIND_ID}):
\"\"\"
{OUTPUT}
\"\"\""""


# --- Italian arm (generalization check) -------------------------------------
# Faithful translation of the fixed scoring prompt. The rubric text is Italian; the
# JSON field names stay English so parsing/schema are unchanged (evidence spans will
# be Italian, copied from Italian prose — span verification is a substring match, so
# it is language-agnostic). NOTE: native-review before a live Italian run.
_SCORING_PROMPT_TEMPLATE_IT = """\
Stai valutando un breve paragrafo per uno studio sull'arte narrativa. Il paragrafo
deve incrociare due storie separate attraverso un unico oggetto condiviso, dove
l'oggetto significa qualcosa di diverso in ciascuna storia, unite da un legame
concreto fisico, causale o temporale.

Valuta ogni dimensione da 0 a __MAX__ (solo numeri interi).

1. DISTINZIONE — l'oggetto significa cose chiaramente diverse nelle due storie,
   senza cambiare ciò che fisicamente è?
   __TOP__ = due significati nettamente diversi, stesso oggetto fisico.
   __MID__ = significati diversi ma sovrapposti o collegati.
   __LOW__ = sostanzialmente lo stesso significato, o l'oggetto cambia fisicamente.

2. COERENZA DEL LEGAME — la connessione è concreta (fisica / causale / temporale),
   non coincidenza, logica onirica, telepatia o solo simbolismo?
   __TOP__ = un legame concreto che il lettore può tracciare.
   __MID__ = un legame esiste ma si appoggia in parte a coincidenza o simbolismo.
   __LOW__ = la connessione è solo simbolica, casuale o onirica.

3. ECONOMIA TONALE — la prosa è concreta e controllata, priva di melodramma,
   moralismo astratto e riempitivo fantasy generico?
   __TOP__ = concreta e controllata, senza riempitivo.
   __MID__ = perlopiù controllata, con qualche astrazione o cliché.
   __LOW__ = melodrammatica, astratta o piena di riempitivo.

4. ORIGINALITÀ INTERPRETATIVA — evita il cliché e i significati simbolici di maniera;
   l'oggetto è interpretato in modo fresco?
   __TOP__ = un'interpretazione fresca e non ovvia.
   __MID__ = competente ma in gran parte prevedibile.
   __LOW__ = simbolismo di maniera o cliché.

5. FEDELTÀ COMBINATORIA (esplorativa)
L'output sembra un vero incrocio di due percorsi narrativi indipendenti attraverso un
unico oggetto condiviso, piuttosto che una semplice transizione di trama? I punteggi
alti mostrano compressione, eleganza strutturale e doppia leggibilità. NON valutare
l'imitazione dello stile di Calvino.

Per ogni punteggio, includi un breve estratto-prova copiato esattamente dall'output.
L'estratto-prova deve comparire alla lettera nel testo della prosa.

Restituisci SOLO un oggetto JSON esattamente in questa forma (chiavi piatte, in inglese), senza altro testo:
{{
  "distinctness_score": <0-__MAX__>,  "distinctness_evidence": "<estratto alla lettera>",
  "bridge_score": <0-__MAX__>,        "bridge_evidence": "<estratto alla lettera>",
  "tonal_score": <0-__MAX__>,         "tonal_evidence": "<estratto alla lettera>",
  "originality_score": <0-__MAX__>,   "originality_evidence": "<estratto alla lettera>",
  "combinatorial_score": <0-__MAX__>, "combinatorial_evidence": "<estratto alla lettera>"
}}

Output da valutare (id {BLIND_ID}):
\"\"\"
{OUTPUT}
\"\"\""""

_SCORING_TEMPLATES = {"en": _SCORING_PROMPT_TEMPLATE, "it": _SCORING_PROMPT_TEMPLATE_IT}

# Holistic execution block, appended ONLY when execution=True (so execution=False reproduces the
# canonical prompt verbatim — see test). Inserted right before the "Output to score" anchor.
# Deliberately no "use the full range / resist clustering" nudge — that would manufacture the
# scale-induced spread the FC-decisiveness control exists to detect.
_EXECUTION_BLOCK = {
    "en": (
        "6. EXECUTION QUALITY (holistic, 0-100) — set the five dimensions above aside and judge "
        "the passage AS A WHOLE: how well-executed is it as a piece of writing, the overall craft "
        "a reader responds to, beyond whether each named criterion is met? Rate 0 to 100 (integer): "
        "0 = inept, 50 = competent, 100 = flawless. A single honest overall impression, NOT the "
        "average of the scores above. Also return \"execution_score\" (0-100) and "
        "\"execution_evidence\" (a short verbatim span) in the JSON object."
    ),
    "it": (
        "6. QUALITÀ DI ESECUZIONE (olistica, 0-100) — metti da parte le cinque dimensioni qui sopra "
        "e valuta il paragrafo NEL SUO COMPLESSO: quanto è ben eseguito come pezzo di scrittura, la "
        "maestria complessiva a cui un lettore risponde, al di là del fatto che ogni criterio sia "
        "soddisfatto? Valuta da 0 a 100 (numero intero): 0 = maldestro, 50 = competente, 100 = "
        "impeccabile. Un'unica impressione complessiva onesta, NON la media dei punteggi sopra. "
        "Restituisci anche \"execution_score\" (0-100) ed \"execution_evidence\" (un breve estratto "
        "alla lettera) nell'oggetto JSON."
    ),
}
_EXECUTION_ANCHOR = {"en": "Output to score (id", "it": "Output da valutare (id"}


def build_scoring_prompt(scale: int = 5, language: str = "en", execution: bool = False) -> str:
    """Render the fixed scoring prompt for a given integer scale (max score) and
    language. scale=5, language='en', execution=False reproduces the canonical prompt verbatim.
    Anchors map linearly onto the 0-5 anchors: top=scale, mid≈0.6·scale, low≈0.2·scale (so
    10↔5, 6↔3, 2↔1). The returned string still carries {BLIND_ID}/{OUTPUT} for
    per-output .format().

    `execution=True` appends a holistic 0-100 execution-quality dimension (the schema then
    requires `execution_score`/`execution_evidence`); the five named dims are untouched."""
    template = _SCORING_TEMPLATES.get(language)
    if template is None:
        raise ValueError(f"no scoring prompt for language {language!r}")
    top, mid, low = scale, round(scale * 3 / 5), round(scale * 1 / 5)
    prompt = (
        template
        .replace("__MAX__", str(scale))
        .replace("__TOP__", str(top))
        .replace("__MID__", str(mid))
        .replace("__LOW__", str(low))
    )
    if execution:
        anchor = _EXECUTION_ANCHOR[language]
        prompt = prompt.replace(anchor, _EXECUTION_BLOCK[language] + "\n\n" + anchor, 1)
    return prompt


SCORING_PROMPT = build_scoring_prompt(5)  # canonical 0-5 prompt (back-compat alias)


@dataclass(frozen=True)
class BlindOutput:
    """One anonymised output as a judge sees it: a blind handle plus prose. The
    real generation_id is kept here for joining results but is NEVER sent to a
    judge (only blind_id and output_text reach the prompt)."""

    blind_id: str
    generation_id: str
    output_text: str


@dataclass(frozen=True)
class Judge:
    """A blind judge. `judge_model` is the recorded identity (and cache key);
    `api_model` is the model id actually called, defaulting to `judge_model`. They
    differ when the same model runs under a distinct label (e.g. an automated
    `claude-opus-4-8-api` judge kept separate from manual `claude-opus-4-8`).
    Use at least one judge family different from the tested models (brief §10)."""

    judge_model: str
    provider: Provider
    api_model: Optional[str] = None
    thinking: bool = False          # score with extended thinking on (reasoning judge)
    effort: str = "high"            # adaptive-thinking effort (Opus 4.8+): high/medium/low
    thinking_budget: int = 8192     # legacy thinking budget (older models' fallback)
    scale: int = 5                  # max score; 10 = finer-grained pass (normalized to 0-5)
    language: str = "en"            # rubric language ("it" for the Italian generalization arm)
    execution: bool = False         # also score a holistic 0-100 execution dim (separate axis)

    @property
    def model_id(self) -> str:
        return self.api_model or self.judge_model


def build_blind_set(generations: list[Generation], *, seed: int = 0) -> list[BlindOutput]:
    """Strip identity and randomise order. Returns blind outputs in shuffled order;
    blind_ids are sequential in that shuffled order so they carry no model signal."""
    shuffled = list(generations)
    random.Random(seed).shuffle(shuffled)
    return [
        BlindOutput(blind_id=f"out-{i:04d}", generation_id=g.id, output_text=g.output_text)
        for i, g in enumerate(shuffled)
    ]


def _balance_braces(text: str) -> Optional[str]:
    """Deterministic structural repair for near-JSON judge replies. Scans from the first
    '{' tracking brace depth (string- and escape-aware): if the object closes before the
    text ends, return just that slice (drops trailing junk such as a doubled '}'); if the
    text ends with the object still open OUTSIDE a string, append the missing closers
    (a truncated-at-the-end reply). Purely structural — never invents content; a reply cut
    off mid-string is unrepairable and returns None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    if depth > 0 and not in_str:
        return text[start:] + "}" * depth
    return None


def _extract_json(text: str) -> Optional[dict]:
    """Parse a JSON object from judge text, tolerating code fences, surrounding prose,
    a doubled trailing '}', or a reply truncated after its last complete value (both
    observed from gemini on the hard bank, 2026-07-02). Returns None if no valid object
    can be recovered without inventing content."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    if not isinstance(text, str):
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    repaired = _balance_braces(text)
    if repaired is not None:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None
    return None


def _parse_scores(payload: dict, scale: int = 5) -> tuple[DimensionScores, DimensionEvidence]:
    """Pull per-dimension score + evidence into the two records. Accepts the FLAT shape
    (`<dim>_score` / `<dim>_evidence`, the current schema) and, for back-compat with
    already-stored data, the old NESTED shape (`<dim>: {score, evidence}`). Raises (caught
    by the caller → PARSE-FAIL) if the shape is wrong or a score is out of range.

    `scale` is the judge's max score. Scores are NORMALIZED to the canonical 0-5 axis
    (× 5/scale) so a 0-10 pass pools with 0-5 data; the verbatim 0-N reply is preserved
    upstream in raw_judge_responses.jsonl. An out-of-range raw score normalizes above 5
    and is rejected by DimensionScores (→ PARSE-FAIL), so the gate still holds."""
    norm = 5.0 / scale
    scores, evidence = {}, {}
    for dim in DIMENSIONS:
        if f"{dim}_score" in payload:                 # flat (current)
            scores[dim] = payload[f"{dim}_score"] * norm
            evidence[dim] = payload[f"{dim}_evidence"]
        else:                                         # nested (legacy)
            scores[dim] = payload[dim]["score"] * norm
            evidence[dim] = payload[dim]["evidence"]
    return DimensionScores(**scores), DimensionEvidence(**evidence)


def _parse_execution(payload: dict) -> tuple[Optional[float], Optional[str]]:
    """Pull the holistic execution score (raw 0-100, NOT normalized) and its evidence span.
    Returns (None, None) if absent. Raises ValueError on an out-of-range score (caught by the
    caller → PARSE-FAIL), same gate as the dimension scores. Kept separate from `_parse_scores`
    so that function's (DimensionScores, DimensionEvidence) shape is unchanged."""
    if "execution_score" not in payload:
        return None, None
    val = float(payload["execution_score"])
    if not (0 <= val <= EXECUTION_MAX):
        raise ValueError(f"execution_score {val} out of range 0-{EXECUTION_MAX}")
    return val, payload.get("execution_evidence", "")


class JudgeOrchestrator:
    def __init__(self, store: Store):
        self.store = store
        self._judged = store.judged_pairs()

    def score_one(self, blind: BlindOutput, judge: Judge) -> Optional[JudgeScore]:
        """Score one blind output with one judge. Returns None if already scored
        (cache hit) or if the response could not be parsed (raw is still saved)."""
        if (blind.generation_id, judge.judge_model) in self._judged:
            return None

        prompt = build_scoring_prompt(judge.scale, judge.language, judge.execution).format(
            BLIND_ID=blind.blind_id, OUTPUT=blind.output_text
        )
        result = judge.provider.generate(
            GenerationRequest(
                prompt=prompt,
                model=judge.model_id,
                condition=Condition.THINKING if judge.thinking else Condition.NORMAL,
                temperature=0.0,
                effort=judge.effort,
                thinking_budget=judge.thinking_budget,
                json_schema=build_judge_schema(judge.scale, judge.execution),
            )
        )

        # Persist the raw response verbatim BEFORE parsing — survives malformed JSON.
        self.store.append_raw(
            RAW_JUDGE_RESPONSES,
            {
                "generation_id": blind.generation_id,
                "blind_id": blind.blind_id,
                "judge_model": judge.judge_model,
                "raw_response": result.output_text,
            },
        )

        payload = _extract_json(result.output_text)
        if payload is None:
            return None
        try:
            scores, evidence = _parse_scores(payload, scale=judge.scale)
            execution, execution_evidence = (
                _parse_execution(payload) if judge.execution else (None, None)
            )
        except (KeyError, TypeError, ValueError):
            return None  # malformed shape / out-of-range score; raw is preserved

        # Verify every evidence span against the prose; flag fabricated ones.
        spans_verified = {
            dim: evidence_span_verified(blind.output_text, getattr(evidence, dim))
            for dim in DIMENSIONS
        }
        if execution_evidence is not None:
            spans_verified["execution"] = evidence_span_verified(
                blind.output_text, execution_evidence
            )

        score = JudgeScore(
            generation_id=blind.generation_id,
            judge_model=judge.judge_model,
            blind_id=blind.blind_id,
            scores=scores,
            evidence=evidence,
            spans_verified=spans_verified,
            raw_response=result.output_text,
            execution=execution,
            execution_evidence=execution_evidence,
        )
        self.store.append(JUDGE_SCORES, score)
        self._judged.add((blind.generation_id, judge.judge_model))
        return score

    def run(
        self,
        blind_set: list[BlindOutput],
        judges: list[Judge],
        progress: Optional[Callable[[Judge, BlindOutput, Optional[JudgeScore]], None]] = None,
        on_quota: Optional[Callable[[Judge, QuotaError], None]] = None,
    ) -> list[JudgeScore]:
        """Score every blind output with every judge, skipping cached pairs. Returns
        only newly produced scores. `progress(judge, blind, score_or_None)` is called
        after each attempt (None = cache hit or parse failure).

        If a judge's provider hits its quota/rate limit (`QuotaError`), skip the REST of
        that judge's outputs and move to the next judge — no point hammering a dead quota,
        and the cache lets a later run resume exactly where this one stopped. `on_quota`
        (if given) is notified once per skipped judge. Other judges still run; non-quota
        provider errors still propagate (a real failure should surface, not be swallowed)."""
        produced: list[JudgeScore] = []
        for judge in judges:
            for blind in blind_set:
                try:
                    score = self.score_one(blind, judge)
                except QuotaError as exc:
                    if on_quota is not None:
                        on_quota(judge, exc)
                    break  # skip remaining outputs for this judge; resume on a later run
                if score is not None:
                    produced.append(score)
                if progress is not None:
                    progress(judge, blind, score)
        return produced
