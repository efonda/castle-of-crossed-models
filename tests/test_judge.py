"""Tests for the judge orchestrator: blinding, JSON parsing (incl. fenced/dirty),
evidence-span verification (incl. fabricated spans), malformed-response handling,
and resumable cache-skip. Uses a JSON-returning mock provider — no API key."""

import json

import pytest

from calvino.judge import (
    Judge,
    JudgeOrchestrator,
    SCORING_PROMPT,
    build_blind_set,
    build_judge_schema,
    build_scoring_prompt,
    _extract_json,
    _parse_scores,
    _parse_execution,
)
from calvino.models import Generation, JudgeScore
from calvino.providers import GenerationRequest, GenerationResult, QuotaError
from calvino.storage import JUDGE_SCORES, RAW_JUDGE_RESPONSES, Store

PROSE = (
    "The smuggler's lantern burned steady on the headland as the patrol passed. "
    "He raised it twice, and far offshore the ship began to move. "
    "At dawn the widow carried the same lantern down into the flooded cellar. "
    "Its light slid over the waterline, naming each thing the tide had taken."
)


def _valid_payload(evidence_override=None):
    ev = {
        "distinctness": "meant something different",  # will be overridden below
        "bridge": "the same lantern",
        "tonal": "flooded cellar",
        "originality": "naming each thing the tide had taken",
        "combinatorial": "raised it twice",
    }
    # Use spans that are genuinely verbatim in PROSE by default.
    ev["distinctness"] = "burned steady on the headland"
    if evidence_override:
        ev.update(evidence_override)
    return {
        "distinctness": {"score": 5, "evidence": ev["distinctness"]},
        "bridge": {"score": 4, "evidence": ev["bridge"]},
        "tonal": {"score": 5, "evidence": ev["tonal"]},
        "originality": {"score": 3, "evidence": ev["originality"]},
        "combinatorial": {"score": 4, "evidence": ev["combinatorial"]},
    }


class JsonJudge:
    """Mock provider returning a canned JSON payload, optionally fence-wrapped.
    Records every prompt it received so tests can assert blinding."""

    def __init__(self, payload, *, fenced=False, raw_override=None):
        self.payload = payload
        self.fenced = fenced
        self.raw_override = raw_override
        self.prompts: list[str] = []
        self.requests: list[GenerationRequest] = []

    def last_request(self) -> GenerationRequest:
        return self.requests[-1]

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.prompts.append(request.prompt)
        self.requests.append(request)
        if self.raw_override is not None:
            body = self.raw_override
        else:
            body = json.dumps(self.payload)
            if self.fenced:
                body = f"Here is my assessment:\n```json\n{body}\n```\n"
        return GenerationResult(output_text=body, provider_meta={"provider": "mock"})


def _gen(gen_id, model="secret-model-7"):
    return Generation(
        id=gen_id,
        item_id="lantern-01",
        object="The Lantern",
        object_type="tarot_image_object",
        model=model,
        condition="normal",
        prompt_variant="base",
        sample_idx=0,
        temperature=0.7,
        rendered_prompt="...",
        output_text=PROSE,
    )


# --- blinding ----------------------------------------------------------------

def test_blind_set_strips_identity_and_is_deterministic():
    gens = [_gen(f"g{i}") for i in range(5)]
    a = build_blind_set(gens, seed=0)
    b = build_blind_set(gens, seed=0)
    assert [x.blind_id for x in a] == [x.blind_id for x in b]  # deterministic
    assert {x.generation_id for x in a} == {g.id for g in gens}  # all present
    assert [x.blind_id for x in a] == [f"out-{i:04d}" for i in range(5)]


def test_blind_set_randomises_order():
    gens = [_gen(f"g{i}") for i in range(10)]
    blind = build_blind_set(gens, seed=1)
    # The blind order should not equal the input order (vanishingly unlikely at n=10).
    assert [x.generation_id for x in blind] != [g.id for g in gens]


def test_judge_prompt_never_leaks_model_or_id(tmp_path):
    store = Store(tmp_path)
    judge_provider = JsonJudge(_valid_payload())
    orch = JudgeOrchestrator(store)
    blind = build_blind_set([_gen("g0", model="secret-model-7")], seed=0)[0]

    orch.score_one(blind, Judge("judge-A", judge_provider))

    prompt = judge_provider.prompts[0]
    assert "secret-model-7" not in prompt
    assert "g0" not in prompt
    assert blind.blind_id in prompt  # the blind handle is fine to show


# --- JSON extraction ---------------------------------------------------------

def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_and_surrounded():
    text = 'Sure!\n```json\n{"a": 1, "b": 2}\n```\nHope that helps.'
    assert _extract_json(text) == {"a": 1, "b": 2}


def test_extract_json_garbage_returns_none():
    assert _extract_json("no json here") is None


# The two real gemini failure shapes observed on the hard bank (2026-07-02, verbatim
# structure from data/runs/hard_fable_fc/raw_judge_responses.jsonl): a doubled closing
# brace, and a reply truncated right after the last complete value (no final brace).


def test_extract_json_repairs_doubled_closing_brace():
    text = '{\n  "distinctness_score": 5,\n  "combinatorial_evidence": "the same rungs"\n}\n}'
    assert _extract_json(text) == {
        "distinctness_score": 5,
        "combinatorial_evidence": "the same rungs",
    }


def test_extract_json_repairs_missing_closing_brace():
    text = '{\n  "distinctness_score": 5,\n  "combinatorial_evidence": "the river mud"'
    assert _extract_json(text) == {
        "distinctness_score": 5,
        "combinatorial_evidence": "the river mud",
    }


def test_extract_json_truncated_mid_string_is_unrepairable():
    # Cut off inside a string value: repairing would require inventing content -> None.
    assert _extract_json('{"distinctness_score": 5, "evidence": "the lantern sli') is None


def test_extract_json_brace_inside_string_not_miscounted():
    text = '{"evidence": "a brace } inside a string", "score": 4}\n}'
    assert _extract_json(text) == {"evidence": "a brace } inside a string", "score": 4}


# --- scoring + span verification ---------------------------------------------

def test_score_one_persists_and_verifies_spans(tmp_path):
    store = Store(tmp_path)
    orch = JudgeOrchestrator(store)
    blind = build_blind_set([_gen("g0")], seed=0)[0]

    score = orch.score_one(blind, Judge("judge-A", JsonJudge(_valid_payload())))

    assert score is not None
    assert score.generation_id == "g0"
    assert score.scores.distinctness == 5
    assert all(score.spans_verified.values())  # all spans are verbatim
    # Persisted and reloadable.
    assert len(store.load(JUDGE_SCORES, JudgeScore)) == 1
    # Raw response written too.
    assert store.path(RAW_JUDGE_RESPONSES).exists()


def test_fabricated_span_is_flagged_not_trusted(tmp_path):
    store = Store(tmp_path)
    orch = JudgeOrchestrator(store)
    blind = build_blind_set([_gen("g0")], seed=0)[0]
    payload = _valid_payload({"bridge": "a quote that is not in the prose at all"})

    score = orch.score_one(blind, Judge("judge-A", JsonJudge(payload)))

    assert score is not None
    assert score.spans_verified["bridge"] is False  # fabricated -> flagged
    assert score.spans_verified["distinctness"] is True  # the real ones still pass


def test_judge_api_model_differs_from_label(tmp_path):
    # Judge labelled distinctly from the model it actually calls: the API receives
    # api_model, but the score is stored (and cached) under judge_model.
    store = Store(tmp_path)
    provider = JsonJudge(_valid_payload())
    blind = build_blind_set([_gen("g0")], seed=0)[0]
    judge = Judge("claude-opus-4-8-api", provider, api_model="claude-opus-4-8")

    score = JudgeOrchestrator(store).score_one(blind, judge)

    assert score.judge_model == "claude-opus-4-8-api"          # label on disk
    assert provider.last_request().model == "claude-opus-4-8"  # what the API got


def test_fenced_response_parses(tmp_path):
    store = Store(tmp_path)
    orch = JudgeOrchestrator(store)
    blind = build_blind_set([_gen("g0")], seed=0)[0]

    score = orch.score_one(blind, Judge("judge-A", JsonJudge(_valid_payload(), fenced=True)))
    assert score is not None
    assert score.scores.bridge == 4


def test_malformed_json_returns_none_but_saves_raw(tmp_path):
    store = Store(tmp_path)
    orch = JudgeOrchestrator(store)
    blind = build_blind_set([_gen("g0")], seed=0)[0]

    judge = Judge("judge-A", JsonJudge(None, raw_override="totally not json"))
    score = orch.score_one(blind, judge)

    assert score is None
    assert len(store.load(JUDGE_SCORES, JudgeScore)) == 0
    # But the raw response is preserved for the failure-mode catalogue.
    raw_lines = store.path(RAW_JUDGE_RESPONSES).read_text().strip().splitlines()
    assert len(raw_lines) == 1


def test_out_of_range_score_rejected(tmp_path):
    store = Store(tmp_path)
    orch = JudgeOrchestrator(store)
    blind = build_blind_set([_gen("g0")], seed=0)[0]
    payload = _valid_payload()
    payload["distinctness"]["score"] = 7  # invalid

    score = orch.score_one(blind, Judge("judge-A", JsonJudge(payload)))
    assert score is None  # Pydantic rejects; raw still saved
    assert len(store.load(JUDGE_SCORES, JudgeScore)) == 0


# --- caching + full run ------------------------------------------------------

def test_cache_skip_within_and_across_runs(tmp_path):
    blind = build_blind_set([_gen("g0")], seed=0)[0]

    p1 = JsonJudge(_valid_payload())
    JudgeOrchestrator(Store(tmp_path)).score_one(blind, Judge("judge-A", p1))
    assert len(p1.prompts) == 1

    # New orchestrator over same dir rebuilds judged-pairs from disk and skips.
    p2 = JsonJudge(_valid_payload())
    result = JudgeOrchestrator(Store(tmp_path)).score_one(blind, Judge("judge-A", p2))
    assert result is None
    assert len(p2.prompts) == 0


# --- scale parameterization (0-10 pass) --------------------------------------

def test_scale5_prompt_is_canonical():
    # build_scoring_prompt(5) must reproduce the canonical prompt verbatim.
    assert build_scoring_prompt(5) == SCORING_PROMPT
    assert "Score each dimension from 0 to 5" in SCORING_PROMPT
    assert "   5 = two sharply different meanings" in SCORING_PROMPT
    assert '"distinctness_score": <0-5>' in SCORING_PROMPT


def test_scale10_prompt_and_schema_reanchor():
    p = build_scoring_prompt(10)
    assert "Score each dimension from 0 to 10" in p
    assert "   10 = two sharply different meanings" in p   # top  5 -> 10
    assert "   6 = different but overlapping" in p          # mid  3 -> 6
    assert "   2 = essentially the same meaning" in p       # low  1 -> 2
    assert '"distinctness_score": <0-10>' in p
    assert build_judge_schema(10)["properties"]["distinctness_score"]["maximum"] == 10


def test_italian_scoring_prompt_translates_rubric_keeps_english_json_keys():
    # The Italian arm uses an Italian rubric but English JSON field names (so parsing
    # and schema are unchanged); scale anchors still fill.
    p = build_scoring_prompt(5, "it")
    assert "Valuta ogni dimensione da 0 a 5" in p   # Italian rubric, scale anchor filled
    assert '"distinctness_score"' in p              # JSON keys stay English
    assert build_scoring_prompt(5, "en") == SCORING_PROMPT  # English path unchanged


def test_italian_judge_scores_via_flat_payload(tmp_path):
    # A language='it' judge still parses the standard flat (English-keyed) payload and
    # verifies an Italian evidence span against Italian prose.
    prose = ("La bilancia pendeva nel mercato. Il mercante chiuse la cassa. "
             "In montagna la vedova pesò la farina. Ne diede una parte ai figli.")
    gen = Generation(id="g0", item_id="bilancia-01", object="La Bilancia",
                     object_type="ordinary_physical_object", model="m", condition="normal",
                     prompt_variant="base", sample_idx=0, temperature=0.7,
                     rendered_prompt="...", output_text=prose)
    payload = {**{f"{d}_score": 4 for d in ("distinctness", "bridge", "tonal", "originality", "combinatorial")},
               **{f"{d}_evidence": "pesò la farina" for d in ("distinctness", "bridge", "tonal", "originality", "combinatorial")}}
    blind = build_blind_set([gen], seed=0)[0]
    provider = JsonJudge(payload)
    score = JudgeOrchestrator(Store(tmp_path)).score_one(blind, Judge("gpt-5.5", provider, language="it"))
    assert score.scores.distinctness == 4.0
    assert "Valuta ogni dimensione" in provider.last_request().prompt  # Italian rubric sent
    assert all(score.spans_verified.values())                          # Italian span verified


def test_parse_scores_normalizes_scale10_to_0_5():
    # A 0-10 payload is normalized to the 0-5 axis (x 0.5): 10->5.0, 7->3.5.
    payload = {**{f"{d}_score": 10 for d in ("distinctness", "bridge", "tonal", "originality", "combinatorial")},
               **{f"{d}_evidence": "x" for d in ("distinctness", "bridge", "tonal", "originality", "combinatorial")}}
    payload["originality_score"] = 7
    scores, _ = _parse_scores(payload, scale=10)
    assert scores.distinctness == 5.0
    assert scores.originality == 3.5


def test_scale10_judge_normalizes_and_uses_scale10_call(tmp_path):
    # End-to-end: a scale-10 judge sends the 0-10 prompt + schema and the stored scores
    # are normalized to the 0-5 axis, so they pool with the 0-5 panel.
    store = Store(tmp_path)
    orch = JudgeOrchestrator(store)
    blind = build_blind_set([_gen("g0")], seed=0)[0]
    payload = {
        "distinctness_score": 10, "distinctness_evidence": "burned steady on the headland",
        "bridge_score": 8, "bridge_evidence": "the same lantern",
        "tonal_score": 6, "tonal_evidence": "flooded cellar",
        "originality_score": 4, "originality_evidence": "naming each thing the tide had taken",
        "combinatorial_score": 10, "combinatorial_evidence": "raised it twice",
    }
    provider = JsonJudge(payload)
    score = orch.score_one(blind, Judge("gpt-5.5-s10", provider, scale=10))
    assert score.scores.distinctness == 5.0 and score.scores.bridge == 4.0
    assert score.scores.tonal == 3.0 and score.scores.originality == 2.0
    req = provider.last_request()
    assert req.json_schema["properties"]["distinctness_score"]["maximum"] == 10
    assert "from 0 to 10" in req.prompt
    # Raw 0-10 reply is preserved verbatim for provenance.
    assert score.raw_response == json.dumps(payload)


# --- holistic execution dimension (opt-in 0-100 axis) ------------------------

def _flat_payload_with_execution(exec_score=82):
    return {
        "distinctness_score": 5, "distinctness_evidence": "burned steady on the headland",
        "bridge_score": 4, "bridge_evidence": "the same lantern",
        "tonal_score": 5, "tonal_evidence": "flooded cellar",
        "originality_score": 3, "originality_evidence": "naming each thing the tide had taken",
        "combinatorial_score": 4, "combinatorial_evidence": "raised it twice",
        "execution_score": exec_score, "execution_evidence": "burned steady on the headland",
    }


def test_execution_off_keeps_prompt_and_schema_canonical():
    # execution=False must change nothing (the canonical-prompt test depends on this).
    assert build_scoring_prompt(5, execution=False) == SCORING_PROMPT
    assert "EXECUTION QUALITY" not in build_scoring_prompt(5)
    assert "execution_score" not in build_judge_schema()["properties"]


def test_execution_on_extends_prompt_and_schema():
    p = build_scoring_prompt(5, execution=True)
    assert "EXECUTION QUALITY (holistic, 0-100)" in p
    # block is inserted BEFORE the output anchor, not after it
    assert p.index("EXECUTION QUALITY") < p.index("Output to score")
    sch = build_judge_schema(5, execution=True)
    assert sch["properties"]["execution_score"]["maximum"] == 100
    assert "execution_score" in sch["required"] and "execution_evidence" in sch["required"]


def test_parse_execution_raw_and_range():
    assert _parse_execution({"execution_score": 82, "execution_evidence": "x"}) == (82.0, "x")
    assert _parse_execution({"distinctness_score": 5}) == (None, None)  # absent -> None
    with pytest.raises(ValueError):
        _parse_execution({"execution_score": 150})  # out of 0-100 -> PARSE-FAIL gate


def test_execution_judge_stores_raw_score_and_verifies_span(tmp_path):
    orch = JudgeOrchestrator(Store(tmp_path))
    blind = build_blind_set([_gen("g0")], seed=0)[0]
    provider = JsonJudge(_flat_payload_with_execution(82))
    score = orch.score_one(blind, Judge("exec-judge", provider, execution=True))
    assert score.execution == 82.0                 # raw 0-100, NOT normalized to 0-5
    assert score.scores.distinctness == 5.0        # five dims parsed normally alongside
    assert score.spans_verified["execution"] is True
    assert score.execution_evidence == "burned steady on the headland"
    assert provider.last_request().json_schema["properties"]["execution_score"]["maximum"] == 100


def test_execution_absent_when_not_requested(tmp_path):
    # A non-execution judge ignores any execution_score in the payload -> field stays None.
    orch = JudgeOrchestrator(Store(tmp_path))
    blind = build_blind_set([_gen("g0")], seed=0)[0]
    score = orch.score_one(blind, Judge("plain", JsonJudge(_flat_payload_with_execution())))
    assert score.execution is None
    assert "execution" not in score.spans_verified


def test_run_scores_every_output_by_every_judge(tmp_path):
    store = Store(tmp_path)
    orch = JudgeOrchestrator(store)
    blind_set = build_blind_set([_gen("g0"), _gen("g1"), _gen("g2")], seed=0)
    judges = [
        Judge("judge-A", JsonJudge(_valid_payload())),
        Judge("judge-B", JsonJudge(_valid_payload())),
        Judge("judge-C", JsonJudge(_valid_payload())),
    ]
    produced = orch.run(blind_set, judges)
    assert len(produced) == 9  # 3 outputs x 3 judges
    assert len(store.load(JUDGE_SCORES, JudgeScore)) == 9


class QuotaJudge:
    """Mock provider that scores `ok_first` outputs then raises QuotaError on the next
    call (simulates a quota cap mid-judge)."""

    def __init__(self, payload, *, ok_first=0):
        self.payload = payload
        self.ok_first = ok_first
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        if self.calls > self.ok_first:
            raise QuotaError("429 RESOURCE_EXHAUSTED: quota exceeded")
        return GenerationResult(output_text=json.dumps(self.payload), provider_meta={})


def test_quota_skips_rest_of_judge_and_continues_with_others(tmp_path):
    store = Store(tmp_path)
    orch = JudgeOrchestrator(store)
    blind_set = build_blind_set([_gen("g0"), _gen("g1"), _gen("g2")], seed=0)
    # judge-A: scores 1 then hits quota (should skip its other 2, not crash).
    quota_provider = QuotaJudge(_valid_payload(), ok_first=1)
    # judge-B: healthy, must still score all 3 after judge-A's quota skip.
    healthy = JsonJudge(_valid_payload())
    skipped = []
    produced = orch.run(
        blind_set,
        [Judge("judge-A", quota_provider), Judge("judge-B", healthy)],
        on_quota=lambda j, exc: skipped.append(j.judge_model),
    )
    # judge-A contributed 1, judge-B contributed 3 → 4 total; no exception raised.
    assert len(produced) == 4
    assert quota_provider.calls == 2          # 1 success + 1 quota; remaining 2 NOT attempted
    assert len(healthy.prompts) == 3          # other judge unaffected
    assert skipped == ["judge-A"]             # on_quota fired once for the skipped judge


def test_quota_skip_is_resumable_on_next_run(tmp_path):
    # First run hits quota immediately (scores nothing); a later run (quota restored)
    # fills everything — nothing is lost or double-counted.
    blind_set = build_blind_set([_gen("g0"), _gen("g1")], seed=0)
    first = QuotaJudge(_valid_payload(), ok_first=0)
    JudgeOrchestrator(Store(tmp_path)).run(blind_set, [Judge("judge-A", first)])
    assert len(Store(tmp_path).load(JUDGE_SCORES, JudgeScore)) == 0  # quota immediately → nothing

    healthy = JsonJudge(_valid_payload())
    JudgeOrchestrator(Store(tmp_path)).run(blind_set, [Judge("judge-A", healthy)])
    assert len(Store(tmp_path).load(JUDGE_SCORES, JudgeScore)) == 2  # resumed, both filled
