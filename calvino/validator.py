"""Deterministic structural validation — pure functions, no LLM calls.

This module exists to fix the failure mode from Phase 0.5: LLM judges miscounted
sentences in 3 of 7 outputs. The sentence count is the hard gate and must be
computed mechanically, never by a model.

Heuristics (documented because they are load-bearing):
- Sentences end on a run of `.!?` followed by whitespace+uppercase (or a closing
  quote then uppercase), or end-of-text.
- Dashes and semicolons are intra-sentence and never split (per the brief).
- Ellipses (`...` / `…`) are protected and never split.
- A `.` is NOT a boundary when it follows a known abbreviation (Mr., Dr., ...)
  or sits inside a decimal number — both are common false positives.
"""

from __future__ import annotations

import re

from .models import Generation, Validation

# Closing punctuation that can trail a terminator before the real break.
_CLOSERS = "\"')]}»”’"
# Opening punctuation a new sentence may legitimately start with.
_OPENERS = "\"'«“‘"
_TERMINATORS = ".!?"
_ELLIPSIS = re.compile(r"\.\.\.|…")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "mt", "vs", "etc",
    "jr", "sr", "prof", "rev", "gen", "sgt", "capt",
}
# Leading articles stripped before object matching. English + Italian (the Italian
# generalization arm): il/lo/la/i/gli/le/l'/un/uno/una. The head-noun fallback in
# object_present catches the rest, but stripping the article improves the full-phrase match.
_ARTICLES = {
    "the", "a", "an",
    "il", "lo", "la", "i", "gli", "le", "l'", "l", "un", "uno", "una",
}
_PLACEHOLDER = "\x00"


def split_sentences(text: str) -> list[str]:
    """Segment prose into sentences. See module docstring for the heuristics."""
    text = text.strip()
    if not text:
        return []

    protected = _ELLIPSIS.sub(_PLACEHOLDER, text)
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(protected)

    while i < n:
        if protected[i] not in _TERMINATORS:
            i += 1
            continue

        # Consume a run of terminators and any closing quotes/brackets.
        j = i + 1
        while j < n and protected[j] in _TERMINATORS + _CLOSERS:
            j += 1

        # Find the next non-space character after the candidate boundary.
        k = j
        while k < n and protected[k].isspace():
            k += 1

        if _is_boundary(protected, start, i, k, n):
            sentences.append(protected[start:j])
            start = k
            i = k
        else:
            i += 1

    if start < n and protected[start:].strip():
        sentences.append(protected[start:])

    return [s.replace(_PLACEHOLDER, "...").strip() for s in sentences if s.strip()]


def _is_boundary(text: str, start: int, term: int, nxt: int, n: int) -> bool:
    """Decide whether the terminator at `text[term]` ends a sentence.

    `nxt` is the index of the next non-space char (or `n` at end-of-text)."""
    if nxt >= n:
        return True
    if not (text[nxt].isupper() or text[nxt] in _OPENERS):
        return False
    if text[term] == ".":
        word = re.search(r"[A-Za-z]+$", text[start:term])
        if word and word.group(0).lower() in _ABBREVIATIONS:
            return False
    return True


def _normalize_ws(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def object_present(text: str, object_name: str) -> bool:
    """Whether the crossing object appears in `text` (case-insensitive).

    Strips a leading article from the object name, then accepts either the full
    phrase as a substring or the head (last) noun — so "The White Horse" matches
    "the white horse" and a later bare "the horse"."""
    haystack = _normalize_ws(text).lower()
    words = _normalize_ws(object_name).lower().split()
    if words and words[0] in _ARTICLES:
        words = words[1:]
    if not words:
        return False
    phrase = " ".join(words)
    if re.search(rf"\b{re.escape(phrase)}\b", haystack):
        return True
    head = words[-1]
    return bool(re.search(rf"\b{re.escape(head)}\b", haystack))


def evidence_span_verified(prose: str, span: str) -> bool:
    """Whether a judge's evidence span appears verbatim (whitespace-normalised)
    in the prose. Case-sensitive — the span is meant to be copied exactly."""
    span_n = _normalize_ws(span)
    if not span_n:
        return False
    return span_n in _normalize_ws(prose)


def validate_output(output_text: str, object_name: str) -> dict:
    """Run all structural checks on raw output text. Returns a plain dict so the
    caller can build a Validation tied to a generation id.

    The A/B split assigns sentences 1-2 to Story A and 3-4 to Story B; it is only
    meaningful when there are exactly 4 sentences (the hard gate)."""
    sentences = split_sentences(output_text)
    count = len(sentences)
    count_ok = count == 4

    half_a = " ".join(sentences[:2])
    half_b = " ".join(sentences[2:4])

    return {
        "sentence_count": count,
        "count_ok": count_ok,
        "split_ok": count_ok,
        "object_in_a": object_present(half_a, object_name),
        "object_in_b": object_present(half_b, object_name),
    }


def validate_generation(generation: Generation) -> Validation:
    """Build a Validation record for a stored Generation."""
    result = validate_output(generation.output_text, generation.object)
    return Validation(generation_id=generation.id, **result)
