"""Construal extraction — the final dilution check for Route A (technical report §11.2).

Whole-passage cliché-distance is *coarse* because the meaning-bearing content (what the
object actually means in each story) is ~15 of ~80 words; embedding the whole passage
averages that signal away. This module isolates it: an extractor model pulls the object's
**claimed meaning in story A** and **in story B** as short phrases, and we measure
*those* against the cliché set, not the whole passage.

Caveats this carries (stated up front): it **reintroduces a model** in the extraction step,
so it is no longer strictly judge-free — though extraction ("what does the object mean
here") is far more *extractive* and lower-subjectivity than judging ("rate originality
0–5"), and can be cross-checked with a second extractor. And even if it sharpens the metric,
the floor may still sit above the frontier trio's ~0.1 gap. It is one bounded shot.

Pure here: the prompt, the schema, the parse, and the construal-distance aggregation. The
provider call + caching live in `scripts/route_a_originality.py`.
"""

from __future__ import annotations

import numpy as np

from .originality import nearest_cliche_distance

CONSTRUAL_PROMPT = """You are extracting, as literally as possible, what a single OBJECT \
means or does in each of two short stories that cross through it. This is an extraction \
task, NOT an evaluation — do not judge quality, originality, or writing; just state the \
object's meaning/role in each story.

OBJECT: {object}

PASSAGE (four sentences; sentences 1–2 are Story A, sentences 3–4 are Story B):
{passage}

Return the object's meaning/role in Story A and in Story B, each as a short phrase of at \
most ~8 words (a noun phrase or short clause). Be concrete and faithful to the text.

Respond with a JSON object with string fields "meaning_a" and "meaning_b"."""

# Forced-JSON schema (Anthropic tool-use / OpenAI response_format), same boundary the judge uses.
CONSTRUAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "meaning_a": {"type": "string", "description": "the object's meaning/role in Story A"},
        "meaning_b": {"type": "string", "description": "the object's meaning/role in Story B"},
    },
    "required": ["meaning_a", "meaning_b"],
    "additionalProperties": False,
}


def render_construal_prompt(object_: str, passage: str) -> str:
    return CONSTRUAL_PROMPT.format(object=object_, passage=passage)


def parse_construal(payload: dict) -> tuple[str, str]:
    """Pull (meaning_a, meaning_b) from a parsed JSON payload; raise on malformed."""
    if not isinstance(payload, dict):
        raise ValueError(f"construal payload not an object: {type(payload)}")
    a, b = payload.get("meaning_a"), payload.get("meaning_b")
    if not isinstance(a, str) or not isinstance(b, str) or not a.strip() or not b.strip():
        raise ValueError(f"construal payload missing meaning_a/meaning_b: {payload!r}")
    return a.strip(), b.strip()


def construal_distance(construal_vecs: np.ndarray, cliche_vecs: np.ndarray) -> float:
    """Distance of a generation's *two extracted construals* from the object's clichés.

    `construal_vecs` is the 2×d stack [embed(meaning_a), embed(meaning_b)]. Each construal
    gets its nearest-cliché distance; we take the **min** — a crossing is unoriginal if
    *either* meaning is a stale cliché (the same 'one cliché is enough' logic as the
    whole-passage nearest-cliché metric). Higher ⇒ both meanings fresh ⇒ more original."""
    d = nearest_cliche_distance(construal_vecs, cliche_vecs)   # length-2
    return float(d.min())
