"""Pairwise forced-choice probe.

Absolute 0-5 scoring has no resolution left near ceiling (every frontier output is a
4 or 5), so within-cluster ICC collapses to ~0. Forced choice — "A or B, which is the
better crossing?" — is far more sensitive: it can surface a consistent preference too
small for the 0-5 scale to register. This module compares two outputs for the same
item and asks a judge to pick one; the runner aggregates votes into inter-judge
agreement and per-model win rates.

The decisive read:
- judges AGREE on A-vs-B (well above chance) AND some model reliably beats another
  => frontier resolution is recoverable; the 0-5 scale was the limiter.
- agreement ~ chance and win rates ~ 50% => the frontier is genuinely unresolvable
  by LLM judges on this construct.
"""

from __future__ import annotations

import hashlib

PAIRWISE_PROMPT = """\
You are comparing two short passages, A and B, written for the SAME task: a single
paragraph where two separate stories cross through one shared object that must mean
something different in each, joined by a grounded physical/causal/temporal link.

Crossing object: {OBJECT}

Passage A:
\"\"\"
{OUTPUT_A}
\"\"\"

Passage B:
\"\"\"
{OUTPUT_B}
\"\"\"

Which passage is the better crossing — sharper distinct meanings for the object, a
more grounded link between the two stories, more controlled and original prose? You
MUST choose one; ties are not allowed.

Output ONLY JSON: {{"winner": "A" or "B", "reason": "<one short sentence>"}}"""

PAIRWISE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B"]},
        "reason": {"type": "string"},
    },
    "required": ["winner"],
}


def render_pairwise_prompt(obj: str, output_a: str, output_b: str) -> str:
    return PAIRWISE_PROMPT.format(OBJECT=obj, OUTPUT_A=output_a, OUTPUT_B=output_b)


def parse_pairwise(payload: dict) -> str:
    """Return 'A' or 'B'. Raises (caught by caller) on a malformed winner."""
    w = str(payload["winner"]).strip().upper()
    if w not in ("A", "B"):
        raise ValueError(f"bad winner {w!r}")
    return w


# --- multi-object pairwise variant (not used in this release's runs) ---------
#
# Differences from the single-crossing prompt, both frozen in the design doc:
# an explicit ABSTAIN option (a comparison is "decided" only when both
# position-flipped presentations agree; abstentions and splits are excluded),
# and the full deck+paths context so the judge can see the global constraint.

TABLEAU_PAIRWISE_PROMPT = """\
You are comparing two TABLEAUX, A and B, written for the SAME task: six short
stories told over one shared 3x3 grid of nine cards — three stories along the
rows, three along the columns — so every card belongs to exactly two stories and
must mean something clearly different in each, without changing what it
physically is. The stories should be independent worlds crossing only through
the shared cards.

The grid and paths both tableaux were given:
{DECK}

Tableau A:
\"\"\"
{OUTPUT_A}
\"\"\"

Tableau B:
\"\"\"
{OUTPUT_B}
\"\"\"

Which tableau is better — sharper double meanings at the crossings, more
controlled and original stories, an arrangement that genuinely holds together?
If you cannot honestly tell them apart, abstain.{EMPHASIS}

Output ONLY JSON: {{"winner": "A" or "B" or "abstain", "reason": "<one short sentence>"}}"""

# Instrument-sensitivity weightings (2026-07-09). "neutral" leaves the frozen prompt
# byte-identical (EMPHASIS=""), so the FC panel is unaffected. The two symmetric arms lean the
# criterion toward the two ends of the dimensional trade-off (composition vs the crossing
# mechanic) to test whether a near-tie's verdict is prompt-weighting-dependent.
# To keep the arms a true like-for-like against the ABSOLUTE rubric, each emphasis carries the SAME
# criterion definition the absolute scoring prompt uses (GESTALT and DISTINCTNESS respectively),
# only re-cast comparatively. gestalt-FC <-> gestalt-absolute; mechanic-FC <-> distinctness-absolute.
_WEIGHTING_EMPHASIS = {
    "neutral": "",
    "gestalt": (
        " Judge THIS comparison ONLY on GESTALT: read each tableau's six stories as one arrangement."
        " The best is six genuinely independent tellings over one shared deck, where the grid"
        " disappears into the prose; the worst is card-mentions distributed to satisfy a checker,"
        " with repeated templates and stories that do not survive as stories. Pick the tableau that"
        " better achieves this whole-arrangement quality; treat the sharpness of individual card"
        " double-meanings and local prose as secondary."
    ),
    "mechanic": (
        " Judge THIS comparison ONLY on DISTINCTNESS: for each card, does it mean clearly different"
        " things in its row-story and its column-story? Pick the tableau whose shared cards carry"
        " sharper, more clearly different dual meanings across their two stories; treat overall"
        " composition and prose flow as secondary."
    ),
}

TABLEAU_PAIRWISE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "abstain"]},
        "reason": {"type": "string"},
    },
    "required": ["winner"],
}


def render_tableau_pairwise_prompt(deck: str, output_a: str, output_b: str,
                                   weighting: str = "neutral") -> str:
    if weighting not in _WEIGHTING_EMPHASIS:
        raise ValueError(f"unknown fc_weighting {weighting!r}; expected one of "
                         f"{sorted(_WEIGHTING_EMPHASIS)}")
    return TABLEAU_PAIRWISE_PROMPT.format(
        DECK=deck, OUTPUT_A=output_a, OUTPUT_B=output_b,
        EMPHASIS=_WEIGHTING_EMPHASIS[weighting])


def parse_pairwise_abstain(payload: dict) -> str:
    """Return 'A', 'B', or 'ABSTAIN'. Raises (caught by caller) on a malformed winner."""
    w = str(payload["winner"]).strip().upper()
    if w not in ("A", "B", "ABSTAIN"):
        raise ValueError(f"bad winner {w!r}")
    return w


def comparison_id(item_id: str, model_a: str, model_b: str, sample_idx: int) -> str:
    """Stable id for one comparison; model order normalised so (a,b) == (b,a)."""
    lo, hi = sorted([model_a, model_b])
    return f"{item_id}|{lo}|{hi}|s{sample_idx}"


def same_model_comparison_id(item_id: str, model: str, sample_a: int, sample_b: int) -> str:
    """Stable id for a same-model cross-sample comparison (tableau re-screen pairs,
    design doc §6/B2): sample order normalised so (i,j) == (j,i)."""
    lo, hi = sorted([sample_a, sample_b])
    return f"{item_id}|{model}|self|s{lo}v{hi}"


def presentation_flip(comparison_id: str) -> bool:
    """Deterministic A/B presentation order per comparison (controls position bias
    without per-judge variation, so inter-judge agreement stays clean)."""
    return int(hashlib.sha256(comparison_id.encode()).hexdigest(), 16) % 2 == 1
