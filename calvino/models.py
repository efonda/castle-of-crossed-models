"""Pydantic models for the Calvino Benchmark data records.

Three records flow through the pipeline and are persisted as JSONL:
- Generation  — one model output for one item under one condition/sample.
- Validation  — deterministic structural checks on a Generation.
- JudgeScore  — one blind judge's scoring of one Generation.

Judge output is adversarial (models malform/fabricate), so the scoring models
validate at the parse boundary. Raw judge bytes are persisted separately before
parsing so malformed replies survive for the failure-mode catalogue.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ObjectType(str, Enum):
    TAROT = "tarot_image_object"
    ORDINARY = "ordinary_physical_object"
    INVENTED = "invented_symbolic_object"
    # Multi-object variant (not used in this release's runs).
    TABLEAU = "tableau_deck"


class Condition(str, Enum):
    NORMAL = "normal"
    THINKING = "thinking"


class PromptVariant(str, Enum):
    BASE = "base"
    MINIMAL = "minimal"
    HARDER = "harder"
    # Free-sentence-count style variants (not part of the scored benchmark):
    LIGHT_MIN = "light_min"          # minimal "write it light" ask, free sentence count
    LIGHT_MAX = "light_max"          # full Six Memos guidance, free sentence count
    GRAVITA_FREE = "gravita_free"    # heavy (BASE) style guidance, free count — count control
    # Multi-object variant (not used in this release's runs):
    TABLEAU = "tableau"


class Generation(BaseModel):
    """One model output. The cache key is derived from the fields that define
    a unique generation request (see storage.generation_cache_key)."""

    id: str
    item_id: str
    object: str
    object_type: ObjectType
    model: str
    condition: Condition
    prompt_variant: PromptVariant
    sample_idx: int
    temperature: float
    seed: Optional[int] = None
    rendered_prompt: str
    output_text: str
    timestamp: datetime = Field(default_factory=_utcnow)
    provider_meta: dict[str, Any] = Field(default_factory=dict)


class Validation(BaseModel):
    """Deterministic structural checks. Computed by pure functions, never an LLM.
    count_ok is the hard gate; the other fields are recorded for audit."""

    generation_id: str
    sentence_count: int
    count_ok: bool
    split_ok: bool
    object_in_a: bool
    object_in_b: bool


class DimensionScores(BaseModel):
    """The five rubric dimensions, each 0-5. combinatorial is exploratory and
    must never be aggregated into the headline score.

    Float (not int): single-crossing judges return integers, but the grid rubric
    averages per-crossing scores into these fields, which can be non-integer."""

    distinctness: float = Field(ge=0, le=5)
    bridge: float = Field(ge=0, le=5)
    tonal: float = Field(ge=0, le=5)
    originality: float = Field(ge=0, le=5)
    combinatorial: float = Field(ge=0, le=5)


class DimensionEvidence(BaseModel):
    """Verbatim evidence span the judge cites for each dimension's score."""

    distinctness: str
    bridge: str
    tonal: str
    originality: str
    combinatorial: str


class JudgeScore(BaseModel):
    """One blind judge's scoring of one (anonymised) generation.

    blind_id is the anonymised handle the judge actually saw; the link back to
    the real generation_id lives only here, never in what the judge is shown.
    spans_verified records, per dimension, whether the evidence span was found
    verbatim in the prose (computed deterministically, post-hoc)."""

    generation_id: str
    judge_model: str
    blind_id: str
    scores: DimensionScores
    evidence: DimensionEvidence
    spans_verified: dict[str, bool] = Field(default_factory=dict)
    raw_response: str
    timestamp: datetime = Field(default_factory=_utcnow)
    # Optional holistic "execution quality" dimension, 0-100 (opt-in via JudgeSpec.execution).
    # Deliberately a SEPARATE 0-100 axis, never normalized to 0-5 and never folded into the
    # primary aggregate (the five named dims saturate at the frontier; this probes the unnamed
    # holistic axis with headroom). None for every judge/run that did not request it — so old
    # records load unchanged.
    execution: Optional[float] = Field(default=None, ge=0, le=100)
    execution_evidence: Optional[str] = None
