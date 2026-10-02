"""Run configuration.

A run is fully described by one YAML config: which item bank, which models (and
the conditions each is run under), how many samples, which judges, and where to
write data. Driving the pipeline from config — rather than CLI flags — keeps runs
reproducible and reviewable (brief §14: config-driven, no UI).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .models import Condition


class ModelSpec(BaseModel):
    """One model under test. `conditions` lets the thinking-paired model declare
    both [normal, thinking]; most models declare [normal] only."""

    name: str
    provider: str = "anthropic"
    conditions: list[Condition] = Field(default_factory=lambda: [Condition.NORMAL])
    # API model id to actually call; defaults to `name`. Mirrors JudgeSpec.model — use a
    # distinct `name` (the on-disk identity / cache key) with a `model` override to run the
    # same underlying API model under a separate identity. E.g. a reasoning-equalised
    # {name: gpt-5.5-noreason, model: gpt-5.5} slice kept separate from the default-reasoning
    # gpt-5.5 rows (2026-07-08 reasoning-confound control).
    model: str | None = None
    # Adaptive-thinking effort, used only when a `thinking` condition is generated (Fable/Mythos
    # 5+: thinking.type.adaptive + output_config.effort). "low"/"medium"/"high"/"xhigh"/"max";
    # default "high" (the API default). Effort is a call-time knob, NOT a cache dimension, so an
    # effort ladder needs DISTINCT `name`s (e.g. claude-fable-5-elow, model: claude-fable-5) — the
    # name is the cache key; effort never enters the request hash.
    effort: str = "high"


class JudgeSpec(BaseModel):
    """One blind judge. Use at least one judge family different from the tested
    models to triangulate (brief §10).

    `judge_model` is the label stored on disk (the judge's identity in analysis).
    `model` is the API model id to actually call; it defaults to `judge_model`.
    Use a distinct label when running the same model under a different identity —
    e.g. an automated `claude-opus-4-8-api` judge kept separate from manual
    `claude-opus-4-8` scores."""

    judge_model: str
    provider: str = "anthropic"
    model: str | None = None
    # Extended thinking for the judge. Off by default (all judges so far scored with
    # thinking OFF). When on, the provider enables thinking with `thinking_budget`
    # tokens; for Anthropic this forces temperature 1.0 (so a thinking judge is not
    # bit-deterministic). Give a thinking judge a DISTINCT judge_model label so its
    # scores stay separable from the non-thinking run of the same model.
    thinking: bool = False
    # Adaptive-thinking effort for Opus 4.8+ (thinking.type.adaptive + output_config.effort):
    # "high" / "medium" / "low". This is the modern reasoning knob.
    effort: str = "high"
    # Legacy token budget — used only as the provider's fallback for older models that
    # reject the adaptive API.
    thinking_budget: int = 8192
    # Max integer score on the rubric. 5 = canonical. 10 = finer-grained ceiling/resolution
    # check; scores are normalized back to the 0-5 axis at parse time so they pool with 0-5
    # data. Run a scale-10 judge under a DISTINCT judge_model label (e.g. `gpt-5.5-s10`).
    scale: int = 5
    # Rubric language ("en" default, "it" for the Italian generalization arm). Selects the
    # fixed scoring prompt; JSON field names stay English so parsing is unchanged.
    language: str = "en"
    # Also score a holistic "execution quality" dimension on a separate 0-100 scale, in the
    # SAME call as the five named dims (so it costs no extra calls). Probes the unnamed
    # holistic axis the five saturating dims can't register; stored raw (never normalized,
    # never aggregated). Off by default — existing runs/records are unaffected.
    execution: bool = False


class RunConfig(BaseModel):
    data_dir: str
    item_bank: str
    models: list[ModelSpec]
    judges: list[JudgeSpec]
    samples: int = 3
    temperature: float = 0.7
    blind_seed: int = 0
    bootstrap: int = 1000
    # Optional item scope: run only these item ids from item_bank (None = all). Lets a
    # pilot exercise a subset of a frozen bank without touching the bank's hash.
    items: list[str] | None = None
    # Pairwise: also emit same-model cross-sample pairs (default off).
    self_pairs: bool = False
    # Pairwise: restrict comparisons to pairs where BOTH models are in this set
    # (None = all cross-model pairs).
    fc_models: list[str] | None = None
    # Pairwise criterion weighting: "neutral" (default; the fixed pairwise prompt) or one of
    # two symmetric alternative weightings used for a prompt-sensitivity check.
    fc_weighting: str = "neutral"


def load_run_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RunConfig.model_validate(raw)
