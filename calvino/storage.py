"""JSONL storage layer.

Append-only JSONL is the source of truth: crash-safe (only a trailing line can
be partial), resumable, human-readable, and loaded into pandas for analysis via
``pd.read_json(path, lines=True)``.

Each record type lives in its own file:
- generations.jsonl
- validations.jsonl
- judge_scores.jsonl
- raw_judge_responses.jsonl  (verbatim judge bytes, written BEFORE parsing)

The generation cache is an in-memory set of content-hash keys rebuilt from
generations.jsonl on startup, so a re-run skips work already on disk without any
API calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator, Type, TypeVar

from pydantic import BaseModel

from .models import Generation, JudgeScore, Validation

M = TypeVar("M", bound=BaseModel)

GENERATIONS = "generations.jsonl"
VALIDATIONS = "validations.jsonl"
JUDGE_SCORES = "judge_scores.jsonl"
RAW_JUDGE_RESPONSES = "raw_judge_responses.jsonl"


class Store:
    """Thin handle over a data directory of JSONL files."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def path(self, filename: str) -> Path:
        return self.data_dir / filename

    def append(self, filename: str, record: BaseModel) -> None:
        """Append one record as a single JSON line."""
        line = record.model_dump_json()
        with self.path(filename).open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load(self, filename: str, model: Type[M]) -> list[M]:
        """Load and validate every record from a JSONL file."""
        return list(self.iter(filename, model))

    def iter(self, filename: str, model: Type[M]) -> Iterator[M]:
        """Stream records one at a time without loading the whole file."""
        path = self.path(filename)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield model.model_validate_json(line)

    def append_raw(self, filename: str, payload: dict) -> None:
        """Append a raw (unvalidated) dict — used for verbatim judge responses
        so malformed replies are preserved before any parsing is attempted."""
        with self.path(filename).open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def generation_ids(self) -> set[str]:
        """Rebuild the set of generation ids already on disk.

        Because a generation's id IS its request hash (see make_request_id), this
        set doubles as the cache: a re-run skips any request whose id is present,
        with no API call."""
        return {g.id for g in self.iter(GENERATIONS, Generation)}

    def validated_ids(self) -> set[str]:
        """Generation ids already validated on disk, so re-validation is skipped."""
        return {v.generation_id for v in self.iter(VALIDATIONS, Validation)}

    def judged_pairs(self) -> set[tuple[str, str]]:
        """Rebuild the set of (generation_id, judge_model) pairs already scored on
        disk, so a re-run of the judge orchestrator skips completed work."""
        return {(s.generation_id, s.judge_model) for s in self.iter(JUDGE_SCORES, JudgeScore)}


def make_request_id(
    *,
    item_id: str,
    model: str,
    condition: str,
    prompt_variant: str,
    sample_idx: int,
    temperature: float,
    seed: int | None = None,
) -> str:
    """Stable hash over the fields that define a unique generation request.

    Identity is the request (what we'd ask the provider), not the response or
    timestamp — so a re-run with the same item/model/condition/variant/sample is
    recognised as already done. Used as the Generation id and the cache key.
    """
    parts = [
        item_id,
        model,
        condition,
        prompt_variant,
        str(sample_idx),
        str(temperature),
        "" if seed is None else str(seed),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def make_text_sha(text: str) -> str:
    """Content fingerprint for downstream cache keys. A request id identifies a
    SPEC, not a text: regenerating a purged row re-occupies its id, so any cache
    keyed on the id alone silently re-attaches old judgments to new text. Judged/validated
    records store this alongside the generation_id and a cache hit requires both
    to match.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
