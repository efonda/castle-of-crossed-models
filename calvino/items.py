"""Item bank — the hand-authored set of crossing objects and story pairs.

Items are config (YAML), not generated data, so they live separately from the
JSONL data layer. Each item supplies the four substitution fields the prompt
renderer needs (object, description, story_a, story_b) plus the design factors
(object_type, which prompt variants to run, label-ablation).

Author the story-pairs so the two stories are thematically orthogonal and a
grounded bridge is *available* but not handed over (brief §8).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .models import ObjectType, PromptVariant


class Item(BaseModel):
    """One crossing item. `description` is a NEUTRAL visual description of the
    image-object, not its divinatory label."""

    item_id: str
    object: str
    object_type: ObjectType
    description: str
    story_a: str
    story_b: str
    # Which prompt variants this item should be generated under. Most items run
    # BASE only; 2-3 items also run MINIMAL to test prompt sensitivity (brief §7).
    prompt_variants: list[PromptVariant] = Field(default_factory=lambda: [PromptVariant.BASE])
    # Label ablation: the divinatory label form (e.g. "Death — a white horse…").
    # Used only when an item is rendered label-visible; absent for most items.
    label: str | None = None
    # Design family for the hard-item (Route B) contrast: "cliche-trap" (an overwhelming
    # canonical reading both stories must subvert) vs "invented" (no settled meaning, forcing
    # fresh construal). Lets analysis report per-family spread separately — the whole point of
    # Route B is the cliché-trap-vs-invented comparison, not a pooled number (§11.3).
    family: str | None = None
    # Language the item is authored in ("en", "it", ...). Selects the generation prompt
    # template so the model writes in that language. The Italian arm is a generalization
    # check — its items MUST be native-authored, never model-translated.
    language: str = "en"


class ItemBank(BaseModel):
    """Versioned collection of items. `version` lets analysis pin which bank a
    generation came from (brief: version the item bank)."""

    version: str
    items: list[Item]

    def by_id(self, item_id: str) -> Item:
        for item in self.items:
            if item.item_id == item_id:
                return item
        raise KeyError(f"no item with id {item_id!r}")


def load_item_bank(path: str | Path) -> ItemBank:
    """Load and validate the item bank from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ItemBank.model_validate(raw)
