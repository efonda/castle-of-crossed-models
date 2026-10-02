"""Tests for the item bank and prompt renderer."""

from pathlib import Path

import pytest

from calvino.items import Item, load_item_bank
from calvino.models import ObjectType, PromptVariant
from calvino.prompts import render_prompt

ITEMS_PATH = Path(__file__).resolve().parent.parent / "data" / "items.yaml"


def _item(**overrides) -> Item:
    base = dict(
        item_id="t-1",
        object="The Lantern",
        object_type=ObjectType.TAROT,
        description="a battered tin lantern",
        story_a="A smuggler waits on the headland.",
        story_b="A widow descends into the flooded cellar.",
    )
    base.update(overrides)
    return Item(**base)


# --- item bank ---------------------------------------------------------------

def test_real_item_bank_loads_and_validates():
    bank = load_item_bank(ITEMS_PATH)
    assert bank.version
    assert len(bank.items) >= 1
    item = bank.by_id("lantern-01")
    assert item.object_type == ObjectType.TAROT
    assert item.prompt_variants == [PromptVariant.BASE]


def test_by_id_raises_on_missing():
    bank = load_item_bank(ITEMS_PATH)
    with pytest.raises(KeyError):
        bank.by_id("does-not-exist")


# --- rendering ---------------------------------------------------------------

def test_base_render_substitutes_all_fields():
    item = _item()
    out = render_prompt(item, PromptVariant.BASE)
    assert "The Lantern — a battered tin lantern" in out
    assert item.story_a in out
    assert item.story_b in out
    assert "exactly 4 sentences" in out
    # No unfilled placeholders left behind.
    assert "{" not in out and "}" not in out


def test_harder_variant_adds_causal_constraint():
    item = _item()
    base = render_prompt(item, PromptVariant.BASE)
    harder = render_prompt(item, PromptVariant.HARDER)
    assert "physically *cause* the transition" in harder
    assert "physically *cause* the transition" not in base
    assert harder.endswith("Output only the paragraph.")


def test_minimal_variant_renders_and_substitutes():
    item = _item()
    out = render_prompt(item, PromptVariant.MINIMAL)
    assert "The Lantern" in out
    assert item.story_a in out
    assert "{" not in out and "}" not in out


def test_label_visible_uses_label_form():
    item = _item(object="The White Horse", label="Death — a pale horse with hollow eyes")
    out = render_prompt(item, PromptVariant.BASE, label_visible=True)
    assert "Death — a pale horse" in out
    assert "The White Horse —" not in out


def test_label_visible_without_label_raises():
    item = _item()  # no label
    with pytest.raises(ValueError):
        render_prompt(item, PromptVariant.BASE, label_visible=True)


def test_story_context_with_braces_does_not_break_format():
    # Defensive: a brace in authored prose must not be treated as a placeholder.
    item = _item(story_a="He counted {the cost} silently.")
    out = render_prompt(item, PromptVariant.BASE)
    assert "{the cost}" in out
