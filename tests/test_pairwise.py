"""Tests for the pairwise forced-choice helpers."""

import pytest

from calvino.pairwise import (
    comparison_id,
    parse_pairwise,
    presentation_flip,
    render_pairwise_prompt,
)


def test_render_pairwise_prompt():
    out = render_pairwise_prompt("The Lantern", "passage A text", "passage B text")
    assert "The Lantern" in out and "passage A text" in out and "passage B text" in out
    assert '"winner"' in out
    assert "{" in out  # JSON example present; format() left no stray field braces
    assert "{OUTPUT_A}" not in out


def test_parse_pairwise():
    assert parse_pairwise({"winner": "A"}) == "A"
    assert parse_pairwise({"winner": "b", "reason": "x"}) == "B"
    with pytest.raises((KeyError, ValueError)):
        parse_pairwise({"winner": "tie"})
    with pytest.raises((KeyError, ValueError)):
        parse_pairwise({})


def test_comparison_id_is_order_invariant():
    assert comparison_id("it", "m1", "m2", 0) == comparison_id("it", "m2", "m1", 0)
    assert comparison_id("it", "m1", "m2", 0) != comparison_id("it", "m1", "m2", 1)


def test_presentation_flip_deterministic():
    cid = comparison_id("it", "m1", "m2", 0)
    assert presentation_flip(cid) == presentation_flip(cid)  # stable
