"""Tests for construal extraction (Route A dilution check, §11.2) — pure parts only;
no provider/embedder/API needed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from calvino.construal import (
    CONSTRUAL_SCHEMA,
    construal_distance,
    parse_construal,
    render_construal_prompt,
)


def test_render_prompt_includes_object_and_passage():
    p = render_construal_prompt("The Lantern", "S1. S2. S3. S4.")
    assert "The Lantern" in p and "S1. S2. S3. S4." in p
    assert "extraction" in p.lower() and "NOT an evaluation" in p
    assert "json" in p.lower()   # OpenAI json_object mode requires the word "json" in the prompt


def test_schema_shape():
    assert CONSTRUAL_SCHEMA["required"] == ["meaning_a", "meaning_b"]
    assert CONSTRUAL_SCHEMA["additionalProperties"] is False


def test_parse_construal_ok():
    assert parse_construal({"meaning_a": " a signal ", "meaning_b": "a drowned hope"}) == (
        "a signal", "a drowned hope")


@pytest.mark.parametrize("bad", [
    {"meaning_a": "x"},                       # missing b
    {"meaning_a": "", "meaning_b": "y"},      # empty a
    {"meaning_a": 1, "meaning_b": "y"},       # wrong type
    ["a", "b"],                                # not a dict
])
def test_parse_construal_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_construal(bad)


def test_construal_distance_takes_min_over_two_meanings():
    # one construal identical to a cliché (distance 0), the other orthogonal (distance 1)
    cliche = np.array([[1.0, 0.0]])
    construals = np.array([[1.0, 0.0],    # = cliché  -> dist 0
                           [0.0, 1.0]])   # orthogonal -> dist 1
    # min over the two = 0 (one cliché is enough to be unoriginal)
    assert construal_distance(construals, cliche) == 0.0


def test_construal_distance_both_fresh():
    cliche = np.array([[1.0, 0.0]])
    construals = np.array([[0.0, 1.0], [0.0, 1.0]])  # both orthogonal to cliché
    assert math.isclose(construal_distance(construals, cliche), 1.0, abs_tol=1e-9)
