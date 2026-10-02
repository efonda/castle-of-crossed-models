"""Tests for the deterministic validator.

This is the component that fixes the LLM judges' counting failure, so the
sentence counter is tested hard: known 3/4/5-sentence cases, plus the tricky
segmentation cases (dashes, semicolons, ellipses, abbreviations, decimals,
quoted dialogue) that naive splitters get wrong.
"""

import pytest

from calvino.validator import (
    evidence_span_verified,
    object_present,
    split_sentences,
    validate_output,
)


# --- sentence counting: the hard gate ---------------------------------------

def test_exactly_four_sentences():
    text = (
        "The lantern guttered as she climbed the last stair. "
        "She set it down and the flame steadied. "
        "Down in the harbor, a different lantern swung from a mast. "
        "The captain read its rhythm and turned the ship north."
    )
    assert len(split_sentences(text)) == 4


def test_three_sentences():
    text = "He opened the door. The room was empty. He left."
    assert len(split_sentences(text)) == 3


def test_five_sentences():
    text = "One. Two. Three. Four. Five."
    assert len(split_sentences(text)) == 5


def test_trailing_text_without_terminator_counts():
    text = "First sentence here. Second one trails off with no period"
    assert len(split_sentences(text)) == 2


# --- segmentation edge cases that naive splitters get wrong ------------------

def test_dashes_and_semicolons_do_not_split():
    text = (
        "The key turned — slowly, reluctantly — and the lock gave; "
        "inside, the box held nothing but salt."
    )
    assert len(split_sentences(text)) == 1


def test_ellipsis_does_not_split():
    assert len(split_sentences("She waited... and waited... then gave up.")) == 1
    assert len(split_sentences("She waited… then left.")) == 1


def test_abbreviations_do_not_split():
    text = "Dr. Reyes set the bell on the table. Mr. Voss did not flinch."
    assert len(split_sentences(text)) == 2


def test_decimal_numbers_do_not_split():
    text = "The scale read 3.5 grams. He frowned at the figure."
    assert len(split_sentences(text)) == 2


def test_quoted_dialogue_splits_after_closing_quote():
    text = 'She whispered, "Go now." He went without looking back.'
    assert len(split_sentences(text)) == 2


def test_multiple_terminators_collapse():
    assert len(split_sentences("What?! He could not believe it.")) == 2


def test_empty_text():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


# --- object presence ---------------------------------------------------------

def test_object_present_strips_article_and_matches_phrase():
    assert object_present("A white horse bolted across the field.", "The White Horse")


def test_object_present_matches_head_noun():
    # Full phrase absent, but the head noun appears later as a bare reference.
    assert object_present("The horse reared at the gate.", "The White Horse")


def test_object_absent():
    assert not object_present("The garden was quiet at dusk.", "The Lantern")


def test_object_head_noun_word_boundary():
    # "keyhole" must not count as the object "The Key".
    assert not object_present("She peered through the keyhole.", "The Key")


# --- evidence span verification ----------------------------------------------

def test_verbatim_span_verified():
    prose = "The lantern guttered as she climbed the last stair."
    assert evidence_span_verified(prose, "guttered as she climbed")


def test_whitespace_normalised_span_verified():
    prose = "The lantern guttered   as she\nclimbed the stair."
    assert evidence_span_verified(prose, "guttered as she climbed")


def test_fabricated_span_rejected():
    prose = "The lantern guttered as she climbed the last stair."
    # A judge fabricating a quote that is not in the prose must be flagged.
    assert not evidence_span_verified(prose, "the captain read its rhythm")


def test_empty_span_rejected():
    assert not evidence_span_verified("Some prose here.", "   ")


# --- end-to-end structural validation ----------------------------------------

def test_validate_output_passing_case():
    text = (
        "The lantern guttered as she climbed the last stair. "
        "She set the lantern down and the flame steadied. "
        "Down in the harbor, a different lantern swung from a mast. "
        "The captain read the lantern's rhythm and turned north."
    )
    result = validate_output(text, "The Lantern")
    assert result["sentence_count"] == 4
    assert result["count_ok"] is True
    assert result["split_ok"] is True
    assert result["object_in_a"] is True
    assert result["object_in_b"] is True


def test_validate_output_failing_count_keeps_record():
    text = "Only one sentence with the key in it."
    result = validate_output(text, "The Key")
    assert result["sentence_count"] == 1
    assert result["count_ok"] is False
    # Object only in the (single) first half; second half is empty.
    assert result["object_in_a"] is True
    assert result["object_in_b"] is False
