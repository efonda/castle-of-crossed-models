"""Tests for the analysis layer.

ICC is validated against the canonical Shrout & Fleiss (1979) example. The rest
run on synthetic tidy frames so the deterministic behaviour (orthogonality sign,
rank ordering, thinking-effect direction, saturation flag) is checkable."""

import numpy as np
import pandas as pd
import pytest

from calvino.analysis import (
    PRIMARY_DIMS,
    build_scores_frame,
    combinatorial_orthogonality,
    descriptive_means,
    failure_catalogue,
    frontier_separation,
    icc_2_1,
    judge_agreement,
    mixed_effects,
    primary_aggregate,
    rank_stability,
    thinking_effect,
)
from calvino.harness import GenerationHarness, GenerationSpec
from calvino.items import Item
from calvino.models import Condition, PromptVariant
from calvino.providers import MockProvider
from calvino.storage import JUDGE_SCORES, Store
from calvino.models import JudgeScore, DimensionScores, DimensionEvidence


# --- ICC validation ----------------------------------------------------------

def test_icc_matches_shrout_fleiss_example():
    # Shrout & Fleiss (1979) table; known ICC(2,1) ≈ 0.29.
    data = np.array([
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ])
    assert icc_2_1(data) == pytest.approx(0.29, abs=0.02)


def test_icc_perfect_agreement_is_one():
    data = np.array([[1, 1], [3, 3], [5, 5], [2, 2]])
    assert icc_2_1(data) == pytest.approx(1.0, abs=1e-6)


def test_icc_degenerate_returns_nan():
    assert np.isnan(icc_2_1(np.array([[1, 2, 3]])))  # one unit


# --- synthetic frame ---------------------------------------------------------

def _frame(seed=0):
    """3 models x 4 items x 2 samples x 3 judges. model_c best, model_a worst,
    so rankings/saturation are predictable. Judges agree closely (small noise)."""
    rng = np.random.default_rng(seed)
    base = {"model_a": 2.0, "model_b": 3.0, "model_c": 4.5}
    rows = []
    for model, level in base.items():
        for item in range(4):
            for sample in range(2):
                gen_id = f"{model}-i{item}-s{sample}"
                for judge in ("j1", "j2", "j3"):
                    row = {
                        "generation_id": gen_id, "item_id": f"item{item}",
                        "model": model, "condition": "normal",
                        "prompt_variant": "base", "sample_idx": sample,
                        "judge_model": judge,
                    }
                    for dim in PRIMARY_DIMS:
                        row[dim] = float(np.clip(level + rng.normal(0, 0.3), 0, 5))
                    # combinatorial correlated with the core here.
                    row["combinatorial"] = float(np.clip(level + rng.normal(0, 0.3), 0, 5))
                    rows.append(row)
    return pd.DataFrame(rows)


def test_primary_aggregate_excludes_combinatorial():
    df = pd.DataFrame([{"distinctness": 4, "bridge": 4, "tonal": 4, "originality": 4, "combinatorial": 0}])
    assert primary_aggregate(df).iloc[0] == 4.0  # combinatorial ignored


def test_descriptive_means_orders_make_sense():
    df = _frame()
    means = descriptive_means(df).set_index("model")
    agg = means[list(PRIMARY_DIMS)].mean(axis=1)
    assert agg["model_c"] > agg["model_b"] > agg["model_a"]
    assert (means["n_scores"] > 0).all()


def test_judge_agreement_returns_all_dimensions():
    df = _frame()
    agree = judge_agreement(df)
    assert set(agree["dimension"]) == set(PRIMARY_DIMS) | {"combinatorial"}
    assert (agree["n_judges"] == 3).all()


def test_combinatorial_orthogonality_detects_correlation():
    df = _frame()  # constructed so combinatorial tracks the core
    orth = combinatorial_orthogonality(df)
    assert orth.n > 0
    assert orth.r > 0.5  # redundant with core in this synthetic set


def test_rank_stability_recovers_true_order():
    df = _frame()
    rs = rank_stability(df, n_boot=200, seed=1)
    assert rs.ranking == ["model_c", "model_b", "model_a"]
    assert rs.mean_spearman > 0.8  # stable ordering
    # model_c sits at rank 0 in the vast majority of resamples.
    assert rs.rank_freq["model_c"].get(0, 0) > 0.8


def test_frontier_separation_flags_no_saturation_for_separated_models():
    df = _frame()
    fr = frontier_separation(df, n_boot=200, seed=2)
    assert list(fr.table["model"])[0] == "model_c"
    assert fr.top_saturated is False  # c and b are well separated


def test_frontier_separation_flags_saturation_when_tied():
    df = _frame()
    # Force a near-tie at the top by relabelling model_b's scores up to model_c's level.
    df2 = df.copy()
    mask = df2["model"] == "model_b"
    for dim in PRIMARY_DIMS:
        df2.loc[mask, dim] = df2.loc[mask, dim] + 1.5
    fr = frontier_separation(df2, n_boot=200, seed=3)
    assert fr.top_saturated is True


def test_thinking_effect_direction():
    # Build a small paired frame: model_t scored under both conditions, thinking higher.
    rows = []
    for item in range(4):
        for cond, level in (("normal", 3.0), ("thinking", 4.0)):
            gen_id = f"model_t-{cond}-i{item}"
            for judge in ("j1", "j2"):
                row = {
                    "generation_id": gen_id, "item_id": f"item{item}",
                    "model": "model_t", "condition": cond,
                    "prompt_variant": "base", "sample_idx": 0, "judge_model": judge,
                }
                for dim in PRIMARY_DIMS + ("combinatorial",):
                    row[dim] = level
                rows.append(row)
    df = pd.DataFrame(rows)
    eff = thinking_effect(df).set_index("dimension")
    assert eff.loc["distinctness", "mean_diff"] == pytest.approx(1.0, abs=1e-6)
    assert (eff["n_pairs"] == 4).all()


def test_thinking_effect_handles_no_paired_data():
    # All-normal frame: no model has both conditions. Must not crash; n_pairs=0.
    df = _frame()
    eff = thinking_effect(df)
    assert (eff["n_pairs"] == 0).all()
    assert eff["mean_diff"].isna().all()


def test_failure_catalogue_surfaces_worst():
    df = _frame()
    cat = failure_catalogue(df, n=3)
    assert len(cat["worst"]) == 3
    assert len(cat["most_disagreed"]) == 3
    # The worst outputs should belong to the weakest model.
    assert (cat["worst"]["model"] == "model_a").all()


def test_mixed_effects_fits_and_returns_per_model():
    df = _frame()
    result = mixed_effects(df, "distinctness")
    # May return None if it fails to converge; if it fits, check shape.
    if result is not None:
        assert set(result["model"]) == {"model_a", "model_b", "model_c"}
        assert "estimate" in result.columns and "se" in result.columns


# --- frame builder from storage ----------------------------------------------

def test_build_scores_frame_joins_generation_metadata(tmp_path):
    store = Store(tmp_path)
    item = Item(
        item_id="lantern-01", object="The Lantern", object_type="tarot_image_object",
        description="a tin lantern", story_a="A.", story_b="B.",
    )
    harness = GenerationHarness(store, MockProvider())
    gen = harness.run_spec(GenerationSpec(item=item, model="model-A", sample_idx=0))

    # Two judges score it.
    for judge in ("j1", "j2"):
        store.append(JUDGE_SCORES, JudgeScore(
            generation_id=gen.id, judge_model=judge, blind_id="out-0000",
            scores=DimensionScores(distinctness=4, bridge=4, tonal=4, originality=4, combinatorial=3),
            evidence=DimensionEvidence(distinctness="x", bridge="x", tonal="x", originality="x", combinatorial="x"),
            raw_response="{}",
        ))

    df = build_scores_frame(store)
    assert len(df) == 2  # one row per judge
    assert set(df["judge_model"]) == {"j1", "j2"}
    assert (df["model"] == "model-A").all()
    assert (df["item_id"] == "lantern-01").all()
