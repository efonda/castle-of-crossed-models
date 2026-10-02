"""Tests for Route A (semantic cliché-distance) — all with a controllable stub
embedder, so they need no model or API. They lock the *logic* of the four
kill-switch steps; whether the real metric works is an empirical question for a
real embedder (technical report §11.2)."""

from __future__ import annotations

import math

import numpy as np

from calvino.originality import (
    CLICHE_REFERENCE,
    CLICHE_REFERENCE_NARRATIVE,
    REFERENCE_SETS,
    DistanceRecord,
    distance_rows,
    frontier_separation,
    generation_level_correlation,
    group_contrast,
    l2_normalize,
    model_level_spearman,
    nearest_cliche_distance,
    normalize_within_item,
    per_model_mean,
    resolution_ladder,
    two_embedder_agreement,
    validate_against_tiers,
)

# English bank + the Italian-arm draft items. The reference dicts must cover every item
# across both banks with no orphans (that's what the coverage tests assert). If the Italian
# items_it.yaml ids change, the cliché refs AND this set must track them together.
ITEMS_EN = {"lantern-01", "tower-01", "wheel-01", "scales-01", "salt-compass-01", "key-01"}
ITEMS_IT = {"bilancia-01", "specchietto-01", "campana-01",
            "bambola-pane-01", "gabbia-sasso-01", "ombrello-cucito-01"}
ITEMS = ITEMS_EN | ITEMS_IT


# --- both reference sets cover exactly the six single-crossing items ---------
def test_cliche_reference_covers_items():
    assert set(CLICHE_REFERENCE) == ITEMS
    assert all(len(v) >= 2 for v in CLICHE_REFERENCE.values())  # multi-phrasing seed


def test_narrative_reference_covers_items_and_is_registered():
    assert set(CLICHE_REFERENCE_NARRATIVE) == ITEMS
    assert all(len(v) >= 2 for v in CLICHE_REFERENCE_NARRATIVE.values())
    assert REFERENCE_SETS == {"statement": CLICHE_REFERENCE, "narrative": CLICHE_REFERENCE_NARRATIVE}


# --- distance metric --------------------------------------------------------
def test_nearest_cliche_distance_zero_when_identical():
    cliche = np.array([[1.0, 0.0], [0.0, 1.0]])
    gen = np.array([[1.0, 0.0]])                     # identical to first cliché
    d = nearest_cliche_distance(gen, cliche)
    assert d[0] == 0.0  # 1 - max cosine (=1)


def test_nearest_cliche_distance_one_when_orthogonal():
    cliche = np.array([[1.0, 0.0]])
    gen = np.array([[0.0, 1.0]])                     # orthogonal
    d = nearest_cliche_distance(gen, cliche)
    assert math.isclose(d[0], 1.0, abs_tol=1e-9)


def test_nearest_takes_closest_not_average():
    # one cliché identical, one orthogonal -> nearest distance is 0 (closest wins)
    cliche = np.array([[1.0, 0.0], [0.0, 1.0]])
    gen = np.array([[1.0, 0.0]])
    assert nearest_cliche_distance(gen, cliche)[0] == 0.0


def test_l2_normalize_handles_zero_rows():
    out = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert not np.isnan(out).any()
    assert math.isclose(np.linalg.norm(out[1]), 1.0)


# --- helpers ----------------------------------------------------------------
def _rec(model, item, gid, dist, orig):
    return DistanceRecord(model=model, item=item, generation_id=gid, distance=dist, judged_originality=orig)


def test_per_model_mean_skips_nan():
    recs = [_rec("m", "i1", "a", 0.5, float("nan")), _rec("m", "i2", "b", 0.7, 4.0)]
    assert per_model_mean(recs, "distance") == {"m": 0.6}
    assert per_model_mean(recs, "judged_originality") == {"m": 4.0}  # NaN dropped


def test_distance_rows_schema_and_nan_to_null():
    recs = [_rec("gpt-5.5", "tower-01", "g1", 0.42, 4.0),
            _rec("gpt-5.5", "tower-01", "g2", 0.55, float("nan"))]
    rows = distance_rows(recs, "openai:te3", "narrative")
    assert rows[0] == {"embedder": "openai:te3", "reference": "narrative", "extractor": None,
                       "generation_id": "g1", "model": "gpt-5.5", "item": "tower-01",
                       "distance": 0.42, "judged_originality": 4.0}
    assert rows[1]["judged_originality"] is None  # NaN → null for clean JSON
    # extractor tag flows through for construal-mode rows
    assert distance_rows(recs, "openai:te3", "statement", extractor="anthropic:x")[0]["extractor"] == "anthropic:x"
    import json as _json
    _json.dumps(rows)  # must be JSON-serialisable


# --- STEP 2: validation kill-switch ----------------------------------------
def test_validation_passes_when_distance_tracks_originality_and_worst_at_floor():
    # distance increases with judged originality; worst model has lowest distance.
    recs = []
    for k, (m, dist, orig) in enumerate([("worst", 0.1, 1.0), ("mid", 0.5, 3.0), ("best", 0.9, 5.0)]):
        for i in range(3):  # 3 items each
            recs.append(_rec(m, f"i{i}", f"{m}{i}", dist, orig))
    v = validate_against_tiers(recs, worst_model="worst")
    assert v.spearman > 0.9
    assert v.worst_in_bottom_half and v.passes


def test_validation_fails_on_coherence_confound():
    # the canary: worst model scores HIGHEST distance (the surface-pilot failure mode).
    recs = []
    for m, dist, orig in [("worst", 0.95, 1.0), ("mid", 0.5, 3.0), ("best", 0.6, 5.0)]:
        for i in range(3):
            recs.append(_rec(m, f"i{i}", f"{m}{i}", dist, orig))
    v = validate_against_tiers(recs, worst_model="worst")
    assert not v.worst_in_bottom_half
    assert not v.passes  # must STOP


# --- STEP 3: frontier + flash control --------------------------------------
def _trio_control_recs(trio_dists, flash_dist, n_items=6, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for m, base in {**trio_dists, "flash": flash_dist}.items():
        for i in range(n_items):
            recs.append(_rec(m, f"i{i}", f"{m}{i}", base + jitter * rng.standard_normal(), 4.0))
    return recs


def test_frontier_branch_A_real_equivalence():
    # trio all equal & clearly above flash -> control resolves, trio does not -> Branch A
    recs = _trio_control_recs({"A": 0.80, "B": 0.80, "C": 0.80}, flash_dist=0.40, jitter=0.01)
    fr = frontier_separation(recs, trio=["A", "B", "C"], control="flash")
    assert fr.control_resolves and not fr.trio_separates
    assert "BRANCH A" in fr.branch()


def test_frontier_branch_B_metric_ceiling():
    # trio ~ flash (no resolved gap anywhere) -> control fails -> Branch B (uninformative)
    recs = _trio_control_recs({"A": 0.50, "B": 0.50, "C": 0.50}, flash_dist=0.50, jitter=0.01)
    fr = frontier_separation(recs, trio=["A", "B", "C"], control="flash")
    assert not fr.control_resolves
    assert "BRANCH B" in fr.branch()


def test_frontier_separation_found():
    # trio genuinely ordered A>B>C and all above flash -> separation found
    recs = _trio_control_recs({"A": 0.90, "B": 0.70, "C": 0.50}, flash_dist=0.20, jitter=0.005)
    fr = frontier_separation(recs, trio=["A", "B", "C"], control="flash")
    assert fr.control_resolves and fr.trio_separates
    assert "SEPARATION FOUND" in fr.branch()


# --- STEP 4: embedder-as-prior guard ---------------------------------------
def test_two_embedder_agreement_high_when_concordant():
    a = {f"g{i}": i / 10 for i in range(10)}
    b = {f"g{i}": i / 10 + 0.01 for i in range(10)}  # same ranking
    assert two_embedder_agreement(a, b) > 0.95


def test_two_embedder_agreement_low_when_discordant():
    a = {f"g{i}": i / 10 for i in range(10)}
    b = {f"g{i}": -i / 10 for i in range(10)}         # reversed ranking
    assert two_embedder_agreement(a, b) < -0.95


# --- STEP 5 signal diagnostics -------------------------------------------------
def _signal_recs():
    # distance increases with judged originality, monotonic across 4 models × 4 items
    recs = []
    for m, dist, orig in [("worst", 0.1, 1.0), ("lo", 0.4, 2.5), ("mid", 0.7, 3.5), ("hi", 0.95, 4.5)]:
        for i in range(4):
            recs.append(_rec(m, f"i{i}", f"{m}{i}", dist + 0.001 * i, orig))
    return recs


def test_generation_level_correlation_detects_monotone_signal():
    glc = generation_level_correlation(_signal_recs())
    assert glc["n"] == 16 and glc["spearman"] > 0.9 and glc["pearson_p"] < 0.01


def test_generation_level_correlation_handles_all_nan():
    recs = [_rec("m", "i", "g", 0.5, float("nan"))]
    assert generation_level_correlation(recs) is None


def test_model_level_spearman_perm_p():
    out = model_level_spearman(_signal_recs(), n_perm=2000)
    assert out["n_models"] == 4 and out["rho"] > 0.9 and 0.0 <= out["perm_p"] <= 1.0


def test_normalize_within_item_zeros_item_mean():
    recs = _signal_recs()
    norm = normalize_within_item(recs)
    # each item's normalised distances should average ~0
    import numpy as _np
    by_item = {}
    for r in norm:
        by_item.setdefault(r.item, []).append(r.distance)
    for vals in by_item.values():
        assert abs(_np.mean(vals)) < 1e-9
    # judged originality is preserved
    assert {r.judged_originality for r in norm} == {1.0, 2.5, 3.5, 4.5}


def test_group_contrast_paired_direction():
    recs = _signal_recs()
    d, p, n = group_contrast(recs, ["hi"], ["worst"])  # hi has larger distance
    assert d > 0 and n == 4


def test_resolution_ladder_orders_by_gap_and_flags_resolution():
    recs = _signal_recs()
    rungs = [("small", ["mid"]), ("large", ["worst"])]
    ladder = resolution_ladder(recs, trio=["hi"], rungs=rungs)
    gaps = {r["rung"]: r["judged_orig_gap"] for r in ladder}
    assert gaps["large"] > gaps["small"] > 0          # worst is a bigger originality gap than mid
    assert all("resolved" in r for r in ladder)
