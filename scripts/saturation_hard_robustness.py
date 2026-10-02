"""Within-cluster resolution on the hard bank (paper §3, App. B).

Checks whether the original trio separates within itself on the hard items or whether
saturation extends there too, weighting items equally and reporting which items contribute.
No API calls.
"""

from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path

CLUSTER = ["gpt-5.5", "claude-opus-4-8", "claude-sonnet-4-6"]   # also the a-priori priority order
CONSENSUS = {"claude-opus-4-8-api", "claude-sonnet-4-6", "gemini-3.1-pro-preview", "gpt-5.5"}
LITE = "gemini-3.1-flash-lite"   # calibration rung: flagship>>lite must resolve, else a
                                 # within-cluster null is an instrument ceiling, not equivalence
HARD_DIR = "data/runs/hard"
SIGNAL_DIR = "data/runs/signal"

# --- GUARD 3: pre-committed, written before the firmed data exists -----------
SEPARATION_THRESHOLD = 0.75   # within-cluster inter-judge agreement must CLEARLY clear the
                              # ~0.62 coin-flip floor to count as resolution. Fixed now.
PRE_COMMITTED = (
    "EXPECTED: confirms at-floor — saturation HOLDS on the hard stratum. Within-cluster "
    "inter-judge agreement at ~the 0.62 coin-flip floor means NO within-cluster resolution "
    "is recovered on hard. A faint lean in the gpt direction is the ALREADY-KNOWN weak gpt "
    "edge, NOT a new axis (gate-survival was refuted as a substring/paraphrase artifact; "
    "sample-0 judged separation was at-floor). It counts as separation ONLY if per-object "
    f"inter-judge agreement exceeds {SEPARATION_THRESHOLD} AND the per-pair win-rate clears "
    "0.5 decisively — a faint underpowered lean does NOT qualify."
)


def _load(dir_: str) -> list[dict]:
    p = Path(dir_) / "pairwise_votes.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _within_cluster_by_object(rows: list[dict]) -> dict[str, list[list[str]]]:
    """object -> list of comparisons, each a list of consensus judges' winner_models."""
    clset = set(CLUSTER)
    comp: dict[tuple, list[str]] = collections.defaultdict(list)
    for v in rows:
        if v["judge_model"] not in CONSENSUS:
            continue
        a, b = v["model_a"], v["model_b"]
        if a not in clset or b not in clset or not v.get("winner_model"):
            continue
        lo, hi = sorted([a, b])
        comp[(v["item_id"], lo, hi, v["sample_idx"])].append(v["winner_model"])
    by_obj: dict[str, list[list[str]]] = collections.defaultdict(list)
    for (item, lo, hi, s), winners in comp.items():
        by_obj[item].append(winners)
    return by_obj


def _agreement(winners: list[str]) -> float | None:
    """Fraction of judge-pairs agreeing on the winner of one comparison."""
    if len(winners) < 2:
        return None
    pairs = [1.0 if x == y else 0.0 for x, y in itertools.combinations(winners, 2)]
    return sum(pairs) / len(pairs)


def _per_object_metrics(rows: list[dict]) -> dict:
    """GUARD 1: per-object weighting only. Each object contributes one agreement value and
    one set of per-pair ref-win-rates; we then mean across objects (equal object weight)."""
    pri = {m: i for i, m in enumerate(CLUSTER)}
    by_obj = _within_cluster_by_object(rows)

    obj_agreements: list[float] = []
    total_comparisons = 0
    for comparisons in by_obj.values():
        ags = [a for a in (_agreement(w) for w in comparisons) if a is not None]
        if ags:
            obj_agreements.append(sum(ags) / len(ags))
        total_comparisons += len(comparisons)

    # per-pair, per-object ref-win-rate computed directly from rows (the pair identity is on
    # the row, not recoverable from the winners list alone)
    clset = set(CLUSTER)
    pair_obj: dict[tuple, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for v in rows:
        if v["judge_model"] not in CONSENSUS:
            continue
        a, b = v["model_a"], v["model_b"]
        if a not in clset or b not in clset or not v.get("winner_model"):
            continue
        lo, hi = sorted([a, b])
        ref = lo if pri[lo] < pri[hi] else hi
        cell = pair_obj[(lo, hi)][v["item_id"]]
        cell[1] += 1
        if v["winner_model"] == ref:
            cell[0] += 1
    per_pair = {}
    for (lo, hi), objs in pair_obj.items():
        ref = lo if pri[lo] < pri[hi] else hi
        rates = [k / n for k, n in objs.values() if n > 0]
        mean = sum(rates) / len(rates) if rates else None
        per_pair[f"{ref}_wins_vs_{(hi if ref == lo else lo)}"] = {
            "per_object_mean_winrate": round(mean, 3) if mean is not None else None,
            "abs_dev_from_0.5": round(abs(mean - 0.5), 3) if mean is not None else None,
            "n_objects": len(rates),
        }

    agreement = sum(obj_agreements) / len(obj_agreements) if obj_agreements else None
    return {
        "inter_judge_agreement_per_object_mean": round(agreement, 3) if agreement is not None else None,
        "per_pair_winrate": per_pair,
        "_n_objects": len(by_obj),
        "_n_comparisons": total_comparisons,
    }


def _calibration_ladder(rows: list[dict]) -> dict:
    """Instrument-works check: cluster vs LITE, per-object-weighted win rate (should be high)."""
    clset = set(CLUSTER)
    by_obj: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for v in rows:
        if v["judge_model"] not in CONSENSUS or not v.get("winner_model"):
            continue
        a, b = v["model_a"], v["model_b"]
        if not ((a in clset and b == LITE) or (b in clset and a == LITE)):
            continue
        by_obj[v["item_id"]][1] += 1
        if v["winner_model"] != LITE:   # the cluster (flagship) won
            by_obj[v["item_id"]][0] += 1
    rates = [k / n for k, n in by_obj.values() if n > 0]
    mean = sum(rates) / len(rates) if rates else None
    return {"flagship_vs_lite_winrate_per_object_mean": round(mean, 3) if mean is not None else None,
            "n_objects": len(rates)}


def run() -> dict:
    hard = _load(HARD_DIR)
    signal = _load(SIGNAL_DIR)
    hard_m = _per_object_metrics(hard)
    sig_m = _per_object_metrics(signal)
    ladder = _calibration_ladder(hard)

    agree = hard_m["inter_judge_agreement_per_object_mean"]
    instrument_ok = (ladder["flagship_vs_lite_winrate_per_object_mean"] or 0) >= 0.75
    separates = agree is not None and agree > SEPARATION_THRESHOLD and instrument_ok
    if separates:
        verdict = ("SEPARATION on hard — cluster resolves within itself (agreement clears the "
                   "floor with the instrument confirmed live). This OVERTURNS the prior; "
                   "investigate, do not assume the gpt-lean.")
    elif not instrument_ok:
        verdict = ("INCONCLUSIVE — calibration ladder did not confirm flagship>>lite on hard, "
                   "so a within-cluster null could be an instrument ceiling, not equivalence.")
    else:
        verdict = ("CONFIRMS AT-FLOOR — saturation holds on the hard stratum (no within-cluster "
                   "resolution; instrument confirmed live by the lite rung). Robustness win.")

    result = {
        "stratum": "hard",
        "weighting": "per-object (structural; per-generation NOT implemented by design — guard 1)",
        # GUARD 2: composition travels with the result
        "surviving_composition": {
            "hard_objects_contributing": hard_m["_n_objects"],
            "within_cluster_comparisons": hard_m["_n_comparisons"],
            "consensus_judges": sorted(CONSENSUS),
            "note": "underpowered until option 2 (hard samples 1-2) is run; n grows ~3x after",
        },
        "hard_within_cluster": {k: v for k, v in hard_m.items() if not k.startswith("_")},
        "signal_within_cluster_reference": {k: v for k, v in sig_m.items() if not k.startswith("_")},
        "calibration_ladder_instrument_works": ladder,
        "separation_threshold_pre_committed": SEPARATION_THRESHOLD,
        "pre_committed_interpretation": PRE_COMMITTED,   # GUARD 3
        "verdict": verdict,
    }
    return result


if __name__ == "__main__":
    out = run()
    Path(HARD_DIR, "saturation_hard_robustness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
