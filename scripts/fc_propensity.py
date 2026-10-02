"""Decision propensity in the fable pairwise votes (paper App. G).

Regresses whether a both-orders comparison is decided on judge, bank, opponent, format checks,
output-length gap and the absolute-rubric margin, over the persisted fable-vs-trio votes on both
banks. No API calls.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from calvino.analysis import PRIMARY_DIMS
from calvino.validator import validate_output

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
NEUTRAL = {"gemini-3.1-pro-preview": "Google", "qwen3.7-max": "Alibaba", "grok-4.3": "xAI"}
CLUSTER = ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-5.5"]
FABLE = "claude-fable-5"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def primary_agg(scores: dict) -> float:
    return sum(scores[d] for d in PRIMARY_DIMS) / len(PRIMARY_DIMS)


def gen_lookup(*paths: Path) -> dict[str, dict]:
    out = {}
    for p in paths:
        for g in load(p):
            out.setdefault(g["id"], g)
    return out


def abs_score_lookup(judges: set[str], *paths: Path) -> dict[str, dict[str, float]]:
    """generation_id -> {judge_model: primary_aggregate}, restricted to `judges`."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for p in paths:
        for r in load(p):
            jm = r["judge_model"]
            if jm not in judges:
                continue
            out[r["generation_id"]][jm] = primary_agg(r["scores"])
    return out


def build_bank(bank_id: int, run_dir: str, item_ids: list[str],
                gens: dict[str, dict], abs_scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    votes = load(RUNS / run_dir / "pairwise_votes.jsonl")
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in votes:
        groups[(r["judge_model"], r["comparison_id"])].append(r["winner_model"])

    rows = []
    for (judge, comp), winners in groups.items():
        parts = comp.split("|")
        item_id, model_a, model_b, s = parts
        sample_idx = int(s[1:])
        opp = model_b if model_a == FABLE else model_a
        decided = len(winners) >= 2 and len(set(winners)) == 1

        # Locate the two generation_ids for this comparison via item/model/sample.
        fable_gid = next((gid for gid, g in gens.items()
                           if g["item_id"] == item_id and g["model"] == FABLE
                           and g["sample_idx"] == sample_idx), None)
        opp_gid = next((gid for gid, g in gens.items()
                         if g["item_id"] == item_id and g["model"] == opp
                         and g["sample_idx"] == sample_idx), None)
        if fable_gid is None or opp_gid is None:
            continue  # should not happen; every FC comparison has both generations

        fable_txt = gens[fable_gid]["output_text"]
        opp_txt = gens[opp_gid]["output_text"]
        v_fable = validate_output(fable_txt, gens[fable_gid]["object"])
        v_opp = validate_output(opp_txt, gens[opp_gid]["object"])
        fable_pass = bool(v_fable["count_ok"] and v_fable["split_ok"])
        opp_pass = bool(v_opp["count_ok"] and v_opp["split_ok"])
        if fable_pass and opp_pass:
            gate = "both_pass"
        elif not fable_pass and not opp_pass:
            gate = "both_fail"
        else:
            # Only one side clears the structural gate. Collapsed into a single
            # "mismatch" level (n=12/486: 3 fable_only + 9 opp_only) rather than
            # kept separate -- the two sub-cells are too small on their own and
            # fable_only (n=3, decided=0 always) causes quasi-complete
            # separation (huge unstable coefficient, non-convergence) if left
            # split. gate_detail retains the split for inspection.
            gate = "mismatch"
        gate_detail = (
            "both_pass" if fable_pass and opp_pass else
            "both_fail" if not fable_pass and not opp_pass else
            "fable_only" if fable_pass else "opp_only"
        )

        fscores = abs_scores.get(fable_gid, {})
        oscores = abs_scores.get(opp_gid, {})
        common = set(fscores) & set(oscores)
        if not common:
            margin = float("nan")
        else:
            margin = float(np.mean([fscores[j] for j in common])) - float(np.mean([oscores[j] for j in common]))

        rows.append(dict(
            bank=bank_id, judge=judge, lineage=NEUTRAL[judge], item_id=item_id,
            opponent=opp, sample_idx=sample_idx, decided=int(decided),
            length_gap=len(fable_txt) - len(opp_txt),
            gate=gate, gate_detail=gate_detail, margin=margin,
            favors_fable=int(margin > 0) if not np.isnan(margin) else np.nan,
            abs_margin=abs(margin),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    gens1 = gen_lookup(RUNS / "fable-fc" / "generations.jsonl")
    gens2 = gen_lookup(RUNS / "hard_fable_fc" / "generations.jsonl")

    # Bank 1 absolute margin source: gemini + grok, pooled across signal_fable
    # (fable's scores) and signal_gemini/signal_grok (opponents' scores, same
    # generation_ids as the original "signal" run via the content-hash cache).
    abs1 = abs_score_lookup(
        {"gemini-3.1-pro-preview", "grok-4.3"},
        RUNS / "signal_fable" / "judge_scores.jsonl",
        RUNS / "signal_gemini" / "judge_scores.jsonl",
        RUNS / "signal_grok" / "judge_scores.jsonl",
    )
    # Bank 2: all three neutral judges score every one of the 144 generations directly.
    abs2 = abs_score_lookup(
        {"gemini-3.1-pro-preview", "qwen3.7-max", "grok-4.3"},
        RUNS / "hard_fable_fc" / "judge_scores.jsonl",
    )

    df1 = build_bank(1, "fable-fc", [], gens1, abs1)
    df2 = build_bank(2, "hard_fable_fc", [], gens2, abs2)
    df = pd.concat([df1, df2], ignore_index=True)

    print("=" * 78)
    print(f"Rows (attempted comparisons): {len(df)}  (bank1={len(df1)}, bank2={len(df2)})")
    print(f"Decided: {df['decided'].sum()}  (rate {df['decided'].mean():.3f})")
    print(f"Margin available for: {df['margin'].notna().sum()}/{len(df)} rows")
    print("=" * 78)

    print("\n-- Decision rate by evaluator lineage --")
    print(df.groupby("lineage")["decided"].agg(["mean", "sum", "count"]))
    print("\n-- Decision rate by bank --")
    print(df.groupby("bank")["decided"].agg(["mean", "sum", "count"]))
    print("\n-- Decision rate by lineage x bank --")
    print(df.groupby(["lineage", "bank"])["decided"].agg(["mean", "sum", "count"]))
    print("\n-- Decision rate by opponent --")
    print(df.groupby("opponent")["decided"].agg(["mean", "sum", "count"]))
    print("\n-- Decision rate by structural gate (collapsed: mismatch = exactly one side fails) --")
    print(df.groupby("gate")["decided"].agg(["mean", "sum", "count"]))
    print("\n-- Decision rate by structural gate (uncollapsed detail) --")
    print(df.groupby("gate_detail")["decided"].agg(["mean", "sum", "count"]))
    print("\n-- Decision rate by margin direction (favors_fable) --")
    print(df.groupby("favors_fable")["decided"].agg(["mean", "sum", "count"]))

    d = df.dropna(subset=["margin"]).copy()
    d["bank"] = d["bank"].astype(str)
    print(f"\nRows with complete margin covariate for regression: {len(d)}/{len(df)}")

    print("\n" + "=" * 78)
    print("LOGISTIC REGRESSION: P(decided) ~ lineage + bank + opponent + gate")
    print("                     + length_gap + abs_margin (gap SIZE) + favors_fable (gap DIRECTION)")
    print("cluster-robust SEs, clustered by item_id")
    print("=" * 78)
    formula = ("decided ~ C(lineage, Treatment('Google')) + C(bank, Treatment('1')) "
               "+ C(opponent, Treatment('gpt-5.5')) + C(gate, Treatment('both_pass')) "
               "+ length_gap + abs_margin + favors_fable")
    model = smf.logit(formula, data=d)
    fit = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["item_id"]})
    print(fit.summary())

    print("\n-- Robustness: same model, clustered by judge instead of item --")
    fit_j = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["judge"]})
    print(fit_j.summary().tables[1])

    print("\n-- Margin-only model (no other covariates), clustered by item --")
    m2 = smf.logit("decided ~ abs_margin + favors_fable", data=d)
    fit2 = m2.fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["item_id"]})
    print(fit2.summary().tables[1])

    print("\n-- Signed-margin model (collinear with direction; shown for transparency only) --")
    m3 = smf.logit("decided ~ margin", data=d)
    fit3 = m3.fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["item_id"]})
    print(fit3.summary().tables[1])

    out_path = ROOT / "outputs" / "fc_propensity_frame.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved frame to {out_path}")


if __name__ == "__main__":
    main()
