"""Same-lab and cross-lab judge agreement on the headline panel (paper §4, §5).

ICC(2,1) for each judge pair of the headline absolute panel on data/runs/signal, with
item-cluster bootstrap 95% intervals. No API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calvino.analysis import PRIMARY_DIMS, icc_2_1

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "data" / "runs" / "signal"
JUDGES = ["claude-opus-4-8-api", "claude-sonnet-4-6", "gpt-5.5"]
PAIRS = [
    ("claude-opus-4-8-api", "claude-sonnet-4-6", "same-lab (Anthropic x Anthropic)"),
    ("claude-opus-4-8-api", "gpt-5.5", "cross-lab (Anthropic x OpenAI)"),
    ("claude-sonnet-4-6", "gpt-5.5", "cross-lab (Anthropic x OpenAI)"),
]
SEED = 20260710
N_BOOT = 5000


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def main() -> None:
    gens = {g["id"]: g for g in load(RUN / "generations.jsonl")}
    rows = []
    for s in load(RUN / "judge_scores.jsonl"):
        if s["judge_model"] not in JUDGES:
            continue
        g = gens.get(s["generation_id"])
        if g is None:
            continue
        agg = sum(s["scores"][d] for d in PRIMARY_DIMS) / len(PRIMARY_DIMS)
        rows.append({"gen": s["generation_id"], "item": g["item_id"], "judge": s["judge_model"], "agg": agg})
    df = pd.DataFrame(rows)
    items = sorted(df["item"].unique())
    print(f"bank-1 items (n={len(items)}): {items}")
    print(f"rows: {len(df)}  judges present: {sorted(df['judge'].unique())}")

    print("\n" + "=" * 78)
    print("Point estimates (full sample, complete cases per pair)")
    print("=" * 78)
    wide_full = df.pivot_table(index="gen", columns="judge", values="agg", aggfunc="mean")
    for a, b, label in PAIRS:
        sub = wide_full[[a, b]].dropna()
        icc = icc_2_1(sub.to_numpy())
        print(f"  {label:32s} ({a} x {b}): ICC(2,1)={icc:.4f}  n_gens={sub.shape[0]}")

    print("\n" + "=" * 78)
    print(f"Item-cluster bootstrap (resample {len(items)} items w/ replacement, B={N_BOOT})")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    by_item = {it: df[df["item"] == it] for it in items}

    for a, b, label in PAIRS:
        boots = []
        n_nan = 0
        for _ in range(N_BOOT):
            draw = rng.choice(items, size=len(items), replace=True)
            frames = []
            for k, it in enumerate(draw):
                sub = by_item[it].copy()
                sub["gen"] = sub["gen"] + f"__rep{k}"  # keep duplicated-item rows distinct units
                frames.append(sub)
            resampled = pd.concat(frames, ignore_index=True)
            wide = resampled.pivot_table(index="gen", columns="judge", values="agg", aggfunc="mean")
            if a not in wide.columns or b not in wide.columns:
                n_nan += 1
                continue
            sub2 = wide[[a, b]].dropna()
            icc = icc_2_1(sub2.to_numpy()) if sub2.shape[0] >= 2 else float("nan")
            if np.isnan(icc):
                n_nan += 1
            boots.append(icc)
        boots = np.array(boots, dtype=float)
        valid = boots[~np.isnan(boots)]
        lo, hi = np.percentile(valid, [2.5, 97.5])
        print(f"  {label:32s} ({a} x {b})")
        print(f"    95% CI [{lo:.3f}, {hi:.3f}]   (median {np.median(valid):.3f}, "
              f"{len(valid)}/{N_BOOT} valid draws, {n_nan} nan/degenerate dropped)")

    print("\n" + "=" * 78)
    print("Bonus: PAIRED bootstrap on the seam DIRECTLY (same-lab minus mean of the two")
    print("cross-lab pairs, same resample draw each iteration -- removes shared item-")
    print("resampling noise that inflates each MARGINAL CI above; sharper test of whether")
    print("the seam direction itself is robust to item resampling)")
    print("=" * 78)
    diffs = []
    for _ in range(N_BOOT):
        draw = rng.choice(items, size=len(items), replace=True)
        frames = []
        for k, it in enumerate(draw):
            sub = by_item[it].copy()
            sub["gen"] = sub["gen"] + f"__rep{k}"
            frames.append(sub)
        resampled = pd.concat(frames, ignore_index=True)
        wide = resampled.pivot_table(index="gen", columns="judge", values="agg", aggfunc="mean")
        same = icc_2_1(wide[["claude-opus-4-8-api", "claude-sonnet-4-6"]].dropna().to_numpy())
        c1 = icc_2_1(wide[["claude-opus-4-8-api", "gpt-5.5"]].dropna().to_numpy())
        c2 = icc_2_1(wide[["claude-sonnet-4-6", "gpt-5.5"]].dropna().to_numpy())
        diffs.append(same - np.nanmean([c1, c2]))
    diffs = np.array(diffs, dtype=float)
    valid = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    print(f"  same-lab minus mean(cross-lab): median={np.median(valid):.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]  ({len(valid)}/{N_BOOT} valid, "
          f"{(valid <= 0).mean():.4f} fraction <=0)")


if __name__ == "__main__":
    main()
