"""Per-dimension variance explained (paper §4, Table 3, App. B, App. C).

eta^2 per rubric dimension over the full roster and restricted to the original trio, on
judge-score rows of the headline panel, plus between-model F-tests on generation means.
Runs entirely from disk; no API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from calvino import analysis as A
from calvino.analysis import ALL_DIMS, PRIMARY_DIMS, icc_2_1
from calvino.storage import Store

_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA_DIR = _ARGS[0] if _ARGS else "data/runs/signal"
# Canonical 0-5 cross-lab absolute panel (exclude s10 / thinking variants, the
# partial gemini judge, and the local gemma judge).
PANEL = {"claude-opus-4-8-api", "claude-sonnet-4-6", "gpt-5.5"}
TRIO = ["gpt-5.5", "claude-opus-4-8", "claude-sonnet-4-6"]
DIMS = list(PRIMARY_DIMS) + [d for d in ALL_DIMS if d not in PRIMARY_DIMS]
RNG = np.random.default_rng(0)
N_BOOT = 2000


def eta_F(df: pd.DataFrame, dim: str, group: str = "model") -> tuple[float, float, float]:
    groups = [g[dim].to_numpy() for _, g in df.groupby(group) if len(g) > 0]
    if len(groups) < 2:
        return float("nan"), float("nan"), float("nan")
    F, p = stats.f_oneway(*groups)
    grand = df[dim].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((df[dim] - grand) ** 2).sum())
    eta2 = ss_between / ss_total if ss_total else float("nan")
    return eta2, float(F), float(p)


def trio_icc(df: pd.DataFrame, dim: str) -> float:
    """ICC(2,1) across the 3 panel judges, restricted to trio generations."""
    piv = df.pivot_table(index="generation_id", columns="judge_model", values=dim)
    piv = piv.dropna()
    if piv.shape[0] < 2 or piv.shape[1] < 2:
        return float("nan")
    return icc_2_1(piv.to_numpy())


def boot_trio_eta(df: pd.DataFrame, dim: str) -> tuple[float, float]:
    """Cluster bootstrap CI on trio eta^2: resample generations within each model."""
    gens_by_model = {m: df[df.model == m]["generation_id"].unique() for m in TRIO}
    rows_by_gen = {gid: g for gid, g in df.groupby("generation_id")}
    vals = []
    for _ in range(N_BOOT):
        parts = []
        for m in TRIO:
            gids = gens_by_model[m]
            pick = RNG.choice(gids, size=len(gids), replace=True)
            parts.append(pd.concat([rows_by_gen[g] for g in pick], ignore_index=True))
        bs = pd.concat(parts, ignore_index=True)
        e, _, _ = eta_F(bs, dim)
        if not np.isnan(e):
            vals.append(e)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    df = A.build_scores_frame(Store(DATA_DIR))
    df = df[df.judge_model.isin(PANEL) & (df.condition == "normal")].copy()
    # Balanced core, so a re-run reproduces the published figures: (a) sample_idx < 3, the
    # three registered samples per model and item (later samples are only partly judged);
    # (b) complete cells only, dropping any model whose cell is short of the full count.
    # Expected: 11 models / 594 rows (signal) and 8 / 864 (hard). --raw shows the
    # unfiltered frame.
    if "--raw" not in sys.argv:
        df = df[df.sample_idx < 3]
        cell = df.groupby("model").size()
        df = df[df.model.isin(cell[cell == cell.max()].index)].copy()

    df["primary"] = df[list(PRIMARY_DIMS)].mean(axis=1)

    model_primary = df.groupby("model")["primary"].mean().sort_values(ascending=False)
    models_ranked = list(model_primary.index)
    n_models = len(models_ranked)
    n_judges = df.judge_model.nunique()
    print(f"# Dimension discrimination — {DATA_DIR}, {n_models} models, "
          f"{n_judges}-judge cross-lab panel ({sorted(PANEL)})"
          f"{'  [RAW, unbalanced]' if '--raw' in sys.argv else '  [balanced core]'}")
    print(f"# rows={len(df)}  (per model: {len(df)//n_models})\n")

    print("## Model ranking (primary, panel mean)")
    for m, v in model_primary.items():
        tag = "  <- TRIO" if m in TRIO else ""
        print(f"  {v:5.2f}  {m}{tag}")

    # --- (A) global discrimination across the full roster ---
    print("\n## (A) GLOBAL tier discrimination — all models")
    print(f"{'dim':<16}{'eta^2':>8}{'F':>9}{'p':>11}{'rho(rank)':>11}{'spread':>9}")
    dim_means = df.groupby("model")[DIMS].mean()
    rank_primary = model_primary.rank(ascending=False)
    globalrows = []
    for dim in DIMS:
        e, F, p = eta_F(df, dim)
        rho = stats.spearmanr(dim_means[dim].reindex(models_ranked), rank_primary.reindex(models_ranked)).correlation
        spread = dim_means[dim].max() - dim_means[dim].min()
        globalrows.append((dim, e, spread))
        print(f"{dim:<16}{e:>8.3f}{F:>9.1f}{p:>11.2e}{-rho:>11.3f}{spread:>9.2f}")
    print("  (rho = |Spearman(dim model-mean, primary rank)|; spread = max-min model mean)")

    # --- range-located: spread within bottom-3 vs top-3 ---
    print("\n## Where each dimension discriminates (model-mean spread within band)")
    bottom = models_ranked[-3:]
    top = models_ranked[:3]
    print(f"{'dim':<16}{'bottom-3':>10}{'top-3':>9}  (top-3 = the trio)")
    for dim in DIMS:
        sb = dim_means[dim].reindex(bottom).max() - dim_means[dim].reindex(bottom).min()
        st = dim_means[dim].reindex(top).max() - dim_means[dim].reindex(top).min()
        print(f"{dim:<16}{sb:>10.2f}{st:>9.2f}")

    # --- (B) frontier discrimination: trio only, with bootstrap CI + ICC ---
    print("\n## (B) FRONTIER discrimination — trio only (gpt-5.5 / opus / sonnet)")
    tdf = df[df.model.isin(TRIO)].copy()
    print(f"{'dim':<16}{'eta^2':>8}{'  95% CI':>16}{'F':>8}{'p':>9}{'trio ICC':>10}")
    for dim in DIMS:
        e, F, p = eta_F(tdf, dim)
        lo, hi = boot_trio_eta(tdf, dim)
        icc = trio_icc(tdf, dim)
        print(f"{dim:<16}{e:>8.3f}  [{lo:5.3f},{hi:5.3f}]{F:>8.2f}{p:>9.3f}{icc:>10.3f}")
    print("  eta^2 CI excluding ~0 => a real frontier difference on that dim; "
          "spanning 0 => within noise.")


if __name__ == "__main__":
    main()
