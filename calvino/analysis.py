"""Analysis layer.

Loads stored data from disk (no API calls) and runs the Phase 1 analyses (brief
§12). Everything starts from one tidy long frame — one row per (generation,
judge) — built by `build_scores_frame`.

Claim discipline is encoded here:
- PRIMARY_DIMS feed the headline aggregate; COMBINATORIAL is exploratory and is
  never folded into `primary_aggregate`.
- Judge agreement (ICC) is reported per dimension so below-threshold dimensions
  can be treated as exploratory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import Generation, JudgeScore
from .storage import GENERATIONS, JUDGE_SCORES, Store

PRIMARY_DIMS = ("distinctness", "bridge", "tonal", "originality")
COMBINATORIAL = "combinatorial"
ALL_DIMS = PRIMARY_DIMS + (COMBINATORIAL,)


def build_scores_frame(store: Store) -> pd.DataFrame:
    """Join generations and judge scores into a tidy long frame: one row per
    (generation, judge), with generation metadata and all five dimension scores.
    """
    gens = {g.id: g for g in store.iter(GENERATIONS, Generation)}
    rows: list[dict] = []
    for s in store.iter(JUDGE_SCORES, JudgeScore):
        g = gens.get(s.generation_id)
        if g is None:
            continue
        row = {
            "generation_id": g.id,
            "item_id": g.item_id,
            "model": g.model,
            "condition": g.condition.value,
            "prompt_variant": g.prompt_variant.value,
            "sample_idx": g.sample_idx,
            "judge_model": s.judge_model,
        }
        for dim in ALL_DIMS:
            row[dim] = getattr(s.scores, dim)
        rows.append(row)
    return pd.DataFrame(rows)


def primary_aggregate(df: pd.DataFrame) -> pd.Series:
    """The headline score per row: mean of the four primary dimensions. Excludes
    Combinatorial Fidelity by design (exploratory, never aggregated)."""
    return df[list(PRIMARY_DIMS)].mean(axis=1)


# --- descriptive means -------------------------------------------------------

def descriptive_means(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-dimension means with n. Descriptive only — see
    `mixed_effects` for uncertainty that respects the nested design."""
    out = df.groupby("model")[list(ALL_DIMS)].mean()
    out["n_scores"] = df.groupby("model").size()
    return out.reset_index()


# --- judge agreement: ICC(2,1) -----------------------------------------------

def icc_2_1(matrix: np.ndarray) -> float:
    """Intraclass correlation ICC(2,1): two-way random effects, single rater,
    absolute agreement (Shrout & Fleiss). `matrix` is units (rows) x raters
    (cols), complete (no NaN). Measures whether judges are interchangeable."""
    data = np.asarray(matrix, dtype=float)
    n, k = data.shape
    if n < 2 or k < 2:
        return float("nan")

    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_total = ((data - grand) ** 2).sum()
    # Degenerate (zero-variance) input yields NaN BY DESIGN. Tolerance rather
    # than == 0: a constant like 4.333... is not exactly representable, so its
    # sums of squares carry ~1e-16 float residue that would otherwise launder a
    # degenerate matrix into ICC = 1.0.
    if ss_total < 1e-9 * max(1.0, abs(grand)):
        return float("nan")
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denom == 0:
        return float("nan")
    return (ms_rows - ms_error) / denom


def judge_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """ICC(2,1) per dimension across judges. Pivots to a units x judges matrix
    per dimension and drops units not scored by every judge (complete cases)."""
    rows = []
    for dim in ALL_DIMS:
        wide = df.pivot_table(
            index="generation_id", columns="judge_model", values=dim, aggfunc="mean"
        ).dropna()
        icc = icc_2_1(wide.to_numpy()) if wide.shape[0] >= 2 and wide.shape[1] >= 2 else float("nan")
        rows.append({"dimension": dim, "icc_2_1": icc, "n_units": wide.shape[0], "n_judges": wide.shape[1]})
    return pd.DataFrame(rows)


# --- combinatorial orthogonality ---------------------------------------------

@dataclass
class Orthogonality:
    r: float
    p: float
    n: int


def combinatorial_orthogonality(df: pd.DataFrame) -> Orthogonality:
    """Correlate Combinatorial Fidelity with the core (Distinctness + Bridge),
    averaged over judges per generation. High r => redundant with the core; low
    r => it carries distinct signal (brief §12, a key Phase 1 output)."""
    from scipy import stats

    per_gen = df.groupby("generation_id")[["distinctness", "bridge", COMBINATORIAL]].mean()
    core = per_gen["distinctness"] + per_gen["bridge"]
    comb = per_gen[COMBINATORIAL]
    if len(per_gen) < 3 or core.nunique() < 2 or comb.nunique() < 2:
        return Orthogonality(r=float("nan"), p=float("nan"), n=len(per_gen))
    r, p = stats.pearsonr(comb, core)
    return Orthogonality(r=float(r), p=float(p), n=len(per_gen))


# --- rank stability ----------------------------------------------------------

@dataclass
class RankStability:
    ranking: list[str]                 # full-data model order, best first
    mean_spearman: float               # mean rank correlation of bootstraps vs full
    rank_freq: dict[str, dict[int, float]]  # model -> {rank: probability}


def _model_scores(df: pd.DataFrame) -> pd.Series:
    agg = df.assign(_p=primary_aggregate(df))
    return agg.groupby("model")["_p"].mean()


def _bootstrap_model_scores(df: pd.DataFrame, *, n_boot: int, seed: int) -> dict[str, list[float]]:
    """Cluster bootstrap over generations, returning each model's resampled mean
    primary-aggregate scores. Groups are precomputed once to avoid O(n^2) masking."""
    gen_ids = df["generation_id"].unique()
    groups = {gid: g for gid, g in df.groupby("generation_id")}
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {m: [] for m in df["model"].unique()}
    for _ in range(n_boot):
        sampled = rng.choice(gen_ids, size=len(gen_ids), replace=True)
        boot = pd.concat([groups[gid] for gid in sampled], ignore_index=True)
        scores = _model_scores(boot)
        for m, v in scores.items():
            out[m].append(v)
    return out


def rank_stability(df: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0) -> RankStability:
    """Bootstrap over generations; re-rank models each resample. Reports the
    full-data ranking, how stable it is (mean Spearman vs full), and each model's
    rank distribution."""
    from scipy import stats

    full = _model_scores(df).sort_values(ascending=False)
    ranking = list(full.index)
    full_rank = {m: i for i, m in enumerate(ranking)}
    models = ranking

    gen_ids = df["generation_id"].unique()
    groups = {gid: g for gid, g in df.groupby("generation_id")}
    rng = np.random.default_rng(seed)
    freq: dict[str, dict[int, int]] = {m: {} for m in models}
    spearmans: list[float] = []

    for _ in range(n_boot):
        sampled = rng.choice(gen_ids, size=len(gen_ids), replace=True)
        # Resample whole generations with replacement (cluster bootstrap).
        boot = pd.concat([groups[gid] for gid in sampled], ignore_index=True)
        scores = _model_scores(boot)
        order = list(scores.sort_values(ascending=False).index)
        for rank, m in enumerate(order):
            freq[m][rank] = freq[m].get(rank, 0) + 1
        if len(models) >= 2:
            full_vec = [full_rank[m] for m in models]
            boot_rank = {m: i for i, m in enumerate(order)}
            boot_vec = [boot_rank.get(m, len(models)) for m in models]
            rho = stats.spearmanr(full_vec, boot_vec).correlation
            if not np.isnan(rho):
                spearmans.append(rho)

    rank_freq = {
        m: {rank: cnt / n_boot for rank, cnt in sorted(d.items())} for m, d in freq.items()
    }
    mean_spearman = float(np.mean(spearmans)) if spearmans else float("nan")
    return RankStability(ranking=ranking, mean_spearman=mean_spearman, rank_freq=rank_freq)


# --- reasoning-depth (thinking) effect ---------------------------------------

def thinking_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Paired thinking-on vs thinking-off comparison, per dimension, for models
    run under both conditions. Pairs by (model, item); returns mean within-pair
    difference (thinking - normal) and a paired t-test p-value."""
    from scipy import stats

    both = df[df["condition"].isin(["normal", "thinking"])]
    models = [
        m for m, g in both.groupby("model")
        if {"normal", "thinking"} <= set(g["condition"].unique())
    ]
    sub = both[both["model"].isin(models)]
    rows = []
    for dim in ALL_DIMS:
        cell = sub.groupby(["model", "item_id", "condition"])[dim].mean().unstack("condition")
        # No paired data (no model run under both conditions): report empty, not crash.
        if {"normal", "thinking"} <= set(cell.columns):
            cell = cell.dropna(subset=["normal", "thinking"])
            diffs = cell["thinking"] - cell["normal"]
        else:
            diffs = pd.Series(dtype=float)
        if len(diffs) >= 2 and diffs.nunique() > 1:
            t = stats.ttest_rel(cell["thinking"], cell["normal"])
            p = float(t.pvalue)
        else:
            p = float("nan")
        rows.append({
            "dimension": dim,
            "mean_diff": float(diffs.mean()) if len(diffs) else float("nan"),
            "n_pairs": int(len(diffs)),
            "p_value": p,
        })
    return pd.DataFrame(rows)


# --- frontier separation / saturation ----------------------------------------

@dataclass
class FrontierResult:
    table: pd.DataFrame          # model, mean, ci_low, ci_high (primary aggregate)
    top_saturated: bool          # do the top two models' CIs overlap?


def frontier_separation(df: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0) -> FrontierResult:
    """Per-model primary-aggregate mean with a bootstrap CI; flags saturation if
    the top two models' CIs overlap (expected at the frontier on one crossing)."""
    full = _model_scores(df).sort_values(ascending=False)
    boots = _bootstrap_model_scores(df, n_boot=n_boot, seed=seed)

    rows = []
    for m in full.index:
        arr = np.array(boots[m]) if boots[m] else np.array([full[m]])
        rows.append({
            "model": m,
            "mean": float(full[m]),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        })
    table = pd.DataFrame(rows)

    saturated = False
    if len(table) >= 2:
        top, second = table.iloc[0], table.iloc[1]
        saturated = top["ci_low"] <= second["ci_high"]
    return FrontierResult(table=table, top_saturated=bool(saturated))


# --- mixed-effects model -----------------------------------------------------

def mixed_effects(df: pd.DataFrame, dimension: str):
    """Fit score ~ model with crossed random intercepts for item and judge, for
    one dimension. Returns a tidy DataFrame of estimated per-model means with
    standard errors. Falls back to None if the fit does not converge.

    Crossed random effects are expressed via variance components on a single
    constant group (the standard statsmodels idiom)."""
    import statsmodels.formula.api as smf

    d = df[["model", "item_id", "judge_model", dimension]].copy()
    d = d.rename(columns={dimension: "score"})
    d["grp"] = 1
    try:
        model = smf.mixedlm(
            "score ~ C(model)",
            data=d,
            groups="grp",
            vc_formula={"item": "0 + C(item_id)", "judge": "0 + C(judge_model)"},
        )
        fit = model.fit(reml=False, method="lbfgs")
    except Exception:
        return None

    intercept = fit.params.get("Intercept", float("nan"))
    rows = []
    models = sorted(d["model"].unique())
    for m in models:
        key = f"C(model)[T.{m}]"
        est = intercept + (fit.params[key] if key in fit.params else 0.0)
        se = fit.bse[key] if key in fit.bse else fit.bse.get("Intercept", float("nan"))
        rows.append({"dimension": dimension, "model": m, "estimate": float(est), "se": float(se)})
    return pd.DataFrame(rows)


# --- failure-mode catalogue --------------------------------------------------

def failure_catalogue(df: pd.DataFrame, *, n: int = 5) -> dict[str, pd.DataFrame]:
    """Collect the worst outputs (lowest primary aggregate) and the most-disagreed
    outputs (highest spread across judges). Both keyed by generation_id."""
    g = df.assign(_p=primary_aggregate(df))
    per_gen = g.groupby(["generation_id", "model", "item_id"])["_p"]
    worst = per_gen.mean().sort_values().head(n).reset_index().rename(columns={"_p": "primary_mean"})
    disagreed = (
        per_gen.std().sort_values(ascending=False).head(n).reset_index().rename(columns={"_p": "primary_std"})
    )
    return {"worst": worst, "most_disagreed": disagreed}
