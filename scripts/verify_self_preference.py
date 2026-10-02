#!/usr/bin/env python3
"""Same-lab score bonus (paper §5, App. C).

Fits primary ~ judge + model + same_lab, with standard errors clustered on item, over the judges
that scored the 198 signal generations. Prints the pooled bonus, per-judge bonuses and the
near-ceiling restriction. load_scores() is the headline-panel loader other scripts reuse
(judges claude-opus-4-8-api, claude-sonnet-4-6, gpt-5.5; sample_idx < 3). No API calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PRIMARY = ["distinctness", "bridge", "tonal", "originality"]
RUNS = Path("data/runs")

# The bank-1 absolute panel at the canonical 0-5 scale. The signal directory also holds judge
# aliases for the 0-10 arm ("-s10") and the reasoning-mode arm ("-thinking"); those are separate
# experiments (App. "The Other Arms") and are excluded here.
BASE_PANEL = ["claude-opus-4-8-api", "claude-sonnet-4-6", "gpt-5.5"]

# Contest-disjoint judges that scored the same 198 generations, each in its own run directory.
NEUTRAL_DIRS = {
    "gemini-3.1-pro-preview": "signal_gemini",
    "grok-4.3": "signal_grok",
    "gemini-3.6-flash": "sibling-seam-google",
    "grok-4.5": "sibling-seam-xai",
}
# Rulers for the sensitivity pass, kept as separate lineages on purpose.
RULERS = {
    "xAI (grok-4.3, grok-4.5)": ["grok-4.3", "grok-4.5"],
    "Google (gemini-3.1-pro, -3.6-flash)": ["gemini-3.1-pro-preview", "gemini-3.6-flash"],
    "all four neutral judges": list(NEUTRAL_DIRS),
}
FRONTIER = ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-5.5"]

CORE_SAMPLES = 3   # the complete 18-generation core every model has (fig:ranking note)
N_BOOT = 5_000
SEED = 0


def family(name: str) -> str:
    """Provider lineage of a model or judge alias.

    Every vendor gets its OWN bucket. An earlier version routed unmatched names to a shared
    "other" bucket, which made the xAI judges look like same-family raters of gpt-oss:20b and
    manufactured a spurious +0.68 own-family bonus for them.
    """
    n = name.lower()
    for suffix in ("-api", "-thinking", "-s10"):
        n = n.replace(suffix, "")
    if n.startswith("claude"):
        return "anthropic"
    if n.startswith("gpt-oss"):
        return "openweight"        # open-weight release, not OpenAI lineage, and judged by nobody
    if n.startswith("gpt"):
        return "openai"
    if n.startswith("gemini") or n.startswith("gemma"):
        return "google"
    if n.startswith("llama"):
        return "meta"
    if n.startswith("grok"):
        return "xai"
    if n.startswith("qwen"):
        return "alibaba"
    return "unknown"


def _flatten(path: Path) -> pd.DataFrame:
    s = pd.read_json(path, lines=True)
    dims = pd.DataFrame(list(s["scores"]))
    s = pd.concat(
        [s.drop(columns=["scores"]).reset_index(drop=True), dims.reset_index(drop=True)], axis=1
    )
    s["primary"] = s[PRIMARY].mean(axis=1)
    return s[["generation_id", "judge_model", "primary"]]


def load_scores() -> pd.DataFrame:
    """The seven canonical 0-5 judges on bank 1, one row per (generation, judge).

    Two things this has to get right. The "-s10" and "-thinking" aliases in signal/ are separate
    instruments (App. "The Other Arms") and are dropped, not pooled. And signal/ already holds a
    partial gemini pass (103 rows) that signal_gemini/ supersedes at full coverage, so rows are
    de-duplicated on (generation, judge) with the dedicated run kept.
    """
    gens = pd.read_json(RUNS / "signal" / "generations.jsonl", lines=True)
    frames = [_flatten(RUNS / d / "judge_scores.jsonl") for d in NEUTRAL_DIRS.values()]
    frames.append(_flatten(RUNS / "signal" / "judge_scores.jsonl"))
    sc = pd.concat(frames)
    sc = sc[sc.judge_model.isin(BASE_PANEL + list(NEUTRAL_DIRS))]
    sc = sc.drop_duplicates(subset=["generation_id", "judge_model"], keep="first")
    sc = sc.merge(
        gens[["id", "model", "sample_idx", "item_id"]], left_on="generation_id", right_on="id"
    )
    sc = sc[sc.sample_idx < CORE_SAMPLES].copy()
    sc["mfam"] = sc.model.map(family)
    return sc


def _boot_diff(a: dict, b: dict) -> tuple[float, float]:
    """95% CI for mean(a) - mean(b), resampling ITEMS with replacement on a common draw."""
    rng = np.random.default_rng(SEED)
    items = sorted(set(a) | set(b))
    out = []
    for _ in range(N_BOOT):
        drawn = [items[i] for i in rng.integers(0, len(items), len(items))]
        A = np.concatenate([a[i] for i in drawn if i in a])
        B = np.concatenate([b[i] for i in drawn if i in b])
        out.append(A.mean() - B.mean())
    return tuple(np.percentile(out, [2.5, 97.5]))


def _by_item(sub: pd.DataFrame, judge: str, ruler: list[str]) -> tuple[dict, float]:
    d = (sub[judge] - sub[ruler].mean(axis=1)).dropna()
    it = sub.loc[d.index, "item_id"]
    return {i: d[it == i].to_numpy() for i in it.unique()}, d.mean()


def verdict(lo: float, hi: float) -> str:
    return "SELF-PREFERS" if lo > 0 else ("NEGATIVE" if hi < 0 else "null")


def fixed_effects(sc: pd.DataFrame) -> pd.DataFrame:
    """PRIMARY estimator: own-family bonus with judge severity and model quality absorbed."""
    d = sc.rename(columns={"judge_model": "J", "model": "M"}).copy()
    d["own"] = (d.J.map(family) == d.M.map(family)).astype(int)
    d["ownj"] = np.where(d.own == 1, d.J, "none")
    pooled = smf.ols("primary ~ C(J) + C(M) + own", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d.item_id}
    )
    fit = smf.ols(
        "primary ~ C(J) + C(M) + C(ownj, Treatment(reference='none'))", data=d
    ).fit(cov_type="cluster", cov_kwds={"groups": d.item_id})
    ci = fit.conf_int()
    rows = []
    for p in fit.params.index:
        if "ownj" not in p:
            continue
        judge = p.split("T.")[1].rstrip("]")
        lo, hi = ci.loc[p, 0], ci.loc[p, 1]
        rows.append(
            {
                "judge": judge,
                "family": family(judge),
                "n_own": int(((d.J == judge) & (d.own == 1)).sum()),
                "bonus": fit.params[p],
                "lo": lo,
                "hi": hi,
                "p": fit.pvalues[p],
                "verdict": verdict(lo, hi),
            }
        )
    out = pd.DataFrame(rows).sort_values("bonus", ascending=False)
    out.attrs["pooled"] = (
        pooled.params["own"],
        pooled.conf_int().loc["own", 0],
        pooled.conf_int().loc["own", 1],
        pooled.pvalues["own"],
    )
    out.attrs["n_obs"] = len(d)
    return out


def severity(wide: pd.DataFrame) -> pd.DataFrame:
    """Why the round-2 estimator fails: each panel judge's global level vs the neutral four."""
    neu = list(NEUTRAL_DIRS)
    rows = []
    for j in BASE_PANEL:
        sub = wide[wide.mfam != family(j)].dropna(subset=[j] + neu)   # rival + unrelated models
        rows.append({"judge": j, "severity_vs_neutral": (sub[j] - sub[neu].mean(axis=1)).mean()})
    return pd.DataFrame(rows)


def frontier_sensitivity(wide: pd.DataFrame) -> pd.DataFrame:
    """SENSITIVITY: level-matched own-vs-rival margin, one row per (judge, neutral ruler)."""
    w = wide[wide.model.isin(FRONTIER)]
    rows = []
    for label, ruler in RULERS.items():
        for j in BASE_PANEL:
            f = family(j)
            sub = w.dropna(subset=[j] + ruler)
            own, rival = sub[sub.mfam == f], sub[sub.mfam != f]
            if own.empty or rival.empty:
                continue
            go, mo = _by_item(own, j, ruler)
            gr, mr = _by_item(rival, j, ruler)
            lo, hi = _boot_diff(go, gr)
            rows.append(
                {
                    "ruler": label,
                    "judge": j,
                    "own": mo,
                    "rival": mr,
                    "bonus": mo - mr,
                    "lo": lo,
                    "hi": hi,
                    "verdict": verdict(lo, hi),
                }
            )
    return pd.DataFrame(rows)


def ruler_scalar(wide: pd.DataFrame, fs: pd.DataFrame) -> pd.DataFrame:
    """How many of the three rulers are independent? Two.

    The sensitivity estimator is [J(own) - J(rival)] - [R(own) - R(rival)], so on the near-ceiling
    three the ruler enters only through D_R = R(Anthropic pair) - R(gpt-5.5). Changing rulers must
    then move the two Anthropic judges by -dD and gpt-5.5 by +dD exactly. This checks that, and
    reports the residual: a non-zero residual would mean the ruler is doing something the scalar
    account does not cover (e.g. unequal item coverage between rulers).
    """
    w = wide[wide.model.isin(FRONTIER)]
    rows = []
    for label, ruler in RULERS.items():
        sub = w.dropna(subset=ruler)
        r = sub[ruler].mean(axis=1)
        a = r[sub.mfam == "anthropic"].mean()
        g = r[sub.mfam == "openai"].mean()
        rows.append({"ruler": label, "R_anthropic": a, "R_gpt": g, "D_R": a - g})
    out = pd.DataFrame(rows)
    piv = fs.pivot(index="judge", columns="ruler", values="bonus")
    lin = [c for c in out.ruler if not c.startswith("all four")]
    dD = out.set_index("ruler").D_R[lin[1]] - out.set_index("ruler").D_R[lin[0]]
    resid = {
        j: (piv.loc[j, lin[1]] - piv.loc[j, lin[0]]) - (dD if family(j) == "openai" else -dD)
        for j in piv.index
    }
    pool = out.ruler[out.ruler.str.startswith("all four")].iloc[0]
    out.attrs["dD"] = dD
    out.attrs["resid"] = resid
    out.attrs["midpoint_err"] = (
        out.set_index("ruler").D_R[pool] - out.set_index("ruler").D_R[lin].mean()
    )
    return out


def round2_estimator(wide: pd.DataFrame) -> pd.DataFrame:
    """The superseded estimator, kept reproducible because the correction record cites it."""
    rows = []
    for j in BASE_PANEL:
        f = family(j)
        others = [c for c in BASE_PANEL if c != j]
        sub = wide[(wide.mfam == f)].dropna(subset=[j] + others)
        g, m = _by_item(sub, j, others)
        rng = np.random.default_rng(SEED)
        items = list(g)
        draws = [
            np.concatenate([g[items[k]] for k in rng.integers(0, len(items), len(items))]).mean()
            for _ in range(N_BOOT)
        ]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"judge": j, "bonus": m, "lo": lo, "hi": hi, "verdict": verdict(lo, hi)})
    return pd.DataFrame(rows)


def icc21(df: pd.DataFrame) -> float:
    """ICC(2,1): two-way random, absolute agreement, single rater (Shrout & Fleiss 1979)."""
    x = df.to_numpy(dtype=float)
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    mse = (
        (x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2
    ).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def seam_without_own_lineage(wide: pd.DataFrame) -> pd.DataFrame:
    """Is the same-lab agreement seam an artifact of one judge's leniency to its own lineage?

    If it were, dropping the judges' own-lineage contestants would remove it. It does not.
    """
    pairs = {
        "same-lab (opus, sonnet)": ["claude-opus-4-8-api", "claude-sonnet-4-6"],
        "cross-lab (opus, gpt-5.5)": ["claude-opus-4-8-api", "gpt-5.5"],
        "cross-lab (sonnet, gpt-5.5)": ["claude-sonnet-4-6", "gpt-5.5"],
    }
    rows = []
    for tag, drop in (("all models", False), ("excluding Anthropic contestants", True)):
        got = {}
        for lab, cols in pairs.items():
            sub = wide.dropna(subset=cols)
            if drop:
                sub = sub[sub.mfam != "anthropic"]
            got[lab] = (icc21(sub[cols]), len(sub))
        same = got["same-lab (opus, sonnet)"][0]
        cross = np.mean([v[0] for k, v in got.items() if k.startswith("cross")])
        rows.append({"sample": tag, "same_lab": same, "cross_lab_mean": cross,
                     "seam": same - cross, "n": got["same-lab (opus, sonnet)"][1]})
    return pd.DataFrame(rows)


def fable_row_check(wide: pd.DataFrame) -> dict:
    """Does the lineage-lenient judge account for fable's absolute margin over gpt-5.5?

    fig:ranking's fable row rests on 4 judges, gpt-5.5's on 3. Recompute both with the
    lineage-lenient judge (sonnet) dropped and see whether the margin moves.
    """
    fab = _flatten(RUNS / "signal_fable" / "judge_scores.jsonl")
    panel = BASE_PANEL + ["gemini-3.1-pro-preview"]
    fw = fab[fab.judge_model.isin(panel)].pivot_table(
        index="generation_id", columns="judge_model", values="primary"
    )
    gw = wide[wide.model == "gpt-5.5"].set_index("generation_id")[BASE_PANEL]
    keep_f = [c for c in fw.columns if c != "claude-sonnet-4-6"]
    keep_g = [c for c in BASE_PANEL if c != "claude-sonnet-4-6"]
    full = fw.mean(axis=1).mean() - gw.mean(axis=1).mean()
    dropped = fw[keep_f].mean(axis=1).mean() - gw[keep_g].mean(axis=1).mean()
    return {"margin_full_panel": full, "margin_sonnet_dropped": dropped, "shift": dropped - full}


def own_lab_removal(wide: pd.DataFrame) -> pd.DataFrame:
    """For the correction record: the round-1 claim that the absolute ordering 'evaporates'
    when each model is scored only by non-own-lab judges. It does not; the gaps widen. But the
    contrast is uninformative, because each Anthropic model's de-confounded score then rests on
    gpt-5.5 alone, the panel's most severe judge (see severity()).
    """
    rows = []
    for m in FRONTIER:
        sub = wide[wide.model == m].dropna(subset=BASE_PANEL)
        non_own = [c for c in BASE_PANEL if family(c) != family(m)]
        rows.append(
            {
                "model": m,
                "full_panel": sub[BASE_PANEL].mean(axis=1).mean(),
                "non_own_lab": sub[non_own].mean(axis=1).mean(),
                "judges_left": ", ".join(c.replace("-api", "") for c in non_own),
            }
        )
    return pd.DataFrame(rows)


def roster_subsets(sc: pd.DataFrame) -> pd.DataFrame:
    """Where on the roster does the own-family bonus live?

    Asked because a reviewer proposed that saturation suppresses the bias, which would explain
    why removing a lenient judge does not move the top-cluster margin. It does not hold: the
    bonus is positive and separates from zero on the three near-ceiling models themselves.
    The median split disagrees with the top-cluster cut because it changes WHICH JUDGES
    contribute an own-family cell (it admits the Google pair, whose second judge is negative),
    not only the score range. Both are reported.
    """
    d = sc.rename(columns={"judge_model": "J", "model": "M"}).copy()
    d["own"] = (d.J.map(family) == d.M.map(family)).astype(int)
    level = (
        sc[sc.judge_model.isin(NEUTRAL_DIRS)].groupby("model").primary.mean().sort_values(ascending=False)
    )
    med = level.median()
    cuts = {
        "all models": d,
        "top cluster (3 near-ceiling)": d[d.M.isin(FRONTIER)],
        "below the top cluster": d[~d.M.isin(FRONTIER)],
        f"upper half (level >= {med:.2f})": d[d.M.isin(level[level >= med].index)],
        f"lower half (level < {med:.2f})": d[d.M.isin(level[level < med].index)],
    }
    rows = []
    for label, sub in cuts.items():
        if sub.M.nunique() < 2 or sub.own.nunique() < 2:
            continue
        fit = smf.ols("primary ~ C(J) + C(M) + own", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub.item_id}
        )
        ci = fit.conf_int()
        rows.append(
            {
                "cut": label,
                "n_models": sub.M.nunique(),
                "bonus": fit.params["own"],
                "lo": ci.loc["own", 0],
                "hi": ci.loc["own", 1],
                "verdict": verdict(ci.loc["own", 0], ci.loc["own", 1]),
            }
        )
    return pd.DataFrame(rows)


def margin_diagnostic(sc: pd.DataFrame) -> dict:
    """Why the drop-one test on Figure 1's newcomer margin cancels.

    NOT because the lenient judge treats both rows alike -- it favours fable by +0.028 more than
    its panel does. The drop-one shifts cancel because the two rows rest on different numbers of
    judges: a deviation of d on a k-judge mean moves it by d/(k-1), and 0.083/3 = 0.056/2.
    """
    lenient = "claude-sonnet-4-6"
    fab = _flatten(RUNS / "signal_fable" / "judge_scores.jsonl")
    panel = BASE_PANEL + ["gemini-3.1-pro-preview"]
    fw = fab[fab.judge_model.isin(panel)].pivot_table(
        index="generation_id", columns="judge_model", values="primary"
    )
    gw = sc[sc.model == "gpt-5.5"].pivot_table(
        index="generation_id", columns="judge_model", values="primary"
    )[BASE_PANEL]
    fm, gm = fw.mean(axis=1).mean(), gw.mean(axis=1).mean()
    fs, gs = fw[lenient].mean(), gw[lenient].mean()
    return {
        "panel_margin": fm - gm,
        "lenient_margin": fs - gs,
        "extra_favour": (fs - gs) - (fm - gm),
        "dev_fable": fs - fm,
        "k_fable": len(fw.columns),
        "dev_gpt": gs - gm,
        "k_gpt": len(gw.columns),
        "margin_after_drop": (fm - (fs - fm) / (len(fw.columns) - 1))
        - (gm - (gs - gm) / (len(gw.columns) - 1)),
    }


def per_judge_subset(sc: pd.DataFrame, models: list[str]) -> dict:
    """Per-judge own-family bonuses refit on a model subset, for the row-level propagation.

    The pooled bonus is the wrong multiplier for a specific row: fable's two own-family judges
    are opus and sonnet, which happen to be the panel's two most self-preferring, while
    gpt-5.5's single own-family judge is itself. Using the pooled figure understates the
    differential. Returns {judge: bonus}; judges with no own-family cell in the subset are absent.
    """
    d = sc[sc.model.isin(models)].rename(columns={"judge_model": "J", "model": "M"}).copy()
    d["own"] = (d.J.map(family) == d.M.map(family)).astype(int)
    d["ownj"] = np.where(d.own == 1, d.J, "none")
    fit = smf.ols(
        "primary ~ C(J) + C(M) + C(ownj, Treatment(reference='none'))", data=d
    ).fit(cov_type="cluster", cov_kwds={"groups": d.item_id})
    return {
        p.split("T.")[1].rstrip("]"): fit.params[p] for p in fit.params.index if "ownj" in p
    }


def row_level_inflation(sc: pd.DataFrame, bonus: float, per_judge: dict | None = None) -> pd.DataFrame:
    """Propagate a per-judge own-family bonus through each fig:ranking row's panel composition.

    A row mean over k judges of which m share the model's lineage is inflated by (m/k) * bonus.
    The rows rest on unequal panels (fable's on four judges, two Anthropic; the rest on the
    three-judge base panel, two Anthropic), so the inflation is NOT common and does not cancel
    out of the gaps. Called with the near-ceiling bonus, which is the level the frontier rows
    are read at; applying it to fable extrapolates, since fable is not in the fitted subset.

    With per_judge supplied, each own-family judge contributes its OWN estimated bonus instead of
    the pooled one. That is the less flattering reading and it is reported alongside the pooled.
    """
    fab = _flatten(RUNS / "signal_fable" / "judge_scores.jsonl")
    panels = {
        "claude-fable-5": (BASE_PANEL + ["gemini-3.1-pro-preview"], fab),
        "gpt-5.5": (BASE_PANEL, None),
        "claude-opus-4-8": (BASE_PANEL, None),
        "claude-sonnet-4-6": (BASE_PANEL, None),
    }
    rows = []
    for m, (panel, src) in panels.items():
        if src is None:
            w = sc[sc.model == m].pivot_table(
                index="generation_id", columns="judge_model", values="primary"
            )
        else:
            w = src[src.judge_model.isin(panel)].pivot_table(
                index="generation_id", columns="judge_model", values="primary"
            )
        cols = [c for c in panel if c in w.columns]
        own = [c for c in cols if family(c) == family(m)]
        share = len(own) / len(cols)
        mean = w[cols].mean(axis=1).mean()
        pj = (
            sum(per_judge.get(c, 0.0) for c in own) / len(cols)
            if per_judge is not None
            else np.nan
        )
        rows.append(
            {
                "model": m,
                "k": len(cols),
                "own_judges": len(own),
                "share": share,
                "published": mean,
                "inflation": share * bonus,
                "adjusted": mean - share * bonus,
                "inflation_pj": pj,
                "adjusted_pj": mean - pj,
            }
        )
    out = pd.DataFrame(rows)
    out["gap_to_next"] = out.published.diff(-1)
    out["adj_gap_to_next"] = out.adjusted.diff(-1)
    out["adj_gap_pj"] = out.adjusted_pj.diff(-1)
    return out


def forced_choice(contestant: str = "gemini-3.1-pro-preview") -> pd.DataFrame:
    votes = pd.read_json(RUNS / "signal" / "pairwise_votes.jsonl", lines=True)
    rel = votes[(votes.model_a == contestant) | (votes.model_b == contestant)]
    fam = family(contestant)
    rows = []
    for judge, grp in rel.groupby("judge_model"):
        wins = int((grp.winner_model == contestant).sum())
        rows.append(
            {
                "judge": judge,
                "own_lineage": family(judge) == fam,
                "in_base_panel": judge in BASE_PANEL,
                "wins": wins,
                "n": len(grp),
                "rate": wins / len(grp),
            }
        )
    return pd.DataFrame(rows).sort_values("rate")


def main() -> int:
    if not (RUNS / "signal").exists():
        print(f"missing {RUNS / 'signal'}", file=sys.stderr)
        return 1
    sc = load_scores()
    wide = sc.pivot_table(
        index=["item_id", "generation_id", "model", "mfam"], columns="judge_model", values="primary"
    ).reset_index()

    print("PRIMARY: own-family bonus, judge severity and model quality absorbed")
    fe = fixed_effects(sc)
    b, lo, hi, p = fe.attrs["pooled"]
    print(f"  {fe.attrs['n_obs']} judge-generation rows, 7 judges, item-clustered SEs")
    print(f"  pooled over judges           {b:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}")
    for r in fe.itertuples():
        print(
            f"  {r.judge:24s} {r.family:9s} {r.bonus:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]  "
            f"n_own={r.n_own:3d}  p={r.p:.3f}  {r.verdict}"
        )
    print("  sibling splits (same lineage, both judges measured):")
    for a, c in (("claude-sonnet-4-6", "claude-opus-4-8-api"),
                 ("gemini-3.1-pro-preview", "gemini-3.6-flash")):
        ra = fe[fe.judge == a].iloc[0]
        rc = fe[fe.judge == c].iloc[0]
        print(f"    {a} {ra.bonus:+.3f} vs {c} {rc.bonus:+.3f}"
              f"   intervals {'DISJOINT' if ra.lo > rc.hi else 'overlap'}")

    print("\nWHY THE SUPERSEDED ESTIMATOR FAILED: panel judges differ in global severity")
    sev = severity(wide)
    for r in sev.itertuples():
        print(f"  {r.judge:24s} {r.severity_vs_neutral:+.3f} vs the neutral four, on rival models")
    print(f"  severity spread {sev.severity_vs_neutral.max() - sev.severity_vs_neutral.min():.3f}"
          f"  (comparable to the own-family bonuses above)")
    print("  superseded estimator (baseline = the other two PANEL judges), for the record:")
    for r in round2_estimator(wide).itertuples():
        print(f"    {r.judge:22s} {r.bonus:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]  {r.verdict}")

    print("\nSENSITIVITY: frontier-restricted own-vs-rival, by neutral ruler")
    fs = frontier_sensitivity(wide)
    for ruler, grp in fs.groupby("ruler", sort=False):
        print(f"  ruler = {ruler}")
        for r in grp.itertuples():
            print(f"    {r.judge:22s} own {r.own:+.3f} rival {r.rival:+.3f}  "
                  f"bonus {r.bonus:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]  {r.verdict}")

    print("  -- how many of these rulers are independent? TWO --")
    rsc = ruler_scalar(wide, fs)
    for r in rsc.itertuples():
        print(f"    {r.ruler:38s} R(anthropic) {r.R_anthropic:.4f}  R(gpt) {r.R_gpt:.4f}  "
              f"D_R {r.D_R:+.4f}")
    print(f"    dD between the two lineage rulers {rsc.attrs['dD']:+.4f}; "
          f"pool minus midpoint of the two {rsc.attrs['midpoint_err']:+.5f}")
    print("    scalar account residual per judge: "
          + ", ".join(f"{k.replace('-api','')} {v:+.5f}" for k, v in rsc.attrs["resid"].items()))

    print("\nFORCED CHOICE (gemini-3.1-pro as contestant)")
    fc = forced_choice()
    base = fc[fc.in_base_panel]
    for r in fc.itertuples():
        tag = "SELF" if r.own_lineage else ("base panel" if r.in_base_panel else "off-panel arm")
        print(f"  {r.judge:26s} {r.wins:3d}/{r.n:3d} = {r.rate:.3f}   {tag}")
    off = sorted(fc[~fc.in_base_panel & ~fc.own_lineage].rate)
    print(f"  base-panel range {base.rate.min():.2f}-{base.rate.max():.2f}; "
          f"off-panel arms {', '.join(f'{x:.3f}' for x in off)}")

    print("\nWHERE ON THE ROSTER THE BONUS LIVES (is it a ceiling artifact? no)")
    rs = roster_subsets(sc)
    for r in rs.itertuples():
        print(f"  {r.cut:34s} {r.bonus:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]  "
              f"models={r.n_models:2d}  {r.verdict}")

    top = rs[rs.cut.str.startswith("top cluster")].iloc[0]
    pj_top = per_judge_subset(sc, FRONTIER)
    pj_all = per_judge_subset(sc, sorted(sc.model.unique()))
    print(f"\nROW-LEVEL EFFECT of the near-ceiling bonus ({top.bonus:+.3f}) on fig:ranking")
    print("  per-judge own-family bonuses, near-ceiling refit: "
          + ", ".join(f"{k.replace('-api','')} {v:+.3f}" for k, v in sorted(pj_top.items())))
    # The level-matched sensitivity's two SINGLE-LINEAGE rulers, as multipliers in their own right.
    # Its third RULERS entry ("all four neutral judges") is excluded here deliberately: it
    # reproduces the near-ceiling refit to three decimals, judge for judge, so it is the same
    # quantity computed twice and cannot corroborate it. The two independent rulers DISAGREE on the
    # judge ordering, which is what decides the answer -- hence the sign is not identified either.
    ruler_pjs = [
        (f"level-matched, ruler = {label.split(' (')[0]}", dict(zip(g.judge, g.bonus)))
        for label, g in fs.groupby("ruler")
        if not label.startswith("all four")
    ]
    diffs = {}
    for label, pj in [("near-ceiling per-judge", pj_top),
                      ("full-roster per-judge", pj_all), *ruler_pjs]:
        ri = row_level_inflation(sc, top.bonus, pj)
        print(f"  -- multiplier: {label} --")
        for r in ri.itertuples():
            gap = "" if pd.isna(r.gap_to_next) else (
                f"   gap {r.gap_to_next:+.3f} -> pooled {r.adj_gap_to_next:+.3f}"
                f" / per-judge {r.adj_gap_pj:+.3f}")
            print(f"    {r.model:20s} {r.own_judges}/{r.k} own  "
                  f"infl pooled {r.inflation:+.3f} per-judge {r.inflation_pj:+.3f}{gap}")
        f, g = ri.inflation_pj.iloc[0], ri.inflation_pj.iloc[1]
        margin = ri.gap_to_next.iloc[0]
        diffs[label] = (f - g, 100 * (f - g) / margin)
        print(f"    fable-minus-gpt differential inflation {f - g:+.3f} "
              f"vs margin {margin:+.3f} = {100 * (f - g) / margin:.0f}% of it")
    ri = row_level_inflation(sc, top.bonus, pj_top)
    pooled_d = ri.inflation.iloc[0] - ri.inflation.iloc[1]
    diffs["near-ceiling pooled"] = (pooled_d, 100 * pooled_d / ri.gap_to_next.iloc[0])
    print("  -- IS THE DIFFERENTIAL IDENTIFIED? no, and not even in sign --")
    for label, (d, pct) in sorted(diffs.items(), key=lambda kv: kv[1][0]):
        print(f"    {label:42s} {d:+.3f}  = {pct:+4.0f}% of the 0.042 margin")
    span = [v[1] for v in diffs.values()]
    print(f"    range {min(span):+.0f}% to {max(span):+.0f}%; "
          f"sign {'FLIPS across estimators' if min(span) < 0 < max(span) else 'is stable'}")
    print(f"  pooled-multiplier inflation spread {ri.inflation.max() - ri.inflation.min():+.3f}, "
          f"against published gaps of {ri.gap_to_next.min():.3f}-{ri.gap_to_next.max():.3f}")

    print("\nIS THE SIBLING SEAM AN ARTIFACT OF OWN-LINEAGE LENIENCY? (no)")
    for r in seam_without_own_lineage(wide).itertuples():
        print(f"  {r.sample:34s} same-lab {r.same_lab:.3f}  cross-lab {r.cross_lab_mean:.3f}  "
              f"seam {r.seam:+.3f}  n={r.n}")

    print("\nTHE DROP-ONE TEST ON fable's MARGIN IS UNINFORMATIVE (and why)")
    fr = fable_row_check(wide)
    print(f"  fable - gpt-5.5, full panels      {fr['margin_full_panel']:+.3f}")
    print(f"  fable - gpt-5.5, sonnet dropped   {fr['margin_sonnet_dropped']:+.3f}"
          f"   shift {fr['shift']:+.3f}")
    md = margin_diagnostic(sc)
    print(f"  BUT the lenient judge's own margin is {md['lenient_margin']:+.3f} vs the panel's "
          f"{md['panel_margin']:+.3f}: it favours fable by {md['extra_favour']:+.3f} MORE")
    print(f"  the shifts cancel only because the rows have unequal panels: "
          f"{md['dev_fable']:+.3f}/(k={md['k_fable']}-1) = {md['dev_gpt']:+.3f}/(k={md['k_gpt']}-1)")

    print("\nFOR THE CORRECTION RECORD: own-lab removal does not reverse the absolute ordering")
    for r in own_lab_removal(wide).itertuples():
        print(f"  {r.model:22s} full {r.full_panel:.3f} -> non-own-lab {r.non_own_lab:.3f}"
              f"   (judges left: {r.judges_left})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
