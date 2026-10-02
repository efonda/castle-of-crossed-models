"""Rank stability and pairwise tier calibration (paper §3).

1. rank stability rho across judges, scales and reasoning modes (generation bootstrap, mean
   Spearman), using calvino.analysis.rank_stability;
2. pairwise win rates across tier gaps: flagship-lite, flagship-mid, mid-lite.

No API calls.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# See the note in scripts/raw_data_lock.py: no build backend means calvino is never installed, and
# a script run by path does not get the repo root on sys.path. Without this the script cannot run
# at all -- not a data problem, and not visible until someone actually invokes it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calvino.analysis import build_scores_frame, rank_stability  # noqa: E402
from calvino.storage import Store  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"

# Signal-run judge subsets that map onto the three claimed axes.
BASE = ["claude-opus-4-8-api", "claude-sonnet-4-6", "gpt-5.5"]
S10 = ["claude-opus-4-8-api-s10", "claude-sonnet-4-6-s10", "gpt-5.5-s10"]
THINKING = ["claude-opus-4-8-thinking", "gpt-5.5-thinking"]


def rho_for(df, judges, label):
    sub = df[df["judge_model"].isin(judges)]
    if sub.empty:
        print(f"  {label:32s}  (no rows)")
        return None
    rs = rank_stability(sub, n_boot=2000, seed=0)
    print(f"  {label:32s}  ρ = {rs.mean_spearman:.3f}   (n_models={len(rs.ranking)})")
    return rs.mean_spearman


print("=" * 72)
print("1. RANK-STABILITY ρ  (claim: 0.97–0.99 across judges, scales, reasoning)")
print("=" * 72)

sig = build_scores_frame(Store(RUNS / "signal"))
rhos = []
print("\nsignal run — by axis:")
for label, judges in [
    ("judges: base trio (0–5)", BASE),
    ("scales: same trio, 0–10 (s10)", S10),
    ("reasoning: thinking judges", THINKING),
]:
    r = rho_for(sig, judges, label)
    if r is not None:
        rhos.append(r)

print("\nsignal run — judges axis, each base judge alone:")
for j in BASE:
    r = rho_for(sig, [j], f"judge: {j}")
    if r is not None:
        rhos.append(r)

# Cross-instrument: the hard run (a different, harder item bank).
hard_dir = RUNS / "hard"
if (hard_dir / "judge_scores.jsonl").exists():
    print("\nhard run (12-item bank) — base trio:")
    hard = build_scores_frame(Store(hard_dir))
    hj = [j for j in BASE if j in set(hard["judge_model"])]
    r = rho_for(hard, hj or BASE, "hard: base trio")
    if r is not None:
        rhos.append(r)

if rhos:
    print(f"\n  ENVELOPE across the above: ρ ∈ [{min(rhos):.3f}, {max(rhos):.3f}]")
    print(f"  claim 0.97–0.99 -> {'HOLDS' if min(rhos) >= 0.965 else 'CHECK: low end below 0.965'}")

print("\nsignal run — CROSS-INSTRUMENT rank agreement (the likely source of the 0.99):")
from scipy import stats  # noqa: E402

from calvino.analysis import primary_aggregate  # noqa: E402


def ordering(df, judges):
    sub = df[df["judge_model"].isin(judges)].copy()
    sub["_p"] = primary_aggregate(sub)
    return sub.groupby("model")["_p"].mean()


def cross_rho(df, ja, jb, label):
    a, b = ordering(df, ja), ordering(df, jb)
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 3:
        print(f"  {label:32s}  (too few common models)")
        return None
    rho = stats.spearmanr(a[common].values, b[common].values).correlation
    print(f"  {label:32s}  ρ = {rho:.3f}   (n_models={len(common)})")
    return rho


cross_rho(sig, BASE, S10, "0–5 vs 0–10 scale")
cross_rho(sig, BASE, THINKING, "normal vs thinking (reasoning)")

print()
print("=" * 72)
print("2. FORCED-CHOICE CALIBRATION LADDER  (claim: F–L 0.88, F–M 0.75, M–L 0.81)")
print("=" * 72)

TIER = {
    "gpt-5.5": "flagship", "claude-opus-4-8": "flagship",
    "claude-sonnet-4-6": "flagship", "gemini-3.1-pro-preview": "flagship",
    "gemini-3.5-flash": "mid", "gpt-5.4-mini": "mid",
    "claude-haiku-4-5": "lite", "claude-haiku-4-5-20251001": "lite",
    "gemini-3.1-flash-lite": "lite",
}
RANK = {"lite": 0, "mid": 1, "flagship": 2}
CROSS_LAB = {"claude-opus-4-8-api", "claude-sonnet-4-6", "gpt-5.5", "gemini-3.1-pro-preview"}


def ladder(votes, judge_filter=None, label=""):
    pair_tot = defaultdict(int)
    pair_strong = defaultdict(float)
    ties = 0
    for v in votes:
        if v.get("sample_idx", 0) != 0:
            continue
        if judge_filter is not None and v["judge_model"] not in judge_filter:
            continue
        ta, tb = TIER.get(v["model_a"]), TIER.get(v["model_b"])
        if ta is None or tb is None or ta == tb:
            continue
        key = tuple(sorted([ta, tb], key=lambda t: -RANK[t]))  # (stronger, weaker)
        strong_model = v["model_a"] if RANK[ta] > RANK[tb] else v["model_b"]
        w = v.get("winner_model")
        pair_tot[key] += 1
        if w == strong_model:
            pair_strong[key] += 1.0
        elif w in (None, "tie", ""):
            pair_strong[key] += 0.5  # split ties
            ties += 1
    print(f"\n  {label}")
    for key in [("flagship", "lite"), ("flagship", "mid"), ("mid", "lite")]:
        n = pair_tot[key]
        if n:
            print(f"    {key[0]:8s}–{key[1]:8s}  win={pair_strong[key]/n:.3f}  (n={n})")
    if ties:
        print(f"    [ties encountered: {ties}, counted as 0.5]")


votes = [json.loads(l) for l in open(RUNS / "signal" / "pairwise_votes.jsonl")]
ladder(votes, judge_filter=None, label="signal, ALL judges, sample_idx==0 (matches report's stated universe)")
ladder(votes, judge_filter=CROSS_LAB, label="signal, CROSS-LAB judges only (de-confounded)")
