"""Hard-bank absolute scores under the neutral judges (paper App. B, App. G).

Per-judge and pooled neutral-judge model means on the hard bank (mean rubric score, combinatorial
excluded) with generation-resampled bootstrap intervals, per-dimension means, and agreement
between pairwise decisions and absolute-rubric margins. Reads data/runs/hard_fable_fc and
data/runs/hard_frontier; no API calls.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "runs" / "hard_fable_fc"
FRONTIER = ROOT / "data" / "runs" / "hard_frontier"
NEUTRAL = ["gemini-3.1-pro-preview", "qwen3.7-max", "grok-4.3"]
MODELS = ["claude-fable-5", "gpt-5.5", "claude-opus-4-8", "claude-sonnet-4-6"]
PRIMARY = ["distinctness", "bridge", "tonal", "originality"]
SEED = 20260703


def load(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def agg(score_row):
    s = score_row["scores"]
    return sum(s[d] for d in PRIMARY) / len(PRIMARY)


def decided(votes_path, judges):
    """Decided comparisons: both position-flipped presentations present and agreeing."""
    by = defaultdict(list)
    meta = {}
    for r in load(votes_path):
        key = (r["judge_model"], r["comparison_id"])
        by[key].append(r["winner_model"])
        meta[key] = r
    out = []
    for (j, cid), ws in by.items():
        if j not in judges or len(ws) < 2 or len(set(ws)) != 1:
            continue
        m = meta[(j, cid)]
        out.append({"judge": j, "item": m["item_id"], "a": m["model_a"], "b": m["model_b"],
                    "sample": m["sample_idx"], "winner": ws[0]})
    return out


def main():
    gens = {g["id"]: g for g in load(DIR / "generations.jsonl")}
    scores = load(DIR / "judge_scores.jsonl")
    # (gen_id, judge) -> aggregate; and index by (model, item, sample, judge)
    aggof = {}
    bykey = {}
    for s in scores:
        g = gens.get(s["generation_id"])
        if not g or s["judge_model"] not in NEUTRAL:
            continue
        a = agg(s)
        aggof[(s["generation_id"], s["judge_model"])] = a
        bykey[(g["model"], g["item_id"], g["sample_idx"], s["judge_model"])] = a

    print("=" * 78)
    print("HARD-BANK ABSOLUTE (3 neutral lineages, 144 gens each) + Q4 CONVERGENT VALIDITY")
    print("=" * 78)

    # ---- (1) per-judge + pooled model means; pre-named (a) ----
    print("\n(1) NEUTRAL-ABSOLUTE MODEL MEANS on hard (primary aggregate, 0-5):")
    permodel_perjudge = {}
    for j in NEUTRAL:
        row = {}
        for m in MODELS:
            vals = [a for (gid, jj), a in aggof.items() if jj == j and gens[gid]["model"] == m]
            row[m] = (np.mean(vals), len(vals))
        permodel_perjudge[j] = row
        order = sorted(MODELS, key=lambda m: -row[m][0])
        print(f"    {j:24s}: " + "  ".join(f"{m.split('-')[1] if m.startswith('claude') else m}={row[m][0]:.3f}(n={row[m][1]})" for m in MODELS))
        print(f"    {'':24s}  nominal order: {' > '.join(order)}")
    # pooled: per generation, mean over the 3 judges, then per model + bootstrap CI
    pooled_gen = defaultdict(list)  # model -> [per-gen pooled agg]
    genlist = defaultdict(list)     # model -> [(item, pooled_agg)]
    for gid, g in gens.items():
        vals = [aggof[(gid, j)] for j in NEUTRAL if (gid, j) in aggof]
        if len(vals) == len(NEUTRAL):
            pooled_gen[g["model"]].append(np.mean(vals))
            genlist[g["model"]].append((g["item_id"], np.mean(vals)))
    rng = np.random.default_rng(SEED)
    print("\n    POOLED (per-gen mean over 3 judges; gen-resampled bootstrap, B=20000):")
    for m in sorted(MODELS, key=lambda m: -np.mean(pooled_gen[m])):
        v = np.array(pooled_gen[m])
        boots = [np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(20000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"      {m:20s} {np.mean(v):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(v)} gens)")
    ordered = sorted(MODELS, key=lambda m: -np.mean(pooled_gen[m]))
    print(f"    pre-named (a): pooled neutral-absolute NOMINAL order on hard: {' > '.join(ordered)}")

    # ---- (2) per-dimension pooled means ----
    print("\n(2) PER-DIMENSION pooled neutral means (carrier check):")
    for m in MODELS:
        dims = {}
        for d in PRIMARY + ["combinatorial"]:
            vals = [s["scores"][d] for s in scores
                    if s["judge_model"] in NEUTRAL and gens.get(s["generation_id"], {}).get("model") == m]
            dims[d] = np.mean(vals)
        print(f"    {m:20s} " + "  ".join(f"{d[:4]}={dims[d]:.2f}" for d in PRIMARY + ["combinatorial"]))

    # ---- (3) Q4 clean-pair FC-vs-rubric agreement ----
    fc = decided(DIR / "pairwise_votes.jsonl", set(NEUTRAL))
    fc += decided(FRONTIER / "pairwise_votes.jsonl", set(NEUTRAL))  # cluster edges, same gen ids
    def genid(model, item, sample):
        for gid, g in gens.items():
            if g["model"] == model and g["item_id"] == item and g["sample_idx"] == sample:
                return gid
        return None
    gid_cache = {(g["model"], g["item_id"], g["sample_idx"]): gid for gid, g in gens.items()}

    def clean_pairs(abs_judge, fc_subset):
        pairs = []
        for c in fc_subset:
            if c["judge"] == abs_judge:
                continue
            ga = gid_cache.get((c["a"], c["item"], c["sample"]))
            gb = gid_cache.get((c["b"], c["item"], c["sample"]))
            if ga is None or gb is None:
                continue
            aa, ab = aggof.get((ga, abs_judge)), aggof.get((gb, abs_judge))
            if aa is None or ab is None or aa == ab:
                continue  # missing or rubric tie -> excluded per pinned rule
            higher = c["a"] if aa > ab else c["b"]
            pairs.append({"agree": higher == c["winner"], "item": c["item"]})
        return pairs

    def wilson(k, n):
        if n == 0:
            return (float("nan"),) * 2
        p, z = k / n, 1.96
        den = 1 + z * z / n
        ctr = (p + z * z / (2 * n)) / den
        w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return ctr - w, ctr + w

    print("\n(3) Q4 CLEAN-PAIR FC-vs-RUBRIC AGREEMENT (FC judge != absolute judge; ties excluded):")
    fable_fc = [c for c in fc if "claude-fable-5" in (c["a"], c["b"])]
    gem_primary = clean_pairs("gemini-3.1-pro-preview", [c for c in fc if c["judge"] != "gemini-3.1-pro-preview"])
    k = sum(p["agree"] for p in gem_primary)
    lo, hi = wilson(k, len(gem_primary))
    print(f"    GEMINI-PRIMARY continuity cut (all hard edges): {k}/{len(gem_primary)} = "
          f"{k/len(gem_primary):.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")
    allpairs = []
    directions = []
    for aj in NEUTRAL:
        ps = clean_pairs(aj, fc)
        allpairs += [dict(p, judge=aj) for p in ps]
        kk = sum(p["agree"] for p in ps)
        lo2, hi2 = wilson(kk, len(ps))
        directions.append(kk / len(ps) > 0.5 if ps else None)
        note = " (descriptive only, <10 pairs)" if len(ps) < 10 else ""
        print(f"    per-abs-judge {aj:24s}: {kk}/{len(ps)} = {kk/len(ps) if ps else float('nan'):.3f} "
              f"[{lo2:.3f}, {hi2:.3f}]{note}")
    K, N = sum(p["agree"] for p in allpairs), len(allpairs)
    lo3, hi3 = wilson(K, N)
    agree_dirs = all(d for d in directions if d is not None)
    print(f"    POOLED (quotable: per-judge directions {'AGREE' if agree_dirs else 'DISAGREE'}): "
          f"{K}/{N} = {K/N:.3f}  Wilson 95% [{lo3:.3f}, {hi3:.3f}]")
    items = sorted(set(p["item"] for p in allpairs))
    byitem = {it: [p["agree"] for p in allpairs if p["item"] == it] for it in items}
    rng2 = np.random.default_rng(SEED + 1)
    boots = []
    for _ in range(20000):
        ch = rng2.choice(items, size=len(items), replace=True)
        w = sum(sum(byitem[c]) for c in ch)
        t = sum(len(byitem[c]) for c in ch)
        boots.append(w / t if t else np.nan)
    blo, bhi = np.nanpercentile(boots, [2.5, 97.5])
    print(f"    pooled item-cluster bootstrap ({len(items)} items): 95% CI [{blo:.3f}, {bhi:.3f}]")
    fable_only = [p for p in allpairs]  # keep full; fable-edge subset:
    fpairs = []
    for aj in NEUTRAL:
        fpairs += clean_pairs(aj, [c for c in fable_fc])
    fk = sum(p["agree"] for p in fpairs)
    flo, fhi = wilson(fk, len(fpairs))
    print(f"    fable-edges-only subset: {fk}/{len(fpairs)} = {fk/len(fpairs):.3f} [{flo:.3f}, {fhi:.3f}]")

    # ---- (4) same-judge halo ----
    halo = []
    for c in fc:
        if c["judge"] != "gemini-3.1-pro-preview":
            continue
        ga = gid_cache.get((c["a"], c["item"], c["sample"]))
        gb = gid_cache.get((c["b"], c["item"], c["sample"]))
        if ga is None or gb is None:
            continue
        aa, ab = aggof.get((ga, "gemini-3.1-pro-preview")), aggof.get((gb, "gemini-3.1-pro-preview"))
        if aa is None or ab is None or aa == ab:
            continue
        halo.append((c["a"] if aa > ab else c["b"]) == c["winner"])
    hk = sum(halo)
    print(f"\n(4) SAME-JUDGE (gemini FC x gemini abs): {hk}/{len(halo)} = {hk/len(halo):.3f}; "
          f"halo gap vs cross-judge pooled = {hk/len(halo) - K/N:+.3f}")

    # ---- (5) exploratory addendum: edge-level sign concordance ----
    print("\n(5) EDGE-LEVEL CONCORDANCE (exploratory, pinned pre-computation): per judge, does the")
    print("    absolute nominal sign (mean aggregate diff) match the SAME judge's FC verdict?")
    edges = [("claude-fable-5", "gpt-5.5"), ("claude-fable-5", "claude-opus-4-8"),
             ("claude-fable-5", "claude-sonnet-4-6"), ("gpt-5.5", "claude-opus-4-8"),
             ("gpt-5.5", "claude-sonnet-4-6"), ("claude-opus-4-8", "claude-sonnet-4-6")]
    conc = tot = 0
    for j in NEUTRAL:
        cells = []
        for (m1, m2) in edges:
            d = [c for c in fc if c["judge"] == j and {c["a"], c["b"]} == {m1, m2}]
            w1 = sum(1 for c in d if c["winner"] == m1)
            w2 = len(d) - w1
            if not d or w1 == w2:
                cells.append(f"{m1.split('-')[1] if m1.startswith('claude') else m1}/{m2.split('-')[1] if m2.startswith('claude') else m2}: FC-unresolved")
                continue
            fcwin = m1 if w1 > w2 else m2
            absdiff = permodel_perjudge[j][m1][0] - permodel_perjudge[j][m2][0]
            if absdiff == 0:
                cells.append("abs-tie")
                continue
            abswin = m1 if absdiff > 0 else m2
            ok = abswin == fcwin
            conc += ok
            tot += 1
            cells.append(f"{'OK' if ok else 'X'}({fcwin.split('-')[1] if fcwin.startswith('claude') else fcwin},{abs(absdiff):.3f})")
        print(f"    {j:24s}: " + "  ".join(cells))
    print(f"    concordance on FC-resolved, abs-non-tied edges: {conc}/{tot} = {conc/tot:.2f}")

    print("\n" + "=" * 78)
    print("Pinned framing (crossbank report, written pre-computation): concordance-in-hindsight,")
    print("never 'the absolute means were suggestive all along'. FC is the resolver; absolute is")
    print("the same construct read at lower gain wherever concordance holds.")
    print("=" * 78)


if __name__ == "__main__":
    main()
