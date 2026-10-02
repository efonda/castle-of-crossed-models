"""Pairwise comparisons within the original trio (paper §3).

For each trio pair, reports win rates pooled over all judges, with same-lab judges removed, and
per neutral judge, so a single-judge result with small n stays visible. Reads
pairwise_votes.jsonl from data/runs/signal and data/runs/hard; no API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy import stats

FAM = {
    "gpt-5.5": "oa", "gpt-5.4-mini": "oa", "gpt-oss:20b": "oa",
    "claude-opus-4-8": "an", "claude-sonnet-4-6": "an", "claude-haiku-4-5-20251001": "an",
    "gemini-3.5-flash": "gg", "gemini-3.1-pro-preview": "gg", "gemini-3.1-flash-lite": "gg",
    "gemma4:26b": "gg", "gemma4:e4b": "gg", "llama3.2:3b": "meta",
}
JFAM = {
    "claude-opus-4-8-api": "an", "claude-sonnet-4-6": "an", "claude-opus-4-8-thinking": "an",
    "gpt-5.5": "oa", "gpt-5.5-thinking": "oa",
    "gemini-3.1-pro-preview": "gg", "gemini-3.5-flash": "gg", "gemma4:26b": "gg",
    "llama3.2:3b": "meta",
}
TRIO = ["gpt-5.5", "claude-opus-4-8", "claude-sonnet-4-6"]
PAIRS = [("gpt-5.5", "claude-sonnet-4-6"), ("gpt-5.5", "claude-opus-4-8"),
         ("claude-opus-4-8", "claude-sonnet-4-6")]


def load(dirs: list[str]) -> list[dict]:
    votes = []
    for d in dirs:
        p = Path(d) / "pairwise_votes.jsonl"
        if p.exists():
            votes += [json.loads(l) for l in open(p)]
    return votes


def rate(votes, a, b, judge_filter):
    w = n = 0
    for v in votes:
        if {v["model_a"], v["model_b"]} != {a, b}:
            continue
        if not judge_filter(v["judge_model"], a, b):
            continue
        n += 1
        if v["winner_model"] == a:
            w += 1
    if n == 0:
        return None
    return w, n, w / n, stats.binomtest(w, n, 0.5).pvalue


def main() -> None:
    dirs = sys.argv[1:] or ["data/runs/signal"]
    votes = load(dirs)
    if not votes:
        print(f"no pairwise_votes.jsonl in {dirs}")
        return
    print(f"# trio cross-lab forced choice — {dirs}  ({len(votes)} votes)\n")

    allj = lambda j, a, b: True
    neutral = lambda j, a, b: JFAM.get(j) not in (FAM[a], FAM[b])  # drop self/sibling-family

    print("== TRIO pairs: pooled (incl self-votes) vs CROSS-LAB (neutral judges) ==")
    for a, b in PAIRS:
        rp, rc = rate(votes, a, b, allj), rate(votes, a, b, neutral)
        line = f"  {a:16s} vs {b:18s}"
        if rp:
            line += f"  pooled {rp[0]:2d}/{rp[1]:2d}={rp[2]:.2f} p={rp[3]:.3f}"
        if rc:
            line += f"   cross-lab {rc[0]:2d}/{rc[1]:2d}={rc[2]:.2f} p={rc[3]:.3f}"
        print(line)
        # per-neutral-judge breakdown (exposes single-judge / small-n flukes)
        for jm in sorted({v["judge_model"] for v in votes}):
            if JFAM.get(jm) in (FAM[a], FAM[b]):
                continue
            r = rate(votes, a, b, lambda j, x, y, _jm=jm: j == _jm)
            if r:
                print(f"        neutral judge {jm:26s} {r[0]:2d}/{r[1]:2d}={r[2]:.2f} p={r[3]:.3f}")

    print("\n== below-cluster models vs the cluster (cross-lab), the bound (should be clearly <0.5) ==")
    present = {v["model_a"] for v in votes} | {v["model_b"] for v in votes}
    for x in ["gpt-5.4-mini", "gemini-3.1-pro-preview", "gemini-3.5-flash",
              "claude-haiku-4-5-20251001", "gemini-3.1-flash-lite"]:
        if x not in present:
            continue
        w = n = 0
        for t in TRIO:
            r = rate(votes, x, t, neutral)
            if r:
                w += r[0]; n += r[1]
        if n:
            print(f"  {x:28s} beats cluster {w:3d}/{n:3d}={w/n:.2f}  p={stats.binomtest(w, n, 0.5).pvalue:.4f}")

    print("\nReading: if a TRIO pair separates pooled but TIES cross-lab, the 'lead' was self-"
          "preference. A separation that SURVIVES cross-lab (esp. on >1 neutral lineage, decent n)"
          " is real — and a stronger result than the tie.")


if __name__ == "__main__":
    main()
