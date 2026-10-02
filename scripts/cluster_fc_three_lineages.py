#!/usr/bin/env python3
"""Pre-registered within-cluster pairwise adjudication (paper §3, Table 2, App. D).

Judges from three labs with no model in the contest (grok-4.3, qwen3.7-max, gemini-3.1-pro)
compare gpt-5.5 with claude-opus-4-8 and claude-sonnet-4-6 on the 12 hard items. A comparison
counts only when both presentation orders agree. Prints per-lab and pooled tallies, decision
rates and item-level sign tests. Reads persisted votes under data/runs/; no API calls.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"

GPT, OPUS, SONNET = "gpt-5.5", "claude-opus-4-8", "claude-sonnet-4-6"
EDGES = [(GPT, OPUS), (GPT, SONNET)]

# lineage -> (label, source dir). One dir each; see WHY NOT A UNION above.
PRIMARY = {
    "grok-4.3": ("xAI", "hard_frontier"),
    "qwen3.7-max": ("Alibaba", "hard_frontier"),
    "gemini-3.1-pro-preview": ("Google", "cluster-gemini-hard"),
}
# the one defensible alternative: grok's independent re-run of the same bank-2 comparisons
SENSITIVITY = dict(PRIMARY, **{"grok-4.3": ("xAI", "hard_grok_bo")})


def sign_test_p(won: int, lost: int) -> float:
    """Two-sided exact binomial at p=0.5 on non-tied items."""
    n = won + lost
    if n == 0:
        return float("nan")
    k = max(won, lost)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def collect(sources: dict[str, tuple[str, str]]):
    """-> {(lineage, edge_opponent): (Counter(winner), set(items), {item: Counter})}"""
    comps: dict[tuple, list[str]] = defaultdict(list)
    meta: dict[tuple, tuple[str, frozenset]] = {}
    for judge, (_lab, d) in sources.items():
        p = RUNS / d / "pairwise_votes.jsonl"
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["judge_model"] != judge:
                continue
            pair = frozenset((r["model_a"], r["model_b"]))
            if not any(pair == frozenset(e) for e in EDGES):
                continue
            k = (judge, r["comparison_id"])
            comps[k].append(r["winner_model"])
            meta[k] = (r["item_id"], pair)

    res: dict[tuple, tuple[Counter, set, dict]] = defaultdict(
        lambda: (Counter(), set(), defaultdict(Counter)))
    for k, winners in comps.items():
        judge = k[0]
        item, pair = meta[k]
        opp = next(m for m in pair if m != GPT)
        if len(winners) < 2 or len(set(winners)) != 1:
            continue
        c, its, per_item = res[(judge, opp)]
        c[winners[0]] += 1
        its.add(item)
        per_item[item][winners[0]] += 1
    return res


def report(sources, title):
    res = collect(sources)
    print("=" * 78)
    print(title)
    print("=" * 78)
    out = {}
    for _a, opp in EDGES:
        tot = Counter()
        all_items: dict[str, Counter] = defaultdict(Counter)
        print(f"\n  gpt-5.5  vs  {opp}")
        for judge, (lab, d) in sources.items():
            c, its, per_item = res.get((judge, opp), (Counter(), set(), {}))
            n = c[GPT] + c[opp]
            rate = c[GPT] / n if n else float("nan")
            print(f"    {lab:8s} {d:22s} {c[GPT]:3d}-{c[opp]:<3d} = {rate:.3f}"
                  f"   n={n:3d}  items={len(its):2d}")
            tot += c
            for it, cc in per_item.items():
                all_items[it] += cc
        n = tot[GPT] + tot[opp]
        rate = tot[GPT] / n if n else float("nan")
        won = sum(1 for c in all_items.values() if c[GPT] > c[opp])
        lost = sum(1 for c in all_items.values() if c[opp] > c[GPT])
        tied = len(all_items) - won - lost
        p = sign_test_p(won, lost)
        print(f"    {'POOLED':<31} {tot[GPT]:3d}-{tot[opp]:<3d} = {rate:.3f}"
              f"   n={n:3d}  items={len(all_items):2d}")
        print(f"    item-level: gpt-5.5 leads on {won} of {len(all_items)} items "
              f"({lost} against, {tied} tied), sign test p = {p:.4f}")
        out[opp] = (tot[GPT], tot[opp], rate, len(all_items), won, lost, tied, p)
    print()
    return out


def main() -> int:
    a = report(PRIMARY, "PRIMARY CUT -- bank 2, three neutral lineages, both-orders decisive")
    b = report(SENSITIVITY, "SENSITIVITY -- identical, but grok from its re-run (hard_grok_bo)")

    print("=" * 78)
    print("ROBUSTNESS TO THE GROK SOURCE")
    print("=" * 78)
    for opp in (OPUS, SONNET):
        print(f"  gpt-5.5 vs {opp:20s} primary {a[opp][0]}-{a[opp][1]} = {a[opp][2]:.3f}"
              f"   |   grok re-run {b[opp][0]}-{b[opp][1]} = {b[opp][2]:.3f}")
    print("\nQuote the PRIMARY pooled line with its item count. 'Both orders' means")
    print("order-CONSISTENT decisions retained (a split abstains), not both presentations")
    print("counted as two votes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
