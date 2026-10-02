"""Fable pairwise counts (paper §3, App. G).

Tallies claude-fable-5 against the original trio under the three neutral labs' judges on the
signal and hard banks. A comparison is decisive only when both presentation orders are present
and agree; position splits and missing presentations abstain. No API calls.

Run: uv run python scripts/verify_fable_fc.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
NEUTRAL = {"gemini-3.1-pro-preview": "Google", "qwen3.7-max": "Alibaba", "grok-4.3": "xAI"}
CLUSTER = ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-5.5"]


def load(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def decisive(votes_by_comp):
    """Map {(judge, comp_id): [winner_model,...]} -> per-judge {winner: count} for
    comparisons where both presentations are present and agree."""
    out = defaultdict(Counter)
    incomplete = Counter()
    abst = Counter()
    for (judge, _comp), winners in votes_by_comp.items():
        if len(winners) < 2:
            incomplete[judge] += 1
            continue
        if len(set(winners)) == 1:
            out[judge][winners[0]] += 1
        else:
            abst[judge] += 1
    return out, abst, incomplete


print("=" * 72)
print("1. FABLE vs CLUSTER  (signal-item neutral forced choice; data/runs/fable-fc)")
print("=" * 72)
rows = load(RUNS / "fable-fc" / "pairwise_votes.jsonl")
by_comp = defaultdict(list)
for r in rows:
    by_comp[(r["judge_model"], r["comparison_id"])].append(r["winner_model"])


def opp_of(comp_id):
    p = comp_id.split("|")
    return p[2] if p[1] == "claude-fable-5" else p[1]


# per lineage: fable wins vs opp wins + decisive-n
dec, abst, incomp = decisive(by_comp)
tf = to = 0
for j, lin in NEUTRAL.items():
    f = dec[j]["claude-fable-5"]
    o = sum(v for k, v in dec[j].items() if k != "claude-fable-5")
    n = f + o
    print(f"  {lin:8s} {j:24s}  fable {f:2d}-{o:1d}   decisive-n {n}/54  (abst {abst[j]}, incomplete {incomp[j]})")
    tf += f
    to += o
print(f"  POOLED fable vs cluster: {tf}-{to}")

# per opponent, pooled across lineages
per_opp = defaultdict(Counter)
for (judge, comp), winners in by_comp.items():
    if len(winners) < 2 or len(set(winners)) != 1:
        continue
    o = opp_of(comp)
    per_opp[o]["fable" if winners[0] == "claude-fable-5" else "opp"] += 1
for o in CLUSTER:
    d = per_opp[o]
    print(f"    fable vs {o:20s} {d['fable']:2d}-{d['opp']:1d}")

print()
print("=" * 72)
print("1b. FABLE vs CLUSTER on the HARD bank (pre-registered cross-bank replication;")
print("    data/runs/hard_fable_fc; pre-registered thresholds)")
print("=" * 72)
rows_h = load(RUNS / "hard_fable_fc" / "pairwise_votes.jsonl")
by_comp_h = defaultdict(list)
for r in rows_h:
    by_comp_h[(r["judge_model"], r["comparison_id"])].append(r["winner_model"])
dec_h, abst_h, incomp_h = decisive(by_comp_h)
tf = to = 0
for j, lin in NEUTRAL.items():
    f = dec_h[j]["claude-fable-5"]
    o = sum(v for k, v in dec_h[j].items() if k != "claude-fable-5")
    n = f + o
    print(f"  {lin:8s} {j:24s}  fable {f:2d}-{o:2d}  decisive-n {n}/108  (abst {abst_h[j]}, incomplete {incomp_h[j]})")
    tf += f
    to += o
print(f"  POOLED fable vs cluster (hard): {tf}-{to}  ({tf/(tf+to):.3f})")
per_opp_h = defaultdict(Counter)
per_item_h = defaultdict(Counter)
for (judge, comp), winners in by_comp_h.items():
    if judge not in NEUTRAL or len(winners) < 2 or len(set(winners)) != 1:
        continue
    w = "fable" if winners[0] == "claude-fable-5" else "opp"
    per_opp_h[opp_of(comp)][w] += 1
    per_item_h[comp.split("|")[0]][w] += 1
for o in CLUSTER:
    d = per_opp_h[o]
    print(f"    fable vs {o:20s} {d['fable']:2d}-{d['opp']:2d}")
fav = sum(1 for c in per_item_h.values() if c["fable"] > c["opp"])
tie = sum(1 for c in per_item_h.values() if c["fable"] == c["opp"])
print(f"    items favoring fable: {fav}/{len(per_item_h)} (+{tie} tie)")

print()
print("=" * 72)
print("2. gpt-5.5 vs opus/sonnet  (neutral lineages at power; data/runs/hard_frontier)")
print("=" * 72)
rows = load(RUNS / "hard_frontier" / "pairwise_votes.jsonl")
pairs = {frozenset(["gpt-5.5", "claude-opus-4-8"]), frozenset(["gpt-5.5", "claude-sonnet-4-6"])}
by_comp = defaultdict(list)
for r in rows:
    if r["judge_model"] not in NEUTRAL:
        continue
    pr = frozenset([r["model_a"], r["model_b"]])
    if pr not in pairs:
        continue
    by_comp[(r["judge_model"], frozenset([r["model_a"], r["model_b"]]), r["item_id"], r.get("sample_idx", 0))].append(r["winner_model"])
tally = defaultdict(Counter)
for (judge, pr, _it, _s), winners in by_comp.items():
    if len(winners) < 2 or len(set(winners)) != 1:
        continue
    tally[(tuple(sorted(pr)), NEUTRAL[judge])][winners[0]] += 1
for pr in [("claude-opus-4-8", "gpt-5.5"), ("claude-sonnet-4-6", "gpt-5.5")]:
    gpt = opp = 0
    detail = []
    for (p, lin), c in tally.items():
        if p != pr:
            continue
        g = c["gpt-5.5"]
        o = sum(v for k, v in c.items() if k != "gpt-5.5")
        gpt += g
        opp += o
        detail.append(f"{lin} {g}-{o}")
    other = [m for m in pr if m != "gpt-5.5"][0]
    n = gpt + opp
    print(f"  gpt-5.5 vs {other:20s}  {gpt}-{opp}  ({gpt/n:.2f})  [{', '.join(detail)}]")
