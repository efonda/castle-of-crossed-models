#!/usr/bin/env python3
"""Same-lab agreement on sibling judge pairs (paper App. G).

For each same-lab pair on the seven-judge panel, the excess agreement is the pair's ICC(2,1)
minus the mean ICC of the cross-lab pairs containing either member, with a paired item-cluster
bootstrap (6 items, 5,000 draws, seed 0). Also reports whether each judge's closest judge is its
sibling. Thresholds were fixed before the added judges ran. No API calls.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from calvino.analysis import PRIMARY_DIMS, icc_2_1  # noqa: E402

RUNS = Path("data/runs")
B, SEED, N_ITEMS_EXPECTED = 5000, 0, 6

# judge -> (lineage, run dir holding its scores)
PANEL = {
    "claude-opus-4-8-api":    ("Anthropic", "signal"),
    "claude-sonnet-4-6":      ("Anthropic", "signal"),
    "gpt-5.5":                ("OpenAI",    "signal"),
    "gemini-3.1-pro-preview": ("Google",    "signal_gemini"),
    "gemini-3.6-flash":       ("Google",    "sibling-seam-google"),
    "grok-4.3":               ("xAI",       "signal_grok"),
    "grok-4.5":               ("xAI",       "sibling-seam-xai"),
}
SAME_LAB_PAIRS = [
    ("claude-opus-4-8-api", "claude-sonnet-4-6"),
    ("grok-4.3", "grok-4.5"),
    ("gemini-3.1-pro-preview", "gemini-3.6-flash"),
]
Q4_SUBPANEL = ["claude-opus-4-8-api", "claude-sonnet-4-6", "gpt-5.5"]
Q4_TARGET = (0.087, 0.038, 0.134)

# --- load ------------------------------------------------------------------------------
gen_item: dict[str, str] = {}
for line in (RUNS / "sibling-seam-google" / "generations.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        gen_item[r["id"]] = r["item_id"]
UNITS = set(gen_item)

score: dict[tuple[str, str], float] = {}
for judge, (_lin, run) in PANEL.items():
    p = RUNS / run / "judge_scores.jsonl"
    if not p.exists():
        continue
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["judge_model"] != judge or r["generation_id"] not in UNITS:
            continue
        s = r["scores"]
        score[(r["generation_id"], judge)] = float(np.mean([s[d] for d in PRIMARY_DIMS]))

complete = [j for j in PANEL if sum((u, j) in score for u in UNITS) == len(UNITS)]
partial = {j: sum((u, j) in score for u in UNITS) for j in PANEL if j not in complete}
print(f"unit set: {len(UNITS)} generations, {len(set(gen_item.values()))} items")
print(f"judges complete: {len(complete)}/{len(PANEL)}  {complete}")
if partial:
    print(f"judges INCOMPLETE: " + ", ".join(f"{j} {n}/{len(UNITS)}" for j, n in partial.items()))
print()


def picc(a: str, b: str, units: list[str]) -> float:
    m = np.array([[score[(u, a)], score[(u, b)]] for u in units], dtype=float)
    return icc_2_1(m)


def seam(pair: tuple[str, str], panel: list[str], units: list[str]) -> tuple[float, float, float]:
    """(ICC of the pair, mean cross-lineage ICC of its members, difference)."""
    a, b = pair
    same = picc(a, b, units)
    cross = [picc(x, y, units) for x, y in itertools.combinations(panel, 2)
             if PANEL[x][0] != PANEL[y][0] and (a in (x, y) or b in (x, y))]
    mc = float(np.mean(cross)) if cross else float("nan")
    return same, mc, same - mc


units_all = sorted(UNITS)
by_item: dict[str, list[str]] = defaultdict(list)
for u in units_all:
    by_item[gen_item[u]].append(u)
items = sorted(by_item)


def bootstrap(fn) -> tuple[float, float, float, int]:
    """Paired item-cluster bootstrap of a scalar statistic. Common draw, fixed seed."""
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(B):
        us = [u for it in rng.choice(items, len(items), replace=True) for u in by_item[it]]
        v = fn(us)
        if not np.isnan(v):
            draws.append(v)
    a = np.array(draws)
    return a.mean(), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), len(a)


# --- Q4: integrity check on the ORIGINAL three-judge panel -----------------------------
print("=" * 78)
print("Q4  INTEGRITY CHECK (not a test) - Anthropic pair on the original 3-judge panel")
print("=" * 78)
if all(j in complete for j in Q4_SUBPANEL):
    pair = ("claude-opus-4-8-api", "claude-sonnet-4-6")
    s, c, d = seam(pair, Q4_SUBPANEL, units_all)
    m, lo, hi, n = bootstrap(lambda us: seam(pair, Q4_SUBPANEL, us)[2])
    print(f"  same-lab ICC {s:.3f} | cross-lab mean {c:.3f} | seam {d:+.3f}")
    print(f"  paired bootstrap {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({n}/{B} draws)")
    t, tlo, thi = Q4_TARGET
    ok = abs(m - t) <= 0.01 and abs(lo - tlo) <= 0.02 and abs(hi - thi) <= 0.02
    print(f"  target {t:+.3f} [{tlo:+.3f}, {thi:+.3f}]  ->  {'REPRODUCED' if ok else 'MISMATCH - HALT'}")
    if not ok:
        print("\n  Per prereg Q4: a pipeline that cannot reproduce a known number may not be")
        print("  used to report an unknown one. Not reading the new pairs.")
        raise SystemExit(1)
else:
    print("  skipped: original panel incomplete")

# --- Q1: the same-lab pairs ------------------------------------------------------------
print()
print("=" * 78)
print("Q1  PER-PAIR SEAM (full panel of prereg §3)")
print("=" * 78)
panel_ready = len(complete) == len(PANEL)
if not panel_ready:
    print("  *** PANEL INCOMPLETE - verdicts are PENDING-PANEL, not final. ***")
    print("  A pair's seam averages over the cross-lineage judges, so adding a judge changes")
    print("  every seam. Numbers below use the complete judges only and WILL move.\n")

for pair in SAME_LAB_PAIRS:
    a, b = pair
    lin = PANEL[a][0]
    if a not in complete or b not in complete:
        have = [(j, partial.get(j, len(UNITS))) for j in pair]
        print(f"{lin:<10} {a} + {b}\n           NOT YET: " +
              ", ".join(f"{j} {n}/{len(UNITS)}" for j, n in have) + "\n")
        continue
    s, c, d = seam(pair, complete, units_all)
    m, lo, hi, n = bootstrap(lambda us, p=pair: seam(p, complete, us)[2])
    sd_a = float(np.std([score[(u, a)] for u in units_all]))
    sd_b = float(np.std([score[(u, b)] for u in units_all]))

    if min(sd_a, sd_b) < 0.10 or len(units_all) < 100:
        verdict = "UNINFORMATIVE (degenerate rater or n<100)"
    elif d > 0 and lo > 0:
        verdict = "REPLICATES"
    elif d < 0 and hi < 0:
        verdict = "REVERSES"
    elif abs(d) <= 0.02:
        verdict = "NULL"
    else:
        verdict = "WEAK-POSITIVE" if d > 0 else "WEAK-NEGATIVE"

    print(f"{lin:<10} {a} + {b}")
    print(f"           ICC {s:.3f} | cross-lineage mean {c:.3f} | seam {d:+.3f}")
    print(f"           bootstrap {m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({n}/{B})  rater SD {sd_a:.2f}/{sd_b:.2f}")
    print(f"           -> {verdict}{'' if panel_ready else '  (PENDING-PANEL)'}\n")

# --- DIAGNOSTIC (not a band): the averaging-free rank test ------------------------------
# seam() subtracts a MEAN over cross-lineage pairs, so one badly-disagreeing partner can
# manufacture a positive seam without the siblings actually preferring each other. The rank
# test is immune to that: for each judge, does its sibling top its own list of partners?
# A genuine sibling effect should show sibling = rank 1 for BOTH members.
print("=" * 78)
print("DIAGNOSTIC (no band): does each judge agree MOST with its sibling?")
print("=" * 78)
for pair in SAME_LAB_PAIRS:
    a, b = pair
    if a not in complete or b not in complete:
        continue
    print(f"{PANEL[a][0]}:")
    for me, sib in ((a, b), (b, a)):
        partners = sorted(
            ((picc(me, o, units_all), o) for o in complete if o != me), reverse=True)
        rank = next(i for i, (_v, o) in enumerate(partners, 1) if o == sib)
        flag = "sibling is TOP" if rank == 1 else f"sibling ranks {rank} of {len(partners)}"
        print(f"  {me:<24} {flag}")
        print("      " + "  ".join(
            f"{'*' if o == sib else ''}{o.split('-')[0][:6]}:{v:.3f}" for v, o in partners))
    print()

# --- the full pairwise matrix, for the record ------------------------------------------
print("=" * 78)
print("All pairwise ICCs among complete judges (unit-consistent, n=%d)" % len(units_all))
print("=" * 78)
rows = []
for x, y in itertools.combinations(complete, 2):
    same = PANEL[x][0] == PANEL[y][0]
    rows.append((same, PANEL[x][0] if same else f"{PANEL[x][0]}/{PANEL[y][0]}",
                 x, y, picc(x, y, units_all)))
for same, lin, x, y, v in sorted(rows, key=lambda r: (not r[0], -r[4])):
    print(f"  {'SAME ' if same else 'cross'} {lin:<20} {x[:24]:<24} + {y[:24]:<24} {v:6.3f}")
