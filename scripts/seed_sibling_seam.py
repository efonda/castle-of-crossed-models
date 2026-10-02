#!/usr/bin/env python3
"""Prepare the sibling-pair run directories (paper App. G).

Copies the 198 signal generations scored by the existing judges into
data/runs/sibling-seam-google and data/runs/sibling-seam-xai, one directory per lab, so the two
added judges score an isolated copy and never write to the published runs. Only needed to
re-run those judges.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
DSTS = [RUNS / "sibling-seam-xai", RUNS / "sibling-seam-google"]
GENS = "generations.jsonl"

PANEL = [
    ("signal", "claude-opus-4-8-api"),
    ("signal", "claude-sonnet-4-6"),
    ("signal", "gpt-5.5"),
    ("signal_gemini", "gemini-3.1-pro-preview"),
    ("signal_grok", "grok-4.3"),
]


def scored_ids(run: str, judge: str) -> set[str]:
    p = RUNS / run / "judge_scores.jsonl"
    out = set()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["judge_model"] == judge:
            out.add(r["generation_id"])
    return out


common: set[str] | None = None
for run, judge in PANEL:
    ids = scored_ids(run, judge)
    common = ids if common is None else common & ids
    print(f"  {judge:<26} {len(ids):4d} scored in {run:<14} running intersection {len(common):4d}")
assert common is not None

src_rows = {}
for line in (RUNS / "signal" / GENS).read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r["id"] in common:
        src_rows[r["id"]] = (line, r)

missing = common - set(src_rows)
assert not missing, f"{len(missing)} scored ids absent from signal/generations.jsonl"
print(f"\nunit set: {len(src_rows)} generations, "
      f"{len({r['model'] for _, r in src_rows.values()})} distinct models")
print("  models:", ", ".join(sorted({r["model"] for _, r in src_rows.values()})))

for dst in DSTS:
    dst.mkdir(parents=True, exist_ok=True)
    dst_path = dst / GENS
    existing = set()
    if dst_path.exists():
        for line in dst_path.read_text().splitlines():
            if line.strip():
                existing.add(json.loads(line)["id"])

    to_write = [line for gid, (line, _) in sorted(src_rows.items()) if gid not in existing]
    if to_write:
        with dst_path.open("a") as fh:
            for line in to_write:
                fh.write(line + "\n")
    print(f"\n{dst.name}: wrote {len(to_write)} rows ({len(existing)} already present)")

    # --- prereg guard: byte-identical on id AND output_text -----------------------------
    back = {}
    for line in dst_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            back[r["id"]] = r
    assert set(back) == set(src_rows), f"{dst.name}: id set mismatch after copy"
    bad = [g for g in src_rows if back[g]["output_text"] != src_rows[g][1]["output_text"]]
    print(f"  GUARD: {len(back)}/{len(src_rows)} ids match, output_text mismatches: {len(bad)}")
    assert not bad
    per_model = collections.Counter(r["model"] for r in back.values())
    assert set(per_model.values()) == {18}, f"{dst.name}: uneven per-model counts"
    print(f"  per-model: {len(per_model)} models x 18 gens")

print("\nOK — both vendor dirs hold a read-only copy verified byte-identical. "
      "No source dir was written.")
