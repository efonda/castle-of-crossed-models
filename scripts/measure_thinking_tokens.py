#!/usr/bin/env python3
"""Reasoning tokens actually spent per model, read from persisted generations (no API calls).

Usage: python scripts/measure_thinking_tokens.py data/runs/<run> [data/runs/<run> ...]

Reasoning tokens live under different provider fields:
  Anthropic (thinking):      usage.output_tokens_details.thinking_tokens
  OpenAI (gpt-5.x):          usage.completion_tokens_details.reasoning_tokens
  Google (gemini):           usage.thoughts_token_count
  Anthropic non-thinking / local / others: absent -> 0
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def thinking_tokens(usage: dict) -> int | None:
    if not usage:
        return None
    otd = usage.get("output_tokens_details") or {}
    if "thinking_tokens" in otd:
        return otd["thinking_tokens"]
    ctd = usage.get("completion_tokens_details") or {}
    if "reasoning_tokens" in ctd:
        return ctd["reasoning_tokens"]
    if "thoughts_token_count" in usage:
        return usage["thoughts_token_count"]
    return None


def main(dirs: list[str]) -> None:
    # key: (model, condition, effort) -> list of thinking-token counts
    agg: dict[tuple, list[int]] = defaultdict(list)
    miss: dict[tuple, int] = defaultdict(int)  # rows with no measurable field
    for d in dirs:
        p = Path(d)
        if not p.exists():
            p = ROOT / "data" / "runs" / d
        f = p / "generations.jsonl"
        if not f.exists():
            print(f"  (no generations.jsonl in {p})")
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            pm = g.get("provider_meta") or {}
            eff = pm.get("effort") or pm.get("reasoning_effort") or ""
            key = (g.get("model", "?"), g.get("condition", "?"), eff)
            tt = thinking_tokens(pm.get("usage") or {})
            if tt is None:
                miss[key] += 1
            else:
                agg[key].append(tt)

    print(f"{'model':30} {'cond':9} {'effort':7} {'n':>4}  {'min':>5} {'mean':>7} {'median':>7} {'max':>6}")
    print("-" * 84)
    for key in sorted(agg):
        v = agg[key]
        m, c, e = key
        print(f"{m:30} {c:9} {e:7} {len(v):4}  {min(v):5} {statistics.mean(v):7.1f} "
              f"{statistics.median(v):7.1f} {max(v):6}")
    for key, n in sorted(miss.items()):
        m, c, e = key
        print(f"{m:30} {c:9} {e:7} {n:4}  (no reasoning-token field -> treat as 0)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
