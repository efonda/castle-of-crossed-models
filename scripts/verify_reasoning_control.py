"""Reasoning-off control for gpt-5.5 (paper Table 2, App. D).

Scores a reasoning-disabled gpt-5.5 against claude-opus-4-8 and claude-sonnet-4-6 under the three
neutral labs' judges, from data/runs/mini-reason-control-signal, with the same both-orders rule
as the main adjudication; also prints absolute means for the same generations. Disabling
reasoning also changed the sampling temperature, so the contrast is not clean. No API calls.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
NEUTRAL = ["gemini-3.1-pro-preview", "qwen3.7-max", "grok-4.3"]
PRIMARY = ["distinctness", "bridge", "tonal", "originality"]
MODELS = ["claude-fable-5", "gpt-5.5", "gpt-5.5-noreason", "claude-opus-4-8", "claude-sonnet-4-6"]
SIGNAL = {"key-01", "lantern-01", "salt-compass-01", "scales-01", "tower-01", "wheel-01"}

SIG_DIRS = ["signal_fable", "signal_gemini", "signal_grok", "signal_qwen", "fable-fc",
            "mini-noreason-gemini", "mini-noreason-grok", "mini-noreason-qwen"]
HARD_DIRS = ["hard_fable_fc", "mini-reason-control-hard"]


def collect(dirs):
    """(gen_id -> (model,item,cond)) and deduped (gen_id,judge) -> primary aggregate."""
    gen, agg = {}, {}
    for d in dirs:
        gn, js = RUNS / d / "generations.jsonl", RUNS / d / "judge_scores.jsonl"
        if gn.exists():
            for l in open(gn):
                g = json.loads(l)
                gen[g["id"]] = (g.get("model"), g.get("item_id"), g.get("condition"))
        if js.exists():
            for l in open(js):
                j = json.loads(l)
                if j.get("judge_model") not in NEUTRAL:
                    continue
                gid, s = j.get("generation_id"), j.get("scores", {})
                if gid not in gen or not all(k in s for k in PRIMARY):
                    continue
                agg[(gid, j["judge_model"])] = sum(s[k] for k in PRIMARY) / len(PRIMARY)
    return gen, agg


def per_model_means(dirs, want_signal):
    gen, agg = collect(dirs)
    per = defaultdict(list)
    for gid in {g for (g, _) in agg}:
        m, it, cond = gen[gid]
        if m not in MODELS or it is None:
            continue
        if (it in SIGNAL) != want_signal:
            continue
        if cond not in (None, "normal"):
            continue
        vals = [agg[(gid, jj)] for jj in NEUTRAL if (gid, jj) in agg]
        if vals:
            per[m].append(float(np.mean(vals)))
    return per


def fc_control():
    votes = RUNS / "mini-reason-control-signal" / "pairwise_votes.jsonl"
    rows = [json.loads(l) for l in open(votes)] if votes.exists() else []
    by = defaultdict(dict)
    for r in rows:
        p = r["comparison_id"].split("|")
        by[(r["judge_model"], r["comparison_id"], frozenset([p[1], p[2]]))][r["presentation"]] = r["winner_model"]
    out = {}
    for pair in [{"claude-opus-4-8", "gpt-5.5-noreason"},
                 {"claude-sonnet-4-6", "gpt-5.5-noreason"},
                 {"gpt-5.5", "gpt-5.5-noreason"}]:
        won = Counter()
        for (_, _, fs), v in by.items():
            if fs != pair:
                continue
            if "ab" in v and "ba" in v and v["ab"] == v["ba"]:
                won[v["ab"]] += 1
        out[frozenset(pair)] = won
    return out


def main():
    sig = per_model_means(SIG_DIRS, want_signal=True)
    hard = per_model_means(HARD_DIRS, want_signal=False)
    print("=" * 72)
    print("COMBINED 18-OBJECT NEUTRAL ABSOLUTE (primary aggregate, per-gen mean over 3 judges)")
    print("=" * 72)
    print(f"{'model':22s} {'signal(6)':>12s} {'hard(12)':>12s} {'COMBINED-18':>14s}")
    for m in MODELS:
        s, h = sig.get(m, []), hard.get(m, [])
        c = s + h

        def f(x):
            return f"{np.mean(x):.3f}(n{len(x)})" if x else "--"
        print(f"{m:22s} {f(s):>12s} {f(h):>12s} {f(c):>14s}")
    print("\ncheck (hard bank must match paper): fable 4.963  gpt-5.5 4.958  opus 4.894  sonnet 4.847")

    print("\n" + "=" * 72)
    print("SECONDARY: forced-choice control (signal bank, small n) -- decided rule")
    print("=" * 72)
    labels = {frozenset(["claude-opus-4-8", "gpt-5.5-noreason"]): "noreason vs opus",
              frozenset(["claude-sonnet-4-6", "gpt-5.5-noreason"]): "noreason vs sonnet",
              frozenset(["gpt-5.5", "gpt-5.5-noreason"]): "reason-on vs noreason"}
    for pair, won in fc_control().items():
        tot = sum(won.values())
        parts = ", ".join(f"{m.split('-')[0][:6]}:{n}({n/tot:.2f})" for m, n in won.most_common())
        print(f"  {labels[pair]:22s} decided={tot:2d}  {parts}")


if __name__ == "__main__":
    main()
