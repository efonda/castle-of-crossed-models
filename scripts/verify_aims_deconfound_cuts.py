#!/usr/bin/env python3
"""Frontier gap under different judge exclusions (paper §3, App. F).

Recomputes the gpt-5.5 minus Anthropic-model gap on the absolute rubric for the headline panel
and for each way of excluding judges (same-lab judges removed; the OpenAI judge removed), using
the headline-panel loader from verify_self_preference. No API calls.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from verify_self_preference import BASE_PANEL, load_scores  # noqa: E402

GPT, SON, OPUS = "gpt-5.5", "claude-sonnet-4-6", "claude-opus-4-8-api"
OPUS_MODEL = "claude-opus-4-8"  # the contestant; OPUS is the judge alias
RNG = np.random.default_rng(0)


def row_mean(sc, model, judges):
    sub = sc[(sc.model == model) & (sc.judge_model.isin(judges))]
    # generation-level mean over the named judges, then mean over generations
    g = sub.groupby(["generation_id", "item_id"]).primary.mean().reset_index()
    return g


def gap(a, b, draws=5000):
    """Paired item-cluster bootstrap of mean(a) - mean(b) over shared items."""
    items = sorted(set(a.item_id) & set(b.item_id))
    ga = a.groupby("item_id").primary.mean()
    gb = b.groupby("item_id").primary.mean()
    d = np.array([ga[i] - gb[i] for i in items])
    point = a.primary.mean() - b.primary.mean()
    boot = np.array([RNG.choice(d, size=len(d), replace=True).mean() for _ in range(draws)])
    # two-sided bootstrap p for H0: diff = 0
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return point, np.percentile(boot, [2.5, 97.5]), max(p, 1 / draws), len(items)


def main():
    sc = load_scores()
    sc = sc[sc.judge_model.isin(BASE_PANEL)]
    anth = [OPUS, SON]

    cuts = {
        "(0) full panel, both rows (published)": (BASE_PANEL, BASE_PANEL),
        "(a) asymmetric: each model by non-own-lab judges": (anth, [GPT]),
        "(b) symmetric: gpt-5.5 judge dropped from both rows": (anth, anth),
        "(c) gpt's self-vote dropped from gpt's row only": (anth, BASE_PANEL),
    }
    print("gpt-5.5 minus claude-sonnet-4-6, absolute primary aggregate, bank 1")
    print(f"{'cut':<52}{'gpt':>7}{'son':>7}{'delta':>8}   95% CI            p")
    for label, (jg, js) in cuts.items():
        a, b = row_mean(sc, GPT, jg), row_mean(sc, SON, js)
        pt, ci, p, n = gap(a, b)
        print(
            f"{label:<52}{a.primary.mean():>7.3f}{b.primary.mean():>7.3f}{pt:>8.3f}"
            f"   [{ci[0]:+.3f}, {ci[1]:+.3f}]   {p:.3f}  (items {n})"
        )

    print("\nsame four cuts, gpt-5.5 minus claude-opus-4-8")
    for label, (jg, js) in cuts.items():
        a, b = row_mean(sc, GPT, jg), row_mean(sc, OPUS_MODEL, js)
        pt, ci, p, n = gap(a, b)
        print(
            f"{label:<52}{a.primary.mean():>7.3f}{b.primary.mean():>7.3f}{pt:>8.3f}"
            f"   [{ci[0]:+.3f}, {ci[1]:+.3f}]   {p:.3f}"
        )

    print("\nWHO JUDGES WHOM under each cut (own-lineage judges / judges on that row)")
    for label, (jg, js) in cuts.items():
        def share(model_fam, judges):
            own = [j for j in judges if ("gpt" in j) == (model_fam == "openai")]
            return f"{len(own)}/{len(judges)}"
        print(
            f"  {label:<52} gpt-5.5 {share('openai', jg)} own"
            f"   |  the Anthropic pair {share('anthropic', js)} own"
        )


if __name__ == "__main__":
    raise SystemExit(main())
