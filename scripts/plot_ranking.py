"""Figure 1 data: full-roster ranking with bootstrap 95% CIs (signal run, headline panel).

Reads from disk (no API). matplotlib is optional and not a project dependency:

    uv run --with matplotlib python scripts/plot_ranking.py   # -> figures/ranking.png

Without matplotlib it prints the table the figure is built from.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calvino import analysis as A
from calvino.analysis import PRIMARY_DIMS
from calvino.storage import Store

PANEL = {"claude-opus-4-8-api", "claude-sonnet-4-6", "gpt-5.5"}
CLUSTER = {"gpt-5.5", "claude-opus-4-8", "claude-sonnet-4-6"}
# Published 11-model signal roster (pinned: an interrupted judge run left stray panel
# scores in signal/ for the local judge gemma4:26b and a partial gemma4:e26b).
ROSTER = {
    "gpt-5.5", "claude-opus-4-8", "claude-sonnet-4-6", "gpt-5.4-mini",
    "gemini-3.1-pro-preview", "gemini-3.5-flash", "claude-haiku-4-5-20251001",
    "gemini-3.1-flash-lite", "gemma4:e4b", "gpt-oss:20b", "llama3.2:3b",
}
RNG = np.random.default_rng(0)


def _panel_gens(data_dir: str):
    """Per-generation primary on the headline panel, condition=normal, first 3 samples.

    sample_idx < 3 keeps every model on the published 3-sample footing and reproduces
    the locked numbers regardless of the trio/flash forced-choice resample (and the
    back-fill from an interrupted judge run) sitting in signal/.
    """
    df = A.build_scores_frame(Store(data_dir))
    df = df[df.judge_model.isin(PANEL) & (df.condition == "normal") & (df.sample_idx < 3)].copy()
    df["primary"] = df[list(PRIMARY_DIMS)].mean(axis=1)
    return df.groupby(["model", "generation_id"])["primary"].mean().reset_index()


def ranking(include_fable: bool = False) -> list[tuple[str, float, float, float]]:
    gen = _panel_gens("data/runs/signal")
    gen = gen[gen.model.isin(ROSTER)]  # drop fable's partial signal scores + stragglers
    if include_fable:
        import pandas as pd
        gen = pd.concat([gen, _panel_gens("data/runs/signal_fable")], ignore_index=True)
    out = []
    for m, g in gen.groupby("model"):
        v = g["primary"].to_numpy()
        boots = [np.mean(RNG.choice(v, len(v), replace=True)) for _ in range(2000)]
        out.append((m, float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))))
    return sorted(out, key=lambda r: r[1])  # ascending for bottom-up plot


def main() -> None:
    with_fable = "--with-fable" in sys.argv
    out_name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else (
        "ranking_fable.png" if with_fable else "ranking.png")
    rows = ranking(include_fable=with_fable)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — printing the figure's data instead:\n")
        for m, mean, lo, hi in reversed(rows):
            print(f"  {mean:5.2f}  [{lo:.2f}, {hi:.2f}]  {m}")
        print("\nRender with:  uv run --with matplotlib python scripts/plot_ranking.py")
        return

    import re
    # Normalize display names to match the paper's Table 1 (strip trailing -YYYYMMDD
    # date suffixes, e.g. claude-haiku-4-5-20251001 -> claude-haiku-4-5).
    labels = [re.sub(r"-\d{8}$", "", m) for m, *_ in rows]
    means = [mean for _, mean, _, _ in rows]
    los = [mean - lo for _, mean, lo, _ in rows]
    his = [hi - mean for _, mean, _, hi in rows]

    n = len(labels)
    # DATA ONLY: points + 95% CIs, ordered by mean. No grouping/tie/floor overlays —
    # interpretation (cluster, tie, bound) is left to the paper text, not the figure.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    y = list(range(n))
    # DATA ONLY: points + 95% CIs, ordered by mean. No grouping/tie/floor overlays.
    ax.errorbar(means, y, xerr=[los, his], fmt="none", capsize=3, ecolor="#999")
    colors = ["#b5651d" if lbl == "claude-fable-5" else "#34495e" for lbl in labels]
    ax.scatter(means, y, color=colors, s=28, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("primary aggregate (0–5), 95% bootstrap CI\n(signal run, 3-judge cross-lab panel)")
    ax.set_title("Primary aggregate by model")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = Path("figures") / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}  ({n} models{', with fable' if with_fable else ''})")


if __name__ == "__main__":
    main()
