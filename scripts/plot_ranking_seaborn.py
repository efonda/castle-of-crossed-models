"""Figure 1 (paper version): full-roster ranking, dot-and-CI plot, seaborn styling.

Same data and bootstrap as scripts/plot_ranking.py (imported); this only restyles it. Reads
from disk, no API. seaborn is optional and not a project dependency. The paper's figure:

    uv run --with seaborn python scripts/plot_ranking_seaborn.py \
        --out ranking-wide.pdf --drop llama3.2:3b --xmin 3 --figsize 8.0,3.2
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot_ranking import ranking  # reuse the exact data builder

FABLE = "claude-fable-5"


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> None:
    with_fable = "--no-fable" not in sys.argv
    show_title = "--title" in sys.argv
    xmin = _arg("--xmin")
    xmin = float(xmin) if xmin is not None else None
    drop = set((_arg("--drop", "") or "").split(",")) - {""}
    # --figsize W,H overrides the default portrait shape. A wide/flat aspect is what a
    # single-column paper wants: width drives height, so a near-square plot costs a page.
    figsize = tuple(float(v) for v in (_arg("--figsize") or "5.8,5.2").split(","))
    out_name = _arg("--out") or ("ranking_seaborn.png" if with_fable else "ranking_seaborn_nofable.png")

    rows = [r for r in ranking(include_fable=with_fable) if r[0] not in drop]  # ascending by mean
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("seaborn/matplotlib not installed — data instead:\n")
        for m, mean, lo, hi in reversed(rows):
            print(f"  {mean:5.2f}  [{lo:.2f}, {hi:.2f}]  {m}")
        print("\nRender with:  uv run --with seaborn python scripts/plot_ranking_seaborn.py")
        return

    labels = [re.sub(r"-\d{8}$", "", m) for m, *_ in rows]  # match Table 1 names
    raw = [m for m, *_ in rows]
    means = [r[1] for r in rows]
    los = [r[1] - r[2] for r in rows]
    his = [r[3] - r[1] for r in rows]
    y = list(range(len(rows)))

    accent, base, capc = "#c1611f", "#3b5168", "#9aa6b2"
    colors = [accent if m == FABLE else base for m in raw]

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(means, y, xerr=[los, his], fmt="none", ecolor=capc,
                elinewidth=1.6, capsize=4, capthick=1.6, zorder=2)
    for yi, mi, ci in zip(y, means, colors):  # halo + point, fable larger
        big = colors[yi] == accent
        ax.scatter(mi, yi, s=150 if big else 90, color=ci, edgecolor="white",
                   linewidth=1.4, zorder=3)
    for yi, mi in zip(y, means):  # value labels to the right of each CI
        ax.annotate(f"{mi:.2f}", (his[yi] + mi, yi), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=accent if colors[yi] == accent else "#55606b",
                    fontweight="bold" if colors[yi] == accent else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    # bold + accent the fable tick label
    for tick, m in zip(ax.get_yticklabels(), raw):
        if m == FABLE:
            tick.set_color(accent); tick.set_fontweight("bold")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    if xmin is not None:
        ax.set_xlim(left=xmin)
    ax.set_xlabel("primary aggregate (0–5), 95% bootstrap CI", labelpad=10)
    ax.set_ylabel("")
    if show_title:
        ax.set_title("Calvino Benchmark — primary aggregate by model\n"
                     "signal run, 3-judge cross-lab panel", loc="left", fontsize=13, pad=12)
    ax.margins(x=0.08)
    sns.despine(left=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()

    out = Path("figures") / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"wrote {out}  ({len(rows)} models{', with fable' if with_fable else ''})")


if __name__ == "__main__":
    main()
