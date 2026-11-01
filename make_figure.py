#!/usr/bin/env python3
"""
make_figure.py

Draws Figure 1 of the paper from the run folders:
  left:  prompts recovered (of 6) at each dose, for the three arms
  right: share of runs recovered, grouped by the widest blend reached in the window

    python make_figure.py                  # reads runs/sweep, runs/matched, runs/rescaled
    python make_figure.py --runs other     # a different runs folder

Writes figures/figure1.png (and .svg). Needs numpy and matplotlib.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [  # folder, condition prefix, label, color, marker
    ("sweep", "dose_", "Blended (collapse removed)", "#2a78d6", "o"),
    ("matched", "sampled_", "Committed (token drawn from the same distribution)", "#eb6834", "s"),
    ("rescaled", "rescaled_", "Blended, length restored", "#1baf7a", "^"),
]
BINS = [("Under 50", 0, 50), ("Hundreds", 50, 1000), ("Thousands", 1000, float("inf"))]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--runs", default=str(here / "runs"))
    ap.add_argument("--out", default=str(here / "figures" / "figure1"))
    args = ap.parse_args()
    R = Path(args.runs)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)

    bar_w = 0.26
    for k, (folder, prefix, label, color, marker) in enumerate(ARMS):
        runs = [r for r in json.loads((R / folder / "summary.json").read_text(encoding="utf-8"))["runs"]
                if r["condition"].startswith(prefix)]
        doses = sorted({float(r["condition"][len(prefix):]) for r in runs})
        rec = [sum(r["after_window"]["recovered"] for r in runs
                   if float(r["condition"][len(prefix):]) == d) for d in doses]
        # small sideways offset so arms that land on the same value stay visible
        ax1.plot([d + (k - 1) * 0.012 for d in doses], rec, color=color, lw=2, marker=marker, ms=8,
                 mec=SURFACE, mew=2, label=label, zorder=3 + k)
        # bins by peak width
        for j, (name, lo, hi) in enumerate(BINS):
            sel = [r for r in runs if lo <= (r["after_window"]["max_blend_width_in_window"] or 0) < hi]
            n, ok = len(sel), sum(r["after_window"]["recovered"] for r in sel)
            x = j + (k - 1) * (bar_w + 0.03)
            if n:
                ax2.bar(x, ok / n, width=bar_w, color=color, zorder=3)
            ax2.text(x, (ok / n if n else 0) + 0.025, f"{ok}/{n}", ha="center", va="bottom",
                     fontsize=8.5, color=INK2)

    ax1.set(xlabel="Temperature in the 20-step window (dose)", ylabel="Prompts recovered (of 6)",
            ylim=(-0.3, 6.5), yticks=range(0, 7), xticks=doses)
    ax1.set_title("Recovery after the window, by dose", loc="left", color=INK, fontsize=11)
    ax1.grid(axis="y", color=GRID, lw=0.8, zorder=0)

    ax2.set_xticks(range(len(BINS)), [f"{b[0]}" for b in BINS])
    ax2.set(xlabel="Widest distribution in the window (effective tokens)", ylabel="Share of runs recovered",
            ylim=(0, 1.12), yticks=[0, 0.25, 0.5, 0.75, 1.0], yticklabels=["0%", "25%", "50%", "75%", "100%"])
    ax2.set_title("Recovery by how wide the distribution got", loc="left", color=INK, fontsize=11)
    ax2.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax2.tick_params(axis="x", length=0)

    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".svg"), facecolor=SURFACE)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
