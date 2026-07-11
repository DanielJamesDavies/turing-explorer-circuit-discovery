"""Fast figure-style tester: every analysis chart archetype on fake data.

Renders a single contact-sheet PNG (plus individual panels) using the real
analysis.style helpers, so any edit to the style module shows up here in
seconds instead of the ~15 min real-data regeneration.

Run from the repo root:
    PYTHONPATH=src python -m debug.style_tester [--out DIR]

Output defaults to ./style-tester/ in the current working directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.style import (
    BAR_WIDTH,
    BLUE,
    CATEGORICAL,
    INK_MUTED,
    METHOD_COLORS,
    NEG_MODE_COLORS,
    SEQUENTIAL_CMAP,
    SERIES2,
    annotate_bars,
    configure_matplotlib,
    grouped_bar_geometry,
    integer_ticks,
    ordinal_blues,
    round_bars,
    styled_boxplot,
    styled_legend,
    tint,
    value_labels,
)

rng = np.random.default_rng(11)

METHODS = ["counterfactual", "ablation", "hybrid"]
MODES = ["close", "random", "distant"]


def grouped_bars(ax):
    x = np.arange(3)
    width, offsets = grouped_bar_geometry(3)
    for offset, (method, color) in zip(offsets, METHOD_COLORS.items()):
        ax.bar(x + offset, rng.uniform(0.4, 0.95, 3), width=width, color=color, label=method.split("_")[0])
    ax.set_xticks(x)
    ax.set_xticklabels(MODES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Grouped Bars + Value Keys")
    styled_legend(ax, loc="upper left")
    round_bars(ax)


def paired_count_bars(ax):
    x = np.arange(3)
    ax.bar(x - 0.18, [1, 4, 9], width=0.3, color=SERIES2[0], label="p50")
    ax.bar(x + 0.18, [4, 11, 20], width=0.3, color=SERIES2[1], label="p90")
    ax.set_xticks(x)
    ax.set_xticklabels(["1 hop", "2 hop", "3 hop"])
    ax.set_ylabel("Reachable latents")
    ax.set_title("Paired Bars + Labels")
    styled_legend(ax, loc="upper left")
    integer_ticks(ax)
    annotate_bars(ax, ".0f")
    round_bars(ax)


def single_bars(ax):
    vals = [0.77, 0.73, 0.72]
    bars = ax.bar(MODES, vals, width=BAR_WIDTH, color=BLUE)
    value_labels(ax, bars)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Single-Series Bars")
    round_bars(ax)


def two_tone_stack(ax):
    hard = np.array([0.20, 0.31, 0.19])
    soft = np.array([0.26, 0.24, 0.20])
    ax.bar(MODES, hard, width=0.55, color=BLUE, label="Very likely")
    ax.bar(MODES, soft, width=0.55, bottom=hard, color=tint(BLUE), label="Likely")
    ax.set_ylim(0, 0.65)
    ax.set_title("Two-Tone Stacks")
    styled_legend(ax, loc="upper right")
    round_bars(ax)


def kde_lines(ax):
    xs = np.linspace(-0.3, 1.7, 250)
    for mu, (method, color) in zip([0.55, 0.78, 0.9], METHOD_COLORS.items()):
        ax.plot(xs, np.exp(-((xs - mu) ** 2) / 0.05), color=color, label=method.split("_")[0])
        ax.axvspan(mu - 0.2, mu + 0.2, color=color, alpha=0.10)
        ax.axvline(mu, color=color, linestyle="--", linewidth=1.1)
    ax.set_xlabel("Eval score")
    ax.set_ylabel("Density")
    ax.set_title("KDE + Bands")
    styled_legend(ax, loc="upper right")


def overlay_hist(ax):
    ax.hist(rng.normal(0, 1, 3000), bins=45, alpha=0.6, color=SERIES2[0], label="same")
    ax.hist(rng.normal(0.8, 1.2, 3000), bins=45, alpha=0.6, color=SERIES2[1], label="cross")
    ax.axvline(0.4, color=INK_MUTED, linestyle="--", linewidth=1.2, label="threshold")
    ax.set_xlabel("PMI")
    ax.set_ylabel("Pair count")
    ax.set_title("Overlay Histogram")
    styled_legend(ax, loc="upper right")


def boxes(ax):
    data = [rng.normal(mu, s, 80) for mu, s in ((0.62, 0.18), (0.74, 0.13), (0.7, 0.2))]
    styled_boxplot(ax, data, MODES, list(NEG_MODE_COLORS.values()))
    ax.set_ylabel("Faithfulness")
    ax.set_title("Boxplots by Mode")


def scatter(ax):
    x = rng.uniform(0, 1, 120)
    y = np.clip(x * 0.8 + rng.normal(0, 0.1, 120), 0, 1.1)
    ax.scatter(x, y, s=22, color=BLUE, alpha=0.7, edgecolors="none")
    ax.plot([0, 1], [0, 0.8], color=INK_MUTED, linestyle="--", linewidth=1.4)
    ax.set_xlabel("Faithfulness")
    ax.set_ylabel("Suppression")
    ax.set_title("Scatter + Reference")


def ordinal_stack(ax):
    buckets = ["singleton", "rare", "shared", "common"]
    colors = ordinal_blues(len(buckets))
    n = 40
    parts = rng.dirichlet(np.ones(4) * 3, n) * 100
    bottoms = np.zeros(n)
    for i, (label, color) in enumerate(zip(buckets, colors)):
        ax.bar(range(n), parts[:, i], bottom=bottoms, width=1.0, color=color, label=label)
        bottoms += parts[:, i]
    ax.set_xticks([])
    ax.set_ylabel("Circuit latents (%)")
    ax.set_title("Ordinal Blue Ramp")
    styled_legend(ax, loc="upper right")


def heatmap(ax):
    matrix = rng.uniform(0, 1, (24, 24))
    im = ax.imshow(matrix, cmap=SEQUENTIAL_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xlabel("Coacting component")
    ax.set_ylabel("Target component")
    ax.set_title("Heatmap (blue->red)")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)


def swatches(ax):
    for i, color in enumerate(CATEGORICAL):
        ax.bar(i, 1.0, width=0.8, color=color)
    ax.set_xticks(range(len(CATEGORICAL)))
    ax.set_xticklabels([f"{i + 1}" for i in range(len(CATEGORICAL))])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Categorical Palette")
    round_bars(ax)


PANELS = [
    grouped_bars,
    paired_count_bars,
    single_bars,
    two_tone_stack,
    kde_lines,
    overlay_hist,
    boxes,
    scatter,
    ordinal_stack,
    heatmap,
    swatches,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render fake-data style specimens.")
    parser.add_argument("--out", type=Path, default=Path("style-tester"), help="Output directory.")
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    plt = configure_matplotlib()

    cols = 3
    rows = (len(PANELS) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 3.6))
    flat = axes.flatten()
    for panel, ax in zip(PANELS, flat):
        panel(ax)
    for ax in flat[len(PANELS):]:
        ax.set_visible(False)
    fig.suptitle("Analysis Style Tester", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "contact_sheet.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    for panel in PANELS:
        f, a = plt.subplots(figsize=(7, 4.5))
        panel(a)
        f.tight_layout()
        f.savefig(out / f"{panel.__name__}.png", bbox_inches="tight", dpi=110)
        plt.close(f)

    print(f"wrote contact_sheet.png + {len(PANELS)} panels to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
