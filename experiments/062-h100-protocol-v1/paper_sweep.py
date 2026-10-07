"""PAPER FIGURES for the WCM-vs-unweighted sweep (DAN-76), from merge.py's OUT/sweep_rows.csv.

  body      paper/figures/wcm-vs-unweighted.pdf           pass rate at lambda = 1e-3 by depth band (bars), and pass rate
                                                           vs circuit size across the sweep for layers 8-11
  appendix  paper/figures/wcm-vs-unweighted-frontier.pdf  the pass-rate / size frontier for every depth band
            paper/figures/wcm-vs-unweighted-zac.pdf       Z / A / C medians at lambda = 1e-3 (the old body figure)
            paper/figures/wcm-vs-unweighted-grid.pdf      depth band x ablation method: median faithfulness + IQR
PNG previews go to OUT. Also prints every number the paper text quotes, so the text can be checked against it.

  OUT=experiments/062-h100-protocol-v1/out PYTHONPATH=src python experiments/062-h100-protocol-v1/paper_sweep.py
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis.style import (BLUE, CATEGORICAL, INK, INK_MUTED, SURFACE, configure_matplotlib,  # noqa: E402
                            grouped_bar_geometry, round_bars, save_figure, styled_legend, tint)

OUT = Path(os.environ.get("OUT", str(HERE / "out")))
FIGS = ROOT / "paper" / "figures"
BANDS = (("L0-4", "Layers 0–4"), ("L5-7", "Layers 5–7"), ("L8-11", "Layers 8–11"))   # merge.py's depth bands
# unweighted colour (Daniel, 2026-09-27; tried in turn: red, teal, honey, pink, grey, now a LIGHTER GREY): red reads
# as C (contrast mean) in the paper; grey marks the baseline. #9aa0a8 vs blue: dE 32.9 normal, 26.2 worst colour-
# blind (lighter separates better than the theme's #6c727c, 26.3 / 22.1), and it sits further from Figure 1's dark
# ink grey for Z. Just under 3:1 contrast on white, relieved by the value printed on every bar.
UNWEIGHTED_GREY = "#9aa0a8"
ARMS = ((True, BLUE, "WCM (fitted coefficients)"), (False, UNWEIGHTED_GREY, "unweighted (α = 1)"))
PROD_LAM = 1e-3                                                    # the production price (protocol v1)
HEAD = (("free0_tk", "zero (Z)"), ("freeM_topk_tk", "activating mean (A)"),
        ("freeN_topk_tk", "contrast mean (C)"))                   # short: the paper caption names each ablation


def curve(rows, weighted):
    """One point per lambda: median nodes, share in band, and the faithfulness quartiles per ablation method."""
    g = rows[rows.weighted == weighted].groupby("lam")
    agg = {"n": ("n", "median"), "in_band": ("in_band", "mean"), "targets": ("seed", "nunique")}
    for col, _ in HEAD:
        agg[col] = (col, "median")
        agg[col + "_lo"] = (col, lambda v: v.quantile(.25))
        agg[col + "_hi"] = (col, lambda v: v.quantile(.75))
    return g.agg(**agg).reset_index().sort_values("n")


def pass_curve(rows, weighted):
    """One point per lambda: median circuit size and the share of targets that pass the DAN-8 rule."""
    g = rows[rows.weighted == weighted].groupby("lam")
    return g.agg(n=("n", "median"), passes=("passes", "mean"), targets=("seed", "nunique")).reset_index().sort_values("n")


def body(plt, sw):
    """BODY FIGURE (2026-09-28, replaces the Z / A / C bars, which now live in the appendix as zac_bars()).
    Left: the share of targets that PASS the paper's criterion (DAN-8: Z, A, C in [0.8, 1.5] and necessity >= 0.9),
    WCM vs unweighted at the production price lambda = 1e-3, grouped bars per depth band, value on every bar and the
    ratio above each pair. The criterion is conjunctive, which is where unweighted circuits fail: their per-score
    medians look only somewhat lower, but they almost never hold all of them at once past layer 4.
    Right: the cost of passing, layers 8-11. Pass rate against median circuit size (log) across the sweep of prices,
    one line per method; an arrow marks the size gap at a matched pass rate."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.2), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.3})
    f = sw[np.isclose(sw.lam, PROD_LAM)]
    width, offsets = grouped_bar_geometry(len(ARMS))
    x = np.arange(len(BANDS))
    vals = {}
    for (weighted, color, name), off in zip(ARMS, offsets):
        v = [100 * f[(f.depth == b) & (f.weighted == weighted)].passes.mean() for b, _ in BANDS]
        vals[weighted] = v
        bars = ax.bar(x + off, v, width, color=color, label=name, zorder=2)
        for bar in bars:                                   # one decimal below 10%, so 2.1 and 1.5 do not both read "2"
            h = bar.get_height()
            ax.annotate(("%.0f" if h >= 10 else "%.1f") % h, (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
    for j in range(len(BANDS)):                                   # the ratio over each pair, where it is dramatic
        w, u = vals[True][j], vals[False][j]
        if u > 0 and w / u >= 2:
            ax.annotate("%.0f×" % (w / u), (x[j], max(w, u) + 9), ha="center", va="bottom", fontsize=9.5,
                        color=INK, fontweight="bold")
    ax.set(xticks=x, xticklabels=["layers " + t.split(" ")[1] for _, t in BANDS], ylim=(0, 100),
           ylabel="targets that pass (%)")
    ax.set_title("Pass rate at the same sparsity penalty (λ = $10^{-3}$)", fontsize=11)
    styled_legend(ax, loc="upper right", fontsize=8.5)
    round_bars(ax)

    rows = sw[sw.depth == "L8-11"]
    curves = {}
    for weighted, color, name in ARMS:
        c = pass_curve(rows, weighted)
        curves[weighted] = c
        bx.plot(c.n, 100 * c.passes, marker="o", color=color, label=name, zorder=3)
    # the size gap at a matched pass rate: WCM's best point against the unweighted point nearest in pass rate
    w_best = curves[True].sort_values("passes").iloc[-1]
    u = curves[False]
    u_match = u.iloc[(u.passes - w_best.passes).abs().argsort().iloc[0]]
    y = 100 * (w_best.passes + u_match.passes) / 2
    bx.annotate("", xy=(u_match.n, y), xytext=(w_best.n, y),
                arrowprops=dict(arrowstyle="<->", color=INK_MUTED, lw=1.2), zorder=2)
    bx.text(np.sqrt(w_best.n * u_match.n), y + 3, "%.0f× more latents" % (u_match.n / w_best.n), ha="center",
            va="bottom", fontsize=9, color=INK)
    bx.set(xscale="log", xlabel="circuit size (median latents)", ylabel="targets that pass (%)", ylim=(0, 60))
    bx.set_title("The cost of passing, layers 8–11", fontsize=11)
    bx.grid(axis="x", visible=False)
    return fig


def frontier(plt, sw):
    """APPENDIX: the pass-rate / size frontier for every depth band (the body shows layers 8-11)."""
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), sharey=True)
    for ax, (band, title) in zip(axes, BANDS):
        rows = sw[sw.depth == band]
        for weighted, color, name in ARMS:
            c = pass_curve(rows, weighted)
            ax.plot(c.n, 100 * c.passes, marker="o", color=color, label=name, zorder=3)
        ax.set(xscale="log", xlabel="circuit size (median latents)", ylim=(0, 100))
        ax.set_title("%s (%d targets)" % (title, rows.seed.nunique()), fontsize=10.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("targets that pass (%)")
    styled_legend(axes[0], loc="upper left", fontsize=8.5)
    return fig


def zac_bars(plt, sw):
    """APPENDIX (the body figure until 2026-09-28): both methods at ONE sparsity price, the production lambda = 1e-3.
    Daniel, 2026-09-27: a fixed price, the Z / A / C scores themselves (not a pass share), and grouped bars (the
    dot-and-IQR version was hard to read). One panel per depth band, x = Z / A / C, one bar per method: the median,
    printed on the bar; median circuit sizes in the panel titles. The spread is in the appendix grid.
    Medians because a few circuits overshoot by orders of magnitude; no average across Z/A/C, which would hide where
    the methods differ."""
    f = sw[np.isclose(sw.lam, PROD_LAM)]
    # drawn near print size (the paper's text width is 5.5 in), so the theme's 10.5-13.5 pt text lands at ~7-9 pt
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.9), sharey=True)
    width, offsets = grouped_bar_geometry(len(ARMS))
    x = np.arange(len(HEAD))
    for ax, (band, title) in zip(axes, BANDS):
        rows = f[f.depth == band]
        ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls=(0, (3, 3)), zorder=1)    # perfect faithfulness, behind the bars
        for (weighted, color, name), off in zip(ARMS, offsets):
            r = rows[rows.weighted == weighted]
            bars = ax.bar(x + off, [r[col].median() for col, _ in HEAD], width, color=color, label=name, zorder=2)
            for bar in bars:                           # value on top; a surface-coloured box keeps it off the 1.0 line
                ax.annotate("%.2f" % bar.get_height(), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9,
                            color=INK, zorder=4, bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
        sizes = [rows[rows.weighted == w].n.median() for w, _, _ in ARMS]
        ax.set_title("%s (%d targets)\n%s vs %s latents" % (title, rows.seed.nunique(), "{:,.0f}".format(sizes[0]),
                                                            "{:,.0f}".format(sizes[1])), fontsize=10.5)
        ax.set(xticks=x, xticklabels=["Z", "A", "C"], ylim=(0, 1.3))
        round_bars(ax)
    axes[0].set_ylabel("held-out faithfulness (median)")
    styled_legend(axes[2], loc="upper right", fontsize=9)
    return fig


def grid(plt, sw):
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 7.6), sharey=True, sharex=True)   # near print size, as in body()
    for r, (band, title) in enumerate(BANDS):
        rows = sw[sw.depth == band]
        for cidx, (col, label) in enumerate(HEAD):
            ax = axes[r, cidx]
            ax.axhspan(0.8, 1.5, color=tint(INK_MUTED, 0.85), zorder=0)      # the DAN-8 pass band
            ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls=(0, (3, 3)), zorder=1)
            for weighted, color, name in ARMS:
                c = curve(rows, weighted)
                ax.fill_between(c.n, c[col + "_lo"], c[col + "_hi"], color=tint(color, 0.75), lw=0, zorder=2)
                ax.plot(c.n, c[col], marker="o", color=color, label=name, zorder=3)
            ax.set_xscale("log")
            ax.set_ylim(-0.1, 1.6)
            ax.set_title("%s: %s" % (title, label) if cidx == 0 else label, fontsize=11.5)
            if r == 2:
                ax.set_xlabel("circuit size (median latents)")
        axes[r, 0].set_ylabel("faithfulness (held-out)")
    styled_legend(axes[0, 0], loc="lower right")
    return fig


def numbers(sw):
    f = sw[np.isclose(sw.lam, PROD_LAM)]
    print("== fixed price (λ = %g, the body figure): per depth band" % PROD_LAM)
    for band, title in BANDS:
        for weighted, _, name in ARMS:
            r = f[(f.depth == band) & (f.weighted == weighted)]
            print("  %-12s %-26s %3d targets | nodes median %5.0f (IQR %5.0f-%5.0f) | in band %3.0f%% | Z %.2f A %.2f C %.2f"
                  % (title, name, r.seed.nunique(), r.n.median(), r.n.quantile(.25), r.n.quantile(.75),
                     100 * r.in_band.mean(), r.free0_tk.median(), r.freeM_topk_tk.median(), r.freeN_topk_tk.median()))
    p = f.pivot_table(index="seed", columns="weighted", values="n")
    ratio = (p[False] / p[True]).rename("r").to_frame().join(f.drop_duplicates("seed").set_index("seed").depth)
    print("  per-target size ratio unweighted / WCM, median: all %.2f | " % ratio.r.median()
          + " | ".join("%s %.2f" % (t, ratio[ratio.depth == b].r.median()) for b, t in BANDS))
    print("  all bands: WCM %.0f nodes %.0f%% in band | unweighted %.0f nodes %.0f%%" % (
        f[f.weighted].n.median(), 100 * f[f.weighted].in_band.mean(),
        f[~f.weighted].n.median(), 100 * f[~f.weighted].in_band.mean()))
    print("== share in band and median size, per arm and depth band")
    for band, title in BANDS:
        rows = sw[sw.depth == band]
        for weighted, _, name in ARMS:
            c = curve(rows, weighted)
            print("  %-12s %-26s " % (title, name) + " | ".join(
                "λ %g: %5.0f nodes %3.0f%%" % (r.lam, r.n, 100 * r.in_band) for r in c.itertuples()))
    c = curve(sw, True), curve(sw, False)
    print("== all bands: WCM", [(r.lam, int(r.n), round(100 * r.in_band)) for r in c[0].itertuples()])
    print("   unweighted   ", [(r.lam, int(r.n), round(100 * r.in_band)) for r in c[1].itertuples()])
    print("== pass rate (DAN-8) per arm and depth band (the body figure)")
    for band, title in BANDS:
        rows = sw[sw.depth == band]
        for weighted, _, name in ARMS:
            c = pass_curve(rows, weighted)
            print("  %-12s %-26s " % (title, name) + " | ".join(
                "λ %g: %5.0f nodes %3.0f%%" % (r.lam, r.n, 100 * r.passes) for r in c.itertuples()))
    print("== cost of passing: smallest PASSING circuit per target, per method (targets where both pass)")
    for band, title in BANDS:
        rows = sw[sw.depth == band]
        cost = []
        for s, g in rows.groupby("seed"):
            pick = lambda w: g[(g.weighted == w) & g.passes].n.min()
            cost.append((pick(True), pick(False)))
        cost = pd.DataFrame(cost, columns=["wcm", "unw"])
        both = cost.dropna()
        print("  %-12s %d targets | WCM passes %d, unweighted %d, both %d | median WCM %.0f, unweighted %.0f, "
              "ratio %.1fx" % (title, len(cost), cost.wcm.notna().sum(), cost.unw.notna().sum(), len(both),
                               both.wcm.median(), both.unw.median(), (both.unw / both.wcm).median()))
    print("== natural-scale cost: smallest circuit with worst-of-3 |faith - 1| <= 0.3, per target")
    for band, title in BANDS:
        rows = sw[sw.depth == band]
        cost = []
        for s, g in rows.groupby("seed"):
            pick = lambda w: g[(g.weighted == w) & (g.dev <= 0.3)].n.min()   # NaN when never reached
            cost.append((pick(True), pick(False)))
        cost = pd.DataFrame(cost, columns=["wcm", "unw"])
        both = cost.dropna()
        print("  %-12s %d targets | WCM reaches %d, unweighted %d, both %d | median WCM %.0f, unweighted %.0f, "
              "ratio %.1fx" % (title, len(cost), cost.wcm.notna().sum(), cost.unw.notna().sum(), len(both),
                               both.wcm.median(), both.unw.median(), (both.unw / both.wcm).median()))


def main():
    sw = pd.read_csv(OUT / "sweep_rows.csv")
    numbers(sw)
    plt = configure_matplotlib()
    FIGS.mkdir(parents=True, exist_ok=True)
    for fig, name in ((body(plt, sw), "wcm-vs-unweighted"), (grid(plt, sw), "wcm-vs-unweighted-grid"),
                      (frontier(plt, sw), "wcm-vs-unweighted-frontier"), (zac_bars(plt, sw), "wcm-vs-unweighted-zac")):
        png = save_figure(fig, OUT / ("%s.png" % name))
        shutil.copy(png.with_suffix(".pdf"), FIGS / ("%s.pdf" % name))
        print("figure:", png, "->", FIGS / ("%s.pdf" % name))

if __name__ == "__main__":
    main()
