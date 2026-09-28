"""TRAINING CURVES of the protocol-v1 fits (train.shard*.jsonl, written by driver.py for every fresh fit).

Figure 1 (OUT/training_curves.png + .pdf), one row of three panels:
  1. circuit size (latents above the keep threshold) per step, median and interquartile band per depth band
  2. data loss (the Z + C + A terms; the sparsity penalty excluded) per step, same bands
  3. how much each layer's circuits still shrank over the last 100 steps (median, interquartile bar)
Figure 2 (OUT/loss_curves.png + .pdf): the total training loss per depth band, the data terms' share of it, and the
data loss split into its Z / C / A terms (plus the rank-keep part inside them).
It also prints the numbers behind the figure, and whether circuits still shrinking at the end are less faithful on
held-out strongest contexts (within each depth band, so depth does not confound it).

  OUT=experiments/062-h100-protocol-v1/out PYTHONPATH=src python experiments/062-h100-protocol-v1/curves.py
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from analysis.style import (CATEGORICAL, INK_MUTED, INK_SECONDARY, configure_matplotlib, panel_figsize,  # noqa: E402
                            save_figure, styled_legend, tint)

OUT = Path(os.environ.get("OUT", str(HERE / "out")))
BANDS = (("layers 0-3", range(0, 4)), ("layers 4-7", range(4, 8)), ("layers 8-11", range(8, 12)))
HEAD = ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")
BAND = (0.8, 1.5)          # the DAN-8 pass band (2026-09-27); "in band" below also requires necessity >= NEC_MIN
NEC_MIN = 0.9
LAST = 100                                                   # the "still shrinking" window, in steps
PERIOD = 12            # batches of 4 cycle through the 48 training contexts in a fixed order: one pass = 12 steps
# near print size (the paper's text width is 5.5 in) with smaller titles and labels, so the text lands at ~6-8 pt
PRINT_SIZE = (9.5, 3.2)
PRINT_RC = {"axes.titlesize": 11, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
            "xtick.labelsize": 9, "ytick.labelsize": 9}
FIGS = HERE.parents[1] / "paper" / "figures"
# depth is ordered: a light-to-dark blue ramp (shallow -> deep) from the theme's blue ramp, not categorical hues. Red
# is reserved for C (contrast mean) and appears in the loss figure's term panel (Daniel's rule: no red elsewhere).
BAND_COLORS = ("#8aa4ff", "#0044ff", "#002894")
# data-loss terms in Figure 1's colours: Z ink grey, C (contrast mean) red, A (activating mean) green
TERMS = {"zero": ("Z (zero)", INK_SECONDARY), "floor": ("0.25 × C (contrast mean)", CATEGORICAL[1]),
         "pos": ("0.25 × A (activating mean)", CATEGORICAL[3])}


def rows(pattern):
    out = []
    for f in glob.glob(str(OUT / "main" / pattern)):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load():
    fits = {}
    for r in rows("train.shard*.jsonl"):
        if r.get("curve") and r["curve"].get("n_members"):
            fits[r["seed"]] = r                              # a refit after a kill repeats the row: keep the last
    ev = {r["seed"]: r for r in rows("eval.shard*.jsonl") if r.get("held") == "strong" and "error" not in r}
    keys = sorted(fits)
    layer = np.array([int(k.split(".")[0]) for k in keys])
    size = np.array([fits[k]["curve"]["n_members"] for k in keys], dtype=float)
    loss = np.array([fits[k]["curve"]["loss"] for k in keys], dtype=float)
    data = loss - np.array([fits[k]["curve"]["penalty"] for k in keys], dtype=float)
    # the engine's per-term curves, already weighted: zero = Z, floor = C (contrast mean, x gamma_C),
    # pos = A (activating mean, x gamma_A); rank = the rank-keep part inside them, before the term weights
    terms = {t: np.array([fits[k]["curve"]["terms"][t] for k in keys], dtype=float) for t in TERMS}
    rank = np.array([fits[k]["curve"]["rank"] for k in keys], dtype=float)
    # the DAN-8 pass rule; a missing score (None) fails, as in merge.py (pandas compares NaN as False)
    nec = lambda r: r.get("phi_sup_blind_tk") is not None and r["phi_sup_blind_tk"] >= NEC_MIN
    in_band = np.array([all(ev[k][h] is not None and BAND[0] <= ev[k][h] <= BAND[1] for h in HEAD) and nec(ev[k])
                        if k in ev else np.nan for k in keys], dtype=float)
    return keys, layer, size, loss, data, terms, rank, in_band


def band_line(ax, steps, values, color, label):
    lo, med, hi = np.percentile(values, [25, 50, 75], axis=0)
    ax.fill_between(steps, lo, hi, color=tint(color, 0.75), linewidth=0, zorder=1)
    ax.plot(steps, med, color=color, label=label, zorder=2)


def per_pass(x):
    """Average each curve over one pass through the training contexts (removes the batch-difficulty sawtooth)."""
    return np.array([np.convolve(r, np.ones(PERIOD) / PERIOD, mode="valid") for r in x])


def loss_figure(plt, layer, loss, data, terms, rank):
    """Second figure: the total training loss, its data share, and the data loss split into its terms."""
    steps = np.arange(loss.shape[1])
    sm = steps[PERIOD - 1:]
    fig, axes = plt.subplots(1, 3, figsize=PRINT_SIZE)
    share = 100 * per_pass(data) / per_pass(loss)
    for (name, ls), color in zip(BANDS, BAND_COLORS):
        m = np.isin(layer, list(ls))
        band_line(axes[0], steps, loss[m], color, "%s (n %d)" % (name, m.sum()))
        band_line(axes[1], sm, np.clip(share[m], 1e-4, None), color, name)
    # explanatory notes (per-pass averaging, "the rest is the penalty") live in the paper caption, not on the axes
    axes[0].set(yscale="log", xlabel="training step", ylabel="loss (data + penalty)")
    axes[0].set_title("Training loss")
    styled_legend(axes[0], loc="upper right")
    axes[1].set(yscale="log", xlabel="training step", ylabel="data share of the loss (%)", ylim=(2e-3, 1.5))
    axes[1].set_title("Data share of the loss")

    for t, (label, color) in TERMS.items():
        band_line(axes[2], sm, np.clip(per_pass(terms[t]), 1e-5, None), color, label)
    band_line(axes[2], sm, np.clip(per_pass(rank), 1e-5, None), CATEGORICAL[6], "rank-keep (inside the terms)")
    axes[2].set(yscale="log", xlabel="training step", ylabel="data loss term", ylim=(1e-5, 30.0))
    axes[2].set_title("Data loss by ablation term")
    styled_legend(axes[2], loc="upper right", fontsize=8)          # the top right is empty above the curves

    final = {t: np.median(per_pass(terms[t])[:, -1]) for t in TERMS}
    print("final data-loss terms (median, last pass): " + ", ".join("%s %.4f" % (TERMS[t][0], v) for t, v in final.items())
          + ", rank-keep %.5f" % np.median(per_pass(rank)[:, -1]))
    print("data share of the loss at the end: median %.2f%%" % np.median(share[:, -1]))
    png = save_figure(fig, OUT / "loss_curves.png")
    print("figure:", png)
    return png


def main():
    keys, layer, size, loss, data, terms, rank, in_band = load()
    steps = np.arange(size.shape[1])
    shrink = 1.0 - size[:, -1] / np.maximum(size[:, -1 - LAST], 1.0)

    print("fits with a training curve: %d (%d steps each)" % (len(keys), size.shape[1]))
    for s in (0, 50, 100, 150, 200, 300, size.shape[1] - 1):
        print("  step %3d  members median %8.0f  (p10 %7.0f, p90 %8.0f)" % (
            s, np.median(size[:, s]), np.percentile(size[:, s], 10), np.percentile(size[:, s], 90)))
    reach = np.array([int(np.argmax(x < 2 * x[-1])) for x in size])
    print("  step at which a circuit is first within 2x of its final size: median %d (p10 %d, p90 %d)" % (
        np.median(reach), np.percentile(reach, 10), np.percentile(reach, 90)))
    print("shrink over the last %d steps: median %.1f%%, p90 %.1f%%; still shrinking >5%%: %.0f%%, >10%%: %.0f%%" % (
        LAST, 100 * np.median(shrink), 100 * np.percentile(shrink, 90), 100 * (shrink > .05).mean(), 100 * (shrink > .10).mean()))
    print("pass rate (DAN-8: Z/A/C in [%.1f, %.1f], necessity >= %.1f; held-out strongest) by last-%d-step shrink, "
          "within depth bands:" % (BAND[0], BAND[1], NEC_MIN, LAST))
    for name, ls in BANDS:
        m = np.isin(layer, list(ls)) & ~np.isnan(in_band)
        cut = np.median(shrink[m])
        lo, hi = m & (shrink <= cut), m & (shrink > cut)
        print("  %-12s settled half %.0f%% in band (n %d) | still-shrinking half %.0f%% (n %d)" % (
            name, 100 * np.nanmean(in_band[lo]), lo.sum(), 100 * np.nanmean(in_band[hi]), hi.sum()))

    plt = configure_matplotlib()
    plt.rcParams.update(PRINT_RC)
    fig, axes = plt.subplots(1, 3, figsize=PRINT_SIZE)
    # Each step's data loss is ONE batch of 4 training contexts, and the batches cycle through the 48 in a fixed order
    # (period 12 steps), so the raw per-step curve is a sawtooth of batch difficulty. Average over each full pass.
    smooth = per_pass(data)
    for (name, ls), color in zip(BANDS, BAND_COLORS):
        m = np.isin(layer, list(ls))
        band_line(axes[0], steps, size[m], color, "%s (n %d)" % (name, m.sum()))
        band_line(axes[1], steps[PERIOD - 1:], np.clip(smooth[m], 1e-5, None), color, name)
    axes[0].set(yscale="log", xlabel="training step", ylabel="latents in the circuit")
    axes[0].set_title("Circuit size during training")
    styled_legend(axes[0], loc="upper right")
    axes[1].set(yscale="log", xlabel="training step", ylabel="data loss (Z + C + A)", ylim=(3e-3, 1.0))
    axes[1].set_title("Data loss during training")          # per-pass averaging is stated in the caption

    med = [100 * np.median(shrink[layer == L]) for L in range(12)]
    q1 = [100 * np.percentile(shrink[layer == L], 25) for L in range(12)]
    q3 = [100 * np.percentile(shrink[layer == L], 75) for L in range(12)]
    axes[2].vlines(range(12), q1, q3, color=tint(CATEGORICAL[0], 0.55), linewidth=2.2, zorder=1)
    axes[2].plot(range(12), med, "o", color=CATEGORICAL[0], markersize=7, zorder=2)
    axes[2].set(xlabel="target layer", ylabel="removed in last %d steps (%%)" % LAST,
                xticks=range(0, 12, 2), ylim=(0, None))
    axes[2].set_title("Still shrinking at step %d" % size.shape[1])      # median + IQR: stated in the caption
    png = save_figure(fig, OUT / "training_curves.png")
    print("figure:", png)
    png2 = loss_figure(plt, layer, loss, data, terms, rank)
    if os.environ.get("PAPER"):                                    # PAPER=1: copy the vector figures into the paper
        FIGS.mkdir(parents=True, exist_ok=True)
        shutil.copy(png.with_suffix(".pdf"), FIGS / "training-curves.pdf")
        shutil.copy(png2.with_suffix(".pdf"), FIGS / "loss-curves.pdf")
        print("copied to", FIGS)


if __name__ == "__main__":
    main()
