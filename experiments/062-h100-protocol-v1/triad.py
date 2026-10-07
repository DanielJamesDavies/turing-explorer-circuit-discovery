"""DAN-20: the coefficient triad on the protocol-v1 circuits (2026-10-07).

Three versions of a circuit, each scored with the production scorer on the held-out strongest contexts:
  fitted   the production circuits as fitted (the full run, ~15k targets)
  alpha=1  the same nodes with every coefficient set to 1 (rescore_alpha1.py: 1,746 targets, stratified by layer and
           site kind)
  random   random circuits of the same size, site by site, drawn from latents live on the target's training contexts,
           with coefficients fitted by the production objective (random_null.py: 747 targets, stratified by depth band
           and site kind)
The three samples are each stratified from the same full run (they share only 81 targets), so each is reported with
its own n. Also the overshoot analysis: how often the same nodes at alpha = 1 overshoot the band (> 1.5), or fall below
the empty circuit's level (score < 0), by ablation method, depth band and site kind.

  OUT=experiments/062-h100-protocol-v1/out_full/out python experiments/062-h100-protocol-v1/triad.py
      prints the numbers; FIG=1 also writes paper/figures/coefficient-triad.pdf (preview PNG in out_alpha1/)
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import merge as M  # noqa: E402
import paper_main as P  # noqa: E402

HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
GROUPS = [("depth", "L0-4", "layers\n0–4"), ("depth", "L5-7", "layers\n5–7"), ("depth", "L8-11", "layers\n8–11"),
          ("kind", "resid", "resid."), ("kind", "mlp", "MLP"), ("kind", "attn", "attn.")]


def rows(path):
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    for c in ("error", "skip"):
        if c in d:
            d = d[d[c].isna()]
    return d


def tag(d):
    d = d.copy()
    d["layer"] = [int(k.split(".")[0]) for k in d.seed]
    d["kind"] = [k.split(".")[1] for k in d.seed]
    d["depth"] = d.layer.map(M.depth_band)
    return d


def load():
    full, _ = P.load()
    full = full.reset_index().rename(columns={"index": "seed"}) if "seed" not in full else full
    a1 = tag(rows(HERE / "out_alpha1" / "alpha1.jsonl").drop_duplicates("seed", keep="last"))
    a1 = a1[~M.vacuous(a1)].copy()
    a1["passes"] = M.passes(a1)
    rn = tag(rows(HERE / "out_random_scale" / "random.jsonl"))
    rn = rn[~M.vacuous(rn)].copy()
    rn["passes"] = M.passes(rn)
    return full, a1, rn


def numbers(full, a1, rn):
    print("n: fitted %d | alpha = 1 %d | random %d" % (len(full), len(a1), len(rn)))
    for name, d in (("fitted", full), ("alpha=1", a1), ("random", rn)):
        print("%-8s pass %.1f%% | " % (name, 100 * d.passes.mean())
              + " ".join("%s %.1f%%" % (g, 100 * d[d[k] == g].passes.mean()) for k, g, _ in GROUPS)
              + " | Z/A/C medians %.2f / %.2f / %.2f" % tuple(d[h].median() for h in HEAD))
    # overshoot / below the empty circuit at alpha = 1
    for rd in ("tk", "pre"):
        cols = [h.replace("_tk", "_" + rd) for h in HEAD]
        over = (a1[cols] > 1.5).any(axis=1)
        print("alpha = 1, %s read: above 1.5 under any ablation %d (%.1f%%, kinds %s); below 0 (under the empty circuit): "
              "Z %.1f%%, A %.1f%%, C %.1f%%, any %.1f%%" % (
                  rd, over.sum(), 100 * over.mean(), a1[over].kind.value_counts().to_dict(),
                  *[100 * (a1[c] < 0).mean() for c in cols], 100 * (a1[cols] < 0).any(axis=1).mean()))
    neg = a1.freeM_topk_tk < 0
    print("below the empty circuit under A, activation read: by depth " + ", ".join(
        "%s %.1f%%" % (b, 100 * neg[a1.depth == b].mean()) for b in ("L0-4", "L5-7", "L8-11")) + "; by kind " + ", ".join(
        "%s %.1f%%" % (k, 100 * neg[a1.kind == k].mean()) for k in ("resid", "mlp", "attn")))
    m = a1[neg]
    print("  there: target natural %.2f, empty circuit (A fill) %.2f, members alone at alpha = 1 %.2f (medians)" % (
        m.a_pos_tk.median(), m.eM_topk_tk.median(), m.freeM_topk_raw_tk.median()))
    fit = full.set_index("seed").passes
    pf = fit.reindex(a1.seed).values
    print("  fitted pass among them %.0f%% vs the other alpha = 1 targets %.0f%%" % (
        100 * np.nanmean(pf[neg.values]), 100 * np.nanmean(pf[~neg.values])))


def figure(full, a1, rn):
    from analysis.style import BLUE, INK, INK_SECONDARY, configure_matplotlib, save_figure, tint
    plt = configure_matplotlib()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.titlepad": 4.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "legend.fontsize": 6.3, "axes.linewidth": 0.7,
                         "grid.linewidth": 0.5, "axes.labelpad": 2.0})
    conds = (("fitted", full, BLUE, "fitted coefficients"), ("alpha=1", a1, "#9aa0a8", "same nodes, $\\alpha = 1$"),
             ("random", rn, "#d3d6db", "random nodes, fitted coefficients"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.25), gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.28})
    w = 0.26
    xs = np.array([0, 1, 2, 3.4, 4.4, 5.4])
    for j, (name, d, col, lab) in enumerate(conds):
        vals = [100 * d[d[k] == g].passes.mean() for k, g, _ in GROUPS]
        x = xs + (j - 1) * w
        ax.bar(x, vals, w * 0.92, color=col, label=lab, zorder=2, linewidth=0)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1.2, ("%.0f" if v >= 9.5 else "%.1f" if v > 0 else "0") % v, ha="center", va="bottom",
                    fontsize=5.2, color=INK, rotation=90 if v < 9.5 else 0)
    ax.set_xticks(xs); ax.set_xticklabels([lab for _, _, lab in GROUPS])
    ax.set(ylim=(0, 80), ylabel="targets that pass (%)")
    ax.grid(axis="x", visible=False)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, handlelength=1.0,
               columnspacing=1.4)
    ax.set_title("(a) Pass rate")
    for j, (name, d, col, lab) in enumerate(conds):
        for i, h in enumerate(HEAD):
            v = d[h].dropna()
            med, lo, hi = v.median(), v.quantile(0.25), v.quantile(0.75)
            x = i + (j - 1) * w
            bx.bar(x, max(med, 0), w * 0.92, color=col, zorder=2, linewidth=0)
            bx.plot([x, x], [max(lo, -0.05), min(hi, 1.6)], color=INK_SECONDARY, linewidth=0.7, zorder=3)
    bx.axhline(1.0, color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (3, 2)))
    bx.set_xticks(range(3)); bx.set_xticklabels(["Z", "A", "C"])
    bx.set(ylim=(0, 1.4), ylabel="faithfulness (median, IQR)")
    bx.grid(axis="x", visible=False)
    bx.set_title("(b) Faithfulness")
    png = save_figure(fig, HERE / "out_alpha1" / "coefficient-triad.png")
    shutil.copy(png.with_suffix(".pdf"), ROOT / "paper" / "figures" / "coefficient-triad.pdf")
    print("figure:", png, "-> paper/figures/coefficient-triad.pdf")


def main():
    full, a1, rn = load()
    numbers(full, a1, rn)
    if os.environ.get("FIG"):
        figure(full, a1, rn)


if __name__ == "__main__":
    main()
