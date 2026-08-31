"""Render the REFRESHED centrepiece from curves2.jsonl: per-band median
curves (3 seeds) with individual seed traces ghosted, plus teal
weighted-circuit markers from 029-panel (triamp400 rows on the
same seeds; amplitudes applied, zero fill — different amplitude
semantics from the curves, named in the caption).

  PYTHONPATH=src python experiments/028-control-vs-faithfulness-figure/render2.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.style import (BASELINE, CATEGORICAL, INK_MUTED,
                            configure_matplotlib, panel_figsize, tint)

configure_matplotlib()
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIGDIR = HERE.parent.parent.parent / "paper" / "figures"
PANEL_ROWS = HERE.parent / "029-panel" / "rows.jsonl"
PINNED, FREE, TEAL = CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2]
BAND_COMP = {"shallow": 10, "mid": 23, "deep": 34}

rows = [json.loads(l) for l in (HERE / "curves2.jsonl").open()]
by_band = defaultdict(lambda: defaultdict(list))
for r in rows:
    by_band[r["band"]][r["latent"]].append(r)

panel = ([json.loads(l) for l in PANEL_ROWS.open()]
         if PANEL_ROWS.exists() else [])

bands = [b for b in ("shallow", "mid", "deep") if by_band.get(b)]
fig, axes = plt.subplots(1, len(bands), figsize=panel_figsize(1, len(bands)),
                         sharey=True)
if len(bands) == 1:
    axes = [axes]
for ax, band in zip(axes, bands):
    seeds = by_band[band]
    ax.axhline(1.0, color=BASELINE, linewidth=0.8, linestyle=(0, (3, 4)),
               alpha=0.6, zorder=1)
    curves = {}
    for metric in ("pinned_mean", "free_mean", "free_zero"):
        per_seed = []
        for sl, rs in seeds.items():
            rs = sorted(rs, key=lambda r: r["n"])
            n = np.array([r["n"] for r in rs], dtype=float)
            v = np.array([r[metric] for r in rs], dtype=float)
            per_seed.append((n, v))
        lo = max(float(n.min()) for n, _ in per_seed)
        hi = min(float(n.max()) for n, _ in per_seed)
        grid = np.logspace(np.log10(lo), np.log10(hi), 60)
        interp = np.stack([np.interp(np.log10(grid), np.log10(n), v)
                           for n, v in per_seed])
        curves[metric] = (grid, np.median(interp, axis=0), per_seed)
    g, med_p, traces_p = curves["pinned_mean"]
    _, med_f, traces_f = curves["free_mean"]
    _, med_z, _ = curves["free_zero"]
    ax.fill_between(g, med_f, med_p, where=med_p >= med_f,
                    color=tint(FREE, 0.75), alpha=0.5, linewidth=0, zorder=2)
    for n, v in traces_p:
        ax.plot(n, v, color=PINNED, linewidth=0.8, alpha=0.28, zorder=3)
    for n, v in traces_f:
        ax.plot(n, v, color=FREE, linewidth=0.8, alpha=0.28, zorder=3)
    ax.plot(g, med_p, color=PINNED, linewidth=2.3, zorder=5,
            label="pinned (node selection)")
    ax.plot(g, med_f, color=FREE, linewidth=2.3, zorder=5,
            label="free, mean fill (circuit alone)")
    ax.plot(g, med_z, color=FREE, linewidth=1.1, linestyle=(0, (2, 2.5)),
            alpha=0.75, zorder=4, label="free, zero fill")
    comp = BAND_COMP[band]
    marks = [(r["n"], r["ampF0"]) for r in panel
             if r["comp_idx"] == comp and r["arm"] == "triamp400"
             and r["latent"] in seeds and r["ampF0"] is not None]
    if marks:
        ax.scatter([m[0] for m in marks], [m[1] for m in marks],
                   marker="D", s=52, color=TEAL, zorder=6,
                   label="weighted circuit ($\\alpha$ applied)")
    r0 = next(iter(seeds.values()))[0]
    ax.set_xscale("log")
    ax.set_ylim(-0.08, 1.3)
    ax.set_title("%s (L%d %s, %d seeds)" % (band, r0["layer"], r0["kind"],
                                            len(seeds)))
    ax.set_xlabel("members added in attribution order")
axes[0].set_ylabel("score")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           fontsize=9.5, labelcolor=INK_MUTED, handlelength=1.6,
           bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.06, 1, 1))
FIGDIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / ("control-vs-faithfulness." + ext),
                bbox_inches="tight")
print("wrote", FIGDIR / "control-vs-faithfulness.pdf")
