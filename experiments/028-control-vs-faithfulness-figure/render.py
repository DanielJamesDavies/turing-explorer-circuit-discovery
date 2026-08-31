"""Render the centrepiece figure from curves.jsonl.

Three panels (shallow / mid / deep seed), x = members added in
attribution order (log), y = score. Per panel: pinned (blue, node
selection) and free under the mean fill (red, the circuit alone), with
the pinned-free gap shaded; free under zero fill as a faint dashed
companion so each free variant names its semantics. Slot 3 (teal) is
reserved for the weighted-circuit markers once matched-seed tri-amp
data exists (the depth-stratified run).

  PYTHONPATH=src python experiments/028-control-vs-faithfulness-figure/render.py
"""
import json
from pathlib import Path

from analysis.style import (BASELINE, CATEGORICAL, INK_MUTED,
                            configure_matplotlib, panel_figsize, tint)

configure_matplotlib()
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIGDIR = HERE.parent.parent.parent / "paper" / "figures"
PINNED, FREE = CATEGORICAL[0], CATEGORICAL[1]

rows = [json.loads(l) for l in (HERE / "curves.jsonl").open()]
bands = []
for band in ("shallow", "mid", "deep"):
    rs = sorted([r for r in rows if r["band"] == band], key=lambda r: r["n"])
    if rs:
        bands.append((band, rs))

fig, axes = plt.subplots(1, len(bands), figsize=panel_figsize(1, len(bands)),
                         sharey=True)
if len(bands) == 1:
    axes = [axes]
for ax, (band, rs) in zip(axes, bands):
    n = [r["n"] for r in rs]
    pinned = [r["pinned_mean"] for r in rs]
    free_m = [r["free_mean"] for r in rs]
    free_0 = [r["free_zero"] for r in rs]
    ax.axhline(1.0, color=BASELINE, linewidth=0.8, linestyle=(0, (3, 4)),
               alpha=0.6, zorder=1)
    ax.fill_between(n, free_m, pinned, where=[p >= f for p, f in
                    zip(pinned, free_m)], color=tint(FREE, 0.75),
                    alpha=0.55, linewidth=0, zorder=2)
    ax.plot(n, pinned, color=PINNED, linewidth=2.2, zorder=4,
            label="pinned (node selection)")
    ax.plot(n, free_m, color=FREE, linewidth=2.2, zorder=4,
            label="free, mean fill (circuit alone)")
    ax.plot(n, free_0, color=FREE, linewidth=1.1, linestyle=(0, (2, 2.5)),
            alpha=0.75, zorder=3, label="free, zero fill")
    ax.set_xscale("log")
    ax.set_ylim(-0.08, 1.3)
    r0 = rs[0]
    ax.set_title("%s (L%d %s)" % (band, r0["layer"], r0["kind"]))
    ax.set_xlabel("members added in attribution order")
axes[0].set_ylabel("score")
axes[0].legend(loc="upper left", frameon=False, fontsize=9.5,
               labelcolor=INK_MUTED, handlelength=1.6)
fig.tight_layout()
FIGDIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / ("control-vs-faithfulness." + ext),
                bbox_inches="tight")
print("wrote", FIGDIR / "control-vs-faithfulness.pdf")
