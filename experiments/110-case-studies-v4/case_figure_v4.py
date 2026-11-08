"""PAPER FIGURE for the case study on the production circuit: 063/case_figure.py's four panels, drawn from
out/case/<KEY>.json (case_v4.py), at the paper's text width (5.5 in), pass band at the production floor 0.75.

  python -X utf8 experiments/110-case-studies-v4/case_figure_v4.py
  env: KEY (9.resid.20419)  FIG (paper/figures/run2026_case_antiquity.png; .pdf written beside it)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib110 as L  # noqa: E402
import case_figure as CF  # noqa: E402  (063)
from analysis import style as S  # noqa: E402

KEY = os.environ.get("KEY", "9.resid.20419")
DATA = L.RESULTS / "case" / ("%s.json" % KEY)
FIG = Path(os.environ.get("FIG", str(L.FIGS / "run2026_case_antiquity.png")))
CF.FS.update(title=7.0, box_title=6.4, box=5.6, edge=5.6, tick=5.8, note=5.5)
CIRC = S.INK                 # circuit-alone read (contrast-mean fill)
# the historical-period box sits lower than in 063 so the early-civilisations -> target arrow can pass above it
CF.POS["historical period (control)"] = (0.645, 0.13)


def panel_circuit(ax, d):
    """063 panel_circuit, plus the early-civilisations -> target arrow (063 left the left-column groups' target drops
    to the caption; this one is the largest drop in the production circuit). All drops are TOTAL effects: the target's
    relative drop when the group is removed and every other member is recomputed live (case_study.edges)."""
    from matplotlib.patches import FancyArrowPatch
    CF.panel_circuit(ax, d)
    g = "early civilisations"
    w = d["edges"]["to_target"][g]
    x, y = CF.POS[g]
    p = (x + CF.BOX_W / 2, y)
    q = (CF.TARGET_POS[0] - CF.TW / 2, CF.TARGET_POS[1] - 0.08)
    col = CF.DISPLAY[g][2]
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=max(0.7, 8 * w), color=col,
                                 zorder=2, shrinkA=1, shrinkB=1))
    f = 0.62
    ax.text(p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1]), "%.2f" % w, fontsize=CF.FS["edge"], ha="center",
            va="center", color=S.INK, zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))


def panel_years(ax, d):
    """(b) clean model (as 063) plus the circuit run alone under contrast-mean ablation (C fill), both as % of the
    reference; the empty circuit under the same fill reads 0 on every sentence."""
    ref = d["probes"]["ref_a_pos"]
    ys = sorted(d["probes"]["years"], key=lambda r: r["year"])
    x = np.arange(len(ys))
    cls = [i for i, r in enumerate(ys) if -800 <= r["year"] <= -300]
    ax.axvspan(min(cls) - 0.4, max(cls) + 0.4, color=S.tint(S.BLUE, 0.9), zorder=0, linewidth=0)
    ax.text((min(cls) + max(cls)) / 2, 128, "classical", ha="center", va="top", fontsize=CF.FS["note"], color=S.BLUE)
    ax.plot(x, [100 * r["clean"] / ref for r in ys], color=S.BLUE, marker="o", markersize=2.2, linewidth=1.1,
            zorder=3, label="model")
    ax.plot(x, [100 * r["C"] / ref for r in ys], color=CIRC, marker="s", markersize=2.0, linewidth=1.0,
            zorder=3, label="circuit alone (C)")
    lab = [("%d BC" % -r["year"]) if r["year"] < 0 else ("AD %d" % r["year"]) for r in ys]
    keep = {"5000 BC", "1000 BC", "AD 100", "AD 2000"}
    ax.set_xticks([i for i, l in enumerate(lab) if l in keep])
    ax.set_xticklabels([l for l in lab if l in keep], rotation=45, ha="right")
    ax.set_ylim(0, 130); ax.set_ylabel("target, % of reference")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.47), handlelength=1.2, borderaxespad=0.1, fontsize=5.2,
              frameon=False)
    ax.set_title("(b) “In [year], …”", fontsize=CF.FS["title"])


def panel_places(ax, d):
    ref = d["probes"]["ref_a_pos"]
    rows = d["probes"]["sentences"]
    pick = [("ancient Athens", "Athens"), ("ancient Greece", "Greece"), ("ancient Rome", "Rome"),
            ("ancient Egypt", "Egypt"), ("ancient China", "China"), ("medieval France", "med. France"),
            ("BC Lions", "“BC Lions”")]
    vals, circ, labs = [], [], []
    for needle, lab in pick:
        r = next(r for r in rows if needle in r["sentence"])
        vals.append(100 * r["clean"] / ref); circ.append(100 * r["C"] / ref); labs.append(lab)
    y = np.arange(len(vals))[::-1]
    cols = [S.BLUE] * 2 + [S.tint(S.BLUE, 0.45)] * 3 + [CF.GREY] * 2
    ax.barh(y, vals, color=cols, height=0.62)
    ax.plot(circ, y, linestyle="none", marker="s", markersize=2.4, color=CIRC, zorder=4)
    for yi, v in zip(y, vals):
        ax.text(v + 3, yi, "%.0f" % v, va="center", fontsize=CF.FS["note"], color=S.INK)
    ax.set_yticks(y); ax.set_yticklabels(labs)
    ax.set_xlim(0, 135); ax.set_xticks([0, 50, 100]); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("target, % of reference")
    ax.set_title("(c) “In [place], …”", fontsize=CF.FS["title"])


def panel_ko(ax, d):
    """063's panel (d), with the production pass band [0.75, 1.5]."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    order = ["classical Greece", "early civilisations", "dates / eras", "philosophy / rhetoric",
             "historical period (control)", "generic core (control)", "all story groups"]
    labs = ["Class.\nGreece", "Early\nciv.", "Dates", "Philos.", "Period", "Generic", "Story\n(all)"]
    K = d["knockouts"]
    w = 0.26
    ax.axhspan(0.75, 1.5, color=S.tint(S.CATEGORICAL[3], 0.9), zorder=0, linewidth=0)
    for gi, g in enumerate(order):
        col = CF.DISPLAY[g][2] if g in CF.DISPLAY else S.INK
        for si, (sc, amt) in enumerate((("Z", 0.6), ("A", 0.3), ("C", 0.0))):
            x = gi + (si - 1) * w
            v = K[g]["ko"][sc]
            ax.bar(x, max(v, 0.0), w * 0.9, color=S.tint(col, amt), zorder=2, linewidth=0)
            if v < 0.03:
                ax.text(x, 0.02, "%.2f" % v, ha="center", va="bottom", fontsize=5.0, color=S.INK, rotation=90)
            rnd = [r[sc] for r in K[g]["random"] if r[sc] is not None]
            m, s = float(np.mean(rnd)), float(np.std(rnd))
            ax.plot([x - w * 0.42, x + w * 0.42], [m, m], color=S.INK, linewidth=0.9, zorder=4)
            ax.plot([x, x], [m - s, m + s], color=S.INK, linewidth=0.5, zorder=4)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(labs); ax.tick_params(axis="x", labelsize=5.2)
    ax.set_xlim(-0.5, len(order) - 0.5); ax.set_ylim(0, 1.42); ax.set_ylabel("faithfulness")
    handles = [Patch(color=S.tint(S.INK_MUTED, a), label=l) for l, a in (("Z", 0.6), ("A", 0.3), ("C", 0.0))]
    handles.append(Line2D([0], [0], color=S.INK, linewidth=0.9, label="random (mean ± sd)"))
    handles.append(Patch(color=S.tint(S.CATEGORICAL[3], 0.9), label="pass band"))
    ax.legend(handles=handles, loc="upper left", ncol=5, handlelength=0.9, columnspacing=0.6, handletextpad=0.3,
              borderaxespad=0.1, fontsize=5.4)
    ax.set_title("(d) Removing a group vs random members", fontsize=CF.FS["title"])


def main():
    plt = S.configure_matplotlib()
    plt.rcParams.update({"font.size": 6.2, "axes.titlesize": CF.FS["title"], "axes.titlepad": 3.5,
                         "axes.labelsize": 6.0, "xtick.labelsize": CF.FS["tick"], "ytick.labelsize": CF.FS["tick"],
                         "legend.fontsize": 5.6, "axes.linewidth": 0.6, "grid.linewidth": 0.45,
                         "xtick.major.pad": 1.2, "ytick.major.pad": 1.2, "axes.labelpad": 1.6})
    d = json.loads(DATA.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(5.5, 4.1))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.85, 1.85], height_ratios=[0.8, 1.0], wspace=0.55, hspace=0.45)
    panel_circuit(fig.add_subplot(gs[0, :]), d)
    panel_years(fig.add_subplot(gs[1, 0]), d)
    panel_places(fig.add_subplot(gs[1, 1]), d)
    panel_ko(fig.add_subplot(gs[1, 2]), d)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    S.save_figure(fig, FIG)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
