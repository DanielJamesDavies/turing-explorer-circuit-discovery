"""PAPER FIGURE for the case study (DAN-149): the weighted circuit of 9.resid.20419, a date or place read as classical
Greek antiquity. Reads results_case/<KEY>.json (case_study.py) and writes paper/figures/case-study.pdf (+ .png).
Drawn at print size (7 in wide, the paper's text width) so the type is 6.5-8 pt on the page.

  (a) the circuit's readable core: role groups (named from each member's own top contexts and logit effect), each
      with its size, layers and contribution share; arrows = causal group edges under the activating-mean fill
      (removing the source group lowers the destination by that fraction; group -> group drawn when >= EDGE_MIN)
  (b) the target in the clean model on "In <year>, the scholar lived in the city of" (max over the sentence, % of its
      mean activation on its held-out strongest contexts)
  (c) the same for places ("In ancient Greece, ...") and the lexical control
  (d) knock-outs: faithfulness under Z, A, C with each group removed vs the same number of rank-matched random
      members (mean +- sd over 8 draws)

  python experiments/063-case-studies/case_figure.py       env: KEY (9.resid.20419)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from analysis import style as S  # noqa: E402

KEY = os.environ.get("KEY", "9.resid.20419")
DATA = HERE / "results_case" / ("%s.json" % KEY)
FIG = ROOT / "paper" / "figures" / "case-study.png"
EDGE_MIN = 0.08
GREY = "#9aa0a8"
FS = dict(title=7.8, box_title=7.4, box=6.4, edge=6.3, tick=6.4, note=6.2)

# display: group -> (label, own-context words, colour); grey = fires about as much on contrast contexts
DISPLAY = {
    "classical Greece": ("Classical Greece", "Athens · Sparta · Socrates", S.CATEGORICAL[0]),
    "dates / eras": ("Dates and eras", "4500 · circa 470 · 6th c. BC", S.CATEGORICAL[1]),
    "early civilisations": ("Early civilisations", "Egypt · Sumer · Babylon", S.CATEGORICAL[2]),
    "philosophy / rhetoric": ("Philosophy, rhetoric", "rhetoric · epistemology", S.CATEGORICAL[3]),
    "historical period (control)": ("Historical period", "Renaissance · Humanism", GREY),
    "generic core (control)": ("Generic core", "fires on most text", GREY),
}
# box centres in axes coordinates: a left-to-right flow across the full-width top row
BOX_W, BOX_H = 0.21, 0.32
POS = {
    "dates / eras": (0.105, 0.80),
    "early civilisations": (0.105, 0.20),
    "classical Greece": (0.395, 0.50),
    "philosophy / rhetoric": (0.645, 0.82),
    "historical period (control)": (0.645, 0.18),
}
TARGET_POS, TW, TH = (0.885, 0.50), 0.22, 0.36
# (source, destination, source side, destination side) for the drawn group -> group arrows
GROUP_EDGES = (("dates / eras", "early civilisations", "b", "t"),
               ("dates / eras", "classical Greece", "r", "l"),
               ("early civilisations", "classical Greece", "r", "l"))


def fmt_layers(ls):
    return "L%d" % ls[0] if len(ls) == 1 else "L%d–%d" % (min(ls), max(ls))


def panel_circuit(ax, d):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    G, E = d["groups"], d["edges"]

    def box(xy, w, h, face, edge, lines, dashed=False):
        x, y = xy
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                    facecolor=face, edgecolor=edge, linewidth=1.0,
                                    linestyle=(0, (3, 2)) if dashed else "-", zorder=3, clip_on=False))
        n = len(lines)
        for i, (txt, kw) in enumerate(lines):
            ax.text(x, y + h / 2 - (i + 0.7) * h / (n + 0.4), txt, ha="center", va="center", zorder=4, **kw)

    for g, (x, y) in POS.items():
        label, words, col = DISPLAY[g]
        info = G[g]
        box((x, y), BOX_W, BOX_H, S.tint(col, 0.88), col,
            [(label, dict(fontsize=FS["box_title"], fontweight="bold", color=S.INK)),
             ("%d latents · %s · %.0f%%" % (info["n"], fmt_layers(info["layers"]), 100 * info["share"]),
              dict(fontsize=FS["box"], color=S.INK_SECONDARY)),
             (words, dict(fontsize=FS["box"], style="italic", color=S.INK_SECONDARY))],
            dashed=col == GREY)

    tx, ty = TARGET_POS
    box((tx, ty), TW, TH, S.INK, S.INK,
        [("Target 9.resid.20419", dict(fontsize=FS["box_title"], fontweight="bold", color="white")),
         ("classical Greek antiquity", dict(fontsize=FS["box"], style="italic", color="white")),
         ("promotes Arist · Spart · Ath", dict(fontsize=FS["box"], color="#d7dbe2"))])

    def arrow(p, q, w, color, dashed=False):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=max(0.7, 8 * w),
                                     color=color, linestyle=(0, (3, 2)) if dashed else "-", zorder=2,
                                     shrinkA=1, shrinkB=1))

    def label(p, q, w, f=0.5):
        ax.text(p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1]), "%.2f" % w, fontsize=FS["edge"], ha="center",
                va="center", color=S.INK, zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    def side(xy, w, h, s):
        x, y = xy
        return {"r": (x + w / 2, y), "l": (x - w / 2, y), "b": (x, y - h / 2), "t": (x, y + h / 2)}[s]

    for src, dst, ps, pd in GROUP_EDGES:                       # group -> group (causal, A fill)
        w = E["edges"][src][dst]
        if w is not None and w >= EDGE_MIN:
            p, q = side(POS[src], BOX_W, BOX_H, ps), side(POS[dst], BOX_W, BOX_H, pd)
            if ps == "r":                                      # land on the destination's left side, offset by source
                q = (q[0], q[1] + (0.07 if POS[src][1] > POS[dst][1] else -0.07))
            arrow(p, q, w, DISPLAY[src][2]); label(p, q, w)
    for g, (x, y) in POS.items():                              # group -> target
        w = E["to_target"][g]
        if w is None or x < POS["classical Greece"][0]:        # left-column groups: their target drop is in the caption
            continue
        p = side(POS[g], BOX_W, BOX_H, "r")
        q = side(TARGET_POS, TW, TH, "l") if g == "classical Greece" else (tx - TW / 2, ty + (0.12 if y > ty else -0.12))
        arrow(p, q, max(w, 0.0), DISPLAY[g][2], dashed=DISPLAY[g][2] == GREY); label(p, q, w, 0.45)
    ax.set_title("(a) The circuit's readable core", fontsize=FS["title"])


def panel_years(ax, d):
    ref = d["probes"]["ref_a_pos"]
    ys = sorted(d["probes"]["years"], key=lambda r: r["year"])
    x = np.arange(len(ys))
    v = [100 * r["clean"] / ref for r in ys]
    cls = [i for i, r in enumerate(ys) if -800 <= r["year"] <= -300]
    ax.axvspan(min(cls) - 0.4, max(cls) + 0.4, color=S.tint(S.BLUE, 0.9), zorder=0, linewidth=0)
    ax.text((min(cls) + max(cls)) / 2, 74, "classical\nperiod", ha="center", va="top", fontsize=FS["note"],
            color=S.BLUE)
    ax.plot(x, v, color=S.BLUE, marker="o", markersize=2.6, linewidth=1.3, zorder=3)
    lab = [("%d BC" % -r["year"]) if r["year"] < 0 else ("AD %d" % r["year"]) for r in ys]
    keep = {"5000 BC", "1000 BC", "500 BC", "100 BC", "AD 500", "AD 1200", "AD 2000"}
    ax.set_xticks([i for i, l in enumerate(lab) if l in keep])
    ax.set_xticklabels([l for l in lab if l in keep], rotation=45, ha="right")
    ax.set_ylim(0, 76); ax.set_ylabel("target, % of reference")
    ax.set_title("(b) “In [year], the scholar …”", fontsize=FS["title"])


def panel_places(ax, d):
    ref = d["probes"]["ref_a_pos"]
    rows = d["probes"]["sentences"]
    pick = [("ancient Athens", "Athens"), ("ancient Greece", "Greece"), ("ancient Rome", "Rome"),
            ("ancient Egypt", "Egypt"), ("ancient China", "China"), ("medieval France", "med. France"),
            ("BC Lions", "“BC Lions”")]
    vals, labs = [], []
    for needle, lab in pick:
        r = next(r for r in rows if needle in r["sentence"])
        vals.append(100 * r["clean"] / ref); labs.append(lab)
    y = np.arange(len(vals))[::-1]
    cols = [S.BLUE] * 2 + [S.tint(S.BLUE, 0.45)] * 3 + [GREY] * 2
    ax.barh(y, vals, color=cols, height=0.64)
    for yi, v in zip(y, vals):
        ax.text(v + 2, yi, "%.0f" % v, va="center", fontsize=FS["note"], color=S.INK)
    ax.set_yticks(y); ax.set_yticklabels(labs)
    ax.set_xlim(0, 112); ax.set_xticks([0, 50, 100]); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("target, % of reference")
    ax.set_title("(c) “In [place], …”", fontsize=FS["title"])


def panel_ko(ax, d):
    """Per group: faithfulness under Z, A, C with the group removed (bars, light -> dark), and the random-removal
    mean +- sd for the same score (black tick with whisker)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    order = ["classical Greece", "early civilisations", "dates / eras", "philosophy / rhetoric",
             "historical period (control)", "generic core (control)", "all story groups"]
    labs = ["Classical\nGreece", "Early\ncivil.", "Dates,\neras", "Philos.,\nrhetoric", "Hist.\nperiod",
            "Generic\ncore", "Story\ngroups"]
    K = d["knockouts"]
    w = 0.26
    ax.axhspan(0.8, 1.5, color=S.tint(S.CATEGORICAL[3], 0.9), zorder=0, linewidth=0)
    for gi, g in enumerate(order):
        col = DISPLAY[g][2] if g in DISPLAY else S.INK
        for si, (sc, amt) in enumerate((("Z", 0.6), ("A", 0.3), ("C", 0.0))):
            x = gi + (si - 1) * w
            v = K[g]["ko"][sc]
            ax.bar(x, max(v, 0.0), w * 0.9, color=S.tint(col, amt), zorder=2, linewidth=0)
            if v < 0.03:                                       # a bar too short to see: print its value
                ax.text(x, 0.02, "%.2f" % v, ha="center", va="bottom", fontsize=5.6, color=S.INK, rotation=90)
            rnd = [r[sc] for r in K[g]["random"] if r[sc] is not None]
            m, s = float(np.mean(rnd)), float(np.std(rnd))
            ax.plot([x - w * 0.42, x + w * 0.42], [m, m], color=S.INK, linewidth=1.0, zorder=4)
            ax.plot([x, x], [m - s, m + s], color=S.INK, linewidth=0.6, zorder=4)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(labs)
    ax.set_xlim(-0.5, len(order) - 0.5); ax.set_ylim(0, 1.38); ax.set_ylabel("faithfulness")
    handles = [Patch(color=S.tint(S.INK_MUTED, a), label=l) for l, a in (("Z", 0.6), ("A", 0.3), ("C", 0.0))]
    handles.append(Line2D([0], [0], color=S.INK, linewidth=1.0, label="random members (mean ± sd)"))
    handles.append(Patch(color=S.tint(S.CATEGORICAL[3], 0.9), label="pass band"))
    ax.legend(handles=handles, loc="upper left", ncol=5, handlelength=1.0, columnspacing=0.8, handletextpad=0.35,
              borderaxespad=0.1)
    ax.set_title("(d) Removing a group vs removing random members", fontsize=FS["title"])


def main():
    plt = S.configure_matplotlib()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": FS["title"], "axes.titlepad": 4.0,
                         "axes.labelsize": 6.6, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
                         "legend.fontsize": 6.2, "axes.linewidth": 0.7, "grid.linewidth": 0.5,
                         "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0})
    d = json.loads(DATA.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(7.0, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.78, 1.9], height_ratios=[0.92, 1.0], wspace=0.38, hspace=0.38)
    panel_circuit(fig.add_subplot(gs[0, :]), d)
    panel_years(fig.add_subplot(gs[1, 0]), d)
    panel_places(fig.add_subplot(gs[1, 1]), d)
    panel_ko(fig.add_subplot(gs[1, 2]), d)
    S.save_figure(fig, FIG)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
