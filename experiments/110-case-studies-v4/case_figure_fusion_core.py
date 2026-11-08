"""PAPER FIGURE, antiquity layout, for the fusion circuit 9.resid.13945 (production v4 circuit): 063/case_figure.py's
four-panel design (readable-core diagram, two probe panels, knock-outs vs random members), reusing its box / arrow
drawing; only the group names, positions, edges and probe groupings are new. Reads out/case/9.resid.13945.json
(case_v4.py with groups_9.resid.13945.json). 5.5 in wide, pass band [0.75, 1.5].

  python -X utf8 experiments/110-case-studies-v4/case_figure_fusion_core.py
  -> paper/figures/run2026_case_fusion_core.pdf/.png
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib110 as L  # noqa: E402
import case_figure as CF  # noqa: E402  (063: FS, GREY, fmt_layers, the drawing style)
from analysis import style as S  # noqa: E402

KEY = "9.resid.13945"
DATA = L.RESULTS / "case" / ("%s.json" % KEY)
FIG = L.FIGS / "run2026_case_fusion_core.png"
GREY = CF.GREY
CIRC = S.INK
FS = dict(CF.FS, title=7.0, box_title=6.4, box=5.6, edge=5.6, tick=5.8, note=5.5)
EDGE_MIN = 0.08

# group -> (label, words from the members' own top contexts, colour); grey dashed = no more necessary than random
DISPLAY = {
    "plasma confinement": ("Plasma confinement", "tokamak · stellarator · torus", S.CATEGORICAL[0]),
    "nuclear and power engineering": ("Nuclear and power", "reactors · turbines · Otto cycle", S.CATEGORICAL[1]),
    "stars, cosmos, particles": ("Stars, cosmos", "stellar cores · dark matter", GREY),
    "quantum and magnetism": ("Quantum, magnetism", "wave function · NMR · qubit", GREY),
}
BOX_W, BOX_H = 0.22, 0.32
POS = {
    "plasma confinement": (0.12, 0.50),
    "nuclear and power engineering": (0.47, 0.82),
    "stars, cosmos, particles": (0.36, 0.17),
    "quantum and magnetism": (0.62, 0.17),
}
TARGET_POS, TW, TH = (0.88, 0.50), 0.235, 0.36
# (source, destination, source side, destination side, destination y offset)
GROUP_EDGES = (("plasma confinement", "nuclear and power engineering", "t", "l", 0.0),
               ("plasma confinement", "stars, cosmos, particles", "b", "l", 0.0))


def panel_circuit(ax, d):
    """063 panel_circuit with this circuit's groups, edges and target box."""
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
             ("%d latents · %s · %.0f%%" % (info["n"], CF.fmt_layers(info["layers"]), 100 * info["share"]),
              dict(fontsize=FS["box"], color=S.INK_SECONDARY)),
             (words, dict(fontsize=FS["box"] - 0.3, style="italic", color=S.INK_SECONDARY))],
            dashed=col == GREY)
    tx, ty = TARGET_POS
    box((tx, ty), TW, TH, S.INK, S.INK,
        [("Target %s" % KEY, dict(fontsize=FS["box_title"], fontweight="bold", color="white")),
         ("magnetic confinement fusion", dict(fontsize=FS["box"] - 0.4, style="italic", color="white")),
         ("promotes Tok · tor · stell", dict(fontsize=FS["box"], color="#d7dbe2"))])

    def arrow(p, q, w, color, dashed=False):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=max(0.7, 8 * w),
                                     color=color, linestyle=(0, (3, 2)) if dashed else "-", zorder=2,
                                     shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=0"))

    def label(p, q, w, f=0.5):
        ax.text(p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1]), "%.2f" % w, fontsize=FS["edge"], ha="center",
                va="center", color=S.INK, zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    def side(xy, w, h, s):
        x, y = xy
        return {"r": (x + w / 2, y), "l": (x - w / 2, y), "b": (x, y - h / 2), "t": (x, y + h / 2)}[s]

    for src, dst, ps, pd_, dy in GROUP_EDGES:                    # group -> group (causal, A fill)
        w = E["edges"][src][dst]
        if w is None or w < EDGE_MIN:
            continue
        p, q = side(POS[src], BOX_W, BOX_H, ps), side(POS[dst], BOX_W, BOX_H, pd_)
        q = (q[0], q[1] + dy)
        if ps == "t":                                            # leave the top, then run to the destination's side
            p = (p[0], p[1])
        arrow(p, q, w, DISPLAY[src][2]); label(p, q, w)
    # group -> target (direct drops)
    tl = tx - TW / 2
    direct = [("plasma confinement", side(POS["plasma confinement"], BOX_W, BOX_H, "r"), (tl, ty)),
              ("nuclear and power engineering", side(POS["nuclear and power engineering"], BOX_W, BOX_H, "r"),
               (tl, ty + 0.12)),
              ("stars, cosmos, particles", side(POS["stars, cosmos, particles"], BOX_W, BOX_H, "t"), (tl, ty - 0.07)),
              ("quantum and magnetism", side(POS["quantum and magnetism"], BOX_W, BOX_H, "r"), (tl, ty - 0.13))]
    for g, p, q in direct:
        w = E["to_target"][g]
        if w is None:
            continue
        arrow(p, q, max(w, 0.0), DISPLAY[g][2], dashed=DISPLAY[g][2] == GREY); label(p, q, w, 0.55)
    ax.set_title("(a) The circuit's readable core", fontsize=FS["title"])


def probe_panel(ax, d, picks, title, cols):
    """063 panel_places, generalised: bars = clean model, squares = circuit alone under C, % of the reference."""
    ref = d["probes"]["ref_a_pos"]
    rows = d["probes"]["sentences"]
    vals, circ, labs = [], [], []
    for needle, lab in picks:
        r = next(r for r in rows if needle in r["sentence"])
        vals.append(100 * r["clean"] / ref); circ.append(100 * r["C"] / ref); labs.append(lab)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    ax.plot(circ, y, linestyle="none", marker="s", markersize=2.4, color=CIRC, zorder=4)
    for yi, v, c in zip(y, vals, circ):
        ax.text(max(v, c) + 5, yi, "%.0f" % v, va="center", fontsize=FS["note"], color=S.INK)
    ax.set_yticks(y); ax.set_yticklabels(labs)
    ax.set_xlim(0, 145); ax.set_xticks([0, 50, 100]); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("target, % of reference")
    ax.set_title(title, fontsize=FS["title"])


def panel_ko(ax, d):
    """063 panel_ko for this circuit's groups, band [0.75, 1.5]."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    order = ["plasma confinement", "nuclear and power engineering", "stars, cosmos, particles",
             "quantum and magnetism", "off-topic (control)", "generic core (control)", "all story groups"]
    labs = ["Plasma", "Nuclear", "Stars", "Quantum", "Off-topic", "Generic", "Story (all)"]
    colour = {g: v[2] for g, v in DISPLAY.items()}
    colour.update({"stars, cosmos, particles": S.CATEGORICAL[2], "quantum and magnetism": S.CATEGORICAL[3],
                   "off-topic (control)": GREY, "generic core (control)": GREY, "all story groups": S.INK})
    K = d["knockouts"]
    w = 0.26
    ax.axhspan(0.75, 1.5, color=S.tint(S.CATEGORICAL[3], 0.9), zorder=0, linewidth=0)
    for gi, g in enumerate(order):
        col = colour[g]
        for si, (sc, amt) in enumerate((("Z", 0.6), ("A", 0.3), ("C", 0.0))):
            x = gi + (si - 1) * w
            v = K[g]["ko"][sc]
            ax.bar(x, max(v, 0.0), w * 0.9, color=S.tint(col, amt), zorder=2, linewidth=0)
            if v < 0.03:
                ax.text(x, 0.02, "%.2f" % v, ha="center", va="bottom", fontsize=4.8, color=S.INK, rotation=90)
            rnd = [r[sc] for r in K[g]["random"] if r[sc] is not None]
            m, s = float(np.mean(rnd)), float(np.std(rnd))
            ax.plot([x - w * 0.42, x + w * 0.42], [m, m], color=S.INK, linewidth=0.9, zorder=4)
            ax.plot([x, x], [m - s, m + s], color=S.INK, linewidth=0.5, zorder=4)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(labs, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=5.2)
    ax.set_xlim(-0.5, len(order) - 0.5); ax.set_ylim(0, 1.42); ax.set_ylabel("faithfulness")
    handles = [Patch(color=S.tint(S.INK_MUTED, a), label=l) for l, a in (("Z", 0.6), ("A", 0.3), ("C", 0.0))]
    handles.append(Line2D([0], [0], color=S.INK, linewidth=0.9, label="random (mean ± sd)"))
    handles.append(Patch(color=S.tint(S.CATEGORICAL[3], 0.9), label="pass band"))
    ax.legend(handles=handles, loc="upper left", ncol=5, handlelength=0.9, columnspacing=0.6, handletextpad=0.3,
              borderaxespad=0.1, fontsize=5.4)
    ax.set_title("(d) Removing a group vs random members", fontsize=FS["title"])


def main():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt = S.configure_matplotlib()
    plt.rcParams.update({"font.size": 6.2, "axes.titlesize": FS["title"], "axes.titlepad": 3.5,
                         "axes.labelsize": 6.0, "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
                         "legend.fontsize": 5.6, "axes.linewidth": 0.6, "grid.linewidth": 0.45,
                         "xtick.major.pad": 1.2, "ytick.major.pad": 1.2, "axes.labelpad": 1.6})
    d = json.loads(DATA.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(5.5, 4.1))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.75], height_ratios=[0.8, 1.0], wspace=0.75, hspace=0.45)
    panel_circuit(fig.add_subplot(gs[0, :]), d)
    blue, light = S.BLUE, S.tint(S.BLUE, 0.45)
    bx = fig.add_subplot(gs[1, 0])
    probe_panel(bx, d, [("tokamak uses", "tokamak"), ("stellarator", "stellarator"), ("helical", "helical field"),
                        ("Inside the tokamak", "tokamak coils"), ("inertial", "inertial conf."),
                        ("Sun produces", "the Sun"), ("fission", "fission reactor")],
                "(b) Fusion and other nuclear", [blue] * 4 + [light] * 3)
    cx = fig.add_subplot(gs[1, 1])
    probe_panel(cx, d,
                [("electric motor", "electric motor"), ("MRI", "MRI scanner"), ("transformer", "transformer"),
                 ("bakery", "bakery*"), ("architect", "architect*")],
                "(c) Magnets, same wording", [light] * 3 + [GREY] * 2)
    cx.set_ylim(-0.6, 6.2)                                         # headroom for the legend above the bars
    cx.legend(handles=[Patch(color=blue, label="model"),
                       Line2D([0], [0], color=CIRC, marker="s", linestyle="none", markersize=2.4,
                              label="circuit alone (C)")], loc="upper right", frameon=False, fontsize=5.0,
              bbox_to_anchor=(1.04, 1.02))
    panel_ko(fig.add_subplot(gs[1, 2]), d)
    S.save_figure(fig, FIG)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
