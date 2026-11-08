"""Two-panel case figure for any out/case/<KEY>.json (case_v4.py with GROUPS_JSON), Lab Bright style, 5.5 in wide.

  (a) knock-outs: Z / A / C with each group removed (bars) vs rank-matched random removals (mean +- sd), pass band
  (b) probes: the target in the clean model (bars) and the circuit run alone under contrast-mean ablation (squares),
      as % of the target's mean activation at its held-out strongest anchors; the empty circuit reads 0 under C.

  python -X utf8 experiments/110-case-studies-v4/case_figure_generic.py    env: KEY  NAME (figure stem suffix)  TITLE
"""
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib110 as L  # noqa: E402
from analysis import style as S  # noqa: E402

KEY = os.environ.get("KEY", "9.resid.13945")
NAME = os.environ.get("NAME", "fusion")
TITLE = os.environ.get("TITLE", "")
GREY = "#9aa0a8"


def main():
    plt = S.configure_matplotlib()
    plt.rcParams.update({"font.size": 6.2, "axes.titlesize": 7.0, "axes.labelsize": 6.0, "xtick.labelsize": 5.4,
                         "ytick.labelsize": 5.6, "legend.fontsize": 5.4, "axes.linewidth": 0.6})
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    d = json.loads((L.RESULTS / "case" / ("%s.json" % KEY)).read_text(encoding="utf-8"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.55))
    K = d["knockouts"]
    order = [g for g in K if g != "full"]
    w = 0.26
    ax.axhspan(0.75, 1.5, color=S.tint(S.CATEGORICAL[3], 0.9), zorder=0, linewidth=0)
    for gi, g in enumerate(order):
        col = GREY if "control" in g else (S.INK if g.startswith("all") else S.CATEGORICAL[gi % 6])
        for si, (sc, amt) in enumerate((("Z", 0.6), ("A", 0.3), ("C", 0.0))):
            x = gi + (si - 1) * w
            v = K[g]["ko"][sc]
            ax.bar(x, max(v, 0.0), w * 0.9, color=S.tint(col, amt), zorder=2, linewidth=0)
            if v < 0.03:
                ax.text(x, 0.02, "%.2f" % v, ha="center", va="bottom", fontsize=4.6, color=S.INK, rotation=90)
            rnd = [r[sc] for r in K[g]["random"] if r[sc] is not None]
            m, s = float(np.mean(rnd)), float(np.std(rnd))
            ax.plot([x - w * 0.42, x + w * 0.42], [m, m], color=S.INK, linewidth=0.9, zorder=4)
            ax.plot([x, x], [m - s, m + s], color=S.INK, linewidth=0.5, zorder=4)
    short = {"plasma confinement": "plasma\nconfin.", "nuclear and power engineering": "nuclear,\npower",
             "stars, cosmos, particles": "stars,\ncosmos", "quantum and magnetism": "quantum,\nmagnet.",
             "off-topic (control)": "off-\ntopic*", "generic core (control)": "generic\ncore*",
             "all story groups": "all\nstory", "name -> law": "name→\nlaw", "physics context": "physics\ncontext",
             "generic possessive (control)": "possess-\nive*",
             "AI word forms (A-I / artificial intelligence)": "AI word\nforms", "machine learning and agents":
             "ML,\nagents", "technology applications": "tech\napplic."}
    labs = [short.get(g, "\n".join(textwrap.wrap(g, 9))) for g in order]
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(labs)
    ax.set_xlim(-0.5, len(order) - 0.5); ax.set_ylim(0, 1.45); ax.set_ylabel("faithfulness")
    handles = [Patch(color=S.tint(S.INK_MUTED, a), label=l) for l, a in (("Z", 0.6), ("A", 0.3), ("C", 0.0))]
    handles.append(Line2D([0], [0], color=S.INK, linewidth=0.9, label="random"))
    ax.legend(handles=handles, loc="upper left", ncol=4, handlelength=0.9, columnspacing=0.6, frameon=False)
    ax.set_title("(a) Removing a group (* = control)")

    P = d["probes"]
    ref = P["ref_a_pos"]
    rows = P["sentences"]
    cats = list(dict.fromkeys(r["category"] for r in rows))
    y = np.arange(len(rows))[::-1]
    cols = [S.CATEGORICAL[cats.index(r["category"]) % 6] if cats.index(r["category"]) == 0 else
            S.tint(S.BLUE, 0.45) if cats.index(r["category"]) < len(cats) - 1 else GREY for r in rows]
    bx.barh(y, [100 * r["clean"] / ref for r in rows], color=cols, height=0.62)
    bx.plot([100 * r["C"] / ref for r in rows], y, linestyle="none", marker="s", markersize=2.4, color=S.INK, zorder=4)
    bx.set_yticks(y)
    bx.set_yticklabels([textwrap.shorten(r["sentence"], 34, placeholder="…") for r in rows], fontsize=4.8)
    bx.set_xlim(0, 135); bx.set_xticks([0, 50, 100]); bx.grid(axis="y", visible=False)
    bx.set_xlabel("target, % of reference")
    bx.legend(handles=[Patch(color=S.BLUE, label="model"),
                       Line2D([0], [0], color=S.INK, marker="s", linestyle="none", markersize=2.4,
                              label="circuit alone (C)")], loc="lower right", frameon=False)
    bx.set_title("(b) Probe sentences")
    if TITLE:
        fig.suptitle(TITLE, fontsize=7.4, y=1.02)
    out = L.FIGS / ("run2026_case_%s.png" % NAME)
    S.save_figure(fig, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
