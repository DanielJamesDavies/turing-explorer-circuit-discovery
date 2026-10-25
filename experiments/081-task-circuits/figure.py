"""Experiment 081 figure (CPU), Lab Bright theme (src/analysis/style.py; the patterns of 071's source_figure.py).

  Panel A  size-faithfulness per competent task: held-out task-metric Z faithfulness against circuit size (log), WCM
           (W, fitted coefficients) and unweighted (U, alpha = 1), filled markers where the circuit passes the v1 rule;
           the random null as grey markers at W's sizes (median Z over the draws, bar from min to max); the pass band
           [0.8, 1.5] shaded
  Panel B  competence: the share of held-out prompts the full model gets right, per task, competent tasks in blue,
           the rest grey, the 70% threshold dashed
Numbers are report.py's (report.task_curves, probe/competence.json); the figure prints what it plots.

  OUT=experiments/081-task-circuits/out PYTHONPATH=src python experiments/081-task-circuits/figure.py
  env: OUT  OUT_PROBE  PAPER=1 (drawn at the paper's 5.5 in text width into paper/figures/task-circuits.{png,pdf})
       METRIC (Z | A | C: the ablation plotted in panel A, default Z)
Writes OUT/figures/task_circuits_<METRIC>.{png,pdf} (or the paper path).
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task_common as C  # noqa: E402
import report as R  # noqa: E402

PAPER = os.environ.get("PAPER") == "1"
METRIC = os.environ.get("METRIC", "Z")
FIELD = {"Z": "task_free0", "A": "task_freeM_topk", "C": "task_freeN_topk"}[METRIC]
SHORT = {"greater_than": "greater-than", "sva_simple": "SVA simple", "sva_within_rc": "SVA within RC",
         "sva_across_rc": "SVA across RC", "sva_across_pp": "SVA across PP", "gender_pronoun": "gendered pronoun",
         "ioi": "IOI", "induction": "induction", "factual_known": "known facts"}


def short(name):
    if name.startswith(C.BLIMP_PREFIX):
        return "BLiMP " + name[len(C.BLIMP_PREFIX):].replace("_", " ")
    return SHORT.get(name, name)


def main():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from analysis import style as S
    rows, _, comp = R.load()
    TC = R.task_curves(rows) if rows else {}
    tasks = [t for t in sorted(TC) if TC[t]["W"]]
    if not tasks and not comp:
        sys.exit("nothing to plot in %s / %s" % (R.OUT, R.OUT_PROBE))
    S.configure_matplotlib()
    cw, cu, cr = S.CATEGORICAL[0], S.CATEGORICAL[1], S.INK_MUTED
    F = (dict(w=5.5, panel=1.25, comp=1.7, tick=6.0, lab=6.8, title=6.8, leg=6.2, ms=3.2) if PAPER
         else dict(w=12.0, panel=2.6, comp=3.2, tick=9.0, lab=10, title=10, leg=9, ms=5))
    ncol = 4 if PAPER else 5
    nrow = max(1, math.ceil(len(tasks) / ncol)) if tasks else 0
    fig = plt.figure(figsize=(F["w"], nrow * F["panel"] + F["comp"] + 0.4))
    gs = GridSpec(nrow + 1, ncol, figure=fig, height_ratios=[F["panel"]] * nrow + [F["comp"]])
    key = {"Z": "Z", "A": "A", "C": "C"}[METRIC]
    print("panel A (%s faithfulness, held-out task metric):" % METRIC)
    legend = {}
    for k, t in enumerate(tasks):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        ax.axhspan(C.BAND_LO, C.BAND_HI, color=S.tint(cw, 0.9), zorder=0)
        ax.axhline(1.0, color=S.BASELINE, lw=0.6, ls=":", zorder=1)
        ax.axhline(0.0, color=S.BASELINE, lw=0.6, zorder=1)
        for arm, col, lab, ls in (("W", cw, "WCM (fitted coefficients)", "-"), ("U", cu, "unweighted (alpha = 1)", "-"),
                                  ("W_ZA", S.CATEGORICAL[2], "WCM trained on Z + A only (sensitivity)", "--")):
            cv = [p for p in TC[t].get(arm, []) if p["n"] > 0 and p[key] is not None]
            if not cv:
                continue
            ns, ys = [p["n"] for p in cv], [p[key] for p in cv]
            ax.plot(ns, ys, ls, color=col, lw=1.2, zorder=3, label=lab)
            for p in cv:
                ax.plot(p["n"], p[key], "o", ms=F["ms"], zorder=4, color=col,
                        mfc=col if p["passed"] else "white", mew=1.0)
            print("  %-30s %s: %s" % (t, arm, "  ".join("n%d:%.2f%s" % (n, y, "*" if p["passed"] else "")
                                                        for n, y, p in zip(ns, ys, cv))))
        rx, rm, rlo, rhi = [], [], [], []
        for lt, rs in TC[t]["RND"].items():
            zs = sorted(C.num(r.get(FIELD)) for r in rs if C.num(r.get(FIELD)) is not None)
            if zs:
                rx.append(float(np.median([r["n"] for r in rs])))
                rm.append(zs[len(zs) // 2]); rlo.append(zs[0]); rhi.append(zs[-1])
        if rx:
            ax.errorbar(rx, rm, yerr=[np.array(rm) - np.array(rlo), np.array(rhi) - np.array(rm)], fmt="s",
                        ms=F["ms"] * 0.8, color=cr, ecolor=cr, elinewidth=0.8, capsize=1.5, zorder=2,
                        label="random, fitted coefficients")
            print("  %-30s RND: %s" % (t, "  ".join("n%.0f:%.2f[%.2f,%.2f]" % v for v in zip(rx, rm, rlo, rhi))))
        ax.set_xscale("log")
        ax.set_title(short(t), fontsize=F["title"])
        ax.tick_params(labelsize=F["tick"])
        lo = min([0.0] + [y for a in ("W", "U") for p in TC[t][a] for y in [p[key]] if y is not None] + rlo)
        hi = max([1.3] + [y for a in ("W", "U") for p in TC[t][a] for y in [p[key]] if y is not None] + rhi)
        ax.set_ylim(max(lo, -0.5) - 0.05, min(hi, 2.0) + 0.05)
        if k % ncol == 0:
            ax.set_ylabel("%s faithfulness" % METRIC, fontsize=F["lab"])
        if k // ncol == nrow - 1:
            ax.set_xlabel("circuit size (latents)", fontsize=F["lab"])
        for h, lab in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(lab, h)
    if legend:
        fig.legend(list(legend.values()), list(legend), loc="upper center", ncol=2 if PAPER else 4, frameon=False,
                   fontsize=F["leg"], bbox_to_anchor=(0.5, 1.0))
    axb = fig.add_subplot(gs[nrow, :])
    names = list(comp)
    vals = [comp[n]["held"]["frac_correct"] for n in names]
    cols = [cw if comp[n]["competent"] else S.tint(S.INK_MUTED, 0.45) for n in names]
    xs = np.arange(len(names))
    axb.bar(xs, vals, S.BAR_WIDTH, color=cols, zorder=2)
    for x, v, n in zip(xs, vals, names):
        axb.text(x, v + 0.02, "%.0f%%" % (100 * v), ha="center", va="bottom", fontsize=F["tick"] * 0.9,
                 color=S.INK_SECONDARY)
    axb.axhline(C.COMPETENCE["min_frac_correct"], color=S.BASELINE, lw=0.8, ls="--", zorder=1)
    axb.set_xticks(xs, [short(n) for n in names], rotation=35, ha="right", fontsize=F["tick"])
    axb.set_ylim(0, 1.12)
    axb.set_ylabel("full model correct\n(held-out)", fontsize=F["lab"])
    axb.tick_params(labelsize=F["tick"])
    S.round_bars(axb, radius=0.3)
    print("panel B (share correct, held-out): %s" % "  ".join("%s %.2f%s" % (n, v, "" if comp[n]["competent"]
                                                                            else " (not competent)")
                                                              for n, v in zip(names, vals)))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if PAPER:
        out = C.ROOT / "paper" / "figures" / ("task-circuits.png" if C.MODEL == "gpt2-small"
                                               else "task-circuits-%s.png" % C.MODEL)
    else:
        out = R.OUT / "figures" / ("task_circuits_%s.png" % METRIC)
    S.save_figure(fig, out)                               # also writes the .pdf sibling
    print("figure: %s" % out)


if __name__ == "__main__":
    main()
