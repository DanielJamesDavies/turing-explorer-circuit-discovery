"""Experiment 080 figure: the 2 x 2 cross-score panel. Rows: the circuit was trained on the latent endpoint (LAT, the
062 production circuit) or on the output endpoint (OUT_W, at the LAT circuit's size). Columns: scored on the latent
metric or on the output metric. Bars: the design's tiers (causal: high / low next-token effect; legacy: direct-path
(064) high / mid / low), median over targets with 95% bootstrap intervals (report.py's numbers: report.matched,
common.paired_bootstrap). The diagonal is each circuit on its own endpoint; the off-diagonal is the cross-score.
Every target is an internal latent; the tier is how much it affects the next token. Descriptive: a tier's bar pools
the kind x band groups, so tier effects are read off report.py's within-group P1 contrasts, not off this figure.

  OUT=experiments/080-latent-vs-output/out PYTHONPATH=src python experiments/080-latent-vs-output/figure.py
  env: METRIC (Z | A | C: the ablation whose faithfulness is plotted, default Z)  B (2000)
       PAPER=1 (drawn at the paper's 5.5 in text width into paper/figures/latent-vs-output.{png,pdf})
Writes OUT/figures/cross_score_<METRIC>.{png,pdf} and prints the plotted values.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import report as R  # noqa: E402

METRIC = os.environ.get("METRIC", "Z")
PAPER = os.environ.get("PAPER") == "1"
B = int(os.environ.get("B", C.B_BOOT))
NAMES = {"Z": "zero", "A": "activating-mean", "C": "contrast-mean"}


def cells(M):
    """{(row, col): [(tier, median, lo, hi, n)]}: row 0 trained on latent / 1 on output; col 0 latent / 1 output."""
    keys = {(0, 0): "LAT_" + METRIC, (0, 1): "LAT_o" + METRIC, (1, 0): "OUT_W_" + METRIC, (1, 1): "OUT_W_o" + METRIC}
    out = {}
    for rc, k in keys.items():
        vals = []
        for t in R.tiers_of(M):
            ms = [m for m in M if m["tier"] == t and not (rc[1] == 1 and m["out_vacuous"])]
            d = C.paired_bootstrap([m.get(k) for m in ms], B=B)
            vals.append((t, d["a"], d["a_lo"], d["a_hi"], d["n"]))
        out[rc] = vals
    return out


def main():
    import matplotlib.pyplot as plt
    from analysis import style as S
    rows, _ = R.load()
    M = R.matched(R.per_target(rows))
    if not M:
        sys.exit("no scored targets in %s" % R.OUT)
    V = cells(M)
    for rc, vals in sorted(V.items()):
        print("trained on %-6s scored on %-6s: %s" % (("latent", "output")[rc[0]], ("latent", "output")[rc[1]],
                                                      "  ".join("%s %s [%s, %s] (n %d)" % (t, R.f3(m), R.f3(lo),
                                                                                          R.f3(hi), n)
                                                                for t, m, lo, hi, n in vals)))
    S.configure_matplotlib()
    tiers = R.tiers_of(M)
    colors = dict(zip(tiers, S.ordinal_blues(max(len(tiers), 2))[::-1]))   # the highest next-token effect darkest
    short = {"high": "high effect", "low": "low effect", "dp_high": "DP high", "dp_mid": "DP mid", "dp_low": "DP low"}
    F = (dict(fig=(5.5, 4.3), val=6.2, tick=7.2, lab=7.6, title=8.2, leg=7.0) if PAPER
         else dict(fig=(9.0, 6.6), val=8.5, tick=10, lab=10.5, title=11.5, leg=10))
    fig, axes = plt.subplots(2, 2, figsize=F["fig"], sharey=True)
    finite = [x for v in V.values() for _, m, lo, hi, _ in v for x in (m, lo, hi) if x == x]
    ylo = min(0.0, min(finite) - 0.05) if finite else 0.0
    yhi = max(1.15, max(finite) + 0.1) if finite else 1.15
    for (r, c), vals in V.items():
        ax = axes[r, c]
        xs = np.arange(len(vals))
        med = np.array([m for _, m, _, _, _ in vals], dtype=float)
        lo = np.array([l for _, _, l, _, _ in vals], dtype=float)
        hi = np.array([h for _, _, _, h, _ in vals], dtype=float)
        ok = np.isfinite(med)                                  # a tier without targets draws nothing
        ax.bar(xs[ok], med[ok], S.BAR_WIDTH, color=[colors[t] for (t, *_), k in zip(vals, ok) if k], zorder=2)
        ax.errorbar(xs[ok], med[ok], yerr=[np.nan_to_num(med - lo)[ok], np.nan_to_num(hi - med)[ok]], fmt="none",
                    ecolor=S.INK_SECONDARY, elinewidth=1.0, capsize=2.5, zorder=3)
        for x, m, h in zip(xs, med, hi):
            if m == m:
                ax.text(x, (h if h == h else m) + 0.02 * (yhi - ylo), "%.2f" % m, ha="center", va="bottom",
                        fontsize=F["val"], color=S.INK_SECONDARY, zorder=4)
        ax.axhline(1.0, color=S.BASELINE, lw=0.8, ls=":", zorder=1)
        ax.axhline(0.0, color=S.BASELINE, lw=0.8, zorder=1)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(xs, [short.get(t, t) for t, *_ in vals])
        ax.tick_params(labelsize=F["tick"])
        diag = r == c
        ax.set_title("%s-trained, scored on %s%s" % (("latent", "output")[r], ("latent", "output")[c],
                                                    "" if diag else " (cross)"), fontsize=F["title"])
        if c == 0:
            ax.set_ylabel("median %s faithfulness" % NAMES[METRIC], fontsize=F["lab"])
        if r == 1:
            ax.set_xlabel("next-token effect tier (within kind x band)" if "high" in tiers
                          else "direct-path (064) tier", fontsize=F["lab"])
    fig.tight_layout()
    if PAPER:
        out = C.EXP.parent / "paper" / "figures" / "latent-vs-output.png"
    else:
        out = R.OUT / "figures" / ("cross_score_%s.png" % METRIC)
    S.save_figure(fig, out)                                # also writes the .pdf sibling
    print("figure: %s" % out)


if __name__ == "__main__":
    main()
