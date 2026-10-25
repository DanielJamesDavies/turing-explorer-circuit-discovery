"""DAN-64 / DAN-160 figure: contrast faithfulness and pass rate by evaluation background (groups) and training contrast
source (bars), at matched circuit size, with 95% bootstrap intervals over targets. The new-run version of the earlier
"median counterfactual faithfulness by method and negative-context mode" bar chart.

Numbers are report.py's: the matched-size values come from logic.matched_cells at n* = the median size of close_silent
at lambda 1e-3; intervals resample targets with n* recomputed per resample, as report.py's bootstrap does.

  OUT=experiments/071-contrast-sweep/out PYTHONPATH=src python experiments/071-contrast-sweep/source_figure.py
  env: B (1000 resamples)  BAND (overall | L0-4 | L5-7 | L8-11)
Writes OUT/figures/source_battery[_<band>].{png,pdf} and prints the plotted values.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logic as L  # noqa: E402
import report as R  # noqa: E402

B = int(os.environ.get("B", 1000))
BAND = os.environ.get("BAND", "overall")
PAPER = os.environ.get("PAPER") == "1"                  # draw at the paper's text width into paper/figures/
NAMES = {"close_silent": "close", "random_silent": "random", "distant_silent": "distant", "fused_silent": "fused"}


def only_c_fails(D):
    """[target, arm, lambda, eval] indicator: Z and A in band and necessity >= 0.9 (everything the pass rule asks
    except C), but C out of band or undefined. Pass + this = 'would pass if the C term were perfect'."""
    lo, hi, nec_min = L.PASS_LO, L.PASS_HI, L.NEC_MIN
    X = D["X"]
    inb = lambda v: np.isfinite(v) & (v >= lo) & (v <= hi)  # noqa: E731
    rest = inb(X["Z"]) & inb(X["A"]) & np.isfinite(X["nec"]) & (X["nec"] >= nec_min)
    return (rest & ~inb(X["C"])).astype(float)


def values(D, idx, ra, rl):
    """(pass [arm, eval], fails-only-C [arm, eval], median C [arm, eval], n*) at matched size over the targets idx."""
    n_star = float(np.median(D["N"][idx, ra, rl]))
    p = L.matched_cells(D["P"], D["N"], idx, n_star, "mean")[0]
    oc = L.matched_cells(D["OC"], D["N"], idx, n_star, "mean")[0]
    c = L.matched_cells(D["X"]["C"], D["N"], idx, n_star, "median")[0]
    return p, oc, c, n_star


def main():
    import matplotlib.pyplot as plt
    from analysis import style as S
    D = R.arrays(R.load("eval", ["seed", "arm", "lam", "eval"]))
    D["OC"] = only_c_fails(D)
    assert not np.any((D["OC"] > 0) & (D["P"] > 0)), "a passing row cannot also fail only C"
    arms, evals, common = D["arms"], D["evals"], D["common"]
    ra = D["ai"][L.REFERENCE]
    rl = [i for l, i in D["li"].items() if abs(l - L.REF_LAM) < 1e-12][0]
    idx = np.arange(len(common))
    if BAND != "overall":
        idx = idx[np.array([L.band(int(t.split(".")[0])) == BAND for t in common])]
    p, oc, c, n_star = values(D, idx, ra, rl)
    rng = np.random.default_rng(0)
    bp, bc = np.zeros((B,) + p.shape), np.zeros((B,) + c.shape)
    for b in range(B):
        s = idx[rng.integers(0, len(idx), len(idx))]
        bp[b], _, bc[b], _ = values(D, s, ra, rl)
    lo_p, hi_p = np.nanpercentile(bp, [2.5, 97.5], axis=0)
    lo_c, hi_c = np.nanpercentile(bc, [2.5, 97.5], axis=0)

    print("%s: %d targets, matched size n* = %.0f members" % (BAND, len(idx), n_star))
    for name, v, lo, hi in (("median C faithfulness", c, lo_c, hi_c), ("pass rate", p, lo_p, hi_p)):
        print("\n%s (rows: trained on; columns: evaluated on)" % name)
        print("%-9s" % "" + "".join("%-22s" % NAMES[e] for e in evals))
        for i, a in enumerate(arms):
            print("%-9s" % NAMES[a] + "".join("%-22s" % ("%.3f [%.3f, %.3f]" % (v[i, j], lo[i, j], hi[i, j]))
                                              for j in range(len(evals))))
    print("\nfail ONLY the contrast ablation (Z, A, necessity hold; C out of band), matched size")
    print("%-9s" % "" + "".join("%-22s" % NAMES[e] for e in evals))
    for i, a in enumerate(arms):
        print("%-9s" % NAMES[a] + "".join("%-22s" % ("%.3f (pass + it %.3f)" % (oc[i, j], p[i, j] + oc[i, j]))
                                          for j in range(len(evals))))

    S.configure_matplotlib()
    # Fused is a mix of the other three sources: purple, the palette slot after green (Daniel, 2026-10-11).
    colors = {"close_silent": S.NEG_MODE_COLORS["close"], "random_silent": S.NEG_MODE_COLORS["random"],
              "distant_silent": S.NEG_MODE_COLORS["distant"], "fused_silent": S.CATEGORICAL[4]}
    width, offs = S.grouped_bar_geometry(len(arms))
    # PAPER=1: drawn at the paper's text width (5.5 in, \linewidth) so it is never scaled down; fonts sized for print.
    # The paper stacks the panels (each gets the full width, so the bars are wide enough for their labels).
    F = (dict(fig=(5.5, 5.1), val=6.0, inside=5.8, ylab=7.5, xlab=7.8, tick=7.2, leg=6.8, ncol=3, top=0.9,
              grid=(2, 1))
         if PAPER else dict(fig=(11.0, 4.0), val=7.5, inside=7, ylab=10, xlab=11, tick=10.5, leg=9, ncol=5, top=0.93,
                            grid=(1, 2)))
    fig, axes = plt.subplots(*F["grid"], figsize=F["fig"])
    panels = ((axes[0], c, "median C faithfulness" if PAPER else "median contrast-ablation faithfulness (C)"),
              (axes[1], p, "pass rate"))
    for k, (ax, v, ylabel) in enumerate(panels):
        for i, a in enumerate(arms):
            xs = np.arange(len(evals)) + offs[i]
            if k == 1:                                   # light segment: fails ONLY the contrast ablation
                ax.bar(xs, oc[i], width, bottom=v[i], color=S.tint(colors[a], 0.6), zorder=2)
            ax.bar(xs, v[i], width, color=colors[a], label="trained on %s" % NAMES[a], zorder=2)
            top = v[i] + oc[i] if k == 1 else v[i]
            for x, y, t in zip(xs, v[i], top):
                ax.text(x, t + 0.008 * (0.55 if k == 1 else 1.05) / 0.5, "%.2f" % t, ha="center", va="bottom",
                        fontsize=F["val"], color=S.INK_SECONDARY, zorder=4)       # the top value, above the bar
                if k == 1:                               # the pass value, inside the solid segment
                    ax.text(x, y - 0.008, "%.2f" % y, ha="center", va="top", fontsize=F["inside"], color="white",
                            fontweight="bold", zorder=4)
        ax.set_xticks(range(len(evals)), [NAMES[e].capitalize() for e in evals])
        ax.tick_params(labelsize=F["tick"])
        ax.set_xlabel("Evaluated on", fontsize=F["xlab"])
        ax.set_ylabel(ylabel, fontsize=F["ylab"])        # a touch smaller than the axis-label default (Daniel)
        ax.set_ylim(0, float(np.nanmax(p + oc)) + 0.05 if k == 1 else 1.05)
        S.round_bars(ax, radius=0.3)
    axes[0].axhline(1.0, color=S.BASELINE, lw=0.8, ls=":", zorder=1)
    from matplotlib.patches import Patch
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=S.tint(S.INK_MUTED, 0.6)))
    labels.append("lighter shade: fails only the contrast ablation")
    fig.legend(handles, labels, loc="upper center", ncol=F["ncol"], frameon=False, fontsize=F["leg"],
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, F["top"]))
    if PAPER:
        out = HERE.parent.parent / "paper" / "figures" / "contrast-source.png"
    else:
        stem = "source_battery" + ("" if BAND == "overall" else "_" + BAND.replace("-", ""))
        (R.OUT / "figures").mkdir(parents=True, exist_ok=True)
        out = R.OUT / "figures" / ("%s.png" % stem)
    S.save_figure(fig, out)                              # also writes the .pdf sibling
    print("\nfigure: %s" % out)


if __name__ == "__main__":
    main()
