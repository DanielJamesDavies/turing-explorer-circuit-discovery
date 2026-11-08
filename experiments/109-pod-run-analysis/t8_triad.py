"""Figure: the coefficient triad redrawn from the 2026 production run (the 062 triad.py figure, fig:coef-triad).

One row per model (TuringLLM, Pythia-70m, GPT-2 small), panels as triad.py:
  (a) pass rate by depth band and site kind, (b) median faithfulness Z / A / C with IQR (1 = perfect, dashed).
Conditions, each scored by the production scorer on the held-out strongest contexts:
  fitted     the production circuits (main/eval, every non-vacuous target)
  alpha = 1  the same members with every coefficient at 1 (alpha1/alpha1, every target; same vacuous rule)
  anchor     084 anchor-active null, fitted like WCM (draw 0)       } public: out_<model>/harder_nulls;
  contrib    084 contribution-matched null, fitted like WCM (draw 0) } TuringLLM: T6 out_turingllm/harder_nulls
Nulls at alpha = 1 are not drawn (0% pass on both public models; t4_robustness.py reports them).
Pass = merge.passes, vacuous = merge.vacuous. A missing condition (TuringLLM nulls until T6 is unpacked) is left out
of that row and printed as "not yet available"; rerun after unpack.sh to fill it.

Writes  paper/figures/run2026_triad.pdf/.png, out/run2026_triad.csv (pass rate and medians per model x condition x group)

  python -X utf8 experiments/109-pod-run-analysis/t8_triad.py
"""
import numpy as np
import pandas as pd

import lib109 as L
from t4_robustness import R84, nulls_dirs

COND = [("fitted", "fitted coefficients"), ("alpha1", "same nodes, $\\alpha = 1$"),
        ("anchor", "anchor-active null, fitted"), ("contrib", "contribution-matched null, fitted")]


def conditions(model):
    out = {}
    ev = L.main_eval(model)
    if ev.empty:
        return out
    out["fitted"] = ev[(ev.held == "strong") & ~ev.vac]
    a1 = L.frame(L.read_rows(L.find(model, "out", "alpha1"), "alpha1"), keys=("seed", "held"))
    if len(a1):
        a1 = L.score(L.add_site(a1, model))
        out["alpha1"] = a1[~a1.vac]
    rows = []
    for d in nulls_dirs(model):
        rows += R84.read_rows(d, "nulls")
    if rows:
        df = R84.frame(rows)
        df = df[(df.split == "held") & (df.draw == 0)]
        df = L.score(L.add_site(df, model))
        for arm in ("anchor", "contrib"):
            x = df[(df.arm == arm) & ~df.vac]
            if len(x):
                out[arm] = x
    return out


def groups(model, kinds):
    return [("band", b, "layers\n%s" % b[1:].replace("-", "–")) for b, _, _ in L.bands_of(model)] + \
           [("kind", k, {"resid": "resid.", "resid_mid": "resid.\nmid", "mlp": "MLP", "attn": "attn."}[k])
            for k in L.KIND_ORDER if k in kinds]


def main():
    plt, st = L.figure_style()
    colors = {"fitted": st.BLUE, "alpha1": "#9aa0a8", "anchor": "#c4c8ce", "contrib": "#e0e2e6"}
    data = {m: conditions(m) for m in L.MODELS}
    models = [m for m in L.MODELS if data[m]]
    for m in L.MODELS:
        for c, _ in COND:
            if c not in data[m]:
                L.not_available("%s triad condition %s%s" % (m, c, " (T6 harder nulls)" if c in ("anchor", "contrib")
                                                              and m == "turingllm" else ""))
    if not models:
        return
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.titlepad": 4.0, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.7,
                         "grid.linewidth": 0.5, "axes.labelpad": 2.0})
    fig, axes = plt.subplots(len(models), 2, figsize=(5.5, 2.3 * len(models)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.65, 1.0], "wspace": 0.25, "hspace": 0.62})
    w = 0.2
    csv = []
    for r, model in enumerate(models):
        conds = [(j, c, lab) for j, (c, lab) in enumerate(COND) if c in data[model]]   # fixed slot per condition
        ax, bx = axes[r]
        gs = groups(model, set(data[model]["fitted"].kind))
        nb = len(L.bands_of(model))
        xs = np.array([i if i < nb else i + 0.4 for i in range(len(gs))])
        off = (len(COND) - 1) / 2
        for j, c, lab in conds:
            d = data[model][c]
            vals = []
            for by, g, _ in gs:
                sub = d[d[by] == g]
                v = 100 * sub["pass"].mean() if len(sub) else np.nan
                vals.append(v)
                csv.append(dict(model=L.LABEL[model], condition=c, group=g, n=len(sub), pass_pct=v))
            csv.append(dict(model=L.LABEL[model], condition=c, group="all", n=len(d), pass_pct=100 * d["pass"].mean(),
                            **{"%s_median" % z: d[h].median() for z, h in zip("ZAC", L.HEAD)},
                            **{"%s_p25" % z: d[h].quantile(.25) for z, h in zip("ZAC", L.HEAD)},
                            **{"%s_p75" % z: d[h].quantile(.75) for z, h in zip("ZAC", L.HEAD)}))
            x = xs + (j - off) * w
            ax.bar(x, vals, w * 0.92, color=colors[c], zorder=2, linewidth=0)
            for xi, v in zip(x, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 1.5, ("%.0f" if v >= 9.5 else "%.1f" if v > 0 else "0") % v, ha="center",
                            va="bottom", fontsize=5.5, color=st.INK, rotation=90 if v < 9.5 else 0)
            for i, h in enumerate(L.HEAD):
                v = d[h].dropna()
                med, lo, hi = v.median(), v.quantile(0.25), v.quantile(0.75)
                xb = i + (j - off) * w
                bx.bar(xb, max(med, 0), w * 0.92, color=colors[c], zorder=2, linewidth=0)
                bx.plot([xb, xb], [max(lo, -0.05), min(hi, 1.6)], color=st.INK_SECONDARY, linewidth=0.7, zorder=3)
            print("%-11s %-8s n %5d pass %5.1f%% | Z/A/C medians %.2f / %.2f / %.2f" % (
                L.LABEL[model], c, len(d), 100 * d["pass"].mean(), *[d[h].median() for h in L.HEAD]))
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, _, lab in gs])
        ax.set(ylim=(0, 108), ylabel="targets that pass (%)")
        ax.grid(axis="x", visible=False)
        short = {"fitted": "fitted", "alpha1": "α=1", "anchor": "anchor", "contrib": "contrib"}
        parts = ["%s %s" % (short[c], "{:,}".format(len(data[model][c]))) for _, c, _ in conds]
        ns = ", ".join(parts[:2]) + ("\n" + ", ".join(parts[2:]) if parts[2:] else "")
        missing = [short[c] for c, _ in COND if c not in data[model]]
        ax.set_title("(%s) %s: pass rate" % ("abc"[r], L.LABEL[model]))
        bx.axhline(1.0, color=st.INK_SECONDARY, linewidth=0.7, linestyle=(0, (3, 2)))
        bx.set_xticks(range(3))
        bx.set_xticklabels(["Z", "A", "C"])
        bx.set(ylim=(0, 1.55), ylabel="median (IQR)")
        bx.grid(axis="x", visible=False)
        bx.set_title("faithfulness")
        bx.text(1.0, 1.0, "n: %s%s" % (ns, ("\n%s: pending" % ", ".join(missing)) if missing else ""),
                transform=bx.transAxes, ha="right", va="top", fontsize=6.3, color=st.INK_MUTED)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=colors[c], linewidth=0) for c, _ in COND], [lab for _, lab in COND], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.0, columnspacing=1.2)
    L.save_with_legend(fig, L.FIGURES / ("%striad.png" % L.PREFIX), top=1 - 0.17 / len(models))
    L.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv).to_csv(L.OUT / ("%striad.csv" % L.PREFIX), index=False)
    print("figure:", L.FIGURES / ("%striad.pdf" % L.PREFIX))


if __name__ == "__main__":
    main()
