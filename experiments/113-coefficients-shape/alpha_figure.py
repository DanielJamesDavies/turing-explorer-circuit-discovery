"""Coefficient (alpha) distribution figure: by member firing frequency (all production circuits) and by member
importance (the case-study circuits, where per-member natural contributions are stored).

(a) alpha quantiles (p10 / p25 / median / p75 / p90) by the member's corpus firing-frequency decile at its site
    (0 = rarest, 9 = most frequent), all non-vacuous held-out-strongest production circuits, three models.
    Source: out/members_<model>_main.parquet joined to the frequency deciles used in run_all.py (out/
    run2026_coef_alpha_by_freq.csv already holds the quantiles; this script re-reads that CSV).
(b) alpha by the member's rank in NATURAL contribution (mean activation at the target's anchor x decoder norm, i.e.
    110's contrib / |alpha|, so alpha is not inside the ranking), pooled over the case-study circuits in
    experiments/110-case-studies-v4/out/members/*.csv. Rank buckets: top 10, 11-50, 51-200, the rest.
(c) histogram of alpha on a log axis, the same three models, split into the rarest nine deciles vs the most frequent
    decile (TuringLLM shown; the public models are in the CSV).

  python -I -X utf8 experiments/113-coefficients-shape/alpha_figure.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import lib113 as C  # noqa: E402  (also puts 109 / src paths on sys.path)

NC = __import__("nulls_core")
sys.path.insert(0, os.path.join(ROOT, "src", "analysis"))
import style  # noqa: E402  Lab Bright theme

style.configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402


def tl_members_with_freq():
    pop = C.strong_population("turingllm")
    df = C.members("turingllm", "main")
    df = df[df.seed.isin(pop.index)].copy()
    cnt = C.freq_store("turingllm")
    dec = {}
    for (l, k), _ in df.groupby(["layer", "kind"]):
        dec[(l, k)] = NC.freq_deciles(cnt[C.comp_idx("turingllm", l, k)])
    df["fdec"] = [int(dec[(l, k)][i]) for l, k, i in zip(df.layer, df.kind, df["index"])]
    return df

OUT = os.path.join(HERE, "out")
FIG = os.path.join(ROOT, "paper", "figures")
MODELS = [("turingllm", "TuringLLM"), ("pythia-70m", "Pythia-70m"), ("gpt2-small", "GPT-2 small")]
COL = dict(zip([m for m, _ in MODELS], style.CATEGORICAL[:3]))


def by_rank():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "experiments", "110-case-studies-v4", "out", "members", "*.csv"))):
        d = pd.read_csv(f)
        d = d[d.alpha.abs() > 0].copy()
        d["natural"] = d.contrib / d.alpha.abs()
        d = d.sort_values("natural", ascending=False).reset_index(drop=True)
        d["rank"] = np.arange(1, len(d) + 1)
        d["bucket"] = pd.cut(d["rank"], [0, 10, 50, 200, np.inf], labels=["top 10", "11-50", "51-200", "rest"])
        d["target"] = os.path.basename(f)[:-4]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def q(s):
    return pd.Series({"p10": s.quantile(.1), "p25": s.quantile(.25), "median": s.median(), "p75": s.quantile(.75),
                      "p90": s.quantile(.9), "share_gt1": (s > 1).mean(), "n": len(s)})


def main():
    fq = pd.read_csv(os.path.join(OUT, "run2026_coef_alpha_by_freq.csv"))
    fq = fq[fq.group.astype(int) >= 0]
    rk = by_rank()
    rk_tab = rk.groupby("bucket", observed=True).alpha.apply(q).unstack()
    rk_tab.to_csv(os.path.join(OUT, "run2026_coef_alpha_by_importance.csv"))
    per_target = rk.groupby(["target", "bucket"], observed=True).alpha.median().unstack()
    per_target.to_csv(os.path.join(OUT, "run2026_coef_alpha_by_importance_targets.csv"))
    print(rk_tab.round(3).to_string())
    print(per_target.round(2).to_string())

    tl = tl_members_with_freq()
    for lab, s in (("rarer 90% (deciles 0-8)", tl.alpha[(tl.fdec >= 0) & (tl.fdec < 9)]),
                   ("most frequent 10%", tl.alpha[tl.fdec == 9])):
        print("TuringLLM %s: n %d median %.2f p75 %.2f p90 %.2f p99 %.2f share>1.5 %.1f%% share>2 %.1f%%" % (
            lab, len(s), s.median(), s.quantile(.75), s.quantile(.9), s.quantile(.99), 100 * (s > 1.5).mean(),
            100 * (s > 2).mean()))

    from matplotlib.ticker import NullLocator
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1, 1.1]))
    # (a) frequency deciles
    for j, (m, lab) in enumerate(MODELS):
        g = fq[fq.model == m].sort_values("group")
        x = g.group.astype(int).values + (j - 1) * 0.22
        ax[0].vlines(x, g.alpha_p10, g.alpha_p90, color=COL[m], lw=1.2, alpha=.6)
        ax[0].plot(x, g.alpha_median, "o", color=COL[m], ms=4, label=lab)
    ax[0].axhline(1, color="0.5", ls="--", lw=.8)
    ax[0].set_yscale("log")
    ax[0].set_yticks([0.5, 1, 2, 3])
    ax[0].set_yticklabels(["0.5", "1", "2", "3"])
    ax[0].set_xticks(range(10))
    ax[0].set_xticklabels(["rare"] + [""] * 8 + ["frequent"])
    ax[0].set_xlabel("member's firing-frequency decile at its site")
    ax[0].set_ylabel(r"fitted $\alpha$ (median, p10-p90)")
    ax[0].yaxis.set_minor_locator(NullLocator())
    ax[0].set_ylim(0.5, 3.2)
    ax[0].set_title("(a) by firing frequency", loc="left")
    ax[0].legend(frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    # (b) importance buckets (case studies)
    labs = list(rk_tab.index)
    for i, b in enumerate(labs):
        s = rk[rk.bucket == b].alpha
        ax[1].boxplot(s, positions=[i], widths=.5, showfliers=False, patch_artist=True,
                      boxprops=dict(facecolor=style.CATEGORICAL[0], alpha=.35, edgecolor=style.CATEGORICAL[0]),
                      medianprops=dict(color=style.CATEGORICAL[0], lw=2), whiskerprops=dict(color="0.4"),
                      capprops=dict(color="0.4"))
    ax[1].axhline(1, color="0.5", ls="--", lw=.8)
    ax[1].set_yscale("log")
    ax[1].set_yticks([0.5, 1, 2, 3])
    ax[1].set_yticklabels(["0.5", "1", "2", "3"])
    ax[1].set_xticks(range(len(labs)))
    ax[1].set_xticklabels(labs)
    ax[1].set_xlabel("member's rank by natural contribution")
    ax[1].yaxis.set_minor_locator(NullLocator())
    ax[1].set_ylim(0.2, 3.5)
    ax[1].set_title("(b) by importance (%d circuits)" % rk.target.nunique(), loc="left")
    # (c) histogram, TuringLLM, rare vs frequent
    bins = np.logspace(np.log10(0.1), np.log10(10), 61)
    dec = tl.fdec
    ax[2].hist(tl.alpha[(dec >= 0) & (dec < 9)].clip(0.1, 10), bins=bins, histtype="step",
               color=style.CATEGORICAL[0], lw=1.5, density=True, label="rarer 90% of latents")
    ax[2].hist(tl.alpha[dec == 9].clip(0.1, 10), bins=bins, histtype="step", color="0.45", lw=1.3,
               density=True, label="most frequent 10%")
    ax[2].axvline(1, color="0.5", ls="--", lw=.8)
    ax[2].set_xscale("log")
    ax[2].set_xlabel(r"fitted $\alpha$ (log scale)")
    ax[2].set_ylabel("density")
    ax[2].set_title("(c) TuringLLM, all members", loc="left")
    ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "run2026_coef_alpha_dist.%s" % ext), dpi=200, bbox_inches="tight")
    print("wrote", os.path.join(FIG, "run2026_coef_alpha_dist.png"))


if __name__ == "__main__":
    main()
