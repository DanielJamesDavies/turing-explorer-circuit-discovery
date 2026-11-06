"""BODY-FIGURE CANDIDATE (DAN-148, depth framing): pass rate against circuit size per depth band, across the lambda
sweep, for weighted (WCM, W_ arms) and unweighted (U_ arms, every alpha = 1) circuit masking. It shows "deeper targets
need larger circuits": each band's WCM curve rises with size and the deep curve sits to the right.

One point per (band, arm): x = median circuit size (upstream latents), y = pass rate (DAN-8 rule via 062/merge.py,
vacuous circuits excluded) with its 95% Wilson interval; the production penalty (W_0.001) is ringed. Rows are the
held-out strongest scores of the sweep (OUT/sweep/eval.shard*.jsonl); W_0.001 falls back to the main run's rows for
sweep targets when the sweep has none (chain.sh: the v2 sweep may reuse the T3 main circuit). Arms with "_s" (step
variants) are not penalty points and are dropped.

  python -X utf8 paper/tables/depth_frontier.py                                    # v2: 094 out_v2/sweep
  python -X utf8 paper/tables/depth_frontier.py --tag v1-validation --out experiments/062-h100-protocol-v1/out_full/out \
      --extra experiments/062-h100-protocol-v1/out_bands experiments/062-h100-protocol-v1/out_depth \
      experiments/062-h100-protocol-v1/out_lowlam          (v1: the weak WCM arms live in three extra OUTs)

Writes paper/tables/out/<tag>/: depth-frontier.pdf/.png (Lab Bright, src/analysis/style.py) and depth_frontier.csv
(band, arm, method, lambda, targets, median size, pass rate and interval) plus the smallest WCM median size at which
each band first reaches 50% pass. The v1 output is validation only; the paper figure is drawn from the v2 sweep.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as K


def load(M, out, extra):
    frames = []
    for d in [out] + list(extra):
        ev = K.clean(M.read(Path(d) / "sweep" / "eval.*.jsonl"))
        if not ev.empty:
            frames.append(ev.assign(src=Path(d).name))
    if not frames:
        raise SystemExit("no sweep rows under %s" % [out] + list(extra))
    ev = pd.concat(frames, ignore_index=True)
    ev = ev[(ev.held == "strong") & ~ev.arm.str.contains("_s")]
    ev = ev.drop_duplicates(["seed", "arm"], keep="last")
    sweep_targets = set(ev.seed)
    if not (ev.arm == "W_0.001").any():                                # v2: the production arm may be the main fit
        main = K.clean(M.read(Path(out) / "main" / "eval.*.jsonl"))
        if not main.empty:
            main = main[(main.held == "strong") & main.seed.isin(sweep_targets)].drop_duplicates("seed", keep="last")
            ev = pd.concat([ev, main.assign(arm="W_0.001", src="main")], ignore_index=True)
    ev = K.add_site(ev, M)
    ev["method"] = np.where(ev.arm.str.startswith("W_"), "WCM", "unweighted")
    ev["lam"] = ev.arm.str[2:].astype(float)
    ev = ev[~M.vacuous(ev)].copy()
    ev["passes"] = M.passes(ev)
    return ev


def points(ev, M):
    rows = []
    for (band, method, lam), g in ev.groupby(["depth", "method", "lam"]):
        k, n = int(g.passes.sum()), len(g)
        lo, hi = K.wilson(k, n)
        rows.append(dict(band=band, method=method, lam=lam, targets=n, size=float(g.n.median()),
                         pass_pct=100 * k / n, ci_lo=100 * lo, ci_hi=100 * hi))
    order = {b: i for i, (b, _, _) in enumerate(M.BANDS)}
    return pd.DataFrame(rows).sort_values(["band", "method", "size"], key=lambda c: c.map(order) if c.name == "band"
                                          else c).reset_index(drop=True)


def first_reach(p, level=50.0):
    out = {}
    for band, g in p[p.method == "WCM"].groupby("band"):
        hit = g[g.pass_pct >= level].sort_values("size")
        out[band] = float(hit["size"].iloc[0]) if len(hit) else float("nan")
    return out


def figure(p, M):
    import sys
    sys.path.insert(0, str(K.ROOT / "src"))
    from analysis.style import CATEGORICAL, INK, INK_MUTED, SURFACE, configure_matplotlib, tint  # noqa: E402
    plt = configure_matplotlib()
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.titlesize": 8, "axes.titlepad": 4, "lines.linewidth": 1.4})
    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    for (band, lo, hi), color in zip(M.BANDS, CATEGORICAL):
        lab = "layers %s" % K.band_tex(lo, hi).replace("--", "–")
        for method, style in (("WCM", "-"), ("unweighted", (0, (3, 2)))):
            g = p[(p.band == band) & (p.method == method)].sort_values("size")
            if g.empty:
                continue
            c = color if method == "WCM" else tint(color, 0.45)
            ax.errorbar(g["size"], g.pass_pct, yerr=[g.pass_pct - g.ci_lo, g.ci_hi - g.pass_pct], color=c,
                        linestyle=style, marker="o", markersize=3, markerfacecolor=c if method == "WCM" else SURFACE,
                        elinewidth=0.7, capsize=0, label="%s, %s" % (lab, method), zorder=3 if method == "WCM" else 2)
            prod = g[(g.method == "WCM") & np.isclose(g.lam, 1e-3)]
            if len(prod):
                ax.plot(prod["size"], prod.pass_pct, marker="o", markersize=7, markerfacecolor="none",
                        markeredgecolor=INK, markeredgewidth=0.9, linestyle="none", zorder=4)
    ax.set(xscale="log", ylim=(0, 100), xlabel="circuit size (median upstream latents)", ylabel="targets that pass (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=2.2)
    ax.text(0.01, 0.97, "ring: production penalty", transform=ax.transAxes, fontsize=6.5, color=INK_MUTED, va="top")
    ax.tick_params(colors=INK_MUTED)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", default="turingllm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="further OUTs whose sweep/ holds extra arms (v1 only)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else K.DEFAULT_OUT[args.family]
    out = out if out.is_absolute() else K.ROOT / out
    extra = [Path(e) if Path(e).is_absolute() else K.ROOT / e for e in args.extra]
    tag = args.tag or "%s-v2" % args.family
    M = K.merge_module(args.family, out)
    ev = load(M, out, extra)
    p = points(ev, M)
    dest = K.HERE / "out" / tag
    dest.mkdir(parents=True, exist_ok=True)
    p.round(4).to_csv(dest / "depth_frontier.csv", index=False)
    from analysis.style import save_figure  # noqa: E402  (src is on sys.path after figure())
    fig = figure(p, M)
    save_figure(fig, dest / "depth-frontier.png")
    print(p.round(1).to_string(index=False))
    print("\nsmallest WCM median size reaching 50%% pass, by band: %s" % {
        b: (None if np.isnan(v) else int(v)) for b, v in first_reach(p).items()})
    print("targets per band (any arm): %s" % ev.groupby("depth", observed=True).seed.nunique().to_dict())
    print("wrote %s" % dest)


if __name__ == "__main__":
    main()
