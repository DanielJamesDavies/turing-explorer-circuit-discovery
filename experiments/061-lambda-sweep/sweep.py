"""PILOT FOR DAN-76 / DAN-77: WCM vs unweighted circuit masking across a sparsity-price sweep, at matched size.

Both methods train on the protocol-v1 contexts of 059 (32 strongest + 16 mid-band, stratified split, close contrast
contexts, rank-keep 3e-3, train-only ablation values) with ablation-term weights gamma = 0.25 (see GAMMA below),
on the 16 pilot targets; the only difference is whether the per-node scaling coefficients are fitted:

  W_<lam>   WCM, coefficients fitted          lam in 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3
  U_<lam>   unweighted circuit masking, a = 1  lam in 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 (lower: it needs more nodes)

Every fit is scored by the eval pass on the held-out strongest contexts (one evaluation reference for all arms).
Output: faithfulness under Z / A / C against node count, one curve per method (medians over targets, IQR), the
first draft of the Caples-style comparison figure, and the natural-scale cost: how many nodes unweighted masking
needs to reach the faithfulness WCM reaches (the abstract's 10^4-10^5 claim rests on old July panels).

  PYTHONPATH=src python experiments/061-lambda-sweep/sweep.py      (STAGE fit | eval | summary; all by default)
  env: STAGE (all)  ARMS (comma list; default all)  ONLY (comma targets, smoke)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "059-context-pool")); sys.path.insert(0, str(EXP / "049-circuit-graph"))
STAGE = os.environ.get("STAGE", "all")
# gamma 0.25 (not 1): the 059 Bw / B2w refit (2026-09-25) found equal weights worse than 0.25 under the final
# protocol (Bw worse than B on 13/15 targets, B2w worse than B2 on 9/15), so the pilot uses the better setting.
# W at lambda 1e-3 reproduces 059 arm B (same contexts, same config): a built-in sanity check.
GAMMA = float(os.environ.get("GAMMA", 0.25))
W_LAMS = (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)
U_LAMS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
ALL_ARMS = [("W_%g" % l, True, l) for l in W_LAMS] + [("U_%g" % l, False, l) for l in U_LAMS]
WANT = [a for a in os.environ.get("ARMS", "").split(",") if a]
ARMS = [a for a in ALL_ARMS if not WANT or a[0] in WANT]
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]
RES = HERE / "results"
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]


def log(s):
    print(s, flush=True)


def targets(P):
    s = [x for x in (P.E055 / "seeds.txt").read_text().split() if x]
    return [x for x in s if x in ONLY] if ONLY else s


def main():
    RES.mkdir(parents=True, exist_ok=True)
    if STAGE == "summary":
        summary(); return
    os.environ.setdefault("TAG", "sweep_unused")
    os.environ.setdefault("CTR_SOURCE", "close")
    import amp_eval_pass_v2 as V
    import protocol_harness as H
    G = V.setup()
    ctx = torch.load(H.P.CTX, weights_only=False)
    tg = targets(H.P)
    if STAGE in ("all", "fit"):
        for name, weighted, lam in ARMS:
            log("fit %s" % name)
            H.fit_arm(G, ctx, tg, HERE / ("data_%s" % name), train_arm="B", gamma=GAMMA, lam=lam, free_amp=weighted, log=log)
    if STAGE in ("all", "eval"):
        pool = H.load_pool()
        for name, weighted, lam in ARMS:
            p = HERE / ("data_%s" % name) / "discovered_circuits.shard0.pt"
            if not p.exists():
                continue
            cs = torch.load(p, weights_only=False, map_location="cpu")
            byk = {V.key_of(c): c for c in cs.values()}
            log("eval %s (%d circuits)" % (name, len(byk)))
            H.eval_arm(G, V, ctx, tg, byk, RES / ("eval_%s.jsonl" % name), helds=("strong",), pool=pool,
                       tag=dict(arm=name, weighted=weighted, lam=lam), log=log)
    if STAGE == "all":
        summary()


def summary():
    import pandas as pd
    rows = []
    for name, weighted, lam in ALL_ARMS:
        p = RES / ("eval_%s.jsonl" % name)
        if p.exists():
            rows += [r for r in map(json.loads, open(p)) if "error" not in r and "skip" not in r]
    if not rows:
        log("no rows"); return
    df = pd.DataFrame(rows)
    df["worst_dev"] = (df[HEAD] - 1).abs().max(axis=1)
    df["in_band"] = ((df[HEAD] >= 0.8) & (df[HEAD] <= 1.25)).all(axis=1)
    g = df.groupby(["weighted", "lam"])
    tab = g.agg(n_targets=("seed", "nunique"), nodes=("n", "median"), free0=("free0_tk", "median"),
                freeM=("freeM_topk_tk", "median"), freeN=("freeN_topk_tk", "median"), worst_dev=("worst_dev", "median"),
                in_band=("in_band", "sum"), necessity=("phi_sup_blind_tk", "median"),
                induce=("phi_cf_alpha_blind_tk", "median")).round(3)
    lines = ["# 061 lambda sweep: WCM vs unweighted circuit masking (held-out strongest, activation read)\n",
             "gamma = %g, protocol-v1 contexts (059 arm B training set), 16 pilot targets (15 with a mid-band pool).\n" % GAMMA,
             tab.to_string()]
    # matched-size read-off: for each weighted arm's median size, the unweighted faithfulness at that size (log-interp)
    u = tab.loc[False].sort_values("nodes") if False in tab.index.get_level_values(0) else None
    w = tab.loc[True].sort_values("nodes") if True in tab.index.get_level_values(0) else None
    if u is not None and w is not None and len(u) > 1:
        lines.append("\n## Matched size (unweighted interpolated in log node count)\n")
        for lam, r in w.iterrows():
            x = np.log10(r.nodes)
            vals = {f: float(np.interp(x, np.log10(u.nodes.values), u[f].values)) for f in ("free0", "freeM", "freeN")}
            lines.append("- WCM lam %g: %d nodes, free0/M/N %.2f/%.2f/%.2f | unweighted at the same size: %.2f/%.2f/%.2f%s"
                         % (lam, r.nodes, r.free0, r.freeM, r.freeN, vals["free0"], vals["freeM"], vals["freeN"],
                            "" if u.nodes.min() <= r.nodes <= u.nodes.max() else " (extrapolated: outside the unweighted range)"))
    txt = "\n".join(lines)
    (RES / "summary.md").write_text(txt, encoding="utf-8")
    log(txt)
    figure(df)


def figure(df):
    sys.path.insert(0, str(EXP.parent / "src"))
    from analysis.style import BLUE, CATEGORICAL, GRID, INK_MUTED, configure_matplotlib, save_figure, style_suptitle, tint
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (f, lab) in zip(axes, (("free0_tk", "zero ablation"), ("freeM_topk_tk", "mean ablation, activating"),
                                   ("freeN_topk_tk", "mean ablation, contrast"))):
        ax.axhspan(0.8, 1.25, color=tint(INK_MUTED, 0.85), zorder=0)
        ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls=(0, (3, 3)), zorder=1)
        for weighted, col, name in ((True, BLUE, "WCM (fitted coefficients)"), (False, CATEGORICAL[1], "unweighted (α = 1)")):
            s = df[df.weighted == weighted].groupby("lam")
            med = s.agg(n=("n", "median"), y=(f, "median"), lo=(f, lambda v: v.quantile(.25)), hi=(f, lambda v: v.quantile(.75))).sort_values("n")
            if med.empty:
                continue
            ax.fill_between(med.n, med.lo, med.hi, color=tint(col, 0.75), lw=0, zorder=2)
            ax.plot(med.n, med.y, marker="o", color=col, label=name, zorder=3)
        ax.set_xscale("log"); ax.set_title(lab); ax.set_xlabel("nodes (median over targets)")
        ax.set_ylim(-0.1, 1.6)
    axes[0].set_ylabel("faithfulness (held-out, activation read)")
    axes[0].legend(loc="lower right")
    style_suptitle(fig, "Faithfulness against circuit size: WCM vs unweighted circuit masking (16 pilot targets)")
    save_figure(fig, RES / "faithfulness_vs_size.png")


if __name__ == "__main__":
    main()
