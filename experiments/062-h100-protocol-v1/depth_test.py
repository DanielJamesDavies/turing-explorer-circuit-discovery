"""DEPTH TEST (DAN-131, Daniel's hypothesis 2026-10-02): do deep circuits fail because one sparsity penalty is used for
every layer, when deeper targets need more latents?

Data: 32 deep targets (layers 8-11, 8 per layer, balanced over kinds; targets_depth32.txt) from the 190-target sweep.
  - existing sweep arms on the same targets (out_full/out/sweep_rows.csv): WCM at lambda 4e-3 ... 2.5e-4, 400 steps
  - new arms (out_depth/sweep/eval.*.jsonl): WCM at lambda 1e-4 and 5e-5 (400 steps), and lambda 1e-3 for 800 steps
Prints, per arm: median circuit size, pass rate (DAN-8 rule), median Z / A / C and necessity. Then the depth-scaled
comparison: the shallow and mid bands' pass rate at matched circuit size, from the full sweep, against the deep curve.

  python experiments/062-h100-protocol-v1/depth_test.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
BAND, NEC_MIN = (0.8, 1.5), 0.9


def passes(df):
    inb = ((df[HEAD] >= BAND[0]) & (df[HEAD] <= BAND[1])).all(axis=1)
    return inb & (df.phi_sup_blind_tk >= NEC_MIN)


def vacuous(df):
    return df.get("vacuous_tk", pd.Series(False, index=df.index)).fillna(False).astype(bool)


def main():
    targets = (HERE / "targets_depth32.txt").read_text().split()
    sw = pd.read_csv(HERE / "out_full" / "out" / "sweep_rows.csv")
    old = sw[sw.weighted & sw.seed.isin(targets)].copy()
    old["arm"] = old.lam.map(lambda l: "W_%g" % l)
    old["steps"] = 400
    rows = []
    for f in sorted((HERE / "out_depth" / "sweep").glob("eval.*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r and "skip" not in r:
                rows.append(r)
    new = pd.DataFrame(rows).drop_duplicates(["seed", "arm"], keep="last") if rows else pd.DataFrame()
    if not new.empty:
        new = new[~vacuous(new)].copy()
        new["passes"] = passes(new)
        new["layer"] = new.seed.str.split(".").str[0].astype(int)
        new["kind"] = new.seed.str.split(".").str[1]
    cols = ["seed", "arm", "lam", "steps", "n", "passes", "layer", "kind"] + HEAD + ["phi_sup_blind_tk"]
    d = pd.concat([old[cols], new[cols]] if not new.empty else [old[cols]], ignore_index=True)

    print("== 32 deep targets (layers 8-11): per arm")
    g = d.groupby(["arm", "lam", "steps"]).agg(
        targets=("seed", "nunique"), n_med=("n", "median"), pass_pct=("passes", "mean"),
        Z=("free0_tk", "median"), A=("freeM_topk_tk", "median"), C=("freeN_topk_tk", "median"),
        nec=("phi_sup_blind_tk", "median")).reset_index().sort_values(["steps", "lam"], ascending=[True, False])
    g["pass_pct"] = (100 * g.pass_pct).round(0)
    show = g.round(2)
    show["lam"] = g.lam.map(lambda l: "%g" % l)                  # rounding would print every lambda as 0.0
    print(show.to_string(index=False))

    print("\n== pass rate by site kind, per arm")
    print((100 * d.pivot_table(index="arm", columns="kind", values="passes", aggfunc="mean")).round(0).to_string())

    # paired: the same targets, 800 vs 400 steps at lambda 1e-3
    a, b = d[d.arm == "W_0.001"].set_index("seed"), d[d.arm == "W_0.001_s800"].set_index("seed")
    both = a.index.intersection(b.index)
    if len(both):
        print("\n== 800 vs 400 steps at lambda 1e-3, paired on %d targets: size %.0f -> %.0f | pass %.0f%% -> %.0f%% | "
              "C %.2f -> %.2f" % (len(both), a.loc[both].n.median(), b.loc[both].n.median(),
                                   100 * a.loc[both].passes.mean(), 100 * b.loc[both].passes.mean(),
                                   a.loc[both].freeN_topk_tk.median(), b.loc[both].freeN_topk_tk.median()))

    # matched size: shallower bands from the full sweep (WCM, 400 steps), every lambda
    print("\n== matched-size comparison (WCM, 400 steps; shallow / mid bands from the 190-target sweep)")
    for band in ("L0-4", "L5-7"):
        r = sw[sw.weighted & (sw.depth == band)].groupby("lam").agg(n=("n", "median"), p=("passes", "mean"))
        print("  %-6s " % band + " | ".join("%5.0f latents %3.0f%%" % (x.n, 100 * x.p) for x in r.sort_values("n").itertuples()))
    deep = g[g.steps == 400].sort_values("n_med")
    print("  L8-11* " + " | ".join("%5.0f latents %3.0f%%" % (x.n_med, x.pass_pct) for x in deep.itertuples())
          + "   (* these 32 targets)")


if __name__ == "__main__":
    main()
