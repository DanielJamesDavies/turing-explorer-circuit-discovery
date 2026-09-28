"""PASS-RULE CALIBRATION (DAN-8): how candidate pass rules score the protocol-v1 circuits.

Stage 1 is the calibration set; the rule is fixed from this table BEFORE the full run is read, and the confirmatory
headline is the full run's non-stage-1 targets. Every rule reads the held-out strongest contexts; the activation read
(post-Top-K, *_tk) is primary, never mixed with the pre-activation read (*_pre) inside one rule (DAN-66).

  OUT=experiments/062-h100-protocol-v1/out_ckpt/out PYTHONPATH=src python experiments/062-h100-protocol-v1/pass_rule.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUT", str(HERE / "out")))
K = 128
FAITH = {"tk": ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"], "pre": ["free0_pre", "freeM_topk_pre", "freeN_topk_pre"]}
NEC, IND = "phi_sup_blind_tk", "phi_cf_alpha_blind_tk"


def load():
    ev = pd.DataFrame([json.loads(l) for f in glob.glob(str(OUT / "main" / "eval.*.jsonl")) for l in open(f)])
    ev = ev[ev.get("error").isna()] if "error" in ev else ev
    ev = ev[ev.held == "strong"].drop_duplicates("seed", keep="last").set_index("seed")
    tg = pd.read_csv(OUT / "targets.csv", index_col="seed")
    d = ev.join(tg[["rank_clean", "amp_any"]], how="left")
    d["layer"] = d.index.str.split(".").str[0].astype(int)
    d["kind"] = d.index.str.split(".").str[1]
    d["depth"] = pd.cut(d.layer, [-1, 3, 7, 11], labels=["L0-3", "L4-7", "L8-11"])
    d["near"] = d.rank_clean >= K / 2
    return d


def band(d, cols, lo, hi):
    x = d[cols].astype(float)
    return ((x >= lo) & (x <= hi)).all(axis=1)          # NaN (missing / vacuous) compares False: counts as a fail


def main():
    d = load()
    n = len(d)
    vac = d.get("vacuous_tk", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    print("circuits (held-out strongest): %d | near-threshold (clean rank >= %d): %d | vacuous denominator: %d"
          % (n, K // 2, int(d.near.sum()), int(vac.sum())))
    print("necessity (%s): median %.3f, >= 0.9 in %.1f%%, >= 0.8 in %.1f%%" % (
        NEC, d[NEC].median(), 100 * (d[NEC] >= .9).mean(), 100 * (d[NEC] >= .8).mean()))
    print("sufficiency to induce (%s): median %.3f, in [0.5, 2] %.1f%%" % (
        IND, d[IND].median(), 100 * d[IND].between(.5, 2).mean()))

    core = ~d.near & ~vac                                  # the activation-primary headline population
    print("\n== candidate rules on the headline population (not near-threshold, not vacuous): %d circuits" % core.sum())
    rows = []
    for lo, hi in ((0.8, 1.25), (0.8, 1.5), (0.8, 2.0), (0.75, 1 / 0.75), (2 / 3, 1.5), (0.5, 2.0)):
        f = band(d, FAITH["tk"], lo, hi)
        for nec in (None, 0.8, 0.9):
            p = f & (d[NEC] >= nec) if nec else f
            for ind in (False, True):
                q = p & d[IND].between(.5, 2) if ind else p
                rows.append(dict(band="[%.2f, %.2f]" % (lo, hi), necessity=nec or "-", induce="[0.5,2]" if ind else "-",
                                 pass_pct=round(100 * q[core].mean(), 1)))
    t = pd.DataFrame(rows)
    print(t.pivot_table(index=["band", "necessity"], columns="induce", values="pass_pct").to_string())

    print("\n== one-ablation relaxations at [0.8, 1.25] + necessity >= 0.9 (headline population)")
    for name, cols in (("Z only", FAITH["tk"][:1]), ("Z and A", FAITH["tk"][:2]), ("Z and C", FAITH["tk"][::2]),
                       ("Z, A, C", FAITH["tk"])):
        p = band(d, cols, .8, 1.25) & (d[NEC] >= .9)
        print("  %-8s %5.1f%%" % (name, 100 * p[core].mean()))

    # the ADOPTED rule (DAN-8, 2026-09-27): asymmetric band, undershoot fails, overshoot to 1.5 accepted
    base = band(d, FAITH["tk"], .8, 1.5) & (d[NEC] >= .9)
    print("\n== ADOPTED rule (Z, A, C in [0.8, 1.5] AND necessity >= 0.9): all non-vacuous %.1f%%" % (100 * base[~vac].mean()))
    print("   by site kind and depth, headline population (near-threshold excluded here for the breakdown only):")
    print(d[core].assign(p=base[core]).groupby("kind").p.mean().mul(100).round(1).to_string())
    print(d[core].assign(p=base[core]).groupby("depth", observed=True).p.mean().mul(100).round(1).to_string())
    pre = band(d, FAITH["pre"], .8, 1.5) & (d[NEC] >= .9)
    print("\n== near-threshold stratum (%d), same rule on the PRE-ACTIVATION read: %.1f%% pass "
          "(activation read: %.1f%%)" % (d.near.sum(), 100 * pre[d.near].mean(), 100 * base[d.near].mean()))
    print("   by kind: %s" % d[d.near].kind.value_counts().to_dict())
    print("\n== graded companion: worst-of-3 |faith - 1| on the headline population: median %.3f, p25 %.3f, p75 %.3f"
          % tuple(np.nanpercentile((d.loc[core, FAITH["tk"]].astype(float) - 1).abs().max(axis=1), [50, 25, 75])))


if __name__ == "__main__":
    main()
