"""118 coefficient stability across refits (CPU only). The T5 stable-core archive holds, for 126 TuringLLM targets, a
second production fit of the same target (fit2, same contexts and config) and a refit of the coefficients on the
shared core alone (core). For members present in both fits: agreement of alpha between fit 1 (T3 production) and
fit 2, and between fit 1 and the core refit.

  python -I -X utf8 experiments/118-coefficient-diagnostics/stability.py
Writes out/stability_members.csv, out/stability_targets.csv and prints the summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
ARCH = ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives"
MAIN = ARCH / "pub-turingllm-T3-main" / "root/turing-tl/experiments/094-tl-v2-records/out_v2/main/circuits"
SC = ARCH / "pub-turingllm-T5-stable" / "root/turing-tl/experiments/094-tl-v2-records/out_v2/stable_core/circuits"


def alphas(path):
    c = torch.load(path, weights_only=False, map_location="cpu")
    if c is None:
        return None
    out = {}
    for n in c.nodes.values():
        if n.metadata.get("role") == "seed":
            continue
        f = n.metadata["feature_id"]
        out[(int(f.layer), str(f.kind), int(f.index))] = float(n.metadata.get("amplitude", 1.0))
    return out


def main():
    rows, trows = [], []
    for p2 in sorted((SC / "fit2").glob("*.pt")):
        key = p2.stem
        a1, a2 = alphas(MAIN / p2.name), alphas(p2)
        pc = SC / "core" / p2.name
        ac = alphas(pc) if pc.exists() else None
        if not a1 or not a2:
            continue
        shared = sorted(set(a1) & set(a2))
        for m in shared:
            rows.append(dict(seed=key, layer=m[0], kind=m[1], latent=m[2], a1=a1[m], a2=a2[m],
                             ac=(ac or {}).get(m)))
        x = np.array([a1[m] for m in shared]); y = np.array([a2[m] for m in shared])
        only1 = np.array([a1[m] for m in set(a1) - set(a2)])
        trows.append(dict(seed=key, n1=len(a1), n2=len(a2), n_shared=len(shared),
                          jaccard=len(shared) / len(set(a1) | set(a2)),
                          spearman=spearmanr(x, y).correlation if len(shared) > 2 else np.nan,
                          med_abs_log=float(np.median(np.abs(np.log(x / y)))) if len(shared) else np.nan,
                          med_a1_shared=float(np.median(x)) if len(x) else np.nan,
                          med_a1_only1=float(np.median(only1)) if len(only1) else np.nan))
    m = pd.DataFrame(rows); t = pd.DataFrame(trows)
    (HERE / "out").mkdir(exist_ok=True)
    m.to_csv(HERE / "out" / "stability_members.csv", index=False)
    t.to_csv(HERE / "out" / "stability_targets.csv", index=False)
    lr = np.log(m.a1 / m.a2)
    print("targets %d, shared members %d" % (len(t), len(m)))
    print("membership Jaccard median %.3f (IQR %.3f-%.3f)" % (t.jaccard.median(), t.jaccard.quantile(.25),
                                                              t.jaccard.quantile(.75)))
    print("pooled Spearman(a1, a2) %.3f; Pearson log %.3f" % (spearmanr(m.a1, m.a2).correlation,
                                                             np.corrcoef(np.log(m.a1), np.log(m.a2))[0, 1]))
    print("per-target Spearman median %.3f (IQR %.3f-%.3f)" % (t.spearman.median(), t.spearman.quantile(.25),
                                                               t.spearman.quantile(.75)))
    print("|log(a1/a2)| median %.3f, p90 %.3f; share within x1.1 %.3f, within x1.25 %.3f" % (
        lr.abs().median(), lr.abs().quantile(.9), (lr.abs() < np.log(1.1)).mean(), (lr.abs() < np.log(1.25)).mean()))
    for lo, hi in ((0, 0.8), (0.8, 1.25), (1.25, 2), (2, 100)):
        s = m[(m.a1 >= lo) & (m.a1 < hi)]
        print("  a1 in [%g, %g): n %d, median a2 %.3f, share a2 on the same side of 1 %.3f" % (
            lo, hi, len(s), s.a2.median(), ((s.a1 > 1) == (s.a2 > 1)).mean()))
    print("alpha of shared members (fit 1) median %.3f vs fit-1-only members %.3f (median over targets)" % (
        t.med_a1_shared.median(), t.med_a1_only1.median()))
    mc = m.dropna(subset=["ac"])
    if len(mc):
        print("core refit: members %d; Spearman(a1, a_core) %.3f; median a_core/a1 %.3f" % (
            len(mc), spearmanr(mc.a1, mc.ac).correlation, float(np.median(mc.ac / mc.a1))))


if __name__ == "__main__":
    main()
