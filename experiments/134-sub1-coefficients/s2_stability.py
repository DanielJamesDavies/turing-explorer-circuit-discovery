"""134 step 2 (CPU): are sub-1 coefficients reproducible? TuringLLM, 126 targets with a second production fit (T5 fit2,
same contexts and config; 118 stability.py's data). For every fit-1 member, binned by fit-1 alpha and gate m: is it a
member of fit 2 at all, and if so is its fit-2 alpha also below 1? Compared against members near 1 and above 1.

Writes out/s2_stability.md.
  PYTHONPATH=src .venv/bin/python -B experiments/134-sub1-coefficients/s2_stability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCH = ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives"
MAIN = ARCH / "pub-turingllm-T3-main" / "root/turing-tl/experiments/094-tl-v2-records/out_v2/main/circuits"
SC = ARCH / "pub-turingllm-T5-stable" / "root/turing-tl/experiments/094-tl-v2-records/out_v2/stable_core/circuits"
BINS = [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.05), (1.05, 1.25), (1.25, 2), (2, 100)]


def members(path):
    c = torch.load(path, weights_only=False, map_location="cpu")
    out = {}
    if c is None:
        return out
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out[(int(f.layer), str(f.kind), int(f.index))] = (float(md.get("amplitude", 1.0)),
                                                          float(md.get("attribution_score", np.nan)))
    return out


def main():
    rows = []
    for p2 in sorted((SC / "fit2").glob("*.pt")):
        a1, a2 = members(MAIN / p2.name), members(p2)
        if not a1 or not a2:
            continue
        for k, (al, m) in a1.items():
            rows.append(dict(seed=p2.stem, a1=al, m1=m, in2=k in a2, a2=a2[k][0] if k in a2 else np.nan))
    df = pd.DataFrame(rows)
    lines = ["# 134 step 2: reproducibility of sub-1 coefficients (TuringLLM, %d targets with a second fit, %d fit-1 "
             "members)" % (df.seed.nunique(), len(df)), "",
             "| fit-1 alpha | members | median gate m | kept in fit 2 | of kept: fit-2 alpha < 1 | of kept: < 0.95 | of kept: median fit-2 alpha |",
             "|---|---|---|---|---|---|---|"]
    for lo, hi in BINS:
        s = df[(df.a1 >= lo) & (df.a1 < hi)]
        k = s[s.in2]
        lines.append("| [%g, %g) | %d | %.2f | %.1f%% | %.1f%% | %.1f%% | %.3f |" % (
            lo, hi, len(s), s.m1.median(), 100 * s.in2.mean(), 100 * (k.a2 < 1).mean(), 100 * (k.a2 < 0.95).mean(),
            k.a2.median()))
    # gate-matched view: low-gate members only, so the comparison is not just "low gate = unstable"
    lines += ["", "Low-gate members only (m < 0.6):", "",
              "| fit-1 alpha | members | kept in fit 2 | of kept: fit-2 alpha < 1 | of kept: median fit-2 alpha |",
              "|---|---|---|---|---|"]
    lo_g = df[df.m1 < 0.6]
    for lo, hi in BINS:
        s = lo_g[(lo_g.a1 >= lo) & (lo_g.a1 < hi)]
        if not len(s):
            continue
        k = s[s.in2]
        lines.append("| [%g, %g) | %d | %.1f%% | %.1f%% | %.3f |" % (
            lo, hi, len(s), 100 * s.in2.mean(), 100 * (k.a2 < 1).mean(), k.a2.median()))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "s2_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
