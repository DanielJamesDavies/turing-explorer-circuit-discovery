"""DAN-163: intervals for the main-body numbers that had none in any 109 / 111 / 112 table. CPU only.

Every interval is a 95% percentile bootstrap over targets, B = 4,000, seed 20261108 (lib109.boot_ci /
boot_paired_diff), computed from existing per-target rows; zero-pass nulls get the exact one-sided 95% upper bound
1 - 0.05^(1/n) (as 112 null_detail.exact_upper).

Reads   109 out/archives (T4 sweep + T3 main, via t6_tl_sweep.load), 109 out/run2026_tl_matched_targets.csv,
        109 out/run2026_unweighted_pythia_targets.csv, 112 out/nulls_detail_meta.json.
Writes  experiments/122-headline-intervals/out/intervals_122.csv and .json

  python -X utf8 experiments/122-headline-intervals/intervals_122.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E109 = ROOT / "experiments" / "109-pod-run-analysis"
sys.path.insert(0, str(E109))

import lib109 as L  # noqa: E402
import t6_tl_sweep as T6  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
rows = []


def add(name, value, lo, hi, n, source, note=""):
    rows.append(dict(number=name, value=value, lo=lo, hi=hi, n=n, source=source, note=note))


# 1. Random nulls: 0 passes of n, exact one-sided 95% upper bound.
meta = json.loads((ROOT / "experiments/112-null-detail/out/nulls_detail_meta.json").read_text())["metas"]
ns = {m: meta[m]["scored"] for m in L.MODELS}
for m, n in list(ns.items()) + [("pooled", sum(ns.values()))]:
    add("random null upper bound, %s" % (L.LABEL.get(m, m)), 0.0, 0.0, 100 * (1 - 0.05 ** (1 / n)), n,
        "112 out/nulls_detail_meta.json (scored)", "0 of n pass every null arm; one-sided exact 95% bound")

# 2. Per-target unweighted / WCM size ratio (smallest passing circuit of each, targets both pass): median + CI.
mt = pd.read_csv(E109 / "out/run2026_tl_matched_targets.csv")
for b in ["L0-4", "L5-7", "L8-11", "all"]:
    g = mt if b == "all" else mt[mt.band == b]
    g = g.dropna(subset=["wcm_pass", "unw_pass"])
    r = (g.unw_pass / g.wcm_pass).values
    lo, hi = L.boot_ci(r, stat=np.median)
    add("TL sweep size ratio U/W (smallest passing, median), %s" % b, float(np.median(r)), lo, hi, len(r),
        "109 out/run2026_tl_matched_targets.csv")

# 3. Pythia unweighted / WCM size ratio at lambda 1e-3, median + CI.
py = pd.read_csv(E109 / "out/run2026_unweighted_pythia_targets.csv")
py = py[(~py.vac.astype(bool)) & (~py.vac_U.astype(bool))]
r = py.ratio.dropna().values
lo, hi = L.boot_ci(r, stat=np.median)
add("Pythia size ratio U/W at 1e-3 (median)", float(np.median(r)), lo, hi, len(r),
    "109 out/run2026_unweighted_pythia_targets.csv")

# 4. Paired pass-rate differences on the TuringLLM sweep (173 scored targets).
sw, _ = T6.load("turingllm")


def arm_pass(arm):
    g = sw[sw.arm == arm].drop_duplicates("seed", keep="last").set_index("seed")
    return g


def paired(a, b, label, by_band=True):
    A, B = arm_pass(a), arm_pass(b)
    common = A.index.intersection(B.index)
    ok = common[(~A.loc[common, "vac"]).values & (~B.loc[common, "vac"]).values]
    groups = [("all", ok)]
    if by_band:
        for bd in ["L0-4", "L5-7", "L8-11"]:
            groups.append((bd, ok[(A.loc[ok, "band"] == bd).values]))
    for bd, idx in groups:
        pa = A.loc[idx, "pass"].astype(float).values
        pb = B.loc[idx, "pass"].astype(float).values
        lo, hi = L.boot_paired_diff(pa, pb)
        add("%s, %s (pp)" % (label, bd), 100 * (pa.mean() - pb.mean()), 100 * lo, 100 * hi, len(idx),
            "109 archives T4 sweep + T3 main (t6_tl_sweep.load)",
            "%s %.1f%% vs %s %.1f%%" % (a, 100 * pa.mean(), b, 100 * pb.mean()))


paired("W_0.0005", "W_0.001", "plateau edge: W 5e-4 minus W 1e-3")
paired("W_5e-05", "W_0.0005", "plateau span: W 5e-5 minus W 5e-4", by_band=False)
paired("W_0.001", "U_1e-05", "W 1e-3 minus best unweighted U 1e-5")

df = pd.DataFrame(rows)
df.to_csv(OUT / "intervals_122.csv", index=False)
(OUT / "intervals_122.json").write_text(json.dumps(rows, indent=1))
pd.set_option("display.width", 250, "display.max_colwidth", 80)
print(df.to_string(index=False, float_format=lambda x: "%.2f" % x))
