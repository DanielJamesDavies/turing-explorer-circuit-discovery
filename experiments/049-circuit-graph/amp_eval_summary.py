"""SUMMARISE the amplitude-aware eval pass (amp_eval.shard*.jsonl).

Adds the BOUNDED scores (overshoot counts as error, like cf_bounded):
  F0_amp_b = 1 - |ampF0 - a_pos| / |a_pos - e0|      (and F0_a1_b, FMd_amp_b, F0_perm_b)
Reports distributions overall and by seed layer / kind, the amplitude
effect (F0_amp - F0_a1), the permutation null, the corrected bare cf vs
the legacy stored cf, and vacuous cases (a_pos ~ e0).

  python experiments/049-circuit-graph/amp_eval_summary.py [DIR with amp_eval.shard*.jsonl]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results_full"
rows = []
for p in sorted(glob.glob(str(DIR / "amp_eval.shard*.jsonl"))):
    for ln in open(p):
        r = json.loads(ln)
        if "skip" in r or "error" in r:
            rows.append(r); continue
        den = abs(r["a_pos_ho"] - r["e0_ho"])
        for k, raw in (("F0_amp_b", "ampF0_ho"), ("F0_a1_b", "F0a1_ho"), ("F0_perm_b", "ampF0_perm")):
            r[k] = 1 - abs(r[raw] - r["a_pos_ho"]) / den if den > 1e-9 else None
        denm = abs(r["a_pos_ho"] - r["eMd_ho"])
        r["FMd_amp_b"] = 1 - abs(r["ampFMd_ho"] - r["a_pos_ho"]) / denm if denm > 1e-9 else None
        rows.append(r)
df = pd.DataFrame(rows)
n_skip = int(df["skip"].notna().sum()) if "skip" in df else 0
n_err = int(df["error"].notna().sum()) if "error" in df else 0
ok = df[df["a_pos_ho"].notna()].copy()
print("rows %d | scored %d | skipped %d | errors %d | vacuous (a_pos ~ e0) %d" % (len(df), len(ok), n_skip, n_err, int(ok["vacuous"].sum())))
ok = ok[~ok["vacuous"]]
print("non-vacuous scored: %d" % len(ok))


def q(s):
    s = s.dropna()
    return "median %.3f | p10 %.3f | p90 %.3f | mean %.3f | >=0.8: %.2f | >=0.5: %.2f" % (s.median(), s.quantile(0.1), s.quantile(0.9), s.mean(), (s >= 0.8).mean(), (s >= 0.5).mean())


print("\n=== HELD-OUT SCORES (fraction of the seed recovered; bounded = overshoot penalised) ===")
for c in ("F0_amp", "F0_amp_b", "F0_a1", "F0_a1_b", "FMd_amp", "FMd_amp_b", "cf_amp", "cf_bare_anch", "cf_bounded", "sup_anch", "F0_perm_b", "F0_amp_tr"):
    if c in ok:
        print("  %-13s " % c + q(ok[c]))
ok["amp_effect"] = ok["F0_amp_b"] - ok["F0_a1_b"]
print("\namplitude effect (F0_amp_b - F0_a1_b): " + q(ok["amp_effect"]).replace(">=0.8", ">=+0.8").replace(">=0.5", ">=+0.5"))
print("  circuits where amplitudes add >= 0.2: %.2f | where a=1 already >= 0.8: %.2f" % ((ok["amp_effect"] >= 0.2).mean(), (ok["F0_a1_b"] >= 0.8).mean()))
print("permutation null (gains shuffled among members), bounded: " + q(ok["F0_perm_b"]))
print("  real - permuted: median %+.3f | real > permuted in %.2f of circuits" % ((ok["F0_amp_b"] - ok["F0_perm_b"]).median(), ((ok["F0_amp_b"] - ok["F0_perm_b"]) > 0).mean()))
print("train vs held-out F0_amp: train median %.3f | held-out median %.3f | gap median %+.3f" % (ok["F0_amp_tr"].median(), ok["F0_amp"].median(), (ok["F0_amp_tr"] - ok["F0_amp"]).median()))
if "cf_bare_legacy" in ok:
    print("bare cf: legacy (stored) median %.3f vs anchor-fixed median %.3f | Spearman %.2f" % (ok["cf_bare_legacy"].median(), ok["cf_bare_anch"].median(), ok[["cf_bare_legacy", "cf_bare_anch"]].corr(method="spearman").iloc[0, 1]))

print("\nby seed layer (medians): F0_amp_b | F0_a1_b | FMd_amp_b | cf_amp | cf_bare_anch | sup | n")
for l, d in ok.groupby("layer"):
    print("  L%-2d %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %d" % (l, d.F0_amp_b.median(), d.F0_a1_b.median(), d.FMd_amp_b.median(), d.cf_amp.median(), d.cf_bare_anch.median(), d.sup_anch.median(), len(d)))
print("by seed kind:")
for k, d in ok.groupby("kind"):
    print("  %-5s %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %d" % (k, d.F0_amp_b.median(), d.F0_a1_b.median(), d.FMd_amp_b.median(), d.cf_amp.median(), d.cf_bare_anch.median(), d.sup_anch.median(), len(d)))
print("\nseconds per circuit: median %.1f | total %.1f GPU-h" % (ok["secs"].median(), ok["secs"].sum() / 3600))
ok.to_csv(DIR / "amp_eval_all.csv", index=False)
print("->", DIR / "amp_eval_all.csv")
