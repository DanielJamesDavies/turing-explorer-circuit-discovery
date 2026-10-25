"""073 / 074 report (CPU). Joins out/rows.jsonl with the 15k run's production eval rows (held-out strongest).

  PYTHONPATH=src python experiments/073-strict-eval/report.py      -> out/report.md (also printed)
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
E62 = HERE.parent / "062-h100-protocol-v1"
OUT73 = Path(os.environ.get("OUT73", str(HERE / "out")))
BAND = (0.8, 1.5)
BANDS = ["L0-4", "L5-7", "L8-11"]


def inband(v):
    return v.notna() & (v >= BAND[0]) & (v <= BAND[1])


def load():
    rows = [json.loads(l) for p in [OUT73 / "rows.jsonl"] + sorted(OUT73.glob("rows.shard*.jsonl")) if p.exists()
            for l in open(p)]                                   # unsharded run, or strict_eval.py under SHARD=i/k
    d = pd.DataFrame([r for r in rows if "error" not in r]).drop_duplicates("seed", keep="last").set_index("seed")
    ev = {}
    run = Path(os.environ.get("OUT", str(E62 / "out_full" / "out")))   # the production run (strict_eval's OUT)
    for f in glob.glob(str(run / "main" / "eval.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "error" not in r and r.get("held") == "strong":
                ev[r["seed"]] = r
    p = pd.DataFrame([ev[k] for k in d.index if k in ev]).set_index("seed")
    keep = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "vacuous_tk", "eM_topk_tk", "eN_topk_tk",
            "a_pos_tk", "n"]
    d = d.join(p[keep], how="left")
    d = d[~d["vacuous_tk"].fillna(False).astype(bool)]
    d["prod_pass"] = inband(d.free0_tk) & inband(d.freeM_topk_tk) & inband(d.freeN_topk_tk) & (d.phi_sup_blind_tk >= 0.9)
    nec = d.phi_sup_blind_tk >= 0.9
    for v in ("wcm", "a1", "rand"):
        if "rs_A_%s" % v not in d:
            continue
        # Z for this variant from the all-position run's anchor means (same anchors as production)
        e = d["ap_Z_empty_anchor_mean"]
        d["Z_%s" % v] = (d["ap_Z_%s_anchor_mean" % v] - e) / (d["a_pos"] - e)
        d["rs_pass_%s" % v] = inband(d["Z_%s" % v]) & inband(d["rs_A_%s" % v]) & inband(d["rs_C_%s" % v]) & nec
        d["rsR_pass_%s" % v] = inband(d["Z_%s" % v]) & inband(d["rs_A_%s" % v]) & inband(d["rs_R_%s" % v]) & nec
    d["leak_A_mean"] = d.eM_topk_tk / d.a_pos_tk
    d["leak_C_mean"] = d.eN_topk_tk / d.a_pos_tk
    return d


def med(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return "%.2f" % s.median() if len(s) else "-"


def pct(s):
    s = s.dropna()
    return "%.1f%%" % (100 * s.mean()) if len(s) else "-"


def main():
    d = load()
    L = []
    w = L.append
    w("# 073 / 074 report\n")
    w("Targets scored: %d (non-vacuous); with a random-null circuit: %d. Production pass rate on these: %s.\n"
      % (len(d), int(d.has_rand.sum()), pct(d.prod_pass)))
    w("Sanity: production Z (free0_tk) vs this run's Z for the same circuit, median |diff| %.3f.\n"
      % float((d.free0_tk - d.Z_wcm).abs().median()))

    w("\n## 073 — resample ablation\n")
    w("Faithfulness (x - e)/(a_pos - e) at held-out anchors, everything outside the circuit resampled from real donor "
      "contexts (A: activating train, C: contrast train, R: random corpus). Medians; pass = Z, A, C (or R) in "
      "[0.8, 1.5] and production necessity >= 0.9.\n")
    w("| band | n | prod pass | resample pass (A,C) | resample pass (A,R) | A wcm | C wcm | R wcm | A a1 | C a1 | R a1 |"
      " rand pass (A,C) | A rand | C rand |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in BANDS + ["all"]:
        x = d if b == "all" else d[d.band == b]
        xr = x[x.has_rand]
        w("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            b, len(x), pct(x.prod_pass), pct(x.rs_pass_wcm), pct(x.rsR_pass_wcm), med(x.rs_A_wcm), med(x.rs_C_wcm),
            med(x.rs_R_wcm), med(x.rs_A_a1), med(x.rs_C_a1), med(x.rs_R_a1),
            pct(xr.rs_pass_rand) if "rs_pass_rand" in xr else "-", med(xr.get("rs_A_rand")),
            med(xr.get("rs_C_rand"))))
    w("\nBy kind (all bands):\n")
    w("| kind | n | prod pass | resample pass (A,C) | A wcm | C wcm | R wcm |")
    w("|---|---|---|---|---|---|---|")
    for k in ("attn", "mlp", "resid"):
        x = d[d.kind == k]
        w("| %s | %d | %s | %s | %s | %s | %s |" % (k, len(x), pct(x.prod_pass), pct(x.rs_pass_wcm), med(x.rs_A_wcm),
                                                    med(x.rs_C_wcm), med(x.rs_R_wcm)))
    w("\nAgreement with production: of production passes, %s also pass resample (A,C); of production fails, %s pass "
      "resample.\n" % (pct(d[d.prod_pass].rs_pass_wcm), pct(d[~d.prod_pass].rs_pass_wcm)))
    w("\nLeak (empty circuit / a_pos): the ablation baseline's own target activation, medians by band.\n")
    w("| band | A mean (prod) | A resample | C mean (prod) | C resample | R resample | R donor natural / a_pos |")
    w("|---|---|---|---|---|---|---|")
    for b in BANDS + ["all"]:
        x = d if b == "all" else d[d.band == b]
        w("| %s | %s | %s | %s | %s | %s | %s |" % (b, med(x.leak_A_mean), med(x.rs_A_leak), med(x.leak_C_mean),
                                                    med(x.rs_C_leak), med(x.rs_R_leak), med(x.rs_R_donor_nat / x.a_pos)))
    # draw stability
    dd = d.rs_C_wcm_draws.dropna()
    if len(dd):
        diffs = [abs(a - b) for a, b in (v[:2] for v in dd if len(v) >= 2 and None not in v[:2])]
        w("\nDraw stability (C, wcm): median |draw1 - draw2| = %.3f over %d targets.\n" % (np.median(diffs), len(diffs)))

    w("\n## 074 — all-position simulation\n")
    w("Circuit-only target read vs the natural read at every position of 16 held-out activating + 16 held-out "
      "contrast contexts. r: Pearson over all positions; r_act: activating contexts only; fve: fraction of variance "
      "explained; prec / rec: firing (> 0) agreement; ctr fire: share of contrast-context positions where the circuit "
      "fires (natural: ~0); fp mass: circuit activation where the target is silent / natural activation where it fires."
      " Medians.\n")
    for fill in ("Z", "C"):
        w("\n**Fill %s** (%s)\n" % (fill, "zero" if fill == "Z" else "contrast mean, sparsity-preserving"))
        w("| band | variant | r | r_act | fve | prec | rec | ctr fire | fp mass |")
        w("|---|---|---|---|---|---|---|---|---|")
        for b in BANDS + ["all"]:
            x = d if b == "all" else d[d.band == b]
            for v in ("wcm", "a1", "rand", "empty"):
                if v == "rand":
                    x_ = x[x.has_rand]
                else:
                    x_ = x
                c = lambda m: x_.get("ap_%s_%s_%s" % (fill, v, m))  # noqa: E731
                if c("r") is None:
                    continue
                w("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (b, v, med(c("r")), med(c("r_act")), med(c("fve")),
                                                                  med(c("prec")), med(c("rec")),
                                                                  med(c("ctr_fire_rate")), med(c("fp_mass"))))
    w("\nPassing vs failing production circuits (fill C, wcm): r %s vs %s; fve %s vs %s; ctr fire %s vs %s.\n" % (
        med(d[d.prod_pass].ap_C_wcm_r), med(d[~d.prod_pass].ap_C_wcm_r), med(d[d.prod_pass].ap_C_wcm_fve),
        med(d[~d.prod_pass].ap_C_wcm_fve), med(d[d.prod_pass].ap_C_wcm_ctr_fire_rate),
        med(d[~d.prod_pass].ap_C_wcm_ctr_fire_rate)))
    w("Natural contrast fire rate (sanity, should be ~0): %s.\n" % med(d.get("ap_C_wcm_ctr_nat_fire_rate")))
    txt = "\n".join(L)
    (OUT73 / "report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
