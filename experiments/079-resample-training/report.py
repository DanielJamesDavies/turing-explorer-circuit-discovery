"""079 report (CPU): out/rows.jsonl -> out/report.md.

Per arm: production pass rule (Z, A, C in [0.8, 1.5], necessity >= 0.9; production scorer), the resample pass
(Z from the production scorer, resample A and resample C with HELD-OUT contrast donors, necessity), resample R, sizes,
074's all-position r under C; paired against W0 (same GPU, same config, no new term) and prod.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT79 = Path(os.environ.get("OUT79", str(HERE / "out")))
BANDS = ["L0-4", "L5-7", "L8-11"]


def ib(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and 0.8 <= v <= 1.5


def load():
    rows = [json.loads(l) for l in open(OUT79 / "rows.jsonl")]
    rows = [r for r in rows if "error" not in r and "skip" not in r]
    out = []
    for r in rows:
        p, s = r["prod_scorer"], r["resample"]
        if p.get("vacuous_tk"):
            continue
        nec = (p.get("phi_sup_blind_tk") or 0) >= 0.9
        out.append(dict(seed=r["seed"], arm=r["arm"], band=r["band"], kind=r["kind"], n=r.get("n"),
                        Z=p.get("free0_tk"), A=p.get("freeM_topk_tk"), C=p.get("freeN_topk_tk"),
                        nec=p.get("phi_sup_blind_tk"), induce=p.get("phi_cf_alpha_blind_tk"),
                        alpha_med=p.get("alpha_median"),
                        rsA=s.get("rs_A_wcm"), rsC=s.get("rs_C_wcm"), rsR=s.get("rs_R_wcm"),
                        r_C=s.get("ap_C_wcm_r"), fve_C=s.get("ap_C_wcm_fve"), ctr_fire_Z=s.get("ap_Z_wcm_ctr_fire_rate"),
                        fp_mass_C=s.get("ap_C_wcm_fp_mass"), ctr_fire_C=s.get("ap_C_wcm_ctr_fire_rate"),
                        prod_pass=ib(p.get("free0_tk")) and ib(p.get("freeM_topk_tk")) and ib(p.get("freeN_topk_tk"))
                        and nec,
                        rs_pass=ib(p.get("free0_tk")) and ib(s.get("rs_A_wcm")) and ib(s.get("rs_C_wcm")) and nec,
                        t_fit=r.get("t_fit")))
    return pd.DataFrame(out)


def main():
    d = load()
    L = ["# 079 — WCM with a donor-resample training term (pilot)\n"]
    w = L.append
    arms = [a for a in ["prod", "W0"] + sorted(set(d.arm) - {"prod", "W0"}) if a in set(d.arm)]
    full = d.groupby("seed").arm.nunique()
    keep = full[full == len(arms)].index
    d = d[d.seed.isin(keep)]
    w("Targets complete for every arm (%s): %d. Resample C uses the 16 held-out contrast contexts as donors; R uses "
      "073's evaluation pool (disjoint from the training donors).\n" % (", ".join(arms), len(keep)))
    for b in ["all"] + BANDS:
        x = d if b == "all" else d[d.band == b]
        if not len(x):
            continue
        w("\n**%s**\n" % b)
        w("| arm | n targets | size (median) | prod pass | resample pass | Z | A | C | nec | rs A | rs C (held) | rs R |"
          " r all-pos (C) | fve (C) | silent mass (C) | ctr fire (C) | ctr fire (Z) | alpha median | fit s |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in arms:
            y = x[x.arm == a]
            med = lambda c: "%.2f" % y[c].dropna().median() if y[c].notna().any() else "-"  # noqa: E731
            w("| %s | %d | %.0f | %.0f%% | %.0f%% | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
              % (a, len(y), y.n.median(), 100 * y.prod_pass.mean(), 100 * y.rs_pass.mean(), med("Z"), med("A"),
                 med("C"), med("nec"), med("rsA"), med("rsC"), med("rsR"), med("r_C"), med("fve_C"), med("fp_mass_C"),
                 med("ctr_fire_C"), med("ctr_fire_Z"), med("alpha_med"), med("t_fit")))
    # paired vs W0
    if "W0" in arms:
        base = d[d.arm == "W0"].set_index("seed")
        w("\n**Paired against W0** (same targets; median of per-target differences, and pass-rate change)\n")
        w("| arm | d size (ratio) | d prod pass | d resample pass | d rs C | d rs R | d Z | d C | d r all-pos (C) |"
          " d silent mass (C) |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for a in arms:
            if a in ("W0",):
                continue
            y = d[d.arm == a].set_index("seed").reindex(base.index)
            w("| %s | %.2fx | %+.0f pp | %+.0f pp | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f |" % (
                a, float((y.n / base.n).median()), 100 * (y.prod_pass.mean() - base.prod_pass.mean()),
                100 * (y.rs_pass.mean() - base.rs_pass.mean()), float((y.rsC - base.rsC).median()),
                float((y.rsR - base.rsR).median()), float((y.Z - base.Z).median()), float((y.C - base.C).median()),
                float((y.r_C - base.r_C).median()), float((y.fp_mass_C - base.fp_mass_C).median())))
        w("\nA full pass for the 079 question = production pass AND resample pass AND all-position fidelity kept: "
          "read the last two columns next to d resample pass.\n")
    txt = "\n".join(L)
    (OUT79 / "report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
