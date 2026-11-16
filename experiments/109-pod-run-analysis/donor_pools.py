"""Donor-background test split by donor pool (2026-11-15, appendix review E / Daniel "do 1 and 2").

Pool A = the 48 strongest TRAINING activating contexts (src/eval/donor_test.py: donor_sets["A"] = pos_tokens[:n_train]);
pool C = the 16 held-out contrast contexts; pool T (code name R) = random corpus text. The reported test requires
production pass AND recovery under A and C in BAND. This script recomputes, from the same stored rows as
t4_robustness.donor (main circuits only, same dedup / vacuous / pass logic), the pass rate with:
  A and C (the reported test; must reproduce run2026_resample), A only, C only (the held-out pool alone),
  C and T (both never-training pools), and T only; plus TuringLLM by depth band and site kind.

  python -X utf8 experiments/109-pod-run-analysis/donor_pools.py
Writes out/run2026_donor_pools.csv and out/run2026_donor_pools_tl_groups.csv.
"""
import numpy as np
import pandas as pd

import lib109 as L
import t4_robustness as T4

COMBOS = {"A and C (reported)": ("A", "C"), "A only": ("A",), "C only": ("C",), "C and T": ("C", "R"), "T only": ("R",)}


def frame(model):
    dirs = T4.donor_dirs(model)["main"]
    rows = [r for r in L.read_rows(dirs, "rows") if "error" not in r and not r.get("skip")]
    last = {}
    for r in rows:
        last[(r["seed"], r["arm"])] = r
    rs = [r for k, r in last.items() if k[1] == "wcm"] or list(last.values())
    arms = {k[1] for k in last}
    prod = pd.DataFrame([r.get("prod", {}) for r in rs])
    prod["seed"] = [r["seed"] for r in rs]
    ok = ~L.vacuous(prod).values
    pp = np.zeros(len(rs), dtype=bool)
    pp[ok] = L.passes(prod[ok]).values
    df = pd.DataFrame({"seed": [r["seed"] for r in rs], "prod_pass": pp, "ok": ok})
    for ds in ("A", "C", "R"):
        df["in_" + ds] = T4.in_band(pd.Series([(r.get("recovery") or {}).get("fit", {}).get(ds) for r in rs],
                                              dtype=float)).values
    df["layer"] = [int(str(s).split(".")[0]) for s in df.seed]
    df["kind"] = [str(s).split(".")[1] for s in df.seed]
    return df[df.ok].copy(), arms


def rates(df):
    out = {"n": len(df)}
    for lab, pools in COMBOS.items():
        p = df.prod_pass.copy()
        for ds in pools:
            p &= df["in_" + ds]
        out[lab] = 100 * p.mean()
    return out


def main():
    allr, grp = [], []
    for m in L.MODELS:
        df, arms = frame(m)
        if df.empty:
            print(m, "no donor rows")
            continue
        r = rates(df)
        r.update(model=L.LABEL[m], prod_pass=100 * df.prod_pass.mean(), arms=",".join(sorted(arms)))
        allr.append(r)
        if m == "turingllm":
            df["band"] = [L.band_of(m, l) for l in df.layer]
            for key in ("band", "kind"):
                for g, d in df.groupby(key):
                    x = rates(d)
                    x.update(group=key, value=g)
                    grp.append(x)
    t = pd.DataFrame(allr).set_index("model")
    t.to_csv(L.HERE / "out" / "run2026_donor_pools.csv")
    print(t.round(1).to_string())
    if grp:
        g = pd.DataFrame(grp)
        g.to_csv(L.HERE / "out" / "run2026_donor_pools_tl_groups.csv", index=False)
        print(g.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
