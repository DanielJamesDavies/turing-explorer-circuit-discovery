"""111 a5: stable core by depth band (and site kind), per model.

Same population and rules as 109 t4_robustness.stable (re-implemented per target here, since t4 only returns
aggregates): stable_core rows (stage fit2 / core), last per (seed, stage); fit 1 = the production held-out strongest
row; targets whose fit-1 row is vacuous are dropped; a missing or vacuous core (or a skipped, e.g. empty, core) counts
as a fail. The pooled numbers are checked against t4's own aggregate.

Writes paper/tables/run2026_stable_core_bands.tex, out/run2026_stable_core_bands.csv

  python -X utf8 experiments/111-failure-anatomy/a5_stable_bands.py
"""
import importlib.util

import pandas as pd

import lib111 as X

L = X.L


def per_target(model):
    rows = L.read_rows(L.find(model, "out", "stable_core"), "stable_core")
    if not rows:
        return None
    d = pd.DataFrame([r for r in rows if r.get("stage") != "error" and "error" not in r])
    d = d.drop_duplicates(["seed", "stage"], keep="last")
    ev = L.main_eval(model)
    f1 = ev[ev.held == "strong"].set_index("seed")
    f2 = L.score(d[d.stage == "fit2"].drop(columns=["layer", "kind"], errors="ignore").assign(held="strong")).set_index("seed")
    core_all = d[d.stage == "core"].set_index("seed")
    core = L.score(core_all[core_all.get("skip").isna()].reset_index() if "skip" in core_all else core_all.reset_index()
                   ).set_index("seed")
    keep = [s for s in f2.index if s in f1.index and not f1.loc[s, "vac"]]
    t = pd.DataFrame(index=keep)
    t["band"], t["kind"] = f1.loc[keep, "band"], f1.loc[keep, "kind"]
    t["fit1"] = f1.loc[keep, "pass"].astype(bool)
    t["fit2"] = (f2.loc[keep, "pass"] & ~f2.loc[keep, "vac"]).astype(bool)
    ck = core.reindex(keep)
    t["core"] = ck["pass"].fillna(False).astype(bool) & ~ck["vac"].fillna(False).astype(bool)
    t["jaccard"] = f2.loc[keep, "jaccard"].astype(float)
    t["n1"], t["n_core"] = f2.loc[keep, "n1"].astype(float), f2.loc[keep, "n_core"].astype(float)
    return t


def main():
    spec = importlib.util.spec_from_file_location("t4", L.HERE / "t4_robustness.py")
    t4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t4)
    out, tex = [], []
    for model in L.MODELS:
        lab = L.LABEL[model]
        t = per_target(model)
        if t is None:
            L.not_available("%s stable core" % model)
            continue
        ref = t4.stable(model)
        assert abs(100 * t.core.mean() - ref["core_pct"]) < 1e-9 and len(t) == ref["scored"], "differs from 109 t4"
        groups = [(b, t[t.band == b]) for b, _, _ in L.bands_of(model)] + [("all", t)] + \
                 [(L.KIND_LABEL[k], t[t.kind == k]) for k in L.KIND_ORDER if (t.kind == k).any()]
        if tex:
            tex.append([r"\midrule"])
        for j, (name, g) in enumerate(groups):
            both = g.fit1 & g.fit2
            c1, c2, cc = X.rate(g.fit1), X.rate(g.fit2), X.rate(g.core)
            r = dict(model=lab, group=name, n=len(g), jaccard_median=g.jaccard.median(),
                     core_over_fit1_median=(g.n_core / g.n1).median(), fit1_pct=c1["pct"], fit2_pct=c2["pct"],
                     core_pct=cc["pct"], core_lo=cc["lo"], core_hi=cc["hi"], both_pass=int(both.sum()),
                     core_pass_given_both=int((g.core & both).sum()))
            out.append(r)
            tex.append([lab if j == 0 else "", L.tex_escape(name), L.tex_int(len(g)), L.tex_num(r["jaccard_median"]),
                        "%.1f" % c1["pct"], "%.1f" % c2["pct"], L.fmt_pass(cc, True),
                        "%d / %d" % (r["core_pass_given_both"], r["both_pass"])])
            print("%-11s %-15s n %3d | Jaccard %.2f | core/fit1 %.2f | pass fit1 %.1f fit2 %.1f core %s | core passes %d of "
                  "%d both-pass" % (lab, name, len(g), r["jaccard_median"], r["core_over_fit1_median"], c1["pct"],
                                    c2["pct"], X.fmt(cc), r["core_pass_given_both"], r["both_pass"]))
    L.write_table("stable_core_bands", tex, ["Model", "Group", "Targets", "Jaccard (median)", r"Pass fit 1 (\%)",
                                             r"Pass fit 2 (\%)", r"Pass core (\%) [95\% CI]", "Core passes / both pass"],
                  "llrrrrll", note="109 t4_robustness.stable rules, per depth band and site kind",
                  csv_df=pd.DataFrame(out))


if __name__ == "__main__":
    main()
