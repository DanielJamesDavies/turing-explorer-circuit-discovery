"""Table 3: the same circuits re-scored with every coefficient set to 1 (alpha = 1) vs their fitted coefficients.

Reads   out/alpha1/alpha1.shard*.jsonl (062 rescore_alpha1.py rows, held strong) and main/eval (held strong), per model
Writes  paper/tables/run2026_alpha1.tex, out/run2026_alpha1.csv (per model and per depth band)

Pairing as rescore_alpha1.py REPORT: inner join on seed, a pair is dropped when either row is vacuous (here
merge.vacuous, the headline's v2 rule, instead of the older vacuous_tk flag); pass = merge.passes on each row.
CI on each pass rate: bootstrap over targets; on the difference: paired bootstrap.

  python -X utf8 experiments/109-pod-run-analysis/t3_alpha1.py
"""
import pandas as pd

import lib109 as L

COLS = L.HEAD + [L.NEC, L.IND]


def paired(model):
    ev = L.main_eval(model)
    a1 = L.frame(L.read_rows(L.find(model, "out", "alpha1"), "alpha1"), keys=("seed", "held"))
    if ev.empty or a1.empty:
        return None
    a1 = L.score(L.add_site(a1[a1.get("held", "strong") == "strong"], model))
    fit = ev[ev.held == "strong"]
    m = fit.set_index("seed")[COLS + ["vac", "pass", "n", "band", "kind"]].join(
        a1.set_index("seed")[COLS + ["vac", "pass"]], rsuffix="_a1", how="inner")
    n_join = len(m)
    m = m[~(m.vac | m.vac_a1)]
    return m, n_join, len(a1)


def row(model, by, group, m):
    pf, pa = L.pass_cell(m["pass"]), L.pass_cell(m["pass_a1"])
    lo, hi = L.boot_paired_diff(m["pass_a1"].astype(float).values, m["pass"].astype(float).values)
    return dict(model=L.LABEL[model], by=by, group=group, paired=len(m), n_median=m.n.median(),
                pass_fit_k=pf["k"], pass_fit_pct=pf["pct"], pass_fit_lo=pf["lo"], pass_fit_hi=pf["hi"],
                pass_a1_k=pa["k"], pass_a1_pct=pa["pct"], pass_a1_lo=pa["lo"], pass_a1_hi=pa["hi"],
                diff_pp=pa["pct"] - pf["pct"], diff_lo=100 * lo, diff_hi=100 * hi,
                lost=int((m["pass"] & ~m["pass_a1"]).sum()), gained=int((~m["pass"] & m["pass_a1"]).sum()),
                **{"%s_fit" % k: m[c].median() for k, c in zip("ZAC", L.HEAD)},
                **{"%s_a1" % k: m[c + "_a1"].median() for k, c in zip("ZAC", L.HEAD)},
                nec_fit=m[L.NEC].median(), nec_a1=m[L.NEC + "_a1"].median(),
                induce_fit=m[L.IND].median(), induce_a1=m[L.IND + "_a1"].median(), _pf=pf, _pa=pa)


def main():
    out, tex = [], []
    for model in L.MODELS:
        p = paired(model)
        if p is None:
            L.not_available("%s alpha = 1 rows" % model)
            continue
        m, n_join, n_a1 = p
        rs = [row(model, "all", "all", m)] + [row(model, "band", b, m[m.band == b]) for b, _, _ in L.bands_of(model)]
        out += rs
        r = rs[0]
        print("%-11s alpha1 rows %d, joined %d, paired non-vacuous %d | pass fitted %s -> alpha=1 %s | diff %.1f pp "
              "[%.1f, %.1f] | lost %d gained %d | Z %.2f->%.2f A %.2f->%.2f C %.2f->%.2f" % (
                  r["model"], n_a1, n_join, r["paired"], L.fmt_pass(r["_pf"]), L.fmt_pass(r["_pa"]), r["diff_pp"],
                  r["diff_lo"], r["diff_hi"], r["lost"], r["gained"], r["Z_fit"], r["Z_a1"], r["A_fit"], r["A_a1"],
                  r["C_fit"], r["C_a1"]))
        if tex:
            tex.append([r"\midrule"])
        for x in rs:
            print("    %-6s n %5d  fitted %5.1f  alpha=1 %5.1f" % (x["group"], x["paired"], x["pass_fit_pct"],
                                                                  x["pass_a1_pct"]))
            tex.append([r["model"] if x is rs[0] else "", "all layers" if x["group"] == "all" else
                        "layers " + x["group"][1:], L.tex_int(x["paired"]), L.fmt_pass(x["_pf"], True),
                        L.fmt_pass(x["_pa"], True), "%+.1f {[}%+.1f, %+.1f{]}" % (x["diff_pp"], x["diff_lo"],
                                                                                 x["diff_hi"]),
                        "%s / %s / %s" % tuple(L.tex_num(x["%s_fit" % k]) for k in "ZAC"),
                        "%s / %s / %s" % tuple(L.tex_num(x["%s_a1" % k]) for k in "ZAC")])
    if not out:
        return
    L.write_table("alpha1", tex, ["Model", "Group", "Circuits", r"Pass, fitted $\alpha$ (\%)",
                                  r"Pass, $\alpha=1$ (\%)", "Difference (pp)", r"Z / A / C, fitted",
                                  r"Z / A / C, $\alpha=1$"], "llrlllll",
                  note="held-out strongest; same members, coefficients set to 1; 95% bootstrap CIs over targets "
                       "(difference: paired)",
                  csv_df=pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in out]))


if __name__ == "__main__":
    main()
