"""Table 5: Pythia-70m WCM (production, lambda = 1e-3) vs unweighted circuit masking (U_0.001: every alpha = 1,
same lambda) on the same targets.

Reads   out_pythia-70m/main/eval (held strong; the W circuit is the production one), out_pythia-70m/sweep/eval
        (arm U_0.001, held strong), 082 targets_pythia-70m_unweighted.txt (the 128-target list)
Writes  paper/tables/run2026_unweighted_pythia.tex, out/run2026_unweighted_pythia.csv (overall + by band),
        out/run2026_unweighted_pythia_targets.csv (per target), paper/figures/run2026_unweighted_pythia.pdf/.png

Paired on targets with both a W and a U row; a target is dropped when either row is vacuous (merge.vacuous).
Size ratio = n_U / n_W per target (median, IQR). CIs: bootstrap over targets; difference: paired bootstrap.

  python -X utf8 experiments/109-pod-run-analysis/t5_unweighted_pythia.py
"""
import numpy as np
import pandas as pd

import lib109 as L

MODEL = "pythia-70m"
ARM = "U_0.001"


def load():
    ev = L.main_eval(MODEL)
    sw = L.frame(L.read_rows(L.find(MODEL, "out", "sweep"), "eval"), keys=("seed", "arm"))
    if ev.empty or sw.empty or ARM not in set(sw.get("arm", [])):
        return None
    u = L.score(L.add_site(sw[sw.arm == ARM], MODEL)).set_index("seed")
    w = ev[ev.held == "strong"].set_index("seed")
    tg = L.targets_list(MODEL, "unweighted") or sorted(u.index)
    keep = [s for s in tg if s in u.index and s in w.index]
    cols = L.HEAD + [L.NEC, L.IND, "n", "vac", "pass"]
    m = w.loc[keep, cols + ["band", "kind"]].join(u.loc[keep, cols], rsuffix="_U")
    return m, len(tg), len(keep)


def row(group, m):
    pw, pu = L.pass_cell(m["pass"]), L.pass_cell(m["pass_U"])
    lo, hi = L.boot_paired_diff(m["pass"].astype(float).values, m["pass_U"].astype(float).values)
    ratio = m.n_U / m.n
    return dict(group=group, targets=len(m), pass_W_k=pw["k"], pass_W_pct=pw["pct"], pass_W_lo=pw["lo"],
                pass_W_hi=pw["hi"], pass_U_k=pu["k"], pass_U_pct=pu["pct"], pass_U_lo=pu["lo"], pass_U_hi=pu["hi"],
                diff_pp=pw["pct"] - pu["pct"], diff_lo=100 * lo, diff_hi=100 * hi,
                n_W_median=m.n.median(), n_U_median=m.n_U.median(), ratio_median=ratio.median(),
                ratio_p25=ratio.quantile(.25), ratio_p75=ratio.quantile(.75), U_larger_pct=100 * (ratio > 1).mean(),
                both_pass=int((m["pass"] & m["pass_U"]).sum()),
                ratio_median_both_pass=ratio[m["pass"] & m["pass_U"]].median(),
                **{"%s_W" % k: m[c].median() for k, c in zip("ZAC", L.HEAD)},
                **{"%s_U" % k: m[c + "_U"].median() for k, c in zip("ZAC", L.HEAD)}, _pw=pw, _pu=pu)


def figure(m):
    plt, st = L.figure_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = (max(1, min(m.n.min(), m.n_U.min()) * 0.7), max(m.n.max(), m.n_U.max()) * 1.4)
    ax.plot(lim, lim, color=st.INK_MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    for (pw, pu), (lab, col, mk) in {(True, True): ("both pass", st.BLUE, "o"),
                                     (True, False): ("WCM only", st.CATEGORICAL[2], "o"),
                                     (False, True): ("unweighted only", st.CATEGORICAL[1], "o"),
                                     (False, False): ("neither", st.INK_MUTED, "x")}.items():
        x = m[(m["pass"] == pw) & (m["pass_U"] == pu)]
        if len(x):
            ax.scatter(x.n, x.n_U, s=18, color=col, marker=mk, label="%s (%d)" % (lab, len(x)), zorder=2,
                       linewidths=1 if mk == "x" else 0)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="WCM circuit size (latents)",
           ylabel="unweighted circuit size (latents)")
    ax.grid(True, axis="both")
    ax.set_title("Pythia-70m, same targets, $\\lambda = 10^{-3}$", fontsize=11.5)
    st.styled_legend(ax, loc="upper left", fontsize=8.5)
    st.save_figure(fig, L.FIGURES / ("%sunweighted_pythia.png" % L.PREFIX))


def main():
    p = load()
    if p is None:
        L.not_available("Pythia unweighted arm %s" % ARM)
        return
    m, n_list, n_both = p
    n_pairs = len(m)
    m = m[~(m.vac | m.vac_U)]
    rows = [row("all", m)] + [row(b, m[m.band == b]) for b, _, _ in L.bands_of(MODEL)]
    print("Pythia W vs U: list %d, both rows %d, non-vacuous pairs %d (of %d)" % (n_list, n_both, len(m), n_pairs))
    tex = []
    for r in rows:
        print("  %-5s n %3d | W %d/%d = %s n %.0f | U %d/%d = %s n %.0f | W-U %+.1f pp [%+.1f, %+.1f] | size ratio U/W "
              "%.2f (%.2f-%.2f), U larger %.0f%%" % (
                  r["group"], r["targets"], r["pass_W_k"], r["targets"], L.fmt_pass(r["_pw"]), r["n_W_median"],
                  r["pass_U_k"], r["targets"], L.fmt_pass(r["_pu"]), r["n_U_median"], r["diff_pp"], r["diff_lo"],
                  r["diff_hi"], r["ratio_median"], r["ratio_p25"], r["ratio_p75"], r["U_larger_pct"]))
        tex.append(["all layers" if r["group"] == "all" else "layers " + r["group"][1:], L.tex_int(r["targets"]),
                    L.fmt_pass(r["_pw"], True), L.fmt_pass(r["_pu"], True),
                    "%+.1f {[}%+.1f, %+.1f{]}" % (r["diff_pp"], r["diff_lo"], r["diff_hi"]),
                    "%s / %s" % (L.tex_int(r["n_W_median"]), L.tex_int(r["n_U_median"])),
                    "%.2f (%.2f--%.2f)" % (r["ratio_median"], r["ratio_p25"], r["ratio_p75"])])
    L.write_table("unweighted_pythia", tex, ["Depth", "Targets", r"Pass, WCM (\%)", r"Pass, unweighted (\%)",
                                             "WCM $-$ unweighted (pp)", "Size WCM / unweighted", "Size ratio U/W (IQR)"],
                  "lrlllll", note="Pythia-70m, lambda = 1e-3 for both; held-out strongest; 95% bootstrap CIs",
                  csv_df=pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]))
    m.assign(ratio=m.n_U / m.n).to_csv(L.OUT / ("%sunweighted_pythia_targets.csv" % L.PREFIX))
    figure(m)


if __name__ == "__main__":
    main()
