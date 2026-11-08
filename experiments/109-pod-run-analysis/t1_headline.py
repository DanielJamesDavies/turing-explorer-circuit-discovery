"""Table 1: headline per model (TuringLLM / Pythia-70m / GPT-2 small) and the by-depth-band / by-site-kind breakdown.

Reads   <archives>/.../main/eval.shard*.jsonl (held strong + mid) and the frozen main target list, per model.
Writes  paper/tables/run2026_headline.tex, run2026_headline_groups.tex
        out/run2026_headline.csv, out/run2026_headline_groups.csv (also band x kind), out/run2026_headline.json
        paper/figures/run2026_pass_by_band.pdf/.png

Population: held-out strongest rows (primary), error rows dropped, last row per (seed, held); vacuous denominators
(merge.vacuous) excluded and counted; pass = merge.passes. Mid-band: the same on the mid rows (own vacuous set).
CIs: 95% percentile bootstrap over targets (BOOT_B resamples, fixed seed). Size: median (IQR) over the non-vacuous
held-out-strongest circuits (paper_main.py's population). Induce: median phi_cf_alpha_blind_tk, same population.

  python -X utf8 experiments/109-pod-run-analysis/t1_headline.py
"""
import pandas as pd

import lib109 as L


def summarise(model, ev):
    s, m = ev[ev.held == "strong"], ev[ev.held == "mid"]
    so, mo = s[~s.vac], m[~m.vac]
    tg = L.targets_list(model) or []
    ps, pm = L.pass_cell(so["pass"]), L.pass_cell(mo["pass"])
    return dict(model=L.LABEL[model], targets=len(tg), fitted=int(s.seed.nunique()), vacuous=int(s.vac.sum()),
                scored=len(so), pass_k=ps["k"], pass_pct=ps["pct"], pass_lo=ps["lo"], pass_hi=ps["hi"],
                mid_rows=len(m), mid_vacuous=int(m.vac.sum()), mid_scored=len(mo), mid_pass_k=pm["k"],
                mid_pass_pct=pm["pct"], mid_pass_lo=pm["lo"], mid_pass_hi=pm["hi"],
                n_median=so.n.median(), n_p25=so.n.quantile(.25), n_p75=so.n.quantile(.75),
                n_median_passing=so[so["pass"]].n.median(), n_median_all_fitted=s.n.median(),
                share_1e2_1e3=100 * ((so.n >= 100) & (so.n <= 1000)).mean(),
                Z_median=so[L.HEAD[0]].median(), A_median=so[L.HEAD[1]].median(), C_median=so[L.HEAD[2]].median(),
                nec_median=so[L.NEC].median(), induce_median=so[L.IND].median(),
                induce_median_mid=mo[L.IND].median()), ps, pm


def group_rows(model, ev):
    s, m = ev[(ev.held == "strong") & ~ev.vac], ev[(ev.held == "mid") & ~ev.vac]
    out = []
    groups = [("band", b) for b, _, _ in L.bands_of(model)] + \
             [("kind", k) for k in L.KIND_ORDER if k in set(s.kind)] + \
             [("band x kind", (b, k)) for b, _, _ in L.bands_of(model) for k in L.KIND_ORDER if k in set(s.kind)]
    for by, g in groups:
        if by == "band x kind":
            sel = lambda d: d[(d.band == g[0]) & (d.kind == g[1])]  # noqa: E731
            name = "%s %s" % g
        else:
            sel = lambda d: d[d[by] == g]  # noqa: E731
            name = g
        gs, gm = sel(s), sel(m)
        ps, pm = L.pass_cell(gs["pass"]), L.pass_cell(gm["pass"])
        out.append(dict(model=L.LABEL[model], by=by, group=name, scored=len(gs), pass_k=ps["k"], pass_pct=ps["pct"],
                        pass_lo=ps["lo"], pass_hi=ps["hi"], mid_scored=len(gm), mid_pass_k=pm["k"],
                        mid_pass_pct=pm["pct"], mid_pass_lo=pm["lo"], mid_pass_hi=pm["hi"],
                        n_median=gs.n.median(), n_p25=gs.n.quantile(.25), n_p75=gs.n.quantile(.75),
                        induce_median=gs[L.IND].median(), _ps=ps, _pm=pm))
    return out


def figure(groups):
    plt, st = L.figure_style()
    models = [m for m in L.MODELS if any(g["model"] == L.LABEL[m] for g in groups)]
    if not models:
        return
    # sized for the 5.5 in text width at 100%: text 7.5-8.5 pt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.titlepad": 4.0, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.8,
                         "grid.linewidth": 0.5})
    fig, axes = plt.subplots(1, len(models), figsize=(5.5, 2.3), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        rows = [g for g in groups if g["model"] == L.LABEL[model] and g["by"] == "band"]
        x = range(len(rows))
        width, offs = st.grouped_bar_geometry(2)
        for (key, lab, col), off in zip((("", "held-out strongest", st.BLUE), ("mid_", "held-out mid-band",
                                                                               st.CATEGORICAL[1])), offs):
            v = [r[key + "pass_pct"] for r in rows]
            lo = [r[key + "pass_pct"] - r[key + "pass_lo"] for r in rows]
            hi = [r[key + "pass_hi"] - r[key + "pass_pct"] for r in rows]
            ax.bar([i + off for i in x], v, width, color=col, label=lab, zorder=2)
            ax.errorbar([i + off for i in x], v, yerr=[lo, hi], fmt="none", ecolor=st.INK_SECONDARY, lw=0.7,
                        capsize=1.5, zorder=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(["L%s" % r["group"][1:].replace("-", "–") for r in rows])
        ax.set_ylim(0, 100)
        ax.set_title(L.LABEL[model])
        ax.set_xlabel("target layers")
    axes[0][0].set_ylabel("circuits that pass (%)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
    L.save_with_legend(fig, L.FIGURES / ("%spass_by_band.png" % L.PREFIX), top=0.88)


def main():
    heads, groups = [], []
    tex_h, tex_g = [], []
    for model in L.MODELS:
        ev = L.main_eval(model)
        if ev.empty:
            L.not_available("%s main eval rows" % model)
            continue
        h, ps, pm = summarise(model, ev)
        heads.append(h)
        # Daniel 2026-11-08: one "Scored" column (non-vacuous held-out strongest rows); targets / fitted / vacuous
        # stay in the CSV and in tab:run2026-exclusions.
        tex_h.append([h["model"], L.tex_int(ps["n"]), L.fmt_pass(ps, True), L.fmt_pass(pm, True), L.med_iqr(ev[(ev.held == "strong") & ~ev.vac].n,
                                                                          tex=True),
                      L.tex_num(h["induce_median"])])
        gr = group_rows(model, ev)
        groups += gr
        if tex_g:
            tex_g.append([r"\midrule"])
        for g in gr:
            if g["by"] == "band x kind":
                continue
            lab = ("layers " + g["group"][1:]) if g["by"] == "band" else L.KIND_LABEL.get(g["group"], g["group"])
            tex_g.append([h["model"] if g is gr[0] else "", lab, L.tex_int(g["scored"]), L.fmt_pass(g["_ps"], True),
                          L.fmt_pass(g["_pm"], True), "%s (%s--%s)" % tuple(L.tex_int(g[c]) for c in
                                                                          ("n_median", "n_p25", "n_p75")),
                          L.tex_num(g["induce_median"])])
        print("%-11s targets %d fitted %d vacuous %d | strongest %d/%d = %s | mid %d/%d = %s | n %s | induce %.2f" % (
            h["model"], h["targets"], h["fitted"], h["vacuous"], ps["k"], ps["n"], L.fmt_pass(ps), pm["k"], pm["n"],
            L.fmt_pass(pm), L.med_iqr(ev[(ev.held == "strong") & ~ev.vac].n), h["induce_median"]))
    if not heads:
        return
    hd = pd.DataFrame(heads)
    gd = pd.DataFrame([{k: v for k, v in g.items() if not k.startswith("_")} for g in groups])
    L.write_table("headline", tex_h, ["Model", "Scored", r"Pass, strongest (\%) [95\% CI]",
                                      r"Pass, mid-band (\%) [95\% CI]", "Size, median (IQR)", "Induce"],
                  "lrllll", note="pass = Z, A, C in [%.2f, %.2f] and necessity >= %.1f (merge.passes); vacuous "
                  "excluded; CI = bootstrap over targets (B=%d)" % (L.BAND[0], L.BAND[1], L.NEC_MIN, L.BOOT_B),
                  csv_df=hd)
    L.write_table("headline_groups", tex_g, ["Model", "Group", "Circuits", r"Pass, strongest (\%) [95\% CI]",
                                             r"Pass, mid-band (\%) [95\% CI]", "Size, median (IQR)", "Induce"],
                  "llrllll", csv_df=gd)
    L.write_json("headline", dict(headline=heads, groups=[{k: v for k, v in g.items() if not k.startswith("_")}
                                                          for g in groups]))
    for g in groups:
        if g["by"] != "band x kind":
            print("   %-11s %-6s %-10s n %5d strongest %s | mid %s | size %.0f" % (
                g["model"], g["by"], g["group"], g["scored"], L.fmt_pass(g["_ps"]), L.fmt_pass(g["_pm"]),
                g["n_median"]))
    figure(groups)


if __name__ == "__main__":
    main()
