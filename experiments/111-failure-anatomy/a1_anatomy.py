"""111 a1: failure anatomy (DAN-183), near-threshold stratum, pre-activation pass rate, layer 0, mid-band deviation,
sufficiency to induce >= 0.5, and vacuous rows by term, per production model.

Population: the main run's eval rows (lib109.main_eval: error / skip rows dropped, last row per (seed, held)); the
headline population is the non-vacuous held-out strongest rows (merge.vacuous v2), as in 109's headline.
Rates carry a 95% percentile bootstrap over targets (lib109.boot_ci: B = 4000, seed 20261108).

Writes
  paper/tables/run2026_anatomy.tex          outcome shares per model and depth band (+ site kind block)
  paper/tables/run2026_near_threshold.tex   pre-activation pass rate; near-threshold stratum on both reads + 2x2 split
  paper/tables/run2026_vacuous_terms.tex    vacuous rows by term, per model and held set
  paper/tables/run2026_induce.tex           share inducing to >= 0.5, overall / passing / by site kind
  paper/figures/run2026_failure_anatomy.pdf/.png   one stacked bar per target layer, panels model x site kind
  out/run2026_anatomy_*.csv, out/run2026_near_threshold.csv, out/run2026_vacuous_terms*.csv, out/run2026_induce.csv,
  out/run2026_mid_deviation.csv, out/a1.json

  python -X utf8 experiments/111-failure-anatomy/a1_anatomy.py
"""
import numpy as np
import pandas as pd

import lib111 as X

L = X.L


def pct(x):
    x = pd.Series(x).astype(float)
    return 100 * x.mean() if len(x) else float("nan")


def groups(s, model):
    """(group type, group name, frame) over all / band / kind / band x kind / layer."""
    yield "all", "all", s
    for b, _, _ in L.bands_of(model):
        yield "band", b, s[s.band == b]
    for k in L.KIND_ORDER:
        if (s.kind == k).any():
            yield "kind", k, s[s.kind == k]
    for b, _, _ in L.bands_of(model):
        for k in L.KIND_ORDER:
            g = s[(s.band == b) & (s.kind == k)]
            if len(g):
                yield "band x kind", "%s %s" % (b, k), g
    for l in sorted(s.layer.unique()):
        yield "layer", "L%d" % l, s[s.layer == l]


def main():
    out_rows, test_rows, combo_rows, near_rows, vac_rows, vac_grp, ind_rows, dev_rows = [], [], [], [], [], [], [], []
    tex_a, tex_n, tex_v, tex_i = [], [], [], []
    summary = {}
    fig_data = {}
    for model in L.MODELS:
        lab = L.LABEL[model]
        strong, mid = X.strong_mid(model)
        if strong.empty:
            L.not_available("%s main eval" % model)
            continue
        # ----------------------------------------------------------------- vacuous by term (both held sets)
        for held, d in (("strong", strong), ("mid", mid)):
            vt = X.vac_terms(d)
            recompute = vt.any(axis=1)
            agree = float((recompute == d.vac).mean())
            v = d.vac
            row = dict(model=lab, held=held, rows=len(d), vacuous=int(v.sum()), recompute_agrees=agree,
                       **{"vac_" + c: int((vt[c] & v).sum()) for c in vt},
                       only_Z_nec=int((vt.Z_nec & ~vt.A & ~vt.C & v).sum()),
                       only_A=int((vt.A & ~vt.Z_nec & ~vt.C & v).sum()),
                       only_C=int((vt.C & ~vt.Z_nec & ~vt.A & v).sum()),
                       A_and_C_only=int((vt.A & vt.C & ~vt.Z_nec & v).sum()),
                       all_three=int((vt.A & vt.C & vt.Z_nec & v).sum()))
            vac_rows.append(row)
            for gt, gn, g in groups(d.assign(**{"t_" + c: vt[c] for c in vt}), model):
                if gt in ("band", "kind", "band x kind"):
                    vac_grp.append(dict(model=lab, held=held, group_type=gt, group=gn, rows=len(g),
                                        vacuous=int(g.vac.sum()),
                                        **{"vac_" + c: int((g["t_" + c] & g.vac).sum()) for c in vt}))
            tex_v.append([lab if held == "strong" else "", {"strong": "strongest", "mid": "mid-band"}[held],
                          L.tex_int(row["rows"]), L.tex_int(row["vacuous"]), L.tex_int(row["vac_Z_nec"]),
                          L.tex_int(row["vac_A"]), L.tex_int(row["vac_C"]), L.tex_int(row["vac_apos_le0"])])

        # ----------------------------------------------------------------- outcome anatomy (strongest, non-vacuous)
        s = strong[~strong.vac].copy()
        s["outcome"] = X.outcome(s)
        s["pass_pre"] = X.pre_passes(s).values
        fl = X.test_flags(s)
        s["combo"] = X.combo(fl)
        s["worst"] = (s[L.HEAD] - 1).abs().max(axis=1)
        sp = X.spec_frame(model)
        tsp = X.per_target_spec(sp) if not sp.empty else pd.DataFrame()
        s["near"] = s.seed.map(tsp["near_threshold"]) if len(tsp) else np.nan
        s["rank_clean"] = s.seed.map(tsp["rank_clean"]) if len(tsp) else np.nan
        fig_data[model] = s
        for gt, gn, g in groups(s, model):
            r = dict(model=lab, group_type=gt, group=gn, n=len(g))
            for o in X.OUTCOME_ORDER:
                r[o + "_pct"] = pct(g.outcome == o)
            c = X.rate(g["pass"])
            r.update(pass_lo=c["lo"], pass_hi=c["hi"])
            cp = X.rate(g["pass_pre"])
            r.update(pass_pre_pct=cp["pct"], pass_pre_lo=cp["lo"], pass_pre_hi=cp["hi"])
            r["near_n"] = int((g.near == True).sum())  # noqa: E712
            r["worst_median"] = g.worst.median()
            for k, c_ in zip("ZAC", L.HEAD):
                r[k + "_median"] = g[c_].median()
            r["nec_median"] = g[L.NEC].median()
            out_rows.append(r)
            fails = g[~g["pass"]]
            fr = fl.loc[fails.index]
            t = dict(model=lab, group_type=gt, group=gn, n=len(g), fails=len(fails))
            for c_ in fl.columns:
                t[c_ + "_pct_of_fails"] = pct(fr[c_]) if len(fails) else float("nan")
            t["A_or_C_low_pct_of_fails"] = pct(fr.A_low | fr.C_low) if len(fails) else float("nan")
            t["any_high_pct_of_fails"] = pct(fr.Z_high | fr.A_high | fr.C_high) if len(fails) else float("nan")
            test_rows.append(t)
            if gt in ("all", "band"):
                vc = fails.combo.value_counts()
                for cname, k in vc.items():
                    combo_rows.append(dict(model=lab, group=gn, combo=cname, n=int(k), pct_of_fails=100 * k / len(fails)))
        # LaTeX: per band, then per kind
        bl = [(b, s[s.band == b]) for b, _, _ in L.bands_of(model)] + [("all", s)]
        kl = [(L.KIND_LABEL[k], s[s.kind == k]) for k in L.KIND_ORDER if (s.kind == k).any()]
        if tex_a:
            tex_a.append([r"\midrule"])
        for j, (name, g) in enumerate(bl + kl):
            c = X.rate(g["pass"])
            fails = g[~g["pass"]]
            fr = fl.loc[fails.index]
            tex_a.append([lab if j == 0 else "", L.tex_escape(name), L.tex_int(len(g)), L.fmt_pass(c, True)]
                         + ["%.1f" % pct(g.outcome == o) for o in ("mean", "zero", "nec", "undef")]
                         + ["%.0f" % pct(fr.Z_high | fr.A_high | fr.C_high) if len(fails) else L.DASH,
                            L.tex_num(g.worst.median())])

        # ----------------------------------------------------------------- near-threshold + pre-activation
        near = s[s.near == True]  # noqa: E712
        rest = s[s.near == False]  # noqa: E712
        nr = dict(model=lab, n=len(s), K=int(sp.K.dropna().iloc[0]) if len(sp) else None,
                  no_spec_row=int(s.near.isna().sum()), near_n=len(near), near_share_pct=pct(s.near == True))  # noqa: E712
        for name, g in (("all", s), ("near", near), ("rest", rest)):
            a, p = X.rate(g["pass"]), X.rate(g["pass_pre"])
            nr.update({"%s_pass_pct" % name: a["pct"], "%s_pass_lo" % name: a["lo"], "%s_pass_hi" % name: a["hi"],
                       "%s_pre_pct" % name: p["pct"], "%s_pre_lo" % name: p["lo"], "%s_pre_hi" % name: p["hi"]})
        nr.update(near_both_pct=pct(near["pass"] & near.pass_pre), near_pre_only_pct=pct(~near["pass"] & near.pass_pre),
                  near_act_only_pct=pct(near["pass"] & ~near.pass_pre), near_neither_pct=pct(~near["pass"] & ~near.pass_pre),
                  all_both_pct=pct(s["pass"] & s.pass_pre), all_pre_only_pct=pct(~s["pass"] & s.pass_pre),
                  all_act_only_pct=pct(s["pass"] & ~s.pass_pre),
                  near_by_kind="; ".join("%s %d/%d" % (k, int((near.kind == k).sum()), int((s.kind == k).sum()))
                                         for k in L.KIND_ORDER if (s.kind == k).any()),
                  near_attn_zero_fail_share_pct=pct(s[(s.kind == "attn") & (s.outcome == "zero")].near == True))  # noqa: E712
        near_rows.append(nr)
        tex_n.append([lab, L.tex_int(len(s)),
                      L.fmt_pass(X.rate(s["pass"]), True), L.fmt_pass(X.rate(s["pass_pre"]), True),
                      "%s (%.1f\\%%)" % (L.tex_int(len(near)), nr["near_share_pct"]),
                      L.fmt_pass(X.rate(near["pass"]), True), L.fmt_pass(X.rate(near["pass_pre"]), True),
                      "%.0f / %.0f / %.0f / %.0f" % (nr["near_both_pct"], nr["near_pre_only_pct"], nr["near_act_only_pct"],
                                                     nr["near_neither_pct"]),
                      L.fmt_pass(X.rate(rest["pass"]), True)])

        # ----------------------------------------------------------------- mid-band vs strongest deviation
        m = mid[~mid.vac].copy()
        m["worst"] = (m[L.HEAD] - 1).abs().max(axis=1)
        for name, gs, gm in [("all", s, m)] + [(b, s[s.band == b], m[m.band == b]) for b, _, _ in L.bands_of(model)]:
            paired = gs.set_index("seed").worst.to_frame("ws").join(gm.set_index("seed").worst.to_frame("wm"), how="inner")
            dev_rows.append(dict(model=lab, group=name, n_strong=len(gs), n_mid=len(gm),
                                 worst_strong_median=gs.worst.median(), worst_mid_median=gm.worst.median(),
                                 paired_n=len(paired), paired_mid_minus_strong_median=(paired.wm - paired.ws).median(),
                                 mid_pass_pct=pct(gm["pass"])))

        # ----------------------------------------------------------------- sufficiency to induce >= 0.5
        ind = pd.to_numeric(s[L.IND], errors="coerce")
        s["ind_half"] = ind >= 0.5                                   # undefined counts as not inducing
        tex_row = [lab]
        for name, g in [("all", s), ("passing", s[s["pass"]])] + [(L.KIND_LABEL[k], s[s.kind == k])
                                                                  for k in L.KIND_ORDER if (s.kind == k).any()] + \
                [(b, s[s.band == b]) for b, _, _ in L.bands_of(model)]:
            c = X.rate(g.ind_half)
            ind_rows.append(dict(model=lab, group=name, n=c["n"], k=c["k"], pct=c["pct"], lo=c["lo"], hi=c["hi"],
                                 induce_median=pd.to_numeric(g[L.IND], errors="coerce").median(),
                                 undefined=int(pd.to_numeric(g[L.IND], errors="coerce").isna().sum())))
        tex_i.append([lab, L.tex_int(len(s)), L.fmt_pass(X.rate(s.ind_half), True),
                      L.fmt_pass(X.rate(s[s["pass"]].ind_half), True)]
                     + [L.fmt_pass(X.rate(s[s.kind == k].ind_half), True) if (s.kind == k).any() else L.DASH
                        for k in L.KIND_ORDER])

        summary[model] = dict(
            scored=len(s), pass_pct=pct(s["pass"]), fails=int((~s["pass"]).sum()),
            outcome_pct={o: pct(s.outcome == o) for o in X.OUTCOME_ORDER},
            mean_only_share_of_fails=100 * (s.outcome == "mean").sum() / max(1, (~s["pass"]).sum()),
            layer0={o: pct(s[s.layer == 0].outcome == o) for o in X.OUTCOME_ORDER} | {"n": int((s.layer == 0).sum())},
            pre_pass_pct=pct(s.pass_pre), near_n=len(near))
        print("%-11s scored %d | pass %.1f | outcomes %s | mean-only share of fails %.1f%% | pre-act pass %.1f | "
              "near-threshold %d (pass %.1f, pre %.1f)" % (
                  lab, len(s), pct(s["pass"]), ", ".join("%s %.1f" % (o, v) for o, v in summary[model]["outcome_pct"].items()),
                  summary[model]["mean_only_share_of_fails"], pct(s.pass_pre), len(near), pct(near["pass"]),
                  pct(near.pass_pre)))

    # ----------------------------------------------------------------- write
    pd.DataFrame(out_rows).to_csv(X.OUT / "run2026_anatomy_outcomes.csv", index=False)
    pd.DataFrame(test_rows).to_csv(X.OUT / "run2026_anatomy_tests.csv", index=False)
    pd.DataFrame(combo_rows).to_csv(X.OUT / "run2026_anatomy_combos.csv", index=False)
    pd.DataFrame(dev_rows).to_csv(X.OUT / "run2026_mid_deviation.csv", index=False)
    pd.DataFrame(vac_grp).to_csv(X.OUT / "run2026_vacuous_terms_groups.csv", index=False)
    L.write_json("a1", summary)
    L.write_table("anatomy", tex_a, ["Model", "Group", "$n$", r"Pass (\%) [95\% CI]", r"A/C only (\%)",
                                     r"Z (\%)", r"Necessity (\%)", r"Undefined (\%)", r"Fails above 1.5 (\% of fails)",
                                     "Largest deviation"],
                  "llrlrrrrrr",
                  note="one outcome per non-vacuous held-out-strongest circuit (062 paper_main.outcome order: pass, "
                       "undefined, necessity < 0.9, Z out of band, else A and/or C only); 'fails above 1.5' = share "
                       "of failing circuits with any of Z, A, C > 1.5; largest deviation = median max|phi - 1|",
                  csv_df=pd.DataFrame(out_rows),
                  footer=[r"Outcome classes are exclusive, in the order pass, undefined, necessity, Z, A/C only; held-out strongest, non-vacuous."])
    L.write_table("near_threshold", tex_n, ["Model", "$n$", r"Pass, activation (\%)", r"Pass, pre-activation (\%)",
                                            "Near-threshold", r"Near: activation (\%)", r"Near: pre-activation (\%)",
                                            r"Near: both / pre only / act.\ only / neither (\%)",
                                            r"Others: activation (\%)"],
                  "lrlllllll", note="near-threshold = median clean rank at the held-out strongest anchors >= K/2 "
                                    "(src/eval/wcm_specificity.target_specificity); pre-activation read = "
                                    "paper/tables/common.pre_passes (Z and necessity from 0)",
                  csv_df=pd.DataFrame(near_rows))
    L.write_table("vacuous_terms", tex_v, ["Model", "Held-out set", "Rows", "Vacuous", "Z / necessity", "A", "C",
                                           r"$a_{\mathrm{pos}} \leq 0$"], "llrrrrrr",
                  note="a row is vacuous if a_pos <= 0 or a_pos - e < 0.05 a_pos for e in e0 (Z, necessity), eM_topk "
                       "(A), eN_topk (C); term columns are non-exclusive",
                  csv_df=pd.DataFrame(vac_rows),
                  footer=[r"Term columns overlap: a row counts under every denominator that is within $5\%$ of $a_{\mathrm{pos}}$."])
    L.write_table("induce", tex_i, ["Model", "$n$", r"All (\%)", r"Passing (\%)"]
                  + [r"%s (\%%)" % L.KIND_LABEL[k] for k in L.KIND_ORDER], "lrllllll",
                  note="share of non-vacuous held-out-strongest circuits with phi_cf_alpha_blind_tk >= 0.5 (undefined "
                       "counts as below)", csv_df=pd.DataFrame(ind_rows))
    pd.DataFrame(vac_rows).to_csv(X.OUT / "run2026_vacuous_terms.csv", index=False)
    figure(fig_data)
    for r in vac_rows:
        print("vacuous %-11s %-6s rows %d vacuous %d | Z/nec %d A %d C %d a_pos<=0 %d | only A %d only C %d A&C %d all %d"
              " | recompute agrees %.4f" % (r["model"], r["held"], r["rows"], r["vacuous"], r["vac_Z_nec"], r["vac_A"],
                                            r["vac_C"], r["vac_apos_le0"], r["only_A"], r["only_C"], r["A_and_C_only"],
                                            r["all_three"], r["recompute_agrees"]))
    for r in near_rows:
        print("near %-11s K %s | near %d (%.1f%%; no spec row %d) by kind %s | pass act %.1f pre %.1f | near act %.1f "
              "pre %.1f | 2x2 %.0f/%.0f/%.0f/%.0f | others act %.1f pre %.1f | near share of attn Z-fails %.0f%%" % (
                  r["model"], r["K"], r["near_n"], r["near_share_pct"], r["no_spec_row"], r["near_by_kind"],
                  r["all_pass_pct"], r["all_pre_pct"], r["near_pass_pct"], r["near_pre_pct"], r["near_both_pct"],
                  r["near_pre_only_pct"], r["near_act_only_pct"], r["near_neither_pct"], r["rest_pass_pct"],
                  r["rest_pre_pct"], r["near_attn_zero_fail_share_pct"]))
    for r in ind_rows:
        if r["group"] in ("all", "passing"):
            print("induce>=0.5 %-11s %-8s %d/%d = %.1f [%.1f, %.1f]; median %.2f" % (
                r["model"], r["group"], r["k"], r["n"], r["pct"], r["lo"], r["hi"], r["induce_median"]))
    for r in dev_rows:
        print("deviation %-11s %-6s strong %.3f mid %.3f (paired diff %.3f, n %d)" % (
            r["model"], r["group"], r["worst_strong_median"], r["worst_mid_median"],
            r["paired_mid_minus_strong_median"], r["paired_n"]))


def figure(data):
    """Failure anatomy: one stacked bar per target layer, rows = models, columns = site kinds (062 figure_v4 style)."""
    plt, style = L.figure_style()
    colors = {"pass": "#0044ff", "mean": style.tint("#0044ff", 0.62), "zero": "#9aa0a8", "nec": style.INK_SECONDARY,
              "undef": "#e3e5e9"}
    models = [m for m in L.MODELS if m in data]
    kinds = [k for k in L.KIND_ORDER if any((data[m].kind == k).any() for m in models)]
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.labelsize": 6.8, "xtick.labelsize": 6.2,
                         "ytick.labelsize": 6.2, "legend.fontsize": 6.4})
    fig, axes = plt.subplots(len(models), len(kinds), figsize=(5.5, 1.75 * len(models) + 0.4), sharey=True,
                             squeeze=False)
    for i, m in enumerate(models):
        s = data[m]
        n_layer = int(s.layer.max()) + 1
        mk = [k for k in kinds if (s.kind == k).any()]           # this model's kinds, left-aligned
        for j in range(len(mk), len(kinds)):
            axes[i, j].axis("off")
        for j, k in enumerate(mk):
            ax = axes[i, j]
            g = s[s.kind == k]
            share = (g.groupby("layer").outcome.value_counts(normalize=True).unstack()
                     .reindex(columns=X.OUTCOME_ORDER).fillna(0) * 100)
            bottom = np.zeros(len(share))
            for o in X.OUTCOME_ORDER:
                ax.bar(share.index, share[o], 0.78, bottom=bottom, color=colors[o], edgecolor=style.SURFACE,
                       linewidth=0.4, label=X.OUTCOME_LABEL[o], zorder=2)
                bottom += share[o].values
            ax.set(xlim=(-0.6, n_layer - 0.4), ylim=(0, 100), xticks=range(0, n_layer, 2 if n_layer > 6 else 1))
            ax.grid(False)
            ax.set_title("%s, %.0f%% pass" % ({"resid_mid": "resid. mid"}.get(k, L.KIND_LABEL[k]), 100 * g["pass"].mean()))
            ax.set_xlabel("target layer")
        axes[i, 0].set_ylabel("%s\ncircuits (%%)" % L.LABEL[m])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, handlelength=1.0)
    L.save_with_legend(fig, L.FIGURES / "run2026_failure_anatomy.png", top=0.94)
    print("figure:", L.FIGURES / "run2026_failure_anatomy.pdf")


if __name__ == "__main__":
    main()
