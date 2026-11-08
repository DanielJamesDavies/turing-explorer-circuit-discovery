"""Table 4: robustness of the production circuits, per model where the arm ran.

  nulls     084 harder nulls (anchor-active and contribution-matched random member sets, fitted like WCM, plus their
            alpha = 1 scoring) vs the production circuit on the same targets. Rows: public out_<model>/harder_nulls/
            nulls.shard*.jsonl; TuringLLM (T6) out_turingllm/harder_nulls/. Frame + per-target quantile logic from
            084 report.py (frame, table_quantiles), with the pass rule and vacuous set replaced by merge.passes /
            merge.vacuous (084's own report still applies the pre-2026-10-31 floor 0.8). Pass on draw 0, held-out
            strongest; bootstrap CIs over targets; paired difference production - null.
  resample  073 strict eval (public out_<model>/strict/rows.shard*.jsonl; not run on TuringLLM): 073 report.py's
            definitions (Z from the all-position anchor means, A / C / R resample faithfulness rs_*), joined with the
            production row; pass = Z, A, C (or R) in merge.BAND and production necessity >= NEC_MIN; vacuous by
            merge.vacuous on the production row.
  donor     src/eval/donor_test.py rows (public out_<model>/donor[_unweighted]/rows.shard*.jsonl; TuringLLM T11
            out_v2/donor/main/, T12 out_v2/donor/sweep/): donor pass = production pass AND pooled donor recovery
            A and C (variant fit) in BAND (donor_test.resample_pass), recomputed with merge's rule from the stored
            prod row and checked against the stored flag; vacuous by merge.vacuous on the prod row.
  stable    062 stable_core.py rows (public out_<model>/stable_core/; TuringLLM T5 out_v2/stable_core/): fit 2 =
            the same protocol with the training rows reordered; core = members in both fits, coefficients refitted.
            Pass rates of fit 1 (production), fit 2 and core on the targets whose fit-1 row is non-vacuous.

Writes  paper/tables/run2026_nulls.tex, run2026_resample.tex, run2026_stable_core.tex and the matching out/*.csv,
        out/run2026_nulls_quantiles.csv

  python -X utf8 experiments/109-pod-run-analysis/t4_robustness.py
"""
import importlib.util

import numpy as np
import pandas as pd

import lib109 as L


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import sys  # noqa: E402
sys.path.insert(0, str(L.ROOT / "experiments" / "084-harder-nulls"))
R84 = _load("report84", L.ROOT / "experiments" / "084-harder-nulls" / "report.py")


ARM_LABEL = {"wcm": "production circuit (WCM)", "wcm_a1": r"same members, $\alpha = 1$",
             "dan17": "random members (weak baseline)", "dan17_a1": r"random members, $\alpha = 1$",
             "anchor": "anchor-active random", "anchor_a1": r"anchor-active random, $\alpha = 1$",
             "contrib": "contribution-matched random", "contrib_a1": r"contribution-matched random, $\alpha = 1$",
             "uniform": "uniform anchor-active random", "uniform_a1": r"uniform anchor-active random, $\alpha = 1$"}


def in_band(x):
    x = pd.to_numeric(x, errors="coerce")
    return x.notna() & (x >= L.BAND[0]) & (x <= L.BAND[1])


# ------------------------------------------------------------------------------------------------- harder nulls
def nulls_dirs(model):
    return L.find(model, "work", "harder_nulls") if model == "turingllm" else L.find(model, "out", "harder_nulls")


def nulls(model):
    rows = []
    for d in nulls_dirs(model):
        rows += R84.read_rows(d, "nulls")
    if not rows:
        return None
    df = R84.frame(rows)
    df["vacuous"] = L.vacuous(df).values
    ok = ~df.vacuous
    df["pass"] = False
    df.loc[ok, "pass"] = L.passes(df[ok]).values
    df["pass"] = df["pass"].astype(bool)
    df = L.add_site(df, model)
    h = df[(df.split == "held") & ~df.vacuous]
    d0 = h[h.draw == 0]
    w = d0[d0.arm == "wcm"].set_index("seed")
    out = []
    ev = L.main_eval(model)
    prod = ev[(ev.held == "strong") & ~ev.vac].set_index("seed")
    for arm in R84.arms_in(h):
        a = d0[d0.arm == arm].set_index("seed")
        both = a.index.intersection(w.index)
        c = L.pass_cell(a["pass"])
        lo, hi = L.boot_paired_diff(w.loc[both, "pass"].astype(float).values, a.loc[both, "pass"].astype(float).values)
        out.append(dict(model=L.LABEL[model], arm=arm, targets=len(a), draws=int((h.arm == arm).sum()),
                        pass_k=c["k"], pass_pct=c["pct"], pass_lo=c["lo"], pass_hi=c["hi"],
                        pass_all_draws_pct=100 * h[h.arm == arm]["pass"].mean(),
                        prod_minus_arm_pp=100 * (w.loc[both, "pass"].mean() - a.loc[both, "pass"].mean()),
                        diff_lo=100 * lo, diff_hi=100 * hi, paired=len(both),
                        n_median=a.n.median() if "n" in a else np.nan,
                        **{"%s_median" % k: a[col].median() for k, col in zip("ZAC", L.HEAD)},
                        nec_median=a[L.NEC].median(), _c=c))
    agree = w.index.intersection(prod.index)
    same = float((w.loc[agree, "pass"] == prod.loc[agree, "pass"]).mean()) if len(agree) else np.nan
    q = R84.table_quantiles(df)
    q.insert(0, "model", L.LABEL[model])
    by_band = []
    for b, _, _ in L.bands_of(model):
        for arm in ("wcm", "anchor", "contrib"):
            x = d0[(d0.arm == arm) & (d0.band == b)]
            if len(x):
                by_band.append(dict(model=L.LABEL[model], band=b, arm=arm, targets=len(x), pass_pct=100 * x["pass"].mean()))
    return out, q, by_band, dict(agree_with_main=same, n_agree=len(agree),
                                 error_rows=sum("error" in r for r in rows), targets=int(df.seed.nunique()))


# ------------------------------------------------------------------------------------------------- 073 resample
def strict(model):
    if model == "turingllm":
        return None
    rows = L.read_rows(L.find(model, "out", "strict"), "rows")
    if not rows:
        return None
    d = L.frame(rows, keys=("seed",)).set_index("seed")
    ev = L.main_eval(model)
    p = ev[ev.held == "strong"].set_index("seed")
    d = d.drop(columns=[c for c in ("layer", "kind", "band") if c in d]).join(
        p[L.HEAD + [L.NEC, "vac", "pass", "n", "band", "kind"]], how="inner")
    n_joined = len(d)
    d = d[~d.vac]
    nec = d[L.NEC] >= L.NEC_MIN
    e = d["ap_Z_empty_anchor_mean"]
    res = {}
    for v in ("wcm", "a1", "rand"):
        if "rs_A_%s" % v not in d:
            continue
        z = (d["ap_Z_%s_anchor_mean" % v] - e) / (d["a_pos"] - e)
        d["rs_pass_%s" % v] = in_band(z) & in_band(d["rs_A_%s" % v]) & in_band(d["rs_C_%s" % v]) & nec
        d["rsR_pass_%s" % v] = in_band(z) & in_band(d["rs_A_%s" % v]) & in_band(d["rs_R_%s" % v]) & nec
        if v == "wcm":
            d["Z_strict_wcm"] = z
    hr = d[d.has_rand.fillna(False).astype(bool)] if "has_rand" in d else d.iloc[0:0]
    for lab, x, col in (("production", d, "pass"), ("resample A,C", d, "rs_pass_wcm"), ("resample A,R", d, "rsR_pass_wcm"),
                        ("resample A,C, alpha=1", d, "rs_pass_a1"), ("resample A,C, random null", hr, "rs_pass_rand")):
        if col in x:
            res[lab] = L.pass_cell(x[col].astype(bool))
    cond = L.pass_cell(d[d["pass"]]["rs_pass_wcm"]) if "rs_pass_wcm" in d else None
    med = {k: d[c].median() for k, c in (("rs_A", "rs_A_wcm"), ("rs_C", "rs_C_wcm"), ("rs_R", "rs_R_wcm")) if c in d}
    zdiff = float((d[L.HEAD[0]] - d["Z_strict_wcm"]).abs().median()) if "Z_strict_wcm" in d else np.nan
    return dict(model=L.LABEL[model], rows=len(rows), errors=sum("error" in r for r in rows), joined=n_joined,
                scored=len(d), with_rand=len(hr), cells=res, cond=cond, med=med, z_sanity=zdiff)


# ------------------------------------------------------------------------------------------------- donor test
def donor_dirs(model):
    if model == "turingllm":
        return {"main": L.find(model, "out", "donor/main"), "sweep": L.find(model, "out", "donor/sweep"),
                "noR": L.find(model, "nor", "donor/sweep") + L.find(model, "nor", "donor")}
    return {"main": L.find(model, "out", "donor"), "unweighted": L.find(model, "out", "donor_unweighted")}


def _lam_tex(s):
    """'0.00025' -> '2.5 \\times 10^{-4}', '0.001' -> '10^{-3}'."""
    m, e = ("%.1e" % float(s)).split("e")
    m = m.rstrip("0").rstrip(".")
    return ("10^{%d}" % int(e)) if m == "1" else (r"%s \times 10^{%d}" % (m, int(e)))


def donor(model):
    out = []
    for src, dirs in donor_dirs(model).items():
        rows = L.read_rows(dirs, "rows")
        if not rows:
            continue
        good = [r for r in rows if "error" not in r and not r.get("skip")]
        last = {}
        for r in good:
            last[(r["seed"], r["arm"])] = r
        for arm in sorted({k[1] for k in last}):
            rs = [r for k, r in last.items() if k[1] == arm]
            prod = pd.DataFrame([r.get("prod", {}) for r in rs])
            prod["seed"] = [r["seed"] for r in rs]
            vac = L.vacuous(prod).values
            prod_pass = np.zeros(len(rs), dtype=bool)
            ok = ~vac
            if ok.any():
                prod_pass[ok] = L.passes(prod[ok]).values
            recA = pd.Series([(r.get("recovery") or {}).get("fit", {}).get("A") for r in rs], dtype=float)
            recC = pd.Series([(r.get("recovery") or {}).get("fit", {}).get("C") for r in rs], dtype=float)
            recR = pd.Series([(r.get("recovery") or {}).get("fit", {}).get("R") for r in rs], dtype=float)
            dp = prod_pass & in_band(recA).values & in_band(recC).values
            stored = np.array([bool(r.get("resample_pass")) for r in rs])
            sc = ok
            c_prod, c_donor = L.pass_cell(pd.Series(prod_pass[sc])), L.pass_cell(pd.Series(dp[sc]))
            c_cond = L.pass_cell(pd.Series(dp[sc & prod_pass]))
            v = lambda var, ds: pd.Series([(r.get("recovery") or {}).get(var, {}).get(ds) for r, s in zip(rs, sc) if s],  # noqa: E731
                                          dtype=float).median()
            out.append(dict(model=L.LABEL[model], source=src, arm=arm, rows=len(rows), errors=sum("error" in r for r in rows),
                            targets=len(rs), scored=int(sc.sum()), prod_pass_pct=c_prod["pct"], donor_pass_k=c_donor["k"],
                            donor_pass_pct=c_donor["pct"], donor_pass_lo=c_donor["lo"], donor_pass_hi=c_donor["hi"],
                            donor_given_prod_pct=c_cond["pct"], donor_given_prod_lo=c_cond["lo"],
                            donor_given_prod_hi=c_cond["hi"], n_prod_pass=c_cond["n"],
                            stored_flag_agrees=float((stored[sc] == dp[sc]).mean()),
                            undefined_A_or_C=int((recA[sc].isna() | recC[sc].isna()).sum()),
                            rec_A_fit=v("fit", "A"), rec_C_fit=v("fit", "C"), rec_R_fit=v("fit", "R"),
                            rec_A_a1=v("a1", "A"), rec_C_a1=v("a1", "C"), rec_A_pinned=v("pinned", "A"),
                            rec_C_pinned=v("pinned", "C"), R_in_band_pct=100 * float(in_band(recR[sc]).mean()),
                            _cd=c_donor, _cc=c_cond, _cp=c_prod))
    return out


# ------------------------------------------------------------------------------------------------- stable core
def stable(model):
    rows = L.read_rows(L.find(model, "out", "stable_core"), "stable_core")
    if not rows:
        return None
    d = pd.DataFrame([r for r in rows if r.get("stage") != "error" and "error" not in r])
    d = d.drop_duplicates(["seed", "stage"], keep="last")
    ev = L.main_eval(model)
    f1 = ev[ev.held == "strong"].set_index("seed")
    f2 = L.score(d[d.stage == "fit2"].drop(columns=["layer", "kind"], errors="ignore").assign(
        held="strong")).set_index("seed")
    core_all = d[d.stage == "core"].set_index("seed")
    core = L.score(core_all[core_all.get("skip").isna()].reset_index() if "skip" in core_all else core_all.reset_index()
                   ).set_index("seed")
    keep = [s for s in f2.index if s in f1.index and not f1.loc[s, "vac"]]
    f1k, f2k = f1.loc[keep], f2.loc[keep]
    ck = core.reindex(keep)
    core_pass = ck["pass"].fillna(False).astype(bool) & ~ck["vac"].fillna(False).astype(bool)
    both = f1k["pass"] & f2k["pass"]
    cells = dict(fit1=L.pass_cell(f1k["pass"]), fit2=L.pass_cell(f2k["pass"] & ~f2k.vac), core=L.pass_cell(core_pass))
    return dict(model=L.LABEL[model], targets=int(d.seed.nunique()), scored=len(keep),
                jaccard_median=f2k.jaccard.median(), jaccard_p25=f2k.jaccard.quantile(.25),
                jaccard_p75=f2k.jaccard.quantile(.75), n1_median=f2k.n1.median(), n2_median=f2k.n2.median(),
                core_median=f2k.n_core.median(), core_over_fit1=(f2k.n_core / f2k.n1).median(),
                empty_core=int((core_all.get("skip") == "empty_core").sum()) if "skip" in core_all else 0,
                both_pass=int(both.sum()), core_pass_given_both=int(core_pass[both].sum()),
                fit1_pct=cells["fit1"]["pct"], fit2_pct=cells["fit2"]["pct"], core_pct=cells["core"]["pct"],
                core_lo=cells["core"]["lo"], core_hi=cells["core"]["hi"], _cells=cells)


def main():
    null_rows, qs, nb, res_rows, don_rows, st_rows = [], [], [], [], [], []
    tex_n, tex_r, tex_s = [], [], []
    for model in L.MODELS:
        lab = L.LABEL[model]
        # ---- nulls
        n = nulls(model)
        if n is None:
            L.not_available("%s harder nulls%s" % (model, " (T6)" if model == "turingllm" else ""))
        else:
            rows, q, by_band, meta = n
            null_rows += rows
            qs.append(q)
            nb += by_band
            print("%-11s harder nulls: %d targets, %d error rows; wcm arm agrees with the main pass on %.1f%% of %d" % (
                lab, meta["targets"], meta["error_rows"], 100 * meta["agree_with_main"], meta["n_agree"]))
            if tex_n:
                tex_n.append([r"\midrule"])
            for r in rows:
                print("    %-11s targets %4d pass %s | prod - arm %+.1f pp [%+.1f, %+.1f] | n %.0f | Z/A/C %.2f/%.2f/%.2f"
                      % (r["arm"], r["targets"], L.fmt_pass(r["_c"]), r["prod_minus_arm_pp"], r["diff_lo"],
                         r["diff_hi"], r["n_median"], r["Z_median"], r["A_median"], r["C_median"]))
                tex_n.append([lab if r is rows[0] else "", ARM_LABEL.get(r["arm"], L.tex_escape(r["arm"])),
                              L.tex_int(r["targets"]),
                              L.fmt_pass(r["_c"], True),
                              L.DASH if r["arm"] == "wcm" else "%+.1f {[}%+.1f, %+.1f{]}" % (
                                  r["prod_minus_arm_pp"], r["diff_lo"], r["diff_hi"]),
                              "%s / %s / %s" % tuple(L.tex_num(r["%s_median" % k]) for k in "ZAC"),
                              L.tex_num(r["nec_median"])])
        # ---- strict
        s = strict(model)
        if s is None:
            L.not_available("%s 073 resample rows%s" % (model, " (not in the TuringLLM chain)" if model == "turingllm"
                                                        else ""))
        else:
            print("%-11s 073 resample: rows %d (errors %d), scored %d (with random null %d); |Z prod - Z strict| median"
                  " %.3f" % (lab, s["rows"], s["errors"], s["scored"], s["with_rand"], s["z_sanity"]))
            for k, c in s["cells"].items():
                print("    %-26s %d/%d = %s" % (k, c["k"], c["n"], L.fmt_pass(c)))
            print("    resample A,C among production passes: %s; medians A %.2f C %.2f R %.2f" % (
                L.fmt_pass(s["cond"]), s["med"].get("rs_A", np.nan), s["med"].get("rs_C", np.nan),
                s["med"].get("rs_R", np.nan)))
            res_rows.append(dict(model=lab, test="073 resample", scored=s["scored"],
                                 **{"%s_pct" % k: c["pct"] for k, c in s["cells"].items()},
                                 **{"%s_ci" % k: L.fmt_pass(c) for k, c in s["cells"].items()},
                                 given_prod_pct=s["cond"]["pct"], **s["med"]))
            c = s["cells"]
            tex_r.append([lab, "strict resample (A, C)", L.tex_int(s["scored"]), L.fmt_pass(c["production"], True),
                          L.fmt_pass(c["resample A,C"], True), L.fmt_pass(s["cond"], True),
                          "%s / %s / %s" % tuple(L.tex_num(s["med"].get(k)) for k in ("rs_A", "rs_C", "rs_R"))])
            if "resample A,C, random null" in c:
                tex_r.append(["", "strict resample, random null", L.tex_int(c["resample A,C, random null"]["n"]), L.DASH,
                              L.fmt_pass(c["resample A,C, random null"], True), L.DASH, L.DASH])
        # ---- donor
        dn = donor(model)
        if not dn:
            L.not_available("%s donor-background rows%s" % (model, " (T11/T12)" if model == "turingllm" else ""))
        # 2026-11-08: rows ordered main, WCM sweep by lambda, without-R arm, unweighted sweep by lambda
        def _key(r):
            a = r["arm"]
            grp = {"main": 0, "noR": 2}.get(r["source"], 1 if a.startswith("W_") else 3)
            lam = float(a[2:]) if a[:2] in ("W_", "U_") else 0.0
            return (grp, lam)
        dn = sorted(dn, key=_key)
        for r in dn:
            print("%-11s donor %-10s %-8s targets %d scored %d (errors %d) | prod pass %.1f | donor pass %s | given prod "
                  "%s | stored flag agrees %.3f | undefined %d | recovery fit A %.2f C %.2f R %.2f; a1 A %.2f; pinned A %.2f"
                  % (lab, r["source"], r["arm"], r["targets"], r["scored"], r["errors"], r["prod_pass_pct"],
                     L.fmt_pass(r["_cd"]), L.fmt_pass(r["_cc"]), r["stored_flag_agrees"], r["undefined_A_or_C"],
                     r["rec_A_fit"], r["rec_C_fit"], r["rec_R_fit"], r["rec_A_a1"], r["rec_A_pinned"]))
            don_rows.append({k: v for k, v in r.items() if not k.startswith("_")})
            arm_lab = ({"main": "WCM"}.get(r["arm"]) or (
                r"WCM without R, $\lambda = %s$" % _lam_tex(r["arm"][2:]) if r["source"] == "noR" else
                r"unweighted, $\lambda = %s$" % _lam_tex(r["arm"][2:]) if r["arm"].startswith("U_") else
                r"WCM, $\lambda = %s$" % _lam_tex(r["arm"][2:]) if r["arm"].startswith("W_") else
                L.tex_escape(r["arm"])))
            if r["source"] in ("sweep", "noR") and model == "turingllm":
                arm_lab = "sweep: " + arm_lab
            tex_r.append([lab if not s and r is dn[0] else "", "donor test (%s)" % arm_lab,
                          L.tex_int(r["scored"]), L.fmt_pass(r["_cp"], True), L.fmt_pass(r["_cd"], True),
                          L.fmt_pass(r["_cc"], True),
                          "%s / %s / %s" % tuple(L.tex_num(r[k]) for k in ("rec_A_fit", "rec_C_fit", "rec_R_fit"))])
        if tex_r and model != L.MODELS[-1] and tex_r[-1] != [r"\midrule"]:
            tex_r.append([r"\midrule"])
        # ---- stable core
        sc = stable(model)
        if sc is None:
            L.not_available("%s stable core rows%s" % (model, " (T5)" if model == "turingllm" else ""))
        else:
            c = sc["_cells"]
            print("%-11s stable core: %d targets (%d non-vacuous) | Jaccard %.2f (%.2f-%.2f) | n1 %.0f n2 %.0f core %.0f "
                  "(%.2f of fit 1) | pass fit1 %s fit2 %s core %s | both pass %d, core passes %d of them" % (
                      lab, sc["targets"], sc["scored"], sc["jaccard_median"], sc["jaccard_p25"], sc["jaccard_p75"],
                      sc["n1_median"], sc["n2_median"], sc["core_median"], sc["core_over_fit1"], L.fmt_pass(c["fit1"]),
                      L.fmt_pass(c["fit2"]), L.fmt_pass(c["core"]), sc["both_pass"], sc["core_pass_given_both"]))
            st_rows.append({k: v for k, v in sc.items() if not k.startswith("_")})
            tex_s.append([lab, L.tex_int(sc["scored"]), "%.2f (%.2f--%.2f)" % (sc["jaccard_median"], sc["jaccard_p25"],
                                                                           sc["jaccard_p75"]),
                          "%s / %s / %s" % tuple(L.tex_int(sc[k]) for k in ("n1_median", "n2_median", "core_median")),
                          L.fmt_pass(c["fit1"], True), L.fmt_pass(c["fit2"], True), L.fmt_pass(c["core"], True),
                          "%d / %d" % (sc["core_pass_given_both"], sc["both_pass"])])
    if null_rows:
        L.write_table("nulls", tex_n, ["Model", "Circuit", "Targets", r"Pass (\%) [95\% CI]", "Production $-$ arm (pp)",
                                       "Z / A / C (median)", "Necessity"], "llrllll",
                      note="held-out strongest, draw 0; random sets are fitted like WCM unless marked alpha = 1; "
                           "code arm names: wcm, wcm_a1, dan17(_a1), anchor(_a1), contrib(_a1) (CSV column arm)",
                      csv_df=pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in null_rows]))
        pd.concat(qs).to_csv(L.OUT / ("%snulls_quantiles.csv" % L.PREFIX), index=False)
        pd.DataFrame(nb).to_csv(L.OUT / ("%snulls_by_band.csv" % L.PREFIX), index=False)
    if tex_r:
        while tex_r and tex_r[-1] == [r"\midrule"]:
            tex_r.pop()
        L.write_table("resample", tex_r, ["Model", "Test", "Circuits", r"Production pass (\%)",
                                          r"Test pass (\%) [95\% CI]", r"Test pass $\mid$ production pass (\%)",
                                          "A / C / T (median)"], "llrllll",
                      note="test pass = production pass AND the resample / donor faithfulness A and C in the band; "
                           "T = the random-text donor pool (code name R; renamed so it is not read as the resample term)",
                      csv_df=pd.DataFrame(res_rows + don_rows))
    if st_rows:
        L.write_table("stable_core", tex_s, ["Model", "Targets", "Jaccard (IQR)", "Size fit 1 / fit 2 / core",
                                             r"Pass fit 1 (\%)", r"Pass fit 2 (\%)", r"Pass core (\%)",
                                             "Core passes / both pass"], "lrllllll", csv_df=pd.DataFrame(st_rows))


if __name__ == "__main__":
    main()
