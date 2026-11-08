"""Table 6 + figure: the TuringLLM lambda sweep (T4) and the without-R arm (T10). Prints "not yet available" until the
T4 / T10 archives are unpacked.

Reads   out_v2/sweep/eval.shard*.jsonl (T4: 190 targets x W_2.5e-05 .. W_0.004 and U_1e-05 .. U_0.001, held strong);
        W_0.001 reuses the T3 circuit, so when the sweep has no W_0.001 row for a target the T3 main held-out-strongest
        row is used (source "main"); out_turingllm/lists/targets_tl8k_sweep190.txt (the population);
        out_v2_noR/sweep/eval.shard*.jsonl (T10: v4nr = v4 without R at W_0.001).
Writes  paper/tables/run2026_tl_sweep.tex (per arm), run2026_tl_sweep_bands.tex (per arm x band),
        run2026_tl_matched.tex (WCM vs unweighted at matched faithfulness), run2026_tl_noR.tex;
        out/run2026_tl_sweep*.csv, out/run2026_tl_matched_targets.csv;
        paper/figures/run2026_tl_sweep_curve.pdf/.png (pass rate and worst deviation vs median size per depth band,
        WCM and unweighted, production lambda 1e-3 marked)

Matched faithfulness, two readings (both per depth band):
  (a) latents needed for a pass rate: along each method's curve (median size vs pass rate over its lambdas, sorted
      by size), the size at which the pass rate first reaches the level, linear in log10(size) between neighbouring
      arms; "not reached" when no arm reaches it. Levels: 50%, 75% and the WCM production (W_0.001) rate of the band.
  (b) per target: the smallest circuit of each method that passes the rule, and (as merge.py's sweep summary) the
      smallest whose worst-of-3 deviation max|Z-1|,|A-1|,|C-1| <= 0.3; ratio unweighted / WCM where both exist.
Vacuous rows (merge.vacuous) are excluded per arm and never pass.

  python -X utf8 experiments/109-pod-run-analysis/t6_tl_sweep.py [--model turingllm] [--test]
  --test writes tables and figures to out/test/ instead of paper/ (for exercising the code on another model's sweep)
"""
import argparse
import re

import numpy as np
import pandas as pd

import lib109 as L

PROD_LAM = 1e-3
DEV_MAX = 0.3


def parse_arm(a):
    m = re.match(r"^([WU])_([0-9.eE+-]+)$", str(a))
    return (m.group(1) == "W", float(m.group(2))) if m else (None, None)


def load(model):
    sw = L.frame(L.read_rows(L.find(model, "out", "sweep"), "eval"), keys=("seed", "arm"))
    if sw.empty:
        return None
    sw = sw[sw.get("held", pd.Series("strong", index=sw.index)).fillna("strong") == "strong"]
    tg = L.targets_list(model, "sweep190") if model == "turingllm" else None
    tg = tg or sorted(sw.seed.unique())
    sw = sw[sw.seed.isin(tg)].copy()
    sw["source"] = "sweep"
    have = set(sw[sw.arm == "W_0.001"].seed)
    ev = L.main_eval(model)
    if len(ev):
        extra = ev[(ev.held == "strong") & ev.seed.isin(tg) & ~ev.seed.isin(have)].copy()
        if len(extra):
            extra["arm"], extra["source"] = "W_0.001", "main"
            sw = pd.concat([sw, extra[[c for c in extra.columns if c not in ("vac", "pass", "band", "layer", "kind")]]],
                           ignore_index=True)
    sw = L.score(L.add_site(sw, model))
    wl = sw.arm.map(parse_arm)
    sw["weighted"], sw["lam"] = [x[0] for x in wl], [x[1] for x in wl]
    sw = sw[sw.weighted.notna()]
    sw["dev"] = (sw[L.HEAD] - 1).abs().max(axis=1)
    return sw, tg


def per_arm(sw, by=None):
    out = []
    keys = ["weighted", "lam"] + ([by] if by else [])
    for k, g in sw.groupby(keys):
        ok = g[~g.vac]
        c = L.pass_cell(ok["pass"])
        out.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), method="WCM" if k[0] else "unweighted",
                        arm=("W_%g" if k[0] else "U_%g") % k[1], targets=int(g.seed.nunique()), vacuous=int(g.vac.sum()),
                        from_main=int((g.source == "main").sum()), pass_k=c["k"], pass_pct=c["pct"], pass_lo=c["lo"],
                        pass_hi=c["hi"], n_median=ok.n.median(), n_p25=ok.n.quantile(.25), n_p75=ok.n.quantile(.75),
                        dev_median=ok.dev.median(), **{"%s_median" % z: ok[col].median() for z, col in zip("ZAC", L.HEAD)},
                        induce_median=ok[L.IND].median(), _c=c))
    return out


def reach(curve, level):
    """(size, at_most) at which a (median size, pass %) curve first reaches level %, interpolated in log10(size).
    at_most = True when the curve's smallest arm already reaches the level (the size is then an upper bound)."""
    pts = sorted((r["n_median"], r["pass_pct"]) for r in curve if not np.isnan(r["n_median"]))
    prev = None
    for n, p in pts:
        if p >= level:
            if prev is None:
                return n, True
            (n0, p0) = prev
            t = (level - p0) / (p - p0)
            return 10 ** (np.log10(n0) + t * (np.log10(n) - np.log10(n0))), False
        prev = (n, p)
    return np.nan, False


def lam_tex(lam):
    """1e-05 -> 10^{-5}, 2.5e-05 -> 2.5 \\times 10^{-5} (math mode body)."""
    m, e = ("%e" % lam).split("e")
    m = float(m)
    return ("10^{%d}" % int(e)) if abs(m - 1) < 1e-9 else ("%g \\times 10^{%d}" % (m, int(e)))


def band_tex(b):
    return "all layers" if b == "all" else "layers " + b[1:].replace("-", "--")


def fragility(curve, level, label):
    """A pass-rate level is fragile for a method when it lies inside the 95% CI of the method's best arm (no arm is
    significantly above it). Returns a note, with whether the curve is monotone in size, or ''."""
    if not curve:
        return ""
    best = max(curve, key=lambda r: r["pass_pct"])
    if best["pass_lo"] > level or best["pass_pct"] < level:
        return ""
    pts = [r["pass_pct"] for r in sorted(curve, key=lambda r: r["n_median"])]
    mono = all(b >= a for a, b in zip(pts, pts[1:]))
    return "%s maximum %.1f%% [%.1f, %.1f] (at $\\lambda = %s$)%s" % (
        label, best["pass_pct"], best["pass_lo"], best["pass_hi"], lam_tex(best["lam"]),
        "" if mono else "; non-monotone curve")


def matched(sw, model):
    bands = [b for b, _, _ in L.bands_of(model)]
    rows, tgt = [], []
    by = per_arm(sw, "band")
    for b in bands + ["all"]:
        cur = per_arm(sw) if b == "all" else [r for r in by if r["band"] == b]
        w = [r for r in cur if r["weighted"]]
        u = [r for r in cur if not r["weighted"]]
        prod = [r for r in w if abs(r["lam"] - PROD_LAM) < 1e-12]
        levels = [("50%", 50.0, r"50\% pass rate"), ("75%", 75.0, r"75\% pass rate")] + (
            [("WCM production rate (%.1f%%)" % prod[0]["pass_pct"], prod[0]["pass_pct"],
              r"WCM's pass rate at $\lambda = 10^{-3}$ (%.1f\%%)" % prod[0]["pass_pct"])] if prod else [])
        for name, lev, tex in levels:
            (nw, bw), (nu, bu) = reach(w, lev), reach(u, lev)
            ratio = nu / nw if nw and not np.isnan(nw) and not np.isnan(nu) else np.nan
            frag = [fragility(cur_m, lev, lab) for cur_m, lab, n_ in ((w, "WCM", nw), (u, "unweighted", nu))
                    if not np.isnan(n_)]
            frag = [f for f in frag if f]
            rows.append(dict(band=b, reading="pass-rate level", level=name, level_tex=tex, wcm_size=nw, wcm_at_most=bw,
                             unweighted_size=nu, unweighted_at_most=bu, ratio=ratio, fragile=bool(frag),
                             fragile_note="; ".join(frag),
                             ratio_note="lower bound" if bw and not bu else ("upper bound" if bu and not bw else
                                                                              ("undetermined" if bu and bw else ""))))
        g = sw if b == "all" else sw[sw.band == b]
        per = []
        for s, x in g.groupby("seed"):
            def smallest(wt, crit):
                y = x[(x.weighted == wt) & ~x.vac & crit(x)]
                return y.n.min() if len(y) else np.nan
            per.append(dict(seed=s, band=x.band.iloc[0],
                            wcm_pass=smallest(True, lambda d: d["pass"]), unw_pass=smallest(False, lambda d: d["pass"]),
                            wcm_dev=smallest(True, lambda d: d.dev <= DEV_MAX),
                            unw_dev=smallest(False, lambda d: d.dev <= DEV_MAX)))
        per = pd.DataFrame(per)
        for crit, cw, cu, tex in (("smallest passing circuit", "wcm_pass", "unw_pass", "smallest passing circuit, per target"),
                                  ("smallest with worst deviation <= %.1f" % DEV_MAX, "wcm_dev", "unw_dev",
                                   r"smallest with Z, A, C within %.1f of 1, per target" % DEV_MAX)):
            both = per.dropna(subset=[cw, cu])
            ratio = both[cu] / both[cw]
            rows.append(dict(band=b, reading="per target", level=crit, level_tex=tex, targets=len(per), wcm_reaches=int(per[cw].notna().sum()),
                             unweighted_reaches=int(per[cu].notna().sum()), both=len(both),
                             wcm_size=both[cw].median(), unweighted_size=both[cu].median(), ratio=ratio.median(),
                             ratio_p25=ratio.quantile(.25), ratio_p75=ratio.quantile(.75)))
        if b == "all":
            tgt = per
    return rows, tgt


def figure(sw, model, figdir):
    plt, st = L.figure_style()
    bands = [b for b, _, _ in L.bands_of(model)]
    by = per_arm(sw, "band")
    # sized for a 5.5 in text width at 100%: text 7.5-8.5 pt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.titlepad": 4.0, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "lines.linewidth": 1.5,
                         "axes.linewidth": 0.8, "grid.linewidth": 0.5})
    fig, axes = plt.subplots(1, len(bands), figsize=(5.5, 2.35), sharey=True, squeeze=False)
    for j, b in enumerate(bands):
        ax = axes[0][j]
        for wt, col, name in ((True, st.BLUE, "WCM (fitted coefficients)"), (False, st.CATEGORICAL[1], "unweighted ($\\alpha = 1$)")):
            cur = sorted([r for r in by if r["band"] == b and r["weighted"] == wt], key=lambda r: r["n_median"])
            if not cur:
                continue
            x = [r["n_median"] for r in cur]
            ax.fill_between(x, [r["pass_lo"] for r in cur], [r["pass_hi"] for r in cur], color=st.tint(col, 0.78),
                            lw=0, zorder=2)
            ax.plot(x, [r["pass_pct"] for r in cur], marker="o", ms=3, color=col, label=name, zorder=3)
            for r in cur:
                if wt and abs(r["lam"] - PROD_LAM) < 1e-12:
                    ax.scatter([r["n_median"]], [r["pass_pct"]], s=70, marker="*", color=col, edgecolor=st.INK, lw=0.6,
                               zorder=4, label="production $\\lambda = 10^{-3}$" if j == 0 else None)
                    ax.annotate("$\\lambda = 10^{-3}$", (r["n_median"], r["pass_pct"]), xytext=(0, -11),
                                textcoords="offset points", ha="center", fontsize=7, color=st.INK)
        ax.set_title("layers %s" % b[1:].replace("-", "–"))
        ax.set_ylim(0, 100)
        ax.set_xscale("log")
        ax.set_xlabel("circuit size (median latents)")
    axes[0][0].set_ylabel("circuits that pass (%)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.4,
               columnspacing=1.2)
    L.save_with_legend(fig, figdir / ("%stl_sweep_curve.png" % L.PREFIX), top=0.86)


def nor(model, sw):
    rows = L.frame(L.read_rows(L.find(model, "nor", "sweep"), "eval"), keys=("seed", "arm"))
    if rows.empty:
        L.not_available("%s without-R arm (T10, out_v2_noR)" % model)
        return None
    r = L.score(L.add_site(rows, model)).set_index("seed")
    w = sw[(sw.arm == "W_0.001")].set_index("seed") if sw is not None else None
    if w is None or w.empty:
        ev = L.main_eval(model)
        w = ev[ev.held == "strong"].set_index("seed")
    both = r.index.intersection(w.index)
    m = w.loc[both, ["pass", "vac", "n", "band"]].join(r.loc[both, ["pass", "vac", "n"]], rsuffix="_noR")
    m = m[~(m.vac | m.vac_noR)]
    out = []
    for b in ["all"] + [x for x, _, _ in L.bands_of(model)]:
        g = m if b == "all" else m[m.band == b]
        pw, pn = L.pass_cell(g["pass"]), L.pass_cell(g["pass_noR"])
        lo, hi = L.boot_paired_diff(g["pass"].astype(float).values, g["pass_noR"].astype(float).values)
        out.append(dict(band=b, targets=len(g), pass_v4=pw["pct"], pass_v4_ci=L.fmt_pass(pw), pass_noR=pn["pct"],
                        pass_noR_ci=L.fmt_pass(pn), diff_pp=pw["pct"] - pn["pct"], diff_lo=100 * lo, diff_hi=100 * hi,
                        n_v4=g.n.median(), n_noR=g.n_noR.median(), _pw=pw, _pn=pn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turingllm")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        L.TABLES = L.FIGURES = L.OUT / "test"
        L.OUT = L.OUT / "test"
    p = load(a.model)
    if p is None:
        L.not_available("%s lambda sweep (T4, out_v2/sweep)" % a.model)
        n = nor(a.model, None)
    else:
        sw, tg = p
        arms = per_arm(sw)
        bands = per_arm(sw, "band")
        print("%s sweep: %d targets in the list, %d with rows; arms %s" % (
            a.model, len(tg), sw.seed.nunique(), ", ".join(r["arm"] for r in arms)))
        tex = []
        for r in sorted(arms, key=lambda r: (not r["weighted"], r["lam"])):
            print("  %-10s targets %3d vac %d (from main %d) pass %s n %.0f dev %.2f" % (
                r["arm"], r["targets"], r["vacuous"], r["from_main"], L.fmt_pass(r["_c"]), r["n_median"], r["dev_median"]))
            tex.append([r["method"], "%g%s" % (r["lam"], " (production)" if r["weighted"] and abs(r["lam"] - PROD_LAM) < 1e-12
                                                else ""), L.tex_int(r["targets"]), L.fmt_pass(r["_c"], True),
                        "%s (%s--%s)" % tuple(L.tex_int(r[k]) for k in ("n_median", "n_p25", "n_p75")),
                        "%s / %s / %s" % tuple(L.tex_num(r["%s_median" % z]) for z in "ZAC"), L.tex_num(r["dev_median"])])
        strip = lambda rs: pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rs])  # noqa: E731
        L.write_table("tl_sweep", tex, ["Method", r"$\lambda$", "Targets", r"Pass (\%) [95\% CI]", "Size, median (IQR)",
                                        "Z / A / C", "Worst dev."], "llrllll", csv_df=strip(arms))
        texb = []
        for r in sorted(bands, key=lambda r: (r["band"], not r["weighted"], r["lam"])):
            texb.append(["layers " + r["band"][1:], r["method"], "%g" % r["lam"], L.tex_int(r["targets"]),
                         L.fmt_pass(r["_c"], True), L.tex_int(r["n_median"])])
        L.write_table("tl_sweep_bands", texb, ["Depth", "Method", r"$\lambda$", "Targets", r"Pass (\%) [95\% CI]",
                                               "Size (median)"], "lllrll", csv_df=strip(bands))
        mrows, per = matched(sw, a.model)
        texm, foot = [], []
        le = lambda r, k: ("$\\le$" if r.get(k + "_at_most") else "") + L.tex_int(r.get(k + "_size"))  # noqa: E731
        for r in mrows:
            if r.get("fragile"):
                foot.append(r"\dag\ %s, %s: %s." % (band_tex(r["band"]), r["level_tex"],
                                                   r["fragile_note"].replace("%", r"\%")))
            rt = r.get("ratio")
            rtx = L.DASH if L._PTC.missing(rt) or r.get("ratio_note") == "undetermined" else (
                {"lower bound": "$\\ge$", "upper bound": "$\\le$"}.get(r.get("ratio_note"), "") +
                L.tex_num(rt, 1) + "$\\times$")
            print("  matched %-6s %-16s %-40s WCM %s unweighted %s ratio %s %s" % (
                r["band"], r["reading"], r["level"], le(r, "wcm"), le(r, "unweighted"), L.tex_num(rt, 1),
                r.get("ratio_note") or ""))
            texm.append([band_tex(r["band"]),
                         r["level_tex"] + ("\\textsuperscript{\\dag}" if r.get("fragile") else ""),
                         L.tex_int(r.get("both")) if r["reading"] == "per target" else L.DASH,
                         le(r, "wcm"), le(r, "unweighted"), rtx])
        if any(r.get("wcm_at_most") or r.get("unweighted_at_most") for r in mrows):
            foot.append(r"$\le$: the method's sparsest arm already reaches the level, so the size is an upper bound.")
        L.write_table("tl_matched", texm, ["Depth", "Matched on", "Targets (both)", "WCM size", "Unweighted size",
                                           "Ratio"], "llrrrr", csv_df=pd.DataFrame(mrows), footer=foot)
        per.to_csv(L.OUT / ("%stl_matched_targets.csv" % L.PREFIX), index=False)
        figure(sw, a.model, L.FIGURES)
        n = nor(a.model, sw)
    if n:
        for r in n:
            print("  noR %-6s n %3d v4 %s vs noR %s diff %+.1f pp [%+.1f, %+.1f] size %.0f vs %.0f" % (
                r["band"], r["targets"], r["pass_v4_ci"], r["pass_noR_ci"], r["diff_pp"], r["diff_lo"], r["diff_hi"],
                r["n_v4"], r["n_noR"]))
        L.write_table("tl_noR", [[r["band"] if r["band"] == "all" else "layers " + r["band"][1:], L.tex_int(r["targets"]),
                                  L.fmt_pass(r["_pw"], True), L.fmt_pass(r["_pn"], True),
                                  "%+.1f {[}%+.1f, %+.1f{]}" % (r["diff_pp"], r["diff_lo"], r["diff_hi"]),
                                  "%s / %s" % (L.tex_int(r["n_v4"]), L.tex_int(r["n_noR"]))] for r in n],
                      ["Depth", "Targets", r"Pass, v4 (\%)", r"Pass, without R (\%)", "Difference (pp)",
                       "Size v4 / without R"], "lrllll",
                      csv_df=pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in n]))


if __name__ == "__main__":
    main()
