"""116 report: error-ablated re-score (§1) and case-study checks (§2). CPU only.

§1. Paired per target: errors zeroed / mean-ablated vs errors live, all three scored on THIS GPU from the same circuit
and record (so hardware float noise does not enter the deltas). Pass rule and vacuous rule: 062/merge.py through 109's
lib109 (imported). Pass rates with 95% bootstrap intervals over targets (lib109.boot_ci); paired deltas with bootstrap
intervals of the mean (lib109.boot_paired_diff) and medians; by depth band and site kind. The sample is drawn
uniformly inside each band x kind cell (select_targets.py); the "weighted" rate re-weights cells to their production
sizes.

§2. The case-check rows (case_checks.py) as tables: named set vs rank-matched random sets vs the top contributors.

  python -X utf8 experiments/116-error-ablated-and-case-checks/report.py   -> out/report.md, out/paired_<family>.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "109-pod-run-analysis"))
import lib109 as L  # noqa: E402

OUT = HERE / "out"
SHORT = {"free0_tk": "Z", "freeM_topk_tk": "A", "freeN_topk_tk": "C", L.NEC: "nec"}
COLS = L.HEAD + [L.NEC]
FAMILIES = ["turingllm", "pythia-70m", "gpt2-small"]


def rows(family):
    p = OUT / ("rs_%s" % family) / "rows.jsonl"
    if not p.exists():
        return None
    df = pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()])
    df = df[df.get("error").isna()] if "error" in df else df
    df = df.drop_duplicates(["seed", "err_mode"], keep="last")
    keys = set((HERE / ("targets_%s.txt" % family)).read_text().split())
    df = df[df.seed.isin(keys)]                       # the smoke-test targets outside the list are dropped
    out = {}
    for m, d in df.groupby("err_mode"):
        d = L.score(L.add_site(d.reset_index(drop=True), family))
        out[m] = d.set_index("seed").sort_index()
    return out


def pct_ci(x):
    x = np.asarray(x, dtype=float)
    lo, hi = L.boot_ci(x)
    return "%.1f%% [%.1f, %.1f]" % (100 * x.mean(), 100 * lo, 100 * hi)


def weighted_rate(d, cells):
    w = cells.set_index(["band", "kind"]).n_prod
    g = d.groupby(["band", "kind"])["pass"].mean()
    g = g[g.index.isin(w.index)]
    return float((g * w.loc[g.index]).sum() / w.loc[g.index].sum())


def section1(lines, family):
    R = rows(family)
    if R is None or "live" not in R:
        lines.append("\n## %s: no rows\n" % family)
        return
    live = R["live"]
    lines.append("\n## §1 %s (%d targets scored live)\n" % (L.LABEL[family], len(live)))
    cells = pd.read_csv(OUT / ("cells_%s.csv" % family))
    # sanity: the local live re-score vs the production rows
    prod = L.main_eval(family)
    prod = prod[prod.held == "strong"].set_index("seed")
    common = live.index.intersection(prod.index)
    dif = (live.loc[common, COLS].astype(float) - prod.loc[common, COLS].astype(float)).abs().max(axis=1)
    agree = int((live.loc[common, "pass"] == prod.loc[common, "pass"]).sum())
    lines.append("Sanity (local live vs production rows, reference only): pass decision agrees on %d / %d; "
                 "max |diff| Z/A/C/nec per target median %.3f, p90 %.3f.\n" % (agree, len(common), dif.median(),
                                                                             dif.quantile(.9)))
    modes = [m for m in ("live", "zero", "mean") if m in R]
    paired = pd.DataFrame(index=live.index)
    paired["band"], paired["kind"] = live.band, live.kind
    for m in modes:
        d = R[m].reindex(live.index)
        paired["pass_" + m] = d["pass"].fillna(False).astype(bool)
        paired["vac_" + m] = d["vac"].fillna(True).astype(bool)
        for c in COLS + ["e0_tk", "eM_topk_tk", "eN_topk_tk", "free0_a1_tk", "phi_cf_alpha_blind_tk",
                         "free0_pre", "freeM_topk_pre", "freeN_topk_pre", "free0_raw_tk"]:
            paired["%s_%s" % (SHORT.get(c, c), m)] = pd.to_numeric(d[c], errors="coerce")
    paired.to_csv(OUT / ("paired_%s.csv" % family))
    lines.append("| errors | targets | vacuous | pass (95% CI) | cell-weighted pass | median Z / A / C / nec |")
    lines.append("|---|---|---|---|---|---|")
    for m in modes:
        d = R[m].reindex(live.index)
        dd = d.copy(); dd["pass"] = paired["pass_" + m]
        lines.append("| %s | %d | %d | %s | %.1f%% | %s |" % (
            m, int(d["seed"].notna().sum()) if "seed" in d else int(d.free0_tk.notna().sum()) or len(d),
            int(paired["vac_" + m].sum()), pct_ci(paired["pass_" + m]), 100 * weighted_rate(dd, cells),
            " / ".join("%.3f" % pd.to_numeric(d[c], errors="coerce").median() for c in COLS)))
    for m in [x for x in modes if x != "live"]:
        a, b = paired["pass_" + m].astype(float), paired["pass_live"].astype(float)
        lo, hi = L.boot_paired_diff(a.values, b.values)
        lines.append("\n**%s vs live** (paired, %d targets): pass %+.1f pp [%.1f, %.1f]; live pass -> fail %d, "
                     "fail -> pass %d." % (m, len(a), 100 * (a - b).mean(), 100 * lo, 100 * hi,
                                          int(((b == 1) & (a == 0)).sum()), int(((b == 0) & (a == 1)).sum())))
        zac = paired[["%s_%s" % (c, m) for c in ("Z", "A", "C")]]
        zac_l = paired[["%s_live" % c for c in ("Z", "A", "C")]]
        lines.append("Targets with any of Z/A/C above %.1f: %d live, %d %s; below %.2f: %d live, %d %s; "
                     "median ratio Z_%s / Z_live %.2f (IQR %.2f-%.2f)." % (
                         L.BAND[1], int((zac_l > L.BAND[1]).any(axis=1).sum()), int((zac > L.BAND[1]).any(axis=1).sum()),
                         m, L.BAND[0], int((zac_l < L.BAND[0]).any(axis=1).sum()),
                         int((zac < L.BAND[0]).any(axis=1).sum()), m, m,
                         (paired["Z_" + m] / paired["Z_live"]).median(),
                         (paired["Z_" + m] / paired["Z_live"]).quantile(.25),
                         (paired["Z_" + m] / paired["Z_live"]).quantile(.75)))
        cens =int(((paired["free0_raw_tk_" + m] <= 0) & (paired["free0_raw_tk_live"] > 0)).sum())
        lines.append("Circuit-alone target read censored to 0 under Z (Top-K): %d targets %s, %d live. "
                     "Pre-activation reads (free0_pre etc.) are the graded diagnostic below." % (
                         int((paired["free0_raw_tk_" + m] <= 0).sum()), m, int((paired["free0_raw_tk_live"] <= 0).sum())))
        lines.append("(%d targets censored only under %s.)" % (cens, m))
        lines.append("")
        lines.append("| score | median Δ | mean Δ (95% CI) | median abs Δ | p90 abs Δ |")
        lines.append("|---|---|---|---|---|")
        for c in ["Z", "A", "C", "nec", "e0_tk", "eM_topk_tk", "eN_topk_tk", "free0_a1_tk", "free0_pre",
                  "freeM_topk_pre", "freeN_topk_pre"]:
            x, y = paired["%s_%s" % (c, m)], paired["%s_live" % c]
            ok = x.notna() & y.notna()
            dlt = (x - y)[ok]
            if dlt.empty:
                continue
            lo, hi = L.boot_paired_diff(x[ok].values, y[ok].values)
            lines.append("| %s | %+.3f | %+.3f [%+.3f, %+.3f] | %.3f | %.3f |" % (
                c, dlt.median(), dlt.mean(), lo, hi, dlt.abs().median(), dlt.abs().quantile(.9)))
        for by in ("band", "kind"):
            lines.append("")
            lines.append("| %s | n | pass live | pass %s | Δ pass pp [95%% CI] | median ΔZ | ΔA | ΔC | Δnec |" % (by, m))
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for g, d in paired.groupby(by):
                a, b = d["pass_" + m].astype(float), d["pass_live"].astype(float)
                lo, hi = L.boot_paired_diff(a.values, b.values)
                lines.append("| %s | %d | %s | %s | %+.1f [%.1f, %.1f] | %s |" % (
                    g, len(d), pct_ci(b), pct_ci(a), 100 * (a - b).mean(), 100 * lo, 100 * hi,
                    " | ".join("%+.3f" % (d["%s_%s" % (c, m)] - d["%s_live" % c]).median() for c in ("Z", "A", "C", "nec"))))
        flips = paired[paired["pass_live"] & ~paired["pass_" + m]]
        if len(flips):
            def why(r):
                bad = []
                for c in ("Z", "A", "C"):
                    v = r["%s_%s" % (c, m)]
                    if pd.isna(v):
                        bad.append("%s_none x" % c)
                    elif v < L.BAND[0]:
                        bad.append("%s<%.2f x" % (c, L.BAND[0]))
                    elif v > L.BAND[1]:
                        bad.append("%s>%.1f x" % (c, L.BAND[1]))
                v = r["nec_%s" % m]
                if pd.isna(v) or v < L.NEC_MIN:
                    bad.append("nec<%.1f x" % L.NEC_MIN)
                if r["vac_" + m]:
                    bad.append("vacuous x")
                return ", ".join(bad)
            fail_by = flips.apply(why, axis=1)
            lines.append("\nlive pass -> %s fail, reasons (counts of the failing condition): %s" % (
                m, ", ".join("%s %d" % (k, v) for k, v in pd.Series(
                    [x.split(" ")[0] for s in fail_by for x in s.split(", ")]).value_counts().items())))


def section_diag(lines):
    """diag_recon.py: the target in the UNABLATED model when every upstream site's stream is replaced by its SAE
    reconstruction (errors zeroed) or reconstruction + mean error, all latents live; by site kind."""
    p = OUT / "diag_recon.jsonl"
    if not p.exists():
        return
    d = pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()]).drop_duplicates("seed", keep="last")
    keys = set((HERE / "targets_turingllm.txt").read_text().split())
    d = d[d.seed.isin(keys)].set_index("seed")
    d = L.add_site(d.reset_index(), "turingllm").set_index("seed")
    lines.append("\n## §1 diagnostic: TuringLLM run on SAE reconstructions, no circuit (%d targets)\n" % len(d))
    lines.append("Target activation / clean activation (held-out strongest, post-Top-K read), median [IQR]; columns = "
                 "which upstream sites have their error zeroed (zero) or set to the strongest-train mean (mean).\n")
    cols = ["zero_attn+mlp+resid_ratio", "mean_attn+mlp+resid_ratio", "zero_resid_ratio", "zero_mlp_ratio",
            "zero_attn_ratio"]
    lines.append("| kind | n | " + " | ".join(c.replace("_ratio", "") for c in cols) + " |")
    lines.append("|---|---|" + "---|" * len(cols))

    def cell(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return "%.2f [%.2f, %.2f]" % (x.median(), x.quantile(.25), x.quantile(.75)) if len(x) else "-"
    for g, dd in list(d.groupby("kind")) + [("all", d)]:
        lines.append("| %s | %d | %s |" % (g, len(dd), " | ".join(cell(dd[c]) for c in cols)))
    pp = OUT / "paired_turingllm.csv"
    if pp.exists():
        pr = pd.read_csv(pp, index_col=0)
        j = d.join(pr, how="inner", rsuffix="_p")
        r = j["zero_attn+mlp+resid_ratio"]
        lines.append("\nShare of targets whose own activation stays within [0.75, 1.5] of clean in the "
                     "reconstruction-only model: %.1f%%; among them, circuit pass with errors zeroed %.1f%% (live %.1f%%); "
                     "outside it: %.1f%% (live %.1f%%), n = %d / %d." % (
                         100 * r.between(.75, 1.5).mean(), 100 * j.loc[r.between(.75, 1.5), "pass_zero"].mean(),
                         100 * j.loc[r.between(.75, 1.5), "pass_live"].mean(),
                         100 * j.loc[~r.between(.75, 1.5), "pass_zero"].mean() if (~r.between(.75, 1.5)).any() else float("nan"),
                         100 * j.loc[~r.between(.75, 1.5), "pass_live"].mean() if (~r.between(.75, 1.5)).any() else float("nan"),
                         int(r.between(.75, 1.5).sum()), int((~r.between(.75, 1.5)).sum())))


def section_refit(lines):
    """refit_err.py pilot: circuits REFITTED with errors live / zeroed in training, each scored live and zeroed."""
    R = {}
    for fe in ("live", "zero"):
        p = OUT / ("refit_%s" % fe) / "rows.jsonl"
        if p.exists():
            df = pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()])
            df = df[df.get("error").isna()] if "error" in df else df
            df = df[df.err_mode.isin(["live", "zero"])].drop_duplicates(["seed", "err_mode"], keep="last")
            for m, d in df.groupby("err_mode"):
                R[(fe, m)] = L.score(L.add_site(d.reset_index(drop=True), "turingllm")).set_index("seed")
    if not R:
        return
    keys = sorted(set.intersection(*[set(d.index) for d in R.values()]))
    lines.append("\n## §1b refit pilot (TuringLLM, %d targets with every cell)\n" % len(keys))
    lines.append("| fitted with errors | scored with errors | pass (95% CI) | median Z / A / C / nec | median nodes |")
    lines.append("|---|---|---|---|---|")
    for (fe, m), d in sorted(R.items()):
        d = d.loc[keys]
        lines.append("| %s | %s | %s | %s | %d |" % (fe, m, pct_ci(d["pass"]),
                                                   " / ".join("%.3f" % d[c].median() for c in COLS),
                                                   int(d["n"].median())))
    pp = OUT / "paired_turingllm.csv"
    if pp.exists():
        pr = pd.read_csv(pp, index_col=0).reindex(keys)
        lines.append("\nSame targets, production circuits (§1): pass live %s, zero %s." % (
            pct_ci(pr.pass_live.astype(float)), pct_ci(pr.pass_zero.astype(float))))
    if ("zero", "zero") in R:
        d = R[("zero", "zero")].loc[keys]
        lines.append("\nBy kind, fitted and scored with errors zeroed: " + ", ".join(
            "%s %d/%d" % (k, int(g["pass"].sum()), len(g)) for k, g in d.groupby("kind")))


def section2(lines):
    p = OUT / "case" / "rows.jsonl"
    if not p.exists():
        lines.append("\n## §2: no case rows\n")
        return
    df = pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()])
    df = df[df.get("error").isna()] if "error" in df else df
    df = df.drop_duplicates(["key", "set", "variant", "draw"], keep="last")
    lines.append("\n## §2 case-study checks\n")
    for key, d in df.groupby("key"):
        full = d[d.set == "full"].iloc[0]
        lines.append("\n### %s (%d members): full circuit Z / A / C / nec %.2f / %.2f / %.2f / %.2f, a_pos %.2f\n" % (
            key, full.n_set, full.free0_tk, full.freeM_topk_tk, full.freeN_topk_tk, full.phi_sup_blind_tk,
            full.a_pos_tk))
        lines.append("| set | n | variant | Z | A | C | nec | KO drop (unmodified model) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in [x for x in d.set.unique() if x != "full"]:
            ds = d[d.set == s]
            for v in ("named_prod", "named_refit", "top_prod", "top_refit", "random_prod", "random_refit"):
                dv = ds[ds.variant == v]
                if dv.empty:
                    continue
                def f(c):
                    x = pd.to_numeric(dv[c], errors="coerce")
                    if len(dv) == 1:
                        return "-" if x.isna().all() else "%.2f" % x.iloc[0]
                    return "%.2f (%.2f-%.2f)" % (x.mean(), x.min(), x.max())
                lines.append("| %s | %d | %s%s | %s | %s | %s | %s | %s |" % (
                    s, int(dv.n_set.iloc[0]), v, "" if len(dv) == 1 else " x%d" % len(dv),
                    f("free0_tk"), f("freeM_topk_tk"), f("freeN_topk_tk"), f("phi_sup_blind_tk"), f("ko_drop")))


def main():
    lines = ["# 116 report (generated by report.py)", ""]
    for fam in FAMILIES:
        section1(lines, fam)
    section_diag(lines)
    section_refit(lines)
    section2(lines)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
