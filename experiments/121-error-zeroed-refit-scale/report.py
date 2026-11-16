"""121 report: error-zeroed refit at scale. CPU only.

Per target (all scored on this GPU, held-out strongest, the production scorer, pass / vacuous rule from 062 merge.py
through 109 lib109):
  (a) refit with every upstream SAE error zeroed in training, scored with errors zeroed      [121 refit_<family>, or
      116 out/refit_zero for the 18 TuringLLM pilot targets: the same script, code and GPU]
  (b) the production circuit (fitted errors live on the pod), scored with errors live         [116 rs_<family> live]
  (c) arm (a)'s circuit scored with errors live                                                [as (a)]
  ref: the production circuit scored with errors zeroed (116 §1, no refit)                    [116 rs_<family> zero]
  pilot only: 116's errors-live refit on this GPU, scored live                                 [116 out/refit_live]
Pass rates with 95% bootstrap CIs over targets (lib109.boot_ci), paired differences (lib109.boot_paired_diff), failure
modes, sizes (latents), by depth band and site kind. TuringLLM also: the error-free full model (116 diag_recon).

  python -X utf8 experiments/121-error-zeroed-refit-scale/report.py   -> out/report.md, out/paired_<family>.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E116 = ROOT / "experiments" / "116-error-ablated-and-case-checks"
sys.path.insert(0, str(ROOT / "experiments" / "109-pod-run-analysis"))
import lib109 as L  # noqa: E402

OUT = HERE / "out"
COLS = L.HEAD + [L.NEC]
SH = dict(zip(COLS, ["Z", "A", "C", "nec"]))
FAMILIES = ["turingllm", "pythia-70m", "gpt2-small"]


def load(p, family, modes=("live", "zero")):
    if not p.exists():
        return {}
    df = pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()])
    if "error" in df:
        df = df[df["error"].isna()]
    df = df[df.err_mode.isin(modes)].drop_duplicates(["seed", "err_mode"], keep="last")
    return {m: L.score(L.add_site(d.reset_index(drop=True), family)).set_index("seed") for m, d in df.groupby("err_mode")}


def pct_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return "-"
    lo, hi = L.boot_ci(x)
    return "%.1f%% [%.1f, %.1f]" % (100 * x.mean(), 100 * lo, 100 * hi)


def diff_ci(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0:
        return "-"
    lo, hi = L.boot_paired_diff(a, b)
    return "%+.1f [%.1f, %.1f]" % (100 * (a - b).mean(), 100 * lo, 100 * hi)


def why(r, p):
    """Failure mode of arm p for one row: vacuous / faithfulness (any of Z, A, C outside the band or missing) /
    necessity only (Z, A, C inside, nec < NEC_MIN)."""
    if r["pass_" + p]:
        return "pass"
    if r["vac_" + p]:
        return "vacuous"
    zac = [r["%s_%s" % (c, p)] for c in ("Z", "A", "C")]
    if any(pd.isna(v) or v < L.BAND[0] or v > L.BAND[1] for v in zac):
        return "faithfulness"
    return "necessity only"


def paired(family):
    t = pd.read_csv(HERE / "out" / ("targets_%s.csv" % family))
    rs = load(E116 / "out" / ("rs_%s" % family) / "rows.jsonl", family)
    a = load(OUT / ("refit_%s" % family) / "rows.jsonl", family)
    if family == "turingllm":
        pz = load(E116 / "out" / "refit_zero" / "rows.jsonl", family)
        for m in pz:
            a[m] = pd.concat([a.get(m, pd.DataFrame()), pz[m][~pz[m].index.isin(a.get(m, pd.DataFrame()).index)]])
    if "zero" not in a:
        return None
    keys = [s for s in t.seed if s in a["zero"].index and "live" in a and s in a["live"].index]
    P = t.set_index("seed").loc[keys, ["band", "kind", "source"]].copy()
    src = {"a": (a, "zero"), "c": (a, "live"), "b": (rs, "live"), "fz": (rs, "zero")}
    if family == "turingllm":
        rl = load(E116 / "out" / "refit_live" / "rows.jsonl", family, ("live",))
        src["rl"] = (rl, "live")
    for p, (R, m) in src.items():
        if m not in R:
            continue
        d = R[m].reindex(keys)
        P["pass_" + p] = d["pass"].fillna(False).astype(bool) if p != "rl" else d["pass"]
        P["vac_" + p] = d["vac"].fillna(True).astype(bool) if p != "rl" else d["vac"]
        for c in COLS:
            P["%s_%s" % (SH[c], p)] = pd.to_numeric(d[c], errors="coerce")
        P["n_" + p] = pd.to_numeric(d["n"], errors="coerce")
    for p in ("a", "b"):
        P["why_" + p] = P.apply(lambda r: why(r, p), axis=1)
    if family == "turingllm":
        dg = pd.DataFrame([json.loads(x) for x in (E116 / "out" / "diag_recon.jsonl").read_text().splitlines()
                           if x.strip()]).drop_duplicates("seed", keep="last").set_index("seed")
        P["recon_ratio"] = pd.to_numeric(dg["zero_attn+mlp+resid_ratio"].reindex(keys), errors="coerce")
    P.to_csv(OUT / ("paired_%s.csv" % family))
    return P


def med(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return "%.0f" % x.median() if len(x) else "-"


def section(lines, family):
    P = paired(family)
    if P is None or P.empty:
        lines.append("\n## %s: no refit rows yet\n" % L.LABEL[family])
        return
    n = len(P)
    tot = len(pd.read_csv(HERE / "out" / ("targets_%s.csv" % family)))
    lines.append("\n## %s (%d of %d planned targets)\n" % (L.LABEL[family], n, tot))
    lines.append("| arm | circuit | scored with errors | pass (95% CI) | median Z / A / C / nec | median latents |")
    lines.append("|---|---|---|---|---|---|")
    arms = [("a", "refit, errors zeroed", "zeroed"), ("b", "production (errors live)", "live"),
            ("c", "refit, errors zeroed", "live"), ("fz", "production (errors live)", "zeroed")]
    for p, circ, sc in arms:
        if "pass_" + p not in P:
            continue
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            {"fz": "ref"}.get(p, p), circ, sc, pct_ci(P["pass_" + p]),
            " / ".join("%.2f" % P["%s_%s" % (c, p)].median() for c in ("Z", "A", "C", "nec")), med(P["n_" + p])))
    lines.append("\nPaired, (a) − (b): %s pp; (a) − ref (production circuit, errors zeroed, no refit): %s pp; "
                 "(c) − (b): %s pp." % (diff_ci(P.pass_a, P.pass_b), diff_ci(P.pass_a, P.pass_fz),
                                        diff_ci(P.pass_c, P.pass_b)))
    ab = pd.crosstab(P.pass_b.map({True: "(b) pass", False: "(b) fail"}),
                     P.pass_a.map({True: "(a) pass", False: "(a) fail"}))
    lines.append("Pass agreement (b) x (a): " + "; ".join("%s & %s %d" % (i, j, ab.loc[i, j])
                                                          for i in ab.index for j in ab.columns) + ".")
    ratio = (P.n_a / P.n_b).replace([np.inf, -np.inf], np.nan).dropna()
    lines.append("Size: median latents (a) %s vs (b) %s; per-target ratio (a)/(b) median %.2f [IQR %.2f-%.2f]." % (
        med(P.n_a), med(P.n_b), ratio.median(), ratio.quantile(.25), ratio.quantile(.75)))
    for p in ("a", "b"):
        vc = P["why_" + p].value_counts()
        lines.append("Outcome %s: %s." % (p, ", ".join("%s %d" % (k, v) for k, v in vc.items())))
    fa = P[P.why_a == "faithfulness"]
    if len(fa):
        low = {c: int((fa["%s_a" % c] < L.BAND[0]).sum() + fa["%s_a" % c].isna().sum()) for c in ("Z", "A", "C")}
        high = {c: int((fa["%s_a" % c] > L.BAND[1]).sum()) for c in ("Z", "A", "C")}
        lines.append("(a) faithfulness failures, by score (a target can count more than once): below %.2f %s; "
                     "above %.1f %s." % (L.BAND[0], ", ".join("%s %d" % kv for kv in low.items()), L.BAND[1],
                                         ", ".join("%s %d" % kv for kv in high.items())))
    nec = P[P.why_a == "necessity only"]
    if len(nec):
        lines.append("(a) necessity-only failures: nec median %.2f (range %.2f-%.2f)." % (
            nec.nec_a.median(), nec.nec_a.min(), nec.nec_a.max()))
    try:
        cells = pd.read_csv(E116 / "out" / ("cells_%s.csv" % family)).set_index(["band", "kind"]).n_prod
        g = P.groupby(["band", "kind"])[["pass_a", "pass_b"]].mean()
        w = cells.loc[g.index]
        lines.append("Cell-weighted (production cell sizes): (a) %.1f%%, (b) %.1f%%." % (
            100 * (g.pass_a * w).sum() / w.sum(), 100 * (g.pass_b * w).sum() / w.sum()))
    except Exception:  # noqa: BLE001
        pass
    for by in ("band", "kind"):
        lines.append("")
        lines.append("| %s | n | (a) refit zeroed / zeroed | (b) production / live | (a) − (b) pp | ref: production / "
                     "zeroed | (c) refit zeroed / live | (a) nec-only fails | median latents (a) / (b) |" % by)
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for gk, d in P.groupby(by):
            lines.append("| %s | %d | %s | %s | %s | %s | %s | %d | %s / %s |" % (
                gk, len(d), pct_ci(d.pass_a), pct_ci(d.pass_b), diff_ci(d.pass_a, d.pass_b), pct_ci(d.pass_fz),
                pct_ci(d.pass_c), int((d.why_a == "necessity only").sum()), med(d.n_a), med(d.n_b)))
    lines.append("\nCells, pass (a) / (b) / ref: " + "; ".join(
        "%s %s %d/%d/%d of %d" % (b, k, d.pass_a.sum(), d.pass_b.sum(), d.pass_fz.sum(), len(d))
        for (b, k), d in P.groupby(["band", "kind"])) + ".")
    if "pass_rl" in P:
        pl = P[P.pass_rl.notna()]
        if len(pl):
            lines.append("\nPilot targets (%d) with 116's errors-live REFIT on this GPU: refit live / live %s, "
                         "(a) %s, (b) %s; median latents refit-live %s." % (
                             len(pl), pct_ci(pl.pass_rl.astype(float)), pct_ci(pl.pass_a), pct_ci(pl.pass_b),
                             med(pl.n_rl)))
    if "recon_ratio" in P:
        r = P.recon_ratio
        ok = r.between(.75, 1.5)
        lines.append("\n**Error-free full model** (116 diag_recon: every upstream site replaced by its SAE "
                     "reconstruction, all latents live; the target's activation / clean). Within [0.75, 1.5]: "
                     "%d / %d targets (%.1f%%). Among them: (a) %s, (b) %s, ref %s. Outside: (a) %s, (b) %s "
                     "(n = %d)." % (int(ok.sum()), int(r.notna().sum()), 100 * ok.mean(), pct_ci(P.pass_a[ok]),
                                    pct_ci(P.pass_b[ok]), pct_ci(P.pass_fz[ok]), pct_ci(P.pass_a[~ok]),
                                    pct_ci(P.pass_b[~ok]), int((~ok).sum())))
        lines.append("Within, paired (a) − (b): %s pp. By kind, within: %s." % (
            diff_ci(P.pass_a[ok], P.pass_b[ok]), ", ".join(
                "%s %d/%d (a) vs %d/%d (b)" % (k, d.pass_a.sum(), len(d), d.pass_b.sum(), len(d))
                for k, d in P[ok].groupby("kind"))))
        lines.append("By kind, share within: " + ", ".join(
            "%s %d/%d" % (k, int(d.recon_ratio.between(.75, 1.5).sum()), len(d)) for k, d in P.groupby("kind")) + ".")


def main():
    lines = ["# 121 report (generated by report.py)", "",
             "Pass = DAN-8 rule (Z, A, C in [%.2f, %.1f], nec >= %.1f; vacuous = fail). 95%% bootstrap CIs over targets."
             % (L.BAND[0], L.BAND[1], L.NEC_MIN)]
    for fam in FAMILIES:
        section(lines, fam)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
