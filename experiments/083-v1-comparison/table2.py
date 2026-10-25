"""C4 (DAN-77 / DAN-159): faithfulness against circuit size for every method (the primary output), and Table 2 as its
summary, from row files only.

Every row is a held-out-strongest score from the production v1 scorer (driver.Runner.score). The DAN-8 rule: Z, A, C
(free0_tk, freeM_topk_tk, freeN_topk_tk) all in [0.8, 1.5] and necessity (phi_sup_blind_tk) >= 0.9; a missing score
fails; vacuous denominators are excluded from the denominator and counted (062/merge.py). Depth bands L0-4 / 5-7 / 8-11.

FIGURE (dan77-faithfulness-vs-size): every method at its genuine sizes. WCM at all eight prices (2.5e-5 .. 4e-3) and
unweighted at all five (1e-5 .. 1e-3), from the reference rows; path attribution (IG) and attribution patching (AtP)
truncated at the target's WCM size at every WCM price (one line per ranking), and at their natural size; Ge's tau
bisection points (binned by size); DAN-43 at the production per-site counts. Pass rate (top) and median A (bottom).
TABLE 2 (summary): each method at its natural operating point (WCM and unweighted at the SAME price, lambda 1e-3;
attribution at its natural position-union size; Ge at its bisected size), then the attribution truncations at the
W_0.001 size and DAN-43.

Reference rows (WCM / unweighted): REF = auto (default: the same-machine re-scores of rescore_ref.py where present,
else the 062 H100 sweep rows), local (re-scores only; the default when CTX_DIR is set, because the rebuilt held-out
sets need re-scored references), or 062. 083 rows are read from OUT083 (attr / ge / fitted / ref).
Outputs (OUT083): dan77_points.csv, dan77-faithfulness-vs-size.{pdf,png}, table2.md, table2.csv, table2_rows.csv
(every per-target row used, with its source file:line), table2.tex (booktabs body).

  PYTHONPATH=src python experiments/083-v1-comparison/table2.py
  env: REF (auto | local | 062)  COMMON=1 (only targets present in every Table 2 row)  FIG_RANKINGS (max,sfc)
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cmp_common as C  # noqa: E402

REF = os.environ.get("REF", "local" if C.CTX_DIR else "auto")
COMMON = os.environ.get("COMMON") == "1"
FIG_RANKINGS = [r for r in os.environ.get("FIG_RANKINGS", "max,sfc").split(",") if r]
PROD = C.REF_ARM
U_PROD = "U_0.001"                                          # unweighted at the same price (Daniel, 2026-10-24)
METHOD_LABEL = {"abl_ig": "path attribution (IG)", "atp": "attribution patching (AtP)"}
RANK_LABEL = {"max": "top-n by max |attr| over contexts and positions", "agg": "top-n by mean |attr|",
              "sfc": "top-n by the aggregate ranking of prior work (SFC: |signed sum|)"}
SECTION_NAT, SECTION_N = "natural operating point", "at the WCM size (λ 1e-3)"
SECTION_X = "ground truth (exact activation patching)"
SOURCE_LABEL = {"abl_ig": "IG-ranked", "atp": "AtP-ranked", "act": "activation-ranked"}
RANK_SHORT = {"max": "max", "agg": "mean |attr|", "sfc": "SFC aggregate", "mean": "mean anchor activation",
              "coact": "co-activation with the target"}


# ---------------------------------------------------------------------------------------------- loading
def frame(rows, label, **extra):
    """Per-target rows -> a frame with the columns the table and figure read."""
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["row"] = label
    for k, v in extra.items():
        d[k] = v
    d["layer"] = d.seed.str.split(".").str[0].astype(int)
    d["kind"] = d.seed.str.split(".").str[1]
    d["depth"] = d.layer.map(C.depth_band)
    d["vacuous"] = [C.vacuous(r) for r in rows]
    d["passes"] = [C.passes(r) and not C.vacuous(r) for r in rows]
    for c in C.HEAD + [C.NEC, C.IND, "source"]:
        if c not in d:
            d[c] = np.nan
    return d


def ours(sub):
    return C.ok_rows(C.read_jsonl(C.OUT / sub / "rows.*.jsonl"))


def reference():
    """WCM / unweighted rows per REF (cmp_common.merge_reference)."""
    local = C.last_by(ours("ref"), ("seed", "arm"))
    return C.merge_reference(C.sweep_rows(), local, REF)


def source_files(f, sep=", "):
    """The row files behind a frame, shard files folded to one glob (each row's file:line is in table2_rows.csv)."""
    return sep.join(sorted({re.sub(r"\.shard\d+\.", ".shard*.", str(s).rsplit(":", 1)[0]) for s in f.source}))


def table_rows(ref):
    """[(section, label, frame)] in table order."""
    out = []
    by = {(r["seed"], r["arm"]): r for r in ref}
    seeds = sorted({r["seed"] for r in ref if r["arm"] == PROD})
    w = [by[(s, PROD)] for s in seeds]
    u = [by[(s, U_PROD)] for s in seeds if (s, U_PROD) in by]
    out.append((SECTION_NAT, "WCM, λ 1e-3 (production)", frame(w, "WCM, λ 1e-3 (production)", method="wcm")))
    out.append((SECTION_NAT, "unweighted, λ 1e-3", frame(u, "unweighted, λ 1e-3", method="unweighted")))
    attr = C.last_by(ours("attr"), ("seed", "method", "ranking", "size_arm"))
    for m in METHOD_LABEL:
        rr = [r for r in attr if r["method"] == m and r["size_arm"] == "natural"]
        lab = "%s, natural size (position union)" % METHOD_LABEL[m]
        out.append((SECTION_NAT, lab, frame(rr, lab, method=m, ranking="union")))
    ge = [r for r in C.last_by(ours("ge"), ("seed", "point")) if r.get("final")]
    lab = "hierarchical attribution (Ge, adapted), τ bisected to the WCM size"
    out.append((SECTION_NAT, lab, frame(ge, lab, method="ge_hier", ranking="bisect")))
    for m in METHOD_LABEL:
        for rk in RANK_LABEL:
            rr = [r for r in attr if r["method"] == m and r["ranking"] == rk and r["size_arm"] == PROD]
            lab = "%s, %s" % (METHOD_LABEL[m], RANK_LABEL[rk])
            out.append((SECTION_N, lab, frame(rr, lab, method=m, ranking=rk)))
    fit = C.last_by(ours("fitted"), ("seed", "method", "ranking", "variant"))
    for m in sorted({r["method"] for r in fit}):
        for rk in sorted({r["ranking"] for r in fit if r["method"] == m}):
            for v, vlab in (("a1", "per-site counts, α = 1"), ("fit", "per-site counts, fitted coefficients (DAN-43)")):
                rr = [r for r in fit if r["method"] == m and r["ranking"] == rk and r["variant"] == v]
                src = SOURCE_LABEL.get(m.replace("fitted_", ""), m)
                lab = "%s members (%s), %s" % (src, RANK_SHORT.get(rk, rk), vlab.replace(" (DAN-43)", "" if m == "fitted_act" else " (DAN-43)"))
                out.append((SECTION_N, lab, frame(rr, lab, method=m, ranking=rk)))
    ex = C.last_by(ours("exact"), ("seed", "variant"))
    for v, vlab in (("a1", "α = 1"), ("fit", "fitted coefficients")):
        rr = [r for r in ex if r["variant"] == v]
        lab = "exact-patching-ranked members (subset of %d targets), %s" % (len({r["seed"] for r in rr}), vlab)
        out.append((SECTION_X, lab, frame(rr, lab, method="exact_patch", ranking="exact")))
    if REF != "local":                                       # scored on the as-run held-out sets
        rnd = C.ok_rows(C.read_jsonl(C.E062 / "out_random" / "random.jsonl"))
        lab = "random members, fitted coefficients (DAN-17 pilot, 63 targets × 2 draws)"
        out.append((SECTION_N, lab, frame(rnd, lab, method="random")))
    return [(sec, lab, f) for sec, lab, f in out if not f.empty]


# ---------------------------------------------------------------------------------------------- table
def summarise(f, wcm_n):
    ok = f[~f.vacuous]
    ratio = (f.set_index("seed").n / wcm_n.reindex(f.seed).values).median() if len(wcm_n) else np.nan
    row = dict(rows=len(f), targets=f.seed.nunique(), vacuous=int(f.vacuous.sum()), n_median=f.n.median(),
               n_over_wcm=ratio)
    for c, name in zip(C.HEAD + [C.IND, C.NEC], ("Z", "A", "C", "induce", "necessity")):
        row[name] = pd.to_numeric(ok[c], errors="coerce").median()
    row["worst_dev"] = (ok[C.HEAD].astype(float) - 1).abs().max(axis=1).median()
    row["pass"] = int(ok.passes.sum())
    row["N"] = len(ok)
    for b, _, _ in C.DEPTH_BANDS:
        g = ok[ok.depth == b]
        row["pass_%s" % b] = "%d/%d" % (int(g.passes.sum()), len(g))
    row["sources"] = source_files(f)
    return row


def build_table(rows):
    if COMMON:
        common = set.intersection(*[set(f.seed) for _, _, f in rows if f.method.iloc[0] != "random"])
        rows = [(sec, lab, f[f.seed.isin(common)]) for sec, lab, f in rows]
    wcm = next((f for _, lab, f in rows if lab.startswith("WCM")), pd.DataFrame())
    wcm_n = wcm.set_index("seed").n if len(wcm) else pd.Series(dtype=float)
    t = pd.DataFrame([dict(section=sec, row=lab, **summarise(f, wcm_n)) for sec, lab, f in rows if len(f)])
    return t, rows


def fmt(v, p=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else ("%.*f" % (p, v))


def size_matching_note(ref):
    """Secondary note (D10, not the comparison): the unweighted arm nearest each target's WCM size."""
    sizes = C.sizes_by_arm(ref)
    picks = [(C.nearest_size(v.get(PROD), {k: x for k, x in v.items() if k.startswith("U_")}), v) for v in sizes.values()]
    picks = [(a, v) for a, v in picks if a]
    if not picks:
        return ""
    share = np.mean([a == U_PROD for a, _ in picks])
    ratio = np.median([v[a] / v[PROD] for a, v in picks])
    return ("Size matching (secondary): the unweighted arm nearest the WCM λ 1e-3 size is %s for %.0f%% of %d "
            "targets, at a median %.2f× the WCM size; no unweighted price on the grid is as small as WCM."
            % (U_PROD, 100 * share, len(picks), ratio))


def write_table(t, rows, note, ctx_dirs):
    C.OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(C.OUT / "table2.csv", index=False)
    pd.concat([f.assign(section=sec) for sec, _, f in rows], ignore_index=True).to_csv(C.OUT / "table2_rows.csv",
                                                                                        index=False)
    lines = ["# Table 2 (protocol v1): summary of the faithfulness-vs-size curves, held-out strongest", "",
             "DAN-8 pass rule (Z, A, C in [0.8, 1.5] and necessity >= 0.9); vacuous denominators excluded and counted. "
             "Medians over non-vacuous rows; size and size ratio over all rows. REF = %s%s. 083 rows scored on: %s."
             % (REF, ", COMMON targets" if COMMON else "", ", ".join(sorted(ctx_dirs)) or "—"), "",
             "| section | row | targets | vac | n (med) | n / WCM | Z | A | C | φ_ind | φ_sup | pass | L0-4 | L5-7 | L8-11 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    recs = t.to_dict("records")                      # (itertuples would rename "pass" and "pass_L0-4")
    for r in recs:
        lines.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %d/%d | %s | %s | %s |" % (
            r["section"], r["row"], r["targets"], r["vacuous"], fmt(r["n_median"], 0), fmt(r["n_over_wcm"]),
            fmt(r["Z"]), fmt(r["A"]), fmt(r["C"]), fmt(r["induce"]), fmt(r["necessity"]), r["pass"], r["N"],
            r["pass_L0-4"], r["pass_L5-7"], r["pass_L8-11"]))
    if len(ctx_dirs) > 1:
        lines += ["", "**WARNING**: 083 rows from more than one context set in this OUT083; use one OUT083 per CTX_DIR."]
    lines += ["", note, "", "## Sources (every per-target row with its file:line is in table2_rows.csv)", ""]
    lines += ["- %s: %s" % (r["row"], r["sources"]) for r in recs]
    (C.OUT / "table2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = ["%% generated by experiments/083-v1-comparison/table2.py from %s/table2_rows.csv" % C.OUT.name]
    for r in recs:
        tex.append("%s & %s & %s & %s & %s & %s & %s & %d/%d \\\\" % (
            r["row"].replace("λ", "$\\lambda$").replace("τ", "$\\tau$").replace("α", "$\\alpha$")
            .replace("φ", "$\\phi$").replace("|", "$|$").replace("×", "$\\times$"),
            "{:,.0f}".format(r["n_median"]).replace(",", "{,}"), fmt(r["Z"]), fmt(r["A"]), fmt(r["C"]),
            fmt(r["induce"]), fmt(r["necessity"]), r["pass"], r["N"]))
    (C.OUT / "table2.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(lines[:len(recs) + 8]))


# ---------------------------------------------------------------------------------------------- DAN-77 figure
def curve_points(ref):
    """One point per (series, size arm, depth band): median size, pass rate (non-vacuous), median A."""
    pts = []

    def add(series, size_arm, f, kind):
        for b, _, _ in C.DEPTH_BANDS + (("all", 0, 99),):
            g = f if b == "all" else f[f.depth == b]
            ok = g[~g.vacuous]
            if not len(ok):
                continue
            pts.append(dict(series=series, kind=kind, size_arm=size_arm, band=b, targets=len(ok), n_median=g.n.median(),
                            pass_rate=ok.passes.mean(), A_median=pd.to_numeric(ok[C.HEAD[1]], errors="coerce").median(),
                            sources=source_files(g, sep=" ")))

    sw = frame(ref, "reference")
    for arm, g in sw.groupby("arm"):
        add("WCM" if arm.startswith("W_") else "unweighted", arm, g, "sweep")
    fa = frame(C.last_by(ours("attr"), ("seed", "method", "ranking", "size_arm")), "attr")
    if len(fa):
        for (m, rk, sz), g in fa.groupby(["method", "ranking", "size_arm"]):
            add("%s/%s" % (m, rk), sz, g, "natural" if sz == "natural" else "attr")
    ge = frame(C.last_by(ours("ge"), ("seed", "point")), "ge")
    if len(ge):
        ge = ge[ge.n.notna()]
        ge["bin"] = pd.cut(np.log10(ge.n.astype(float).clip(lower=1)), np.arange(0.0, 6.01, 0.5), include_lowest=True)
        for b, g in ge.groupby("bin", observed=True):
            add("ge_hier", "log10 n in %s" % b, g, "ge")
    ff = frame(C.last_by(ours("fitted"), ("seed", "method", "ranking", "variant"))
               + [r for r in C.last_by(ours("exact"), ("seed", "variant")) if r["variant"] in ("a1", "fit")], "fitted")
    if len(ff):
        for (m, rk, v), g in ff.groupby(["method", "ranking", "variant"]):
            add("%s/%s/%s" % (m, rk, v), PROD, g, "fitted")
    return pd.DataFrame(pts)


def figure(pts):
    from analysis.style import BLUE, CATEGORICAL, INK_MUTED, configure_matplotlib, save_figure, style_suptitle, tint
    if pts.empty:
        return None
    plt = configure_matplotlib()
    colors = {"WCM": BLUE, "unweighted": "#9aa0a8", "abl_ig": CATEGORICAL[2], "atp": CATEGORICAL[3],
              "ge_hier": CATEGORICAL[6], "fitted_abl_ig": CATEGORICAL[5], "fitted_atp": CATEGORICAL[7],
              "fitted_act": CATEGORICAL[4], "exact_patch": "#20242b"}
    names = {"abl_ig": "IG", "atp": "AtP", "ge_hier": "Ge (τ bisection points)"}
    dashes = {"max": "-", "agg": ":", "sfc": "--"}
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.8), sharex=True, sharey="row")
    for j, (b, lo, hi) in enumerate(C.DEPTH_BANDS):
        P = pts[pts.band == b]
        for i, (col, ylab) in enumerate((("pass_rate", "targets that pass (%)"), ("A_median", "A (activating mean), median"))):
            ax = axes[i, j]
            scale = 100 if col == "pass_rate" else 1
            for series, g in P.groupby("series"):
                g = g.sort_values("n_median")
                kind, head = g.kind.iloc[0], series.split("/")[0]
                rk = series.split("/")[1] if "/" in series else None
                if kind in ("attr", "natural") and rk not in FIG_RANKINGS + ["union"]:
                    continue
                c = colors.get(head, INK_MUTED)
                if head == "fitted_act" and rk == "coact":
                    c = tint(c, 0.5)                               # the co-activation variant: a lighter purple
                name = names.get(head, head) + ("" if rk is None or kind == "natural" else " (%s)" % rk)
                if kind in ("sweep", "attr"):
                    ax.plot(g.n_median, scale * g[col], ls=dashes.get(rk, "-"), marker="o", ms=3.5, color=c,
                            label=name, zorder=3)
                elif kind == "natural":
                    ax.scatter(g.n_median, scale * g[col], marker="*", s=60, color=c, label=name + ", natural size", zorder=5)
                elif kind == "ge":
                    ax.scatter(g.n_median, scale * g[col], marker="D", s=14, color=c, label=name, zorder=4)
                else:
                    src = {"fitted_abl_ig": "IG-ranked", "fitted_atp": "AtP-ranked", "exact_patch": "exact-ranked"}.get(
                        head, "activation-ranked (%s)" % rk if head == "fitted_act" else head)
                    lab = src + (", fitted α" if series.endswith("/fit") else ", α = 1")
                    ax.scatter(g.n_median, scale * g[col], marker="s" if series.endswith("/fit") else "^", s=24,
                               color=c, label=lab, zorder=5)
            if col == "A_median":
                ax.axhspan(0.8, 1.5, color="#eef0f4", zorder=0)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title("Layers %d–%d" % (lo, hi), fontsize=11)
            if i == 1:
                ax.set_xlabel("circuit size (median latents)")
            if j == 0:
                ax.set_ylabel(ylab)
            ax.grid(axis="x", visible=False)
    seen = {}
    for ax in axes.flat:                                   # one legend for the figure, below the panels
        for hnd, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, hnd)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=4, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02 - 0.035 * ((len(seen) + 3) // 4)))
    style_suptitle(fig, "Faithfulness against circuit size, protocol v1 (held-out strongest)")
    return save_figure(fig, C.OUT / "dan77-faithfulness-vs-size.png")


def main():
    ref = reference()
    pts = curve_points(ref)
    C.OUT.mkdir(parents=True, exist_ok=True)
    pts.to_csv(C.OUT / "dan77_points.csv", index=False)
    png = figure(pts)
    t, rows = build_table(table_rows(ref))
    ctx_dirs = {str(x) for sub in ("attr", "ge", "fitted", "ref") for x in (r.get("ctx_dir") for r in ours(sub)) if x}
    write_table(t, rows, size_matching_note(ref), ctx_dirs)
    print("\nwrote", C.OUT / "dan77_points.csv", png or "(no figure: no points)", C.OUT / "table2.md",
          C.OUT / "table2.csv", C.OUT / "table2_rows.csv", C.OUT / "table2.tex")


if __name__ == "__main__":
    main()
