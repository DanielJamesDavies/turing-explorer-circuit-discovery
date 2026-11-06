"""098 report: the mini Table 2 (WCM vs CircuitLasso vs the 083 ranked-member rows), the Lasso diagnostics and a
small faithfulness-vs-size plot, from the row files in OUT083 (CPU, seconds).

Rules (protocol v2, §12.1):
- post-Top-K pass (headline): Z, A_sp, C_sp in [0.8, 1.5] and necessity >= 0.9 on the activation read; vacuous rows
  are excluded from the denominator (cmp_common.passes);
- pre-activation pass: the same rule on the pre-activation read, with baseline 0 for Z and necessity
  (paper/tables/common.pre_passes).

Outputs (in experiments/098-circuitlasso-baseline/out/, or OUT098):
- table_mini.md: summary, per-target cells and diagnostics;
- table_mini.csv;
- faithfulness_vs_size.png.

  PYTHONPATH=src python experiments/098-circuitlasso-baseline/report.py     (env OUT083, REF_RUN as for the run)
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E083 = HERE.parent / "083-v1-comparison"
sys.path.insert(0, str(E083))
import cmp_common as C  # noqa: E402

OUT098 = Path(os.environ.get("OUT098", str(HERE / "out")))
PROD = C.REF_ARM


def _pre_passes():
    spec = importlib.util.spec_from_file_location("_paper_tables_common", ROOT / "paper" / "tables" / "common.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.pre_passes


pre_passes = _pre_passes()


def rows(sub, pattern="rows.*.jsonl"):
    return C.ok_rows(C.read_jsonl(C.OUT / sub / pattern))


# (label, source, filter): the mini Table 2 in display order
def groups():
    ref = C.last_by(rows("ref"), ("seed", "arm"))
    attr = C.last_by(rows("attr"), ("seed", "method", "ranking", "size_arm"))
    fit = C.last_by(rows("fitted"), ("seed", "method", "ranking", "variant"))
    sel = lambda R, **kw: [r for r in R if all(r.get(k) == v for k, v in kw.items())]  # noqa: E731
    return [
        ("WCM v2 (093 circuit, λ 1e-3), fitted α", sel(ref, arm=PROD)),
        ("Lasso top-n, |coef| (paper), α = 1", sel(attr, method="lasso", ranking="coef", size_arm=PROD)),
        ("Lasso top-n, |std coef|, α = 1", sel(attr, method="lasso", ranking="std", size_arm=PROD)),
        ("Lasso per-site counts, |coef|, α = 1", sel(fit, method="fitted_lasso", ranking="coef", variant="a1")),
        ("Lasso per-site counts, |coef|, fitted α", sel(fit, method="fitted_lasso", ranking="coef", variant="fit")),
        ("Lasso per-site counts, |std coef|, α = 1", sel(fit, method="fitted_lasso", ranking="std", variant="a1")),
        ("Lasso per-site counts, |std coef|, fitted α", sel(fit, method="fitted_lasso", ranking="std", variant="fit")),
        ("Lasso natural (support at CV λ), α = 1", sel(attr, method="lasso", ranking="cv", size_arm="natural")),
        ("AtP top-n, max, α = 1", sel(attr, method="atp", ranking="max", size_arm=PROD)),
        ("AtP top-n, SFC aggregate, α = 1", sel(attr, method="atp", ranking="sfc", size_arm=PROD)),
        ("AtP natural (position union), α = 1", sel(attr, method="atp", size_arm="natural")),
        ("IG top-n, max, α = 1", sel(attr, method="abl_ig", ranking="max", size_arm=PROD)),
        ("IG top-n, SFC aggregate, α = 1", sel(attr, method="abl_ig", ranking="sfc", size_arm=PROD)),
        ("IG natural (position union), α = 1", sel(attr, method="abl_ig", size_arm="natural")),
        ("DAN-43: IG (max) per-site counts, α = 1", sel(fit, method="fitted_abl_ig", ranking="max", variant="a1")),
        ("DAN-43: IG (max) per-site counts, fitted α", sel(fit, method="fitted_abl_ig", ranking="max", variant="fit")),
        ("activation-ranked (mean) per-site, α = 1", sel(fit, method="fitted_act", ranking="mean", variant="a1")),
        ("activation-ranked (mean) per-site, fitted α", sel(fit, method="fitted_act", ranking="mean", variant="fit")),
    ]


def frame(rs, label):
    d = pd.DataFrame(rs)
    for c in C.HEAD + [C.NEC, "a_pos_pre", "free0_raw_pre", "phi_sup_blind_raw_pre", "freeM_topk_pre",
                       "freeN_topk_pre", "n"]:
        if c not in d:
            d[c] = np.nan
    d["row"] = label
    d["layer"] = d.seed.str.split(".").str[0].astype(int)
    d["depth"] = d.layer.map(C.depth_band)
    d["vacuous"] = [C.vacuous(r) for r in rs]
    d["pass_tk"] = [C.passes(r) and not C.vacuous(r) for r in rs]
    d["pass_pre"] = pre_passes(d).values
    d["z_pre"] = pd.to_numeric(d.free0_raw_pre, errors="coerce") / pd.to_numeric(d.a_pos_pre, errors="coerce")
    d["nec_pre"] = 1 - pd.to_numeric(d.phi_sup_blind_raw_pre, errors="coerce") / pd.to_numeric(d.a_pos_pre, errors="coerce")
    return d


def summary(d, wcm_n):
    ok = d[~d.vacuous]
    med = lambda c: pd.to_numeric(ok[c], errors="coerce").median()  # noqa: E731
    r = dict(row=d.row.iloc[0], targets=d.seed.nunique(), n_med=d.n.median(),
             n_over_wcm=(d.set_index("seed").n / wcm_n.reindex(d.seed).values).median(),
             Z=med("free0_tk"), A=med("freeM_topk_tk"), C=med("freeN_topk_tk"), nec=med(C.NEC),
             Z_pre=med("z_pre"), A_pre=med("freeM_topk_pre"), C_pre=med("freeN_topk_pre"), nec_pre=med("nec_pre"),
             pass_tk="%d/%d" % (int(ok.pass_tk.sum()), len(ok)), pass_pre="%d/%d" % (int(d.pass_pre.sum()), len(d)),
             vacuous=int(d.vacuous.sum()))
    for b, _, _ in C.DEPTH_BANDS:
        g = ok[ok.depth == b]
        r["tk_" + b] = "%d/%d" % (int(g.pass_tk.sum()), len(g))
    return r


def fmt(v, p=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return "%.*f" % (p, v) if isinstance(v, float) else str(v)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


def flat(members):
    return {(s, int(i)) for s, v in members.items() for i in v}


def diagnostics(targets):
    """Per target: the Lasso diagnostics and overlaps (needs the cached payloads and rankings)."""
    sizes = C.sizes_by_arm(C.sweep_rows())
    diag = {r["seed"]: r for r in C.last_by(rows("lasso", "diag.*.jsonl"), ("seed",))}
    out = []
    for key in targets:
        pl = sorted((C.OUT / "rankings" / "lasso").glob("%s.*.pt" % key))
        if not pl:
            continue
        p = torch.load(pl[-1], weights_only=False)
        h = pl[-1].name.split(".")[-2]
        dg = p["diag"]
        n = sizes.get(key, {}).get(PROD)
        ref_path = C.circuit_path(PROD, key)
        W = flat(C.members_by_site(torch.load(ref_path, weights_only=False))) if ref_path else set()
        L = set(p["global_members"].get(PROD, {}).get("coef", []))
        Ls = set(p["global_members"].get(PROD, {}).get("std", []))
        row = dict(seed=key, n_wcm=n, samples=dg["n_samples"], p_active=dg["p_active"], nnz_X=dg["nnz_design"],
                   sites=dg["n_sites"], nat_nnz=(p["natural"] or {}).get("nnz"),
                   nat_lam_rel=(p["natural"] or {}).get("lam_rel"), r2_nat=(p["natural"] or {}).get("r2_all"),
                   r2_nat_anchor=(p["natural"] or {}).get("r2_anchor"),
                   lam_rel_n=p["global_info"].get(PROD, {}).get("lam_rel"),
                   support_n=p["global_info"].get(PROD, {}).get("support"),
                   short_n=p["global_info"].get(PROD, {}).get("short"),
                   per_site_short=sum(1 for x in p["per_site_info"].values() if x["short"]),
                   J_lasso_wcm=jaccard(L, W), J_lassostd_wcm=jaccard(Ls, W), J_coef_std=jaccard(L, Ls),
                   t_lasso=dg.get("t_total"), t_collect=dg.get("t_collect"), t_cv=dg.get("t_cv"), t_path=dg.get("t_path"),
                   path_iters=dg.get("path_iters"), unconverged=dg.get("path_unconverged"),
                   pre_err=dg.get("pre_vs_post_maxerr"), cv_at_end=(dg.get("cv") or {}).get("at_end"))
        for arm in ("atp", "abl_ig"):
            rp = C.OUT / "rankings" / arm / ("%s.%s.pt" % (key, h))
            if rp.exists() and n:
                rk = torch.load(rp, weights_only=False)
                for name in ("sfc", "max"):
                    A = flat(C.truncate_global(rk["rankings"][name], rk["sites"], n))
                    row["J_lasso_%s_%s" % (arm, name)] = jaccard(L, A)
                    row["J_%s_%s_wcm" % (arm, name)] = jaccard(A, W)
        if key in diag:
            row["diag_source"] = diag[key]["source"]
        out.append(row)
    return pd.DataFrame(out)


def plot(summ, path):
    from analysis.style import BLUE, CATEGORICAL, INK_MUTED, configure_matplotlib, save_figure, style_suptitle
    plt = configure_matplotlib()
    col = lambda lab: (BLUE if lab.startswith("WCM") else CATEGORICAL[1] if lab.startswith("Lasso")  # noqa: E731
                       else CATEGORICAL[3] if lab.startswith("AtP") else CATEGORICAL[2] if lab.startswith("IG")
                       else CATEGORICAL[6] if lab.startswith("DAN-43") else CATEGORICAL[4] if lab.startswith("activ")
                       else INK_MUTED)
    mk = lambda lab: "*" if "natural" in lab else "s" if "fitted" in lab else "o"  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for _, r in summ.iterrows():
        k, n = (int(x) for x in r.pass_tk.split("/"))
        for ax, y in ((axes[0], 100 * k / max(n, 1)), (axes[1], r.A)):
            ax.scatter(r.n_med, y, color=col(r.row), marker=mk(r.row), s=60 if mk(r.row) == "*" else 26, zorder=3,
                       label=r.row)
    axes[1].axhspan(0.8, 1.5, color="#eef0f4", zorder=0)
    for ax, ylab in zip(axes, ("targets that pass, post-Top-K (%)", "A (sparsity-preserving), median")):
        ax.set_xscale("log")
        ax.set_xlabel("circuit size (median latents)")
        ax.set_ylabel(ylab)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.06 - 0.035 * ((len(l_) + 2) // 3)))
    style_suptitle(fig, "CircuitLasso vs WCM and the ranked-member baselines (8 targets, protocol v2)")
    return save_figure(fig, path)


def main():
    OUT098.mkdir(parents=True, exist_ok=True)
    gs = [(lab, frame(rs, lab)) for lab, rs in groups() if rs]
    wcm = gs[0][1].set_index("seed").n if gs and gs[0][0].startswith("WCM") else pd.Series(dtype=float)
    summ = pd.DataFrame([summary(d, wcm) for _, d in gs])
    targets = [t for t in Path(C.TARGETS).read_text().split() if t]
    dg = diagnostics(targets)
    allrows = pd.concat([d for _, d in gs], ignore_index=True)
    summ.to_csv(OUT098 / "table_mini.csv", index=False)
    allrows.to_csv(OUT098 / "table_mini_rows.csv", index=False)
    dg.to_csv(OUT098 / "lasso_diagnostics.csv", index=False)
    L = ["# 098 mini Table 2: CircuitLasso vs WCM and the ranked-member rows (protocol v2, held-out strongest)", "",
         "%d targets (`%s`); WCM reference = `%s` re-scored locally; records = `%s`; 083 rows in `%s`. Medians over "
         "non-vacuous rows; post-Top-K pass = DAN-8 rule on `*_tk`; pre-activation pass = §12.1 rule, baseline 0."
         % (len(targets), os.path.relpath(C.TARGETS, ROOT), os.path.relpath(str(C.RUN062), ROOT),
            os.environ.get("CTX_DIR", "—"), os.path.relpath(str(C.OUT), ROOT)), "",
         "| row | targets | n (med) | n/WCM | Z | A | C | nec | pass post-Top-K | pass pre-act | L0-4 | L5-7 | L8-11 "
         "| Z_pre | A_pre | C_pre | nec_pre |", "|---|" + "---|" * 16]
    for _, r in summ.iterrows():
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r.row, r.targets, fmt(r.n_med, 0), fmt(r.n_over_wcm), fmt(r.Z), fmt(r.A), fmt(r.C), fmt(r.nec), r.pass_tk,
            r.pass_pre, r["tk_L0-4"], r["tk_L5-7"], r["tk_L8-11"], fmt(r.Z_pre), fmt(r.A_pre), fmt(r.C_pre),
            fmt(r.nec_pre)))
    # per-target Z / A / C (post-Top-K) of the main rows
    keep = [lab for lab, _ in gs if lab.startswith(("WCM", "Lasso top-n, |coef|", "Lasso per-site counts, |coef|",
                                                    "Lasso natural", "AtP top-n", "IG top-n", "DAN-43", "activation"))]
    L += ["", "## Per target (post-Top-K Z / A / C, necessity; **P** = passes; n)", "",
          "| target | " + " | ".join(keep) + " |", "|---|" + "---|" * len(keep)]
    for t in targets:
        cells = []
        for lab in keep:
            d = allrows[(allrows.row == lab) & (allrows.seed == t)]
            if not len(d):
                cells.append("—")
                continue
            x = d.iloc[0]
            cells.append("%s/%s/%s, %s%s (n %s)" % (fmt(x.free0_tk), fmt(x.freeM_topk_tk), fmt(x.freeN_topk_tk),
                                                   fmt(x[C.NEC]), " **P**" if x.pass_tk else "", fmt(x.n, 0)))
        L.append("| %s | %s |" % (t, " | ".join(cells)))
    if len(dg):
        L += ["", "## Lasso diagnostics", "",
              "R² at the cross-validated (natural) λ, in sample over all samples and over the activating anchors; "
              "support at that λ; λ / λ_max where the support first reaches the W_0.001 size; Jaccard of the Lasso top-n "
              "(|coef|) with the WCM members and with the AtP / IG top-n at the same size; seconds per target (collect = "
              "forward pass; cv = validation path; path = full-data path).", "",
              "| target | n_WCM | samples | p active | nnz(X) | natural nnz | λ_cv/λ_max | R² nat | R² anchor | λ_n/λ_max "
              "| J(Lasso,WCM) | J(Lasso,AtP sfc) | J(Lasso,AtP max) | J(AtP sfc,WCM) | J(IG sfc,WCM) | J(coef,std) | s total "
              "(collect/cv/path) |", "|---|" + "---|" * 17]
        for _, r in dg.iterrows():
            L.append("| %s | %s | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s (%s/%s/%s) |" % (
                r.seed, fmt(r.n_wcm, 0), r.samples, r.p_active, r.nnz_X, fmt(r.nat_nnz, 0), fmt(r.nat_lam_rel, 4),
                fmt(r.r2_nat), fmt(r.r2_nat_anchor), fmt(r.lam_rel_n, 4), fmt(r.J_lasso_wcm, 3),
                fmt(r.get("J_lasso_atp_sfc"), 3), fmt(r.get("J_lasso_atp_max"), 3), fmt(r.get("J_atp_sfc_wcm"), 3),
                fmt(r.get("J_abl_ig_sfc_wcm"), 3), fmt(r.J_coef_std, 3), fmt(r.t_lasso, 0), fmt(r.t_collect, 0),
                fmt(r.t_cv, 0), fmt(r.t_path, 0)))
        L += ["", "Medians: R² nat %s, R² anchor %s, natural nnz %s, J(Lasso,WCM) %s, J(Lasso,AtP sfc) %s, "
              "J(AtP sfc,WCM) %s, Lasso seconds/target %s." % (
                  fmt(dg.r2_nat.median()), fmt(dg.r2_nat_anchor.median()), fmt(dg.nat_nnz.median(), 0),
                  fmt(dg.J_lasso_wcm.median(), 3), fmt(dg.get("J_lasso_atp_sfc", pd.Series(dtype=float)).median(), 3),
                  fmt(dg.get("J_atp_sfc_wcm", pd.Series(dtype=float)).median(), 3), fmt(dg.t_lasso.median(), 0))]
    png = plot(summ, OUT098 / "faithfulness_vs_size.png") if len(summ) else None
    L += ["", "Plot: `%s`. Every per-target row (with file:line) is in `table_mini_rows.csv`; diagnostics in "
          "`lasso_diagnostics.csv`." % (os.path.relpath(str(png), ROOT) if png else "—")]
    (OUT098 / "table_mini.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
