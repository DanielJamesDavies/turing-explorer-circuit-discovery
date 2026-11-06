"""088 SAE PROPERTIES, step 2: do failing targets share SAE properties? Pass vs fail medians, pooled and within-cell
Spearman rho, and partial rho controlling for the target's held-out activation.

Labels: 062 full run, main/eval.shard*.jsonl, held == "strong", vacuous_tk rows dropped. PASS = free0_tk,
freeM_topk_tk, freeN_topk_tk all in [0.8, 1.5] and phi_sup_blind_tk >= 0.9. Scores Z = free0_tk, A = freeM_topk_tk,
C = freeN_topk_tk. Cells = depth band (L0-4 / L5-7 / L8-11) x kind; "within" = the unweighted mean of the per-cell
Spearman rho over cells with >= MIN_CLASS passes and fails. Partial rho: within each cell, ranks of x and y are
residualised on the rank of the control (a_pos_tk) and correlated (Pearson of residuals), then averaged over cells.

  PYTHONPATH=src .venv/bin/python experiments/088-sae-properties/analyze.py   (after compute.py)
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = HERE / "out"
MAIN = REPO / "experiments/062-h100-protocol-v1/out_full/out/main"
MIN_CLASS = 20

PROPS = [  # (column, label)
    ("enc_norm", "encoder row norm"), ("enc_norm_pct", "encoder norm, site percentile"),
    ("b_enc", "encoder bias b_enc"), ("b_eff", "effective bias b_enc - W_enc.b_dec"),
    ("b_eff_pct", "effective bias, site percentile"), ("b_eff_z", "b_eff / std(pre-act)"),
    ("dec_norm", "decoder column norm"), ("dec_norm_pct", "decoder norm, site percentile"),
    ("enc_dec_cos", "cos(encoder row, decoder col)"),
    ("enc_nn", "max encoder cosine to another latent"), ("enc_n03", "# encoder rivals cos>0.3"),
    ("enc_n05", "# encoder rivals cos>0.5"),
    ("enc_n07", "# encoder rivals cos>0.7"),
    ("dec_nn", "max decoder cosine to another latent"), ("dec_n03", "# decoder rivals cos>0.3"),
    ("dec_n05", "# decoder rivals cos>0.5"),
    ("dec_n07", "# decoder rivals cos>0.7"),
    ("log_freq", "log10 firing freq (latent_stats)"), ("log_freq_sample", "log10 Top-K freq (corpus sample)"),
    ("mean_act_rel", "mean act when active / site cut"), ("pre_max_rel", "sample max pre-act / site cut"),
    ("apos_rel", "held-out a_pos_tk / site cut"), ("a_pos_tk", "held-out a_pos_tk (control)"),
    ("rank_clean_A", "natural Top-K rank, clean (spec, pi A)"),
]


def band(l):
    return "L0-4" if l <= 4 else ("L5-7" if l <= 7 else "L8-11")


def load():
    rows = []
    for p in glob.glob(str(MAIN / "eval.shard*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("held") == "strong" and "error" not in r:
                rows.append({k: r.get(k) for k in ("seed", "layer", "kind", "a_pos_tk", "free0_tk", "freeM_topk_tk",
                                                   "freeN_topk_tk", "phi_sup_blind_tk", "vacuous_tk", "n")})
    ev = pd.DataFrame(rows).rename(columns={"seed": "key"})
    n_all = len(ev)
    ev = ev[ev.vacuous_tk != True].copy()  # noqa: E712
    f = ev[["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]].astype(float)
    ev["pass"] = (((f >= 0.8) & (f <= 1.5)).all(axis=1) & (ev.phi_sup_blind_tk.astype(float) >= 0.9)).astype(int)
    sp = []
    for p in glob.glob(str(MAIN / "spec.shard*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("pi") == "A":
                sp.append((r["seed"], r.get("rank_clean")))
    sp = pd.DataFrame(sp, columns=["key", "rank_clean_A"]).drop_duplicates("key")
    pr = pd.read_csv(OUT / "props.csv")
    d = ev.merge(pr.drop(columns=["layer", "kind"]), on="key", how="inner").merge(sp, on="key", how="left")
    d["band"] = d.layer.map(band)
    d["cell"] = d.band + "/" + d.kind
    d["log_freq"] = np.log10(d.freq.clip(lower=1e-9))
    d["log_freq_sample"] = np.log10(d.freq_sample.clip(lower=1e-5))
    d["mean_act_rel"] = d.mean_act / d.cut_med
    d["pre_max_rel"] = d.pre_max / d.cut_med
    d["apos_rel"] = d.a_pos_tk / d.cut_med
    d["b_eff_z"] = d.b_eff / d.pre_std
    return d, n_all


def rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return spearmanr(x[m], y[m]).statistic


def prho(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 10:
        return np.nan
    rx, ry, rz = (rankdata(v[m]) for v in (x, y, z))
    Z = np.c_[np.ones_like(rz), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def within(d, fn):
    vals = []
    for _, g in d.groupby("cell"):
        if g["pass"].sum() >= MIN_CLASS and (1 - g["pass"]).sum() >= MIN_CLASS:
            v = fn(g)
            if np.isfinite(v):
                vals.append(v)
    return (np.mean(vals) if vals else np.nan), len(vals)


def table(d):
    out = []
    for c, lab in PROPS:
        x = d[c].astype(float).values
        ps, fl = d.loc[d["pass"] == 1, c].astype(float), d.loc[d["pass"] == 0, c].astype(float)
        r = dict(prop=c, label=lab, med_pass=ps.median(), med_fail=fl.median(),
                 rho_pass_pooled=rho(x, d["pass"].values.astype(float)))
        for nm, col in (("pass", "pass"), ("A", "freeM_topk_tk"), ("C", "freeN_topk_tk"), ("Z", "free0_tk")):
            r["rho_%s_within" % nm], r["n_cells"] = within(d, lambda g: rho(g[c].astype(float).values,
                                                                            g[col].astype(float).values))
        r["prho_pass_ctrl_apos"], _ = within(d, lambda g: prho(g[c].astype(float).values, g["pass"].values.astype(float),
                                                               g["a_pos_tk"].astype(float).values))
        r["prho_A_ctrl_apos"], _ = within(d, lambda g: prho(g[c].astype(float).values,
                                                            g["freeM_topk_tk"].astype(float).values,
                                                            g["a_pos_tk"].astype(float).values))
        r["prho_pass_ctrl_rank"], _ = within(d, lambda g: prho(g[c].astype(float).values,
                                                               g["pass"].values.astype(float),
                                                               g["rank_clean_A"].astype(float).values))
        out.append(r)
    return pd.DataFrame(out)


def fmt(t, title):
    cols = ["label", "med_pass", "med_fail", "rho_pass_pooled", "rho_pass_within", "rho_A_within", "rho_C_within",
            "prho_pass_ctrl_apos", "prho_A_ctrl_apos", "prho_pass_ctrl_rank"]
    s = ["### " + title, "", "| property | med pass | med fail | rho pass (pooled) | rho pass (within) | rho A (within) "
         "| rho C (within) | partial rho pass, ctrl a_pos | partial rho A, ctrl a_pos | partial rho pass, ctrl rank |",
         "|---|" + "---:|" * (len(cols) - 1)]
    for _, r in t.iterrows():
        s.append("| " + " | ".join([r["label"]] + ["%.3g" % r[c] if np.isfinite(r[c]) else "-" for c in cols[1:]]) + " |")
    return "\n".join(s) + "\n"


GEOM = ["enc_norm_pct", "b_eff_pct", "b_eff_z", "enc_dec_cos", "enc_nn", "dec_nn", "dec_n03", "log_freq"]


def cv_auc(d):
    """How much the SAE properties add to the activation predictors: 5-fold CV logistic-regression AUC per cell
    (standardised features, ranks of the activation features), activation only vs activation + SAE geometry."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    sets = {"a_pos_tk + a_pos/cut": ["a_pos_tk", "apos_rel"],
            "+ rank_clean": ["a_pos_tk", "apos_rel", "rank_clean_A"],
            "SAE geometry only": GEOM,
            "a_pos + a_pos/cut + geometry": ["a_pos_tk", "apos_rel"] + GEOM,
            "a_pos + a_pos/cut + rank + geometry": ["a_pos_tk", "apos_rel", "rank_clean_A"] + GEOM}
    s = ["### cross-validated AUC for pass, per cell (5-fold logistic regression on log/standardised features)", "",
         "| cell | n | pass % | " + " | ".join(sets) + " |", "|---|---:|---:|" + "---:|" * len(sets)]
    for c, g in d.groupby("cell"):
        g = g.dropna(subset=sorted(set(sum(sets.values(), []))))
        y = g["pass"].values
        if min(y.sum(), (1 - y).sum()) < MIN_CLASS:
            continue
        aucs = []
        for cols in sets.values():
            X = g[cols].astype(float).copy()
            for col in ("a_pos_tk", "apos_rel", "rank_clean_A"):
                if col in X:
                    X[col] = np.log1p(X[col].clip(lower=0))
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            p = cross_val_predict(m, X.values, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                  method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(y, p))
        s.append("| %s | %d | %.0f | %s |" % (c, len(g), 100 * y.mean(), " | ".join("%.3f" % a for a in aucs)))
    return "\n".join(s) + "\n"


def main():
    d, n_all = load()
    d.to_csv(OUT / "merged.csv", index=False)
    md = ["# 088 SAE properties of passing vs failing targets", "",
          "strong eval rows %d; non-vacuous with SAE props %d; pass %d (%.1f%%)" % (
              n_all, len(d), d["pass"].sum(), 100 * d["pass"].mean()), ""]
    md.append("pass rate by cell: " + ", ".join("%s %.0f%% (n=%d)" % (c, 100 * g["pass"].mean(), len(g))
                                                 for c, g in d.groupby("cell")) + "\n")
    t = table(d); t.to_csv(OUT / "table_all.csv", index=False)
    md.append(fmt(t, "All kinds (within = mean over band x kind cells, n_cells=%d)" % int(t.n_cells.max())))
    for kd in ("attn", "mlp", "resid"):
        tk = table(d[d.kind == kd]); tk.to_csv(OUT / ("table_%s.csv" % kd), index=False)
        md.append(fmt(tk, "%s only (within = mean over depth bands, n_cells=%d)" % (kd, int(tk.n_cells.max()))))
    # attention: rivals and norms by pass, per band
    a = d[d.kind == "attn"]
    md.append("### attention, per band: medians pass / fail\n")
    md.append("| band | n pass | n fail | enc_n05 | dec_n05 | enc_nn | dec_nn | enc_norm_pct | dec_norm_pct | b_eff_z | apos_rel |")
    md.append("|---|---:|---:|---|---|---|---|---|---|---|---|")
    for b, g in a.groupby("band"):
        cell = []
        for c in ("enc_n05", "dec_n05", "enc_nn", "dec_nn", "enc_norm_pct", "dec_norm_pct", "b_eff_z", "apos_rel"):
            cell.append("%.3g / %.3g" % (g.loc[g["pass"] == 1, c].median(), g.loc[g["pass"] == 0, c].median()))
        md.append("| %s | %d | %d | %s |" % (b, g["pass"].sum(), (1 - g["pass"]).sum(), " | ".join(cell)))
    md.append("")
    # fraction with any near-duplicate rival
    md.append("### share of targets with >= 1 rival (cos > 0.5 / > 0.7), pass vs fail\n")
    md.append("| kind | enc>0.5 pass | enc>0.5 fail | dec>0.5 pass | dec>0.5 fail | dec>0.7 pass | dec>0.7 fail |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for kd, g in d.groupby("kind"):
        P, F = g[g["pass"] == 1], g[g["pass"] == 0]
        md.append("| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
            kd, 100 * (P.enc_n05 > 0).mean(), 100 * (F.enc_n05 > 0).mean(), 100 * (P.dec_n05 > 0).mean(),
            100 * (F.dec_n05 > 0).mean(), 100 * (P.dec_n07 > 0).mean(), 100 * (F.dec_n07 > 0).mean()))
    md.append("")
    md.append(cv_auc(d))
    sites = pd.read_csv(OUT / "sites.csv")
    md.append("### site Top-K cut (128th largest post-ReLU pre-act, corpus sample)\n")
    md.append("```\n" + sites.round(3).to_string(index=False) + "\n```")
    (OUT / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
