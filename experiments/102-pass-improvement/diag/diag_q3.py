"""102 diag Q3: do context-set / target / fit features predict pass vs fail? CPU, no model.

Outcomes on 095's 72 targets: v2CR production, v2CR resample, v2 production (also v2R both, for reference).
Features: out/ctxfeat_72.csv (diag_ctxfeat.py) + target features from the production row (identical across arms:
a_pos, empty-circuit baselines e0 / eM / eN relative to a_pos, contrast flags) + 062 targets.csv rank_clean + fit
features of the arm itself (size, alpha stats, member sites).
Per feature: AUC for pass (> 0.5: higher value -> more passes), Mann-Whitney p, Benjamini-Hochberg q over the
feature family per outcome, band-stratified AUC (pairs within the same depth band only), tertile pass rates with n.
Model: L2 logistic regression, standardised, repeated stratified 6-fold CV (one row per target, so folds are grouped
by target), CV AUC vs a 200-label-permutation null; feature sets: layer + kind only, rank_clean only, context
features only, pre-fit (context + target), pre-fit + fit.

Then v1 15k (062 eval rows, held-out strongest, v1 rule at floor 0.75) for the features that need no refit; caveat:
leaky held-out sets.

Writes out/q3.md, out/q3_auc.csv, out/q3_feat72.csv, out/q3_15k_auc.csv.
"""
import csv
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from diag_common import BANDS, DIAG, E62, E93, auc, band_of, inband, load_062_eval, load_095, md_table, prod_pass

warnings.filterwarnings("ignore")
OUT = DIAG / "out"
RNG = np.random.default_rng(20261031)

CTX_FEATS = ["tr_anchor_distinct", "tr_anchor_entropy", "tr_anchor_top1", "tr_bigram_top1", "tr_trigram_top1",
             "tr_bigram_distinct", "tr_trigram_distinct", "tr_dup16", "tr_win_jacc", "tr_anchor_pos_mean", "tr_early",
             "ho_anchor_seen", "ho_bigram_seen", "ho_trigram_seen", "ho_dup16", "ho_nn_jacc", "ho_anchor_distinct",
             "tr_peak_cv", "tr_peak_maxmin", "ho_tr_ratio", "st_peak_maxmin", "mid_strong_ratio",
             "ct_has_top_anchor", "ct_has_any_anchor", "ct_nn_jacc"]
TGT_FEATS = ["layer", "log_a_pos", "rank_clean", "e0_rel", "eM_rel", "eN_rel", "a_base_rel", "ctr_fallback",
             "is_attn", "is_mlp", "is_resid"]
FIT_FEATS = ["log_n", "alpha_median", "alpha_p90", "alpha_max", "frac_alpha_gt2", "n_member_sites", "n_upstream_sites",
             "frac_alpha_suppress", "log_t_fit"]


def rel(p, k):
    a = p.get("a_pos_tk")
    v = p.get(k)
    if a is None or v is None or a <= 1e-9:
        return np.nan
    return float(np.log10(max(v / a, 1e-4)))


def alpha_stats(arm):
    """{seed: (alpha_max, frac_alpha_gt2)} from the fits' amp_stats (095 train.jsonl) or 093 rows (v2)."""
    from diag_common import load_jsonl, E95
    out = {}
    if arm == "v2":
        for r in load_jsonl(E93 / "out" / "rows.jsonl"):
            if r.get("arm") == "v2":
                out[r["seed"]] = (r.get("alpha_max"), r.get("frac_alpha_gt2"))
    else:
        for r in load_jsonl(E95 / "out" / "train.jsonl"):
            if r.get("arm95") == arm:
                a = r.get("amp_stats") or {}
                out[r["seed"]] = (a.get("max"), None)
    return out


def build72():
    by = load_095()
    ctx = pd.read_csv(OUT / "ctxfeat_72.csv").set_index("seed")
    tg = pd.read_csv(E62 / "out_full" / "out" / "targets.csv").set_index("seed")
    rows = []
    for arm in ("v2", "v2CR", "v2R"):
        ast = alpha_stats(arm)
        for seed, r in by[arm].items():
            p = r["prod"]
            d = dict(seed=seed, arm=arm, band=r["band"], kind=r["kind"], layer=r["layer"], P=r["P"], RS=r["RS"])
            d.update(log_a_pos=np.log10(max(p["a_pos_tk"], 1e-4)), rank_clean=tg.rank_clean.get(seed, np.nan),
                     e0_rel=rel(p, "e0_tk"), eM_rel=rel(p, "eM_topk_tk"), eN_rel=rel(p, "eN_topk_tk"),
                     a_base_rel=rel(p, "a_base_tk"), ctr_fallback=float(bool(p.get("ctr_fallback"))),
                     is_attn=float(r["kind"] == "attn"), is_mlp=float(r["kind"] == "mlp"),
                     is_resid=float(r["kind"] == "resid"))
            amax, fgt2 = ast.get(seed, (None, None))
            d.update(log_n=np.log10(max(r.get("n") or 1, 1)), alpha_median=p.get("alpha_median"),
                     alpha_p90=p.get("alpha_p90"), alpha_max=amax, frac_alpha_gt2=fgt2,
                     n_member_sites=p.get("n_member_sites"), n_upstream_sites=p.get("n_upstream_sites"),
                     frac_alpha_suppress=p.get("frac_alpha_suppress"),
                     log_t_fit=np.log10(r["t_fit"]) if r.get("t_fit") else np.nan)
            for c in CTX_FEATS:
                d[c] = ctx.at[seed, c]
            rows.append(d)
    df = pd.DataFrame(rows)
    for c in CTX_FEATS + TGT_FEATS + FIT_FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_csv(OUT / "q3_feat72.csv", index=False)
    return df


def bh(ps):
    ps = np.asarray(ps, float)
    n = len(ps)
    o = np.argsort(ps)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, ps[i] * n / rank)
        q[i] = prev
    return q


def strat_auc(x, y, g):
    """AUC from pass/fail pairs within the same stratum only."""
    num = den = 0.0
    for s in np.unique(g):
        m = (g == s) & ~np.isnan(x)
        xs, ys = x[m], y[m]
        pos, neg = xs[ys], xs[~ys]
        if len(pos) == 0 or len(neg) == 0:
            continue
        cmp = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        num += cmp
        den += len(pos) * len(neg)
    return num / den if den else np.nan


def tertiles(x, y):
    m = ~np.isnan(x)
    x, y = x[m], y[m]
    if len(np.unique(x)) < 3:
        return "-"
    q = np.quantile(x, [1 / 3, 2 / 3])
    bins = np.digitize(x, q, right=True)
    out = []
    for b in range(3):
        sel = bins == b
        out.append("%d/%d" % (y[sel].sum(), sel.sum()))
    return " | ".join(out)


def feature_table(df, outcome, feats):
    y = df[outcome].astype(bool).values
    g = df["band"].values
    res = []
    for f in feats:
        x = df[f].values.astype(float)
        m = ~np.isnan(x)
        if m.sum() < 10 or len(np.unique(x[m])) < 2:
            continue
        a = auc(x, y)
        try:
            p = mannwhitneyu(x[m & y], x[m & ~y]).pvalue
        except ValueError:
            p = np.nan
        res.append(dict(feature=f, n=int(m.sum()), auc=a, p=p, auc_band=strat_auc(x, y, g), tertiles=tertiles(x, y)))
    q = bh([r["p"] if r["p"] == r["p"] else 1.0 for r in res])
    for r, qq in zip(res, q):
        r["q"] = qq
    return sorted(res, key=lambda r: -abs((r["auc"] or 0.5) - 0.5))


def cv_auc(X, y, n_rep=10, k=6, seed=0, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    aucs = []
    for rep in range(n_rep):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + rep)
        pred = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=C, max_iter=2000))
            mdl.fit(X[tr], y[tr])
            pred[te] = mdl.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, pred))
    return float(np.mean(aucs))


def cv_with_null(X, y, n_perm=200):
    obs = cv_auc(X, y)
    null = [cv_auc(X, RNG.permutation(y), n_rep=1, seed=i) for i in range(n_perm)]
    return obs, float(np.mean(np.array(null) >= obs)), float(np.percentile(null, 95))


def main():
    df = build72()
    md = ["# Q3: features vs pass / fail (diag_q3.py)\n",
          "AUC > 0.5: higher value -> more passes. auc_band: pairs within the same depth band only. q: BH over this"
          " outcome's features. Tertiles: passes / n in the low | mid | high third of the feature. n = 72 targets;"
          " %d features per outcome -> expect ~%d with p < 0.05 by chance.\n" % (
              len(CTX_FEATS + TGT_FEATS + FIT_FEATS), round(0.05 * len(CTX_FEATS + TGT_FEATS + FIT_FEATS)))]
    allres = []
    outcomes = (("v2CR", "P", "v2CR production"), ("v2CR", "RS", "v2CR resample"), ("v2", "P", "v2 production"),
                ("v2R", "P", "v2R production"), ("v2R", "RS", "v2R resample"))
    for arm, oc, lab in outcomes:
        d = df[df.arm == arm].reset_index(drop=True)
        res = feature_table(d, oc, CTX_FEATS + TGT_FEATS + FIT_FEATS)
        for r in res:
            r.update(outcome=lab)
        allres += res
        md.append("\n## %s (%d/%d pass)\n" % (lab, d[oc].sum(), len(d)))
        md.append(md_table(["feature", "n", "AUC", "p", "q (BH)", "AUC within band", "tertiles low | mid | high"],
                           [[r["feature"], r["n"], r["auc"], r["p"], r["q"], r["auc_band"], r["tertiles"]]
                            for r in res[:14]]))
        y = d[oc].astype(bool).values.astype(int)
        md.append("\nCross-validated L2 logistic (6-fold x 10, C 0.3), AUC vs 200-permutation null (p, null 95th pct):\n")
        trs = []
        sets = [("layer + kind", ["layer", "is_attn", "is_mlp"]), ("rank_clean only", ["rank_clean"]),
                ("context features only", CTX_FEATS), ("pre-fit (context + target)", CTX_FEATS + TGT_FEATS),
                ("pre-fit + fit", CTX_FEATS + TGT_FEATS + FIT_FEATS),
                ("compact: layer, kind, rank_clean, log a_pos, eM_rel, tr_anchor_top1, ho_dup16",
                 ["layer", "is_attn", "is_mlp", "rank_clean", "log_a_pos", "eM_rel", "tr_anchor_top1", "ho_dup16"])]
        for name, cols in sets:
            X = d[cols].values.astype(float)
            obs, pp, n95 = cv_with_null(X, y, n_perm=100)
            trs.append([name, len(cols), obs, pp, n95])
        md.append(md_table(["feature set", "k", "CV AUC", "perm p", "null 95%"], trs))
        (OUT / "q3.md").write_text("\n".join(md))
        print("\n".join(md[-4:]), flush=True)
    pd.DataFrame(allres).to_csv(OUT / "q3_auc.csv", index=False)
    md += strata_tables(df)
    md += v1_15k()
    (OUT / "q3.md").write_text("\n".join(md))
    print("\n".join(md))


def strata_tables(df):
    """Key features within band x kind cells (8 per cell): median among pass vs fail, v2CR production / resample."""
    md = ["\n## Key features within band x kind (v2CR; median feature among pass / fail, n pass / n fail)\n"]
    feats = ["rank_clean", "log_a_pos", "eM_rel", "tr_anchor_top1", "tr_anchor_distinct", "ho_dup16", "tr_dup16",
             "ho_anchor_seen"]
    for oc in ("P", "RS"):
        d = df[df.arm == "v2CR"]
        trs = []
        for (b, k), g in d.groupby(["band", "kind"]):
            y = g[oc].astype(bool)
            row = ["%s/%s" % (b, k), "%d/%d" % (y.sum(), (~y).sum())]
            for f in feats:
                row.append("%.2g / %.2g" % (g[f][y].median(), g[f][~y].median()) if y.any() and (~y).any() else "-")
            trs.append(row)
        md.append("\n**v2CR %s**\n" % ("production" if oc == "P" else "resample"))
        md.append(md_table(["cell", "pass/fail"] + feats, trs))
    return md


def v1_15k():
    """v1 15k: pass at floor 0.75 (v1 rule, post-Top-K) vs no-refit features."""
    ev = load_062_eval()
    ctx = pd.read_csv(OUT / "ctxfeat_15k.csv")
    ctx = ctx[ctx.get("error").isna()] if "error" in ctx else ctx
    ctx = ctx.set_index("seed")
    tg = pd.read_csv(E62 / "out_full" / "out" / "targets.csv").set_index("seed")
    rows = []
    for seed, r in ev.items():
        if seed not in ctx.index:
            continue
        P = prod_pass(r)
        if P is None:
            continue
        layer = int(seed.split(".")[0])
        kind = seed.split(".")[1]
        d = dict(seed=seed, P=P, layer=layer, kind=kind, band=band_of(layer),
                 log_a_pos=np.log10(max(r.get("a_pos_tk") or 1e-4, 1e-4)), rank_clean=tg.rank_clean.get(seed, np.nan),
                 e0_rel=rel(r, "e0_tk"), eM_rel=rel(r, "eM_topk_tk"), eN_rel=rel(r, "eN_topk_tk"),
                 ctr_fallback=float(bool(r.get("ctr_fallback"))) if r.get("ctr_fallback") is not None else np.nan,
                 is_attn=float(kind == "attn"), is_mlp=float(kind == "mlp"), is_resid=float(kind == "resid"),
                 log_n=np.log10(max(r.get("n") or 1, 1)))
        for c in CTX_FEATS:
            d[c] = ctx.at[seed, c]
        rows.append(d)
    df = pd.DataFrame(rows)
    for c in CTX_FEATS + ["rank_clean", "e0_rel", "eM_rel", "eN_rel", "log_a_pos", "ctr_fallback", "log_n"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_csv(OUT / "q3_15k_feat.csv", index=False)
    md = ["\n## v1 15k (062 out_full, held-out strongest, v1 circuits, floor 0.75; LEAKY held-out sets)\n",
          "n = %d targets, pass %d (%.1f%%). Features that need no refit. AUC overall / within band x kind (pairs in"
          " the same cell only).\n" % (len(df), df.P.sum(), 100 * df.P.mean())]
    feats = CTX_FEATS + ["layer", "log_a_pos", "rank_clean", "e0_rel", "eM_rel", "eN_rel", "ctr_fallback", "log_n"]
    y = df.P.astype(bool).values
    cell = (df.band + "/" + df.kind).values
    res = []
    for f in feats:
        x = df[f].values.astype(float)
        m = ~np.isnan(x)
        if m.sum() < 50 or len(np.unique(x[m])) < 2:
            continue
        res.append(dict(feature=f, n=int(m.sum()), auc=auc(x, y), auc_cell=strat_auc(x, y, cell),
                        tertiles=tertiles(x, y)))
    res.sort(key=lambda r: -abs(r["auc_cell"] - 0.5))
    pd.DataFrame(res).to_csv(OUT / "q3_15k_auc.csv", index=False)
    md.append(md_table(["feature", "n", "AUC", "AUC within band x kind", "tertiles"],
                       [[r["feature"], r["n"], r["auc"], r["auc_cell"], r["tertiles"]] for r in res]))
    # CV on 15k (subsample-free; 5-fold x 2)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    trs = []
    base = ["layer", "is_attn", "is_mlp"]
    for name, cols in (("layer + kind", base), ("+ rank_clean, log a_pos, e*_rel", base + ["rank_clean", "log_a_pos",
                                                                                            "e0_rel", "eM_rel", "eN_rel"]),
                       ("context features only", CTX_FEATS),
                       ("layer + kind + context", base + CTX_FEATS),
                       ("all no-refit", base + CTX_FEATS + ["rank_clean", "log_a_pos", "e0_rel", "eM_rel", "eN_rel",
                                                            "ctr_fallback"])):
        X = df[cols].values.astype(float)
        skf = StratifiedKFold(5, shuffle=True, random_state=1)
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        p1 = cross_val_predict(lr, X, y, cv=skf, method="predict_proba")[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15)
        p2 = cross_val_predict(gb, X, y, cv=skf, method="predict_proba")[:, 1]
        trs.append([name, len(cols), roc_auc_score(y, p1), roc_auc_score(y, p2)])
    md.append("\nCross-validated prediction of v1 pass (5-fold):\n")
    md.append(md_table(["feature set", "k", "logistic CV AUC", "GBM CV AUC"], trs))
    return md


if __name__ == "__main__":
    main()
