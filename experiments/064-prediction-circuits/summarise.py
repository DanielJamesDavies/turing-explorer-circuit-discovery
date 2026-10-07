"""Tables, statistics and figures for 064 from the outputs of target_effects.py and inference_dla.py.

Prediction-like tiers (per target; thresholds fixed before looking at the inference results):
  writer          boost_peak >= 1 nat: at the target's peak activation, its direct push on its top promoted token
                  (vs the average token, typical final-norm scale) is at least 1 nat
  prediction-like writer AND ctx_z >= 1: the latent's empirical next tokens (logit_ctx top-10) sit >= 1 sd above
                  the average token in its direct logit vector, i.e. what it writes is what actually comes next
  focused         prediction-like AND top-1 z-score >= the 95th percentile of random non-target latents at its site

  PYTHONPATH=src python experiments/064-prediction-circuits/summarise.py
Writes results/summary.json, results/tables.md, figures/*.png (+ .pdf).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import common as C
from analysis.style import (BLUE, BLUE_LIGHT, CATEGORICAL, INK_MUTED, configure_matplotlib, save_figure,
                            style_suptitle, styled_boxplot, styled_legend, panel_figsize)

FIG = C.HERE / "figures"
R = C.RESULTS
KIND_COL = {"attn": CATEGORICAL[0], "mlp": CATEGORICAL[1], "resid": CATEGORICAL[2]}


def tiers(t, n):
    z95 = n.groupby(["layer", "kind"]).z1.quantile(0.95).rename("z95")
    t = t.join(z95, on=["layer", "kind"])
    t["writer"] = t.boost_peak >= 1.0
    t["pred_like"] = t.writer & (t.ctx_z >= 1.0)
    t["focused"] = t.pred_like & (t.z1 >= t.z95)
    return t


def md_table(df, floatfmt="{:.3g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                     for v in r.tolist()) + " |")
    return "\n".join(out)


def boot_median(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, (n, len(x))), axis=1)
    return float(np.median(x)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    t = pd.read_parquet(R / "target_effects.parquet")
    n = pd.read_parquet(R / "null_effects.parquet")
    P = pd.read_parquet(R / "dla_positions.parquet")
    F = pd.read_parquet(R / "dla_fired.parquet")
    Z = pd.read_parquet(R / "causal.parquet")
    rms = json.loads(C.RMS_TYP_FILE.read_text())
    t = tiers(t, n)
    t[["gid", "writer", "pred_like", "focused", "z95"]].to_parquet(R / "target_tiers.parquet")
    S, md = {"rms_typ": rms}, []

    # ---------------------------------------------------------------- 1. per-target output effect
    tot = len(t)
    S["targets"] = {k: int(t[k].sum()) for k in ("writer", "pred_like", "focused")}
    S["targets_pass"] = {k: int((t[k] & t["pass"]).sum()) for k in ("writer", "pred_like", "focused")}
    S["n_targets"], S["n_pass"] = tot, int(t["pass"].sum())
    S["pass_rate"] = {k: float(t.loc[t[k], "pass"].mean()) for k in ("writer", "pred_like", "focused")}
    S["pass_rate_all"] = float(t["pass"].mean())
    S["ctx_z_ge1"] = {"targets": float((t.ctx_z >= 1).mean()), "targets_shuffled": float((t.ctx_z_shuf >= 1).mean()),
                      "null": float((n.ctx_z >= 1).mean()), "null_shuffled": float((n.ctx_z_shuf >= 1).mean())}
    by = t.groupby(["kind"]).agg(n=("gid", "size"), writer=("writer", "sum"), pred_like=("pred_like", "sum"),
                                 focused=("focused", "sum"),
                                 pred_like_pass=("pred_like", lambda s: int((s & t.loc[s.index, "pass"]).sum())),
                                 med_boost_peak=("boost_peak", "median"), med_ctx_z=("ctx_z", "median"))
    by["pct_pred_like"] = 100 * by.pred_like / by.n
    md.append("### Prediction-like targets by kind\n\n" + md_table(by.reset_index()))
    bl = t.groupby(["layer", "kind"]).pred_like.agg(["size", "sum"]).reset_index()
    piv = bl.pivot(index="layer", columns="kind", values="sum").reindex(columns=list(C.KINDS)).fillna(0).astype(int)
    pn = bl.pivot(index="layer", columns="kind", values="size").reindex(columns=list(C.KINDS)).fillna(0).astype(int)
    tab = pd.DataFrame({"layer (1-based)": piv.index + 1})
    for k in C.KINDS:
        tab[k] = [f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-" for a, b in zip(piv[k], pn[k])]
    pl_pass = t[t.pred_like].groupby("layer")["pass"].sum().reindex(piv.index, fill_value=0)
    tab["pred-like passing"] = pl_pass.to_numpy()
    md.append("### Prediction-like targets by layer (count / targets at that site)\n\n" + md_table(tab))
    # targets vs null (same-site random latents): per-unit-activation metrics
    cmp_rows = []
    for k in C.KINDS:
        a, b = t[t.kind == k], n[n.kind == k]
        cmp_rows.append({"kind": k, "z1 targets": a.z1.median(), "z1 null": b.z1.median(),
                         "gain targets": a.gain.median(), "gain null": b.gain.median(),
                         "boost@mean targets": a.boost_mean.median(), "boost@mean null": b.boost_mean.median(),
                         "ctx_z targets": a.ctx_z.median(), "ctx_z null": b.ctx_z.median(),
                         "ctx_z shuffled": a.ctx_z_shuf.median()})
    md.append("### Targets vs random non-target latents at the same sites (medians)\n\n"
              + md_table(pd.DataFrame(cmp_rows)))
    S["top_boost_peak_quantiles"] = t.boost_peak.quantile([.5, .9, .99]).to_dict()

    # ---------------------------------------------------------------- 2. inference
    g = F.groupby(["seq", "t"])
    best = F.loc[g.dla_top1.idxmax()]
    bsc = F.loc[g.score.idxmax()]
    tl = t.set_index("gid")
    F["pred_like"] = tl.pred_like.reindex(F.gid).to_numpy()
    best = best.assign(pred_like=tl.pred_like.reindex(best.gid).to_numpy())
    rho = g[["score", "dla_top1"]].apply(lambda d: spearmanr(d.score, d.dla_top1)[0] if len(d) > 3 else np.nan)

    def rank_under_score(d):
        o = d.sort_values("score", ascending=False).reset_index(drop=True)
        return int(o.dla_top1.idxmax())
    rk = g[["score", "dla_top1"]].apply(rank_under_score)
    ov5 = g[["row", "score", "dla_top1"]].apply(
        lambda d: len(set(d.nlargest(5, "dla_top1").row) & set(d.nlargest(5, "score").row)))
    S["inference"] = {
        "n_positions": len(P), "by_src": P.src.value_counts().to_dict(),
        "median_fired": float(P.n_fired.median()), "median_active": float(P.n_active.median()),
        "active_target_frac": float(P.active_target_frac.median()),
        "best_fired_dla_quantiles": best.dla_top1.quantile([.1, .25, .5, .75, .9]).to_dict(),
        "frac_pos_best_fired_dla_ge": {str(x): float((best.dla_top1 >= x).mean()) for x in (0.1, 0.25, 0.5, 1.0)},
        "best_all_dla_quantiles": P.best_all_dla.quantile([.25, .5, .75]).to_dict(),
        "frac_pos_best_all_dla_ge": {str(x): float((P.best_all_dla >= x).mean()) for x in (0.1, 0.25, 0.5, 1.0)},
        "best_all_is_target": float(P.best_all_is_target.mean()),
        "top10_all_target_frac": float(P.top10_all_target_frac.mean()),
        "best_target_rank_among_all_median": float(P.best_target_rank_all.median()),
        "share_pos_dla_from_targets_median": float((P.sum_pos_dla_targets / P.sum_pos_dla_all).median()),
        "best_dla_layer_mean": float(best.layer.mean()), "best_score_layer_mean": float(bsc.layer.mean()),
        "best_dla_kind": best.kind.value_counts(normalize=True).to_dict(),
        "best_score_kind": bsc.kind.value_counts(normalize=True).to_dict(),
        "best_score_circuit_dla_median": float(bsc.dla_top1.median()),
        "spearman_score_dla_median": float(rho.median()),
        "rank_of_top_dla_under_score": rk.describe().to_dict(),
        "top5_overlap_mean": float(ov5.mean()),
        "best_dla_pass_frac": float(best["pass"].mean()), "fired_pass_frac": float(F["pass"].mean()),
        "fired_pred_like_frac": float(F.pred_like.mean()), "best_dla_pred_like_frac": float(best.pred_like.mean()),
        "dla_median_pred_like_fired": float(F.loc[F.pred_like, "dla_top1"].median()),
        "absdla_median_pred_like_fired": float(F.loc[F.pred_like, "dla_top1"].abs().median()),
        "absdla_median_other_fired": float(F.loc[~F.pred_like, "dla_top1"].abs().median()),
        "dla_true_best_quantiles": F.groupby(["seq", "t"]).dla_true.max().quantile([.5, .9]).to_dict(),
    }
    inf_tab = pd.DataFrame([
        {"ranking": "explorer score (top-1)", "mean layer (0-based)": bsc.layer.mean(),
         "resid/mlp/attn %": "/".join(f"{100 * bsc.kind.eq(k).mean():.0f}" for k in ("resid", "mlp", "attn")),
         "median DLA to top-1 (nats)": bsc.dla_top1.median(), "pass %": 100 * bsc["pass"].mean()},
        {"ranking": "DLA to top-1 (top-1)", "mean layer (0-based)": best.layer.mean(),
         "resid/mlp/attn %": "/".join(f"{100 * best.kind.eq(k).mean():.0f}" for k in ("resid", "mlp", "attn")),
         "median DLA to top-1 (nats)": best.dla_top1.median(), "pass %": 100 * best["pass"].mean()}])
    md.append("### Top circuit per position: explorer score vs DLA (1,258 positions)\n\n" + md_table(inf_tab))

    # ---------------------------------------------------------------- 3. causal
    s = Z[~Z.joint]
    j = Z[Z.joint]
    per_pos = s.groupby(["seq", "t", "set"]).d_logp_top1.mean().unstack()
    per_kl = s.groupby(["seq", "t", "set"]).kl.mean().unstack()
    caus = []
    for name in ("dla", "score", "act", "random"):
        a, b = s[s.set == name], j[j.set == name]
        m, lo, hi = boot_median(a.d_logp_top1)
        caus.append({"set": name, "pred. DLA (single)": a.dla_top1.median(), "act": a.act.median(),
                     "d logit top1": a.d_logit_top1.median(), "d logp top1": m, "95% CI": f"[{lo:.4f}, {hi:.4f}]",
                     "KL": a.kl.median(), "top1 flips %": 100 * a.top1_changed.mean(),
                     "joint d logp": b.d_logp_top1.median(), "joint KL": b.kl.median(),
                     "joint flips %": 100 * b.top1_changed.mean()})
    md.append("### Causal check: single-target ablations (144 per set, 48 positions) and set-joint ablations\n\n"
              + md_table(pd.DataFrame(caus), "{:.4f}"))
    tests = {}
    for ctrl in ("score", "act", "random"):
        d = per_pos["dla"] - per_pos[ctrl]
        tests[ctrl] = {"median_diff_dlogp": float(d.median()), "frac_dla_more_negative": float((d < 0).mean()),
                       "wilcoxon_p": float(wilcoxon(per_pos["dla"], per_pos[ctrl]).pvalue),
                       "kl_frac_dla_larger": float((per_kl["dla"] > per_kl[ctrl]).mean()),
                       "kl_wilcoxon_p": float(wilcoxon(per_kl["dla"], per_kl[ctrl]).pvalue)}
    S["causal"] = {"table": caus, "paired_per_position": tests,
                   "dla_score_set_overlap": int(s[s.set == "dla"].merge(s[s.set == "score"], on=["seq", "t", "gid"]).shape[0])}
    # direct-path fidelity: predicted first-order change (-DLA) vs measured, single ablations
    fid = []
    bands = [(0, 3, "L1-4"), (4, 7, "L5-8"), (8, 11, "L9-12")]
    for lo_, hi_, name in bands:
        d = s[(s.layer >= lo_) & (s.layer <= hi_)]
        fid.append({"layers": name, "n": len(d), "pearson": np.corrcoef(-d.dla_top1, d.d_logp_top1)[0, 1],
                    "spearman": spearmanr(-d.dla_top1, d.d_logp_top1)[0],
                    "slope measured/predicted": np.polyfit(-d.dla_top1, d.d_logp_top1, 1)[0]})
    fid.append({"layers": "all", "n": len(s), "pearson": np.corrcoef(-s.dla_top1, s.d_logp_top1)[0, 1],
                "spearman": spearmanr(-s.dla_top1, s.d_logp_top1)[0],
                "slope measured/predicted": np.polyfit(-s.dla_top1, s.d_logp_top1, 1)[0]})
    S["direct_path_fidelity"] = fid
    md.append("### Direct-path fidelity: predicted (-DLA) vs measured change of log p(top-1), single ablations\n\n"
              + md_table(pd.DataFrame(fid)))

    # ---------------------------------------------------------------- examples
    tok = C.load_tokenizer()
    dec = C.Dec(tok)
    tk = np.load(R / "target_topk.npz")
    rowof = {int(x): i for i, x in enumerate(tk["gid"])}
    ex = F.nlargest(15, "dla_top1").merge(P[["seq", "t", "token", "top1", "p_top1", "src"]], on=["seq", "t"])
    exr = [{"token": repr(dec(int(r.token))), "model top-1": f"{dec(int(r.top1))!r} ({r.p_top1:.2f})",
            "target": C.full_label(int(r.gid)), "act": r.act, "DLA": r.dla_top1, "score": r.score,
            "pass": bool(r["pass"]),
            "target promotes": " ".join(repr(dec(int(x))) for x in tk["top_ids"][rowof[int(r.gid)]][:5])}
           for _, r in ex.iterrows()]
    md.append("### The 15 largest fired-circuit DLAs to the model's top-1 (all 1,258 positions)\n\n"
              + md_table(pd.DataFrame(exr), "{:.3f}"))
    top = t[t.pred_like].nlargest(15, "boost_peak")
    lc_tok, lc_prob, _ = C.load_logit_ctx()
    tpr = [{"target": C.full_label(int(r.gid)), "pass": bool(r["pass"]), "boost@peak": r.boost_peak,
            "z1": r.z1, "ctx_z": r.ctx_z,
            "promotes": " ".join(repr(dec(int(x))) for x in tk["top_ids"][rowof[int(r.gid)]][:6]),
            "logit_ctx next": " ".join(repr(dec(x)) for x, _ in
                                       C.distinct_next_tokens(lc_tok, lc_prob, int(r.gid), 5))}
           for _, r in top.iterrows()]
    md.append("### Most prediction-like targets (largest boost at peak among prediction-like)\n\n"
              + md_table(pd.DataFrame(tpr), "{:.2f}"))

    (R / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    (R / "tables.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    figures(t, n, P, F, best, bsc, s, j)
    print(json.dumps(S, indent=1, default=float)[:6000])


def figures(t, n, P, F, best, bsc, s, j):
    plt = configure_matplotlib()
    FIG.mkdir(exist_ok=True)
    layers = np.arange(12)

    # fig 1: where are the prediction-like targets
    fig, ax = plt.subplots(1, 2, figsize=panel_figsize(1, 2))
    for k in C.KINDS:
        d = t[t.kind == k].groupby("layer")
        ax[0].plot(layers + 1, 100 * d.pred_like.mean().reindex(layers), marker="o", color=KIND_COL[k], label=k)
    ax[0].set_xlabel("layer (1-based)"); ax[0].set_ylabel("% of targets prediction-like")
    ax[0].set_title("Prediction-like targets by site"); ax[0].set_xticks(layers + 1)
    styled_legend(ax[0], loc="upper left")
    bins = np.linspace(-3, 5, 61)
    ax[1].hist(t.ctx_z.dropna(), bins=bins, color=BLUE, alpha=0.75, label="own logit_ctx tokens")
    ax[1].hist(t.ctx_z_shuf.dropna(), bins=bins, color=CATEGORICAL[1], alpha=0.5, label="shuffled pairing (null)")
    ax[1].axvline(1, color=INK_MUTED, lw=1.2, ls="--")
    ax[1].set_xlabel("ctx_z: direct logit of empirical next tokens (sd above avg)")
    ax[1].set_ylabel("targets"); ax[1].set_title("Direct output agrees with logit_ctx")
    styled_legend(ax[1], loc="upper right")
    style_suptitle(fig, "Per-target direct output effect (15,004 targets)")
    save_figure(fig, FIG / "fig1_target_effects.png")

    # fig 2: inference
    fig, ax = plt.subplots(1, 2, figsize=panel_figsize(1, 2))
    bins = np.logspace(-4, 1, 51)
    ax[0].hist(np.clip(best.dla_top1, 1e-4, None), bins=bins, color=BLUE, alpha=0.75, label="best fired circuit target")
    ax[0].hist(np.clip(P.best_all_dla, 1e-4, None), bins=bins, color=CATEGORICAL[1], alpha=0.5, label="best active latent (any)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("DLA to log p(top-1), nats"); ax[0].set_ylabel("positions")
    ax[0].set_title("Best DLA per position"); styled_legend(ax[0], loc="upper left")
    w = 0.4
    cb = np.bincount(best.layer, minlength=12) / len(best)
    cs = np.bincount(bsc.layer, minlength=12) / len(bsc)
    ax[1].bar(layers + 1 - w / 2, 100 * cs, w, color=BLUE_LIGHT, label="top by explorer score")
    ax[1].bar(layers + 1 + w / 2, 100 * cb, w, color=BLUE, label="top by DLA to top-1")
    ax[1].set_xticks(layers + 1); ax[1].set_xlabel("layer (1-based)"); ax[1].set_ylabel("% of positions")
    ax[1].set_title("Layer of the top-ranked fired circuit"); styled_legend(ax[1], loc="upper left")
    style_suptitle(fig, "Circuits that fired: do they drive the prediction? (1,258 positions)")
    save_figure(fig, FIG / "fig2_inference_dla.png")

    # fig 3: causal
    fig, ax = plt.subplots(1, 2, figsize=panel_figsize(1, 2))
    names = ["dla", "score", "act", "random"]
    lab = ["top-3 DLA", "top-3 score", "top-3 act", "3 random"]
    cols = [CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[3]]
    styled_boxplot(ax[0], [j[j.set == x].d_logp_top1.to_numpy() for x in names], lab, cols, edge="match")
    ax[0].set_ylabel("change of log p(top-1), nats"); ax[0].set_title("Joint ablation of the 3 targets (48 positions)")
    ax[0].axhline(0, color=INK_MUTED, lw=1)
    for (lo_, hi_, name), c in zip([(0, 3, "L1-4"), (4, 7, "L5-8"), (8, 11, "L9-12")], CATEGORICAL):
        d = s[(s.layer >= lo_) & (s.layer <= hi_)]
        ax[1].scatter(-d.dla_top1, d.d_logp_top1, s=14, color=c, alpha=0.7, label=name, edgecolors="none")
    lim = float(np.abs(np.r_[s.dla_top1, s.d_logp_top1]).max()) * 1.05
    ax[1].plot([-lim, lim], [-lim, lim], color=INK_MUTED, lw=1, ls="--")
    ax[1].set_xlim(-lim, lim * 0.3); ax[1].set_ylim(-lim, lim * 0.3)
    ax[1].set_xlabel("predicted change (-DLA), nats"); ax[1].set_ylabel("measured change, nats")
    ax[1].set_title("Direct path vs ablation (single targets)"); styled_legend(ax[1], loc="upper left")
    style_suptitle(fig, "Causal check: ablating fired targets at the position")
    save_figure(fig, FIG / "fig3_causal.png")


if __name__ == "__main__":
    main()
