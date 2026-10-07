"""Attribution patching vs ablation for fired circuit targets (Linear DAN-135).

Candidate replacement for DLA in the explorer's "Prediction" sort: a first-order TOTAL effect
    est_i = -a_i * (d_i . d log p(top-1) / d x_site)      (and the same with the top-1 logit)
where x_site is the activation exactly as captured for the SAEs (attn output / mlp output / post-block resid),
a_i the target's activation and d_i its decoder column. One backward per position gives every site's gradient.
It is the first-order estimate of 064's ablation (x_site -> x_site - a_i d_i at that position, SAE error kept).

Data
  * 064 causal.parquet (inference_dla.py): 576 single + 192 joint ablations at 48 positions. est is computed for
    every row (joint rows: sum of members' est).
  * EXHAUSTIVE extension: at the same 48 positions plus N_NEW fresh positions (same eligibility, n_fired >= 12,
    half prompt / half corpus) EVERY fired target is ablated (same method, exact) -> per-band sample and a
    per-position ranking test (does est recover the biggest-effect target better than DLA / score?).
Fired lists, activations, score and dla_top1 come from 064's dla_fired.parquet; positions/top-1 from
dla_positions.parquet; sequences from inference_dla.sequences. Inputs are truncated to ids[:t+1] (causal model:
logits at t do not depend on later tokens; verified against the full-sequence ablation in the sanity step).

  PYTHONPATH=src python experiments/064-prediction-circuits/attribution_check.py [--n-new 32] [--sanity-only]
  PYTHONPATH=src python experiments/064-prediction-circuits/attribution_check.py --analyse-only
Writes results/attribution_check.parquet, results/attribution_check_summary.json,
figures/fig4_attribution_check.{png,pdf}.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common as C
import inference_dla as I
from model.hooks import multi_patch

OUT = C.RESULTS / "attribution_check.parquet"
SUMMARY = C.RESULTS / "attribution_check_summary.json"
FIG = C.HERE / "figures" / "fig4_attribution_check.png"
BANDS = ("L1–4", "L5–8", "L9–12")
SITES = [(l, k) for l in range(C.N_LAYERS) for k in C.KINDS]
CHUNK = 32


def band_of(layer):
    return BANDS[int(layer) // 4]


# ----------------------------------------------------------------------------- model maths
def site_grads(model, ids_t, top1):
    """Gradients of log p(top1) and logit(top1) at the last position w.r.t. every site at that position.

    -> (logits [V], {site: g_logp [d]}, {site: g_logit [d]})"""
    T = len(ids_t)
    deltas = {s: torch.zeros(1, T, 1024, requires_grad=True) for s in SITES}

    def tf(layer, kind, x):
        return x + deltas[(layer, kind)]

    with torch.enable_grad(), multi_patch(model, tf):
        lg, _ = model(torch.tensor([ids_t]))          # last position only
        lg = lg[0, -1].float()
        lp = F.log_softmax(lg, -1)[top1]
        keys = list(deltas)
        g1 = torch.autograd.grad(lp, [deltas[s] for s in keys], retain_graph=True)
        g2 = torch.autograd.grad(lg[top1], [deltas[s] for s in keys])
    return (lg.detach(), {s: g[0, -1] for s, g in zip(keys, g1)}, {s: g[0, -1] for s, g in zip(keys, g2)})


@torch.no_grad()
def ablate_batch(model, ids_t, items):
    """items: list of lists of (layer, kind, vec=a*d); each inner list is ONE ablation (joint if > 1 items),
    removed at the last position. -> logits [B, V] at the last position."""
    out = []
    for c0 in range(0, len(items), CHUNK):
        chunk = items[c0:c0 + CHUNK]
        by_site: dict = {}
        for b, abl in enumerate(chunk):
            for l, k, v in abl:
                by_site.setdefault((l, k), ([], []))
                by_site[(l, k)][0].append(b)
                by_site[(l, k)][1].append(v)
        by_site = {s: (torch.tensor(bs), torch.stack(vs)) for s, (bs, vs) in by_site.items()}

        def tf(layer, kind, x):
            e = by_site.get((layer, kind))
            if e is None:
                return None
            y = x.clone()
            y[:, -1].index_add_(0, e[0], -e[1])
            return y

        x = torch.tensor([ids_t] * len(chunk))
        with multi_patch(model, tf):
            lg, _ = model(x)
        out.append(lg[:, -1].float())
    return torch.cat(out)


def effects(base, lg, top1):
    lp0 = F.log_softmax(base, -1)
    lp = F.log_softmax(lg, -1)
    return {"d_logit_top1": (lg[:, top1] - base[top1]).numpy(),
            "d_logp_top1": (lp[:, top1] - lp0[top1]).numpy(),
            "kl": (lp0.exp()[None] * (lp0[None] - lp)).sum(-1).numpy(),
            "top1_changed": (lg.argmax(-1) != top1).numpy()}


# ----------------------------------------------------------------------------- compute
def compute(n_new, sanity_only):
    t0 = time.time()
    torch.set_num_threads(8)
    model = C.load_model()
    model.requires_grad_(False)
    tok = C.load_tokenizer()
    bank = C.load_bank()
    seqs = I.sequences(tok)
    P = pd.read_parquet(C.RESULTS / "dla_positions.parquet")
    Fd = pd.read_parquet(C.RESULTS / "dla_fired.parquet")
    Cz = pd.read_parquet(C.RESULTS / "causal.parquet")
    for r in P.sample(50, random_state=0).itertuples():     # same sequences / token convention as 064
        assert seqs[r.seq][2][r.t] == r.token, (r.seq, r.t)
    gids_all = np.unique(Fd.gid.to_numpy())
    D = C.decoder_dirs(bank, gids_all)
    dir_of = {int(g): D[j] for j, g in enumerate(gids_all)}
    print(f"loaded {time.time() - t0:.0f}s")

    # ---------------- sanity: reproduce 064's single ablations (full sequence, 064's own function + batched/truncated)
    single = Cz[~Cz.joint].reset_index(drop=True)
    sn = single.sample(20, random_state=135)
    diffs = []
    for r in sn.itertuples():
        ids = seqs[r.seq][2]
        t = int(r.t)
        top1 = int(P[(P.seq == r.seq) & (P.t == t)].top1.iloc[0])
        base_full, _ = model(torch.tensor([ids]), return_all_logits=True)
        base_full = base_full[0, t].float().detach()
        item = (int(r.layer), r.kind, float(r.act), dir_of[int(r.gid)])
        lg_full = I.ablate_logits(model, ids, t, [item])
        d_full = float(F.log_softmax(lg_full, -1)[top1] - F.log_softmax(base_full, -1)[top1])
        with torch.no_grad():
            base_tr = model(torch.tensor([ids[:t + 1]]))[0][0, -1].float()
        lg_tr = ablate_batch(model, ids[:t + 1], [[(int(r.layer), r.kind, float(r.act) * dir_of[int(r.gid)])]])
        d_tr = float(effects(base_tr, lg_tr, top1)["d_logp_top1"][0])
        diffs.append((r.d_logp_top1, d_full, d_tr))
    dd = np.array(diffs)
    err_full = np.abs(dd[:, 0] - dd[:, 1])
    err_tr = np.abs(dd[:, 0] - dd[:, 2])
    print(f"sanity (20 rows): |measured| median {np.median(np.abs(dd[:, 0])):.4f}; max |repro_full - 064| "
          f"{err_full.max():.2e}; max |repro_trunc - 064| {err_tr.max():.2e}")
    if max(err_full.max(), err_tr.max()) > 1e-3:
        raise SystemExit("sanity check FAILED")
    if sanity_only:
        return

    # ---------------- positions: 48 causal + n_new fresh
    cpos = Cz[["seq", "t"]].drop_duplicates()
    elig = P[P.n_fired >= I.MIN_FIRED].merge(cpos.assign(_c=1), on=["seq", "t"], how="left")
    elig = elig[elig._c.isna()]
    half = n_new // 2
    newpos = pd.concat([elig[elig.src == "prompt"].sample(half, random_state=135),
                        elig[elig.src == "corpus"].sample(n_new - half, random_state=136)])[["seq", "t"]]
    positions = pd.concat([cpos.assign(origin="causal"), newpos.assign(origin="new")]).reset_index(drop=True)
    print(f"{len(positions)} positions, {len(Fd.merge(positions, on=['seq', 't']))} fired targets to ablate")

    rows = []
    for pi, pr in enumerate(positions.itertuples()):
        seq, t = int(pr.seq), int(pr.t)
        ids = seqs[seq][2]
        ids_t = ids[:t + 1]
        ppos = P[(P.seq == seq) & (P.t == t)].iloc[0]
        top1 = int(ppos.top1)
        base, g_lp, g_lg = site_grads(model, ids_t, top1)
        assert int(base.argmax()) == top1, (seq, t)
        p_top1 = float(F.softmax(base, -1)[top1])

        def est(layer, kind, act, gid):
            d = dir_of[int(gid)]
            s = (int(layer), kind)
            return -act * float(d @ g_lp[s]), -act * float(d @ g_lg[s])

        # (a) 064 causal rows at this position
        if pr.origin == "causal":
            cz = Cz[(Cz.seq == seq) & (Cz.t == t)]
            f = Fd[(Fd.seq == seq) & (Fd.t == t)]
            for r in cz.itertuples():
                rec = {k: getattr(r, k) for k in ["seq", "t", "src", "set", "rank", "joint", "gid", "layer", "kind",
                                                  "act", "score", "dla_top1", "d_logit_top1", "d_logp_top1", "kl",
                                                  "top1_changed"]}
                if r.joint:
                    sub = cz[(cz.set == r.set) & ~cz.joint]
                    es = [est(x.layer, x.kind, x.act, x.gid) for x in sub.itertuples()]
                    rec["est_logp"], rec["est_logit"] = float(sum(e[0] for e in es)), float(sum(e[1] for e in es))
                    rec["members"] = sub.gid.astype(np.int64).tolist()
                else:
                    rec["est_logp"], rec["est_logit"] = est(r.layer, r.kind, r.act, r.gid)
                    rec["members"] = [int(r.gid)]
                rows.append(dict(rec, source="causal", origin="causal", p_top1=p_top1, top1=top1))

        # (b) exhaustive: every fired target at this position
        f = Fd[(Fd.seq == seq) & (Fd.t == t)].reset_index(drop=True)
        items = [[(int(r.layer), r.kind, float(r.act) * dir_of[int(r.gid)])] for r in f.itertuples()]
        eff = effects(base, ablate_batch(model, ids_t, items), top1)
        for j, r in enumerate(f.itertuples()):
            e1, e2 = est(r.layer, r.kind, r.act, r.gid)
            rows.append({"seq": seq, "t": t, "src": seqs[seq][0], "set": "all", "rank": -1, "joint": False,
                         "gid": int(r.gid), "layer": int(r.layer), "kind": r.kind, "act": float(r.act),
                         "score": float(r.score), "dla_top1": float(r.dla_top1),
                         **{k: (bool(v[j]) if k == "top1_changed" else float(v[j])) for k, v in eff.items()},
                         "est_logp": e1, "est_logit": e2, "members": [int(r.gid)],
                         "source": "exhaustive", "origin": pr.origin, "p_top1": p_top1, "top1": top1})
        print(f"pos {pi + 1}/{len(positions)} seq {seq} t {t} fired {len(f)} {time.time() - t0:.0f}s", flush=True)

    R = pd.DataFrame(rows)
    R["layer"] = R.layer.astype("float")
    R["band"] = [band_of(l) if l == l else None for l in R.layer]
    R.to_parquet(OUT)

    # exhaustive vs 064 causal agreement on the same (seq, t, gid)
    ex = R[R.source == "exhaustive"][["seq", "t", "gid", "d_logp_top1"]]
    cz = R[(R.source == "causal") & ~R.joint][["seq", "t", "gid", "d_logp_top1"]]
    m = cz.merge(ex, on=["seq", "t", "gid"], suffixes=("_064", "_ex"))
    print(f"exhaustive vs 064 causal on {len(m)} shared rows: max |diff| "
          f"{(m.d_logp_top1_064 - m.d_logp_top1_ex).abs().max():.2e}")
    print(f"done {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- analysis
def corr(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan, np.nan
    return float(stats.pearsonr(x, y)[0]), float(stats.spearmanr(x, y)[0])


def slope(pred, meas):
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    return float(np.polyfit(pred, meas, 1)[0])


def fidelity(df):
    """Per band + all: est / DLA / score vs measured, on single rows."""
    out = []
    for b in list(BANDS) + ["all"]:
        d = df if b == "all" else df[df.band == b]
        meas, meas_lg = d.d_logp_top1.to_numpy(), d.d_logit_top1.to_numpy()
        big = np.abs(meas) >= 0.01
        r = {"band": b, "n": len(d), "n_big": int(big.sum())}
        r["est_pearson"], r["est_spearman"] = corr(d.est_logp, meas)
        r["dla_pearson"], r["dla_spearman"] = corr(-d.dla_top1, meas)
        r["score_spearman"] = corr(d.score, -meas)[1]
        r["est_logit_pearson"], r["est_logit_spearman"] = corr(d.est_logit, meas_lg)
        r["dla_vs_logit_pearson"], r["dla_vs_logit_spearman"] = corr(-d.dla_top1, meas_lg)
        r["est_slope"] = slope(d.est_logp, meas)
        r["dla_slope"] = slope(-d.dla_top1, meas)
        r["est_logit_slope"] = slope(d.est_logit, meas_lg)
        r["est_sign"] = float((np.sign(d.est_logp[big]) == np.sign(meas[big])).mean()) if big.any() else np.nan
        r["dla_sign"] = float((np.sign(-d.dla_top1[big]) == np.sign(meas[big])).mean()) if big.any() else np.nan
        r["abs_err_est_med"] = float(np.median(np.abs(d.est_logp - meas)))
        r["abs_err_dla_med"] = float(np.median(np.abs(-d.dla_top1 - meas)))
        out.append(r)
    return pd.DataFrame(out)


def ranking(ex):
    """Per position (exhaustive): does each ranker find the target whose ablation hurts top-1 most?"""
    rankers = {"est": -ex.est_logp, "dla": ex.dla_top1, "score": ex.score, "act": ex.act}
    ex = ex.assign(**{f"k_{k}": v.to_numpy() for k, v in rankers.items()},
                   k_abs_est=ex.est_logp.abs(), k_abs_dla=ex.dla_top1.abs(), sup=-ex.d_logp_top1,
                   mag=ex.d_logp_top1.abs())
    res = {k: {"hit1": [], "in_top3": [], "top3_sup": [], "rho": [], "top3_mag": [], "hit1_mag": []}
           for k in ["est", "dla", "score", "act", "oracle"]}
    for _, g in ex.groupby(["seq", "t"]):
        best = g.sup.idxmax()
        best_mag = g.mag.idxmax()
        for k in res:
            kk = "sup" if k == "oracle" else f"k_{k}"
            km = {"est": "k_abs_est", "dla": "k_abs_dla", "oracle": "mag"}.get(k, kk)
            top3 = g.nlargest(3, kk).index
            res[k]["hit1"].append(g[kk].idxmax() == best)
            res[k]["in_top3"].append(best in top3)
            res[k]["top3_sup"].append(g.loc[top3, "sup"].mean())
            res[k]["rho"].append(corr(g[kk], g.sup)[1])
            top3m = g.nlargest(3, km).index
            res[k]["top3_mag"].append(g.loc[top3m, "mag"].mean())
            res[k]["hit1_mag"].append(g[km].idxmax() == best_mag)
    rows = []
    for k, v in res.items():
        rows.append({"ranker": k, "n_pos": len(v["hit1"]), "hit1": float(np.mean(v["hit1"])),
                     "best_in_top3": float(np.mean(v["in_top3"])),
                     "top3_mean_drop": float(np.mean(v["top3_sup"])),
                     "median_rho": float(np.nanmedian(v["rho"])),
                     "hit1_abs": float(np.mean(v["hit1_mag"])), "top3_mean_absdelta": float(np.mean(v["top3_mag"]))})
    return pd.DataFrame(rows)


def ranking_by_band_of_best(ex):
    """Share of positions whose biggest-drop target lies in each band, and est/DLA hit rates split by it."""
    out = []
    for (s, t), g in ex.groupby(["seq", "t"]):
        b = g.loc[g.d_logp_top1.idxmin()]
        out.append({"band": b.band, "est_hit": g.est_logp.idxmin() == b.name, "dla_hit": g.dla_top1.idxmax() == b.name,
                    "score_hit": g.score.idxmax() == b.name, "drop": -b.d_logp_top1})
    o = pd.DataFrame(out)
    return o.groupby("band").agg(n=("est_hit", "size"), est_hit1=("est_hit", "mean"), dla_hit1=("dla_hit", "mean"),
                                 score_hit1=("score_hit", "mean"), median_best_drop=("drop", "median")).reset_index()


def analyse():
    R = pd.read_parquet(OUT)
    ex = R[R.source == "exhaustive"].copy()
    cz = R[(R.source == "causal") & ~R.joint].copy()
    jt = R[(R.source == "causal") & R.joint].copy()
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)

    fx = fidelity(ex)
    fc = fidelity(cz)
    rk = ranking(ex)
    rb = ranking_by_band_of_best(ex)
    sat = []
    for name, m in [("p_top1 < 0.5", ex.p_top1 < 0.5), ("0.5 <= p_top1 < 0.9", (ex.p_top1 >= 0.5) & (ex.p_top1 < 0.9)),
                    ("p_top1 >= 0.9", ex.p_top1 >= 0.9)]:
        d = ex[m]
        sat.append({"group": name, "n": len(d), "n_pos": d[["seq", "t"]].drop_duplicates().shape[0],
                    "est_pearson": corr(d.est_logp, d.d_logp_top1)[0], "est_spearman": corr(d.est_logp, d.d_logp_top1)[1],
                    "est_slope": slope(d.est_logp, d.d_logp_top1) if len(d) > 2 else np.nan,
                    "est_logit_pearson": corr(d.est_logit, d.d_logit_top1)[0],
                    "dla_pearson": corr(-d.dla_top1, d.d_logp_top1)[0]})
    sat = pd.DataFrame(sat)
    # big-effect subset and the effect-size regime
    big = ex[ex.d_logp_top1.abs() >= 0.05]
    jrows = []
    for s in ["all"] + sorted(jt.set.unique()):
        d = jt if s == "all" else jt[jt.set == s]
        jrows.append({"set": s, "n": len(d), "est_pearson": corr(d.est_logp, d.d_logp_top1)[0],
                      "est_spearman": corr(d.est_logp, d.d_logp_top1)[1], "est_slope": slope(d.est_logp, d.d_logp_top1),
                      "dla_pearson": corr(-d.dla_top1, d.d_logp_top1)[0],
                      "dla_spearman": corr(-d.dla_top1, d.d_logp_top1)[1],
                      "est_logit_pearson": corr(d.est_logit, d.d_logit_top1)[0]})
    jr = pd.DataFrame(jrows)
    nbig = {b: int(((big.band == b)).sum()) for b in BANDS}
    big_corr = {b: corr(big[big.band == b].est_logp, big[big.band == b].d_logp_top1) for b in BANDS}
    big_corr_dla = {b: corr(-big[big.band == b].dla_top1, big[big.band == b].d_logp_top1) for b in BANDS}

    print("\n== exhaustive single ablations: fidelity (est = attribution patching; DLA predicted = -dla_top1) ==")
    print(fx.to_string(index=False))
    print("\n== 064 causal sample (576 rows): fidelity ==")
    print(fc.to_string(index=False))
    print("\n== exhaustive: ranking per position ==")
    print(rk.to_string(index=False))
    print("\n== exhaustive: hit rate split by band of the true best ==")
    print(rb.to_string(index=False))
    print("\n== exhaustive: saturation ==")
    print(sat.to_string(index=False))
    print("\n== |measured| >= 0.05 nats: n, (pearson, spearman) est | dla ==")
    for b in BANDS:
        print(b, nbig[b], big_corr[b], big_corr_dla[b])
    print("\n== joint rows (064 sets) ==")
    print(jr.to_string(index=False))

    summ = {"n_rows_exhaustive": len(ex), "n_positions": int(ex[["seq", "t"]].drop_duplicates().shape[0]),
            "fidelity_exhaustive": fx.to_dict("records"), "fidelity_causal064": fc.to_dict("records"),
            "ranking": rk.to_dict("records"), "ranking_by_best_band": rb.to_dict("records"),
            "saturation": sat.to_dict("records"), "joint": jr.to_dict("records"),
            "big_effects": {b: {"n": nbig[b], "est": big_corr[b], "dla": big_corr_dla[b]} for b in BANDS}}
    SUMMARY.write_text(json.dumps(summ, indent=1, default=float))
    figure(ex)


def figure(ex):
    from analysis.style import CATEGORICAL, INK_MUTED, configure_matplotlib, save_figure, style_suptitle, styled_legend
    plt = configure_matplotlib()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    lim = float(np.abs(np.concatenate([ex.d_logp_top1, ex.est_logp, ex.dla_top1])).max()) * 1.2   # no clipping
    for a, (col, sign, title) in zip(ax, [("est_logp", 1, "Attribution patching (total effect)"),
                                         ("dla_top1", -1, "Direct logit attribution (064)")]):
        for i, (b, c) in enumerate(zip(BANDS, CATEGORICAL[:3])):
            d = ex[ex.band == b]
            r, rho = corr(sign * d[col], d.d_logp_top1)
            a.scatter(sign * d[col], d.d_logp_top1, s=7, alpha=0.45, color=c, edgecolor="none", zorder=5 - i,
                      label=f"{b}  (r {r:.2f}, ρ {rho:.2f}, n {len(d)})", rasterized=True)
        a.plot([-lim, lim], [-lim, lim], color=INK_MUTED, lw=1, ls=(0, (4, 3)))
        a.set_xscale("symlog", linthresh=1e-3)
        a.set_yscale("symlog", linthresh=1e-3)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_title(title)
        a.set_xlabel("predicted Δ log p(top-1)  [nats]")
        styled_legend(a, loc="upper left")
    ax[0].set_ylabel("measured Δ log p(top-1) on ablation  [nats]")
    style_suptitle(fig, f"Predicted vs measured ablation effect, every fired target at "
                        f"{ex[['seq', 't']].drop_duplicates().shape[0]} positions ({len(ex):,} ablations)")
    save_figure(fig, FIG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-new", type=int, default=32)
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    if not a.analyse_only:
        compute(a.n_new, a.sanity_only)
    if not a.sanity_only:
        analyse()
