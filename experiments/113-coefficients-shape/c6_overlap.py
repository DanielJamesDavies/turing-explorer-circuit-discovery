"""Weighted vs unweighted member overlap on the TuringLLM lambda sweep (T4), and whether unweighted-only members are
near-duplicates of kept WCM members (SAE decoder cosine).

Pairing (primary): per target, the smallest passing WCM circuit (any W arm) and the smallest passing unweighted circuit
(any U arm), as 109 t6_tl_sweep.py reading (b) (`load` reused: held strong, merge.vacuous excluded, W_0.001 falls back
to the T3 main row). Secondary: same lambda, W_0.001 vs U_0.001, whatever their pass status.
Circuits: T4 sweep/circuits/<arm>/<seed>.pt; a W_0.001 row whose source is "main" reads the T3 main circuit.

Per target pair: share of WCM members inside the unweighted circuit |W & U| / |W|, share of unweighted members
inside WCM |W & U| / |U|, Jaccard |W & U| / |W | U|.
Near-duplicates: for every unweighted-only member u (in U, not in W), the largest cosine between its decoder column
and the decoder columns of the WCM members at the SAME site (TuringLLM SAEs, models/TuringLLM/SAE, decoder.weight
[d_model, d_sae], columns L2-normalised). Baseline: the same statistic for random latents of the same site (uniform
over the site's latents not in W, as many as the site's unweighted-only members, capped at 500 per target and site;
rng seed 20261114). Unweighted-only members at sites where the WCM circuit has no member have no comparison and are
counted separately.

Writes out/run2026_coef_overlap_targets.csv, out/run2026_coef_overlap.csv, out/run2026_coef_overlap_cosine.csv,
paper/tables/run2026_coef_overlap.tex
  python -I -X utf8 experiments/113-coefficients-shape/c6_overlap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import lib113 as C  # noqa: E402
import t6_tl_sweep as T6  # noqa: E402  (109)

L = C.L
SAE = C.ROOT / "models" / "TuringLLM" / "SAE"
CAP = 500
THR = (0.2, 0.3, 0.5, 0.7, 0.9)


def member_sets(arm, seeds, main_seeds):
    """{seed: set of (layer, kind, index)} for the arm; main_seeds read from the T3 main circuits."""
    out = {}
    df = C.members("turingllm", arm)
    for s, g in df[df.seed.isin(seeds)].groupby("seed"):
        out[s] = set(zip(g.layer, g.kind, g["index"]))
    if main_seeds:
        m = C.members("turingllm", "main")
        for s, g in m[m.seed.isin(main_seeds)].groupby("seed"):
            out[s] = set(zip(g.layer, g.kind, g["index"]))
    return out


def pairs(sw, how):
    rows = []
    for s, x in sw.groupby("seed"):
        if how == "smallest":
            w = x[x.weighted & ~x.vac & x["pass"]].sort_values("n")
            u = x[~x.weighted & ~x.vac & x["pass"]].sort_values("n")
        else:
            w = x[(x.arm == "W_0.001") & ~x.vac]
            u = x[(x.arm == "U_0.001") & ~x.vac]
        if len(w) and len(u):
            rows.append(dict(seed=s, band=x.band.iloc[0], w_arm=w.arm.iloc[0], w_src=w.source.iloc[0], w_n=int(w.n.iloc[0]),
                             u_arm=u.arm.iloc[0], u_n=int(u.n.iloc[0]), w_pass=bool(w["pass"].iloc[0]),
                             u_pass=bool(u["pass"].iloc[0])))
    return pd.DataFrame(rows)


def overlap(P):
    sets = {}
    for (arm, src), g in P.groupby(["w_arm", "w_src"]):
        main = list(g.seed) if src == "main" else []
        sw = [] if src == "main" else list(g.seed)
        for s, v in member_sets(arm, sw, main).items():
            sets[("W", s)] = v
    for arm, g in P.groupby("u_arm"):
        for s, v in member_sets(arm, list(g.seed), []).items():
            sets[("U", s)] = v
    out = []
    for r in P.itertuples():
        W, U = sets.get(("W", r.seed)), sets.get(("U", r.seed))
        if W is None or U is None:
            continue
        i = len(W & U)
        out.append(dict(seed=r.seed, n_w=len(W), n_u=len(U), inter=i, w_in_u=i / len(W), u_in_w=i / len(U),
                        jaccard=i / len(W | U), w_subset=(i == len(W))))
    return pd.DataFrame(out), sets


def cosine(P, sets):
    """Per (target, site): unweighted-only vs WCM same-site max cosine, and the random-latent baseline."""
    rng = np.random.default_rng(C.BOOT_SEED)
    by_site = {}
    for r in P.itertuples():
        W, U = sets.get(("W", r.seed)), sets.get(("U", r.seed))
        if W is None or U is None:
            continue
        for (l, k, i) in U - W:
            by_site.setdefault((l, k), {}).setdefault(r.seed, [[], []])[0].append(i)
        for (l, k, i) in W:
            by_site.setdefault((l, k), {}).setdefault(r.seed, [[], []])[1].append(i)
    recs = []
    for (l, k) in sorted(by_site):
        sd = torch.load(SAE / ("sae-%s" % k) / ("sae_%s_layer_%d.pth" % (k, l)), map_location="cpu", weights_only=True)
        D = sd["decoder.weight"].float()
        D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)
        d_sae = D.shape[1]
        for s, (uo, w) in by_site[(l, k)].items():
            if not uo:
                continue
            if not w:
                recs.append(dict(seed=s, layer=l, kind=k, n_uonly=len(uo), no_wcm_site=True))
                continue
            uo = np.array(uo)
            if len(uo) > CAP:
                uo = rng.choice(uo, CAP, replace=False)
            wset = set(w)
            cand = np.setdiff1d(np.arange(d_sae), np.array(w))
            rnd = rng.choice(cand, len(uo), replace=False)
            Dw = D[:, w]
            cu = (D[:, uo].T @ Dw).max(dim=1).values.numpy()
            cr = (D[:, rnd].T @ Dw).max(dim=1).values.numpy()
            # a fresh random pair per sampled latent: the plain random-pair cosine
            rp = rng.choice(d_sae, (len(uo), 2))
            cp = (D[:, rp[:, 0]] * D[:, rp[:, 1]]).sum(0).numpy()
            recs.append(dict(seed=s, layer=l, kind=k, n_uonly=len(by_site[(l, k)][s][0]), n_sampled=len(uo),
                             no_wcm_site=False, n_w_site=len(wset), cu=cu, cr=cr, cp=cp))
        print("site L%d.%s done (%d targets)" % (l, k, len(by_site[(l, k)])), flush=True)
    return recs


def summarize_cos(recs):
    good = [r for r in recs if not r["no_wcm_site"]]
    cu = np.concatenate([r["cu"] for r in good])
    cr = np.concatenate([r["cr"] for r in good])
    cp = np.concatenate([r["cp"] for r in good])
    n_tot = sum(r["n_uonly"] for r in recs)
    n_none = sum(r["n_uonly"] for r in recs if r["no_wcm_site"])
    # per-target shares above thresholds, for target-level CIs
    per = {}
    for r in good:
        p = per.setdefault(r["seed"], {"n": 0, **{"u%g" % t: 0 for t in THR}, **{"r%g" % t: 0 for t in THR}})
        p["n"] += len(r["cu"])
        for t in THR:
            p["u%g" % t] += int((r["cu"] > t).sum())
            p["r%g" % t] += int((r["cr"] > t).sum())
    pt = pd.DataFrame(per).T
    out = dict(uonly_total=n_tot, uonly_no_wcm_site=n_none, sampled=len(cu), targets=len(pt),
               cos_uonly_median=float(np.median(cu)), cos_random_median=float(np.median(cr)),
               cos_pair_median=float(np.median(cp)), cos_uonly_p90=float(np.percentile(cu, 90)),
               cos_random_p90=float(np.percentile(cr, 90)))
    for t in THR:
        for a in ("u", "r"):
            lo, hi = C.cluster_share_ci(pt["%s%g" % (a, t)].values, pt.n.values)
            out["%s_gt%g" % ({"u": "uonly", "r": "random"}[a], t)] = float(pt["%s%g" % (a, t)].sum() / pt.n.sum())
            out["%s_gt%g_lo" % ({"u": "uonly", "r": "random"}[a], t)] = lo
            out["%s_gt%g_hi" % ({"u": "uonly", "r": "random"}[a], t)] = hi
    return out, cu, cr


def main():
    sw, tg = T6.load("turingllm")
    print("sweep: %d targets with rows, %d in list" % (sw.seed.nunique(), len(tg)))
    summary, tex = [], []
    allt = []
    for how, lab in (("smallest", "smallest passing WCM vs smallest passing unweighted"),
                     ("lam1e-3", r"same $\lambda = 10^{-3}$ (W\_0.001 vs U\_0.001, any pass status)")):
        P = pairs(sw, how)
        O, sets = overlap(P)
        O = O.merge(P, on="seed")
        O["pairing"] = how
        allt.append(O)
        r = dict(pairing=how, targets=len(O), n_w_median=O.n_w.median(), n_u_median=O.n_u.median(),
                 w_in_u_median=O.w_in_u.median(), w_in_u_mean=O.w_in_u.mean(), w_in_u_ci=C.boot_ci(O.w_in_u.values),
                 u_in_w_median=O.u_in_w.median(), jaccard_median=O.jaccard.median(), jaccard_mean=O.jaccard.mean(),
                 jaccard_ci=C.boot_ci(O.jaccard.values), w_subset_share=O.w_subset.mean(),
                 w_in_u_ge90=(O.w_in_u >= 0.9).mean())
        if how == "smallest":
            r["w_arms"] = O.w_arm.value_counts().to_dict()
            r["u_arms"] = O.u_arm.value_counts().to_dict()
            cs, cu, cr = summarize_cos(cosine(P, sets))
            r.update(cs)
            pd.DataFrame(dict(cos_uonly=cu, cos_random=cr)).describe(percentiles=[.1, .25, .5, .75, .9, .99]).to_csv(
                C.OUT / (C.PREFIX + "overlap_cosine.csv"))
        summary.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        tex.append([lab, L.tex_int(len(O)), "%s / %s" % (L.tex_int(r["n_w_median"]), L.tex_int(r["n_u_median"])),
                    "%.2f {[}%.2f, %.2f{]}" % (r["w_in_u_mean"], *r["w_in_u_ci"]), "%.0f" % (100 * r["w_subset_share"]),
                    "%.2f {[}%.2f, %.2f{]}" % (r["jaccard_mean"], *r["jaccard_ci"])])
    pd.concat(allt).to_csv(C.OUT / (C.PREFIX + "overlap_targets.csv"), index=False)
    S = pd.DataFrame(summary)
    s0 = summary[0]
    foot = [r"Means over targets with 95\%% bootstrap CI. Unweighted-only members, largest decoder cosine to a WCM "
            r"member at the same site: median %.3f (random latents of the same site: %.3f); above 0.3: %.2f\%% "
            r"(random %.2f\%%); above 0.5: %.2f\%% (random %.2f\%%)." % (
                s0["cos_uonly_median"], s0["cos_random_median"], 100 * s0["uonly_gt0.3"], 100 * s0["random_gt0.3"],
                100 * s0["uonly_gt0.5"], 100 * s0["random_gt0.5"])]
    C.write_table("overlap", tex, ["Pairing", "Targets", "Size WCM / unweighted",
                                   r"WCM members in unweighted", r"WCM $\subseteq$ unweighted (\%)", "Jaccard"],
                  "lrllrl", footer=foot, csv_df=S)


if __name__ == "__main__":
    main()
