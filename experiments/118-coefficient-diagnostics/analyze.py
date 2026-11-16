"""118 analysis (CPU only): reads out/rows.jsonl and out/members/*.npz written by run118.py.

  python -I -X utf8 experiments/118-coefficient-diagnostics/analyze.py
Writes out/report.md (all tables) and out/*.csv.

Pass rule: 062 merge (Z, A, C in [0.75, 1.5], necessity >= 0.9); necessity and vacuity come from the local full
production score of the fitted circuit (alpha-independent). Every comparison is paired on this GPU.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "062-h100-protocol-v1"))
from merge import BAND, NEC_MIN  # noqa: E402

OUT = HERE / "out"
B_BOOT, SEED = 4000, 20261115
FITDIR = "fit4"          # the refit run (bs 4); out/fit is the aborted bs-8 attempt (deep targets spilled GPU memory)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def band_of(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def ok3(d, nec):
    if d is None or nec is None:
        return False
    v = [d.get(f) for f in "ZAC"]
    return all(x is not None and BAND[0] <= x <= BAND[1] for x in v) and nec >= NEC_MIN


def boot(x, stat=np.mean):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(SEED)
    b = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B_BOOT)]
    return np.percentile(b, [2.5, 97.5])


def load():
    rows = []
    for line in open(OUT / "rows.jsonl"):
        r = json.loads(line)
        if "error" in r:
            continue
        rows.append(r)
    return pd.DataFrame(rows).drop_duplicates("seed", keep="last").set_index("seed")


def vac(r):
    ap = r["full_a_pos_tk"]
    if ap is None or ap <= 0:
        return True
    return any((r["full_" + k] or 0) > 0.95 * ap for k in ("e0_tk", "eM_topk_tk", "eN_topk_tk"))


def pass_table(df):
    say("## 1. Pass rates of coefficient variants (paired, local GPU)")
    say()
    df = df[~df.apply(vac, axis=1)].copy()
    df["layer"] = [int(k.split(".")[0]) for k in df.index]
    df["kind"] = [k.split(".")[1] for k in df.index]
    df["band"] = df.layer.map(band_of)
    nec = df.full_phi_sup_blind_tk
    cols = {"fitted (local re-score)": df.red_fitted}
    for k in ("alpha1", "med_global", "med_kind", "med_site", "clip_1.25", "clip_1.5", "clip_2", "clip_3", "cap_hi2"):
        cols[k] = df["var"].map(lambda v, k=k: v.get(k))
    arms = sorted({a for f in df.get("fit", pd.Series(dtype=object)).dropna() for a in f})
    for a in ("fit_global", "fit_kind", "fit_site", "fit_member", "fit_member_c2"):
        if a in arms:
            cols[a] = df["fit"].map(lambda f, a=a: (f or {}).get(a, {}).get("red"))
    P = pd.DataFrame({k: [ok3(d, n) for d, n in zip(v, nec)] for k, v in cols.items()}, index=df.index)
    have_all = P.index
    say("Targets: %d non-vacuous of %d scored. Reduced-scorer check (max |diff| vs the full scorer at fitted alpha): "
        "median %.2g, max %.2g." % (len(df), len(load()), df.red_check.median(), df.red_check.max()))
    full_pass = [ok3(dict(Z=r.full_free0_tk, A=r.full_freeM_topk_tk, C=r.full_freeN_topk_tk), r.full_phi_sup_blind_tk)
                 for r in df.itertuples()]
    say("Full-scorer pass at fitted alpha: %.1f%%; reduced: %.1f%% (agreement %d / %d)." % (
        100 * np.mean(full_pass), 100 * P.iloc[:, 0].mean(), int((np.array(full_pass) == P.iloc[:, 0].values).sum()),
        len(P)))
    say()
    p_fit, p_a1 = P.iloc[:, 0].mean(), P["alpha1"].mean()
    say("| variant | pass % [95% CI] | Δ vs fitted (pp) [CI] | gap recovered | median Z / A / C |")
    say("|---|---|---|---|---|")
    meds = {}
    for k in P.columns:
        lo, hi = boot(P[k].astype(float))
        d = P[k].astype(float) - P.iloc[:, 0].astype(float)
        dlo, dhi = boot(d)
        rec = (P[k].mean() - p_a1) / (p_fit - p_a1) if p_fit > p_a1 else np.nan
        zac = [np.nanmedian([(v or {}).get(f, np.nan) if (v or {}).get(f) is not None else np.nan for v in cols[k]])
               for f in "ZAC"]
        meds[k] = zac
        say("| %s | %.1f [%.1f, %.1f] | %+.1f [%+.1f, %+.1f] | %s | %.2f / %.2f / %.2f |" % (
            k, 100 * P[k].mean(), 100 * lo, 100 * hi, 100 * d.mean(), 100 * dlo, 100 * dhi,
            "%.0f%%" % (100 * rec) if np.isfinite(rec) else "-", *zac))
    say()
    say("By band (pass %):")
    say()
    show = [c for c in P.columns if c in ("fitted (local re-score)", "alpha1", "med_site", "fit_global", "fit_kind",
                                          "fit_site", "fit_member", "fit_member_c2", "clip_2")]
    say("| band | n | " + " | ".join(show) + " |")
    say("|---|---|" + "---|" * len(show))
    for g, idx in df.groupby("band").groups.items():
        say("| %s | %d | " % (g, len(idx)) + " | ".join("%.1f" % (100 * P.loc[idx, c].mean()) for c in show) + " |")
    for g, idx in df.groupby("kind").groups.items():
        say("| %s | %d | " % (g, len(idx)) + " | ".join("%.1f" % (100 * P.loc[idx, c].mean()) for c in show) + " |")
    say()
    # failure side for the tied arms
    for a in ("fit_site", "fit_global", "alpha1"):
        if a not in cols:
            continue
        low = hi_ = 0
        for v, n, p in zip(cols[a], nec, P[a]):
            if p or v is None:
                continue
            vals = [v.get(f) for f in "ZAC"]
            low += any(x is not None and x < BAND[0] for x in vals) or any(x is None for x in vals)
            hi_ += any(x is not None and x > BAND[1] for x in vals)
        say("%s failures: %d with some score below 0.75 (or undefined), %d with some score above 1.5." % (a, low, hi_))
    if "fit_member" in arms:
        lt = df["fit"].map(lambda f: f)
        tl = pd.DataFrame({a: lt.map(lambda f, a=a: (f or {}).get(a, {}).get("loss_tr")) for a in arms})
        tl["fitted"] = df.loss_tr_fitted
        tl["alpha1"] = df.loss_tr_alpha1
        say()
        say("Training loss of the fitter's objective (median over targets): " + ", ".join(
            "%s %.3f" % (c, tl[c].median()) for c in tl.columns))
        nps = df["fit"].map(lambda f: (f or {}).get("fit_site", {}).get("n_par"))
        say("Per-site arm: median %d coefficients per circuit (members: median %d)." % (nps.median(), df.n.median()))
    P.to_csv(OUT / "pass_variants.csv")
    return df


def fit_table(df, FITDIR=FITDIR, title="1b. Refits (OBJ=pre)"):
    """The refit arms (out/<FITDIR>/rows.jsonl, the balanced subset), paired with the same targets' no-refit variants."""
    p = OUT / FITDIR / "rows.jsonl"
    if not p.exists():
        return
    F = pd.DataFrame([json.loads(l) for l in open(p) if '"error"' not in l]).drop_duplicates("seed", keep="last")
    F = F.set_index("seed")
    F = F[F.index.isin(df.index)]
    say("## %s: members fixed, only the coefficients refitted (from alpha = 1 unless 'warm')" % title)
    say()
    nec = df.loc[F.index, "full_phi_sup_blind_tk"]
    cols = {"fitted (production alpha)": df.loc[F.index, "red_fitted"],
            "alpha1": df.loc[F.index, "var"].map(lambda v: v["alpha1"]),
            "med_site": df.loc[F.index, "var"].map(lambda v: v["med_site"]),
            "clip_2": df.loc[F.index, "var"].map(lambda v: v["clip_2"]),
            "clip_3": df.loc[F.index, "var"].map(lambda v: v["clip_3"])}
    arms = [a for a in ("fit_global", "fit_kind", "fit_site", "fit_member", "fit_member_c2", "fit_member_warm")
            if a in F.fit.iloc[0]]
    for a in arms:
        cols[a] = F.fit.map(lambda f, a=a: f[a]["red"])
    P = pd.DataFrame({k: [ok3(d, n) for d, n in zip(v, nec)] for k, v in cols.items()}, index=F.index)
    p_fit, p_a1 = P.iloc[:, 0].mean(), P["alpha1"].mean()
    tk = "obj" in F and (F.obj == "tk").all()
    n_plan = 54 if tk else 108
    say("Targets: %d (balanced: %d per band x kind cell, interleaved; %s). Fitter: Adam, %s steps, cosine lr from 0.1, "
        "4 training rows per step, loss = sum over Z / A / C of (ratio - 1)^2 on %s of the 48 strongest-training rows; "
        "held-out rows are never seen." % (
            len(F), n_plan // 9, "complete" if len(F) >= n_plan else "partial", len(F.traces.iloc[0][arms[0]]),
            "the post-Top-K read (straight-through) plus the censor hinge" if tk else "the relu pre-activation"))
    say()
    npar = {"fitted (production alpha)": F.n.median(), "alpha1": 0, "med_site": np.nan, "clip_2": F.n.median(),
            "clip_3": F.n.median()}
    tl = {"fitted (production alpha)": df.loc[F.index, "loss_tr_fitted"].median(),
          "alpha1": df.loc[F.index, "loss_tr_alpha1"].median()}
    for a in arms:
        npar[a] = F.fit.map(lambda f, a=a: f[a]["n_par"]).median()
        tl[a] = F.fit.map(lambda f, a=a: f[a].get("loss_tr_tk", f[a]["loss_tr"])).median()
    if "loss_tr_tk_fitted" in F:
        tl["fitted (production alpha)"] = F.loss_tr_tk_fitted.median()
    say("Training loss column: the %s-read objective on all 48 training rows." % (
        "post-Top-K" if "loss_tr_tk_fitted" in F else "pre-activation"))
    say()
    say("| arm | coefficients per circuit (median) | pass % [95% CI] | Δ vs production alpha (pp) [CI] | gap recovered | "
        "median Z / A / C | training loss (median) |")
    say("|---|---|---|---|---|---|---|")
    for k in P.columns:
        lo, hi = boot(P[k].astype(float))
        d = P[k].astype(float) - P.iloc[:, 0].astype(float)
        dlo, dhi = boot(d)
        rec = (P[k].mean() - p_a1) / (p_fit - p_a1) if p_fit > p_a1 else np.nan
        zac = [np.nanmedian([np.nan if (v or {}).get(f) is None else v[f] for v in cols[k]]) for f in "ZAC"]
        say("| %s | %s | %.1f [%.1f, %.1f] | %+.1f [%+.1f, %+.1f] | %.0f%% | %.2f / %.2f / %.2f | %s |" % (
            k, "-" if not np.isfinite(npar.get(k, np.nan)) else "%d" % npar[k], 100 * P[k].mean(), 100 * lo, 100 * hi,
            100 * d.mean(), 100 * dlo, 100 * dhi, 100 * rec, *zac,
            "%.3f" % tl[k] if k in tl else "-"))
    say()
    for a in arms:
        tr = F.traces.map(lambda t, a=a: t[a])
        last = tr.map(lambda t: np.mean(t[-5:]))
        mid = tr.map(lambda t: np.mean(t[len(t) // 2 - 2:len(t) // 2 + 3]))
        say("- %s: alpha median %.3f (circuit min / max, medians: %.3f / %.3f); training-loss trace, mean of the "
            "last 5 steps vs the middle 5: median ratio %.3f." % (
                a, F.fit.map(lambda f, a=a: f[a]["alpha_med"]).median(), F.fit.map(lambda f, a=a: f[a]["alpha_min"]).median(),
                F.fit.map(lambda f, a=a: f[a]["alpha_max"]).median(), (last / mid.clip(lower=1e-9)).median()))
    F["band"] = [band_of(int(k.split(".")[0])) for k in F.index]
    F["kind"] = [k.split(".")[1] for k in F.index]
    say()
    show = list(P.columns)
    say("| cell | n | " + " | ".join(show) + " |")
    say("|---|---|" + "---|" * len(show))
    for by in ("band", "kind"):
        for g, idx in F.groupby(by).groups.items():
            say("| %s | %d | " % (g, len(idx)) + " | ".join("%.0f" % (100 * P.loc[idx, c].mean()) for c in show) + " |")
    say()
    # site scalars vs SAE error
    S = []
    for key in F.index:
        pf, pm = OUT / FITDIR / "members" / ("%s.npz" % key), OUT / "members" / ("%s.npz" % key)
        if not (pf.exists() and pm.exists()) or "fit_site" not in arms:
            continue
        zf, zm = np.load(pf), np.load(pm)
        lab = np.array(["%d.%s" % (l, k) for l, k in zip(zm["site_layer"], zm["site_kind"])])
        for j, s in enumerate(zm["sites"].tolist()):
            m = lab == s
            S.append(dict(seed=key, site=s, scalar=float(zf["alpha_fit_site"][m][0]), med=float(np.median(zm["alpha"][m])),
                          relerr=float(zm["relerr"][:, j].mean()), n_m=int(m.sum())))
    if S:
        S = pd.DataFrame(S)
        S.to_csv(OUT / "fit_site_scalars.csv", index=False)
        say("Per-site fitted scalars (%d sites): median %.3f (p10 %.3f, p90 %.3f); Spearman with the site's median "
            "production alpha %.3f, with the site's SAE relative error %.3f." % (
                len(S), S.scalar.median(), S.scalar.quantile(.1), S.scalar.quantile(.9),
                spearmanr(S.scalar, S.med).correlation, spearmanr(S.scalar, S.relerr).correlation))
        say()
    if FITDIR != "fit4" and (OUT / "fit4" / "rows.jsonl").exists():
        G = pd.DataFrame([json.loads(l) for l in open(OUT / "fit4" / "rows.jsonl") if '"error"' not in l])
        G = G.drop_duplicates("seed", keep="last").set_index("seed")
        both = [k for k in F.index if k in G.index]
        say("Paired with the OBJ=pre run on the same %d targets (pass %%, [95%% CI] of the tk - pre difference):" % len(both))
        say()
        for a in ("fit_global", "fit_site", "fit_member"):
            if a not in arms or a not in G.fit.iloc[0]:
                continue
            pp = np.array([ok3(G.loc[k, "fit"][a]["red"], nec[k]) for k in both], dtype=float)
            pt = np.array([ok3(F.loc[k, "fit"][a]["red"], nec[k]) for k in both], dtype=float)
            lo, hi = boot(pt - pp)
            say("- %s: pre %.1f, tk %.1f, difference %+.1f pp [%+.1f, %+.1f]" % (
                a, 100 * pp.mean(), 100 * pt.mean(), 100 * (pt - pp).mean(), 100 * lo, 100 * hi))
        say()
    if "fit_member_warm" in arms:
        pf = P.iloc[:, 0]
        say("Warm start: of %d targets passing at production alpha, %d still pass after the warm refit; %d of %d failing "
            "ones pass after it." % (int(pf.sum()), int((pf & P.fit_member_warm).sum()), int((~pf & P.fit_member_warm).sum()),
                                     int((~pf).sum())))
        say()
    if "fit_member" in arms and tk:
        lt = F.fit.map(lambda f: f["fit_member"].get("loss_tr_tk"))
        zz = F.fit.map(lambda f: f["fit_member"]["red"].get("Z"))
        say("fit_member: training tk-loss median %.4f vs production %.4f, yet held-out median Z %.2f; failures of "
            "fit_member among targets that pass at production alpha: %d." % (
                lt.median(), F.loss_tr_tk_fitted.median() if "loss_tr_tk_fitted" in F else np.nan,
                np.nanmedian(zz.astype(float)), int((P.iloc[:, 0] & ~P.fit_member).sum())))
        say()
    P.to_csv(OUT / ("pass_refits_%s.csv" % FITDIR))


def members_frame(df):
    recs = []
    for key in df.index:
        p = OUT / "members" / ("%s.npz" % key)
        if p.exists():
            recs.append((key, dict(np.load(p, allow_pickle=False))))
    return recs


def contributions(recs):
    say("## 2. Effective size by contribution (alpha x activation x decoder norm, at the anchor, held-out rows)")
    say()
    out = []
    for key, z in recs:
        n_tr = int(z["n_train"])
        a = z["alpha"]
        act = z["a_anchor"][n_tr:]                         # [16, n]
        actm = z["a_max"][n_tr:]
        c = a[None, :] * act * z["dec_norm"][None, :]
        cm = a[None, :] * actm * z["dec_norm"][None, :]
        n = len(a)
        r = dict(seed=key, n=n, active_anchor=float((act > 0).mean()), active_any=float((actm > 0).mean()),
                 member_ever_active_anchor=float((act > 0).any(0).mean()))
        for tag, M in (("anchor", c), ("posmax", cm)):
            tot = M.mean(0)
            s = np.sort(tot)[::-1]
            cs = np.cumsum(s) / max(s.sum(), 1e-12)
            for q in (0.5, 0.8, 0.9):
                k = int(np.searchsorted(cs, q) + 1)
                r["N%d_%s" % (int(q * 100), tag)] = k
                r["f%d_%s" % (int(q * 100), tag)] = k / n
            # per-context N80 and the overlap of per-context top sets
            ks, sets = [], []
            for b in range(M.shape[0]):
                sb = np.sort(M[b])[::-1]
                if sb.sum() <= 0:
                    continue
                cb = np.cumsum(sb) / sb.sum()
                kb = int(np.searchsorted(cb, 0.8) + 1)
                ks.append(kb)
                sets.append(set(np.argsort(-M[b])[:kb].tolist()))
            r["N80_ctx_med_%s" % tag] = float(np.median(ks)) if ks else np.nan
            r["N80_ctx_cv_%s" % tag] = float(np.std(ks) / np.mean(ks)) if ks else np.nan
            jac = [len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
            r["top80_jaccard_%s" % tag] = float(np.median(jac)) if jac else np.nan
            # per-member spread over contexts (members with nonzero mean contribution)
            mu, sd = M.mean(0), M.std(0)
            nz = mu > 0
            r["member_cv_med_%s" % tag] = float(np.median(sd[nz] / mu[nz])) if nz.any() else np.nan
            # share of the pooled contribution carried by members with alpha > 1 / alpha > 2; vs alpha = 1 contributions
            r["share_alpha_gt1_%s" % tag] = float(tot[a > 1].sum() / max(tot.sum(), 1e-12))
            r["share_alpha_gt2_%s" % tag] = float(tot[a > 2].sum() / max(tot.sum(), 1e-12))
            r["frac_alpha_gt2_%s" % tag] = float((a > 2).mean())
            r["spearman_alpha_contrib1_%s" % tag] = spearmanr(a, tot / np.maximum(a, 1e-12)).correlation
        out.append(r)
    E = pd.DataFrame(out).set_index("seed")
    E["band"] = [band_of(int(k.split(".")[0])) for k in E.index]
    E.to_csv(OUT / "contrib_targets.csv")
    say("Targets: %d. Median over targets (IQR):" % len(E))
    say()
    say("| quantity | at the anchor | max over positions <= anchor |")
    say("|---|---|---|")
    def q(col):
        s = E[col]
        return "%.3g (%.3g-%.3g)" % (s.median(), s.quantile(.25), s.quantile(.75))
    for lab, base in (("members carrying 50%", "N50"), ("members carrying 80%", "N80"), ("members carrying 90%", "N90"),
                      ("share of members for 50%", "f50"), ("share of members for 80%", "f80"),
                      ("share of members for 90%", "f90"), ("per-context N80", "N80_ctx_med"),
                      ("CV of per-context N80", "N80_ctx_cv"), ("Jaccard of per-context top-80% sets", "top80_jaccard"),
                      ("per-member CV of contribution over contexts", "member_cv_med"),
                      ("contribution share of alpha > 1 members", "share_alpha_gt1"),
                      ("contribution share of alpha > 2 members", "share_alpha_gt2"),
                      ("Spearman(alpha, natural contribution)", "spearman_alpha_contrib1")):
        say("| %s | %s | %s |" % (lab, q(base + "_anchor"), q(base + "_posmax")))
    say("| members active at the anchor (share of member x context) | %s | %s |" % (q("active_anchor"), q("active_any")))
    say("| members active at the anchor in >= 1 held-out context | %s | |" % q("member_ever_active_anchor"))
    say("| circuit size n | %s | |" % q("n"))
    say("| alpha > 2 share of members | %s | |" % q("frac_alpha_gt2_anchor"))
    say()
    say("By band (median f80 anchor / posmax; N80 anchor): " + "; ".join(
        "%s %.3f / %.3f; %d" % (g, d.f80_anchor.median(), d.f80_posmax.median(), d.N80_anchor.median())
        for g, d in E.groupby("band")))
    say()
    return E


def off_manifold(recs):
    say("## 3. Off-manifold check (fitted alpha, held-out rows)")
    say()
    # References. (1) LOCAL: each member's largest natural value over the target's 64 activating rows (positions <=
    # anchor; the held-out rows are among them). top_ctx.pt was meant as a corpus maximum but its ctx_seq_val is a
    # per-sequence MEAN (10-25x below the target peaks), so it is not used. (2) CORPUS z: the member's active-value
    # mean and sd over the corpus pass (outputs/latent_stats.pt: mean over active positions, m2 / active_count).
    import torch
    ls = torch.load(ROOT / "outputs" / "latent_stats.pt", weights_only=False, map_location="cpu")
    mu_all = ls["mean"].numpy()
    sd_all = np.sqrt(ls["m2"].numpy() / np.maximum(ls["active_count"].numpy(), 1))
    KI = {"attn": 0, "mlp": 1, "resid": 2}
    out = []
    for key, z in recs:
        n_tr = int(z["n_train"])
        a = z["alpha"]
        comp = z["site_layer"] * 3 + np.array([KI[k] for k in z["site_kind"]])
        mu, sd = mu_all[comp, z["latent"]], np.maximum(sd_all[comp, z["latent"]], 1e-9)
        loc = z["a_max"].max(0)                                    # [n] local natural max over 64 rows
        act = z["a_max"][n_tr:]
        r = dict(seed=key)
        r["n_member_ctx_active"] = int((act > 0).sum())
        tags = [("nat_scaled", a[None, :] * act), ("Z", z["vZ_max"]), ("A", z["vA_max"]), ("natural", act)]
        pa1 = OUT / "diag_a1" / "members" / ("%s.npz" % key)
        if pa1.exists():
            z1 = np.load(pa1)
            tags += [("Z_a1", z1["vZ_max_a1"]), ("A_a1", z1["vA_max_a1"])]
            for t_ in ("Z", "A"):
                ratio1 = z1["recon_%s_a1" % t_] / np.maximum(z["recon_nat"][n_tr:], 1e-9)
                r["recon_ratio_med_%s_a1" % t_] = float(np.median(ratio1))
                r["recon_ratio_p90_%s_a1" % t_] = float(np.percentile(ratio1, 90))
                r["recon_gt_p99_%s_a1" % t_] = float((z1["recon_%s_a1" % t_] > z["recon_p99"][None, :]).mean())
        for tag, V in tags:
            act_ = V > 0
            den = max(act_.sum(), 1)
            r["gt_max_%s" % tag] = float((V > loc[None, :] * (1 + 1e-6)).sum() / den)
            r["gt_max1.5_%s" % tag] = float((V > 1.5 * loc[None, :]).sum() / den)
            r["gt_64th_%s" % tag] = float(((V - mu[None, :]) / sd[None, :] > 6).sum() / den)
            r["any_gt_max_%s" % tag] = float((V > 1.5 * loc[None, :]).any(0).mean())
        rn = z["recon_nat"][n_tr:]
        for tag in ("Z", "A"):
            ratio = z["recon_%s" % tag] / np.maximum(rn, 1e-9)
            r["recon_ratio_med_%s" % tag] = float(np.median(ratio))
            r["recon_ratio_p90_%s" % tag] = float(np.percentile(ratio, 90))
            r["recon_gt_p99_%s" % tag] = float((z["recon_%s" % tag] > z["recon_p99"][None, :]).mean())
        r["recon_gt_p99_natural"] = float((rn > z["recon_p99"][None, :]).mean())
        out.append(r)
    O = pd.DataFrame(out).set_index("seed")
    O.to_csv(OUT / "offmanifold_targets.csv")
    say("Targets: %d. 'Local max' = the member's largest natural value over the target's 64 activating rows (train + "
        "held-out, positions <= anchor). 'z > 6' = more than 6 corpus sd above the member's corpus active mean "
        "(latent_stats). Shares are over active (member, held-out context) pairs; value = max over positions <= anchor. "
        "Median over targets (mean in brackets)." % len(O))
    say()
    tg = ["natural", "nat_scaled", "Z", "A"] + (["Z_a1", "A_a1"] if "gt_max_Z_a1" in O else [])
    lab_ = {"natural": "natural (unscaled)", "nat_scaled": "natural x alpha", "Z": "Z run, fitted alpha",
            "A": "A run, fitted alpha", "Z_a1": "Z run, alpha = 1", "A_a1": "A run, alpha = 1"}
    say("| quantity | " + " | ".join(lab_[t] for t in tg) + " |")
    say("|---|" + "---|" * len(tg))
    for lab, base in (("share above the local natural max", "gt_max"), ("share above 1.5 x local max", "gt_max1.5"),
                      ("share with corpus z > 6", "gt_64th"),
                      ("share of members ever above 1.5 x local max", "any_gt_max")):
        say("| %s | " % lab + " | ".join("%.4f (%.4f)" % (O["%s_%s" % (base, t)].median(), O["%s_%s" % (base, t)].mean())
                                          for t in tg) + " |")
    if "recon_ratio_med_Z_a1" in O:
        say()
        say("Paired (fitted alpha - alpha = 1), n = %d targets with both:" % int(O.gt_max_Z_a1.notna().sum()))
        say()
        for base, lab in (("gt_max", "above the local max"), ("gt_max1.5", "above 1.5 x local max"),
                          ("gt_64th", "corpus z > 6")):
            for t_ in ("Z", "A"):
                d = (O["%s_%s" % (base, t_)] - O["%s_%s_a1" % (base, t_)]).dropna()
                lo, hi = boot(d.values)
                sh = O["%s_%s_a1" % (base, t_)].mean() / max(O["%s_%s" % (base, t_)].mean(), 1e-12)
                say("- %s, %s run: mean difference %+.4f [%+.4f, %+.4f]; alpha = 1 share of the fitted-alpha mean %.2f" % (
                    lab, t_, d.mean(), lo, hi, sh))
        say()
        say("alpha = 1 control: edited / natural reconstruction norm Z median %.3f (p90 %.3f), A median %.3f (p90 %.3f); "
            "share above the natural p99: Z %.4f, A %.4f." % (
                O.recon_ratio_med_Z_a1.median(), O.recon_ratio_p90_Z_a1.median(), O.recon_ratio_med_A_a1.median(),
                O.recon_ratio_p90_A_a1.median(), O.recon_gt_p99_Z_a1.median(), O.recon_gt_p99_A_a1.median()))
    say()
    say("Site reconstruction norm at the anchor, edited / natural (same context): Z run median %.3f (p90 %.3f), "
        "A run median %.3f (p90 %.3f). Share of (site, context) above the site's natural p99 reconstruction norm: "
        "Z %.4f, A %.4f, natural %.4f (medians over targets)." % (
            O.recon_ratio_med_Z.median(), O.recon_ratio_p90_Z.median(), O.recon_ratio_med_A.median(),
            O.recon_ratio_p90_A.median(), O.recon_gt_p99_Z.median(), O.recon_gt_p99_A.median(),
            O.recon_gt_p99_natural.median()))
    say()
    return O


def gain_vs_error(recs):
    say("## 4. Gain vs SAE reconstruction error")
    say()
    rows = []
    for key, z in recs:
        sites = z["sites"].tolist()
        lab = np.array(["%d.%s" % (l, k) for l, k in zip(z["site_layer"], z["site_kind"])])
        re_ = z["relerr"]                                           # [64, n_sites]
        tl = int(key.split(".")[0])
        for j, s in enumerate(sites):
            m = lab == s
            if not m.any():
                continue
            rows.append(dict(seed=key, site=s, layer=int(s.split(".")[0]), kind=s.split(".")[1], dist=tl - int(s.split(".")[0]),
                             n_m=int(m.sum()), alpha_med=float(np.median(z["alpha"][m])),
                             alpha_mean_log=float(np.mean(np.log(z["alpha"][m]))),
                             relerr=float(np.mean(re_[:, j])),
                             fit_site=float(z["alpha_fit_site"][m][0]) if "alpha_fit_site" in z else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "gain_vs_error_sites.csv", index=False)
    say("(target, member site) pairs: %d over %d targets. relerr = mean over the 64 activating rows of "
        "||x - x_hat|| / ||x|| at the anchor." % (len(S), S.seed.nunique()))
    say()
    say("- Pooled Spearman(site median alpha, relerr): %.3f; with the member-weighted log-mean alpha: %.3f." % (
        spearmanr(S.alpha_med, S.relerr).correlation, spearmanr(S.alpha_mean_log, S.relerr).correlation))
    w = [spearmanr(d.alpha_med, d.relerr).correlation for _, d in S.groupby("seed") if len(d) >= 5]
    say("- Within-circuit Spearman (circuits with >= 5 member sites, n = %d): median %.3f (IQR %.3f to %.3f)." % (
        len(w), np.nanmedian(w), np.nanpercentile(w, 25), np.nanpercentile(w, 75)))
    wk = []
    for (_, kd), d in S.groupby(["layer", "kind"]):
        if len(d) >= 10:
            wk.append((len(d), spearmanr(d.alpha_med, d.relerr).correlation))
    say("- Within the same (layer, kind) site across circuits (relerr varies only by context): n-weighted mean "
        "Spearman %.3f over %d sites." % (np.average([c for _, c in wk], weights=[n for n, _ in wk]), len(wk)))
    say("- Site relerr by kind (median): " + ", ".join("%s %.3f" % (k, d.relerr.median()) for k, d in S.groupby("kind")))
    say("- Site median alpha by kind: " + ", ".join("%s %.3f" % (k, d.alpha_med.median()) for k, d in S.groupby("kind")))
    q = pd.qcut(S.relerr, 5, labels=False, duplicates="drop")
    say("- Site median alpha by relerr quintile: " + ", ".join(
        "Q%d (relerr %.3f) %.3f" % (i + 1, S.relerr[q == i].median(), S.alpha_med[q == i].median()) for i in sorted(q.unique())))
    say("- Site median alpha by layer distance: " + ", ".join(
        "%d: %.3f" % (d, g.alpha_med.median()) for d, g in S.groupby(S.dist.clip(upper=8))))
    if S.fit_site.notna().any():
        say("- Per-site fitted scalar vs relerr: Spearman %.3f; vs the site's median fitted alpha: %.3f." % (
            spearmanr(S.fit_site, S.relerr, nan_policy="omit").correlation,
            spearmanr(S.fit_site, S.alpha_med, nan_policy="omit").correlation))
    # member level: alpha vs the member's natural activation level relative to its corpus max
    mrow = []
    for key, z in recs:
        n_tr = int(z["n_train"])
        act = z["a_anchor"][n_tr:].mean(0)
        mrow.append(pd.DataFrame(dict(alpha=z["alpha"], rel=act / np.maximum(z["a_max"].max(0), 1e-9), active=act > 0)))
    Mm = pd.concat(mrow)
    a_ = Mm[Mm.active]
    say("- Member level: Spearman(alpha, mean held-out anchor activation / local natural max) %.3f over %d active members; "
        "alpha median for members inactive at every held-out anchor %.3f vs active %.3f." % (
            spearmanr(a_.alpha, a_.rel).correlation, len(a_), Mm[~Mm.active].alpha.median(), a_.alpha.median()))
    say()
    return S


def main():
    df = load()
    say("# 118 results (auto-generated by analyze.py)")
    say()
    d2 = pass_table(df)
    fit_table(d2)
    fit_table(d2, "fit_tk", "1c. Refits (OBJ=tk: post-Top-K read + censor hinge)")
    recs = members_frame(d2)
    contributions(recs)
    off_manifold(recs)
    gain_vs_error(recs)
    (OUT / "report.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
