"""098: CircuitLasso (Yin et al. 2026) as a ranked-members baseline in the DAN-159 method comparison (083).

Per target, on the target's training contexts (the rows every 083 discovery uses, asserted through
cmp_common.training_contexts) plus the contrast-train rows:
  1. one clean forward pass. Each (context, position) is one sample: the target's pre-activation there
     (x @ W_enc[i] + b_eff[i], the quantity IG / AtP attribute) and every upstream site's post-Top-K code at the SAME
     position (sparse);
  2. the Lasso of lasso_core (columns scaled to unit l2 norm, unpenalised intercept, FISTA, warm-started geometric
     lambda path);
  3. the natural lambda by the paper's rule adapted: the lambda minimising validation MSE on a context-level 3:1 split
     of these training rows (the held-out evaluation sets are never touched);
  4. member sets: global top-n by |coefficient| at each WCM size (the largest path lambda whose support reaches n);
     per-site top-c at the reference circuit's per-site counts; the support at the natural lambda.
Scored with driver.Runner.score on held-out strongest, the same scorer every 083 row uses:
  attr/rows.lasso.<shard>.jsonl     method "lasso", ranking coef | std, size_arm W_* (global top-n, alpha = 1) and
                                    ranking "cv", size_arm "natural" (the support at the cross-validated lambda)
  fitted/rows.lasso.<shard>.jsonl   method "fitted_lasso", ranking coef | std, variant a1 | fit: the per-site counts at
                                    alpha = 1 and with coefficients fitted exactly as DAN-43 (attr_fitted.py's fit)
  lasso/diag.<shard>.jsonl          R^2, supports, lambdas, path cost, timings
  rankings/lasso/<key>.<hash>.pt    every member set + the coefficients at the natural lambda (report.py reads it)
table2.py reads the attr / fitted rows like any other method (METHOD_LABEL "lasso", SOURCE_LABEL "lasso").

  wsl: WCM_VERSION=v2 CTX_DIR=... REF_RUN=... OUT083=... TARGETS=... PYTHONPATH=src .venv/bin/python -u \
       experiments/098-circuitlasso-baseline/lasso_rows.py
  env: LASSO_RANKINGS (coef,std)  FIT_RANKINGS (coef,std; "" = no per-site rows)  FIT (1: also the fitted variant)
       SIZES (W_all)  NATURAL (1)  PER_DECADE (20)  MIN_RATIO (1e-4)  TOL (1e-6)  MAX_ITER (3000)  SUPPORT_CAP (8192)
       CV_FRAC (0.25)  POS_FROM (0: first position used as a sample)  STEPS (driver)  TARGETS LIMIT SHARD OUT083
Resumable: the Lasso payload is cached per (target, training contexts); rows are keyed like the 083 rows.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
E083 = HERE.parent / "083-v1-comparison"
for p in (str(E083), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import cmp_common as C  # noqa: E402
import lasso_core as LC  # noqa: E402

LASSO_RANKINGS = [r for r in os.environ.get("LASSO_RANKINGS", "coef,std").split(",") if r]
FIT_RANKINGS = [r for r in os.environ.get("FIT_RANKINGS", "coef,std").split(",") if r]
FIT = os.environ.get("FIT", "1") == "1"
NATURAL = os.environ.get("NATURAL", "1") == "1"
PER_DECADE = int(os.environ.get("PER_DECADE", 20))
MIN_RATIO = float(os.environ.get("MIN_RATIO", 1e-4))
TOL = float(os.environ.get("TOL", 1e-6))
MAX_ITER = int(os.environ.get("MAX_ITER", 3000))
SUPPORT_CAP = int(os.environ.get("SUPPORT_CAP", 8192))
CV_FRAC = float(os.environ.get("CV_FRAC", 0.25))
CV_PATIENCE = int(os.environ.get("CV_PATIENCE", 10))      # stop the CV path this many steps past the minimum
POS_FROM = int(os.environ.get("POS_FROM", 0))
STEPS = int(os.environ["STEPS"]) if os.environ.get("STEPS") else None
ATTR_KEYS = ("seed", "method", "ranking", "size_arm", "ctx_dir")
FIT_KEYS = ("seed", "method", "ranking", "variant", "ctx_dir")
VERSION = "098-v1"                                         # bump to invalidate cached payloads


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), C.TAG, s), flush=True)


# ---------------------------------------------------------------------------------------------- data
def contrast_train(rec):
    """The contrast-train rows: rec["neg"]'s first n - round(n * holdout_frac) rows (the engine's and the scorer's
    3:1 slice; protocol_contexts.eval_contrast keeps the same slice)."""
    from circuit.protocol_contexts import _engine_holdout
    neg = rec["neg"]
    n = int(neg.shape[0])
    return neg[:max(1, n - int(round(n * _engine_holdout(None))))]


@torch.no_grad()
def collect(G, key, tokens, batch=16):
    """One clean forward over `tokens` [N, T]. Returns sites, the COO triplets of the upstream post-Top-K codes (row =
    context * T + position, col = site_index * d_sae + latent), the target's pre-activation y [N * T] and its
    post-Top-K activation post [N * T]."""
    from eval.ablation_faithfulness import upstream_sites
    from sae.dense import target_latent_activations
    inference, bank = G["inference"], G["bank"]
    layer, kind, i = C.parse(key)
    sites = C.site_order(G["KINDS"], upstream_sites(bank, layer, kind))
    sidx = {s: j for j, s in enumerate(sites)}
    k2i = {k: n for n, k in enumerate(G["KINDS"])}
    sae = bank.saes[kind][layer]
    w = sae.encoder.weight[i].detach().float()
    b = sae._get_bias_eff()[i].detach().float()
    d = int(bank.d_sae)
    N, T = int(tokens.shape[0]), int(tokens.shape[1])
    R_, C_, V_, Y, P = [], [], [], [], []
    inference.disable_compile()
    try:
        for s0 in range(0, N, batch):
            tk = tokens[s0:s0 + batch]
            Bc = int(tk.shape[0])
            base = (torch.arange(Bc, device=tk.device)[:, None] + s0) * T + torch.arange(T, device=tk.device)[None]

            def hook(layer_idx, activations):
                for kd in G["KINDS"]:
                    s = (layer_idx, kd)
                    if s != (layer, kind) and s not in sidx:
                        continue
                    x = activations[k2i[kd]]
                    ta, ti = bank.encode(x, kd, layer_idx)                     # [B, T, k]
                    if s == (layer, kind):
                        Y.append((x.float() @ w.to(x.device) + b.to(x.device)).reshape(-1).cpu())
                        P.append(target_latent_activations(ta, ti, i).float().reshape(-1).cpu())
                    else:
                        kk = ta.shape[-1]
                        R_.append(base.to(ta.device)[..., None].expand(-1, -1, kk).reshape(-1).cpu())
                        C_.append((ti.long() + sidx[s] * d).reshape(-1).cpu())
                        V_.append(ta.float().reshape(-1).cpu())
            inference.forward(tk, activations_callback=hook, return_activations=False, tokenize_final=False)
    finally:
        inference.enable_compile()
    return dict(sites=sites, rows=torch.cat(R_), cols=torch.cat(C_), vals=torch.cat(V_), y=torch.cat(Y),
                post=torch.cat(P), N=N, T=T, d_sae=d)


def sample_mask(N, T, pos_from=POS_FROM):
    m = torch.ones(N, T, dtype=torch.bool)
    m[:, :pos_from] = False
    return m.reshape(-1)


def subset(data, keep_samples):
    """COO triplets and y restricted to the samples in keep_samples ([N*T] bool), rows renumbered 0..n-1."""
    new = torch.full((keep_samples.numel(),), -1, dtype=torch.long)
    new[keep_samples] = torch.arange(int(keep_samples.sum()))
    r = new[data["rows"]]
    ok = r >= 0
    return r[ok], data["cols"][ok], data["vals"][ok], data["y"][keep_samples]


# ---------------------------------------------------------------------------------------------- the Lasso
def cv_lambda(data, lams, ctx_val, dev, seed):
    """The natural lambda (the paper picks lambda by held-out prediction; D.2): fit the path on the training fold of
    contexts, return (index into lams of the minimum validation MSE, the curve). ctx_val [N] bool = validation contexts."""
    N, T = data["N"], data["T"]
    smask = sample_mask(N, T)
    tr = smask & ~ctx_val.repeat_interleave(T)
    va = smask & ctx_val.repeat_interleave(T)
    full_cols = torch.unique(data["cols"][data["vals"] != 0])
    r, c, v, y = subset(data, tr)
    D = LC.build_design(r.to(dev), c.to(dev), v.to(dev), y.to(dev), col_ids=full_cols.to(dev))
    rv, cv, vv, yv = subset(data, va)
    keep = vv != 0
    rv, cv, vv = rv[keep].to(dev), cv[keep].to(dev), vv[keep].to(dev)
    cidx = torch.searchsorted(D.col, cv)
    yv = yv.to(dev)
    curve = []

    def on_step(k, lam, beta, info):
        bo, b0 = LC.original_units(D, beta)
        yhat = LC.predict((rv, cidx, vv), bo, b0, n=yv.numel())
        mse = float(((yv.double() - yhat) ** 2).mean())
        curve.append(dict(k=k, lam=lam, nnz=info["nnz"], val_mse=mse, iters=info["iters"]))
        best = min(range(len(curve)), key=lambda j: curve[j]["val_mse"])
        return len(curve) - 1 - best >= CV_PATIENCE
    t = time.time()
    LC.lasso_path(D, lams, on_step, tol=TOL, max_iter=MAX_ITER, support_cap=SUPPORT_CAP)
    best = min(range(len(curve)), key=lambda j: curve[j]["val_mse"])
    var = float(yv.double().var(unbiased=False))
    return best, dict(curve=curve, val_var=var, secs=round(time.time() - t, 1), n_train=D.n, n_val=int(yv.numel()),
                      at_end=best == len(curve) - 1)


def run_lasso(data, sizes, counts, dev, seed, need_natural=True):
    """The full-data path with captures. sizes {arm: n}; counts {site index: c}. Returns the payload pieces."""
    N, T = data["N"], data["T"]
    smask = sample_mask(N, T)
    r, c, v, y = subset(data, smask)
    t0 = time.time()
    D = LC.build_design(r.to(dev), c.to(dev), v.to(dev), y.to(dev))
    t_build = time.time() - t0
    lmax = LC.lam_max(D)
    lams = LC.lam_grid(lmax, PER_DECADE, MIN_RATIO)
    site_of = (D.col // data["d_sae"]).long()
    cv = None
    k_nat = None
    if need_natural:
        g = torch.Generator().manual_seed(seed)
        ctx_val = torch.zeros(N, dtype=torch.bool)
        for grp in (data["is_pos"], ~data["is_pos"]):             # stratified: 25% of activating, 25% of contrast
            ids = grp.nonzero(as_tuple=True)[0]
            ids = ids[torch.randperm(ids.numel(), generator=g)]
            ctx_val[ids[:max(1, int(round(ids.numel() * CV_FRAC)))]] = True
        k_nat, cv = cv_lambda(data, lams, ctx_val, dev, seed)
    cap = LC.PathCapture(sizes=dict(sizes), counts=dict(counts), site_of=site_of, s=D.s, rankings=LASSO_RANKINGS)
    nat = {}
    anchor_rows = data["anchor_samples"]                       # sample indices (after smask) of the activating anchors

    def stats(beta):
        bo, b0 = LC.original_units(D, beta)
        yhat = D.ybar + D.mv(beta).double()
        return dict(r2_all=LC.r2(D.y, yhat), r2_anchor=LC.r2(D.y[anchor_rows], yhat[anchor_rows]),
                    nnz=int((beta != 0).sum()), intercept=b0)

    def on_step(k, lam, beta, info):
        cap.step(k, lam, beta)
        if k_nat is not None and k == k_nat:
            nat.update(stats(beta), lam=lam, k=k, members=beta.nonzero(as_tuple=True)[0].clone(),
                       beta_scaled=beta[beta != 0].clone())
        return cap.done() and (k_nat is None or k >= k_nat)
    t0 = time.time()
    path = LC.lasso_path(D, lams, on_step, tol=TOL, max_iter=MAX_ITER, support_cap=SUPPORT_CAP)
    t_path = time.time() - t0
    cap.finish()
    last = stats(cap.last["beta"]) if cap.last is not None else {}
    return dict(D=D, lmax=lmax, lams=lams, cap=cap, nat=nat, cv=cv, path=path, last=last, t_build=round(t_build, 1),
                t_path=round(t_path, 1))


def to_members(D, sel, sites, d_sae):
    """[(site, latent)] in rank order from Design column indices."""
    g = D.col[sel].cpu().tolist()
    return [(sites[x // d_sae], int(x % d_sae)) for x in g]


def lasso_payload(G, key, rec, pos, arg, sizes_t, counts, dev):
    """Collect, fit, capture: the cached payload for one target (rankings/lasso/<key>.<hash>.pt)."""
    neg = contrast_train(rec).to(pos.device)
    T = int(pos.shape[1])
    tokens = torch.cat([pos, neg[:, :T]], 0)
    t0 = time.time()
    data = collect(G, key, tokens)
    t_collect = time.time() - t0
    N = data["N"]
    is_pos = torch.zeros(N, dtype=torch.bool)
    is_pos[:int(pos.shape[0])] = True
    data["is_pos"] = is_pos
    smask = sample_mask(N, T)
    new = torch.cumsum(smask.long(), 0) - 1
    anc = torch.arange(int(pos.shape[0])) * T + arg.cpu().long().clamp(0, T - 1)
    data["anchor_samples"] = new[anc[smask[anc]]].to(dev)
    # sanity: the pre-activation read here reproduces the target's post-Top-K value where it is active
    post, y = data["post"], data["y"]
    on = post > 0
    pre_err = float((y[on] - post[on]).abs().max()) if bool(on.any()) else None
    sites = data["sites"]
    s_index = {s: j for j, s in enumerate(sites)}
    out = run_lasso(data, sizes_t, {s_index[s]: c for s, c in counts.items() if s in s_index}, dev, C.key_seed(key))
    D, cap, nat = out["D"], out["cap"], out["nat"]
    d_sae = data["d_sae"]
    glob = {arm: {r: to_members(D, g["sel"][r], sites, d_sae) for r in LASSO_RANKINGS} for arm, g in cap.got_sizes.items()}
    glob_info = {arm: dict(lam_rel=g["lam"] / out["lmax"], k=g["k"], support=g["support"], short=g["short"])
                 for arm, g in cap.got_sizes.items()}
    per_site = {r: {} for r in LASSO_RANKINGS}
    per_site_info = {}
    for j, g in cap.got_sites.items():
        for r in LASSO_RANKINGS:
            per_site[r][sites[j]] = [lat for _, lat in to_members(D, g["sel"][r], sites, d_sae)]
        per_site_info[sites[j]] = dict(lam_rel=g["lam"] / out["lmax"], support=g["support"], short=g["short"])
    natural = None
    if nat:
        natural = dict(members=to_members(D, nat["members"], sites, d_sae), beta_scaled=nat["beta_scaled"].cpu(),
                       s=D.s[nat["members"]].cpu(), lam_rel=nat["lam"] / out["lmax"],
                       **{k: nat[k] for k in ("r2_all", "r2_anchor", "nnz", "intercept", "k")})
    diag = dict(n_samples=D.n, n_ctx=N, n_pos_ctx=int(pos.shape[0]), n_neg_ctx=int(neg.shape[0]), T=T, p_active=D.p,
                nnz_design=int(D.A.values().numel()), n_sites=len(sites), lam_max=out["lmax"],
                path_steps=len(out["path"]), path_iters=int(sum(x["iters"] for x in out["path"])),
                path_unconverged=int(sum(not x["converged"] for x in out["path"])),
                last_step=dict(nnz=out["last"].get("nnz"), lam_rel=(cap.last["lam"] / out["lmax"]) if cap.last else None,
                               r2_all=out["last"].get("r2_all"), r2_anchor=out["last"].get("r2_anchor")),
                y_mean=D.ybar, y_std=float(D.y.double().std()), pre_vs_post_maxerr=pre_err,
                t_collect=round(t_collect, 1), t_build=out["t_build"], t_path=out["t_path"],
                t_cv=(out["cv"] or {}).get("secs"), cv=(None if out["cv"] is None else
                                                       {k: v for k, v in out["cv"].items() if k != "curve"}),
                cv_curve=None if out["cv"] is None else [(x["lam"] / out["lmax"], x["nnz"], x["val_mse"])
                                                         for x in out["cv"]["curve"]],
                path=[(x["lam"] / out["lmax"], x["nnz"], x["iters"], x["secs"]) for x in out["path"]])
    return dict(key=key, version=VERSION, sites=sites, d_sae=d_sae, global_members=glob, global_info=glob_info,
                per_site=per_site, per_site_info=per_site_info, natural=natural, diag=diag)


# ---------------------------------------------------------------------------------------------- main
def members_dict(items):
    out = {}
    for s, lat in items:
        out.setdefault(s, []).append(int(lat))
    return out


def main():
    driver = C.import_driver()
    sys.path.insert(0, str(E083))
    import attr_fitted as F
    targets = C.shard_targets()
    sizes = C.sizes_by_arm(C.sweep_rows())
    da, df, dl = C.OUT / "attr", C.OUT / "fitted", C.OUT / "lasso"
    for x in (da, df, dl):
        x.mkdir(parents=True, exist_ok=True)
    pa, pf = da / ("rows.lasso.%s.jsonl" % C.TAG), df / ("rows.lasso.%s.jsonl" % C.TAG)
    done_a, done_f = C.done_keys(pa, ATTR_KEYS), C.done_keys(pf, FIT_KEYS)
    log("lasso rows | %d targets | rankings %s | per-site %s fit %s | sizes %s | natural %s | out %s | WCM %s"
        % (len(targets), LASSO_RANKINGS, FIT_RANKINGS, FIT, C.SIZES, NATURAL, C.OUT, driver.WCM_VERSION))
    if FIT_RANKINGS and FIT:
        F.install_support_wrapper()
    R = driver.Runner()
    ctx_dir = str(R.ctx_dir)
    M = R.method(driver.GAMMA, driver.LAM, True, steps=STEPS or driver.STEPS) if FIT_RANKINGS and FIT else None
    fa, ff, fd, fr = open(pa, "a"), open(pf, "a"), open(dl / ("diag.%s.jsonl" % C.TAG), "a"), open(
        df / ("train.lasso.%s.jsonl" % C.TAG), "a")
    for n_, key in enumerate(targets):
        rec = R.contexts(key)
        if rec.get("strong") is None:
            log("%4d/%d %-16s skip (no contexts)" % (n_ + 1, len(targets), key))
            continue
        ref_path = C.circuit_path(C.REF_ARM, key)
        ref = torch.load(ref_path, weights_only=False) if ref_path is not None else None
        counts = {s: len(v) for s, v in C.members_by_site(ref).items()} if ref is not None else {}
        size_arms = C.resolve_sizes(sizes.get(key, {}))
        tr_arm, thin, pos, arg, h = C.training_contexts(R, rec)
        jobs_a = [(rk, sz) for sz in size_arms for rk in LASSO_RANKINGS if (key, "lasso", rk, sz, ctx_dir) not in done_a]
        if NATURAL and (key, "lasso", "cv", "natural", ctx_dir) not in done_a:
            jobs_a.append(("cv", "natural"))
        variants = ("a1", "fit") if FIT else ("a1",)
        jobs_f = [(rk, v) for rk in FIT_RANKINGS for v in variants if counts
                  and (key, "fitted_lasso", rk, v, ctx_dir) not in done_f]
        if not jobs_a and not jobs_f:
            continue
        base = dict(seed=key, train_arm=tr_arm, thin=thin, held="strong", shard=C.TAG, **C.provenance(R, h))
        rpath = C.OUT / "rankings" / "lasso" / ("%s.%s.pt" % (key, h))
        try:
            payload = torch.load(rpath, weights_only=False) if rpath.exists() else None
            if payload is not None and (payload.get("version") != VERSION
                                        or not set(size_arms) <= set(payload["global_members"])):
                payload = None
            if payload is None:
                t = time.time()
                payload = lasso_payload(R.G, key, rec, pos, arg, {a: sizes[key][a] for a in size_arms}, counts, R.dev)
                payload["diag"]["t_total"] = round(time.time() - t, 1)
                C.save_ranking_file(rpath, payload)
                dg = payload["diag"]
                C.write_row(fd, dict(base, **{k: v for k, v in dg.items() if k not in ("path", "cv_curve")},
                                     natural=None if payload["natural"] is None else {
                                         k: payload["natural"][k] for k in ("r2_all", "r2_anchor", "nnz", "lam_rel")},
                                     global_info=payload["global_info"],
                                     per_site_short=sum(1 for x in payload["per_site_info"].values() if x["short"])))
                log("%4d/%d %-16s lasso: n %d p %d nnz %d | natural nnz %s R2 %.3f (anchor %.3f) | %.0fs (collect %.0f "
                    "cv %s path %.0f)" % (n_ + 1, len(targets), key, dg["n_samples"], dg["p_active"], dg["nnz_design"],
                                          (payload["natural"] or {}).get("nnz"),
                                          (payload["natural"] or {}).get("r2_all", float("nan")),
                                          (payload["natural"] or {}).get("r2_anchor", float("nan")), dg["t_total"],
                                          dg["t_collect"], dg["t_cv"], dg["t_path"]))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            C.write_row(fa, dict(base, method="lasso", ranking=None, size_arm=None,
                                 error="%s: %s" % (type(e).__name__, str(e)[:300])))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        t_lasso = payload["diag"].get("t_total")
        sites = payload["sites"]
        for rk, sz in jobs_a:
            row = dict(base, method="lasso", ranking=rk, size_arm=sz, t_attr=t_lasso)
            try:
                if sz == "natural":
                    if payload["natural"] is None:
                        continue
                    members = members_dict(payload["natural"]["members"])
                    row.update(n_target=None, lam_rel=payload["natural"]["lam_rel"])
                else:
                    gi = payload["global_info"][sz]
                    row.update(n_target=sizes[key][sz], lam_rel=gi["lam_rel"], support=gi["support"],
                               short=gi["short"])
                    rank = C.ranking_from_items([(s, lat, 0.0) for s, lat in payload["global_members"][sz][rk]], sites,
                                                keep_top=0)
                    members = C.truncate_global(rank, sites, row["n_target"])
                t = time.time()
                r = C.scalar_row(R.score(C.make_circuit(key, members), rec, "strong"))
                r.update(row)
                r["t_score"] = round(time.time() - t, 1)
                row = r
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            C.write_row(fa, row)
            log("%4d/%d %-16s lasso %-4s %-8s n=%s/%s Z %s A %s C %s nec %s" % (
                n_ + 1, len(targets), key, rk, sz, row.get("n"), row.get("n_target"),
                *("%.2f" % row[c] if isinstance(row.get(c), float) else "-" for c in C.HEAD + [C.NEC])))
        for rk in FIT_RANKINGS:
            todo = [v for r_, v in jobs_f if r_ == rk]
            if not todo:
                continue
            members = {s: v for s, v in payload["per_site"][rk].items() if v}
            short = sorted(s for s, x in payload["per_site_info"].items() if x["short"])
            fb = dict(base, method="fitted_lasso", ranking=rk, size_arm=C.REF_ARM, n_ref=sum(counts.values()),
                      n_support=C.n_members(members), short_sites=len(short), t_attr=t_lasso)
            final = df / "circuits" / ("lasso_%s" % rk) / ("%s.%s.pt" % (key, h))
            for row in F.score_variants(R, M, rec, key, tr_arm, members, todo, fb, final, fr,
                                        dict(method="fitted_lasso", ranking=rk)):
                C.write_row(ff, row)
                log("%4d/%d %-16s fitted_lasso %-4s %-3s n=%s/%s short %s Z %s A %s C %s nec %s" % (
                    n_ + 1, len(targets), key, rk, row["variant"], row.get("n"), row["n_ref"], row.get("short_sites"),
                    *("%.2f" % row[c] if isinstance(row.get(c), float) else "-" for c in C.HEAD + [C.NEC])))
    log("DONE")


if __name__ == "__main__":
    main()
