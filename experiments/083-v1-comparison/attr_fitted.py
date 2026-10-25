"""C3 (DAN-43) and the activation-ranked row: ranked members with fitted coefficients, the middle rows between
truncated attribution (ranked members, alpha = 1) and WCM (optimised members and coefficients).

For each target: the production circuit's per-site member counts (062 sweep W_0.001, the main lambda 1e-3 circuit);
the support at each site = that many latents from the top of a ranking restricted to the site; then the coefficients
are fitted exactly as random_null.py fits its random supports (DAN-17): the production engine call wrapped so that
`support` = these members, sparsity penalty 0, gates fully open (theta_init 40, binarize none), so only the
coefficients move; everything else (contexts, ablation values, gamma, rank-keep, 400 steps, lr) is the v1 production
configuration through driver.Runner.fit. Scored by driver.Runner.score on held-out strongest, twice: the support at
alpha = 1 (variant a1: the per-site truncation itself) and with the fitted coefficients (variant fit). The a1 -> fit
difference on the same members is the coefficient-fitting effect; fit vs WCM is the member-choice effect.

Sources (FIT_FROM = comma list of source:ranking):
  abl_ig:max   (DAN-43) attr_baselines.py's IG ranking, max reduction (also atp:max, abl_ig:sfc, ...)
  act:mean     the no-gradient heuristic (Daniel, 2026-10-24): at each upstream site, latents ranked by their mean
               post-Top-K activation AT THE TARGET'S ANCHOR over the v1 training contexts
  act:coact    the co-activation variant: Pearson correlation, across the training contexts, of the latent's anchor
               activation with the target's (latents never active at an anchor are not ranked)
  The pre-v1 analogue of the activation row (activation-ranked members with fitted amplitudes) passed 0/128.

  CUDA_VISIBLE_DEVICES=0 SHARD=0/1 PYTHONPATH=src python experiments/083-v1-comparison/attr_fitted.py
  env: FIT_FROM (abl_ig:max,act:mean,act:coact)  STEPS (400)  TARGETS  LIMIT  SHARD  OUT083  CTX_DIR (read by driver)
The attribution sources need attr_baselines.py's rankings (rankings/<arm>/<key>.<train hash>.pt); a target without
one gets an error row (no_ranking), retried on the next run. The activation rankings are computed here (one forward
pass) and cached in rankings/act/. Rows are keyed (seed, method, ranking, variant, ctx_dir).
"""
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cmp_common as C  # noqa: E402

FIT_FROM = [tuple(x.split(":")) for x in os.environ.get("FIT_FROM", "abl_ig:max,act:mean,act:coact").split(",") if x]
STEPS = int(os.environ["STEPS"]) if os.environ.get("STEPS") else None
_SUPPORT = {}                                              # set per fit; read by the wrapped engine call
METHOD = {"act": "fitted_act"}                             # row method names: fitted_<source>


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), C.TAG, s), flush=True)


def install_support_wrapper():
    """random_null.py's wrapper (DAN-17), reproduced here: when _SUPPORT is set, the fit is restricted to it with no
    sparsity penalty and fully open gates. ablation_gradient imports run_learned_mask at call time, so patching the
    module attribute reaches the production hop."""
    import circuit.instrument.learned_mask as LM
    orig = getattr(LM, "_v1cmp_orig_run_learned_mask", LM.run_learned_mask)
    LM._v1cmp_orig_run_learned_mask = orig

    def wrapped(inference, bank, **kw):
        if _SUPPORT:
            kw.update(support={s: torch.tensor(v, dtype=torch.long) for s, v in _SUPPORT.items()},
                      l1_lambda=0.0, binarize="none", theta_init=40.0)
        return orig(inference, bank, **kw)
    LM.run_learned_mask = wrapped


# ---------------------------------------------------------------------------------------------- activation ranking
def anchor_codes(G, key, pos, arg, batch=16):
    """Per upstream site, the dense post-Top-K code AT EACH CONTEXT'S ANCHOR [B, d_sae] (CPU), and the target's own
    anchor activation [B]: one clean forward pass over the training contexts."""
    from sae.dense import sparse_topk_to_dense, target_latent_activations
    from eval.ablation_faithfulness import upstream_sites
    inference, bank = G["inference"], G["bank"]
    layer, kind, i = C.parse(key)
    sites = C.site_order(G["KINDS"], upstream_sites(bank, layer, kind))
    k2i = {k: n for n, k in enumerate(G["KINDS"])}
    codes = {s: [] for s in sites}
    tgt = []
    inference.disable_compile()
    try:
        for s0 in range(0, int(pos.shape[0]), batch):
            tk, an = pos[s0:s0 + batch], arg[s0:s0 + batch]

            def hook(layer_idx, activations):
                for kd in G["KINDS"]:
                    s = (layer_idx, kd)
                    if s not in codes and s != (layer, kind):
                        continue
                    ta, ti = bank.encode(activations[k2i[kd]], kd, layer_idx)
                    rr = torch.arange(ta.shape[0], device=ta.device)
                    a = an.to(ta.device).clamp(0, ta.shape[1] - 1)
                    if s == (layer, kind):
                        tgt.append(target_latent_activations(ta, ti, i)[rr, a].float().cpu())
                    else:
                        codes[s].append(sparse_topk_to_dense(ta[rr, a], ti[rr, a], bank.d_sae, dtype=torch.float32).cpu())
            with torch.no_grad():
                inference.forward(tk, activations_callback=hook, return_activations=False, tokenize_final=False)
    finally:
        inference.enable_compile()
    return sites, {s: torch.cat(v) for s, v in codes.items()}, torch.cat(tgt)


def activation_scores(codes, target):
    """{"mean": mean anchor activation, "coact": Pearson correlation of each latent's anchor activation with the
    target's across contexts (0 where the latent or the target is constant)} per site, [d_sae] each."""
    y = target.double() - target.double().mean()
    out = {"mean": {}, "coact": {}}
    for s, x in codes.items():
        x = x.double()
        out["mean"][s] = x.mean(0).float()
        xc = x - x.mean(0, keepdim=True)
        den = xc.pow(2).sum(0).sqrt() * y.pow(2).sum().sqrt()
        out["coact"][s] = torch.where(den > 0, (xc * y[:, None]).sum(0) / den.clamp_min(1e-30),
                                      torch.zeros_like(den)).float()
    return out


def activation_ranking(G, key, pos, arg):
    sites, codes, target = anchor_codes(G, key, pos, arg)
    sc = activation_scores(codes, target)
    ranks = {"mean": C.ranking_from_site_scores(sc["mean"], sites),
             # coact ranks by the correlation itself (most positively co-active first), among latents ever active
             "coact": C.ranking_from_site_scores({s: torch.where(codes[s].abs().sum(0) > 0, sc["coact"][s],
                                                                 torch.zeros_like(sc["coact"][s])) for s in sites},
                                                 sites, key={s: sc["coact"][s] for s in sites})}
    return dict(key=key, arm="act", sites=sites, rankings=ranks,
                info=dict(n_train=int(pos.shape[0]), target_mean=float(target.mean()),
                          n_active_at_anchor=int(sum(int((codes[s].abs().sum(0) > 0).sum()) for s in sites))))


# ---------------------------------------------------------------------------------------------- fit + score
def fit_circuit(R, M, rec, key, arm, members, final, fr, label):
    """The support-restricted coefficient fit (cached at `final`; tmp + rename so a kill never leaves a half file)."""
    if final.exists():
        return torch.load(final, weights_only=False), 0.0
    tmp = final.with_name("%s.%d.tmp.pt" % (final.stem, os.getpid()))
    _SUPPORT.clear(); _SUPPORT.update(members)
    try:
        c, t_fit = R.fit(M, rec, key, arm, tmp, train_fh=fr, label=label)
    finally:
        _SUPPORT.clear()
    if c is not None:
        os.replace(tmp, final)
    return c, t_fit


def score_variants(R, M, rec, key, arm, members, todo, base, final, fr, label):
    """Yield one row per variant: a1 (the members at alpha = 1) and fit (fitted coefficients)."""
    for v in todo:
        row = dict(base, variant=v)
        try:
            t = time.time()
            if v == "a1":
                c = C.make_circuit(key, members)
            else:
                c, t_fit = fit_circuit(R, M, rec, key, arm, members, final, fr, label)
                row["t_fit"] = round(t_fit, 1)
            if c is None:
                row["skip"] = "rejected"
            else:
                r = C.scalar_row(R.score(c, rec, "strong"))
                r.update(row)
                r["t_score"] = round(time.time() - t - row.get("t_fit", 0.0), 1)
                row = r
        except Exception as e:  # noqa: BLE001
            _SUPPORT.clear()
            traceback.print_exc()
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        yield row


def main():
    driver = C.import_driver()
    targets = C.shard_targets()
    d = C.OUT / "fitted"
    d.mkdir(parents=True, exist_ok=True)
    rows_path = d / ("rows.%s.jsonl" % C.TAG)
    done = C.done_keys(rows_path, ("seed", "method", "ranking", "variant", "ctx_dir"))
    log("ranked members + fitted coefficients | %d targets | from %s | steps %s"
        % (len(targets), FIT_FROM, STEPS or driver.STEPS))
    install_support_wrapper()
    R = driver.Runner()
    ctx_dir = str(R.ctx_dir)
    M = R.method(driver.GAMMA, driver.LAM, True, steps=STEPS or driver.STEPS)
    fh, fr = open(rows_path, "a"), open(d / ("train.%s.jsonl" % C.TAG), "a")
    for n_, key in enumerate(targets):
        ref_path = C.circuit_path(C.REF_ARM, key)
        if ref_path is None:
            continue
        ref = torch.load(ref_path, weights_only=False)
        if ref is None:
            continue
        counts = {s: len(v) for s, v in C.members_by_site(ref).items()}
        rec = R.contexts(key)
        if rec.get("strong") is None:
            continue
        arm, thin, pos, arg, h = C.training_contexts(R, rec)
        for src, rk in FIT_FROM:
            method = METHOD.get(src, "fitted_%s" % src)
            todo = [v for v in ("a1", "fit") if (key, method, rk, v, ctx_dir) not in done]
            if not todo:
                continue
            base = dict(seed=key, method=method, ranking=rk, size_arm=C.REF_ARM, n_ref=sum(counts.values()),
                        train_arm=arm, thin=thin, held="strong", shard=C.TAG, **C.provenance(R, h))
            rpath = C.OUT / "rankings" / src / ("%s.%s.pt" % (key, h))
            try:
                if src == "act":
                    payload = torch.load(rpath, weights_only=False) if rpath.exists() else None
                    if payload is None:
                        payload = activation_ranking(R.G, key, pos, arg)
                        C.save_ranking_file(rpath, payload)
                elif rpath.exists():
                    payload = torch.load(rpath, weights_only=False)
                else:                                 # an error row: retried once attr_baselines has run
                    for v in todo:
                        C.write_row(fh, dict(base, variant=v, error="no_ranking: run attr_baselines.py first"))
                    continue
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                for v in todo:
                    C.write_row(fh, dict(base, variant=v, error="%s: %s" % (type(e).__name__, str(e)[:300])))
                continue
            members, short = C.truncate_per_site(payload["rankings"][rk], payload["sites"], counts)
            base.update(n_support=C.n_members(members), short_sites=len(short))
            final = d / "circuits" / ("%s_%s" % (src, rk)) / ("%s.%s.pt" % (key, h))
            for row in score_variants(R, M, rec, key, arm, members, todo, base, final, fr,
                                      dict(method=method, ranking=rk)):
                C.write_row(fh, row)
                log("%4d/%d %-16s %-14s %-5s %-3s n=%s/%s short %s Z %s A %s C %s nec %s" % (
                    n_ + 1, len(targets), key, method, rk, row["variant"], row.get("n"), row["n_ref"],
                    row.get("short_sites"),
                    *("%.2f" % row[c_] if isinstance(row.get(c_), float) else "-" for c_ in C.HEAD + [C.NEC])))
    log("DONE")


if __name__ == "__main__":
    main()
