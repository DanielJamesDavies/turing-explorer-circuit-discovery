"""C1 (DAN-159 / DAN-77): the attribution baselines under protocol v1, truncated to the WCM sizes and scored by the
production v1 scorer.

Per target, on the target's v1 TRAINING contexts (arm B: 32 strongest + 16 mid-band; arm A: the strongest-train set
for thin targets; exactly the rows driver.Runner.fit hands the engine, asserted per target), two estimators of each
upstream latent's effect on the target's pre-activation at each context's anchor ("drive"), both on the path
between every upstream site's mean value over the training contexts (the A ablation value) and its natural code,
all sites together (circuit/instrument/ig_baseline's InterpolatedCodeInstrument, unchanged):

  abl_ig   integrated gradients (ig_mean): gradients at IG_STEPS points along the path, x (natural - mean)
           (sparse feature circuits' IE_ig in form; 031's settings: 10 steps, batches of 4)
  atp      attribution patching (Nanda 2023; Syed et al. 2023; Kramar et al. 2024): the gradient at the CLEAN point
           only (path end, alpha = 1, where the instrument reproduces the unedited stream) x (natural - mean): the
           first-order estimate of mean-ablating the latent (its estimated effect is the negative of this score).
           Not the library's "local" ablation mode, which scores activation x grad(pre^2), i.e. zero ablation of
           active latents only

Each arm keeps its ranking (rankings/<arm>/<key>.<train hash>.pt, top KEEP_TOP entries) under three reductions of the
per-(context, position) attribution:
  max    the largest |attribution| at any context and position (031's ranking)
  agg    mean over contexts of the summed |attribution| over the causal prefix
  sfc    |mean over contexts of the summed signed attribution|: the classic collapsed score, i.e. the aggregate
         ranking of prior work (sparse feature circuits)
Each ranking is truncated (global top-n) to the target's WCM size at every WCM price (SIZES, default W_all: the DAN-77
curve; W_0.001 is Table 2) and, with NATURAL=1 (default), scored at its natural size (031's position union: per batch
one pooled 50th-percentile cut on |attribution|, entries in each context's causal prefix). Scoring is
driver.Runner.score on held-out strongest: the same scorer, contexts, reference set and fills as every WCM row.

The reduction is computed here rather than through discover(), which keeps only one reduction (the max) and builds
the 10^5-member natural union as a Circuit plus the pipeline eval. Contexts come through driver.Runner.contexts, so the
driver's CTX_DIR (rebuilt held-out records) applies.

  CUDA_VISIBLE_DEVICES=0 SHARD=0/1 PYTHONPATH=src python experiments/083-v1-comparison/attr_baselines.py
  env: ARMS (abl_ig,atp)  RANKINGS (max,agg,sfc)  SIZES (W_all)  IG_STEPS (10)  ATTR_BS (4, sequences per grad pass)
       NATURAL (1)  KEEP_TOP (65536)  TARGETS  LIMIT  SHARD  OUT083  CTX_DIR (read by driver)
Resumable: rankings are cached per (target, arm, training contexts); rows are keyed (seed, method, ranking, size_arm,
ctx_dir).
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

ARMS = [a for a in os.environ.get("ARMS", "abl_ig,atp").split(",") if a]
RANKINGS = [r for r in os.environ.get("RANKINGS", "max,agg,sfc").split(",") if r]
IG_STEPS = int(os.environ.get("IG_STEPS", 10))          # config.discovery.ablation_gradient.ig_steps, as in 031
ATTR_BS = int(os.environ.get("ATTR_BS", 4))              # 031's probe_batch_size
NATURAL = os.environ.get("NATURAL", "1") == "1"
UNION_PCTL = 50.0                                        # 031: position_aware_select abs_pctl, threshold 50
KEY_FIELDS = ("seed", "method", "ranking", "size_arm", "ctx_dir")


def arm_alphas(arm):
    """Where the gradient is taken along the mean -> natural path: IG's left Riemann points, or AtP's clean point."""
    if arm == "abl_ig":
        return [step / IG_STEPS for step in range(IG_STEPS)]
    if arm == "atp":
        return [1.0]
    raise ValueError("unknown arm %r (abl_ig | atp)" % arm)


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), C.TAG, s), flush=True)


# ---------------------------------------------------------------------------------------------- the estimator
def ig_reduce(inference, bank, sites, tokens, anchors, *, seed, w_seed, b_seed, objective, target_act, floors,
              ig_steps=None, alphas=None, batch_size=None, union_pctl=None):
    """integrated_baseline_scores' estimator (same instrument, path and metric), reduced per latent by
    C.AttrReducer instead of collapsed. The gradient is taken at `alphas` (default IG's ig_steps left Riemann points
    0, 1/N, .., (N-1)/N; [1.0] = attribution patching at the clean point) and the attribution is mean_alpha(grad) x
    (natural - floor). Returns ({site: {"max", "agg", "sfc"}}, metric at alpha 0 (None if not evaluated), metric at
    alpha 1, union). Chunks of `batch_size` sequences; each sequence's attribution is to its own anchor read (the
    chunk metric is a mean, so the chunk's attribution is rescaled by the chunk size). `union` (union_pctl given) is
    031's natural-size position union: per chunk, one pooled percentile cut over every site's nonzero |attr|
    (chunk-mean scale, as ig_baseline merges chunks), entries in each context's causal prefix (this chunk's anchors)."""
    from circuit.instrument.ig_baseline import InterpolatedCodeInstrument, collect_natural_codes
    from circuit.instrument.position_aware import pooled_abs_threshold
    if alphas is None:
        alphas = [step / ig_steps for step in range(ig_steps)]
    alphas = [float(a) for a in alphas]
    passes = [(a, True) for a in alphas] + ([] if 1.0 in alphas else [(1.0, False)])   # + the endpoint read
    layer, kind = seed
    sset = set(sites)
    dev = tokens.device
    red = {s: C.AttrReducer(bank.d_sae, device=dev) for s in sites}
    union = ({s: torch.zeros(bank.d_sae, dtype=torch.bool, device=dev) for s in sites}
             if union_pctl is not None else None)
    B_total = int(tokens.shape[0])
    bs = max(1, int(batch_size or B_total))
    m_start = 0.0 if 0.0 in alphas else None
    m_end = 0.0
    inference.disable_compile()
    try:
        for s0 in range(0, B_total, bs):
            tk, an = tokens[s0:s0 + bs], anchors[s0:s0 + bs]
            Bc = int(tk.shape[0])
            codes, residuals = collect_natural_codes(inference, bank, tk, sset)
            acc = {}
            for alpha, grad_on in passes:
                inst = InterpolatedCodeInstrument(bank, sset, codes, residuals, floors, alpha, layer, kind,
                                                  w_seed, b_seed)
                inference.forward(tk, patcher=inst, grad_enabled=grad_on, return_activations=False,
                                  tokenize_final=False)
                pre = inst.seed_pre_act
                if pre is None:
                    raise RuntimeError("seed pre-activation was not captured")
                rr = torch.arange(Bc, device=pre.device)
                peak = pre[rr, an.to(pre.device).clamp(0, pre.shape[1] - 1)]
                metric = -((peak - target_act) ** 2).mean() if objective == "gap" else peak.mean()
                if alpha == 0.0 and m_start is not None:
                    m_start += Bc / B_total * float(metric.detach())
                if alpha == 1.0:
                    m_end += Bc / B_total * float(metric.detach())
                if grad_on:
                    order = sorted(inst.anchors)
                    grads = torch.autograd.grad(metric, [inst.anchors[s] for s in order], allow_unused=True)
                    for s, g in zip(order, grads):
                        if g is None:
                            continue
                        c = (g.float() * inst.deltas[s].float() / len(alphas)).detach()
                        acc[s] = c if s not in acc else acc[s] + c
                    del grads
                inst.release()
                del inst
            for s, a in acc.items():
                red[s].update(a * Bc, an)
            if union is not None and acc:
                scaled = {s: a * (Bc / B_total) for s, a in acc.items()}
                thr = pooled_abs_threshold(list(scaled.values()), union_pctl)
                for s, a in scaled.items():
                    T = a.shape[1]
                    pre_mask = torch.arange(T, device=a.device)[None, :] <= an.to(a.device)[:, None]
                    union[s] |= ((a.abs() >= thr) & pre_mask[..., None]).any(dim=1).any(dim=0)
            del acc
    finally:
        inference.enable_compile()
    return {s: r.result() for s, r in red.items()}, m_start, m_end, union


def upstream(G, key):
    from eval.ablation_faithfulness import upstream_sites
    layer, kind, _ = C.parse(key)
    return C.site_order(G["KINDS"], upstream_sites(G["bank"], layer, kind))


def attribute(G, key, arm, pos, arg):
    """One arm on one target: the rankings (and the natural union) as a saveable payload."""
    from eval.floors import collect_site_means
    inference, bank = G["inference"], G["bank"]
    layer, kind, i = C.parse(key)
    sites = upstream(G, key)
    floors = collect_site_means(inference, bank, pos, set(sites))          # the mean over the training contexts
    sae = bank.saes[kind][layer]
    w_seed, b_seed = sae.encoder.weight[i].detach(), sae._get_bias_eff()[i].detach()
    torch.manual_seed(C.key_seed(key))                     # the union's pooled-quantile sampling (> 200k entries)
    red, m0, m1, union = ig_reduce(inference, bank, sites, pos, arg, seed=(layer, kind), w_seed=w_seed, b_seed=b_seed,
                                   objective="drive", target_act=0.0, floors=floors, alphas=arm_alphas(arm),
                                   batch_size=ATTR_BS, union_pctl=UNION_PCTL if NATURAL else None)
    ranks = {name: C.ranking_from_site_scores({s: red[s][name] for s in sites}, sites) for name in ("max", "agg", "sfc")}
    info = dict(objective="drive", floor="A_train", alphas=len(arm_alphas(arm)), metric_start=m0, metric_end=m1,
                sfc_total=float(sum(float(red[s]["sfc"].double().sum()) for s in sites)), batch=ATTR_BS,
                n_train=int(pos.shape[0]), n_sites=len(sites),
                n_nonzero=int(sum(int((red[s]["max"] != 0).sum()) for s in sites)))
    payload = dict(key=key, arm=arm, sites=sites, rankings=ranks, info=info)
    if union is not None:
        payload["natural"] = {s: union[s].nonzero(as_tuple=True)[0].cpu().numpy().astype("int32") for s in sites}
    return payload


def load_or_compute(path, fn, need_natural):
    if path.exists():
        payload = torch.load(path, weights_only=False)
        if not need_natural or "natural" in payload:
            return payload
    t = time.time()
    payload = fn()
    payload["info"]["secs"] = round(time.time() - t, 1)
    C.save_ranking_file(path, payload)
    return payload


def main():
    driver = C.import_driver()
    targets = C.shard_targets()
    sizes = C.sizes_by_arm(C.sweep_rows())
    d = C.OUT / "attr"
    d.mkdir(parents=True, exist_ok=True)
    rows_path = d / ("rows.%s.jsonl" % C.TAG)
    done = C.done_keys(rows_path, KEY_FIELDS)
    log("attr baselines | %d targets | arms %s | rankings %s | sizes %s | ig_steps %d | natural %s | out %s"
        % (len(targets), ARMS, RANKINGS, C.SIZES, IG_STEPS, NATURAL, d))
    R = driver.Runner()
    ctx_dir = str(R.ctx_dir)
    fh = open(rows_path, "a")
    for n_, key in enumerate(targets):
        rec = R.contexts(key)
        if rec.get("strong") is None:
            log("%4d/%d %-16s skip (%s)" % (n_ + 1, len(targets), key, "no_contrast" if rec.get("no_contrast") else "too_few"))
            continue
        tr_arm, thin, pos, arg, h = C.training_contexts(R, rec)
        size_arms = C.resolve_sizes(sizes.get(key, {}))
        for arm in ARMS:
            jobs = [(rk, sz) for sz in size_arms for rk in RANKINGS if (key, arm, rk, sz, ctx_dir) not in done]
            if NATURAL and (key, arm, "union", "natural", ctx_dir) not in done:
                jobs.append(("union", "natural"))
            if not jobs:
                continue
            base = dict(method=arm, train_arm=tr_arm, thin=thin, held="strong", shard=C.TAG, **C.provenance(R, h))
            try:
                payload = load_or_compute(d.parent / "rankings" / arm / ("%s.%s.pt" % (key, h)),
                                          lambda: attribute(R.G, key, arm, pos, arg), NATURAL)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                C.write_row(fh, dict(seed=key, ranking=None, size_arm=None, error="%s: %s" % (type(e).__name__, str(e)[:300]), **base))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            sites = payload["sites"]
            for rk, sz in jobs:
                row = dict(seed=key, ranking=rk, size_arm=sz, t_attr=payload["info"].get("secs"), **base)
                try:
                    if sz == "natural":
                        members = {s: [int(x) for x in v] for s, v in payload["natural"].items() if len(v)}
                        row["n_target"] = None
                    else:
                        row["n_target"] = sizes[key][sz]
                        row["n_ranked"] = int(len(payload["rankings"][rk]["latent"]))
                        members = C.truncate_global(payload["rankings"][rk], sites, row["n_target"])
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
                C.write_row(fh, row)
                log("%4d/%d %-16s %-6s %-5s %-10s n=%s/%s Z %s A %s C %s nec %s  attr %ss score %ss" % (
                    n_ + 1, len(targets), key, arm, rk, sz, row.get("n"), row.get("n_target"),
                    *("%.2f" % row[c] if isinstance(row.get(c), float) else "-" for c in C.HEAD + [C.NEC]),
                    row.get("t_attr"), row.get("t_score")))
    log("DONE")


if __name__ == "__main__":
    main()
