"""Experiment 080: the CAUSAL, in-context effect of each target latent on the next token (replaces 064's boost_peak as
the sampling axis; 2026-10-18).

Every target is an internal latent. The question is how much removing it changes the model's prediction of the
token that actually comes next, on the contexts where it fires.

Contexts (per target): its 32 strongest TRAINING contexts from the 062 context cache (SRC_OUT/ctx/<key>.pt, read only):
rec["D_train"], the strongest-train contexts protocol v1 trains on (a thin target without them: the 32 strongest of
strong["train"] by peak). Anchor t = the target's peak position (rec["strong"]["arg"], as production). Next token =
the actual token at t + 1 = rec["strong"]["tgt"][i, t] (the cache loads 65 tokens: 64 of context, tgt shifted by one,
so the next token exists even for t = 63). Edge cases: an anchor outside [0, 63] is clamped and counted; a context
whose next token falls in zero padding (next token 0 with only zeros after it in tgt: a sequence shorter than 65
tokens) is excluded and counted (n_excluded).

Two forward passes per context, at the anchor:
  clean     log p(next token) and the full next-token distribution
  ablated   the target latent removed at its own site at EVERY position: x <- x - a_f(x, t) d_f. Implemented by
            TargetAblation: at the target's (layer, kind) site, bank.encode -> the post-Top-K activation a_f
            (sae.dense.target_latent_activations, exactly the read wcm_eval._tap and the engine use), then the edit is
            the DECODED DIFFERENCE only: x + bank.decode(delta, add_bias=False), delta = -a_f at latent f and 0
            elsewhere (the single-decode form of learned_mask.LearnedMaskPatcher). So the SAE reconstruction error is
            untouched, no other SAE is in the loop, every later layer recomputes natively, and a latent that is
            silent everywhere changes nothing (delta = 0, the patch returns x unchanged, bit for bit). This is the
            zero-ablation sense the paper uses.
Clean and ablated rows of a context run in the SAME forward (interleaved rows; the patcher edits only the ablated
rows), and several targets' rows are packed into one forward when they fit (BATCH rows). Inference.forward has no
resume-from-layer entry point, so "run to the site once, then the rest twice" would mean bypassing the engine's
forward; the interleaved batch is the shared-path saving available through the engine.

Recorded per target (effect.jsonl, then effects.parquet):
  effect_abs     mean over contexts of |delta|, delta = log p_clean(next) - log p_ablated(next)
  effect_signed  mean delta: POSITIVE = the latent helps predict the actual next token (removing it lowers log p)
  kl             mean KL(clean || ablated) over the full next-token distribution at the anchor
  n_contexts, n_excluded, n_clamped, mean_act_at_anchor (the target's post-Top-K activation at the anchor, clean),
  lp_clean_mean, and per-context arrays in effect_ctx.jsonl.gz (delta, kl, act, lp_clean, next token, anchor).
WHY effect_abs is the sampling measure: a latent that SUPPRESSES the actual next token (negative delta) is as
output-relevant as one that promotes it; the signed mean would let promoting and suppressing contexts cancel and
would rank strong suppressors as "low effect". The output arm's endpoint (log p of the actual next token) is
sensitive to both directions, so the axis should be too. effect_signed and kl are kept for description.

Pool (the targets measured): every target that passes make_targets' population filters (non-vacuous eval row,
circuit, context record), either all of them (POOL=all, ~15k) or N_POOL per kind x band group drawn uniformly with a
fixed seed (POOL=subsample, default 300 per group, 2,700 + the 2 smoke targets); or TARGETS=<file>.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -X utf8 experiments/080-latent-vs-output/next_token_effect.py
  env: OUT_EFF (experiments/080-latent-vs-output/effect)  SRC_OUT (062 out_full/out)  TARGETS (a key file) or
       POOL (subsample | all)  N_POOL (300)  POOL_SEED (20261018)  N_CTX (32)  BATCH (64 rows per forward)
       LIMIT (first N targets)  BENCH=N (time N targets after a warm-up one, print sequences/s and the projected hours
       for the full population and the subsample, then stop)  FINALIZE=1 (rebuild effects.parquet only; no model)
       DRY=1 (print the pool and exit; no model)
"""
import gzip
import json
import math
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUT_EFF = Path(os.environ.get("OUT_EFF", str(C.EFFECT_DIR)))
SRC_OUT = Path(os.environ.get("SRC_OUT", str(C.EXP / "062-h100-protocol-v1" / "out_full" / "out")))
N_CTX = int(os.environ.get("N_CTX", 32))
BATCH = int(os.environ.get("BATCH", 64))
N_POOL = int(os.environ.get("N_POOL", 300))
POOL_SEED = int(os.environ.get("POOL_SEED", C.TARGET_SEED))


# ------------------------------------------------------------------------------------------- contexts
def contexts_of(rec, n=N_CTX):
    """(tokens [m, T], next tokens [m], anchors [m], info) for the target's n strongest training contexts, padded
    contexts excluded (see the module docstring)."""
    s = rec["strong"]
    idx = list(rec.get("D_train") or [])
    if not idx:                                          # thin target: the n strongest of strongest-train by peak
        idx = sorted(s["train"], key=lambda j: -float(s["peak"][j]))
    if len(idx) > n:
        idx = sorted(idx, key=lambda j: -float(s["peak"][j]))[:n]
    idx = torch.as_tensor(idx, dtype=torch.long)
    tok, tgt = s["pos"][idx].long(), s["tgt"][idx].long()
    T = int(tok.shape[1])
    arg = torch.as_tensor(s["arg"])[idx].long()
    n_clamped = int(((arg < 0) | (arg > T - 1)).sum())
    arg = arg.clamp(0, T - 1)
    rr = torch.arange(len(idx))
    nxt = tgt[rr, arg]
    # zero padding: the next token is 0 and so is everything after it in tgt (a sequence shorter than 65 tokens)
    tail_zero = torch.stack([(tgt[i, int(arg[i]):] == 0).all() for i in range(len(idx))]) if len(idx) else \
        torch.zeros(0, dtype=torch.bool)
    keep = ~((nxt == 0) & tail_zero)
    return tok[keep], nxt[keep], arg[keep], dict(n_excluded=int((~keep).sum()), n_clamped=n_clamped,
                                                 ctx_index=idx[keep].tolist())


# ------------------------------------------------------------------------------------------- the ablation
class TargetAblation:
    """multi_patch patcher: row r of the batch runs with latent f_r removed at its site (x <- x - a_f d_f at every
    position, the decoded difference only), or clean when assign[r] is None. Records each ablated row's activation
    a_f [T] (clean stream: nothing upstream is edited) in self.acts."""

    def __init__(self, bank, assign):
        self.bank = bank
        self.by_site = defaultdict(list)                 # site -> [(row, latent)]
        for r, a in enumerate(assign):
            if a is not None:
                layer, kind, latent = a
                self.by_site[(int(layer), str(kind))].append((r, int(latent)))
        self.acts = {}

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        from sae.dense import target_latent_activations
        rl = self.by_site.get((layer_idx, kind))
        if not rl:
            return x
        rows = torch.tensor([r for r, _ in rl], device=x.device, dtype=torch.long)
        ta, ti = self.bank.encode(x[rows], kind, layer_idx)
        a = torch.zeros(ta.shape[:-1], device=ta.device, dtype=ta.dtype)          # [n, T]
        for j, (_, f) in enumerate(rl):
            a[j] = target_latent_activations(ta[j], ti[j], f)
        for j, (r, _) in enumerate(rl):
            self.acts[r] = a[j].detach().float().cpu()
        if not bool((a != 0).any()):
            return x                                     # nothing to remove: the stream is untouched, exactly
        delta = torch.zeros(*a.shape, int(self.bank.d_sae), device=x.device, dtype=x.dtype)
        for j, (_, f) in enumerate(rl):
            delta[j, :, f] = -a[j].to(x.dtype)
        diff = self.bank.decode(delta, kind, layer_idx, add_bias=False).to(x.dtype)
        out = x.clone()
        out[rows] = x[rows] + diff
        return out


# ------------------------------------------------------------------------------------------- measurement
def run_pack(inference, bank, jobs):
    """One forward over several targets. jobs = [(key, (layer, kind, latent), tokens [m, T], next [m], anchors [m])].
    Rows interleave clean / ablated per context. Returns {key: per-context dict of tensors}."""
    toks, assign, meta = [], [], []
    for key, site, tk, nx, an in jobs:
        for i in range(int(tk.shape[0])):
            toks += [tk[i], tk[i]]
            assign += [None, site]
            meta.append((key, int(nx[i]), int(an[i])))
    dev = getattr(bank, "device", None) or toks[0].device
    batch = torch.stack(toks).to(dev)
    p = TargetAblation(bank, assign)
    inference.disable_compile()
    try:
        with torch.no_grad():
            out = inference.forward(batch, patcher=p, grad_enabled=False, all_logits=True, return_activations=False,
                                    tokenize_final=False)
    finally:
        inference.enable_compile()
    lg = out[1] if isinstance(out, (tuple, list)) else out
    n = len(meta)
    an = torch.tensor([m[2] for m in meta], device=lg.device)
    nx = torch.tensor([m[1] for m in meta], device=lg.device)
    cr = torch.arange(n, device=lg.device)
    lc = torch.log_softmax(lg[2 * cr, an].float(), -1)                    # [n, V] clean, at each anchor
    la = torch.log_softmax(lg[2 * cr + 1, an].float(), -1)                # [n, V] ablated
    lpc = lc[cr, nx]
    vals = torch.stack([lpc, lpc - la[cr, nx], (lc.exp() * (lc - la)).sum(-1)]).cpu().tolist()   # one host copy
    res = defaultdict(lambda: defaultdict(list))
    for c, (key, nxt, anc) in enumerate(meta):
        d = res[key]
        d["lp_clean"].append(vals[0][c])
        d["delta"].append(vals[1][c])
        d["kl"].append(vals[2][c])
        d["act"].append(float(p.acts[2 * c + 1][anc]))                    # already on the CPU
        d["next"].append(nxt)
        d["anchor"].append(anc)
    return res


def summarise(key, per, info):
    """The per-target row from its per-context values."""
    n = len(per["delta"])
    mean = lambda v: float(sum(v) / len(v)) if v else None  # noqa: E731
    layer, kind, latent = C.parse(key)
    return dict(seed=key, layer=layer, kind=kind, latent=latent, band=C.band(layer), n_contexts=n,
                n_excluded=info["n_excluded"], n_clamped=info["n_clamped"],
                effect_abs=mean([abs(x) for x in per["delta"]]), effect_signed=mean(per["delta"]),
                kl=mean(per["kl"]), mean_act_at_anchor=mean(per["act"]), lp_clean_mean=mean(per["lp_clean"]),
                frac_act_zero=mean([float(a == 0.0) for a in per["act"]]))


# ------------------------------------------------------------------------------------------- pool
def pool_keys():
    """The targets to measure (see the module docstring), interleaved across kind x band groups."""
    if os.environ.get("TARGETS"):
        return [k for k in Path(os.environ["TARGETS"]).read_text().split() if k]
    import make_targets as MT
    meta, _ = MT.population()
    if os.environ.get("POOL", "subsample") == "all":
        keys = sorted(meta)
    else:
        groups = defaultdict(list)
        for k, m in meta.items():
            groups[(m["kind"], m["band"])].append(k)
        rng = random.Random(POOL_SEED)
        picks = [rng.sample(sorted(groups[g]), min(N_POOL, len(groups[g]))) for g in sorted(groups)]
        keys = [p[i] for i in range(max(map(len, picks))) for p in picks if i < len(p)]
        keys += [k for k in C.SMOKE_TARGETS if k in meta and k not in set(keys)]
    return keys


def done_keys(path):
    out = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if not r.get("error"):
                out.add(r["seed"])
    return out


def finalize():
    """effects.parquet from effect.jsonl (the last row per target), with 064's boost_peak joined for the proxy check."""
    import pandas as pd
    rows = {}
    for line in open(OUT_EFF / "effect.jsonl"):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if r["seed"] in rows and r.get("error") and not rows[r["seed"]].get("error"):
            continue                                     # a later error never hides an earlier success
        rows[r["seed"]] = r
    d = pd.DataFrame(list(rows.values()))
    te = pd.read_parquet(C.EXP / "064-prediction-circuits" / "results" / "target_effects.parquet")
    te["seed"] = te.layer.astype(str) + "." + te.kind + "." + te.latent.astype(str)
    d = d.merge(te[["seed", "boost_peak"]], on="seed", how="left")
    if "error" not in d:
        d["error"] = None
    d.to_parquet(OUT_EFF / "effects.parquet", index=False)
    ok = d[d.error.isna()]
    print("effects.parquet: %d targets (%d errors) -> %s" % (len(d), len(d) - len(ok), OUT_EFF / "effects.parquet"))
    if len(ok):
        print(ok.groupby(["kind", "band"]).effect_abs.describe(percentiles=[.33, .5, .67]).round(4).to_string())
        print("Spearman(effect_abs, boost_peak) = %.3f over %d targets" % (
            ok.effect_abs.rank().corr(ok.boost_peak.rank()), len(ok)))


def main():
    if os.environ.get("FINALIZE") == "1":
        return finalize()
    keys = pool_keys()
    if os.environ.get("LIMIT"):
        keys = keys[:int(os.environ["LIMIT"])]
    if os.environ.get("DRY") == "1":
        g = defaultdict(int)
        for k in keys:
            l, kd, _ = C.parse(k)
            g["%s %s" % (kd, C.band(l))] += 1
        print("%d targets to measure: %s" % (len(keys), dict(sorted(g.items()))))
        return
    bench = int(os.environ.get("BENCH", 0))
    OUT_EFF.mkdir(parents=True, exist_ok=True)
    path = OUT_EFF / "effect.jsonl"
    done = done_keys(path)
    todo = [k for k in keys if k not in done]
    if bench:
        todo = todo[:bench + 1]
    sys.path.insert(0, str(C.EXP / "049-circuit-graph"))
    import amp_eval_pass_v2 as V
    t0 = time.time()
    G = V.setup()
    print("engine up in %.0fs | %d targets (%d done, %d to measure) | %d contexts each | batch %d rows -> %s" % (
        time.time() - t0, len(keys), len(done), len(todo), N_CTX, BATCH, OUT_EFF), flush=True)
    inference, bank = G["inference"], G["bank"]
    fh, fc = open(path, "a"), gzip.open(OUT_EFF / "effect_ctx.jsonl.gz", "at")
    t_start, n_seq, n_tgt = None, 0, 0
    pack, rows_in_pack = [], 0

    def flush():
        nonlocal pack, rows_in_pack, n_seq, n_tgt
        if not pack:
            return
        try:
            res = run_pack(inference, bank, [(k, s, tk, nx, an) for k, s, tk, nx, an, _ in pack])
            for k, s, tk, nx, an, info in pack:
                row = summarise(k, res[k], info)
                fh.write(json.dumps(row) + "\n")
                fc.write(json.dumps(dict(seed=k, ctx_index=info["ctx_index"],
                                         **{a: [round(v, 5) if isinstance(v, float) else v for v in res[k][a]]
                                            for a in ("delta", "kl", "act", "lp_clean", "next", "anchor")})) + "\n")
                n_seq += 2 * int(tk.shape[0])
                n_tgt += 1
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            for k, *_ in pack:
                fh.write(json.dumps(dict(seed=k, error="%s: %s" % (type(e).__name__, str(e)[:300]))) + "\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        fh.flush(); fc.flush()
        pack, rows_in_pack = [], 0

    for i, key in enumerate(todo):
        if bench and i == 1:                             # the first target warms the engine up; time the rest
            flush()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start, n_seq, n_tgt = time.time(), 0, 0
        try:
            rec = torch.load(SRC_OUT / "ctx" / ("%s.pt" % key), weights_only=False)[key]
            if rec.get("strong") is None:
                fh.write(json.dumps(dict(seed=key, error="skip record (%s)" % C.parse(key)[1])) + "\n"); continue
            tk, nx, an, info = contexts_of(rec)
            if not int(tk.shape[0]):
                fh.write(json.dumps(dict(seed=key, error="no usable context")) + "\n"); continue
        except Exception as e:  # noqa: BLE001
            fh.write(json.dumps(dict(seed=key, error="load %s: %s" % (type(e).__name__, str(e)[:300]))) + "\n")
            continue
        need = 2 * int(tk.shape[0])
        if rows_in_pack and rows_in_pack + need > BATCH:
            flush()
        pack.append((key, C.parse(key), tk, nx, an, info))
        rows_in_pack += need
        if rows_in_pack >= BATCH:
            flush()
        if (i + 1) % 100 == 0:
            print("  %d/%d  %.0fs" % (i + 1, len(todo), time.time() - t0), flush=True)
    flush()
    fh.close(); fc.close()
    if bench and t_start is not None and n_tgt:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t_start
        sps, spt = n_seq / dt, n_seq / n_tgt
        import make_targets as MT
        n_all = len(MT.population()[0])
        n_sub = 9 * N_POOL + len(C.SMOKE_TARGETS)
        print("BENCH: %d targets, %d sequences in %.1fs -> %.1f sequences/s, %.2f s/target" % (n_tgt, n_seq, dt, sps,
                                                                                          dt / n_tgt))
        print("  projected (engine start-up excluded): full population %d targets -> %.2f h; subsample %d -> %.2f h" % (
            n_all, n_all * spt / sps / 3600, n_sub, n_sub * spt / sps / 3600))
        (OUT_EFF / "bench.json").write_text(json.dumps(dict(n_targets=n_tgt, n_sequences=n_seq, secs=dt, seq_per_s=sps,
                                                            batch=BATCH, n_ctx=N_CTX, hours_all=n_all * spt / sps / 3600,
                                                            hours_subsample=n_sub * spt / sps / 3600), indent=1))
        return
    finalize()
    print("DONE in %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
