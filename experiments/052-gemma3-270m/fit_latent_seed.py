"""LATENT-ENDPOINT TRI-AMP CIRCUITS on Gemma 3 270M + Gemma Scope 2 — the
paper's object, on the new substrate.

Seed = one SAE latent (kind, layer, index). Recipe mirrors the production
engine (src/circuit/instrument/learned_mask.py, 045 driver): objective "pos"
(reproduce the seed's natural pre-activation at each context's anchor), TRIPLE
floor (zero fill w 1.0 + negctx-mean fill w 0.25 + posctx-mean fill w 0.10 if
layer <= 7 else 0.05), free amplitudes alpha = softplus(psi) with the leak
charge LAM*(1-sigmoid(theta))*|alpha-1|, gate L1 LAM*sum sigmoid(theta),
annealed binarisation m = sigmoid(theta/T), T 1 -> 0.05, theta init 4,
400 steps, AdamW lr 0.05 wd 0.05, data terms divided by mean(target^2).

Upstream sites = every (kind, layer) before the seed in forward order
(att -> mlp -> res within a layer). Edits: x <- x + (c_hat - c) @ W_dec at
positions >= 1 (SAE error passes through, BOS untouched). The forward stops
at the seed site.

Contexts:
  positives  Gemma Scope 2 examples.safetensors for the seed's SAE,
             deduplicated by SEQUENCE (highest activation per sequence),
             top N_POS; each truncated to its anchor. N_TR train / rest held out.
  negatives  random corpus windows from the same file, read position p in
             [16, 255], VERIFIED silent (seed code == 0 at p); N_NEG, split alike.

Scores (held-out; relu(pre-activation) at the anchor):
  F0 / FMd / FMn   (a_circ - a_empty) / (a_pos - a_empty) under zero /
                   posctx-mean / negctx-mean fill of non-members
  F0 a1            the same set at alpha = 1 (zero frame)
  perm null        random latents, same per-site counts, shuffled amplitudes
  fitted null      random latents, same per-site counts, gates FROZEN open,
                   amplitudes FITTED with the same schedule and floors (R16)
  sup              1 - a(members zeroed, all else natural) / a_pos
  cf               members SET to alpha*pin in held-out negatives, all else
                   natural: (a - a_base) / (a_pos - a_base)

  SEEDS=res:6:123,att:9:40 python experiments/052-gemma3-270m/fit_latent_seed.py
  AUTO_SITES=mlp:3,res:6,att:9 python ...   (one random seed per site)
Env: N_POS 64 | N_NEG 64 | N_TR 48 | STEPS 400 | LR 0.05 | LAM 1e-3 |
     MICRO 2 | ACCUM 2 | N_FNULL 1 | FREQ_LO 1e-4 | FREQ_HI 5e-3 | RNG 0
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gemmascope2 as GS  # noqa: E402

MODEL = os.environ.get("MODEL", "unsloth/gemma-3-270m")
N_POS = int(os.environ.get("N_POS", 64)); N_NEG = int(os.environ.get("N_NEG", 64))
N_TR = int(os.environ.get("N_TR", 48))
STEPS = int(os.environ.get("STEPS", 400)); LR = float(os.environ.get("LR", 0.05))
LAM = float(os.environ.get("LAM", 1e-3))
# THETA0: gate init. 4.0 (production) starts every gate open (sigmoid 0.98 = the
# model). On uncapped SAEs the partly-closed states between "all open" and
# "empty" are unstable (member re-encoding runs away), so a fit starting open can
# be trapped there; a negative THETA0 (e.g. -4 -> 2% open) starts at the stable
# empty end and grows the circuit instead. Same objective/floors/semantics.
THETA0 = float(os.environ.get("THETA0", 4.0))
# Floor weights: DUAL_W on the negctx-mean term (production 0.25); TRIPLE_W on the
# posctx-mean term (unset -> the 029-panel depth rule: 0.10 if layer <= 7 else 0.05).
DUAL_W = float(os.environ.get("DUAL_W", 0.25))
TRIPLE_W = os.environ.get("TRIPLE_W")
MICRO = int(os.environ.get("MICRO", 2)); ACCUM = int(os.environ.get("ACCUM", 2))
N_FNULL = int(os.environ.get("N_FNULL", 1))
FREQ_LO = float(os.environ.get("FREQ_LO", 1e-4)); FREQ_HI = float(os.environ.get("FREQ_HI", 5e-3))
RNG = int(os.environ.get("RNG", 0))
# UP_KINDS: which site kinds upstream of the seed are editable circuit sites.
# att,mlp,res mirrors the TuringLLM engine; att,mlp leaves the residual stream
# unedited (on Gemma 3, zero-filling residual SAE sites is ill-defined: the
# empty circuit drove a L6 seed to 4.8M against a natural 76).
UP_KINDS = [k for k in os.environ.get("UP_KINDS", "att,mlp,res").split(",") if k]
# R12 vacuity rule (026 floor-isolation): a frame whose empty circuit sits more
# than VAC x a_pos away from a_pos makes its faithfulness band near-vacuous.
VAC = float(os.environ.get("VAC", 5.0))
# ERROR_MODE: "current" = the site keeps the SAE error of the edited stream (what
# TuringLLM and 033 do; fine when TopK caps how many latents can fire). "clean" =
# hold each site's SAE error at its clean-run value; required here because the
# uncapped JumpReLU reconstruction lets an edit's residue feed itself and diverge
# (empty circuit -> inf by L13; see res_explosion_diag.py).
ERROR_MODE = os.environ.get("ERROR_MODE", "current")
# BOUND: keep the RE-ENCODED code of an edited stream inside the token's natural
# envelope (solution_diag.py). "cap" = at most the token's clean-run latent count
# (top by value; an exact no-op on TopK), "clamp" = no latent above the token's
# largest clean-run latent value. Both are exact at identity. cap+clamp with
# ERROR_MODE=clean stayed within 0.76-1.83x of the clean residual norm at every
# layer for 0/10/50/90% random keeps; either alone, or clean alone, diverges.
BOUND = [b for b in os.environ.get("BOUND", "").split("+") if b]
# MEMBERS_FROM: a result jsonl in this folder; re-score that circuit (necessity
# panel etc.) instead of fitting. RED_COS: decoder-cosine cut for "copies".
MEMBERS_FROM = os.environ.get("MEMBERS_FROM", "")
RED_COS = float(os.environ.get("RED_COS", 0.7))
# FLOOR_MODE: how the mean frames fill NON-members. "dense" = every latent at
# its mean (what SFC does; 9k-15k nonzero here vs 60-123 natural, floor_density.py).
# "topk" = the engine's _respect_topk_fill (src/eval/ablation_faithfulness.py:267,
# freeM_topk): members first, then the highest-mean non-members up to the token's
# NATURAL latent count, everything else exactly zero — a code the model could
# emit. 027 R6a: the two disagree 29x at depth on TuringLLM (L9 dense 0.051 vs
# topk 1.460) and freeM_topk is the PRIMARY metric there. Both are always scored;
# this switch only chooses which one the FIT's floors use.
FLOOR_MODE = os.environ.get("FLOOR_MODE", "dense")
# EVAL_FAMILY=1 adds the paper's Table 1 rows that this port was missing, all in
# their AMPLITUDE-AWARE form (Daniel, 2026-09-20). Roles come from attribution
# (sign of d seed / d c_i times the natural value, over train positives), because
# the engine's phi_cf / phi_sup intervene BY ROLE
# (src/eval/counterfactual_faithfulness.py:484) and this port did not:
#   phi_cf^a   contrast contexts: activators <- a * posctx mean, inhibitors <- 0
#   phi_sup    positives: activators <- 0, inhibitors <- negctx mean (engine)
#   phi_sup^a  positives: every member <- max(0, 1-a) * natural, i.e. remove
#              exactly the contribution the circuit CLAIMS (= phi_sup when a=1)
#   phi_pin^a  positives: members <- a * clean value, non-members per fill;
#              pinned isolates node selection, free adds the re-encoding burden
# Default off until tested; flip after the 2026-09-20 run.
EVAL_FAMILY = os.environ.get("EVAL_FAMILY", "0") == "1"
_CUR_REF = {}   # site -> clean-run ref for the batch currently in the forward
OUT =HERE / os.environ.get("OUT_FILE", "latent_circuits.jsonl")
DEV = torch.device("cuda"); DTYPE = torch.bfloat16
ORDER = {"att": 0, "mlp": 1, "res": 2}

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV).eval()
for p_ in model.parameters():
    p_.requires_grad_(False)
LAYERS = model.model.layers
NL = len(LAYERS)
_SAE = {}


def sae(site):
    if site not in _SAE:
        _SAE[site] = GS.load_sae(site[0], site[1], device=DEV, dtype=DTYPE)
    return _SAE[site]


def module_for(site):
    k, l = site
    return {"att": LAYERS[l].self_attn.o_proj, "mlp": LAYERS[l].post_feedforward_layernorm, "res": LAYERS[l]}[k]


class _Stop(Exception):
    pass


def _sae_error(t, x, c):
    """x - (c W_dec + b_dec): the part of x the SAE does not model. ONE function
    for both runs, so the clean and live errors are bitwise equal at identity."""
    return x - (c @ t["W_dec"] + t["b_dec"])


def _bound(c, ref):
    """BOUND: clip a re-encoded code to the token's natural count / max value."""
    if "cap" in BOUND:
        k = ref["l0"]
        kmax = int(k.max())
        if kmax <= 0:
            c = torch.zeros_like(c)
        else:
            tv, ti = c.topk(kmax, dim=-1)
            keep = torch.arange(kmax, device=c.device)[None, None, :] < k[..., None]
            c = torch.zeros_like(c).scatter(-1, ti, tv * keep)
    if "clamp" in BOUND:
        c = torch.minimum(c, ref["mx"][..., None])
    return c


def _topk_fill(c, mu, kept_mask, l0):
    """the engine's k-sparse floor: members take the budget first, the rest of
    the token's NATURAL latent count goes to the highest-mean non-member
    latents at their means, every other latent is exactly zero."""
    busy = ((c > 0) & kept_mask).sum(-1)
    budget = (l0 - busy).clamp(min=0)
    rank = mu.float().masked_fill(kept_mask, float("-inf")).argsort(descending=True)
    out = torch.zeros_like(c)
    max_b = int(budget.max())
    if max_b > 0:
        fill = rank[:max_b]
        act = torch.arange(max_b, device=c.device)[None, None, :] < budget[..., None]
        out[..., fill] = (mu[fill].float().view(1, 1, -1) * act).to(c.dtype)
    return out


def _edit(site, x, fn, ref=None):
    t = sae(site)
    pre = x @ t["W_enc"] + t["b_enc"]
    c = pre * (pre > t["threshold"])
    if BOUND and ref is not None:
        c = _bound(c, ref)
    _CUR_REF[site] = ref
    err_clean = ref["err"] if (ref is not None and ERROR_MODE == "clean") else None
    # ERROR_MODE=current (TuringLLM/033 semantics): x + (c_hat - c(x)) W_dec
    #   = c_hat W_dec + b_dec + err(x): the site is members/floor PLUS the SAE error
    #   of the EDITED stream. Fine when TopK caps active latents; on uncapped
    #   JumpReLU err(x) feeds itself and diverges (res_explosion_diag.py).
    # ERROR_MODE=clean: add (err_clean - err(x)), giving
    #   c_hat W_dec + b_dec + err_clean: members/floor EXACTLY as in current mode
    #   (non-members at the floor, members re-encoded, so closure is still
    #   measured — NOT pinning), with only the error term held at its clean value
    #   (SFC's error-node convention). Both correction terms are bitwise zero at
    #   identity, so identity is exact.
    #   NB the earlier form x + (c_hat - c_clean) W_dec is NOT equivalent: it equals
    #   this PLUS (x - x_clean), letting upstream deviations leak through every
    #   non-member (033 check: circuits 1.4-2.4x larger, drive overshoot 2.5->3.5).
    delta = (fn(c) - c).to(x.dtype) @ t["W_dec"]
    if err_clean is not None:
        delta = delta + (err_clean - _sae_error(t, x, c))
    delta[:, 0] = 0
    return x + delta


def _reg(hs, site, fn_x):
    mod = module_for(site)
    if site[0] == "att":
        hs.append(mod.register_forward_pre_hook(lambda m, a: (fn_x(a[0]),) + tuple(a[1:])))
    else:
        def post(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            y = fn_x(x)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hs.append(mod.register_forward_hook(post))


def clean_errors(tokens, seed, sites, keep=None):
    """per site on the UNEDITED run, detached: the SAE error x - (c(x) W_dec + b_dec),
    the per-token latent count l0 and the per-token largest latent value mx."""
    kind, layer, idx = seed
    err, hs = {}, []

    def capf(site):
        def f(x):
            t = sae(site)
            pre = x @ t["W_enc"] + t["b_enc"]
            c = pre * (pre > t["threshold"])
            err[site] = dict(err=_sae_error(t, x, c).detach(), l0=(c > 0).sum(-1), mx=c.max(-1).values.detach())
            if keep is not None and site in keep:
                err[site]["clean_c"] = c[..., keep[site]].detach()   # for phi_pin
            return x
        return f

    def stop(x):
        raise _Stop()
    for s in sites:
        _reg(hs, s, capf(s))
    _reg(hs, (kind, layer), stop)
    try:
        with torch.no_grad():
            model.model(tokens.to(DEV))
    except _Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    return err


def run(tokens, seed, transforms, grad, err=None):
    """seed pre-activation [B, T] (float32) with `transforms` {site: fn(c) -> c_hat} applied upstream."""
    kind, layer, idx = seed
    out, handles = {}, []
    # always available: the k-sparse fill needs each site's natural L0 even when
    # the error mode and bound do not (scoring always reports freeM_topk).
    if transforms and err is None:
        err = clean_errors(tokens, seed, list(transforms))
    errs = err or {}
    for site, fn in transforms.items():
        _reg(handles, site, lambda x, _s=site, _f=fn, _e=errs.get(site): _edit(_s, x, _f, _e))

    def read(x):
        t = sae((kind, layer))
        out["pre"] = (x.float() @ t["W_enc"][:, idx].float() + t["b_enc"][idx].float())
        raise _Stop()
    _reg(handles, (kind, layer), read)
    try:
        with torch.set_grad_enabled(grad):
            model.model(tokens.to(DEV))
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
    return out["pre"]


def at(pre, anchors):
    return pre[torch.arange(pre.shape[0], device=pre.device), anchors.to(pre.device)]


def batched(tokens, anchors, seed, transforms, bs=8):
    vals = []
    with torch.no_grad():
        for s in range(0, tokens.shape[0], bs):
            vals.append(at(run(tokens[s:s + bs], seed, transforms, False), anchors[s:s + bs]))
    return torch.cat(vals)


def pad(seqs):
    T = max(len(s) for s in seqs)
    return torch.tensor([s + [0] * (T - len(s)) for s in seqs], dtype=torch.long)


# ---------------------------------------------------------------- seeds
def examples(kind, layer):
    return load_file(hf_hub_download(GS.REPO, GS.path(kind, layer) + "/examples.safetensors"))


seed_specs = []
for s in [x for x in os.environ.get("SEEDS", "").split(",") if x]:
    k, l, i = s.split(":"); seed_specs.append((k, int(l), int(i)))
rng = random.Random(RNG)
for s in [x for x in os.environ.get("AUTO_SITES", "").split(",") if x]:
    k, l = s.split(":"); l = int(l)
    E = examples(k, l)
    fq = E["feature_frequencies"].numpy()
    ok = [i for i in range(len(fq)) if FREQ_LO <= fq[i] <= FREQ_HI
          and len(set(E["seq_ids"][i][E["activations"][i] > 0].tolist())) >= N_POS]
    seed_specs.append((k, l, rng.choice(ok)))
    print("auto seed at %s L%d: %d eligible latents (freq %.0e-%.0e, >=%d sequences) -> latent %d"
          % (k, l, len(ok), FREQ_LO, FREQ_HI, N_POS, seed_specs[-1][2]), flush=True)
    del E

for seed in seed_specs:
    kind, layer, idx = seed
    t_seed = time.time()
    E = examples(kind, layer)
    TOKS = E["tokens"]
    A, S, P = E["activations"][idx], E["seq_ids"][idx], E["positions"][idx]
    order = torch.argsort(A, descending=True)
    seen, pos = set(), []
    for j in order.tolist():
        if float(A[j]) <= 0:
            break
        s_, p_ = int(S[j]), int(P[j])
        if s_ in seen or p_ < 1:
            continue
        seen.add(s_); pos.append((s_, p_, float(A[j])))
        if len(pos) >= N_POS:
            break
    prng = random.Random(RNG + idx)
    prng.shuffle(pos)
    pos_tok = pad([TOKS[s_, :p_ + 1].tolist() for s_, p_, _ in pos])
    pos_anc = torch.tensor([p_ for _, p_, _ in pos])
    stored = torch.tensor([a for _, _, a in pos])
    UP = [(k, l) for l in range(NL) for k in ("att", "mlp", "res")
          if k in UP_KINDS and (l, ORDER[k]) < (layer, ORDER[kind])]
    triple_w = float(TRIPLE_W) if TRIPLE_W else (0.10 if layer <= 7 else 0.05)

    # negatives: random windows, verified silent at the read position
    thr = float(sae((kind, layer))["threshold"][idx])
    neg, tries = [], 0
    while len(neg) < N_NEG and tries < 50:
        tries += 1
        cand = [(prng.randrange(TOKS.shape[0]), prng.randrange(16, TOKS.shape[1])) for _ in range(32)]
        cand = [(s_, p_) for s_, p_ in cand if s_ not in seen]
        ct = pad([TOKS[s_, :p_ + 1].tolist() for s_, p_ in cand])
        pre = batched(ct, torch.tensor([p_ for _, p_ in cand]), seed, {})
        neg += [c for c, v in zip(cand, pre.tolist()) if v <= thr][:N_NEG - len(neg)]
    neg_tok = pad([TOKS[s_, :p_ + 1].tolist() for s_, p_ in neg])
    neg_anc = torch.tensor([p_ for _, p_ in neg])
    del E, TOKS

    tr = (pos_tok[:N_TR], pos_anc[:N_TR]); ho = (pos_tok[N_TR:], pos_anc[N_TR:])
    ntr = (neg_tok[:N_TR], neg_anc[:N_TR]); nho = (neg_tok[N_TR:], neg_anc[N_TR:])
    nat_tr = batched(*tr, seed, {})
    nat_ho = batched(*ho, seed, {})
    a_pos = float(torch.relu(nat_ho).mean())
    dn = max(float((nat_tr ** 2).mean()), 1e-6)
    print("\n" + "=" * 100)
    print("SEED %s L%d #%d | %d upstream sites | positives %d distinct sequences (stored vs recomputed pre at "
          "anchor: median rel err %.3f) | negatives %d verified silent | a_pos %.1f"
          % (kind, layer, idx, len(UP), len(pos),
             float(((torch.cat([nat_tr, nat_ho]).cpu() - stored).abs() / stored.abs()).median()), len(neg), a_pos),
          flush=True)
    ex = pos[:3]
    toks_view = [tok.decode(pos_tok[j, max(0, int(pos_anc[j]) - 10):int(pos_anc[j]) + 1].tolist())
                 .replace("\n", "\\n") for j in range(3)]
    print("  contexts: " + " || ".join("...%s" % v[-60:] for v in toks_view))

    # site means over real positions (1..anchor) of train positives / negatives
    def site_means(split):
        tk_, an_ = split
        acc = {s: torch.zeros(sae(s)["W_enc"].shape[1], device=DEV, dtype=torch.float32) for s in UP}
        cnt = [0.0]
        for s0 in range(0, tk_.shape[0], 8):
            tk, an = tk_[s0:s0 + 8], an_[s0:s0 + 8]
            keep = ((torch.arange(tk.shape[1])[None, :] >= 1) & (torch.arange(tk.shape[1])[None, :] <= an[:, None])).float().to(DEV)
            cnt[0] += float(keep.sum())

            def mk(_s, _k=keep):
                def fn(c):
                    acc[_s] += (c.float() * _k[..., None]).sum(dim=(0, 1))
                    return c
                return fn
            with torch.no_grad():
                run(tk, seed, {s: mk(s) for s in UP}, False)
        return {s: (acc[s] / max(cnt[0], 1.0)).to(DTYPE) for s in UP}

    MU = {"neg": site_means(ntr), "pos": site_means(tr)} if UP else {"neg": {}, "pos": {}}

    # ANCHOR SUPPORT, 030-cross-sae definition (anchor_support.py): per upstream site,
    # codes at the anchor positions of the train positives; "live" = nonzero at the
    # anchor in any context; support = mean fraction of live latents nonzero per
    # context, averaged over sites. TopK 0.2-0.3%, dense ReLU 6.5-37% (030/033).
    def anchor_support(split):
        tk_, an_ = split
        cap = {s: [] for s in UP}
        for s0 in range(0, tk_.shape[0], 8):
            tk, an = tk_[s0:s0 + 8], an_[s0:s0 + 8]

            def mk(_s, _an=an):
                def fn(c):
                    cap[_s].append((c[torch.arange(c.shape[0], device=DEV), _an.to(DEV)] > 0).cpu())
                    return c
                return fn
            with torch.no_grad():
                run(tk, seed, {s: mk(s) for s in UP}, False)
        fr, live, l0 = [], [], []
        for s in UP:
            nz = torch.cat(cap[s]).float()
            lm = nz.bool().any(0)
            fr.append(float(nz[:, lm].mean()) if lm.any() else 0.0)
            live.append(int(lm.sum()))
            l0.append(float(nz.sum(1).mean()))
        return (float(np.mean(fr)) if fr else float("nan"), float(np.mean(live)) if live else 0.0,
                float(np.mean(l0)) if l0 else 0.0)

    supp, live_site, l0_anchor = anchor_support(tr) if UP else (float("nan"), 0.0, 0.0)
    print("  anchor support %.4f | live latents per site %.0f | L0 at anchor %.0f"
          % (supp, live_site, l0_anchor), flush=True)

    def fit(support=None, steps=STEPS):
        """support None -> learn gates + amplitudes; else {site: idx tensor}: gates frozen open on
        exactly those latents, amplitudes fitted, no L1 (the fitted null)."""
        P_ = {}
        for s in UP:
            W = sae(s)["W_enc"].shape[1]
            if support is None:
                th = torch.full((W,), THETA0, device=DEV, requires_grad=True)
            else:
                th = torch.full((W,), -40.0, device=DEV)
                if s in support and len(support[s]):
                    th[support[s].to(DEV)] = 40.0
            ps = torch.full((W,), math.log(math.e - 1.0), device=DEV, requires_grad=True)
            P_[s] = (th, ps)
        params = [ps for _, ps in P_.values()] + ([th for th, _ in P_.values()] if support is None else [])
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.05)
        T_ = [1.0]

        def trf(frame):
            t = {}
            for s in UP:
                th, ps = P_[s]

                def fn(c, _th=th, _ps=ps, _s=s):
                    m = torch.sigmoid(_th / T_[0])
                    kept = (m * F.softplus(_ps)).to(c.dtype) * c
                    if frame == "zero":
                        return kept
                    if FLOOR_MODE == "topk":
                        # gates are soft during the anneal; the engine's fill is
                        # binary, so membership for the BUDGET is theta > 0 (m > 0.5)
                        fl = _topk_fill(c.detach(), MU[frame][_s], _th.detach() > 0,
                                        _CUR_REF[_s]["l0"])
                    else:
                        fl = MU[frame][_s].to(c.dtype)
                    return kept + (1.0 - m).to(c.dtype) * fl
                t[s] = fn
            return t
        terms = [("zero", 1.0), ("neg", DUAL_W), ("pos", triple_w)]
        for step in range(steps):
            T_[0] = 1.0 * (0.05 ** (step / max(steps - 1, 1)))
            opt.zero_grad()
            for j in range(ACCUM):
                s0 = ((step * ACCUM + j) * MICRO) % N_TR
                tk, an, tg = tr[0][s0:s0 + MICRO], tr[1][s0:s0 + MICRO], nat_tr[s0:s0 + MICRO]
                # one clean pass per micro-batch, shared by the three floor terms
                eb = clean_errors(tk, seed, UP) if ((ERROR_MODE == "clean" or BOUND or FLOOR_MODE == "topk") and UP) else None
                for frame, w in terms:
                    v = at(run(tk, seed, trf(frame), True, err=eb), an)
                    (w * ((v - tg) ** 2).mean() / dn / ACCUM).backward()
            if support is None:
                pen = LAM * torch.stack([torch.sigmoid(th).sum() for th, _ in P_.values()]).sum()
                pen = pen + LAM * torch.stack([((1.0 - torch.sigmoid(th)) * (F.softplus(ps) - 1.0).abs()).sum()
                                               for th, ps in P_.values()]).sum()
                pen.backward()
            opt.step()
        mem = {}
        with torch.no_grad():
            for s, (th, ps) in P_.items():
                keep = (th > 0).nonzero(as_tuple=True)[0]
                if len(keep):
                    mem[s] = {int(i): float(a) for i, a in zip(keep.tolist(), F.softplus(ps)[keep].tolist())}
        return mem

    def circuit(members, frame="zero", use_amps=True, fill="dense"):
        t = {}
        for s in UP:
            d = members.get(s, {})
            ii = torch.tensor(sorted(d), device=DEV, dtype=torch.long)
            aa = torch.tensor([d[int(i)] if use_amps else 1.0 for i in ii.tolist()], device=DEV)
            km = torch.zeros(sae(s)["W_enc"].shape[1], dtype=torch.bool, device=DEV)
            km[ii] = True

            def fn(c, _ii=ii, _aa=aa, _s=s, _km=km):
                if frame == "zero":
                    ch = torch.zeros_like(c)
                elif fill == "topk":
                    ch = _topk_fill(c, MU[frame][_s], _km, _CUR_REF[_s]["l0"])
                else:
                    ch = MU[frame][_s].to(c.dtype).expand_as(c).clone()
                if len(_ii):
                    ch[..., _ii] = c[..., _ii] * _aa.to(c.dtype)
                return ch
            t[s] = fn
        return t

    def score(members, frame="zero", use_amps=True, split=ho, fill="dense"):
        return float(torch.relu(batched(*split, seed, circuit(members, frame, use_amps, fill))).mean())

    def frac(a, e):
        return (a - e) / (a_pos - e) if abs(a_pos - e) > 1e-9 else float("nan")

    t0 = time.time()
    if MEMBERS_FROM:
        # re-score a circuit fitted earlier (no refit, no fitted nulls)
        prev = [r for r in map(json.loads, open(HERE / MEMBERS_FROM)) if r["seed"] == "%s:%d:%d" % seed]
        members = {(k_.split("/")[0], int(k_.split("/")[1])): {int(i): a for i, a in d.items()}
                   for k_, d in prev[-1]["members"].items()} if prev else {}
        print("  members loaded from %s: %d" % (MEMBERS_FROM, sum(len(d) for d in members.values())), flush=True)
    else:
        members = fit() if UP else {}
    fit_s = time.time() - t0
    n_mem = sum(len(d) for d in members.values())
    per_kind = {k: sum(len(d) for s, d in members.items() if s[0] == k) for k in ("att", "mlp", "res")}
    empty = {f: score({}, f) for f in ("zero", "pos", "neg")}
    circ = {f: score(members, f) for f in ("zero", "pos", "neg")}
    a1 = score(members, "zero", use_amps=False)
    # the same circuit under the k-sparse fill (freeM_topk), always reported
    empty_tk = {f: score({}, f, fill="topk") for f in ("pos", "neg")}
    circ_tk = {f: score(members, f, fill="topk") for f in ("pos", "neg")}
    # permuted null
    prn = random.Random(RNG + 7 * idx)
    perm, fsup = {}, {}
    amps_all = [a for d in members.values() for a in d.values()]
    prn.shuffle(amps_all)
    ptr = 0
    for s, d in members.items():
        W = sae(s)["W_enc"].shape[1]
        ids = prn.sample(range(W), len(d))
        perm[s] = dict(zip(ids, amps_all[ptr:ptr + len(d)])); ptr += len(d)
        fsup[s] = torch.tensor(prn.sample(range(W), len(d)), dtype=torch.long)
    perm_s = score(perm, "zero")
    fnull = []
    for _ in range(0 if MEMBERS_FROM else N_FNULL):
        nm = fit(support=fsup)
        fnull.append(dict({f: frac(score(nm, f), empty[f]) for f in ("zero", "pos", "neg")},
                          **{f + "_tk": frac(score(nm, f, fill="topk"), empty_tk[f]) for f in ("pos", "neg")}))
    # suppression: zero the members, everything else natural
    abl = {}
    for s, d in members.items():
        ii = torch.tensor(sorted(d), device=DEV, dtype=torch.long)

        def fn(c, _ii=ii):
            ch = c.clone(); ch[..., _ii] = 0; return ch
        abl[s] = fn
    a_abl = float(torch.relu(batched(*ho, seed, abl)).mean()) if members else a_pos
    sup = 1 - a_abl / a_pos if a_pos > 1e-9 else float("nan")

    # NECESSITY PANEL (solution F + leak test). Why is sup weak / sign-flipping?
    #   by kind   zero only the res members / only the att+mlp members
    #   mean      members set to their negctx mean, not zero (zero ablation is
    #             itself off-distribution)
    #   leak      the same zero ablation under the OTHER error mode — if clean
    #             error re-injects the members' information, sup_current > sup
    #   redundant members PLUS every latent at other upstream res sites whose
    #             decoder has cosine >= RED_COS with a res member's decoder (the
    #             copies a residual feature leaves at neighbouring layers), vs a
    #             matched random expansion (same count per site)
    def sup_of(abl_sets, fill="zero"):
        tr_ = {}
        for s, ii in abl_sets.items():
            if not len(ii):
                continue

            def fn(c, _ii=ii, _s=s):
                ch = c.clone()
                ch[..., _ii] = 0 if fill == "zero" else MU["neg"][_s][_ii].to(c.dtype)
                return ch
            tr_[s] = fn
        if not tr_:
            return float("nan")
        return 1 - float(torch.relu(batched(*ho, seed, tr_)).mean()) / a_pos if a_pos > 1e-9 else float("nan")
    msets = {s: torch.tensor(sorted(d), device=DEV, dtype=torch.long) for s, d in members.items()}
    panel = {}
    if members:
        panel["sup_res"] = sup_of({s: v for s, v in msets.items() if s[0] == "res"})
        panel["sup_attmlp"] = sup_of({s: v for s, v in msets.items() if s[0] != "res"})
        panel["sup_mean"] = sup_of(msets, "mean")
        em0 = ERROR_MODE
        ERROR_MODE = "current" if em0 == "clean" else "clean"
        panel["sup_other_error_mode"] = sup_of(msets)
        ERROR_MODE = em0
        res_dirs = [F.normalize(sae(s)["W_dec"][v].float(), dim=-1) for s, v in msets.items() if s[0] == "res"]
        if res_dirs:
            D = torch.cat(res_dirs)
            red, rnd = {}, {}
            grng = torch.Generator(device="cpu").manual_seed(RNG + idx)
            for s in UP:
                if s[0] != "res":
                    continue
                cs = F.normalize(sae(s)["W_dec"].float(), dim=-1) @ D.T
                extra = (cs.max(1).values >= RED_COS).nonzero(as_tuple=True)[0]
                own = msets.get(s, torch.empty(0, dtype=torch.long, device=DEV))
                extra = extra[~torch.isin(extra, own)]
                red[s] = torch.cat([own, extra])
                W = cs.shape[0]
                rnd[s] = torch.cat([own, torch.randperm(W, generator=grng)[:len(extra)].to(DEV)])
            full = {**msets, **red}
            fullr = {**msets, **rnd}
            panel["red_added"] = int(sum(len(v) for v in red.values()) - sum(len(v) for s, v in msets.items() if s[0] == "res"))
            panel["sup_redundant"] = sup_of(full)
            panel["sup_random_expansion"] = sup_of(fullr)
    print("  necessity panel " + " | ".join("%s %s" % (k_, ("%.3f" % v) if isinstance(v, float) else v)
                                            for k_, v in panel.items()), flush=True)

    # ---- the paper's Table 1 rows, amplitude-aware (EVAL_FAMILY=1) ----------
    family = {}

    def table1_family():
        def member_roles():
            """activator / inhibitor per member by attribution: scale latent i by
            w_i (ones), backprop the seed's anchor pre-activation, and read
            d seed / d w_i = (d seed / d c_i) * c_i — gradient x activation, summed
            over train positives. w is used instead of c itself because the
            identity transform (c_hat - c) cancels and would give zero gradient.
            One backward per micro-batch covers every site at once."""
            acc = {s: torch.zeros(len(v), device=DEV) for s, v in msets.items()}
            for s0 in range(0, N_TR, 4):
                tk, an = tr[0][s0:s0 + 4], tr[1][s0:s0 + 4]
                W_ = {s: torch.ones(len(v), device=DEV, requires_grad=True) for s, v in msets.items()}

                def mk(_s):
                    def fn(c):
                        ch = c.clone()
                        ch[..., msets[_s]] = c[..., msets[_s]] * W_[_s].to(c.dtype)
                        return ch
                    return fn
                pre = run(tk, seed, {s: mk(s) for s in msets}, True)
                g = torch.autograd.grad(at(pre, an).sum(), [W_[s] for s in msets], allow_unused=True)
                for s, gi in zip(msets, g):
                    if gi is not None:
                        acc[s] += gi.detach().float()
            return {s: (v >= 0) for s, v in acc.items()}, {s: v.tolist() for s, v in acc.items()}

        act, attr = member_roles()
        amps_of = {s: torch.tensor([members[s][int(i)] for i in v.tolist()], device=DEV)
                   for s, v in msets.items()}

        def role_run(split, mode):
            trf_ = {}
            for s, ii in msets.items():
                a_, is_act = amps_of[s], act[s]

                def fn(c, _ii=ii, _a=a_, _k=is_act, _s=s):
                    ch = c.clone()
                    if mode == "cf":        # contrast: activators injected, inhibitors off
                        v = torch.where(_k, _a * pins[_s][_ii], torch.zeros_like(_a))
                        ch[..., _ii] = v.to(c.dtype)
                    elif mode == "sup":     # positives: activators off, inhibitors at negctx mean
                        v = torch.where(_k, torch.zeros_like(_a), MU["neg"][_s][_ii].float())
                        ch[..., _ii] = v.to(c.dtype)
                    else:                   # sup_alpha: remove the CLAIMED contribution
                        ch[..., _ii] = c[..., _ii] * (1.0 - _a).clamp(min=0).to(c.dtype)
                    return ch
                trf_[s] = fn
            return float(torch.relu(batched(*split, seed, trf_)).mean())

        a_cf_r = role_run(nho, "cf")
        family["cf_alpha_role"] = ((a_cf_r - a_base) / (a_pos - a_base)) if abs(a_pos - a_base) > 1e-9 else float("nan")
        family["sup_role"] = 1 - role_run(ho, "sup") / a_pos if a_pos > 1e-9 else float("nan")
        family["sup_alpha"] = 1 - role_run(ho, "sup_a") / a_pos if a_pos > 1e-9 else float("nan")
        family["n_activator"] = int(sum(int(v.sum()) for v in act.values()))
        family["n_inhibitor"] = int(sum(int((~v).sum()) for v in act.values()))

        def value_ratio():
            """WHY IS phi_pin^a ~ 0 WHILE free0 ~ 1? Compare each member's value in
            the FREE counterfactual (re-encoded from the edited stream, zero fill)
            with its CLEAN value at the same anchor. Ratio >> 1 means the circuit
            drives the seed with INFLATED members, which pinning to clean removes.
            Also reports how close those values sit to the per-token clamp, which
            is what our cap+clamp actually bounds (token max over the dictionary,
            NOT the latent's own maximum)."""
            rr, near = [], []
            for s0 in range(0, ho[0].shape[0], 8):
                tk, an = ho[0][s0:s0 + 8], ho[1][s0:s0 + 8]
                ref = clean_errors(tk, seed, UP, keep=msets)
                cap_, trf_ = {}, {}
                for s in UP:
                    ii = msets.get(s)

                    def fn(c, _ii=ii, _s=s):
                        ch = torch.zeros_like(c)
                        if _ii is not None and len(_ii):
                            ch[..., _ii] = c[..., _ii] * amps_of[_s].to(c.dtype)
                            cap_[_s] = c[..., _ii].detach()
                        return ch
                    trf_[s] = fn
                with torch.no_grad():
                    run(tk, seed, trf_, False, err=ref)
                # compare over the WHOLE window the circuit acts on (positions
                # 1..anchor), not just the anchor: upstream members act earlier,
                # and phi_pin pins them at every position.
                pos = torch.arange(tk.shape[1], device=DEV)[None, :]
                win = (pos >= 1) & (pos <= an.to(DEV)[:, None])
                for s in msets:
                    if s not in cap_:
                        continue
                    fv = cap_[s].float()
                    cv = ref[s]["clean_c"].float()
                    mx = ref[s]["mx"].float()[..., None]
                    m = (cv > 0) & win[..., None]
                    if m.any():
                        rr.append((fv[m] / cv[m]).cpu())
                        near.append((fv / mx.clamp_min(1e-6)).flatten().cpu())
            if not rr:
                return
            r_ = torch.cat(rr); n_ = torch.cat(near)
            family["val_ratio_median"] = float(r_.median())
            family["val_ratio_p90"] = float(r_.quantile(0.9))
            family["val_frac_above_clean"] = float((r_ > 1.5).float().mean())
            family["val_frac_at_clamp"] = float((n_ > 0.9).float().mean())
            print("  member values free/clean: median %.2f p90 %.2f | %.1f%% above 1.5x clean | "
                  "%.1f%% at the per-token clamp"
                  % (family["val_ratio_median"], family["val_ratio_p90"],
                     100 * family["val_frac_above_clean"], 100 * family["val_frac_at_clamp"]), flush=True)

        def pin_alpha(fill, who="all"):
            """phi_pin^a: members CLAMPED to alpha * their clean value, non-members
            filled per pi. Isolates node selection from the re-encoding burden."""
            def one(split):
                vals = []
                for s0 in range(0, split[0].shape[0], 8):
                    tk, an = split[0][s0:s0 + 8], split[1][s0:s0 + 8]
                    ref = clean_errors(tk, seed, UP, keep=msets)
                    trf_ = {}
                    for s in UP:
                        ii = msets.get(s)

                        def fn(c, _ii=ii, _s=s, _r=ref):
                            if fill == "zero":
                                ch = torch.zeros_like(c)
                            elif fill == "topk":
                                km = torch.zeros(c.shape[-1], dtype=torch.bool, device=c.device)
                                if _ii is not None:
                                    km[_ii] = True
                                ch = _topk_fill(c, MU["pos"][_s], km, _r[_s]["l0"])
                            else:
                                ch = MU["pos"][_s].to(c.dtype).expand_as(c).clone()
                            if _ii is not None and len(_ii):
                                v = _r[_s]["clean_c"] * amps_of[_s].to(c.dtype)
                                if who == "act":
                                    # pin ACTIVATORS to clean, leave inhibitors free:
                                    # separates "restoring inhibition" from "members
                                    # are silent in the free counterfactual"
                                    v = torch.where(act[_s], v, c[..., _ii] * amps_of[_s].to(c.dtype))
                                ch[..., _ii] = v
                            return ch
                        trf_[s] = fn
                    with torch.no_grad():
                        vals.append(at(run(tk, seed, trf_, False, err=ref), an))
                return float(torch.relu(torch.cat(vals)).mean())
            return one(ho)
        value_ratio()
        for fl in ("zero", "dense", "topk"):
            e_ = empty["zero"] if fl == "zero" else (empty["pos"] if fl == "dense" else empty_tk["pos"])
            for who in ("all", "act"):
                a_p = pin_alpha(fl, who)
                key = "pin_alpha_" + fl + ("" if who == "all" else "_activators")
                family[key] = (a_p - e_) / (a_pos - e_) if abs(a_pos - e_) > 1e-9 else float("nan")
        print("  TABLE 1 (alpha-aware)  free0_a %.3f | freeM_a dense %.3f topk %.3f | freeN_a topk %.3f"
              % (frac(circ["zero"], empty["zero"]), frac(circ["pos"], empty["pos"]),
                 frac(circ_tk["pos"], empty_tk["pos"]), frac(circ_tk["neg"], empty_tk["neg"])))
        print("     phi_cf^a role %.3f (vs no-role %.3f) | phi_sup role %.3f | phi_sup^a %.3f | "
              "activators %d inhibitors %d | phi_pin^a zero %.3f dense %.3f topk %.3f"
              % (family["cf_alpha_role"], cf, family["sup_role"], family["sup_alpha"],
                 family["n_activator"], family["n_inhibitor"],
                 family["pin_alpha_zero"], family["pin_alpha_dense"], family["pin_alpha_topk"]), flush=True)
    # drive: members set to alpha * pin (mean code at the anchor over train positives) in held-out negatives
    pins = {}
    if members:
        acc = {s: torch.zeros(sae(s)["W_enc"].shape[1], device=DEV) for s in members}

        def mkpin(_s):
            def fn(c):
                acc[_s] += c[torch.arange(c.shape[0], device=DEV), cur_an.to(DEV)].float().sum(0)
                return c
            return fn
        for s0 in range(0, N_TR, 8):
            cur_an = tr[1][s0:s0 + 8]
            with torch.no_grad():
                run(tr[0][s0:s0 + 8], seed, {s: mkpin(s) for s in members}, False)
        pins = {s: acc[s] / N_TR for s in members}
    inj = {}
    for s, d in members.items():
        ii = torch.tensor(sorted(d), device=DEV, dtype=torch.long)
        vv = torch.tensor([d[int(i)] for i in ii.tolist()], device=DEV) * pins[s][ii]

        def fn(c, _ii=ii, _vv=vv):
            ch = c.clone(); ch[..., _ii] = _vv.to(c.dtype); return ch
        inj[s] = fn
    a_base = float(torch.relu(batched(*nho, seed, {})).mean())
    a_inj = float(torch.relu(batched(*nho, seed, inj)).mean()) if members else a_base
    cf = (a_inj - a_base) / (a_pos - a_base) if abs(a_pos - a_base) > 1e-9 else float("nan")
    if EVAL_FAMILY and members:
        table1_family()   # needs pins / a_base / cf, so it runs here

    amps = np.array([a for d in members.values() for a in d.values()]) if n_mem else np.array([1.0])
    vacuous = {f: bool(abs(a_pos - empty[f]) > VAC * a_pos) for f in ("zero", "pos", "neg")}
    rec = dict(seed="%s:%d:%d" % seed, up_kinds=UP_KINDS, error_mode=ERROR_MODE, bound="+".join(BOUND), vacuous=vacuous, a_abl=a_abl,
               anchor_support=supp, live_per_site=live_site, l0_at_anchor=l0_anchor,
               n_upstream_sites=len(UP), n_pos=len(pos), n_neg=len(neg), a_pos=a_pos,
               lam=LAM, steps=STEPS, triple_w=triple_w, n_members=n_mem, per_kind=per_kind, fit_s=round(fit_s, 1),
               F0=frac(circ["zero"], empty["zero"]), FMd=frac(circ["pos"], empty["pos"]),
               FMn=frac(circ["neg"], empty["neg"]), F0_a1=frac(a1, empty["zero"]),
               floor_mode=FLOOR_MODE,
               FMd_tk=frac(circ_tk["pos"], empty_tk["pos"]), FMn_tk=frac(circ_tk["neg"], empty_tk["neg"]),
               empty_tk=empty_tk,
               perm_null_F0=frac(perm_s, empty["zero"]), fitted_nulls=fnull, sup=sup, cf=cf,
               necessity_panel=panel, members_from=MEMBERS_FROM or None, table1=family,
               empty={f: v for f, v in empty.items()}, a_base=a_base,
               alpha_median=float(np.median(amps)), alpha_p90=float(np.percentile(amps, 90)),
               members={"%s/%d" % s: {str(i): round(a, 4) for i, a in d.items()} for s, d in members.items()})
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    fmt = lambda x: "%.3f" % x if x == x else "nan"
    print("  members %d %s | fit %.0fs | alpha median %.2f p90 %.2f" % (n_mem, per_kind, fit_s, rec["alpha_median"], rec["alpha_p90"]))
    vflag = lambda f: " [VACUOUS]" if vacuous[f] else ""
    print("  held-out faithfulness  zero %s%s | posctx-mean %s%s | negctx-mean %s%s | zero@alpha=1 %s"
          % (fmt(rec["F0"]), vflag("zero"), fmt(rec["FMd"]), vflag("pos"), fmt(rec["FMn"]), vflag("neg"),
             fmt(rec["F0_a1"])))
    print("  k-sparse fill (freeM_topk, fit floor = %s)  posctx-mean %s | negctx-mean %s | empties %.2f / %.2f"
          % (FLOOR_MODE, fmt(rec["FMd_tk"]), fmt(rec["FMn_tk"]), empty_tk["pos"], empty_tk["neg"]))
    print("  nulls                  permuted %s | fitted %s"
          % (fmt(rec["perm_null_F0"]), " ; ".join("zero %s pos %s neg %s" % (fmt(n["zero"]), fmt(n["pos"]), fmt(n["neg"]))
                                                   for n in fnull)))
    print("  necessity sup %s | drive cf %s | empties zero %.2f pos %.2f neg %.2f | seed total %.0fs"
          % (fmt(sup), fmt(cf), empty["zero"], empty["pos"], empty["neg"], time.time() - t_seed), flush=True)
