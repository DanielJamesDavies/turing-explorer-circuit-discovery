"""LATENT-ENDPOINT TRI-AMP CIRCUITS on GPT-2 small + the OpenAI v5 TopK SAEs.

The paper's object on a PUBLIC model with a CAPPED dictionary and FOUR site
kinds. Recipe mirrors the production engine and 052: objective "pos" (reproduce
the seed's natural pre-activation at each context's anchor), TRIPLE floor (zero
1.0 + negctx-mean DUAL_W + posctx-mean TRIPLE_W), free amplitudes
alpha = softplus(psi) with leak charge LAM*(1-sigmoid(theta))*|alpha-1|, gate L1
LAM*sum sigmoid(theta), annealed binarisation m = sigmoid(theta/T), T 1 -> 0.05,
400 steps, AdamW lr 0.05 wd 0.05, data terms divided by mean(target^2).

SUBSTRATE (verified in README: identity edit is bitwise exact, L0 == 32.0):
  sites      attn-out < resid-mid < mlp-out < resid-post, 12 layers = 48 sites
  edit       x <- x + ((c_hat - c) @ W_dec) * std     (per-token std; the SAE's
             layer_norm input convention, mean and b_dec cancel in the delta)
  code       topk_32(relu((x_n - b_dec) @ W_enc + b_enc))  — capped, so 052's
             cap/clamp are unnecessary here and ERROR_MODE=current (the
             TuringLLM/033 reference semantics) is the default.

CONTEXTS
  positives  Neuronpedia top-activating contexts for the seed latent (~45 per
             latent), each truncated to its peak position; N_TR_FRAC train.
  negatives  wikitext windows, read position verified SILENT (seed code == 0).

SCORES (held-out, relu of the seed pre-activation at the anchor), all
amplitude-aware, the paper's Table 1 family:
  free0 / freeM_dense / freeM_topk / freeN_topk, phi_pin^a (members clamped to
  alpha x clean), phi_sup (role-aware: activators -> 0, inhibitors -> negctx
  mean), phi_cf^a (contrast contexts, role-aware), plus permuted and fitted
  nulls. Roles from attribution (grad x activation on train positives).

  SEEDS=resid-post:6:100,mlp-out:6:1234 python experiments/053-gpt2-topk/fit_latent_seed.py
Env: N_POS 48 | N_NEG 48 | N_TR_FRAC 0.75 | STEPS 400 | LR 0.05 | LAM 1e-3 |
     MICRO 2 | ACCUM 2 | N_FNULL 1 | DUAL_W 0.25 | TRIPLE_W (depth rule) |
     ERROR_MODE current | DTYPE float32 | RNG 0 | OUT_FILE
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gpt2saes as G  # noqa: E402
import neuronpedia as NP  # noqa: E402

N_POS = int(os.environ.get("N_POS", 48)); N_NEG = int(os.environ.get("N_NEG", 48))
N_TR_FRAC = float(os.environ.get("N_TR_FRAC", 0.75))
STEPS = int(os.environ.get("STEPS", 400)); LR = float(os.environ.get("LR", 0.05))
LAM = float(os.environ.get("LAM", 1e-3))
THETA0 = float(os.environ.get("THETA0", 4.0))
DUAL_W = float(os.environ.get("DUAL_W", 0.25)); TRIPLE_W = os.environ.get("TRIPLE_W")
MICRO = int(os.environ.get("MICRO", 2)); ACCUM = int(os.environ.get("ACCUM", 2))
N_FNULL = int(os.environ.get("N_FNULL", 1))
ERROR_MODE = os.environ.get("ERROR_MODE", "current")
RNG = int(os.environ.get("RNG", 0))
SEQ = int(os.environ.get("SEQ", 64))
DTYPE = getattr(torch, os.environ.get("DTYPE", "float32"))
OUT = HERE / os.environ.get("OUT_FILE", "latent_circuits.jsonl")
DEV = torch.device("cuda")

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=DTYPE).to(DEV).eval()
for p_ in model.parameters():
    p_.requires_grad_(False)
NL = model.config.n_layer
_SAE = {}


def sae(site):
    if site not in _SAE:
        _SAE[site] = G.load_sae(site[0], site[1], device=DEV, dtype=DTYPE)
    return _SAE[site]


class _Stop(Exception):
    pass


def _reg(hs, site, fn_x):
    """fn_x(x) -> x' replaces the site's activation."""
    mod = G.module_for(model, site[0], site[1])
    if site[0] == "resid-mid":
        hs.append(mod.register_forward_pre_hook(lambda m, a: (fn_x(a[0]),) + tuple(a[1:])))
    elif site[0] == "mlp-out":
        hs.append(mod.register_forward_hook(lambda m, a, o: fn_x(o)))
    else:
        def post(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            y = fn_x(x)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hs.append(mod.register_forward_hook(post))


def _edit(site, x, fn, ref=None):
    t = sae(site)
    xn, std, mu = G.norm_in(x)
    c = G.encode(t, xn)
    delta = (fn(c) - c).to(xn.dtype) @ t["W_dec"]
    if ERROR_MODE == "clean" and ref is not None:
        delta = delta + (ref["err"] - (xn - (c @ t["W_dec"] + t["b_dec"])))
    return x + delta * std


def clean_refs(tokens, seed, sites):
    """per site on the UNEDITED run: SAE error (normalised space), the clean code
    at member latents is taken separately by pin_vals."""
    err, hs = {}, []

    def capf(site):
        def f(x):
            t = sae(site)
            xn, _, _ = G.norm_in(x)
            c = G.encode(t, xn)
            err[site] = dict(err=(xn - (c @ t["W_dec"] + t["b_dec"])).detach())
            return x
        return f

    def stop(x):
        raise _Stop()
    for s in sites:
        _reg(hs, s, capf(s))
    _reg(hs, (seed[0], seed[1]), stop)
    try:
        with torch.no_grad():
            model(tokens.to(DEV))
    except _Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    return err


def run(tokens, seed, transforms, grad, err=None):
    """seed pre-activation [B, T] (float32) with `transforms` applied upstream."""
    kind, layer, idx = seed
    out, handles = {}, []
    if ERROR_MODE == "clean" and transforms and err is None:
        err = clean_refs(tokens, seed, list(transforms))
    errs = err or {}
    for site, fn in transforms.items():
        _reg(handles, site, lambda x, _s=site, _f=fn, _e=errs.get(site): _edit(_s, x, _f, _e))

    def read(x):
        t = sae((kind, layer))
        xn, _, _ = G.norm_in(x)
        out["pre"] = ((xn.float() - t["b_dec"].float()) @ t["W_enc"][:, idx].float()
                      + t["b_enc"][idx].float())
        raise _Stop()
    _reg(handles, (kind, layer), read)
    try:
        with torch.set_grad_enabled(grad):
            model(tokens.to(DEV))
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
    return torch.tensor([s + [tok.eos_token_id] * (T - len(s)) for s in seqs], dtype=torch.long)


# ---------------------------------------------------------------- corpus
def wikitext_windows(n, seq=SEQ):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                        filename="wikitext-103-raw-v1/train-00000-of-00002.parquet")
    txt = "".join(t for t in pq.read_table(p).column("text").to_pylist()[:40000] if len(t) > 200)
    ids = tok(txt, return_tensors="pt").input_ids[0]
    n = min(n, len(ids) // seq)
    return torch.stack([ids[i * seq:(i + 1) * seq] for i in range(n)])


seed_specs = []
for s in [x for x in os.environ.get("SEEDS", "").split(",") if x]:
    k, l, i = s.split(":"); seed_specs.append((k, int(l), int(i)))

CORPUS = None
for seed in seed_specs:
    kind, layer, idx = seed
    t_seed = time.time()
    rng = random.Random(RNG + idx)

    # positives: Neuronpedia contexts, truncated at the peak position
    rows = NP.contexts(kind, layer, idx, N_POS)
    if len(rows) < 8:
        print("SEED %s:%d:%d — only %d Neuronpedia contexts, skipping"
              % (kind, layer, idx, len(rows)), flush=True)
        continue
    # SHUFFLE before splitting: Neuronpedia returns contexts strongest-first, so
    # an unshuffled split puts the WEAKEST contexts in held-out and deflates
    # a_pos (measured: 1.26 against a stored peak of 3.8), which makes every
    # ratio noisy. 052 shuffles for the same reason.
    rng.shuffle(rows)
    pos_ids, pos_anc, stored = [], [], []
    for toks_, a_, v_ in rows:
        ids = NP.to_ids(tok, toks_)[:a_ + 1]
        if len(ids) < 2:
            continue
        pos_ids.append(ids); pos_anc.append(len(ids) - 1); stored.append(v_)
    pos_tok = pad(pos_ids); pos_anchor = torch.tensor(pos_anc)
    n_tr = max(2, int(round(len(pos_ids) * N_TR_FRAC)))

    UP = [(k, l) for l in range(NL) for k in G.KINDS
          if (l, G.ORDER[k]) < (layer, G.ORDER[kind])]
    triple_w = float(TRIPLE_W) if TRIPLE_W else (0.10 if layer <= 5 else 0.05)

    # negatives: wikitext windows whose read position is verified silent
    if CORPUS is None:
        CORPUS = wikitext_windows(4000)
    neg_ids, neg_anc, tries = [], [], 0
    while len(neg_ids) < N_NEG and tries < 40:
        tries += 1
        pick = [rng.randrange(CORPUS.shape[0]) for _ in range(32)]
        ps = [rng.randrange(16, SEQ) for _ in pick]
        ct = pad([CORPUS[j, :p + 1].tolist() for j, p in zip(pick, ps)])
        ca = torch.tensor([p for p in ps])
        pre = batched(ct, ca, seed, {})
        t = sae((kind, layer))
        # silent = the latent is not among the token's top-k (its code is zero)
        for j, (jj, p) in enumerate(zip(pick, ps)):
            if len(neg_ids) >= N_NEG:
                break
            if float(pre[j]) <= 0:
                neg_ids.append(CORPUS[jj, :p + 1].tolist()); neg_anc.append(p)
    neg_tok = pad(neg_ids) if neg_ids else pos_tok[:1]
    neg_anchor = torch.tensor(neg_anc) if neg_anc else pos_anchor[:1]

    tr = (pos_tok[:n_tr], pos_anchor[:n_tr]); ho = (pos_tok[n_tr:], pos_anchor[n_tr:])
    n_ntr = max(1, int(round(len(neg_ids) * N_TR_FRAC)))
    ntr = (neg_tok[:n_ntr], neg_anchor[:n_ntr]); nho = (neg_tok[n_ntr:], neg_anchor[n_ntr:])
    nat_tr = batched(*tr, seed, {})
    nat_ho = batched(*ho, seed, {})
    a_pos = float(torch.relu(nat_ho).mean())
    dn = max(float((nat_tr ** 2).mean()), 1e-6)
    rel = float((torch.cat([nat_tr, nat_ho]).cpu() - torch.tensor(stored)).abs().div(
        torch.tensor(stored).abs().clamp_min(1e-6)).median())
    print("\n" + "=" * 100)
    print("SEED %s L%d #%d | %d upstream sites | %d positives (train %d, stored vs recomputed "
          "median rel err %.3f) | %d negatives verified silent | a_pos %.2f"
          % (kind, layer, idx, len(UP), len(pos_ids), n_tr, rel, len(neg_ids), a_pos), flush=True)
    lab = [e.get("description") for e in (NP.feature(kind, layer, idx).get("explanations") or [])]
    if lab:
        print("  neuronpedia label (hint only): %s" % lab[0])

    def site_means(split):
        tk_, an_ = split
        acc = {s: torch.zeros(sae(s)["W_enc"].shape[1], device=DEV, dtype=torch.float32) for s in UP}
        cnt = [0.0]
        for s0 in range(0, tk_.shape[0], 8):
            tkb, anb = tk_[s0:s0 + 8], an_[s0:s0 + 8]
            keep = (torch.arange(tkb.shape[1])[None, :] <= anb[:, None]).float().to(DEV)
            cnt[0] += float(keep.sum())

            def mk(_s, _k=keep):
                def fn(c):
                    acc[_s] += (c.float() * _k[..., None]).sum(dim=(0, 1))
                    return c
                return fn
            with torch.no_grad():
                run(tkb, seed, {s: mk(s) for s in UP}, False)
        return {s: (acc[s] / max(cnt[0], 1.0)).to(DTYPE) for s in UP}

    MU = {"neg": site_means(ntr), "pos": site_means(tr)} if UP else {"neg": {}, "pos": {}}

    def fit(support=None, steps=STEPS):
        P_ = {}
        for s in UP:
            W = sae(s)["W_enc"].shape[1]
            if support is None:
                th = torch.full((W,), THETA0, device=DEV, requires_grad=True)
            else:
                th = torch.full((W,), -40.0, device=DEV)
                if s in support and len(support[s]):
                    th[support[s].to(DEV)] = 40.0
            ps = torch.zeros(W, device=DEV, requires_grad=True)
            P_[s] = (th, ps)
        params = [ps for _, ps in P_.values()] + ([th for th, _ in P_.values()] if support is None else [])
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.05)
        T_ = [1.0]

        def trf(frame):
            t_ = {}
            for s in UP:
                th, ps = P_[s]

                def fn(c, _th=th, _ps=ps, _s=s):
                    m = torch.sigmoid(_th / T_[0])
                    kept = (m * F.softplus(_ps)).to(c.dtype) * c
                    if frame == "zero":
                        return kept
                    return kept + (1.0 - m).to(c.dtype) * MU[frame][_s].to(c.dtype)
                t_[s] = fn
            return t_
        terms = [("zero", 1.0), ("neg", DUAL_W), ("pos", triple_w)]
        N_TR_ = tr[0].shape[0]
        for step in range(steps):
            T_[0] = 1.0 * (0.05 ** (step / max(steps - 1, 1)))
            opt.zero_grad()
            for j in range(ACCUM):
                s0 = ((step * ACCUM + j) * MICRO) % max(N_TR_, 1)
                tkb, anb, tgb = tr[0][s0:s0 + MICRO], tr[1][s0:s0 + MICRO], nat_tr[s0:s0 + MICRO]
                if not len(tkb):
                    continue
                eb = clean_refs(tkb, seed, UP) if (ERROR_MODE == "clean" and UP) else None
                for frame, w in terms:
                    v = at(run(tkb, seed, trf(frame), True, err=eb), anb)
                    (w * ((v - tgb) ** 2).mean() / dn / ACCUM).backward()
            pen = 0.0
            for s in UP:
                th, ps = P_[s]
                m = torch.sigmoid(th / T_[0])
                if support is None:
                    pen = pen + LAM * m.sum()
                pen = pen + LAM * ((1 - m) * (F.softplus(ps) - 1).abs()).sum()
            if isinstance(pen, torch.Tensor):
                pen.backward()
            opt.step()
        mem = {}
        with torch.no_grad():
            for s, (th, ps) in P_.items():
                keep = (th > 0).nonzero(as_tuple=True)[0]
                if len(keep):
                    mem[s] = {int(i): float(a) for i, a in zip(keep.tolist(), F.softplus(ps)[keep].tolist())}
        return mem

    def topk_fill(c, mu, kept_mask, k):
        """freeM_topk: members take the budget first, the rest of the token's k
        goes to the highest-mean non-members, everything else exactly zero."""
        busy = ((c > 0) & kept_mask).sum(-1)
        budget = (k - busy).clamp(min=0)
        rank = mu.float().masked_fill(kept_mask, float("-inf")).argsort(descending=True)
        out = torch.zeros_like(c)
        mb = int(budget.max())
        if mb > 0:
            fill = rank[:mb]
            act = torch.arange(mb, device=c.device)[None, None, :] < budget[..., None]
            out[..., fill] = (mu[fill].float().view(1, 1, -1) * act).to(c.dtype)
        return out

    def circuit(members, frame="zero", use_amps=True, fill="dense"):
        t_ = {}
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
                    ch = topk_fill(c, MU[frame][_s], _km, sae(_s)["k"])
                else:
                    ch = MU[frame][_s].to(c.dtype).expand_as(c).clone()
                if len(_ii):
                    ch[..., _ii] = c[..., _ii] * _aa.to(c.dtype)
                return ch
            t_[s] = fn
        return t_

    def score(members, frame="zero", use_amps=True, split=ho, fill="dense"):
        return float(torch.relu(batched(*split, seed, circuit(members, frame, use_amps, fill))).mean())

    def frac(a, e):
        return (a - e) / (a_pos - e) if abs(a_pos - e) > 1e-9 else float("nan")

    t0 = time.time()
    members = fit() if UP else {}
    fit_s = time.time() - t0
    n_mem = sum(len(d) for d in members.values())
    per_kind = {k: sum(len(d) for s, d in members.items() if s[0] == k) for k in G.KINDS}
    empty = {f: score({}, f) for f in ("zero", "pos", "neg")}
    circ = {f: score(members, f) for f in ("zero", "pos", "neg")}
    empty_tk = {f: score({}, f, fill="topk") for f in ("pos", "neg")}
    circ_tk = {f: score(members, f, fill="topk") for f in ("pos", "neg")}
    a1 = score(members, "zero", use_amps=False)

    prn = random.Random(RNG + 7 * idx)
    perm, fsup = {}, {}
    amps_all = [a for d in members.values() for a in d.values()]
    prn.shuffle(amps_all)
    ptr = 0
    for s, d in members.items():
        W = sae(s)["W_enc"].shape[1]
        ids_ = prn.sample(range(W), len(d))
        perm[s] = dict(zip(ids_, amps_all[ptr:ptr + len(d)])); ptr += len(d)
        fsup[s] = torch.tensor(prn.sample(range(W), len(d)), dtype=torch.long)
    perm_s = score(perm, "zero") if members else empty["zero"]
    fnull = []
    for _ in range(N_FNULL if members else 0):
        nm = fit(support=fsup)
        fnull.append({f: frac(score(nm, f), empty[f]) for f in ("zero", "pos", "neg")})

    msets = {s: torch.tensor(sorted(d), device=DEV, dtype=torch.long) for s, d in members.items()}
    amps_of = {s: torch.tensor([members[s][int(i)] for i in v.tolist()], device=DEV)
               for s, v in msets.items()}

    def member_roles():
        """activator / inhibitor by attribution: scale member i by w_i (ones) and
        read d seed / d w_i = (d seed / d c_i) * c_i. A scaling weight is used
        because the identity transform (c_hat - c) cancels and would give zero."""
        acc = {s: torch.zeros(len(v), device=DEV) for s, v in msets.items()}
        for s0 in range(0, tr[0].shape[0], 4):
            tkb, anb = tr[0][s0:s0 + 4], tr[1][s0:s0 + 4]
            W_ = {s: torch.ones(len(v), device=DEV, requires_grad=True) for s, v in msets.items()}

            def mk(_s):
                def fn(c):
                    ch = c.clone()
                    ch[..., msets[_s]] = c[..., msets[_s]] * W_[_s].to(c.dtype)
                    return ch
                return fn
            pre = run(tkb, seed, {s: mk(s) for s in msets}, True)
            g = torch.autograd.grad(at(pre, anb).sum(), [W_[s] for s in msets], allow_unused=True)
            for s, gi in zip(msets, g):
                if gi is not None:
                    acc[s] += gi.detach().float()
        return {s: (v >= 0) for s, v in acc.items()}

    act = member_roles() if members else {}
    pins = {}
    if members:
        accp = {s: torch.zeros(sae(s)["W_enc"].shape[1], device=DEV) for s in members}

        def mkpin(_s):
            def fn(c):
                accp[_s] += c[torch.arange(c.shape[0], device=DEV), cur_an.to(DEV)].float().sum(0)
                return c
            return fn
        for s0 in range(0, tr[0].shape[0], 8):
            cur_an = tr[1][s0:s0 + 8]
            with torch.no_grad():
                run(tr[0][s0:s0 + 8], seed, {s: mkpin(s) for s in members}, False)
        pins = {s: accp[s] / max(tr[0].shape[0], 1) for s in members}

    def role_run(split, mode):
        t_ = {}
        for s, ii in msets.items():
            a_, kk = amps_of[s], act[s]

            def fn(c, _ii=ii, _a=a_, _k=kk, _s=s):
                ch = c.clone()
                if mode == "cf":
                    v = torch.where(_k, _a * pins[_s][_ii], torch.zeros_like(_a))
                elif mode == "sup":
                    v = torch.where(_k, torch.zeros_like(_a), MU["neg"][_s][_ii].float())
                else:
                    v = None
                ch[..., _ii] = (c[..., _ii] * (1.0 - _a).clamp(min=0).to(c.dtype)
                                if v is None else v.to(c.dtype))
                return ch
            t_[s] = fn
        return float(torch.relu(batched(*split, seed, t_)).mean()) if t_ else float("nan")

    def pin_alpha(fill):
        """phi_pin^a: members clamped to alpha x their CLEAN values."""
        vals = []
        for s0 in range(0, ho[0].shape[0], 8):
            tkb, anb = ho[0][s0:s0 + 8], ho[1][s0:s0 + 8]
            capc, hs_ = {}, []
            for s in msets:
                def grab(x, _s=s):
                    xn, _, _ = G.norm_in(x)
                    capc[_s] = G.encode(sae(_s), xn)[..., msets[_s]].detach()
                    return x
                _reg(hs_, s, grab)
            try:
                with torch.no_grad():
                    model(tkb.to(DEV))
            finally:
                for h in hs_:
                    h.remove()
            t_ = {}
            for s in UP:
                ii = msets.get(s)

                def fn(c, _ii=ii, _s=s):
                    if fill == "zero":
                        ch = torch.zeros_like(c)
                    elif fill == "topk":
                        km = torch.zeros(c.shape[-1], dtype=torch.bool, device=c.device)
                        if _ii is not None:
                            km[_ii] = True
                        ch = topk_fill(c, MU["pos"][_s], km, sae(_s)["k"])
                    else:
                        ch = MU["pos"][_s].to(c.dtype).expand_as(c).clone()
                    if _ii is not None and len(_ii):
                        ch[..., _ii] = capc[_s] * amps_of[_s].to(c.dtype)
                    return ch
                t_[s] = fn
            with torch.no_grad():
                vals.append(at(run(tkb, seed, t_, False), anb))
        return float(torch.relu(torch.cat(vals)).mean()) if vals else float("nan")

    a_base = float(torch.relu(batched(*nho, seed, {})).mean()) if len(nho[0]) else 0.0
    fam = {}
    if members:
        fam["sup_role"] = 1 - role_run(ho, "sup") / a_pos if a_pos > 1e-9 else float("nan")
        fam["sup_alpha"] = 1 - role_run(ho, "sup_a") / a_pos if a_pos > 1e-9 else float("nan")
        a_cf = role_run(nho, "cf")
        fam["cf_alpha_role"] = ((a_cf - a_base) / (a_pos - a_base)) if abs(a_pos - a_base) > 1e-9 else float("nan")
        fam["n_activator"] = int(sum(int(v.sum()) for v in act.values()))
        fam["n_inhibitor"] = int(sum(int((~v).sum()) for v in act.values()))
        for fl in ("zero", "dense", "topk"):
            e_ = empty["zero"] if fl == "zero" else (empty["pos"] if fl == "dense" else empty_tk["pos"])
            a_p = pin_alpha(fl)
            fam["pin_alpha_" + fl] = (a_p - e_) / (a_pos - e_) if abs(a_pos - e_) > 1e-9 else float("nan")

    amps = np.array([a for d in members.values() for a in d.values()]) if n_mem else np.array([1.0])
    rec = dict(seed="%s:%d:%d" % seed, model="gpt2", width=G.WIDTH, kinds=list(G.KINDS),
               error_mode=ERROR_MODE, n_upstream_sites=len(UP), n_pos=len(pos_ids), n_neg=len(neg_ids),
               n_train=n_tr, a_pos=a_pos, lam=LAM, steps=STEPS, triple_w=triple_w,
               n_members=n_mem, per_kind=per_kind, fit_s=round(fit_s, 1),
               F0=frac(circ["zero"], empty["zero"]), FMd=frac(circ["pos"], empty["pos"]),
               FMn=frac(circ["neg"], empty["neg"]), F0_a1=frac(a1, empty["zero"]),
               FMd_tk=frac(circ_tk["pos"], empty_tk["pos"]), FMn_tk=frac(circ_tk["neg"], empty_tk["neg"]),
               perm_null_F0=frac(perm_s, empty["zero"]), fitted_nulls=fnull, table1=fam,
               empty=empty, empty_tk=empty_tk, a_base=a_base,
               alpha_median=float(np.median(amps)), alpha_p90=float(np.percentile(amps, 90)),
               label=(lab[0] if lab else None),
               members={"%s/%d" % s: {str(i): round(a, 4) for i, a in d.items()} for s, d in members.items()})
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    f_ = lambda x: "%.3f" % x if x == x else "nan"
    print("  members %d %s | fit %.0fs | alpha median %.2f p90 %.2f"
          % (n_mem, per_kind, fit_s, rec["alpha_median"], rec["alpha_p90"]))
    print("  held-out  free0 %s | freeM dense %s topk %s | freeN dense %s topk %s | free0@a=1 %s"
          % (f_(rec["F0"]), f_(rec["FMd"]), f_(rec["FMd_tk"]), f_(rec["FMn"]), f_(rec["FMn_tk"]), f_(rec["F0_a1"])))
    print("  nulls     permuted %s | fitted %s" % (f_(rec["perm_null_F0"]),
          " ; ".join("zero %s pos %s neg %s" % (f_(n["zero"]), f_(n["pos"]), f_(n["neg"])) for n in fnull)))
    if fam:
        print("  TABLE 1   phi_sup role %s | phi_sup^a %s | phi_cf^a %s | activators %d inhibitors %d"
              % (f_(fam["sup_role"]), f_(fam["sup_alpha"]), f_(fam["cf_alpha_role"]),
                 fam["n_activator"], fam["n_inhibitor"]))
        print("            phi_pin^a zero %s dense %s topk %s | seed total %.0fs"
              % (f_(fam["pin_alpha_zero"]), f_(fam["pin_alpha_dense"]), f_(fam["pin_alpha_topk"]),
                 time.time() - t_seed), flush=True)
