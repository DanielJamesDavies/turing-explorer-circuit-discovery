"""Exp 135 (GPU, one process): parent closure and candidate attribution for exp 133's IOI WCM circuits.

GPT-2 small + OAI v5 32k Top-K SAEs (081's engine, 48 sites). Nothing in src / 081 / 132 / 133 is edited; their code is
imported (081 engine + task_common, src SAE bank / CircuitOnlyPatcher / multi_patch). Three stages (STAGES env):

  attr      gradient x activation attribution of the task metric m = logit(IO) - logit(S) at the anchor, per (site,
            latent, ROLE of the position): z = g_f * f (first-order effect of zeroing the latent: the scorer's Z
            frame) and atp = g_f * (f_ABC - f) (attribution patching to the item's own ABC prompt; 132's AtP).
            g_f = W_dec g_x, g_x = d m / d x_site in SAE-input space (132 attrib.py's additive-leaf method).
            Summed over the positions of a role, averaged over items. Roles: io, s1, s1_next, s2, end (the anchor),
            other. Sets: train (the 1,600 training items; also gives the train-active masks), dis (081 held-out,
            disjoint names), sn (133's same-name held-out).
  closure   for each circuit and set, the NATURAL (unablated) pre-activation of every active residual-stream member
            (resid_mid and resid sites) decomposed EXACTLY into upstream parts, at frozen per-token LN scales:
              pre_i(t) = sum_c  w~_i . comp_c(t) / s_tgt(t)  +  b_eff_i,   w~_i = W_enc_i centred (SAE input LN)
            view A (the brief's parents): resid_mid_L = resid_{L-1} + attn_L ; resid_L = resid_{L-1} + attn_L + mlp_L
                    (resid_{-1} = the embedding). Each component is split by ITS site's SAE into in-circuit latents
                    (members at that site), out-of-circuit latents, SAE error and b_dec; plus b_eff (bias).
            view B: resid_L = resid_mid_L + mlp_L (the nearest residual parent, same layer).
            view C (positions explicit): the attn_L component replaced by its exact source-position decomposition,
                    attn_L(t) = sum_h sum_s A_h[t,s] (LN1(x_in[s]) W_V^h + b_V^h) W_O^h + b_O, with LN1's scale
                    frozen at s, and x_in[s] = resid_{L-1}[s] split by the resid_{L-1} SAE (in / out / err / b_dec)
                    or the embedding (L = 0), per ROLE of the source position s.
            Checks: each view's parts sum to the natural activation (max relative residual is logged and saved).
            Plus a FUNCTIONAL closure read: the scorer's Z frame (CircuitOnlyPatcher: members kept at alpha,
            non-members 0, SAE errors live, scope = the circuit's), capturing every member's re-encoded code; retention
            = circuit code / natural code at the member's natural occurrences (before alpha).
  Outputs: OUT/attr_<set>.pt, OUT/closure_<circuit>_<set>.pt, OUT/log lines. analyse135.py makes the tables (CPU).

  WSL, repo root:
  PYTHONPATH=src HF_HUB_OFFLINE=1 MODEL=gpt2-small WCM_VERSION=v4 .venv/bin/python -X utf8 -u \
      experiments/135-ioi-parent-closure/closure135.py
  env: STAGES (attr,closure)  CIRCUITS (comma list of 133 circuit names)  N_ATTR_TR (1600) N_ATTR_HO (500)
       N_DEC_TR (400) N_DEC_HO (500)  BS (16)  OUT (experiments/135-ioi-parent-closure/out)  DEVICE (cuda if any)
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E81 = ROOT / "experiments" / "081-task-circuits"
E133 = ROOT / "experiments" / "133-ioi-wcm-sweep"
sys.path.insert(0, str(E81))
os.environ.setdefault("MODEL", "gpt2-small")
os.environ.setdefault("WCM_VERSION", "v4")
os.environ.setdefault("OUT", str(HERE / "out"))
os.environ.pop("HELDOUT_N", None)
import torch  # noqa: E402

import task_common as C  # noqa: E402

OUT = Path(os.environ.get("OUT135", str(HERE / "out")))
STAGES = [s for s in os.environ.get("STAGES", "attr,closure").split(",") if s]
CIRCUITS = [s for s in os.environ.get(
    "CIRCUITS", "W_1e-3_lr05_pair,W_3e-4_lr05_pair,W_1e-4_lr05_pair,AM_W_3e-3_lr05_pair,AM_W_1e-6_lr05_pair"
).split(",") if s]
N_ATTR_TR = int(os.environ.get("N_ATTR_TR", 1600))
N_ATTR_HO = int(os.environ.get("N_ATTR_HO", 500))
N_DEC_TR = int(os.environ.get("N_DEC_TR", 400))
N_DEC_HO = int(os.environ.get("N_DEC_HO", 500))
BS = int(os.environ.get("BS", 16))
ROLES = ("io", "s1", "s1_next", "s2", "end", "other")
R_IDX = {r: i for i, r in enumerate(ROLES)}
NR = len(ROLES)

# per-occurrence features of the closure decomposition (column names; analyse135.py reads them from the .pt)
_CATS = ("in", "out", "err", "bdec")
FEATS = (["act", "emb", "beff"]
         + ["A_skip_%s" % c for c in _CATS] + ["A_attn_%s" % c for c in _CATS] + ["A_mlp_%s" % c for c in _CATS]
         + ["A_maxpar", "A_maxpar_ge25", "A_maxpar_ge50", "A_in_ge50"]
         + ["B_rmid_%s" % c for c in _CATS] + ["B_maxpar", "B_maxpar_ge50", "B_in_ge50"]
         + ["C_attn_%s_%s" % (c, r) for c in ("in", "out", "err", "bdec", "emb") for r in ROLES]
         + ["C_attn_const", "C_in_ge50", "chkA", "chkB", "chkC", "n"])
F = {k: i for i, k in enumerate(FEATS)}
DBG = {}


def log(s):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), s), flush=True)


# ------------------------------------------------------------------------------------------------ data
def load():
    import engine as E
    eng = E.Engine(C.MODEL)
    T = json.loads((E133 / "tasks" / "ioi_sn.json").read_text())
    tr = list(T["split"]["train"])
    sets = {}
    for name, held in (("dis", list(T["split"]["held"])), ("sn", list(T["split"]["held_sn"]))):
        B = eng.batch(dict(T, split=dict(train=tr, held=held[:max(N_ATTR_HO, N_DEC_HO)])))
        sets[name] = B
    return eng, sets


def rows_of(B, which, n):
    """(tokens, anchors, valid, good, bad, role_idx [N, T], cf_tokens) for the first n rows of the training part
    (which = 'train') or the held-out part."""
    n_tr = B["n_train"]
    lo, hi = (0, min(n, n_tr)) if which == "train" else (n_tr, min(n_tr + n, int(B["tokens"].shape[0])))
    sl = slice(lo, hi)
    tok, anc, val = B["tokens"][sl], B["anchors"][sl], B["valid"][sl]
    good, bad = B["good"][sl, 0], B["bad"][sl, 0]
    W = int(tok.shape[1])
    rid = torch.full((hi - lo, W), R_IDX["other"], dtype=torch.long)
    for k, i in enumerate(range(lo, hi)):
        for t, r in enumerate(B["roles"][i]):
            if r in R_IDX:
                rid[k, t] = R_IDX[r]
        rid[k, int(anc[k])] = R_IDX["end"]
    rid = rid.to(tok.device)
    ct = B["cf_all"][0][sl]
    return tok, anc, val, good, bad, rid, ct


# ------------------------------------------------------------------------------------------------ model access
class Nat:
    """Natural (unablated) forward with model-space captures: x_in[L] (block input), a[L] (attn out), m[L] (mlp out),
    r[L] = x_in + a (resid_mid), o[L] (block output), attention patterns A[L] [B, H, T, T], logits."""

    def __init__(self, eng):
        self.eng = eng
        self.hf = eng.inf.model.hf
        self.blocks = self.hf.transformer.h

    def run(self, tok):
        cap = {}
        hs = []
        for l, blk in enumerate(self.blocks):
            def pre(_m, args, kwargs, l=l):
                cap[("x", l)] = (args[0] if args else kwargs["hidden_states"]).detach()

            def ah(_m, _i, out, l=l):
                cap[("a", l)] = out[0].detach()

            def mh(_m, _i, out, l=l):
                cap[("m", l)] = (out[0] if isinstance(out, tuple) else out).detach()

            def bh(_m, _i, out, l=l):
                cap[("o", l)] = (out[0] if isinstance(out, tuple) else out).detach()
            hs += [blk.register_forward_pre_hook(pre, with_kwargs=True), blk.attn.register_forward_hook(ah),
                   blk.mlp.register_forward_hook(mh), blk.register_forward_hook(bh)]
        try:
            with torch.no_grad():
                o = self.hf(input_ids=tok, use_cache=False, output_attentions=True)
        finally:
            for h in hs:
                h.remove()
        att = o.attentions
        if att is None or len(att) != len(self.blocks) or att[0] is None:
            raise RuntimeError("no attention patterns returned (output_attentions)")
        n = len(self.blocks)
        out = dict(x=[cap[("x", l)].float() for l in range(n)], a=[cap[("a", l)].float() for l in range(n)],
                   m=[cap[("m", l)].float() for l in range(n)], o=[cap[("o", l)].float() for l in range(n)],
                   A=[a.float() for a in att], logits=o.logits.float())
        out["r"] = [out["x"][l] + out["a"][l] for l in range(n)]
        return out


def site_x(nat, l, kind):
    return {"attn": nat["a"], "resid_mid": nat["r"], "mlp": nat["m"], "resid": nat["o"]}[kind][l]


def codes(eng, nat):
    """Per site: x_n, scale s, top-k acts / idx, error eps (all in SAE-input space)."""
    from model.hf_adapters import layer_norm_in
    from sae.dense import sparse_topk_to_dense
    bank = eng.bank
    eps_ln = float(bank.norm_eps)
    out = {}
    for (l, kd) in eng.sites:
        x = site_x(nat, l, kd)
        xn, s = layer_norm_in(x, eps_ln)
        ta, ti = bank.encode(xn, kd, l)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32)
        err = xn - bank.decode(dense, kd, l)
        out[(l, kd)] = dict(xn=xn, s=s[..., 0], a=ta.float(), i=ti.long(), err=err.float())
    return out


def sae_w(eng, l, kd):
    sae = eng.bank.saes[kd][l]
    # views only (no copies: 48 transposed decoders would be another 4.8 GB); Wdec[idx] gathers rows of the view
    return dict(Wenc=sae.encoder.weight, beff=sae._get_bias_eff(), Wdec=sae.decoder.weight.T, bdec=sae.decoder_bias)


# ------------------------------------------------------------------------------------------------ stage: attr
def stage_attr(eng, sets):
    from config import config
    from model.hooks import multi_patch
    config.discovery.autocast_bf16 = False
    inf, bank = eng.inf, eng.bank
    sites = list(eng.sites)
    D = int(bank.d_sae)
    dev = eng.dev
    Wdec = {s: bank.saes[s[1]][s[0]].decoder.weight.float() for s in sites}     # [d_model, D]

    class Add:
        def __init__(self, leaves):
            self.leaves = leaves

        def __call__(self, model):
            return multi_patch(model, self.transform)

        def transform(self, l, kd, x):
            s = (l, kd)
            return x + self.leaves[s].to(x.dtype) if s in self.leaves else None

    nat = Nat(eng)
    jobs = [("train", sets["dis"], "train", N_ATTR_TR), ("dis", sets["dis"], "held", N_ATTR_HO),
            ("sn", sets["sn"], "held", N_ATTR_HO)]
    for name, B, which, n in jobs:
        p = OUT / ("attr_%s.pt" % name)
        if p.exists():
            log("attr %s: exists, skipped" % name)
            continue
        t0 = time.time()
        tok, anc, val, good, bad, rid, ct = rows_of(B, which, n)
        N = int(tok.shape[0])
        acc = {k: {s: torch.zeros(NR, D, dtype=torch.float64, device=dev) for s in sites}
               for k in ("z", "z_abs", "atp", "atp_abs", "act_n")}
        m_full = []
        for s0 in range(0, N, BS):
            sl = slice(s0, min(N, s0 + BS))
            tk, an, vv, g_, b_, rr = tok[sl], anc[sl], val[sl], good[sl], bad[sl], rid[sl]
            nb = int(tk.shape[0])
            cn = codes(eng, nat.run(tk))
            cc = codes(eng, nat.run(ct[sl]))
            leaves = {s: torch.zeros(nb, tk.shape[1], bank.d_model, device=dev, requires_grad=True) for s in sites}
            o = inf.forward(tk, patcher=Add(leaves), grad_enabled=True, all_logits=True, return_activations=False,
                            tokenize_final=False)
            lg = o[1] if isinstance(o, (tuple, list)) else o
            ar = torch.arange(nb, device=lg.device)
            z = lg[ar, an.to(lg.device)].float()
            m = z[ar, g_.to(lg.device)] - z[ar, b_.to(lg.device)]
            grads = torch.autograd.grad(m.sum(), [leaves[s] for s in sites])
            m_full.append(m.detach().cpu())
            oh = torch.nn.functional.one_hot(rr, NR).float() * vv[..., None].float()       # [b, T, R]
            for s, gx in zip(sites, grads):
                gf = gx.float() @ Wdec[s]                                                    # [b, T, D]
                f = torch.zeros_like(gf).scatter_(-1, cn[s]["i"], cn[s]["a"])
                fc = torch.zeros_like(gf).scatter_(-1, cc[s]["i"], cc[s]["a"])
                zt = torch.einsum("btd,btr->brd", gf * f, oh)
                at = torch.einsum("btd,btr->brd", gf * (fc - f), oh)
                acc["z"][s] += zt.sum(0).double()
                acc["z_abs"][s] += zt.abs().sum(0).double()
                acc["atp"][s] += at.sum(0).double()
                acc["atp_abs"][s] += at.abs().sum(0).double()
                acc["act_n"][s] += torch.einsum("btd,btr->rd", (f > 0).float(), oh).double()
                del gf, f, fc, zt, at
            del leaves, grads, cn, cc, lg, o
        mf = torch.cat(m_full)
        d = dict(set=name, n_items=N, roles=ROLES, sites=sites, m_full_mean=float(mf.mean()),
                 **{k: {s: v[s].float().cpu() for s in sites} for k, v in acc.items()})
        torch.save(d, p)
        log("attr %s: %d items, full m %.3f, %.0fs" % (name, N, float(mf.mean()), time.time() - t0))


# ------------------------------------------------------------------------------------------------ stage: closure
def load_circuit(name):
    d = json.loads((E133 / "out" / "circuits" / ("%s.json" % name)).read_text())
    al = C.alphas_from_json(d["alphas"])
    al = {s: v for s, v in al.items() if v}
    scope = d.get("cfg", {}).get("sites", "all")
    return al, scope


class Prep:
    """Per circuit: member lookups, centred encoder rows of resid-type members, view-C OV kernels."""

    def __init__(self, eng, alphas):
        self.eng = eng
        D = int(eng.bank.d_sae)
        dev = eng.dev
        self.memb = {}
        for s in eng.sites:
            mk = torch.zeros(D, dtype=torch.bool, device=dev)
            if s in alphas:
                mk[torch.tensor(sorted(alphas[s]), device=dev)] = True
            self.memb[s] = mk
        self.tgt = {}
        hf = eng.inf.model.hf
        nh = int(hf.config.n_head)
        for s in eng.sites:
            l, kd = s
            if kd not in ("resid_mid", "resid") or s not in alphas:
                continue
            idx = torch.tensor(sorted(alphas[s]), device=dev, dtype=torch.long)
            w = sae_w(eng, l, kd)
            We = w["Wenc"][idx]
            Wt = We - We.mean(1, keepdim=True)                                                # [nM, d]
            loc = torch.full((D,), -1, dtype=torch.long, device=dev)
            loc[idx] = torch.arange(len(idx), device=dev)
            blk = hf.transformer.h[l]
            Wqkv, bqkv = blk.attn.c_attn.weight.float(), blk.attn.c_attn.bias.float()
            d = int(Wqkv.shape[0])
            WV, bV = Wqkv[:, 2 * d:], bqkv[2 * d:]
            WO, bO = blk.attn.c_proj.weight.float(), blk.attn.c_proj.bias.float()
            dh = d // nh
            g1, be1 = blk.ln_1.weight.float(), blk.ln_1.bias.float()
            K = []
            const = bO.clone()
            for h in range(nh):
                ov = WV[:, h * dh:(h + 1) * dh] @ WO[h * dh:(h + 1) * dh, :]                  # [d, d]
                k = g1[:, None] * (ov @ Wt.T)                                                 # [d, nM]
                K.append(k - k.mean(0, keepdim=True))
                const = const + (be1 @ WV[:, h * dh:(h + 1) * dh] + bV[h * dh:(h + 1) * dh]) @ WO[h * dh:(h + 1) * dh, :]
            self.tgt[s] = dict(idx=idx, Wt=Wt, beff=w["beff"][idx], loc=loc, K=torch.stack(K), const=const,
                               ln1_eps=float(blk.ln_1.eps), alphas=torch.tensor([alphas[s][int(i)] for i in idx.tolist()],
                                                                               device=dev))
        self.W = {s: sae_w(self.eng, *s) for s in eng.sites}


def _site_parts(P, cd, src, b, t, Wt, sT, chunk=2048):
    """in / out / err / bdec / maxpar contributions of the source site src at positions (b, t) to members with
    centred encoder rows Wt [n, d]; sT [n] the target scales."""
    c = cd[src]
    W = P.W[src]
    ratio = c["s"][b, t] / sT
    a_s, i_s = c["a"][b, t], c["i"][b, t]                                                       # [n, k]
    inm = P.memb[src][i_s]
    outs = []
    for c0 in range(0, int(Wt.shape[0]), chunk):
        cs = slice(c0, c0 + chunk)
        dots = torch.einsum("nkd,nd->nk", W["Wdec"][i_s[cs]], Wt[cs])
        outs.append(dots)
    dots = torch.cat(outs) if outs else torch.zeros_like(a_s)
    con = a_s * dots * ratio[:, None]
    vin = (con * inm).sum(1)
    vout = (con * ~inm).sum(1)
    mx = con.masked_fill(~inm, float("-inf")).max(1).values
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    err = (c["err"][b, t] * Wt).sum(1) * ratio
    bd = (Wt @ W["bdec"]) * ratio
    return vin, vout, err, bd, mx


def _viewC(P, cd, nat, L, tgt, rid, vv):
    """[B, T, nM, cat, R] source-position decomposition of attn_L(t) read by the target members (divided by s_tgt
    later), cat in (in, out, err, bdec, emb)."""
    T_ = P.tgt[tgt]
    K = T_["K"]                                                                                  # [H, d, nM]
    x_in = nat["x"][L]                                                                           # [B, S, d]
    var = x_in.var(-1, unbiased=False)
    sig1 = torch.sqrt(var + T_["ln1_eps"])                                                       # [B, S]
    Bn, S, d = x_in.shape
    pieces = {}
    if L == 0:
        pieces["emb"] = x_in / sig1[..., None]
    else:
        src = (L - 1, "resid")
        c = cd[src]
        W = P.W[src]
        Wd = W["Wdec"][c["i"]]                                                                   # [B, S, k, d]
        inm = P.memb[src][c["i"]].float()
        r = (c["s"] / sig1)[..., None]
        pieces["in"] = torch.einsum("bsk,bskd->bsd", c["a"] * inm, Wd) * r
        pieces["out"] = torch.einsum("bsk,bskd->bsd", c["a"] * (1 - inm), Wd) * r
        pieces["err"] = c["err"] * r
        pieces["bdec"] = W["bdec"].expand(Bn, S, d) * r
    A = nat["A"][L]                                                                              # [B, H, T, S]
    oh = torch.nn.functional.one_hot(rid, NR).float() * vv[..., None].float()                    # [B, S, R]
    out = {}
    for cat, p in pieces.items():
        Y = torch.einsum("bsd,hdm->bshm", p, K)                                                  # [B, S, H, nM]
        out[cat] = torch.stack([torch.einsum("bhts,bshm->btm", A, Y * oh[:, :, r, None, None])
                                for r in range(NR)], -1)                                         # [B, T, nM, R]
    return out


def closure_batch(P, eng, nat, cd, rid, vv, acc_m, acc_r, chk):
    for tgt, T_ in P.tgt.items():
        L, kd = tgt
        c = cd[tgt]
        loc = T_["loc"][c["i"]]                                                                  # [B, T, k]
        occ = (loc >= 0) & (c["a"] > 0) & vv[..., None]
        b, t, k = occ.nonzero(as_tuple=True)
        if b.numel() == 0:
            continue
        mi = loc[b, t, k]
        act = c["a"][b, t, k]
        Wt = T_["Wt"][mi]
        sT = c["s"][b, t]
        n = int(b.numel())
        f = torch.zeros(n, len(FEATS), device=act.device, dtype=torch.float32)
        f[:, F["act"]] = act
        f[:, F["beff"]] = T_["beff"][mi]
        f[:, F["n"]] = 1.0
        # view A
        if L == 0:
            f[:, F["emb"]] = (nat["x"][0][b, t] * Wt).sum(1) / sT
            skip = f[:, F["emb"]]
            skip_in = torch.zeros_like(act)
            skip_mx = torch.full_like(act, 0.0)
        else:
            vin, vout, err, bd, mx = _site_parts(P, cd, (L - 1, "resid"), b, t, Wt, sT)
            f[:, F["A_skip_in"]], f[:, F["A_skip_out"]], f[:, F["A_skip_err"]], f[:, F["A_skip_bdec"]] = vin, vout, err, bd
            skip = vin + vout + err + bd
            skip_in, skip_mx = vin, mx
        vin, vout, err, bd, mxa = _site_parts(P, cd, (L, "attn"), b, t, Wt, sT)
        f[:, F["A_attn_in"]], f[:, F["A_attn_out"]], f[:, F["A_attn_err"]], f[:, F["A_attn_bdec"]] = vin, vout, err, bd
        attnA = vin + vout + err + bd
        a_in = vin
        mx_all = torch.maximum(skip_mx, mxa)
        mlpA = torch.zeros_like(act)
        m_in = torch.zeros_like(act)
        if kd == "resid":
            vin, vout, err, bd, mxm = _site_parts(P, cd, (L, "mlp"), b, t, Wt, sT)
            f[:, F["A_mlp_in"]], f[:, F["A_mlp_out"]], f[:, F["A_mlp_err"]], f[:, F["A_mlp_bdec"]] = vin, vout, err, bd
            mlpA = vin + vout + err + bd
            m_in = vin
            mx_all = torch.maximum(mx_all, mxm)
        totA = skip + attnA + mlpA + f[:, F["beff"]]
        f[:, F["A_maxpar"]] = mx_all
        f[:, F["A_maxpar_ge25"]] = (mx_all >= 0.25 * act).float()
        f[:, F["A_maxpar_ge50"]] = (mx_all >= 0.5 * act).float()
        f[:, F["A_in_ge50"]] = ((skip_in + a_in + m_in) >= 0.5 * act).float()
        f[:, F["chkA"]] = (act - totA).abs() / act.clamp_min(1e-6)
        # view B (resid only): resid_mid_L + mlp_L
        if kd == "resid":
            vin, vout, err, bd, mxr = _site_parts(P, cd, (L, "resid_mid"), b, t, Wt, sT)
            f[:, F["B_rmid_in"]], f[:, F["B_rmid_out"]], f[:, F["B_rmid_err"]], f[:, F["B_rmid_bdec"]] = vin, vout, err, bd
            mxB = torch.maximum(mxr, mxm)
            f[:, F["B_maxpar"]] = mxB
            f[:, F["B_maxpar_ge50"]] = (mxB >= 0.5 * act).float()
            f[:, F["B_in_ge50"]] = ((vin + m_in) >= 0.5 * act).float()
            totB = vin + vout + err + bd + mlpA + f[:, F["beff"]]
            f[:, F["chkB"]] = (act - totB).abs() / act.clamp_min(1e-6)
        # view C: attn_L by source position / role
        vc = _viewC(P, cd, nat, L, tgt, rid, vv)
        totC = skip + mlpA + f[:, F["beff"]]
        c_in = skip_in + m_in
        for cat, z in vc.items():
            g = z[b, t, mi] / sT[:, None]                                                         # [n, R]
            for r in range(NR):
                f[:, F["C_attn_%s_%s" % (cat, ROLES[r])]] = g[:, r]
            totC = totC + g.sum(1)
            if cat == "in":
                c_in = c_in + g.sum(1)
        cst = (Wt @ T_["const"]) / sT
        f[:, F["C_attn_const"]] = cst
        totC = totC + cst
        f[:, F["C_in_ge50"]] = (c_in >= 0.5 * act).float()
        f[:, F["chkC"]] = (act - totC).abs() / act.clamp_min(1e-6)
        acc_m[tgt].index_add_(0, mi, f.double())
        acc_r[tgt].index_add_(0, rid[b, t], f.double())
        for v in ("chkA", "chkB", "chkC"):
            chk[v] = max(chk[v], float(f[:, F[v]].max()))


def retention_batch(eng, alphas, scope, tk, an, nat_cd, vv, ret):
    """Z frame (the scorer's CircuitOnlyPatcher: members at alpha, non-members 0, errors live, the circuit's scope):
    each member's re-encoded code at its natural occurrences."""
    from eval.ablation_faithfulness import CircuitOnlyPatcher
    bank = eng.bank
    dev = eng.dev
    D = int(bank.d_sae)
    keep = {s: set(v) for s, v in alphas.items()}
    scales = {}
    for s, v in alphas.items():
        sv = torch.ones(D, device=dev)
        ii = torch.tensor(sorted(v), device=dev)
        sv[ii] = torch.tensor([v[int(i)] for i in ii.tolist()], device=dev)
        scales[s] = sv
    seen = {}

    class Cap(CircuitOnlyPatcher):
        def transform(self, l, kd, x):
            DBG.setdefault("calls", set()).add((l, kd))
            if (l, kd) in keep:
                with torch.no_grad():
                    ta, ti = bank.encode(x, kd, l)
                seen[(l, kd)] = (ta.float(), ti.long())
            return super().transform(l, kd, x)

    p = Cap(bank=bank, keep_indices=keep, in_scope=scope, seed_layer=-1, seed_kind="", seed_latent_idx=0,
            pos_argmax=an, site_means=None, respect_topk=False, topk=int(bank.k), keep_scales=scales)
    with torch.no_grad():
        o = eng.inf.forward(tk, patcher=p, grad_enabled=False, all_logits=True, return_activations=False,
                            tokenize_final=False)
    lg = o[1] if isinstance(o, (tuple, list)) else o
    for s, v in alphas.items():
        c = nat_cd[s]
        ii = torch.tensor(sorted(v), device=dev)
        loc = torch.full((D,), -1, dtype=torch.long, device=dev)
        loc[ii] = torch.arange(len(ii), device=dev)
        lo = loc[c["i"]]
        occ = (lo >= 0) & (c["a"] > 0) & vv[..., None]
        b, t, k = occ.nonzero(as_tuple=True)
        if b.numel() == 0:
            continue
        mi = lo[b, t, k]
        nat_v = c["a"][b, t, k]
        ta, ti = seen[s]
        dense = torch.zeros(ta.shape[:-1] + (D,), device=dev).scatter_(-1, ti, ta)
        cir_v = dense[b, t, ii[mi]]
        r = ret[s]
        r[:, 0].index_add_(0, mi, nat_v.double())
        r[:, 1].index_add_(0, mi, cir_v.double())
        r[:, 2].index_add_(0, mi, (cir_v > 0).double())
        r[:, 3].index_add_(0, mi, torch.ones_like(nat_v).double())
        r[:, 4].index_add_(0, mi, (cir_v * scales[s][ii[mi]]).double())                  # alpha x circuit code
        del dense
    return lg


def stage_closure(eng, sets):
    nat_run = Nat(eng)
    dev = eng.dev
    D = int(eng.bank.d_sae)
    for cname in CIRCUITS:
        alphas, scope_name = load_circuit(cname)
        scope = set(eng.sites) if scope_name == "all" else {s for s in eng.sites if s[1] in ("attn", "mlp")}
        alphas = {s: v for s, v in alphas.items() if s in scope}
        P = Prep(eng, alphas)
        for sname, B, which, n in (("train", sets["dis"], "train", N_DEC_TR), ("dis", sets["dis"], "held", N_DEC_HO),
                                   ("sn", sets["sn"], "held", N_DEC_HO)):
            p = OUT / ("closure_%s_%s.pt" % (cname, sname))
            if p.exists():
                log("closure %s %s: exists, skipped" % (cname, sname))
                continue
            t0 = time.time()
            tok, anc, val, good, bad, rid, ct = rows_of(B, which, n)
            N = int(tok.shape[0])
            acc_m = {s: torch.zeros(len(T_["idx"]), len(FEATS), dtype=torch.float64, device=dev) for s, T_ in P.tgt.items()}
            acc_r = {s: torch.zeros(NR, len(FEATS), dtype=torch.float64, device=dev) for s in P.tgt}
            ret = {s: torch.zeros(len(v), 5, dtype=torch.float64, device=dev) for s, v in alphas.items()}
            chk = dict(chkA=0.0, chkB=0.0, chkC=0.0)
            m_nat, m_cir = [], []
            for s0 in range(0, N, BS):
                sl = slice(s0, min(N, s0 + BS))
                tk, an, vv, rr = tok[sl], anc[sl], val[sl], rid[sl]
                nat = nat_run.run(tk)
                cd = codes(eng, nat)
                closure_batch(P, eng, nat, cd, rr, vv, acc_m, acc_r, chk)
                lg = retention_batch(eng, alphas, scope, tk, an, cd, vv, ret)
                ar = torch.arange(int(tk.shape[0]), device=dev)
                if s0 == 0:
                    log("  first batch: patcher saw %d sites (%d with members); max |logit circuit - natural| %.3f"
                        % (len(DBG.get("calls", ())), len(alphas), float((lg.float() - nat["logits"]).abs().max())))
                    DBG.clear()
                for lgx, dst in ((nat["logits"], m_nat), (lg.float(), m_cir)):
                    z = lgx[ar, an]
                    dst.append((z[ar, good[sl]] - z[ar, bad[sl]]).cpu())
                del nat, cd
            mn, mc = torch.cat(m_nat), torch.cat(m_cir)
            torch.save(dict(circuit=cname, set=sname, scope=scope_name, n_items=N, feats=FEATS, roles=ROLES,
                            members={s: T_["idx"].cpu() for s, T_ in P.tgt.items()},
                            alphas_t={s: T_["alphas"].cpu() for s, T_ in P.tgt.items()},
                            acc_m={s: v.cpu() for s, v in acc_m.items()}, acc_r={s: v.cpu() for s, v in acc_r.items()},
                            ret={s: v.cpu() for s, v in ret.items()},
                            ret_members={s: torch.tensor(sorted(v)) for s, v in alphas.items()},
                            chk=chk, m_full=float(mn.mean()), m_circuit_Z=float(mc.mean())), p)
            log("closure %s %s: %d items, D full %.3f, D circuit(Z) %.3f, max rel residual A %.2e B %.2e C %.2e, %.0fs"
                % (cname, sname, N, float(mn.mean()), float(mc.mean()), chk["chkA"], chk["chkB"], chk["chkC"],
                   time.time() - t0))
        del P
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eng, sets = load()
    log("engine + batches %.0fs; device %s; stages %s; circuits %s" % (time.time() - t0, eng.dev, STAGES, CIRCUITS))
    eng.inf.disable_compile()
    if "attr" in STAGES:
        stage_attr(eng, sets)
    if "closure" in STAGES:
        with torch.no_grad():
            stage_closure(eng, sets)
    if torch.cuda.is_available():
        log("peak GPU %.1f GB" % (torch.cuda.max_memory_allocated() / 2 ** 30))
    log("DONE %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
