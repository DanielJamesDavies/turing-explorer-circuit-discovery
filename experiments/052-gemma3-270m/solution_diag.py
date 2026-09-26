"""DO THE CANDIDATE STABILISERS FIX BOTH FAILURE MODES? (before any fitting)

Two counterfactuals run through all 18 layers, every site att/mlp/res edited:
  EMPTY     every latent removed (zero fill)              -> the removal blow-up
  HALF      a random 50% of latents kept at alpha=1, the rest removed
            -> the partly-kept states the open-start fit gets trapped in
Residual-stream norm vs clean is reported per layer (1.0 = unchanged; the
current form reached inf by L13 on EMPTY).

Edit forms (site = c_hat W_dec + b_dec + error term):
  current   x + (c_hat - c(x)) W_dec                 (TuringLLM / 033)
  clean     + (err_clean - err(x))                   (fix 11)
  cap       current, but c(x) keeps at most the token's NATURAL latent count
            (top-k by value, k = clean-run L0 at that token and site) —
            solution A; exact at identity, bounded off-distribution
  norm      current, but encode/decode at the token's CLEAN norm:
            x_s = x * |x_clean|/|x|, edit in that frame, scale back —
            solution B; exact at identity, removes absolute-scale drift
  cap+norm  both
  cap+clean cap the re-encoded code AND hold the error clean:
            c_hat W_dec + b_dec + err_clean with c = cap(c(x)) — removal
            cannot feed the error, keeping cannot feed thousands of latents
  +clamp    no re-encoded latent may exceed the token's largest NATURAL
            latent value at that site (exact at identity) — bounds magnitude,
            which the count cap does not

  python experiments/052-gemma3-270m/solution_diag.py
Env: N_SEQ (8) | SEQ (128) | EXAMPLES_DIR | KEEP (0.5)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gemmascope2 as GS  # noqa: E402

MODEL = os.environ.get("MODEL", "unsloth/gemma-3-270m")
N_SEQ = int(os.environ.get("N_SEQ", 8)); SEQ = int(os.environ.get("SEQ", 128))
KEEP = float(os.environ.get("KEEP", 0.5))
EX = Path(os.environ.get("EXAMPLES_DIR", str(Path.home() / "gemmascope2_examples")))
DEV = torch.device("cuda"); DTYPE = torch.bfloat16
torch.set_grad_enabled(False)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV).eval()
LAYERS = model.model.layers
NL = len(LAYERS)
corpus = np.load(next(EX.glob("corpus_*.npy")), mmap_mode="r")
tokens = torch.tensor(np.asarray(corpus[200000:200000 + N_SEQ, :SEQ]), dtype=torch.long, device=DEV)
SITES = [(k, l) for l in range(NL) for k in ("att", "mlp", "res")]
SAE = {s: GS.load_sae(s[0], s[1], device=DEV, dtype=DTYPE) for s in SITES}
gen = torch.Generator(device="cpu").manual_seed(0)
KEEPMASK = {s: (torch.rand(SAE[s]["W_enc"].shape[1], generator=gen) < KEEP).to(DEV) for s in SITES}


def module_for(site):
    k, l = site
    return {"att": LAYERS[l].self_attn.o_proj, "mlp": LAYERS[l].post_feedforward_layernorm, "res": LAYERS[l]}[k]


def hook(site, fn):
    mod = module_for(site)
    if site[0] == "att":
        return mod.register_forward_pre_hook(lambda m, a: (fn(a[0]),) + tuple(a[1:]))

    def post(m, a, o):
        x = o[0] if isinstance(o, tuple) else o
        y = fn(x)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    return mod.register_forward_hook(post)


def enc(site, x):
    t = SAE[site]
    pre = x @ t["W_enc"] + t["b_enc"]
    return pre * (pre > t["threshold"])


def err_of(site, x, c):
    t = SAE[site]
    return x - (c @ t["W_dec"] + t["b_dec"])


def cap_code(c, k_tok):
    """keep at most k_tok[b, t] largest entries per token (k varies by token)."""
    kmax = int(k_tok.max())
    if kmax <= 0:
        return torch.zeros_like(c)
    tv, ti = c.topk(kmax, dim=-1)
    keep = torch.arange(kmax, device=c.device)[None, None, :] < k_tok[..., None]
    return torch.zeros_like(c).scatter(-1, ti, tv * keep)


# clean run: per-site stream, natural L0 per token, clean error
clean, nat_l0, clean_err, nat_max = {}, {}, {}, {}


def cap_clean(site):
    def fn(x):
        c = enc(site, x)
        clean[site] = x.detach()
        nat_l0[site] = (c > 0).sum(-1)
        nat_max[site] = c.max(-1).values
        clean_err[site] = err_of(site, x, c).detach()
        return x
    return fn


hs = [hook(s, cap_clean(s)) for s in SITES]
model.model(tokens)
for h in hs:
    h.remove()


def run(mode, keep_frac):
    rows = {}
    use_cap = "cap" in mode
    use_norm = "norm" in mode

    def mk(site):
        def fn(x):
            if site[0] == "res":
                rows[l_of(site)] = float(x[:, 1:].float().norm(dim=-1).mean() / clean[site][:, 1:].float().norm(dim=-1).mean())
            t = SAE[site]
            xs, s = x, None
            if use_norm:
                s = (clean[site].float().norm(dim=-1, keepdim=True) /
                     x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(x.dtype)
                xs = x * s
            c = enc(site, xs)
            if use_cap:
                c = cap_code(c, nat_l0[site])
            if "clamp" in mode:
                c = torch.minimum(c, nat_max[site][..., None])
            chat = c * KEEPMASK[site].to(c.dtype) if keep_frac > 0 else torch.zeros_like(c)
            delta = (chat - c) @ t["W_dec"]
            if "clean" in mode:
                delta = delta + (clean_err[site] - err_of(site, xs, c))
            y = xs + delta
            if use_norm:
                y = y / s
            y = y.clone()
            y[:, 0] = x[:, 0]
            return y
        return fn
    hs_ = [hook(s, mk(s)) for s in SITES]
    try:
        model.model(tokens)
    finally:
        for h in hs_:
            h.remove()
    return rows


def l_of(site):
    return site[1]


modes = [m for m in os.environ.get("MODES", "current,clean,cap,norm,cap+norm,cap+clean").split(",") if m]

# identity check: every mode with "keep everything" must reproduce the clean stream
for mode in [m for m in modes if m != "current"]:
    KM = KEEPMASK
    KEEPMASK = {s: torch.ones_like(v) for s, v in KM.items()}
    r = run(mode, 1.0)
    KEEPMASK = KM
    print("identity check %-15s max |norm ratio - 1| over layers = %.2e" % (mode, max(abs(v - 1) for v in r.values())))

KEEPS = [float(k) for k in os.environ.get("KEEPS", "0,%g" % KEEP).split(",")]
for kf in KEEPS:
    title = "EMPTY (all latents removed)" if kf == 0 else "PARTIAL (random %.0f%% kept at alpha=1)" % (100 * kf)
    if kf > 0:
        g = torch.Generator(device="cpu").manual_seed(0)
        KEEPMASK = {s: (torch.rand(SAE[s]["W_enc"].shape[1], generator=g) < kf).to(DEV) for s in SITES}
    res = {m: run(m, kf) for m in modes}
    print("\n%s — residual norm vs clean, %d x %d tokens" % (title, N_SEQ, SEQ))
    print("%3s | " % "L" + " | ".join("%10s" % m for m in modes))
    for l in range(NL):
        print("%3d | " % l + " | ".join("%10.3g" % res[m][l] for m in modes))
