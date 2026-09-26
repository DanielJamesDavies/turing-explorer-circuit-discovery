"""WHY DO EMPTY CIRCUITS EXPLODE WHEN RESIDUAL SITES ARE EDITED?

Every site of an empty (zero-fill) circuit, run through ALL 18 layers on corpus
text (real positions only, BOS excluded; clean run in parallel for reference).
At each site, on the stream arriving there:

  norm ratio   ||x|| / ||x_clean||            growth vs the clean run
  L0           active latents of c(x)          (clean L0 in brackets)
  gain         ||c(x) @ W_dec|| / ||x - b_dec||  reconstruction size vs input;
               a zero-fill edit carries forward the CURRENT error
               x - c(x) W_dec, so gain > 2 flips and amplifies the stream

Modes:
  amr/current   att+mlp+res zero-filled, error recomputed from the edited stream
                (what fit_latent_seed.py does — the TuringLLM engine semantics)
  res/current   residual sites only
  am/current    attention + MLP sites only
  amr/clean     att+mlp+res zero-filled, each site's SAE error HELD at its clean
                value: x' = c_hat @ W_dec + b_dec + error_clean (SFC-style)

  python experiments/052-gemma3-270m/res_explosion_diag.py
Env: N_SEQ (8) | SEQ (128) | EXAMPLES_DIR
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


# clean run: per-site stream and SAE error
clean, err = {}, {}


def cap(site):
    def fn(x):
        clean[site] = x.float()
        err[site] = (x - (enc(site, x) @ SAE[site]["W_dec"] + SAE[site]["b_dec"])).detach()
        return None if False else x
    return fn


hs = [hook(s, cap(s)) for s in SITES]
model.model(tokens)
for h in hs:
    h.remove()


def stats(site, x):
    x1 = x[:, 1:].float()
    c = enc(site, x[:, 1:])
    rec = (c @ SAE[site]["W_dec"]).float()
    inp = x1 - SAE[site]["b_dec"].float()
    ratio = float(x1.norm(dim=-1).mean() / clean[site][:, 1:].norm(dim=-1).mean())
    l0 = float((c > 0).sum(-1).float().mean())
    l0c = float((enc(site, clean[site][:, 1:].to(DTYPE)) > 0).sum(-1).float().mean())
    gain = float((rec.norm(dim=-1) / inp.norm(dim=-1).clamp_min(1e-6)).mean())
    return ratio, l0, l0c, gain


def run_mode(kinds, error_mode):
    rows = {}

    def mk(site):
        def fn(x):
            rows[site] = stats(site, x)
            if site[0] not in kinds:
                return x
            y = x.clone()
            if error_mode == "current":
                c = enc(site, x[:, 1:])
                y[:, 1:] = x[:, 1:] - (c @ SAE[site]["W_dec"])            # zero-fill: c_hat = 0
            else:
                y[:, 1:] = SAE[site]["b_dec"] + err[site][:, 1:]         # c_hat = 0, clean error held
            return y
        return fn
    hs_ = [hook(s, mk(s)) for s in SITES]
    try:
        model.model(tokens)
    finally:
        for h in hs_:
            h.remove()
    return rows


modes = [("amr/current", ("att", "mlp", "res"), "current"), ("res/current", ("res",), "current"),
         ("am/current", ("att", "mlp"), "current"), ("amr/clean", ("att", "mlp", "res"), "clean")]
results = {name: run_mode(kinds, em) for name, kinds, em in modes}
print("empty zero-fill circuit through all %d layers | %d x %d corpus tokens\n" % (NL, N_SEQ, SEQ))
for kind in ("res", "att", "mlp"):
    print("--- %s sites: norm ratio vs clean | L0 (clean L0) | gain" % kind)
    print("%3s | " % "L" + " | ".join("%-30s" % n for n, _, _ in modes))
    for l in range(NL):
        cells = []
        for n, _, _ in modes:
            r, l0, l0c, g = results[n][(kind, l)]
            cells.append("%9.3g  L0 %5.0f (%3.0f) g %5.2f" % (r, l0, l0c, g))
        print("%3d | " % l + " | ".join("%-30s" % c for c in cells))
    print()
