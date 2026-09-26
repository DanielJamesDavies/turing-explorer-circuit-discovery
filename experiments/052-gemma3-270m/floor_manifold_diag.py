"""IS THE ZERO FLOOR OFF-MANIFOLD FOR A NO-PRE-BIAS SAE, AND DOES THE MEAN FLOOR FIX IT?

Our TopK SAEs encode topk(relu(W_enc (x - b_dec) + b_enc)) — subtracting b_dec
first — so deleting every latent leaves the site at ~= data mean + residue (a
mean ablation). Gemma Scope 2 has NO pre-bias (pre = x W_enc + b_enc), so
deleting every latent leaves b_dec + residue, which need not be near the mean.

  (1) per residual site: ||b_dec|| vs ||mean activation||, cos(b_dec, mean),
      residue share ||x - x_hat|| / ||x||, and how far the ZERO-filled state
      (b_dec + err) and the MEAN-filled state (b_dec + mu_code W_dec + err)
      sit from the true mean activation
  (2) single site, no chaining: latents firing and reconstruction gain when the
      SAE re-encodes its own clean / zero-filled / mean-filled state
  (3) chained through all 18 layers, empty circuit: residual-stream norm vs
      clean under zero fill everywhere, MEAN fill everywhere, and the hybrid
      (zero at att/mlp, mean at res)

  python experiments/052-gemma3-270m/floor_manifold_diag.py
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
    return hook_wrap(mod, post)


def hook_wrap(mod, post):
    return mod.register_forward_hook(post)


def enc(site, x):
    t = SAE[site]
    pre = x @ t["W_enc"] + t["b_enc"]
    return pre * (pre > t["threshold"])


clean, err, mu_code, mu_act = {}, {}, {}, {}


def cap(site):
    def fn(x):
        x1 = x[:, 1:]
        c = enc(site, x1)
        clean[site] = x1.float()
        err[site] = (x1 - (c @ SAE[site]["W_dec"] + SAE[site]["b_dec"])).detach()
        mu_code[site] = c.float().mean(dim=(0, 1)).to(DTYPE)
        mu_act[site] = x1.float().mean(dim=(0, 1))
        return x
    return fn


hs = [hook(s, cap(s)) for s in SITES]
model.model(tokens)
for h in hs:
    h.remove()

print("(1) residual sites: is the ZERO-filled state near the data mean?  (%d x %d tokens)\n" % (N_SEQ, SEQ))
print("%3s | %9s %9s %8s | %8s | %11s %11s" %
      ("L", "||b_dec||", "||mean x||", "cos", "residue", "|zero-mean|", "|mean-mean|"))
for l in range(NL):
    s = ("res", l)
    bd = SAE[s]["b_dec"].float()
    ma = mu_act[s]
    resid_share = float((err[s].float().norm(dim=-1) / clean[s].norm(dim=-1)).mean())
    zero_state = bd + err[s].float().mean(dim=(0, 1))
    mean_state = zero_state + (mu_code[s].float() @ SAE[s]["W_dec"].float())
    print("%3d | %9.1f %9.1f %8.3f | %7.1f%% | %10.2f%% %10.2f%%"
          % (l, float(bd.norm()), float(ma.norm()),
             float(torch.nn.functional.cosine_similarity(bd[None], ma[None])),
             100 * resid_share,
             100 * float((zero_state - ma).norm() / ma.norm()),
             100 * float((mean_state - ma).norm() / ma.norm())))

print("\n(2) single site, no chaining: the SAE re-encodes its own clean / zero-filled / mean-filled state")
print("%3s | %26s | %26s | %26s" % ("L", "clean", "zero fill (b_dec+err)", "mean fill (+mu W_dec)"))
for l in range(NL):
    s = ("res", l)
    t = SAE[s]
    x = clean[s].to(DTYPE)
    states = {"clean": x,
              "zero": (t["b_dec"] + err[s]),
              "mean": (t["b_dec"] + err[s] + (mu_code[s] @ t["W_dec"]))}
    cells = []
    for name, st in states.items():
        c = enc(s, st)
        l0 = float((c > 0).sum(-1).float().mean())
        gain = float(((c @ t["W_dec"]).float().norm(dim=-1) / st.float().norm(dim=-1).clamp_min(1e-6)).mean())
        cells.append("L0 %6.0f  gain %7.2f" % (l0, gain))
    print("%3d | %26s | %26s | %26s" % (l, cells[0], cells[1], cells[2]))


def run_mode(fill_for_kind):
    rows = {}

    def mk(site):
        def fn(x):
            if site[0] == "res":
                rows[site] = (float(x[:, 1:].float().norm(dim=-1).mean() / clean[site].norm(dim=-1).mean()),
                              float((enc(site, x[:, 1:]) > 0).sum(-1).float().mean()))
            fill = fill_for_kind[site[0]]
            if fill is None:
                return x
            c = enc(site, x[:, 1:])
            t = SAE[site]
            chat = torch.zeros_like(c) if fill == "zero" else mu_code[site].expand_as(c)
            y = x.clone()
            y[:, 1:] = x[:, 1:] + (chat - c) @ t["W_dec"]
            return y
        return fn
    hs_ = [hook(s, mk(s)) for s in SITES]
    try:
        model.model(tokens)
    finally:
        for h in hs_:
            h.remove()
    return rows


modes = [("zero everywhere", {"att": "zero", "mlp": "zero", "res": "zero"}),
         ("MEAN everywhere", {"att": "mean", "mlp": "mean", "res": "mean"}),
         ("zero att/mlp + MEAN res", {"att": "zero", "mlp": "zero", "res": "mean"})]
res = {n: run_mode(f) for n, f in modes}
print("\n(3) chained, empty circuit through all 18 layers: residual norm vs clean (and res-site L0)\n")
print("%3s | " % "L" + " | ".join("%-24s" % n for n, _ in modes))
for l in range(NL):
    print("%3d | " % l + " | ".join("%-24s" % ("%9.3g   L0 %6.0f" % res[n][("res", l)]) for n, _ in modes))
