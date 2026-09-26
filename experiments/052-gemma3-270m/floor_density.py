"""HOW DENSE IS THE MEAN FLOOR, AND HOW MUCH OF IT DOES THE CAP TOUCH?

The mean frames fill every NON-member latent with its mean code over the
context set. A natural token has ~120 active latents out of 16,384; the mean
over many tokens is nonzero for every latent that ever fires. This measures:

  nat L0      natural latents per token at this site
  floor nz    latents with nonzero mean (the fill's L0)
  top120 %    share of the mean code's mass in its 120 largest entries
  |dec| /|x|  norm of the decoded mean floor vs the natural stream
  cos(b_dec)  cosine of the decoded floor with b_dec (the data mean)

  python experiments/052-gemma3-270m/floor_density.py
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
corpus = np.load(next(EX.glob("corpus_*.npy")), mmap_mode="r")
tokens = torch.tensor(np.asarray(corpus[200000:200000 + N_SEQ, :SEQ]), dtype=torch.long, device=DEV)
SITES = [(k, l) for l in (2, 5, 8, 11, 14) for k in ("att", "mlp", "res")]


def module_for(site):
    k, l = site
    return {"att": LAYERS[l].self_attn.o_proj, "mlp": LAYERS[l].post_feedforward_layernorm, "res": LAYERS[l]}[k]


stats = {}
hs = []
for s in SITES:
    t = GS.load_sae(s[0], s[1], device=DEV, dtype=DTYPE)

    def grab(x, _s=s, _t=t):
        x = x[:, 1:]                      # drop BOS
        pre = x @ _t["W_enc"] + _t["b_enc"]
        c = (pre * (pre > _t["threshold"])).float()
        mu = c.mean(dim=(0, 1))           # the mean floor for this site
        dec = mu.to(DTYPE) @ _t["W_dec"] + _t["b_dec"]
        top = mu.sort(descending=True).values[:120].sum() / mu.sum().clamp_min(1e-9)
        stats[_s] = (float((c > 0).sum(-1).float().mean()), int((mu > 0).sum()), float(top),
                     float(dec.float().norm() / x.float().norm(dim=-1).mean()),
                     float(torch.nn.functional.cosine_similarity(dec.float(), _t["b_dec"].float(), dim=0)))
        return x
    mod = module_for(s)
    if s[0] == "att":
        hs.append(mod.register_forward_pre_hook(lambda m, a, _f=grab: (_f(a[0]), *a[1:])[:1] and None or a))
    else:
        def post(m, a, o, _f=grab):
            _f(o[0] if isinstance(o, tuple) else o)
        hs.append(mod.register_forward_hook(post))
model.model(tokens)
for h in hs:
    h.remove()

print("dictionary 16,384 | %d x %d tokens, BOS excluded\n" % (N_SEQ, SEQ - 1))
print("site      nat L0   floor nz   top120 %   |dec|/|x|   cos(b_dec)")
for s in SITES:
    if s in stats:
        l0, nz, top, nr, cb = stats[s]
        print("%-3s L%-4d %6.0f   %8d   %7.1f%%   %9.3f   %+.3f" % (s[0], s[1], l0, nz, 100 * top, nr, cb))
