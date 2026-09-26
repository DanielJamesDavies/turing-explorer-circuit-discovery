"""HOW SPARSE IS GEMMA SCOPE 2 IN PRACTICE, AND IS ANY LATENT ALWAYS ON?

TopK fires exactly k latents per token, and WHICH k varies. JumpReLU fires
however many cross threshold, so both the count and the identity vary. On clean
corpus text (BOS excluded), per site:

  L0 per token   mean / p5 / p50 / p95 / max, and as a % of the 16,384 dictionary
  always-on      latents firing on >= 90% / >= 50% of tokens (TopK has none by
                 construction unless a latent is genuinely in the top k always)
  coverage       distinct latents that ever fire in the sample

  python experiments/052-gemma3-270m/sparsity_stats.py
Env: N_SEQ (16) | SEQ (256) | EXAMPLES_DIR
"""
import json
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
N_SEQ = int(os.environ.get("N_SEQ", 16)); SEQ = int(os.environ.get("SEQ", 256))
EX = Path(os.environ.get("EXAMPLES_DIR", str(Path.home() / "gemmascope2_examples")))
DEV = torch.device("cuda")
torch.set_grad_enabled(False)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
LAYERS = model.model.layers
NL = len(LAYERS)
corpus = np.load(next(EX.glob("corpus_*.npy")), mmap_mode="r")
tokens = torch.tensor(np.asarray(corpus[300000:300000 + N_SEQ, :SEQ]), dtype=torch.long, device=DEV)
SITES = [(k, l) for l in range(NL) for k in ("att", "mlp", "res")]


def module_for(site):
    k, l = site
    return {"att": LAYERS[l].self_attn.o_proj, "mlp": LAYERS[l].post_feedforward_layernorm, "res": LAYERS[l]}[k]


rows = {}
for site in SITES:
    t = GS.load_sae(site[0], site[1], device=DEV, dtype=torch.bfloat16)
    cap = {}

    def fn(x, _t=t):
        pre = x[:, 1:] @ _t["W_enc"] + _t["b_enc"]
        cap["nz"] = (pre > _t["threshold"])
        return None
    mod = module_for(site)
    if site[0] == "att":
        h = mod.register_forward_pre_hook(lambda m, a: fn(a[0]))
    else:
        def post(m, a, o):
            fn(o[0] if isinstance(o, tuple) else o)
            return None
        h = mod.register_forward_hook(post)
    model.model(tokens)
    h.remove()
    nz = cap["nz"].reshape(-1, cap["nz"].shape[-1])          # [tokens, W]
    per_tok = nz.sum(-1).float().cpu().numpy()
    per_lat = nz.float().mean(0).cpu().numpy()               # firing rate per latent
    W = nz.shape[-1]
    rows[site] = dict(mean=float(per_tok.mean()), p5=float(np.percentile(per_tok, 5)),
                      p50=float(np.percentile(per_tok, 50)), p95=float(np.percentile(per_tok, 95)),
                      mx=float(per_tok.max()), pct=100 * float(per_tok.mean()) / W,
                      always90=int((per_lat >= 0.9).sum()), always50=int((per_lat >= 0.5).sum()),
                      ever=int((per_lat > 0).sum()), W=W)
    del t
    torch.cuda.empty_cache()

print("clean corpus text, %d x %d tokens (BOS excluded), dictionary 16,384\n" % (N_SEQ, SEQ))
print("%-4s %3s | %7s %5s %5s %5s %6s | %7s | %8s %8s | %7s"
      % ("kind", "L", "mean L0", "p5", "p50", "p95", "max", "% dict", ">=90% on", ">=50% on", "ever on"))
for kind in ("att", "mlp", "res"):
    for l in range(NL):
        r = rows[(kind, l)]
        print("%-4s %3d | %7.1f %5.0f %5.0f %5.0f %6.0f | %6.2f%% | %8d %8d | %7d"
              % (kind, l, r["mean"], r["p5"], r["p50"], r["p95"], r["mx"], r["pct"], r["always90"], r["always50"], r["ever"]))
print("\nby kind (median over layers):")
for kind in ("att", "mlp", "res"):
    rs = [rows[(kind, l)] for l in range(NL)]
    f = lambda k: np.median([r[k] for r in rs])
    print("  %-3s mean L0 %5.1f (%.2f%% of dict) | p5 %4.0f p95 %4.0f max %5.0f | always-on >=90%% %3.0f >=50%% %4.0f | ever on %6.0f"
          % (kind, f("mean"), f("pct"), f("p5"), f("p95"), f("mx"), f("always90"), f("always50"), f("ever")))
json.dump({"%s_%d" % s: r for s, r in rows.items()}, open(HERE / "sparsity_stats.json", "w"), indent=1)
