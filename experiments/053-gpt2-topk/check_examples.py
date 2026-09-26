"""STEP 2 — can we trust Neuronpedia's contexts as an examples store?

For a handful of latents across all four site kinds: fetch the contexts, map the
display tokens back to GPT-2 ids, re-run the model, and compare OUR computed
latent activation at the stored peak position with Neuronpedia's stored value.
This is the 052 "stored vs recomputed" check (there: median rel err 0.007-0.012)
and it validates the loader, the layer-norm transcription and the token mapping
in one go.

  python experiments/053-gpt2-topk/check_examples.py
Env: GPT2_SAE_WIDTH (32k) | SEEDS ("resid-post:6:100,...") | N_CTX (16)
"""
import os
import statistics as st
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gpt2saes as G  # noqa: E402
import neuronpedia as NP  # noqa: E402

N_CTX = int(os.environ.get("N_CTX", 16))
SEEDS = [s for s in os.environ.get(
    "SEEDS", "resid-post:6:100,resid-post:9:2000,attn-out:6:500,mlp-out:6:1234,resid-mid:3:777").split(",") if s]
DEV = torch.device("cuda")
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()


def read_latent(kind, layer, idx, tokens):
    """our computed activation of one latent at every position."""
    t = G.load_sae(kind, layer, device=DEV)
    out = {}

    def fn(x):
        xn, _, _ = G.norm_in(x)
        out["c"] = G.encode(t, xn)[..., idx].float()
    # capture only: these hooks must return None, or a tuple-valued module
    # output (block, attn) gets replaced by a bare tensor
    mod = G.module_for(model, kind, layer)
    if kind == "resid-mid":
        h = mod.register_forward_pre_hook(lambda m, a: fn(a[0]))
    elif kind == "mlp-out":
        h = mod.register_forward_hook(lambda m, a, o: fn(o))
    else:
        h = mod.register_forward_hook(lambda m, a, o: fn(o[0] if isinstance(o, tuple) else o))
    try:
        model(tokens.to(DEV))
    finally:
        h.remove()
    del t
    torch.cuda.empty_cache()
    return out["c"]


print("GPT-2 small v5 %s | Neuronpedia examples check\n" % G.WIDTH)
print("%-22s %6s %8s %9s %9s %8s" % ("seed", "n_ctx", "frac_nz", "peak@stored", "recomputed", "rel err"))
for spec in SEEDS:
    kind, layer, idx = spec.split(":"); layer, idx = int(layer), int(idx)
    rows = NP.contexts(kind, layer, idx, N_CTX)
    if not rows:
        print("%-22s  no activating contexts" % spec)
        continue
    ids = [NP.to_ids(tok, r[0]) for r in rows]
    T = min(len(i) for i in ids)
    toks = torch.tensor([i[:T] for i in ids], dtype=torch.long)
    anchors = [min(r[1], T - 1) for r in rows]
    stored = [r[2] for r in rows]
    c = read_latent(kind, layer, idx, toks)
    mine = [float(c[j, anchors[j]]) for j in range(len(rows))]
    rel = [abs(m - s) / max(abs(s), 1e-6) for m, s in zip(mine, stored)]
    fq = NP.feature(kind, layer, idx).get("frac_nonzero")
    print("%-22s %6d %8s %9.3f %9.3f %8.3f"
          % (spec, len(rows), ("%.4f" % fq) if fq else "-", st.median(stored), st.median(mine), st.median(rel)))
