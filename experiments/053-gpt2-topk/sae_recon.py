"""STEP 1 — do these SAEs reconstruct, and does our hook/normalisation
transcription match theirs? (the 052 step-1 audit, on GPT-2 small)

Per site kind x layer, on wikitext windows of the SAEs' own training length:
  FVU        1 - explained variance of the reconstruction (lower is better)
  L0         non-zeros per token (must be exactly k = 32 for Top-K)
  dCE        cross-entropy increase when the site is REPLACED by its
             reconstruction (the number that matters for circuit work)
  ident      max |x - splice(x)| when the reconstruction is replaced by x
             itself through the same hook path: a wiring check, must be 0

  python experiments/053-gpt2-topk/sae_recon.py
Env: GPT2_SAE_WIDTH (32k) | N_SEQ (16) | SEQ (64) | LAYERS (0,3,6,9,11)
"""
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gpt2saes as G  # noqa: E402

N_SEQ = int(os.environ.get("N_SEQ", 16)); SEQ = int(os.environ.get("SEQ", 64))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,3,6,9,11").split(",")]
DEV = torch.device("cuda")
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()

# wikitext windows, tokenised without BOS (the SAEs set prepend_bos False)
from huggingface_hub import hf_hub_download  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
pqf = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                      filename="wikitext-103-raw-v1/train-00000-of-00002.parquet")
text = "".join(t for t in pq.read_table(pqf).column("text").to_pylist()[:6000] if len(t) > 200)
ids = tok(text, return_tensors="pt").input_ids[0]
tokens = torch.stack([ids[i * SEQ:(i + 1) * SEQ] for i in range(N_SEQ)]).to(DEV)

base = model(tokens, labels=tokens).loss.item()
print("GPT-2 small | %d x %d wikitext tokens | width %s | clean CE %.4f\n"
      % (N_SEQ, SEQ, G.WIDTH, base))


def splice(kind, layer, fn):
    """register fn on the site, return (CE, captured stats)."""
    mod = G.module_for(model, kind, layer)
    if kind == "resid-mid":
        h = mod.register_forward_pre_hook(lambda m, a: (fn(a[0]),))
    elif kind == "mlp-out":
        h = mod.register_forward_hook(lambda m, a, o: fn(o))
    else:
        def post(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            y = fn(x)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        h = mod.register_forward_hook(post)
    try:
        return model(tokens, labels=tokens).loss.item()
    finally:
        h.remove()


rows = []
print("%-10s %-3s %8s %6s %8s %9s" % ("kind", "L", "FVU", "L0", "dCE", "ident"))
for layer in LAYERS:
    for kind in G.KINDS:
        t = G.load_sae(kind, layer, device=DEV)
        st = {}

        def recon(x, _t=t, _s=st):
            xn, std, mu = G.norm_in(x)
            c = G.encode(_t, xn)
            xh = (G.decode(_t, c)) * std + mu
            _s["fvu"] = float(((x - xh) ** 2).sum() / ((x - x.mean((0, 1))) ** 2).sum())
            _s["l0"] = float((c > 0).sum(-1).float().mean())
            return xh

        def ident(x, _t=t, _s=st):
            # same code path, but the delta form our fitter uses with c_hat = c:
            # x + ((c - c) @ W_dec) * std must be EXACTLY x
            xn, std, mu = G.norm_in(x)
            c = G.encode(_t, xn)
            d = ((c - c) @ _t["W_dec"]) * std
            _s["ident"] = float(d.abs().max())
            return x + d
        ce = splice(kind, layer, recon)
        splice(kind, layer, ident)
        r = dict(kind=kind, layer=layer, fvu=st["fvu"], l0=st["l0"], dce=ce - base, ident=st["ident"])
        rows.append(r)
        print("%-10s %-3d %8.4f %6.1f %+8.4f %9.1e" % (kind, layer, r["fvu"], r["l0"], r["dce"], r["ident"]))
        del t
        torch.cuda.empty_cache()

json.dump(rows, open(HERE / ("sae_recon_%s.json" % G.WIDTH), "w"), indent=1)
print("\nwrote sae_recon_%s.json" % G.WIDTH)
