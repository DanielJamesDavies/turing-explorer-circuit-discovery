"""Does the activation circuit-tracer feeds the transcoder match the one
our HF harness feeds it?

Our probe (llama_loader.py) established that these transcoders consume
the HF MLP module's input, i.e. post_attention_layernorm(x) = (x/scale)*w.
TransformerLens places `ln2.hook_normalized` BEFORE the RMSNorm weight,
so circuit-tracer would feed x/scale instead — a factor-of-w error, the
same class of bug as the GPT-2 LayerNorm convention.

This script measures three relative errors per layer:
    raw hook   : TL hook              vs HF MLP input   (expect mismatch)
    hook * w   : TL hook * rms weight vs HF MLP input   (expect match)
    mlp_out    : TL output hook       vs HF MLP output  (sanity)

If `hook * w` matches, the fix is to fold the RMS weight into the
encoder for the circuit-tracer side only: W_enc <- W_enc * w.

  ../../dev-notes/data/venv-ct/bin/python check_hooks.py
"""
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "unsloth/Llama-3.2-1B"
TL_NAME = "meta-llama/Llama-3.2-1B"
LAYERS = [0, 4, 8, 15]
TEXT = "The Eiffel Tower is in Paris, the capital of France."

hf = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(MID)
tl = HookedTransformer.from_pretrained(
    TL_NAME, hf_model=hf, tokenizer=tok, fold_ln=False,
    center_writing_weights=False, center_unembed=False, device="cpu")

ids = tok(TEXT, return_tensors="pt")["input_ids"]

cap = {}
handles = []
for L in LAYERS:
    def mk(layer):
        def hook(mod, inp, out):
            cap[("in", layer)] = inp[0].detach()
            cap[("out", layer)] = out.detach()
        return hook
    handles.append(hf.model.layers[L].mlp.register_forward_hook(mk(L)))
with torch.no_grad():
    hf(ids)
for h in handles:
    h.remove()

names = set()
for L in LAYERS:
    names.add("blocks.%d.ln2.hook_normalized" % L)
    names.add("blocks.%d.hook_mlp_out" % L)
_, tlc = tl.run_with_cache(ids, names_filter=lambda n: n in names)

print("layer | raw hook rel | hook*w rel | mlp_out rel")
ok = True
for L in LAYERS:
    ti = tlc["blocks.%d.ln2.hook_normalized" % L]
    to = tlc["blocks.%d.hook_mlp_out" % L]
    hi = cap[("in", L)]
    ho = cap[("out", L)]
    w = hf.model.layers[L].post_attention_layernorm.weight.data
    denom_i = max(float(hi.abs().max()), 1e-9)
    denom_o = max(float(ho.abs().max()), 1e-9)
    r_raw = float((ti - hi).abs().max()) / denom_i
    r_scaled = float((ti * w - hi).abs().max()) / denom_i
    r_out = float((to - ho).abs().max()) / denom_o
    ok = ok and (r_scaled < 1e-3) and (r_out < 1e-3)
    print("  %2d  |   %.2e   |  %.2e  |  %.2e" % (L, r_raw, r_scaled, r_out))

print("")
print("DIAGNOSIS: TransformerLens hooks the RMSNorm output BEFORE its")
print("weight; the HF MLP input includes it. Fix for the circuit-tracer")
print("side only: W_enc <- W_enc * w (per layer).")
print("VERDICT: " + ("hook*w MATCHES the HF input - fold confirmed" if ok
                     else "still mismatched - investigate before running"))
