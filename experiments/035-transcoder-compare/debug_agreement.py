"""Why do the two sides disagree? Two very different possible causes:

  (A) MY BUG — a wrong fold/convention, in which case the comparison is
      broken until fixed.
  (B) THEIR MODEL — circuit-tracer's ReplacementModel SUBSTITUTES every
      MLP output with its transcoder reconstruction. The residual stream
      therefore diverges from the original model from layer 0's MLP
      onward, so features at layer L are computed on a different stream.
      That is inherent to attribution graphs (they analyse a surrogate),
      not a bug — but it changes what a fair comparison means.

Discriminating test, at layer 0 (whose MLP input is upstream of any MLP
substitution, so A and B make different predictions):

  1. TL hook * w  vs  HF mlp input          -> tests the RMS fold alone
  2. features from TL's own hook            vs their reported activations
  3. features from HF's mlp input           vs their reported activations

If (1) and (2) match but (3) does not, the cause is (B) at deeper layers
and something else at layer 0. If (2) matches everywhere, our maths is
right and the divergence is purely the surrogate stream.

  ../../dev-notes/data/venv-ct/bin/python debug_agreement.py
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HERE = Path(__file__).parent
TC_DIR = HERE / "transcoders_llama_ct"
MODEL_ID = "unsloth/Llama-3.2-1B"
TL_NAME = "meta-llama/Llama-3.2-1B"
K = 32
TEXT = "The Eiffel Tower is in Paris, the capital of France."


def code(pre):
    v, i = pre.topk(K, dim=-1)
    return torch.zeros_like(pre).scatter(-1, i, F.relu(v))


def main():
    import torch as t
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.transcoder.activation_functions import TopK
    from circuit_tracer.transcoder.single_layer_transcoder import (
        TranscoderSet, load_transcoder)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=t.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(TEXT, return_tensors="pt")["input_ids"]

    transcoders = {}
    for L in range(16):
        tc = load_transcoder(str(TC_DIR / ("layer_%d.safetensors" % L)),
                             layer=L, activation_fn=TopK(K),
                             device=t.device("cpu"), dtype=t.float32,
                             lazy_encoder=False, lazy_decoder=False)
        w = hf.model.layers[L].post_attention_layernorm.weight.data
        with t.no_grad():
            tc.W_enc.mul_(w.to(t.float32))
        transcoders[L] = tc
    tset = TranscoderSet(transcoders, feature_input_hook="ln2.hook_normalized",
                         feature_output_hook="hook_mlp_out",
                         scan_name="debug")
    model = ReplacementModel.from_pretrained_and_transcoders(
        TL_NAME, tset, device=t.device("cpu"), dtype=t.float32,
        hf_model=hf, tokenizer=tok)

    # their activations + the hook values their model actually saw
    names = {"blocks.%d.ln2.hook_normalized" % L for L in range(16)}
    _, cache = model.run_with_cache(ids, names_filter=lambda n: n in names)
    _, acts = model.get_activations(ids, sparse=False)

    # HF's own mlp inputs (unsubstituted model)
    hfin = {}
    hs = []
    for L in range(16):
        def mk(layer):
            def hook(mod, inp, out):
                hfin[layer] = inp[0].detach()
            return hook
        hs.append(hf.model.layers[L].mlp.register_forward_hook(mk(L)))
    with torch.no_grad():
        hf(ids)
    for h in hs:
        h.remove()

    raw = {L: {k: v.float() for k, v in load_file(
        str(TC_DIR / ("layer_%d.safetensors" % L))).items()} for L in [0, 4, 8]}

    print("layer | (1) TLhook*w vs HFin | (2) ours-from-TLhook vs theirs | "
          "(3) ours-from-HFin vs theirs")
    for L in [0, 4, 8]:
        w = hf.model.layers[L].post_attention_layernorm.weight.data
        tl_hook = cache["blocks.%d.ln2.hook_normalized" % L].squeeze(0)
        their = acts[L].squeeze(0) if acts[L].dim() == 3 else acts[L]
        sd = raw[L]

        x_from_tl = tl_hook * w
        f_tl = code(x_from_tl @ sd["W_enc"].T + sd["b_enc"])
        f_hf = code(hfin[L].squeeze(0) @ sd["W_enc"].T + sd["b_enc"])

        r1 = float((x_from_tl - hfin[L].squeeze(0)).abs().max()
                   / max(float(hfin[L].abs().max()), 1e-9))
        r2 = float((f_tl - their).abs().max()
                   / max(float(their.abs().max()), 1e-9))
        r3 = float((f_hf - their).abs().max()
                   / max(float(their.abs().max()), 1e-9))
        print("  %2d  |        %.2e        |        %.2e        |     %.2e"
              % (L, r1, r2, r3))

    # concrete comparison at one position: are the values a permutation,
    # a scalar multiple, or unrelated?
    L, pos = 0, 5
    sd = raw[L]
    w = hf.model.layers[L].post_attention_layernorm.weight.data
    tl_hook = cache["blocks.%d.ln2.hook_normalized" % L].squeeze(0)
    their = (acts[L].squeeze(0) if acts[L].dim() == 3 else acts[L])[pos]
    ours = code((tl_hook * w) @ sd["W_enc"].T + sd["b_enc"])[pos]
    ti = their.nonzero().flatten()
    oi = ours.nonzero().flatten()
    print("")
    print("layer 0, position %d" % pos)
    print("  their active idx (first 8):", ti[:8].tolist())
    print("  our   active idx (first 8):", oi[:8].tolist())
    print("  shared indices: %d / %d" % (len(set(ti.tolist()) & set(oi.tolist())), len(oi)))
    print("  their values (first 5):", [round(float(v), 4) for v in their[ti[:5]]])
    print("  our   values (first 5):", [round(float(v), 4) for v in ours[oi[:5]]])
    common = sorted(set(ti.tolist()) & set(oi.tolist()))[:5]
    if common:
        print("  on shared idx -> theirs:", [round(float(their[i]), 4) for i in common])
        print("                   ours  :", [round(float(ours[i]), 4) for i in common])
        ratios = [float(their[i]) / float(ours[i]) for i in common if abs(float(ours[i])) > 1e-6]
        print("  ratio theirs/ours:", [round(r, 4) for r in ratios])
    print("")
    print("READING: r2 small  => our feature maths matches theirs exactly,")
    print("         and any r1/r3 gap is the SURROGATE STREAM diverging")
    print("         (their replacement model substitutes every MLP).")
    print("         r2 large  => a genuine convention/fold bug remains.")


if __name__ == "__main__":
    main()
