"""THE end-to-end agreement test: do both sides compute the SAME feature
activations for the same token?

Three convention traps have already been caught in this comparison by
measuring rather than assuming (GPT-2's LayerNorm placement, sparsify's
input centring, TransformerLens's RMSNorm hook placement). Each produced
plausible-looking but meaningless features. This script is the direct
check that no fourth one remains: it runs circuit-tracer's
ReplacementModel and our HF harness on identical tokens and compares the
transcoder feature activations feature-by-feature.

If this passes, the two sides are provably looking at the same object
and any difference in their circuits is a difference in METHOD, which is
the whole point of the comparison.

Runs in the circuit-tracer venv; imports our side's maths inline (a
dozen lines) rather than importing ours_llama.py, so the two
computations stay genuinely independent.

  ../../dev-notes/data/venv-ct/bin/python check_agreement.py
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HERE = Path(__file__).parent
TC_DIR = HERE / "transcoders_llama_ct"
MODEL_ID = "unsloth/Llama-3.2-1B"
TL_NAME = "meta-llama/Llama-3.2-1B"
K_SAE = 32
LAYERS = [int(p.stem.split("_")[1]) for p in sorted(TC_DIR.glob("layer_*.safetensors"))]
TEXT = "The Eiffel Tower is in Paris, the capital of France."


def ours_features(hf, layer, ids):
    """Our side: features from the HF MLP module input, TopK code."""
    sd = {k: v.float() for k, v in
          load_file(str(TC_DIR / ("layer_%d.safetensors" % layer))).items()}
    cap = {}

    def hook(mod, inp, out):
        cap["x"] = inp[0].detach()

    h = hf.model.layers[layer].mlp.register_forward_hook(hook)
    with torch.no_grad():
        hf(ids)
    h.remove()
    pre = cap["x"] @ sd["W_enc"].T + sd["b_enc"]
    vals, idx = pre.topk(K_SAE, dim=-1)
    return torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))


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
    for L in LAYERS:
        tc = load_transcoder(
            str(TC_DIR / ("layer_%d.safetensors" % L)), layer=L,
            activation_fn=TopK(K_SAE), device=t.device("cpu"),
            dtype=t.float32, lazy_encoder=False, lazy_decoder=False)
        w = hf.model.layers[L].post_attention_layernorm.weight.data
        with t.no_grad():
            tc.W_enc.mul_(w.to(tc.W_enc.dtype))
        transcoders[L] = tc

    if set(LAYERS) != set(range(max(LAYERS) + 1)):
        print("NOTE: only layers %s converted so far; comparing those that "
              "form a prefix." % LAYERS)
    prefix = [L for L in range(max(LAYERS) + 1) if L in transcoders]
    tset = TranscoderSet({L: transcoders[L] for L in prefix},
                         feature_input_hook="ln2.hook_normalized",
                         feature_output_hook="hook_mlp_out",
                         scan_name="agreement-check")
    model = ReplacementModel.from_pretrained_and_transcoders(
        TL_NAME, tset, device=t.device("cpu"), dtype=t.float32,
        hf_model=hf, tokenizer=tok)

    _, acts = model.get_activations(ids, sparse=False)
    # circuit-tracer ZEROES position 0 (the prepended BOS) by design:
    # zero_positions = slice(0, 1), "this prepended token is later
    # ignored". Comparing it would always look like total disagreement,
    # so the gate compares positions >= 1 and reports position 0
    # separately as a convention check rather than a failure.
    print("layer |    L0 | max|d| p>=1 |   rel   | same support | pos0 theirs")
    ok = True
    for L in prefix:
        theirs = acts[L].squeeze(0) if acts[L].dim() == 3 else acts[L]
        ours = ours_features(hf, L, ids).squeeze(0)
        t1, o1 = theirs[1:], ours[1:]
        d = float((t1 - o1).abs().max())
        rel = d / max(float(o1.abs().max()), 1e-9)
        same = bool(torch.equal((t1 != 0), (o1 != 0)))
        pos0 = float(theirs[0].abs().max())
        ok = ok and rel < 1e-3 and same
        print("  %2d  | %5.1f |   %.2e   | %.1e |     %-5s    |   %.1e"
              % (L, float((o1 != 0).sum(-1).float().mean()), d, rel, same,
                 pos0))
    print("")
    print("VERDICT: " + ("BOTH SIDES AGREE - the comparison is measuring "
                         "method, not convention"
                         if ok else "DISAGREEMENT - resolve before running"))


if __name__ == "__main__":
    main()
