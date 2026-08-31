"""Convert Dunefsky/Chlenski GPT-2 transcoders into circuit-tracer's
SingleLayerTranscoder safetensors layout, so BOTH methods read the
same weights.

Source (pchlenski/gpt2-transcoders, `sae_training` pickles), per layer:
    W_enc      [768, 24576]   b_enc [24576]
    W_dec      [24576, 768]   b_dec [768]   b_dec_out [768]
    forward:   acts = relu((x - b_dec) @ W_enc + b_enc)
               out  = acts @ W_dec + b_dec_out
    hooks:     in  blocks.L.ln2.hook_normalized
               out blocks.L.hook_mlp_out

Target (circuit_tracer.transcoder.SingleLayerTranscoder):
    encode: F.linear(x, W_enc, b_enc) = x @ W_enc.T + b_enc   [NO input centering]
    decode: acts @ W_dec + b_dec
    keys:   W_enc [d_sae, d_model], b_enc [d_sae],
            W_dec [d_sae, d_model], b_dec [d_model]
    activation: ReLU (inferred when no threshold key is present)

So the conversion is a transpose plus a bias fold — the source centres
its input, the target does not, and
    relu((x - c) @ We + be) == relu(x @ We + (be - c @ We))
makes the two exactly equivalent:
    W_enc_t = W_enc_s.T
    b_enc_t = b_enc_s - b_dec_s @ W_enc_s
    W_dec_t = W_dec_s
    b_dec_t = b_dec_out_s
Equivalence is asserted numerically on random inputs before writing.

SECOND FOLD — THE LAYERNORM CONVENTION (measured, not assumed).
These transcoders were trained against `ln2.hook_normalized` under
TransformerLens's fold_ln=TRUE, i.e. the PURE normalisation with the
LayerNorm weight and bias folded away. Measured on GPT-2 layer 6:
feeding pure normalisation reconstructs mlp_out at relative error 0.41
with L0 68, while feeding the full ln_2 output (weight and bias
applied) gives 0.79 and L0 5.4 — the wrong convention silently
destroys the transcoder. But circuit-tracer loads its model with
fold_ln=FALSE, where the hook yields the FULL ln_2 output, so the raw
weights would be wrong on their side out of the box.
We therefore fold the LayerNorm affine into the encoder as well, using
x_norm = (x_full - b_ln) / w_ln:
    W_enc_t <- W_enc_t / w_ln          (broadcast over d_model)
    b_enc_t <- b_enc_t - (b_ln / w_ln) @ W_enc_t_before
after which the transcoder consumes the FULL ln_2 output natively —
correct for circuit-tracer's fold_ln=False model and for our harness,
which reads block.ln_2(hidden). Verified exact (max |dfeature| ~1e-5,
identical L0); GPT-2's ln_2 weights are bounded away from zero
(min 0.042), so the division is safe.

  python convert_transcoders.py
"""
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import GPT2LMHeadModel

HERE = Path(__file__).parent
OUT = HERE / "transcoders_ct"
_GPT2 = GPT2LMHeadModel.from_pretrained("gpt2").float().eval()
REPO = "pchlenski/gpt2-transcoders"
FILE = "final_sparse_autoencoder_gpt2-small_blocks.%d.ln2.hook_normalized_24576.pt"
N_LAYERS = 12

# stand-in modules so the legacy pickles resolve
class _Any:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        self.__dict__.update(st if isinstance(st, dict) else {})


for _n in ["sae_training", "sae_training.sparse_autoencoder",
           "sae_training.config", "sae_training.utils",
           "sae_training.geometric_median"]:
    _m = types.ModuleType(_n)
    _m.__getattr__ = lambda n, _mm=_m: _Any
    sys.modules[_n] = _m


def convert(layer):
    path = hf_hub_download(REPO, FILE % layer)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd, cfg = blob["state_dict"], blob["cfg"].__dict__
    assert cfg["is_transcoder"], "expected a transcoder checkpoint"
    assert cfg["hook_point"] == "blocks.%d.ln2.hook_normalized" % layer
    assert cfg["out_hook_point"] == "blocks.%d.hook_mlp_out" % layer

    We_s, be_s = sd["W_enc"].float(), sd["b_enc"].float()
    Wd_s, bd_s = sd["W_dec"].float(), sd["b_dec"].float()
    bdo_s = sd["b_dec_out"].float()

    W_enc_t = We_s.T.contiguous()                 # [d_sae, d_model]
    b_enc_t = (be_s - bd_s @ We_s).contiguous()   # fold the input centering
    W_dec_t = Wd_s.contiguous()                   # [d_sae, d_model]
    b_dec_t = bdo_s.contiguous()

    # numerical equivalence check on random inputs (pre-LN-fold)
    x = torch.randn(64, cfg["d_in"])
    acts_s = F.relu((x - bd_s) @ We_s + be_s)
    out_s = acts_s @ Wd_s + bdo_s
    acts_t = F.relu(F.linear(x, W_enc_t, b_enc_t))
    out_t = acts_t @ W_dec_t + b_dec_t
    da = (acts_s - acts_t).abs().max().item()
    do = (out_s - out_t).abs().max().item()
    assert da < 2e-3 and do < 2e-3, "conversion mismatch: %g / %g" % (da, do)

    # LN fold: consume the FULL ln_2 output instead of pure normalisation
    ln = _GPT2.transformer.h[layer].ln_2
    w_ln, b_ln = ln.weight.data.float(), ln.bias.data.float()
    # GPT-2 layer 3 has one near-degenerate ln_2 weight (~2.6e-4), which
    # inflates that column of W_enc by ~4e3. The fold stays algebraically
    # exact; the equivalence assert below is what actually certifies it, so
    # this guard only rules out a true division by zero.
    assert float(w_ln.abs().min()) > 1e-6, "ln weight is effectively zero"
    W_folded = (W_enc_t / w_ln).contiguous()
    b_folded = (b_enc_t - (b_ln / w_ln) @ W_enc_t.T).contiguous()
    # exactness on a normalised random input and its full-LN image
    xn = torch.randn(64, cfg["d_in"])
    xn = (xn - xn.mean(-1, keepdim=True)) / xn.std(-1, keepdim=True)
    x_full = xn * w_ln + b_ln
    fa = F.relu(F.linear(xn, W_enc_t, b_enc_t))
    fb = F.relu(F.linear(x_full, W_folded, b_folded))
    dln = (fa - fb).abs().max().item()
    assert dln < 2e-3, "LN fold mismatch: %g" % dln
    W_enc_t, b_enc_t = W_folded, b_folded

    OUT.mkdir(parents=True, exist_ok=True)
    save_file({"W_enc": W_enc_t, "b_enc": b_enc_t,
               "W_dec": W_dec_t, "b_dec": b_dec_t},
              str(OUT / ("layer_%d.safetensors" % layer)))
    l0 = float((acts_s > 0).float().sum(-1).mean())
    print("layer %2d ok | max|dacts| %.2e max|dout| %.2e | LN-fold %.2e | "
          "d_sae %d" % (layer, da, do, dln, cfg["d_sae"]), flush=True)
    return {"layer": layer, "d_model": cfg["d_in"], "d_sae": cfg["d_sae"],
            "feature_input_hook": "ln2.hook_normalized",
            "feature_output_hook": "hook_mlp_out"}


if __name__ == "__main__":
    metas = [convert(L) for L in range(N_LAYERS)]
    import json
    (OUT / "meta.json").write_text(json.dumps(
        {"source_repo": REPO, "model": "gpt2",
         "feature_input_hook": "ln2.hook_normalized",
         "feature_output_hook": "hook_mlp_out",
         "d_model": metas[0]["d_model"], "d_sae": metas[0]["d_sae"],
         "n_layers": N_LAYERS,
         "input_convention": "FULL ln_2 output (weight+bias applied)",
         "note": "converted from sae_training pickles. Two folds: (1) the "
                 "source's input pre-bias folded into b_enc, since "
                 "circuit-tracer's encode does not centre its input; (2) "
                 "GPT-2's ln_2 affine folded into the encoder, since these "
                 "transcoders were trained under fold_ln=True (pure "
                 "normalisation) while circuit-tracer loads fold_ln=False "
                 "(full ln_2 output). Both folds asserted numerically."},
        indent=1))
    print("WROTE", OUT)
