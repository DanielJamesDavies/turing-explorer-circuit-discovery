"""Bake the RMSNorm fold into W_enc ON DISK, in bf16, on native disk.

WHY. A 16k-node attribution graph is the smallest that includes our seed
feature (at 4096 the seed gets no row at all), and it does not fit in
16 GB alongside 8.6 GB of resident encoders -- even at batch 256, under
a hard VRAM cap. Lazy encoders would free that, but circuit-tracer's
lazy path reads W_enc straight from the file, so the RMSNorm fold this
side needs (`W_enc * w`, worth a 2.3-6.3x error if skipped) can no
longer be applied in memory.

Folding on disk resolves the conflict: the lazy read then returns
already-folded weights, and the fold is the same algebra as before,
x_full @ W_enc.T == x_hook @ (W_enc * w).T, just computed once.

Also stores bf16 rather than fp32: that is the dtype everything is cast
to anyway, and it halves the bytes each lazy read pulls.

The fold is computed in fp32 and cast afterwards, so it does not inherit
bf16 rounding -- matching what build_model did in memory.

  ../../dev-notes/data/venv-ct/bin/python prefold_weights.py            # $HOME/tc_llama -> $HOME/tc_llama_folded
"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path(os.environ.get("SRC", str(Path.home() / "tc_llama")))
DST = Path(os.environ.get("DST", str(Path.home() / "tc_llama_folded")))
MODEL_ID = "unsloth/Llama-3.2-1B"
N_LAYERS = 16
DTYPE = torch.bfloat16


def main():
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    DST.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    for L in range(N_LAYERS):
        sd = load_file(str(SRC / ("layer_%d.safetensors" % L)), device="cpu")
        w = hf.model.layers[L].post_attention_layernorm.weight.data.float()
        out = {}
        for k, v in sd.items():
            v = v.float()
            if k == "W_enc":
                v = v * w            # the fold, in fp32
            out[k] = v.to(DTYPE).contiguous()

        # Verify the fold on random inputs: folded-and-cast weights must
        # reproduce the in-memory computation to bf16 precision.
        torch.manual_seed(0)
        x = torch.randn(32, sd["W_enc"].shape[1])
        ref = (x * w) @ sd["W_enc"].float().T + sd["b_enc"].float()
        got = x.to(DTYPE) @ out["W_enc"].T + out["b_enc"]
        rel = float((got.float() - ref).abs().max()
                    / max(float(ref.abs().max()), 1e-9))
        worst = max(worst, rel)
        assert rel < 5e-2, "layer %d fold mismatch: rel %g" % (L, rel)

        save_file(out, str(DST / ("layer_%d.safetensors" % L)))
        print("layer %2d folded | rel %.2e | keys %s"
              % (L, rel, list(out)), flush=True)

    (DST / "meta.json").write_text(json.dumps(
        {"source": str(SRC), "dtype": "bfloat16", "rms_fold": "baked in",
         "note": "W_enc pre-multiplied by post_attention_layernorm.weight "
                 "so circuit-tracer's LAZY encoder path reads already-"
                 "folded weights. Do NOT fold again in memory.",
         "worst_rel": worst}, indent=1))
    print("WROTE %s | worst rel %.2e" % (DST, worst), flush=True)


if __name__ == "__main__":
    sys.exit(main())
