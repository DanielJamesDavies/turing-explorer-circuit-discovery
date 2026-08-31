"""Convention prober for circuit-tracer's HOME-TURF transcoders:
mwhanna/gemma-scope-transcoders (the tool's default "gemma" scan; a
repackaging of google/gemma-scope-2b-pt-transcoders, JumpReLU, 16k, one
fixed L0 pick per layer, already in circuit-tracer's key layout).

Both sides of the comparison will read THESE EXACT FILES — no
conversion — so the only thing that can go wrong is a convention, and
Gemma-2 offers three fresh ways:

  1. WHICH INPUT. Gemma-2 has four layernorms per block; the MLP path is
     pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm.
     Candidate inputs: pre_ffw_ln's output (the HF MLP input) and the
     residual before it.
  2. WHICH TARGET. "hook_mlp_out" in the shipped config is a
     TransformerLens name; on the HF side the candidates are the raw MLP
     output and its post_feedforward_layernorm image (which is what is
     actually added to the residual). The intervention delta must be
     injected at whichever tensor the decoder predicts, so this choice
     is not cosmetic.
  3. THE (1+w) TRAP. Gemma RMSNorm scales by (1 + weight), not weight.
     Any fold between a TL-style hook and the HF tensor must use (1+w);
     folding w is silently wrong by roughly the weight's magnitude.

Scores every (input, target, centering) combination by FVU and checks
measured L0 against the JumpReLU threshold's implied sparsity.

  ../../../.venv/bin/python probe_gemma_tc.py
"""
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

HERE = Path(__file__).parent
TC_REPO = "mwhanna/gemma-scope-transcoders"
MODEL_ID = "unsloth/gemma-2-2b"
PROBE_LAYER = 4
TEXT = ("The Eiffel Tower is in Paris, the capital of France. "
        "Machine learning models process text one token at a time.")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = hf_hub_download(TC_REPO, "layer_%d.safetensors" % PROBE_LAYER)
    sd = {k: v.float() for k, v in load_file(p).items()}
    print("tensors:", {k: tuple(v.shape) for k, v in sd.items()})

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32).eval()
    ids = tok(TEXT, return_tensors="pt")["input_ids"]

    blk = model.model.layers[PROBE_LAYER]
    cap = {}
    hs = [blk.mlp.register_forward_hook(
        lambda m, i, o: cap.update(mlp_in=i[0].detach(),
                                   mlp_out=o.detach())),
          blk.pre_feedforward_layernorm.register_forward_hook(
        lambda m, i, o: cap.update(pre_ffw_in=i[0].detach())),
          blk.post_feedforward_layernorm.register_forward_hook(
        lambda m, i, o: cap.update(post_ffw_out=o.detach()))]
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()

    W_enc = sd["W_enc"]            # circuit-tracer layout: [d_sae, d_model]
    b_enc, W_dec, b_dec = sd["b_enc"], sd["W_dec"], sd["b_dec"]
    thr = sd.get("activation_function.threshold", sd.get("threshold"))
    print("||b_dec|| %.3f | threshold: min %.3f med %.3f"
          % (b_dec.norm(), thr.min(), thr.median()))

    # THE (1+w) CANDIDATES. TL hooks fire before the norm WEIGHT, and
    # Gemma RMSNorm scales by (1 + w). So the TL-trained tensor is
    # rmsnorm(x) with NO weight: divide the HF module output by (1+w).
    w_pre = blk.pre_feedforward_layernorm.weight.data
    w_post = blk.post_feedforward_layernorm.weight.data
    inputs = {"mlp_in (pre_ffw_ln out)": cap["mlp_in"],
              "mlp_in/(1+w)  [TL hook]": cap["mlp_in"] / (1 + w_pre),
              "pre_ffw_ln in (residual)": cap["pre_ffw_in"]}
    targets = {"mlp_out (raw)": cap["mlp_out"],
               "post_ffw_ln(mlp_out)": cap["post_ffw_out"],
               "postln/(1+w) [TL hook]": cap["post_ffw_out"] / (1 + w_post)}
    print("\n%-28s %-22s %-9s %7s %8s %8s"
          % ("input", "target", "centering", "L0", "FVU", "FVU>=1"))
    best = None
    for iname, x in inputs.items():
        for centering in (False, True):
            xin = x - b_dec if centering else x
            pre = xin @ W_enc.T + b_enc
            acts = pre * (pre > thr)
            rec = acts @ W_dec + b_dec
            l0 = float((acts > 0).sum(-1).float().mean())
            for tname, tgt in targets.items():
                fvu = float(((rec - tgt) ** 2).sum()
                            / ((tgt - tgt.mean()) ** 2).sum())
                # Gemma BOS activations are anomalously large and can
                # dominate an FVU; position 0 is excluded from the
                # decisive number (circuit-tracer zeroes it anyway).
                r1, t1 = rec[:, 1:], tgt[:, 1:]
                fvu1 = float(((r1 - t1) ** 2).sum()
                             / ((t1 - t1.mean()) ** 2).sum())
                print("{:<28} {:<22} {!s:<9} {:7.1f} {:8.4f} {:8.4f}".format(
                    iname, tname, centering, l0, fvu, fvu1))
                if best is None or fvu1 < best[0]:
                    best = (fvu1, iname, tname, centering, l0)
    print("\nBEST: FVU %.4f | input=%r target=%r centering=%s L0=%.1f"
          % best[::1][0:1] + ())


if __name__ == "__main__":
    main()
