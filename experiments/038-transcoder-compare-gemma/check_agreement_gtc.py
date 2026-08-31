"""THE AGREEMENT GATE on their turf: do circuit-tracer's native "gemma"
ReplacementModel and our HF harness compute the SAME transcoder
features for the same tokens?

Our side's maths is reimplemented inline (a dozen lines) rather than
imported from ours_gtc.py, so the two computations stay independent.
Positions >= 1 only: circuit-tracer zeroes position 0 by design and
Gemma's BOS is anomalous anyway (probe_gemma_tc.py).

If this passes, any difference in circuits is METHOD, not convention —
on the tool's own default scan, with zero weight conversion.

  ../../dev-notes/data/venv-ct/bin/python check_agreement_gtc.py
"""
from pathlib import Path

import torch
from safetensors.torch import load_file

TC_DIR = Path.home() / "gemma_tc"
MODEL_ID = "unsloth/gemma-2-2b"
TL_NAME = "google/gemma-2-2b"
TEXT = "The Eiffel Tower is in Paris, the capital of France."
LAYERS = [0, 2, 4, 6, 8]


def ours_features(hf, layer, ids):
    sd = {k: v.float() for k, v in
          load_file(str(TC_DIR / ("layer_%d.safetensors" % layer))).items()}
    thr = sd.get("activation_function.threshold", sd.get("threshold"))
    blk = hf.model.layers[layer]
    cap = {}
    h = blk.pre_feedforward_layernorm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("x", o.detach()))
    with torch.no_grad():
        hf(ids)
    h.remove()
    x_in = cap["x"] / (1.0 + blk.pre_feedforward_layernorm.weight.data)
    pre = x_in @ sd["W_enc"].T + sd["b_enc"]
    return pre * (pre > thr)


def main():
    import torch as t
    from circuit_tracer.replacement_model import ReplacementModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=t.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(TEXT, return_tensors="pt")["input_ids"]
    model = ReplacementModel.from_pretrained(
        TL_NAME, "gemma", device=t.device("cpu"), dtype=t.float32,
        hf_model=hf, tokenizer=tok)
    print("their hooks:", getattr(model, "feature_input_hook", "?"), "->",
          getattr(model, "feature_output_hook", "?"))
    _, acts = model.get_activations(ids, sparse=False)

    print("layer |   L0  | max|d| p>=1 |   rel   | support flips / magnitudes | pos0")
    ok = True
    for L in LAYERS:
        theirs = acts[L].squeeze(0) if acts[L].dim() == 3 else acts[L]
        ours = ours_features(hf, L, ids).squeeze(0)
        t1, o1 = theirs[1:], ours[1:]
        d = float((t1 - o1).abs().max())
        rel = d / max(float(o1.abs().max()), 1e-9)
        same = bool(torch.equal((t1 != 0), (o1 != 0)))
        # Drift budget. debug_agreement_gtc.py localised the gap to TL-vs-HF
        # ATTENTION numerics (~4e-4 on the stream, configs identical), which
        # JumpReLU thresholds turn into support flips for near-threshold
        # features. So the gate is the FLIP RATE, not bit-exactness: per
        # position, features active on one side only, as a share of L0.
        ta, oa = (t1 != 0), (o1 != 0)
        flips = (ta ^ oa).sum(-1).float()
        l0 = oa.sum(-1).float().clamp(min=1)
        flip_rate = float((flips / l0).mean())
        # and are the flipped features near threshold (tiny), not real?
        flipped_mag = float(torch.where(ta ^ oa, (t1 - o1).abs(),
                                        torch.zeros_like(t1)).max())
        both = ta & oa
        agree_mag = float(torch.where(both, (t1 - o1).abs(),
                                      torch.zeros_like(t1)).max())
        ok = ok and flip_rate < 0.05 and agree_mag / max(float(o1.abs().max()), 1e-9) < 1e-2
        print("  %2d  | %5.1f |   %.2e   | %.1e |  flips/L0 %.3f  max|flip| %.3f  "
              "max|d| on shared %.2e |  %.1e"
              % (L, float(oa.sum(-1).float().mean()), d, rel, flip_rate,
                 flipped_mag, agree_mag, float(theirs[0].abs().max())))
    print("\nVERDICT: " + ("AGREE WITHIN DRIFT BUDGET (flips<5%% of L0, shared-feature err<1%%)" if ok else
                           "DISAGREEMENT - resolve before running"))


if __name__ == "__main__":
    main()
