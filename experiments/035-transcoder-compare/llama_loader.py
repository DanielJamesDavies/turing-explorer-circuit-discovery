"""Loader + convention prober for EleutherAI `sparsify` skip-transcoders
(TopK), and converter into circuit-tracer's SingleLayerTranscoder keys.

Target: EleutherAI/skip-transcoder-Llama-3.2-1B-131k
        TopK k=32, 131,072 latents, d_in 2048, skip_connection, 16 layers
        (model: unsloth/Llama-3.2-1B, an ungated mirror of meta-llama's)

Weight layout (identical to the Pythia Top-K SAEs we already handled,
plus a skip term):
    encoder.weight [d_sae, d_model]   encoder.bias [d_sae]
    W_dec          [d_sae, d_model]   b_dec        [d_model]
    W_skip         [d_model, d_model]

TWO CONVENTIONS THAT MUST BE MEASURED, NOT ASSUMED
1. Input centering. sparsify's encode subtracts b_dec for SAEs but NOT
   for transcoders (`if not cfg.transcode: x = x - b_dec`). The Llama
   cfg does not carry the `transcode` flag (SmolLM2's does), so which
   branch trained these is not readable from the config.
2. Which activation is the input: the MLP module's input
   (post_attention_layernorm output) or the pre-norm residual.
Getting either wrong silently destroys the transcoder — the same class
of bug as the GPT-2 LayerNorm convention (see convert_transcoders.py).
`probe` scores every combination by fraction of variance unexplained
against the true MLP output and reports the winner.

circuit-tracer expects keys W_enc/b_enc/W_dec/b_dec/W_skip with the
same shapes, so conversion is a rename once the convention is known.

  python llama_loader.py probe      # downloads ONE layer + model, CPU
  python llama_loader.py convert    # writes all layers for circuit-tracer
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

HERE = Path(__file__).parent
OUT = HERE / "transcoders_llama_ct"
TC_REPO = "EleutherAI/skip-transcoder-Llama-3.2-1B-131k"
MODEL = "unsloth/Llama-3.2-1B"
N_LAYERS, K_SAE = 16, 32
PROBE_LAYER = 4
TEXT = ("The Eiffel Tower is in Paris, the capital of France. "
        "Machine learning models process text one token at a time.")


def load_raw(layer):
    path = hf_hub_download(TC_REPO, "layers.%d.mlp/sae.safetensors" % layer)
    sd = load_file(path, device="cpu")
    return {k: v.float() for k, v in sd.items()}


def topk_code(pre, k=K_SAE):
    vals, idx = pre.topk(k, dim=-1)
    return torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))


def probe():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32).eval()
    ids = tok(TEXT, return_tensors="pt")["input_ids"]
    sd = load_raw(PROBE_LAYER)
    We, be = sd["encoder.weight"], sd["encoder.bias"]
    Wd, bd = sd["W_dec"], sd["b_dec"]
    Ws = sd.get("W_skip")
    print("tensors:", {k: tuple(v.shape) for k, v in sd.items()})

    blk = model.model.layers[PROBE_LAYER]
    cap = {}

    def mlp_hook(mod, inp, out):
        cap["mlp_in"] = inp[0].detach()
        cap["mlp_out"] = out.detach()

    def ln_hook(mod, inp, out):
        cap["pre_norm_resid"] = inp[0].detach()

    h1 = blk.mlp.register_forward_hook(mlp_hook)
    h2 = blk.post_attention_layernorm.register_forward_hook(ln_hook)
    with torch.no_grad():
        model(ids)
    h1.remove(); h2.remove()

    tgt = cap["mlp_out"]
    inputs = {"mlp module input (post_attn_ln out)": cap["mlp_in"],
              "pre-norm residual": cap["pre_norm_resid"]}
    print("\n%-38s %-9s %-7s %6s %8s" % ("input", "centering", "skip",
                                         "L0", "FVU"))
    best = None
    for iname, x in inputs.items():
        for centering in (False, True):
            xin = x - bd if centering else x
            code = topk_code(xin @ We.T + be)
            base = code @ Wd + bd
            for use_skip in ([False, True] if Ws is not None else [False]):
                rec = base + (x @ Ws.T if use_skip else 0.0)
                fvu = float(((rec - tgt) ** 2).sum()
                            / ((tgt - tgt.mean()) ** 2).sum())
                l0 = float((code > 0).sum(-1).float().mean())
                print("%-38s %-9s %-7s %6.1f %8.4f"
                      % (iname, centering, use_skip, l0, fvu))
                if best is None or fvu < best[0]:
                    best = (fvu, iname, centering, use_skip)
    print("\nBEST: FVU %.4f with input=%r centering=%s skip=%s"
          % (best[0], best[1], best[2], best[3]))
    (HERE / "llama_convention.json").write_text(json.dumps(
        {"fvu": round(best[0], 5), "input": best[1],
         "subtract_b_dec": best[2], "use_skip": best[3],
         "k": K_SAE, "layer_probed": PROBE_LAYER}, indent=1))
    print("wrote llama_convention.json")


def convert(layers=None):
    """Rename into circuit-tracer's key layout, folding the input
    centering into b_enc when the probe found one (their encode does not
    centre):  topk((x - c) @ We.T + be) == topk(x @ We.T + (be - c @ We.T))
    Equivalence is asserted numerically per layer."""
    conv = json.loads((HERE / "llama_convention.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for L in (layers if layers is not None else range(N_LAYERS)):
        sd = load_raw(L)
        We, be = sd["encoder.weight"], sd["encoder.bias"]
        bd = sd["b_dec"]
        if conv["subtract_b_dec"]:
            be_folded = be - bd @ We.T
            torch.manual_seed(0)
            x = torch.randn(64, We.shape[1])
            pa = (x - bd) @ We.T + be
            pb = x @ We.T + be_folded
            d = float((pa - pb).abs().max())
            assert d < 2e-3, "centering fold mismatch: %g" % d
            # Selection: the fold is algebraically exact, so indices can
            # only differ where two features are near-tied at the k-th
            # boundary and float noise (~1e-6) reorders them. Asserting
            # exact index equality is therefore wrong — it fails on
            # harmless ties. Assert instead that the resulting CODE
            # agrees, which is what every downstream computation sees.
            def code_of(p):
                v, i = p.topk(K_SAE, dim=-1)
                return torch.zeros_like(p).scatter(-1, i, F.relu(v))
            dc = float((code_of(pa) - code_of(pb)).abs().max())
            scale = float(code_of(pa).abs().max())
            assert dc < max(1e-3 * scale, 1e-3), (
                "code mismatch under the fold: %g (scale %g)" % (dc, scale))
            be = be_folded
        else:
            d = 0.0
        out = {"W_enc": We.contiguous(), "b_enc": be.contiguous(),
               "W_dec": sd["W_dec"].contiguous(), "b_dec": bd.contiguous()}
        if "W_skip" in sd:
            out["W_skip"] = sd["W_skip"].contiguous()
        save_file(out, str(OUT / ("layer_%d.safetensors" % L)))
        print("layer %2d written | centering fold max|d| %.2e | keys %s"
              % (L, d, list(out)), flush=True)
    (OUT / "meta.json").write_text(json.dumps(
        {"source_repo": TC_REPO, "model": MODEL, "k": K_SAE,
         "n_layers": N_LAYERS, "convention": conv,
         "note": "sparsify TopK skip-transcoders; keys renamed to "
                 "circuit-tracer's SingleLayerTranscoder layout. Shapes "
                 "already match [d_sae, d_model]."}, indent=1))
    print("WROTE", OUT)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "probe":
        probe()
    elif cmd == "convert":
        # optional layer list: `convert 4` converts only layer 4 (smoke test)
        ls = [int(a) for a in sys.argv[2:]] or None
        convert(ls)
    else:
        raise SystemExit("usage: llama_loader.py probe|convert [layers...]")
