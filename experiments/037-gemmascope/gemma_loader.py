"""Loader + convention prober for GemmaScope JumpReLU SAEs on Gemma-2-2B.

WHY THIS EXPERIMENT EXISTS
Every dictionary the tri-amp mask has been validated on is top-k-like
(home bank k=128/40,960; Pythia TopK; Llama TopK skip-transcoders). The
one ReLU/L1 bank we tried went vacuous: random size-matched sets scored
~1.0, so the null could not discriminate and we scoped it out of domain.

The suspected mechanism is a RATIO -- latents active at the anchor over
live latents at that site. Low ratio (TopK): a random set is mostly
exact zeros, so no fitted amplitude can rescue it and the null is easy.
High ratio (ReLU/L1): enough latents are live that many different sets
span the target, the circuit stops being identified, and the null is
unbeatable. Two opposite failure modes of the same quantity.

GemmaScope is the instrument that turns this from a yes/no into a CURVE.
It ships FIVE L0 variants per layer at fixed width, so we can sweep
per-position density at 16k width from ~0.15% to ~4.3% (layer 6: L0 25,
55, 133, 328, 699) and watch identifiability degrade -- rather than
sampling one point and arguing about it.

JumpReLU, the third activation family:
    pre  = x @ W_enc + b_enc
    acts = pre * (pre > threshold)          [per-latent learned threshold]
Unlike TopK there is no fixed L0, and unlike ReLU/L1 there is no
shrinkage tail below the threshold -- it sits between the two regimes,
which is exactly why it is the informative test.

CONVENTIONS ARE PROBED, NOT ASSUMED. Five convention traps were caught
by measuring on the Llama side (LayerNorm placement, input centring,
RMSNorm hook placement, BOS handling, top-k tie-breaking); each produced
plausible-looking but meaningless features. `probe` scores every
combination by fraction of variance unexplained against the true
activation and reports the winner.

  python gemma_loader.py list                 # what variants exist
  python gemma_loader.py probe                # ONE sae + model, CPU
  python gemma_loader.py fetch <tier>         # cache a sparsity tier
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
# Native ext4, not /mnt/x: measured 216 MB/s (barely cached) vs 28.5 GB/s
# cached on ext4, which was worth ~6x on the Llama run.
CACHE = Path(os.environ.get("SAE_CACHE", str(Path.home() / "gemmascope")))
SAE_REPO = "google/gemma-scope-2b-pt-mlp"
# ungated mirror, same trick the Llama side used for meta-llama
MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
WIDTH = os.environ.get("WIDTH", "16k")
PROBE_LAYER = int(os.environ.get("PROBE_LAYER", 4))
TEXT = ("The Eiffel Tower is in Paris, the capital of France. "
        "Machine learning models process text one token at a time.")


def variants():
    """{layer: [l0 ...]} available at WIDTH, from the repo listing."""
    import re
    from huggingface_hub import list_repo_files
    pat = re.compile(r"layer_(\d+)/width_%s/average_l0_(\d+)/" % WIDTH)
    out = {}
    for f in list_repo_files(SAE_REPO):
        m = pat.search(f)
        if m:
            out.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    return {k: sorted(v) for k, v in sorted(out.items())}


def tier_l0(layer, tier, v=None):
    """L0 for a sparsity TIER (0 = sparsest) at this layer.

    Each layer publishes its own L0 ladder, so a tier index -- not an
    absolute L0 -- is what makes 'the same sparsity setting' meaningful
    across the upstream sites of one circuit.
    """
    v = v or variants()
    ls = v[layer]
    return ls[min(tier, len(ls) - 1)]


def load_sae(layer, l0, device="cpu"):
    """Fetch (cached) one SAE's params as tensors."""
    from huggingface_hub import hf_hub_download
    name = "layer_%d/width_%s/average_l0_%d/params.npz" % (layer, WIDTH, l0)
    local = CACHE / ("layer_%d_w%s_l0_%d.npz" % (layer, WIDTH, l0))
    if not local.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(SAE_REPO, name)
        local.write_bytes(Path(p).read_bytes())
    z = np.load(local)
    return {k: torch.tensor(z[k], dtype=torch.float32, device=device)
            for k in z.files}


def jumprelu(pre, threshold, gate="raw"):
    """JumpReLU code. `gate` distinguishes two readings of the rule that
    differ whenever a pre-activation is negative but above threshold --
    which is why it is probed rather than assumed."""
    if gate == "raw":
        return pre * (pre > threshold)
    return torch.relu(pre) * (pre > threshold)


def probe():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    v = variants()
    l0 = tier_l0(PROBE_LAYER, 2, v)
    print("variants at layer %d width %s: %s" % (PROBE_LAYER, WIDTH,
                                                 v[PROBE_LAYER]))
    print("probing with average_l0 = %d" % l0)
    sd = load_sae(PROBE_LAYER, l0)
    print("tensors:", {k: tuple(t.shape) for k, t in sd.items()})

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32).eval()
    ids = tok(TEXT, return_tensors="pt")["input_ids"]

    blk = model.model.layers[PROBE_LAYER]
    print("block submodules:", [n for n, _ in blk.named_children()])
    cap = {}
    hs = [blk.mlp.register_forward_hook(
        lambda m, i, o: cap.update(mlp_in=i[0].detach(),
                                   mlp_out=o.detach()))]
    # GEMMA-2 IS NOT LLAMA. It has FOUR layernorms per block, and
    # post_attention_layernorm sits on the ATTENTION path -- it is not
    # the MLP's input as it is in Llama. The MLP's actual contribution
    # to the residual is post_feedforward_layernorm(mlp(...)), so that
    # tensor has to be a candidate or the real convention is never
    # tested. Assuming Llama's layout here is what produced FVU 0.69
    # with no combination anywhere near the advertised L0.
    for name in ("post_attention_layernorm", "pre_feedforward_layernorm",
                 "post_feedforward_layernorm"):
        mod = getattr(blk, name, None)
        if mod is None:
            continue
        hs.append(mod.register_forward_hook(
            lambda m, i, o, _n=name: cap.update(
                **{_n + "_in": i[0].detach(), _n + "_out": o.detach()})))
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()

    W_enc, b_enc = sd["W_enc"], sd["b_enc"]
    W_dec, b_dec, thr = sd["W_dec"], sd["b_dec"], sd["threshold"]

    # These SAEs are labelled "-mlp"; which tensor that means is exactly
    # the kind of thing that must be measured. An SAE reconstructs its
    # OWN input, so the target is whatever the input is.
    cands = {k: v for k, v in [
        ("mlp_out (raw MLP output)", cap.get("mlp_out")),
        ("post_ffw_ln_out (MLP's residual contribution)",
         cap.get("post_feedforward_layernorm_out")),
        ("mlp_in (pre_ffw_ln out)", cap.get("mlp_in")),
        ("pre_ffw_ln_in (residual before MLP)",
         cap.get("pre_feedforward_layernorm_in")),
        ("post_attn_ln_out (attention path)",
         cap.get("post_attention_layernorm_out")),
    ] if v is not None}
    # Scale sanity: b_dec is fitted to the target's mean, so a target
    # whose norm is wildly off b_dec's is the wrong tensor regardless of
    # what FVU says.
    print("\n||b_dec|| = %.3f" % float(b_dec.norm()))
    for name, x in cands.items():
        print("   %-46s mean|x| %8.3f  ||x||/tok %8.3f"
              % (name, float(x.abs().mean()),
                 float(x.reshape(-1, x.shape[-1]).norm(dim=-1).mean())))
    print("\n%-40s %-9s %-6s %7s %8s" % ("input=target", "centering",
                                         "gate", "L0", "FVU"))
    best = None
    for name, x in cands.items():
        for centering in (False, True):
            xin = x - b_dec if centering else x
            for gate in ("raw", "relu"):
                pre = xin @ W_enc + b_enc
                acts = jumprelu(pre, thr, gate)
                rec = acts @ W_dec + b_dec
                fvu = float(((rec - x) ** 2).sum()
                            / ((x - x.mean()) ** 2).sum())
                l0m = float((acts > 0).sum(-1).float().mean())
                print("%-40s %-9s %-6s %7.1f %8.4f"
                      % (name, centering, gate, l0m, fvu))
                if best is None or fvu < best[0]:
                    best = (fvu, name, centering, gate, l0m)
    print("\nBEST: FVU %.4f | input=%r centering=%s gate=%s L0=%.1f"
          % (best[0], best[1], best[2], best[3], best[4]))
    if best[0] > 0.5:
        print("WARNING: FVU > 0.5 means no convention reconstructs well; "
              "do NOT build on this until it is resolved.")
    (HERE / "gemma_convention.json").write_text(json.dumps(
        {"fvu": round(best[0], 5), "input": best[1],
         "subtract_b_dec": best[2], "gate": best[3],
         "measured_l0": round(best[4], 2), "layer_probed": PROBE_LAYER,
         "width": WIDTH, "l0_variant": l0, "sae_repo": SAE_REPO,
         "model": MODEL_ID}, indent=1))
    print("wrote gemma_convention.json")


def fetch(tier, layers):
    """Pre-cache every SAE for one sparsity tier onto native disk."""
    v = variants()
    tot = 0
    for L in layers:
        l0 = tier_l0(L, tier, v)
        sd = load_sae(L, l0)
        mb = sum(t.numel() * 4 for t in sd.values()) / 1024 ** 2
        tot += mb
        print("layer %-2d tier %d -> average_l0 %-4d | %6.1f MB | %s"
              % (L, tier, l0, mb, {k: tuple(t.shape) for k, t in sd.items()}),
              flush=True)
    print("tier %d cached, %.2f GB total, at %s" % (tier, tot / 1024, CACHE))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for k, val in variants().items():
            print("layer %-2d width %-4s L0: %s" % (k, WIDTH, val))
    elif cmd == "probe":
        probe()
    elif cmd == "fetch":
        tier = int(sys.argv[2])
        layers = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 \
            else list(range(7))
        fetch(tier, layers)
    else:
        raise SystemExit("usage: gemma_loader.py list|probe|fetch <tier> [layers]")
