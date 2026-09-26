"""GPT-2 small + the OpenAI v5 TopK SAEs — loader, verified against the repos.

WHY THIS SUBSTRATE (2026-09-20): 033's phi_pin control showed free and pinned
faithfulness AGREE on genuine Top-K dictionaries (pin0 median 0.789, 24/28
seeds >= 0.5, no depth decay) and collapse on Gemma Scope 2's uncapped JumpReLU
(phi_pin 0.00-0.39 against free0 0.67-1.13). Top-K bounds what a re-encode of a
degraded stream can do. These SAEs are the first PUBLIC set that is Top-K AND
covers every site kind, so the method's full three-kind form can run on a public
model without the uncapped-dictionary failure.

Verified from the repo configs and weights (not from a spec page):
  repos       jbloom/GPT2-Small-OAI-v5-{32k,128k}-{resid-post,resid-mid,
              mlp-out,attn-out}-SAEs     (resid-mid/mlp-out/attn-out: 12 layers
              in folders "v5_32k_layer_N"; resid-post uses "v5_32k_layer_N.pt")
  weights     W_enc [768, 32768], W_dec [32768, 768], b_enc, b_dec, float32
  config      activation_fn_str "topk", k = 32, d_in 768, d_sae 32768,
              context_size 64, prepend_bos False, normalize_activations
              "layer_norm", apply_b_dec_to_input True

SEMANTICS (SAELens sae.py, transcribed):
  in    x_n = (x - mean(x)) / (std(x) + 1e-5)     per token, std is torch's
        (unbiased) std over d_in
  code  topk_k(relu((x_n - b_dec) @ W_enc + b_enc))      apply_b_dec_to_input
  out   x_hat = (code @ W_dec + b_dec) * std + mean
So an edit in MODEL space is the usual delta, scaled by the token's std:
  x <- x + ((c_hat - c) @ W_dec) * std          (mean and b_dec cancel)
and the SAE error in model space is err = (x_n - (c @ W_dec + b_dec)) * std.

HOOK POINTS in plain HF GPT2 (no TransformerLens):
  attn-out    block.attn   output[0]        (after c_proj) = hook_attn_out
  resid-mid   block.ln_2   INPUT            = x + attn_out = hook_resid_mid
  mlp-out     block.mlp    output           = hook_mlp_out
  resid-post  block        output[0]        = hook_resid_post
Within-layer causal order: attn-out < resid-mid < mlp-out < resid-post.
"""
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

WIDTH = os.environ.get("GPT2_SAE_WIDTH", "32k")
KINDS = ("attn-out", "resid-mid", "mlp-out", "resid-post")
ORDER = {k: i for i, k in enumerate(KINDS)}


def repo(kind):
    return "jbloom/GPT2-Small-OAI-v5-%s-%s-SAEs" % (WIDTH, kind)


def folder(kind, layer):
    # resid-post is the only set whose folders keep the ".pt" suffix
    suffix = ".pt" if kind == "resid-post" else ""
    return "v5_%s_layer_%d%s" % (WIDTH, layer, suffix)


def load_sae(kind, layer, device="cuda", dtype=torch.float32):
    d = folder(kind, layer)
    sd = load_file(hf_hub_download(repo(kind), d + "/sae_weights.safetensors"))
    import json
    cfg = json.load(open(hf_hub_download(repo(kind), d + "/cfg.json")))
    t = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}
    t["k"] = int(cfg["activation_fn_kwargs"]["k"])
    t["hook_name"] = cfg["hook_name"]
    return t


def norm_in(x, eps=1e-5):
    """x_n, std, mean — SAELens run_time_activation_ln_in."""
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    std = xc.std(dim=-1, keepdim=True)
    return xc / (std + eps), std + eps, mu


def encode(t, x_n):
    """topk_k(relu(pre)) as a dense code; exactly k non-zeros per token."""
    pre = (x_n - t["b_dec"]) @ t["W_enc"] + t["b_enc"]
    v, i = pre.relu().topk(t["k"], dim=-1)
    return torch.zeros_like(pre).scatter(-1, i, v)


def decode(t, c):
    return c @ t["W_dec"] + t["b_dec"]


def module_for(model, kind, layer):
    b = model.transformer.h[layer]
    return {"attn-out": b.attn, "resid-mid": b.ln_2, "mlp-out": b.mlp, "resid-post": b}[kind]
