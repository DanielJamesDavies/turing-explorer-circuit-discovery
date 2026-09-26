"""Gemma Scope 2 loader for Gemma 3 270M (google/gemma-scope-2-270m-pt).

Kind names follow the 048 fitter (att / mlp / res) and the returned tensor
keys follow the GemmaScope 1 convention the 048-051 code already uses
(W_enc, b_enc, threshold, W_dec, b_dec), so callers can switch loaders
without touching their encode/decode code.

  W_enc  (d_in, W)     pre  = x @ W_enc + b_enc        [SUB_BDEC=0]
  b_enc  (W,)          pre  = (x - b_dec) @ W_enc + b_enc  [SUB_BDEC=1]
  threshold (W,)       code = pre * (pre > threshold)   (JumpReLU)
  W_dec  (W, d_out)    x_hat = code @ W_dec + b_dec
  b_dec  (d_out,)

Hook points (from each SAE's config.json):
  att = model.layers.N.self_attn.o_proj INPUT  (n_heads x head_dim)
  mlp = model.layers.N.post_feedforward_layernorm OUTPUT
  res = model.layers.N OUTPUT

Env: GS2_WIDTH (16k) | GS2_L0 (big) | GS2_REPO
"""
import json
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = os.environ.get("GS2_REPO", "google/gemma-scope-2-270m-pt")
WIDTH = os.environ.get("GS2_WIDTH", "16k")
L0 = os.environ.get("GS2_L0", "big")
FOLDER = {"att": "attn_out_all", "mlp": "mlp_out_all", "res": "resid_post_all"}
KEYMAP = {"w_enc": "W_enc", "b_enc": "b_enc", "threshold": "threshold", "w_dec": "W_dec", "b_dec": "b_dec"}


def path(kind, layer):
    return "%s/layer_%d_width_%s_l0_%s" % (FOLDER[kind], layer, WIDTH, L0)


def config(kind, layer):
    return json.load(open(hf_hub_download(REPO, path(kind, layer) + "/config.json")))


def load_sae(kind, layer, device="cpu", dtype=torch.float32):
    raw = load_file(hf_hub_download(REPO, path(kind, layer) + "/params.safetensors"))
    return {KEYMAP[k]: v.to(device=device, dtype=dtype) for k, v in raw.items()}


def encode(sae, x, sub_bdec=False):
    if sub_bdec:
        x = x - sae["b_dec"]
    pre = x @ sae["W_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"])


def decode(sae, code):
    return code @ sae["W_dec"] + sae["b_dec"]
