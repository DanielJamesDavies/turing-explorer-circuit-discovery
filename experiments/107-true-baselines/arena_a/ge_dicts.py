"""Arena A1 (DAN-192): the public GPT-2 dictionaries as OpenMOSS `lm_saes.sae.SparseAutoEncoder`s (v0.1.0, unchanged), so
Ge et al.'s own attributor (`lm_saes.circuit`) runs on them. Conversion glue, all ours, labelled in README.md section 3.
Run in ~/venvs/ge107 (source ~/venvs/ge107/bin/ge_env.sh; that puts lm_saes and the vendored TL fork on PYTHONPATH).

  transcoder L  Dunefsky et al. (pchlenski/gpt2-transcoders @4f3ca345, sae_training pickles, cuda:0 tensors)
                source:  f = relu((x - b_dec) W_enc + b_enc), out = f W_dec + b_dec_out,  x = blocks.L.ln2.hook_normalized
                lm_saes: hook_point_in blocks.L.ln2.hook_normalized -> hook_point_out blocks.L.hook_mlp_out;
                         encoder.weight = W_enc^T, encoder.bias = b_enc - b_dec W_enc (centring folded, exact);
                         decoder.weight = W_dec^T, decoder.bias = b_dec_out; apply_decoder_bias_to_pre_encoder False
  z-SAE L       Kissane et al. (ckkissane/attn-saes-gpt2-small-all-layers @97e0da0; raw {W_enc, W_dec, b_enc, b_dec})
                source:  f = relu((x - b_dec) W_enc + b_enc), x_hat = f W_dec + b_dec,  x = blocks.L.attn.hook_z (cat heads)
                lm_saes: hook_point_in = hook_point_out = blocks.L.attn.hook_z; decoder.bias = b_dec,
                         apply_decoder_bias_to_pre_encoder True (the same centring, done by their code)
  both          norm_activation "none" (factor 1), feature_act_mask = feature_act_scale = 1, float32.

ZSparseAutoEncoder: hook_z is [batch, pos, n_heads, d_head]; their SparseAutoEncoder acts on the last dimension, so the
subclass only flattens the head axes around the parent's forward (encode, hook_feature_acts and decode are theirs).
"""
from __future__ import annotations

import glob
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Dict

import torch

from lm_saes.config import SAEConfig
from lm_saes.sae import SparseAutoEncoder

HF = Path("/home/daniel/.cache/huggingface/hub")
TC_SNAP = HF / "models--pchlenski--gpt2-transcoders/snapshots/4f3ca345572ce2d75018a60d10b4dca8bd899e99"
TC_FILE = "final_sparse_autoencoder_gpt2-small_blocks.%d.ln2.hook_normalized_24576.pt"
Z_SNAP = HF / "models--ckkissane--attn-saes-gpt2-small-all-layers/snapshots/97e0da0e9182695e9813d15ea32184da83f5c874"
N_LAYER, D_MODEL = 12, 768


# ------------------------------------------------------------------------------------------- raw loaders (CPU)
class _Any:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        self.__dict__.update(st if isinstance(st, dict) else {})


def _stub_sae_training():
    """the transcoder pickles reference sae_training.config.LanguageModelSAERunnerConfig; ge107 does not carry that
    package (it belongs to tc107), so stand-in modules let the pickle resolve (as 035's convert_transcoders.py)."""
    for n in ["sae_training", "sae_training.sparse_autoencoder", "sae_training.config", "sae_training.utils",
              "sae_training.geometric_median"]:
        if n not in sys.modules:
            m = types.ModuleType(n)
            m.__getattr__ = lambda name, _m=m: _Any
            sys.modules[n] = m


def load_tc_raw(layer: int) -> Dict[str, torch.Tensor]:
    _stub_sae_training()
    blob = torch.load(TC_SNAP / (TC_FILE % layer), map_location="cpu", weights_only=False)
    cfg = blob["cfg"].__dict__
    assert cfg["is_transcoder"] and cfg["hook_point"] == "blocks.%d.ln2.hook_normalized" % layer
    assert cfg["out_hook_point"] == "blocks.%d.hook_mlp_out" % layer
    sd = {k: v.float() for k, v in blob["state_dict"].items()}
    return dict(W_enc=sd["W_enc"], b_enc=sd["b_enc"], W_dec=sd["W_dec"], b_dec=sd["b_dec"], b_dec_out=sd["b_dec_out"])


def z_path(layer: int) -> Path:
    hits = glob.glob(str(Z_SNAP / ("gpt2-small_L%d_Hcat_z_*.pt" % layer)))
    assert len(hits) == 1, hits
    return Path(hits[0])


def load_z_raw(layer: int) -> Dict[str, torch.Tensor]:
    sd = torch.load(z_path(layer), map_location="cpu", weights_only=True)   # raw state dict, saved from cuda:0
    assert set(sd) == {"W_enc", "W_dec", "b_enc", "b_dec"}, set(sd)
    return {k: v.float() for k, v in sd.items()}


# ------------------------------------------------------------------------------------------- into lm_saes
class ZSparseAutoEncoder(SparseAutoEncoder):
    """[ours: shape glue] flattens [.., n_heads, d_head] -> [.., d_model] around the parent's forward."""

    def forward(self, x, label=None):
        if x.dim() >= 2 and x.shape[-1] != self.cfg.d_model:
            shape = x.shape
            y = super().forward(x.flatten(-2), None if label is None else label.flatten(-2))
            return y.view(shape)
        return super().forward(x, label)


def _build(cls, cfg: SAEConfig, W_enc, b_enc, W_dec, b_dec) -> SparseAutoEncoder:
    sae = cls(cfg)
    sd = {"encoder.weight": W_enc.T.contiguous(), "encoder.bias": b_enc.contiguous(),
          "decoder.weight": W_dec.T.contiguous(), "decoder.bias": b_dec.contiguous(),
          "feature_act_mask": torch.ones(cfg.d_sae), "feature_act_scale": torch.ones(cfg.d_sae)}
    sae.load_state_dict(sd, strict=True)
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae.eval()


def tc_sae(layer: int, device: str = "cpu") -> SparseAutoEncoder:
    r = load_tc_raw(layer)
    d_sae = r["W_enc"].shape[1]
    cfg = SAEConfig(hook_point_in="blocks.%d.ln2.hook_normalized" % layer, hook_point_out="blocks.%d.hook_mlp_out" % layer,
                    d_model=D_MODEL, d_sae=d_sae, use_decoder_bias=True, apply_decoder_bias_to_pre_encoder=False,
                    norm_activation="none", device=device, dtype=torch.float32)
    b_enc = r["b_enc"] - r["b_dec"] @ r["W_enc"]                       # fold the input centring (exact)
    return _build(SparseAutoEncoder, cfg, r["W_enc"], b_enc, r["W_dec"], r["b_dec_out"])


def z_sae(layer: int, device: str = "cpu") -> SparseAutoEncoder:
    r = load_z_raw(layer)
    d_sae = r["W_enc"].shape[1]
    cfg = SAEConfig(hook_point_in="blocks.%d.attn.hook_z" % layer, d_model=D_MODEL, d_sae=d_sae, use_decoder_bias=True,
                    apply_decoder_bias_to_pre_encoder=True, norm_activation="none", device=device, dtype=torch.float32)
    return _build(ZSparseAutoEncoder, cfg, r["W_enc"], r["b_enc"], r["W_dec"], r["b_dec"])


def all_saes(device: str = "cpu", kinds=("z", "tc")):
    out = []
    for l in range(N_LAYER):
        if "z" in kinds:
            out.append(z_sae(l, device))
        if "tc" in kinds:
            out.append(tc_sae(l, device))
    return out


def feature_node_name(sae: SparseAutoEncoder) -> str:
    """the hook point Ge's code reads a dictionary's features at (transformer.py:40,73)"""
    return "%s.sae.hook_feature_acts" % sae.cfg.hook_point_out


@lru_cache(maxsize=None)
def source_forward_ref(kind: str, layer: int):
    """the source formula of each dictionary, for the parity tests"""
    r = load_tc_raw(layer) if kind == "tc" else load_z_raw(layer)

    def f(x):
        acts = torch.relu((x - r["b_dec"]) @ r["W_enc"] + r["b_enc"])
        out = acts @ r["W_dec"] + (r["b_dec_out"] if kind == "tc" else r["b_dec"])
        return acts, out
    return f
