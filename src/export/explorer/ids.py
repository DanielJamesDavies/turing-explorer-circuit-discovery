"""Latent and circuit identifiers shared by every converter stage.

gid = (layer * N_KINDS + kind_idx) * D_SAE + latent, with kinds in SAEBank order.
The same formula lives in the explorer backend (services/v2/identifiers.py).
"""
from __future__ import annotations

from typing import Tuple

KINDS = ("attn", "mlp", "resid")  # must match sae.bank.SAEBank.kinds
N_KINDS = len(KINDS)
N_LAYERS = 12
D_SAE = 40960
N_COMPONENTS = N_LAYERS * N_KINDS
N_LATENTS = N_COMPONENTS * D_SAE


def comp_of(layer: int, kind: str) -> int:
    return layer * N_KINDS + KINDS.index(kind)


def gid_of(layer: int, kind: str, latent: int) -> int:
    return comp_of(layer, kind) * D_SAE + latent


def split_gid(gid: int) -> Tuple[int, str, int]:
    comp, latent = divmod(int(gid), D_SAE)
    layer, kind_idx = divmod(comp, N_KINDS)
    return layer, KINDS[kind_idx], latent


def parse_key(key: str) -> Tuple[int, str, int]:
    """'5.mlp.2277' -> (5, 'mlp', 2277)."""
    layer, kind, latent = key.split(".")
    if kind not in KINDS:
        raise ValueError(f"unknown kind in key {key!r}")
    return int(layer), kind, int(latent)


def key_of(layer: int, kind: str, latent: int) -> str:
    return f"{layer}.{kind}.{latent}"


def circuit_key(run: str, arm: str, target_key: str) -> str:
    return f"{run}/{arm}/{target_key}"
