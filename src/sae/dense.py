"""Utilities for expanding sparse SAE top-k activations."""

from __future__ import annotations

import torch


def sparse_topk_to_dense(
    top_acts: torch.Tensor,
    top_indices: torch.Tensor,
    d_sae: int,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Expand sparse top-k SAE activations to a dense latent tensor.

    Some top-k encoders use index ``0`` as padding for inactive slots. If latent
    0 is genuinely active, plain ``scatter_`` can let later padded zeros
    overwrite the real value. Max reduction preserves the active value while
    retaining the usual zero-fill behavior for absent latents.
    """

    target_dtype = dtype or top_acts.dtype
    dense = torch.zeros(*top_acts.shape[:-1], int(d_sae), device=top_acts.device, dtype=target_dtype)
    # scatter_add_ instead of scatter_reduce_(amax): for top-k output the
    # real indices are unique and the values are post-ReLU (>= 0), while any
    # index-0 padding carries value 0 — so add == amax exactly. The amax
    # backward (ScatterReduceBackward + an eq mask pass) was 17% of fit
    # device time (H100 profile 2026-09-05); add's backward is a gather.
    dense.scatter_add_(
        dim=-1,
        index=top_indices.long(),
        src=top_acts.to(target_dtype),
    )
    return dense


def target_latent_activations(
    top_acts: torch.Tensor,
    top_indices: torch.Tensor,
    latent_idx: int,
) -> torch.Tensor:
    """Return dense activation values for one latent over the top-k axis."""

    is_target = top_indices == int(latent_idx)
    return torch.where(is_target, top_acts, torch.zeros_like(top_acts)).amax(dim=-1)
