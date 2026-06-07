"""Utilities for compact coactivation-profile comparisons."""

from __future__ import annotations

import torch


def deterministic_sample_indices(total: int, max_samples: int) -> torch.Tensor:
    """Return evenly spaced row indices for deterministic large-tensor sampling."""

    if total <= 0:
        raise ValueError("total must be positive")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    sample_count = min(int(total), int(max_samples))
    return torch.linspace(0, total - 1, steps=sample_count, dtype=torch.float64).round().to(torch.int64).unique()


def build_hashed_coact_profiles(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    sample_indices: torch.Tensor,
    hash_bins: int = 256,
) -> torch.Tensor:
    """
    Build dense hashed fingerprints for target latents from their top coacts.

    Positive PMI values are used as weights; non-positive PMI contributes no
    evidence for similarity. Coacting latent IDs are feature-hashed into a fixed
    number of bins, then each target profile is L2-normalized.
    """

    if top_values.ndim != 3 or top_indices.ndim != 3:
        raise ValueError("top_values and top_indices must have shape [components, d_sae, top_k]")
    if tuple(top_values.shape) != tuple(top_indices.shape):
        raise ValueError("top_values and top_indices shapes must match")
    if hash_bins <= 0:
        raise ValueError("hash_bins must be positive")

    flat_values = top_values.detach().cpu().to(torch.float32).reshape(-1, top_values.shape[-1])
    flat_indices = top_indices.detach().cpu().to(torch.int64).reshape(-1, top_indices.shape[-1])
    rows = sample_indices.detach().cpu().to(torch.int64)
    if rows.numel() == 0:
        raise ValueError("sample_indices must not be empty")
    if int(rows.min().item()) < 0 or int(rows.max().item()) >= flat_values.shape[0]:
        raise ValueError("sample_indices contain rows outside the target latent range")

    sampled_values = flat_values[rows].clamp(min=0.0)
    sampled_bins = flat_indices[rows] % int(hash_bins)
    profiles = torch.zeros((rows.numel(), int(hash_bins)), dtype=torch.float32)
    profiles.scatter_add_(1, sampled_bins, sampled_values)
    return normalize_rows(profiles)


def normalize_rows(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize rows while leaving all-zero rows at zero."""

    return matrix / matrix.norm(dim=1, keepdim=True).clamp(min=eps)

