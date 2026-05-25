"""Experimental two-stage blockwise top-k for SAE activations.

This backend is a correctness-first prototype. It keeps the same public contract
as ``torch.topk(sorted=False)`` while reducing the final top-k problem to a
smaller candidate set: top-k within each feature block, then top-k across the
concatenated block candidates.
"""

from __future__ import annotations

import os

import torch


def _block_size(default: int = 4096) -> int:
    raw = os.environ.get("TURINGLLM_BLOCKWISE_TOPK_BLOCK_N")
    if raw is None:
        return default
    value = int(raw)
    if value < 1:
        raise ValueError("TURINGLLM_BLOCKWISE_TOPK_BLOCK_N must be >= 1")
    return value


def topk_blockwise(
    x: torch.Tensor,
    k: int,
    *,
    block_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k values/indices over the last dimension using two stages."""

    orig_shape = x.shape
    n_features = int(orig_shape[-1])
    rows = x.reshape(-1, n_features)
    block_n = int(block_n or _block_size())

    if block_n >= n_features:
        values, indices = rows.topk(k, dim=-1, sorted=False)
    else:
        value_chunks: list[torch.Tensor] = []
        index_chunks: list[torch.Tensor] = []
        for start in range(0, n_features, block_n):
            end = min(start + block_n, n_features)
            local_k = min(k, end - start)
            local_values, local_indices = rows[:, start:end].topk(
                local_k,
                dim=-1,
                sorted=False,
            )
            value_chunks.append(local_values)
            index_chunks.append(local_indices + start)

        candidates = torch.cat(value_chunks, dim=-1)
        candidate_indices = torch.cat(index_chunks, dim=-1)
        values, positions = candidates.topk(k, dim=-1, sorted=False)
        indices = candidate_indices.gather(-1, positions)

    out_shape = orig_shape[:-1] + (k,)
    return values.reshape(out_shape), indices.reshape(out_shape).long()
