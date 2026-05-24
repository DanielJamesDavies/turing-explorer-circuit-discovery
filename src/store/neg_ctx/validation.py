"""Validation helpers for negative-context artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from store.context import Context


def validate_neg_ctx_output(
    neg_ctx: "Context",
    *,
    total_n_seqs: int,
    n_sequences: int,
) -> None:
    """Validate populated neg_ctx rows after exact retrieval."""

    if neg_ctx.ctx_seq_idx.ndim != 3 or neg_ctx.ctx_seq_val.ndim != 3:
        raise ValueError("neg_ctx tensors must be rank-3")
    if neg_ctx.ctx_seq_idx.shape != neg_ctx.ctx_seq_val.shape:
        raise ValueError("neg_ctx tensor shape mismatch")
    if neg_ctx.ctx_seq_idx.shape[2] > n_sequences:
        raise ValueError("neg_ctx rows exceed configured n_sequences")
    if (neg_ctx.ctx_seq_idx < 0).any():
        raise ValueError("neg_ctx sequence IDs must be non-negative")
    if not torch.isfinite(neg_ctx.ctx_seq_val.float()).all():
        raise ValueError("neg_ctx similarities must be finite")
    if (neg_ctx.ctx_seq_val < 0).any():
        raise ValueError("neg_ctx similarities must be non-negative")

    populated = neg_ctx.ctx_seq_idx > 0
    if bool((neg_ctx.ctx_seq_idx[populated] > total_n_seqs).any().item()):
        raise ValueError("neg_ctx sequence ID exceeds seq_repr n_seqs")
    if bool(((neg_ctx.ctx_seq_val > 0) & ~populated).any().item()):
        raise ValueError("neg_ctx positive similarities must have sequence IDs")


__all__ = ["validate_neg_ctx_output"]
