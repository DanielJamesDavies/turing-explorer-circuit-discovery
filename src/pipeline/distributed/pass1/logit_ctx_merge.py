"""Logit-context merge helpers for distributed pass 1."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch

from ..pass1_partials import load_pass1_partial, validate_pass1_partial
from .contracts import LogitCtxPartial


def load_and_merge_logit_ctx_partials(
    partial_paths: Sequence[str | Path],
    *,
    expected_config_hash: str | None = None,
    vocab_size: int | None = None,
) -> Dict[str, object]:
    """Load logit-context partial files and merge event top-K rows."""

    partials = [
        load_pass1_partial(
            path,
            expected_artifact_name="logit_ctx",
            expected_config_hash=expected_config_hash,
        )
        for path in partial_paths
    ]
    return merge_logit_ctx_partials(partials, vocab_size=vocab_size)


def merge_logit_ctx_partials(
    partials: Sequence[LogitCtxPartial],
    *,
    vocab_size: int | None = None,
) -> Dict[str, object]:
    """Merge logit-context partials with exact event top-K semantics."""

    if not partials:
        raise ValueError("at least one logit_ctx partial is required")
    _validate_logit_ctx_partial_set(partials, vocab_size=vocab_size)

    first_payload = partials[0][1]
    latent_counts = sum(
        (payload["latent_counts"].to(torch.int64) for _metadata, payload in partials),
        start=torch.zeros_like(first_payload["latent_counts"], dtype=torch.int64),
    )
    output_k = int(first_payload["top_tokens"].shape[2])
    top_tokens = torch.zeros_like(first_payload["top_tokens"], dtype=torch.int32)
    top_probs = torch.zeros_like(first_payload["top_probs"], dtype=torch.float32)

    candidate_tokens = torch.cat(
        [payload["top_tokens"].to(torch.int64) for _metadata, payload in partials],
        dim=2,
    )
    candidate_probs = torch.cat(
        [payload["top_probs"].to(torch.float32) for _metadata, payload in partials],
        dim=2,
    )
    candidate_workers = torch.cat(
        [
            torch.full_like(payload["top_tokens"].to(torch.int64), metadata.worker_id)
            for metadata, payload in partials
        ],
        dim=2,
    )
    candidate_rows = torch.cat(
        [
            torch.arange(payload["top_tokens"].shape[2], dtype=torch.int64)
            .view(1, 1, -1)
            .expand_as(payload["top_tokens"])
            for _metadata, payload in partials
        ],
        dim=2,
    )

    valid = candidate_probs > 0
    if vocab_size is not None:
        valid &= candidate_tokens < int(vocab_size)
    candidate_probs = candidate_probs.masked_fill(~valid, 0.0)
    candidate_tokens = candidate_tokens.masked_fill(~valid, 0)

    _select_logit_ctx_events(
        top_tokens,
        top_probs,
        candidate_tokens,
        candidate_probs,
        candidate_workers,
        candidate_rows,
        output_k,
    )
    merged = {
        "latent_counts": latent_counts,
        "top_tokens": top_tokens,
        "top_probs": top_probs,
        "merge_report": {
            "num_partials": len(partials),
            "event_top_k": output_k,
            "tie_breaking": "probability_desc_token_asc_worker_asc_candidate_row_asc",
        },
    }
    _validate_merged_logit_ctx(merged, partials, vocab_size=vocab_size)
    return merged


def _validate_logit_ctx_partial_set(
    partials: Sequence[LogitCtxPartial],
    *,
    vocab_size: int | None = None,
) -> None:
    seen_workers: set[int] = set()
    first_metadata = partials[0][0]
    first_shape = partials[0][1]["top_tokens"].shape
    for metadata, payload in partials:
        if metadata.artifact_name != "logit_ctx":
            raise ValueError("all partials must be logit_ctx artifacts")
        if metadata.worker_id in seen_workers:
            raise ValueError(f"duplicate logit_ctx partial for worker {metadata.worker_id}")
        seen_workers.add(metadata.worker_id)
        if metadata.run_id != first_metadata.run_id:
            raise ValueError("logit_ctx partial run_id mismatch")
        if metadata.config_hash != first_metadata.config_hash:
            raise ValueError("logit_ctx partial config hash mismatch")
        if metadata.component_count != first_metadata.component_count:
            raise ValueError("logit_ctx partial component count mismatch")
        if metadata.d_sae != first_metadata.d_sae:
            raise ValueError("logit_ctx partial d_sae mismatch")
        if payload["top_tokens"].shape != first_shape:
            raise ValueError("logit_ctx partial top tensor shape mismatch")
        validate_pass1_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_artifact_name="logit_ctx",
            expected_config_hash=first_metadata.config_hash,
        )
        _validate_logit_ctx_token_range(payload, vocab_size=vocab_size)


def _select_logit_ctx_events(
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    candidate_tokens: torch.Tensor,
    candidate_probs: torch.Tensor,
    candidate_workers: torch.Tensor,
    candidate_rows: torch.Tensor,
    output_k: int,
) -> None:
    component_count, d_sae, _candidate_count = candidate_tokens.shape
    for component_id in range(component_count):
        for latent_id in range(d_sae):
            probs = candidate_probs[component_id, latent_id]
            valid = probs > 0
            if not bool(valid.any()):
                continue
            tokens = candidate_tokens[component_id, latent_id][valid]
            workers = candidate_workers[component_id, latent_id][valid]
            rows = candidate_rows[component_id, latent_id][valid]
            probs = probs[valid]

            order = torch.argsort(rows, stable=True)
            tokens = tokens[order]
            workers = workers[order]
            rows = rows[order]
            probs = probs[order]

            order = torch.argsort(workers, stable=True)
            tokens = tokens[order]
            workers = workers[order]
            rows = rows[order]
            probs = probs[order]

            order = torch.argsort(tokens, stable=True)
            tokens = tokens[order]
            probs = probs[order]

            order = torch.argsort(probs, descending=True, stable=True)
            selected = order[:output_k]
            selected_count = int(selected.numel())
            top_tokens[component_id, latent_id, :selected_count] = tokens[selected].to(
                torch.int32
            )
            top_probs[component_id, latent_id, :selected_count] = probs[selected].to(
                torch.float32
            )


def _validate_logit_ctx_token_range(
    payload: Dict[str, object],
    *,
    vocab_size: int | None = None,
) -> None:
    tokens = payload["top_tokens"]
    if tokens.numel() > 0 and int(tokens.min()) < 0:
        raise ValueError("logit_ctx token IDs must be non-negative")
    if vocab_size is not None and tokens.numel() > 0 and int(tokens.max()) >= int(vocab_size):
        raise ValueError("logit_ctx token IDs above vocabulary range")


def _validate_merged_logit_ctx(
    merged: Dict[str, object],
    partials: Sequence[LogitCtxPartial],
    *,
    vocab_size: int | None = None,
) -> None:
    expected_counts = sum(
        (payload["latent_counts"].to(torch.int64) for _metadata, payload in partials),
        start=torch.zeros_like(partials[0][1]["latent_counts"], dtype=torch.int64),
    )
    if not torch.equal(merged["latent_counts"], expected_counts):
        raise ValueError("merged logit_ctx latent_counts does not equal sum of partial counts")
    top_tokens = merged["top_tokens"]
    top_probs = merged["top_probs"]
    if top_tokens.dtype != torch.int32:
        raise ValueError("merged logit_ctx top_tokens must be int32")
    if top_probs.shape != top_tokens.shape or top_probs.ndim != 3:
        raise ValueError("merged logit_ctx top tensors have invalid shape")
    if not torch.isfinite(top_probs.float()).all():
        raise ValueError("merged logit_ctx top_probs contains non-finite values")
    if bool((top_probs < 0).any()):
        raise ValueError("merged logit_ctx top_probs must be non-negative")
    _validate_logit_ctx_token_range(merged, vocab_size=vocab_size)
    invalid = top_probs == 0
    if bool((top_tokens[invalid] != 0).any()):
        raise ValueError("merged logit_ctx invalid sentinel tokens must be zero")


__all__ = [
    "load_and_merge_logit_ctx_partials",
    "merge_logit_ctx_partials",
]
