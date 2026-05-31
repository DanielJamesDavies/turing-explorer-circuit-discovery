"""Logit-context merge helpers for distributed pass 1."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, Sequence

import torch

from ..pass1_partials import load_pass1_partial, validate_pass1_partial
from .contracts import LogitCtxPartial


_LOGIT_CTX_CHUNK_ROWS = 8192


def load_and_merge_logit_ctx_partials(
    partial_paths: Sequence[str | Path],
    *,
    expected_config_hash: str | None = None,
    vocab_size: int | None = None,
) -> Dict[str, object]:
    """Load logit-context partial files and merge event top-K rows."""

    load_start = time.perf_counter()
    partials = []
    print(f"[pass1_merge] loading {len(partial_paths)} logit_ctx partials", flush=True)
    for index, path in enumerate(partial_paths, start=1):
        partial_start = time.perf_counter()
        print(
            f"[pass1_merge] loading logit_ctx partial {index}/{len(partial_paths)} -> {path}",
            flush=True,
        )
        partials.append(
            load_pass1_partial(
                path,
                expected_artifact_name="logit_ctx",
                expected_config_hash=expected_config_hash,
            )
        )
        print(
            f"[pass1_merge] loaded logit_ctx partial {index}/{len(partial_paths)} "
            f"elapsed={time.perf_counter() - partial_start:.1f}s",
            flush=True,
        )
    print(
        f"[pass1_merge] loaded all logit_ctx partials elapsed={time.perf_counter() - load_start:.1f}s",
        flush=True,
    )
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

    merge_start = time.perf_counter()
    _select_logit_ctx_events_chunked(
        top_tokens,
        top_probs,
        partials,
        output_k,
        vocab_size=vocab_size,
        merge_start=merge_start,
    )
    print(
        f"[pass1_merge] logit_ctx event top-k merge elapsed={time.perf_counter() - merge_start:.1f}s",
        flush=True,
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


def _select_logit_ctx_events_reference(
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


def _select_logit_ctx_events_chunked(
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    partials: Sequence[LogitCtxPartial],
    output_k: int,
    *,
    vocab_size: int | None = None,
    merge_start: float | None = None,
    chunk_rows: int = _LOGIT_CTX_CHUNK_ROWS,
) -> None:
    sorted_partials = sorted(partials, key=lambda item: item[0].worker_id)
    component_count, d_sae, input_k = sorted_partials[0][1]["top_tokens"].shape
    total_rows = component_count * d_sae
    candidate_count = len(sorted_partials) * input_k
    selected_k = min(output_k, candidate_count)
    worker_ids = torch.tensor(
        [int(metadata.worker_id) for metadata, _payload in sorted_partials],
        dtype=torch.int64,
    ).view(1, len(sorted_partials), 1)
    candidate_rows = torch.arange(input_k, dtype=torch.int64).view(1, 1, input_k)
    progress_start = merge_start if merge_start is not None else time.perf_counter()
    progress_interval = max(chunk_rows, max(1, total_rows // 20))
    next_progress = 0

    print(
        "[pass1_merge] merging logit_ctx event top-k "
        f"workers={len(sorted_partials)} components={component_count} d_sae={d_sae} "
        f"rows={total_rows} output_k={output_k}",
        flush=True,
    )

    for row_start in range(0, total_rows, chunk_rows):
        row_end = min(total_rows, row_start + chunk_rows)
        row_count = row_end - row_start
        chunk_start = time.perf_counter()
        token_chunk = torch.stack(
            [
                payload["top_tokens"].reshape(total_rows, input_k)[row_start:row_end].to(torch.int64)
                for _metadata, payload in sorted_partials
            ],
            dim=1,
        )
        prob_chunk = torch.stack(
            [
                payload["top_probs"].reshape(total_rows, input_k)[row_start:row_end].to(torch.float32)
                for _metadata, payload in sorted_partials
            ],
            dim=1,
        )
        worker_chunk = worker_ids.expand(row_count, -1, input_k)
        row_chunk = candidate_rows.expand(row_count, len(sorted_partials), input_k)
        flat_tokens = token_chunk.reshape(row_count, candidate_count)
        flat_probs = prob_chunk.reshape(row_count, candidate_count)
        flat_workers = worker_chunk.reshape(row_count, candidate_count)
        flat_rows = row_chunk.reshape(row_count, candidate_count)

        valid = flat_probs > 0
        if vocab_size is not None:
            valid &= flat_tokens < int(vocab_size)
        flat_probs = flat_probs.masked_fill(~valid, float("-inf"))
        flat_tokens = flat_tokens.masked_fill(~valid, 0)

        order = torch.argsort(flat_rows, dim=1, stable=True)
        flat_tokens = flat_tokens.gather(1, order)
        flat_workers = flat_workers.gather(1, order)
        flat_rows = flat_rows.gather(1, order)
        flat_probs = flat_probs.gather(1, order)

        order = torch.argsort(flat_workers, dim=1, stable=True)
        flat_tokens = flat_tokens.gather(1, order)
        flat_rows = flat_rows.gather(1, order)
        flat_probs = flat_probs.gather(1, order)

        order = torch.argsort(flat_tokens, dim=1, stable=True)
        flat_tokens = flat_tokens.gather(1, order)
        flat_probs = flat_probs.gather(1, order)

        order = torch.argsort(flat_probs, dim=1, descending=True, stable=True)
        selected = order[:, :selected_k]
        selected_tokens = flat_tokens.gather(1, selected).to(torch.int32)
        selected_probs = flat_probs.gather(1, selected).to(torch.float32)
        selected_valid = torch.isfinite(selected_probs)
        selected_tokens = selected_tokens.masked_fill(~selected_valid, 0)
        selected_probs = selected_probs.masked_fill(~selected_valid, 0.0)

        out_tokens = torch.zeros((row_count, output_k), dtype=torch.int32)
        out_probs = torch.zeros((row_count, output_k), dtype=torch.float32)
        out_tokens[:, :selected_k] = selected_tokens
        out_probs[:, :selected_k] = selected_probs
        top_tokens.reshape(total_rows, output_k)[row_start:row_end] = out_tokens
        top_probs.reshape(total_rows, output_k)[row_start:row_end] = out_probs

        rows_done = row_end
        if rows_done >= next_progress or rows_done == total_rows:
            elapsed_s = time.perf_counter() - progress_start
            rows_per_s = rows_done / max(elapsed_s, 1e-9)
            remaining_s = (total_rows - rows_done) / max(rows_per_s, 1e-9)
            print(
                "[pass1_merge] logit_ctx chunk progress "
                f"{rows_done}/{total_rows} rows "
                f"({rows_done / total_rows:.1%}) "
                f"chunk_elapsed={time.perf_counter() - chunk_start:.1f}s "
                f"elapsed={elapsed_s:.1f}s eta={remaining_s:.1f}s",
                flush=True,
            )
            next_progress = rows_done + progress_interval


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
