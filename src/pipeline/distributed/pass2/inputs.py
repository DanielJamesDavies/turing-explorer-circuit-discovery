"""Reducer input loading and global mapping helpers for distributed pass 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from tqdm import tqdm

from ..pass2_partials import (
    CandidateDumpMetadata,
    CandidatePreAggregationMetadata,
    load_candidate_dump_partial,
    load_candidate_preaggregation_partial,
    validate_candidate_dump_partial,
    validate_candidate_preaggregation_partial,
)
from ..pass2_replay import build_pass2_replay_list
from .contracts import (
    CandidateDumpReducerEntry,
    CandidateDumpReducerInputs,
    CandidatePreAggregationReducerEntry,
    CandidatePreAggregationReducerInputs,
    GlobalTopCtxTargetMapping,
)


def load_candidate_dump_reducer_inputs(
    paths: Sequence[str | Path],
    *,
    expected_config_hash: Optional[str] = None,
    expected_mode: Optional[str] = None,
) -> CandidateDumpReducerInputs:
    """Load and validate simple exact worker candidate dumps for reduction."""

    print(f"[pass2_reduce] loading {len(paths)} candidate dump partials", flush=True)
    entries = []
    for path in tqdm(paths, desc="  [pass2_reduce:load_dumps]", unit="dump"):
        entries.append(
            CandidateDumpReducerEntry(
                *load_candidate_dump_partial(path, expected_config_hash=expected_config_hash)
            )
        )
    return validate_candidate_dump_reducer_inputs(
        entries,
        expected_config_hash=expected_config_hash,
        expected_mode=expected_mode,
    )


def validate_candidate_dump_reducer_inputs(
    entries: Sequence[CandidateDumpReducerEntry | tuple[CandidateDumpMetadata, Dict[str, Any]]],
    *,
    expected_config_hash: Optional[str] = None,
    expected_mode: Optional[str] = None,
) -> CandidateDumpReducerInputs:
    """Validate cross-worker simple dump agreement before global reduce."""

    normalized = _normalize_candidate_dump_entries(entries, expected_config_hash=expected_config_hash)
    if not normalized:
        raise ValueError("at least one candidate dump partial is required")

    first = normalized[0].metadata
    if expected_mode is not None and first.mode != expected_mode:
        raise ValueError("candidate dump mode mismatch")
    worker_ids = set()
    for entry in normalized:
        metadata = entry.metadata
        if metadata.worker_id in worker_ids:
            raise ValueError("duplicate candidate dump worker_id")
        worker_ids.add(metadata.worker_id)
        if expected_mode is not None and metadata.mode != expected_mode:
            raise ValueError("candidate dump mode mismatch")
        if metadata.mode != first.mode:
            raise ValueError("candidate dump mode mismatch")
        if metadata.num_components != first.num_components:
            raise ValueError("candidate dump num_components mismatch")
        if metadata.d_sae != first.d_sae:
            raise ValueError("candidate dump d_sae mismatch")
        if metadata.n_latents_per_latent != first.n_latents_per_latent:
            raise ValueError("candidate dump n_latents_per_latent mismatch")
        if metadata.n_candidates_per_component != first.n_candidates_per_component:
            raise ValueError("candidate dump n_candidates_per_component mismatch")
        if metadata.m != first.m:
            raise ValueError("candidate dump M mismatch")

    ordered = tuple(sorted(normalized, key=lambda entry: entry.metadata.worker_id))
    return CandidateDumpReducerInputs(
        entries=ordered,
        mode=first.mode,
        m=first.m,
        n_candidates_per_component=first.n_candidates_per_component,
        n_latents_per_latent=first.n_latents_per_latent,
        num_components=first.num_components,
        d_sae=first.d_sae,
        total_sequence_count=sum(entry.metadata.sequence_count for entry in ordered),
        total_token_count=sum(entry.metadata.token_count for entry in ordered),
    )


def load_candidate_preaggregation_reducer_inputs(
    paths: Sequence[str | Path],
    *,
    expected_config_hash: Optional[str] = None,
    expected_mode: Optional[str] = None,
    expected_target_start_id: Optional[int] = None,
    expected_target_end_id: Optional[int] = None,
) -> CandidatePreAggregationReducerInputs:
    """Load and validate worker preaggregation shards for one reducer range."""

    entries = [
        CandidatePreAggregationReducerEntry(
            *load_candidate_preaggregation_partial(path, expected_config_hash=expected_config_hash)
        )
        for path in paths
    ]
    return validate_candidate_preaggregation_reducer_inputs(
        entries,
        expected_config_hash=expected_config_hash,
        expected_mode=expected_mode,
        expected_target_start_id=expected_target_start_id,
        expected_target_end_id=expected_target_end_id,
    )


def load_global_top_ctx_target_mapping(
    path: str | Path,
    *,
    dump_inputs: Optional[CandidateDumpReducerInputs] = None,
) -> GlobalTopCtxTargetMapping:
    """Load merged global top_ctx.pt and build reducer CSR plus dump row mapping."""

    print(f"[pass2_reduce] loading top_ctx mapping input -> {path}", flush=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return build_global_top_ctx_target_mapping(payload, dump_inputs=dump_inputs)


def load_global_active_count(
    path: str | Path,
    *,
    expected_config_hash: Optional[str] = None,
    expected_num_components: Optional[int] = None,
    expected_d_sae: Optional[int] = None,
) -> torch.Tensor:
    """Load merged global latent_stats.active_count for PMI postprocess."""

    print(f"[pass2_reduce] loading active_count input -> {path}", flush=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("latent_stats payload must be a mapping")
    if expected_config_hash is not None:
        payload_hash = payload.get("config_hash")
        metadata = payload.get("metadata")
        metadata_hash = metadata.get("config_hash") if isinstance(metadata, Mapping) else None
        if payload_hash != expected_config_hash and metadata_hash != expected_config_hash:
            raise ValueError("latent_stats config hash mismatch")
    active_count = payload.get("active_count")
    if not isinstance(active_count, torch.Tensor):
        raise ValueError("latent_stats payload missing tensor: active_count")
    return validate_global_active_count(
        active_count,
        expected_num_components=expected_num_components,
        expected_d_sae=expected_d_sae,
    )


def validate_global_active_count(
    active_count: torch.Tensor,
    *,
    expected_num_components: Optional[int] = None,
    expected_d_sae: Optional[int] = None,
) -> torch.Tensor:
    """Validate global pass-1 active_count used for PMI candidate firing rates."""

    if active_count.ndim != 2:
        raise ValueError("active_count must have shape [num_components, d_sae]")
    if expected_num_components is not None and int(active_count.shape[0]) != expected_num_components:
        raise ValueError("active_count num_components mismatch")
    if expected_d_sae is not None and int(active_count.shape[1]) != expected_d_sae:
        raise ValueError("active_count d_sae mismatch")
    if not torch.isfinite(active_count.float()).all():
        raise ValueError("active_count must be finite")
    if (active_count < 0).any():
        raise ValueError("active_count must be non-negative")
    if active_count.numel() == 0 or int(active_count[0].sum().item()) <= 0:
        raise ValueError("active_count must contain positive global counts")
    return active_count.detach().cpu()


def build_global_top_ctx_target_mapping(
    top_ctx_payload: Mapping[str, object],
    *,
    dump_inputs: Optional[CandidateDumpReducerInputs] = None,
) -> GlobalTopCtxTargetMapping:
    """Build global sequence-to-target CSR from merged top_ctx, not worker state."""

    ctx_seq_idx = _required_top_ctx_tensor(top_ctx_payload, "ctx_seq_idx")
    ctx_seq_val = _required_top_ctx_tensor(top_ctx_payload, "ctx_seq_val")
    if ctx_seq_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("top_ctx ctx_seq_idx must be int32 or int64")
    if not torch.is_floating_point(ctx_seq_val):
        raise ValueError("top_ctx ctx_seq_val must be floating point")
    if ctx_seq_idx.shape != ctx_seq_val.shape:
        raise ValueError("top_ctx ctx_seq_idx and ctx_seq_val shapes must match")
    if ctx_seq_idx.ndim != 3:
        raise ValueError("top_ctx tensors must have shape [components, d_sae, contexts]")

    replay = build_pass2_replay_list(top_ctx_payload)
    seq_offsets, seq_targets_global = _build_sequence_to_targets_csr(ctx_seq_idx, ctx_seq_val)
    sequence_ids = tuple(replay.sequence_ids)
    sid_to_row = {int(sequence_id): row for row, sequence_id in enumerate(sequence_ids)}
    sid_to_row_tensor = _build_sid_to_row_tensor(sequence_ids)
    mapping = GlobalTopCtxTargetMapping(
        replay=replay,
        seq_offsets=seq_offsets,
        seq_targets_global=seq_targets_global,
        sequence_ids=sequence_ids,
        sid_to_row=sid_to_row,
        sid_to_row_tensor=sid_to_row_tensor,
    )
    if dump_inputs is not None:
        validate_candidate_dump_sequence_coverage(dump_inputs, mapping)
    return mapping


def validate_candidate_dump_sequence_coverage(
    dump_inputs: CandidateDumpReducerInputs,
    mapping: GlobalTopCtxTargetMapping,
) -> None:
    """Ensure simple exact dumps cover every global replay sequence exactly once."""

    expected_count = len(mapping.sequence_ids)
    seen_rows = torch.zeros(expected_count, dtype=torch.bool)
    sid_to_row = mapping.sid_to_row_tensor.to(torch.int64)
    max_valid_sid = int(sid_to_row.numel()) - 1
    seen_count = 0
    for entry in tqdm(
        dump_inputs.entries,
        desc="  [pass2_reduce:validate_coverage]",
        unit="dump",
    ):
        sequence_ids = entry.payload["sequence_ids"].to(torch.int64).cpu()
        if bool((sequence_ids == 0).any()):
            raise ValueError("candidate dump contains sentinel sequence ID 0")
        if sequence_ids.numel() and (
            int(sequence_ids.min().item()) < 0 or int(sequence_ids.max().item()) > max_valid_sid
        ):
            raise ValueError("candidate dump contains sequence ID outside global replay set")
        rows = sid_to_row[sequence_ids]
        if bool((rows < 0).any()):
            bad_sequence_id = int(sequence_ids[rows < 0][0].item())
            raise ValueError(f"candidate dump contains sequence ID outside global replay set: {bad_sequence_id}")
        if bool(seen_rows[rows].any()):
            duplicate_sequence_id = int(sequence_ids[seen_rows[rows]][0].item())
            raise ValueError(f"candidate dumps contain duplicate sequence IDs: [{duplicate_sequence_id}]")
        seen_rows[rows] = True
        seen_count += int(sequence_ids.numel())
    if seen_count != expected_count or not bool(seen_rows.all()):
        missing_row = int((~seen_rows).nonzero(as_tuple=False)[0].item())
        missing_sequence_id = int(mapping.sequence_ids[missing_row])
        raise ValueError(f"candidate dumps missing replay sequence IDs: [{missing_sequence_id}]")


def validate_candidate_preaggregation_reducer_inputs(
    entries: Sequence[
        CandidatePreAggregationReducerEntry | tuple[CandidatePreAggregationMetadata, Dict[str, Any]]
    ],
    *,
    expected_config_hash: Optional[str] = None,
    expected_mode: Optional[str] = None,
    expected_target_start_id: Optional[int] = None,
    expected_target_end_id: Optional[int] = None,
    expected_worker_ids: Optional[Sequence[int]] = None,
) -> CandidatePreAggregationReducerInputs:
    """Validate cross-worker MapReduce partial-sum agreement for one target range."""

    normalized = _normalize_preaggregation_entries(entries, expected_config_hash=expected_config_hash)
    if not normalized:
        raise ValueError("at least one candidate preaggregation partial is required")

    first = normalized[0].metadata
    if expected_mode is not None and first.mode != expected_mode:
        raise ValueError("candidate preaggregation mode mismatch")
    if expected_target_start_id is not None and first.target_start_id != expected_target_start_id:
        raise ValueError("candidate preaggregation target range mismatch")
    if expected_target_end_id is not None and first.target_end_id != expected_target_end_id:
        raise ValueError("candidate preaggregation target range mismatch")

    worker_ids = set()
    for entry in normalized:
        metadata = entry.metadata
        if metadata.worker_id in worker_ids:
            raise ValueError("duplicate candidate preaggregation worker_id")
        worker_ids.add(metadata.worker_id)
        if expected_mode is not None and metadata.mode != expected_mode:
            raise ValueError("candidate preaggregation mode mismatch")
        if metadata.mode != first.mode:
            raise ValueError("candidate preaggregation mode mismatch")
        if metadata.num_components != first.num_components:
            raise ValueError("candidate preaggregation num_components mismatch")
        if metadata.d_sae != first.d_sae:
            raise ValueError("candidate preaggregation d_sae mismatch")
        if metadata.m != first.m:
            raise ValueError("candidate preaggregation M mismatch")
        if metadata.target_start_id != first.target_start_id or metadata.target_end_id != first.target_end_id:
            raise ValueError("candidate preaggregation target range mismatch")
    if expected_worker_ids is not None and worker_ids != {int(worker_id) for worker_id in expected_worker_ids}:
        raise ValueError("candidate preaggregation worker coverage mismatch")

    ordered = tuple(sorted(normalized, key=lambda entry: entry.metadata.worker_id))
    return CandidatePreAggregationReducerInputs(
        entries=ordered,
        mode=first.mode,
        m=first.m,
        num_components=first.num_components,
        d_sae=first.d_sae,
        target_start_id=first.target_start_id,
        target_end_id=int(first.target_end_id),
        total_sequence_count=sum(entry.metadata.sequence_count for entry in ordered),
        total_contribution_count=sum(entry.metadata.contribution_count for entry in ordered),
    )


def _normalize_candidate_dump_entries(
    entries: Sequence[CandidateDumpReducerEntry | tuple[CandidateDumpMetadata, Dict[str, Any]]],
    *,
    expected_config_hash: Optional[str],
) -> list[CandidateDumpReducerEntry]:
    normalized: list[CandidateDumpReducerEntry] = []
    for entry in entries:
        if isinstance(entry, CandidateDumpReducerEntry):
            metadata, payload = entry.metadata, entry.payload
        else:
            metadata, payload = entry
        metadata, payload = validate_candidate_dump_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_config_hash=expected_config_hash,
        )
        normalized.append(CandidateDumpReducerEntry(metadata, payload))
    return normalized


def _normalize_preaggregation_entries(
    entries: Sequence[
        CandidatePreAggregationReducerEntry | tuple[CandidatePreAggregationMetadata, Dict[str, Any]]
    ],
    *,
    expected_config_hash: Optional[str],
) -> list[CandidatePreAggregationReducerEntry]:
    normalized: list[CandidatePreAggregationReducerEntry] = []
    for entry in entries:
        if isinstance(entry, CandidatePreAggregationReducerEntry):
            metadata, payload = entry.metadata, entry.payload
        else:
            metadata, payload = entry
        metadata, payload = validate_candidate_preaggregation_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_config_hash=expected_config_hash,
        )
        normalized.append(CandidatePreAggregationReducerEntry(metadata, payload))
    return normalized


def _required_top_ctx_tensor(top_ctx_payload: Mapping[str, object], key: str) -> torch.Tensor:
    value = top_ctx_payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"top_ctx payload missing tensor: {key}")
    return value.cpu()


def _build_sequence_to_targets_csr(
    ctx_seq_idx: torch.Tensor,
    ctx_seq_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = (ctx_seq_idx != 0) & (ctx_seq_val > 0)
    if not torch.any(valid_mask):
        return torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)

    sequence_ids = ctx_seq_idx[valid_mask].to(torch.int64)
    component_ids, latent_ids = torch.nonzero(valid_mask, as_tuple=True)[:2]
    d_sae = int(ctx_seq_idx.shape[1])
    global_target_ids = component_ids.to(torch.int64) * d_sae + latent_ids.to(torch.int64)

    order = torch.argsort(sequence_ids)
    sorted_sequence_ids = sequence_ids[order]
    sorted_target_ids = global_target_ids[order]
    max_sequence_id = int(sorted_sequence_ids[-1].item())
    counts = torch.bincount(sorted_sequence_ids, minlength=max_sequence_id + 1)
    seq_offsets = torch.cumsum(counts, dim=0).to(torch.int64)
    return seq_offsets.cpu(), sorted_target_ids.cpu()


def _build_sid_to_row_tensor(sequence_ids: tuple[int, ...]) -> torch.Tensor:
    if not sequence_ids:
        return torch.empty(0, dtype=torch.int64)
    max_sequence_id = max(sequence_ids)
    sid_to_row = torch.full((max_sequence_id + 1,), -1, dtype=torch.int64)
    sid_to_row[torch.tensor(sequence_ids, dtype=torch.int64)] = torch.arange(
        len(sequence_ids),
        dtype=torch.int64,
    )
    return sid_to_row


__all__ = [
    "build_global_top_ctx_target_mapping",
    "load_candidate_dump_reducer_inputs",
    "load_candidate_preaggregation_reducer_inputs",
    "load_global_active_count",
    "load_global_top_ctx_target_mapping",
    "validate_candidate_dump_reducer_inputs",
    "validate_candidate_dump_sequence_coverage",
    "validate_candidate_preaggregation_reducer_inputs",
    "validate_global_active_count",
]
