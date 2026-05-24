"""MapReduce pass-2 shard persistence, checksums, and artifact validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from ..pass2_partials import (
    CandidatePreAggregationMetadata,
    validate_candidate_preaggregation_partial,
)
from .contracts import (
    CandidatePreAggregationReducerEntry,
    CandidatePreAggregationReducerInputs,
    MapReduceShardMemoryEstimate,
    MapReduceTargetShardArtifact,
    MapReduceTargetShardResult,
    TargetRange,
)
from .reports import atomic_torch_save


def sorted_coo_preaggregation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sorted COO payload ordered by (target_id, candidate_id)."""

    target_ids = payload["target_ids"].to(torch.int64).cpu()
    candidate_ids = payload["candidate_ids"].to(torch.int32).cpu()
    values = payload["values"].to(torch.float32).cpu()
    sequence_ids = payload["sequence_ids"].to(torch.int64).cpu()
    order = sorted(
        range(int(target_ids.numel())),
        key=lambda idx: (int(target_ids[idx].item()), int(candidate_ids[idx].item())),
    )
    if not order:
        return {
            "target_ids": target_ids.clone(),
            "candidate_ids": candidate_ids.clone(),
            "values": values.clone(),
            "sequence_ids": sequence_ids.clone(),
        }
    order_tensor = torch.tensor(order, dtype=torch.int64)
    return {
        "target_ids": target_ids[order_tensor].clone(),
        "candidate_ids": candidate_ids[order_tensor].clone(),
        "values": values[order_tensor].clone(),
        "sequence_ids": sequence_ids[order_tensor].clone(),
    }


def build_mapreduce_storage_metadata(
    metadata: CandidatePreAggregationMetadata,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Build inspectable metadata for one sorted COO partial-sum shard."""

    checksum = checksum_coo_payload(payload)
    unique_candidates = (
        int(torch.unique(payload["candidate_ids"]).numel())
        if payload["candidate_ids"].numel()
        else 0
    )
    return {
        "storage_schema_version": 1,
        "format": "sorted_coo",
        "sort_order": ["target_id", "candidate_id"],
        "run_id": metadata.run_id,
        "worker_id": metadata.worker_id,
        "mode": metadata.mode,
        "target_start_id": metadata.target_start_id,
        "target_end_id": metadata.target_end_id,
        "row_count": metadata.contribution_count,
        "candidate_count": unique_candidates,
        "value_dtype": str(payload["values"].dtype),
        "checksum_sha256": checksum,
        "tensor_bytes": estimate_mapreduce_shard_tensor_bytes(metadata.contribution_count),
    }


def save_mapreduce_partial_sum_shard(
    path: str | Path,
    metadata: CandidatePreAggregationMetadata,
    payload: Dict[str, Any],
) -> None:
    """Save one sorted COO MapReduce partial-sum shard atomically."""

    sorted_payload = sorted_coo_preaggregation_payload(payload)
    metadata = metadata.model_copy(update={"contribution_count": int(sorted_payload["target_ids"].numel())})
    metadata, sorted_payload = validate_candidate_preaggregation_partial(
        {"metadata": metadata.model_dump(mode="json"), "payload": sorted_payload}
    )
    storage_metadata = build_mapreduce_storage_metadata(metadata, sorted_payload)
    data = {
        "metadata": metadata.model_dump(mode="json"),
        "storage_metadata": storage_metadata,
        "payload": sorted_payload,
    }
    validate_mapreduce_partial_sum_shard(data)
    atomic_torch_save(data, path)


def load_mapreduce_partial_sum_shard(
    path: str | Path,
    *,
    expected_config_hash: Optional[str] = None,
    expected_target_start_id: Optional[int] = None,
    expected_target_end_id: Optional[int] = None,
) -> CandidatePreAggregationReducerEntry:
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata, payload = validate_mapreduce_partial_sum_shard(
        data,
        expected_config_hash=expected_config_hash,
        expected_target_start_id=expected_target_start_id,
        expected_target_end_id=expected_target_end_id,
    )
    return CandidatePreAggregationReducerEntry(metadata, payload)


def validate_mapreduce_partial_sum_shard(
    data: Dict[str, Any],
    *,
    expected_config_hash: Optional[str] = None,
    expected_target_start_id: Optional[int] = None,
    expected_target_end_id: Optional[int] = None,
) -> tuple[CandidatePreAggregationMetadata, Dict[str, Any]]:
    if not isinstance(data, dict) or "metadata" not in data or "payload" not in data:
        raise ValueError("MapReduce shard must contain metadata and payload")
    metadata, payload = validate_candidate_preaggregation_partial(
        {"metadata": data["metadata"], "payload": data["payload"]},
        expected_config_hash=expected_config_hash,
    )
    if expected_target_start_id is not None and metadata.target_start_id != expected_target_start_id:
        raise ValueError("MapReduce shard target range mismatch")
    if expected_target_end_id is not None and metadata.target_end_id != expected_target_end_id:
        raise ValueError("MapReduce shard target range mismatch")
    _validate_sorted_coo_payload(payload)
    storage_metadata = data.get("storage_metadata")
    if not isinstance(storage_metadata, dict):
        raise ValueError("MapReduce shard missing storage_metadata")
    if storage_metadata.get("format") != "sorted_coo":
        raise ValueError("MapReduce shard format must be sorted_coo")
    if int(storage_metadata.get("target_start_id", -1)) != metadata.target_start_id:
        raise ValueError("MapReduce shard storage target_start_id mismatch")
    if int(storage_metadata.get("target_end_id", -1)) != int(metadata.target_end_id):
        raise ValueError("MapReduce shard storage target_end_id mismatch")
    if int(storage_metadata.get("row_count", -1)) != metadata.contribution_count:
        raise ValueError("MapReduce shard row_count mismatch")
    if storage_metadata.get("value_dtype") != str(payload["values"].dtype):
        raise ValueError("MapReduce shard value_dtype mismatch")
    expected_checksum = checksum_coo_payload(payload)
    if storage_metadata.get("checksum_sha256") != expected_checksum:
        raise ValueError("MapReduce shard checksum mismatch")
    return metadata, payload


def load_mapreduce_reducer_shards(
    paths: Sequence[str | Path],
    *,
    guardrail_bytes: Optional[int] = None,
    fail_on_guardrail: bool = True,
    expected_config_hash: Optional[str] = None,
    expected_target_start_id: Optional[int] = None,
    expected_target_end_id: Optional[int] = None,
) -> tuple[CandidatePreAggregationReducerEntry, ...]:
    """Load reducer shards after estimating memory from file sizes."""

    estimate = estimate_mapreduce_reducer_input_bytes(paths=paths, guardrail_bytes=guardrail_bytes)
    if estimate.exceeds_guardrail:
        message = (
            "MapReduce reducer input estimate exceeds guardrail: "
            f"{estimate.file_bytes} bytes > {guardrail_bytes} bytes"
        )
        if fail_on_guardrail:
            raise MemoryError(message)
        print(f"  [pass2_reduce] WARNING: {message}")
    return tuple(
        load_mapreduce_partial_sum_shard(
            path,
            expected_config_hash=expected_config_hash,
            expected_target_start_id=expected_target_start_id,
            expected_target_end_id=expected_target_end_id,
        )
        for path in paths
    )


def estimate_mapreduce_shard_tensor_bytes(contribution_count: int) -> int:
    if contribution_count < 0:
        raise ValueError("contribution_count must be >= 0")
    # target_ids int64 + candidate_ids int32 + values float32 + sequence_ids int64.
    return int(contribution_count) * (8 + 4 + 4 + 8)


def estimate_mapreduce_reducer_input_bytes(
    *,
    entries: Optional[Sequence[CandidatePreAggregationReducerEntry]] = None,
    paths: Optional[Sequence[str | Path]] = None,
    guardrail_bytes: Optional[int] = None,
) -> MapReduceShardMemoryEstimate:
    if entries is None and paths is None:
        raise ValueError("entries or paths are required")
    contribution_count = 0
    tensor_bytes = 0
    file_bytes = 0
    shard_count = 0
    if entries is not None:
        shard_count = len(entries)
        contribution_count = sum(entry.metadata.contribution_count for entry in entries)
        tensor_bytes = estimate_mapreduce_shard_tensor_bytes(contribution_count)
    if paths is not None:
        shard_count = len(paths)
        file_bytes = sum(Path(path).stat().st_size for path in paths)
    total_estimate = file_bytes if file_bytes else tensor_bytes
    return MapReduceShardMemoryEstimate(
        shard_count=shard_count,
        contribution_count=contribution_count,
        tensor_bytes=tensor_bytes,
        file_bytes=file_bytes,
        guardrail_bytes=guardrail_bytes,
        exceeds_guardrail=guardrail_bytes is not None and total_estimate > guardrail_bytes,
    )


def checksum_coo_payload(payload: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("target_ids", "candidate_ids", "values", "sequence_ids"):
        tensor = payload[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_mapreduce_target_shard_result(
    path: str | Path,
    result: MapReduceTargetShardResult,
    inputs: CandidatePreAggregationReducerInputs,
    *,
    n_latents_per_latent: int,
) -> None:
    """Persist one reduced target-range shard for resume and final stitching."""

    validate_mapreduce_target_shard_result(
        result,
        target_range=result.target_range,
        num_components=inputs.num_components,
        d_sae=inputs.d_sae,
        n_latents_per_latent=n_latents_per_latent,
    )
    first = inputs.entries[0].metadata
    metadata = {
        "schema_version": 1,
        "reducer_mode": "mapreduce_target_ranges",
        "backend": "cpu",
        "run_id": first.run_id,
        "config_hash": first.config_hash,
        "mode": inputs.mode,
        "num_components": inputs.num_components,
        "d_sae": inputs.d_sae,
        "n_latents_per_latent": n_latents_per_latent,
        "reducer_id": result.target_range.reducer_id,
        "target_start_id": result.target_range.target_start_id,
        "target_end_id": result.target_range.target_end_id,
        "worker_ids": [entry.metadata.worker_id for entry in inputs.entries],
        "input_contribution_count": inputs.total_contribution_count,
    }
    payload = {
        "metadata": metadata,
        "top_indices": result.top_indices.detach().cpu().to(torch.int32),
        "top_values": result.top_values.detach().cpu().to(torch.float32),
    }
    validate_mapreduce_target_shard_artifact(
        payload,
        expected_target_range=result.target_range,
        expected_num_components=inputs.num_components,
        expected_d_sae=inputs.d_sae,
        expected_n_latents_per_latent=n_latents_per_latent,
        expected_mode=inputs.mode,
    )
    atomic_torch_save(payload, path)


def load_mapreduce_target_shard_result(
    path: str | Path,
    *,
    expected_target_range: Optional[TargetRange] = None,
    expected_config_hash: Optional[str] = None,
    expected_mode: Optional[str] = None,
    expected_num_components: Optional[int] = None,
    expected_d_sae: Optional[int] = None,
    expected_n_latents_per_latent: Optional[int] = None,
) -> MapReduceTargetShardArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return validate_mapreduce_target_shard_artifact(
        payload,
        path=path,
        expected_target_range=expected_target_range,
        expected_config_hash=expected_config_hash,
        expected_mode=expected_mode,
        expected_num_components=expected_num_components,
        expected_d_sae=expected_d_sae,
        expected_n_latents_per_latent=expected_n_latents_per_latent,
    )


def validate_mapreduce_target_shard_artifact(
    payload: Mapping[str, Any],
    *,
    path: Optional[str | Path] = None,
    expected_target_range: Optional[TargetRange] = None,
    expected_config_hash: Optional[str] = None,
    expected_mode: Optional[str] = None,
    expected_num_components: Optional[int] = None,
    expected_d_sae: Optional[int] = None,
    expected_n_latents_per_latent: Optional[int] = None,
) -> MapReduceTargetShardArtifact:
    if not isinstance(payload, Mapping):
        raise ValueError("MapReduce target shard must be a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("MapReduce target shard missing metadata")
    if metadata.get("reducer_mode") != "mapreduce_target_ranges":
        raise ValueError("MapReduce target shard reducer_mode mismatch")
    top_indices = payload.get("top_indices")
    top_values = payload.get("top_values")
    if not isinstance(top_indices, torch.Tensor) or not isinstance(top_values, torch.Tensor):
        raise ValueError("MapReduce target shard missing top tensors")
    target_range = TargetRange(
        reducer_id=int(metadata["reducer_id"]),
        target_start_id=int(metadata["target_start_id"]),
        target_end_id=int(metadata["target_end_id"]),
    )
    if expected_target_range is not None and target_range != expected_target_range:
        raise ValueError("MapReduce target shard range mismatch")
    if expected_config_hash is not None and metadata.get("config_hash") != expected_config_hash:
        raise ValueError("MapReduce target shard config hash mismatch")
    if expected_mode is not None and metadata.get("mode") != expected_mode:
        raise ValueError("MapReduce target shard mode mismatch")
    num_components = int(metadata["num_components"])
    d_sae = int(metadata["d_sae"])
    n_latents_per_latent = int(metadata["n_latents_per_latent"])
    if expected_num_components is not None and num_components != expected_num_components:
        raise ValueError("MapReduce target shard num_components mismatch")
    if expected_d_sae is not None and d_sae != expected_d_sae:
        raise ValueError("MapReduce target shard d_sae mismatch")
    if expected_n_latents_per_latent is not None and n_latents_per_latent != expected_n_latents_per_latent:
        raise ValueError("MapReduce target shard top-K mismatch")
    validate_mapreduce_target_shard_result(
        MapReduceTargetShardResult(
            target_range=target_range,
            top_indices=top_indices,
            top_values=top_values,
            summed_target_ids=torch.empty(0, dtype=torch.int64),
            summed_candidate_ids=torch.empty(0, dtype=torch.int32),
            summed_values=torch.empty(0, dtype=torch.float32),
        ),
        target_range=target_range,
        num_components=num_components,
        d_sae=d_sae,
        n_latents_per_latent=n_latents_per_latent,
    )
    return MapReduceTargetShardArtifact(
        path=Path(path) if path is not None else Path(),
        target_range=target_range,
        top_indices=top_indices.detach().cpu().to(torch.int32),
        top_values=top_values.detach().cpu().to(torch.float32),
        metadata=dict(metadata),
    )


def validate_mapreduce_target_shard_result(
    result: MapReduceTargetShardResult,
    *,
    target_range: TargetRange,
    num_components: int,
    d_sae: int,
    n_latents_per_latent: int,
) -> None:
    if target_range.target_start_id < 0 or target_range.target_end_id < target_range.target_start_id:
        raise ValueError("invalid MapReduce target range")
    if target_range.target_end_id > num_components * d_sae:
        raise ValueError("MapReduce target range exceeds flattened target count")
    expected_shape = (target_range.target_end_id - target_range.target_start_id, n_latents_per_latent)
    if tuple(result.top_indices.shape) != expected_shape or tuple(result.top_values.shape) != expected_shape:
        raise ValueError("MapReduce target shard tensor shape mismatch")
    if not torch.isfinite(result.top_values.float()).all():
        raise ValueError("MapReduce target shard values must be finite")


def _validate_sorted_coo_payload(payload: Dict[str, Any]) -> None:
    target_ids = payload["target_ids"].to(torch.int64).cpu().tolist()
    candidate_ids = payload["candidate_ids"].to(torch.int32).cpu().tolist()
    pairs = list(zip(target_ids, candidate_ids))
    if pairs != sorted(pairs):
        raise ValueError("MapReduce shard COO records must be sorted by (target_id, candidate_id)")


__all__ = [
    "build_mapreduce_storage_metadata",
    "checksum_coo_payload",
    "estimate_mapreduce_reducer_input_bytes",
    "estimate_mapreduce_shard_tensor_bytes",
    "load_mapreduce_partial_sum_shard",
    "load_mapreduce_reducer_shards",
    "load_mapreduce_target_shard_result",
    "save_mapreduce_partial_sum_shard",
    "save_mapreduce_target_shard_result",
    "sorted_coo_preaggregation_payload",
    "validate_mapreduce_partial_sum_shard",
    "validate_mapreduce_target_shard_artifact",
    "validate_mapreduce_target_shard_result",
]
