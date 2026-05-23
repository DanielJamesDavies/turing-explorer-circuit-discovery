"""Reducer-side input contracts for distributed pass-2 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import time
import hashlib
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from .pass2_partials import (
    CandidateDumpMetadata,
    CandidatePreAggregationMetadata,
    load_candidate_dump_partial,
    load_candidate_preaggregation_partial,
    validate_candidate_dump_partial,
    validate_candidate_preaggregation_partial,
)
from .pass2_replay import Pass2ReplayList, build_pass2_replay_list
from .interfaces import build_output_paths


@dataclass(frozen=True)
class CandidateDumpReducerEntry:
    metadata: CandidateDumpMetadata
    payload: Dict[str, Any]


@dataclass(frozen=True)
class CandidateDumpReducerInputs:
    entries: tuple[CandidateDumpReducerEntry, ...]
    mode: str
    m: int
    n_candidates_per_component: int
    n_latents_per_latent: int
    num_components: int
    d_sae: int
    total_sequence_count: int
    total_token_count: int


@dataclass(frozen=True)
class GlobalTopCtxTargetMapping:
    replay: Pass2ReplayList
    seq_offsets: torch.Tensor
    seq_targets_global: torch.Tensor
    sequence_ids: tuple[int, ...]
    sid_to_row: Dict[int, int]
    sid_to_row_tensor: torch.Tensor


@dataclass(frozen=True)
class SimpleExactCandidateDump:
    sequence_ids: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_vals: torch.Tensor
    sid_to_row: Dict[int, int]
    sid_to_row_tensor: torch.Tensor
    mode: str
    m: int
    n_candidates_per_component: int
    n_latents_per_latent: int
    num_components: int
    d_sae: int
    seq_len: int
    total_token_count: int


@dataclass(frozen=True)
class PmiReduceInputs:
    active_count: torch.Tensor
    total_replay_tokens: int
    total_worker_tokens: int


@dataclass(frozen=True)
class CandidatePreAggregationReducerEntry:
    metadata: CandidatePreAggregationMetadata
    payload: Dict[str, Any]


@dataclass(frozen=True)
class CandidatePreAggregationReducerInputs:
    entries: tuple[CandidatePreAggregationReducerEntry, ...]
    mode: str
    m: int
    num_components: int
    d_sae: int
    target_start_id: int
    target_end_id: int
    total_sequence_count: int
    total_contribution_count: int


@dataclass(frozen=True)
class TargetRange:
    reducer_id: int
    target_start_id: int
    target_end_id: int


@dataclass(frozen=True)
class MapReduceTargetShardResult:
    target_range: TargetRange
    top_indices: torch.Tensor
    top_values: torch.Tensor
    summed_target_ids: torch.Tensor
    summed_candidate_ids: torch.Tensor
    summed_values: torch.Tensor


@dataclass(frozen=True)
class MapReduceShardMemoryEstimate:
    shard_count: int
    contribution_count: int
    tensor_bytes: int
    file_bytes: int
    guardrail_bytes: Optional[int] = None
    exceeds_guardrail: bool = False


@dataclass(frozen=True)
class Pass2ReduceSchedulerConfig:
    reducer_mode: str = "mapreduce_target_ranges"
    reducer_count: int = 1
    execution_mode: str = "sequential"
    backend: str = "cpu"
    resume: bool = False
    cleanup: bool = False
    memory_guardrail_bytes: Optional[int] = None
    chunk_size: Optional[int] = None


@dataclass(frozen=True)
class MapReduceTargetShardArtifact:
    path: Path
    target_range: TargetRange
    top_indices: torch.Tensor
    top_values: torch.Tensor
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class MapReduceReduceResult:
    artifact_path: Path
    report_path: Path
    shard_paths: tuple[Path, ...]
    report: Dict[str, Any]


@dataclass(frozen=True)
class SimpleExactReduceResult:
    artifact_path: Path
    report_path: Path
    report: Dict[str, Any]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the simple exact pass-2 reduce CLI parser."""

    parser = argparse.ArgumentParser(
        description="Reduce distributed pass-2 candidate dumps into top_coactivation.pt."
    )
    parser.add_argument("--output-root", required=True, help="Canonical run root")
    parser.add_argument(
        "--candidate-dump",
        action="append",
        required=True,
        help="Path to a worker candidate_dump.partial.pt file. Repeat for every worker.",
    )
    parser.add_argument("--top-ctx", default=None, help="Path to top_ctx.pt")
    parser.add_argument("--latent-stats", default=None, help="Path to latent_stats.pt for PMI mode")
    parser.add_argument("--expected-config-hash", default=None)
    parser.add_argument("--mode", default=None, help="Expected coactivation mode")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for simple exact distributed pass-2 reduce."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_simple_exact_reduce_stage(
        output_root=args.output_root,
        candidate_dump_paths=args.candidate_dump,
        top_ctx_path=args.top_ctx,
        latent_stats_path=args.latent_stats,
        expected_config_hash=args.expected_config_hash,
        expected_mode=args.mode,
    )
    print(json.dumps(result.report, indent=2, sort_keys=True))


def run_simple_exact_reduce_stage(
    *,
    output_root: str | Path,
    candidate_dump_paths: Sequence[str | Path],
    top_ctx_path: str | Path | None = None,
    latent_stats_path: str | Path | None = None,
    expected_config_hash: Optional[str] = None,
    expected_mode: Optional[str] = None,
) -> SimpleExactReduceResult:
    """Load candidate dumps and run the simple exact reducer from canonical artifacts."""

    output_paths = build_output_paths(output_root)
    dump_inputs = load_candidate_dump_reducer_inputs(
        candidate_dump_paths,
        expected_config_hash=expected_config_hash,
        expected_mode=expected_mode,
    )
    mapping = load_global_top_ctx_target_mapping(
        top_ctx_path or output_paths.top_ctx,
        dump_inputs=dump_inputs,
    )
    active_count = None
    if dump_inputs.mode == "pmi":
        active_count = load_global_active_count(
            latent_stats_path or output_paths.latent_stats,
            expected_config_hash=expected_config_hash,
            expected_num_components=dump_inputs.num_components,
            expected_d_sae=dump_inputs.d_sae,
        )
    from store.top_coactivation import top_coactivation

    return run_simple_exact_reduce_and_write(
        top_coactivation,
        dump_inputs,
        mapping,
        output_root,
        active_count=active_count,
    )


def load_candidate_dump_reducer_inputs(
    paths: Sequence[str | Path],
    *,
    expected_config_hash: Optional[str] = None,
    expected_mode: Optional[str] = None,
) -> CandidateDumpReducerInputs:
    """Load and validate simple exact worker candidate dumps for reduction."""

    entries = [
        CandidateDumpReducerEntry(*load_candidate_dump_partial(path, expected_config_hash=expected_config_hash))
        for path in paths
    ]
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


def partition_target_ranges(num_targets: int, reducer_count: int) -> tuple[TargetRange, ...]:
    """Split flattened target IDs into balanced contiguous reducer ranges."""

    if num_targets < 0:
        raise ValueError("num_targets must be >= 0")
    if reducer_count < 1:
        raise ValueError("reducer_count must be >= 1")
    base, remainder = divmod(num_targets, reducer_count)
    ranges: list[TargetRange] = []
    start = 0
    for reducer_id in range(reducer_count):
        width = base + (1 if reducer_id < remainder else 0)
        end = start + width
        ranges.append(
            TargetRange(
                reducer_id=reducer_id,
                target_start_id=start,
                target_end_id=end,
            )
        )
        start = end
    return tuple(ranges)


def shard_preaggregation_by_target_range(
    metadata: CandidatePreAggregationMetadata,
    payload: Dict[str, Any],
    target_ranges: Sequence[TargetRange],
) -> tuple[CandidatePreAggregationReducerEntry, ...]:
    """Partition raw contribution records by target range only."""

    _metadata, validated_payload = validate_candidate_preaggregation_partial(
        {"metadata": metadata.model_dump(mode="json"), "payload": payload}
    )
    target_ids = validated_payload["target_ids"].to(torch.int64).cpu()
    candidate_ids = validated_payload["candidate_ids"].to(torch.int32).cpu()
    values = validated_payload["values"].to(torch.float32).cpu()
    sequence_ids = validated_payload["sequence_ids"].to(torch.int64).cpu()
    shards: list[CandidatePreAggregationReducerEntry] = []
    for target_range in target_ranges:
        mask = (target_ids >= target_range.target_start_id) & (target_ids < target_range.target_end_id)
        shard_payload = {
            "target_ids": target_ids[mask].clone(),
            "candidate_ids": candidate_ids[mask].clone(),
            "values": values[mask].clone(),
            "sequence_ids": sequence_ids[mask].clone(),
        }
        shard_metadata = CandidatePreAggregationMetadata.model_validate(
            {
                **metadata.model_dump(mode="json"),
                "contribution_count": int(mask.sum().item()),
                "target_start_id": target_range.target_start_id,
                "target_end_id": target_range.target_end_id,
            }
        )
        shard_metadata, shard_payload = validate_candidate_preaggregation_partial(
            {"metadata": shard_metadata.model_dump(mode="json"), "payload": shard_payload}
        )
        shards.append(CandidatePreAggregationReducerEntry(shard_metadata, shard_payload))
    return tuple(shards)


def reduce_mapreduce_target_range(
    inputs: CandidatePreAggregationReducerInputs,
    *,
    n_latents_per_latent: int,
    chunk_size: Optional[int] = None,
    reducer_id: int = 0,
) -> MapReduceTargetShardResult:
    """Merge partial sums for one target range and keep deterministic top-K."""

    if n_latents_per_latent < 1:
        raise ValueError("n_latents_per_latent must be >= 1")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if reducer_id < 0:
        raise ValueError("reducer_id must be >= 0")
    aggregate: dict[tuple[int, int], float] = {}
    for entry in inputs.entries:
        target_ids = entry.payload["target_ids"].to(torch.int64).cpu()
        candidate_ids = entry.payload["candidate_ids"].to(torch.int32).cpu()
        values = entry.payload["values"].to(torch.float32).cpu()
        width = int(target_ids.numel())
        step = chunk_size or width or 1
        for start in range(0, width, step):
            stop = min(start + step, width)
            for target_id, candidate_id, value in zip(
                target_ids[start:stop].tolist(),
                candidate_ids[start:stop].tolist(),
                values[start:stop].tolist(),
            ):
                target_id = int(target_id)
                candidate_id = int(candidate_id)
                if target_id < inputs.target_start_id or target_id >= inputs.target_end_id:
                    raise ValueError("target ID outside reducer range")
                key = (target_id, candidate_id)
                aggregate[key] = aggregate.get(key, 0.0) + float(value)

    sorted_records = sorted(
        ((target_id, candidate_id, value) for (target_id, candidate_id), value in aggregate.items()),
        key=lambda record: (record[0], record[1]),
    )
    target_width = inputs.target_end_id - inputs.target_start_id
    top_indices = torch.zeros((target_width, n_latents_per_latent), dtype=torch.int32)
    top_values = torch.zeros((target_width, n_latents_per_latent), dtype=torch.float32)
    by_target: dict[int, list[tuple[int, float]]] = {}
    for target_id, candidate_id, value in sorted_records:
        by_target.setdefault(target_id, []).append((candidate_id, value))
    for target_id, candidate_values in by_target.items():
        row = target_id - inputs.target_start_id
        selected = sorted(candidate_values, key=lambda item: (-item[1], item[0]))[:n_latents_per_latent]
        for column, (candidate_id, value) in enumerate(selected):
            top_indices[row, column] = int(candidate_id)
            top_values[row, column] = float(value)

    return MapReduceTargetShardResult(
        target_range=TargetRange(
            reducer_id=reducer_id,
            target_start_id=inputs.target_start_id,
            target_end_id=inputs.target_end_id,
        ),
        top_indices=top_indices,
        top_values=top_values,
        summed_target_ids=torch.tensor([record[0] for record in sorted_records], dtype=torch.int64),
        summed_candidate_ids=torch.tensor([record[1] for record in sorted_records], dtype=torch.int32),
        summed_values=torch.tensor([record[2] for record in sorted_records], dtype=torch.float32),
    )


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
    _atomic_torch_save(data, path)


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


def validate_pass2_reduce_scheduler_config(config: Pass2ReduceSchedulerConfig) -> Pass2ReduceSchedulerConfig:
    """Validate explicit reducer scheduling fields before dispatch."""

    if config.reducer_mode not in {"simple_exact", "target_sharded", "mapreduce_target_ranges"}:
        raise ValueError("unknown pass-2 reducer_mode")
    if config.reducer_count < 1:
        raise ValueError("reducer_count must be >= 1")
    if config.execution_mode not in {"sequential", "parallel"}:
        raise ValueError("unknown pass-2 reducer execution_mode")
    if config.execution_mode == "parallel":
        raise NotImplementedError("parallel MapReduce reducer execution is not implemented yet")
    if config.backend not in {"cpu", "openmp"}:
        raise ValueError("MapReduce reducer backend must be cpu or openmp")
    if config.memory_guardrail_bytes is not None and config.memory_guardrail_bytes < 1:
        raise ValueError("memory_guardrail_bytes must be >= 1")
    if config.chunk_size is not None and config.chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    return config


def mapreduce_target_shard_path(shard_dir: str | Path, target_range: TargetRange) -> Path:
    return Path(shard_dir) / (
        f"reducer_{target_range.reducer_id:04d}_"
        f"targets_{target_range.target_start_id:08d}_{target_range.target_end_id:08d}.pt"
    )


def cleanup_mapreduce_target_shards(shard_dir: str | Path) -> int:
    """Remove reducer output shards from a previous MapReduce scheduling attempt."""

    root = Path(shard_dir)
    if not root.exists():
        return 0
    removed = 0
    for path in root.glob("reducer_*_targets_*.pt"):
        path.unlink()
        removed += 1
    return removed


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
    _atomic_torch_save(payload, path)


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


def run_mapreduce_reduce_and_write(
    shard_paths_by_reducer: Mapping[int, Sequence[str | Path]],
    output_root: str | Path,
    *,
    config: Pass2ReduceSchedulerConfig,
    num_components: int,
    d_sae: int,
    n_latents_per_latent: int,
    mode: str,
    total_tokens_processed: int = 0,
    expected_config_hash: Optional[str] = None,
    expected_worker_ids: Optional[Sequence[int]] = None,
    active_count: Optional[torch.Tensor] = None,
    seq_offsets: Optional[torch.Tensor] = None,
    seq_targets_global: Optional[torch.Tensor] = None,
    seq_len: Optional[int] = None,
    pmi_sae_k: int = 1,
) -> MapReduceReduceResult:
    """Run sequential CPU MapReduce reducer shards and write canonical output."""

    config = validate_pass2_reduce_scheduler_config(config)
    if config.reducer_mode != "mapreduce_target_ranges":
        raise ValueError("run_mapreduce_reduce_and_write requires mapreduce_target_ranges mode")
    output_paths = build_output_paths(output_root)
    shard_dir = output_paths.run_root / "distributed" / "pass2_reduce" / "mapreduce_shards"
    report_dir = output_paths.run_root / "distributed" / "reports"
    shard_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    removed_shards = cleanup_mapreduce_target_shards(shard_dir) if config.cleanup and not config.resume else 0
    target_ranges = partition_target_ranges(num_components * d_sae, config.reducer_count)
    artifacts: list[MapReduceTargetShardArtifact] = []
    reused_shards = 0
    written_shards = 0
    input_partial_sum_bytes = sum(_file_size_or_zero(path) for paths in shard_paths_by_reducer.values() for path in paths)
    timing = {
        "load_shards_s": 0.0,
        "reduce_s": 0.0,
        "shard_write_s": 0.0,
        "shard_load_s": 0.0,
        "stitch_s": 0.0,
        "pmi_s": 0.0,
        "save_s": 0.0,
    }

    memory_trace = _start_memory_trace()
    started = time.perf_counter()
    for target_range in target_ranges:
        if target_range.target_start_id == target_range.target_end_id:
            continue
        output_shard_path = mapreduce_target_shard_path(shard_dir, target_range)
        if config.resume and output_shard_path.exists():
            load_started = time.perf_counter()
            artifacts.append(
                load_mapreduce_target_shard_result(
                    output_shard_path,
                    expected_target_range=target_range,
                    expected_config_hash=expected_config_hash,
                    expected_mode=mode,
                    expected_num_components=num_components,
                    expected_d_sae=d_sae,
                    expected_n_latents_per_latent=n_latents_per_latent,
                )
            )
            timing["shard_load_s"] += time.perf_counter() - load_started
            reused_shards += 1
            continue

        paths = tuple(shard_paths_by_reducer.get(target_range.reducer_id, ()))
        load_started = time.perf_counter()
        entries = load_mapreduce_reducer_shards(
            paths,
            guardrail_bytes=config.memory_guardrail_bytes,
            expected_config_hash=expected_config_hash,
            expected_target_start_id=target_range.target_start_id,
            expected_target_end_id=target_range.target_end_id,
        )
        timing["load_shards_s"] += time.perf_counter() - load_started
        inputs = validate_candidate_preaggregation_reducer_inputs(
            entries,
            expected_config_hash=expected_config_hash,
            expected_mode=mode,
            expected_target_start_id=target_range.target_start_id,
            expected_target_end_id=target_range.target_end_id,
            expected_worker_ids=expected_worker_ids,
        )
        reduce_started = time.perf_counter()
        result = reduce_mapreduce_target_range(
            inputs,
            n_latents_per_latent=n_latents_per_latent,
            chunk_size=config.chunk_size,
            reducer_id=target_range.reducer_id,
        )
        timing["reduce_s"] += time.perf_counter() - reduce_started
        write_started = time.perf_counter()
        save_mapreduce_target_shard_result(
            output_shard_path,
            result,
            inputs,
            n_latents_per_latent=n_latents_per_latent,
        )
        timing["shard_write_s"] += time.perf_counter() - write_started
        load_started = time.perf_counter()
        artifacts.append(
            load_mapreduce_target_shard_result(
                output_shard_path,
                expected_target_range=target_range,
                expected_config_hash=expected_config_hash,
                expected_mode=mode,
                expected_num_components=num_components,
                expected_d_sae=d_sae,
                expected_n_latents_per_latent=n_latents_per_latent,
            )
        )
        timing["shard_load_s"] += time.perf_counter() - load_started
        written_shards += 1

    stitch_started = time.perf_counter()
    top_indices, top_values = stitch_mapreduce_target_shards(
        artifacts,
        num_components=num_components,
        d_sae=d_sae,
        n_latents_per_latent=n_latents_per_latent,
    )
    timing["stitch_s"] = time.perf_counter() - stitch_started
    if mode == "pmi":
        if active_count is None or seq_offsets is None or seq_targets_global is None or seq_len is None:
            raise ValueError("MapReduce PMI reduction requires active_count, seq_offsets, seq_targets_global, and seq_len")
        pmi_started = time.perf_counter()
        top_values = apply_pmi_postprocess_to_topk(
            top_indices,
            top_values,
            active_count=active_count,
            seq_offsets=seq_offsets,
            seq_targets_global=seq_targets_global,
            seq_len=seq_len,
            num_components=num_components,
            d_sae=d_sae,
            sae_k=pmi_sae_k,
        )
        timing["pmi_s"] = time.perf_counter() - pmi_started
    save_started = time.perf_counter()
    _write_mapreduce_top_coactivation_artifact(
        output_paths.top_coactivation,
        top_indices=top_indices,
        top_values=top_values,
        mode=mode,
        total_tokens_processed=total_tokens_processed,
    )
    timing["save_s"] = time.perf_counter() - save_started
    elapsed_s = time.perf_counter() - started
    peak_cpu_memory_bytes = _stop_memory_trace(memory_trace)
    output_artifact_size_bytes = _file_size_or_zero(output_paths.top_coactivation)
    output_shard_bytes = sum(_file_size_or_zero(artifact.path) for artifact in artifacts)
    timing["total_s"] = elapsed_s
    report = {
        "schema_version": 1,
        "reducer_mode": config.reducer_mode,
        "coactivation_mode": mode,
        "execution_mode": config.execution_mode,
        "backend": config.backend,
        "reducer_count": config.reducer_count,
        "shards_written": written_shards,
        "shards_reused": reused_shards,
        "shards_cleaned": removed_shards,
        "input_partial_sum_bytes": input_partial_sum_bytes,
        "output_shard_bytes": output_shard_bytes,
        "output_artifact": str(output_paths.top_coactivation),
        "output_artifact_size_bytes": output_artifact_size_bytes,
        "output_shape": [num_components, d_sae, n_latents_per_latent],
        "peak_cpu_memory_bytes": peak_cpu_memory_bytes,
        "timing": timing,
        "elapsed_s": elapsed_s,
    }
    report["manifest_metrics"] = build_pass2_reduce_manifest_metrics(report)
    report_path = report_dir / "pass2_mapreduce_reduce_report.json"
    _atomic_write_json(report_path, report)
    return MapReduceReduceResult(
        artifact_path=output_paths.top_coactivation,
        report_path=report_path,
        shard_paths=tuple(artifact.path for artifact in artifacts),
        report=report,
    )


def stitch_mapreduce_target_shards(
    artifacts: Sequence[MapReduceTargetShardArtifact | MapReduceTargetShardResult],
    *,
    num_components: int,
    d_sae: int,
    n_latents_per_latent: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch contiguous reducer target shards into canonical [C, d_sae, K] tensors."""

    total_targets = num_components * d_sae
    ordered = sorted(artifacts, key=lambda artifact: artifact.target_range.target_start_id)
    expected_start = 0
    flat_indices = torch.zeros((total_targets, n_latents_per_latent), dtype=torch.int32)
    flat_values = torch.zeros((total_targets, n_latents_per_latent), dtype=torch.float32)
    for artifact in ordered:
        target_range = artifact.target_range
        if target_range.target_start_id != expected_start:
            raise ValueError("MapReduce target shards do not cover targets contiguously")
        if target_range.target_end_id > total_targets:
            raise ValueError("MapReduce target shard exceeds flattened target count")
        expected_shape = (target_range.target_end_id - target_range.target_start_id, n_latents_per_latent)
        if tuple(artifact.top_indices.shape) != expected_shape or tuple(artifact.top_values.shape) != expected_shape:
            raise ValueError("MapReduce target shard tensor shape mismatch")
        flat_indices[target_range.target_start_id : target_range.target_end_id] = artifact.top_indices.to(torch.int32)
        flat_values[target_range.target_start_id : target_range.target_end_id] = artifact.top_values.to(torch.float32)
        expected_start = target_range.target_end_id
    if expected_start != total_targets:
        raise ValueError("MapReduce target shards do not cover all targets")
    return (
        flat_indices.reshape(num_components, d_sae, n_latents_per_latent),
        flat_values.reshape(num_components, d_sae, n_latents_per_latent),
    )


def apply_pmi_postprocess_to_topk(
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    *,
    active_count: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_targets_global: torch.Tensor,
    seq_len: int,
    num_components: int,
    d_sae: int,
    sae_k: int,
    pmi_clamp_min: float = -5.0,
    pmi_clamp_max: float = 10.0,
) -> torch.Tensor:
    """Apply the same global PMI postprocess used by TopCoactivation to stitched top-K counts."""

    if sae_k < 1:
        raise ValueError("sae_k must be >= 1")
    if pmi_clamp_min > pmi_clamp_max:
        raise ValueError("pmi_clamp_min must be <= pmi_clamp_max")
    active_count = validate_global_active_count(
        active_count,
        expected_num_components=num_components,
        expected_d_sae=d_sae,
    )
    expected_prefix = (num_components, d_sae)
    if tuple(top_indices.shape[:2]) != expected_prefix or tuple(top_values.shape[:2]) != expected_prefix:
        raise ValueError("PMI top-K tensors must match top_coactivation dimensions")
    if tuple(top_indices.shape) != tuple(top_values.shape):
        raise ValueError("PMI top_indices and top_values shapes must match")
    if seq_offsets.dtype != torch.int64 or seq_targets_global.dtype != torch.int64:
        raise ValueError("PMI seq_offsets and seq_targets_global must be int64")
    if seq_len <= 0:
        raise ValueError("PMI seq_len must be positive")

    total_tokens_globally = max(1, int(active_count[0].sum().item()) // int(sae_k))
    global_rate = active_count.flatten().float() / total_tokens_globally
    per_target_tokens = compute_total_tokens_per_target(
        seq_offsets,
        seq_targets_global,
        seq_len=seq_len,
        num_targets=num_components * d_sae,
    )
    context_rate = top_values.detach().cpu().to(torch.float32) / per_target_tokens.view(num_components, d_sae, 1).clamp(min=1)
    j_rate = global_rate[top_indices.detach().cpu().long()]
    pmi = (context_rate / j_rate.clamp(min=1e-10)).log().clamp(pmi_clamp_min, pmi_clamp_max)
    if not torch.isfinite(pmi).all():
        raise ValueError("PMI postprocess produced non-finite values")
    return pmi.to(torch.float32)


def compute_total_tokens_per_target(
    seq_offsets: torch.Tensor,
    seq_targets_global: torch.Tensor,
    *,
    seq_len: int,
    num_targets: int,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if num_targets < 0:
        raise ValueError("num_targets must be >= 0")
    valid_mask = (seq_targets_global >= 0) & (seq_targets_global < num_targets)
    valid_targets = seq_targets_global[valid_mask].long()
    target_seq_counts = torch.zeros(num_targets, dtype=torch.float32)
    target_seq_counts.scatter_add_(0, valid_targets, torch.ones(valid_targets.shape[0], dtype=torch.float32))
    return target_seq_counts * int(seq_len)


def build_pass2_reduce_manifest_metrics(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract stable reducer metrics suitable for manifest/report embedding."""

    timing = report.get("timing", {})
    if not isinstance(timing, Mapping):
        timing = {}
    return {
        "reducer_mode": report.get("reducer_mode"),
        "coactivation_mode": report.get("coactivation_mode"),
        "backend": report.get("backend"),
        "reducer_count": int(report.get("reducer_count", 1) or 1),
        "input_bytes": int(
            report.get("input_candidate_dump_bytes", report.get("input_partial_sum_bytes", 0)) or 0
        ),
        "output_artifact_size_bytes": int(report.get("output_artifact_size_bytes", 0) or 0),
        "peak_cpu_memory_bytes": report.get("peak_cpu_memory_bytes"),
        "total_s": float(timing.get("total_s", report.get("elapsed_s", 0.0)) or 0.0),
        "reduce_s": float(timing.get("reduce_s", 0.0) or 0.0),
        "pmi_s": float(timing.get("pmi_s", 0.0) or 0.0),
    }


def format_pass2_reduce_benchmark_report(report: Mapping[str, Any]) -> str:
    """Format a reducer report into a short benchmark summary."""

    timing = report.get("timing", {})
    if not isinstance(timing, Mapping):
        timing = {}
    lines = [
        "pass2 reduce benchmark:",
        f"  reducer_mode: {report.get('reducer_mode')}",
        f"  coactivation_mode: {report.get('coactivation_mode')}",
        f"  backend: {report.get('backend')}",
        f"  reducer_count: {report.get('reducer_count', 1)}",
        f"  input_bytes: {report.get('input_candidate_dump_bytes', report.get('input_partial_sum_bytes', 0))}",
        f"  output_artifact_size_bytes: {report.get('output_artifact_size_bytes', 0)}",
        f"  peak_cpu_memory_bytes: {report.get('peak_cpu_memory_bytes')}",
        f"  total_s: {float(timing.get('total_s', report.get('elapsed_s', 0.0)) or 0.0):.6f}",
        f"  reduce_s: {float(timing.get('reduce_s', 0.0) or 0.0):.6f}",
        f"  pmi_s: {float(timing.get('pmi_s', 0.0) or 0.0):.6f}",
    ]
    if "shard_write_s" in timing or "stitch_s" in timing:
        lines.extend(
            [
                f"  shard_write_s: {float(timing.get('shard_write_s', 0.0) or 0.0):.6f}",
                f"  stitch_s: {float(timing.get('stitch_s', 0.0) or 0.0):.6f}",
            ]
        )
    return "\n".join(lines)


def load_global_top_ctx_target_mapping(
    path: str | Path,
    *,
    dump_inputs: Optional[CandidateDumpReducerInputs] = None,
) -> GlobalTopCtxTargetMapping:
    """Load merged global top_ctx.pt and build reducer CSR plus dump row mapping."""

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

    seen: set[int] = set()
    duplicates: set[int] = set()
    replay_set = set(mapping.sequence_ids)
    for entry in dump_inputs.entries:
        sequence_ids = entry.payload["sequence_ids"]
        for sequence_id in sequence_ids.tolist():
            sequence_id = int(sequence_id)
            if sequence_id == 0:
                raise ValueError("candidate dump contains sentinel sequence ID 0")
            if sequence_id not in replay_set:
                raise ValueError(f"candidate dump contains sequence ID outside global replay set: {sequence_id}")
            if sequence_id in seen:
                duplicates.add(sequence_id)
            seen.add(sequence_id)
    if duplicates:
        raise ValueError(f"candidate dumps contain duplicate sequence IDs: {sorted(duplicates)}")
    missing = sorted(replay_set - seen)
    if missing:
        raise ValueError(f"candidate dumps missing replay sequence IDs: {missing}")
    extras = sorted(seen - replay_set)
    if extras:
        raise ValueError(f"candidate dumps contain extra replay sequence IDs: {extras}")


def build_simple_exact_candidate_dump(
    dump_inputs: CandidateDumpReducerInputs,
    mapping: GlobalTopCtxTargetMapping,
) -> SimpleExactCandidateDump:
    """
    Concatenate worker dumps into one sequence-ID ordered reducer input.

    The resulting tensors match the in-memory dump contract consumed by
    TopCoactivation.reduce(): rows are ordered by the global replay list, while
    sid_to_row maps arbitrary sequence IDs back to those rows.
    """

    validate_candidate_dump_sequence_coverage(dump_inputs, mapping)
    _validate_simple_dump_reduce_dimensions(dump_inputs, mapping)
    sequence_count = len(mapping.sequence_ids)
    candidate_ids = torch.zeros((sequence_count, dump_inputs.m), dtype=torch.int32)
    candidate_vals = torch.zeros((sequence_count, dump_inputs.m), dtype=torch.float32)
    for entry in dump_inputs.entries:
        sequence_ids = entry.payload["sequence_ids"].to(torch.int64).cpu()
        worker_candidate_ids = entry.payload["candidate_ids"].to(torch.int32).cpu()
        worker_candidate_vals = entry.payload["candidate_vals"].to(torch.float32).cpu()
        for source_row, sequence_id_tensor in enumerate(sequence_ids):
            sequence_id = int(sequence_id_tensor.item())
            destination_row = mapping.sid_to_row[sequence_id]
            candidate_ids[destination_row] = worker_candidate_ids[source_row]
            candidate_vals[destination_row] = worker_candidate_vals[source_row]

    first_metadata = dump_inputs.entries[0].metadata
    return SimpleExactCandidateDump(
        sequence_ids=torch.tensor(mapping.sequence_ids, dtype=torch.int64),
        candidate_ids=candidate_ids,
        candidate_vals=candidate_vals,
        sid_to_row=dict(mapping.sid_to_row),
        sid_to_row_tensor=mapping.sid_to_row_tensor.clone(),
        mode=dump_inputs.mode,
        m=dump_inputs.m,
        n_candidates_per_component=dump_inputs.n_candidates_per_component,
        n_latents_per_latent=dump_inputs.n_latents_per_latent,
        num_components=dump_inputs.num_components,
        d_sae=dump_inputs.d_sae,
        seq_len=first_metadata.seq_len,
        total_token_count=dump_inputs.total_token_count,
    )


def attach_simple_exact_dump_to_store(
    top_coactivation_store,
    dump: SimpleExactCandidateDump,
) -> None:
    """Attach a merged dump to a TopCoactivation-compatible store."""

    if int(top_coactivation_store.num_components) != dump.num_components:
        raise ValueError("top_coactivation store num_components mismatch")
    if int(top_coactivation_store.d_sae) != dump.d_sae:
        raise ValueError("top_coactivation store d_sae mismatch")
    if int(top_coactivation_store.n_latents_per_latent) != dump.n_latents_per_latent:
        raise ValueError("top_coactivation store n_latents_per_latent mismatch")
    if int(top_coactivation_store.n_candidates_per_component) != dump.n_candidates_per_component:
        raise ValueError("top_coactivation store n_candidates_per_component mismatch")
    if int(top_coactivation_store.M) != dump.m:
        raise ValueError("top_coactivation store M mismatch")
    if top_coactivation_store.mode != dump.mode:
        raise ValueError("top_coactivation store mode mismatch")

    top_coactivation_store.candidate_ids = dump.candidate_ids.clone()
    top_coactivation_store.candidate_vals = dump.candidate_vals.clone()
    top_coactivation_store.seq_id_to_row = dict(dump.sid_to_row)
    top_coactivation_store.sid_to_row_tensor = dump.sid_to_row_tensor.clone()
    top_coactivation_store.total_tokens_processed = dump.total_token_count


def reduce_simple_exact_candidate_dump(
    top_coactivation_store,
    dump: SimpleExactCandidateDump,
    mapping: GlobalTopCtxTargetMapping,
    *,
    active_count: Optional[torch.Tensor] = None,
) -> None:
    """Run the existing TopCoactivation reducer over a merged simple exact dump."""

    pmi_inputs = validate_pmi_reduce_inputs(
        dump,
        mapping,
        active_count=active_count,
    )
    attach_simple_exact_dump_to_store(top_coactivation_store, dump)
    top_coactivation_store.reduce(
        mapping.seq_offsets,
        mapping.seq_targets_global,
        seq_len=dump.seq_len,
        active_count=pmi_inputs.active_count if pmi_inputs is not None else active_count,
    )
    validate_top_coactivation_reduce_output(top_coactivation_store, dump)


def run_simple_exact_reduce_and_write(
    top_coactivation_store,
    dump_inputs: CandidateDumpReducerInputs,
    mapping: GlobalTopCtxTargetMapping,
    output_root: str | Path,
    *,
    active_count: Optional[torch.Tensor] = None,
    report_name: str = "pass2_reduce_report.json",
) -> SimpleExactReduceResult:
    """Reduce simple exact worker dumps and write canonical top_coactivation.pt."""

    output_paths = build_output_paths(output_root)
    reports_dir = output_paths.run_root / "distributed" / "reports"
    output_paths.run_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    memory_trace = _start_memory_trace()
    build_started = time.perf_counter()
    dump = build_simple_exact_candidate_dump(dump_inputs, mapping)
    build_elapsed_s = time.perf_counter() - build_started

    reduce_started = time.perf_counter()
    reduce_simple_exact_candidate_dump(
        top_coactivation_store,
        dump,
        mapping,
        active_count=active_count,
    )
    reduce_elapsed_s = time.perf_counter() - reduce_started

    save_started = time.perf_counter()
    _atomic_store_save(top_coactivation_store, output_paths.top_coactivation)
    save_elapsed_s = time.perf_counter() - save_started
    peak_cpu_memory_bytes = _stop_memory_trace(memory_trace)
    validate_saved_top_coactivation_artifact(
        top_coactivation_store,
        output_paths.top_coactivation,
        dump=dump,
    )

    report = build_simple_exact_reduce_report(
        top_coactivation_store,
        dump_inputs,
        dump,
        mapping,
        artifact_path=output_paths.top_coactivation,
        build_elapsed_s=build_elapsed_s,
        reduce_elapsed_s=reduce_elapsed_s,
        save_elapsed_s=save_elapsed_s,
        peak_cpu_memory_bytes=peak_cpu_memory_bytes,
    )
    report_path = reports_dir / report_name
    _atomic_write_json(report_path, report)
    return SimpleExactReduceResult(
        artifact_path=output_paths.top_coactivation,
        report_path=report_path,
        report=report,
    )


def build_simple_exact_reduce_report(
    top_coactivation_store,
    dump_inputs: CandidateDumpReducerInputs,
    dump: SimpleExactCandidateDump,
    mapping: GlobalTopCtxTargetMapping,
    *,
    artifact_path: str | Path,
    build_elapsed_s: float,
    reduce_elapsed_s: float,
    save_elapsed_s: float,
    peak_cpu_memory_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable reducer report for canonical output validation."""

    top_indices = getattr(top_coactivation_store, "top_indices", None)
    top_values = getattr(top_coactivation_store, "top_values", None)
    output_nonzero_count = 0
    output_finite = True
    output_shape: list[int] = []
    if isinstance(top_values, torch.Tensor):
        output_nonzero_count = int((top_values != 0).sum().item())
        output_finite = bool(torch.isfinite(top_values).all().item())
        output_shape = [int(dim) for dim in top_values.shape]
    if isinstance(top_indices, torch.Tensor):
        output_shape = [int(dim) for dim in top_indices.shape]

    output_artifact_size_bytes = _file_size_or_zero(artifact_path)
    candidate_dump_bytes = int(dump.candidate_ids.numel() * dump.candidate_ids.element_size()) + int(
        dump.candidate_vals.numel() * dump.candidate_vals.element_size()
    )
    input_dump_bytes = sum(
        _candidate_dump_entry_tensor_bytes(entry)
        for entry in dump_inputs.entries
    )
    report = {
        "schema_version": 1,
        "reducer_mode": "simple_exact",
        "coactivation_mode": dump.mode,
        "backend": "top_coactivation_reduce",
        "worker_count": len(dump_inputs.entries),
        "replay_sequence_count": len(mapping.sequence_ids),
        "candidate_dump_sequence_count": dump_inputs.total_sequence_count,
        "candidate_width": dump.m,
        "num_components": dump.num_components,
        "d_sae": dump.d_sae,
        "n_latents_per_latent": dump.n_latents_per_latent,
        "seq_len": dump.seq_len,
        "total_worker_token_count": dump.total_token_count,
        "input_candidate_dump_bytes": input_dump_bytes,
        "merged_candidate_dump_bytes": candidate_dump_bytes,
        "output_artifact": str(artifact_path),
        "output_artifact_size_bytes": output_artifact_size_bytes,
        "output_shape": output_shape,
        "output_nonzero_count": output_nonzero_count,
        "output_finite": output_finite,
        "peak_cpu_memory_bytes": peak_cpu_memory_bytes,
        "timing": {
            "build_dump_s": float(build_elapsed_s),
            "reduce_s": float(reduce_elapsed_s),
            "pmi_s": float(reduce_elapsed_s) if dump.mode == "pmi" else 0.0,
            "save_s": float(save_elapsed_s),
            "total_s": float(build_elapsed_s + reduce_elapsed_s + save_elapsed_s),
        },
    }
    report["manifest_metrics"] = build_pass2_reduce_manifest_metrics(report)
    return report


def validate_saved_top_coactivation_artifact(
    top_coactivation_store,
    path: str | Path,
    *,
    dump: SimpleExactCandidateDump,
) -> None:
    """Validate canonical top_coactivation.pt can be loaded by the existing store."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("top_indices", "top_values", "freq_factors", "total_tokens_processed", "mode"):
        if key not in payload:
            raise ValueError(f"top_coactivation artifact missing field: {key}")
    expected_shape = (dump.num_components, dump.d_sae, dump.n_latents_per_latent)
    if tuple(payload["top_indices"].shape) != expected_shape:
        raise ValueError("saved top_indices shape mismatch")
    if tuple(payload["top_values"].shape) != expected_shape:
        raise ValueError("saved top_values shape mismatch")
    if not torch.isfinite(payload["top_values"]).all():
        raise ValueError("saved top_values must be finite")
    if payload["mode"] != dump.mode:
        raise ValueError("saved top_coactivation mode mismatch")
    if hasattr(top_coactivation_store, "load"):
        top_coactivation_store.load(str(path))


def validate_pmi_reduce_inputs(
    dump: SimpleExactCandidateDump,
    mapping: GlobalTopCtxTargetMapping,
    *,
    active_count: Optional[torch.Tensor],
) -> Optional[PmiReduceInputs]:
    """Validate global inputs required to apply PMI exactly once after reduce."""

    if dump.mode != "pmi":
        return None
    if active_count is None:
        raise ValueError("PMI reduction requires merged global latent_stats.active_count")
    validated_active_count = validate_global_active_count(
        active_count,
        expected_num_components=dump.num_components,
        expected_d_sae=dump.d_sae,
    )
    total_replay_tokens = len(mapping.sequence_ids) * int(dump.seq_len)
    if total_replay_tokens <= 0:
        raise ValueError("PMI reduction requires a non-empty replay sequence set")
    if int(dump.total_token_count) != total_replay_tokens:
        raise ValueError("PMI worker token-count metadata does not match replay sequence count")
    return PmiReduceInputs(
        active_count=validated_active_count,
        total_replay_tokens=total_replay_tokens,
        total_worker_tokens=int(dump.total_token_count),
    )


def validate_top_coactivation_reduce_output(
    top_coactivation_store,
    dump: SimpleExactCandidateDump,
) -> None:
    """Validate reducer output shape and PMI finite values after postprocess."""

    if not hasattr(top_coactivation_store, "top_indices") or not hasattr(top_coactivation_store, "top_values"):
        return
    top_indices = top_coactivation_store.top_indices
    top_values = top_coactivation_store.top_values
    if not isinstance(top_indices, torch.Tensor) or not isinstance(top_values, torch.Tensor):
        return
    expected_shape = (dump.num_components, dump.d_sae, dump.n_latents_per_latent)
    if tuple(top_indices.shape) != expected_shape or tuple(top_values.shape) != expected_shape:
        raise ValueError("top_coactivation reducer output shape mismatch")
    if dump.mode == "pmi" and not torch.isfinite(top_values).all():
        raise ValueError("PMI top_coactivation values must be finite")


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


def _validate_simple_dump_reduce_dimensions(
    dump_inputs: CandidateDumpReducerInputs,
    mapping: GlobalTopCtxTargetMapping,
) -> None:
    for entry in dump_inputs.entries:
        metadata = entry.metadata
        if metadata.seq_len != dump_inputs.entries[0].metadata.seq_len:
            raise ValueError("candidate dump seq_len mismatch")
    if mapping.seq_targets_global.numel():
        max_target_id = int(mapping.seq_targets_global.max().item())
        if max_target_id >= dump_inputs.num_components * dump_inputs.d_sae:
            raise ValueError("top_ctx target IDs exceed candidate dump dimensions")


def _atomic_store_save(top_coactivation_store, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    top_coactivation_store.save(str(tmp_path))
    if not tmp_path.exists():
        raise ValueError("top_coactivation store did not write an artifact")
    os.replace(tmp_path, output_path)


def _atomic_torch_save(data: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    torch.save(data, tmp_path)
    os.replace(tmp_path, output_path)


def _write_mapreduce_top_coactivation_artifact(
    path: str | Path,
    *,
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    mode: str,
    total_tokens_processed: int,
) -> None:
    payload = {
        "top_indices": top_indices.detach().cpu().to(torch.int32),
        "top_values": top_values.detach().cpu().to(torch.float32),
        "freq_factors": torch.ones(int(top_indices.shape[0]) * int(top_indices.shape[1]), dtype=torch.float32),
        "total_tokens_processed": int(total_tokens_processed),
        "mode": mode,
    }
    _atomic_torch_save(payload, path)


def _atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, output_path)


def _candidate_dump_entry_tensor_bytes(entry: CandidateDumpReducerEntry) -> int:
    total = 0
    for key in ("sequence_ids", "candidate_ids", "candidate_vals"):
        value = entry.payload.get(key)
        if isinstance(value, torch.Tensor):
            total += int(value.numel() * value.element_size())
    return total


def _file_size_or_zero(path: str | Path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _start_memory_trace() -> bool:
    if tracemalloc.is_tracing():
        return False
    tracemalloc.start()
    return True


def _stop_memory_trace(started_here: bool) -> Optional[int]:
    if not tracemalloc.is_tracing():
        return None
    _current, peak = tracemalloc.get_traced_memory()
    if started_here:
        tracemalloc.stop()
    return int(peak)


def _validate_sorted_coo_payload(payload: Dict[str, Any]) -> None:
    target_ids = payload["target_ids"].to(torch.int64).cpu().tolist()
    candidate_ids = payload["candidate_ids"].to(torch.int32).cpu().tolist()
    pairs = list(zip(target_ids, candidate_ids))
    if pairs != sorted(pairs):
        raise ValueError("MapReduce shard COO records must be sorted by (target_id, candidate_id)")


if __name__ == "__main__":
    main()
