"""MapReduce reducer semantics and orchestration for distributed pass 2."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from ..interfaces import build_output_paths
from ..pass2_partials import (
    CandidatePreAggregationMetadata,
    validate_candidate_preaggregation_partial,
)
from .contracts import (
    CandidatePreAggregationReducerEntry,
    CandidatePreAggregationReducerInputs,
    MapReduceReduceResult,
    MapReduceTargetShardArtifact,
    MapReduceTargetShardResult,
    Pass2ReduceSchedulerConfig,
    TargetRange,
)
from .inputs import (
    validate_candidate_preaggregation_reducer_inputs,
    validate_global_active_count,
)
from .mapreduce_io import (
    load_mapreduce_reducer_shards,
    load_mapreduce_target_shard_result,
    save_mapreduce_target_shard_result,
)
from .reports import (
    atomic_torch_save,
    atomic_write_json,
    build_pass2_reduce_manifest_metrics,
    file_size_or_zero,
    start_memory_trace,
    stop_memory_trace,
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
    input_partial_sum_bytes = sum(file_size_or_zero(path) for paths in shard_paths_by_reducer.values() for path in paths)
    timing = {
        "load_shards_s": 0.0,
        "reduce_s": 0.0,
        "shard_write_s": 0.0,
        "shard_load_s": 0.0,
        "stitch_s": 0.0,
        "pmi_s": 0.0,
        "save_s": 0.0,
    }

    memory_trace = start_memory_trace()
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
    peak_cpu_memory_bytes = stop_memory_trace(memory_trace)
    output_artifact_size_bytes = file_size_or_zero(output_paths.top_coactivation)
    output_shard_bytes = sum(file_size_or_zero(artifact.path) for artifact in artifacts)
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
    atomic_write_json(report_path, report)
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
    atomic_torch_save(payload, path)


__all__ = [
    "apply_pmi_postprocess_to_topk",
    "cleanup_mapreduce_target_shards",
    "compute_total_tokens_per_target",
    "mapreduce_target_shard_path",
    "partition_target_ranges",
    "reduce_mapreduce_target_range",
    "run_mapreduce_reduce_and_write",
    "shard_preaggregation_by_target_range",
    "stitch_mapreduce_target_shards",
    "validate_pass2_reduce_scheduler_config",
]
