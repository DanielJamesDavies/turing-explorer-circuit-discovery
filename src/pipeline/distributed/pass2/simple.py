"""Simple exact reducer path for distributed pass 2."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from tqdm import tqdm

from ..interfaces import build_output_paths
from .contracts import (
    CandidateDumpReducerInputs,
    GlobalTopCtxTargetMapping,
    PmiReduceInputs,
    SimpleExactCandidateDump,
    SimpleExactReduceResult,
)
from .inputs import (
    load_candidate_dump_reducer_inputs,
    load_global_active_count,
    load_global_top_ctx_target_mapping,
    validate_candidate_dump_sequence_coverage,
    validate_global_active_count,
)
from .reports import (
    atomic_write_json,
    build_pass2_reduce_manifest_metrics,
    candidate_dump_entry_tensor_bytes,
    file_size_or_zero,
    start_memory_trace,
    stop_memory_trace,
)


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

    stage_started = time.perf_counter()
    output_paths = build_output_paths(output_root)
    print(
        f"[pass2_reduce] starting simple exact reduce dumps={len(candidate_dump_paths)} "
        f"output_root={output_root}",
        flush=True,
    )
    load_started = time.perf_counter()
    dump_inputs = load_candidate_dump_reducer_inputs(
        candidate_dump_paths,
        expected_config_hash=expected_config_hash,
        expected_mode=expected_mode,
    )
    print(
        "[pass2_reduce] candidate dump partials loaded "
        f"workers={len(dump_inputs.entries)} sequences={dump_inputs.total_sequence_count} "
        f"M={dump_inputs.m} elapsed={time.perf_counter() - load_started:.1f}s",
        flush=True,
    )
    mapping_started = time.perf_counter()
    mapping = load_global_top_ctx_target_mapping(
        top_ctx_path or output_paths.top_ctx,
        dump_inputs=dump_inputs,
    )
    print(
        "[pass2_reduce] top_ctx mapping built "
        f"replay_sequences={len(mapping.sequence_ids)} targets={mapping.seq_targets_global.numel()} "
        f"elapsed={time.perf_counter() - mapping_started:.1f}s",
        flush=True,
    )
    active_count = None
    if dump_inputs.mode == "pmi":
        active_started = time.perf_counter()
        active_count = load_global_active_count(
            latent_stats_path or output_paths.latent_stats,
            expected_config_hash=expected_config_hash,
            expected_num_components=dump_inputs.num_components,
            expected_d_sae=dump_inputs.d_sae,
        )
        print(
            f"[pass2_reduce] active_count loaded elapsed={time.perf_counter() - active_started:.1f}s",
            flush=True,
        )
    from store.top_coactivation import top_coactivation

    result = run_simple_exact_reduce_and_write(
        top_coactivation,
        dump_inputs,
        mapping,
        output_root,
        active_count=active_count,
    )
    print(
        f"[pass2_reduce] complete elapsed={time.perf_counter() - stage_started:.1f}s "
        f"artifact={result.artifact_path}",
        flush=True,
    )
    return result


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

    _validate_simple_dump_reduce_dimensions(dump_inputs, mapping)
    sequence_count = len(mapping.sequence_ids)
    candidate_ids = torch.zeros((sequence_count, dump_inputs.m), dtype=torch.int32)
    candidate_vals = torch.zeros((sequence_count, dump_inputs.m), dtype=torch.float32)
    sid_to_row_tensor = mapping.sid_to_row_tensor.to(torch.int64)
    total_rows = sum(int(entry.metadata.sequence_count) for entry in dump_inputs.entries)
    progress = tqdm(
        total=total_rows,
        desc="  [pass2_reduce:assemble_dump]",
        unit="row",
    )
    for entry in dump_inputs.entries:
        sequence_ids = entry.payload["sequence_ids"].to(torch.int64).cpu()
        worker_candidate_ids = entry.payload["candidate_ids"].to(torch.int32).cpu()
        worker_candidate_vals = entry.payload["candidate_vals"].to(torch.float32).cpu()
        chunk_rows = _candidate_dump_assembly_chunk_rows(dump_inputs.m)
        for row_start in range(0, int(sequence_ids.numel()), chunk_rows):
            row_end = min(row_start + chunk_rows, int(sequence_ids.numel()))
            chunk_sequence_ids = sequence_ids[row_start:row_end]
            destination_rows = sid_to_row_tensor[chunk_sequence_ids]
            if bool((destination_rows < 0).any()):
                bad_sequence_id = int(chunk_sequence_ids[destination_rows < 0][0].item())
                raise ValueError(f"candidate dump contains sequence ID outside global replay set: {bad_sequence_id}")
            candidate_ids[destination_rows] = worker_candidate_ids[row_start:row_end]
            candidate_vals[destination_rows] = worker_candidate_vals[row_start:row_end]
            progress.update(row_end - row_start)
    progress.close()

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

    memory_trace = start_memory_trace()
    print("[pass2_reduce] assembling global candidate dump", flush=True)
    build_started = time.perf_counter()
    dump = build_simple_exact_candidate_dump(dump_inputs, mapping)
    build_elapsed_s = time.perf_counter() - build_started
    print(f"[pass2_reduce] global candidate dump assembled elapsed={build_elapsed_s:.1f}s", flush=True)

    print("[pass2_reduce] reducing top coactivation", flush=True)
    reduce_started = time.perf_counter()
    reduce_simple_exact_candidate_dump(
        top_coactivation_store,
        dump,
        mapping,
        active_count=active_count,
    )
    reduce_elapsed_s = time.perf_counter() - reduce_started
    print(f"[pass2_reduce] top coactivation reduce complete elapsed={reduce_elapsed_s:.1f}s", flush=True)

    print(f"[pass2_reduce] writing top_coactivation -> {output_paths.top_coactivation}", flush=True)
    save_started = time.perf_counter()
    _atomic_store_save(top_coactivation_store, output_paths.top_coactivation)
    save_elapsed_s = time.perf_counter() - save_started
    print(f"[pass2_reduce] wrote top_coactivation elapsed={save_elapsed_s:.1f}s", flush=True)
    peak_cpu_memory_bytes = stop_memory_trace(memory_trace)
    print("[pass2_reduce] validating top_coactivation artifact", flush=True)
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
    atomic_write_json(report_path, report)
    print(f"[pass2_reduce] wrote reduce report -> {report_path}", flush=True)
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

    output_artifact_size_bytes = file_size_or_zero(artifact_path)
    candidate_dump_bytes = int(dump.candidate_ids.numel() * dump.candidate_ids.element_size()) + int(
        dump.candidate_vals.numel() * dump.candidate_vals.element_size()
    )
    input_dump_bytes = sum(
        candidate_dump_entry_tensor_bytes(entry)
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


def _candidate_dump_assembly_chunk_rows(candidate_width: int) -> int:
    target_bytes = int(os.environ.get("TURING_PASS2_REDUCE_ASSEMBLY_CHUNK_BYTES", str(512 * 1024 * 1024)))
    row_bytes = max(1, int(candidate_width) * (torch.empty((), dtype=torch.int32).element_size() + torch.empty((), dtype=torch.float32).element_size()))
    return max(1, target_bytes // row_bytes)


__all__ = [
    "attach_simple_exact_dump_to_store",
    "build_simple_exact_candidate_dump",
    "build_simple_exact_reduce_report",
    "reduce_simple_exact_candidate_dump",
    "run_simple_exact_reduce_and_write",
    "run_simple_exact_reduce_stage",
    "validate_pmi_reduce_inputs",
    "validate_saved_top_coactivation_artifact",
    "validate_top_coactivation_reduce_output",
]
