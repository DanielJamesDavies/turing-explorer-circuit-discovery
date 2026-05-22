from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol, Sequence

import torch

from .distributed.interfaces import PipelineOutputPaths, build_output_paths
from .distributed.manifest import (
    DistributedRunManifest,
    NegativeContextRunConfig,
    load_manifest,
    save_manifest,
)
from .runtime import get_runtime
from config import config
from store.context import mid_ctx, neg_ctx, top_ctx
from store.neg_context import (
    NegCtxStats,
    build_neg_ctx,
    build_neg_ctx_multi_gpu,
    build_neg_ctx_single_gpu_exact,
    validate_neg_ctx_output,
)


class SeqReprLike(Protocol):
    repr_buf: torch.Tensor
    repr_mode: str
    repr_dim: int
    n_seqs: int
    n_stored: int
    is_capped: bool
    slot_to_id: Optional[torch.Tensor]
    id_to_slot: Optional[torch.Tensor]


@dataclass
class LoadedContext:
    ctx_type: str
    ctx_seq_idx: torch.Tensor
    ctx_seq_val: torch.Tensor
    num_components: int
    d_sae: int
    num_ctx_sequences: int
    mode: Optional[str] = None
    reservoir_fill: Optional[torch.Tensor] = None
    reservoir_n: Optional[torch.Tensor] = None

    def save(self, path: str | Path) -> None:
        checkpoint: Dict[str, object] = {
            "ctx_seq_idx": self.ctx_seq_idx,
            "ctx_seq_val": self.ctx_seq_val,
            "ctx_type": self.ctx_type,
        }
        if self.ctx_type == "mid":
            if self.mode is not None:
                checkpoint["mode"] = self.mode
            checkpoint["num_ctx_sequences"] = self.num_ctx_sequences
            if self.reservoir_fill is not None:
                checkpoint["reservoir_fill"] = self.reservoir_fill
            if self.reservoir_n is not None:
                checkpoint["reservoir_n"] = self.reservoir_n
        torch.save(checkpoint, path)


@dataclass
class LoadedSeqRepr:
    repr_buf: torch.Tensor
    repr_mode: str
    repr_dim: int
    n_seqs: int
    n_stored: int
    is_capped: bool
    slot_to_id: Optional[torch.Tensor] = None
    id_to_slot: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class NegativeContextInputs:
    top_ctx: LoadedContext
    mid_ctx: LoadedContext
    seq_repr: SeqReprLike
    paths: PipelineOutputPaths


@dataclass(frozen=True)
class NegativeContextRunResult:
    neg_ctx_path: Path
    stats_path: Path
    stats: NegCtxStats


@dataclass(frozen=True)
class NegativeContextComparisonResult:
    report_path: Path
    report: Dict[str, object]


@dataclass(frozen=True)
class NegativeContextStagePlan:
    output_root: Path
    part_dir: Path
    metadata: Dict[str, object]
    resume_status: str
    reason: str


BuildNegCtxFn = Callable[
    [SeqReprLike, LoadedContext, LoadedContext, LoadedContext],
    NegCtxStats,
]


def load_negative_context_inputs(
    output_root: str | Path = "outputs",
    *,
    expected_config_hash: Optional[str] = None,
) -> NegativeContextInputs:
    """Load and validate merged pass-1 artifacts for a standalone neg_ctx stage."""

    paths = build_output_paths(output_root)
    _require_artifacts(
        {
            "top_ctx": paths.top_ctx,
            "mid_ctx": paths.mid_ctx,
            "seq_repr": paths.seq_repr,
        }
    )
    top_payload = _load_torch_payload(paths.top_ctx, expected_config_hash=expected_config_hash)
    mid_payload = _load_torch_payload(paths.mid_ctx, expected_config_hash=expected_config_hash)
    seq_payload = _load_torch_payload(paths.seq_repr, expected_config_hash=expected_config_hash)

    loaded_top_ctx = _context_from_payload(top_payload, expected_ctx_type="top")
    loaded_mid_ctx = _context_from_payload(mid_payload, expected_ctx_type="mid")
    loaded_seq_repr = _seq_repr_from_payload(seq_payload)
    _validate_negative_context_inputs(loaded_top_ctx, loaded_mid_ctx, loaded_seq_repr)
    return NegativeContextInputs(
        top_ctx=loaded_top_ctx,
        mid_ctx=loaded_mid_ctx,
        seq_repr=loaded_seq_repr,
        paths=paths,
    )


def run_negative_context_stage(
    output_root: str | Path = "outputs",
    *,
    expected_config_hash: Optional[str] = None,
    manifest_path: str | Path | None = None,
    resume: bool = False,
    dry_run: bool = False,
    build_fn: Optional[BuildNegCtxFn] = None,
) -> NegativeContextRunResult:
    """Build neg_ctx from merged pass-1 artifacts under a run root."""

    manifest = load_manifest(manifest_path) if manifest_path is not None else None
    effective_config_hash = expected_config_hash or (
        manifest.normalized_config_hash if manifest is not None else None
    )
    inputs = load_negative_context_inputs(
        output_root,
        expected_config_hash=effective_config_hash,
    )
    output_neg_ctx = _empty_neg_context_like(inputs.top_ctx)
    selected_devices = _manifest_neg_ctx_devices_from_manifest(manifest)
    metadata = build_negative_context_stage_metadata(
        inputs,
        manifest=manifest,
        expected_config_hash=effective_config_hash,
        selected_devices=selected_devices,
    )
    plan = classify_negative_context_stage(inputs.paths.run_root, metadata=metadata)
    if dry_run:
        return NegativeContextRunResult(
            neg_ctx_path=inputs.paths.neg_ctx,
            stats_path=inputs.paths.run_root / "neg_ctx_stats.json",
            stats=NegCtxStats(backend=str(metadata["backend"])),
        )
    if resume and plan.resume_status == "completed":
        return NegativeContextRunResult(
            neg_ctx_path=inputs.paths.neg_ctx,
            stats_path=inputs.paths.run_root / "neg_ctx_stats.json",
            stats=NegCtxStats(backend=str(metadata["backend"])),
        )
    part_dir = _neg_ctx_part_dir(inputs.paths.run_root)
    _write_part_marker(part_dir / "started.json", "running", metadata)
    if build_fn is None:
        try:
            stats = build_neg_ctx(
                inputs.seq_repr,
                inputs.top_ctx,
                inputs.mid_ctx,
                output_neg_ctx,
                selected_devices=selected_devices,
            )
        except Exception as error:
            _write_part_marker(part_dir / "failed.json", "failed", metadata, error=str(error))
            raise
    else:
        try:
            stats = build_fn(inputs.seq_repr, inputs.top_ctx, inputs.mid_ctx, output_neg_ctx)
        except Exception as error:
            _write_part_marker(part_dir / "failed.json", "failed", metadata, error=str(error))
            raise
    try:
        validate_neg_ctx_output(
            output_neg_ctx,
            total_n_seqs=inputs.seq_repr.n_seqs,
            n_sequences=output_neg_ctx.num_ctx_sequences,
        )
    except Exception as error:
        _write_part_marker(part_dir / "failed.json", "failed", metadata, error=str(error))
        raise
    inputs.paths.run_root.mkdir(parents=True, exist_ok=True)
    output_neg_ctx.save(inputs.paths.neg_ctx)
    stats.save(str(inputs.paths.run_root / "neg_ctx_stats.json"))
    sanity_report = build_negative_context_sanity_report(
        inputs.paths,
        output_neg_ctx,
        stats,
        metadata,
        seq_repr=inputs.seq_repr,
    )
    _atomic_write_json(part_dir / "neg_ctx_sanity_report.json", sanity_report)
    print_negative_context_sanity_summary(sanity_report)
    _write_part_marker(
        part_dir / "completed.json",
        "completed",
        metadata,
        artifacts={
            "neg_ctx": str(inputs.paths.neg_ctx),
            "neg_ctx_stats": str(inputs.paths.run_root / "neg_ctx_stats.json"),
            "sanity_report": str(part_dir / "neg_ctx_sanity_report.json"),
        },
    )
    if manifest is not None:
        updated_manifest = manifest.model_copy(
            update={"neg_ctx": _manifest_neg_ctx_config(metadata)}
        )
        save_manifest(updated_manifest, manifest.manifest_path)
    return NegativeContextRunResult(
        neg_ctx_path=inputs.paths.neg_ctx,
        stats_path=inputs.paths.run_root / "neg_ctx_stats.json",
        stats=stats,
    )


def plan_negative_context_stage(
    output_root: str | Path = "outputs",
    *,
    expected_config_hash: Optional[str] = None,
    manifest_path: str | Path | None = None,
) -> NegativeContextStagePlan:
    manifest = load_manifest(manifest_path) if manifest_path is not None else None
    effective_config_hash = expected_config_hash or (
        manifest.normalized_config_hash if manifest is not None else None
    )
    inputs = load_negative_context_inputs(
        output_root,
        expected_config_hash=None,
    )
    metadata = build_negative_context_stage_metadata(
        inputs,
        manifest=manifest,
        expected_config_hash=effective_config_hash,
        selected_devices=_manifest_neg_ctx_devices_from_manifest(manifest),
    )
    classification = classify_negative_context_stage(inputs.paths.run_root, metadata=metadata)
    return NegativeContextStagePlan(
        output_root=inputs.paths.run_root,
        part_dir=_neg_ctx_part_dir(inputs.paths.run_root),
        metadata=metadata,
        resume_status=classification.resume_status,
        reason=classification.reason,
    )


@dataclass(frozen=True)
class NegativeContextStageClassification:
    resume_status: str
    reason: str


def classify_negative_context_stage(
    run_root: str | Path,
    *,
    metadata: Dict[str, object],
) -> NegativeContextStageClassification:
    root = Path(run_root)
    part_dir = _neg_ctx_part_dir(root)
    failed_marker = part_dir / "failed.json"
    completed_marker = part_dir / "completed.json"
    if failed_marker.exists():
        return NegativeContextStageClassification("failed", "failed marker exists")
    if not completed_marker.exists():
        return NegativeContextStageClassification("missing", "completed marker missing")
    required_outputs = [
        root / "neg_ctx.pt",
        root / "neg_ctx_stats.json",
        part_dir / "neg_ctx_sanity_report.json",
    ]
    if any(not path.exists() for path in required_outputs):
        return NegativeContextStageClassification("missing", "required neg_ctx outputs missing")
    try:
        marker = json.loads(completed_marker.read_text(encoding="utf-8"))
        sanity = json.loads((part_dir / "neg_ctx_sanity_report.json").read_text(encoding="utf-8"))
    except Exception:
        return NegativeContextStageClassification("stale", "status metadata is unreadable")
    if marker.get("metadata") != metadata:
        return NegativeContextStageClassification("stale", "completed marker metadata mismatch")
    if sanity.get("metadata") != metadata:
        return NegativeContextStageClassification("stale", "sanity report metadata mismatch")
    return NegativeContextStageClassification("completed", "outputs and metadata match")


def build_negative_context_stage_metadata(
    inputs: NegativeContextInputs,
    *,
    manifest: DistributedRunManifest | None,
    expected_config_hash: Optional[str],
    selected_devices: Sequence[int] | None,
) -> Dict[str, object]:
    backend = str(config.latents.neg_ctx.backend or "single_gpu_exact")
    configured_devices = [str(device) for device in list(config.latents.neg_ctx.devices)]
    if selected_devices:
        selected_device_labels = [f"cuda:{device}" for device in selected_devices]
        device_source = "manifest_declared_devices"
    elif configured_devices:
        selected_device_labels = configured_devices
        device_source = "config_override"
    elif backend in {"multi_gpu_exact", "multi_gpu_index_sharded_exact"}:
        selected_device_labels = []
        device_source = "standalone_all_visible"
    else:
        selected_device_labels = [str(config.hardware.ann_device or "auto")]
        device_source = "single_device"
    return {
        "schema_version": 1,
        "run_id": manifest.run_id if manifest is not None else None,
        "config_hash": expected_config_hash
        or (manifest.normalized_config_hash if manifest is not None else None),
        "backend": backend,
        "used_backend": backend,
        "selected_devices": selected_device_labels,
        "device_selection_source": device_source,
        "n_neighbors": int(config.latents.neg_ctx.n_neighbors or 512),
        "n_sequences": int(config.latents.neg_ctx.n_sequences or 64),
        "min_pos_ctx": int(config.latents.neg_ctx.min_pos_ctx or 8),
        "repr_mode": str(config.latents.neg_ctx.repr_mode or "mean_pool"),
        "max_repr_seqs": config.latents.neg_ctx.max_repr_seqs,
        "memory_guardrail_fraction": float(config.latents.neg_ctx.memory_guardrail_fraction),
        "fail_on_memory_guardrail": bool(config.latents.neg_ctx.fail_on_memory_guardrail),
        "inputs": {
            "top_ctx": _artifact_metadata(inputs.paths.top_ctx),
            "mid_ctx": _artifact_metadata(inputs.paths.mid_ctx),
            "seq_repr": _artifact_metadata(inputs.paths.seq_repr),
        },
    }


def build_negative_context_sanity_report(
    paths: PipelineOutputPaths,
    output_neg_ctx: LoadedContext,
    stats: NegCtxStats,
    metadata: Dict[str, object],
    *,
    seq_repr: SeqReprLike,
) -> Dict[str, object]:
    validation = _neg_ctx_validation_summary(
        output_neg_ctx,
        total_n_seqs=seq_repr.n_seqs,
        n_sequences=output_neg_ctx.num_ctx_sequences,
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "metadata": metadata,
        "artifacts": {
            "neg_ctx": str(paths.neg_ctx),
            "neg_ctx_stats": str(paths.run_root / "neg_ctx_stats.json"),
        },
        "backend": stats.backend,
        "devices": stats.devices,
        "shape": list(output_neg_ctx.ctx_seq_idx.shape),
        "dtype": {
            "ctx_seq_idx": str(output_neg_ctx.ctx_seq_idx.dtype),
            "ctx_seq_val": str(output_neg_ctx.ctx_seq_val.dtype),
        },
        "seq_repr": {
            "n_seqs": int(seq_repr.n_seqs),
            "n_stored": int(seq_repr.n_stored),
            "repr_dim": int(seq_repr.repr_dim),
            "is_capped": bool(seq_repr.is_capped),
            "cap_percent": (
                float(seq_repr.n_stored) / float(seq_repr.n_seqs) * 100.0
                if int(seq_repr.n_seqs) > 0
                else 0.0
            ),
        },
        "populated_rows": _populated_row_count(output_neg_ctx),
        "zero_negative_rows": int(stats.n_latents_zero_negatives),
        "valid_entry_count": validation["valid_entry_count"],
        "validation": validation,
        "fill_distribution": _fill_summary(output_neg_ctx),
        "timing_ms": _stats_timing_ms(stats),
        "memory": {
            "ann_index_memory_estimate_bytes": stats.ann_index_memory_estimate_bytes,
            "ann_query_working_memory_bytes": stats.ann_query_working_memory_bytes,
            "ann_total_memory_estimate_bytes": stats.ann_total_memory_estimate_bytes,
            "ann_memory_guardrail_fraction": stats.ann_memory_guardrail_fraction,
            "ann_memory_guardrail_limit_bytes": stats.ann_memory_guardrail_limit_bytes,
        },
    }


def print_negative_context_sanity_summary(report: Dict[str, object]) -> None:
    fill = report.get("fill_distribution", {})
    timing = report.get("timing_ms", {})
    seq_repr = report.get("seq_repr", {})
    validation = report.get("validation", {})
    memory = report.get("memory", {})
    print(
        "  [neg_ctx] summary "
        f"backend={report.get('backend')} "
        f"devices={','.join(str(device) for device in report.get('devices', []))} "
        f"populated_rows={report.get('populated_rows')} "
        f"fill_mean={float(fill.get('mean', 0.0)):.1f} "
        f"fill_min={fill.get('min', 0)} "
        f"fill_max={fill.get('max', 0)} "
        f"zero_neg={report.get('zero_negative_rows', 0)} "
        f"invalid_seq={validation.get('invalid_sequence_count', 0)} "
        f"non_finite={validation.get('non_finite_similarity_count', 0)} "
        f"seq_repr={seq_repr.get('n_stored', 0)}/{seq_repr.get('n_seqs', 0)} "
        f"cap={float(seq_repr.get('cap_percent', 0.0)):.1f}% "
        f"ann_mem_gib={int(memory.get('ann_total_memory_estimate_bytes', 0)) / 1024**3:.2f} "
        f"total_ms={float(timing.get('total_ms', 0.0)):.1f}"
    )


def compare_negative_context_backends(
    output_root: str | Path = "outputs",
    *,
    expected_config_hash: Optional[str] = None,
    manifest_path: str | Path | None = None,
    selected_devices: Sequence[int | str | torch.device] | None = None,
    single_build_fn: BuildNegCtxFn = build_neg_ctx_single_gpu_exact,
    multi_build_fn: Callable[..., NegCtxStats] = build_neg_ctx_multi_gpu,
    sample_rows: int = 8,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> NegativeContextComparisonResult:
    """Build both exact neg_ctx backends from one input set and compare outputs."""

    inputs = load_negative_context_inputs(
        output_root,
        expected_config_hash=expected_config_hash
        or (load_manifest(manifest_path).normalized_config_hash if manifest_path is not None else None),
    )
    devices = (
        list(selected_devices)
        if selected_devices is not None
        else _manifest_neg_ctx_devices(manifest_path)
    )
    single_output = _empty_neg_context_like(inputs.top_ctx)
    multi_output = _empty_neg_context_like(inputs.top_ctx)
    single_stats = single_build_fn(
        inputs.seq_repr,
        inputs.top_ctx,
        inputs.mid_ctx,
        single_output,
    )
    multi_stats = multi_build_fn(
        inputs.seq_repr,
        inputs.top_ctx,
        inputs.mid_ctx,
        multi_output,
        selected_devices=devices,
    )
    report = build_negative_context_comparison_report(
        single_output,
        multi_output,
        single_stats=single_stats,
        multi_stats=multi_stats,
        sample_rows=sample_rows,
        atol=atol,
        rtol=rtol,
    )
    report_path = inputs.paths.run_root / "neg_ctx_equivalence_report.json"
    _atomic_write_json(report_path, report)
    return NegativeContextComparisonResult(report_path=report_path, report=report)


def build_negative_context_comparison_report(
    single_output: LoadedContext,
    multi_output: LoadedContext,
    *,
    single_stats: NegCtxStats,
    multi_stats: NegCtxStats,
    sample_rows: int = 8,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Dict[str, object]:
    """Compare two exact neg_ctx artifacts and summarize quality/timing."""

    shape_match = single_output.ctx_seq_idx.shape == multi_output.ctx_seq_idx.shape
    dtype_match = (
        single_output.ctx_seq_idx.dtype == multi_output.ctx_seq_idx.dtype
        and single_output.ctx_seq_val.dtype == multi_output.ctx_seq_val.dtype
    )
    indices_equal = torch.equal(single_output.ctx_seq_idx, multi_output.ctx_seq_idx)
    values_exact = torch.equal(single_output.ctx_seq_val, multi_output.ctx_seq_val)
    values_close = shape_match and torch.allclose(
        single_output.ctx_seq_val.float(),
        multi_output.ctx_seq_val.float(),
        atol=atol,
        rtol=rtol,
    )
    max_abs_diff = 0.0
    if shape_match and single_output.ctx_seq_val.numel():
        max_abs_diff = float(
            (single_output.ctx_seq_val.float() - multi_output.ctx_seq_val.float())
            .abs()
            .max()
            .item()
        )
    single_fill = _fill_summary(single_output)
    multi_fill = _fill_summary(multi_output)
    populated_single = _populated_row_count(single_output)
    populated_multi = _populated_row_count(multi_output)
    exact_equivalent = bool(
        shape_match
        and dtype_match
        and indices_equal
        and values_close
        and populated_single == populated_multi
        and single_fill == multi_fill
    )
    return {
        "schema_version": 1,
        "status": "equivalent" if exact_equivalent else "different",
        "exact_equivalent": exact_equivalent,
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "indices_equal": indices_equal,
        "values_exact": values_exact,
        "values_close": values_close,
        "value_tolerance": {"atol": float(atol), "rtol": float(rtol)},
        "max_abs_value_diff": max_abs_diff,
        "populated_rows": {
            "single_gpu_exact": populated_single,
            "multi_gpu_exact": populated_multi,
            "match": populated_single == populated_multi,
        },
        "fill_distribution": {
            "single_gpu_exact": single_fill,
            "multi_gpu_exact": multi_fill,
            "match": single_fill == multi_fill,
        },
        "sample_rows": _sample_row_comparisons(
            single_output,
            multi_output,
            limit=sample_rows,
        ),
        "timing_ms": {
            "single_gpu_exact": _stats_timing_ms(single_stats),
            "multi_gpu_exact": _stats_timing_ms(multi_stats),
        },
        "ordering_or_tie_note": (
            "No ordering/tie differences detected."
            if indices_equal and values_close
            else "Differences may indicate backend drift or device-level tie ordering; inspect sample_rows."
        ),
    }


def build_negative_contexts(output_root: str | Path = "outputs") -> None:
    runtime = get_runtime()
    output_paths = build_output_paths(output_root)
    print("--- ANN Step: Building Negative Contexts ---")
    assert runtime.seq_repr is not None

    try:
        neg_stats: NegCtxStats = build_neg_ctx(runtime.seq_repr, top_ctx, mid_ctx, neg_ctx)
        output_paths.run_root.mkdir(parents=True, exist_ok=True)
        neg_ctx.save(str(output_paths.neg_ctx))
        neg_stats.save(str(output_paths.run_root / "neg_ctx_stats.json"))
        neg_stats.print_summary(neg_ctx.num_ctx_sequences)
        print(f"  ✓ neg_ctx built and saved to {output_paths.neg_ctx}")
    except ImportError as error:
        print(f"  ✗ neg_ctx skipped: {error}")
    print("")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build negative contexts from merged pass-1 artifacts."
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Run root containing top_ctx.pt, mid_ctx.pt, and seq_repr.pt",
    )
    parser.add_argument(
        "--expected-config-hash",
        default=None,
        help="Optional config hash to validate when artifact metadata includes one",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional distributed manifest; neg_ctx uses manifest physical devices by default",
    )
    parser.add_argument(
        "--compare-backends",
        action="store_true",
        help="Build single_gpu_exact and multi_gpu_exact and write an equivalence report",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip neg_ctx rebuild when completed outputs and status metadata match",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned neg_ctx metadata and resume classification without building",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        plan = plan_negative_context_stage(
            args.output_root,
            expected_config_hash=args.expected_config_hash,
            manifest_path=args.manifest,
        )
        print(json.dumps({
            "output_root": str(plan.output_root),
            "part_dir": str(plan.part_dir),
            "resume_status": plan.resume_status,
            "reason": plan.reason,
            "metadata": plan.metadata,
        }, indent=2))
        return
    if args.compare_backends:
        result = compare_negative_context_backends(
            args.output_root,
            expected_config_hash=args.expected_config_hash,
            manifest_path=args.manifest,
        )
        print(f"  ✓ neg_ctx equivalence report saved to {result.report_path}")
        return
    result = run_negative_context_stage(
        args.output_root,
        expected_config_hash=args.expected_config_hash,
        manifest_path=args.manifest,
        resume=args.resume,
    )
    result.stats.print_summary(int(configured_neg_ctx_sequences(result.neg_ctx_path)))
    print(f"  ✓ neg_ctx saved to {result.neg_ctx_path}")
    print(f"  ✓ neg_ctx stats saved to {result.stats_path}")


def configured_neg_ctx_sequences(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["ctx_seq_idx"].shape[2])


def _require_artifacts(paths: Dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        details = ", ".join(f"{name} ({paths[name]})" for name in missing)
        raise FileNotFoundError(f"missing required pass-1 artifact(s): {details}")


def _load_torch_payload(
    path: Path,
    *,
    expected_config_hash: Optional[str],
) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact payload must be a dict: {path}")
    _validate_config_hash_if_present(payload, path, expected_config_hash)
    return payload


def _validate_config_hash_if_present(
    payload: Dict[str, object],
    path: Path,
    expected_config_hash: Optional[str],
) -> None:
    if expected_config_hash is None:
        return
    metadata = payload.get("metadata")
    observed = payload.get("config_hash")
    if observed is None and isinstance(metadata, dict):
        observed = metadata.get("config_hash")
    if observed is None:
        raise ValueError(f"artifact config hash missing for {path}")
    if observed is not None and str(observed) != expected_config_hash:
        raise ValueError(f"artifact config hash mismatch for {path}")


def _context_from_payload(
    payload: Dict[str, object],
    *,
    expected_ctx_type: str,
) -> LoadedContext:
    ctx_seq_idx = payload.get("ctx_seq_idx")
    ctx_seq_val = payload.get("ctx_seq_val")
    if not isinstance(ctx_seq_idx, torch.Tensor) or not isinstance(ctx_seq_val, torch.Tensor):
        raise ValueError(f"{expected_ctx_type}_ctx artifact must contain context tensors")
    if ctx_seq_idx.ndim != 3 or ctx_seq_val.ndim != 3:
        raise ValueError(f"{expected_ctx_type}_ctx tensors must be rank-3")
    if ctx_seq_idx.shape != ctx_seq_val.shape:
        raise ValueError(f"{expected_ctx_type}_ctx tensor shape mismatch")
    if payload.get("ctx_type") != expected_ctx_type:
        raise ValueError(f"expected ctx_type={expected_ctx_type!r}")
    if not torch.is_floating_point(ctx_seq_val):
        raise ValueError(f"{expected_ctx_type}_ctx values must be floating point")
    if not torch.isfinite(ctx_seq_val.float()).all():
        raise ValueError(f"{expected_ctx_type}_ctx values contain non-finite entries")
    if (ctx_seq_idx < 0).any():
        raise ValueError(f"{expected_ctx_type}_ctx sequence IDs must be non-negative")
    return LoadedContext(
        ctx_type=expected_ctx_type,
        ctx_seq_idx=ctx_seq_idx.to(torch.int32).cpu(),
        ctx_seq_val=ctx_seq_val.cpu(),
        num_components=int(ctx_seq_idx.shape[0]),
        d_sae=int(ctx_seq_idx.shape[1]),
        num_ctx_sequences=int(ctx_seq_idx.shape[2]),
        mode=str(payload["mode"]) if "mode" in payload else None,
        reservoir_fill=payload.get("reservoir_fill")
        if isinstance(payload.get("reservoir_fill"), torch.Tensor)
        else None,
        reservoir_n=payload.get("reservoir_n")
        if isinstance(payload.get("reservoir_n"), torch.Tensor)
        else None,
    )


def _seq_repr_from_payload(payload: Dict[str, object]) -> LoadedSeqRepr:
    repr_buf = payload.get("repr_buf")
    if not isinstance(repr_buf, torch.Tensor):
        raise ValueError("seq_repr artifact must contain repr_buf")
    if repr_buf.ndim != 2:
        raise ValueError("seq_repr repr_buf must be rank-2")
    if not torch.isfinite(repr_buf.float()).all():
        raise ValueError("seq_repr repr_buf contains non-finite entries")
    n_seqs = int(payload.get("n_seqs", 0))
    n_stored = int(payload.get("n_stored", n_seqs))
    repr_dim = int(payload.get("repr_dim", repr_buf.shape[1]))
    is_capped = bool(payload.get("is_capped", False))
    if n_seqs < 1:
        raise ValueError("seq_repr n_seqs must be positive")
    if n_stored < 1 or n_stored > n_seqs:
        raise ValueError("seq_repr n_stored must be in [1, n_seqs]")
    if repr_buf.shape != (n_stored + 1, repr_dim):
        raise ValueError("seq_repr repr_buf shape does not match n_stored/repr_dim")

    slot_to_id = payload.get("slot_to_id")
    id_to_slot = payload.get("id_to_slot")
    if is_capped:
        if not isinstance(slot_to_id, torch.Tensor) or not isinstance(id_to_slot, torch.Tensor):
            raise ValueError("capped seq_repr requires slot_to_id and id_to_slot")
        _validate_seq_repr_cap_mapping(slot_to_id, id_to_slot, n_seqs, n_stored)
        loaded_slot_to_id = slot_to_id.to(torch.int64).cpu()
        loaded_id_to_slot = id_to_slot.to(torch.int32).cpu()
    else:
        loaded_slot_to_id = None
        loaded_id_to_slot = None

    return LoadedSeqRepr(
        repr_buf=repr_buf.cpu(),
        repr_mode=str(payload.get("repr_mode", "mean_pool")),
        repr_dim=repr_dim,
        n_seqs=n_seqs,
        n_stored=n_stored,
        is_capped=is_capped,
        slot_to_id=loaded_slot_to_id,
        id_to_slot=loaded_id_to_slot,
    )


def _validate_seq_repr_cap_mapping(
    slot_to_id: torch.Tensor,
    id_to_slot: torch.Tensor,
    n_seqs: int,
    n_stored: int,
) -> None:
    if slot_to_id.shape != (n_stored + 1,):
        raise ValueError("seq_repr slot_to_id shape mismatch")
    if id_to_slot.shape != (n_seqs + 1,):
        raise ValueError("seq_repr id_to_slot shape mismatch")
    slot_to_id_i64 = slot_to_id.to(torch.int64)
    id_to_slot_i64 = id_to_slot.to(torch.int64)
    if int(slot_to_id_i64[0].item()) != 0 or int(id_to_slot_i64[0].item()) != 0:
        raise ValueError("seq_repr cap mappings must keep sentinel zero")
    selected = slot_to_id_i64[1:]
    if ((selected < 1) | (selected > n_seqs)).any():
        raise ValueError("seq_repr slot_to_id contains out-of-range sequence IDs")
    expected_slots = torch.arange(1, n_stored + 1, dtype=torch.int64)
    if not torch.equal(id_to_slot_i64[selected], expected_slots):
        raise ValueError("seq_repr cap mappings are inconsistent")


def _validate_negative_context_inputs(
    loaded_top_ctx: LoadedContext,
    loaded_mid_ctx: LoadedContext,
    loaded_seq_repr: SeqReprLike,
) -> None:
    if loaded_top_ctx.num_components != loaded_mid_ctx.num_components:
        raise ValueError("top_ctx and mid_ctx component counts differ")
    if loaded_top_ctx.d_sae != loaded_mid_ctx.d_sae:
        raise ValueError("top_ctx and mid_ctx SAE widths differ")
    if loaded_top_ctx.num_ctx_sequences < 1 or loaded_mid_ctx.num_ctx_sequences < 1:
        raise ValueError("context artifacts must have at least one context slot")
    max_sequence_id = max(
        int(loaded_top_ctx.ctx_seq_idx.max().item()),
        int(loaded_mid_ctx.ctx_seq_idx.max().item()),
    )
    if max_sequence_id > loaded_seq_repr.n_seqs:
        raise ValueError("context sequence ID exceeds seq_repr n_seqs")


def _empty_neg_context_like(loaded_top_ctx: LoadedContext) -> LoadedContext:
    n_sequences = int(neg_ctx.num_ctx_sequences)
    return LoadedContext(
        ctx_type="neg",
        ctx_seq_idx=torch.zeros(
            (loaded_top_ctx.num_components, loaded_top_ctx.d_sae, n_sequences),
            dtype=torch.int32,
        ),
        ctx_seq_val=torch.zeros(
            (loaded_top_ctx.num_components, loaded_top_ctx.d_sae, n_sequences),
            dtype=torch.float32,
        ),
        num_components=loaded_top_ctx.num_components,
        d_sae=loaded_top_ctx.d_sae,
        num_ctx_sequences=n_sequences,
    )


def _manifest_neg_ctx_devices(manifest_path: str | Path | None) -> list[int] | None:
    if manifest_path is None:
        return None
    manifest = load_manifest(manifest_path)
    return _manifest_neg_ctx_devices_from_manifest(manifest)


def _manifest_neg_ctx_devices_from_manifest(
    manifest: DistributedRunManifest | None,
) -> list[int] | None:
    if manifest is None:
        return None
    physical_ids = [
        int(device.physical_id)
        for device in sorted(manifest.devices, key=lambda assignment: assignment.worker_id)
        if device.physical_id is not None
    ]
    if not physical_ids:
        return None
    return physical_ids


def _manifest_neg_ctx_config(metadata: Dict[str, object]) -> NegativeContextRunConfig:
    return NegativeContextRunConfig(
        backend=str(metadata["backend"]),
        selected_devices=[str(device) for device in metadata["selected_devices"]],
        device_selection_source=metadata["device_selection_source"],  # type: ignore[arg-type]
        n_neighbors=int(metadata["n_neighbors"]),
        n_sequences=int(metadata["n_sequences"]),
        min_pos_ctx=int(metadata["min_pos_ctx"]),
        repr_mode=str(metadata["repr_mode"]),
        max_repr_seqs=metadata["max_repr_seqs"],  # type: ignore[arg-type]
        memory_guardrail_fraction=float(metadata["memory_guardrail_fraction"]),
        fail_on_memory_guardrail=bool(metadata["fail_on_memory_guardrail"]),
    )


def _neg_ctx_part_dir(run_root: str | Path) -> Path:
    return Path(run_root) / "distributed" / "parts" / "neg_ctx"


def _artifact_metadata(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _write_part_marker(
    path: Path,
    status: str,
    metadata: Dict[str, object],
    *,
    artifacts: Optional[Dict[str, str]] = None,
    error: Optional[str] = None,
) -> None:
    payload: Dict[str, object] = {
        "schema_version": 1,
        "part": "neg_ctx",
        "status": status,
        "metadata": metadata,
        "artifacts": artifacts or {},
    }
    if error is not None:
        payload["error"] = error
    _atomic_write_json(path, payload)


def _populated_row_count(output: LoadedContext) -> int:
    row_has_entries = (output.ctx_seq_idx > 0).any(dim=2)
    return int(row_has_entries.sum().item())


def _fill_summary(output: LoadedContext) -> Dict[str, object]:
    fill_counts = (output.ctx_seq_idx > 0).sum(dim=2).reshape(-1).to(torch.int64)
    histogram = torch.bincount(fill_counts).cpu()
    return {
        "min": int(fill_counts.min().item()) if fill_counts.numel() else 0,
        "max": int(fill_counts.max().item()) if fill_counts.numel() else 0,
        "mean": float(fill_counts.float().mean().item()) if fill_counts.numel() else 0.0,
        "histogram": {
            str(idx): int(count.item())
            for idx, count in enumerate(histogram)
        },
    }


def _neg_ctx_validation_summary(
    output: LoadedContext,
    *,
    total_n_seqs: int,
    n_sequences: int,
) -> Dict[str, object]:
    idx = output.ctx_seq_idx
    vals = output.ctx_seq_val.float()
    populated = idx > 0
    non_finite = ~torch.isfinite(vals)
    invalid_sequence = populated & ((idx < 1) | (idx > total_n_seqs))
    negative_similarity = vals < 0
    value_without_sequence = (vals > 0) & ~populated
    valid_entries = populated & (vals > 0) & ~invalid_sequence & ~non_finite & ~negative_similarity
    return {
        "checked": True,
        "total_rows": int(idx.shape[0] * idx.shape[1]),
        "configured_n_sequences": int(n_sequences),
        "invalid_sequence_count": int(invalid_sequence.sum().item()),
        "non_finite_similarity_count": int(non_finite.sum().item()),
        "negative_similarity_count": int(negative_similarity.sum().item()),
        "value_without_sequence_count": int(value_without_sequence.sum().item()),
        "valid_entry_count": int(valid_entries.sum().item()),
    }


def _sample_row_comparisons(
    single_output: LoadedContext,
    multi_output: LoadedContext,
    *,
    limit: int,
) -> list[Dict[str, object]]:
    populated = torch.nonzero(
        (single_output.ctx_seq_idx > 0).any(dim=2)
        | (multi_output.ctx_seq_idx > 0).any(dim=2),
        as_tuple=False,
    )
    samples: list[Dict[str, object]] = []
    for comp_idx, latent_idx in populated[: max(0, limit)].tolist():
        single_ids = single_output.ctx_seq_idx[comp_idx, latent_idx]
        multi_ids = multi_output.ctx_seq_idx[comp_idx, latent_idx]
        single_vals = single_output.ctx_seq_val[comp_idx, latent_idx].float()
        multi_vals = multi_output.ctx_seq_val[comp_idx, latent_idx].float()
        samples.append(
            {
                "component": int(comp_idx),
                "latent": int(latent_idx),
                "ids_equal": bool(torch.equal(single_ids, multi_ids)),
                "values_close": bool(torch.allclose(single_vals, multi_vals)),
                "single_ids": single_ids.tolist(),
                "multi_ids": multi_ids.tolist(),
                "single_values": [float(value) for value in single_vals.tolist()],
                "multi_values": [float(value) for value in multi_vals.tolist()],
            }
        )
    return samples


def _stats_timing_ms(stats: NegCtxStats) -> Dict[str, float]:
    return {
        "index_build_ms": round(stats.t_index_build * 1000, 1),
        "pos_collect_ms": round(stats.t_pos_collect * 1000, 1),
        "qmat_build_ms": round(stats.t_qmat_build * 1000, 1),
        "query_ms": round(stats.t_query * 1000, 1),
        "filter_ms": round(stats.t_filter * 1000, 1),
        "write_ms": round(stats.t_write * 1000, 1),
        "total_ms": round(stats.t_total * 1000, 1),
    }


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    main()
