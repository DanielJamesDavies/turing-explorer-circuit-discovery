"""Stage execution for negative-context artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pipeline.distributed.manifest import load_manifest, save_manifest
from store.neg_context import NegCtxStats

from .inputs import BuildNegCtxFn, _empty_neg_context_like, load_negative_context_inputs
from .planning import (
    _manifest_neg_ctx_config,
    _manifest_neg_ctx_devices_from_manifest,
    _neg_ctx_part_dir,
    build_negative_context_stage_metadata,
    classify_negative_context_stage,
)


@dataclass(frozen=True)
class NegativeContextRunResult:
    neg_ctx_path: Path
    stats_path: Path
    stats: NegCtxStats


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
    compat = _compat_module()
    if build_fn is None:
        try:
            stats = compat.build_neg_ctx(
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
        compat.validate_neg_ctx_output(
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
    sanity_report = compat.build_negative_context_sanity_report(
        inputs.paths,
        output_neg_ctx,
        stats,
        metadata,
        seq_repr=inputs.seq_repr,
    )
    compat._atomic_write_json(part_dir / "neg_ctx_sanity_report.json", sanity_report)
    compat.print_negative_context_sanity_summary(sanity_report)
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
    _compat_module()._atomic_write_json(path, payload)


def _compat_module():
    import pipeline.negative_context as compat

    return compat


__all__ = [
    "NegativeContextRunResult",
    "run_negative_context_stage",
]
