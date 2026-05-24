"""Preflight checks for distributed controller planning."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from .controller_contracts import PreflightReport
from .devices import build_device_assignments
from .shard_table import ShardRecord, build_shard_table


REQUIRED_NATIVE_EXTENSIONS = {
    "pass2_reduce": ("top_coactivation_reduce",),
    "mid_ctx": ("mid_reservoir",),
}


def run_preflight_checks(
    *,
    config_path: str | Path,
    normalized_config_hash: str,
    output_root: str | Path,
    dataset_path: str | Path,
    n_shards: Optional[int],
    resume: bool,
    worker_count: int,
    physical_ids: Optional[Iterable[int]],
    use_cpu: bool,
    selected_parts: Sequence[str],
) -> PreflightReport:
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    config_path = Path(config_path)
    output_root = Path(output_root)
    if not config_path.exists():
        raise FileNotFoundError(f"config path does not exist: {config_path}")
    if not normalized_config_hash:
        raise ValueError("normalized_config_hash must be non-empty")
    if output_root.exists() and not resume:
        raise FileExistsError("run ID collision; pass resume=True to reuse output_root")
    _check_output_writable(output_root.parent)

    shard_table = build_shard_table(dataset_path, n_shards=n_shards)
    total_shard_bytes = sum(record.shard_size_bytes for record in shard_table)

    visible_count = _visible_cuda_device_count()
    build_device_assignments(
        worker_count,
        physical_ids=physical_ids,
        visible_device_count=None if use_cpu else visible_count,
        use_cpu=use_cpu,
    )

    missing_native = native_extension_availability(selected_parts)
    unavailable = [
        name for name, available in missing_native.items() if not available
    ]
    if unavailable:
        raise RuntimeError(f"required native extensions unavailable: {unavailable}")

    rough_required_disk_bytes = _estimate_preflight_disk_bytes(
        shard_table,
        worker_count=worker_count,
        selected_parts=selected_parts,
    )
    free_disk_bytes = shutil.disk_usage(output_root.parent).free
    if free_disk_bytes < rough_required_disk_bytes:
        raise OSError(
            "insufficient disk space for distributed run: "
            f"free={free_disk_bytes} required={rough_required_disk_bytes}"
        )
    return PreflightReport(
        config_path=config_path,
        normalized_config_hash=normalized_config_hash,
        output_root=output_root,
        run_id_collision=output_root.exists(),
        free_disk_bytes=free_disk_bytes,
        rough_required_disk_bytes=rough_required_disk_bytes,
        native_extensions=missing_native,
        selected_parts=tuple(selected_parts),
        shard_count=len(shard_table),
        total_shard_bytes=total_shard_bytes,
    )


def native_extension_availability(selected_parts: Sequence[str]) -> Dict[str, bool]:
    required: set[str] = set()
    for part in selected_parts:
        required.update(REQUIRED_NATIVE_EXTENSIONS.get(part, ()))
    return {name: importlib.util.find_spec(name) is not None for name in sorted(required)}


def _estimate_preflight_disk_bytes(
    shard_table: Sequence[ShardRecord],
    *,
    worker_count: int,
    selected_parts: Sequence[str],
) -> int:
    """Conservative low-cost estimate for manifests, reports, metrics, and partials."""

    selected = set(selected_parts)
    metadata_bytes = 1_000_000 + worker_count * 250_000
    shard_bytes = sum(record.shard_size_bytes for record in shard_table)
    partial_factor = 0.0
    if not selected or "pass1" in selected:
        partial_factor += 0.05
    if not selected or "pass2" in selected or "pass2_reduce" in selected:
        partial_factor += 0.10
    if not selected or "discovery" in selected:
        partial_factor += 0.02
    return int(metadata_bytes + shard_bytes * partial_factor)


def _check_output_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".distributed_write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()


def _visible_cuda_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


__all__ = [
    "REQUIRED_NATIVE_EXTENSIONS",
    "native_extension_availability",
    "run_preflight_checks",
]
