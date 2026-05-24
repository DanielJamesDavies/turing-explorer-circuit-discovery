"""Core distributed controller planning."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .assignments import build_work_assignments
from .controller_commands import build_worker_commands
from .controller_config import (
    _candidate_dump_m_from_config,
    _resolve_config_path,
    load_and_hash_config,
)
from .controller_contracts import ControllerPlan
from .controller_dry_run import format_dry_run
from .controller_preflight import _visible_cuda_device_count, run_preflight_checks
from .controller_reports import (
    build_discovery_dry_run_estimate,
    build_h100_exact_mode_report,
    build_local_compatibility_report,
)
from .devices import build_device_assignments
from .experimental_modes import build_experimental_fast_config
from .layout import build_run_layout, create_output_layout
from .manifest import (
    CleanupPolicy,
    DistributedRunManifest,
    ManifestStatus,
    RunMode,
    generate_run_id,
    save_manifest,
)
from .shard_table import build_shard_table


def plan_distributed_run(
    *,
    config_path: str | Path,
    project_root: str | Path,
    output_base: str | Path,
    worker_count: int,
    run_mode: RunMode = RunMode.DISTRIBUTED_SIMPLE_EXACT,
    run_id: Optional[str] = None,
    resume: bool = False,
    create_layout: bool = True,
    physical_ids: Optional[Iterable[int]] = None,
    use_cpu: bool = False,
    cleanup_policy: CleanupPolicy = CleanupPolicy.KEEP_ALL,
    selected_parts: Sequence[str] = (),
    selected_shards: Optional[Iterable[int]] = None,
    pass2_sequence_ids: Optional[Sequence[int]] = None,
    discovery_seed_ids: Optional[Sequence[int]] = None,
    timestamp: Optional[datetime] = None,
) -> ControllerPlan:
    """Create a manifest, output layout, and worker commands without running work."""

    project_root_path = Path(project_root)
    config_path = Path(config_path)
    output_base = Path(output_base)
    physical_id_list = list(physical_ids) if physical_ids is not None else None
    normalized_config, config_hash = load_and_hash_config(config_path)
    weights = normalized_config["weights"]
    data = normalized_config.get("data", {})
    distributed = normalized_config.get("distributed", {})
    dataset_path = Path(data.get("dataset_path", ""))
    if not dataset_path.is_absolute():
        dataset_path = project_root_path / dataset_path
    n_shards = data.get("n_shards")

    actual_run_id = run_id or generate_run_id(
        config_hash,
        timestamp=timestamp or datetime.now(timezone.utc),
    )
    output_root = output_base / actual_run_id
    distributed_root = output_root / "distributed"

    preflight = run_preflight_checks(
        config_path=config_path,
        normalized_config_hash=config_hash,
        output_root=output_root,
        dataset_path=dataset_path,
        n_shards=n_shards,
        resume=resume,
        worker_count=worker_count,
        physical_ids=physical_id_list,
        use_cpu=use_cpu,
        selected_parts=selected_parts,
    )

    shard_table = build_shard_table(dataset_path, n_shards=n_shards)
    work_assignments = build_work_assignments(
        shard_table,
        worker_count,
        pass2_sequence_ids=pass2_sequence_ids,
        discovery_seed_ids=discovery_seed_ids,
        selected_shards=selected_shards,
    )
    devices = build_device_assignments(
        worker_count,
        physical_ids=physical_id_list,
        visible_device_count=None if use_cpu else _visible_cuda_device_count(),
        use_cpu=use_cpu,
    )
    manifest = DistributedRunManifest(
        run_id=actual_run_id,
        run_mode=run_mode,
        status=ManifestStatus.PLANNED,
        cleanup_policy=cleanup_policy,
        sampling_seed=int(distributed.get("sampling_seed", 0)) if isinstance(distributed, dict) else 0,
        created_at=(timestamp or datetime.now(timezone.utc))
        .isoformat()
        .replace("+00:00", "Z"),
        config_path=str(config_path),
        normalized_config_hash=config_hash,
        environment_overrides={},
        project_root=str(project_root_path),
        output_root=str(output_root),
        distributed_root=str(distributed_root),
        manifest_path=str(distributed_root / "manifest.json"),
        metrics_path=str(distributed_root / "reports" / "run_metrics.jsonl"),
        run_summary_path=str(distributed_root / "reports" / "run_summary.json"),
        model_path=str(_resolve_config_path(project_root_path, weights["model_path"])),
        sae_path=str(_resolve_config_path(project_root_path, weights["sae_path"])),
        dataset_path=str(dataset_path),
        worker_count=worker_count,
        devices=devices,
        shard_table=shard_table,
        work_assignments=work_assignments,
        experimental_fast=build_experimental_fast_config(
            acknowledged=bool(distributed.get("experimental_acknowledgement", False))
            if isinstance(distributed, dict)
            else False,
            exact_baseline_root=(
                distributed.get("experimental_exact_baseline_root")
                if isinstance(distributed, dict)
                else None
            ),
            quality_toggles=(
                distributed.get("experimental_quality_toggles", {})
                if isinstance(distributed, dict)
                else {}
            ),
        ),
    )
    layout = create_output_layout(manifest) if create_layout else build_run_layout(manifest)
    if create_layout:
        save_manifest(manifest, layout.manifest_path)

    worker_commands = build_worker_commands(manifest, project_root_path)
    discovery_estimate = build_discovery_dry_run_estimate(normalized_config, worker_count)
    local_compatibility = build_local_compatibility_report(
        normalized_config,
        worker_count=worker_count,
        use_cpu=use_cpu,
        physical_ids=physical_id_list,
    )
    h100_exact_mode = build_h100_exact_mode_report(
        normalized_config,
        manifest=manifest,
    )
    return ControllerPlan(
        manifest=manifest,
        layout=layout,
        preflight=preflight,
        worker_commands=worker_commands,
        discovery_estimate=discovery_estimate,
        local_compatibility=local_compatibility,
        h100_exact_mode=h100_exact_mode,
        dry_run_text=format_dry_run(
            manifest,
            preflight,
            worker_commands,
            pass2_dump_m=_candidate_dump_m_from_config(normalized_config),
            discovery_estimate=discovery_estimate,
            local_compatibility=local_compatibility,
            h100_exact_mode=h100_exact_mode,
        ),
    )


__all__ = [
    "plan_distributed_run",
]
