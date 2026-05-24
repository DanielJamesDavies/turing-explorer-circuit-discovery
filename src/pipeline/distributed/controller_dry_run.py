"""Dry-run text formatting for distributed controller plans."""

from __future__ import annotations

from typing import Optional, Sequence

from .controller_contracts import (
    DiscoveryDryRunEstimate,
    H100ExactModeReport,
    LocalCompatibilityReport,
    PreflightReport,
    WorkerCommand,
)
from .controller_reports import (
    _format_discovery_dry_run_estimate,
    _format_h100_exact_mode_report,
    _format_local_compatibility_report,
)
from .manifest import DistributedRunManifest
from .operating_modes import operating_mode_definition
from .pass2_benchmark import build_pass2_benchmark_estimate, format_pass2_benchmark_estimate


def format_dry_run(
    manifest: DistributedRunManifest,
    preflight: PreflightReport,
    worker_commands: Sequence[WorkerCommand],
    *,
    pass2_dump_m: Optional[int] = None,
    discovery_estimate: Optional[DiscoveryDryRunEstimate] = None,
    local_compatibility: Optional[LocalCompatibilityReport] = None,
    h100_exact_mode: Optional[H100ExactModeReport] = None,
) -> str:
    mode_definition = operating_mode_definition(manifest.run_mode)
    lines = [
        f"run_id: {manifest.run_id}",
        f"run_mode: {manifest.run_mode.value}",
        f"exactness: {mode_definition.exactness_status}",
        f"config_hash: {preflight.normalized_config_hash}",
        f"output_root: {manifest.output_root}",
        f"worker_count: {manifest.worker_count}",
        f"selected_parts: {list(preflight.selected_parts)}",
        f"preflight_shards: {preflight.shard_count}",
        f"preflight_required_disk_bytes: {preflight.rough_required_disk_bytes}",
        f"preflight_free_disk_bytes: {preflight.free_disk_bytes}",
        "workers:",
    ]
    for command in worker_commands:
        assignment = manifest.devices[command.worker_id]
        shard_ids = manifest.work_assignments.pass1_shards.get(str(command.worker_id), [])
        env = " ".join(f"{key}={value}" for key, value in command.environment.items())
        lines.append(
            f"  worker_{command.worker_id:03d}: physical={assignment.physical_id} "
            f"logical={assignment.logical_id} shards={shard_ids}"
        )
        lines.append(f"    {env} {' '.join(command.command)}")
    if manifest.work_assignments.pass2_sequence_ids and pass2_dump_m is not None:
        lines.append(
            format_pass2_benchmark_estimate(
                build_pass2_benchmark_estimate(manifest, m=pass2_dump_m)
            )
        )
    if discovery_estimate is not None:
        lines.extend(_format_discovery_dry_run_estimate(discovery_estimate))
    if local_compatibility is not None and local_compatibility.mode == "local_one_worker":
        lines.extend(_format_local_compatibility_report(local_compatibility))
    if h100_exact_mode is not None:
        lines.extend(_format_h100_exact_mode_report(h100_exact_mode))
    return "\n".join(lines)


__all__ = [
    "format_dry_run",
]
