"""Compatibility facade for distributed controller planning."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

from . import controller_cli as _controller_cli, controller_commands as _controller_commands, controller_planning as _controller_planning, controller_preflight as _controller_preflight
from .controller_commands import _worker_pythonpath
from .controller_config import (
    _candidate_dump_m_from_config,
    _distributed_cli_defaults,
    _normalize_for_hash,
    _parse_physical_ids,
    _resolve_config_path,
    _root_config_dump,
    _validate_config_strict,
    load_and_hash_config,
)
from .controller_contracts import (
    ControllerPlan,
    DiscoveryDryRunEstimate,
    DistributedParts1To3Result,
    H100ExactModeReport,
    LocalCompatibilityReport,
    PreflightReport,
    WorkerCommand,
)
from .controller_dry_run import format_dry_run
from .controller_preflight import (
    REQUIRED_NATIVE_EXTENSIONS,
    _check_output_writable,
    _estimate_preflight_disk_bytes,
    _visible_cuda_device_count,
    native_extension_availability,
)
from .controller_reports import (
    _format_discovery_dry_run_estimate,
    _format_h100_exact_mode_report,
    _format_local_compatibility_report,
    build_discovery_dry_run_estimate,
    build_h100_exact_mode_report,
    build_local_compatibility_report,
)
from .controller_resume import _validate_marker_identity, classify_resume_workers
from .controller_stages import run_parts_1_to_3
from .manifest import CleanupPolicy, DistributedRunManifest, RunMode


def build_arg_parser() -> argparse.ArgumentParser:
    return _controller_cli.build_arg_parser()


def main(argv: Optional[Sequence[str]] = None) -> None:
    _sync_cli_compat()
    _controller_cli.main(argv)


def plan_distributed_run_from_args(args: argparse.Namespace) -> ControllerPlan:
    _sync_cli_compat()
    return _controller_cli.plan_distributed_run_from_args(args)


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
    _sync_planning_compat()
    return _controller_planning.plan_distributed_run(
        config_path=config_path,
        project_root=project_root,
        output_base=output_base,
        worker_count=worker_count,
        run_mode=run_mode,
        run_id=run_id,
        resume=resume,
        create_layout=create_layout,
        physical_ids=physical_ids,
        use_cpu=use_cpu,
        cleanup_policy=cleanup_policy,
        selected_parts=selected_parts,
        pass2_sequence_ids=pass2_sequence_ids,
        discovery_seed_ids=discovery_seed_ids,
        selected_shards=selected_shards,
        timestamp=timestamp,
    )


def run_preflight_checks(**kwargs) -> PreflightReport:
    _controller_preflight._check_output_writable = _check_output_writable
    _controller_preflight._visible_cuda_device_count = _visible_cuda_device_count
    return _controller_preflight.run_preflight_checks(**kwargs)


def build_worker_commands(
    manifest: DistributedRunManifest,
    project_root: str | Path,
    *,
    phase: Literal["pass1", "pass2", "discovery"] = "pass1",
) -> list[WorkerCommand]:
    _controller_commands._worker_pythonpath = _worker_pythonpath
    return _controller_commands.build_worker_commands(manifest, project_root, phase=phase)


def launch_worker_processes(worker_commands: Sequence[WorkerCommand]) -> list[subprocess.Popen]:
    _controller_commands.subprocess = subprocess
    return _controller_commands.launch_worker_processes(worker_commands)


def _sync_cli_compat() -> None:
    _controller_cli.build_worker_commands = build_worker_commands
    _controller_cli.launch_worker_processes = launch_worker_processes
    _controller_cli.plan_distributed_run = plan_distributed_run
    _controller_cli.format_dry_run = format_dry_run


def _sync_planning_compat() -> None:
    _controller_planning.run_preflight_checks = run_preflight_checks
    _controller_planning._visible_cuda_device_count = _visible_cuda_device_count
    _controller_planning.build_worker_commands = build_worker_commands


__all__ = ["ControllerPlan", "DiscoveryDryRunEstimate", "DistributedParts1To3Result", "H100ExactModeReport", "LocalCompatibilityReport", "PreflightReport", "WorkerCommand", "build_arg_parser", "build_discovery_dry_run_estimate", "build_h100_exact_mode_report", "build_local_compatibility_report", "build_worker_commands", "classify_resume_workers", "format_dry_run", "launch_worker_processes", "load_and_hash_config", "main", "native_extension_availability", "plan_distributed_run", "plan_distributed_run_from_args", "run_parts_1_to_3", "run_preflight_checks"]


if __name__ == "__main__":
    main()
