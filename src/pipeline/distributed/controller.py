"""Controller skeleton for planning distributed pipeline runs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence

import yaml

from .assignments import build_work_assignments
from .devices import build_device_assignments, worker_environment
from .layout import RunLayout, build_run_layout, create_output_layout, read_worker_marker
from .manifest import (
    CleanupPolicy,
    DistributedRunManifest,
    ManifestStatus,
    RunMode,
    generate_run_id,
    save_manifest,
)
from .pass2_benchmark import build_pass2_benchmark_estimate, format_pass2_benchmark_estimate
from .shard_table import ShardRecord, build_shard_table


REQUIRED_NATIVE_EXTENSIONS = {
    "pass2_reduce": ("top_coactivation_reduce",),
    "mid_ctx": ("mid_reservoir",),
}


@dataclass(frozen=True)
class PreflightReport:
    config_path: Path
    normalized_config_hash: str
    output_root: Path
    run_id_collision: bool
    free_disk_bytes: int
    rough_required_disk_bytes: int
    native_extensions: Dict[str, bool]


@dataclass(frozen=True)
class WorkerCommand:
    worker_id: int
    command: List[str]
    environment: Dict[str, str]
    cwd: Path


@dataclass(frozen=True)
class ControllerPlan:
    manifest: DistributedRunManifest
    layout: RunLayout
    preflight: PreflightReport
    worker_commands: List[WorkerCommand]
    dry_run_text: str


@dataclass(frozen=True)
class DistributedParts1To3Result:
    worker_artifacts: Dict[int, Dict[str, str]]
    pass1_merge: Dict[str, object]
    negative_context: object


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
    )
    layout = create_output_layout(manifest) if create_layout else build_run_layout(manifest)
    if create_layout:
        save_manifest(manifest, layout.manifest_path)

    worker_commands = build_worker_commands(manifest, project_root_path)
    return ControllerPlan(
        manifest=manifest,
        layout=layout,
        preflight=preflight,
        worker_commands=worker_commands,
        dry_run_text=format_dry_run(
            manifest,
            preflight,
            worker_commands,
            pass2_dump_m=_candidate_dump_m_from_config(normalized_config),
        ),
    )


def load_and_hash_config(config_path: str | Path) -> tuple[Dict[str, object], str]:
    """Strictly load config data and return normalized data plus SHA-256 hash."""

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _validate_config_strict(raw)
    normalized = _normalize_for_hash(raw)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return normalized, hashlib.sha256(encoded).hexdigest()


def run_preflight_checks(
    *,
    config_path: str | Path,
    normalized_config_hash: str,
    output_root: str | Path,
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
    if output_root.exists() and not resume:
        raise FileExistsError("run ID collision; pass resume=True to reuse output_root")
    _check_output_writable(output_root.parent)

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

    free_disk_bytes = shutil.disk_usage(output_root.parent).free
    return PreflightReport(
        config_path=config_path,
        normalized_config_hash=normalized_config_hash,
        output_root=output_root,
        run_id_collision=output_root.exists(),
        free_disk_bytes=free_disk_bytes,
        rough_required_disk_bytes=0,
        native_extensions=missing_native,
    )


def native_extension_availability(selected_parts: Sequence[str]) -> Dict[str, bool]:
    required: set[str] = set()
    for part in selected_parts:
        required.update(REQUIRED_NATIVE_EXTENSIONS.get(part, ()))
    return {name: importlib.util.find_spec(name) is not None for name in sorted(required)}


def build_worker_commands(
    manifest: DistributedRunManifest,
    project_root: str | Path,
) -> List[WorkerCommand]:
    commands: List[WorkerCommand] = []
    by_worker_id = {assignment.worker_id: assignment for assignment in manifest.devices}
    for worker_id in range(manifest.worker_count):
        assignment = by_worker_id[worker_id]
        environment = worker_environment(assignment)
        environment["PYTHONPATH"] = _worker_pythonpath(Path(project_root))
        commands.append(
            WorkerCommand(
                worker_id=worker_id,
                command=[
                    sys.executable,
                    "-m",
                    "pipeline.distributed.worker",
                    "--manifest",
                    manifest.manifest_path,
                    "--worker-id",
                    str(worker_id),
                ],
                environment=environment,
                cwd=Path(project_root),
            )
        )
    return commands


def run_parts_1_to_3(
    manifest: DistributedRunManifest,
    *,
    worker_runner: Optional[Callable[[DistributedRunManifest, int], Dict[str, str]]] = None,
    merge_runner: Optional[Callable[..., Dict[str, object]]] = None,
    neg_ctx_runner: Optional[Callable[..., object]] = None,
    seq_latent_index_enabled: bool = True,
    vocab_size: int | None = None,
    resume_neg_ctx: bool = True,
) -> DistributedParts1To3Result:
    """
    Execute the current integrated distributed path: pass-1 workers, pass-1 merge,
    then standalone negative context over merged canonical artifacts.
    """

    if worker_runner is None:
        from .worker import run_pass1_worker

        worker_runner = run_pass1_worker
    if merge_runner is None:
        from .pass1_merge import merge_pass1_worker_outputs

        merge_runner = merge_pass1_worker_outputs
    if neg_ctx_runner is None:
        from pipeline.negative_context import run_negative_context_stage

        neg_ctx_runner = run_negative_context_stage

    worker_artifacts = {
        worker_id: worker_runner(manifest, worker_id)
        for worker_id in range(manifest.worker_count)
    }
    pass1_merge = merge_runner(
        manifest,
        seq_latent_index_enabled=seq_latent_index_enabled,
        vocab_size=vocab_size,
    )
    negative_context = neg_ctx_runner(
        manifest.output_root,
        manifest_path=manifest.manifest_path,
        resume=resume_neg_ctx,
    )
    return DistributedParts1To3Result(
        worker_artifacts=worker_artifacts,
        pass1_merge=pass1_merge,
        negative_context=negative_context,
    )


def launch_worker_processes(
    worker_commands: Sequence[WorkerCommand],
) -> List[subprocess.Popen]:
    processes: List[subprocess.Popen] = []
    for worker_command in worker_commands:
        env = os.environ.copy()
        env.update(worker_command.environment)
        processes.append(
            subprocess.Popen(
                worker_command.command,
                cwd=worker_command.cwd,
                env=env,
            )
        )
    return processes


def format_dry_run(
    manifest: DistributedRunManifest,
    preflight: PreflightReport,
    worker_commands: Sequence[WorkerCommand],
    *,
    pass2_dump_m: Optional[int] = None,
) -> str:
    lines = [
        f"run_id: {manifest.run_id}",
        f"run_mode: {manifest.run_mode.value}",
        f"config_hash: {preflight.normalized_config_hash}",
        f"output_root: {manifest.output_root}",
        f"worker_count: {manifest.worker_count}",
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
    return "\n".join(lines)


def classify_resume_workers(
    manifest: DistributedRunManifest,
    *,
    current_config_hash: Optional[str] = None,
) -> Dict[str, List[int]]:
    """Classify workers from marker files as pending/completed/failed/stale."""

    result = {"pending": [], "completed": [], "failed": [], "stale": []}
    if current_config_hash is not None and current_config_hash != manifest.normalized_config_hash:
        result["stale"] = list(range(manifest.worker_count))
        return result

    layout = build_run_layout(manifest)
    for worker_id, worker_layout in layout.workers.items():
        try:
            if worker_layout.failed_marker.exists():
                marker = read_worker_marker(worker_layout.failed_marker)
                _validate_marker_identity(manifest, worker_id, marker.run_id, marker.worker_id)
                result["failed"].append(worker_id)
            elif worker_layout.completed_marker.exists():
                marker = read_worker_marker(worker_layout.completed_marker)
                _validate_marker_identity(manifest, worker_id, marker.run_id, marker.worker_id)
                result["completed"].append(worker_id)
            else:
                result["pending"].append(worker_id)
        except Exception:
            result["stale"].append(worker_id)
    return result


def _validate_config_strict(data: Dict[str, object]) -> None:
    from config import RootConfig

    RootConfig.model_validate(data)


def _normalize_for_hash(value):
    if isinstance(value, dict):
        return {key: _normalize_for_hash(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_config_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _check_output_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".distributed_write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()


def _worker_pythonpath(project_root: Path) -> str:
    src_path = str(project_root / "src")
    existing = os.environ.get("PYTHONPATH")
    if existing:
        return os.pathsep.join([src_path, existing])
    return src_path


def _candidate_dump_m_from_config(normalized_config: Dict[str, object]) -> int:
    latents = normalized_config.get("latents", {})
    top_coactivation = (
        latents.get("top_coactivation", {})
        if isinstance(latents, dict)
        else {}
    )
    n_latents_per_latent = int(
        top_coactivation.get("n_latents_per_latent", 64)
        if isinstance(top_coactivation, dict)
        else 64
    )
    n_candidates_per_component = int(
        top_coactivation.get("n_candidates_per_component", 16)
        if isinstance(top_coactivation, dict)
        else 16
    )
    # The model config currently defaults to 12 layers with three SAE components per layer.
    default_num_components = 36
    return min(
        n_latents_per_latent * 4,
        default_num_components * n_candidates_per_component,
    )


def _visible_cuda_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _validate_marker_identity(
    manifest: DistributedRunManifest,
    worker_id: int,
    marker_run_id: str,
    marker_worker_id: int,
) -> None:
    if marker_run_id != manifest.run_id or marker_worker_id != worker_id:
        raise ValueError("worker marker identity does not match manifest")
