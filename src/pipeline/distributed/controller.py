"""Controller skeleton for planning distributed pipeline runs."""

from __future__ import annotations

import argparse
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
from .assignments import (
    SEED_FREE_DISCOVERY_METHODS,
    assign_seed_free_method_owners,
    build_discovery_task_assignments,
)
from .devices import build_device_assignments, worker_environment
from .experimental_modes import build_experimental_fast_config
from .layout import RunLayout, build_run_layout, create_output_layout, read_worker_marker
from .manifest import (
    CleanupPolicy,
    DistributedRunManifest,
    ManifestStatus,
    RunMode,
    generate_run_id,
    save_manifest,
)
from .operating_modes import operating_mode_definition
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
    selected_parts: tuple[str, ...] = ()
    shard_count: int = 0
    total_shard_bytes: int = 0


@dataclass(frozen=True)
class WorkerCommand:
    worker_id: int
    command: List[str]
    environment: Dict[str, str]
    cwd: Path


@dataclass(frozen=True)
class DiscoveryDryRunEstimate:
    mode: str
    candidate_count: int
    seed_method_count: int
    seed_free_method_count: int
    expected_worker_task_counts: Dict[str, int]
    expected_worker_estimated_costs: Dict[str, float]
    probe_batch_size: int
    neg_ctx_eval_max: int
    replicated_model_sae_workers: int


@dataclass(frozen=True)
class LocalCompatibilityReport:
    mode: str
    worker_count: int
    device_mode: str
    h100_required: bool
    memory: str
    keep_model_loaded_for_neg_ctx: bool
    search_cache_deferred: bool
    n_shards: int
    n_seeds: int
    probe_batch_size: int
    neg_ctx_eval_max: int


@dataclass(frozen=True)
class H100ExactModeReport:
    mode: str
    worker_count: int
    one_worker_per_gpu: bool
    worker_logical_device: str
    manifest_declared_devices: tuple[int, ...]
    replicated_model_sae_workers: int
    neg_ctx_device_source: str
    pass2_reduce_strategy: str
    mapreduce_entry_criterion: str
    gpu_phases: tuple[str, ...]
    cpu_or_io_phases: tuple[str, ...]


@dataclass(frozen=True)
class ControllerPlan:
    manifest: DistributedRunManifest
    layout: RunLayout
    preflight: PreflightReport
    worker_commands: List[WorkerCommand]
    dry_run_text: str
    discovery_estimate: DiscoveryDryRunEstimate
    local_compatibility: LocalCompatibilityReport
    h100_exact_mode: Optional[H100ExactModeReport]


@dataclass(frozen=True)
class DistributedParts1To3Result:
    worker_artifacts: Dict[int, Dict[str, str]]
    pass1_merge: Dict[str, object]
    negative_context: object


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the model-free distributed controller CLI parser."""

    parser = argparse.ArgumentParser(
        description="Plan or launch manifest-assigned distributed pipeline workers."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--project-root", default=".", help="Repository/project root")
    parser.add_argument("--output-base", default=None, help="Base output directory")
    parser.add_argument("--run-id", default=None, help="Run ID, or config value if omitted")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RunMode],
        default=None,
        help="Operating mode, or config distributed.mode if omitted",
    )
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated physical CUDA device IDs, e.g. 0,1,2,3",
    )
    parser.add_argument("--use-cpu", action="store_true", help="Plan CPU worker devices")
    parser.add_argument(
        "--cleanup-policy",
        choices=[policy.value for policy in CleanupPolicy],
        default=None,
    )
    parser.add_argument(
        "--part",
        action="append",
        default=None,
        help="Selected part/native preflight gate. May be supplied more than once.",
    )
    parser.add_argument(
        "--phase",
        choices=["pass1", "pass2", "discovery"],
        default="pass1",
        help="Worker phase for emitted commands",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing run root")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch worker subprocesses after planning",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for controller dry-runs and optional subprocess launch."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    plan = plan_distributed_run_from_args(args)
    print(plan.dry_run_text)
    if args.launch:
        processes = launch_worker_processes(
            build_worker_commands(plan.manifest, args.project_root, phase=args.phase)
        )
        print(f"launched_workers: {len(processes)}")


def plan_distributed_run_from_args(args: argparse.Namespace) -> ControllerPlan:
    """Create a controller plan from CLI args and Phase 2 config defaults."""

    defaults = _distributed_cli_defaults(args.config)
    worker_count = args.worker_count or int(defaults["worker_count"])
    output_base = Path(args.output_base or str(defaults["output_base"]))
    run_mode = RunMode(args.mode or str(defaults["mode"]))
    cleanup_policy = CleanupPolicy(args.cleanup_policy or str(defaults["cleanup_policy"]))
    selected_parts = tuple(args.part if args.part is not None else defaults["parts"])
    physical_ids = _parse_physical_ids(args.devices, defaults["devices"])
    resume = bool(args.resume or defaults["resume_policy"] in {"resume", "auto"})

    plan = plan_distributed_run(
        config_path=args.config,
        project_root=args.project_root,
        output_base=output_base,
        worker_count=worker_count,
        run_mode=run_mode,
        run_id=args.run_id or defaults["run_id"],
        resume=resume,
        physical_ids=physical_ids,
        use_cpu=bool(args.use_cpu),
        cleanup_policy=cleanup_policy,
        selected_parts=selected_parts,
    )
    if args.phase != "pass1":
        worker_commands = build_worker_commands(plan.manifest, args.project_root, phase=args.phase)
        return ControllerPlan(
            manifest=plan.manifest,
            layout=plan.layout,
            preflight=plan.preflight,
            worker_commands=worker_commands,
            discovery_estimate=plan.discovery_estimate,
            local_compatibility=plan.local_compatibility,
            h100_exact_mode=plan.h100_exact_mode,
            dry_run_text=format_dry_run(
                plan.manifest,
                plan.preflight,
                worker_commands,
                pass2_dump_m=_candidate_dump_m_from_config(load_and_hash_config(args.config)[0]),
                discovery_estimate=plan.discovery_estimate,
                local_compatibility=plan.local_compatibility,
                h100_exact_mode=plan.h100_exact_mode,
            ),
        )
    return plan


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


def build_worker_commands(
    manifest: DistributedRunManifest,
    project_root: str | Path,
    *,
    phase: Literal["pass1", "pass2", "discovery"] = "pass1",
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
                    "--phase",
                    phase,
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


def build_discovery_dry_run_estimate(
    normalized_config: Dict[str, object],
    worker_count: int,
) -> DiscoveryDryRunEstimate:
    """Estimate discovery work before candidate selection has materialized candidates."""

    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    discovery = _root_config_dump(normalized_config)["discovery"]
    candidate_count = int(discovery["n_seeds"])
    methods = [str(method) for method in discovery["methods"]]
    seed_methods = [
        method for method in methods if method not in SEED_FREE_DISCOVERY_METHODS
    ]
    seed_free_owners = assign_seed_free_method_owners(methods, worker_count)
    synthetic_candidates = [
        {"comp_idx": index, "latent_idx": index}
        for index in range(candidate_count)
    ]
    task_assignments, worker_costs = build_discovery_task_assignments(
        synthetic_candidates,
        worker_count,
        methods=methods,
        seed_free_method_owners=seed_free_owners,
    )
    if worker_count == 1:
        mode = "local_one_worker"
    elif worker_count == 8:
        mode = "h100_one_worker_per_gpu"
    else:
        mode = "distributed_multi_worker"
    return DiscoveryDryRunEstimate(
        mode=mode,
        candidate_count=candidate_count,
        seed_method_count=len(seed_methods),
        seed_free_method_count=len(seed_free_owners),
        expected_worker_task_counts={
            worker_id: len(tasks)
            for worker_id, tasks in task_assignments.items()
        },
        expected_worker_estimated_costs=worker_costs,
        probe_batch_size=int(discovery["probe_batch_size"]),
        neg_ctx_eval_max=int(discovery["neg_ctx_eval_max"]),
        replicated_model_sae_workers=worker_count,
    )


def _format_discovery_dry_run_estimate(
    estimate: DiscoveryDryRunEstimate,
) -> List[str]:
    lines = [
        "discovery estimate:",
        f"  mode: {estimate.mode}",
        f"  candidate_count: {estimate.candidate_count}",
        f"  seed_method_count: {estimate.seed_method_count}",
        f"  seed_free_method_count: {estimate.seed_free_method_count}",
        f"  probe_batch_size: {estimate.probe_batch_size}",
        f"  neg_ctx_eval_max: {estimate.neg_ctx_eval_max}",
        f"  replicated_model_sae_workers: {estimate.replicated_model_sae_workers}",
    ]
    for worker_id in sorted(estimate.expected_worker_task_counts, key=int):
        lines.append(
            f"  worker_{int(worker_id):03d}: discovery_tasks="
            f"{estimate.expected_worker_task_counts[worker_id]} "
            f"estimated_cost={estimate.expected_worker_estimated_costs[worker_id]:.1f}"
        )
    return lines


def build_local_compatibility_report(
    normalized_config: Dict[str, object],
    *,
    worker_count: int,
    use_cpu: bool,
    physical_ids: Optional[Sequence[int]],
) -> LocalCompatibilityReport:
    """Summarize local one-worker compatibility knobs without model loading."""

    config_dump = _root_config_dump(normalized_config)
    hardware = config_dump["hardware"]
    data = config_dump["data"]
    discovery = config_dump["discovery"]
    persist = config_dump["persist"]
    device_mode = (
        "cpu"
        if use_cpu
        else "single_cuda"
        if worker_count == 1 and physical_ids is not None and len(physical_ids) == 1
        else "auto"
        if worker_count == 1
        else "multi_worker"
    )
    return LocalCompatibilityReport(
        mode="local_one_worker" if worker_count == 1 else "distributed_multi_worker",
        worker_count=worker_count,
        device_mode=device_mode,
        h100_required=False if worker_count == 1 else True,
        memory=str(hardware["memory"]),
        keep_model_loaded_for_neg_ctx=bool(hardware["keep_model_loaded_for_neg_ctx"]),
        search_cache_deferred=not bool(persist["build_search_cache_after_pipeline"]),
        n_shards=int(data["n_shards"]),
        n_seeds=int(discovery["n_seeds"]),
        probe_batch_size=int(discovery["probe_batch_size"]),
        neg_ctx_eval_max=int(discovery["neg_ctx_eval_max"]),
    )


def _format_local_compatibility_report(report: LocalCompatibilityReport) -> List[str]:
    return [
        "local compatibility:",
        f"  mode: {report.mode}",
        f"  device_mode: {report.device_mode}",
        f"  h100_required: {str(report.h100_required).lower()}",
        f"  memory: {report.memory}",
        f"  keep_model_loaded_for_neg_ctx: {str(report.keep_model_loaded_for_neg_ctx).lower()}",
        f"  search_cache_deferred: {str(report.search_cache_deferred).lower()}",
        f"  n_shards: {report.n_shards}",
        f"  n_seeds: {report.n_seeds}",
        f"  probe_batch_size: {report.probe_batch_size}",
        f"  neg_ctx_eval_max: {report.neg_ctx_eval_max}",
    ]


def build_h100_exact_mode_report(
    normalized_config: Dict[str, object],
    *,
    manifest: DistributedRunManifest,
) -> Optional[H100ExactModeReport]:
    """Summarize recommended H100 exact-mode execution without model loading."""

    if manifest.worker_count != 8 or manifest.run_mode not in {
        RunMode.DISTRIBUTED_SIMPLE_EXACT,
        RunMode.DISTRIBUTED_MAPREDUCE_EXACT,
    }:
        return None
    physical_ids = tuple(
        int(device.physical_id)
        for device in manifest.devices
        if device.physical_id is not None
    )
    if len(physical_ids) != manifest.worker_count:
        return None
    config_dump = _root_config_dump(normalized_config)
    top_coactivation = config_dump["latents"]["top_coactivation"]
    pass2_strategy = (
        "simple_exact_candidate_dump_reduce"
        if manifest.run_mode == RunMode.DISTRIBUTED_SIMPLE_EXACT
        else "mapreduce_target_range_reduce"
    )
    if manifest.run_mode == RunMode.DISTRIBUTED_SIMPLE_EXACT:
        pass2_strategy += f":{top_coactivation['reduce_backend']}"
    return H100ExactModeReport(
        mode=manifest.run_mode.value,
        worker_count=manifest.worker_count,
        one_worker_per_gpu=len(set(physical_ids)) == manifest.worker_count,
        worker_logical_device="cuda:0",
        manifest_declared_devices=physical_ids,
        replicated_model_sae_workers=manifest.worker_count,
        neg_ctx_device_source="manifest_declared_devices",
        pass2_reduce_strategy=pass2_strategy,
        mapreduce_entry_criterion=(
            "enable distributed_mapreduce_exact only after simple exact benchmarks show "
            "candidate-dump merge or reducer input memory is a bottleneck"
        ),
        gpu_phases=("pass1", "neg_ctx", "pass2_dump", "discovery"),
        cpu_or_io_phases=("pass1_merge", "pass2_reduce", "candidate_selection", "circuit_merge"),
    )


def _format_h100_exact_mode_report(report: H100ExactModeReport) -> List[str]:
    return [
        "h100 exact mode:",
        f"  mode: {report.mode}",
        f"  one_worker_per_gpu: {str(report.one_worker_per_gpu).lower()}",
        f"  worker_logical_device: {report.worker_logical_device}",
        f"  manifest_declared_devices: {list(report.manifest_declared_devices)}",
        f"  replicated_model_sae_workers: {report.replicated_model_sae_workers}",
        f"  neg_ctx_device_source: {report.neg_ctx_device_source}",
        f"  pass2_reduce_strategy: {report.pass2_reduce_strategy}",
        f"  gpu_phases: {list(report.gpu_phases)}",
        f"  cpu_or_io_phases: {list(report.cpu_or_io_phases)}",
        f"  mapreduce_entry_criterion: {report.mapreduce_entry_criterion}",
    ]


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


def _root_config_dump(normalized_config: Dict[str, object]) -> Dict[str, object]:
    from config import RootConfig

    return RootConfig.model_validate(normalized_config).model_dump()


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


def _distributed_cli_defaults(config_path: str | Path) -> Dict[str, object]:
    normalized_config, _config_hash = load_and_hash_config(config_path)
    return dict(_root_config_dump(normalized_config)["distributed"])


def _parse_physical_ids(
    raw_devices: Optional[str],
    config_devices: object,
) -> Optional[List[int]]:
    devices = (
        [part.strip() for part in raw_devices.split(",") if part.strip()]
        if raw_devices is not None
        else list(config_devices) if isinstance(config_devices, list) else []
    )
    if not devices:
        return None
    parsed: List[int] = []
    for device in devices:
        if isinstance(device, int):
            parsed.append(device)
            continue
        text = str(device)
        if text.startswith("cuda:"):
            text = text.split(":", 1)[1]
        parsed.append(int(text))
    return parsed


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


if __name__ == "__main__":
    main()
