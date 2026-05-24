"""Report builders for distributed controller dry-runs."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .assignments import (
    SEED_FREE_DISCOVERY_METHODS,
    assign_seed_free_method_owners,
    build_discovery_task_assignments,
)
from .controller_config import _root_config_dump
from .controller_contracts import (
    DiscoveryDryRunEstimate,
    H100ExactModeReport,
    LocalCompatibilityReport,
)
from .manifest import DistributedRunManifest, RunMode


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


__all__ = [
    "build_discovery_dry_run_estimate",
    "build_h100_exact_mode_report",
    "build_local_compatibility_report",
]
