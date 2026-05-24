"""Shared contracts for distributed controller planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .layout import RunLayout
from .manifest import DistributedRunManifest


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


__all__ = [
    "ControllerPlan",
    "DiscoveryDryRunEstimate",
    "DistributedParts1To3Result",
    "H100ExactModeReport",
    "LocalCompatibilityReport",
    "PreflightReport",
    "WorkerCommand",
]
