"""Discovery distributed worker implementation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from config import config
from data.loader import DataLoader
from model.inference import Inference
from pipeline.discovery_artifacts import load_discovery_artifacts, validate_discovery_artifacts
from pipeline.runtime import build_distributed_worker_runtime, clear_runtime, get_runtime, set_runtime
from sae.bank import SAEBank

from ..layout import build_run_layout, build_worker_marker, write_worker_marker
from ..manifest import DistributedRunManifest
from ..shard_table import validate_shard_table
from ..worker_common import (
    _device_assignment_for_worker,
    _peak_cuda_memory_bytes,
    _utc_now,
    _validate_worker_id,
)
from .assignments import load_assigned_discovery_candidates, save_discovery_worker_inputs
from .method_filtering import (
    discovery_methods_for_worker,
    seed_free_methods_for_worker,
)
from .stats import (
    _discovery_output_artifacts,
    reset_discovery_worker_state,
    save_worker_discovery_stats,
)


def run_discovery_worker(
    manifest: DistributedRunManifest,
    worker_id: int,
    *,
    validate_inputs_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None,
    load_artifacts_fn: Optional[Callable[[DistributedRunManifest], None]] = None,
    initialize_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None,
    run_discovery_fn: Optional[Callable[[List[Dict[str, Any]], str], None]] = None,
) -> Dict[str, str]:
    """Run discovery for one worker's assigned candidate subset."""

    validate_inputs = validate_inputs_fn or validate_discovery_worker_inputs
    load_artifacts = load_artifacts_fn or load_discovery_global_artifacts
    initialize = initialize_fn or initialize_discovery_worker_resources
    run_discovery = run_discovery_fn or run_worker_discovery_window
    layout = build_run_layout(manifest)
    worker_layout = layout.workers[worker_id]
    seed_count = len(manifest.work_assignments.discovery_seed_ids.get(str(worker_id), []))
    start_time = _utc_now()
    started_at = time.perf_counter()

    write_worker_marker(
        build_worker_marker(
            manifest,
            worker_id,
            phase="discovery",
            status="started",
            start_time=start_time,
            seed_count=seed_count,
        ),
        worker_layout.started_marker,
    )

    try:
        validate_inputs(manifest, worker_id)
        assigned_candidates = load_assigned_discovery_candidates(manifest, worker_id)
        artifacts = save_discovery_worker_inputs(manifest, worker_id, assigned_candidates)
        reset_discovery_worker_state()
        load_artifacts(manifest)
        initialize(manifest, worker_id)
        circuits_dir = worker_layout.discovery_dir / "circuits"
        if run_discovery_fn is None:
            task_metrics = run_worker_discovery_window(
                assigned_candidates,
                str(circuits_dir),
                seed_free_methods=seed_free_methods_for_worker(manifest, worker_id),
            )
        else:
            run_discovery(assigned_candidates, str(circuits_dir))
            task_metrics = []
        artifacts["worker_discovery_stats"] = str(
            save_worker_discovery_stats(
                manifest,
                worker_id,
                assigned_candidates,
                task_metrics=task_metrics,
            )
        )
        artifacts.update(_discovery_output_artifacts(worker_layout))
        end_time = _utc_now()
        write_worker_marker(
            build_worker_marker(
                manifest,
                worker_id,
                phase="discovery",
                status="completed",
                start_time=start_time,
                end_time=end_time,
                duration_s=time.perf_counter() - started_at,
                seed_count=seed_count,
                peak_cuda_memory_bytes=_peak_cuda_memory_bytes(),
                artifacts=artifacts,
            ),
            worker_layout.completed_marker,
        )
        return artifacts
    except Exception as error:
        write_worker_marker(
            build_worker_marker(
                manifest,
                worker_id,
                phase="discovery",
                status="failed",
                start_time=start_time,
                end_time=_utc_now(),
                duration_s=time.perf_counter() - started_at,
                seed_count=seed_count,
                error=str(error),
            ),
            worker_layout.failed_marker,
        )
        raise
    finally:
        clear_runtime()


def validate_discovery_worker_inputs(
    manifest: DistributedRunManifest,
    worker_id: int,
    *,
    validate_on_disk: bool = True,
) -> None:
    """Validate discovery worker inputs before model/SAE initialization."""

    _validate_worker_id(manifest, worker_id)
    if validate_on_disk:
        validate_shard_table(manifest.dataset_path, manifest.shard_table)
    assignments = manifest.work_assignments.discovery_candidate_assignments.get(str(worker_id))
    seed_ids = manifest.work_assignments.discovery_seed_ids.get(str(worker_id))
    if assignments is None or seed_ids is None:
        raise ValueError("manifest missing discovery assignments for worker")
    if [assignment.candidate_index for assignment in assignments] != seed_ids:
        raise ValueError("discovery assignment metadata does not match seed IDs")
    validate_discovery_artifacts(
        manifest.output_root,
        candidates_path=Path(manifest.output_root) / "candidates.pt",
    )


def load_discovery_global_artifacts(manifest: DistributedRunManifest) -> None:
    """Load merged global stores needed for discovery."""

    load_discovery_artifacts(manifest.output_root)


def initialize_discovery_worker_resources(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> None:
    """Initialize DataLoader, model, and SAE bank for one discovery worker."""

    assignment = _device_assignment_for_worker(manifest, worker_id)
    runtime = build_distributed_worker_runtime(assignment)
    set_runtime(runtime)
    runtime.loader = DataLoader(device=runtime.device, pin_memory=runtime.fast)
    runtime.model = Inference(device=runtime.device, compile=runtime.compile)
    runtime.bank = SAEBank(
        devices=runtime.devices,
        load_decoders=runtime.fast,
        compile=runtime.compile,
    )


def run_worker_discovery_window(
    assigned_candidates: List[Dict[str, Any]],
    output_dir: str,
    *,
    seed_free_methods: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Run DiscoveryWindow on an already-initialized distributed worker runtime."""

    from circuit.discovery_window import DiscoveryWindow

    runtime = get_runtime()
    assert runtime.model is not None
    assert runtime.bank is not None
    assert runtime.loader is not None
    with discovery_methods_for_worker(seed_free_methods or ()):
        window = DiscoveryWindow(
            runtime.model,
            runtime.bank,
            runtime.loader,
            output_dir=output_dir,
        )
        return window.run(assigned_candidates)


__all__ = [
    "initialize_discovery_worker_resources",
    "load_discovery_global_artifacts",
    "run_discovery_worker",
    "run_worker_discovery_window",
    "validate_discovery_worker_inputs",
]
