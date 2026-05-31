"""Compatibility facade and CLI for distributed worker phases."""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from config import config
from data.loader import DataLoader
from model.inference import Inference
from pipeline.discovery_artifacts import (
    hash_discovery_artifacts,
    load_discovery_artifacts,
    validate_discovery_artifacts,
)
from pipeline.first_pass import run_first_pass
from pipeline.runtime import build_distributed_worker_runtime, clear_runtime, set_runtime
from pipeline.second_pass import SecondPassDumpResult, run_second_pass_dump
from sae.bank import SAEBank
from store.circuits import circuit_store
from store.context import mid_ctx, top_ctx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store.seq_repr import SeqRepr
from store.top_coactivation import top_coactivation

from .discovery import assignments as _discovery_assignments
from .discovery import method_filtering as _discovery_method_filtering
from .discovery import stats as _discovery_stats
from .discovery import worker as _discovery_worker
from .interfaces import get_worker_shard_ids
from .manifest import DistributedRunManifest, load_manifest
from .pass1 import worker as _pass1_worker
from .pass1.contracts import PASS1_PARTIAL_FILENAMES
from .pass1_partials import (
    build_pass1_partial_metadata,
    latent_stats_payload,
    logit_ctx_payload,
    mid_ctx_candidates_payload,
    mid_ctx_reservoir_payload,
    save_pass1_partial,
    seq_repr_payload,
    top_ctx_payload,
)
from .pass2 import worker as _pass2_worker
from .pass2.worker import PASS2_PARTIAL_FILENAMES
from .seq_repr_mapping import build_seq_repr_cap_mapping, shard_table_fingerprint
from .shard_table import sequence_ids_for_shards, validate_shard_table
from .worker_common import (
    _atomic_torch_save,
    _atomic_write_json,
    _device_assignment_for_worker,
    _peak_cuda_memory_bytes,
    _total_sequences,
    _validate_worker_id,
    _worker_batch_count,
)

SEED_FREE_DISCOVERY_METHODS = _discovery_method_filtering.SEED_FREE_DISCOVERY_METHODS
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_runtime_seq_repr_payload = _pass1_worker._runtime_seq_repr_payload
_component_count = _pass1_worker._component_count
_d_sae = _pass1_worker._d_sae
_store_mode_for = _pass1_worker._store_mode_for
_mid_ctx_partial_payload = _pass1_worker._mid_ctx_partial_payload
_PASS1_IMPL_VALIDATE = _pass1_worker.validate_pass1_worker_inputs


def _sync(module: object, names: Sequence[str]) -> None:
    for name in names:
        setattr(module, name, globals()[name])


def _sync_pass1() -> None:
    _sync(
        _pass1_worker,
        (
            "config", "DataLoader", "Inference", "SAEBank", "SeqRepr",
            "top_ctx", "mid_ctx", "latent_stats", "logit_ctx", "PASS1_PARTIAL_FILENAMES",
            "build_distributed_worker_runtime", "set_runtime", "clear_runtime",
            "latent_stats_payload", "top_ctx_payload", "mid_ctx_candidates_payload",
            "mid_ctx_reservoir_payload",
            "seq_repr_payload", "logit_ctx_payload", "build_pass1_partial_metadata",
            "save_pass1_partial", "get_worker_shard_ids", "build_seq_repr_cap_mapping",
            "shard_table_fingerprint", "sequence_ids_for_shards", "validate_shard_table",
            "_runtime_seq_repr_payload", "_component_count", "_d_sae", "_store_mode_for",
            "_mid_ctx_partial_payload",
            "_device_assignment_for_worker", "_total_sequences", "_worker_batch_count",
        ),
    )
    _pass1_worker.validate_pass1_worker_inputs = (
        validate_pass1_worker_inputs
        if validate_pass1_worker_inputs is not _FACADE_VALIDATE_PASS1
        else _PASS1_IMPL_VALIDATE
    )


def _sync_pass2() -> None:
    _sync(
        _pass2_worker,
        (
            "config", "DataLoader", "Inference", "SAEBank", "top_ctx", "latent_stats",
            "top_coactivation", "PASS2_PARTIAL_FILENAMES", "build_distributed_worker_runtime",
            "set_runtime", "clear_runtime", "_device_assignment_for_worker",
            "_peak_cuda_memory_bytes", "_atomic_write_json",
        ),
    )


def _sync_discovery() -> None:
    _sync(
        _discovery_worker,
        (
            "config", "DataLoader", "Inference", "SAEBank", "load_discovery_artifacts",
            "validate_discovery_artifacts", "build_distributed_worker_runtime", "set_runtime",
            "clear_runtime", "_device_assignment_for_worker", "_peak_cuda_memory_bytes",
            "_validate_worker_id", "validate_shard_table", "load_assigned_discovery_candidates",
            "save_discovery_worker_inputs", "save_worker_discovery_stats",
            "reset_discovery_worker_state", "discovery_methods_for_worker",
            "seed_free_methods_for_worker", "_discovery_output_artifacts",
        ),
    )
    _sync(_discovery_assignments, ("torch", "hash_discovery_artifacts", "_atomic_torch_save", "_atomic_write_json", "_validate_worker_id", "seed_free_methods_for_worker"))
    _sync(_discovery_stats, ("circuit_store", "_atomic_write_json", "seed_free_methods_for_worker"))
    _discovery_method_filtering.config = config
    _discovery_method_filtering.SEED_FREE_DISCOVERY_METHODS = SEED_FREE_DISCOVERY_METHODS


def run_worker(manifest_path: str | Path, worker_id: int, *, phase: str = "pass1") -> Dict[str, str]:
    manifest = load_manifest(manifest_path)
    if phase == "pass1":
        return run_pass1_worker(manifest, worker_id)
    if phase == "pass2":
        return run_pass2_worker(manifest, worker_id)
    if phase == "discovery":
        return run_discovery_worker(manifest, worker_id)
    raise ValueError(f"unsupported worker phase: {phase}")


def run_pass1_worker(manifest: DistributedRunManifest, worker_id: int, *, initialize_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None, run_first_pass_fn: Callable[..., None] = run_first_pass, save_partials_fn: Optional[Callable[[DistributedRunManifest, int], Dict[str, str]]] = None, validate_inputs_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None) -> Dict[str, str]:
    _sync_pass1()
    return _pass1_worker.run_pass1_worker(manifest, worker_id, initialize_fn=initialize_fn or initialize_pass1_worker_resources, run_first_pass_fn=run_first_pass_fn, save_partials_fn=save_partials_fn or save_pass1_partials, validate_inputs_fn=validate_inputs_fn or validate_pass1_worker_inputs)


def initialize_pass1_worker_resources(manifest: DistributedRunManifest, worker_id: int) -> None:
    _sync_pass1()
    _pass1_worker.initialize_pass1_worker_resources(manifest, worker_id)


def validate_pass1_worker_inputs(manifest: DistributedRunManifest, worker_id: int, *, validate_on_disk: bool = True) -> None:
    _sync_pass1()
    _pass1_worker.validate_pass1_worker_inputs(manifest, worker_id, validate_on_disk=validate_on_disk)


_FACADE_VALIDATE_PASS1 = validate_pass1_worker_inputs


def save_pass1_partials(manifest: DistributedRunManifest, worker_id: int) -> Dict[str, str]:
    _sync_pass1()
    return _pass1_worker.save_pass1_partials(manifest, worker_id)


def configure_mid_ctx_candidate_pool(manifest: DistributedRunManifest) -> None:
    _sync_pass1()
    _pass1_worker.configure_mid_ctx_candidate_pool(manifest)


def run_pass2_worker(manifest: DistributedRunManifest, worker_id: int, *, validate_inputs_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None, load_artifacts_fn: Optional[Callable[[DistributedRunManifest], None]] = None, initialize_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None, run_dump_fn: Callable[[list[int]], SecondPassDumpResult] = run_second_pass_dump, save_dump_fn: Optional[Callable[[DistributedRunManifest, int, SecondPassDumpResult], Dict[str, str]]] = None) -> Dict[str, str]:
    _sync_pass2()
    return _pass2_worker.run_pass2_worker(manifest, worker_id, validate_inputs_fn=validate_inputs_fn or validate_pass2_worker_inputs, load_artifacts_fn=load_artifacts_fn or load_pass2_global_artifacts, initialize_fn=initialize_fn or initialize_pass2_worker_resources, run_dump_fn=run_dump_fn, save_dump_fn=save_dump_fn or save_pass2_candidate_dump)


def validate_pass2_worker_inputs(manifest: DistributedRunManifest, worker_id: int, *, validate_on_disk: bool = True) -> None:
    _sync_pass2()
    _pass2_worker.validate_pass2_worker_inputs(manifest, worker_id, validate_on_disk=validate_on_disk)


def load_pass2_global_artifacts(manifest: DistributedRunManifest) -> None:
    _sync_pass2()
    _pass2_worker.load_pass2_global_artifacts(manifest)


def initialize_pass2_worker_resources(manifest: DistributedRunManifest, worker_id: int) -> None:
    _sync_pass2()
    _pass2_worker.initialize_pass2_worker_resources(manifest, worker_id)


def save_pass2_candidate_dump(manifest: DistributedRunManifest, worker_id: int, dump_result: SecondPassDumpResult) -> Dict[str, str]:
    _sync_pass2()
    return _pass2_worker.save_pass2_candidate_dump(manifest, worker_id, dump_result)


def build_pass2_worker_summary(manifest: DistributedRunManifest, worker_id: int, dump_result: SecondPassDumpResult, *, artifact_path: str | Path, save_elapsed_s: float) -> Dict[str, object]:
    _sync_pass2()
    return _pass2_worker.build_pass2_worker_summary(manifest, worker_id, dump_result, artifact_path=artifact_path, save_elapsed_s=save_elapsed_s)


def run_discovery_worker(manifest: DistributedRunManifest, worker_id: int, *, validate_inputs_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None, load_artifacts_fn: Optional[Callable[[DistributedRunManifest], None]] = None, initialize_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None, run_discovery_fn: Optional[Callable[[List[Dict[str, Any]], str], None]] = None) -> Dict[str, str]:
    _sync_discovery()
    return _discovery_worker.run_discovery_worker(manifest, worker_id, validate_inputs_fn=validate_inputs_fn or validate_discovery_worker_inputs, load_artifacts_fn=load_artifacts_fn or load_discovery_global_artifacts, initialize_fn=initialize_fn or initialize_discovery_worker_resources, run_discovery_fn=run_discovery_fn)


def validate_discovery_worker_inputs(manifest: DistributedRunManifest, worker_id: int, *, validate_on_disk: bool = True) -> None:
    _sync_discovery()
    _discovery_worker.validate_discovery_worker_inputs(manifest, worker_id, validate_on_disk=validate_on_disk)


def load_discovery_global_artifacts(manifest: DistributedRunManifest) -> None:
    _sync_discovery()
    _discovery_worker.load_discovery_global_artifacts(manifest)


def initialize_discovery_worker_resources(manifest: DistributedRunManifest, worker_id: int) -> None:
    _sync_discovery()
    _discovery_worker.initialize_discovery_worker_resources(manifest, worker_id)


def load_assigned_discovery_candidates(manifest: DistributedRunManifest, worker_id: int) -> List[Dict[str, Any]]:
    _sync_discovery()
    return _discovery_assignments.load_assigned_discovery_candidates(manifest, worker_id)


def save_discovery_worker_inputs(manifest: DistributedRunManifest, worker_id: int, assigned_candidates: List[Dict[str, Any]]) -> Dict[str, str]:
    _sync_discovery()
    return _discovery_assignments.save_discovery_worker_inputs(manifest, worker_id, assigned_candidates)


def save_worker_discovery_stats(manifest: DistributedRunManifest, worker_id: int, assigned_candidates: List[Dict[str, Any]], *, task_metrics: Optional[List[Dict[str, Any]]] = None) -> Path:
    _sync_discovery()
    return _discovery_stats.save_worker_discovery_stats(manifest, worker_id, assigned_candidates, task_metrics=task_metrics)


def _discovery_output_artifacts(worker_layout) -> Dict[str, str]:
    return _discovery_stats._discovery_output_artifacts(worker_layout)


def reset_discovery_worker_state() -> None:
    _sync_discovery()
    _discovery_stats.reset_discovery_worker_state()


def run_worker_discovery_window(assigned_candidates: List[Dict[str, Any]], output_dir: str, *, seed_free_methods: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    _sync_discovery()
    return _discovery_worker.run_worker_discovery_window(assigned_candidates, output_dir, seed_free_methods=seed_free_methods)


def seed_free_methods_for_worker(manifest: DistributedRunManifest, worker_id: int) -> List[str]:
    _sync_discovery()
    return _discovery_method_filtering.seed_free_methods_for_worker(manifest, worker_id)


def discovery_methods_for_worker_filter(methods: Sequence[str], seed_free_methods: Sequence[str]) -> List[str]:
    _sync_discovery()
    return _discovery_method_filtering.discovery_methods_for_worker_filter(methods, seed_free_methods)


@contextmanager
def discovery_methods_for_worker(seed_free_methods: Sequence[str]):
    _sync_discovery()
    with _discovery_method_filtering.discovery_methods_for_worker(seed_free_methods):
        yield


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a distributed pipeline worker")
    parser.add_argument("--manifest", required=True, help="Path to distributed manifest JSON")
    parser.add_argument("--worker-id", required=True, type=int, help="Worker ID from manifest")
    parser.add_argument("--phase", default="pass1", choices=["pass1", "pass2", "discovery"], help="Worker phase to run")
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=int(os.environ.get("TURING_WORKER_THREADS", "4")),
        help=(
            "Default CPU thread cap for direct worker launches. Set to 0 to leave "
            "thread env vars unchanged. Defaults to 4, or TURING_WORKER_THREADS when set."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.worker_threads < 0:
        raise ValueError("--worker-threads must be >= 0")
    _apply_worker_thread_limits(args.worker_threads)
    run_worker(args.manifest, args.worker_id, phase=args.phase)


def _apply_worker_thread_limits(worker_threads: int) -> None:
    if worker_threads <= 0:
        return
    value = str(worker_threads)
    for name in THREAD_ENV_VARS:
        os.environ.setdefault(name, value)


__all__ = [
    "PASS1_PARTIAL_FILENAMES", "PASS2_PARTIAL_FILENAMES", "SEED_FREE_DISCOVERY_METHODS",
    "build_arg_parser", "run_worker", "main",
    "run_pass1_worker", "initialize_pass1_worker_resources", "validate_pass1_worker_inputs", "save_pass1_partials", "configure_mid_ctx_candidate_pool",
    "run_pass2_worker", "validate_pass2_worker_inputs", "load_pass2_global_artifacts", "initialize_pass2_worker_resources", "save_pass2_candidate_dump", "build_pass2_worker_summary",
    "run_discovery_worker", "validate_discovery_worker_inputs", "load_discovery_global_artifacts", "initialize_discovery_worker_resources", "load_assigned_discovery_candidates", "save_discovery_worker_inputs", "save_worker_discovery_stats", "run_worker_discovery_window", "seed_free_methods_for_worker", "discovery_methods_for_worker_filter", "discovery_methods_for_worker", "reset_discovery_worker_state",
]


if __name__ == "__main__":
    main()
