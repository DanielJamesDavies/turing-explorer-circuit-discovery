import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

import torch

from circuit.feature_selection import CandidateSelector
from config import config
from observability.timing import format_duration
from store.context import mid_ctx, neg_ctx, top_ctx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store.top_coactivation import top_coactivation
from .distributed.assignments import (
    assign_seed_free_method_owners,
    build_discovery_candidate_assignments,
    build_discovery_scheduling_report,
    build_discovery_task_assignments,
)
from .distributed.manifest import DistributedRunManifest, WorkAssignments, load_manifest, save_manifest
from .distributed.interfaces import build_output_paths


@dataclass(frozen=True)
class CandidateSelectionStageResult:
    candidates_path: Path
    metadata_path: Path
    candidates: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def run_candidate_selection(output_root: str = "outputs") -> List[Dict[str, Any]]:
    print("--- Candidate Selection: Finding Seeds ---")
    output_paths = build_output_paths(output_root)
    n_seeds = cast(int, config.discovery.n_seeds or 1000)
    selector = CandidateSelector(n_seeds=n_seeds)
    select_t0 = time.perf_counter()
    candidates = selector.select_candidates()
    print(f"  [timing] candidate scoring: {format_duration(time.perf_counter() - select_t0)}")
    selector.get_summary_stats(candidates)

    save_t0 = time.perf_counter()
    output_paths.run_root.mkdir(parents=True, exist_ok=True)
    torch.save(candidates, output_paths.candidates)
    print(f"  [timing] candidates save: {format_duration(time.perf_counter() - save_t0)}")
    print(f"  ✓ candidates saved to {output_paths.candidates}")
    print("")
    return candidates


def run_candidate_selection_stage(
    output_root: str | Path = "outputs",
    *,
    expected_config_hash: Optional[str] = None,
    manifest_path: str | Path | None = None,
    selector_cls=None,
) -> CandidateSelectionStageResult:
    """Run centralized candidate selection over merged global run-root artifacts."""

    stage_t0 = time.perf_counter()
    selector_class = selector_cls or CandidateSelector
    print(f"[candidate_selection] starting output_root={output_root}", flush=True)
    manifest_t0 = time.perf_counter()
    manifest = load_manifest(manifest_path) if manifest_path is not None else None
    if manifest is not None:
        print(
            f"[candidate_selection] loaded manifest run_id={manifest.run_id} "
            f"elapsed={format_duration(time.perf_counter() - manifest_t0)}",
            flush=True,
        )
    effective_config_hash = expected_config_hash or (
        manifest.normalized_config_hash if manifest is not None else None
    )
    output_paths = build_output_paths(output_root)
    part_dir = _candidate_selection_part_dir(output_paths.run_root)
    artifact_paths = _required_candidate_selection_artifacts(output_paths)
    print("[candidate_selection] validating required artifacts", flush=True)
    _validate_required_artifacts(artifact_paths)
    hash_t0 = time.perf_counter()
    artifact_hashes = _hash_artifacts(artifact_paths)
    print(
        f"[candidate_selection] artifact hashes computed "
        f"elapsed={format_duration(time.perf_counter() - hash_t0)}",
        flush=True,
    )
    metadata = _candidate_selection_metadata(
        artifact_hashes=artifact_hashes,
        expected_config_hash=effective_config_hash,
        manifest=manifest,
    )

    _write_part_marker(part_dir / "started.json", "running", metadata)
    try:
        load_t0 = time.perf_counter()
        load_candidate_selection_inputs(output_paths.run_root)
        print(
            f"[candidate_selection] inputs loaded elapsed={format_duration(time.perf_counter() - load_t0)}",
            flush=True,
        )
        candidates = _run_candidate_selection_with_selector(
            output_paths.run_root,
            selector_cls=selector_class,
        )
        metadata = _candidate_selection_metadata(
            artifact_hashes=artifact_hashes,
            expected_config_hash=effective_config_hash,
            manifest=manifest,
            candidates=candidates,
        )
        if manifest is not None:
            assign_t0 = time.perf_counter()
            print(
                f"[candidate_selection] assigning {len(candidates)} candidates to discovery workers",
                flush=True,
            )
            manifest = assign_discovery_candidates_to_manifest(manifest, candidates)
            save_manifest(manifest, manifest.manifest_path)
            scheduling_report_path = _write_discovery_scheduling_report(manifest)
            print(
                "[candidate_selection] discovery assignments saved "
                f"elapsed={format_duration(time.perf_counter() - assign_t0)}",
                flush=True,
            )
        else:
            scheduling_report_path = None
        metadata_path = part_dir / "candidate_selection_metadata.json"
        metadata_t0 = time.perf_counter()
        _atomic_write_json(metadata_path, metadata)
        print(
            f"[candidate_selection] metadata written elapsed={format_duration(time.perf_counter() - metadata_t0)}",
            flush=True,
        )
        _write_part_marker(
            part_dir / "completed.json",
            "completed",
            metadata,
            artifacts={
                "candidates": str(output_paths.candidates),
                "metadata": str(metadata_path),
                **(
                    {"discovery_scheduling_report": str(scheduling_report_path)}
                    if scheduling_report_path is not None
                    else {}
                ),
            },
        )
        print(
            f"[candidate_selection] complete elapsed={format_duration(time.perf_counter() - stage_t0)}",
            flush=True,
        )
        return CandidateSelectionStageResult(
            candidates_path=output_paths.candidates,
            metadata_path=metadata_path,
            candidates=candidates,
            metadata=metadata,
        )
    except Exception as error:
        _write_part_marker(part_dir / "failed.json", "failed", metadata, error=str(error))
        raise


def load_candidate_selection_inputs(output_root: str | Path = "outputs") -> None:
    """Load all merged global artifacts required by CandidateSelector."""

    output_paths = build_output_paths(output_root)
    artifact_paths = _required_candidate_selection_artifacts(output_paths)
    _validate_required_artifacts(artifact_paths)
    print("[candidate_selection] loading latent_stats", flush=True)
    t0 = time.perf_counter()
    latent_stats.load(str(artifact_paths["latent_stats"]))
    print(f"[candidate_selection] loaded latent_stats elapsed={format_duration(time.perf_counter() - t0)}", flush=True)
    print("[candidate_selection] loading top_ctx", flush=True)
    t0 = time.perf_counter()
    top_ctx.load(str(artifact_paths["top_ctx"]))
    print(f"[candidate_selection] loaded top_ctx elapsed={format_duration(time.perf_counter() - t0)}", flush=True)
    print("[candidate_selection] loading mid_ctx", flush=True)
    t0 = time.perf_counter()
    mid_ctx.load(str(artifact_paths["mid_ctx"]))
    print(f"[candidate_selection] loaded mid_ctx elapsed={format_duration(time.perf_counter() - t0)}", flush=True)
    print("[candidate_selection] loading neg_ctx", flush=True)
    t0 = time.perf_counter()
    neg_ctx.load(str(artifact_paths["neg_ctx"]))
    print(f"[candidate_selection] loaded neg_ctx elapsed={format_duration(time.perf_counter() - t0)}", flush=True)
    print("[candidate_selection] loading logit_ctx", flush=True)
    t0 = time.perf_counter()
    logit_ctx.load(str(artifact_paths["logit_ctx"]))
    print(f"[candidate_selection] loaded logit_ctx elapsed={format_duration(time.perf_counter() - t0)}", flush=True)
    print("[candidate_selection] loading top_coactivation", flush=True)
    t0 = time.perf_counter()
    top_coactivation.load(str(artifact_paths["top_coactivation"]))
    print(
        f"[candidate_selection] loaded top_coactivation elapsed={format_duration(time.perf_counter() - t0)}",
        flush=True,
    )


def assign_discovery_candidates_to_manifest(
    manifest: DistributedRunManifest,
    candidates: List[Dict[str, Any]],
    *,
    methods: Optional[List[str]] = None,
) -> DistributedRunManifest:
    """Return a manifest with deterministic candidate assignments for discovery workers."""

    method_list = list(methods if methods is not None else config.discovery.methods)
    seed_ids, candidate_assignments = build_discovery_candidate_assignments(
        candidates,
        manifest.worker_count,
        methods=method_list,
    )
    seed_free_owners = assign_seed_free_method_owners(
        method_list,
        manifest.worker_count,
        owner_worker_id=0,
    )
    task_assignments, worker_costs = build_discovery_task_assignments(
        candidates,
        manifest.worker_count,
        methods=method_list,
        seed_free_method_owners=seed_free_owners,
    )
    existing = manifest.work_assignments
    updated_work = WorkAssignments(
        pass1_shards=existing.pass1_shards,
        pass1_sequence_totals=existing.pass1_sequence_totals,
        pass2_sequence_ids=existing.pass2_sequence_ids,
        pass2_replay_sequence_count=existing.pass2_replay_sequence_count,
        pass2_replay_sequence_hash=existing.pass2_replay_sequence_hash,
        discovery_seed_ids=seed_ids,
        discovery_candidate_assignments=candidate_assignments,
        discovery_seed_free_method_owners=seed_free_owners,
        discovery_scheduling_strategy="candidate_contiguous",
        discovery_task_assignments=task_assignments,
        discovery_worker_estimated_costs=worker_costs,
        discovery_failed_task_ranges=existing.discovery_failed_task_ranges,
    )
    return manifest.model_copy(update={"work_assignments": updated_work})


def _write_discovery_scheduling_report(manifest: DistributedRunManifest) -> Path:
    report_path = Path(manifest.distributed_root) / "reports" / "discovery_scheduling_report.json"
    report = build_discovery_scheduling_report(manifest.work_assignments, manifest.worker_count)
    _atomic_write_json(report_path, report)
    return report_path


def _run_candidate_selection_with_selector(
    output_root: str | Path,
    *,
    selector_cls,
) -> List[Dict[str, Any]]:
    print("--- Candidate Selection: Finding Seeds ---")
    output_paths = build_output_paths(output_root)
    n_seeds = cast(int, config.discovery.n_seeds or 1000)
    print(
        f"[candidate_selection] initializing selector n_seeds={n_seeds} "
        f"criteria={list(config.discovery.seed_criteria)}",
        flush=True,
    )
    selector = selector_cls(n_seeds=n_seeds)
    select_t0 = time.perf_counter()
    print("[candidate_selection] scoring candidates", flush=True)
    candidates = selector.select_candidates()
    print(f"  [timing] candidate scoring: {format_duration(time.perf_counter() - select_t0)}")
    print(f"[candidate_selection] selected candidates={len(candidates)}", flush=True)
    selector.get_summary_stats(candidates)

    save_t0 = time.perf_counter()
    output_paths.run_root.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(candidates, output_paths.candidates)
    print(f"  [timing] candidates save: {format_duration(time.perf_counter() - save_t0)}")
    print(f"  ✓ candidates saved to {output_paths.candidates}")
    print("")
    return candidates


def _required_candidate_selection_artifacts(output_paths) -> Dict[str, Path]:
    return {
        "latent_stats": output_paths.latent_stats,
        "top_ctx": output_paths.top_ctx,
        "mid_ctx": output_paths.mid_ctx,
        "neg_ctx": output_paths.neg_ctx,
        "logit_ctx": output_paths.logit_ctx,
        "top_coactivation": output_paths.top_coactivation,
    }


def _validate_required_artifacts(paths: Mapping[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not Path(path).exists()]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise FileNotFoundError(f"missing candidate-selection input artifacts: {missing_list}")


def _candidate_selection_metadata(
    *,
    artifact_hashes: Mapping[str, str],
    expected_config_hash: Optional[str],
    manifest: Optional[DistributedRunManifest],
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    candidate_list = candidates or []
    return {
        "schema_version": 1,
        "stage": "candidate_selection",
        "run_id": manifest.run_id if manifest is not None else None,
        "run_mode": manifest.run_mode.value if manifest is not None else None,
        "config_hash": expected_config_hash,
        "criteria": list(config.discovery.seed_criteria),
        "seed_filter": {
            "layers": list(config.discovery.seed_filter.layers),
            "kinds": list(config.discovery.seed_filter.kinds),
        },
        "n_seeds": int(config.discovery.n_seeds or 1000),
        "artifact_hashes": dict(artifact_hashes),
        "selected_count": len(candidate_list),
        "per_candidate_criterion_scores": [
            {
                "candidate_index": index,
                "comp_idx": int(candidate.get("comp_idx", -1)),
                "latent_idx": int(candidate.get("latent_idx", -1)),
                "score": float(candidate.get("score", 0.0)),
                "criteria_scores": {
                    str(name): float(value)
                    for name, value in dict(candidate.get("criteria_scores", {})).items()
                },
            }
            for index, candidate in enumerate(candidate_list)
        ],
    }


def _hash_artifacts(paths: Mapping[str, Path]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for name, path in paths.items():
        t0 = time.perf_counter()
        size_bytes = Path(path).stat().st_size
        print(
            f"[candidate_selection] hashing {name} size={size_bytes / 1024 ** 2:.1f} MiB -> {path}",
            flush=True,
        )
        hashes[name] = _sha256_file(path)
        print(
            f"[candidate_selection] hashed {name} elapsed={format_duration(time.perf_counter() - t0)}",
            flush=True,
        )
    return hashes


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_selection_part_dir(run_root: str | Path) -> Path:
    return Path(run_root) / "distributed" / "parts" / "candidate_selection"


def _write_part_marker(
    path: str | Path,
    status: str,
    metadata: Mapping[str, Any],
    *,
    artifacts: Optional[Mapping[str, str]] = None,
    error: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "stage": "candidate_selection",
        "metadata": dict(metadata),
        "artifacts": dict(artifacts or {}),
    }
    if error is not None:
        payload["error"] = error
    _atomic_write_json(path, payload)


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, output_path)


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run centralized candidate selection for a run root.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--expected-config-hash", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args(argv)
    result = run_candidate_selection_stage(
        args.output_root,
        expected_config_hash=args.expected_config_hash,
        manifest_path=args.manifest,
    )
    print(f"  ✓ candidate-selection metadata saved to {result.metadata_path}")


if __name__ == "__main__":
    main()
