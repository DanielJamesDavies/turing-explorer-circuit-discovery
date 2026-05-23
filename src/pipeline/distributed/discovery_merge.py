"""Merge distributed discovery worker circuit stores into canonical artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from store.circuits import Circuit, CircuitStore

from .interfaces import build_output_paths
from .layout import build_run_layout, read_worker_marker
from .manifest import DistributedRunManifest


@dataclass(frozen=True)
class CircuitStoreMergeResult:
    circuit_store_path: Path
    summary_path: Path
    summary_xlsx_path: Path | None
    report_path: Path
    merged_circuit_count: int
    worker_count: int
    worker_circuit_counts: Dict[int, int]
    validation: Dict[str, Any]


def run_circuit_store_merge(
    manifest: DistributedRunManifest,
    *,
    allow_uuid_rewrite: bool = False,
) -> CircuitStoreMergeResult:
    """Merge every completed discovery worker store into canonical circuits artifacts."""

    if allow_uuid_rewrite:
        raise NotImplementedError("UUID rewrite merge mode is not implemented; duplicate UUIDs fail loudly")
    worker_stores = load_completed_worker_circuit_stores(manifest)
    merged_store = merge_circuit_stores(worker_stores)
    output_paths = build_output_paths(manifest.output_root)
    output_paths.circuits_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_paths.circuits_dir / "discovered_circuits.pt"
    summary_path = output_paths.circuits_dir / "summary.json"
    summary_xlsx_path = output_paths.circuits_dir / "summary.xlsx"
    _atomic_torch_save(merged_store.circuits, store_path)
    summary = build_circuit_summary(merged_store)
    _atomic_write_json(summary_path, summary)
    written_xlsx = _write_summary_xlsx(summary_xlsx_path, summary)
    worker_counts = {
        worker_id: len(store.circuits)
        for worker_id, store in worker_stores.items()
    }
    validation = validate_merged_discovery_outputs(
        manifest,
        merged_store,
        summary,
        worker_counts,
    )
    report = build_merged_discovery_report(
        manifest,
        merged_store,
        summary,
        worker_counts,
        validation,
        artifacts={
            "discovered_circuits": str(store_path),
            "summary": str(summary_path),
            "summary_xlsx": str(written_xlsx) if written_xlsx is not None else None,
        },
    )
    report_path = Path(manifest.distributed_root) / "reports" / "discovery_merge_report.json"
    _atomic_write_json(report_path, report)
    return CircuitStoreMergeResult(
        circuit_store_path=store_path,
        summary_path=summary_path,
        summary_xlsx_path=written_xlsx,
        report_path=report_path,
        merged_circuit_count=len(merged_store.circuits),
        worker_count=len(worker_stores),
        worker_circuit_counts=worker_counts,
        validation=validation,
    )


def build_merged_discovery_report(
    manifest: DistributedRunManifest,
    merged_store: CircuitStore,
    summary: Sequence[Dict[str, Any]],
    worker_counts: Dict[int, int],
    validation: Dict[str, Any],
    *,
    artifacts: Dict[str, str | None],
) -> Dict[str, Any]:
    """Build the canonical distributed discovery merge report."""

    worker_reports = _worker_reports(manifest, worker_counts)
    method_counts = _method_counts(merged_store)
    eval_summary = _eval_summary(merged_store)
    return {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "worker_count": len(worker_counts),
        "merged_circuit_count": len(merged_store.circuits),
        "worker_circuit_counts": {str(k): v for k, v in worker_counts.items()},
        "seed_free_method_counts": {
            method: count
            for method, count in method_counts.items()
            if _is_seed_free_method(method)
        },
        "method_counts": method_counts,
        "eval_summary": eval_summary,
        "worker_reports": worker_reports,
        "failed_task_ranges": {
            str(worker_id): ranges
            for worker_id, ranges in manifest.work_assignments.discovery_failed_task_ranges.items()
        },
        "validation": validation,
        "artifacts": artifacts,
    }


def validate_merged_discovery_outputs(
    manifest: DistributedRunManifest,
    merged_store: CircuitStore,
    summary: Sequence[Dict[str, Any]],
    worker_counts: Dict[int, int],
) -> Dict[str, Any]:
    """Validate merged circuit counts, circuit metadata, and summary consistency."""

    issues: List[str] = []
    merged_count = len(merged_store.circuits)
    expected_count = sum(worker_counts.values())
    if merged_count != expected_count:
        issues.append(
            f"merged circuit count {merged_count} does not match worker sum {expected_count}"
        )
    summary_issues = _summary_consistency_issues(merged_store, summary)
    issues.extend(summary_issues)
    metadata_issues = _circuit_metadata_issues(merged_store)
    issues.extend(metadata_issues)
    method_counts = _method_counts(merged_store)
    seed_free_count = sum(
        count
        for method, count in method_counts.items()
        if _is_seed_free_method(method)
    )
    validation = {
        "ok": not issues,
        "issues": issues,
        "merged_count_matches_worker_sum": merged_count == expected_count,
        "summary_rows_match_store": not summary_issues,
        "circuit_metadata_valid": not metadata_issues,
        "expected_circuit_count": expected_count,
        "actual_circuit_count": merged_count,
        "seed_free_circuit_count": seed_free_count,
        "seed_free_method_owners": dict(manifest.work_assignments.discovery_seed_free_method_owners),
    }
    if issues:
        raise ValueError("; ".join(issues))
    return validation


def load_completed_worker_circuit_stores(
    manifest: DistributedRunManifest,
) -> Dict[int, CircuitStore]:
    """Load circuit stores declared by completed discovery worker markers."""

    layout = build_run_layout(manifest)
    stores: Dict[int, CircuitStore] = {}
    for worker_id, worker_layout in layout.workers.items():
        if not worker_layout.completed_marker.exists():
            raise FileNotFoundError(f"missing completed discovery marker for worker {worker_id}")
        marker = read_worker_marker(worker_layout.completed_marker)
        if marker.phase != "discovery":
            raise ValueError(f"worker {worker_id} completed marker is not for discovery")
        artifact_path = marker.artifacts.get("discovered_circuits")
        if artifact_path is None:
            default_path = worker_layout.discovery_dir / "circuits" / "discovered_circuits.pt"
            artifact_path = str(default_path)
        store = CircuitStore()
        store.load(artifact_path)
        stores[worker_id] = store
    return stores


def merge_circuit_stores(worker_stores: Dict[int, CircuitStore]) -> CircuitStore:
    """Append worker circuits into a fresh CircuitStore, rejecting UUID collisions."""

    merged = CircuitStore()
    for worker_id in sorted(worker_stores):
        for circuit_uuid, circuit in worker_stores[worker_id].circuits.items():
            if circuit_uuid in merged.circuits:
                raise ValueError(f"duplicate circuit UUID during merge: {circuit_uuid}")
            merged.add_circuit(circuit)
    return merged


def build_circuit_summary(store: CircuitStore) -> List[Dict[str, Any]]:
    """Build the canonical JSON summary rows from a CircuitStore."""

    summary: List[Dict[str, Any]] = []
    for circuit in store.circuits.values():
        summary.append(
            {
                "name": circuit.name,
                "uuid": circuit.uuid,
                "nodes": len(circuit.nodes),
                "edges": len(circuit.edges),
                "metadata": {
                    key: value
                    for key, value in circuit.metadata.items()
                    if isinstance(value, (int, float, str, bool, dict))
                },
            }
        )
    return summary


def _summary_consistency_issues(
    store: CircuitStore,
    summary: Sequence[Dict[str, Any]],
) -> List[str]:
    issues: List[str] = []
    store_uuids = list(store.circuits)
    summary_uuids = [str(row.get("uuid")) for row in summary]
    if summary_uuids != store_uuids:
        issues.append("summary UUID order does not match merged circuit store")
        return issues
    for row in summary:
        uuid = str(row.get("uuid"))
        circuit = store.circuits[uuid]
        # Build a single-row summary using the same canonical formatter.
        single = CircuitStore()
        single.add_circuit(circuit)
        expected = build_circuit_summary(single)[0]
        if row != expected:
            issues.append(f"summary row does not match circuit store for {uuid}")
    return issues


def _circuit_metadata_issues(store: CircuitStore) -> List[str]:
    issues: List[str] = []
    for uuid, circuit in store.circuits.items():
        metadata = circuit.metadata
        method = metadata.get("discovery_method")
        if not method:
            issues.append(f"circuit {uuid} missing discovery_method metadata")
            continue
        if _is_seed_free_method(str(method)):
            if "cluster_id" not in metadata:
                issues.append(f"seed-free circuit {uuid} missing cluster_id metadata")
        else:
            for key in ("candidate_index", "seed_comp", "seed_latent"):
                if key not in metadata:
                    issues.append(f"seed-based circuit {uuid} missing {key} metadata")
        if not _has_eval_metadata(metadata):
            issues.append(f"circuit {uuid} missing eval metadata")
    return issues


def _has_eval_metadata(metadata: Dict[str, Any]) -> bool:
    evals = metadata.get("evals")
    if isinstance(evals, dict) and evals:
        return True
    return any(key in metadata for key in ("faithfulness", "kl_faithfulness"))


def _method_counts(store: CircuitStore) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for circuit in store.circuits.values():
        method = str(circuit.metadata.get("discovery_method", "unknown"))
        counts[method] = counts.get(method, 0) + 1
    return counts


def _eval_summary(store: CircuitStore) -> Dict[str, Dict[str, float | int]]:
    values_by_metric: Dict[str, List[float]] = {}
    for circuit in store.circuits.values():
        metadata = circuit.metadata
        evals = metadata.get("evals")
        if isinstance(evals, dict):
            source = evals
        else:
            source = metadata
        for key in ("faithfulness", "specificity", "kl_faithfulness"):
            value = source.get(key)
            if isinstance(value, (int, float)):
                values_by_metric.setdefault(key, []).append(float(value))
    summary: Dict[str, Dict[str, float | int]] = {}
    for key, values in values_by_metric.items():
        summary[key] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def _worker_reports(
    manifest: DistributedRunManifest,
    worker_counts: Dict[int, int],
) -> List[Dict[str, Any]]:
    layout = build_run_layout(manifest)
    reports: List[Dict[str, Any]] = []
    for worker_id in sorted(worker_counts):
        worker_layout = layout.workers[worker_id]
        marker = read_worker_marker(worker_layout.completed_marker)
        stats = _load_worker_discovery_stats(worker_layout, marker.artifacts)
        method_counts: Dict[str, int] = {}
        for method in stats.get("methods", []):
            method_counts[str(method)] = method_counts.get(str(method), 0) + 1
        for metric in stats.get("task_metrics", []):
            method = metric.get("method")
            if method is not None:
                method_counts[str(method)] = method_counts.get(str(method), 0) + 0
        reports.append(
            {
                "worker_id": worker_id,
                "duration_s": marker.duration_s,
                "seed_count": marker.seed_count,
                "peak_cuda_memory_bytes": marker.peak_cuda_memory_bytes,
                "accepted_circuit_count": worker_counts[worker_id],
                "method_count": stats.get("method_count", len(method_counts)),
                "methods": stats.get("methods", sorted(method_counts)),
                "planned_task_count": stats.get("planned_task_count", 0),
                "estimated_task_cost": stats.get("estimated_task_cost", 0.0),
                "failed_task_ranges": manifest.work_assignments.discovery_failed_task_ranges.get(
                    str(worker_id),
                    [],
                ),
                "task_metrics": stats.get("task_metrics", []),
            }
        )
    return reports


def _load_worker_discovery_stats(worker_layout, artifacts: Dict[str, str]) -> Dict[str, Any]:
    stats_path = artifacts.get("worker_discovery_stats")
    path = Path(stats_path) if stats_path is not None else worker_layout.discovery_dir / "worker_discovery_stats.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _is_seed_free_method(method: str) -> bool:
    return method == "cluster_contrast"


def _write_summary_xlsx(path: Path, summary: Sequence[Dict[str, Any]]) -> Path | None:
    if not summary:
        return None
    try:
        import pandas as pd
    except Exception:
        return None
    rows: List[Dict[str, Any]] = []
    for item in summary:
        row = {
            "name": item["name"],
            "uuid": item["uuid"],
            "nodes": item["nodes"],
            "edges": item["edges"],
        }
        row.update(_flatten_dict(item.get("metadata", {})))
        rows.append(row)
    tmp_path = path.with_name(f"{path.name}.tmp.xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Circuits", index=False)
    os.replace(tmp_path, path)
    return path


def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, next_key))
        else:
            flattened[next_key] = value
    return flattened


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)
