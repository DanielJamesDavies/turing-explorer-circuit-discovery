import json
from pathlib import Path

import pytest
import torch

from pipeline.distributed.discovery_merge import (
    build_circuit_summary,
    load_completed_worker_circuit_stores,
    merge_circuit_stores,
    run_circuit_store_merge,
    validate_merged_discovery_outputs,
)
from pipeline.distributed.layout import build_run_layout, build_worker_marker, write_worker_marker
from pipeline.distributed.manifest import CleanupPolicy, DeviceAssignment, DistributedRunManifest, WorkAssignments
from store.circuits import Circuit, CircuitNode, CircuitStore


def _manifest(tmp_path: Path, worker_count: int = 2) -> DistributedRunManifest:
    run_id = "20260517-002500-abcdef12"
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    return DistributedRunManifest(
        run_id=run_id,
        run_mode="distributed_simple_exact",
        status="planned",
        cleanup_policy=CleanupPolicy.KEEP_ALL,
        created_at="2026-05-17T00:25:00Z",
        config_path=str(tmp_path / "config.yaml"),
        normalized_config_hash="abcdef1234567890",
        project_root=str(tmp_path),
        output_root=str(output_root),
        distributed_root=str(distributed_root),
        manifest_path=str(distributed_root / "manifest.json"),
        metrics_path=str(distributed_root / "reports" / "run_metrics.jsonl"),
        run_summary_path=str(distributed_root / "reports" / "run_summary.json"),
        model_path=str(tmp_path / "model.pt"),
        sae_path=str(tmp_path / "sae"),
        dataset_path=str(tmp_path / "data"),
        worker_count=worker_count,
        devices=[
            DeviceAssignment(worker_id=worker_id, physical_id=worker_id, logical_id="cuda:0")
            for worker_id in range(worker_count)
        ],
        work_assignments=WorkAssignments(),
    )


def _circuit(uuid: str, *, worker_id: int, method: str = "mock_method") -> Circuit:
    circuit = Circuit(name=f"circuit-{uuid}", uuid=uuid)
    circuit.add_node(CircuitNode(metadata={"kind": "mlp"}))
    circuit.metadata.update(
        {
            "run_id": "20260517-002500-abcdef12",
            "worker_id": worker_id,
            "candidate_index": worker_id,
            "seed_comp": worker_id + 1,
            "seed_latent": worker_id + 10,
            "discovery_method": method,
            "evals": {"faithfulness": 0.5 + worker_id},
            "post_analysis": {"layer_mean": float(worker_id)},
            "seed_criteria": {"connectivity": 1.0},
        }
    )
    return circuit


def _cluster_circuit(uuid: str, *, worker_id: int) -> Circuit:
    circuit = Circuit(name=f"cluster-{uuid}", uuid=uuid)
    circuit.add_node(CircuitNode(metadata={"kind": "mlp"}))
    circuit.metadata.update(
        {
            "run_id": "20260517-002500-abcdef12",
            "worker_id": worker_id,
            "cluster_id": 7,
            "cluster_size": 11,
            "discovery_method": "cluster_contrast",
            "faithfulness": 0.7,
            "specificity": 0.6,
        }
    )
    return circuit


def _write_worker_store(
    manifest: DistributedRunManifest,
    worker_id: int,
    circuits,
    *,
    task_metrics=None,
) -> Path:
    layout = build_run_layout(manifest)
    worker_layout = layout.workers[worker_id]
    path = worker_layout.discovery_dir / "circuits" / "discovered_circuits.pt"
    stats_path = worker_layout.discovery_dir / "worker_discovery_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({circuit.uuid: circuit for circuit in circuits}, path)
    (worker_layout.discovery_dir / "circuits" / "summary.json").write_text("[]", encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "worker_id": worker_id,
                "accepted_circuit_count": len(circuits),
                "method_count": len({circuit.metadata.get("discovery_method") for circuit in circuits}),
                "methods": sorted(
                    {
                        str(circuit.metadata.get("discovery_method"))
                        for circuit in circuits
                    }
                ),
                "planned_task_count": len(task_metrics or []),
                "estimated_task_cost": float(len(task_metrics or [])),
                "task_metrics": list(task_metrics or []),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_worker_marker(
        build_worker_marker(
            manifest,
            worker_id,
            phase="discovery",
            status="started",
            start_time="2026-05-17T00:25:00Z",
        ),
        worker_layout.started_marker,
    )
    write_worker_marker(
        build_worker_marker(
            manifest,
            worker_id,
            phase="discovery",
            status="completed",
            start_time="2026-05-17T00:25:00Z",
            end_time="2026-05-17T00:26:00Z",
            duration_s=60.0,
            seed_count=len(circuits),
            artifacts={
                "discovered_circuits": str(path),
                "worker_discovery_stats": str(stats_path),
            },
        ),
        worker_layout.completed_marker,
    )
    return path


def test_merge_circuit_stores_preserves_metadata_and_order():
    store0 = CircuitStore()
    store1 = CircuitStore()
    first = _circuit("circuit-a", worker_id=0, method="method_a")
    second = _circuit("circuit-b", worker_id=1, method="method_b")
    store0.add_circuit(first)
    store1.add_circuit(second)

    merged = merge_circuit_stores({1: store1, 0: store0})

    assert list(merged.circuits) == ["circuit-a", "circuit-b"]
    assert merged.circuits["circuit-b"].metadata["evals"]["faithfulness"] == 1.5
    assert merged.circuits["circuit-b"].metadata["post_analysis"]["layer_mean"] == 1.0
    assert merged.circuits["circuit-b"].metadata["seed_criteria"]["connectivity"] == 1.0


def test_merge_circuit_stores_rejects_duplicate_uuid():
    store0 = CircuitStore()
    store1 = CircuitStore()
    store0.add_circuit(_circuit("duplicate", worker_id=0))
    store1.add_circuit(_circuit("duplicate", worker_id=1))

    with pytest.raises(ValueError, match="duplicate circuit UUID"):
        merge_circuit_stores({0: store0, 1: store1})


def test_load_completed_worker_circuit_stores_accepts_empty_worker_store(tmp_path):
    manifest = _manifest(tmp_path, worker_count=2)
    _write_worker_store(manifest, 0, [_circuit("circuit-a", worker_id=0)])
    _write_worker_store(manifest, 1, [])

    stores = load_completed_worker_circuit_stores(manifest)

    assert len(stores[0].circuits) == 1
    assert stores[1].circuits == {}


def test_run_circuit_store_merge_writes_canonical_outputs(tmp_path):
    manifest = _manifest(tmp_path, worker_count=2)
    _write_worker_store(manifest, 0, [_circuit("circuit-a", worker_id=0, method="method_a")])
    _write_worker_store(manifest, 1, [_circuit("circuit-b", worker_id=1, method="method_b")])

    result = run_circuit_store_merge(manifest)

    assert result.merged_circuit_count == 2
    assert result.worker_circuit_counts == {0: 1, 1: 1}
    loaded = torch.load(result.circuit_store_path, weights_only=False)
    assert list(loaded) == ["circuit-a", "circuit-b"]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert [row["uuid"] for row in summary] == ["circuit-a", "circuit-b"]
    assert summary[0]["metadata"]["discovery_method"] == "method_a"
    assert summary[1]["metadata"]["discovery_method"] == "method_b"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["merged_circuit_count"] == 2
    assert report["worker_circuit_counts"] == {"0": 1, "1": 1}
    assert report["validation"]["ok"] is True
    assert report["validation"]["merged_count_matches_worker_sum"] is True
    assert report["validation"]["summary_rows_match_store"] is True
    assert report["method_counts"] == {"method_a": 1, "method_b": 1}
    assert report["eval_summary"]["faithfulness"]["count"] == 2
    assert report["worker_reports"][0]["duration_s"] == 60.0
    assert report["worker_reports"][0]["accepted_circuit_count"] == 1


def test_build_circuit_summary_handles_mixed_method_outputs():
    store = CircuitStore()
    store.add_circuit(_circuit("statistical", worker_id=0, method="coactivation_statistical"))
    store.add_circuit(_circuit("gradient", worker_id=1, method="logit_attribution"))

    summary = build_circuit_summary(store)

    assert [item["metadata"]["discovery_method"] for item in summary] == [
        "coactivation_statistical",
        "logit_attribution",
    ]


def test_run_circuit_store_merge_reports_seed_free_and_failed_ranges(tmp_path):
    manifest = _manifest(tmp_path, worker_count=2).model_copy(
        update={
            "work_assignments": WorkAssignments(
                discovery_seed_free_method_owners={"cluster_contrast": 0},
                discovery_failed_task_ranges={"1": [[3, 4]]},
            )
        }
    )
    _write_worker_store(
        manifest,
        0,
        [_cluster_circuit("cluster-a", worker_id=0)],
        task_metrics=[
            {
                "method": "cluster_contrast",
                "duration_s": 1.5,
                "forward_pass_count": 4,
                "accepted_circuit_count": 1,
            }
        ],
    )
    _write_worker_store(manifest, 1, [_circuit("seed-a", worker_id=1)])

    result = run_circuit_store_merge(manifest)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["seed_free_method_counts"] == {"cluster_contrast": 1}
    assert report["validation"]["seed_free_circuit_count"] == 1
    assert report["failed_task_ranges"] == {"1": [[3, 4]]}
    assert report["worker_reports"][0]["task_metrics"][0]["forward_pass_count"] == 4
    assert report["eval_summary"]["faithfulness"]["count"] == 2


def test_one_worker_merge_matches_single_worker_store_on_synthetic_setup(tmp_path):
    manifest = _manifest(tmp_path, worker_count=1)
    circuit = _circuit("single-worker", worker_id=0, method="coactivation_statistical")
    _write_worker_store(manifest, 0, [circuit])

    result = run_circuit_store_merge(manifest)

    merged = torch.load(result.circuit_store_path, weights_only=False)
    assert merged == {circuit.uuid: circuit}
    store = CircuitStore()
    store.add_circuit(circuit)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary == build_circuit_summary(store)


def test_two_worker_synthetic_discovery_eval_smoke_merges_and_reports(tmp_path):
    manifest = _manifest(tmp_path, worker_count=2).model_copy(
        update={
            "work_assignments": WorkAssignments(
                discovery_seed_ids={"0": [0], "1": [1]},
                discovery_failed_task_ranges={},
            )
        }
    )
    first = _circuit("worker-0-circuit", worker_id=0, method="coactivation_statistical")
    second = _circuit("worker-1-circuit", worker_id=1, method="logit_attribution")
    _write_worker_store(
        manifest,
        0,
        [first],
        task_metrics=[
            {
                "method": "coactivation_statistical",
                "duration_s": 0.2,
                "forward_pass_count": 1,
                "accepted_circuit_count": 1,
            }
        ],
    )
    _write_worker_store(
        manifest,
        1,
        [second],
        task_metrics=[
            {
                "method": "logit_attribution",
                "duration_s": 0.3,
                "forward_pass_count": 2,
                "accepted_circuit_count": 1,
            }
        ],
    )

    result = run_circuit_store_merge(manifest)

    assert result.validation["ok"] is True
    assert result.worker_circuit_counts == {0: 1, 1: 1}
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert [row["uuid"] for row in summary] == [first.uuid, second.uuid]
    assert report["eval_summary"]["faithfulness"]["count"] == 2
    assert report["method_counts"] == {
        "coactivation_statistical": 1,
        "logit_attribution": 1,
    }
    assert [item["task_metrics"][0]["forward_pass_count"] for item in report["worker_reports"]] == [1, 2]


def test_validate_merged_discovery_outputs_rejects_bad_summary(tmp_path):
    manifest = _manifest(tmp_path, worker_count=1)
    store = CircuitStore()
    store.add_circuit(_circuit("circuit-a", worker_id=0))
    bad_summary = build_circuit_summary(store)
    bad_summary[0]["nodes"] = 999

    with pytest.raises(ValueError, match="summary row does not match"):
        validate_merged_discovery_outputs(manifest, store, bad_summary, {0: 1})


def test_validate_merged_discovery_outputs_rejects_missing_metadata(tmp_path):
    manifest = _manifest(tmp_path, worker_count=1)
    store = CircuitStore()
    bad = Circuit(name="bad", uuid="bad")
    bad.metadata["discovery_method"] = "coactivation_statistical"
    store.add_circuit(bad)

    with pytest.raises(ValueError, match="missing candidate_index"):
        validate_merged_discovery_outputs(manifest, store, build_circuit_summary(store), {0: 1})
