import json
from pathlib import Path

import pytest
import torch

from pipeline import candidate_selection
from pipeline.distributed.manifest import (
    DistributedRunManifest,
    WorkAssignments,
    generate_run_id,
    load_manifest,
    save_manifest,
)
from pipeline.distributed.shard_table import ShardRecord


REQUIRED_ARTIFACTS = (
    "latent_stats.pt",
    "top_ctx.pt",
    "mid_ctx.pt",
    "neg_ctx.pt",
    "logit_ctx.pt",
    "top_coactivation.pt",
)


class FakeSelector:
    def __init__(self, n_seeds):
        self.n_seeds = n_seeds

    def select_candidates(self):
        return [
            {
                "comp_idx": 1,
                "latent_idx": 2,
                "score": 1.5,
                "reason": "connectivity",
                "criteria_scores": {"connectivity": 1.0, "surprise": 0.5},
            }
        ]

    def get_summary_stats(self, candidates):
        assert candidates


def _records(counts):
    records = []
    next_id = 1
    for shard_index, count in enumerate(counts):
        records.append(
            ShardRecord(
                shard_index=shard_index,
                shard_filename=f"shard_{shard_index}.npy",
                sequence_count=count,
                global_start_id=next_id,
                global_end_id=next_id + count,
                shard_size_bytes=1,
                shard_mtime_ns=1,
                index_filename=f".shard_indices/shard_{shard_index}.idx.npy",
                index_size_bytes=1,
                index_mtime_ns=1,
            )
        )
        next_id += count
    return records


def _manifest(tmp_path: Path, worker_count: int = 3) -> DistributedRunManifest:
    config_hash = "abcdef1234567890"
    run_id = generate_run_id(config_hash)
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    if worker_count == 1:
        pass1_shards = {"0": [0, 1]}
        pass1_totals = {"0": 4}
    else:
        pass1_shards = {"0": [0], "1": [1]}
        pass1_totals = {"0": 2, "1": 2}
        for worker_id in range(2, worker_count):
            pass1_shards[str(worker_id)] = []
            pass1_totals[str(worker_id)] = 0
    return DistributedRunManifest(
        run_id=run_id,
        run_mode="distributed_simple_exact",
        created_at="2026-05-23T01:00:00Z",
        config_path=str(tmp_path / "config.yaml"),
        normalized_config_hash=config_hash,
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
        shard_table=_records([2, 2]),
        work_assignments=WorkAssignments(
            pass1_shards=pass1_shards,
            pass1_sequence_totals=pass1_totals,
        ),
    )


def _write_required_artifacts(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ARTIFACTS:
        (run_root / name).write_bytes(f"{name}\n".encode("utf-8"))


def test_candidate_selection_stage_writes_candidates_metadata_and_markers(monkeypatch, tmp_path):
    run_root = tmp_path / "outputs" / "20260523-010203-abcdef12"
    _write_required_artifacts(run_root)
    loaded = {}

    def fake_load_inputs(output_root):
        loaded["output_root"] = Path(output_root)

    monkeypatch.setattr(candidate_selection, "load_candidate_selection_inputs", fake_load_inputs)

    result = candidate_selection.run_candidate_selection_stage(
        run_root,
        expected_config_hash="abcdef123456",
        selector_cls=FakeSelector,
    )

    assert loaded["output_root"] == run_root
    assert result.candidates_path == run_root / "candidates.pt"
    assert result.metadata_path == (
        run_root
        / "distributed"
        / "parts"
        / "candidate_selection"
        / "candidate_selection_metadata.json"
    )
    assert torch.load(result.candidates_path, weights_only=False) == FakeSelector(1).select_candidates()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["config_hash"] == "abcdef123456"
    assert metadata["selected_count"] == 1
    assert set(metadata["artifact_hashes"]) == {
        "latent_stats",
        "top_ctx",
        "mid_ctx",
        "neg_ctx",
        "logit_ctx",
        "top_coactivation",
    }
    assert metadata["per_candidate_criterion_scores"] == [
        {
            "candidate_index": 0,
            "comp_idx": 1,
            "latent_idx": 2,
            "score": 1.5,
            "criteria_scores": {"connectivity": 1.0, "surprise": 0.5},
        }
    ]
    completed = json.loads(
        (
            run_root
            / "distributed"
            / "parts"
            / "candidate_selection"
            / "completed.json"
        ).read_text(encoding="utf-8")
    )
    assert completed["status"] == "completed"
    assert completed["artifacts"]["candidates"] == str(run_root / "candidates.pt")


def test_candidate_selection_stage_fails_before_selection_when_inputs_missing(monkeypatch, tmp_path):
    run_root = tmp_path / "outputs" / "20260523-010203-abcdef12"
    run_root.mkdir(parents=True)
    (run_root / "latent_stats.pt").write_bytes(b"latent_stats")

    def fail_if_called(_output_root):
        raise AssertionError("load should not run when required artifacts are missing")

    monkeypatch.setattr(candidate_selection, "load_candidate_selection_inputs", fail_if_called)

    with pytest.raises(FileNotFoundError, match="missing candidate-selection input artifacts"):
        candidate_selection.run_candidate_selection_stage(
            run_root,
            expected_config_hash="abcdef123456",
            selector_cls=FakeSelector,
        )


def test_single_process_and_distributed_stage_use_same_candidate_selection(monkeypatch, tmp_path):
    single_root = tmp_path / "single"
    distributed_root = tmp_path / "outputs" / "20260523-010203-abcdef12"
    _write_required_artifacts(distributed_root)
    monkeypatch.setattr(candidate_selection, "CandidateSelector", FakeSelector)
    monkeypatch.setattr(candidate_selection, "load_candidate_selection_inputs", lambda _root: None)

    single_candidates = candidate_selection.run_candidate_selection(output_root=str(single_root))
    stage_result = candidate_selection.run_candidate_selection_stage(
        distributed_root,
        expected_config_hash="abcdef123456",
        selector_cls=FakeSelector,
    )

    assert stage_result.candidates == single_candidates
    assert torch.load(distributed_root / "candidates.pt", weights_only=False) == single_candidates


def test_assign_discovery_candidates_to_manifest_preserves_order_for_one_worker(tmp_path):
    manifest = _manifest(tmp_path, worker_count=1)
    candidates = [
        {"comp_idx": 0, "latent_idx": 7},
        {"comp_idx": 2, "latent_idx": 9},
    ]

    updated = candidate_selection.assign_discovery_candidates_to_manifest(
        manifest,
        candidates,
        methods=["coactivation_statistical", "logit_attribution"],
    )

    assert updated.work_assignments.discovery_seed_ids == {"0": [0, 1]}
    assignments = updated.work_assignments.discovery_candidate_assignments["0"]
    assert [(item.comp_idx, item.latent_idx) for item in assignments] == [(0, 7), (2, 9)]
    assert [item.methods for item in assignments] == [
        ["coactivation_statistical", "logit_attribution"],
        ["coactivation_statistical", "logit_attribution"],
    ]
    assert [item.estimated_task_count for item in assignments] == [2, 2]


def test_assign_discovery_candidates_records_seed_free_owner(tmp_path):
    manifest = _manifest(tmp_path, worker_count=3)
    candidates = [
        {"comp_idx": 0, "latent_idx": 7},
        {"comp_idx": 2, "latent_idx": 9},
    ]

    updated = candidate_selection.assign_discovery_candidates_to_manifest(
        manifest,
        candidates,
        methods=["coactivation_statistical", "cluster_contrast"],
    )

    assert updated.work_assignments.discovery_seed_free_method_owners == {
        "cluster_contrast": 0
    }
    assert updated.work_assignments.discovery_scheduling_strategy == "candidate_contiguous"
    assert updated.work_assignments.discovery_worker_estimated_costs == {
        "0": 2.0,
        "1": 1.0,
        "2": 0.0,
    }
    for assignments in updated.work_assignments.discovery_candidate_assignments.values():
        for assignment in assignments:
            assert assignment.methods == ["coactivation_statistical"]
            assert assignment.estimated_task_count == 1
    assert [
        task.method
        for task in updated.work_assignments.discovery_task_assignments["0"]
    ] == ["coactivation_statistical", "cluster_contrast"]


def test_assign_discovery_candidates_handles_uneven_and_more_workers_than_candidates(tmp_path):
    candidates = [
        {"comp_idx": index, "latent_idx": index + 10}
        for index in range(5)
    ]
    updated = candidate_selection.assign_discovery_candidates_to_manifest(
        _manifest(tmp_path, worker_count=3),
        candidates,
        methods=["method_a"],
    )

    assert updated.work_assignments.discovery_seed_ids == {
        "0": [0, 1],
        "1": [2, 3],
        "2": [4],
    }
    assert [
        item.candidate_index
        for worker_items in updated.work_assignments.discovery_candidate_assignments.values()
        for item in worker_items
    ] == [0, 1, 2, 3, 4]

    sparse = candidate_selection.assign_discovery_candidates_to_manifest(
        _manifest(tmp_path, worker_count=4),
        candidates[:2],
        methods=["method_a"],
    )
    assert sparse.work_assignments.discovery_seed_ids == {
        "0": [0],
        "1": [1],
        "2": [],
        "3": [],
    }
    assert sparse.work_assignments.discovery_candidate_assignments["2"] == []
    assert sparse.work_assignments.discovery_candidate_assignments["3"] == []


def test_candidate_selection_stage_records_assignments_in_manifest(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path, worker_count=2)
    save_manifest(manifest, manifest.manifest_path)
    run_root = Path(manifest.output_root)
    _write_required_artifacts(run_root)
    monkeypatch.setattr(candidate_selection, "load_candidate_selection_inputs", lambda _root: None)

    candidate_selection.run_candidate_selection_stage(
        run_root,
        manifest_path=manifest.manifest_path,
        selector_cls=FakeSelector,
    )

    updated = load_manifest(manifest.manifest_path)
    assert updated.work_assignments.discovery_seed_ids == {"0": [0], "1": []}
    assigned = updated.work_assignments.discovery_candidate_assignments["0"][0]
    assert (assigned.comp_idx, assigned.latent_idx) == (1, 2)
    assert assigned.methods == list(candidate_selection.config.discovery.methods)
    report_path = Path(manifest.distributed_root) / "reports" / "discovery_scheduling_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scheduling_strategy"] == "candidate_contiguous"
    assert report["workers"][0]["task_count"] == len(candidate_selection.config.discovery.methods)
    assert report["workers"][1]["task_count"] == 0
