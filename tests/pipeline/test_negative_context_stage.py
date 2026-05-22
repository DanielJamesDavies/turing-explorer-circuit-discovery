from pathlib import Path
from datetime import datetime, timezone
import json

import pytest
import torch

from pipeline.distributed.interfaces import build_output_paths
from pipeline.distributed.manifest import (
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
)
from pipeline.negative_context import (
    build_negative_context_comparison_report,
    compare_negative_context_backends,
    LoadedContext,
    load_negative_context_inputs,
    plan_negative_context_stage,
    run_negative_context_stage,
)
from store.neg_context import NegCtxStats


def _write_pass1_artifacts(
    run_root: Path,
    *,
    top_ids: torch.Tensor | None = None,
    mid_ids: torch.Tensor | None = None,
    n_seqs: int = 4,
    metadata: dict | None = None,
) -> None:
    paths = build_output_paths(run_root)
    paths.run_root.mkdir(parents=True, exist_ok=True)
    top_ctx_ids = (
        top_ids
        if top_ids is not None
        else torch.tensor([[[1, 2], [3, 0]]], dtype=torch.int32)
    )
    mid_ctx_ids = (
        mid_ids
        if mid_ids is not None
        else torch.tensor([[[4, 0], [2, 0]]], dtype=torch.int32)
    )
    common = {"metadata": metadata or {}}
    torch.save(
        {
            **common,
            "ctx_seq_idx": top_ctx_ids,
            "ctx_seq_val": torch.ones_like(top_ctx_ids, dtype=torch.float32),
            "ctx_type": "top",
        },
        paths.top_ctx,
    )
    torch.save(
        {
            **common,
            "ctx_seq_idx": mid_ctx_ids,
            "ctx_seq_val": torch.ones_like(mid_ctx_ids, dtype=torch.float32),
            "ctx_type": "mid",
            "mode": "distributed_priority_reservoir",
            "num_ctx_sequences": mid_ctx_ids.shape[2],
            "reservoir_fill": torch.ones(mid_ctx_ids.shape[:2], dtype=torch.int32),
            "reservoir_n": torch.ones(mid_ctx_ids.shape[:2], dtype=torch.int64),
        },
        paths.mid_ctx,
    )
    torch.save(
        {
            **common,
            "repr_buf": torch.arange((n_seqs + 1) * 3, dtype=torch.float32).reshape(
                n_seqs + 1,
                3,
            ),
            "repr_mode": "mean_pool",
            "repr_dim": 3,
            "n_seqs": n_seqs,
            "n_stored": n_seqs,
            "is_capped": False,
        },
        paths.seq_repr,
    )


def test_load_negative_context_inputs_accepts_tiny_pass1_artifacts(tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(
        run_root,
        metadata={"config_hash": "abcdef1234567890"},
    )

    inputs = load_negative_context_inputs(
        run_root,
        expected_config_hash="abcdef1234567890",
    )

    assert inputs.top_ctx.ctx_seq_idx.shape == (1, 2, 2)
    assert inputs.mid_ctx.mode == "distributed_priority_reservoir"
    assert inputs.seq_repr.n_seqs == 4
    assert inputs.seq_repr.repr_buf.shape == (5, 3)


def test_load_negative_context_inputs_rejects_missing_required_artifacts(tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    paths = build_output_paths(run_root)
    paths.run_root.mkdir(parents=True, exist_ok=True)
    torch.save({"ctx_seq_idx": torch.zeros((1, 1, 1)), "ctx_seq_val": torch.zeros((1, 1, 1)), "ctx_type": "top"}, paths.top_ctx)

    with pytest.raises(FileNotFoundError, match="mid_ctx"):
        load_negative_context_inputs(run_root)


def test_load_negative_context_inputs_rejects_incompatible_sequence_range(tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(run_root, n_seqs=3)

    with pytest.raises(ValueError, match="exceeds seq_repr n_seqs"):
        load_negative_context_inputs(run_root)


def test_load_negative_context_inputs_rejects_config_hash_mismatch(tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(run_root, metadata={"config_hash": "oldhash"})

    with pytest.raises(ValueError, match="config hash mismatch"):
        load_negative_context_inputs(run_root, expected_config_hash="newhash")


def test_load_negative_context_inputs_requires_config_hash_when_expected(tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(run_root)

    with pytest.raises(ValueError, match="config hash missing"):
        load_negative_context_inputs(run_root, expected_config_hash="abcdef1234567890")


def test_run_negative_context_stage_writes_run_root_outputs(monkeypatch, tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(run_root)
    monkeypatch.setattr("pipeline.negative_context.neg_ctx.num_ctx_sequences", 2)

    def fake_build_neg_ctx(seq_repr, top_ctx, mid_ctx, output_neg_ctx):
        assert seq_repr.n_seqs == 4
        assert top_ctx.ctx_seq_idx.shape == (1, 2, 2)
        assert mid_ctx.ctx_seq_idx.shape == (1, 2, 2)
        output_neg_ctx.ctx_seq_idx[0, 0] = torch.tensor([4, 3], dtype=torch.int32)
        output_neg_ctx.ctx_seq_val[0, 0] = torch.tensor([0.9, 0.8], dtype=torch.float32)
        return NegCtxStats(
            n_latents_attempted=2,
            n_latents_populated=1,
            fill_counts=[2],
            backend="single_gpu_exact",
            devices=["cpu"],
        )

    result = run_negative_context_stage(run_root, build_fn=fake_build_neg_ctx)

    assert result.neg_ctx_path == run_root / "neg_ctx.pt"
    assert result.stats_path == run_root / "neg_ctx_stats.json"
    assert result.neg_ctx_path.exists()
    assert result.stats_path.exists()
    payload = torch.load(result.neg_ctx_path, map_location="cpu", weights_only=False)
    assert payload["ctx_type"] == "neg"
    assert payload["ctx_seq_idx"][0, 0].tolist() == [4, 3]


def test_run_negative_context_stage_actual_cpu_backend_smoke(monkeypatch, tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    n_components = 36
    top_ids = torch.zeros((n_components, 2, 2), dtype=torch.int32)
    top_ids[:, 0] = torch.tensor([1, 2], dtype=torch.int32)
    top_ids[:, 1] = torch.tensor([3, 0], dtype=torch.int32)
    mid_ids = torch.zeros_like(top_ids)
    _write_pass1_artifacts(run_root, top_ids=top_ids, mid_ids=mid_ids, n_seqs=4)
    monkeypatch.setattr("pipeline.negative_context.config.hardware.ann_device", "cpu")
    monkeypatch.setattr("pipeline.negative_context.config.latents.neg_ctx.backend", "single_gpu_exact")
    monkeypatch.setattr("pipeline.negative_context.config.latents.neg_ctx.n_neighbors", 4)
    monkeypatch.setattr("pipeline.negative_context.config.latents.neg_ctx.min_pos_ctx", 1)
    monkeypatch.setattr("pipeline.negative_context.neg_ctx.num_ctx_sequences", 2)

    result = run_negative_context_stage(run_root)

    payload = torch.load(result.neg_ctx_path, map_location="cpu", weights_only=False)
    sanity = json.loads(
        (run_root / "distributed" / "parts" / "neg_ctx" / "neg_ctx_sanity_report.json")
        .read_text(encoding="utf-8")
    )
    assert payload["ctx_type"] == "neg"
    assert payload["ctx_seq_idx"].shape == (n_components, 2, 2)
    assert result.stats.backend == "single_gpu_exact"
    assert sanity["validation"]["invalid_sequence_count"] == 0
    assert sanity["validation"]["non_finite_similarity_count"] == 0


def test_run_negative_context_stage_uses_manifest_physical_devices(monkeypatch, tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    distributed_root = run_root / "distributed"
    manifest_path = distributed_root / "manifest.json"
    _write_pass1_artifacts(run_root, metadata={"config_hash": "abcdef1234567890"})
    manifest = DistributedRunManifest(
        run_id="20260517-002500-abcdef12",
        run_mode="distributed_simple_exact",
        status="planned",
        cleanup_policy=CleanupPolicy.KEEP_ALL,
        created_at=datetime(2026, 5, 17, 0, 25, 0, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        config_path=str(tmp_path / "config.yaml"),
        normalized_config_hash="abcdef1234567890",
        project_root=str(tmp_path),
        output_root=str(run_root),
        distributed_root=str(distributed_root),
        manifest_path=str(manifest_path),
        metrics_path=str(distributed_root / "reports" / "run_metrics.jsonl"),
        run_summary_path=str(distributed_root / "reports" / "run_summary.json"),
        model_path=str(tmp_path / "model.pt"),
        sae_path=str(tmp_path / "sae"),
        dataset_path=str(tmp_path / "data"),
        worker_count=2,
        devices=[
            DeviceAssignment(worker_id=0, physical_id=3, logical_id="cuda:0"),
            DeviceAssignment(worker_id=1, physical_id=1, logical_id="cuda:0"),
        ],
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    seen = {}

    def fake_build_neg_ctx(seq_repr, top_ctx, mid_ctx, output_neg_ctx, *, selected_devices=None):
        seen["selected_devices"] = selected_devices
        return NegCtxStats()

    monkeypatch.setattr("pipeline.negative_context.build_neg_ctx", fake_build_neg_ctx)

    run_negative_context_stage(run_root, manifest_path=manifest_path)

    assert seen["selected_devices"] == [3, 1]
    loaded_manifest = DistributedRunManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    assert loaded_manifest.neg_ctx.selected_devices == ["cuda:3", "cuda:1"]
    assert loaded_manifest.neg_ctx.device_selection_source == "manifest_declared_devices"


def test_negative_context_stage_writes_part_markers_and_sanity_report(capsys, tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(run_root)

    def fake_build_neg_ctx(_seq_repr, _top_ctx, _mid_ctx, output_neg_ctx):
        output_neg_ctx.ctx_seq_idx[0, 0, :2] = torch.tensor([4, 3], dtype=torch.int32)
        output_neg_ctx.ctx_seq_val[0, 0, :2] = torch.tensor([0.9, 0.8], dtype=torch.float32)
        return NegCtxStats(
            backend="single_gpu_exact",
            devices=["cpu"],
            fill_counts=[2],
            n_latents_zero_negatives=1,
            ann_index_memory_estimate_bytes=12,
            ann_query_working_memory_bytes=32,
            ann_total_memory_estimate_bytes=44,
            ann_memory_guardrail_fraction=0.9,
        )

    run_negative_context_stage(run_root, build_fn=fake_build_neg_ctx)

    part_dir = run_root / "distributed" / "parts" / "neg_ctx"
    assert (part_dir / "started.json").exists()
    assert (part_dir / "completed.json").exists()
    assert (part_dir / "neg_ctx_sanity_report.json").exists()
    completed = json.loads((part_dir / "completed.json").read_text(encoding="utf-8"))
    sanity = json.loads((part_dir / "neg_ctx_sanity_report.json").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["metadata"] == sanity["metadata"]
    assert sanity["status"] == "completed"
    assert sanity["populated_rows"] == 1
    assert sanity["zero_negative_rows"] == 1
    assert sanity["seq_repr"] == {
        "n_seqs": 4,
        "n_stored": 4,
        "repr_dim": 3,
        "is_capped": False,
        "cap_percent": 100.0,
    }
    assert sanity["validation"]["invalid_sequence_count"] == 0
    assert sanity["validation"]["non_finite_similarity_count"] == 0
    assert sanity["validation"]["negative_similarity_count"] == 0
    assert sanity["validation"]["valid_entry_count"] == 2
    assert sanity["metadata"]["memory_guardrail_fraction"] == 0.9
    assert sanity["memory"]["ann_total_memory_estimate_bytes"] == 44
    assert "[neg_ctx] summary" in capsys.readouterr().out


def test_negative_context_stage_rejects_invalid_output_and_writes_failed_marker(tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(run_root)

    def fake_build_neg_ctx(_seq_repr, _top_ctx, _mid_ctx, output_neg_ctx):
        output_neg_ctx.ctx_seq_idx[0, 0, 0] = 999
        output_neg_ctx.ctx_seq_val[0, 0, 0] = 0.9
        return NegCtxStats(backend="single_gpu_exact")

    with pytest.raises(ValueError, match="exceeds seq_repr"):
        run_negative_context_stage(run_root, build_fn=fake_build_neg_ctx)

    part_dir = run_root / "distributed" / "parts" / "neg_ctx"
    failed = json.loads((part_dir / "failed.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert "exceeds seq_repr" in failed["error"]
    assert not (part_dir / "completed.json").exists()
    assert not (run_root / "neg_ctx.pt").exists()


def test_negative_context_resume_skips_completed_matching_outputs(tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(run_root)
    calls = {"count": 0}

    def fake_build_neg_ctx(_seq_repr, _top_ctx, _mid_ctx, output_neg_ctx):
        calls["count"] += 1
        output_neg_ctx.ctx_seq_idx[0, 0, :2] = torch.tensor([4, 3], dtype=torch.int32)
        output_neg_ctx.ctx_seq_val[0, 0, :2] = torch.tensor([0.9, 0.8], dtype=torch.float32)
        return NegCtxStats(backend="single_gpu_exact")

    run_negative_context_stage(run_root, build_fn=fake_build_neg_ctx)
    plan = plan_negative_context_stage(run_root)
    assert plan.resume_status == "completed"

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("resume should skip rebuild")

    result = run_negative_context_stage(run_root, resume=True, build_fn=should_not_run)

    assert calls["count"] == 1
    assert result.neg_ctx_path.exists()


def test_negative_context_resume_classifies_missing_stale_and_failed(tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(run_root)
    assert plan_negative_context_stage(run_root).resume_status == "missing"

    def fake_build_neg_ctx(_seq_repr, _top_ctx, _mid_ctx, output_neg_ctx):
        output_neg_ctx.ctx_seq_idx[0, 0, :2] = torch.tensor([4, 3], dtype=torch.int32)
        output_neg_ctx.ctx_seq_val[0, 0, :2] = torch.tensor([0.9, 0.8], dtype=torch.float32)
        return NegCtxStats(backend="single_gpu_exact")

    run_negative_context_stage(run_root, build_fn=fake_build_neg_ctx)
    assert plan_negative_context_stage(run_root, expected_config_hash="newhash").resume_status == "stale"
    part_dir = run_root / "distributed" / "parts" / "neg_ctx"
    (part_dir / "failed.json").write_text(
        json.dumps({"status": "failed", "metadata": {}}),
        encoding="utf-8",
    )
    assert plan_negative_context_stage(run_root).resume_status == "failed"


def test_negative_context_stage_metadata_marks_sharded_backend_all_visible(monkeypatch, tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(run_root)
    monkeypatch.setattr(
        "pipeline.negative_context.config.latents.neg_ctx.backend",
        "multi_gpu_index_sharded_exact",
    )
    monkeypatch.setattr("pipeline.negative_context.config.latents.neg_ctx.devices", [])

    plan = plan_negative_context_stage(run_root)

    assert plan.metadata["device_selection_source"] == "standalone_all_visible"
    assert plan.metadata["selected_devices"] == []


def test_build_negative_context_comparison_report_detects_equivalent_outputs():
    single = LoadedContext(
        ctx_type="neg",
        ctx_seq_idx=torch.tensor([[[1, 2], [0, 0]]], dtype=torch.int32),
        ctx_seq_val=torch.tensor([[[0.9, 0.8], [0.0, 0.0]]], dtype=torch.float32),
        num_components=1,
        d_sae=2,
        num_ctx_sequences=2,
    )
    multi = LoadedContext(
        ctx_type="neg",
        ctx_seq_idx=single.ctx_seq_idx.clone(),
        ctx_seq_val=single.ctx_seq_val.clone(),
        num_components=1,
        d_sae=2,
        num_ctx_sequences=2,
    )

    report = build_negative_context_comparison_report(
        single,
        multi,
        single_stats=NegCtxStats(t_query=0.1),
        multi_stats=NegCtxStats(t_query=0.2),
    )

    assert report["status"] == "equivalent"
    assert report["exact_equivalent"] is True
    assert report["populated_rows"]["match"] is True
    assert report["fill_distribution"]["single_gpu_exact"]["histogram"] == {"0": 1, "1": 0, "2": 1}
    assert report["timing_ms"]["multi_gpu_exact"]["query_ms"] == 200.0


def test_build_negative_context_comparison_report_reports_differences():
    single = LoadedContext(
        ctx_type="neg",
        ctx_seq_idx=torch.tensor([[[1, 2]]], dtype=torch.int32),
        ctx_seq_val=torch.tensor([[[0.9, 0.8]]], dtype=torch.float32),
        num_components=1,
        d_sae=1,
        num_ctx_sequences=2,
    )
    multi = LoadedContext(
        ctx_type="neg",
        ctx_seq_idx=torch.tensor([[[1, 3]]], dtype=torch.int32),
        ctx_seq_val=torch.tensor([[[0.9, 0.7]]], dtype=torch.float32),
        num_components=1,
        d_sae=1,
        num_ctx_sequences=2,
    )

    report = build_negative_context_comparison_report(
        single,
        multi,
        single_stats=NegCtxStats(),
        multi_stats=NegCtxStats(),
    )

    assert report["status"] == "different"
    assert report["exact_equivalent"] is False
    assert report["indices_equal"] is False
    assert report["max_abs_value_diff"] > 0
    assert report["sample_rows"][0]["ids_equal"] is False


def test_compare_negative_context_backends_writes_equivalence_report(tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    _write_pass1_artifacts(run_root)

    def fill_output(_seq_repr, _top_ctx, _mid_ctx, output_neg_ctx):
        output_neg_ctx.ctx_seq_idx[0, 0, :2] = torch.tensor([4, 3], dtype=torch.int32)
        output_neg_ctx.ctx_seq_val[0, 0, :2] = torch.tensor([0.9, 0.8], dtype=torch.float32)
        return NegCtxStats(backend="single_gpu_exact", fill_counts=[2], t_total=0.1)

    def fill_multi_output(_seq_repr, _top_ctx, _mid_ctx, output_neg_ctx, *, selected_devices=None):
        assert selected_devices == [0, 1]
        output_neg_ctx.ctx_seq_idx[0, 0, :2] = torch.tensor([4, 3], dtype=torch.int32)
        output_neg_ctx.ctx_seq_val[0, 0, :2] = torch.tensor([0.9, 0.8], dtype=torch.float32)
        return NegCtxStats(backend="multi_gpu_exact", fill_counts=[2], t_total=0.2)

    result = compare_negative_context_backends(
        run_root,
        selected_devices=[0, 1],
        single_build_fn=fill_output,
        multi_build_fn=fill_multi_output,
    )

    assert result.report_path == run_root / "neg_ctx_equivalence_report.json"
    assert result.report_path.exists()
    assert result.report["exact_equivalent"] is True
