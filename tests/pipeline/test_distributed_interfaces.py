import inspect
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import config
from pipeline import candidate_selection, persist, run_pipeline, second_pass
from pipeline.distributed.interfaces import (
    build_output_paths,
    get_worker_output_paths,
    get_worker_seed_ids,
    get_worker_sequence_ids,
    get_worker_shard_ids,
    manifest_field_consumers,
    resolve_output_path,
)

from pipeline.distributed.manifest import (
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
    WorkAssignments,
)
from pipeline.runtime import PipelineRuntime, clear_runtime, set_runtime


def _minimal_manifest(tmp_path: Path, worker_count: int = 1) -> DistributedRunManifest:
    config_hash = "abcdef1234567890"
    run_id = "20260517-002500-abcdef12"
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    return DistributedRunManifest(
        run_id=run_id,
        run_mode="distributed_simple_exact",
        status="planned",
        cleanup_policy=CleanupPolicy.KEEP_ALL,
        created_at=datetime(2026, 5, 17, 0, 25, 0, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
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
        devices=[
            DeviceAssignment(worker_id=worker_id, physical_id=worker_id, logical_id="cuda:0")
            for worker_id in range(worker_count)
        ],
    )


def test_normal_pipeline_run_signature_stays_unchanged():
    assert list(inspect.signature(run_pipeline.run).parameters) == []


def test_output_paths_resolve_under_run_root_without_schema_changes(tmp_path):
    run_root = tmp_path / "outputs" / "20260517-002500-abcdef12"
    paths = build_output_paths(run_root)

    assert paths.latent_stats == run_root / "latent_stats.pt"
    assert paths.top_coactivation == run_root / "top_coactivation.pt"
    assert paths.candidates == run_root / "candidates.pt"
    assert paths.search_cache == run_root / "search_cache.parquet"
    assert paths.seq_latent_index_dir == run_root / "seq_latent_index"
    assert resolve_output_path(run_root, "circuits/summary.json") == (
        run_root / "circuits" / "summary.json"
    )


def test_single_process_output_paths_resolve_under_outputs_run_id():
    paths = build_output_paths(Path("outputs") / "20260517-002500-abcdef12")

    assert paths.run_root == Path("outputs") / "20260517-002500-abcdef12"
    assert paths.latent_stats == Path("outputs") / "20260517-002500-abcdef12" / "latent_stats.pt"
    assert paths.search_cache == Path("outputs") / "20260517-002500-abcdef12" / "search_cache.parquet"


def test_worker_assignment_and_output_interfaces(tmp_path):
    manifest = _minimal_manifest(tmp_path, worker_count=2).model_copy(
        update={
            "work_assignments": WorkAssignments(
                pass1_shards={"0": [0, 2], "1": [1]},
                pass1_sequence_totals={"0": 8, "1": 4},
                pass2_sequence_ids={"0": [1, 2], "1": [3]},
                discovery_seed_ids={"0": [10], "1": [20, 30]},
            )
        }
    )

    assert get_worker_shard_ids(manifest, 0) == [0, 2]
    assert get_worker_sequence_ids(manifest, 1) == [3]
    assert get_worker_seed_ids(manifest, 1) == [20, 30]
    assert get_worker_output_paths(manifest, 0).root == (
        Path(manifest.distributed_root) / "workers" / "worker_000"
    )


def test_manifest_field_consumers_documents_later_parts():
    consumers = manifest_field_consumers()

    assert "pass 1" in consumers["shard_table"]
    assert "pass 2 dump workers" in consumers["work_assignments.pass2_sequence_ids"]
    assert "distributed discovery workers" in consumers["work_assignments.discovery_seed_ids"]
    assert "runtime device isolation" in consumers["devices"]


def test_save_results_can_write_to_run_root(monkeypatch, tmp_path):
    saved = {}

    class FakeStore:
        def __init__(self, name):
            self.name = name

        def save(self, path):
            saved[self.name] = Path(path)
            Path(path).write_text(self.name, encoding="utf-8")

    class FakeSeqRepr:
        def save(self, path):
            saved["seq_repr"] = Path(path)
            Path(path).write_text("seq_repr", encoding="utf-8")

    runtime = PipelineRuntime(
        fast=False,
        compile=False,
        devices=[torch.device("cpu")],
        device=torch.device("cpu"),
        cpu_device=torch.device("cpu"),
        multi_gpu=False,
        mid_ctx_warmup=0,
        loader=object(),
        bank=object(),
        seq_repr=FakeSeqRepr(),
    )
    set_runtime(runtime)
    monkeypatch.setattr(config.persist, "save_workers", 1)
    monkeypatch.setattr(config.latents.seq_latent_index, "enabled", False)
    monkeypatch.setattr(persist, "latent_stats", FakeStore("latent_stats"))
    monkeypatch.setattr(persist, "top_ctx", FakeStore("top_ctx"))
    monkeypatch.setattr(persist, "mid_ctx", FakeStore("mid_ctx"))
    monkeypatch.setattr(persist, "logit_ctx", FakeStore("logit_ctx"))

    try:
        persist.save_results(output_root=str(tmp_path / "run-root"))
    finally:
        clear_runtime()

    assert (tmp_path / "run-root" / "latent_stats.pt").read_text(encoding="utf-8") == "latent_stats"
    assert (tmp_path / "run-root" / "top_ctx.pt").read_text(encoding="utf-8") == "top_ctx"
    assert (tmp_path / "run-root" / "mid_ctx.pt").read_text(encoding="utf-8") == "mid_ctx"
    assert (tmp_path / "run-root" / "seq_repr.pt").read_text(encoding="utf-8") == "seq_repr"
    assert (tmp_path / "run-root" / "logit_ctx.pt").read_text(encoding="utf-8") == "logit_ctx"


def test_candidate_selection_can_write_to_run_root(monkeypatch, tmp_path):
    class FakeSelector:
        def __init__(self, n_seeds):
            self.n_seeds = n_seeds

        def select_candidates(self):
            return [{"latent": 1}]

        def get_summary_stats(self, candidates):
            assert candidates == [{"latent": 1}]

    saved = {}

    def fake_save(obj, path):
        saved["obj"] = obj
        saved["path"] = Path(path)

    monkeypatch.setattr(candidate_selection, "CandidateSelector", FakeSelector)
    monkeypatch.setattr(candidate_selection.torch, "save", fake_save)

    candidates = candidate_selection.run_candidate_selection(
        output_root=str(tmp_path / "run-root")
    )

    assert candidates == [{"latent": 1}]
    assert saved["path"] == tmp_path / "run-root" / "candidates.pt"


def test_second_pass_can_write_to_run_root(monkeypatch, tmp_path):
    seen = {}

    class FakeLoader:
        batch_size = 1

        def get_batches_by_ids(self, _ids):
            yield torch.tensor([1], dtype=torch.int32), torch.ones((1, 2), dtype=torch.long)

    class FakeModel:
        def forward(self, *args, **kwargs):
            kwargs["activations_callback"](0, (torch.ones((1, 2, 3)),))

    class FakeTopCtx:
        def get_all_sequence_ids(self):
            return [1]

        def get_sequence_to_latents_csr(self, device):
            assert device == torch.device("cpu")
            return torch.tensor([0, 1]), torch.tensor([0])

    class FakeTopCoactivation:
        mode = "raw"

        def set_device(self, device):
            seen["set_device"] = device

        def prepare_dump(self, sequence_ids):
            seen["sequence_ids"] = sequence_ids

        def update_batch(self, batch_ids, current_batch_latents, dump_row_start):
            seen["batch_ids"] = batch_ids.tolist()
            seen["dump_row_start"] = dump_row_start
            assert 0 in current_batch_latents

        def dump_timing_summary(self):
            return "timing"

        def reduce(self, seq_offsets, seq_targets_global, seq_len, active_count):
            seen["seq_len"] = seq_len

        def save(self, path):
            seen["save_path"] = Path(path)

    runtime = PipelineRuntime(
        fast=True,
        compile=False,
        devices=[torch.device("cpu")],
        device=torch.device("cpu"),
        cpu_device=torch.device("cpu"),
        multi_gpu=False,
        mid_ctx_warmup=0,
        loader=FakeLoader(),
        model=FakeModel(),
        bank=object(),
    )
    set_runtime(runtime)
    monkeypatch.setattr(second_pass, "top_ctx", FakeTopCtx())
    monkeypatch.setattr(second_pass, "top_coactivation", FakeTopCoactivation())
    monkeypatch.setattr(second_pass.latent_stats, "active_count", torch.ones(1))
    monkeypatch.setattr(
        second_pass,
        "encode_layer_components",
        lambda *_args, **_kwargs: {0: (torch.ones((1, 2)), torch.ones((1, 2), dtype=torch.int32))},
    )
    monkeypatch.setattr(config.latents.top_coactivation, "dump_profile", False)

    try:
        second_pass.run_second_pass(output_root=str(tmp_path / "run-root"))
    finally:
        clear_runtime()

    assert seen["save_path"] == tmp_path / "run-root" / "top_coactivation.pt"
