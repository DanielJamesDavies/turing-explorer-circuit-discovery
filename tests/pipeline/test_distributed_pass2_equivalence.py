from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from pipeline.distributed.layout import build_run_layout
from pipeline.distributed.manifest import (
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
    ShardRecord,
    WorkAssignments,
)
from pipeline.distributed.pass2_partials import load_candidate_dump_partial
from pipeline.distributed.pass2_replay import hash_replay_sequence_ids
from pipeline.distributed.worker import save_pass2_candidate_dump
from pipeline.runtime import PipelineRuntime, clear_runtime, set_runtime
from pipeline.second_pass import SecondPassDumpResult, run_second_pass_dump


REPLAY_IDS = [1, 2, 3, 4]
SEQ_LEN = 3


class FakeLoader:
    batch_size = 2

    def get_batches_by_ids(self, sequence_ids):
        for offset in range(0, len(sequence_ids), self.batch_size):
            chunk = [int(sequence_id) for sequence_id in sequence_ids[offset : offset + self.batch_size]]
            yield (
                torch.tensor(chunk, dtype=torch.int64),
                torch.tensor(
                    [[sequence_id, sequence_id + 10, sequence_id + 20] for sequence_id in chunk],
                    dtype=torch.int64,
                ),
            )


class FakeModel:
    def forward(
        self,
        tokens,
        *,
        num_gen,
        tokenize_final,
        activations_callback,
        return_activations,
    ):
        activations_callback(0, (tokens,))
        return None


def _manifest(tmp_path: Path, worker_count: int) -> DistributedRunManifest:
    if worker_count == 1:
        assignments = {"0": REPLAY_IDS}
    else:
        assignments = {"0": REPLAY_IDS[:2], "1": REPLAY_IDS[2:]}
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
        shard_table=[
            ShardRecord(
                shard_index=0,
                shard_filename="shard_0.npy",
                sequence_count=len(REPLAY_IDS),
                global_start_id=1,
                global_end_id=len(REPLAY_IDS) + 1,
                shard_size_bytes=1,
                shard_mtime_ns=1,
                index_filename=".shard_indices/shard_0.npy_sft1.idx.npy",
                index_size_bytes=1,
                index_mtime_ns=1,
            ),
        ],
        work_assignments=WorkAssignments(
            pass1_shards={str(worker_id): [] for worker_id in range(worker_count)},
            pass1_sequence_totals={str(worker_id): 0 for worker_id in range(worker_count)},
            pass2_sequence_ids=assignments,
            pass2_replay_sequence_count=len(REPLAY_IDS),
            pass2_replay_sequence_hash=hash_replay_sequence_ids(REPLAY_IDS),
        ),
    )


def _fake_encode_layer_components(_bank, _layer_idx, activations, *, primary_device, multi_gpu):
    tokens = activations[0].to(primary_device)
    sequence_ids = tokens[:, 0].long()
    comp0_indices = torch.stack(
        [
            sequence_ids % 6,
            (sequence_ids + 1) % 6,
            (sequence_ids + 2) % 6,
        ],
        dim=1,
    ).to(torch.int32)
    comp1_indices = torch.stack(
        [
            (sequence_ids + 3) % 6,
            (sequence_ids + 4) % 6,
            (sequence_ids + 5) % 6,
        ],
        dim=1,
    ).to(torch.int32)
    sequence_values = sequence_ids.float()
    comp0_acts = torch.stack(
        [
            sequence_values + 1.0,
            sequence_values + 2.0,
            sequence_values + 3.0,
        ],
        dim=1,
    )
    comp1_acts = torch.stack(
        [
            sequence_values + 4.0,
            sequence_values + 5.0,
            sequence_values + 6.0,
        ],
        dim=1,
    )
    return {
        0: (comp0_acts, comp0_indices),
        1: (comp1_acts, comp1_indices),
    }


def _configure_synthetic_runtime(monkeypatch, mode: str):
    import pipeline.distributed.worker as worker_module
    import pipeline.second_pass as second_pass_module

    top_coactivation = second_pass_module.top_coactivation
    top_coactivation.device = torch.device("cpu")
    top_coactivation.num_components = 2
    top_coactivation.d_sae = 6
    top_coactivation.n_candidates_per_component = 3
    top_coactivation.n_latents_per_latent = 4
    top_coactivation.M = 6
    top_coactivation._allocated = False
    top_coactivation._mode = mode
    top_coactivation.total_tokens_processed = 0
    top_coactivation.candidate_ids = None
    top_coactivation.candidate_vals = None
    top_coactivation.seq_id_to_row = {}
    top_coactivation.sid_to_row_tensor = None
    top_coactivation.dump_timing = {}
    top_coactivation.dump_batches = 0
    top_coactivation.dump_components = 0

    monkeypatch.setattr(
        "pipeline.second_pass.encode_layer_components",
        _fake_encode_layer_components,
    )
    monkeypatch.setattr(
        second_pass_module.top_ctx,
        "get_all_sequence_ids",
        lambda: list(REPLAY_IDS),
    )
    monkeypatch.setattr(
        "pipeline.second_pass.config.latents.top_coactivation.dump_device",
        "cpu",
    )
    monkeypatch.setattr(
        "pipeline.second_pass.config.latents.top_coactivation.dump_profile",
        False,
    )
    monkeypatch.setattr(
        second_pass_module.latent_stats,
        "active_count",
        torch.arange(1, 13, dtype=torch.int64).reshape(2, 6),
    )
    monkeypatch.setattr(worker_module, "top_coactivation", top_coactivation)

    set_runtime(
        PipelineRuntime(
            fast=False,
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
    )
    return top_coactivation


def _run_dump(monkeypatch, mode: str, sequence_ids: list[int] | None):
    top_coactivation = _configure_synthetic_runtime(monkeypatch, mode)
    try:
        result = run_second_pass_dump(sequence_ids)
        return (
            result,
            torch.tensor(
                sequence_ids if sequence_ids is not None else REPLAY_IDS,
                dtype=torch.int64,
            ),
            top_coactivation.candidate_ids.detach().cpu().clone(),
            top_coactivation.candidate_vals.detach().cpu().clone(),
            int(top_coactivation.total_tokens_processed),
        )
    finally:
        clear_runtime()


def _save_worker_dump(monkeypatch, tmp_path: Path, mode: str, worker_count: int, worker_id: int):
    manifest = _manifest(tmp_path, worker_count)
    sequence_ids = manifest.work_assignments.pass2_sequence_ids[str(worker_id)]
    dump_result, expected_sequence_ids, expected_ids, expected_vals, token_count = _run_dump(
        monkeypatch,
        mode,
        sequence_ids,
    )
    artifacts = save_pass2_candidate_dump(manifest, worker_id, dump_result)
    metadata, payload = load_candidate_dump_partial(
        artifacts["candidate_dump"],
        expected_config_hash=manifest.normalized_config_hash,
    )
    assert metadata.worker_id == worker_id
    assert torch.equal(payload["sequence_ids"], expected_sequence_ids)
    assert torch.equal(payload["candidate_ids"], expected_ids)
    assert torch.allclose(payload["candidate_vals"], expected_vals)
    assert int(payload["total_tokens_processed"]) == token_count
    return metadata, payload


@pytest.mark.parametrize("mode", ["raw", "freq_weighted", "pmi"])
def test_one_worker_distributed_dump_matches_single_process(monkeypatch, tmp_path, mode):
    baseline_result, baseline_sequence_ids, baseline_ids, baseline_vals, baseline_tokens = _run_dump(
        monkeypatch,
        mode,
        None,
    )

    metadata, payload = _save_worker_dump(monkeypatch, tmp_path, mode, worker_count=1, worker_id=0)

    assert metadata.sequence_count == baseline_result.sequence_count
    assert torch.equal(payload["sequence_ids"], baseline_sequence_ids)
    assert torch.equal(payload["candidate_ids"], baseline_ids)
    assert torch.allclose(payload["candidate_vals"], baseline_vals)
    assert int(payload["total_tokens_processed"]) == baseline_tokens
    if mode == "pmi":
        assert baseline_tokens == len(REPLAY_IDS) * SEQ_LEN


@pytest.mark.parametrize("mode", ["raw", "freq_weighted", "pmi"])
def test_two_worker_dump_concatenation_matches_single_process(monkeypatch, tmp_path, mode):
    _, baseline_sequence_ids, baseline_ids, baseline_vals, baseline_tokens = _run_dump(
        monkeypatch,
        mode,
        None,
    )

    partials = [
        _save_worker_dump(monkeypatch, tmp_path, mode, worker_count=2, worker_id=worker_id)
        for worker_id in range(2)
    ]
    concatenated_sequence_ids = torch.cat([payload["sequence_ids"] for _, payload in partials])
    concatenated_ids = torch.cat([payload["candidate_ids"] for _, payload in partials], dim=0)
    concatenated_vals = torch.cat([payload["candidate_vals"] for _, payload in partials], dim=0)

    assert torch.equal(concatenated_sequence_ids, baseline_sequence_ids)
    baseline_rows = {
        int(sequence_id): row_idx
        for row_idx, sequence_id in enumerate(baseline_sequence_ids.tolist())
    }
    for row_idx, sequence_id in enumerate(concatenated_sequence_ids.tolist()):
        baseline_row_idx = baseline_rows[int(sequence_id)]
        assert torch.equal(concatenated_ids[row_idx], baseline_ids[baseline_row_idx])
        assert torch.allclose(concatenated_vals[row_idx], baseline_vals[baseline_row_idx])

    worker_token_sum = sum(
        int(payload["total_tokens_processed"])
        for _, payload in partials
    )
    if mode == "pmi":
        assert worker_token_sum == baseline_tokens
        assert worker_token_sum == len(REPLAY_IDS) * SEQ_LEN
    else:
        assert worker_token_sum == 0


def test_pass2_worker_artifacts_use_deterministic_worker_order(monkeypatch, tmp_path):
    _save_worker_dump(monkeypatch, tmp_path, "raw", worker_count=2, worker_id=0)
    _save_worker_dump(monkeypatch, tmp_path, "raw", worker_count=2, worker_id=1)

    layout = build_run_layout(_manifest(tmp_path, worker_count=2))
    assert layout.workers[0].pass2_dir.name == "pass2"
    assert layout.workers[1].pass2_dir.name == "pass2"
    assert layout.workers[0].root.name < layout.workers[1].root.name
