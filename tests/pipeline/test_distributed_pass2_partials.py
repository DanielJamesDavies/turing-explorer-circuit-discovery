from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from pipeline.distributed.manifest import (
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
    ShardRecord,
    WorkAssignments,
)
from pipeline.distributed.pass2_partials import (
    CandidateDumpMetadata,
    candidate_dump_payload,
    check_candidate_dump_memory_guardrail,
    estimate_candidate_dump_bytes,
    expand_candidate_dump_to_contributions,
    load_candidate_preaggregation_partial,
    load_candidate_dump_partial,
    save_candidate_preaggregation_partial,
    save_candidate_dump_partial,
    validate_candidate_preaggregation_partial,
    validate_candidate_dump_partial,
)
from pipeline.distributed.pass2_replay import hash_replay_sequence_ids
from pipeline.second_pass import SecondPassDumpResult


def _manifest(tmp_path: Path) -> DistributedRunManifest:
    run_id = "20260517-002500-abcdef12"
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    replay_ids = [1, 2]
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
        worker_count=1,
        devices=[DeviceAssignment(worker_id=0, physical_id=0, logical_id="cuda:0")],
        shard_table=[
            ShardRecord(
                shard_index=0,
                shard_filename="shard_0.npy",
                sequence_count=2,
                global_start_id=1,
                global_end_id=3,
                shard_size_bytes=1,
                shard_mtime_ns=1,
                index_filename=".shard_indices/shard_0.npy_sft1.idx.npy",
                index_size_bytes=1,
                index_mtime_ns=1,
            ),
        ],
        work_assignments=WorkAssignments(
            pass1_shards={"0": [0]},
            pass1_sequence_totals={"0": 2},
            pass2_sequence_ids={"0": replay_ids},
            pass2_replay_sequence_count=len(replay_ids),
            pass2_replay_sequence_hash=hash_replay_sequence_ids(replay_ids),
        ),
    )


def _metadata(**overrides) -> CandidateDumpMetadata:
    data = {
        "run_id": "20260517-002500-abcdef12",
        "worker_id": 0,
        "sequence_count": 2,
        "sequence_id_min": 1,
        "sequence_id_max": 2,
        "replay_sequence_hash": hash_replay_sequence_ids([1, 2]),
        "config_hash": "abcdef1234567890",
        "physical_id": 0,
        "logical_id": "cuda:0",
        "created_at": "2026-05-17T00:25:00Z",
        "mode": "raw",
        "m": 3,
        "n_candidates_per_component": 2,
        "n_latents_per_latent": 4,
        "num_components": 2,
        "d_sae": 8,
        "token_count": 128,
        "seq_len": 64,
        "batch_count": 1,
        "estimated_dump_bytes": 48,
    }
    data.update(overrides)
    return CandidateDumpMetadata.model_validate(data)


def _payload() -> dict[str, object]:
    return {
        "sequence_ids": torch.tensor([1, 2], dtype=torch.int64),
        "candidate_ids": torch.tensor([[1, 2, 0], [4, 5, 0]], dtype=torch.int32),
        "candidate_vals": torch.tensor([[1.5, 0.5, 0.0], [2.5, 0.25, 0.0]], dtype=torch.float32),
        "total_tokens_processed": 128,
    }


def test_candidate_dump_partial_round_trip(tmp_path):
    path = tmp_path / "candidate_dump.partial.pt"
    metadata = _metadata()
    payload = _payload()

    save_candidate_dump_partial(path, metadata, payload)
    loaded_metadata, loaded_payload = load_candidate_dump_partial(
        path,
        expected_config_hash="abcdef1234567890",
    )

    assert loaded_metadata == metadata
    assert torch.equal(loaded_payload["sequence_ids"], payload["sequence_ids"])
    assert torch.equal(loaded_payload["candidate_ids"], payload["candidate_ids"])
    assert torch.allclose(loaded_payload["candidate_vals"], payload["candidate_vals"])


def test_estimate_candidate_dump_bytes_counts_ids_and_values():
    estimate = estimate_candidate_dump_bytes(10, 3, guardrail_bytes=1_000)

    assert estimate.sequence_count == 10
    assert estimate.m == 3
    assert estimate.candidate_ids_bytes == 120
    assert estimate.candidate_vals_bytes == 120
    assert estimate.total_bytes == 240
    assert estimate.exceeds_guardrail is False


def test_candidate_dump_memory_guardrail_can_fail_or_warn(capsys):
    with pytest.raises(MemoryError, match="exceeds guardrail"):
        check_candidate_dump_memory_guardrail(
            10,
            3,
            guardrail_bytes=100,
            fail_on_guardrail=True,
        )

    estimate = check_candidate_dump_memory_guardrail(
        10,
        3,
        guardrail_bytes=100,
        fail_on_guardrail=False,
    )

    assert estimate.exceeds_guardrail is True
    assert "WARNING" in capsys.readouterr().out


def test_candidate_dump_payload_uses_top_coactivation_buffers():
    class FakeTopCoactivation:
        candidate_ids = torch.tensor([[1, 2]], dtype=torch.int64)
        candidate_vals = torch.tensor([[0.5, 1.0]], dtype=torch.float64)
        total_tokens_processed = 12

    payload = candidate_dump_payload(FakeTopCoactivation(), [10])

    assert payload["sequence_ids"].dtype == torch.int64
    assert payload["candidate_ids"].dtype == torch.int32
    assert payload["candidate_vals"].dtype == torch.float32
    assert payload["sequence_ids"].tolist() == [10]
    assert payload["total_tokens_processed"] == 12


def test_candidate_dump_validation_rejects_bad_shape():
    data = {"metadata": _metadata().model_dump(mode="json"), "payload": _payload()}
    data["payload"]["candidate_vals"] = torch.zeros((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="unexpected shape"):
        validate_candidate_dump_partial(data)


def test_candidate_dump_validation_rejects_missing_sequence_ids():
    data = {"metadata": _metadata().model_dump(mode="json"), "payload": _payload()}
    del data["payload"]["sequence_ids"]

    with pytest.raises(ValueError, match="missing tensor: sequence_ids"):
        validate_candidate_dump_partial(data)


def test_candidate_dump_validation_rejects_invalid_candidate_ids():
    data = {"metadata": _metadata().model_dump(mode="json"), "payload": _payload()}
    data["payload"]["candidate_ids"] = torch.tensor(
        [[1, 16, 0], [4, 5, 0]],
        dtype=torch.int32,
    )

    with pytest.raises(ValueError, match="candidate_ids out of range"):
        validate_candidate_dump_partial(data)


def test_candidate_dump_validation_rejects_non_finite_values():
    data = {"metadata": _metadata().model_dump(mode="json"), "payload": _payload()}
    data["payload"]["candidate_vals"] = torch.tensor(
        [[1.0, float("nan"), 0.0], [2.0, 0.25, 0.0]],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="candidate_vals must be finite"):
        validate_candidate_dump_partial(data)


def test_candidate_dump_validation_rejects_token_count_mismatch():
    data = {"metadata": _metadata(token_count=64).model_dump(mode="json"), "payload": _payload()}

    with pytest.raises(ValueError, match="token-count metadata mismatch"):
        validate_candidate_dump_partial(data)


def test_build_metadata_from_manifest_and_dump_result(tmp_path):
    from pipeline.distributed.pass2_partials import build_candidate_dump_metadata

    class FakeTopCoactivation:
        mode = "pmi"
        M = 3
        n_candidates_per_component = 2
        n_latents_per_latent = 4
        num_components = 2
        d_sae = 8
        total_tokens_processed = 128

    metadata = build_candidate_dump_metadata(
        _manifest(tmp_path),
        0,
        FakeTopCoactivation(),
        SecondPassDumpResult(
            sequence_count=2,
            batch_count=1,
            seq_len=64,
            elapsed_s=0.1,
        ),
    )

    assert metadata.mode == "pmi"
    assert metadata.sequence_count == 2
    assert metadata.sequence_id_min == 1
    assert metadata.sequence_id_max == 2
    assert metadata.token_count == 128
    assert metadata.seq_len == 64
    assert metadata.estimated_dump_bytes == 48


def test_expand_candidate_dump_to_contributions_matches_reducer_multiset():
    metadata = _metadata(
        m=3,
        num_components=2,
        d_sae=8,
    )
    payload = {
        "sequence_ids": torch.tensor([1, 2], dtype=torch.int64),
        "candidate_ids": torch.tensor(
            [
                [0, 1, 2],
                [1, 0, 3],
            ],
            dtype=torch.int32,
        ),
        "candidate_vals": torch.tensor(
            [
                [9.0, 1.0, 2.0],
                [3.0, 4.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "total_tokens_processed": 128,
    }
    seq_offsets = torch.tensor([0, 2, 3], dtype=torch.int64)
    seq_targets = torch.tensor([0, 5, 0], dtype=torch.int64)

    preagg_metadata, preagg_payload = expand_candidate_dump_to_contributions(
        metadata,
        payload,
        seq_offsets,
        seq_targets,
    )

    expected = _reference_contributions(payload, seq_offsets, seq_targets)
    actual = list(
        zip(
            preagg_payload["target_ids"].tolist(),
            preagg_payload["candidate_ids"].tolist(),
            preagg_payload["values"].tolist(),
            preagg_payload["sequence_ids"].tolist(),
        )
    )
    assert actual == expected
    assert preagg_metadata.artifact_name == "candidate_preaggregation"
    assert preagg_metadata.contribution_count == len(expected)
    assert preagg_metadata.source_candidate_dump_schema_version == metadata.partial_schema_version


def test_expand_candidate_dump_preserves_duplicate_target_entries():
    metadata = _metadata(
        sequence_count=1,
        sequence_id_min=1,
        sequence_id_max=1,
        replay_sequence_hash=hash_replay_sequence_ids([1]),
        m=2,
        num_components=1,
        d_sae=4,
    )
    payload = {
        "sequence_ids": torch.tensor([1], dtype=torch.int64),
        "candidate_ids": torch.tensor([[1, 2]], dtype=torch.int32),
        "candidate_vals": torch.tensor([[0.5, 1.5]], dtype=torch.float32),
        "total_tokens_processed": 128,
    }
    seq_offsets = torch.tensor([0, 2], dtype=torch.int64)
    seq_targets = torch.tensor([0, 0], dtype=torch.int64)

    _preagg_metadata, preagg_payload = expand_candidate_dump_to_contributions(
        metadata,
        payload,
        seq_offsets,
        seq_targets,
    )

    assert preagg_payload["target_ids"].tolist() == [0, 0, 0, 0]
    assert preagg_payload["candidate_ids"].tolist() == [1, 2, 1, 2]
    assert preagg_payload["sequence_ids"].tolist() == [1, 1, 1, 1]


def test_candidate_preaggregation_round_trip(tmp_path):
    metadata = _metadata()
    preagg_metadata, preagg_payload = expand_candidate_dump_to_contributions(
        metadata,
        _payload(),
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([0, 4], dtype=torch.int64),
    )
    path = tmp_path / "candidate_preaggregation.partial.pt"

    save_candidate_preaggregation_partial(path, preagg_metadata, preagg_payload)
    loaded_metadata, loaded_payload = load_candidate_preaggregation_partial(
        path,
        expected_config_hash="abcdef1234567890",
    )

    assert loaded_metadata == preagg_metadata
    assert torch.equal(loaded_payload["target_ids"], preagg_payload["target_ids"])
    assert torch.equal(loaded_payload["candidate_ids"], preagg_payload["candidate_ids"])
    assert torch.allclose(loaded_payload["values"], preagg_payload["values"])
    assert torch.equal(loaded_payload["sequence_ids"], preagg_payload["sequence_ids"])


def test_candidate_preaggregation_validation_rejects_self_candidates():
    metadata = {
        "partial_schema_version": 1,
        "artifact_name": "candidate_preaggregation",
        "run_id": "20260517-002500-abcdef12",
        "worker_id": 0,
        "source_candidate_dump_schema_version": 1,
        "sequence_count": 1,
        "contribution_count": 1,
        "config_hash": "abcdef1234567890",
        "mode": "raw",
        "num_components": 1,
        "d_sae": 4,
        "m": 2,
        "created_at": "2026-05-17T00:25:00Z",
    }
    payload = {
        "target_ids": torch.tensor([1], dtype=torch.int64),
        "candidate_ids": torch.tensor([1], dtype=torch.int32),
        "values": torch.tensor([0.5], dtype=torch.float32),
        "sequence_ids": torch.tensor([1], dtype=torch.int64),
    }

    with pytest.raises(ValueError, match="self-candidate"):
        validate_candidate_preaggregation_partial({"metadata": metadata, "payload": payload})


def _reference_contributions(payload, seq_offsets, seq_targets):
    records = []
    for row_idx, sequence_id in enumerate(payload["sequence_ids"].tolist()):
        start = int(seq_offsets[sequence_id - 1])
        end = int(seq_offsets[sequence_id])
        for target_id in seq_targets[start:end].tolist():
            for candidate_id, value in zip(
                payload["candidate_ids"][row_idx].tolist(),
                payload["candidate_vals"][row_idx].tolist(),
            ):
                if candidate_id == target_id or value <= 0.0:
                    continue
                records.append((int(target_id), int(candidate_id), float(value), int(sequence_id)))
    return records
