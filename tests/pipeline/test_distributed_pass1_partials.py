from types import SimpleNamespace

from pathlib import Path

import pytest
import torch

from pipeline.distributed.pass1_partials import (
    MID_CTX_PRIORITY_HASH_VERSION,
    Pass1PartialMetadata,
    load_pass1_partial,
    mid_ctx_candidates_payload,
    save_pass1_partial,
    validate_pass1_partial,
)


def _metadata(artifact_name: str) -> Pass1PartialMetadata:
    return Pass1PartialMetadata(
        artifact_name=artifact_name,
        run_id="20260517-002500-abcdef12",
        worker_id=0,
        shard_ids=[0],
        sequence_id_min=1,
        sequence_id_max=2,
        sequence_count=2,
        config_hash="abcdef1234567890",
        physical_id=0,
        logical_id="cuda:0",
        created_at="2026-05-17T00:25:00Z",
        component_count=2,
        d_sae=3,
    )


def _payloads():
    shape = (2, 3)
    ctx_idx = torch.tensor(
        [
            [[1, 2], [0, 0], [0, 0]],
            [[0, 0], [1, 0], [2, 0]],
        ],
        dtype=torch.int32,
    )
    ctx_val = torch.tensor(
        [
            [[0.5, 0.25], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.7, 0.0], [0.3, 0.0]],
        ],
        dtype=torch.float32,
    )
    return {
        "latent_stats": {
            "active_count": torch.zeros(shape, dtype=torch.int64),
            "mean": torch.zeros(shape, dtype=torch.float32),
            "mean_abs": torch.zeros(shape, dtype=torch.float32),
            "m2": torch.zeros(shape, dtype=torch.float32),
            "m2_abs": torch.zeros(shape, dtype=torch.float32),
            "seq_count": torch.zeros(shape, dtype=torch.int64),
            "mean_seq": torch.zeros(shape, dtype=torch.float32),
            "m2_seq": torch.zeros(shape, dtype=torch.float32),
            "component_steps": {0: 1, 1: 1},
        },
        "top_ctx": {
            "ctx_seq_idx": ctx_idx,
            "ctx_seq_val": ctx_val,
            "ctx_type": "top",
        },
        "mid_ctx_candidates": {
            "component_ids": torch.tensor([0, 1], dtype=torch.int16),
            "latent_ids": torch.tensor([0, 2], dtype=torch.int32),
            "sequence_ids": torch.tensor([1, 2], dtype=torch.int32),
            "activation_values": torch.tensor([0.5, 0.3], dtype=torch.float32),
            "priorities": torch.tensor([10, 20], dtype=torch.int64),
            "candidate_pool_settings": {"mode": "worker_local_mid_ctx_checkpoint"},
            "truncation_counters": torch.zeros(shape, dtype=torch.int64),
            "ctx_seq_idx": ctx_idx,
            "ctx_seq_val": ctx_val,
        },
        "seq_repr": {
            "repr_buf": torch.zeros((3, 4), dtype=torch.float16),
            "repr_mode": "mean_pool",
            "repr_dim": 4,
            "n_seqs": 2,
            "n_stored": 2,
            "is_capped": False,
        },
        "logit_ctx": {
            "latent_counts": torch.zeros(shape, dtype=torch.int64),
            "top_tokens": torch.zeros((2, 3, 2), dtype=torch.int32),
            "top_probs": torch.zeros((2, 3, 2), dtype=torch.float32),
        },
    }


@pytest.mark.parametrize(
    "artifact_name",
    ["latent_stats", "top_ctx", "mid_ctx_candidates", "seq_repr", "logit_ctx"],
)
def test_pass1_partial_round_trip_for_each_artifact(tmp_path, artifact_name):
    path = tmp_path / f"{artifact_name}.partial.pt"

    save_pass1_partial(path, _metadata(artifact_name), _payloads()[artifact_name])
    metadata, payload = load_pass1_partial(path, expected_artifact_name=artifact_name)

    assert metadata.artifact_name == artifact_name
    assert metadata.config_hash == "abcdef1234567890"
    assert isinstance(payload, dict)
    assert path.exists()
    assert not Path(f"{path}.tmp").exists()


def test_pass1_partial_rejects_wrong_artifact_name():
    data = {
        "metadata": _metadata("top_ctx").model_dump(mode="json"),
        "payload": _payloads()["top_ctx"],
    }

    with pytest.raises(ValueError, match="artifact name mismatch"):
        validate_pass1_partial(data, expected_artifact_name="latent_stats")


def test_pass1_partial_rejects_non_finite_tensor():
    payload = _payloads()["latent_stats"]
    payload["mean"][0, 0] = float("nan")
    data = {
        "metadata": _metadata("latent_stats").model_dump(mode="json"),
        "payload": payload,
    }

    with pytest.raises(ValueError, match="mean contains non-finite"):
        validate_pass1_partial(data)


def test_pass1_partial_rejects_stale_config_hash():
    data = {
        "metadata": _metadata("logit_ctx").model_dump(mode="json"),
        "payload": _payloads()["logit_ctx"],
    }

    with pytest.raises(ValueError, match="config hash mismatch"):
        validate_pass1_partial(data, expected_config_hash="different")


def test_mid_ctx_candidate_priorities_are_seeded_and_reproducible():
    store = SimpleNamespace(
        ctx_seq_idx=torch.tensor([[[1, 2]]], dtype=torch.int32),
        ctx_seq_val=torch.tensor([[[0.5, 0.7]]], dtype=torch.float32),
        mid_mode="test",
        _band_low=0.5,
        _band_high=1.5,
        num_ctx_sequences=2,
        num_components=1,
        d_sae=1,
        reservoir_fill=torch.ones((1, 1), dtype=torch.int32),
        reservoir_n=torch.ones((1, 1), dtype=torch.int64),
    )

    first = mid_ctx_candidates_payload(
        store,
        sampling_seed=123,
        dataset_fingerprint="dataset-a",
    )
    second = mid_ctx_candidates_payload(
        store,
        sampling_seed=123,
        dataset_fingerprint="dataset-a",
    )
    changed = mid_ctx_candidates_payload(
        store,
        sampling_seed=124,
        dataset_fingerprint="dataset-a",
    )
    changed_dataset = mid_ctx_candidates_payload(
        store,
        sampling_seed=123,
        dataset_fingerprint="dataset-b",
    )

    assert torch.equal(first["priorities"], second["priorities"])
    assert not torch.equal(first["priorities"], changed["priorities"])
    assert not torch.equal(first["priorities"], changed_dataset["priorities"])
    assert first["candidate_pool_settings"]["sampling_seed"] == 123
    assert first["candidate_pool_settings"]["dataset_fingerprint"] == "dataset-a"
    assert first["candidate_pool_settings"]["priority_hash_version"] == MID_CTX_PRIORITY_HASH_VERSION
    assert first["priorities"].dtype == torch.int64


def test_mid_ctx_candidate_priorities_change_with_band_settings_not_run_id():
    store = SimpleNamespace(
        ctx_seq_idx=torch.tensor([[[1, 2]]], dtype=torch.int32),
        ctx_seq_val=torch.tensor([[[0.5, 0.7]]], dtype=torch.float32),
        mid_mode="test",
        _band_low=0.5,
        _band_high=1.5,
        num_ctx_sequences=2,
        num_components=1,
        d_sae=1,
        reservoir_fill=torch.ones((1, 1), dtype=torch.int32),
        reservoir_n=torch.ones((1, 1), dtype=torch.int64),
    )
    widened_store = SimpleNamespace(**{**store.__dict__, "_band_high": 2.5})

    first = mid_ctx_candidates_payload(
        store,
        sampling_seed=123,
        dataset_fingerprint="dataset-a",
    )
    rerun_with_new_run_id_not_supplied = mid_ctx_candidates_payload(
        store,
        sampling_seed=123,
        dataset_fingerprint="dataset-a",
    )
    changed_band = mid_ctx_candidates_payload(
        widened_store,
        sampling_seed=123,
        dataset_fingerprint="dataset-a",
    )

    assert torch.equal(first["priorities"], rerun_with_new_run_id_not_supplied["priorities"])
    assert not torch.equal(first["priorities"], changed_band["priorities"])


def test_mid_ctx_candidate_priorities_are_uniform_across_seeded_trials():
    sequence_ids = torch.arange(1, 1001, dtype=torch.int32)
    store = SimpleNamespace(
        ctx_seq_idx=sequence_ids.view(1, 1, -1),
        ctx_seq_val=torch.ones((1, 1, sequence_ids.numel()), dtype=torch.float32),
        mid_mode="test",
        _band_low=0.5,
        _band_high=1.5,
        num_ctx_sequences=sequence_ids.numel(),
        num_components=1,
        d_sae=1,
        reservoir_fill=torch.ones((1, 1), dtype=torch.int32),
        reservoir_n=torch.ones((1, 1), dtype=torch.int64),
    )
    means = []
    for seed in range(10):
        payload = mid_ctx_candidates_payload(
            store,
            sampling_seed=seed,
            dataset_fingerprint="dataset-a",
        )
        priorities = payload["priorities"]
        normalized = priorities.to(torch.float64) / float(1 << 63)
        means.append(float(normalized.mean()))
        assert 0.45 < float((normalized < 0.5).float().mean()) < 0.55

    assert 0.45 < sum(means) / len(means) < 0.55
