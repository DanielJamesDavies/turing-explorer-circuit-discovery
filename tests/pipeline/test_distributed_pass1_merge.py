import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pipeline.distributed.manifest import (
    DeviceAssignment,
    DistributedRunManifest,
    WorkAssignments,
    load_manifest,
)
from pipeline.distributed.pass1_merge import (
    load_and_merge_latent_stats_partials,
    load_and_merge_logit_ctx_partials,
    load_and_merge_mid_ctx_candidate_partials,
    load_and_merge_seq_repr_partials,
    load_and_merge_top_ctx_partials,
    merge_latent_stats_partials,
    merge_logit_ctx_partials,
    merge_mid_ctx_candidate_partials,
    merge_pass1_worker_outputs,
    merge_seq_latent_index_shards,
    merge_seq_repr_partials,
    merge_top_ctx_partials,
)
from pipeline.distributed.pass1_partials import latent_stats_payload
from store.latent_stats import LatentStats
from pipeline.distributed.pass1_partials import (
    Pass1PartialMetadata,
    save_pass1_partial,
)
from pipeline.distributed.seq_repr_mapping import (
    build_seq_repr_cap_mapping,
    derive_seq_repr_cap_seed,
)
from pipeline.negative_context import run_negative_context_stage
from store.neg_context import NegCtxStats


def _metadata(worker_id: int) -> Pass1PartialMetadata:
    return Pass1PartialMetadata(
        artifact_name="latent_stats",
        run_id="20260517-002500-abcdef12",
        worker_id=worker_id,
        shard_ids=[worker_id],
        sequence_id_min=1 + worker_id * 10,
        sequence_id_max=2 + worker_id * 10,
        sequence_count=2,
        config_hash="abcdef1234567890",
        physical_id=worker_id,
        logical_id="cuda:0",
        created_at="2026-05-17T00:25:00Z",
        component_count=2,
        d_sae=3,
    )


def _top_ctx_metadata(worker_id: int) -> Pass1PartialMetadata:
    return Pass1PartialMetadata(
        artifact_name="top_ctx",
        run_id="20260517-002500-abcdef12",
        worker_id=worker_id,
        shard_ids=[worker_id],
        sequence_id_min=1 + worker_id * 2,
        sequence_id_max=2 + worker_id * 2,
        sequence_count=2,
        config_hash="abcdef1234567890",
        physical_id=worker_id,
        logical_id="cuda:0",
        created_at="2026-05-17T00:25:00Z",
        component_count=1,
        d_sae=2,
    )


def _mid_ctx_metadata(worker_id: int) -> Pass1PartialMetadata:
    return Pass1PartialMetadata(
        artifact_name="mid_ctx_candidates",
        run_id="20260517-002500-abcdef12",
        worker_id=worker_id,
        shard_ids=[worker_id],
        sequence_id_min=1 + worker_id * 2,
        sequence_id_max=2 + worker_id * 2,
        sequence_count=2,
        config_hash="abcdef1234567890",
        physical_id=worker_id,
        logical_id="cuda:0",
        created_at="2026-05-17T00:25:00Z",
        component_count=1,
        d_sae=2,
    )


def _seq_repr_metadata(worker_id: int) -> Pass1PartialMetadata:
    return Pass1PartialMetadata(
        artifact_name="seq_repr",
        run_id="20260517-002500-abcdef12",
        worker_id=worker_id,
        shard_ids=[worker_id],
        sequence_id_min=1 + worker_id * 2,
        sequence_id_max=2 + worker_id * 2,
        sequence_count=2,
        config_hash="abcdef1234567890",
        physical_id=worker_id,
        logical_id="cuda:0",
        created_at="2026-05-17T00:25:00Z",
        component_count=1,
        d_sae=1,
    )


def _logit_ctx_metadata(worker_id: int) -> Pass1PartialMetadata:
    return Pass1PartialMetadata(
        artifact_name="logit_ctx",
        run_id="20260517-002500-abcdef12",
        worker_id=worker_id,
        shard_ids=[worker_id],
        sequence_id_min=1 + worker_id * 2,
        sequence_id_max=2 + worker_id * 2,
        sequence_count=2,
        config_hash="abcdef1234567890",
        physical_id=worker_id,
        logical_id="cuda:0",
        created_at="2026-05-17T00:25:00Z",
        component_count=1,
        d_sae=2,
    )


def _payload(
    token_values: dict[tuple[int, int], list[float]],
    seq_values: dict[tuple[int, int], list[float]],
    *,
    component_steps: dict[int, int] | None = None,
) -> dict[str, object]:
    shape = (2, 3)
    active_count = torch.zeros(shape, dtype=torch.int64)
    mean = torch.zeros(shape, dtype=torch.float32)
    mean_abs = torch.zeros(shape, dtype=torch.float32)
    m2 = torch.zeros(shape, dtype=torch.float32)
    m2_abs = torch.zeros(shape, dtype=torch.float32)
    seq_count = torch.zeros(shape, dtype=torch.int64)
    mean_seq = torch.zeros(shape, dtype=torch.float32)
    m2_seq = torch.zeros(shape, dtype=torch.float32)

    for (component, latent), values in token_values.items():
        stats = _stats(values)
        active_count[component, latent] = stats["count"]
        mean[component, latent] = stats["mean"]
        m2[component, latent] = stats["m2"]
        abs_stats = _stats([abs(value) for value in values])
        mean_abs[component, latent] = abs_stats["mean"]
        m2_abs[component, latent] = abs_stats["m2"]

    for (component, latent), values in seq_values.items():
        stats = _stats(values)
        seq_count[component, latent] = stats["count"]
        mean_seq[component, latent] = stats["mean"]
        m2_seq[component, latent] = stats["m2"]

    return {
        "active_count": active_count,
        "mean": mean,
        "mean_abs": mean_abs,
        "m2": m2,
        "m2_abs": m2_abs,
        "seq_count": seq_count,
        "mean_seq": mean_seq,
        "m2_seq": m2_seq,
        "component_steps": component_steps or {0: 1, 1: 1},
    }


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "m2": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = tensor.mean()
    m2 = (tensor - mean).square().sum()
    return {"count": len(values), "mean": float(mean), "m2": float(m2)}


def test_merge_latent_stats_partials_matches_single_global_stream():
    worker_0_tokens = {(0, 0): [1.0, 3.0], (0, 1): [-2.0], (1, 2): [0.5, 0.75]}
    worker_1_tokens = {(0, 0): [5.0], (0, 1): [4.0, 6.0], (1, 2): [1.25]}
    worker_0_seq = {(0, 0): [0.5, 1.5], (1, 1): [2.0]}
    worker_1_seq = {(0, 0): [2.5], (1, 1): [3.0, 4.0]}

    worker_0 = _payload(worker_0_tokens, worker_0_seq, component_steps={0: 2, 1: 1})
    worker_1 = _payload(worker_1_tokens, worker_1_seq, component_steps={0: 1, 1: 2})
    expected = _payload(
        _combine_value_maps(worker_0_tokens, worker_1_tokens),
        _combine_value_maps(worker_0_seq, worker_1_seq),
        component_steps={0: 3, 1: 3},
    )

    merged = merge_latent_stats_partials(
        [(_metadata(0), worker_0), (_metadata(1), worker_1)]
    )

    assert torch.equal(merged["active_count"], expected["active_count"])
    assert torch.equal(merged["seq_count"], expected["seq_count"])
    for key in ["mean", "mean_abs", "m2", "m2_abs", "mean_seq", "m2_seq"]:
        assert torch.allclose(merged[key], expected[key], atol=1e-6)
    assert merged["component_steps"] == {0: 3, 1: 3}


def test_merge_latent_stats_partials_is_order_invariant_for_welford_state():
    partials = [
        (
            _metadata(0).model_copy(update={"worker_id": 0}),
            _payload({(0, 0): [1.0, 4.0]}, {(0, 0): [1.5]}),
        ),
        (
            _metadata(1).model_copy(update={"worker_id": 1}),
            _payload({(0, 0): [2.0, 8.0]}, {(0, 0): [2.5, 3.5]}),
        ),
        (
            _metadata(2).model_copy(update={"worker_id": 2}),
            _payload({(0, 0): [16.0]}, {(0, 0): [4.5]}),
        ),
    ]

    merged_forward = merge_latent_stats_partials(partials)
    merged_reversed = merge_latent_stats_partials(list(reversed(partials)))

    assert torch.equal(merged_forward["active_count"], merged_reversed["active_count"])
    assert torch.equal(merged_forward["seq_count"], merged_reversed["seq_count"])
    for key in ["mean", "mean_abs", "m2", "m2_abs", "mean_seq", "m2_seq"]:
        assert torch.allclose(merged_forward[key], merged_reversed[key], atol=1e-6)


def test_merge_latent_stats_partials_matches_latent_stats_update_component():
    batches = [
        (
            torch.tensor([[[1.0, 3.0], [0.0, 2.0]]]),
            torch.tensor([[[0, 1], [0, 2]]], dtype=torch.int32),
        ),
        (
            torch.tensor([[[5.0, 4.0], [6.0, 0.0]]]),
            torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int32),
        ),
        (
            torch.tensor([[[0.5, 0.75], [1.25, 0.0]]]),
            torch.tensor([[[2, 2], [2, 1]]], dtype=torch.int32),
        ),
    ]
    single = _stats_store_for_batches(batches)
    worker_0 = _stats_store_for_batches(batches[:2])
    worker_1 = _stats_store_for_batches(batches[2:])

    merged = merge_latent_stats_partials(
        [
            (_metadata(0), latent_stats_payload(worker_0)),
            (_metadata(1), latent_stats_payload(worker_1)),
        ]
    )
    expected = latent_stats_payload(single)

    assert torch.equal(merged["active_count"], expected["active_count"])
    assert torch.equal(merged["seq_count"], expected["seq_count"])
    for key in ["mean", "mean_abs", "m2", "m2_abs", "mean_seq", "m2_seq"]:
        assert torch.allclose(merged[key], expected[key], atol=1e-6)


def test_load_and_merge_latent_stats_partials_round_trip(tmp_path):
    worker_0 = (_metadata(0), _payload({(0, 0): [1.0, 2.0]}, {(0, 0): [0.5]}))
    worker_1 = (_metadata(1), _payload({(0, 0): [3.0]}, {(0, 0): [1.5]}))
    paths = []
    for metadata, payload in [worker_0, worker_1]:
        path = tmp_path / f"worker_{metadata.worker_id}.pt"
        save_pass1_partial(path, metadata, payload)
        paths.append(path)

    merged = load_and_merge_latent_stats_partials(
        paths,
        expected_config_hash="abcdef1234567890",
    )

    assert merged["active_count"][0, 0].item() == 3
    assert torch.isclose(merged["mean"][0, 0], torch.tensor(2.0))


def test_merge_top_ctx_partials_selects_global_topk_across_workers():
    worker_0 = {
        "ctx_seq_idx": torch.tensor([[[1, 2], [2, 1]]], dtype=torch.int32),
        "ctx_seq_val": torch.tensor([[[0.9, 0.4], [0.5, 0.1]]], dtype=torch.float32),
        "ctx_type": "top",
    }
    worker_1 = {
        "ctx_seq_idx": torch.tensor([[[3, 4], [3, 4]]], dtype=torch.int32),
        "ctx_seq_val": torch.tensor([[[0.8, 0.7], [0.5, 0.2]]], dtype=torch.float32),
        "ctx_type": "top",
    }

    merged = merge_top_ctx_partials(
        [(_top_ctx_metadata(0), worker_0), (_top_ctx_metadata(1), worker_1)]
    )

    assert merged["ctx_type"] == "top"
    assert merged["ctx_seq_idx"].tolist() == [[[1, 3], [2, 3]]]
    assert torch.allclose(
        merged["ctx_seq_val"],
        torch.tensor([[[0.9, 0.8], [0.5, 0.5]]]),
    )


def test_merge_top_ctx_partials_zeroes_invalid_sentinel_rows():
    worker_0 = {
        "ctx_seq_idx": torch.tensor([[[0, 1], [0, 0]]], dtype=torch.int32),
        "ctx_seq_val": torch.tensor([[[0.0, 0.3], [0.0, 0.0]]], dtype=torch.float32),
        "ctx_type": "top",
    }
    worker_1 = {
        "ctx_seq_idx": torch.tensor([[[3, 0], [0, 0]]], dtype=torch.int32),
        "ctx_seq_val": torch.tensor([[[0.4, 99.0], [0.0, 0.0]]], dtype=torch.float32),
        "ctx_type": "top",
    }

    merged = merge_top_ctx_partials(
        [(_top_ctx_metadata(0), worker_0), (_top_ctx_metadata(1), worker_1)]
    )

    assert merged["ctx_seq_idx"].tolist() == [[[3, 1], [0, 0]]]
    assert torch.allclose(
        merged["ctx_seq_val"],
        torch.tensor([[[0.4, 0.3], [0.0, 0.0]]]),
    )


def test_load_and_merge_top_ctx_partials_round_trip(tmp_path):
    worker_0 = {
        "ctx_seq_idx": torch.tensor([[[1, 2], [0, 0]]], dtype=torch.int32),
        "ctx_seq_val": torch.tensor([[[0.2, 0.1], [0.0, 0.0]]], dtype=torch.float32),
        "ctx_type": "top",
    }
    worker_1 = {
        "ctx_seq_idx": torch.tensor([[[3, 4], [3, 0]]], dtype=torch.int32),
        "ctx_seq_val": torch.tensor([[[0.5, 0.4], [0.9, 0.0]]], dtype=torch.float32),
        "ctx_type": "top",
    }
    paths = []
    for metadata, payload in [(_top_ctx_metadata(0), worker_0), (_top_ctx_metadata(1), worker_1)]:
        path = tmp_path / f"top_ctx_worker_{metadata.worker_id}.pt"
        save_pass1_partial(path, metadata, payload)
        paths.append(path)

    merged = load_and_merge_top_ctx_partials(
        paths,
        expected_config_hash="abcdef1234567890",
    )

    assert merged["ctx_seq_idx"].tolist() == [[[3, 4], [3, 0]]]


def test_merge_top_ctx_partials_rejects_sequence_outside_worker_range():
    worker_0 = {
        "ctx_seq_idx": torch.tensor([[[3, 1], [0, 0]]], dtype=torch.int32),
        "ctx_seq_val": torch.tensor([[[0.8, 0.3], [0.0, 0.0]]], dtype=torch.float32),
        "ctx_type": "top",
    }

    with pytest.raises(ValueError, match="above worker range"):
        merge_top_ctx_partials([(_top_ctx_metadata(0), worker_0)])


def test_merge_mid_ctx_candidates_filters_by_global_stats_and_priority():
    worker_0 = _mid_ctx_payload(
        component_ids=[0, 0, 0],
        latent_ids=[0, 0, 1],
        sequence_ids=[1, 2, 1],
        activation_values=[2.0, 1.0, 12.0],
        priorities=[0.3, 0.1, 0.5],
    )
    worker_1 = _mid_ctx_payload(
        component_ids=[0, 0],
        latent_ids=[0, 0],
        sequence_ids=[3, 4],
        activation_values=[2.2, 2.1],
        priorities=[0.1, 0.2],
    )

    merged = merge_mid_ctx_candidate_partials(
        [(_mid_ctx_metadata(0), worker_0), (_mid_ctx_metadata(1), worker_1)],
        latent_stats_payload=_mid_ctx_latent_stats_payload(),
        num_ctx_sequences=2,
        band_low_sigma=0.5,
        band_high_sigma=1.5,
    )

    assert merged["ctx_type"] == "mid"
    assert merged["mode"] == "distributed_priority_reservoir"
    assert merged["ctx_seq_idx"].tolist() == [[[3, 4], [1, 0]]]
    assert torch.allclose(
        merged["ctx_seq_val"],
        torch.tensor([[[2.2, 2.1], [12.0, 0.0]]]),
    )
    assert merged["reservoir_n"].tolist() == [[3, 1]]
    assert merged["reservoir_fill"].tolist() == [[2, 1]]
    assert merged["merge_report"]["candidate_count"].tolist() == [[4, 1]]
    assert merged["merge_report"]["valid_count"].tolist() == [[3, 1]]


def test_load_and_merge_mid_ctx_candidates_round_trip(tmp_path):
    worker_0 = _mid_ctx_payload(
        component_ids=[0],
        latent_ids=[0],
        sequence_ids=[1],
        activation_values=[2.0],
        priorities=[0.3],
    )
    worker_1 = _mid_ctx_payload(
        component_ids=[0],
        latent_ids=[0],
        sequence_ids=[3],
        activation_values=[2.2],
        priorities=[0.1],
    )
    paths = []
    for metadata, payload in [(_mid_ctx_metadata(0), worker_0), (_mid_ctx_metadata(1), worker_1)]:
        path = tmp_path / f"mid_ctx_worker_{metadata.worker_id}.pt"
        save_pass1_partial(path, metadata, payload)
        paths.append(path)

    merged = load_and_merge_mid_ctx_candidate_partials(
        paths,
        latent_stats_payload=_mid_ctx_latent_stats_payload(),
        expected_config_hash="abcdef1234567890",
        num_ctx_sequences=2,
    )

    assert merged["ctx_seq_idx"].tolist() == [[[3, 1], [0, 0]]]


def test_merge_mid_ctx_candidates_fail_policy_rejects_truncation():
    payload = _mid_ctx_payload(
        component_ids=[0],
        latent_ids=[0],
        sequence_ids=[1],
        activation_values=[2.0],
        priorities=[0.3],
    )
    payload["truncation_counters"][0, 0] = 1

    with pytest.raises(ValueError, match="truncation detected"):
        merge_mid_ctx_candidate_partials(
            [(_mid_ctx_metadata(0), payload)],
            latent_stats_payload=_mid_ctx_latent_stats_payload(),
            on_truncation="fail",
        )


def test_merge_mid_ctx_candidates_replay_fallback_reports_truncation():
    payload = _mid_ctx_payload(
        component_ids=[0],
        latent_ids=[0],
        sequence_ids=[1],
        activation_values=[2.0],
        priorities=[0.3],
    )
    payload["truncation_counters"][0, 0] = 1

    merged = merge_mid_ctx_candidate_partials(
        [(_mid_ctx_metadata(0), payload)],
        latent_stats_payload=_mid_ctx_latent_stats_payload(),
        on_truncation="replay_fallback",
    )

    assert merged["merge_report"]["requires_replay_fallback"] is True
    assert merged["merge_report"]["replay_fallback_executed"] is False
    assert merged["merge_report"]["bounded_approximation"] is False
    assert merged["merge_report"]["candidate_pool_cleanup_eligible"] is False


def test_merge_mid_ctx_candidates_executes_stats_aware_replay_fallback():
    payload = _mid_ctx_payload(
        component_ids=[0],
        latent_ids=[0],
        sequence_ids=[1],
        activation_values=[2.0],
        priorities=[0.3],
    )
    payload["truncation_counters"][0, 0] = 1

    def replay_fallback_fn(**kwargs):
        assert kwargs["num_ctx_sequences"] == 2
        return {
            "ctx_seq_idx": torch.tensor([[[2, 1], [0, 0]]], dtype=torch.int32),
            "ctx_seq_val": torch.tensor([[[2.1, 2.0], [0.0, 0.0]]], dtype=torch.float32),
            "ctx_type": "mid",
            "mode": "distributed_priority_reservoir",
            "band_low_sigma": kwargs["band_low_sigma"],
            "band_high_sigma": kwargs["band_high_sigma"],
            "num_ctx_sequences": kwargs["num_ctx_sequences"],
            "reservoir_fill": torch.tensor([[2, 0]], dtype=torch.int32),
            "reservoir_n": torch.tensor([[2, 0]], dtype=torch.int64),
            "merge_report": {},
        }

    merged = merge_mid_ctx_candidate_partials(
        [(_mid_ctx_metadata(0), payload)],
        latent_stats_payload=_mid_ctx_latent_stats_payload(),
        on_truncation="replay_fallback",
        replay_fallback_fn=replay_fallback_fn,
    )

    assert merged["ctx_seq_idx"].tolist() == [[[2, 1], [0, 0]]]
    assert merged["merge_report"]["replay_fallback_executed"] is True
    assert merged["merge_report"]["mode"] == "stats_aware_replay_fallback"


def test_merge_mid_ctx_candidates_allow_bounded_approx_reports_cleanup_blocked():
    payload = _mid_ctx_payload(
        component_ids=[0],
        latent_ids=[0],
        sequence_ids=[1],
        activation_values=[2.0],
        priorities=[0.3],
    )
    payload["truncation_counters"][0, 0] = 1

    merged = merge_mid_ctx_candidate_partials(
        [(_mid_ctx_metadata(0), payload)],
        latent_stats_payload=_mid_ctx_latent_stats_payload(),
        on_truncation="allow_bounded_approx",
    )

    assert merged["merge_report"]["bounded_approximation"] is True
    assert merged["merge_report"]["candidate_pool_cleanup_eligible"] is False


def test_merge_mid_ctx_candidates_priority_selection_is_stable_across_worker_order():
    worker_0 = _mid_ctx_payload(
        component_ids=[0, 0],
        latent_ids=[0, 0],
        sequence_ids=[1, 2],
        activation_values=[2.0, 2.1],
        priorities=[0.4, 0.1],
    )
    worker_1 = _mid_ctx_payload(
        component_ids=[0, 0],
        latent_ids=[0, 0],
        sequence_ids=[3, 4],
        activation_values=[2.2, 2.3],
        priorities=[0.3, 0.2],
    )
    partials = [(_mid_ctx_metadata(0), worker_0), (_mid_ctx_metadata(1), worker_1)]

    merged_forward = merge_mid_ctx_candidate_partials(
        partials,
        latent_stats_payload=_mid_ctx_latent_stats_payload(),
        num_ctx_sequences=2,
    )
    merged_reversed = merge_mid_ctx_candidate_partials(
        list(reversed(partials)),
        latent_stats_payload=_mid_ctx_latent_stats_payload(),
        num_ctx_sequences=2,
    )

    assert merged_forward["ctx_seq_idx"].tolist() == [[[2, 4], [0, 0]]]
    assert torch.equal(merged_forward["ctx_seq_idx"], merged_reversed["ctx_seq_idx"])
    assert torch.equal(merged_forward["ctx_seq_val"], merged_reversed["ctx_seq_val"])


def test_seq_repr_cap_mapping_is_deterministic_and_seeded():
    first = build_seq_repr_cap_mapping(
        total_sequence_count=8,
        max_repr_seqs=4,
        sampling_seed=123,
        dataset_fingerprint="dataset-a",
    )
    second = build_seq_repr_cap_mapping(
        total_sequence_count=8,
        max_repr_seqs=4,
        sampling_seed=123,
        dataset_fingerprint="dataset-a",
    )
    changed = build_seq_repr_cap_mapping(
        total_sequence_count=8,
        max_repr_seqs=4,
        sampling_seed=124,
        dataset_fingerprint="dataset-a",
    )

    assert torch.equal(first["slot_to_id"], second["slot_to_id"])
    assert torch.equal(first["id_to_slot"], second["id_to_slot"])
    assert not torch.equal(first["slot_to_id"], changed["slot_to_id"])


def test_seq_repr_cap_seed_excludes_run_id_and_uses_sampling_seed_inputs():
    seed_a = derive_seq_repr_cap_seed(
        sampling_seed=123,
        dataset_fingerprint="dataset-a",
        cap_size=4,
        total_sequence_count=8,
    )
    seed_b = derive_seq_repr_cap_seed(
        sampling_seed=123,
        dataset_fingerprint="dataset-a",
        cap_size=4,
        total_sequence_count=8,
    )
    seed_changed = derive_seq_repr_cap_seed(
        sampling_seed=124,
        dataset_fingerprint="dataset-a",
        cap_size=4,
        total_sequence_count=8,
    )

    assert seed_a == seed_b
    assert seed_a != seed_changed


def test_merge_seq_repr_partials_uncapped_copies_global_sequence_rows():
    worker_0 = _seq_repr_payload(
        rows={1: [1.0, 1.1], 2: [2.0, 2.2]},
        n_seqs=4,
    )
    worker_1 = _seq_repr_payload(
        rows={3: [3.0, 3.3], 4: [4.0, 4.4]},
        n_seqs=4,
    )

    merged = merge_seq_repr_partials(
        [(_seq_repr_metadata(0), worker_0), (_seq_repr_metadata(1), worker_1)]
    )

    assert merged["is_capped"] is False
    assert merged["repr_buf"].tolist() == [
        [0.0, 0.0],
        [1.0, 1.099609375],
        [2.0, 2.19921875],
        [3.0, 3.30078125],
        [4.0, 4.3984375],
    ]
    assert merged["merge_report"]["written_slots"] == 4


def test_merge_seq_repr_partials_capped_uses_global_slot_mapping():
    mapping = build_seq_repr_cap_mapping(
        total_sequence_count=4,
        max_repr_seqs=2,
        sampling_seed=123,
        dataset_fingerprint="dataset-a",
    )
    selected = mapping["slot_to_id"][1:].tolist()
    worker_0 = _seq_repr_payload(rows={1: [1.0, 1.1], 2: [2.0, 2.2]}, n_seqs=4)
    worker_1 = _seq_repr_payload(rows={3: [3.0, 3.3], 4: [4.0, 4.4]}, n_seqs=4)

    merged = merge_seq_repr_partials(
        [(_seq_repr_metadata(0), worker_0), (_seq_repr_metadata(1), worker_1)],
        seq_repr_mapping=mapping,
    )

    assert merged["is_capped"] is True
    assert merged["slot_to_id"].tolist() == mapping["slot_to_id"].tolist()
    for slot, sequence_id in enumerate(selected, start=1):
        assert torch.allclose(
            merged["repr_buf"][slot].float(),
            torch.tensor([float(sequence_id), float(sequence_id) + sequence_id / 10]),
            atol=1e-3,
        )


def test_merge_seq_repr_partials_uses_assigned_sequence_ids_for_noncontiguous_shards():
    worker_0 = _seq_repr_payload(rows={1: [1.0, 1.1], 3: [3.0, 3.3]}, n_seqs=4)
    worker_1 = _seq_repr_payload(rows={2: [2.0, 2.2], 4: [4.0, 4.4]}, n_seqs=4)
    worker_0_metadata = _seq_repr_metadata(0).model_copy(
        update={"sequence_id_min": 1, "sequence_id_max": 3}
    )
    worker_1_metadata = _seq_repr_metadata(1).model_copy(
        update={"sequence_id_min": 2, "sequence_id_max": 4}
    )

    merged = merge_seq_repr_partials(
        [(worker_0_metadata, worker_0), (worker_1_metadata, worker_1)],
        sequence_ids_by_worker={0: [1, 3], 1: [2, 4]},
    )

    assert merged["repr_buf"].tolist() == [
        [0.0, 0.0],
        [1.0, 1.099609375],
        [2.0, 2.19921875],
        [3.0, 3.30078125],
        [4.0, 4.3984375],
    ]


def test_load_and_merge_seq_repr_partials_round_trip(tmp_path):
    worker_0 = _seq_repr_payload(rows={1: [1.0, 1.1], 2: [2.0, 2.2]}, n_seqs=4)
    worker_1 = _seq_repr_payload(rows={3: [3.0, 3.3], 4: [4.0, 4.4]}, n_seqs=4)
    paths = []
    for metadata, payload in [(_seq_repr_metadata(0), worker_0), (_seq_repr_metadata(1), worker_1)]:
        path = tmp_path / f"seq_repr_worker_{metadata.worker_id}.pt"
        save_pass1_partial(path, metadata, payload)
        paths.append(path)

    merged = load_and_merge_seq_repr_partials(
        paths,
        expected_config_hash="abcdef1234567890",
    )

    assert merged["repr_buf"][4].tolist() == [4.0, 4.3984375]


def test_merge_seq_repr_partials_rejects_duplicate_selected_slot():
    first = _seq_repr_payload(rows={1: [1.0, 1.1]}, n_seqs=4)
    second = _seq_repr_payload(rows={1: [9.0, 9.9]}, n_seqs=4)
    duplicate_metadata = _seq_repr_metadata(1).model_copy(
        update={"sequence_id_min": 1, "sequence_id_max": 1, "worker_id": 1}
    )

    with pytest.raises(ValueError, match="written more than once"):
        merge_seq_repr_partials([(_seq_repr_metadata(0), first), (duplicate_metadata, second)])


def test_merge_logit_ctx_partials_selects_global_event_topk_and_sums_counts():
    worker_0 = _logit_ctx_payload(
        counts=[[2, 1]],
        tokens=[[[10, 11], [20, 21]]],
        probs=[[[0.7, 0.5], [0.1, 0.9]]],
    )
    worker_1 = _logit_ctx_payload(
        counts=[[3, 4]],
        tokens=[[[12, 13], [22, 23]]],
        probs=[[[0.8, 0.6], [0.8, 0.7]]],
    )

    merged = merge_logit_ctx_partials(
        [(_logit_ctx_metadata(0), worker_0), (_logit_ctx_metadata(1), worker_1)],
        vocab_size=100,
    )

    assert merged["latent_counts"].tolist() == [[5, 5]]
    assert merged["top_tokens"].tolist() == [[[12, 10], [21, 22]]]
    assert torch.allclose(
        merged["top_probs"],
        torch.tensor([[[0.8, 0.7], [0.9, 0.8]]]),
    )


def test_merge_logit_ctx_partials_tie_breaks_by_token_then_worker_then_row():
    worker_1 = _logit_ctx_payload(
        counts=[[1, 0]],
        tokens=[[[9, 4, 4], [0, 0, 0]]],
        probs=[[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]],
    )
    worker_0 = _logit_ctx_payload(
        counts=[[1, 0]],
        tokens=[[[5, 4, 7], [0, 0, 0]]],
        probs=[[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]],
    )

    merged = merge_logit_ctx_partials(
        [(_logit_ctx_metadata(1), worker_1), (_logit_ctx_metadata(0), worker_0)],
        vocab_size=100,
    )

    assert merged["top_tokens"][0, 0].tolist() == [4, 4, 4]
    assert merged["merge_report"]["tie_breaking"] == (
        "probability_desc_token_asc_worker_asc_candidate_row_asc"
    )


def test_load_and_merge_logit_ctx_partials_round_trip(tmp_path):
    worker_0 = _logit_ctx_payload(
        counts=[[2, 1]],
        tokens=[[[10, 11], [20, 21]]],
        probs=[[[0.7, 0.5], [0.1, 0.9]]],
    )
    worker_1 = _logit_ctx_payload(
        counts=[[3, 4]],
        tokens=[[[12, 13], [22, 23]]],
        probs=[[[0.8, 0.6], [0.8, 0.7]]],
    )
    paths = []
    for metadata, payload in [(_logit_ctx_metadata(0), worker_0), (_logit_ctx_metadata(1), worker_1)]:
        path = tmp_path / f"logit_ctx_worker_{metadata.worker_id}.pt"
        save_pass1_partial(path, metadata, payload)
        paths.append(path)

    merged = load_and_merge_logit_ctx_partials(
        paths,
        expected_config_hash="abcdef1234567890",
        vocab_size=100,
    )

    assert merged["top_tokens"].tolist() == [[[12, 10], [21, 22]]]


def test_merge_logit_ctx_partials_rejects_token_above_vocab():
    payload = _logit_ctx_payload(
        counts=[[1, 0]],
        tokens=[[[100, 1], [0, 0]]],
        probs=[[[0.5, 0.4], [0.0, 0.0]]],
    )

    with pytest.raises(ValueError, match="above vocabulary range"):
        merge_logit_ctx_partials([(_logit_ctx_metadata(0), payload)], vocab_size=100)


def test_merge_logit_ctx_partials_rejects_non_finite_probabilities():
    payload = _logit_ctx_payload(
        counts=[[1, 0]],
        tokens=[[[2, 1], [0, 0]]],
        probs=[[[float("nan"), 0.4], [0.0, 0.0]]],
    )

    with pytest.raises(ValueError, match="top_probs contains non-finite"):
        merge_logit_ctx_partials([(_logit_ctx_metadata(0), payload)], vocab_size=100)


def test_merge_logit_ctx_partials_matches_single_event_stream_for_split_workers():
    worker_0 = _logit_ctx_payload(
        counts=[[1, 0]],
        tokens=[[[3, 8, 5], [0, 0, 0]]],
        probs=[[[0.3, 0.8, 0.5], [0.0, 0.0, 0.0]]],
    )
    worker_1 = _logit_ctx_payload(
        counts=[[2, 0]],
        tokens=[[[1, 2, 4], [0, 0, 0]]],
        probs=[[[0.9, 0.4, 0.7], [0.0, 0.0, 0.0]]],
    )

    merged = merge_logit_ctx_partials(
        [(_logit_ctx_metadata(0), worker_0), (_logit_ctx_metadata(1), worker_1)],
        vocab_size=100,
    )

    assert merged["latent_counts"].tolist() == [[3, 0]]
    assert merged["top_tokens"][0, 0].tolist() == [1, 8, 4]
    assert torch.allclose(
        merged["top_probs"][0, 0],
        torch.tensor([0.9, 0.8, 0.7]),
    )


def test_merge_seq_latent_index_shards_copies_disjoint_worker_outputs(tmp_path):
    worker_0 = tmp_path / "worker_0"
    worker_1 = tmp_path / "worker_1"
    output_dir = tmp_path / "merged"
    _write_seq_latent_index_shard(
        worker_0 / "shard_0.pt",
        {0: torch.tensor([[1, 10], [2, 11]], dtype=torch.int32)},
    )
    _write_seq_latent_index_shard(
        worker_1 / "shard_1.pt",
        {0: torch.tensor([[3, 12], [4, 13]], dtype=torch.int32)},
    )

    report = merge_seq_latent_index_shards(
        [worker_0, worker_1],
        output_dir,
        expected_shard_ids=[0, 1],
        shard_id_ranges={0: (1, 2), 1: (3, 4)},
    )

    assert report["copied_shards"] == [0, 1]
    assert (output_dir / "shard_0.pt").exists()
    assert (output_dir / "shard_1.pt").exists()
    assert torch.equal(
        torch.load(output_dir / "shard_1.pt", map_location="cpu")[0],
        torch.tensor([[3, 12], [4, 13]], dtype=torch.int32),
    )


def test_merge_seq_latent_index_shards_allows_identical_duplicate(tmp_path):
    worker_0 = tmp_path / "worker_0"
    worker_1 = tmp_path / "worker_1"
    output_dir = tmp_path / "merged"
    payload = {0: torch.tensor([[1, 10], [2, 11]], dtype=torch.int32)}
    _write_seq_latent_index_shard(worker_0 / "shard_0.pt", payload)
    _write_seq_latent_index_shard(worker_1 / "shard_0.pt", payload)

    report = merge_seq_latent_index_shards(
        [worker_0, worker_1],
        output_dir,
        expected_shard_ids=[0],
    )

    assert report["copied_shards"] == [0]
    assert report["duplicate_identical_shards"] == [0]


def test_merge_seq_latent_index_shards_rejects_different_duplicate(tmp_path):
    worker_0 = tmp_path / "worker_0"
    worker_1 = tmp_path / "worker_1"
    output_dir = tmp_path / "merged"
    _write_seq_latent_index_shard(
        worker_0 / "shard_0.pt",
        {0: torch.tensor([[1, 10]], dtype=torch.int32)},
    )
    _write_seq_latent_index_shard(
        worker_1 / "shard_0.pt",
        {0: torch.tensor([[1, 99]], dtype=torch.int32)},
    )

    with pytest.raises(ValueError, match="duplicate seq_latent_index shard differs"):
        merge_seq_latent_index_shards([worker_0, worker_1], output_dir, expected_shard_ids=[0])


def test_merge_seq_latent_index_shards_requires_expected_outputs(tmp_path):
    worker_0 = tmp_path / "worker_0"
    output_dir = tmp_path / "merged"
    _write_seq_latent_index_shard(
        worker_0 / "shard_0.pt",
        {0: torch.tensor([[1, 10]], dtype=torch.int32)},
    )

    with pytest.raises(ValueError, match="missing seq_latent_index shard outputs"):
        merge_seq_latent_index_shards([worker_0], output_dir, expected_shard_ids=[0, 1])


def test_merge_seq_latent_index_shards_disabled_noops(tmp_path):
    report = merge_seq_latent_index_shards(
        [tmp_path / "missing_worker"],
        tmp_path / "merged",
        expected_shard_ids=[0],
        enabled=False,
    )

    assert report["enabled"] is False
    assert not (tmp_path / "merged").exists()


def test_merge_seq_latent_index_shards_rejects_out_of_range_sequence_ids(tmp_path):
    worker_0 = tmp_path / "worker_0"
    _write_seq_latent_index_shard(
        worker_0 / "shard_0.pt",
        {0: torch.tensor([[3, 10]], dtype=torch.int32)},
    )

    with pytest.raises(ValueError, match="outside shard range"):
        merge_seq_latent_index_shards(
            [worker_0],
            tmp_path / "merged",
            expected_shard_ids=[0],
            shard_id_ranges={0: (1, 2)},
        )


def test_merge_pass1_worker_outputs_writes_canonical_artifacts_and_report(tmp_path):
    manifest = _pass1_merge_manifest(tmp_path)
    for worker_id in range(2):
        pass1_dir = (
            tmp_path
            / "outputs"
            / manifest.run_id
            / "distributed"
            / "workers"
            / f"worker_{worker_id:03d}"
            / "pass1"
        )
        pass1_dir.mkdir(parents=True, exist_ok=True)
        _write_pass1_worker_partials(pass1_dir, worker_id)
        _write_seq_latent_index_shard(
            pass1_dir / "seq_latent_index" / f"shard_{worker_id}.pt",
            {0: torch.tensor([[1 + worker_id * 2, 10 + worker_id]], dtype=torch.int32)},
        )

    result = merge_pass1_worker_outputs(manifest, vocab_size=100)

    for artifact_path in result["artifacts"].values():
        assert Path(artifact_path).exists()
        assert not Path(f"{artifact_path}.tmp").exists()
    assert (Path(manifest.output_root) / "seq_latent_index" / "shard_0.pt").exists()
    assert Path(result["sanity_report"]).exists()
    report = json.loads(Path(result["sanity_report"]).read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["context_fill_rates"]["top_ctx"] > 0
    assert report["seq_repr_fill"]["filled"] == 4
    assert report["logit_ctx_counts"]["total"] == 6
    top_ctx = torch.load(result["artifacts"]["top_ctx"], map_location="cpu", weights_only=False)
    assert top_ctx["metadata"]["run_id"] == manifest.run_id
    assert top_ctx["metadata"]["config_hash"] == manifest.normalized_config_hash
    assert top_ctx["config_hash"] == manifest.normalized_config_hash
    saved_manifest = load_manifest(manifest.manifest_path)
    assert saved_manifest.status == "completed"
    assert saved_manifest.work_assignments.pass2_replay_sequence_count == 2
    assert saved_manifest.work_assignments.pass2_sequence_ids == {
        "0": [1],
        "1": [3],
    }


def test_merge_pass1_worker_outputs_one_worker_matches_partial_schema(tmp_path):
    base_manifest = _pass1_merge_manifest(tmp_path)
    manifest = base_manifest.model_copy(
        update={
            "worker_count": 1,
            "devices": [DeviceAssignment(worker_id=0, physical_id=None, logical_id="cpu")],
            "shard_table": [base_manifest.shard_table[0]],
            "work_assignments": WorkAssignments(
                pass1_shards={"0": [0]},
                pass1_sequence_totals={"0": 2},
            ),
        }
    )
    pass1_dir = (
        Path(manifest.distributed_root)
        / "workers"
        / "worker_000"
        / "pass1"
    )
    pass1_dir.mkdir(parents=True, exist_ok=True)
    _write_pass1_worker_partials(pass1_dir, 0)
    _write_seq_latent_index_shard(
        pass1_dir / "seq_latent_index" / "shard_0.pt",
        {0: torch.tensor([[1, 10]], dtype=torch.int32)},
    )

    result = merge_pass1_worker_outputs(manifest, vocab_size=100)

    latent_stats = torch.load(result["artifacts"]["latent_stats"], map_location="cpu")
    top_ctx = torch.load(result["artifacts"]["top_ctx"], map_location="cpu")
    seq_repr = torch.load(result["artifacts"]["seq_repr"], map_location="cpu")
    assert latent_stats["active_count"].shape == (1, 2)
    assert top_ctx["ctx_seq_idx"].shape == (1, 2, 2)
    assert seq_repr["repr_buf"].shape == (5, 2)
    saved_manifest = load_manifest(manifest.manifest_path)
    assert saved_manifest.status == "completed"
    assert saved_manifest.work_assignments.pass2_sequence_ids == {"0": [1, 2]}


def test_one_worker_merge_outputs_feed_negative_context_stage(tmp_path):
    base_manifest = _pass1_merge_manifest(tmp_path)
    manifest = base_manifest.model_copy(
        update={
            "worker_count": 1,
            "devices": [DeviceAssignment(worker_id=0, physical_id=None, logical_id="cpu")],
            "shard_table": [base_manifest.shard_table[0]],
            "work_assignments": WorkAssignments(
                pass1_shards={"0": [0]},
                pass1_sequence_totals={"0": 2},
            ),
        }
    )
    pass1_dir = Path(manifest.distributed_root) / "workers" / "worker_000" / "pass1"
    pass1_dir.mkdir(parents=True, exist_ok=True)
    _write_pass1_worker_partials(pass1_dir, 0)

    merge_pass1_worker_outputs(manifest, seq_latent_index_enabled=False, vocab_size=100)

    def fake_build_neg_ctx(seq_repr, top_ctx, mid_ctx, output_neg_ctx):
        assert seq_repr.n_seqs == 4
        assert top_ctx.ctx_seq_idx.shape == (1, 2, 2)
        assert mid_ctx.ctx_type == "mid"
        output_neg_ctx.ctx_seq_idx[0, 0, :1] = torch.tensor([2], dtype=torch.int32)
        output_neg_ctx.ctx_seq_val[0, 0, :1] = torch.tensor([0.9], dtype=torch.float32)
        return NegCtxStats(backend="single_gpu_exact", devices=["cpu"], fill_counts=[1])

    result = run_negative_context_stage(
        manifest.output_root,
        manifest_path=manifest.manifest_path,
        build_fn=fake_build_neg_ctx,
    )

    payload = torch.load(result.neg_ctx_path, map_location="cpu", weights_only=False)
    assert payload["ctx_type"] == "neg"
    assert payload["ctx_seq_idx"][0, 0, 0].item() == 2
    assert (Path(manifest.output_root) / "distributed" / "parts" / "neg_ctx" / "completed.json").exists()


def test_merge_latent_stats_partials_rejects_duplicate_worker():
    partial = (_metadata(0), _payload({(0, 0): [1.0]}, {(0, 0): [0.5]}))

    with pytest.raises(ValueError, match="duplicate latent_stats partial"):
        merge_latent_stats_partials([partial, partial])


def test_merge_latent_stats_partials_rejects_negative_variance_state():
    payload = _payload({(0, 0): [1.0]}, {(0, 0): [0.5]})
    payload["m2"][0, 0] = -1.0

    with pytest.raises(ValueError, match="m2 contains non-finite|negative"):
        merge_latent_stats_partials([(_metadata(0), payload)])


def test_merge_latent_stats_partials_clamps_tiny_negative_variance_noise():
    payload = _payload({(0, 0): [1.0]}, {(0, 0): [0.5]})
    payload["m2"][0, 0] = -5e-4

    merged = merge_latent_stats_partials([(_metadata(0), payload)])

    assert merged["m2"][0, 0].item() == 0.0


def test_merge_latent_stats_partials_clamps_relative_negative_variance_noise():
    payload = _payload({(0, 0): [1.0, 3.0]}, {(0, 0): [0.5]})
    payload["m2"][0, 0] = -0.05
    payload["m2"][0, 1] = 100000.0

    merged = merge_latent_stats_partials([(_metadata(0), payload)])

    assert merged["m2"][0, 0].item() == 0.0


def _combine_value_maps(
    first: dict[tuple[int, int], list[float]],
    second: dict[tuple[int, int], list[float]],
) -> dict[tuple[int, int], list[float]]:
    keys = set(first) | set(second)
    return {key: list(first.get(key, [])) + list(second.get(key, [])) for key in keys}


def _stats_store_for_batches(
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> LatentStats:
    store = LatentStats(device=torch.device("cpu"))
    store.num_components = 2
    store.sae_config = SimpleNamespace(d_sae=3)
    for top_acts, top_indices in batches:
        store.update_component(0, (top_acts, top_indices))
    if not store._allocated:
        store.allocate()
    return store


def _mid_ctx_payload(
    *,
    component_ids: list[int],
    latent_ids: list[int],
    sequence_ids: list[int],
    activation_values: list[float],
    priorities: list[float],
) -> dict[str, object]:
    return {
        "component_ids": torch.tensor(component_ids, dtype=torch.int16),
        "latent_ids": torch.tensor(latent_ids, dtype=torch.int32),
        "sequence_ids": torch.tensor(sequence_ids, dtype=torch.int32),
        "activation_values": torch.tensor(activation_values, dtype=torch.float32),
        "priorities": torch.tensor(priorities, dtype=torch.float32),
        "candidate_pool_settings": {
            "mode": "test",
            "num_ctx_sequences": 2,
        },
        "truncation_counters": torch.zeros((1, 2), dtype=torch.int64),
        "ctx_seq_idx": torch.zeros((1, 2, 2), dtype=torch.int32),
        "ctx_seq_val": torch.zeros((1, 2, 2), dtype=torch.float32),
    }


def _mid_ctx_latent_stats_payload() -> dict[str, torch.Tensor]:
    return {
        "seq_count": torch.tensor([[2, 2]], dtype=torch.int64),
        "mean_seq": torch.tensor([[1.0, 10.0]], dtype=torch.float32),
        "m2_seq": torch.tensor([[1.0, 4.0]], dtype=torch.float32),
    }


def _seq_repr_payload(
    *,
    rows: dict[int, list[float]],
    n_seqs: int,
) -> dict[str, object]:
    repr_buf = torch.zeros((n_seqs + 1, 2), dtype=torch.float16)
    for sequence_id, values in rows.items():
        repr_buf[sequence_id] = torch.tensor(values, dtype=torch.float16)
    return {
        "repr_buf": repr_buf,
        "repr_mode": "mean_pool",
        "repr_dim": 2,
        "n_seqs": n_seqs,
        "n_stored": n_seqs,
        "is_capped": False,
    }


def _logit_ctx_payload(
    *,
    counts: list[list[int]],
    tokens: list[list[list[int]]],
    probs: list[list[list[float]]],
) -> dict[str, object]:
    return {
        "latent_counts": torch.tensor(counts, dtype=torch.int64),
        "top_tokens": torch.tensor(tokens, dtype=torch.int32),
        "top_probs": torch.tensor(probs, dtype=torch.float32),
    }


def _write_seq_latent_index_shard(
    path,
    payload: dict[int, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _pass1_merge_manifest(tmp_path) -> DistributedRunManifest:
    run_id = "20260517-002500-abcdef12"
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    return DistributedRunManifest.model_validate(
        {
            "run_id": run_id,
            "run_mode": "distributed_simple_exact",
            "status": "running",
            "created_at": "2026-05-17T00:25:00Z",
            "config_path": str(tmp_path / "config.yaml"),
            "normalized_config_hash": "abcdef1234567890",
            "project_root": str(tmp_path),
            "output_root": str(output_root),
            "distributed_root": str(distributed_root),
            "manifest_path": str(distributed_root / "manifest.json"),
            "metrics_path": str(distributed_root / "reports" / "run_metrics.jsonl"),
            "run_summary_path": str(distributed_root / "reports" / "run_summary.json"),
            "model_path": str(tmp_path / "model.pt"),
            "sae_path": str(tmp_path / "sae"),
            "dataset_path": str(tmp_path / "data"),
            "worker_count": 2,
            "devices": [
                {"worker_id": 0, "physical_id": None, "logical_id": "cpu"},
                {"worker_id": 1, "physical_id": None, "logical_id": "cpu"},
            ],
            "shard_table": [
                {
                    "shard_index": 0,
                    "shard_filename": "shard_0.pt",
                    "sequence_count": 2,
                    "global_start_id": 1,
                    "global_end_id": 3,
                    "shard_size_bytes": 1,
                    "shard_mtime_ns": 1,
                    "index_filename": "shard_0.idx.npy",
                    "index_size_bytes": 1,
                    "index_mtime_ns": 1,
                },
                {
                    "shard_index": 1,
                    "shard_filename": "shard_1.pt",
                    "sequence_count": 2,
                    "global_start_id": 3,
                    "global_end_id": 5,
                    "shard_size_bytes": 1,
                    "shard_mtime_ns": 1,
                    "index_filename": "shard_1.idx.npy",
                    "index_size_bytes": 1,
                    "index_mtime_ns": 1,
                },
            ],
            "work_assignments": {
                "pass1_shards": {"0": [0], "1": [1]},
                "pass1_sequence_totals": {"0": 2, "1": 2},
            },
        }
    )


def _write_pass1_worker_partials(pass1_dir, worker_id: int) -> None:
    artifacts = {
        "latent_stats": ("latent_stats.partial.pt", _small_latent_stats_payload(worker_id)),
        "top_ctx": (
            "top_ctx.partial.pt",
            {
                "ctx_seq_idx": torch.tensor(
                    [[[1 + worker_id * 2, 2 + worker_id * 2], [0, 0]]],
                    dtype=torch.int32,
                ),
                "ctx_seq_val": torch.tensor([[[0.5, 0.4], [0.0, 0.0]]], dtype=torch.float32),
                "ctx_type": "top",
            },
        ),
        "mid_ctx_candidates": (
            "mid_ctx_candidates.partial.pt",
            _mid_ctx_payload(
                component_ids=[],
                latent_ids=[],
                sequence_ids=[],
                activation_values=[],
                priorities=[],
            ),
        ),
        "seq_repr": (
            "seq_repr.partial.pt",
            _seq_repr_payload(
                rows={
                    1 + worker_id * 2: [1.0 + worker_id, 1.1 + worker_id],
                    2 + worker_id * 2: [2.0 + worker_id, 2.2 + worker_id],
                },
                n_seqs=4,
            ),
        ),
        "logit_ctx": (
            "logit_ctx.partial.pt",
            _logit_ctx_payload(
                counts=[[2 + worker_id, worker_id]],
                tokens=[[[10 + worker_id, 11 + worker_id], [20 + worker_id, 0]]],
                probs=[[[0.6 + worker_id * 0.1, 0.5], [0.3, 0.0]]],
            ),
        ),
    }
    for artifact_name, (filename, payload) in artifacts.items():
        save_pass1_partial(
            pass1_dir / filename,
            _pass1_merge_metadata(artifact_name, worker_id),
            payload,
        )


def _pass1_merge_metadata(artifact_name: str, worker_id: int) -> Pass1PartialMetadata:
    return Pass1PartialMetadata(
        artifact_name=artifact_name,
        run_id="20260517-002500-abcdef12",
        worker_id=worker_id,
        shard_ids=[worker_id],
        sequence_id_min=1 + worker_id * 2,
        sequence_id_max=2 + worker_id * 2,
        sequence_count=2,
        config_hash="abcdef1234567890",
        physical_id=None,
        logical_id="cpu",
        created_at="2026-05-17T00:25:00Z",
        component_count=1,
        d_sae=2,
    )


def _small_latent_stats_payload(worker_id: int) -> dict[str, object]:
    shape = (1, 2)
    active_count = torch.tensor([[1 + worker_id, 0]], dtype=torch.int64)
    mean = torch.tensor([[1.0 + worker_id, 0.0]], dtype=torch.float32)
    return {
        "active_count": active_count,
        "mean": mean,
        "mean_abs": mean.abs(),
        "m2": torch.zeros(shape, dtype=torch.float32),
        "m2_abs": torch.zeros(shape, dtype=torch.float32),
        "seq_count": torch.tensor([[2, 0]], dtype=torch.int64),
        "mean_seq": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "m2_seq": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "component_steps": {0: 1},
    }
