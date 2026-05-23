import json

import pytest
import torch

from pipeline.distributed.pass2_partials import (
    CandidateDumpMetadata,
    CandidatePreAggregationMetadata,
    expand_candidate_dump_to_contributions,
    save_candidate_dump_partial,
    save_candidate_preaggregation_partial,
)
from pipeline.distributed.pass2_reduce import (
    apply_pmi_postprocess_to_topk,
    attach_simple_exact_dump_to_store,
    build_global_top_ctx_target_mapping,
    build_simple_exact_candidate_dump,
    build_pass2_reduce_manifest_metrics,
    cleanup_mapreduce_target_shards,
    estimate_mapreduce_reducer_input_bytes,
    estimate_mapreduce_shard_tensor_bytes,
    format_pass2_reduce_benchmark_report,
    load_candidate_dump_reducer_inputs,
    load_candidate_preaggregation_reducer_inputs,
    load_global_active_count,
    load_global_top_ctx_target_mapping,
    load_mapreduce_partial_sum_shard,
    load_mapreduce_reducer_shards,
    MapReduceTargetShardResult,
    Pass2ReduceSchedulerConfig,
    partition_target_ranges,
    reduce_mapreduce_target_range,
    reduce_simple_exact_candidate_dump,
    run_mapreduce_reduce_and_write,
    run_simple_exact_reduce_and_write,
    save_mapreduce_partial_sum_shard,
    shard_preaggregation_by_target_range,
    stitch_mapreduce_target_shards,
    TargetRange,
    validate_mapreduce_partial_sum_shard,
    validate_pass2_reduce_scheduler_config,
    validate_saved_top_coactivation_artifact,
    validate_candidate_dump_reducer_inputs,
    validate_candidate_preaggregation_reducer_inputs,
    validate_pmi_reduce_inputs,
)
from pipeline.distributed.pass2_replay import hash_replay_sequence_ids


def _candidate_metadata(worker_id: int = 0, sequence_ids=None, **overrides) -> CandidateDumpMetadata:
    sequence_ids = list(sequence_ids or [1, 2])
    data = {
        "run_id": "20260517-002500-abcdef12",
        "worker_id": worker_id,
        "sequence_count": len(sequence_ids),
        "sequence_id_min": min(sequence_ids) if sequence_ids else None,
        "sequence_id_max": max(sequence_ids) if sequence_ids else None,
        "replay_sequence_hash": hash_replay_sequence_ids(sequence_ids),
        "config_hash": "abcdef1234567890",
        "physical_id": worker_id,
        "logical_id": "cuda:0",
        "created_at": "2026-05-17T00:25:00Z",
        "mode": "raw",
        "m": 3,
        "n_candidates_per_component": 2,
        "n_latents_per_latent": 4,
        "num_components": 2,
        "d_sae": 8,
        "token_count": 64 * len(sequence_ids),
        "seq_len": 64,
        "batch_count": 1,
        "estimated_dump_bytes": 24 * len(sequence_ids),
    }
    data.update(overrides)
    return CandidateDumpMetadata.model_validate(data)


def _candidate_payload(sequence_ids=None, *, candidate_ids=None, candidate_vals=None):
    sequence_ids = list(sequence_ids or [1, 2])
    row_count = len(sequence_ids)
    if candidate_ids is None:
        candidate_ids = torch.tensor(
            [[(sequence_id + 0) % 15, (sequence_id + 1) % 15, 0] for sequence_id in sequence_ids],
            dtype=torch.int32,
        )
    if candidate_vals is None:
        candidate_vals = torch.tensor(
            [[float(row + 1), 0.5, 0.0] for row in range(row_count)],
            dtype=torch.float32,
        )
    return {
        "sequence_ids": torch.tensor(sequence_ids, dtype=torch.int64),
        "candidate_ids": candidate_ids,
        "candidate_vals": candidate_vals,
        "total_tokens_processed": 64 * row_count,
    }


def _preaggregation_metadata(worker_id: int = 0, **overrides) -> CandidatePreAggregationMetadata:
    data = {
        "partial_schema_version": 1,
        "artifact_name": "candidate_preaggregation",
        "run_id": "20260517-002500-abcdef12",
        "worker_id": worker_id,
        "source_candidate_dump_schema_version": 1,
        "sequence_count": 2,
        "contribution_count": 2,
        "config_hash": "abcdef1234567890",
        "mode": "raw",
        "num_components": 2,
        "d_sae": 8,
        "m": 3,
        "target_start_id": 0,
        "target_end_id": 8,
        "created_at": "2026-05-17T00:25:00Z",
    }
    data.update(overrides)
    return CandidatePreAggregationMetadata.model_validate(data)


def _preaggregation_payload(*, target_ids=None):
    if target_ids is None:
        target_ids = torch.tensor([1, 6], dtype=torch.int64)
    return {
        "target_ids": target_ids,
        "candidate_ids": torch.tensor([2, 7], dtype=torch.int32),
        "values": torch.tensor([0.5, 1.25], dtype=torch.float32),
        "sequence_ids": torch.tensor([1, 2], dtype=torch.int64),
    }


def _top_ctx_payload():
    ctx_seq_idx = torch.zeros((2, 8, 2), dtype=torch.int32)
    ctx_seq_val = torch.zeros((2, 8, 2), dtype=torch.float32)
    ctx_seq_idx[0, 1, 0] = 1
    ctx_seq_val[0, 1, 0] = 3.0
    ctx_seq_idx[1, 4, 0] = 2
    ctx_seq_val[1, 4, 0] = 2.0
    ctx_seq_idx[0, 2, 0] = 3
    ctx_seq_val[0, 2, 0] = 1.0
    return {
        "ctx_seq_idx": ctx_seq_idx,
        "ctx_seq_val": ctx_seq_val,
        "ctx_type": "top",
    }


def _equivalence_candidate_rows():
    return {
        1: ([4, 7, 6, 0], [2.0, 1.0, 1.0, 0.0]),
        2: ([4, 5, 6, 0], [3.0, 1.0, 1.0, 0.0]),
        3: ([7, 4, 5, 0], [2.0, 2.0, 1.0, 0.0]),
    }


def _candidate_payload_from_rows(sequence_ids, rows):
    return _candidate_payload(
        sequence_ids,
        candidate_ids=torch.tensor([rows[int(sequence_id)][0] for sequence_id in sequence_ids], dtype=torch.int32),
        candidate_vals=torch.tensor([rows[int(sequence_id)][1] for sequence_id in sequence_ids], dtype=torch.float32),
    )


def _reduce_candidate_dump_fixture(metadata, payload, mapping, *, top_k, active_count=None):
    pre_metadata, pre_payload = expand_candidate_dump_to_contributions(
        metadata,
        payload,
        mapping.seq_offsets,
        mapping.seq_targets_global,
    )
    inputs = validate_candidate_preaggregation_reducer_inputs(
        [(pre_metadata, pre_payload)],
        expected_target_start_id=0,
        expected_target_end_id=metadata.num_components * metadata.d_sae,
    )
    result = reduce_mapreduce_target_range(inputs, n_latents_per_latent=top_k)
    top_indices, top_values = stitch_mapreduce_target_shards(
        [result],
        num_components=metadata.num_components,
        d_sae=metadata.d_sae,
        n_latents_per_latent=top_k,
    )
    if metadata.mode == "pmi":
        top_values = apply_pmi_postprocess_to_topk(
            top_indices,
            top_values,
            active_count=active_count,
            seq_offsets=mapping.seq_offsets,
            seq_targets_global=mapping.seq_targets_global,
            seq_len=metadata.seq_len,
            num_components=metadata.num_components,
            d_sae=metadata.d_sae,
            sae_k=1,
        )
    return top_indices, top_values


def _reduce_mapreduce_fixture(entries, mapping, *, top_k, active_count=None):
    first_metadata = entries[0][0]
    ranges = partition_target_ranges(first_metadata.num_components * first_metadata.d_sae, reducer_count=3)
    entries_by_reducer = {target_range.reducer_id: [] for target_range in ranges}
    for metadata, payload in entries:
        pre_metadata, pre_payload = expand_candidate_dump_to_contributions(
            metadata,
            payload,
            mapping.seq_offsets,
            mapping.seq_targets_global,
        )
        for target_range, shard in zip(ranges, shard_preaggregation_by_target_range(pre_metadata, pre_payload, ranges)):
            entries_by_reducer[target_range.reducer_id].append(shard)

    results = []
    for target_range in ranges:
        reducer_inputs = validate_candidate_preaggregation_reducer_inputs(
            entries_by_reducer[target_range.reducer_id],
            expected_mode=first_metadata.mode,
            expected_target_start_id=target_range.target_start_id,
            expected_target_end_id=target_range.target_end_id,
            expected_worker_ids=[metadata.worker_id for metadata, _payload in entries],
        )
        results.append(
            reduce_mapreduce_target_range(
                reducer_inputs,
                n_latents_per_latent=top_k,
                reducer_id=target_range.reducer_id,
            )
        )
    top_indices, top_values = stitch_mapreduce_target_shards(
        results,
        num_components=first_metadata.num_components,
        d_sae=first_metadata.d_sae,
        n_latents_per_latent=top_k,
    )
    if first_metadata.mode == "pmi":
        top_values = apply_pmi_postprocess_to_topk(
            top_indices,
            top_values,
            active_count=active_count,
            seq_offsets=mapping.seq_offsets,
            seq_targets_global=mapping.seq_targets_global,
            seq_len=first_metadata.seq_len,
            num_components=first_metadata.num_components,
            d_sae=first_metadata.d_sae,
            sae_k=1,
        )
    return top_indices, top_values


def test_candidate_dump_reducer_inputs_accept_good_worker_dumps(tmp_path):
    path_1 = tmp_path / "worker_001.pt"
    path_0 = tmp_path / "worker_000.pt"
    save_candidate_dump_partial(
        path_1,
        _candidate_metadata(worker_id=1, sequence_ids=[3]),
        _candidate_payload([3], candidate_ids=torch.tensor([[7, 0, 0]], dtype=torch.int32),
                           candidate_vals=torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float32)),
    )
    save_candidate_dump_partial(path_0, _candidate_metadata(worker_id=0), _candidate_payload())

    inputs = load_candidate_dump_reducer_inputs(
        [path_1, path_0],
        expected_config_hash="abcdef1234567890",
        expected_mode="raw",
    )

    assert [entry.metadata.worker_id for entry in inputs.entries] == [0, 1]
    assert inputs.mode == "raw"
    assert inputs.m == 3
    assert inputs.num_components == 2
    assert inputs.d_sae == 8
    assert inputs.total_sequence_count == 3
    assert inputs.total_token_count == 192


def test_global_top_ctx_mapping_builds_csr_and_sid_to_row():
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload())

    assert mapping.sequence_ids == (1, 2, 3)
    assert mapping.replay.sequence_count == 3
    assert mapping.seq_offsets.tolist() == [0, 1, 2, 3]
    assert mapping.seq_targets_global.tolist() == [1, 12, 2]
    assert mapping.sid_to_row == {1: 0, 2: 1, 3: 2}
    assert mapping.sid_to_row_tensor.tolist() == [-1, 0, 1, 2]


def test_global_top_ctx_mapping_loads_from_path_and_validates_dump_coverage(tmp_path):
    top_ctx_path = tmp_path / "top_ctx.pt"
    torch.save(_top_ctx_payload(), top_ctx_path)
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2]), _candidate_payload([1, 2])),
            (_candidate_metadata(worker_id=1, sequence_ids=[3]), _candidate_payload([3])),
        ]
    )

    mapping = load_global_top_ctx_target_mapping(top_ctx_path, dump_inputs=dump_inputs)

    assert mapping.sequence_ids == (1, 2, 3)


def test_simple_exact_candidate_dump_concatenates_in_sequence_id_order():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (
                _candidate_metadata(worker_id=1, sequence_ids=[3]),
                _candidate_payload(
                    [3],
                    candidate_ids=torch.tensor([[9, 8, 0]], dtype=torch.int32),
                    candidate_vals=torch.tensor([[3.0, 0.3, 0.0]], dtype=torch.float32),
                ),
            ),
            (
                _candidate_metadata(worker_id=0, sequence_ids=[1, 2]),
                _candidate_payload(
                    [1, 2],
                    candidate_ids=torch.tensor([[1, 2, 0], [4, 5, 0]], dtype=torch.int32),
                    candidate_vals=torch.tensor([[1.0, 0.1, 0.0], [2.0, 0.2, 0.0]], dtype=torch.float32),
                ),
            ),
        ]
    )
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)

    merged = build_simple_exact_candidate_dump(dump_inputs, mapping)

    assert merged.sequence_ids.tolist() == [1, 2, 3]
    assert merged.candidate_ids.tolist() == [[1, 2, 0], [4, 5, 0], [9, 8, 0]]
    assert merged.candidate_vals.tolist() == [[1.0, 0.10000000149011612, 0.0], [2.0, 0.20000000298023224, 0.0], [3.0, 0.30000001192092896, 0.0]]
    assert merged.sid_to_row == {1: 0, 2: 1, 3: 2}
    assert merged.sid_to_row_tensor.tolist() == [-1, 0, 1, 2]
    assert merged.seq_len == 64
    assert merged.total_token_count == 192


def test_simple_exact_candidate_dump_rejects_seq_len_mismatch():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2], seq_len=64), _candidate_payload([1, 2])),
            (_candidate_metadata(worker_id=1, sequence_ids=[3], seq_len=32), _candidate_payload([3])),
        ]
    )
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)

    with pytest.raises(ValueError, match="seq_len mismatch"):
        build_simple_exact_candidate_dump(dump_inputs, mapping)


def test_attach_simple_exact_dump_to_store_populates_reduce_contract():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2]), _candidate_payload([1, 2])),
            (_candidate_metadata(worker_id=1, sequence_ids=[3]), _candidate_payload([3])),
        ]
    )
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)
    merged = build_simple_exact_candidate_dump(dump_inputs, mapping)

    class FakeTopCoactivation:
        num_components = 2
        d_sae = 8
        n_latents_per_latent = 4
        n_candidates_per_component = 2
        M = 3
        mode = "raw"
        candidate_ids = None
        candidate_vals = None
        seq_id_to_row = {}
        sid_to_row_tensor = None
        total_tokens_processed = 0

    store = FakeTopCoactivation()
    attach_simple_exact_dump_to_store(store, merged)

    assert torch.equal(store.candidate_ids, merged.candidate_ids)
    assert torch.equal(store.candidate_vals, merged.candidate_vals)
    assert store.seq_id_to_row == {1: 0, 2: 1, 3: 2}
    assert store.sid_to_row_tensor.tolist() == [-1, 0, 1, 2]
    assert store.total_tokens_processed == 192


def test_reduce_simple_exact_candidate_dump_calls_existing_store_reduce():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2]), _candidate_payload([1, 2])),
            (_candidate_metadata(worker_id=1, sequence_ids=[3]), _candidate_payload([3])),
        ]
    )
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)
    merged = build_simple_exact_candidate_dump(dump_inputs, mapping)
    calls = []

    class FakeTopCoactivation:
        num_components = 2
        d_sae = 8
        n_latents_per_latent = 4
        n_candidates_per_component = 2
        M = 3
        mode = "raw"
        candidate_ids = None
        candidate_vals = None
        seq_id_to_row = {}
        sid_to_row_tensor = None
        total_tokens_processed = 0

        def reduce(self, seq_offsets, seq_targets_global, *, seq_len, active_count):
            calls.append((seq_offsets.clone(), seq_targets_global.clone(), seq_len, active_count))

    store = FakeTopCoactivation()
    reduce_simple_exact_candidate_dump(store, merged, mapping)

    assert len(calls) == 1
    assert torch.equal(calls[0][0], mapping.seq_offsets)
    assert torch.equal(calls[0][1], mapping.seq_targets_global)
    assert calls[0][2] == 64
    assert calls[0][3] is None
    assert torch.equal(store.candidate_ids, merged.candidate_ids)


def test_load_global_active_count_validates_config_and_shape(tmp_path):
    path = tmp_path / "latent_stats.pt"
    active_count = torch.ones((2, 8), dtype=torch.int64)
    torch.save(
        {
            "active_count": active_count,
            "config_hash": "abcdef1234567890",
            "metadata": {"config_hash": "abcdef1234567890"},
        },
        path,
    )

    loaded = load_global_active_count(
        path,
        expected_config_hash="abcdef1234567890",
        expected_num_components=2,
        expected_d_sae=8,
    )

    assert torch.equal(loaded, active_count)


def test_load_global_active_count_rejects_stale_config(tmp_path):
    path = tmp_path / "latent_stats.pt"
    torch.save({"active_count": torch.ones((2, 8), dtype=torch.int64), "config_hash": "old"}, path)

    with pytest.raises(ValueError, match="config hash mismatch"):
        load_global_active_count(path, expected_config_hash="abcdef1234567890")


def test_pmi_reduce_inputs_require_global_active_count():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2], mode="pmi"), _candidate_payload([1, 2])),
            (_candidate_metadata(worker_id=1, sequence_ids=[3], mode="pmi"), _candidate_payload([3])),
        ]
    )
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)
    merged = build_simple_exact_candidate_dump(dump_inputs, mapping)

    with pytest.raises(ValueError, match="active_count"):
        validate_pmi_reduce_inputs(merged, mapping, active_count=None)


def test_pmi_reduce_inputs_reject_worker_token_count_mismatch():
    payload_0 = _candidate_payload([1, 2])
    payload_0["total_tokens_processed"] = 1
    payload_1 = _candidate_payload([3])
    payload_1["total_tokens_processed"] = 2
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2], mode="pmi", token_count=1), payload_0),
            (_candidate_metadata(worker_id=1, sequence_ids=[3], mode="pmi", token_count=2), payload_1),
        ]
    )
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)
    merged = build_simple_exact_candidate_dump(dump_inputs, mapping)

    with pytest.raises(ValueError, match="token-count metadata"):
        validate_pmi_reduce_inputs(merged, mapping, active_count=torch.ones((2, 8), dtype=torch.int64))


def test_reduce_simple_exact_candidate_dump_passes_global_active_count_for_pmi():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2], mode="pmi"), _candidate_payload([1, 2])),
            (_candidate_metadata(worker_id=1, sequence_ids=[3], mode="pmi"), _candidate_payload([3])),
        ]
    )
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)
    merged = build_simple_exact_candidate_dump(dump_inputs, mapping)
    active_count = torch.ones((2, 8), dtype=torch.int64)
    calls = []

    class FakeTopCoactivation:
        num_components = 2
        d_sae = 8
        n_latents_per_latent = 4
        n_candidates_per_component = 2
        M = 3
        mode = "pmi"
        candidate_ids = None
        candidate_vals = None
        seq_id_to_row = {}
        sid_to_row_tensor = None
        total_tokens_processed = 0
        top_indices = torch.zeros((2, 8, 4), dtype=torch.int32)
        top_values = torch.zeros((2, 8, 4), dtype=torch.float32)

        def reduce(self, seq_offsets, seq_targets_global, *, seq_len, active_count):
            calls.append(active_count)
            self.top_values = torch.ones((2, 8, 4), dtype=torch.float32)

    store = FakeTopCoactivation()
    reduce_simple_exact_candidate_dump(store, merged, mapping, active_count=active_count)

    assert len(calls) == 1
    assert torch.equal(calls[0], active_count)


def test_reduce_simple_exact_candidate_dump_rejects_non_finite_pmi_output():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2], mode="pmi"), _candidate_payload([1, 2])),
            (_candidate_metadata(worker_id=1, sequence_ids=[3], mode="pmi"), _candidate_payload([3])),
        ]
    )
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)
    merged = build_simple_exact_candidate_dump(dump_inputs, mapping)

    class FakeTopCoactivation:
        num_components = 2
        d_sae = 8
        n_latents_per_latent = 4
        n_candidates_per_component = 2
        M = 3
        mode = "pmi"
        candidate_ids = None
        candidate_vals = None
        seq_id_to_row = {}
        sid_to_row_tensor = None
        total_tokens_processed = 0
        top_indices = torch.zeros((2, 8, 4), dtype=torch.int32)
        top_values = torch.zeros((2, 8, 4), dtype=torch.float32)

        def reduce(self, seq_offsets, seq_targets_global, *, seq_len, active_count):
            self.top_values = torch.full((2, 8, 4), float("nan"), dtype=torch.float32)

    with pytest.raises(ValueError, match="finite"):
        reduce_simple_exact_candidate_dump(
            FakeTopCoactivation(),
            merged,
            mapping,
            active_count=torch.ones((2, 8), dtype=torch.int64),
        )


def test_run_simple_exact_reduce_and_write_saves_canonical_artifact_and_report(tmp_path):
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2]), _candidate_payload([1, 2])),
            (_candidate_metadata(worker_id=1, sequence_ids=[3]), _candidate_payload([3])),
        ]
    )
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)

    class FakeTopCoactivation:
        num_components = 2
        d_sae = 8
        n_latents_per_latent = 4
        n_candidates_per_component = 2
        M = 3
        mode = "raw"
        candidate_ids = None
        candidate_vals = None
        seq_id_to_row = {}
        sid_to_row_tensor = None
        total_tokens_processed = 0
        top_indices = torch.zeros((2, 8, 4), dtype=torch.int32)
        top_values = torch.zeros((2, 8, 4), dtype=torch.float32)
        freq_factors = torch.ones(16, dtype=torch.float32)
        loaded_path = None

        def reduce(self, seq_offsets, seq_targets_global, *, seq_len, active_count):
            self.top_indices = torch.arange(64, dtype=torch.int32).reshape(2, 8, 4)
            self.top_values = torch.ones((2, 8, 4), dtype=torch.float32)

        def save(self, path):
            torch.save(
                {
                    "top_indices": self.top_indices,
                    "top_values": self.top_values,
                    "freq_factors": self.freq_factors,
                    "total_tokens_processed": self.total_tokens_processed,
                    "mode": self.mode,
                },
                path,
            )

        def load(self, path):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.loaded_path = path
            self.top_indices = payload["top_indices"]
            self.top_values = payload["top_values"]

    store = FakeTopCoactivation()
    result = run_simple_exact_reduce_and_write(store, dump_inputs, mapping, tmp_path / "outputs" / "run-1")

    assert result.artifact_path == tmp_path / "outputs" / "run-1" / "top_coactivation.pt"
    assert result.report_path == tmp_path / "outputs" / "run-1" / "distributed" / "reports" / "pass2_reduce_report.json"
    assert result.artifact_path.exists()
    assert result.report_path.exists()
    assert not result.artifact_path.with_name("top_coactivation.pt.tmp").exists()
    assert store.loaded_path == str(result.artifact_path)

    payload = torch.load(result.artifact_path, map_location="cpu", weights_only=False)
    assert set(payload) >= {"top_indices", "top_values", "freq_factors", "total_tokens_processed", "mode"}
    assert tuple(payload["top_indices"].shape) == (2, 8, 4)
    assert tuple(payload["top_values"].shape) == (2, 8, 4)
    assert payload["mode"] == "raw"

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["reducer_mode"] == "simple_exact"
    assert report["coactivation_mode"] == "raw"
    assert report["worker_count"] == 2
    assert report["replay_sequence_count"] == 3
    assert report["input_candidate_dump_bytes"] > 0
    assert report["merged_candidate_dump_bytes"] > 0
    assert report["output_artifact_size_bytes"] == result.artifact_path.stat().st_size
    assert report["peak_cpu_memory_bytes"] is None or report["peak_cpu_memory_bytes"] > 0
    assert report["output_nonzero_count"] == 64
    assert report["output_finite"] is True
    assert set(report["timing"]) >= {"build_dump_s", "reduce_s", "pmi_s", "save_s", "total_s"}
    assert report["timing"]["total_s"] >= report["timing"]["save_s"]
    assert report["manifest_metrics"]["reducer_mode"] == "simple_exact"
    assert report["manifest_metrics"]["input_bytes"] == report["input_candidate_dump_bytes"]


def test_pass2_reduce_benchmark_report_format_and_manifest_metrics():
    report = {
        "reducer_mode": "mapreduce_target_ranges",
        "coactivation_mode": "pmi",
        "backend": "cpu",
        "reducer_count": 2,
        "input_partial_sum_bytes": 123,
        "output_artifact_size_bytes": 456,
        "peak_cpu_memory_bytes": 789,
        "timing": {"total_s": 1.25, "reduce_s": 0.5, "pmi_s": 0.25, "shard_write_s": 0.1, "stitch_s": 0.2},
    }

    metrics = build_pass2_reduce_manifest_metrics(report)
    formatted = format_pass2_reduce_benchmark_report(report)

    assert metrics["input_bytes"] == 123
    assert metrics["output_artifact_size_bytes"] == 456
    assert metrics["total_s"] == 1.25
    assert "pass2 reduce benchmark:" in formatted
    assert "reducer_mode: mapreduce_target_ranges" in formatted
    assert "pmi_s: 0.250000" in formatted
    assert "shard_write_s: 0.100000" in formatted


def test_validate_saved_top_coactivation_artifact_rejects_bad_shape(tmp_path):
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2]), _candidate_payload([1, 2])),
            (_candidate_metadata(worker_id=1, sequence_ids=[3]), _candidate_payload([3])),
        ]
    )
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)
    dump = build_simple_exact_candidate_dump(dump_inputs, mapping)
    path = tmp_path / "top_coactivation.pt"
    torch.save(
        {
            "top_indices": torch.zeros((1, 1, 1), dtype=torch.int32),
            "top_values": torch.zeros((1, 1, 1), dtype=torch.float32),
            "freq_factors": torch.ones(16, dtype=torch.float32),
            "total_tokens_processed": 0,
            "mode": "raw",
        },
        path,
    )

    with pytest.raises(ValueError, match="top_indices shape mismatch"):
        validate_saved_top_coactivation_artifact(object(), path, dump=dump)


def test_partition_target_ranges_balances_remainders_without_dropping_targets():
    ranges = partition_target_ranges(num_targets=10, reducer_count=3)

    assert [(r.target_start_id, r.target_end_id) for r in ranges] == [(0, 4), (4, 7), (7, 10)]
    covered = [target for r in ranges for target in range(r.target_start_id, r.target_end_id)]
    assert covered == list(range(10))


def test_partition_target_ranges_allows_more_reducers_than_targets():
    ranges = partition_target_ranges(num_targets=2, reducer_count=4)

    assert [(r.target_start_id, r.target_end_id) for r in ranges] == [(0, 1), (1, 2), (2, 2), (2, 2)]
    covered = [target for r in ranges for target in range(r.target_start_id, r.target_end_id)]
    assert covered == [0, 1]


def test_shard_preaggregation_by_target_range_preserves_cross_range_candidates():
    metadata = _preaggregation_metadata(worker_id=0, target_start_id=0, target_end_id=16, contribution_count=4)
    payload = {
        "target_ids": torch.tensor([1, 2, 9, 10], dtype=torch.int64),
        "candidate_ids": torch.tensor([12, 13, 1, 2], dtype=torch.int32),
        "values": torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32),
        "sequence_ids": torch.tensor([1, 1, 2, 2], dtype=torch.int64),
    }
    ranges = partition_target_ranges(num_targets=16, reducer_count=2)

    shards = shard_preaggregation_by_target_range(metadata, payload, ranges)

    assert [shard.metadata.contribution_count for shard in shards] == [2, 2]
    assert shards[0].payload["target_ids"].tolist() == [1, 2]
    assert shards[0].payload["candidate_ids"].tolist() == [12, 13]
    assert shards[1].payload["target_ids"].tolist() == [9, 10]
    assert shards[1].payload["candidate_ids"].tolist() == [1, 2]


def test_preaggregation_reducer_validates_expected_worker_coverage():
    entries = [
        (_preaggregation_metadata(worker_id=0), _preaggregation_payload()),
    ]

    with pytest.raises(ValueError, match="worker coverage"):
        validate_candidate_preaggregation_reducer_inputs(entries, expected_worker_ids=[0, 1])


def test_reduce_mapreduce_target_range_merges_duplicates_and_uses_deterministic_topk():
    metadata_0 = _preaggregation_metadata(worker_id=0, contribution_count=4, target_start_id=0, target_end_id=8)
    payload_0 = {
        "target_ids": torch.tensor([1, 1, 1, 2], dtype=torch.int64),
        "candidate_ids": torch.tensor([7, 6, 5, 7], dtype=torch.int32),
        "values": torch.tensor([1.0, 3.0, 3.0, 4.0], dtype=torch.float32),
        "sequence_ids": torch.tensor([1, 1, 2, 2], dtype=torch.int64),
    }
    metadata_1 = _preaggregation_metadata(worker_id=1, contribution_count=3, target_start_id=0, target_end_id=8)
    payload_1 = {
        "target_ids": torch.tensor([1, 1, 2], dtype=torch.int64),
        "candidate_ids": torch.tensor([7, 4, 3], dtype=torch.int32),
        "values": torch.tensor([3.0, 4.0, 4.0], dtype=torch.float32),
        "sequence_ids": torch.tensor([3, 3, 3], dtype=torch.int64),
    }
    inputs = validate_candidate_preaggregation_reducer_inputs(
        [(metadata_1, payload_1), (metadata_0, payload_0)],
        expected_target_start_id=0,
        expected_target_end_id=8,
        expected_worker_ids=[0, 1],
    )

    result = reduce_mapreduce_target_range(inputs, n_latents_per_latent=2)

    # target 1 has candidates: 7 -> 4.0, 4 -> 4.0, 6 -> 3.0, 5 -> 3.0.
    # Equal values tie-break by lower candidate ID.
    assert result.top_indices[1].tolist() == [4, 7]
    assert result.top_values[1].tolist() == [4.0, 4.0]
    # target 2 keeps cross-range candidate IDs; candidate range is not sharded.
    assert result.top_indices[2].tolist() == [3, 7]
    assert result.top_values[2].tolist() == [4.0, 4.0]
    summed = list(
        zip(
            result.summed_target_ids.tolist(),
            result.summed_candidate_ids.tolist(),
            result.summed_values.tolist(),
        )
    )
    assert (1, 7, 4.0) in summed


def test_reduce_mapreduce_target_range_rejects_out_of_range_records():
    metadata = _preaggregation_metadata(worker_id=0, contribution_count=1, target_start_id=0, target_end_id=8)
    payload = {
        "target_ids": torch.tensor([7], dtype=torch.int64),
        "candidate_ids": torch.tensor([1], dtype=torch.int32),
        "values": torch.tensor([1.0], dtype=torch.float32),
        "sequence_ids": torch.tensor([1], dtype=torch.int64),
    }
    inputs = validate_candidate_preaggregation_reducer_inputs(
        [(metadata, payload)],
        expected_target_start_id=0,
        expected_target_end_id=8,
    )
    bad_inputs = inputs.__class__(
        entries=inputs.entries,
        mode=inputs.mode,
        m=inputs.m,
        num_components=inputs.num_components,
        d_sae=inputs.d_sae,
        target_start_id=0,
        target_end_id=4,
        total_sequence_count=inputs.total_sequence_count,
        total_contribution_count=inputs.total_contribution_count,
    )

    with pytest.raises(ValueError, match="outside reducer range"):
        reduce_mapreduce_target_range(bad_inputs, n_latents_per_latent=2)


def test_mapreduce_partial_sum_shard_round_trips_sorted_coo_with_metadata(tmp_path):
    metadata = _preaggregation_metadata(worker_id=3, contribution_count=4, target_start_id=0, target_end_id=8)
    payload = {
        "target_ids": torch.tensor([3, 1, 3, 1], dtype=torch.int64),
        "candidate_ids": torch.tensor([4, 9, 2, 5], dtype=torch.int32),
        "values": torch.tensor([4.0, 9.0, 2.0, 5.0], dtype=torch.float32),
        "sequence_ids": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
    }
    path = tmp_path / "worker_003.reducer_000.partial_sum.pt"

    save_mapreduce_partial_sum_shard(path, metadata, payload)
    loaded = load_mapreduce_partial_sum_shard(
        path,
        expected_config_hash="abcdef1234567890",
        expected_target_start_id=0,
        expected_target_end_id=8,
    )
    raw = torch.load(path, map_location="cpu", weights_only=False)

    assert loaded.metadata.worker_id == 3
    assert loaded.payload["target_ids"].tolist() == [1, 1, 3, 3]
    assert loaded.payload["candidate_ids"].tolist() == [5, 9, 2, 4]
    assert raw["storage_metadata"]["format"] == "sorted_coo"
    assert raw["storage_metadata"]["worker_id"] == 3
    assert raw["storage_metadata"]["row_count"] == 4
    assert raw["storage_metadata"]["candidate_count"] == 4
    assert raw["storage_metadata"]["value_dtype"] == "torch.float32"
    assert raw["storage_metadata"]["tensor_bytes"] == estimate_mapreduce_shard_tensor_bytes(4)


def test_mapreduce_partial_sum_shard_rejects_checksum_mismatch(tmp_path):
    metadata = _preaggregation_metadata(contribution_count=2, target_start_id=0, target_end_id=8)
    payload = _preaggregation_payload()
    path = tmp_path / "worker_000.reducer_000.partial_sum.pt"
    save_mapreduce_partial_sum_shard(path, metadata, payload)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    raw["payload"]["values"][0] = 999.0

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_mapreduce_partial_sum_shard(raw)


def test_mapreduce_reducer_memory_estimate_and_guardrail(tmp_path):
    metadata_0 = _preaggregation_metadata(worker_id=0, contribution_count=2, target_start_id=0, target_end_id=8)
    metadata_1 = _preaggregation_metadata(worker_id=1, contribution_count=2, target_start_id=0, target_end_id=8)
    path_0 = tmp_path / "worker_000.reducer_000.partial_sum.pt"
    path_1 = tmp_path / "worker_001.reducer_000.partial_sum.pt"
    save_mapreduce_partial_sum_shard(path_0, metadata_0, _preaggregation_payload())
    save_mapreduce_partial_sum_shard(path_1, metadata_1, _preaggregation_payload())

    entries = [
        load_mapreduce_partial_sum_shard(path_0),
        load_mapreduce_partial_sum_shard(path_1),
    ]
    estimate = estimate_mapreduce_reducer_input_bytes(entries=entries, guardrail_bytes=estimate_mapreduce_shard_tensor_bytes(4))
    assert estimate.shard_count == 2
    assert estimate.contribution_count == 4
    assert estimate.tensor_bytes == estimate_mapreduce_shard_tensor_bytes(4)
    assert estimate.exceeds_guardrail is False

    with pytest.raises(MemoryError, match="exceeds guardrail"):
        load_mapreduce_reducer_shards([path_0, path_1], guardrail_bytes=1)

    loaded = load_mapreduce_reducer_shards([path_0, path_1], guardrail_bytes=1, fail_on_guardrail=False)
    assert [entry.metadata.worker_id for entry in loaded] == [0, 1]


def test_reduce_mapreduce_target_range_chunked_matches_full_merge():
    metadata_0 = _preaggregation_metadata(worker_id=0, contribution_count=4, target_start_id=0, target_end_id=8)
    payload_0 = {
        "target_ids": torch.tensor([1, 1, 1, 2], dtype=torch.int64),
        "candidate_ids": torch.tensor([7, 6, 5, 7], dtype=torch.int32),
        "values": torch.tensor([1.0, 3.0, 3.0, 4.0], dtype=torch.float32),
        "sequence_ids": torch.tensor([1, 1, 2, 2], dtype=torch.int64),
    }
    metadata_1 = _preaggregation_metadata(worker_id=1, contribution_count=3, target_start_id=0, target_end_id=8)
    payload_1 = {
        "target_ids": torch.tensor([1, 1, 2], dtype=torch.int64),
        "candidate_ids": torch.tensor([7, 4, 3], dtype=torch.int32),
        "values": torch.tensor([3.0, 4.0, 4.0], dtype=torch.float32),
        "sequence_ids": torch.tensor([3, 3, 3], dtype=torch.int64),
    }
    inputs = validate_candidate_preaggregation_reducer_inputs(
        [(metadata_1, payload_1), (metadata_0, payload_0)],
        expected_target_start_id=0,
        expected_target_end_id=8,
        expected_worker_ids=[0, 1],
    )

    full = reduce_mapreduce_target_range(inputs, n_latents_per_latent=2)
    chunked = reduce_mapreduce_target_range(inputs, n_latents_per_latent=2, chunk_size=2)

    assert torch.equal(chunked.top_indices, full.top_indices)
    assert torch.equal(chunked.top_values, full.top_values)
    assert torch.equal(chunked.summed_target_ids, full.summed_target_ids)
    assert torch.equal(chunked.summed_candidate_ids, full.summed_candidate_ids)
    assert torch.equal(chunked.summed_values, full.summed_values)


def test_pass2_reduce_scheduler_config_validates_explicit_modes():
    config = validate_pass2_reduce_scheduler_config(
        Pass2ReduceSchedulerConfig(reducer_mode="mapreduce_target_ranges", reducer_count=2, backend="openmp")
    )
    assert config.reducer_count == 2

    with pytest.raises(ValueError, match="unknown pass-2 reducer_mode"):
        validate_pass2_reduce_scheduler_config(Pass2ReduceSchedulerConfig(reducer_mode="unknown"))

    with pytest.raises(NotImplementedError, match="parallel"):
        validate_pass2_reduce_scheduler_config(Pass2ReduceSchedulerConfig(execution_mode="parallel"))


def test_cleanup_mapreduce_target_shards_removes_only_reducer_outputs(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    stale = shard_dir / "reducer_0000_targets_00000000_00000008.pt"
    keep = shard_dir / "worker_000.reducer_000.partial_sum.pt"
    stale.write_bytes(b"stale")
    keep.write_bytes(b"keep")

    removed = cleanup_mapreduce_target_shards(shard_dir)

    assert removed == 1
    assert not stale.exists()
    assert keep.exists()


def test_stitch_mapreduce_target_shards_rejects_gaps():
    result = MapReduceTargetShardResult(
        target_range=TargetRange(reducer_id=1, target_start_id=4, target_end_id=8),
        top_indices=torch.zeros((4, 2), dtype=torch.int32),
        top_values=torch.zeros((4, 2), dtype=torch.float32),
        summed_target_ids=torch.empty(0, dtype=torch.int64),
        summed_candidate_ids=torch.empty(0, dtype=torch.int32),
        summed_values=torch.empty(0, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="contiguously"):
        stitch_mapreduce_target_shards([result], num_components=2, d_sae=8, n_latents_per_latent=2)


def test_run_mapreduce_reduce_and_write_stitches_canonical_artifact_and_resumes(tmp_path):
    shard_root = tmp_path / "input_shards"
    shard_root.mkdir()
    paths_by_reducer = {0: [], 1: []}
    payloads = {
        (0, 0): {
            "target_ids": torch.tensor([1, 1, 2], dtype=torch.int64),
            "candidate_ids": torch.tensor([7, 4, 3], dtype=torch.int32),
            "values": torch.tensor([1.0, 4.0, 2.0], dtype=torch.float32),
            "sequence_ids": torch.tensor([1, 1, 2], dtype=torch.int64),
        },
        (0, 1): {
            "target_ids": torch.tensor([1, 2], dtype=torch.int64),
            "candidate_ids": torch.tensor([7, 5], dtype=torch.int32),
            "values": torch.tensor([3.0, 5.0], dtype=torch.float32),
            "sequence_ids": torch.tensor([3, 3], dtype=torch.int64),
        },
        (1, 0): {
            "target_ids": torch.tensor([9, 9], dtype=torch.int64),
            "candidate_ids": torch.tensor([1, 2], dtype=torch.int32),
            "values": torch.tensor([2.0, 6.0], dtype=torch.float32),
            "sequence_ids": torch.tensor([1, 2], dtype=torch.int64),
        },
        (1, 1): {
            "target_ids": torch.tensor([9, 10], dtype=torch.int64),
            "candidate_ids": torch.tensor([1, 3], dtype=torch.int32),
            "values": torch.tensor([5.0, 7.0], dtype=torch.float32),
            "sequence_ids": torch.tensor([3, 3], dtype=torch.int64),
        },
    }
    for reducer_id, worker_id in payloads:
        target_start_id, target_end_id = (0, 8) if reducer_id == 0 else (8, 16)
        payload = payloads[(reducer_id, worker_id)]
        metadata = _preaggregation_metadata(
            worker_id=worker_id,
            contribution_count=int(payload["target_ids"].numel()),
            target_start_id=target_start_id,
            target_end_id=target_end_id,
        )
        path = shard_root / f"worker_{worker_id:03d}.reducer_{reducer_id:03d}.partial_sum.pt"
        save_mapreduce_partial_sum_shard(path, metadata, payload)
        paths_by_reducer[reducer_id].append(path)

    output_root = tmp_path / "run"
    result = run_mapreduce_reduce_and_write(
        paths_by_reducer,
        output_root,
        config=Pass2ReduceSchedulerConfig(reducer_count=2, cleanup=True, chunk_size=2),
        num_components=2,
        d_sae=8,
        n_latents_per_latent=2,
        mode="raw",
        total_tokens_processed=128,
        expected_config_hash="abcdef1234567890",
        expected_worker_ids=[0, 1],
    )
    artifact = torch.load(result.artifact_path, map_location="cpu", weights_only=False)

    assert result.artifact_path == output_root / "top_coactivation.pt"
    assert artifact["top_indices"].shape == (2, 8, 2)
    assert artifact["top_values"].shape == (2, 8, 2)
    assert artifact["top_indices"][0, 1].tolist() == [4, 7]
    assert artifact["top_values"][0, 1].tolist() == [4.0, 4.0]
    assert artifact["top_indices"][1, 1].tolist() == [1, 2]
    assert artifact["top_values"][1, 1].tolist() == [7.0, 6.0]
    assert artifact["total_tokens_processed"] == 128
    assert result.report["shards_written"] == 2
    assert result.report["shards_reused"] == 0
    assert result.report["input_partial_sum_bytes"] > 0
    assert result.report["output_shard_bytes"] > 0
    assert result.report["output_artifact_size_bytes"] == result.artifact_path.stat().st_size
    assert result.report["peak_cpu_memory_bytes"] is None or result.report["peak_cpu_memory_bytes"] > 0
    assert set(result.report["timing"]) >= {
        "load_shards_s",
        "reduce_s",
        "shard_write_s",
        "shard_load_s",
        "stitch_s",
        "pmi_s",
        "save_s",
        "total_s",
    }
    assert result.report["timing"]["total_s"] >= result.report["timing"]["save_s"]
    assert result.report["manifest_metrics"]["reducer_mode"] == "mapreduce_target_ranges"
    assert result.report["manifest_metrics"]["input_bytes"] == result.report["input_partial_sum_bytes"]

    resumed = run_mapreduce_reduce_and_write(
        paths_by_reducer,
        output_root,
        config=Pass2ReduceSchedulerConfig(reducer_count=2, resume=True),
        num_components=2,
        d_sae=8,
        n_latents_per_latent=2,
        mode="raw",
        expected_config_hash="abcdef1234567890",
        expected_worker_ids=[0, 1],
    )

    assert resumed.report["shards_written"] == 0
    assert resumed.report["shards_reused"] == 2


def test_run_mapreduce_reduce_and_write_rejects_stale_resume_shard(tmp_path):
    shard_root = tmp_path / "input_shards"
    shard_root.mkdir()
    output_root = tmp_path / "run"
    payload = _preaggregation_payload(target_ids=torch.tensor([1], dtype=torch.int64))
    payload["candidate_ids"] = torch.tensor([2], dtype=torch.int32)
    payload["values"] = torch.tensor([1.0], dtype=torch.float32)
    payload["sequence_ids"] = torch.tensor([1], dtype=torch.int64)
    input_path = shard_root / "worker_000.reducer_000.partial_sum.pt"
    save_mapreduce_partial_sum_shard(
        input_path,
        _preaggregation_metadata(worker_id=0, contribution_count=1, target_start_id=0, target_end_id=16),
        payload,
    )
    paths_by_reducer = {0: [input_path]}
    run_mapreduce_reduce_and_write(
        paths_by_reducer,
        output_root,
        config=Pass2ReduceSchedulerConfig(reducer_count=1),
        num_components=2,
        d_sae=8,
        n_latents_per_latent=2,
        mode="raw",
        expected_config_hash="abcdef1234567890",
        expected_worker_ids=[0],
    )

    with pytest.raises(ValueError, match="config hash mismatch"):
        run_mapreduce_reduce_and_write(
            paths_by_reducer,
            output_root,
            config=Pass2ReduceSchedulerConfig(reducer_count=1, resume=True),
            num_components=2,
            d_sae=8,
            n_latents_per_latent=2,
            mode="raw",
            expected_config_hash="stale",
            expected_worker_ids=[0],
        )


@pytest.mark.parametrize("mode", ["raw", "freq_weighted", "pmi"])
@pytest.mark.parametrize("worker_sequence_groups", [[[1, 2, 3]], [[1, 2], [3]], [[1], [2], [3]]])
def test_phase10_single_mode_a_and_mapreduce_mode_b_equivalence(mode, worker_sequence_groups):
    rows = _equivalence_candidate_rows()
    top_k = 2
    active_count = torch.full((2, 8), 8, dtype=torch.int64)
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload())
    single_sequence_ids = [sequence_id for group in worker_sequence_groups for sequence_id in group]
    single_metadata = _candidate_metadata(
        worker_id=99,
        sequence_ids=single_sequence_ids,
        mode=mode,
        m=4,
        n_latents_per_latent=top_k,
    )
    single_payload = _candidate_payload_from_rows(single_sequence_ids, rows)
    single_top_indices, single_top_values = _reduce_candidate_dump_fixture(
        single_metadata,
        single_payload,
        mapping,
        top_k=top_k,
        active_count=active_count,
    )

    distributed_entries = []
    for worker_id, sequence_ids in enumerate(worker_sequence_groups):
        distributed_entries.append(
            (
                _candidate_metadata(
                    worker_id=worker_id,
                    sequence_ids=sequence_ids,
                    mode=mode,
                    m=4,
                    n_latents_per_latent=top_k,
                ),
                _candidate_payload_from_rows(sequence_ids, rows),
            )
        )
    dump_inputs = validate_candidate_dump_reducer_inputs(distributed_entries, expected_mode=mode)
    mapping_with_coverage = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)
    mode_a_dump = build_simple_exact_candidate_dump(dump_inputs, mapping_with_coverage)
    mode_a_metadata = _candidate_metadata(
        worker_id=100,
        sequence_ids=mode_a_dump.sequence_ids.tolist(),
        mode=mode,
        m=mode_a_dump.m,
        n_latents_per_latent=top_k,
    )
    mode_a_payload = {
        "sequence_ids": mode_a_dump.sequence_ids,
        "candidate_ids": mode_a_dump.candidate_ids,
        "candidate_vals": mode_a_dump.candidate_vals,
        "total_tokens_processed": mode_a_dump.total_token_count,
    }
    mode_a_top_indices, mode_a_top_values = _reduce_candidate_dump_fixture(
        mode_a_metadata,
        mode_a_payload,
        mapping_with_coverage,
        top_k=top_k,
        active_count=active_count,
    )
    mode_b_top_indices, mode_b_top_values = _reduce_mapreduce_fixture(
        distributed_entries,
        mapping_with_coverage,
        top_k=top_k,
        active_count=active_count,
    )

    assert torch.equal(mode_a_top_indices, single_top_indices)
    assert torch.allclose(mode_a_top_values, single_top_values, atol=1e-6, rtol=0.0)
    assert torch.equal(mode_b_top_indices, mode_a_top_indices)
    assert torch.allclose(mode_b_top_values, mode_a_top_values, atol=1e-6, rtol=0.0)
    # Target 2 has candidates 7 and 4 with equal values; lower candidate ID wins.
    assert mode_b_top_indices[0, 2].tolist() == [4, 7]


def test_phase10_mapreduce_keeps_candidate_not_local_topk_but_global_topk():
    metadata_0 = _preaggregation_metadata(worker_id=0, contribution_count=3, target_start_id=0, target_end_id=8)
    payload_0 = {
        "target_ids": torch.tensor([1, 1, 1], dtype=torch.int64),
        "candidate_ids": torch.tensor([2, 4, 3], dtype=torch.int32),
        "values": torch.tensor([10.0, 6.0, 5.0], dtype=torch.float32),
        "sequence_ids": torch.tensor([1, 1, 1], dtype=torch.int64),
    }
    metadata_1 = _preaggregation_metadata(worker_id=1, contribution_count=3, target_start_id=0, target_end_id=8)
    payload_1 = {
        "target_ids": torch.tensor([1, 1, 1], dtype=torch.int64),
        "candidate_ids": torch.tensor([5, 4, 6], dtype=torch.int32),
        "values": torch.tensor([9.0, 6.0, 4.0], dtype=torch.float32),
        "sequence_ids": torch.tensor([2, 2, 2], dtype=torch.int64),
    }
    inputs = validate_candidate_preaggregation_reducer_inputs(
        [(metadata_0, payload_0), (metadata_1, payload_1)],
        expected_worker_ids=[0, 1],
    )

    exact = reduce_mapreduce_target_range(inputs, n_latents_per_latent=1)
    naive_local_top1 = max([(2, 10.0), (5, 9.0)], key=lambda item: (item[1], -item[0]))

    assert exact.top_indices[1].tolist() == [4]
    assert exact.top_values[1].tolist() == [12.0]
    assert naive_local_top1 == (2, 10.0)


def test_phase12_file_backed_synthetic_pass2_dump_to_mapreduce_reduce_equivalence(tmp_path):
    rows = _equivalence_candidate_rows()
    top_k = 2
    sequence_groups = {0: [1, 2], 1: [3]}
    dump_paths = []
    for worker_id, sequence_ids in sequence_groups.items():
        path = tmp_path / "candidate_dumps" / f"worker_{worker_id:03d}.candidate_dump.partial.pt"
        save_candidate_dump_partial(
            path,
            _candidate_metadata(
                worker_id=worker_id,
                sequence_ids=sequence_ids,
                m=4,
                n_latents_per_latent=top_k,
            ),
            _candidate_payload_from_rows(sequence_ids, rows),
        )
        dump_paths.append(path)

    dump_inputs = load_candidate_dump_reducer_inputs(dump_paths, expected_config_hash="abcdef1234567890")
    mapping = build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)
    simple_dump = build_simple_exact_candidate_dump(dump_inputs, mapping)
    simple_metadata = _candidate_metadata(
        worker_id=99,
        sequence_ids=simple_dump.sequence_ids.tolist(),
        m=simple_dump.m,
        n_latents_per_latent=top_k,
    )
    simple_payload = {
        "sequence_ids": simple_dump.sequence_ids,
        "candidate_ids": simple_dump.candidate_ids,
        "candidate_vals": simple_dump.candidate_vals,
        "total_tokens_processed": simple_dump.total_token_count,
    }
    expected_indices, expected_values = _reduce_candidate_dump_fixture(
        simple_metadata,
        simple_payload,
        mapping,
        top_k=top_k,
    )

    target_ranges = partition_target_ranges(16, reducer_count=2)
    paths_by_reducer = {target_range.reducer_id: [] for target_range in target_ranges}
    for entry in dump_inputs.entries:
        pre_metadata, pre_payload = expand_candidate_dump_to_contributions(
            entry.metadata,
            entry.payload,
            mapping.seq_offsets,
            mapping.seq_targets_global,
        )
        for target_range, shard in zip(
            target_ranges,
            shard_preaggregation_by_target_range(pre_metadata, pre_payload, target_ranges),
        ):
            path = (
                tmp_path
                / "partial_sums"
                / f"worker_{entry.metadata.worker_id:03d}.reducer_{target_range.reducer_id:03d}.partial_sum.pt"
            )
            save_mapreduce_partial_sum_shard(path, shard.metadata, shard.payload)
            paths_by_reducer[target_range.reducer_id].append(path)

    result = run_mapreduce_reduce_and_write(
        paths_by_reducer,
        tmp_path / "outputs" / "synthetic-run",
        config=Pass2ReduceSchedulerConfig(reducer_count=2, cleanup=True, chunk_size=1),
        num_components=2,
        d_sae=8,
        n_latents_per_latent=top_k,
        mode="raw",
        total_tokens_processed=simple_dump.total_token_count,
        expected_config_hash="abcdef1234567890",
        expected_worker_ids=[0, 1],
    )
    artifact = torch.load(result.artifact_path, map_location="cpu", weights_only=False)

    assert result.artifact_path == tmp_path / "outputs" / "synthetic-run" / "top_coactivation.pt"
    assert torch.equal(artifact["top_indices"], expected_indices)
    assert torch.allclose(artifact["top_values"], expected_values, atol=1e-6, rtol=0.0)
    assert result.report["shards_written"] == 2
    assert result.report["output_artifact_size_bytes"] == result.artifact_path.stat().st_size


def test_global_top_ctx_mapping_rejects_duplicate_dump_sequences():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [
            (_candidate_metadata(worker_id=0, sequence_ids=[1, 2]), _candidate_payload([1, 2])),
            (_candidate_metadata(worker_id=1, sequence_ids=[2, 3]), _candidate_payload([2, 3])),
        ]
    )

    with pytest.raises(ValueError, match="duplicate sequence IDs"):
        build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)


def test_global_top_ctx_mapping_rejects_missing_replay_sequences():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [(_candidate_metadata(worker_id=0, sequence_ids=[1, 2]), _candidate_payload([1, 2]))]
    )

    with pytest.raises(ValueError, match="missing replay sequence IDs"):
        build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)


def test_global_top_ctx_mapping_rejects_extra_dump_sequences():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [(_candidate_metadata(worker_id=0, sequence_ids=[1, 2, 4]), _candidate_payload([1, 2, 4]))]
    )

    with pytest.raises(ValueError, match="outside global replay set"):
        build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)


def test_global_top_ctx_mapping_rejects_zero_dump_sequence_id():
    dump_inputs = validate_candidate_dump_reducer_inputs(
        [(_candidate_metadata(worker_id=0, sequence_ids=[0, 1, 2, 3]), _candidate_payload([0, 1, 2, 3]))]
    )

    with pytest.raises(ValueError, match="sentinel sequence ID 0"):
        build_global_top_ctx_target_mapping(_top_ctx_payload(), dump_inputs=dump_inputs)


def test_candidate_dump_reducer_rejects_unsorted_worker_rows():
    with pytest.raises(ValueError, match="sorted replay order"):
        validate_candidate_dump_reducer_inputs(
            [(_candidate_metadata(worker_id=0, sequence_ids=[1, 2]), _candidate_payload([2, 1]))]
        )


def test_candidate_dump_reducer_rejects_stale_config():
    with pytest.raises(ValueError, match="config hash mismatch"):
        validate_candidate_dump_reducer_inputs(
            [(_candidate_metadata(), _candidate_payload())],
            expected_config_hash="deadbeef",
        )


def test_candidate_dump_reducer_rejects_wrong_mode():
    with pytest.raises(ValueError, match="mode mismatch"):
        validate_candidate_dump_reducer_inputs(
            [
                (_candidate_metadata(worker_id=0), _candidate_payload()),
                (_candidate_metadata(worker_id=1, mode="pmi"), _candidate_payload()),
            ]
        )


def test_candidate_dump_reducer_rejects_wrong_shape():
    payload = _candidate_payload()
    payload["candidate_vals"] = torch.zeros((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="unexpected shape"):
        validate_candidate_dump_reducer_inputs([(_candidate_metadata(), payload)])


def test_candidate_dump_reducer_rejects_invalid_candidate_ids():
    payload = _candidate_payload(
        candidate_ids=torch.tensor([[1, 16, 0], [4, 5, 0]], dtype=torch.int32)
    )

    with pytest.raises(ValueError, match="candidate_ids out of range"):
        validate_candidate_dump_reducer_inputs([(_candidate_metadata(), payload)])


def test_candidate_dump_reducer_rejects_non_finite_values():
    payload = _candidate_payload(
        candidate_vals=torch.tensor([[1.0, float("inf"), 0.0], [2.0, 0.25, 0.0]], dtype=torch.float32)
    )

    with pytest.raises(ValueError, match="candidate_vals must be finite"):
        validate_candidate_dump_reducer_inputs([(_candidate_metadata(), payload)])


def test_preaggregation_reducer_inputs_accept_one_target_range(tmp_path):
    path_0 = tmp_path / "worker_000_preagg.pt"
    path_1 = tmp_path / "worker_001_preagg.pt"
    save_candidate_preaggregation_partial(path_0, _preaggregation_metadata(0), _preaggregation_payload())
    save_candidate_preaggregation_partial(path_1, _preaggregation_metadata(1), _preaggregation_payload())

    inputs = load_candidate_preaggregation_reducer_inputs(
        [path_1, path_0],
        expected_config_hash="abcdef1234567890",
        expected_mode="raw",
        expected_target_start_id=0,
        expected_target_end_id=8,
    )

    assert [entry.metadata.worker_id for entry in inputs.entries] == [0, 1]
    assert inputs.target_start_id == 0
    assert inputs.target_end_id == 8
    assert inputs.total_contribution_count == 4


def test_preaggregation_schema_rejects_target_ids_outside_range():
    metadata = _preaggregation_metadata(target_start_id=4, target_end_id=8)

    with pytest.raises(ValueError, match="outside reducer target range"):
        save_candidate_preaggregation_partial(
            "unused.pt",
            metadata,
            _preaggregation_payload(target_ids=torch.tensor([3, 6], dtype=torch.int64)),
        )
