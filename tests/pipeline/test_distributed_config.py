import pytest
import yaml
from pydantic import ValidationError

from config import RootConfig


def _base_config(**overrides):
    data = {
        "weights": {
            "model_path": "weights/model.pt",
            "sae_path": "weights/sae",
        },
        "data": {
            "dataset_path": "data",
        },
    }
    data.update(overrides)
    return data


def test_default_local_config_keeps_single_process_defaults():
    parsed = RootConfig.model_validate(_base_config())

    assert parsed.distributed.mode == "single_process"
    assert parsed.distributed.worker_count == 1
    assert parsed.distributed.output_base == "outputs"
    assert parsed.distributed.launch_strategy == "manual_commands"
    assert parsed.distributed.cleanup_policy == "keep_all"
    assert parsed.distributed.strict_equivalence is True
    assert parsed.distributed.observability_sample_interval_s == 30.0
    assert parsed.distributed.schema_versions.manifest == 1
    assert parsed.distributed.mid_ctx_candidate_pool.enabled is True
    assert parsed.distributed.mid_ctx_candidate_pool.band_margin_sigma == 1.0
    assert parsed.distributed.mid_ctx_candidate_pool.on_truncation == "replay_fallback"
    assert parsed.persist.build_search_cache_after_pipeline is True


def test_one_worker_distributed_config_defaults_search_cache_to_deferred():
    parsed = RootConfig.model_validate(
        _base_config(
            distributed={
                "mode": "distributed_simple_exact",
                "worker_count": 1,
                "sampling_seed": 123,
            }
        )
    )

    assert parsed.distributed.mode == "distributed_simple_exact"
    assert parsed.distributed.worker_count == 1
    assert parsed.distributed.sampling_seed == 123
    assert parsed.persist.build_search_cache_after_pipeline is False


def test_local_distributed_smoke_example_config_is_valid():
    with open("config_examples/local-distributed-smoke.yaml", encoding="utf-8") as handle:
        parsed = RootConfig.model_validate(yaml.safe_load(handle))

    assert parsed.distributed.mode == "distributed_simple_exact"
    assert parsed.distributed.worker_count == 1
    assert parsed.distributed.devices == [0]
    assert parsed.hardware.memory == "efficient"
    assert parsed.hardware.keep_model_loaded_for_neg_ctx is False
    assert parsed.data.n_shards == 4
    assert parsed.discovery.n_seeds == 16
    assert parsed.discovery.probe_batch_size == 4
    assert parsed.persist.build_search_cache_after_pipeline is False


def test_distributed_config_allows_explicit_search_cache_override():
    parsed = RootConfig.model_validate(
        _base_config(
            distributed={
                "mode": "distributed_simple_exact",
                "worker_count": 1,
            },
            persist={
                "build_search_cache_after_pipeline": True,
            },
        )
    )

    assert parsed.persist.build_search_cache_after_pipeline is True


def test_h100_simple_exact_config_validates_devices_and_runtime_fields():
    parsed = RootConfig.model_validate(
        _base_config(
            distributed={
                "mode": "distributed_simple_exact",
                "run_id": "20260517-002500-abcdef12",
                "output_base": "outputs",
                "worker_count": 8,
                "devices": list(range(8)),
                "launch_strategy": "manual_commands",
                "resume_policy": "fresh",
                "cleanup_policy": "delete_large_partials_on_success",
                "parts": ["pass1", "neg_ctx", "pass2", "discovery"],
                "strict_equivalence": True,
            }
        )
    )

    assert parsed.distributed.run_id == "20260517-002500-abcdef12"
    assert parsed.distributed.worker_count == 8
    assert parsed.distributed.devices == list(range(8))
    assert parsed.distributed.parts == ["pass1", "neg_ctx", "pass2", "discovery"]
    assert parsed.distributed.cleanup_policy == "delete_large_partials_on_success"


def test_distributed_h100_example_config_is_valid_and_one_worker_per_gpu():
    with open("config_examples/h100-8x-distributed-simple-exact.yaml", encoding="utf-8") as handle:
        parsed = RootConfig.model_validate(yaml.safe_load(handle))

    assert parsed.distributed.mode == "distributed_simple_exact"
    assert parsed.distributed.worker_count == 8
    assert parsed.distributed.devices == list(range(8))
    assert parsed.hardware.memory == "fast"
    assert parsed.hardware.multi_gpu is False
    assert parsed.latents.neg_ctx.backend == "multi_gpu_exact"
    assert parsed.latents.neg_ctx.devices == list(range(8))
    assert parsed.latents.top_coactivation.reduce_backend == "single_process"
    assert parsed.persist.build_search_cache_after_pipeline is False


def test_mapreduce_exact_config_is_allowed_after_mode_b_contracts_exist():
    parsed = RootConfig.model_validate(
        _base_config(
            distributed={
                "mode": "distributed_mapreduce_exact",
                "worker_count": 2,
                "devices": [0, 1],
                "schema_versions": {
                    "manifest": 1,
                    "partial_artifacts": 1,
                    "metrics_jsonl": 1,
                    "sanity_reports": 1,
                    "run_summaries": 1,
                },
            }
        )
    )

    assert parsed.distributed.mode == "distributed_mapreduce_exact"
    assert parsed.distributed.schema_versions.partial_artifacts == 1


def test_distributed_observability_sample_interval_is_configurable():
    parsed = RootConfig.model_validate(
        _base_config(
            distributed={
                "mode": "distributed_simple_exact",
                "worker_count": 1,
                "observability_sample_interval_s": 10.0,
            }
        )
    )

    assert parsed.distributed.observability_sample_interval_s == 10.0

    with pytest.raises(ValidationError, match="observability_sample_interval_s"):
        RootConfig.model_validate(
            _base_config(
                distributed={
                    "mode": "distributed_simple_exact",
                    "worker_count": 1,
                    "observability_sample_interval_s": 0,
                }
            )
        )


def test_experimental_fast_requires_explicit_acknowledgement():
    with pytest.raises(ValidationError, match="experimental_acknowledgement"):
        RootConfig.model_validate(
            _base_config(
                distributed={
                    "mode": "distributed_experimental_fast",
                    "worker_count": 1,
                    "output_base": "outputs/experimental_fast",
                    "experimental_exact_baseline_root": "outputs/exact-baseline",
                    "experimental_quality_toggles": {
                        "local_topk_merge": True,
                    },
                }
            )
        )

    parsed = RootConfig.model_validate(
        _base_config(
            distributed={
                "mode": "distributed_experimental_fast",
                "worker_count": 1,
                "output_base": "outputs/experimental_fast",
                "experimental_acknowledgement": True,
                "experimental_exact_baseline_root": "outputs/exact-baseline",
                "experimental_quality_toggles": {
                    "local_topk_merge": True,
                },
                "mid_ctx_candidate_pool": {
                    "on_truncation": "allow_bounded_approx",
                },
            }
        )
    )
    assert parsed.distributed.experimental_acknowledgement is True
    assert parsed.distributed.experimental_exact_baseline_root == "outputs/exact-baseline"
    assert parsed.distributed.experimental_quality_toggles == {"local_topk_merge": True}


def test_experimental_fast_requires_baseline_toggles_and_marked_output_base():
    invalid_configs = [
        {
            "mode": "distributed_experimental_fast",
            "worker_count": 1,
            "output_base": "outputs/experimental_fast",
            "experimental_acknowledgement": True,
            "experimental_quality_toggles": {"local_topk_merge": True},
        },
        {
            "mode": "distributed_experimental_fast",
            "worker_count": 1,
            "output_base": "outputs/experimental_fast",
            "experimental_acknowledgement": True,
            "experimental_exact_baseline_root": "outputs/exact-baseline",
        },
        {
            "mode": "distributed_experimental_fast",
            "worker_count": 1,
            "output_base": "outputs",
            "experimental_acknowledgement": True,
            "experimental_exact_baseline_root": "outputs/exact-baseline",
            "experimental_quality_toggles": {"local_topk_merge": True},
        },
    ]

    for distributed_config in invalid_configs:
        with pytest.raises(ValidationError):
            RootConfig.model_validate(_base_config(distributed=distributed_config))


def test_invalid_distributed_config_combinations_fail():
    invalid_configs = [
        {"mode": "single_process", "worker_count": 2},
        {"mode": "single_process", "devices": [0]},
        {"mode": "distributed_simple_exact", "worker_count": 2, "devices": [0]},
        {"mode": "distributed_simple_exact", "worker_count": 2, "devices": [0, 0]},
        {
            "mode": "distributed_simple_exact",
            "worker_count": 1,
            "mid_ctx_candidate_pool": {"on_truncation": "allow_bounded_approx"},
        },
        {"mode": "distributed_simple_exact", "worker_count": 1, "experimental_acknowledgement": True},
    ]

    for distributed_config in invalid_configs:
        with pytest.raises(ValidationError):
            RootConfig.model_validate(_base_config(distributed=distributed_config))


def test_strict_distributed_config_rejects_unknown_keys_and_schema_versions():
    with pytest.raises(ValidationError):
        RootConfig.model_validate(
            _base_config(
                distributed={
                    "mode": "distributed_simple_exact",
                    "worker_count": 1,
                    "unknown_key": True,
                }
            )
        )

    with pytest.raises(ValidationError, match="schema versions"):
        RootConfig.model_validate(
            _base_config(
                distributed={
                    "mode": "distributed_simple_exact",
                    "worker_count": 1,
                    "schema_versions": {"manifest": 2},
                }
            )
        )
