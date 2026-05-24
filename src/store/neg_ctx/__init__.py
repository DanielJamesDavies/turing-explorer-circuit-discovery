"""Package entrypoint for negative-context store helpers."""

from __future__ import annotations

from .ann import (
    TorchANNIndex,
    _record_ann_memory_estimate,
    check_neg_ctx_memory_guardrail,
    estimate_neg_ctx_ann_memory,
    estimate_neg_ctx_ann_memory_for_shape,
)
from .backends import (
    build_neg_ctx,
    build_neg_ctx_index_sharded,
    build_neg_ctx_multi_gpu,
    build_neg_ctx_single_gpu_exact,
)
from .component import _PAIR_CHUNK, _process_component, _process_component_sharded
from .devices import (
    _ann_device,
    _validate_cuda_devices,
    parse_neg_ctx_devices,
    partition_components,
)
from .sharded_ann import (
    ANNIndexShard,
    ShardedANNIndex,
    merge_shard_search_results,
    partition_index_slots,
)
from .stats import NegCtxStats
from .validation import validate_neg_ctx_output

__all__ = [
    "ANNIndexShard",
    "NegCtxStats",
    "ShardedANNIndex",
    "TorchANNIndex",
    "_PAIR_CHUNK",
    "_ann_device",
    "_process_component",
    "_process_component_sharded",
    "_record_ann_memory_estimate",
    "_validate_cuda_devices",
    "build_neg_ctx",
    "build_neg_ctx_index_sharded",
    "build_neg_ctx_multi_gpu",
    "build_neg_ctx_single_gpu_exact",
    "check_neg_ctx_memory_guardrail",
    "estimate_neg_ctx_ann_memory",
    "estimate_neg_ctx_ann_memory_for_shape",
    "merge_shard_search_results",
    "parse_neg_ctx_devices",
    "partition_components",
    "partition_index_slots",
    "validate_neg_ctx_output",
]
