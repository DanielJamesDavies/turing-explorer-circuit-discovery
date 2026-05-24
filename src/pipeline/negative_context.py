from __future__ import annotations

import torch

from .distributed.interfaces import build_output_paths
from .runtime import get_runtime
from config import config
from store.context import mid_ctx, neg_ctx, top_ctx
from store.neg_context import (
    NegCtxStats,
    build_neg_ctx,
    validate_neg_ctx_output,
)
from .negative_context_stage import (
    BuildNegCtxFn,
    LoadedContext,
    LoadedSeqRepr,
    NegativeContextComparisonResult,
    NegativeContextInputs,
    NegativeContextRunResult,
    NegativeContextStageClassification,
    NegativeContextStagePlan,
    SeqReprLike,
    build_negative_context_comparison_report,
    build_negative_context_sanity_report,
    build_negative_context_stage_metadata,
    build_negative_contexts,
    classify_negative_context_stage,
    compare_negative_context_backends,
    configured_neg_ctx_sequences,
    load_negative_context_inputs,
    main,
    plan_negative_context_stage,
    print_negative_context_sanity_summary,
    run_negative_context_stage,
)
from .negative_context_stage.comparison import _sample_row_comparisons
from .negative_context_stage.inputs import (
    _context_from_payload,
    _empty_neg_context_like,
    _load_torch_payload,
    _require_artifacts,
    _seq_repr_from_payload,
    _validate_config_hash_if_present,
    _validate_negative_context_inputs,
    _validate_seq_repr_cap_mapping,
)
from .negative_context_stage.planning import (
    _artifact_metadata,
    _manifest_neg_ctx_config,
    _manifest_neg_ctx_devices,
    _manifest_neg_ctx_devices_from_manifest,
    _neg_ctx_part_dir,
)
from .negative_context_stage.reports import (
    _atomic_write_json,
    _fill_summary,
    _neg_ctx_validation_summary,
    _populated_row_count,
    _stats_timing_ms,
)
from .negative_context_stage.stage import _write_part_marker

__all__ = [
    "BuildNegCtxFn",
    "LoadedContext",
    "LoadedSeqRepr",
    "NegativeContextComparisonResult",
    "NegativeContextInputs",
    "NegativeContextRunResult",
    "NegativeContextStageClassification",
    "NegativeContextStagePlan",
    "SeqReprLike",
    "build_negative_context_comparison_report",
    "build_negative_context_sanity_report",
    "build_negative_context_stage_metadata",
    "build_negative_contexts",
    "classify_negative_context_stage",
    "compare_negative_context_backends",
    "configured_neg_ctx_sequences",
    "load_negative_context_inputs",
    "main",
    "plan_negative_context_stage",
    "print_negative_context_sanity_summary",
    "run_negative_context_stage",
]


if __name__ == "__main__":
    main()
