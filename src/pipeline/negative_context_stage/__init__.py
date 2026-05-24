"""Package entrypoint for the negative-context pipeline stage."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .cli import configured_neg_ctx_sequences, main
from .comparison import (
    NegativeContextComparisonResult,
    build_negative_context_comparison_report,
    compare_negative_context_backends,
)
from .inputs import (
    BuildNegCtxFn,
    LoadedContext,
    LoadedSeqRepr,
    NegativeContextInputs,
    SeqReprLike,
    load_negative_context_inputs,
)
from .planning import (
    NegativeContextStageClassification,
    NegativeContextStagePlan,
    build_negative_context_stage_metadata,
    classify_negative_context_stage,
    plan_negative_context_stage,
)
from .reports import (
    build_negative_context_sanity_report,
    print_negative_context_sanity_summary,
)
from .runtime import build_negative_contexts
from .stage import NegativeContextRunResult, run_negative_context_stage

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


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("pipeline.negative_context"), name)
