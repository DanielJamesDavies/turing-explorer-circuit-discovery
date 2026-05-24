"""Backend comparison for negative-context implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import torch

from pipeline.distributed.manifest import load_manifest
from store.neg_context import (
    NegCtxStats,
    build_neg_ctx_multi_gpu,
    build_neg_ctx_single_gpu_exact,
)

from .inputs import BuildNegCtxFn, LoadedContext, _empty_neg_context_like, load_negative_context_inputs
from .planning import _manifest_neg_ctx_devices
from .reports import _atomic_write_json, _fill_summary, _populated_row_count, _stats_timing_ms


@dataclass(frozen=True)
class NegativeContextComparisonResult:
    report_path: Path
    report: Dict[str, object]


def compare_negative_context_backends(
    output_root: str | Path = "outputs",
    *,
    expected_config_hash: Optional[str] = None,
    manifest_path: str | Path | None = None,
    selected_devices: Sequence[int | str | torch.device] | None = None,
    single_build_fn: BuildNegCtxFn = build_neg_ctx_single_gpu_exact,
    multi_build_fn: Callable[..., NegCtxStats] = build_neg_ctx_multi_gpu,
    sample_rows: int = 8,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> NegativeContextComparisonResult:
    """Build both exact neg_ctx backends from one input set and compare outputs."""

    inputs = load_negative_context_inputs(
        output_root,
        expected_config_hash=expected_config_hash
        or (load_manifest(manifest_path).normalized_config_hash if manifest_path is not None else None),
    )
    devices = (
        list(selected_devices)
        if selected_devices is not None
        else _manifest_neg_ctx_devices(manifest_path)
    )
    single_output = _empty_neg_context_like(inputs.top_ctx)
    multi_output = _empty_neg_context_like(inputs.top_ctx)
    single_stats = single_build_fn(
        inputs.seq_repr,
        inputs.top_ctx,
        inputs.mid_ctx,
        single_output,
    )
    multi_stats = multi_build_fn(
        inputs.seq_repr,
        inputs.top_ctx,
        inputs.mid_ctx,
        multi_output,
        selected_devices=devices,
    )
    report = build_negative_context_comparison_report(
        single_output,
        multi_output,
        single_stats=single_stats,
        multi_stats=multi_stats,
        sample_rows=sample_rows,
        atol=atol,
        rtol=rtol,
    )
    report_path = inputs.paths.run_root / "neg_ctx_equivalence_report.json"
    _atomic_write_json(report_path, report)
    return NegativeContextComparisonResult(report_path=report_path, report=report)


def build_negative_context_comparison_report(
    single_output: LoadedContext,
    multi_output: LoadedContext,
    *,
    single_stats: NegCtxStats,
    multi_stats: NegCtxStats,
    sample_rows: int = 8,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Dict[str, object]:
    """Compare two exact neg_ctx artifacts and summarize quality/timing."""

    shape_match = single_output.ctx_seq_idx.shape == multi_output.ctx_seq_idx.shape
    dtype_match = (
        single_output.ctx_seq_idx.dtype == multi_output.ctx_seq_idx.dtype
        and single_output.ctx_seq_val.dtype == multi_output.ctx_seq_val.dtype
    )
    indices_equal = torch.equal(single_output.ctx_seq_idx, multi_output.ctx_seq_idx)
    values_exact = torch.equal(single_output.ctx_seq_val, multi_output.ctx_seq_val)
    values_close = shape_match and torch.allclose(
        single_output.ctx_seq_val.float(),
        multi_output.ctx_seq_val.float(),
        atol=atol,
        rtol=rtol,
    )
    max_abs_diff = 0.0
    if shape_match and single_output.ctx_seq_val.numel():
        max_abs_diff = float(
            (single_output.ctx_seq_val.float() - multi_output.ctx_seq_val.float())
            .abs()
            .max()
            .item()
        )
    single_fill = _fill_summary(single_output)
    multi_fill = _fill_summary(multi_output)
    populated_single = _populated_row_count(single_output)
    populated_multi = _populated_row_count(multi_output)
    exact_equivalent = bool(
        shape_match
        and dtype_match
        and indices_equal
        and values_close
        and populated_single == populated_multi
        and single_fill == multi_fill
    )
    return {
        "schema_version": 1,
        "status": "equivalent" if exact_equivalent else "different",
        "exact_equivalent": exact_equivalent,
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "indices_equal": indices_equal,
        "values_exact": values_exact,
        "values_close": values_close,
        "value_tolerance": {"atol": float(atol), "rtol": float(rtol)},
        "max_abs_value_diff": max_abs_diff,
        "populated_rows": {
            "single_gpu_exact": populated_single,
            "multi_gpu_exact": populated_multi,
            "match": populated_single == populated_multi,
        },
        "fill_distribution": {
            "single_gpu_exact": single_fill,
            "multi_gpu_exact": multi_fill,
            "match": single_fill == multi_fill,
        },
        "sample_rows": _sample_row_comparisons(
            single_output,
            multi_output,
            limit=sample_rows,
        ),
        "timing_ms": {
            "single_gpu_exact": _stats_timing_ms(single_stats),
            "multi_gpu_exact": _stats_timing_ms(multi_stats),
        },
        "ordering_or_tie_note": (
            "No ordering/tie differences detected."
            if indices_equal and values_close
            else "Differences may indicate backend drift or device-level tie ordering; inspect sample_rows."
        ),
    }


def _sample_row_comparisons(
    single_output: LoadedContext,
    multi_output: LoadedContext,
    *,
    limit: int,
) -> list[Dict[str, object]]:
    populated = torch.nonzero(
        (single_output.ctx_seq_idx > 0).any(dim=2)
        | (multi_output.ctx_seq_idx > 0).any(dim=2),
        as_tuple=False,
    )
    samples: list[Dict[str, object]] = []
    for comp_idx, latent_idx in populated[: max(0, limit)].tolist():
        single_ids = single_output.ctx_seq_idx[comp_idx, latent_idx]
        multi_ids = multi_output.ctx_seq_idx[comp_idx, latent_idx]
        single_vals = single_output.ctx_seq_val[comp_idx, latent_idx].float()
        multi_vals = multi_output.ctx_seq_val[comp_idx, latent_idx].float()
        samples.append(
            {
                "component": int(comp_idx),
                "latent": int(latent_idx),
                "ids_equal": bool(torch.equal(single_ids, multi_ids)),
                "values_close": bool(torch.allclose(single_vals, multi_vals)),
                "single_ids": single_ids.tolist(),
                "multi_ids": multi_ids.tolist(),
                "single_values": [float(value) for value in single_vals.tolist()],
                "multi_values": [float(value) for value in multi_vals.tolist()],
            }
        )
    return samples


__all__ = [
    "NegativeContextComparisonResult",
    "build_negative_context_comparison_report",
    "compare_negative_context_backends",
]
