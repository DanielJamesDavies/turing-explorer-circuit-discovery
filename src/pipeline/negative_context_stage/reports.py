"""Reports and summaries for negative-context stage outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from pipeline.distributed.interfaces import PipelineOutputPaths
from store.neg_context import NegCtxStats

from .inputs import LoadedContext, SeqReprLike


def build_negative_context_sanity_report(
    paths: PipelineOutputPaths,
    output_neg_ctx: LoadedContext,
    stats: NegCtxStats,
    metadata: Dict[str, object],
    *,
    seq_repr: SeqReprLike,
) -> Dict[str, object]:
    validation = _neg_ctx_validation_summary(
        output_neg_ctx,
        total_n_seqs=seq_repr.n_seqs,
        n_sequences=output_neg_ctx.num_ctx_sequences,
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "metadata": metadata,
        "artifacts": {
            "neg_ctx": str(paths.neg_ctx),
            "neg_ctx_stats": str(paths.run_root / "neg_ctx_stats.json"),
        },
        "backend": stats.backend,
        "devices": stats.devices,
        "shape": list(output_neg_ctx.ctx_seq_idx.shape),
        "dtype": {
            "ctx_seq_idx": str(output_neg_ctx.ctx_seq_idx.dtype),
            "ctx_seq_val": str(output_neg_ctx.ctx_seq_val.dtype),
        },
        "seq_repr": {
            "n_seqs": int(seq_repr.n_seqs),
            "n_stored": int(seq_repr.n_stored),
            "repr_dim": int(seq_repr.repr_dim),
            "is_capped": bool(seq_repr.is_capped),
            "cap_percent": (
                float(seq_repr.n_stored) / float(seq_repr.n_seqs) * 100.0
                if int(seq_repr.n_seqs) > 0
                else 0.0
            ),
        },
        "populated_rows": _populated_row_count(output_neg_ctx),
        "zero_negative_rows": int(stats.n_latents_zero_negatives),
        "valid_entry_count": validation["valid_entry_count"],
        "validation": validation,
        "fill_distribution": _fill_summary(output_neg_ctx),
        "timing_ms": _stats_timing_ms(stats),
        "memory": {
            "ann_index_memory_estimate_bytes": stats.ann_index_memory_estimate_bytes,
            "ann_query_working_memory_bytes": stats.ann_query_working_memory_bytes,
            "ann_total_memory_estimate_bytes": stats.ann_total_memory_estimate_bytes,
            "ann_memory_guardrail_fraction": stats.ann_memory_guardrail_fraction,
            "ann_memory_guardrail_limit_bytes": stats.ann_memory_guardrail_limit_bytes,
        },
    }


def print_negative_context_sanity_summary(report: Dict[str, object]) -> None:
    fill = report.get("fill_distribution", {})
    timing = report.get("timing_ms", {})
    seq_repr = report.get("seq_repr", {})
    validation = report.get("validation", {})
    memory = report.get("memory", {})
    print(
        "  [neg_ctx] summary "
        f"backend={report.get('backend')} "
        f"devices={','.join(str(device) for device in report.get('devices', []))} "
        f"populated_rows={report.get('populated_rows')} "
        f"fill_mean={float(fill.get('mean', 0.0)):.1f} "
        f"fill_min={fill.get('min', 0)} "
        f"fill_max={fill.get('max', 0)} "
        f"zero_neg={report.get('zero_negative_rows', 0)} "
        f"invalid_seq={validation.get('invalid_sequence_count', 0)} "
        f"non_finite={validation.get('non_finite_similarity_count', 0)} "
        f"seq_repr={seq_repr.get('n_stored', 0)}/{seq_repr.get('n_seqs', 0)} "
        f"cap={float(seq_repr.get('cap_percent', 0.0)):.1f}% "
        f"ann_mem_gib={int(memory.get('ann_total_memory_estimate_bytes', 0)) / 1024**3:.2f} "
        f"total_ms={float(timing.get('total_ms', 0.0)):.1f}"
    )


def _populated_row_count(output: LoadedContext) -> int:
    row_has_entries = (output.ctx_seq_idx > 0).any(dim=2)
    return int(row_has_entries.sum().item())


def _fill_summary(output: LoadedContext) -> Dict[str, object]:
    fill_counts = (output.ctx_seq_idx > 0).sum(dim=2).reshape(-1).to(torch.int64)
    histogram = torch.bincount(fill_counts).cpu()
    return {
        "min": int(fill_counts.min().item()) if fill_counts.numel() else 0,
        "max": int(fill_counts.max().item()) if fill_counts.numel() else 0,
        "mean": float(fill_counts.float().mean().item()) if fill_counts.numel() else 0.0,
        "histogram": {
            str(idx): int(count.item())
            for idx, count in enumerate(histogram)
        },
    }


def _neg_ctx_validation_summary(
    output: LoadedContext,
    *,
    total_n_seqs: int,
    n_sequences: int,
) -> Dict[str, object]:
    idx = output.ctx_seq_idx
    vals = output.ctx_seq_val.float()
    populated = idx > 0
    non_finite = ~torch.isfinite(vals)
    invalid_sequence = populated & ((idx < 1) | (idx > total_n_seqs))
    negative_similarity = vals < 0
    value_without_sequence = (vals > 0) & ~populated
    valid_entries = populated & (vals > 0) & ~invalid_sequence & ~non_finite & ~negative_similarity
    return {
        "checked": True,
        "total_rows": int(idx.shape[0] * idx.shape[1]),
        "configured_n_sequences": int(n_sequences),
        "invalid_sequence_count": int(invalid_sequence.sum().item()),
        "non_finite_similarity_count": int(non_finite.sum().item()),
        "negative_similarity_count": int(negative_similarity.sum().item()),
        "value_without_sequence_count": int(value_without_sequence.sum().item()),
        "valid_entry_count": int(valid_entries.sum().item()),
    }


def _stats_timing_ms(stats: NegCtxStats) -> Dict[str, float]:
    return {
        "index_build_ms": round(stats.t_index_build * 1000, 1),
        "pos_collect_ms": round(stats.t_pos_collect * 1000, 1),
        "qmat_build_ms": round(stats.t_qmat_build * 1000, 1),
        "query_ms": round(stats.t_query * 1000, 1),
        "filter_ms": round(stats.t_filter * 1000, 1),
        "write_ms": round(stats.t_write * 1000, 1),
        "total_ms": round(stats.t_total * 1000, 1),
    }


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


__all__ = [
    "build_negative_context_sanity_report",
    "print_negative_context_sanity_summary",
]
