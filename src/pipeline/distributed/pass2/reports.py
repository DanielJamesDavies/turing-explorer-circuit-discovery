"""Reporting and shared IO helpers for distributed pass-2 reducers."""

from __future__ import annotations

import json
import os
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from .contracts import CandidateDumpReducerEntry


def build_pass2_reduce_manifest_metrics(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract stable reducer metrics suitable for manifest/report embedding."""

    timing = report.get("timing", {})
    if not isinstance(timing, Mapping):
        timing = {}
    return {
        "reducer_mode": report.get("reducer_mode"),
        "coactivation_mode": report.get("coactivation_mode"),
        "backend": report.get("backend"),
        "reducer_count": int(report.get("reducer_count", 1) or 1),
        "input_bytes": int(
            report.get("input_candidate_dump_bytes", report.get("input_partial_sum_bytes", 0)) or 0
        ),
        "output_artifact_size_bytes": int(report.get("output_artifact_size_bytes", 0) or 0),
        "peak_cpu_memory_bytes": report.get("peak_cpu_memory_bytes"),
        "total_s": float(timing.get("total_s", report.get("elapsed_s", 0.0)) or 0.0),
        "reduce_s": float(timing.get("reduce_s", 0.0) or 0.0),
        "pmi_s": float(timing.get("pmi_s", 0.0) or 0.0),
    }


def format_pass2_reduce_benchmark_report(report: Mapping[str, Any]) -> str:
    """Format a reducer report into a short benchmark summary."""

    timing = report.get("timing", {})
    if not isinstance(timing, Mapping):
        timing = {}
    lines = [
        "pass2 reduce benchmark:",
        f"  reducer_mode: {report.get('reducer_mode')}",
        f"  coactivation_mode: {report.get('coactivation_mode')}",
        f"  backend: {report.get('backend')}",
        f"  reducer_count: {report.get('reducer_count', 1)}",
        f"  input_bytes: {report.get('input_candidate_dump_bytes', report.get('input_partial_sum_bytes', 0))}",
        f"  output_artifact_size_bytes: {report.get('output_artifact_size_bytes', 0)}",
        f"  peak_cpu_memory_bytes: {report.get('peak_cpu_memory_bytes')}",
        f"  total_s: {float(timing.get('total_s', report.get('elapsed_s', 0.0)) or 0.0):.6f}",
        f"  reduce_s: {float(timing.get('reduce_s', 0.0) or 0.0):.6f}",
        f"  pmi_s: {float(timing.get('pmi_s', 0.0) or 0.0):.6f}",
    ]
    if "shard_write_s" in timing or "stitch_s" in timing:
        lines.extend(
            [
                f"  shard_write_s: {float(timing.get('shard_write_s', 0.0) or 0.0):.6f}",
                f"  stitch_s: {float(timing.get('stitch_s', 0.0) or 0.0):.6f}",
            ]
        )
    return "\n".join(lines)


def candidate_dump_entry_tensor_bytes(entry: CandidateDumpReducerEntry) -> int:
    total = 0
    for key in ("sequence_ids", "candidate_ids", "candidate_vals"):
        value = entry.payload.get(key)
        if isinstance(value, torch.Tensor):
            total += int(value.numel() * value.element_size())
    return total


def file_size_or_zero(path: str | Path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def start_memory_trace() -> bool:
    if tracemalloc.is_tracing():
        return False
    tracemalloc.start()
    return True


def stop_memory_trace(started_here: bool) -> Optional[int]:
    if not tracemalloc.is_tracing():
        return None
    _current, peak = tracemalloc.get_traced_memory()
    if started_here:
        tracemalloc.stop()
    return int(peak)


def atomic_torch_save(data: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    torch.save(data, tmp_path)
    os.replace(tmp_path, output_path)


def atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, output_path)


__all__ = [
    "atomic_torch_save",
    "atomic_write_json",
    "build_pass2_reduce_manifest_metrics",
    "candidate_dump_entry_tensor_bytes",
    "file_size_or_zero",
    "format_pass2_reduce_benchmark_report",
    "start_memory_trace",
    "stop_memory_trace",
]
