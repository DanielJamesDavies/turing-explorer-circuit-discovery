"""Negative-context stats and reporting helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from store.seq_repr import SeqRepr


@dataclass
class NegCtxStats:
    n_latents_attempted: int = 0
    n_latents_skipped_low_pos: int = 0
    n_latents_populated: int = 0
    n_latents_zero_negatives: int = 0
    fill_counts: list[int] = field(default_factory=list)
    backend: str = "single_gpu_exact"
    devices: list[str] = field(default_factory=list)
    ann_device: str = ""
    seq_repr_n_seqs: int = 0
    seq_repr_n_stored: int = 0
    seq_repr_repr_dim: int = 0
    seq_repr_is_capped: bool = False
    component_assignments: dict[str, list[int]] = field(default_factory=dict)
    index_shard_assignments: dict[str, dict[str, int]] = field(default_factory=dict)
    per_device_timing_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    ann_index_memory_estimate_bytes: int = 0
    ann_query_working_memory_bytes: int = 0
    ann_total_memory_estimate_bytes: int = 0
    ann_memory_guardrail_fraction: float = 0.0
    ann_memory_guardrail_limit_bytes: int = 0
    ann_shard_memory_estimates: dict[str, dict[str, int]] = field(default_factory=dict)

    t_index_build: float = 0.0
    t_pos_collect: float = 0.0   # active detection + pair gather
    t_qmat_build: float = 0.0    # scatter-mean query matrix
    t_query: float = 0.0         # matmul + topk search
    t_filter: float = 0.0        # searchsorted filter
    t_write: float = 0.0         # index_copy_ write to neg_ctx
    t_total: float = 0.0

    @property
    def fill_rate_mean(self) -> float:
        return float(np.mean(self.fill_counts)) if self.fill_counts else 0.0

    @property
    def fill_rate_p10(self) -> float:
        return float(np.percentile(self.fill_counts, 10)) if self.fill_counts else 0.0

    @property
    def fill_rate_p50(self) -> float:
        return float(np.percentile(self.fill_counts, 50)) if self.fill_counts else 0.0

    @property
    def fill_rate_p90(self) -> float:
        return float(np.percentile(self.fill_counts, 90)) if self.fill_counts else 0.0

    @property
    def seq_repr_cap_percent(self) -> float:
        if self.seq_repr_n_seqs <= 0:
            return 0.0
        return float(self.seq_repr_n_stored) / float(self.seq_repr_n_seqs) * 100.0

    def record_seq_repr(self, seq_repr: "SeqRepr") -> None:
        self.seq_repr_n_seqs = int(seq_repr.n_seqs)
        self.seq_repr_n_stored = int(seq_repr.n_stored)
        self.seq_repr_repr_dim = int(seq_repr.repr_dim)
        self.seq_repr_is_capped = bool(seq_repr.is_capped)

    def print_summary(self, n_sequences: int) -> None:
        print(f"  [neg_ctx] Latents attempted:       {self.n_latents_attempted:,}")
        if self.backend:
            print(f"  [neg_ctx] Backend:                 {self.backend}")
        if self.ann_device:
            print(f"  [neg_ctx] ANN device:              {self.ann_device}")
        if self.devices:
            print(f"  [neg_ctx] Devices:                 {', '.join(self.devices)}")
        if self.seq_repr_n_seqs:
            print(
                "  [neg_ctx] Seq repr index:          "
                f"{self.seq_repr_n_stored:,} / {self.seq_repr_n_seqs:,} sequences "
                f"({self.seq_repr_cap_percent:.1f}%, dim={self.seq_repr_repr_dim}, "
                f"capped={self.seq_repr_is_capped})"
            )
        if self.ann_total_memory_estimate_bytes:
            print(
                "  [neg_ctx] ANN memory estimate:     "
                f"{self.ann_total_memory_estimate_bytes / 1024**3:.2f} GiB "
                f"(index={self.ann_index_memory_estimate_bytes / 1024**3:.2f} GiB, "
                f"query_working={self.ann_query_working_memory_bytes / 1024**3:.2f} GiB)"
            )
        print(f"  [neg_ctx] Skipped (low PosCtx):    {self.n_latents_skipped_low_pos:,}")
        print(f"  [neg_ctx] Populated:               {self.n_latents_populated:,}")
        print(f"  [neg_ctx] Zero negatives found:    {self.n_latents_zero_negatives:,}")
        if self.fill_counts:
            print(
                f"  [neg_ctx] Fill count (/{n_sequences}) "
                f"mean={self.fill_rate_mean:.1f}  "
                f"p10={self.fill_rate_p10:.1f}  "
                f"p50={self.fill_rate_p50:.1f}  "
                f"p90={self.fill_rate_p90:.1f}  "
                f"min={min(self.fill_counts)}  "
                f"max={max(self.fill_counts)}"
            )
        print(f"  [neg_ctx] Timing breakdown:")
        print(f"    Index build:      {self.t_index_build * 1000:8.1f} ms")
        print(f"    PosCtx collect:   {self.t_pos_collect * 1000:8.1f} ms")
        print(f"    Qmat scatter:     {self.t_qmat_build  * 1000:8.1f} ms")
        print(f"    Matmul + topk:    {self.t_query       * 1000:8.1f} ms")
        print(f"    Filter:           {self.t_filter      * 1000:8.1f} ms")
        print(f"    Write:            {self.t_write       * 1000:8.1f} ms")
        print(f"    Total:            {self.t_total       * 1000:8.1f} ms  ({self.t_total:.1f} s)")

    def save(self, path: str) -> None:
        data = {
            "n_latents_attempted": self.n_latents_attempted,
            "n_latents_skipped_low_pos": self.n_latents_skipped_low_pos,
            "n_latents_populated": self.n_latents_populated,
            "n_latents_zero_negatives": self.n_latents_zero_negatives,
            "backend": self.backend,
            "devices": self.devices,
            "ann_device": self.ann_device,
            "seq_repr_n_seqs": self.seq_repr_n_seqs,
            "seq_repr_n_stored": self.seq_repr_n_stored,
            "seq_repr_repr_dim": self.seq_repr_repr_dim,
            "seq_repr_is_capped": self.seq_repr_is_capped,
            "seq_repr_cap_percent": round(self.seq_repr_cap_percent, 2),
            "component_assignments": self.component_assignments,
            "index_shard_assignments": self.index_shard_assignments,
            "per_device_timing_ms": self.per_device_timing_ms,
            "ann_index_memory_estimate_bytes": self.ann_index_memory_estimate_bytes,
            "ann_query_working_memory_bytes": self.ann_query_working_memory_bytes,
            "ann_total_memory_estimate_bytes": self.ann_total_memory_estimate_bytes,
            "ann_memory_guardrail_fraction": self.ann_memory_guardrail_fraction,
            "ann_memory_guardrail_limit_bytes": self.ann_memory_guardrail_limit_bytes,
            "ann_shard_memory_estimates": self.ann_shard_memory_estimates,
            "fill_rate_mean": round(self.fill_rate_mean, 2),
            "fill_rate_p10": round(self.fill_rate_p10, 2),
            "fill_rate_p50": round(self.fill_rate_p50, 2),
            "fill_rate_p90": round(self.fill_rate_p90, 2),
            "fill_count_min": int(min(self.fill_counts)) if self.fill_counts else 0,
            "fill_count_max": int(max(self.fill_counts)) if self.fill_counts else 0,
            "fill_counts": self.fill_counts,
            "t_index_build_ms": round(self.t_index_build * 1000, 1),
            "t_pos_collect_ms": round(self.t_pos_collect * 1000, 1),
            "t_qmat_build_ms": round(self.t_qmat_build * 1000, 1),
            "t_query_ms": round(self.t_query * 1000, 1),
            "t_filter_ms": round(self.t_filter * 1000, 1),
            "t_write_ms": round(self.t_write * 1000, 1),
            "t_total_ms": round(self.t_total * 1000, 1),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def merge_from(self, other: "NegCtxStats") -> None:
        self.n_latents_attempted += other.n_latents_attempted
        self.n_latents_skipped_low_pos += other.n_latents_skipped_low_pos
        self.n_latents_populated += other.n_latents_populated
        self.n_latents_zero_negatives += other.n_latents_zero_negatives
        self.fill_counts.extend(other.fill_counts)
        self.t_index_build += other.t_index_build
        self.t_pos_collect += other.t_pos_collect
        self.t_qmat_build += other.t_qmat_build
        self.t_query += other.t_query
        self.t_filter += other.t_filter
        self.t_write += other.t_write
        self.per_device_timing_ms.update(other.per_device_timing_ms)


__all__ = ["NegCtxStats"]
