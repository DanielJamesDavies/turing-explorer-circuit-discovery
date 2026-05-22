"""
Negative context builder (ANN retrieval).

After Pass 1 has collected seq_repr, top_ctx, and mid_ctx, this module:
  1. Builds a TorchANNIndex (exact cosine via chunked matmul + topk) over all
     sequence representations.
  2. For each SAE component, vectorises the entire pipeline end-to-end on the
     compute device — no Python loops over individual latents:
       a. Active latent detection via tensor nonzero.
       b. Query matrix built with a single gather + index_add_ scatter-mean.
       c. Batched search (F.normalize @ index.T + topk).
       d. Membership filter via torch.searchsorted on encoded (qi, seq_id) keys.
       e. Bulk write to neg_ctx via index_copy_.
  3. Stores the top N_neg hardest negatives per latent (cosine similarity order).

Compute device: config.hardware.ann_device  ("auto" | "gpu" | "cpu")
  "auto" — GPU if CUDA available, else CPU.

No external ANN library is required — the entire pipeline uses PyTorch primitives
(cuBLAS on GPU, MKL/OpenBLAS on CPU).

All timing is printed via tqdm for performance analysis and optimisation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Iterable, List, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import config
from model.turingllm import TuringLLMConfig

if TYPE_CHECKING:
    from store.context import Context
    from store.seq_repr import SeqRepr


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class NegCtxStats:
    n_latents_attempted:       int = 0
    n_latents_skipped_low_pos: int = 0
    n_latents_populated:       int = 0
    n_latents_zero_negatives:  int = 0
    fill_counts: List[int] = field(default_factory=list)
    backend: str = "single_gpu_exact"
    devices: List[str] = field(default_factory=list)
    ann_device: str = ""
    seq_repr_n_seqs: int = 0
    seq_repr_n_stored: int = 0
    seq_repr_repr_dim: int = 0
    seq_repr_is_capped: bool = False
    component_assignments: dict[str, List[int]] = field(default_factory=dict)
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
    t_qmat_build:  float = 0.0   # scatter-mean query matrix
    t_query:       float = 0.0   # matmul + topk search
    t_filter:      float = 0.0   # searchsorted filter
    t_write:       float = 0.0   # index_copy_ write to neg_ctx
    t_total:       float = 0.0

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
            print(f"  [neg_ctx] Fill count (/{n_sequences}) "
                  f"mean={self.fill_rate_mean:.1f}  "
                  f"p10={self.fill_rate_p10:.1f}  "
                  f"p50={self.fill_rate_p50:.1f}  "
                  f"p90={self.fill_rate_p90:.1f}  "
                  f"min={min(self.fill_counts)}  "
                  f"max={max(self.fill_counts)}")
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
            "n_latents_attempted":       self.n_latents_attempted,
            "n_latents_skipped_low_pos": self.n_latents_skipped_low_pos,
            "n_latents_populated":       self.n_latents_populated,
            "n_latents_zero_negatives":  self.n_latents_zero_negatives,
            "backend":                   self.backend,
            "devices":                   self.devices,
            "ann_device":                self.ann_device,
            "seq_repr_n_seqs":           self.seq_repr_n_seqs,
            "seq_repr_n_stored":         self.seq_repr_n_stored,
            "seq_repr_repr_dim":         self.seq_repr_repr_dim,
            "seq_repr_is_capped":        self.seq_repr_is_capped,
            "seq_repr_cap_percent":      round(self.seq_repr_cap_percent, 2),
            "component_assignments":     self.component_assignments,
            "index_shard_assignments":   self.index_shard_assignments,
            "per_device_timing_ms":      self.per_device_timing_ms,
            "ann_index_memory_estimate_bytes": self.ann_index_memory_estimate_bytes,
            "ann_query_working_memory_bytes": self.ann_query_working_memory_bytes,
            "ann_total_memory_estimate_bytes": self.ann_total_memory_estimate_bytes,
            "ann_memory_guardrail_fraction": self.ann_memory_guardrail_fraction,
            "ann_memory_guardrail_limit_bytes": self.ann_memory_guardrail_limit_bytes,
            "ann_shard_memory_estimates": self.ann_shard_memory_estimates,
            "fill_rate_mean": round(self.fill_rate_mean, 2),
            "fill_rate_p10":  round(self.fill_rate_p10,  2),
            "fill_rate_p50":  round(self.fill_rate_p50,  2),
            "fill_rate_p90":  round(self.fill_rate_p90,  2),
            "fill_count_min": int(min(self.fill_counts)) if self.fill_counts else 0,
            "fill_count_max": int(max(self.fill_counts)) if self.fill_counts else 0,
            "fill_counts": self.fill_counts,
            "t_index_build_ms": round(self.t_index_build * 1000, 1),
            "t_pos_collect_ms": round(self.t_pos_collect * 1000, 1),
            "t_qmat_build_ms":  round(self.t_qmat_build  * 1000, 1),
            "t_query_ms":       round(self.t_query       * 1000, 1),
            "t_filter_ms":      round(self.t_filter      * 1000, 1),
            "t_write_ms":       round(self.t_write       * 1000, 1),
            "t_total_ms":       round(self.t_total       * 1000, 1),
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


# ---------------------------------------------------------------------------
# TorchANNIndex — pure PyTorch, no external dependency
# ---------------------------------------------------------------------------

class TorchANNIndex:
    """
    Exact cosine similarity index.

    Build: L2-normalise vecs → store on device.
    Search: F.normalize(queries) @ index.T → topk  (cuBLAS / MKL).

    Results are returned on the same device as the index (no CPU round-trip),
    so downstream GPU filtering can start immediately.
    """

    def __init__(self, vecs: torch.Tensor, device: torch.device):
        t0 = time.perf_counter()
        self.device = device
        self.n      = vecs.shape[0]
        self.dim    = vecs.shape[1]

        # chunk_size: larger = fewer kernel launches = better GPU utilisation.
        # Peak intermediate tensor = chunk_size × N × 4 B.
        # GPU:  4096 × 65536 × 4 = 1 GB — safe for 16 GB+ VRAM.
        # CPU:  512  × 65536 × 4 = 128 MB — conservative.
        self.chunk_size = 4096 if device.type == "cuda" else 512

        self.index = F.normalize(vecs.float(), dim=1).to(device)

        build_ms = (time.perf_counter() - t0) * 1000
        print(f"  [neg_ctx] ANN index on {device} "
              f"— {self.n:,} vecs × {self.dim} dims "
              f"— built in {build_ms:.1f} ms "
              f"— chunk_size={self.chunk_size}")

    @torch.no_grad()
    def search(
        self,
        queries: torch.Tensor,   # [Q, D], any device
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (similarities [Q, k], indices [Q, k]) on self.device.
        Similarities are in descending order (highest first = hardest negatives first).
        """
        K      = min(k, self.n)
        q_norm = F.normalize(queries.float(), dim=1)
        if q_norm.device != self.device:
            q_norm = q_norm.to(self.device)

        Q        = q_norm.shape[0]
        all_sims = torch.empty(Q, K, dtype=torch.float32, device=self.device)
        all_idxs = torch.empty(Q, K, dtype=torch.int64,   device=self.device)

        for start in range(0, Q, self.chunk_size):
            end   = min(start + self.chunk_size, Q)
            sims  = q_norm[start:end] @ self.index.T                   # [C, N]
            ts, ti = sims.topk(K, dim=1, sorted=True)
            all_sims[start:end] = ts
            all_idxs[start:end] = ti

        return all_sims, all_idxs   # on self.device


def estimate_neg_ctx_ann_memory(
    seq_repr: "SeqRepr",
    *,
    query_chunk_size: int,
) -> dict[str, int]:
    """Estimate per-device memory for replicated exact ANN retrieval."""

    return estimate_neg_ctx_ann_memory_for_shape(
        n_stored=int(seq_repr.n_stored),
        n_seqs=int(seq_repr.n_seqs),
        repr_dim=int(seq_repr.repr_dim),
        is_capped=bool(seq_repr.is_capped),
        query_chunk_size=query_chunk_size,
    )


def estimate_neg_ctx_ann_memory_for_shape(
    *,
    n_stored: int,
    n_seqs: int,
    repr_dim: int,
    is_capped: bool,
    query_chunk_size: int,
) -> dict[str, int]:
    """Estimate ANN memory for a specific index shard shape."""

    index_bytes = n_stored * repr_dim * 4
    slot_to_id_bytes = (n_stored + 1) * 8 if is_capped else 0
    id_to_slot_bytes = (n_seqs + 1) * 4 if is_capped else 0
    query_working_bytes = min(int(query_chunk_size), n_stored) * n_stored * 4
    total_bytes = index_bytes + slot_to_id_bytes + id_to_slot_bytes + query_working_bytes
    return {
        "index_bytes": int(index_bytes),
        "slot_to_id_bytes": int(slot_to_id_bytes),
        "id_to_slot_bytes": int(id_to_slot_bytes),
        "query_working_bytes": int(query_working_bytes),
        "total_bytes": int(total_bytes),
        "query_chunk_size": int(query_chunk_size),
    }


def check_neg_ctx_memory_guardrail(
    device: torch.device,
    estimate: dict[str, int],
    *,
    fraction: float,
    fail_on_exceed: bool,
    total_vram_bytes: int | None = None,
) -> dict[str, object]:
    """Check ANN memory estimate against a per-device VRAM fraction."""

    if device.type != "cuda":
        return {
            "device": str(device),
            "checked": False,
            "reason": "non_cuda_device",
            "estimate": estimate,
            "fraction": float(fraction),
            "limit_bytes": 0,
            "exceeds_limit": False,
        }
    if total_vram_bytes is None:
        idx = device.index if device.index is not None else 0
        total_vram_bytes = int(torch.cuda.get_device_properties(idx).total_memory)
    limit_bytes = int(total_vram_bytes * fraction)
    exceeds_limit = int(estimate["total_bytes"]) > limit_bytes
    result = {
        "device": str(device),
        "checked": True,
        "estimate": estimate,
        "fraction": float(fraction),
        "total_vram_bytes": int(total_vram_bytes),
        "limit_bytes": int(limit_bytes),
        "exceeds_limit": bool(exceeds_limit),
    }
    if exceeds_limit:
        message = (
            f"neg_ctx ANN memory estimate for {device} is "
            f"{estimate['total_bytes'] / 1024**3:.2f} GiB, exceeding "
            f"{fraction:.0%} of VRAM ({limit_bytes / 1024**3:.2f} GiB). "
            "Reduce latents.neg_ctx.max_repr_seqs or use a smaller backend/config."
        )
        if fail_on_exceed:
            raise RuntimeError(message)
        print(f"  [neg_ctx] WARNING: {message}")
    return result


def _record_ann_memory_estimate(
    stats: NegCtxStats,
    estimate: dict[str, int],
    guardrail: dict[str, object],
) -> None:
    stats.ann_index_memory_estimate_bytes = int(estimate["index_bytes"])
    stats.ann_query_working_memory_bytes = int(estimate["query_working_bytes"])
    stats.ann_total_memory_estimate_bytes = int(estimate["total_bytes"])
    stats.ann_memory_guardrail_fraction = float(guardrail.get("fraction", 0.0))
    stats.ann_memory_guardrail_limit_bytes = int(guardrail.get("limit_bytes", 0))


@dataclass(frozen=True)
class ANNIndexShard:
    device: torch.device
    start_slot: int
    end_slot: int
    index: TorchANNIndex


def partition_index_slots(n_stored: int, devices: Sequence[torch.device]) -> dict[str, tuple[int, int]]:
    """Split 0-indexed ANN slot rows contiguously across devices."""

    if not devices:
        raise ValueError("At least one device is required for index sharding.")
    result: dict[str, tuple[int, int]] = {}
    base = n_stored // len(devices)
    remainder = n_stored % len(devices)
    start = 0
    for idx, device in enumerate(devices):
        size = base + (1 if idx < remainder else 0)
        end = start + size
        result[str(device)] = (start, end)
        start = end
    return result


def merge_shard_search_results(
    shard_results: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    k: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge shard-local top-K results into global top-K slot IDs."""

    if not shard_results:
        raise ValueError("At least one shard result is required.")
    target_device = device or shard_results[0][0].device
    sims = torch.cat([result[0].to(target_device) for result in shard_results], dim=1)
    idxs = torch.cat([result[1].to(target_device) for result in shard_results], dim=1)
    merged_k = min(k, sims.shape[1])
    top_sims, top_pos = sims.topk(merged_k, dim=1, sorted=True)
    top_idxs = idxs.gather(1, top_pos)
    return top_sims, top_idxs


class ShardedANNIndex:
    """Exact ANN index with sequence-representation rows split across devices."""

    def __init__(
        self,
        vecs: torch.Tensor,
        devices: Sequence[torch.device],
    ) -> None:
        self.devices = list(devices)
        self.n = vecs.shape[0]
        self.dim = vecs.shape[1]
        self.shards: list[ANNIndexShard] = []
        assignments = partition_index_slots(self.n, self.devices)
        for device in self.devices:
            start, end = assignments[str(device)]
            if start == end:
                continue
            index = TorchANNIndex(vecs[start:end], device=device)
            self.shards.append(
                ANNIndexShard(
                    device=device,
                    start_slot=start,
                    end_slot=end,
                    index=index,
                )
            )
        if not self.shards:
            raise ValueError("Index sharding produced no non-empty shards.")

    @property
    def shard_assignments(self) -> dict[str, tuple[int, int]]:
        return {str(shard.device): (shard.start_slot, shard.end_slot) for shard in self.shards}

    @torch.no_grad()
    def search(self, queries: torch.Tensor, k: int, *, merge_device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        shard_results: list[tuple[torch.Tensor, torch.Tensor]] = []
        for shard in self.shards:
            shard_k = min(k, shard.end_slot - shard.start_slot)
            if shard_k <= 0:
                continue
            sims, local_idxs = shard.index.search(queries, shard_k)
            global_idxs = local_idxs + shard.start_slot
            shard_results.append((sims, global_idxs))
        return merge_shard_search_results(shard_results, k=k, device=merge_device)


# ---------------------------------------------------------------------------
# Device selector
# ---------------------------------------------------------------------------

def _ann_device() -> torch.device:
    cfg = cast(str, config.hardware.ann_device or "auto")
    if cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("hardware.ann_device = 'gpu' but CUDA is not available.")
        return torch.device("cuda")
    if cfg == "cpu":
        return torch.device("cpu")
    if cfg.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"hardware.ann_device = {cfg!r} but CUDA is not available.")
        device = torch.device(cfg)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"hardware.ann_device = {cfg!r} is outside visible CUDA range "
                f"0..{torch.cuda.device_count() - 1}."
            )
        return device
    raise ValueError(
        "hardware.ann_device must be one of 'auto', 'cpu', 'gpu', 'cuda', or 'cuda:N'."
    )


def parse_neg_ctx_devices(
    configured_devices: Sequence[int | str],
    cuda_count: int | None = None,
) -> list[torch.device]:
    if cuda_count is None:
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if not configured_devices:
        return [torch.device(f"cuda:{idx}") for idx in range(cuda_count)]

    devices: list[torch.device] = []
    for raw in configured_devices:
        if isinstance(raw, int):
            devices.append(torch.device(f"cuda:{raw}"))
            continue
        text = str(raw)
        if text.isdigit():
            devices.append(torch.device(f"cuda:{text}"))
        elif text == "cuda":
            devices.append(torch.device("cuda:0"))
        elif text.startswith("cuda:"):
            devices.append(torch.device(text))
        else:
            raise ValueError(f"Invalid neg_ctx device {raw!r}; use CUDA ids like 0 or 'cuda:0'.")

    seen: set[str] = set()
    deduped: list[torch.device] = []
    for device in devices:
        key = str(device)
        if key not in seen:
            seen.add(key)
            deduped.append(device)
    return deduped


def partition_components(n_components: int, devices: Sequence[torch.device]) -> dict[str, list[int]]:
    if not devices:
        raise ValueError("At least one device is required for component partitioning.")
    result = {str(device): [] for device in devices}
    for comp_idx in range(n_components):
        result[str(devices[comp_idx % len(devices)])].append(comp_idx)
    return result


def _validate_cuda_devices(devices: Sequence[torch.device]) -> None:
    if not devices:
        raise RuntimeError(
            "latents.neg_ctx.backend='multi_gpu_exact' requires at least one CUDA device."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "latents.neg_ctx.backend='multi_gpu_exact' requires CUDA, but CUDA is not available."
        )
    cuda_count = torch.cuda.device_count()
    for device in devices:
        if device.type != "cuda":
            raise RuntimeError(f"multi_gpu_exact only supports CUDA devices, got {device}.")
        idx = device.index if device.index is not None else 0
        if idx < 0 or idx >= cuda_count:
            raise RuntimeError(f"Configured CUDA device {device} is outside visible range 0..{cuda_count - 1}.")


def validate_neg_ctx_output(
    neg_ctx: "Context",
    *,
    total_n_seqs: int,
    n_sequences: int,
) -> None:
    """Validate populated neg_ctx rows after exact retrieval."""

    if neg_ctx.ctx_seq_idx.ndim != 3 or neg_ctx.ctx_seq_val.ndim != 3:
        raise ValueError("neg_ctx tensors must be rank-3")
    if neg_ctx.ctx_seq_idx.shape != neg_ctx.ctx_seq_val.shape:
        raise ValueError("neg_ctx tensor shape mismatch")
    if neg_ctx.ctx_seq_idx.shape[2] > n_sequences:
        raise ValueError("neg_ctx rows exceed configured n_sequences")
    if (neg_ctx.ctx_seq_idx < 0).any():
        raise ValueError("neg_ctx sequence IDs must be non-negative")
    if not torch.isfinite(neg_ctx.ctx_seq_val.float()).all():
        raise ValueError("neg_ctx similarities must be finite")
    if (neg_ctx.ctx_seq_val < 0).any():
        raise ValueError("neg_ctx similarities must be non-negative")

    populated = neg_ctx.ctx_seq_idx > 0
    if bool((neg_ctx.ctx_seq_idx[populated] > total_n_seqs).any().item()):
        raise ValueError("neg_ctx sequence ID exceeds seq_repr n_seqs")
    if bool(((neg_ctx.ctx_seq_val > 0) & ~populated).any().item()):
        raise ValueError("neg_ctx positive similarities must have sequence IDs")


# ---------------------------------------------------------------------------
# Per-component processing — fully vectorised, no Python loop over latents
# ---------------------------------------------------------------------------

# Max pairs processed in one scatter-mean chunk (bounds peak GPU tensor size).
# 65536 pairs × 1024 × 4 B = 256 MB — safe even on 8 GB VRAM.
_PAIR_CHUNK = 65536


@torch.no_grad()
def _process_component(
    comp_idx:      int,
    top_ctx:       "Context",
    mid_ctx:       "Context",
    neg_ctx:       "Context",
    index:         TorchANNIndex,
    K:             int,
    n_neg:         int,
    min_pos_ctx:   int,
    stats:         NegCtxStats,
    total_n_seqs:  int,
    slot_to_id_d:  "torch.Tensor | None",
    id_to_slot_d:  "torch.Tensor | None",
) -> dict:
    """
    Process one SAE component end-to-end without any Python loop over latents.

    slot_to_id_d / id_to_slot_d are None when seq_repr is uncapped (slot == seq_id).

    Returns a dict of per-step timing (seconds) for the tqdm postfix.
    """
    device = index.device
    d_sae  = top_ctx.ctx_seq_idx.shape[1]
    timing: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # 1. Move component slices to compute device                          #
    # ------------------------------------------------------------------ #

    top_ids_d  = top_ctx.ctx_seq_idx[comp_idx].to(device, dtype=torch.int64)  # [d_sae, N_top]
    top_mask_d = top_ctx.ctx_seq_val[comp_idx].to(device).float() > 0          # [d_sae, N_top]
    mid_ids_d  = mid_ctx.ctx_seq_idx[comp_idx].to(device, dtype=torch.int64)  # [d_sae, N_mid]
    mid_mask_d = mid_ctx.ctx_seq_val[comp_idx].to(device).float() > 0          # [d_sae, N_mid]

    # ------------------------------------------------------------------ #
    # 2. Active latent detection + flatten (qi, seq_id) pairs            #
    # ------------------------------------------------------------------ #

    t0 = time.perf_counter()

    pos_counts = top_mask_d.sum(dim=1) + mid_mask_d.sum(dim=1)   # [d_sae]
    active_js  = (pos_counts >= min_pos_ctx).nonzero(as_tuple=True)[0]  # [Q]
    Q          = active_js.shape[0]

    stats.n_latents_attempted       += d_sae
    stats.n_latents_skipped_low_pos += d_sae - Q

    if Q == 0:
        timing["pos"] = time.perf_counter() - t0
        return timing

    # Gather (qi_within_Q, seq_id) pairs for all active latents in one pass.
    at_ids  = top_ids_d[active_js]    # [Q, N_top]
    at_mask = top_mask_d[active_js]   # [Q, N_top]
    am_ids  = mid_ids_d[active_js]    # [Q, N_mid]
    am_mask = mid_mask_d[active_js]   # [Q, N_mid]

    t_qi, t_ki = at_mask.nonzero(as_tuple=True)   # [M_top]
    m_qi, m_ki = am_mask.nonzero(as_tuple=True)   # [M_mid]

    all_qi     = torch.cat([t_qi, m_qi])
    all_seqids = torch.cat([at_ids[t_qi, t_ki], am_ids[m_qi, m_ki]])   # 1-indexed

    # Remove sentinel 0 only — range check is handled by the slot filter below.
    valid      = (all_seqids > 0) & (all_seqids <= total_n_seqs)
    all_qi     = all_qi[valid]
    all_seqids = all_seqids[valid]
    M          = all_qi.shape[0]

    timing["pos"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 3. Build query matrix — scatter-mean of PosCtx reps                #
    #    Chunked over pairs to bound peak intermediate tensor size.       #
    #    When capped, filter to pairs whose seq is actually in the index, #
    #    then convert seq_ids → 0-indexed slot rows.                      #
    # ------------------------------------------------------------------ #

    t0   = time.perf_counter()
    D    = index.dim
    sums = torch.zeros(Q, D, dtype=torch.float32, device=device)

    if id_to_slot_d is not None:
        # Capped: only sequences stored in the ANN index contribute to the centroid.
        slots    = id_to_slot_d[all_seqids]      # [M] slot (0 = not stored)
        in_index = slots > 0
        qi_qmat  = all_qi[in_index]
        rows_all = (slots[in_index] - 1)         # 0-indexed slot row
    else:
        qi_qmat  = all_qi
        rows_all = all_seqids - 1                # 0-indexed seq_id row (slot == seq_id)

    Mq  = qi_qmat.shape[0]
    cnt = torch.bincount(qi_qmat, minlength=Q).float().unsqueeze(1)    # [Q, 1]

    for pair_start in range(0, Mq, _PAIR_CHUNK):
        pair_end    = min(pair_start + _PAIR_CHUNK, Mq)
        chunk_rows  = rows_all[pair_start:pair_end]
        chunk_qi    = qi_qmat[pair_start:pair_end]
        chunk_reps  = index.index[chunk_rows]            # [C, D] float32
        sums.index_add_(0, chunk_qi, chunk_reps)

    qmat = F.normalize(sums / cnt.clamp(min=1e-8), dim=1)   # [Q, D]
    timing["qmat"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 4. Batched search — matmul + topk on device                        #
    # ------------------------------------------------------------------ #

    t0 = time.perf_counter()
    nn_sims, nn_idx = index.search(qmat, K)   # [Q, K] on device
    timing["query"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 5. Membership filter — GPU searchsorted on encoded (qi, seq_id)    #
    #    Encodes every pair as  qi × stride + seq_id  (int64)            #
    #    to allow a single sorted-set lookup across all Q×K candidates.  #
    # ------------------------------------------------------------------ #

    t0 = time.perf_counter()

    # stride must be > max possible seq_id to avoid encoding collisions.
    stride = total_n_seqs + 2
    # Positive set: encode all (qi, seq_id) pairs from PosCtx.
    encoded_pos, _ = (all_qi * stride + all_seqids).sort()   # [M] sorted

    # Candidate set: convert ANN slot indices → actual seq_ids, then encode.
    qi_range   = torch.arange(Q, dtype=torch.int64, device=device)
    if slot_to_id_d is not None:
        nn_seq_ids = slot_to_id_d[(nn_idx + 1).long()]       # [Q, K] slot → seq_id
    else:
        nn_seq_ids = nn_idx + 1                              # [Q, K] slot == seq_id
    encoded_cands  = (qi_range[:, None] * stride + nn_seq_ids).reshape(-1)  # [Q*K]

    # Binary search: is each candidate in the positive set?
    idxs   = torch.searchsorted(encoded_pos, encoded_cands)
    idxs   = idxs.clamp(0, M - 1)
    is_neg = (encoded_pos[idxs] != encoded_cands).reshape(Q, K)          # [Q, K]

    # Select the first n_neg negatives per row (already sorted by desc similarity).
    cum_neg  = is_neg.long().cumsum(dim=1)              # [Q, K]
    selected = is_neg & (cum_neg <= n_neg)              # [Q, K]

    n_found = selected.sum(dim=1)                       # [Q] int64 on device
    timing["filter"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 6. Bulk write to neg_ctx                                            #
    # ------------------------------------------------------------------ #

    t0 = time.perf_counter()

    n_found_cpu = n_found.cpu()
    stats.fill_counts.extend(n_found_cpu.tolist())
    stats.n_latents_zero_negatives += int((n_found_cpu == 0).sum().item())
    stats.n_latents_populated      += int((n_found_cpu >  0).sum().item())

    # Build selected active rows on the compute device, then transfer only
    # those rows back to the CPU store.
    # Fast path: all rows filled to exactly n_neg (virtually always true when K >> n_neg).
    if bool((n_found_cpu == n_neg).all().item()):
        q_ids  = nn_seq_ids[selected].reshape(Q, n_neg).to(torch.int32)   # [Q, n_neg]
        q_sims = nn_sims[selected].reshape(Q, n_neg)                       # [Q, n_neg]
    else:
        # Variable-fill path: some rows have fewer than n_neg negatives.
        q_ids  = torch.zeros(Q, n_neg, dtype=torch.int32,   device=device)
        q_sims = torch.zeros(Q, n_neg, dtype=torch.float32, device=device)
        for qi in range(Q):
            nf = int(n_found_cpu[qi].item())
            if nf == 0:
                continue
            sel_pos = selected[qi].nonzero(as_tuple=True)[0][:n_neg]
            q_ids[qi,  :nf] = nn_seq_ids[qi, sel_pos].to(torch.int32)
            q_sims[qi, :nf] = nn_sims[qi, sel_pos]

    comp_ids = neg_ctx.ctx_seq_idx[comp_idx]
    comp_vals = neg_ctx.ctx_seq_val[comp_idx]
    comp_ids.zero_()
    comp_vals.zero_()

    active_js_cpu = active_js.cpu()
    comp_ids[active_js_cpu] = q_ids.cpu()
    comp_vals[active_js_cpu] = q_sims.cpu()

    timing["write"] = time.perf_counter() - t0
    return timing


@torch.no_grad()
def _process_component_sharded(
    comp_idx:      int,
    top_ctx:       "Context",
    mid_ctx:       "Context",
    neg_ctx:       "Context",
    seq_repr:      "SeqRepr",
    index:         ShardedANNIndex,
    K:             int,
    n_neg:         int,
    min_pos_ctx:   int,
    stats:         NegCtxStats,
    total_n_seqs:  int,
    query_device:  torch.device,
    slot_to_id_d:  "torch.Tensor | None",
    id_to_slot_d:  "torch.Tensor | None",
) -> dict:
    """Process one component using globally merged results from sharded indexes."""

    d_sae = top_ctx.ctx_seq_idx.shape[1]
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    top_ids_d  = top_ctx.ctx_seq_idx[comp_idx].to(query_device, dtype=torch.int64)
    top_mask_d = top_ctx.ctx_seq_val[comp_idx].to(query_device).float() > 0
    mid_ids_d  = mid_ctx.ctx_seq_idx[comp_idx].to(query_device, dtype=torch.int64)
    mid_mask_d = mid_ctx.ctx_seq_val[comp_idx].to(query_device).float() > 0

    pos_counts = top_mask_d.sum(dim=1) + mid_mask_d.sum(dim=1)
    active_js  = (pos_counts >= min_pos_ctx).nonzero(as_tuple=True)[0]
    Q          = active_js.shape[0]

    stats.n_latents_attempted       += d_sae
    stats.n_latents_skipped_low_pos += d_sae - Q

    if Q == 0:
        timing["pos"] = time.perf_counter() - t0
        return timing

    at_ids  = top_ids_d[active_js]
    at_mask = top_mask_d[active_js]
    am_ids  = mid_ids_d[active_js]
    am_mask = mid_mask_d[active_js]

    t_qi, t_ki = at_mask.nonzero(as_tuple=True)
    m_qi, m_ki = am_mask.nonzero(as_tuple=True)

    all_qi     = torch.cat([t_qi, m_qi])
    all_seqids = torch.cat([at_ids[t_qi, t_ki], am_ids[m_qi, m_ki]])
    valid      = (all_seqids > 0) & (all_seqids <= total_n_seqs)
    all_qi     = all_qi[valid]
    all_seqids = all_seqids[valid]
    M          = all_qi.shape[0]
    timing["pos"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    D = index.dim
    sums = torch.zeros(Q, D, dtype=torch.float32, device=query_device)

    if id_to_slot_d is not None:
        slots    = id_to_slot_d[all_seqids]
        in_index = slots > 0
        qi_qmat  = all_qi[in_index]
        rows_all = slots[in_index] - 1
    else:
        qi_qmat  = all_qi
        rows_all = all_seqids - 1

    Mq  = qi_qmat.shape[0]
    cnt = torch.bincount(qi_qmat, minlength=Q).float().unsqueeze(1).to(query_device)

    for pair_start in range(0, Mq, _PAIR_CHUNK):
        pair_end = min(pair_start + _PAIR_CHUNK, Mq)
        chunk_rows = rows_all[pair_start:pair_end].cpu() + 1
        chunk_qi = qi_qmat[pair_start:pair_end]
        chunk_reps = seq_repr.repr_buf[chunk_rows].float().to(query_device)
        chunk_reps = F.normalize(chunk_reps, dim=1)
        sums.index_add_(0, chunk_qi, chunk_reps)

    qmat = F.normalize(sums / cnt.clamp(min=1e-8), dim=1)
    timing["qmat"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    nn_sims, nn_idx = index.search(qmat, K, merge_device=query_device)
    timing["query"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stride = total_n_seqs + 2
    encoded_pos, _ = (all_qi * stride + all_seqids).sort()
    qi_range = torch.arange(Q, dtype=torch.int64, device=query_device)
    if slot_to_id_d is not None:
        nn_seq_ids = slot_to_id_d[(nn_idx + 1).long()]
    else:
        nn_seq_ids = nn_idx + 1
    encoded_cands = (qi_range[:, None] * stride + nn_seq_ids).reshape(-1)
    idxs = torch.searchsorted(encoded_pos, encoded_cands)
    idxs = idxs.clamp(0, M - 1)
    is_neg = (encoded_pos[idxs] != encoded_cands).reshape(Q, nn_seq_ids.shape[1])
    cum_neg = is_neg.long().cumsum(dim=1)
    selected = is_neg & (cum_neg <= n_neg)
    n_found = selected.sum(dim=1)
    timing["filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_found_cpu = n_found.cpu()
    stats.fill_counts.extend(n_found_cpu.tolist())
    stats.n_latents_zero_negatives += int((n_found_cpu == 0).sum().item())
    stats.n_latents_populated      += int((n_found_cpu >  0).sum().item())

    if bool((n_found_cpu == n_neg).all().item()):
        q_ids  = nn_seq_ids[selected].reshape(Q, n_neg).to(torch.int32)
        q_sims = nn_sims[selected].reshape(Q, n_neg)
    else:
        q_ids  = torch.zeros(Q, n_neg, dtype=torch.int32,   device=query_device)
        q_sims = torch.zeros(Q, n_neg, dtype=torch.float32, device=query_device)
        for qi in range(Q):
            nf = int(n_found_cpu[qi].item())
            if nf == 0:
                continue
            sel_pos = selected[qi].nonzero(as_tuple=True)[0][:n_neg]
            q_ids[qi,  :nf] = nn_seq_ids[qi, sel_pos].to(torch.int32)
            q_sims[qi, :nf] = nn_sims[qi, sel_pos]

    comp_ids = neg_ctx.ctx_seq_idx[comp_idx]
    comp_vals = neg_ctx.ctx_seq_val[comp_idx]
    comp_ids.zero_()
    comp_vals.zero_()
    active_js_cpu = active_js.cpu()
    comp_ids[active_js_cpu] = q_ids.cpu()
    comp_vals[active_js_cpu] = q_sims.cpu()

    timing["write"] = time.perf_counter() - t0
    return timing


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_neg_ctx(
    seq_repr: "SeqRepr",
    top_ctx:  "Context",
    mid_ctx:  "Context",
    neg_ctx:  "Context",
    *,
    selected_devices: Sequence[int | str | torch.device] | None = None,
) -> NegCtxStats:
    """
    Populate neg_ctx for all latents with sufficient PosCtx data.

    Each of the 36 SAE components is processed fully vectorised:
    no Python loops over the ~40K active latents per component.

    Returns a NegCtxStats instance with fill-rate distribution and timing.
    """
    backend = cast(str, config.latents.neg_ctx.backend or "single_gpu_exact")
    if backend == "multi_gpu_exact":
        return build_neg_ctx_multi_gpu(
            seq_repr,
            top_ctx,
            mid_ctx,
            neg_ctx,
            selected_devices=selected_devices,
        )
    if backend == "multi_gpu_index_sharded_exact":
        return build_neg_ctx_index_sharded(
            seq_repr,
            top_ctx,
            mid_ctx,
            neg_ctx,
            selected_devices=selected_devices,
        )

    return build_neg_ctx_single_gpu_exact(seq_repr, top_ctx, mid_ctx, neg_ctx)


def build_neg_ctx_single_gpu_exact(
    seq_repr: "SeqRepr",
    top_ctx:  "Context",
    mid_ctx:  "Context",
    neg_ctx:  "Context",
) -> NegCtxStats:
    """Single-device exact negative-context backend used as the correctness baseline."""

    llm_cfg = TuringLLMConfig()
    n_comp      = llm_cfg.n_layer * 3
    n_neg       = neg_ctx.num_ctx_sequences
    n_neighbors = cast(int, config.latents.neg_ctx.n_neighbors or 500)
    min_pos_ctx = cast(int, config.latents.neg_ctx.min_pos_ctx  or 8)

    stats   = NegCtxStats(backend="single_gpu_exact")
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Build ANN index                                                      #
    # ------------------------------------------------------------------ #

    t0           = time.perf_counter()
    total_n_seqs = seq_repr.n_seqs     # full dataset count (for stride + valid filter)
    n_stored     = seq_repr.n_stored   # actual ANN index size (≤ total_n_seqs)
    device       = _ann_device()
    stats.devices = [str(device)]
    stats.ann_device = str(device)
    stats.record_seq_repr(seq_repr)
    ann_memory_estimate = estimate_neg_ctx_ann_memory(
        seq_repr,
        query_chunk_size=4096 if device.type == "cuda" else 512,
    )
    guardrail = check_neg_ctx_memory_guardrail(
        device,
        ann_memory_estimate,
        fraction=float(config.latents.neg_ctx.memory_guardrail_fraction),
        fail_on_exceed=bool(config.latents.neg_ctx.fail_on_memory_guardrail),
    )
    _record_ann_memory_estimate(stats, ann_memory_estimate, guardrail)

    raw_vecs = seq_repr.repr_buf[1:n_stored + 1].float()   # [n_stored, D] float32, CPU
    index    = TorchANNIndex(raw_vecs, device=device)
    stats.t_index_build = time.perf_counter() - t0

    K = min(n_neighbors, n_stored)

    # Move slot-mapping tensors to compute device once (reused across components).
    # None when uncapped (slot == seq_id, no mapping needed).
    if seq_repr.is_capped and seq_repr.slot_to_id is not None and seq_repr.id_to_slot is not None:
        slot_to_id_d: torch.Tensor | None = seq_repr.slot_to_id.to(device)
        id_to_slot_d:  torch.Tensor | None = seq_repr.id_to_slot.to(device)
    else:
        slot_to_id_d = None
        id_to_slot_d = None

    # ------------------------------------------------------------------ #
    # Per-component loop — tqdm for progress; timing breakdown in postfix #
    # ------------------------------------------------------------------ #

    pbar = tqdm(range(n_comp), desc="  [neg_ctx]", unit="comp", leave=True)
    for comp_idx in pbar:
        tc0 = time.perf_counter()

        timing = _process_component(
            comp_idx, top_ctx, mid_ctx, neg_ctx,
            index, K, n_neg, min_pos_ctx, stats,
            total_n_seqs, slot_to_id_d, id_to_slot_d,
        )

        comp_s = time.perf_counter() - tc0
        stats.t_pos_collect += timing.get("pos",    0.0)
        stats.t_qmat_build  += timing.get("qmat",   0.0)
        stats.t_query       += timing.get("query",  0.0)
        stats.t_filter      += timing.get("filter", 0.0)
        stats.t_write       += timing.get("write",  0.0)

        active = stats.n_latents_attempted - stats.n_latents_skipped_low_pos
        pbar.set_postfix({
            "active": f"{active // (comp_idx + 1):,}",
            "ms":     f"{comp_s * 1000:.0f}",
            "pos_s":  f"{stats.t_pos_collect:.1f}",
            "qmat_s": f"{stats.t_qmat_build:.1f}",
            "srch_s": f"{stats.t_query:.1f}",
            "flt_s":  f"{stats.t_filter:.1f}",
        })

    pbar.close()
    stats.t_total = time.perf_counter() - t_start
    validate_neg_ctx_output(neg_ctx, total_n_seqs=total_n_seqs, n_sequences=n_neg)
    return stats


def build_neg_ctx_multi_gpu(
    seq_repr: "SeqRepr",
    top_ctx:  "Context",
    mid_ctx:  "Context",
    neg_ctx:  "Context",
    *,
    selected_devices: Sequence[int | str | torch.device] | None = None,
) -> NegCtxStats:
    """
    Component-parallel exact backend.

    Each selected GPU builds its own exact ANN index and owns a disjoint subset
    of SAE components. Component writes target disjoint neg_ctx slices, so the
    final artifact shape and semantics match the single-device backend.
    """
    llm_cfg = TuringLLMConfig()
    n_comp      = llm_cfg.n_layer * 3
    n_neg       = neg_ctx.num_ctx_sequences
    n_neighbors = cast(int, config.latents.neg_ctx.n_neighbors or 500)
    min_pos_ctx = cast(int, config.latents.neg_ctx.min_pos_ctx  or 8)

    configured_devices = (
        list(selected_devices)
        if selected_devices is not None
        else list(config.latents.neg_ctx.devices)
    )
    devices = parse_neg_ctx_devices(configured_devices)
    _validate_cuda_devices(devices)
    assignments = partition_components(n_comp, devices)

    print("  [neg_ctx] backend=multi_gpu_exact")
    for device in devices:
        comps = assignments[str(device)]
        if comps:
            print(f"  [neg_ctx] {device}: {len(comps)} components ({comps[0]}..{comps[-1]})")
        else:
            print(f"  [neg_ctx] {device}: 0 components")

    total_n_seqs = seq_repr.n_seqs
    n_stored     = seq_repr.n_stored
    raw_vecs     = seq_repr.repr_buf[1:n_stored + 1].float()
    K            = min(n_neighbors, n_stored)

    final_stats = NegCtxStats(
        backend="multi_gpu_exact",
        devices=[str(device) for device in devices],
        ann_device="multi_gpu_exact",
        component_assignments=assignments,
    )
    final_stats.record_seq_repr(seq_repr)
    ann_memory_estimate = estimate_neg_ctx_ann_memory(
        seq_repr,
        query_chunk_size=4096,
    )
    for device in devices:
        guardrail = check_neg_ctx_memory_guardrail(
            device,
            ann_memory_estimate,
            fraction=float(config.latents.neg_ctx.memory_guardrail_fraction),
            fail_on_exceed=bool(config.latents.neg_ctx.fail_on_memory_guardrail),
        )
        _record_ann_memory_estimate(final_stats, ann_memory_estimate, guardrail)
    t_start = time.perf_counter()

    def worker(device: torch.device, component_indices: list[int]) -> NegCtxStats:
        torch.cuda.set_device(device)
        worker_stats = NegCtxStats(
            backend="multi_gpu_exact",
            devices=[str(device)],
            ann_device=str(device),
        )
        worker_stats.record_seq_repr(seq_repr)
        _record_ann_memory_estimate(
            worker_stats,
            ann_memory_estimate,
            {
                "fraction": float(config.latents.neg_ctx.memory_guardrail_fraction),
                "limit_bytes": final_stats.ann_memory_guardrail_limit_bytes,
            },
        )

        t0 = time.perf_counter()
        index = TorchANNIndex(raw_vecs, device=device)
        worker_stats.t_index_build = time.perf_counter() - t0

        if seq_repr.is_capped and seq_repr.slot_to_id is not None and seq_repr.id_to_slot is not None:
            slot_to_id_d: torch.Tensor | None = seq_repr.slot_to_id.to(device)
            id_to_slot_d:  torch.Tensor | None = seq_repr.id_to_slot.to(device)
        else:
            slot_to_id_d = None
            id_to_slot_d = None

        for comp_idx in component_indices:
            tc0 = time.perf_counter()
            timing = _process_component(
                comp_idx, top_ctx, mid_ctx, neg_ctx,
                index, K, n_neg, min_pos_ctx, worker_stats,
                total_n_seqs, slot_to_id_d, id_to_slot_d,
            )
            comp_s = time.perf_counter() - tc0
            worker_stats.t_pos_collect += timing.get("pos",    0.0)
            worker_stats.t_qmat_build  += timing.get("qmat",   0.0)
            worker_stats.t_query       += timing.get("query",  0.0)
            worker_stats.t_filter      += timing.get("filter", 0.0)
            worker_stats.t_write       += timing.get("write",  0.0)
            print(f"  [neg_ctx:{device}] comp={comp_idx} {comp_s * 1000:.0f} ms")
        worker_stats.per_device_timing_ms[str(device)] = {
            "index_build_ms": round(worker_stats.t_index_build * 1000, 1),
            "pos_collect_ms": round(worker_stats.t_pos_collect * 1000, 1),
            "qmat_build_ms": round(worker_stats.t_qmat_build * 1000, 1),
            "query_ms": round(worker_stats.t_query * 1000, 1),
            "filter_ms": round(worker_stats.t_filter * 1000, 1),
            "write_ms": round(worker_stats.t_write * 1000, 1),
        }
        return worker_stats

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(worker, device, assignments[str(device)])
            for device in devices
            if assignments[str(device)]
        ]
        for future in as_completed(futures):
            final_stats.merge_from(future.result())

    final_stats.t_total = time.perf_counter() - t_start
    validate_neg_ctx_output(neg_ctx, total_n_seqs=total_n_seqs, n_sequences=n_neg)
    return final_stats


def build_neg_ctx_index_sharded(
    seq_repr: "SeqRepr",
    top_ctx:  "Context",
    mid_ctx:  "Context",
    neg_ctx:  "Context",
    *,
    selected_devices: Sequence[int | str | torch.device] | None = None,
) -> NegCtxStats:
    """
    Exact backend that shards seq_repr rows across CUDA devices.

    Every query is searched against every shard, then shard-local top-K rows are
    merged into global slot IDs before applying the standard positive filter.
    """

    llm_cfg = TuringLLMConfig()
    n_comp      = llm_cfg.n_layer * 3
    n_neg       = neg_ctx.num_ctx_sequences
    n_neighbors = cast(int, config.latents.neg_ctx.n_neighbors or 500)
    min_pos_ctx = cast(int, config.latents.neg_ctx.min_pos_ctx  or 8)

    configured_devices = (
        list(selected_devices)
        if selected_devices is not None
        else list(config.latents.neg_ctx.devices)
    )
    devices = parse_neg_ctx_devices(configured_devices)
    _validate_cuda_devices(devices)

    total_n_seqs = seq_repr.n_seqs
    n_stored     = seq_repr.n_stored
    raw_vecs     = seq_repr.repr_buf[1:n_stored + 1].float()
    K            = min(n_neighbors, n_stored)
    query_device = devices[0]
    shard_assignments = partition_index_slots(n_stored, devices)

    print("  [neg_ctx] backend=multi_gpu_index_sharded_exact")
    for device in devices:
        start, end = shard_assignments[str(device)]
        print(f"  [neg_ctx] {device}: index slots [{start}, {end}) ({end - start:,} rows)")

    final_stats = NegCtxStats(
        backend="multi_gpu_index_sharded_exact",
        devices=[str(device) for device in devices],
        ann_device="multi_gpu_index_sharded_exact",
        index_shard_assignments={
            str(device): {"start_slot": start, "end_slot": end, "n_rows": end - start}
            for device, (start, end) in (
                (device, shard_assignments[str(device)]) for device in devices
            )
        },
    )
    final_stats.record_seq_repr(seq_repr)

    largest_estimate: dict[str, int] | None = None
    largest_guardrail: dict[str, object] | None = None
    for device in devices:
        start, end = shard_assignments[str(device)]
        shard_estimate = estimate_neg_ctx_ann_memory_for_shape(
            n_stored=end - start,
            n_seqs=total_n_seqs,
            repr_dim=seq_repr.repr_dim,
            is_capped=seq_repr.is_capped,
            query_chunk_size=4096,
        )
        final_stats.ann_shard_memory_estimates[str(device)] = shard_estimate
        guardrail = check_neg_ctx_memory_guardrail(
            device,
            shard_estimate,
            fraction=float(config.latents.neg_ctx.memory_guardrail_fraction),
            fail_on_exceed=bool(config.latents.neg_ctx.fail_on_memory_guardrail),
        )
        if largest_estimate is None or shard_estimate["total_bytes"] > largest_estimate["total_bytes"]:
            largest_estimate = shard_estimate
            largest_guardrail = guardrail
    if largest_estimate is not None and largest_guardrail is not None:
        _record_ann_memory_estimate(final_stats, largest_estimate, largest_guardrail)

    t_start = time.perf_counter()
    t0 = time.perf_counter()
    index = ShardedANNIndex(raw_vecs, devices)
    final_stats.t_index_build = time.perf_counter() - t0

    if seq_repr.is_capped and seq_repr.slot_to_id is not None and seq_repr.id_to_slot is not None:
        slot_to_id_d: torch.Tensor | None = seq_repr.slot_to_id.to(query_device)
        id_to_slot_d:  torch.Tensor | None = seq_repr.id_to_slot.to(query_device)
    else:
        slot_to_id_d = None
        id_to_slot_d = None

    pbar = tqdm(range(n_comp), desc="  [neg_ctx:sharded]", unit="comp", leave=True)
    for comp_idx in pbar:
        tc0 = time.perf_counter()
        timing = _process_component_sharded(
            comp_idx, top_ctx, mid_ctx, neg_ctx, seq_repr,
            index, K, n_neg, min_pos_ctx, final_stats,
            total_n_seqs, query_device, slot_to_id_d, id_to_slot_d,
        )
        comp_s = time.perf_counter() - tc0
        final_stats.t_pos_collect += timing.get("pos",    0.0)
        final_stats.t_qmat_build  += timing.get("qmat",   0.0)
        final_stats.t_query       += timing.get("query",  0.0)
        final_stats.t_filter      += timing.get("filter", 0.0)
        final_stats.t_write       += timing.get("write",  0.0)
        pbar.set_postfix({
            "ms":     f"{comp_s * 1000:.0f}",
            "qmat_s": f"{final_stats.t_qmat_build:.1f}",
            "srch_s": f"{final_stats.t_query:.1f}",
            "flt_s":  f"{final_stats.t_filter:.1f}",
        })
    pbar.close()

    final_stats.t_total = time.perf_counter() - t_start
    validate_neg_ctx_output(neg_ctx, total_n_seqs=total_n_seqs, n_sequences=n_neg)
    return final_stats
