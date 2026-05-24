"""Exact ANN index and memory guardrails for negative-context retrieval."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .stats import NegCtxStats

if TYPE_CHECKING:
    from store.seq_repr import SeqRepr


class TorchANNIndex:
    """
    Exact cosine similarity index.

    Build: L2-normalise vecs -> store on device.
    Search: F.normalize(queries) @ index.T -> topk  (cuBLAS / MKL).

    Results are returned on the same device as the index (no CPU round-trip),
    so downstream GPU filtering can start immediately.
    """

    def __init__(self, vecs: torch.Tensor, device: torch.device):
        t0 = time.perf_counter()
        self.device = device
        self.n = vecs.shape[0]
        self.dim = vecs.shape[1]

        # chunk_size: larger = fewer kernel launches = better GPU utilisation.
        # Peak intermediate tensor = chunk_size x N x 4 B.
        # GPU:  4096 x 65536 x 4 = 1 GB - safe for 16 GB+ VRAM.
        # CPU:  512  x 65536 x 4 = 128 MB - conservative.
        self.chunk_size = 4096 if device.type == "cuda" else 512

        self.index = F.normalize(vecs.float(), dim=1).to(device)

        build_ms = (time.perf_counter() - t0) * 1000
        print(
            f"  [neg_ctx] ANN index on {device} "
            f"- {self.n:,} vecs x {self.dim} dims "
            f"- built in {build_ms:.1f} ms "
            f"- chunk_size={self.chunk_size}"
        )

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
        K = min(k, self.n)
        q_norm = F.normalize(queries.float(), dim=1)
        if q_norm.device != self.device:
            q_norm = q_norm.to(self.device)

        Q = q_norm.shape[0]
        all_sims = torch.empty(Q, K, dtype=torch.float32, device=self.device)
        all_idxs = torch.empty(Q, K, dtype=torch.int64, device=self.device)

        for start in range(0, Q, self.chunk_size):
            end = min(start + self.chunk_size, Q)
            sims = q_norm[start:end] @ self.index.T                   # [C, N]
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


__all__ = [
    "TorchANNIndex",
    "_record_ann_memory_estimate",
    "check_neg_ctx_memory_guardrail",
    "estimate_neg_ctx_ann_memory",
    "estimate_neg_ctx_ann_memory_for_shape",
]
