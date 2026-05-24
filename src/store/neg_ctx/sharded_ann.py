"""Sharded exact ANN support for negative-context retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .ann import TorchANNIndex


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
    def search(
        self,
        queries: torch.Tensor,
        k: int,
        *,
        merge_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shard_results: list[tuple[torch.Tensor, torch.Tensor]] = []
        for shard in self.shards:
            shard_k = min(k, shard.end_slot - shard.start_slot)
            if shard_k <= 0:
                continue
            sims, local_idxs = shard.index.search(queries, shard_k)
            global_idxs = local_idxs + shard.start_slot
            shard_results.append((sims, global_idxs))
        return merge_shard_search_results(shard_results, k=k, device=merge_device)


__all__ = [
    "ANNIndexShard",
    "ShardedANNIndex",
    "merge_shard_search_results",
    "partition_index_slots",
]
