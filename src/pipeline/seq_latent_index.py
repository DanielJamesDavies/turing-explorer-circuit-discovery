import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


class SeqLatentIndexAccumulator:
    """
    Accumulates (sequence_id, latent_id) pairs for the top-K active latents
    per sequence per component during the first pass, then writes one
    ``outputs/seq_latent_index/shard_{i}.pt`` file per data shard.

    Each saved file is a ``Dict[int, torch.Tensor]`` mapping component index
    to an int32 tensor of shape ``[N, 2]`` where each row is
    ``[sequence_id, latent_id]``.

    Shard files are flushed incrementally as shard boundaries are crossed
    (call ``on_batch_complete`` after every batch) to keep peak memory low.
    Any remaining data is written by ``flush_all`` after the pass ends.
    """

    def __init__(
        self,
        shard_id_ranges: List[Tuple[int, int]],
        top_k_per_component: int,
        output_dir: str,
    ) -> None:
        self.shard_id_ranges = shard_id_ranges
        self.top_k = top_k_per_component
        self.output_dir = output_dir

        # Sorted array of shard start IDs for fast binary search.
        self._shard_starts = np.array(
            [r[0] for r in shard_id_ranges], dtype=np.int64
        )

        # shard_idx -> comp_idx -> list of [M*top_k, 2] int32 pair tensors.
        self._buffers: Dict[int, Dict[int, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        comp_idx: int,
        seq_ids: torch.Tensor,
        latents: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Record the top-K latent IDs per sequence for one component.

        Args:
            comp_idx:  Flat component index (layer * n_kinds + kind_idx).
            seq_ids:   Global sequence IDs for this batch, shape [B], int32.
            latents:   (top_acts [B,T,k], top_indices [B,T,k]) from the SAE.
        """
        top_acts, top_indices = latents  # [B, T, k_sae]
        B = seq_ids.shape[0]

        with torch.no_grad():
            # Flatten the T and k_sae dims so we can take a global top-K per
            # sequence across all token positions without a scatter-max over
            # the full d_sae vocabulary (which would be ~40 K entries × B).
            flat_acts = top_acts.detach().reshape(B, -1)    # [B, T*k_sae]
            flat_idx  = top_indices.detach().reshape(B, -1) # [B, T*k_sae]

            actual_k = min(self.top_k, flat_acts.shape[1])
            topk_pos = flat_acts.topk(actual_k, dim=-1).indices  # [B, actual_k]
            result_latents = (
                flat_idx.gather(-1, topk_pos).cpu().to(torch.int32)
            )  # [B, actual_k]

        seq_ids_cpu = seq_ids.cpu()  # [B], int32
        shard_idx_per_seq = self._shard_indices_for(
            seq_ids_cpu.numpy().astype(np.int64)
        )  # [B]

        for shard_idx in np.unique(shard_idx_per_seq):
            mask = shard_idx_per_seq == shard_idx
            s_seq = seq_ids_cpu[mask]      # [M]
            s_lat = result_latents[mask]   # [M, actual_k]
            M = s_seq.shape[0]

            seq_col = s_seq.unsqueeze(1).expand(M, actual_k).reshape(-1)  # [M*actual_k]
            lat_col = s_lat.reshape(-1)                                    # [M*actual_k]
            pairs = torch.stack([seq_col, lat_col], dim=1)                 # [M*actual_k, 2]

            self._buffers[int(shard_idx)][comp_idx].append(pairs)

    def on_batch_complete(self, batch_max_id: int) -> None:
        """
        Flush any shards whose last sequence ID is strictly less than
        ``batch_max_id``.  Call this after all component updates for a batch
        have been processed.

        Because global sequence IDs are assigned in strictly increasing order
        across shards, once we have processed a sequence ID beyond a shard's
        end, that shard is guaranteed to be fully accumulated.
        """
        for shard_idx, (start_id, end_id) in enumerate(self.shard_id_ranges):
            if end_id != -1 and batch_max_id > end_id and shard_idx in self._buffers:
                self._flush_shard(shard_idx)

    def flush_all(self) -> None:
        """Flush and save all remaining buffered shards.  Call after the pass."""
        for shard_idx in list(self._buffers.keys()):
            self._flush_shard(shard_idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shard_indices_for(self, seq_ids_np: np.ndarray) -> np.ndarray:
        """Return the shard index for each sequence ID (vectorised binary search)."""
        return np.searchsorted(self._shard_starts, seq_ids_np, side="right") - 1

    def _flush_shard(self, shard_idx: int) -> None:
        """Concatenate buffered pairs for each component and write to disk."""
        comp_data: Dict[int, torch.Tensor] = {}
        for comp_idx, tensor_list in self._buffers[shard_idx].items():
            if tensor_list:
                comp_data[comp_idx] = torch.cat(tensor_list, dim=0)

        path = os.path.join(self.output_dir, f"shard_{shard_idx}.pt")
        torch.save(comp_data, path)
        del self._buffers[shard_idx]
