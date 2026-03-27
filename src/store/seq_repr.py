"""
Per-sequence representation store.

Captures a single pooled vector per sequence during Pass 1 (from the last
transformer layer's residual stream) and persists it as a permanent artifact.
Used by the ANN step to build the TorchANNIndex for neg_ctx retrieval.

Two pooling modes are supported, selected by config.latents.neg_ctx.repr_mode:
  "mean_pool"  — average residual stream over all token positions (default)
  "last_token" — residual stream at the final token position only

Capacity cap (config.latents.neg_ctx.max_repr_seqs):
  When max_repr_seqs < n_seqs, a random subset of sequence IDs is pre-selected
  at init time and only those are stored. The selection is uniform without
  replacement, so cluster density is preserved in proportion to the true
  distribution — important for neg_ctx nearest-neighbour quality.

  Two slot-mapping tensors are maintained:
    slot_to_id[slot]  → seq_id   (1-indexed; slot 0 = sentinel 0)
    id_to_slot[seq_id] → slot    (0 = sequence not in store)

  repr_buf is indexed by slot, not by seq_id, keeping it compact at
  [n_stored + 1, repr_dim] regardless of total dataset size.

  When uncapped (null or >= n_seqs), slot == seq_id and these tensors are None.
"""

from __future__ import annotations

import time
from typing import Optional, cast

import torch

from config import config
from model.turingllm import TuringLLMConfig


class SeqRepr:

    def __init__(self, n_seqs: int, device: Optional[torch.device] = None):
        """
        Args:
            n_seqs: Total number of unique sequences in the dataset.
                    Sequence IDs are 1-indexed; index 0 is an unused sentinel.
        """
        self.llm_config = TuringLLMConfig()
        self.repr_dim   = self.llm_config.n_embd   # 1024
        self.n_seqs     = n_seqs
        self.repr_mode  = cast(str, config.latents.neg_ctx.repr_mode or "mean_pool")

        _max_cfg = config.latents.neg_ctx.max_repr_seqs
        _max     = int(_max_cfg) if _max_cfg is not None else None

        self.is_capped = (_max is not None) and (_max < n_seqs)
        self.n_stored  = min(_max, n_seqs) if _max is not None else n_seqs

        if self.is_capped:
            # Pre-select a uniform random sample of seq IDs (without replacement).
            # Sorted so slot indices are stable and compact.
            kept = (torch.randperm(n_seqs)[: self.n_stored].sort().values + 1)  # [n_stored] 1-indexed

            # slot_to_id[slot] = seq_id  (slot 0 = sentinel 0)
            self.slot_to_id = torch.zeros(self.n_stored + 1, dtype=torch.int64)
            self.slot_to_id[1:] = kept.long()

            # id_to_slot[seq_id] = slot  (0 = not stored)
            self.id_to_slot = torch.zeros(n_seqs + 1, dtype=torch.int32)
            self.id_to_slot[kept] = torch.arange(1, self.n_stored + 1, dtype=torch.int32)

            buf_size = self.n_stored + 1
            pct = self.n_stored / n_seqs * 100
            print(f"  [seq_repr] Capped at {self.n_stored:,} / {n_seqs:,} sequences "
                  f"(uniform random sample, {pct:.1f}%)")
        else:
            self.slot_to_id = None   # identity: slot == seq_id
            self.id_to_slot = None
            buf_size = n_seqs + 1

        # Always on CPU — only extracted once per batch at one layer.
        # float16 for compact storage (2 bytes × repr_dim × buf_size).
        self.repr_buf = torch.zeros((buf_size, self.repr_dim), dtype=torch.float16)

        self._n_updates = 0
        self._total_ms  = 0.0

    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, seq_ids: torch.Tensor, resid: torch.Tensor) -> None:
        """
        Store pooled residual stream representations for a batch of sequences.

        Args:
            seq_ids: [B] int32/int64 — global sequence IDs (1-indexed).
            resid:   [B, T, D] float32/bfloat16 — last-layer residual stream,
                     where T = sequence length and D = repr_dim (1024).
        """
        t0 = time.perf_counter()

        # OLD METHOD (Commented out for easy revert)
        # if self.repr_mode == "last_token":
        #     pooled = resid[:, -1, :].float()
        # else:
        #     pooled = resid.float().mean(dim=1)
        # pooled_h = pooled.half().cpu()
        # ids      = seq_ids.long().cpu()
        # if self.is_capped:
        #     valid = (ids >= 1) & (ids <= self.n_seqs)
        #     if valid.any():
        #         valid_ids = ids[valid]
        #         slots     = self.id_to_slot[valid_ids].long()   # 0 = not in store
        #         in_store  = slots > 0
        #         if in_store.any():
        #             self.repr_buf[slots[in_store]] = pooled_h[valid][in_store]
        #             self._n_updates += int(in_store.sum().item())
        # else:
        #     valid = (ids >= 1) & (ids <= self.n_seqs)
        #     if valid.any():
        #         self.repr_buf[ids[valid]] = pooled_h[valid]
        #         self._n_updates += int(valid.sum().item())

        # NEW IMPROVED METHOD (VRAM-safe on GPU)
        # 1. Pool and cast to float16 on GPU (reduces transfer size)
        if self.repr_mode == "last_token":
            pooled_gpu = resid[:, -1, :].to(torch.float16)
        else:
            pooled_gpu = resid.to(torch.float32).mean(dim=1).to(torch.float16)
        
        # 2. Move pooled results to CPU once
        pooled_cpu = pooled_gpu.cpu()
        ids_cpu = seq_ids.cpu().long()

        # Bounds-check all IDs at once
        valid_mask = (ids_cpu >= 1) & (ids_cpu <= self.n_seqs)
        
        if valid_mask.any():
            valid_ids = ids_cpu[valid_mask]
            valid_pooled = pooled_cpu[valid_mask]

            if self.is_capped:
                # 3. Use id_to_slot as a direct lookup (VRAM-safe on CPU)
                # id_to_slot[valid_ids] gives us the slot for every ID in the batch.
                # IDs not in store will have slot 0 (sentinel).
                slots = self.id_to_slot[valid_ids].long()
                in_store_mask = slots > 0
                
                if in_store_mask.any():
                    active_slots = slots[in_store_mask]
                    self.repr_buf[active_slots] = valid_pooled[in_store_mask]
                    self._n_updates += int(active_slots.shape[0])
            else:
                # Uncapped path
                self.repr_buf[valid_ids] = valid_pooled
                self._n_updates += int(valid_ids.shape[0])

        self._total_ms += (time.perf_counter() - t0) * 1000.0

    def get_repr(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns float32 representations for the requested sequence IDs.

        Args:
            seq_ids: [N] int — sequence IDs (1-indexed).

        Returns:
            [N, repr_dim] float32. Rows for seq_ids not in the store are zeros.
        """
        ids = seq_ids.long().cpu()
        if self.is_capped:
            slots = self.id_to_slot[ids].long()
            return self.repr_buf[slots].float()
        return self.repr_buf[ids].float()

    # ------------------------------------------------------------------

    def print_stats(self) -> None:
        filled  = int((self.repr_buf[1:].abs().sum(dim=1) > 0).sum().item())
        rate    = self._total_ms / max(1, self._n_updates) * 1000  # µs/seq
        cap_str = f" (capped from {self.n_seqs:,})" if self.is_capped else ""
        mb      = self.repr_buf.numel() * 2 / 1024 ** 2
        print(f"  [seq_repr] {filled:,} / {self.n_stored:,}{cap_str} sequences filled "
              f"| {mb:.0f} MB | mode={self.repr_mode} | repr_dim={self.repr_dim} "
              f"| update time: {self._total_ms:.1f} ms total  ({rate:.2f} µs/seq)")

    def save(self, path: str) -> None:
        data: dict = {
            "repr_buf":  self.repr_buf,
            "repr_mode": self.repr_mode,
            "repr_dim":  self.repr_dim,
            "n_seqs":    self.n_seqs,
            "n_stored":  self.n_stored,
            "is_capped": self.is_capped,
        }
        if self.is_capped:
            data["slot_to_id"] = self.slot_to_id
            data["id_to_slot"] = self.id_to_slot
        torch.save(data, path)

    def load(self, path: str) -> None:
        ckpt           = torch.load(path, map_location="cpu")
        self.repr_buf  = ckpt["repr_buf"]
        self.repr_mode = ckpt["repr_mode"]
        self.repr_dim  = ckpt["repr_dim"]
        self.n_seqs    = ckpt["n_seqs"]
        self.n_stored  = ckpt.get("n_stored", self.n_seqs)
        self.is_capped = ckpt.get("is_capped", False)
        if self.is_capped:
            self.slot_to_id = ckpt["slot_to_id"]
            self.id_to_slot = ckpt["id_to_slot"]
        else:
            self.slot_to_id = None
            self.id_to_slot = None


# Module-level singleton — n_seqs filled in by pipeline after DataLoader init.
seq_repr: Optional[SeqRepr] = None
