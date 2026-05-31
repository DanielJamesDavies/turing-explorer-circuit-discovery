import importlib.util
import hashlib
import os
import sys
import torch
from typing import cast, Optional, Dict, List, Tuple

from config import config
from model.turingllm import TuringLLMConfig
from sae.topk_sae import SAEConfig
from store.utils import _AutoAllocTensor


def _load_mid_reservoir_ext():
    native_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "native"))
    try:
        for fname in os.listdir(native_dir):
            if fname.startswith("mid_reservoir") and fname.endswith(".so"):
                so_path = os.path.join(native_dir, fname)
                spec = importlib.util.spec_from_file_location("mid_reservoir", so_path)
                mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                spec.loader.exec_module(mod)                  # type: ignore[union-attr]
                return mod
    except Exception:
        pass
    return None


_mid_reservoir_ext = _load_mid_reservoir_ext()
_HAS_MID_RESERVOIR = _mid_reservoir_ext is not None
if _HAS_MID_RESERVOIR:
    print("[context] mid_reservoir extension loaded.")
else:
    print("[context] mid_reservoir extension not found — mid_ctx updates will be skipped. "
          "Build with: cd src/native && python setup.py build_ext --inplace")


def compute_seq_scores(
    top_acts: torch.Tensor,    # [batch, seq_len, k]
    top_indices: torch.Tensor, # [batch, seq_len, k]
    d_sae: int,
) -> torch.Tensor:
    """
    Returns [d_sae, batch] float32 mean activation score per latent per sequence.
    Shared between top_ctx and mid_ctx to avoid recomputing per-latent sequence scores.
    """
    batch, seq_len, _k = top_acts.shape
    scores = torch.zeros(batch, d_sae, device=top_acts.device, dtype=torch.float32)
    scores.scatter_add_(
        1,
        top_indices.reshape(batch, -1).long(),
        top_acts.reshape(batch, -1).float(),
    )
    scores /= seq_len
    return scores.T  # [d_sae, batch]


def _signed_int64_from_material(material: str) -> int:
    value = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    if value >= (1 << 63):
        value -= 1 << 64
    return value


MID_CTX_PRIORITY_HASH_VERSION = "splitmix64-v1"

# SplitMix64 constants, represented as signed int64 values so Torch int64
# arithmetic wraps the same way on CPU/GPU tensor operations.
_SPLITMIX_GOLDEN_GAMMA = -7046029254386353131  # unsigned: 0x9E3779B97F4A7C15
_SPLITMIX_MIX_1 = -4658895280553007687        # unsigned: 0xBF58476D1CE4E5B9
_SPLITMIX_MIX_2 = -7723592293110705685        # unsigned: 0x94D049BB133111EB


class Context:

    ctx_seq_idx    = _AutoAllocTensor()
    ctx_seq_val    = _AutoAllocTensor()
    reservoir_fill = _AutoAllocTensor()
    reservoir_n    = _AutoAllocTensor()

    def __init__(self, ctx_type: str, device: Optional[torch.device] = None):
        self.ctx_type = ctx_type  # "top" | "mid" | "neg"
        self.llm_config = TuringLLMConfig()
        self.sae_config = SAEConfig()
        self.num_components = self.llm_config.n_layer * 3
        self.d_sae = self.sae_config.d_sae

        if ctx_type == "mid":
            self.device = torch.device("cpu")
            self.num_ctx_sequences = cast(int, config.latents.mid_ctx.n_sequences or 64)
            self.mid_mode = cast(str, config.latents.mid_ctx.mode or "reservoir_cpu")
            self._band_low  = cast(float, config.latents.mid_ctx.band_low_sigma  or 0.5)
            self._band_high = cast(float, config.latents.mid_ctx.band_high_sigma or 1.5)
            self._priority_seed = cast(int, config.distributed.sampling_seed or 0)
            self.val_dtype = torch.float32
        elif ctx_type == "neg":
            self.device = torch.device("cpu")
            self.num_ctx_sequences = cast(int, config.latents.neg_ctx.n_sequences or 64)
            self.mid_mode = ""
            self._priority_seed = 0
            self.val_dtype = torch.float32
        else:
            self.device = device if device is not None else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.num_ctx_sequences = cast(int, config.latents.top_ctx.n_sequences)
            self.mid_mode = ""
            self._priority_seed = 0
            self.val_dtype = torch.bfloat16

        self._allocated = False

    def allocate(self, device: Optional[torch.device] = None) -> None:
        if self._allocated:
            if device is not None and device != self.device:
                self.set_device(device)
            return

        # mid_ctx and neg_ctx stores stay on CPU so downstream readers and neg_ctx
        # can use one artifact shape regardless of update backend.
        if self.ctx_type in ("mid", "neg"):
            device = None

        if device is not None:
            self.device = device

        self.ctx_seq_idx = torch.zeros(
            (self.num_components, self.d_sae, self.num_ctx_sequences),
            dtype=torch.int32, device=self.device,
        )
        self.ctx_seq_val = torch.zeros(
            (self.num_components, self.d_sae, self.num_ctx_sequences),
            dtype=self.val_dtype, device=self.device,
        )

        if self.ctx_type == "mid":
            self.reservoir_fill = torch.zeros(
                (self.num_components, self.d_sae), dtype=torch.int32,
            )
            self.reservoir_n = torch.zeros(
                (self.num_components, self.d_sae), dtype=torch.int64,
            )
            if self.mid_mode == "gpu_priority_reservoir":
                self._priority_val = torch.full(
                    (self.num_components, self.d_sae, self.num_ctx_sequences),
                    torch.iinfo(torch.int64).max,
                    dtype=torch.int64,
                )
        self._allocated = True

    # ------------------------------------------------------------------
    # Public update entry point
    # ------------------------------------------------------------------

    def update_component(
        self,
        component_idx: int,
        sequence_indices: torch.Tensor,
        latents: tuple[torch.Tensor, torch.Tensor],
        latent_mean_seq: Optional[torch.Tensor] = None,
        latent_std_seq: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update stored contexts for one SAE component over a batch.

        For "top": latent_mean_seq / latent_std_seq are ignored.
        For "mid": both must be supplied (shape [d_sae], any device).
                   These are per-sequence-score statistics from LatentStats
                   (mean_seq / std_seq), which live in the same value range
                   as compute_seq_scores() so the band is correctly calibrated.
        """
        self.allocate(sequence_indices.device if sequence_indices.is_cuda else None)
        if self.ctx_type == "top":
            self._update_top(component_idx, sequence_indices, latents)
        elif self.ctx_type == "mid":
            if latent_mean_seq is None or latent_std_seq is None:
                raise ValueError(
                    "mid_ctx.update_component requires latent_mean_seq and latent_std_seq"
                )
            if self.mid_mode == "gpu_topk_mid":
                self._update_mid_gpu_topk(component_idx, sequence_indices, latents, latent_mean_seq, latent_std_seq)
            elif self.mid_mode == "gpu_priority_reservoir":
                self._update_mid_gpu_priority_reservoir(
                    component_idx,
                    sequence_indices,
                    latents,
                    latent_mean_seq,
                    latent_std_seq,
                )
            else:
                self._update_mid_reservoir(component_idx, sequence_indices, latents, latent_mean_seq, latent_std_seq)
        else:
            raise ValueError(f"Invalid context type: {self.ctx_type}")

    # ------------------------------------------------------------------
    # Top-N context (highest mean-activation sequences per latent)
    # ------------------------------------------------------------------

    def _update_top(
        self,
        component_idx: int,
        sequence_indices: torch.Tensor,
        latents: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            top_acts, top_indices = latents
            scores = compute_seq_scores(top_acts, top_indices, self.d_sae)  # [d_sae, B]

            new_indices = sequence_indices.unsqueeze(0).expand(self.d_sae, -1).to(self.device)

            ctx_seq_idx_temp = torch.cat([self.ctx_seq_idx[component_idx], new_indices], dim=1)
            ctx_seq_val_temp = torch.cat(
                [self.ctx_seq_val[component_idx], scores.to(self.ctx_seq_val.dtype)], dim=1
            )

            topk_values, topk_indices = torch.topk(ctx_seq_val_temp, k=self.num_ctx_sequences, dim=1)
            self.ctx_seq_idx[component_idx] = ctx_seq_idx_temp.gather(1, topk_indices)
            self.ctx_seq_val[component_idx] = topk_values

    # ------------------------------------------------------------------
    # Mid context (reservoir-sampled sequences in the mid activation band)
    # ------------------------------------------------------------------

    def _mid_band_bounds(
        self,
        latent_mean_seq: torch.Tensor,
        latent_std_seq: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_seq = latent_mean_seq.to(device)
        std_seq = latent_std_seq.to(device).clamp(min=1e-6)
        low = mean_seq + self._band_low * std_seq
        high = mean_seq + self._band_high * std_seq
        midpoint = mean_seq + ((self._band_low + self._band_high) * 0.5) * std_seq
        return low, high, midpoint

    def _update_mid_reservoir(
        self,
        component_idx: int,
        sequence_indices: torch.Tensor,
        latents: tuple[torch.Tensor, torch.Tensor],
        latent_mean_seq: torch.Tensor,  # [d_sae], any device — per-sequence score mean
        latent_std_seq: torch.Tensor,   # [d_sae], any device — per-sequence score std
    ) -> None:
        if not _HAS_MID_RESERVOIR:
            return

        with torch.no_grad():
            top_acts, top_indices = latents
            compute_device = top_acts.device

            scores = compute_seq_scores(top_acts, top_indices, self.d_sae)  # [d_sae, B]

            # Band bounds in per-sequence-score space.
            # mean_seq / std_seq come from LatentStats.mean_seq / std_seq, which track
            # the distribution of compute_seq_scores() values — the same space as scores.
            # A floor on std prevents a degenerate zero-width band during warmup.
            low, high, _midpoint = self._mid_band_bounds(latent_mean_seq, latent_std_seq, compute_device)

            # In-band mask: only positions where the latent fired with a
            # mean score in the mid band (score == 0 means the latent did
            # not fire in this sequence, which is always below the band).
            in_band = (scores > low.unsqueeze(1)) & (scores < high.unsqueeze(1))  # [d_sae, B]

            pairs = in_band.nonzero()  # [N_pairs, 2]: (latent_idx, batch_idx)
            if pairs.numel() == 0:
                return

            lat_idxs   = pairs[:, 0]
            bat_idxs   = pairs[:, 1]
            seq_ids_d  = sequence_indices.to(compute_device)[bat_idxs]
            pair_scores = scores[lat_idxs, bat_idxs]

            # Sort by latent index so C++ can scan groups in one pass.
            order = torch.argsort(lat_idxs)
            lat_sorted   = lat_idxs[order].cpu().to(torch.int32).contiguous()
            seq_sorted   = seq_ids_d[order].cpu().to(torch.int32).contiguous()
            score_sorted = pair_scores[order].cpu().to(torch.float32).contiguous()

            _mid_reservoir_ext.reservoir_update(
                lat_sorted,
                seq_sorted,
                score_sorted,
                self.ctx_seq_idx[component_idx].contiguous(),    # [d_sae, N_mid] int32  CPU
                self.ctx_seq_val[component_idx].contiguous(),    # [d_sae, N_mid] float32 CPU
                self.reservoir_fill[component_idx].contiguous(), # [d_sae] int32  CPU
                self.reservoir_n[component_idx].contiguous(),    # [d_sae] int64  CPU
                self.num_ctx_sequences,
            )

    def _update_mid_gpu_topk(
        self,
        component_idx: int,
        sequence_indices: torch.Tensor,
        latents: tuple[torch.Tensor, torch.Tensor],
        latent_mean_seq: torch.Tensor,
        latent_std_seq: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            top_acts, top_indices = latents
            compute_device = top_acts.device
            scores = compute_seq_scores(top_acts, top_indices, self.d_sae)  # [d_sae, B]
            low, high, midpoint = self._mid_band_bounds(latent_mean_seq, latent_std_seq, compute_device)

            existing_idx = self.ctx_seq_idx[component_idx].to(compute_device)
            existing_val = self.ctx_seq_val[component_idx].to(compute_device).float()
            existing_valid = (
                (existing_idx != 0)
                & (existing_val > low.unsqueeze(1))
                & (existing_val < high.unsqueeze(1))
            )

            new_idx = sequence_indices.to(compute_device).unsqueeze(0).expand(self.d_sae, -1)
            new_valid = (scores > low.unsqueeze(1)) & (scores < high.unsqueeze(1))

            candidate_idx = torch.cat([existing_idx, new_idx], dim=1)
            candidate_val = torch.cat([existing_val, scores], dim=1)
            candidate_valid = torch.cat([existing_valid, new_valid], dim=1)

            distance = (candidate_val - midpoint.unsqueeze(1)).abs()
            distance = distance.masked_fill(~candidate_valid, float("inf"))
            selected_distance, selected_pos = torch.topk(
                -distance,
                k=self.num_ctx_sequences,
                dim=1,
                largest=True,
                sorted=True,
            )
            selected_valid = torch.isfinite(-selected_distance)
            selected_idx = candidate_idx.gather(1, selected_pos).to(torch.int32)
            selected_val = candidate_val.gather(1, selected_pos).to(torch.float32)

            selected_idx = selected_idx.masked_fill(~selected_valid, 0)
            selected_val = selected_val.masked_fill(~selected_valid, 0.0)

            self.ctx_seq_idx[component_idx].copy_(selected_idx.cpu())
            self.ctx_seq_val[component_idx].copy_(selected_val.cpu().to(self.ctx_seq_val.dtype))
            self.reservoir_fill[component_idx].copy_(selected_valid.sum(dim=1).cpu().to(torch.int32))
            self.reservoir_n[component_idx] += new_valid.sum(dim=1).cpu().to(torch.int64)

    def _update_mid_gpu_priority_reservoir(
        self,
        component_idx: int,
        sequence_indices: torch.Tensor,
        latents: tuple[torch.Tensor, torch.Tensor],
        latent_mean_seq: torch.Tensor,
        latent_std_seq: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            top_acts, top_indices = latents
            compute_device = top_acts.device
            scores = compute_seq_scores(top_acts, top_indices, self.d_sae)  # [d_sae, B]
            low, high, _midpoint = self._mid_band_bounds(latent_mean_seq, latent_std_seq, compute_device)

            existing_idx = self.ctx_seq_idx[component_idx].to(compute_device)
            existing_val = self.ctx_seq_val[component_idx].to(compute_device).float()
            existing_priority = self._priority_val[component_idx].to(compute_device)
            existing_valid = (
                (existing_idx != 0)
                & (existing_priority != torch.iinfo(torch.int64).max)
            )

            new_valid = (scores > low.unsqueeze(1)) & (scores < high.unsqueeze(1))
            pairs = new_valid.nonzero()
            if pairs.numel() == 0:
                return

            new_latents = pairs[:, 0]
            new_batch_positions = pairs[:, 1]
            sequence_ids_d = sequence_indices.to(compute_device)
            new_sequences = sequence_ids_d[new_batch_positions]
            new_scores = scores[new_latents, new_batch_positions]
            new_priorities = self._mid_priority_values_for_candidates(
                component_idx,
                new_latents,
                new_sequences,
                compute_device,
            )

            new_counts = torch.bincount(new_latents, minlength=self.d_sae).cpu().to(torch.int64)
            self.reservoir_n[component_idx] += new_counts

            invalid_priority = torch.iinfo(torch.int64).max
            for latent_idx in torch.unique(new_latents).tolist():
                latent = int(latent_idx)
                existing_latent_valid = existing_valid[latent]
                new_latent_mask = new_latents == latent

                candidate_idx = torch.cat(
                    [
                        existing_idx[latent, existing_latent_valid],
                        new_sequences[new_latent_mask].to(existing_idx.dtype),
                    ],
                    dim=0,
                )
                candidate_val = torch.cat(
                    [
                        existing_val[latent, existing_latent_valid],
                        new_scores[new_latent_mask].to(torch.float32),
                    ],
                    dim=0,
                )
                candidate_priority = torch.cat(
                    [
                        existing_priority[latent, existing_latent_valid],
                        new_priorities[new_latent_mask],
                    ],
                    dim=0,
                )

                order_by_sequence = torch.argsort(candidate_idx.to(torch.long), stable=True)
                candidate_idx = candidate_idx[order_by_sequence]
                candidate_val = candidate_val[order_by_sequence]
                candidate_priority = candidate_priority[order_by_sequence]

                selected_count = min(int(candidate_priority.numel()), self.num_ctx_sequences)
                selected_priority, selected_pos = torch.topk(
                    candidate_priority,
                    k=selected_count,
                    largest=False,
                    sorted=True,
                )
                selected_idx = candidate_idx[selected_pos].to(torch.int32)
                selected_val = candidate_val[selected_pos].to(torch.float32)

                self.ctx_seq_idx[component_idx, latent].zero_()
                self.ctx_seq_val[component_idx, latent].zero_()
                self._priority_val[component_idx, latent].fill_(invalid_priority)
                self.ctx_seq_idx[component_idx, latent, :selected_count].copy_(selected_idx.cpu())
                self.ctx_seq_val[component_idx, latent, :selected_count].copy_(
                    selected_val.cpu().to(self.ctx_seq_val.dtype)
                )
                self._priority_val[component_idx, latent, :selected_count].copy_(
                    selected_priority.cpu().to(torch.int64)
                )
                self.reservoir_fill[component_idx, latent] = selected_count

    def _mid_priority_values(
        self,
        component_idx: int,
        sequence_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        final_band_low = float(getattr(self, "_final_band_low", self._band_low))
        final_band_high = float(getattr(self, "_final_band_high", self._band_high))
        final_num_ctx_sequences = int(
            getattr(self, "_final_num_ctx_sequences", self.num_ctx_sequences)
        )
        material = "|".join(
            [
                MID_CTX_PRIORITY_HASH_VERSION,
                str(int(self._priority_seed)),
                "mid_ctx",
                str(getattr(self, "_candidate_pool_dataset_fingerprint", "")),
                str(final_band_low),
                str(final_band_high),
                str(float(self._band_low)),
                str(float(self._band_high)),
                str(float(getattr(self, "_candidate_band_margin", 0.0))),
                str(final_num_ctx_sequences),
            ]
        )
        base = _signed_int64_from_material(material)
        seq_ids = sequence_indices.to(device=device, dtype=torch.int64).unsqueeze(0)
        latent_ids = torch.arange(self.d_sae, device=device, dtype=torch.int64).unsqueeze(1)
        component_ids = torch.full((1, 1), int(component_idx), device=device, dtype=torch.int64)
        values = seq_ids * _SPLITMIX_MIX_1
        values = values + latent_ids * _SPLITMIX_MIX_2
        values = values + component_ids * _SPLITMIX_GOLDEN_GAMMA
        values = values + base
        values = self._splitmix64(values)
        return torch.bitwise_and(values, 0x7FFFFFFFFFFFFFFF)

    def _mid_priority_values_for_candidates(
        self,
        component_idx: int,
        latent_indices: torch.Tensor,
        sequence_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        final_band_low = float(getattr(self, "_final_band_low", self._band_low))
        final_band_high = float(getattr(self, "_final_band_high", self._band_high))
        final_num_ctx_sequences = int(
            getattr(self, "_final_num_ctx_sequences", self.num_ctx_sequences)
        )
        material = "|".join(
            [
                MID_CTX_PRIORITY_HASH_VERSION,
                str(int(self._priority_seed)),
                "mid_ctx",
                str(getattr(self, "_candidate_pool_dataset_fingerprint", "")),
                str(final_band_low),
                str(final_band_high),
                str(float(self._band_low)),
                str(float(self._band_high)),
                str(float(getattr(self, "_candidate_band_margin", 0.0))),
                str(final_num_ctx_sequences),
            ]
        )
        base = _signed_int64_from_material(material)
        seq_ids = sequence_indices.to(device=device, dtype=torch.int64)
        latent_ids = latent_indices.to(device=device, dtype=torch.int64)
        component_ids = torch.full_like(seq_ids, int(component_idx), dtype=torch.int64, device=device)
        values = seq_ids * _SPLITMIX_MIX_1
        values = values + latent_ids * _SPLITMIX_MIX_2
        values = values + component_ids * _SPLITMIX_GOLDEN_GAMMA
        values = values + base
        values = self._splitmix64(values)
        return torch.bitwise_and(values, 0x7FFFFFFFFFFFFFFF)

    @staticmethod
    def _splitmix64(values: torch.Tensor) -> torch.Tensor:
        values = values + _SPLITMIX_GOLDEN_GAMMA
        values = torch.bitwise_xor(values, values >> 30) * _SPLITMIX_MIX_1
        values = torch.bitwise_xor(values, values >> 27) * _SPLITMIX_MIX_2
        return torch.bitwise_xor(values, values >> 31)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if not self._allocated:
            return
        checkpoint: dict = {
            "ctx_seq_idx": self.ctx_seq_idx,
            "ctx_seq_val": self.ctx_seq_val,
            "ctx_type": self.ctx_type,
        }
        if self.ctx_type == "mid":
            checkpoint["mode"] = self.mid_mode
            checkpoint["band_low_sigma"] = self._band_low
            checkpoint["band_high_sigma"] = self._band_high
            checkpoint["num_ctx_sequences"] = self.num_ctx_sequences
            checkpoint["reservoir_fill"] = self.reservoir_fill
            checkpoint["reservoir_n"]    = self.reservoir_n
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.allocate()
        self.ctx_seq_idx.copy_(checkpoint["ctx_seq_idx"])
        self.ctx_seq_val.copy_(checkpoint["ctx_seq_val"])
        if self.ctx_type == "mid":
            if "reservoir_fill" in checkpoint:
                self.reservoir_fill.copy_(checkpoint["reservoir_fill"])
            if "reservoir_n" in checkpoint:
                self.reservoir_n.copy_(checkpoint["reservoir_n"])

    def set_device(self, device: torch.device) -> None:
        if self.ctx_type in ("mid", "neg"):
            # mid_ctx and neg_ctx always stay on CPU; no VRAM allocation needed.
            return
        self.device = device
        if self._allocated:
            self.ctx_seq_idx = self.ctx_seq_idx.to(device)
            self.ctx_seq_val = self.ctx_seq_val.to(device)

    # ------------------------------------------------------------------
    # Query helpers (shared by all context types)
    # ------------------------------------------------------------------

    def get_all_sequence_ids(self) -> list[int]:
        """Returns a sorted list of all unique sequence IDs stored (excludes sentinel 0)."""
        if not self._allocated:
            return []
        unique_ids = torch.unique(self.ctx_seq_idx)
        unique_ids = unique_ids[unique_ids != 0]
        return unique_ids.tolist()

    def get_sequence_to_latents_map(self) -> Dict[int, List[Tuple[int, int]]]:
        """Maps sequence ID → list of (component_idx, latent_idx) pairs."""
        if not self._allocated:
            return {}
        full_mask = (self.ctx_seq_val > 0) & (self.ctx_seq_idx != 0)
        if not torch.any(full_mask):
            return {}
        sids    = self.ctx_seq_idx[full_mask]
        indices = torch.nonzero(full_mask)[:, :2]
        sids_sorted, sort_indices = torch.sort(sids)
        unique_sids, counts = torch.unique_consecutive(sids_sorted, return_counts=True)
        unique_sids_cpu = unique_sids.cpu().tolist()
        counts_cpu      = counts.cpu().tolist()
        pairs_cpu       = indices[sort_indices].cpu()
        pair_splits     = torch.split(pairs_cpu, counts_cpu)
        result = {}
        for i, sid in enumerate(unique_sids_cpu):
            c, l = pair_splits[i].t().tolist()
            result[int(sid)] = list(zip(c, l))
        return result

    def get_sequence_to_latents_csr(
        self,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CSR-style mapping: (seq_offsets, seq_targets_global).
        seq_offsets[sid] = end offset in seq_targets_global for sequence sid.
        seq_targets_global holds global latent IDs (comp_idx * d_sae + latent_idx).
        """
        if not self._allocated:
            target_device = self.device if device is None else device
            return (
                torch.zeros(0, dtype=torch.int64, device=target_device),
                torch.zeros(0, dtype=torch.int64, device=target_device),
            )
        target_device = self.device if device is None else device
        ctx_seq_idx = self.ctx_seq_idx.to(target_device)
        ctx_seq_val = self.ctx_seq_val.to(target_device)

        full_mask = (ctx_seq_val > 0) & (ctx_seq_idx != 0)
        if not torch.any(full_mask):
            return (
                torch.zeros(0, dtype=torch.int64, device=target_device),
                torch.zeros(0, dtype=torch.int64, device=target_device),
            )

        sids       = ctx_seq_idx[full_mask].to(torch.long)
        indices    = torch.nonzero(full_mask)[:, :2]
        global_ids = indices[:, 0].to(torch.long) * self.d_sae + indices[:, 1].to(torch.long)

        order            = torch.argsort(sids)
        sids_sorted      = sids[order]
        global_ids_sorted = global_ids[order]

        max_sid    = int(sids_sorted[-1].item())
        counts     = torch.bincount(sids_sorted, minlength=max_sid + 1)
        seq_offsets = torch.cumsum(counts, dim=0)

        return seq_offsets, global_ids_sorted


top_ctx = Context("top")
mid_ctx = Context("mid")
neg_ctx = Context("neg")
