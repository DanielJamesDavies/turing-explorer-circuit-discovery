import importlib.util
import os
import sys
import torch
from typing import cast, Optional, Dict, List, Tuple

from config import config
from model.turingllm import TuringLLMConfig
from sae.topk_sae import SAEConfig


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


class Context:

    def __init__(self, ctx_type: str, device: Optional[torch.device] = None):
        self.ctx_type = ctx_type  # "top" | "mid" | "neg"
        self.llm_config = TuringLLMConfig()
        self.sae_config = SAEConfig()
        self.num_components = self.llm_config.n_layer * 3
        self.d_sae = self.sae_config.d_sae

        if ctx_type == "mid":
            # mid_ctx always lives on CPU: the C++ reservoir extension requires CPU tensors,
            # and the reservoir (~540 MB) is not worth keeping in VRAM.
            self.device = torch.device("cpu")
            self.num_ctx_sequences = cast(int, config.latents.mid_ctx.n_sequences or 64)
            self._band_low  = cast(float, config.latents.mid_ctx.band_low_sigma  or 0.5)
            self._band_high = cast(float, config.latents.mid_ctx.band_high_sigma or 1.5)
            val_dtype = torch.float32
        elif ctx_type == "neg":
            # neg_ctx always lives on CPU: populated by the offline ANN step, not
            # during the GPU forward pass.  ctx_seq_val stores cosine similarities
            # (float32) rather than activation scores.
            self.device = torch.device("cpu")
            self.num_ctx_sequences = cast(int, config.latents.neg_ctx.n_sequences or 64)
            val_dtype = torch.float32
        else:
            self.device = device if device is not None else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.num_ctx_sequences = cast(int, config.latents.top_ctx.n_sequences)
            val_dtype = torch.bfloat16

        self.ctx_seq_idx = torch.zeros(
            (self.num_components, self.d_sae, self.num_ctx_sequences),
            dtype=torch.int32, device=self.device,
        )
        self.ctx_seq_val = torch.zeros(
            (self.num_components, self.d_sae, self.num_ctx_sequences),
            dtype=val_dtype, device=self.device,
        )

        if ctx_type == "mid":
            # Auxiliary reservoir state (always CPU, never moved to another device).
            self.reservoir_fill = torch.zeros(
                (self.num_components, self.d_sae), dtype=torch.int32,
            )
            self.reservoir_n = torch.zeros(
                (self.num_components, self.d_sae), dtype=torch.int64,
            )

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
        if self.ctx_type == "top":
            self._update_top(component_idx, sequence_indices, latents)
        elif self.ctx_type == "mid":
            if latent_mean_seq is None or latent_std_seq is None:
                raise ValueError(
                    "mid_ctx.update_component requires latent_mean_seq and latent_std_seq"
                )
            self._update_mid(component_idx, sequence_indices, latents, latent_mean_seq, latent_std_seq)
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

    def _update_mid(
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
            mean_seq = latent_mean_seq.to(compute_device)
            std_seq  = latent_std_seq.to(compute_device).clamp(min=1e-6)
            low  = mean_seq + self._band_low  * std_seq  # [d_sae]
            high = mean_seq + self._band_high * std_seq  # [d_sae]

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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        checkpoint: dict = {
            "ctx_seq_idx": self.ctx_seq_idx,
            "ctx_seq_val": self.ctx_seq_val,
        }
        if self.ctx_type == "mid":
            checkpoint["reservoir_fill"] = self.reservoir_fill
            checkpoint["reservoir_n"]    = self.reservoir_n
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.ctx_seq_idx = checkpoint["ctx_seq_idx"].to(self.device)
        self.ctx_seq_val = checkpoint["ctx_seq_val"].to(self.device)
        if self.ctx_type == "mid":
            if "reservoir_fill" in checkpoint:
                self.reservoir_fill = checkpoint["reservoir_fill"].cpu()
            if "reservoir_n" in checkpoint:
                self.reservoir_n = checkpoint["reservoir_n"].cpu()

    def set_device(self, device: torch.device) -> None:
        if self.ctx_type in ("mid", "neg"):
            # mid_ctx and neg_ctx always stay on CPU; no VRAM allocation needed.
            return
        self.device = device
        self.ctx_seq_idx = self.ctx_seq_idx.to(device)
        self.ctx_seq_val = self.ctx_seq_val.to(device)

    # ------------------------------------------------------------------
    # Query helpers (shared by all context types)
    # ------------------------------------------------------------------

    def get_all_sequence_ids(self) -> list[int]:
        """Returns a sorted list of all unique sequence IDs stored (excludes sentinel 0)."""
        unique_ids = torch.unique(self.ctx_seq_idx)
        unique_ids = unique_ids[unique_ids != 0]
        return unique_ids.tolist()

    def get_sequence_to_latents_map(self) -> Dict[int, List[Tuple[int, int]]]:
        """Maps sequence ID → list of (component_idx, latent_idx) pairs."""
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
