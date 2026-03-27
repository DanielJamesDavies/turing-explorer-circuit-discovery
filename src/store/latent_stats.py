import importlib.util
import os
import torch
from collections import defaultdict
from typing import Optional

from model.turingllm import TuringLLMConfig
from sae.topk_sae import SAEConfig
from store.utils import _AutoAllocTensor


def _load_cuda_ext():
    """Load the pre-compiled latent_stats_cuda extension from src/native/."""
    native_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "native")
    )
    try:
        for fname in os.listdir(native_dir):
            if fname.startswith("latent_stats_cuda") and fname.endswith(".so"):
                so_path = os.path.join(native_dir, fname)
                spec = importlib.util.spec_from_file_location("latent_stats_cuda", so_path)
                mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
                spec.loader.exec_module(mod)                   # type: ignore[union-attr]
                return mod
    except Exception:
        pass
    return None


_cuda_ext = _load_cuda_ext()
_HAS_CUDA_EXT = _cuda_ext is not None
if _HAS_CUDA_EXT:
    print("[latent_stats] CUDA extension loaded.")
else:
    print("[latent_stats] CUDA extension not found, using PyTorch fallback.")


class LatentStats:

    active_count = _AutoAllocTensor()
    mean         = _AutoAllocTensor()
    mean_abs     = _AutoAllocTensor()
    m2           = _AutoAllocTensor()
    m2_abs       = _AutoAllocTensor()
    seq_count    = _AutoAllocTensor()
    mean_seq     = _AutoAllocTensor()
    m2_seq       = _AutoAllocTensor()

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_config = TuringLLMConfig()
        self.sae_config = SAEConfig()
        self.num_components = self.llm_config.n_layer * 3
        self.component_steps: dict[int, int] = defaultdict(int)
        
        self._allocated = False

    def allocate(self, device: Optional[torch.device] = None) -> None:
        """Explicitly allocate the large GPU tensors. Safe to call multiple times."""
        if self._allocated:
            if device is not None and device != self.device:
                self.set_device(device)
            return

        if device is not None:
            self.device = device

        d = self.sae_config.d_sae
        shape = (self.num_components, d)

        # Activation counts (also serves as Welford n)
        self.active_count = torch.zeros(shape, dtype=torch.int64, device=self.device)

        # Per-token Welford statistics (tracks individual token activations)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.mean_abs = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.m2 = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.m2_abs = torch.zeros(shape, dtype=torch.float32, device=self.device)

        # Per-sequence Welford statistics
        self.seq_count = torch.zeros(shape, dtype=torch.int64,   device=self.device)
        self.mean_seq  = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.m2_seq    = torch.zeros(shape, dtype=torch.float32, device=self.device)
        
        self._allocated = True

    def update_component(
        self,
        component_idx: int,
        latents: tuple[torch.Tensor, torch.Tensor]  # [0: top_acts, 1: top_indices]
    ) -> None:
        self.allocate(latents[0].device if latents[0].is_cuda else None)
        with torch.no_grad():
            top_acts = latents[0]    # [batch_size, seq_len, k]
            top_indices = latents[1] # [batch_size, seq_len, k]

            if _HAS_CUDA_EXT and top_acts.is_cuda:
                _cuda_ext.update_latent_stats(
                    top_acts.float().contiguous(),
                    top_indices.int().contiguous(),
                    self.mean[component_idx],
                    self.m2[component_idx],
                    self.mean_abs[component_idx],
                    self.m2_abs[component_idx],
                    self.active_count[component_idx],
                    self.sae_config.d_sae,
                )
                self.component_steps[component_idx] += 1
                self._update_seq_scores(component_idx, top_acts, top_indices)
                return

            # Fallback: pure PyTorch path (CPU or no CUDA extension)
            mask = top_acts > 0
            if not mask.any():
                self.component_steps[component_idx] += 1
                return

            indices = top_indices[mask].flatten().long()
            values = top_acts[mask].flatten().float()
            values_abs = values.abs()

            d = self.sae_config.d_sae

            ones = torch.ones_like(values)
            n_b = torch.zeros(d, dtype=torch.float32, device=self.device).scatter_add_(0, indices, ones)

            sum_b = torch.zeros(d, dtype=torch.float32, device=self.device).scatter_add_(0, indices, values)
            sum_abs_b = torch.zeros(d, dtype=torch.float32, device=self.device).scatter_add_(0, indices, values_abs)

            sum_sq_b = torch.zeros(d, dtype=torch.float32, device=self.device).scatter_add_(0, indices, values.square())
            sum_sq_abs_b = torch.zeros(d, dtype=torch.float32, device=self.device).scatter_add_(0, indices, values_abs.square())

            safe_n_b = n_b.clamp(min=1)
            mean_b = sum_b / safe_n_b
            mean_abs_b = sum_abs_b / safe_n_b
            m2_b = sum_sq_b - n_b * mean_b.square()
            m2_abs_b = sum_sq_abs_b - n_b * mean_abs_b.square()

            n_a = self.active_count[component_idx].float()
            self._welford_merge(self.mean[component_idx], self.m2[component_idx], n_a, mean_b, m2_b, n_b)
            self._welford_merge(self.mean_abs[component_idx], self.m2_abs[component_idx], n_a, mean_abs_b, m2_abs_b, n_b)

            self.active_count[component_idx] += n_b.long()
            self.component_steps[component_idx] += 1

        # Always update per-sequence stats (pure PyTorch, no CUDA kernel needed).
        self._update_seq_scores(component_idx, top_acts, top_indices)

    def _update_seq_scores(
        self,
        component_idx: int,
        top_acts: torch.Tensor,    # [batch, seq_len, k]
        top_indices: torch.Tensor, # [batch, seq_len, k]
    ) -> None:
        """Welford update for per-sequence activation scores.

        Tracks the distribution of mean-activation-per-sequence for sequences
        where each latent fired at least once. This lives in the same value
        range as compute_seq_scores(), making it suitable for the mid_ctx band.
        """
        with torch.no_grad():
            batch, seq_len, _k = top_acts.shape
            d   = self.sae_config.d_sae
            dev = top_acts.device

            # [batch, d_sae] — mean activation over all seq_len tokens (0 if not fired)
            scores = torch.zeros(batch, d, device=dev, dtype=torch.float32)
            scores.scatter_add_(
                1,
                top_indices.reshape(batch, -1).long(),
                top_acts.reshape(batch, -1).float(),
            )
            scores /= seq_len
            scores = scores.T.contiguous()  # [d_sae, batch]

            # Only count sequences where the latent actually fired.
            mask  = scores > 0              # [d_sae, batch]
            n_b   = mask.float().sum(dim=1) # [d_sae]
            if not n_b.any():
                return

            scores_m = scores * mask
            sum_b    = scores_m.sum(dim=1)            # [d_sae]
            sum_sq_b = scores_m.pow(2).sum(dim=1)     # [d_sae]

            safe_n = n_b.clamp(min=1)
            mean_b = sum_b / safe_n
            m2_b   = (sum_sq_b - n_b * mean_b.pow(2)).clamp(min=0)

            # Transfer to stats device for the Welford merge.
            mean_b = mean_b.to(self.device)
            m2_b   = m2_b.to(self.device)
            n_b    = n_b.to(self.device)

            n_a = self.seq_count[component_idx].float()
            self._welford_merge(
                self.mean_seq[component_idx], self.m2_seq[component_idx],
                n_a, mean_b, m2_b, n_b,
            )
            self.seq_count[component_idx] += n_b.long()

    @staticmethod
    def _welford_merge(
        mean_a: torch.Tensor, m2_a: torch.Tensor, n_a: torch.Tensor,
        mean_b: torch.Tensor, m2_b: torch.Tensor, n_b: torch.Tensor,
    ) -> None:
        """Parallel Welford merge: combines batch stats (B) into global stats (A) in-place."""
        n_total = n_a + n_b
        safe_n = n_total.clamp(min=1)
        delta = mean_b - mean_a
        mean_a += delta * (n_b / safe_n)
        m2_a += m2_b + delta.square() * (n_a * n_b / safe_n)

    def variance(self, component_idx: Optional[int] = None) -> torch.Tensor:
        """Sample variance of a: M2 / max(1, n - 1)."""
        if not self._allocated:
            return torch.zeros((self.num_components, self.sae_config.d_sae) if component_idx is None else (self.sae_config.d_sae,), device=self.device)
        count = self.active_count if component_idx is None else self.active_count[component_idx]
        m2 = self.m2 if component_idx is None else self.m2[component_idx]
        return m2 / (count.float() - 1).clamp(min=1)

    def variance_abs(self, component_idx: Optional[int] = None) -> torch.Tensor:
        """Sample variance of |a|: M2_abs / max(1, n - 1)."""
        if not self._allocated:
            return torch.zeros((self.num_components, self.sae_config.d_sae) if component_idx is None else (self.sae_config.d_sae,), device=self.device)
        count = self.active_count if component_idx is None else self.active_count[component_idx]
        m2_abs = self.m2_abs if component_idx is None else self.m2_abs[component_idx]
        return m2_abs / (count.float() - 1).clamp(min=1)

    def std(self, component_idx: Optional[int] = None) -> torch.Tensor:
        """Sample standard deviation of a."""
        return self.variance(component_idx).sqrt()

    def std_abs(self, component_idx: Optional[int] = None) -> torch.Tensor:
        """Sample standard deviation of |a|."""
        return self.variance_abs(component_idx).sqrt()

    def std_seq(self, component_idx: Optional[int] = None) -> torch.Tensor:
        """Sample standard deviation of per-sequence activation scores."""
        if not self._allocated:
            return torch.zeros((self.num_components, self.sae_config.d_sae) if component_idx is None else (self.sae_config.d_sae,), device=self.device)
        count = self.seq_count if component_idx is None else self.seq_count[component_idx]
        m2    = self.m2_seq    if component_idx is None else self.m2_seq[component_idx]
        return (m2 / (count.float() - 1).clamp(min=1)).sqrt()

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.allocate()
        self.active_count.copy_(checkpoint["active_count"])
        self.mean.copy_(checkpoint["mean"])
        self.mean_abs.copy_(checkpoint["mean_abs"])
        self.m2.copy_(checkpoint["m2"])
        self.m2_abs.copy_(checkpoint["m2_abs"])
        self.component_steps.update(checkpoint["component_steps"])
        # Per-sequence stats (may be absent in older checkpoints).
        if "seq_count" in checkpoint:
            self.seq_count.copy_(checkpoint["seq_count"])
        if "mean_seq" in checkpoint:
            self.mean_seq.copy_(checkpoint["mean_seq"])
        if "m2_seq" in checkpoint:
            self.m2_seq.copy_(checkpoint["m2_seq"])

    def save(self, path: str) -> None:
        if not self._allocated:
            return # Nothing to save
        torch.save({
            "active_count":    self.active_count,
            "mean":            self.mean,
            "mean_abs":        self.mean_abs,
            "m2":              self.m2,
            "m2_abs":          self.m2_abs,
            "seq_count":       self.seq_count,
            "mean_seq":        self.mean_seq,
            "m2_seq":          self.m2_seq,
            "component_steps": dict(self.component_steps),
        }, path)

    def set_device(self, device: torch.device) -> None:
        self.device       = device
        if self._allocated:
            self.active_count = self.active_count.to(device)
            self.mean         = self.mean.to(device)
            self.mean_abs     = self.mean_abs.to(device)
            self.m2           = self.m2.to(device)
            self.m2_abs       = self.m2_abs.to(device)
            self.seq_count    = self.seq_count.to(device)
            self.mean_seq     = self.mean_seq.to(device)
            self.m2_seq       = self.m2_seq.to(device)


latent_stats = LatentStats()
