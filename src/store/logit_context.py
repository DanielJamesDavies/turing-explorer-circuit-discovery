import torch
from typing import Optional, Dict, cast
from config import config
from model.turingllm import TuringLLMConfig
from sae.topk_sae import SAEConfig
from store.utils import _AutoAllocTensor

class LogitContext:
    """
    Stores and updates the tokens most likely to be generated when specific 
    SAE latents are activated, using efficient GPU-vectorized Top-K merging.
    """

    top_tokens     = _AutoAllocTensor()
    top_probs      = _AutoAllocTensor()
    latent_counts  = _AutoAllocTensor()

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_config = TuringLLMConfig()
        self.sae_config = SAEConfig()
        
        self.num_components = self.llm_config.n_layer * 3
        self.d_sae = self.sae_config.d_sae
        self.n_tokens_per_latent = cast(int, config.latents.logit_ctx.n_tokens_per_latent)
        self.topk_output_tokens = cast(int, config.latents.logit_ctx.topk_output_tokens)

        self._allocated = False

    def allocate(self, device: Optional[torch.device] = None) -> None:
        """Explicitly allocate the large GPU tensors. Safe to call multiple times."""
        if self._allocated:
            if device is not None and device != self.device:
                self.set_device(device)
            return

        if device is not None:
            self.device = device

        self.top_tokens = torch.zeros((self.num_components, self.d_sae, self.n_tokens_per_latent),  dtype=torch.int32, device=self.device)
        self.top_probs = torch.zeros((self.num_components, self.d_sae, self.n_tokens_per_latent), dtype=torch.float32, device=self.device)
        self.latent_counts = torch.zeros((self.num_components, self.d_sae), dtype=torch.int64, device=self.device)
        self._allocated = True

    @torch.no_grad()
    def update(
        self,
        component_last_indices: Dict[int, torch.Tensor],
        final_probs: torch.Tensor,
    ) -> None:
        """
        Updates the top token mapping using the final probabilities and active latents.
        Uses a 'Stack and Flatten' strategy to process all components and samples in one
        parallelized GPU pass, bypassing expensive Python loops and sequential updates.
        """
        self.allocate(final_probs.device if final_probs.is_cuda else None)
        # Get top predicted tokens for this batch
        batch_size = final_probs.shape[0]
        batch_v_probs, batch_v_indices = final_probs.topk(self.topk_output_tokens, dim=-1)

        # Stack all active latents from all components
        active_comp_ids = sorted(component_last_indices.keys())
        if not active_comp_ids:
            return
            
        # OLD METHOD (Commented out for easy revert)
        # all_b_latents = torch.stack([component_last_indices[i] for i in active_comp_ids]) # (num_active_comps, batch_size, k_sae)
        # num_active_comps, _, k_sae = all_b_latents.shape
        # comp_offsets = torch.tensor(active_comp_ids, device=self.device).view(-1, 1, 1) * self.d_sae
        # global_latents = (all_b_latents + comp_offsets).flatten()
        # batch_indices = torch.arange(batch_size, device=self.device).view(1, batch_size, 1).expand(num_active_comps, -1, k_sae).flatten()

        # NEW MEMORY-EFFICIENT METHOD (VRAM-safe on GPU)
        # Avoids the intermediate 3D stack by computing global IDs directly
        k_sae = component_last_indices[active_comp_ids[0]].shape[-1]
        global_latents = torch.cat([
            (component_last_indices[i].flatten() + (i * self.d_sae))
            for i in active_comp_ids
        ])
        batch_indices = torch.cat([
            torch.arange(batch_size, device=self.device).repeat_interleave(k_sae)
            for _ in active_comp_ids
        ])
        
        # Update firing counts in bulk
        self.latent_counts.view(-1).scatter_add_(
            0, global_latents, torch.ones_like(global_latents, dtype=torch.int64)
        )

        # Deduplicate firing events within this batch
        # We use sort-and-mask to find unique latents and one batch index where each fired
        sorted_latents, perm = global_latents.sort()
        is_first = torch.cat([
            torch.tensor([True], device=self.device), 
            sorted_latents[1:] != sorted_latents[:-1]
        ])
        
        unique_latents = sorted_latents[is_first]
        first_indices = perm[is_first]
        unique_batch_idx = batch_indices[first_indices]

        # Parallel Merge Pass
        # We process in chunks if unique_latents is massive, to prevent OOM
        chunk_size = 131072 # 128k latents per pass
        for i in range(0, unique_latents.shape[0], chunk_size):
            u_latents = unique_latents[i : i + chunk_size]
            u_batch_idx = unique_batch_idx[i : i + chunk_size]

            # Fetch current top-K
            curr_p = self.top_probs.view(-1, self.n_tokens_per_latent)[u_latents]
            curr_t = self.top_tokens.view(-1, self.n_tokens_per_latent)[u_latents]

            # Fetch new predictions
            new_p = batch_v_probs[u_batch_idx]
            new_t = batch_v_indices[u_batch_idx]

            # Concatenate and take new Top-K
            combined_p = torch.cat([curr_p, new_p], dim=-1)
            combined_t = torch.cat([curr_t, new_t], dim=-1)
            
            new_top_p, top_idx = combined_p.topk(self.n_tokens_per_latent, dim=-1)
            new_top_t = torch.gather(combined_t, -1, top_idx)

            # Bulk Write Back
            self.top_probs.view(-1, self.n_tokens_per_latent)[u_latents] = new_top_p
            self.top_tokens.view(-1, self.n_tokens_per_latent)[u_latents] = new_top_t.to(torch.int32)

    def get_top_tokens(
        self, component_idx: int, latent_idx: int
    ) -> list[tuple[int, float]]:
        """Returns the top stored tokens for a specific latent, filtering out zeros."""
        if not self._allocated:
            return []
        tokens = self.top_tokens[component_idx, latent_idx].tolist()
        probs = self.top_probs[component_idx, latent_idx].tolist()
        
        # Filter out zero probabilities (e.g. from initialization)
        results = [(int(t), float(p)) for t, p in zip(tokens, probs) if p > 0]
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def load(self, path: str) -> None:
        """
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.allocate()
            if "latent_counts" in checkpoint:
                self.latent_counts.copy_(checkpoint["latent_counts"])
            if "top_tokens" in checkpoint:
                self.top_tokens.copy_(checkpoint["top_tokens"])
            if "top_probs" in checkpoint:
                self.top_probs.copy_(checkpoint["top_probs"])
        except Exception as e:
            print(f"LogitContext load failed (likely no file yet): {e}")

    def save(self, path: str) -> None:
        """
        """
        if not self._allocated:
            return
        torch.save({
            "latent_counts": self.latent_counts,
            "top_tokens": self.top_tokens,
            "top_probs": self.top_probs,
        }, path)

    def set_device(self, device: torch.device) -> None:
        self.device = device
        if self._allocated:
            self.top_tokens = self.top_tokens.to(device)
            self.top_probs = self.top_probs.to(device)
            self.latent_counts = self.latent_counts.to(device)


logit_ctx = LogitContext()
