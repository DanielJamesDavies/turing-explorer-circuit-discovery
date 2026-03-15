import torch
from typing import Optional, Dict, cast
from config import config
from model.turingllm import TuringLLMConfig
from sae.topk_sae import SAEConfig

class LogitContext:
    """
    Stores and updates the tokens most likely to be generated when specific 
    SAE latents are activated, using efficient GPU-vectorized Top-K merging.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_config = TuringLLMConfig()
        self.sae_config = SAEConfig()
        
        self.num_components = self.llm_config.n_layer * 3
        self.d_sae = self.sae_config.d_sae
        self.n_tokens_per_latent = cast(int, config.latents.logit_ctx.n_tokens_per_latent)
        self.topk_output_tokens = cast(int, config.latents.logit_ctx.topk_output_tokens)

        self.top_tokens = torch.zeros((self.num_components, self.d_sae, self.n_tokens_per_latent),  dtype=torch.int32, device=self.device)
        self.top_probs = torch.zeros((self.num_components, self.d_sae, self.n_tokens_per_latent), dtype=torch.float32, device=self.device)
        self.latent_counts = torch.zeros((self.num_components, self.d_sae), dtype=torch.int64, device=self.device)

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
        # Get top predicted tokens for this batch
        batch_size = final_probs.shape[0]
        batch_v_probs, batch_v_indices = final_probs.topk(self.topk_output_tokens, dim=-1)

        # Stack all active latents from all components
        active_comp_ids = sorted(component_last_indices.keys())
        if not active_comp_ids:
            return
            
        all_b_latents = torch.stack([component_last_indices[i] for i in active_comp_ids]) # (num_active_comps, batch_size, k_sae)
        num_active_comps, _, k_sae = all_b_latents.shape

        # Flatten to Global Latent IDs: (comp_idx * d_sae) + latent_idx
        comp_offsets = torch.tensor(active_comp_ids, device=self.device).view(-1, 1, 1) * self.d_sae
        global_latents = (all_b_latents + comp_offsets).flatten()
        
        # Update firing counts in bulk
        self.latent_counts.view(-1).scatter_add_(
            0, global_latents, torch.ones_like(global_latents, dtype=torch.int64)
        )

        # Deduplicate firing events within this batch
        # If a latent fires multiple times in one batch, we just process the first occurrence.
        # We use a sort-and-mask approach since torch.unique(return_index=True) is not universally supported.
        sorted_latents, perm = global_latents.sort()
        is_first = torch.cat([torch.tensor([True], device=self.device), sorted_latents[1:] != sorted_latents[:-1]])
        
        unique_latents = sorted_latents[is_first]
        first_indices = perm[is_first]
        
        # batch_indices maps each flat index back to which batch row it originated from
        batch_indices = torch.arange(batch_size, device=self.device).view(1, batch_size, 1).expand(num_active_comps, -1, k_sae).flatten()
        unique_batch_idx = batch_indices[first_indices]

        # Parallel Merge Pass
        # Fetch current top-K for all unique latents found in this batch
        curr_p = self.top_probs.view(-1, self.n_tokens_per_latent)[unique_latents]
        curr_t = self.top_tokens.view(-1, self.n_tokens_per_latent)[unique_latents]

        # Fetch new predictions from the corresponding batch samples
        new_p = batch_v_probs[unique_batch_idx]
        new_t = batch_v_indices[unique_batch_idx]

        # Concatenate and take new Top-K
        combined_p = torch.cat([curr_p, new_p], dim=-1)
        combined_t = torch.cat([curr_t, new_t], dim=-1)
        
        new_top_p, top_idx = combined_p.topk(self.n_tokens_per_latent, dim=-1)
        new_top_t = torch.gather(combined_t, -1, top_idx)

        # Bulk Write Back
        self.top_probs.view(-1, self.n_tokens_per_latent)[unique_latents] = new_top_p
        self.top_tokens.view(-1, self.n_tokens_per_latent)[unique_latents] = new_top_t.to(torch.int32)

    def get_top_tokens(
        self, component_idx: int, latent_idx: int
    ) -> list[tuple[int, float]]:
        """Returns the top stored tokens for a specific latent, filtering out zeros."""
        tokens = self.top_tokens[component_idx, latent_idx].tolist()
        probs = self.top_probs[component_idx, latent_idx].tolist()
        
        # Filter out zero probabilities (e.g. from initialization)
        results = [(int(t), float(p)) for t, p in zip(tokens, probs) if p > 0]
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def load(self, path: str) -> None:
        """
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if "latent_counts" in checkpoint:
                self.latent_counts = checkpoint["latent_counts"].to(self.device)
            if "top_tokens" in checkpoint:
                self.top_tokens = checkpoint["top_tokens"].to(self.device)
            if "top_probs" in checkpoint:
                self.top_probs = checkpoint["top_probs"].to(self.device)
        except Exception as e:
            print(f"LogitContext load failed (likely no file yet): {e}")

    def save(self, path: str) -> None:
        """
        """
        torch.save({
            "latent_counts": self.latent_counts,
            "top_tokens": self.top_tokens,
            "top_probs": self.top_probs,
        }, path)

    def set_device(self, device: torch.device) -> None:
        self.device = device
        self.top_tokens = self.top_tokens.to(device)
        self.top_probs = self.top_probs.to(device)
        self.latent_counts = self.latent_counts.to(device)


logit_ctx = LogitContext()
