import torch
import math
from typing import List, Dict, Any, Tuple, Optional, cast
from config import config
from pipeline.component_index import split_component_idx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store.top_coactivation import top_coactivation
from store.context import top_ctx

class CandidateSelector:
    """
    Identifies 'Seed Latents' for global circuit discovery.
    Uses multi-dimensional scoring from Pass 1 & 2 stores to rank latents
    that are likely to be part of interesting mechanisms.
    """
    def __init__(self, n_seeds: Optional[int] = None, device: Optional[torch.device] = None):
        self.n_seeds = n_seeds or cast(int, config.discovery.n_seeds or 1000)
        self.device = device if device is not None else torch.device("cpu")

    @torch.no_grad()
    def select_candidates(self) -> List[Dict[str, Any]]:
        """
        Runs the tiered selection process and returns a list of candidate dictionaries.
        """
        # Ensure stores are on the correct device for calculation
        # (Usually CPU to avoid VRAM pressure during selection)
        
        # 1. Direct Logit Impact (Direct Effectors)
        # sum of top-k probabilities for each latent
        logit_impact = logit_ctx.top_probs.to(self.device).sum(dim=-1)
        logit_seeds = self._top_k_indices(logit_impact, self.n_seeds, "logit_impact")

        # 2. Network Centrality (Structural Hubs)
        # sum of frequency-adjusted co-occurrence magnitudes
        connectivity = top_coactivation.top_values.to(self.device).sum(dim=-1)
        conn_seeds = self._top_k_indices(connectivity, self.n_seeds, "connectivity")

        # 3. "Surprise" / Salient Rarity
        # Salience (mean_seq) * Rarity (log boost)
        # Total sequences is roughly max(seq_count) or we can infer it
        total_seqs = float(latent_stats.seq_count.max().item()) + 1.0
        p_fire_seq = (latent_stats.seq_count.to(self.device).float() + 1e-6) / total_seqs
        rarity_boost = torch.log10(1.0 / p_fire_seq).clamp(1.0, 8.0)
        
        # Surprise = Strength when firing * Rarity
        surprise = latent_stats.mean_seq.to(self.device) * rarity_boost
        # Only consider latents that have fired at least a few times to avoid noise
        surprise = surprise.masked_fill(latent_stats.seq_count.to(self.device) < 5, -1e9)
        surprise_seeds = self._top_k_indices(surprise, self.n_seeds, "surprise")

        # 4. Context Stability / Coherence
        # Latents whose activation values in top contexts are highest/most stable
        coherence = top_ctx.ctx_seq_val.to(self.device).mean(dim=-1)
        coherence_seeds = self._top_k_indices(coherence, self.n_seeds, "context_coherence")

        # Merge and deduplicate
        merged = {}
        for seeds in [logit_seeds, conn_seeds, surprise_seeds, coherence_seeds]:
            for seed in seeds:
                key = (seed['comp_idx'], seed['latent_idx'])
                if key not in merged:
                    merged[key] = seed
                else:
                    # Accumulate score and append reason if already exists
                    merged[key]['score'] += seed['score']
                    merged[key]['reason'] += f", {seed['reason']}"
        
        # Sort by combined score and return the top n_seeds
        results = sorted(merged.values(), key=lambda x: x['score'], reverse=True)
        print(f"[CandidateSelector] Selected {len(results)} unique seed latents. Returning top {self.n_seeds}.")
        
        return results[:self.n_seeds]

    def _top_k_indices(self, score_tensor: torch.Tensor, k: int, reason: str) -> List[Dict[str, Any]]:
        """Helper to get top-k (comp_idx, latent_idx) from a [C, D] score tensor."""
        C, D = score_tensor.shape
        flat_scores = score_tensor.view(-1)
        actual_k = min(k, flat_scores.numel())
        
        top_vals, top_flat_indices = torch.topk(flat_scores, actual_k)
        
        results = []
        for val, flat_idx in zip(top_vals.tolist(), top_flat_indices.tolist()):
            if val <= 0: continue # Skip inactive/zero-score latents
            
            comp_idx = flat_idx // D
            latent_idx = flat_idx % D
            results.append({
                "comp_idx": comp_idx,
                "latent_idx": latent_idx,
                "score": float(val),
                "reason": reason
            })
        return results

    def get_summary_stats(self, candidates: List[Dict[str, Any]]):
        """Prints a summary of selected candidates by layer and kind."""
        from model.turingllm import TuringLLMConfig
        n_layers = TuringLLMConfig().n_layer
        kinds = ["attn", "mlp", "resid"]
        
        layer_counts = [0] * n_layers
        kind_counts = {"attn": 0, "mlp": 0, "resid": 0}
        
        for c in candidates:
            layer_idx, kind_idx = split_component_idx(c["comp_idx"], len(kinds))
            layer_counts[layer_idx] += 1
            kind_counts[kinds[kind_idx]] += 1
            
        print("\nCandidate Summary:")
        print(f"By Kind: {kind_counts}")
        print(f"By Layer: {layer_counts}")
