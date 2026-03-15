import torch
import sys
import os

# Ensure src is in the path when running from the root
sys.path.append(os.path.join(os.getcwd(), "src"))

from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store.top_coactivation import top_coactivation
from store.context import top_ctx
from circuit.feature_selection import CandidateSelector
from pipeline.component_index import split_component_idx

def print_top_candidates(n_seeds: int = 10):
    print("Loading stores from outputs/...")
    device = torch.device("cpu")
    
    # Load each store
    # These expect paths relative to the current working directory (usually root)
    latent_stats.load("outputs/latent_stats.pt")
    logit_ctx.load("outputs/logit_ctx.pt")
    top_coactivation.load("outputs/top_coactivation.pt")
    top_ctx.load("outputs/top_ctx.pt")
    
    print("Selecting candidates...")
    selector = CandidateSelector(n_seeds=n_seeds, device=device)
    candidates = selector.select_candidates()
    
    kinds = ["attn", "mlp", "resid"]
    
    print(f"\nTop {n_seeds} Seed Candidates:")
    print("-" * 60)
    for c in candidates[:n_seeds]:
        layer, kind_idx = split_component_idx(c["comp_idx"], len(kinds))
        kind = kinds[kind_idx]
        latent = c['latent_idx']
        reason = c['reason']
        print(f"Layer {layer:2d} | Kind: {kind:5s} | Latent: {latent:5d} | Reasons: {reason}")

if __name__ == "__main__":
    try:
        # Default to 10 seeds, but could be adjusted via arguments
        n = 10
        if len(sys.argv) > 1:
            n = int(sys.argv[1])
        print_top_candidates(n)
    except Exception as e:
        print(f"Error selecting candidates: {e}")
        import traceback
        traceback.print_exc()
