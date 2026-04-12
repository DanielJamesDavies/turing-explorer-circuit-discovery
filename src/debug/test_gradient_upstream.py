import torch
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from model.inference import Inference
from sae.bank import SAEBank
from data.loader import DataLoader
from config import config
from circuit.discovery_window import DiscoveryWindow
from store.latent_stats import latent_stats
from store.top_coactivation import top_coactivation
from store.context import neg_ctx

def main():
    # 1. Load resources
    print("Loading model and SAEs...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = Inference(device=device)
    bank = SAEBank(device=inference.device)
    loader = DataLoader(device=inference.device)
    
    # Load stats
    print("Loading latent statistics...")
    from store.context import top_ctx, mid_ctx, neg_ctx
    if not latent_stats._allocated:
        latent_stats.load("outputs/latent_stats.pt")
    if not top_coactivation._allocated:
        top_coactivation.load("outputs/top_coactivation.pt")
    if not top_ctx._allocated:
        top_ctx.load("outputs/top_ctx.pt")
    if not mid_ctx._allocated:
        mid_ctx.load("outputs/mid_ctx.pt")
    if not neg_ctx._allocated:
        neg_ctx.load("outputs/neg_ctx.pt")

    # 2. Select seeds
    # Let's try some seeds from layer 8 resid if possible, or just the first few from candidates.pt
    candidates_path = "outputs/candidates.pt"
    if os.path.exists(candidates_path):
        candidates = torch.load(candidates_path, weights_only=False)[:2] # Just test 2 seeds
        print(f"Loaded {len(candidates)} seeds from {candidates_path}")
    else:
        # Fallback to manual seeds if file doesn't exist
        candidates = [{"comp_idx": 26, "latent_idx": 100}] # Layer 8 resid (assuming 3 kinds)
        print("Using manual seed")

    # 3. Run discovery
    window = DiscoveryWindow(inference, bank, loader, output_dir="outputs/debug_circuits")
    window.run(candidates)

    print("\nValidation complete. Check outputs/debug_circuits/summary.json for results.")

if __name__ == "__main__":
    main()
