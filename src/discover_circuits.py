import torch
import os
import argparse
from hardware import is_fast_memory, should_compile, detect_devices
from model.inference import Inference
from sae.bank import SAEBank
from data.loader import DataLoader
from store.latent_stats import latent_stats
from store.top_coactivation import top_coactivation
from store.context import top_ctx, neg_ctx
from store.logit_context import logit_ctx
from circuit.discovery_window import DiscoveryWindow
from circuit.feature_selection import CandidateSelector

def discover_circuits(candidates_path: str = "outputs/candidates.pt", reselect: bool = False, n_seeds: int = 16):
    devices = detect_devices()
    device = devices[0]
    fast = is_fast_memory()
    compile = should_compile()

    # Create outputs dir if not exists
    os.makedirs("outputs", exist_ok=True)

    print("Loading stores from outputs/...")
    
    # Load required stores
    latent_stats.load("outputs/latent_stats.pt")
    top_coactivation.load("outputs/top_coactivation.pt")
    top_ctx.load("outputs/top_ctx.pt")
    neg_ctx.load("outputs/neg_ctx.pt")
    logit_ctx.load("outputs/logit_ctx.pt")

    if reselect:
        print(f"Reselecting top {n_seeds} candidates...")
        selector = CandidateSelector(n_seeds=n_seeds)
        candidates = selector.select_candidates()
        selector.get_summary_stats(candidates)
        torch.save(candidates, candidates_path)
    else:
        if not os.path.exists(candidates_path):
            print(f"Candidates file {candidates_path} not found. Running selection...")
            selector = CandidateSelector(n_seeds=n_seeds)
            candidates = selector.select_candidates()
            selector.get_summary_stats(candidates)
            torch.save(candidates, candidates_path)
        else:
            candidates = torch.load(candidates_path, weights_only=False)
            print(f"Loaded {len(candidates)} candidates from {candidates_path}")

    print("Initializing model and SAE bank...")
    loader = DataLoader(device=device, pin_memory=fast)
    model = Inference(device=device, compile=compile)
    bank = SAEBank(devices=devices, load_decoders=fast, compile=compile)

    print("--- Standalone Discovery Window ---")
    window = DiscoveryWindow(model, bank, loader)
    window.run(candidates)
    print("Discovery complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run circuit discovery standalone from saved outputs.")
    parser.add_argument("--candidates", default="outputs/candidates.pt", help="Path to saved candidates .pt file")
    parser.add_argument("--reselect", action="store_true", help="Force re-selection of seed candidates")
    parser.add_argument("--n-seeds", type=int, default=16, help="Number of seeds if reselecting")
    args = parser.parse_args()
    
    discover_circuits(
        candidates_path=args.candidates, 
        reselect=args.reselect, 
        n_seeds=args.n_seeds
    )
