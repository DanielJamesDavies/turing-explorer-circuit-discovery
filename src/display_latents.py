import os
import torch
import argparse
from data.loader import DataLoader
from model.inference import Inference
from sae.bank import SAEBank
from store.context import top_ctx, mid_ctx, neg_ctx
from store.logit_context import logit_ctx
from store.top_coactivation import top_coactivation
from display.display import display

VALID_KINDS = ["attn", "mlp", "resid"]


def _try_load(ctx, path: str, name: str) -> None:
    """Load a context store if the file exists; print a notice otherwise."""
    if os.path.exists(path):
        ctx.load(path)
        print(f"  loaded {name}")
    else:
        print(f"  {name} not found at {path} — skipping")


def parse_latent_input(raw: str):
    """Parse '<layer> <kind> <latent>' from a string. Returns (layer, kind, latent) or None."""
    parts = raw.strip().split()
    if len(parts) != 3:
        return None
    try:
        layer = int(parts[0])
    except ValueError:
        return None
    kind = parts[1].lower()
    if kind not in VALID_KINDS:
        return None
    try:
        latent = int(parts[2])
    except ValueError:
        return None
    return layer, kind, latent


def analyze(model, bank, loader, layer_idx, kind, latent_idx, n_sequences):
    print(f"\nAnalyzing latent {latent_idx} in {kind} layer {layer_idx}...")
    display.analyze_and_print_specific_latent(
        top_ctx=top_ctx,
        model=model,
        bank=bank,
        loader=loader,
        layer_idx=layer_idx,
        kind=kind,
        latent_idx=latent_idx,
        n_sequences=n_sequences,
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze a specific latent by layer, kind, and index.")
    parser.add_argument("--layer", type=int, help="Layer index")
    parser.add_argument("--kind", type=str, choices=VALID_KINDS, help="Kind of component")
    parser.add_argument("--latent", type=int, help="Latent index")
    parser.add_argument("--sequences", type=int, default=5, help="Number of top sequences to display")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("")
    print("Loading stores...")
    top_ctx.load("outputs/top_ctx.pt")
    logit_ctx.load("outputs/logit_ctx.pt")
    top_coactivation.load("outputs/top_coactivation.pt")
    _try_load(mid_ctx, "outputs/mid_ctx.pt", "mid_ctx")
    _try_load(neg_ctx, "outputs/neg_ctx.pt", "neg_ctx")

    print("Initializing components...")
    loader = DataLoader(device=device)
    model = Inference(device=device)
    bank = SAEBank(device=device, load_decoders=False)

    if args.layer is not None and args.kind is not None and args.latent is not None:
        analyze(model, bank, loader, args.layer, args.kind, args.latent, args.sequences)
    else:
        print("\nEntering interactive mode (CTRL+C to exit)")
        print("Format: <layer> <kind> <latent>  e.g. 5 resid 12345\n")
        try:
            while True:
                raw = input("Analyze latent: ").strip()
                if not raw:
                    continue
                parsed = parse_latent_input(raw)
                if parsed is None:
                    print(f"Invalid input. Expected: <layer> <kind> <latent>  (kind: {', '.join(VALID_KINDS)})")
                    continue
                layer, kind, latent = parsed
                analyze(model, bank, loader, layer, kind, latent, args.sequences)
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()
