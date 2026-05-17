from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch

from data.loader import DataLoader
from store.context import top_ctx
from pipeline.persist import build_search_cache_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build outputs/search_cache.parquet from saved top_ctx and token shards."
    )
    parser.add_argument(
        "--top-ctx",
        default="outputs/top_ctx.pt",
        help="Path to a saved top_ctx.pt artifact.",
    )
    parser.add_argument(
        "--output",
        default="outputs/search_cache.parquet",
        help="Path for the generated Parquet search cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.top_ctx):
        raise FileNotFoundError(f"top_ctx artifact not found: {args.top_ctx}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = torch.device("cpu")
    loader = DataLoader(device=device, pin_memory=False)
    top_ctx.load(args.top_ctx)

    # Search-cache generation only needs the SAE kind names, not the weights.
    bank_ref = SimpleNamespace(kinds=["attn", "mlp", "resid"])
    build_search_cache_artifact(top_ctx, bank_ref, loader, output_path=args.output)


if __name__ == "__main__":
    main()
