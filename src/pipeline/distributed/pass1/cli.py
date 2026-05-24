"""CLI entrypoint for distributed pass-1 merge."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from ..manifest import load_manifest
from .writer import merge_pass1_worker_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the pass-1 merge CLI parser."""

    parser = argparse.ArgumentParser(
        description="Merge distributed pass-1 worker partials into canonical run-root artifacts."
    )
    parser.add_argument("--manifest", required=True, help="Path to distributed manifest JSON")
    parser.add_argument(
        "--disable-seq-latent-index",
        action="store_true",
        help="Skip seq_latent_index directory merge",
    )
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--mid-ctx-num-ctx-sequences", type=int, default=None)
    parser.add_argument("--mid-ctx-band-low-sigma", type=float, default=0.5)
    parser.add_argument("--mid-ctx-band-high-sigma", type=float, default=1.5)
    parser.add_argument(
        "--mid-ctx-on-truncation",
        choices=["fail", "replay_fallback", "allow_bounded_approx"],
        default="replay_fallback",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for standalone pass-1 artifact merge."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    result = merge_pass1_worker_outputs(
        manifest,
        seq_latent_index_enabled=not args.disable_seq_latent_index,
        vocab_size=args.vocab_size,
        mid_ctx_num_ctx_sequences=args.mid_ctx_num_ctx_sequences,
        mid_ctx_band_low_sigma=args.mid_ctx_band_low_sigma,
        mid_ctx_band_high_sigma=args.mid_ctx_band_high_sigma,
        mid_ctx_on_truncation=args.mid_ctx_on_truncation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = ["build_arg_parser", "main"]
