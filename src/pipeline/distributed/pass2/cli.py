"""CLI entrypoint for distributed pass-2 reduce."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .simple import run_simple_exact_reduce_stage


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the simple exact pass-2 reduce CLI parser."""

    parser = argparse.ArgumentParser(
        description="Reduce distributed pass-2 candidate dumps into top_coactivation.pt."
    )
    parser.add_argument("--output-root", required=True, help="Canonical run root")
    parser.add_argument(
        "--candidate-dump",
        action="append",
        required=True,
        help="Path to a worker candidate_dump.partial.pt file. Repeat for every worker.",
    )
    parser.add_argument("--top-ctx", default=None, help="Path to top_ctx.pt")
    parser.add_argument("--latent-stats", default=None, help="Path to latent_stats.pt for PMI mode")
    parser.add_argument("--expected-config-hash", default=None)
    parser.add_argument("--mode", default=None, help="Expected coactivation mode")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for simple exact distributed pass-2 reduce."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_simple_exact_reduce_stage(
        output_root=args.output_root,
        candidate_dump_paths=args.candidate_dump,
        top_ctx_path=args.top_ctx,
        latent_stats_path=args.latent_stats,
        expected_config_hash=args.expected_config_hash,
        expected_mode=args.mode,
    )
    print(json.dumps(result.report, indent=2, sort_keys=True))


__all__ = ["build_arg_parser", "main"]
