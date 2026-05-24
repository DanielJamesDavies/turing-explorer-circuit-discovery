"""CLI for the negative-context pipeline stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from .comparison import compare_negative_context_backends
from .planning import plan_negative_context_stage
from .stage import run_negative_context_stage


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build negative contexts from merged pass-1 artifacts."
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Run root containing top_ctx.pt, mid_ctx.pt, and seq_repr.pt",
    )
    parser.add_argument(
        "--expected-config-hash",
        default=None,
        help="Optional config hash to validate when artifact metadata includes one",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional distributed manifest; neg_ctx uses manifest physical devices by default",
    )
    parser.add_argument(
        "--compare-backends",
        action="store_true",
        help="Build single_gpu_exact and multi_gpu_exact and write an equivalence report",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip neg_ctx rebuild when completed outputs and status metadata match",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned neg_ctx metadata and resume classification without building",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        plan = plan_negative_context_stage(
            args.output_root,
            expected_config_hash=args.expected_config_hash,
            manifest_path=args.manifest,
        )
        print(json.dumps({
            "output_root": str(plan.output_root),
            "part_dir": str(plan.part_dir),
            "resume_status": plan.resume_status,
            "reason": plan.reason,
            "metadata": plan.metadata,
        }, indent=2))
        return
    if args.compare_backends:
        result = compare_negative_context_backends(
            args.output_root,
            expected_config_hash=args.expected_config_hash,
            manifest_path=args.manifest,
        )
        print(f"  ✓ neg_ctx equivalence report saved to {result.report_path}")
        return
    result = run_negative_context_stage(
        args.output_root,
        expected_config_hash=args.expected_config_hash,
        manifest_path=args.manifest,
        resume=args.resume,
    )
    result.stats.print_summary(int(configured_neg_ctx_sequences(result.neg_ctx_path)))
    print(f"  ✓ neg_ctx saved to {result.neg_ctx_path}")
    print(f"  ✓ neg_ctx stats saved to {result.stats_path}")


def configured_neg_ctx_sequences(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["ctx_seq_idx"].shape[2])


__all__ = [
    "configured_neg_ctx_sequences",
    "main",
]
