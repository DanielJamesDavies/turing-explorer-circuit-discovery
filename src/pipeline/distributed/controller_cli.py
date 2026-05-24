"""CLI entrypoint helpers for distributed controller planning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .controller_commands import build_worker_commands, launch_worker_processes
from .controller_config import (
    _candidate_dump_m_from_config,
    _distributed_cli_defaults,
    _parse_physical_ids,
    load_and_hash_config,
)
from .controller_contracts import ControllerPlan
from .controller_dry_run import format_dry_run
from .controller_planning import plan_distributed_run
from .manifest import CleanupPolicy, RunMode


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the model-free distributed controller CLI parser."""

    parser = argparse.ArgumentParser(
        description="Plan or launch manifest-assigned distributed pipeline workers."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--project-root", default=".", help="Repository/project root")
    parser.add_argument("--output-base", default=None, help="Base output directory")
    parser.add_argument("--run-id", default=None, help="Run ID, or config value if omitted")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RunMode],
        default=None,
        help="Operating mode, or config distributed.mode if omitted",
    )
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated physical CUDA device IDs, e.g. 0,1,2,3",
    )
    parser.add_argument("--use-cpu", action="store_true", help="Plan CPU worker devices")
    parser.add_argument(
        "--cleanup-policy",
        choices=[policy.value for policy in CleanupPolicy],
        default=None,
    )
    parser.add_argument(
        "--part",
        action="append",
        default=None,
        help="Selected part/native preflight gate. May be supplied more than once.",
    )
    parser.add_argument(
        "--phase",
        choices=["pass1", "pass2", "discovery"],
        default="pass1",
        help="Worker phase for emitted commands",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing run root")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch worker subprocesses after planning",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for controller dry-runs and optional subprocess launch."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    plan = plan_distributed_run_from_args(args)
    print(plan.dry_run_text)
    if args.launch:
        processes = launch_worker_processes(
            build_worker_commands(plan.manifest, args.project_root, phase=args.phase)
        )
        print(f"launched_workers: {len(processes)}")


def plan_distributed_run_from_args(args: argparse.Namespace) -> ControllerPlan:
    """Create a controller plan from CLI args and Phase 2 config defaults."""

    defaults = _distributed_cli_defaults(args.config)
    worker_count = args.worker_count or int(defaults["worker_count"])
    output_base = Path(args.output_base or str(defaults["output_base"]))
    run_mode = RunMode(args.mode or str(defaults["mode"]))
    cleanup_policy = CleanupPolicy(args.cleanup_policy or str(defaults["cleanup_policy"]))
    selected_parts = tuple(args.part if args.part is not None else defaults["parts"])
    physical_ids = _parse_physical_ids(args.devices, defaults["devices"])
    resume = bool(args.resume or defaults["resume_policy"] in {"resume", "auto"})

    plan = plan_distributed_run(
        config_path=args.config,
        project_root=args.project_root,
        output_base=output_base,
        worker_count=worker_count,
        run_mode=run_mode,
        run_id=args.run_id or defaults["run_id"],
        resume=resume,
        physical_ids=physical_ids,
        use_cpu=bool(args.use_cpu),
        cleanup_policy=cleanup_policy,
        selected_parts=selected_parts,
    )
    if args.phase != "pass1":
        worker_commands = build_worker_commands(plan.manifest, args.project_root, phase=args.phase)
        return ControllerPlan(
            manifest=plan.manifest,
            layout=plan.layout,
            preflight=plan.preflight,
            worker_commands=worker_commands,
            discovery_estimate=plan.discovery_estimate,
            local_compatibility=plan.local_compatibility,
            h100_exact_mode=plan.h100_exact_mode,
            dry_run_text=format_dry_run(
                plan.manifest,
                plan.preflight,
                worker_commands,
                pass2_dump_m=_candidate_dump_m_from_config(load_and_hash_config(args.config)[0]),
                discovery_estimate=plan.discovery_estimate,
                local_compatibility=plan.local_compatibility,
                h100_exact_mode=plan.h100_exact_mode,
            ),
        )
    return plan


__all__ = [
    "build_arg_parser",
    "main",
    "plan_distributed_run_from_args",
]
