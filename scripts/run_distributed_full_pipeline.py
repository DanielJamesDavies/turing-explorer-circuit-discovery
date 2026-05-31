#!/usr/bin/env python
"""Run a distributed manifest through the full exact pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all distributed pipeline stages for an existing manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to distributed/manifest.json")
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--keep-pass1-partials",
        action="store_true",
        help="Do not delete worker pass1 partials after pass1 merge succeeds.",
    )
    parser.add_argument(
        "--cleanup-pass2-partials",
        action="store_true",
        help="Delete worker pass2 candidate dumps after pass2 reduce succeeds.",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=int(os.environ.get("TURING_WORKER_THREADS", "4")),
        help=(
            "Default CPU thread cap for distributed worker subprocesses. "
            "Set to 0 to leave thread env vars unchanged. Defaults to 4, "
            "or TURING_WORKER_THREADS when set."
        ),
    )
    args = parser.parse_args(argv)
    if args.worker_threads < 0:
        parser.error("--worker-threads must be >= 0")

    from pipeline.distributed.manifest import load_manifest

    manifest = load_manifest(args.manifest)
    project_root = Path(args.project_root).resolve()
    output_root = Path(manifest.output_root)
    distributed_root = Path(manifest.distributed_root)

    print(f"project_root: {project_root}")
    print(f"output_root: {output_root}")
    print(f"manifest: {manifest.manifest_path}")
    print(f"worker_count: {manifest.worker_count}")
    print(f"worker_threads: {args.worker_threads or 'unchanged'}")

    env = _base_env(project_root)
    _run_worker_phase(manifest, "pass1", project_root, env, worker_threads=args.worker_threads)
    _run_logged(
        [
            sys.executable,
            "-m",
            "pipeline.distributed.pass1_merge",
            "--manifest",
            manifest.manifest_path,
        ],
        log_path=distributed_root / "pass1_merge.log",
        cwd=project_root,
        env=env,
    )

    if not args.keep_pass1_partials:
        _cleanup_worker_partials(manifest, "pass1")

    _remove_if_exists(output_root / "neg_ctx.pt")
    _remove_if_exists(output_root / "neg_ctx_stats.json")
    shutil.rmtree(distributed_root / "parts" / "neg_ctx", ignore_errors=True)
    _run_logged(
        [
            sys.executable,
            "-m",
            "pipeline.negative_context",
            "--output-root",
            str(output_root),
            "--manifest",
            manifest.manifest_path,
        ],
        log_path=distributed_root / "neg_ctx_multi_gpu.log",
        cwd=project_root,
        env=env,
    )

    _run_worker_phase(manifest, "pass2", project_root, env, worker_threads=args.worker_threads)
    _run_logged(
        [
            sys.executable,
            "-m",
            "pipeline.distributed.pass2_reduce",
            "--output-root",
            str(output_root),
            "--top-ctx",
            str(output_root / "top_ctx.pt"),
            "--latent-stats",
            str(output_root / "latent_stats.pt"),
            *_candidate_dump_args(manifest),
        ],
        log_path=distributed_root / "pass2_reduce.log",
        cwd=project_root,
        env=env,
    )

    if args.cleanup_pass2_partials:
        _cleanup_worker_partials(manifest, "pass2")

    _run_logged(
        [
            sys.executable,
            "-m",
            "pipeline.candidate_selection",
            "--output-root",
            str(output_root),
            "--manifest",
            manifest.manifest_path,
        ],
        log_path=distributed_root / "candidate_selection.log",
        cwd=project_root,
        env=env,
    )

    _run_worker_phase(manifest, "discovery", project_root, env, worker_threads=args.worker_threads)
    _run_discovery_merge(manifest)
    _print_final_artifact_check(output_root)
    return 0


def _base_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(project_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _run_worker_phase(
    manifest: Any,
    phase: str,
    project_root: Path,
    env: dict[str, str],
    *,
    worker_threads: int,
) -> None:
    print(f"\n=== {phase} workers ===")
    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
    distributed_root = Path(manifest.distributed_root)
    for worker_id in range(manifest.worker_count):
        log_path = distributed_root / f"{phase}_worker_{worker_id:03d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("wb")
        worker_env = env.copy()
        _apply_worker_thread_limits(worker_env, worker_threads)
        physical_id = _physical_device_id(manifest, worker_id)
        if physical_id is not None:
            worker_env["CUDA_VISIBLE_DEVICES"] = str(physical_id)
        cmd = [
            sys.executable,
            "-m",
            "pipeline.distributed.worker",
            "--manifest",
            manifest.manifest_path,
            "--phase",
            phase,
            "--worker-id",
            str(worker_id),
        ]
        print(f"worker {worker_id}: {' '.join(cmd)} > {log_path}")
        processes.append(
            (
                worker_id,
                subprocess.Popen(
                    cmd,
                    cwd=project_root,
                    env=worker_env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                ),
                log_handle,
            )
        )

    failures: list[str] = []
    for worker_id, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append(f"worker {worker_id} exited {return_code}")

    if failures:
        raise RuntimeError(f"{phase} failed: {', '.join(failures)}")


def _apply_worker_thread_limits(env: dict[str, str], worker_threads: int) -> None:
    if worker_threads <= 0:
        return
    value = str(worker_threads)
    for name in THREAD_ENV_VARS:
        env.setdefault(name, value)


def _physical_device_id(manifest: Any, worker_id: int) -> int | None:
    for device in manifest.devices:
        if device.worker_id == worker_id:
            return device.physical_id
    return worker_id


def _run_logged(
    cmd: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
) -> None:
    print(f"\n=== {' '.join(cmd)} ===")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            list(cmd),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.readline(), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_handle.write(chunk)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {' '.join(cmd)}")


def _candidate_dump_args(manifest: Any) -> list[str]:
    args: list[str] = []
    for worker_id in range(manifest.worker_count):
        args.extend(
            [
                "--candidate-dump",
                str(
                    Path(manifest.distributed_root)
                    / "workers"
                    / f"worker_{worker_id:03d}"
                    / "pass2"
                    / "candidate_dump.partial.pt"
                ),
            ]
        )
    return args


def _cleanup_worker_partials(manifest: Any, phase: str) -> None:
    print(f"\n=== cleanup {phase} worker partials ===")
    for worker_id in range(manifest.worker_count):
        path = (
            Path(manifest.distributed_root)
            / "workers"
            / f"worker_{worker_id:03d}"
            / phase
        )
        shutil.rmtree(path, ignore_errors=True)
        print(f"removed {path}")


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
        print(f"removed {path}")
    except FileNotFoundError:
        return


def _run_discovery_merge(manifest: Any) -> None:
    from pipeline.distributed.discovery_merge import run_circuit_store_merge

    print("\n=== discovery merge ===")
    result = run_circuit_store_merge(manifest)
    print(
        json.dumps(
            {
                "merged_circuit_count": result.merged_circuit_count,
                "worker_circuit_counts": result.worker_circuit_counts,
                "report_path": str(result.report_path),
                "summary_path": str(result.summary_path),
            },
            indent=2,
        )
    )


def _print_final_artifact_check(output_root: Path) -> None:
    print("\n=== final artifact check ===")
    for path in [
        output_root,
        output_root / "circuits",
        output_root / "distributed" / "reports",
    ]:
        print(f"{path}:")
        if path.exists():
            for child in sorted(path.iterdir()):
                print(f"  {child.name}\t{_format_size(child)}")
        else:
            print("  missing")
    usage = shutil.disk_usage(output_root)
    print(
        "disk:",
        f"used={_format_bytes(usage.used)}",
        f"free={_format_bytes(usage.free)}",
        f"total={_format_bytes(usage.total)}",
    )


def _format_size(path: Path) -> str:
    if path.is_dir():
        return "<dir>"
    return _format_bytes(path.stat().st_size)


def _format_bytes(n_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


if __name__ == "__main__":
    raise SystemExit(main())
