"""Worker command construction and launching for distributed controller planning."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Literal, Sequence

from .controller_contracts import WorkerCommand
from .devices import worker_environment
from .manifest import DistributedRunManifest


def build_worker_commands(
    manifest: DistributedRunManifest,
    project_root: str | Path,
    *,
    phase: Literal["pass1", "pass2", "discovery"] = "pass1",
) -> List[WorkerCommand]:
    commands: List[WorkerCommand] = []
    by_worker_id = {assignment.worker_id: assignment for assignment in manifest.devices}
    for worker_id in range(manifest.worker_count):
        assignment = by_worker_id[worker_id]
        environment = worker_environment(assignment)
        environment["PYTHONPATH"] = _worker_pythonpath(Path(project_root))
        commands.append(
            WorkerCommand(
                worker_id=worker_id,
                command=[
                    sys.executable,
                    "-m",
                    "pipeline.distributed.worker",
                    "--manifest",
                    manifest.manifest_path,
                    "--phase",
                    phase,
                    "--worker-id",
                    str(worker_id),
                ],
                environment=environment,
                cwd=Path(project_root),
            )
        )
    return commands


def launch_worker_processes(
    worker_commands: Sequence[WorkerCommand],
) -> List[subprocess.Popen]:
    processes: List[subprocess.Popen] = []
    for worker_command in worker_commands:
        env = os.environ.copy()
        env.update(worker_command.environment)
        processes.append(
            subprocess.Popen(
                worker_command.command,
                cwd=worker_command.cwd,
                env=env,
            )
        )
    return processes


def _worker_pythonpath(project_root: Path) -> str:
    src_path = str(project_root / "src")
    existing = os.environ.get("PYTHONPATH")
    if existing:
        return os.pathsep.join([src_path, existing])
    return src_path


__all__ = [
    "build_worker_commands",
    "launch_worker_processes",
]
