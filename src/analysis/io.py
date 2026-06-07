"""Filesystem helpers for run-local analysis outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def resolve_run_root(run_root: str | Path) -> Path:
    """Resolve and validate a pipeline run root."""

    path = Path(run_root).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"run root does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"run root is not a directory: {path}")
    return path


def analysis_output_dirs(
    run_root: str | Path,
    suite_name: str,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Create and return the standard output directories for one analysis suite."""

    root = Path(output_root).expanduser().resolve() if output_root is not None else resolve_run_root(run_root)
    suite_root = root / "analysis" / suite_name
    dirs = {
        "root": suite_root,
        "figures": suite_root / "figures",
        "summaries": suite_root / "summaries",
        "tables": suite_root / "tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write deterministic, human-readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> Path:
    """Write rows to CSV with a stable header."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path

