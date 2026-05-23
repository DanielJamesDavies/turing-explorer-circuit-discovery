"""Run summary, UX reporting, and lightweight observability helpers."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .layout import build_run_layout
from .manifest import DistributedRunManifest, RunMode
from .operating_modes import operating_mode_definition
from .rollout_gates import RolloutGateReport


OBSERVABILITY_SAMPLE_SCHEMA_VERSION = 1


class ObservabilitySample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = OBSERVABILITY_SAMPLE_SCHEMA_VERSION
    run_id: str
    phase: str
    timestamp: str
    worker_id: Optional[int] = None
    physical_id: Optional[int] = None
    logical_id: Optional[str] = None
    pid: int = Field(default_factory=os.getpid)
    hostname: str = Field(default_factory=socket.gethostname)
    gpu_utilization_percent: Optional[float] = None
    vram_used_bytes: Optional[int] = None
    vram_total_bytes: Optional[int] = None
    power_watts: Optional[float] = None
    temperature_c: Optional[float] = None
    cpu_ram_used_bytes: Optional[int] = None
    cpu_ram_total_bytes: Optional[int] = None
    disk_used_bytes: Optional[int] = None
    disk_total_bytes: Optional[int] = None
    disk_write_bytes_per_s: Optional[float] = None

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: int) -> int:
        if value != OBSERVABILITY_SAMPLE_SCHEMA_VERSION:
            raise ValueError("unsupported observability sample schema version")
        return value

    @field_validator(
        "gpu_utilization_percent",
        "power_watts",
        "temperature_c",
        "disk_write_bytes_per_s",
    )
    @classmethod
    def optional_floats_are_non_negative(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("observability float values must be >= 0")
        return value

    @field_validator(
        "vram_used_bytes",
        "vram_total_bytes",
        "cpu_ram_used_bytes",
        "cpu_ram_total_bytes",
        "disk_used_bytes",
        "disk_total_bytes",
    )
    @classmethod
    def optional_ints_are_non_negative(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError("observability byte values must be >= 0")
        return value

    @model_validator(mode="after")
    def utilization_is_percent(self) -> "ObservabilitySample":
        if self.gpu_utilization_percent is not None and self.gpu_utilization_percent > 100:
            raise ValueError("gpu_utilization_percent must be <= 100")
        return self


def build_mode_summary_report(
    manifest: DistributedRunManifest,
    *,
    rollout_report: Optional[RolloutGateReport] = None,
    warnings: Sequence[str] = (),
) -> Dict[str, object]:
    """Build a concise mode summary for logs and run reports."""

    mode_definition = operating_mode_definition(manifest.run_mode)
    return {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "run_mode": manifest.run_mode.value,
        "description": mode_definition.description,
        "exactness_status": _exactness_status(manifest.run_mode),
        "rollout_ok": rollout_report.ok if rollout_report is not None else None,
        "worker_count": manifest.worker_count,
        "output_root": manifest.output_root,
        "distributed_root": manifest.distributed_root,
        "cleanup_policy": manifest.cleanup_policy.value,
        "warnings": list(warnings),
        "hardware": build_hardware_context(manifest),
    }


def build_final_run_report(
    manifest: DistributedRunManifest,
    *,
    part_statuses: Mapping[str, str],
    artifacts: Mapping[str, str | Path],
    rollout_report: Optional[RolloutGateReport] = None,
    equivalence_reports: Mapping[str, str | Path] = {},
    benchmark_report: str | Path | None = None,
    warnings: Sequence[str] = (),
) -> Dict[str, object]:
    """Build the final report linking parts, artifacts, equivalence, and benchmarks."""

    mode_summary = build_mode_summary_report(
        manifest,
        rollout_report=rollout_report,
        warnings=warnings,
    )
    return {
        "schema_version": 1,
        "mode_summary": mode_summary,
        "part_statuses": dict(part_statuses),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "equivalence_reports": {
            name: _load_report_summary(path) for name, path in equivalence_reports.items()
        },
        "benchmark_report": (
            _load_report_summary(benchmark_report) if benchmark_report is not None else None
        ),
        "rollout": (
            {
                "ok": rollout_report.ok,
                "issues": list(rollout_report.issues),
                "required_paths": [str(path) for path in rollout_report.required_paths],
            }
            if rollout_report is not None
            else None
        ),
        "warnings": list(warnings),
    }


def build_hardware_context(manifest: DistributedRunManifest) -> Dict[str, object]:
    """Summarize physical/logical device metadata recorded in the manifest."""

    devices = []
    total_vram_bytes = 0
    for assignment in manifest.devices:
        if assignment.total_vram_bytes is not None:
            total_vram_bytes += int(assignment.total_vram_bytes)
        devices.append(
            {
                "worker_id": assignment.worker_id,
                "physical_id": assignment.physical_id,
                "logical_id": assignment.logical_id,
                "uuid": assignment.uuid,
                "name": assignment.name,
                "pci_bus_id": assignment.pci_bus_id,
                "total_vram_bytes": assignment.total_vram_bytes,
                "hostname": assignment.hostname,
            }
        )
    return {
        "device_count": len(devices),
        "total_vram_bytes": total_vram_bytes or None,
        "devices": devices,
    }


def save_run_report(report: Mapping[str, object], path: str | Path) -> None:
    """Write a stable JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def save_mode_summary_report(
    manifest: DistributedRunManifest,
    *,
    rollout_report: Optional[RolloutGateReport] = None,
    warnings: Sequence[str] = (),
) -> Path:
    """Save `mode_summary.json` under the distributed reports directory."""

    layout = build_run_layout(manifest)
    path = layout.reports_dir / "mode_summary.json"
    save_run_report(
        build_mode_summary_report(manifest, rollout_report=rollout_report, warnings=warnings),
        path,
    )
    return path


def append_observability_sample(sample: ObservabilitySample, path: str | Path) -> None:
    """Append one observability sample as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(sample.model_dump_json() + "\n")


def _exactness_status(mode: RunMode) -> str:
    if mode == RunMode.SINGLE_PROCESS:
        return "single_process_oracle"
    if mode == RunMode.DISTRIBUTED_SIMPLE_EXACT:
        return "exact_equivalent"
    if mode == RunMode.DISTRIBUTED_MAPREDUCE_EXACT:
        return "exact_mapreduce_equivalent"
    return "experimental_non_exact"


def _load_report_summary(path: str | Path) -> Dict[str, object]:
    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must be a JSON object: {report_path}")
    return {
        "path": str(report_path),
        "status": payload.get("status"),
        "ok": payload.get("ok"),
        "validation": payload.get("validation"),
        "equivalence": payload.get("equivalence"),
        "benchmark": payload.get("benchmark"),
    }
