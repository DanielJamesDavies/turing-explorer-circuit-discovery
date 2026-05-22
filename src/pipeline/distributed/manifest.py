"""Versioned manifest contract for distributed and run-root pipeline execution."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .shard_table import ShardRecord, contains_sequence_id, validate_shard_records


MANIFEST_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1
RUN_SUMMARY_SCHEMA_VERSION = 1
SANITY_REPORT_SCHEMA_VERSION = 1

_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-fA-F]{8}$")


class RunMode(str, Enum):
    SINGLE_PROCESS = "single_process"
    DISTRIBUTED_SIMPLE_EXACT = "distributed_simple_exact"
    DISTRIBUTED_MAPREDUCE_EXACT = "distributed_mapreduce_exact"
    DISTRIBUTED_EXPERIMENTAL_FAST = "distributed_experimental_fast"


class ManifestStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class CleanupPolicy(str, Enum):
    KEEP_ALL = "keep_all"
    DELETE_LARGE_PARTIALS_ON_SUCCESS = "delete_large_partials_on_success"
    DELETE_ALL_PARTIALS_ON_SUCCESS = "delete_all_partials_on_success"
    MANUAL_CLEANUP_ONLY = "manual_cleanup_only"


class ArtifactSchemaVersions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latent_stats: int = 1
    top_ctx: int = 1
    mid_ctx_candidates: int = 1
    seq_repr: int = 1
    logit_ctx: int = 1
    seq_latent_index: int = 1
    candidate_dump: int = 1
    top_coactivation: int = 1
    neg_ctx: int = 1
    circuits: int = 1

    @field_validator("*")
    @classmethod
    def schema_versions_start_at_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("schema versions must be >= 1")
        return value


class DeviceAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: int
    physical_id: Optional[int] = None
    logical_id: str = "cuda:0"
    uuid: Optional[str] = None
    name: Optional[str] = None
    pci_bus_id: Optional[str] = None
    total_vram_bytes: Optional[int] = None
    hostname: Optional[str] = None

    @field_validator("worker_id")
    @classmethod
    def worker_id_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("worker_id must be >= 0")
        return value

    @field_validator("total_vram_bytes")
    @classmethod
    def vram_is_positive_when_present(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("total_vram_bytes must be positive when provided")
        return value

    @model_validator(mode="after")
    def validate_worker_local_device(self) -> "DeviceAssignment":
        if self.physical_id is None:
            if self.logical_id != "cpu":
                raise ValueError("CPU workers must use logical_id='cpu'")
        elif self.logical_id != "cuda:0":
            raise ValueError("CUDA workers must use worker-local logical_id='cuda:0'")
        return self


class WorkAssignments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pass1_shards: Dict[str, List[int]] = Field(default_factory=dict)
    pass1_sequence_totals: Dict[str, int] = Field(default_factory=dict)
    pass2_sequence_ids: Dict[str, List[int]] = Field(default_factory=dict)
    discovery_seed_ids: Dict[str, List[int]] = Field(default_factory=dict)

    @field_validator("pass1_sequence_totals")
    @classmethod
    def pass1_totals_are_non_negative(
        cls,
        value: Dict[str, int],
    ) -> Dict[str, int]:
        for total in value.values():
            if total < 0:
                raise ValueError("pass1 sequence totals must be >= 0")
        return value


class NegativeContextRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "single_gpu_exact"
    selected_devices: List[str] = Field(default_factory=list)
    device_selection_source: Literal[
        "manifest_declared_devices",
        "config_override",
        "standalone_all_visible",
        "single_device",
    ] = "single_device"
    n_neighbors: int = 512
    n_sequences: int = 64
    min_pos_ctx: int = 8
    repr_mode: str = "mean_pool"
    max_repr_seqs: Optional[int] = 200000
    memory_guardrail_fraction: float = 0.90
    fail_on_memory_guardrail: bool = True

    @field_validator("n_neighbors", "n_sequences", "min_pos_ctx")
    @classmethod
    def positive_counts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("negative-context counts must be >= 1")
        return value

    @field_validator("backend")
    @classmethod
    def backend_is_supported(cls, value: str) -> str:
        allowed = {"single_gpu_exact", "multi_gpu_exact", "multi_gpu_index_sharded_exact"}
        if value not in allowed:
            raise ValueError(f"negative-context backend must be one of {sorted(allowed)}")
        return value

    @field_validator("memory_guardrail_fraction")
    @classmethod
    def guardrail_fraction_is_valid(cls, value: float) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError("memory_guardrail_fraction must be in (0, 1]")
        return value


class DistributedRunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION
    metrics_schema_version: int = METRICS_SCHEMA_VERSION
    run_summary_schema_version: int = RUN_SUMMARY_SCHEMA_VERSION
    sanity_report_schema_version: int = SANITY_REPORT_SCHEMA_VERSION
    artifact_schema_versions: ArtifactSchemaVersions = Field(
        default_factory=ArtifactSchemaVersions
    )

    run_id: str
    run_mode: RunMode
    status: ManifestStatus = ManifestStatus.PLANNED
    cleanup_policy: CleanupPolicy = CleanupPolicy.KEEP_ALL
    sampling_seed: int = 0

    created_at: str
    config_path: str
    normalized_config_hash: str
    git_sha: Optional[str] = None
    environment_overrides: Dict[str, str] = Field(default_factory=dict)

    project_root: str
    output_root: str
    distributed_root: str
    manifest_path: str
    metrics_path: str
    run_summary_path: str

    model_path: str
    sae_path: str
    dataset_path: str

    worker_count: int
    devices: List[DeviceAssignment] = Field(default_factory=list)
    shard_table: List[ShardRecord] = Field(default_factory=list)
    work_assignments: WorkAssignments = Field(default_factory=WorkAssignments)
    neg_ctx: NegativeContextRunConfig = Field(default_factory=NegativeContextRunConfig)

    @field_validator(
        "manifest_schema_version",
        "metrics_schema_version",
        "run_summary_schema_version",
        "sanity_report_schema_version",
    )
    @classmethod
    def known_contract_versions(cls, value: int) -> int:
        if value != 1:
            raise ValueError("unsupported schema version")
        return value

    @field_validator("run_id")
    @classmethod
    def run_id_has_expected_shape(cls, value: str) -> str:
        if not _RUN_ID_RE.match(value):
            raise ValueError("run_id must match YYYYMMDD-HHMMSS-<config_hash_8>")
        return value

    @field_validator("normalized_config_hash")
    @classmethod
    def config_hash_is_long_enough(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("normalized_config_hash must be at least 8 characters")
        return value

    @field_validator("worker_count")
    @classmethod
    def worker_count_is_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("worker_count must be >= 1")
        return value

    @field_validator("sampling_seed")
    @classmethod
    def sampling_seed_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("sampling_seed must be >= 0")
        return value

    @model_validator(mode="after")
    def validate_worker_and_output_contract(self) -> "DistributedRunManifest":
        worker_ids = [device.worker_id for device in self.devices]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("device worker IDs must be unique")
        if any(worker_id >= self.worker_count for worker_id in worker_ids):
            raise ValueError("device worker IDs must be less than worker_count")

        physical_ids = [
            device.physical_id for device in self.devices if device.physical_id is not None
        ]
        if len(physical_ids) != len(set(physical_ids)):
            raise ValueError("physical device IDs must be unique")

        expected_output_root = str(Path("outputs") / self.run_id)
        normalized_output_root = self.output_root.replace("\\", "/")
        if not normalized_output_root.endswith(expected_output_root.replace("\\", "/")):
            raise ValueError("output_root must end with outputs/<run_id>")

        expected_distributed_root = str(Path(self.output_root) / "distributed")
        if Path(self.distributed_root) != Path(expected_distributed_root):
            raise ValueError("distributed_root must be output_root/distributed")

        expected_manifest_path = str(Path(self.distributed_root) / "manifest.json")
        if Path(self.manifest_path) != Path(expected_manifest_path):
            raise ValueError("manifest_path must be distributed_root/manifest.json")

        if self.shard_table:
            validate_shard_records(self.shard_table)
            valid_shards = {record.shard_index for record in self.shard_table}
            for worker_id, shard_ids in self.work_assignments.pass1_shards.items():
                _validate_worker_key(worker_id, self.worker_count)
                expected_total = 0
                for shard_id in shard_ids:
                    if shard_id not in valid_shards:
                        raise ValueError("assigned shard index out of range")
                    expected_total += next(
                        record.sequence_count
                        for record in self.shard_table
                        if record.shard_index == shard_id
                    )
                declared_total = self.work_assignments.pass1_sequence_totals.get(worker_id)
                if declared_total is not None and declared_total != expected_total:
                    raise ValueError("pass1 sequence total does not match assigned shards")

            for worker_id in self.work_assignments.pass1_sequence_totals:
                _validate_worker_key(worker_id, self.worker_count)

            seen_sequence_ids: set[int] = set()
            for worker_id, sequence_ids in self.work_assignments.pass2_sequence_ids.items():
                _validate_worker_key(worker_id, self.worker_count)
                for sequence_id in sequence_ids:
                    if sequence_id in seen_sequence_ids:
                        raise ValueError("duplicated assigned sequence ID")
                    seen_sequence_ids.add(sequence_id)
                    if not contains_sequence_id(self.shard_table, sequence_id):
                        raise ValueError("assigned sequence ID out of range")

        return self


def generate_run_id(
    normalized_config_hash: str,
    *,
    timestamp: Optional[datetime] = None,
) -> str:
    """Generate a stable-shaped run ID from UTC timestamp and config hash."""

    if len(normalized_config_hash) < 8:
        raise ValueError("normalized_config_hash must be at least 8 characters")
    now = timestamp or datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{normalized_config_hash[:8].lower()}"


def save_manifest(manifest: DistributedRunManifest, path: str | Path) -> None:
    """Write a manifest JSON file, creating parent directories if needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_manifest(path: str | Path) -> DistributedRunManifest:
    """Load and validate a manifest JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DistributedRunManifest.model_validate(data)


def _validate_worker_key(worker_id: str, worker_count: int) -> None:
    try:
        parsed = int(worker_id)
    except ValueError as exc:
        raise ValueError("work assignment worker keys must be integer strings") from exc
    if parsed < 0 or parsed >= worker_count:
        raise ValueError("work assignment worker key out of range")
