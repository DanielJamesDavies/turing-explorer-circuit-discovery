"""Versioned partial artifact schema for distributed pass-2 candidate dumps."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from pipeline.second_pass import SecondPassDumpResult

from .manifest import DistributedRunManifest
from .pass2_replay import get_pass2_worker_input


PASS2_CANDIDATE_DUMP_SCHEMA_VERSION = 1
PASS2_PREAGGREGATION_SCHEMA_VERSION = 1
CANDIDATE_DUMP_BYTES_PER_ENTRY = 8


@dataclass(frozen=True)
class CandidateDumpMemoryEstimate:
    sequence_count: int
    m: int
    candidate_ids_bytes: int
    candidate_vals_bytes: int
    total_bytes: int
    guardrail_bytes: Optional[int] = None
    exceeds_guardrail: bool = False


class CandidateDumpMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    partial_schema_version: int = PASS2_CANDIDATE_DUMP_SCHEMA_VERSION
    artifact_name: str = "candidate_dump"
    run_id: str
    worker_id: int
    sequence_count: int
    sequence_id_min: Optional[int] = None
    sequence_id_max: Optional[int] = None
    replay_sequence_hash: Optional[str] = None
    config_hash: str
    physical_id: Optional[int] = None
    logical_id: str
    created_at: str
    mode: str
    m: int
    n_candidates_per_component: int
    n_latents_per_latent: int
    num_components: int
    d_sae: int
    token_count: int = 0
    seq_len: int = 0
    batch_count: int = 0
    estimated_dump_bytes: int = 0

    @field_validator("partial_schema_version")
    @classmethod
    def schema_is_supported(cls, value: int) -> int:
        if value != PASS2_CANDIDATE_DUMP_SCHEMA_VERSION:
            raise ValueError("unsupported candidate dump schema version")
        return value

    @field_validator(
        "worker_id",
        "sequence_count",
        "m",
        "n_candidates_per_component",
        "n_latents_per_latent",
        "num_components",
        "d_sae",
        "token_count",
        "seq_len",
        "batch_count",
        "estimated_dump_bytes",
    )
    @classmethod
    def counts_are_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("candidate dump counts must be >= 0")
        return value

    @model_validator(mode="after")
    def sequence_bounds_match_count(self) -> "CandidateDumpMetadata":
        if self.artifact_name != "candidate_dump":
            raise ValueError("candidate dump artifact_name must be candidate_dump")
        if self.sequence_count == 0:
            if self.sequence_id_min is not None or self.sequence_id_max is not None:
                raise ValueError("empty candidate dumps must not declare sequence ID bounds")
        elif self.sequence_id_min is None or self.sequence_id_max is None:
            raise ValueError("non-empty candidate dumps must declare sequence ID bounds")
        elif self.sequence_id_min > self.sequence_id_max:
            raise ValueError("sequence_id_min must be <= sequence_id_max")
        return self


class CandidatePreAggregationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    partial_schema_version: int = PASS2_PREAGGREGATION_SCHEMA_VERSION
    artifact_name: str = "candidate_preaggregation"
    run_id: str
    worker_id: int
    source_candidate_dump_schema_version: int
    sequence_count: int
    contribution_count: int
    config_hash: str
    mode: str
    num_components: int
    d_sae: int
    m: int
    target_start_id: int = 0
    target_end_id: Optional[int] = None
    created_at: str

    @field_validator("partial_schema_version")
    @classmethod
    def schema_is_supported(cls, value: int) -> int:
        if value != PASS2_PREAGGREGATION_SCHEMA_VERSION:
            raise ValueError("unsupported pass2 preaggregation schema version")
        return value

    @field_validator(
        "source_candidate_dump_schema_version",
        "sequence_count",
        "contribution_count",
        "num_components",
        "d_sae",
        "m",
        "target_start_id",
    )
    @classmethod
    def counts_are_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("pass2 preaggregation counts must be >= 0")
        return value

    @model_validator(mode="after")
    def validate_preaggregation_contract(self) -> "CandidatePreAggregationMetadata":
        if self.artifact_name != "candidate_preaggregation":
            raise ValueError("preaggregation artifact_name must be candidate_preaggregation")
        flattened_target_count = self.num_components * self.d_sae
        if self.target_end_id is None:
            self.target_end_id = flattened_target_count
        if self.target_end_id < self.target_start_id:
            raise ValueError("target_end_id must be greater than or equal to target_start_id")
        if self.target_end_id > flattened_target_count:
            raise ValueError("target_end_id exceeds flattened target count")
        return self


def build_candidate_dump_metadata(
    manifest: DistributedRunManifest,
    worker_id: int,
    top_coactivation_store,
    dump_result: SecondPassDumpResult,
) -> CandidateDumpMetadata:
    worker_input = get_pass2_worker_input(manifest, worker_id)
    device = _device_for_worker(manifest, worker_id)
    return CandidateDumpMetadata(
        run_id=manifest.run_id,
        worker_id=worker_id,
        sequence_count=worker_input.sequence_count,
        sequence_id_min=worker_input.sequence_id_min,
        sequence_id_max=worker_input.sequence_id_max,
        replay_sequence_hash=worker_input.replay_sequence_hash,
        config_hash=manifest.normalized_config_hash,
        physical_id=device.physical_id if device is not None else None,
        logical_id=device.logical_id if device is not None else "cpu",
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        mode=top_coactivation_store.mode,
        m=int(top_coactivation_store.M),
        n_candidates_per_component=int(top_coactivation_store.n_candidates_per_component),
        n_latents_per_latent=int(top_coactivation_store.n_latents_per_latent),
        num_components=int(top_coactivation_store.num_components),
        d_sae=int(top_coactivation_store.d_sae),
        token_count=int(top_coactivation_store.total_tokens_processed),
        seq_len=int(dump_result.seq_len),
        batch_count=int(dump_result.batch_count),
        estimated_dump_bytes=estimate_candidate_dump_bytes(
            worker_input.sequence_count,
            int(top_coactivation_store.M),
        ).total_bytes,
    )


def estimate_candidate_dump_bytes(
    sequence_count: int,
    m: int,
    *,
    guardrail_bytes: Optional[int] = None,
) -> CandidateDumpMemoryEstimate:
    """Estimate simple dump tensor memory: int32 IDs plus float32 values."""

    if sequence_count < 0:
        raise ValueError("sequence_count must be >= 0")
    if m < 0:
        raise ValueError("m must be >= 0")
    candidate_ids_bytes = int(sequence_count) * int(m) * 4
    candidate_vals_bytes = int(sequence_count) * int(m) * 4
    total_bytes = candidate_ids_bytes + candidate_vals_bytes
    return CandidateDumpMemoryEstimate(
        sequence_count=int(sequence_count),
        m=int(m),
        candidate_ids_bytes=candidate_ids_bytes,
        candidate_vals_bytes=candidate_vals_bytes,
        total_bytes=total_bytes,
        guardrail_bytes=guardrail_bytes,
        exceeds_guardrail=guardrail_bytes is not None and total_bytes > guardrail_bytes,
    )


def check_candidate_dump_memory_guardrail(
    sequence_count: int,
    m: int,
    *,
    guardrail_bytes: Optional[int],
    fail_on_guardrail: bool,
) -> CandidateDumpMemoryEstimate:
    """Warn or fail when a worker candidate dump exceeds configured memory."""

    estimate = estimate_candidate_dump_bytes(
        sequence_count,
        m,
        guardrail_bytes=guardrail_bytes,
    )
    if estimate.exceeds_guardrail:
        message = (
            "pass2 candidate dump estimate exceeds guardrail: "
            f"{estimate.total_bytes} bytes > {guardrail_bytes} bytes "
            f"(sequences={sequence_count}, M={m})"
        )
        if fail_on_guardrail:
            raise MemoryError(message)
        print(f"  [pass2] WARNING: {message}")
    return estimate


def build_candidate_preaggregation_metadata(
    candidate_metadata: CandidateDumpMetadata,
    contribution_count: int,
) -> CandidatePreAggregationMetadata:
    return CandidatePreAggregationMetadata(
        run_id=candidate_metadata.run_id,
        worker_id=candidate_metadata.worker_id,
        source_candidate_dump_schema_version=candidate_metadata.partial_schema_version,
        sequence_count=candidate_metadata.sequence_count,
        contribution_count=contribution_count,
        config_hash=candidate_metadata.config_hash,
        mode=candidate_metadata.mode,
        num_components=candidate_metadata.num_components,
        d_sae=candidate_metadata.d_sae,
        m=candidate_metadata.m,
        target_start_id=0,
        target_end_id=candidate_metadata.num_components * candidate_metadata.d_sae,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def candidate_dump_payload(
    top_coactivation_store,
    sequence_ids: list[int],
) -> Dict[str, Any]:
    if top_coactivation_store.candidate_ids is None or top_coactivation_store.candidate_vals is None:
        raise RuntimeError("candidate dump buffers are not initialized")
    return {
        "sequence_ids": torch.tensor(sequence_ids, dtype=torch.int64),
        "candidate_ids": top_coactivation_store.candidate_ids.detach().cpu().to(torch.int32),
        "candidate_vals": top_coactivation_store.candidate_vals.detach().cpu().to(torch.float32),
        "total_tokens_processed": int(top_coactivation_store.total_tokens_processed),
    }


def save_candidate_dump_partial(
    path: str | Path,
    metadata: CandidateDumpMetadata,
    payload: Dict[str, Any],
) -> None:
    data = {"metadata": metadata.model_dump(mode="json"), "payload": payload}
    validate_candidate_dump_partial(data)
    _atomic_torch_save(data, path)


def load_candidate_dump_partial(
    path: str | Path,
    *,
    expected_config_hash: Optional[str] = None,
) -> tuple[CandidateDumpMetadata, Dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return validate_candidate_dump_partial(
        data,
        expected_config_hash=expected_config_hash,
    )


def validate_candidate_dump_partial(
    data: Dict[str, Any],
    *,
    expected_config_hash: Optional[str] = None,
) -> tuple[CandidateDumpMetadata, Dict[str, Any]]:
    if not isinstance(data, dict) or "metadata" not in data or "payload" not in data:
        raise ValueError("candidate dump partial must contain metadata and payload")
    metadata = CandidateDumpMetadata.model_validate(data["metadata"])
    payload = data["payload"]
    if not isinstance(payload, dict):
        raise ValueError("candidate dump partial payload must be a dict")
    if expected_config_hash is not None and metadata.config_hash != expected_config_hash:
        raise ValueError("candidate dump config hash mismatch")

    sequence_ids = _required_tensor(payload, "sequence_ids")
    candidate_ids = _required_tensor(payload, "candidate_ids")
    candidate_vals = _required_tensor(payload, "candidate_vals")
    if sequence_ids.dtype != torch.int64:
        raise ValueError("sequence_ids must be int64")
    if candidate_ids.dtype != torch.int32:
        raise ValueError("candidate_ids must be int32")
    if candidate_vals.dtype != torch.float32:
        raise ValueError("candidate_vals must be float32")

    expected_shape = (metadata.sequence_count, metadata.m)
    if tuple(candidate_ids.shape) != expected_shape or tuple(candidate_vals.shape) != expected_shape:
        raise ValueError("candidate dump tensors have unexpected shape")
    if tuple(sequence_ids.shape) != (metadata.sequence_count,):
        raise ValueError("sequence_ids shape does not match sequence_count")
    if sequence_ids.tolist() != sorted(sequence_ids.tolist()):
        raise ValueError("sequence_ids must preserve sorted replay order")
    if len(set(int(sequence_id) for sequence_id in sequence_ids.tolist())) != metadata.sequence_count:
        raise ValueError("sequence_ids must not contain duplicates")
    if metadata.sequence_count:
        if int(sequence_ids.min().item()) != metadata.sequence_id_min:
            raise ValueError("sequence_id_min does not match sequence_ids")
        if int(sequence_ids.max().item()) != metadata.sequence_id_max:
            raise ValueError("sequence_id_max does not match sequence_ids")

    if not torch.isfinite(candidate_vals).all():
        raise ValueError("candidate_vals must be finite")
    if (candidate_vals < 0).any():
        raise ValueError("candidate_vals must be non-negative")
    if (candidate_ids < 0).any():
        raise ValueError("candidate_ids must be non-negative")
    max_candidate_id = metadata.num_components * metadata.d_sae
    if candidate_ids.numel() and int(candidate_ids.max().item()) >= max_candidate_id:
        raise ValueError("candidate_ids out of range")

    total_tokens_processed = payload.get("total_tokens_processed", metadata.token_count)
    if int(total_tokens_processed) != metadata.token_count:
        raise ValueError("token-count metadata mismatch")
    return metadata, payload


def expand_candidate_dump_to_contributions(
    candidate_metadata: CandidateDumpMetadata,
    candidate_payload: Dict[str, Any],
    seq_offsets: torch.Tensor,
    seq_targets_global: torch.Tensor,
) -> tuple[CandidatePreAggregationMetadata, Dict[str, Any]]:
    """
    Expand simple dump rows into raw contribution records for future MapReduce.

    This preserves native reducer semantics: duplicate target entries contribute
    duplicate rows, self-candidates are skipped, and only positive candidate
    values contribute.
    """

    validate_candidate_dump_partial(
        {
            "metadata": candidate_metadata.model_dump(mode="json"),
            "payload": candidate_payload,
        }
    )
    if seq_offsets.dtype != torch.int64 or seq_targets_global.dtype != torch.int64:
        raise ValueError("seq_offsets and seq_targets_global must be int64")
    if seq_offsets.ndim != 1 or seq_targets_global.ndim != 1:
        raise ValueError("seq_offsets and seq_targets_global must be 1D tensors")
    if seq_offsets.numel() == 0:
        raise ValueError("seq_offsets must include at least sequence ID 0")

    sequence_ids = candidate_payload["sequence_ids"].to(torch.int64).cpu()
    candidate_ids = candidate_payload["candidate_ids"].to(torch.int32).cpu()
    candidate_vals = candidate_payload["candidate_vals"].to(torch.float32).cpu()
    offsets = seq_offsets.to(torch.int64).cpu()
    targets = seq_targets_global.to(torch.int64).cpu()
    max_target_id = candidate_metadata.num_components * candidate_metadata.d_sae

    target_records: list[int] = []
    candidate_records: list[int] = []
    value_records: list[float] = []
    sequence_records: list[int] = []

    for row_idx, sequence_id_tensor in enumerate(sequence_ids):
        sequence_id = int(sequence_id_tensor.item())
        if sequence_id <= 0 or sequence_id >= offsets.numel():
            raise ValueError(f"sequence ID missing from CSR offsets: {sequence_id}")
        start = int(offsets[sequence_id - 1].item())
        end = int(offsets[sequence_id].item())
        if start < 0 or end < start or end > targets.numel():
            raise ValueError(f"invalid CSR range for sequence ID: {sequence_id}")

        row_candidate_ids = candidate_ids[row_idx]
        row_candidate_vals = candidate_vals[row_idx]
        for target_id in targets[start:end].tolist():
            target_id = int(target_id)
            if target_id < 0 or target_id >= max_target_id:
                continue
            for candidate_id, candidate_value in zip(
                row_candidate_ids.tolist(),
                row_candidate_vals.tolist(),
            ):
                candidate_id = int(candidate_id)
                candidate_value = float(candidate_value)
                if candidate_id == target_id or candidate_value <= 0.0:
                    continue
                target_records.append(target_id)
                candidate_records.append(candidate_id)
                value_records.append(candidate_value)
                sequence_records.append(sequence_id)

    payload = {
        "target_ids": torch.tensor(target_records, dtype=torch.int64),
        "candidate_ids": torch.tensor(candidate_records, dtype=torch.int32),
        "values": torch.tensor(value_records, dtype=torch.float32),
        "sequence_ids": torch.tensor(sequence_records, dtype=torch.int64),
    }
    metadata = build_candidate_preaggregation_metadata(
        candidate_metadata,
        contribution_count=len(target_records),
    )
    validate_candidate_preaggregation_partial(
        {"metadata": metadata.model_dump(mode="json"), "payload": payload}
    )
    return metadata, payload


def save_candidate_preaggregation_partial(
    path: str | Path,
    metadata: CandidatePreAggregationMetadata,
    payload: Dict[str, Any],
) -> None:
    data = {"metadata": metadata.model_dump(mode="json"), "payload": payload}
    validate_candidate_preaggregation_partial(data)
    _atomic_torch_save(data, path)


def load_candidate_preaggregation_partial(
    path: str | Path,
    *,
    expected_config_hash: Optional[str] = None,
) -> tuple[CandidatePreAggregationMetadata, Dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return validate_candidate_preaggregation_partial(
        data,
        expected_config_hash=expected_config_hash,
    )


def validate_candidate_preaggregation_partial(
    data: Dict[str, Any],
    *,
    expected_config_hash: Optional[str] = None,
) -> tuple[CandidatePreAggregationMetadata, Dict[str, Any]]:
    if not isinstance(data, dict) or "metadata" not in data or "payload" not in data:
        raise ValueError("candidate preaggregation partial must contain metadata and payload")
    metadata = CandidatePreAggregationMetadata.model_validate(data["metadata"])
    payload = data["payload"]
    if not isinstance(payload, dict):
        raise ValueError("candidate preaggregation payload must be a dict")
    if expected_config_hash is not None and metadata.config_hash != expected_config_hash:
        raise ValueError("candidate preaggregation config hash mismatch")

    target_ids = _required_tensor(payload, "target_ids")
    candidate_ids = _required_tensor(payload, "candidate_ids")
    values = _required_tensor(payload, "values")
    sequence_ids = _required_tensor(payload, "sequence_ids")
    if target_ids.dtype != torch.int64:
        raise ValueError("target_ids must be int64")
    if candidate_ids.dtype != torch.int32:
        raise ValueError("candidate_ids must be int32")
    if values.dtype != torch.float32:
        raise ValueError("values must be float32")
    if sequence_ids.dtype != torch.int64:
        raise ValueError("sequence_ids must be int64")
    expected_shape = (metadata.contribution_count,)
    if (
        tuple(target_ids.shape) != expected_shape
        or tuple(candidate_ids.shape) != expected_shape
        or tuple(values.shape) != expected_shape
        or tuple(sequence_ids.shape) != expected_shape
    ):
        raise ValueError("preaggregation tensors have unexpected shape")
    if not torch.isfinite(values).all():
        raise ValueError("preaggregation values must be finite")
    if (values < 0).any():
        raise ValueError("preaggregation values must be non-negative")
    max_target_id = metadata.num_components * metadata.d_sae
    if target_ids.numel() and int(target_ids.max().item()) >= max_target_id:
        raise ValueError("target_ids out of range")
    if candidate_ids.numel() and int(candidate_ids.max().item()) >= max_target_id:
        raise ValueError("candidate_ids out of range")
    if (target_ids < 0).any() or (candidate_ids < 0).any():
        raise ValueError("preaggregation IDs must be non-negative")
    if target_ids.numel():
        if int(target_ids.min().item()) < metadata.target_start_id:
            raise ValueError("target_ids outside reducer target range")
        if int(target_ids.max().item()) >= int(metadata.target_end_id):
            raise ValueError("target_ids outside reducer target range")
    if (target_ids.to(torch.int32) == candidate_ids).any():
        raise ValueError("preaggregation must not contain self-candidate records")
    return metadata, payload


def _required_tensor(payload: Dict[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"candidate dump payload missing tensor: {key}")
    return value


def _device_for_worker(manifest: DistributedRunManifest, worker_id: int):
    for assignment in manifest.devices:
        if assignment.worker_id == worker_id:
            return assignment
    return None


def _atomic_torch_save(data: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    torch.save(data, tmp_path)
    os.replace(tmp_path, output_path)
