"""Versioned partial artifact schemas for distributed pass 1."""

from __future__ import annotations

import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .manifest import DistributedRunManifest
from .shard_table import sequence_ids_for_shards


PASS1_PARTIAL_SCHEMA_VERSION = 1
MID_CTX_PRIORITY_HASH_VERSION = "sha256-v1"

Pass1ArtifactName = Literal[
    "latent_stats",
    "top_ctx",
    "mid_ctx_candidates",
    "seq_repr",
    "logit_ctx",
]


class Pass1PartialMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    partial_schema_version: int = PASS1_PARTIAL_SCHEMA_VERSION
    artifact_name: Pass1ArtifactName
    run_id: str
    worker_id: int
    shard_ids: list[int]
    sequence_id_min: Optional[int] = None
    sequence_id_max: Optional[int] = None
    sequence_count: int
    config_hash: str
    physical_id: Optional[int] = None
    logical_id: str
    created_at: str
    component_count: int
    d_sae: int
    store_mode: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("partial_schema_version")
    @classmethod
    def schema_is_supported(cls, value: int) -> int:
        if value != PASS1_PARTIAL_SCHEMA_VERSION:
            raise ValueError("unsupported pass1 partial schema version")
        return value

    @field_validator("worker_id", "sequence_count", "component_count", "d_sae")
    @classmethod
    def counts_are_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("pass1 partial counts must be >= 0")
        return value

    @model_validator(mode="after")
    def sequence_bounds_match_count(self) -> "Pass1PartialMetadata":
        if self.sequence_count == 0:
            if self.sequence_id_min is not None or self.sequence_id_max is not None:
                raise ValueError("empty partials must not declare sequence ID bounds")
        elif self.sequence_id_min is None or self.sequence_id_max is None:
            raise ValueError("non-empty partials must declare sequence ID bounds")
        elif self.sequence_id_min > self.sequence_id_max:
            raise ValueError("sequence_id_min must be <= sequence_id_max")
        return self


def build_pass1_partial_metadata(
    manifest: DistributedRunManifest,
    worker_id: int,
    artifact_name: Pass1ArtifactName,
    *,
    component_count: int,
    d_sae: int,
    store_mode: Optional[Dict[str, Any]] = None,
) -> Pass1PartialMetadata:
    shard_ids = list(manifest.work_assignments.pass1_shards.get(str(worker_id), []))
    sequence_ids = sequence_ids_for_shards(manifest.shard_table, shard_ids)
    device = _device_for_worker(manifest, worker_id)
    return Pass1PartialMetadata(
        artifact_name=artifact_name,
        run_id=manifest.run_id,
        worker_id=worker_id,
        shard_ids=shard_ids,
        sequence_id_min=min(sequence_ids) if sequence_ids else None,
        sequence_id_max=max(sequence_ids) if sequence_ids else None,
        sequence_count=len(sequence_ids),
        config_hash=manifest.normalized_config_hash,
        physical_id=device.physical_id if device is not None else None,
        logical_id=device.logical_id if device is not None else "cpu",
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        component_count=component_count,
        d_sae=d_sae,
        store_mode=store_mode or {},
    )


def save_pass1_partial(
    path: str | Path,
    metadata: Pass1PartialMetadata,
    payload: Dict[str, Any],
) -> None:
    data = {"metadata": metadata.model_dump(mode="json"), "payload": payload}
    validate_pass1_partial(data, expected_artifact_name=metadata.artifact_name)
    _atomic_torch_save(data, path)


def load_pass1_partial(
    path: str | Path,
    *,
    expected_artifact_name: Optional[Pass1ArtifactName] = None,
    expected_config_hash: Optional[str] = None,
) -> tuple[Pass1PartialMetadata, Dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return validate_pass1_partial(
        data,
        expected_artifact_name=expected_artifact_name,
        expected_config_hash=expected_config_hash,
    )


def validate_pass1_partial(
    data: Dict[str, Any],
    *,
    expected_artifact_name: Optional[Pass1ArtifactName] = None,
    expected_config_hash: Optional[str] = None,
) -> tuple[Pass1PartialMetadata, Dict[str, Any]]:
    if not isinstance(data, dict) or "metadata" not in data or "payload" not in data:
        raise ValueError("pass1 partial must contain metadata and payload")
    metadata = Pass1PartialMetadata.model_validate(data["metadata"])
    payload = data["payload"]
    if not isinstance(payload, dict):
        raise ValueError("pass1 partial payload must be a dict")
    if expected_artifact_name is not None and metadata.artifact_name != expected_artifact_name:
        raise ValueError("pass1 partial artifact name mismatch")
    if expected_config_hash is not None and metadata.config_hash != expected_config_hash:
        raise ValueError("pass1 partial config hash mismatch")

    if metadata.artifact_name == "latent_stats":
        _validate_latent_stats_payload(metadata, payload)
    elif metadata.artifact_name == "top_ctx":
        _validate_context_payload(metadata, payload, expected_ctx_type="top")
    elif metadata.artifact_name == "mid_ctx_candidates":
        _validate_mid_ctx_candidates_payload(metadata, payload)
    elif metadata.artifact_name == "seq_repr":
        _validate_seq_repr_payload(payload)
    elif metadata.artifact_name == "logit_ctx":
        _validate_logit_ctx_payload(metadata, payload)
    else:
        raise ValueError(f"unsupported pass1 partial artifact: {metadata.artifact_name}")
    return metadata, payload


def latent_stats_payload(store) -> Dict[str, Any]:
    return {
        "active_count": store.active_count.cpu(),
        "mean": store.mean.cpu(),
        "mean_abs": store.mean_abs.cpu(),
        "m2": store.m2.cpu(),
        "m2_abs": store.m2_abs.cpu(),
        "seq_count": store.seq_count.cpu(),
        "mean_seq": store.mean_seq.cpu(),
        "m2_seq": store.m2_seq.cpu(),
        "component_steps": dict(store.component_steps),
    }


def top_ctx_payload(store) -> Dict[str, Any]:
    return {
        "ctx_seq_idx": store.ctx_seq_idx.cpu(),
        "ctx_seq_val": store.ctx_seq_val.cpu(),
        "ctx_type": store.ctx_type,
    }


def mid_ctx_candidates_payload(
    store,
    *,
    sampling_seed: int = 0,
    dataset_fingerprint: str = "",
    artifact_name: str = "mid_ctx",
) -> Dict[str, Any]:
    ctx_seq_idx = store.ctx_seq_idx.cpu()
    ctx_seq_val = store.ctx_seq_val.cpu().float()
    valid = (ctx_seq_idx != 0) & torch.isfinite(ctx_seq_val)
    component_ids, latent_ids, _slot_ids = torch.nonzero(valid, as_tuple=True)
    sequence_ids = ctx_seq_idx[valid].to(torch.int32)
    activation_values = ctx_seq_val[valid].to(torch.float32)
    candidate_pool_settings = {
        "mode": "widened_worker_candidate_pool"
        if bool(getattr(store, "_distributed_candidate_pool", False))
        else "worker_local_mid_ctx_checkpoint",
        "source_mid_mode": store.mid_mode,
        "candidate_band_low_sigma": float(store._band_low),
        "candidate_band_high_sigma": float(store._band_high),
        "band_low_sigma": float(getattr(store, "_final_band_low", store._band_low)),
        "band_high_sigma": float(getattr(store, "_final_band_high", store._band_high)),
        "band_margin_sigma": float(getattr(store, "_candidate_band_margin", 0.0)),
        "num_ctx_sequences": int(getattr(store, "_final_num_ctx_sequences", store.num_ctx_sequences)),
        "max_candidates_per_latent": int(store.num_ctx_sequences),
        "sampling_seed": int(sampling_seed),
        "dataset_fingerprint": str(dataset_fingerprint),
        "priority_hash_version": MID_CTX_PRIORITY_HASH_VERSION,
        "priority_artifact_name": str(artifact_name),
    }
    priorities = _candidate_priorities(
        component_ids,
        latent_ids,
        sequence_ids,
        sampling_seed=sampling_seed,
        dataset_fingerprint=dataset_fingerprint,
        candidate_pool_settings=candidate_pool_settings,
        artifact_name=artifact_name,
    )
    return {
        "component_ids": component_ids.to(torch.int16),
        "latent_ids": latent_ids.to(torch.int32),
        "sequence_ids": sequence_ids,
        "activation_values": activation_values,
        "priorities": priorities,
        "candidate_pool_settings": candidate_pool_settings,
        "truncation_counters": torch.zeros(
            (store.num_components, store.d_sae), dtype=torch.int64
        ),
        "ctx_seq_idx": ctx_seq_idx,
        "ctx_seq_val": ctx_seq_val,
        "reservoir_fill": store.reservoir_fill.cpu(),
        "reservoir_n": store.reservoir_n.cpu(),
    }


def seq_repr_payload(seq_repr) -> Dict[str, Any]:
    payload = {
        "repr_buf": seq_repr.repr_buf.cpu(),
        "repr_mode": seq_repr.repr_mode,
        "repr_dim": seq_repr.repr_dim,
        "n_seqs": seq_repr.n_seqs,
        "n_stored": seq_repr.n_stored,
        "is_capped": seq_repr.is_capped,
    }
    if seq_repr.is_capped:
        payload["slot_to_id"] = seq_repr.slot_to_id.cpu()
        payload["id_to_slot"] = seq_repr.id_to_slot.cpu()
    return payload


def logit_ctx_payload(store) -> Dict[str, Any]:
    return {
        "latent_counts": store.latent_counts.cpu(),
        "top_tokens": store.top_tokens.cpu(),
        "top_probs": store.top_probs.cpu(),
    }


def _validate_latent_stats_payload(
    metadata: Pass1PartialMetadata,
    payload: Dict[str, Any],
) -> None:
    shape = (metadata.component_count, metadata.d_sae)
    for name in ["active_count", "seq_count"]:
        _require_tensor(payload, name, dtype=torch.int64, shape=shape)
    for name in ["mean", "mean_abs", "m2", "m2_abs", "mean_seq", "m2_seq"]:
        tensor = _require_tensor(payload, name, shape=shape)
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        _require_finite(tensor, name)
    if not isinstance(payload.get("component_steps"), dict):
        raise ValueError("component_steps must be a dict")


def _validate_context_payload(
    metadata: Pass1PartialMetadata,
    payload: Dict[str, Any],
    *,
    expected_ctx_type: str,
) -> None:
    idx = _require_tensor(payload, "ctx_seq_idx", dtype=torch.int32)
    vals = _require_tensor(payload, "ctx_seq_val", shape=tuple(idx.shape))
    if idx.ndim != 3 or idx.shape[0] != metadata.component_count or idx.shape[1] != metadata.d_sae:
        raise ValueError("context tensors must have shape [component_count, d_sae, k]")
    if payload.get("ctx_type") != expected_ctx_type:
        raise ValueError("context partial ctx_type mismatch")
    if not vals.is_floating_point():
        raise ValueError("ctx_seq_val must be floating point")
    _require_finite(vals, "ctx_seq_val")


def _validate_mid_ctx_candidates_payload(
    metadata: Pass1PartialMetadata,
    payload: Dict[str, Any],
) -> None:
    required = [
        "component_ids",
        "latent_ids",
        "sequence_ids",
        "activation_values",
        "priorities",
    ]
    tensors = [_require_tensor(payload, name) for name in required]
    lengths = {int(tensor.numel()) for tensor in tensors}
    if len(lengths) != 1:
        raise ValueError("mid_ctx candidate tensors must have the same length")
    component_ids, latent_ids, sequence_ids, activation_values, priorities = tensors
    if component_ids.numel() > 0:
        if int(component_ids.min()) < 0 or int(component_ids.max()) >= metadata.component_count:
            raise ValueError("mid_ctx component_ids out of range")
        if int(latent_ids.min()) < 0 or int(latent_ids.max()) >= metadata.d_sae:
            raise ValueError("mid_ctx latent_ids out of range")
        if metadata.sequence_id_min is not None and int(sequence_ids.min()) < metadata.sequence_id_min:
            raise ValueError("mid_ctx sequence_ids below worker range")
        if metadata.sequence_id_max is not None and int(sequence_ids.max()) > metadata.sequence_id_max:
            raise ValueError("mid_ctx sequence_ids above worker range")
    _require_finite(activation_values, "activation_values")
    _require_finite(priorities, "priorities")
    if not isinstance(payload.get("candidate_pool_settings"), dict):
        raise ValueError("candidate_pool_settings must be a dict")
    _require_tensor(
        payload,
        "truncation_counters",
        dtype=torch.int64,
        shape=(metadata.component_count, metadata.d_sae),
    )
    _validate_context_payload(metadata, {**payload, "ctx_type": "mid"}, expected_ctx_type="mid")


def _validate_seq_repr_payload(payload: Dict[str, Any]) -> None:
    repr_buf = _require_tensor(payload, "repr_buf")
    if repr_buf.ndim != 2 or not repr_buf.is_floating_point():
        raise ValueError("repr_buf must be a 2D floating point tensor")
    _require_finite(repr_buf, "repr_buf")
    for key in ["repr_mode", "repr_dim", "n_seqs", "n_stored", "is_capped"]:
        if key not in payload:
            raise ValueError(f"seq_repr payload missing {key}")
    if bool(payload["is_capped"]):
        _require_tensor(payload, "slot_to_id", dtype=torch.int64)
        _require_tensor(payload, "id_to_slot", dtype=torch.int32)


def _validate_logit_ctx_payload(
    metadata: Pass1PartialMetadata,
    payload: Dict[str, Any],
) -> None:
    counts = _require_tensor(
        payload,
        "latent_counts",
        dtype=torch.int64,
        shape=(metadata.component_count, metadata.d_sae),
    )
    tokens = _require_tensor(payload, "top_tokens", dtype=torch.int32)
    probs = _require_tensor(payload, "top_probs", shape=tuple(tokens.shape))
    if tokens.ndim != 3 or tokens.shape[:2] != counts.shape:
        raise ValueError("logit_ctx top tensors must have shape [component_count, d_sae, k]")
    if tokens.numel() > 0 and int(tokens.min()) < 0:
        raise ValueError("logit_ctx top_tokens must be non-negative")
    if not probs.is_floating_point():
        raise ValueError("top_probs must be floating point")
    _require_finite(probs, "top_probs")


def _require_tensor(
    payload: Dict[str, Any],
    name: str,
    *,
    dtype: Optional[torch.dtype] = None,
    shape: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} has invalid shape")
    return value


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor.float()).all():
        raise ValueError(f"{name} contains non-finite values")


def _candidate_priorities(
    component_ids: torch.Tensor,
    latent_ids: torch.Tensor,
    sequence_ids: torch.Tensor,
    *,
    sampling_seed: int = 0,
    dataset_fingerprint: str = "",
    candidate_pool_settings: Optional[Dict[str, Any]] = None,
    artifact_name: str = "mid_ctx",
) -> torch.Tensor:
    if component_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    settings = candidate_pool_settings or {}
    priorities = []
    for component_id, latent_id, sequence_id in zip(
        component_ids.tolist(),
        latent_ids.tolist(),
        sequence_ids.tolist(),
    ):
        material = "|".join(
            [
                MID_CTX_PRIORITY_HASH_VERSION,
                str(int(sampling_seed)),
                str(artifact_name),
                str(dataset_fingerprint),
                str(settings.get("band_low_sigma", "")),
                str(settings.get("band_high_sigma", "")),
                str(settings.get("candidate_band_low_sigma", "")),
                str(settings.get("candidate_band_high_sigma", "")),
                str(settings.get("band_margin_sigma", "")),
                str(settings.get("num_ctx_sequences", "")),
                str(component_id),
                str(latent_id),
                str(sequence_id),
            ]
        )
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(1 << 64)
        priorities.append(value)
    return torch.tensor(priorities, dtype=torch.float32)


def _atomic_torch_save(data: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(data, tmp_path)
    os.replace(tmp_path, output_path)


def _device_for_worker(manifest: DistributedRunManifest, worker_id: int):
    for device in manifest.devices:
        if device.worker_id == worker_id:
            return device
    return None
