"""Sequence-representation merge helpers for distributed pass 1."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch

from ..pass1_partials import load_pass1_partial, validate_pass1_partial
from ..seq_repr_mapping import validate_seq_repr_mapping
from .contracts import SeqReprPartial


def load_and_merge_seq_repr_partials(
    partial_paths: Sequence[str | Path],
    *,
    expected_config_hash: str | None = None,
    seq_repr_mapping: dict[str, object] | None = None,
) -> Dict[str, object]:
    """Load seq-repr partial files and merge them by global sequence ID."""

    partials = [
        load_pass1_partial(
            path,
            expected_artifact_name="seq_repr",
            expected_config_hash=expected_config_hash,
        )
        for path in partial_paths
    ]
    return merge_seq_repr_partials(partials, seq_repr_mapping=seq_repr_mapping)


def merge_seq_repr_partials(
    partials: Sequence[SeqReprPartial],
    *,
    seq_repr_mapping: dict[str, object] | None = None,
) -> Dict[str, object]:
    """Merge capped or uncapped seq_repr partials into one global store payload."""

    if not partials:
        raise ValueError("at least one seq_repr partial is required")
    _validate_seq_repr_partial_set(partials)
    target_mapping = seq_repr_mapping or _mapping_from_seq_repr_payload(partials[0][1])
    validate_seq_repr_mapping(target_mapping)

    n_seqs = int(target_mapping["n_seqs"])
    n_stored = int(target_mapping["n_stored"])
    slot_to_id = target_mapping["slot_to_id"]
    id_to_slot = target_mapping["id_to_slot"]
    is_capped = bool(target_mapping["is_capped"])
    repr_dim = int(partials[0][1]["repr_dim"])
    repr_mode = str(partials[0][1]["repr_mode"])
    repr_buf = torch.zeros((n_stored + 1, repr_dim), dtype=partials[0][1]["repr_buf"].dtype)
    written_slots = torch.zeros(n_stored + 1, dtype=torch.bool)

    for metadata, payload in partials:
        source_buf = payload["repr_buf"]
        source_id_to_slot = payload.get("id_to_slot")
        if source_id_to_slot is not None:
            source_id_to_slot = source_id_to_slot.to(torch.int64)
        for sequence_id in range(metadata.sequence_id_min or 1, (metadata.sequence_id_max or 0) + 1):
            if sequence_id < 1 or sequence_id > n_seqs:
                raise ValueError("seq_repr sequence ID out of global range")
            target_slot = int(id_to_slot[sequence_id].item())
            if target_slot == 0:
                continue
            source_slot = (
                int(source_id_to_slot[sequence_id].item())
                if source_id_to_slot is not None
                else sequence_id
            )
            if source_slot == 0:
                continue
            if source_slot >= source_buf.shape[0]:
                raise ValueError("seq_repr source slot out of range")
            if written_slots[target_slot]:
                raise ValueError(f"seq_repr slot written more than once: {target_slot}")
            row = source_buf[source_slot]
            if not torch.isfinite(row.float()).all():
                raise ValueError("seq_repr row contains non-finite values")
            repr_buf[target_slot] = row.to(repr_buf.dtype)
            written_slots[target_slot] = True

    merged = {
        "repr_buf": repr_buf,
        "repr_mode": repr_mode,
        "repr_dim": repr_dim,
        "n_seqs": n_seqs,
        "n_stored": n_stored,
        "is_capped": is_capped,
        "merge_report": {
            "selected_slots": int(n_stored),
            "written_slots": int(written_slots[1:].sum().item()),
            "missing_slots": int((~written_slots[1:]).sum().item()),
            "sampling_seed": target_mapping.get("sampling_seed"),
            "derived_seed": target_mapping.get("derived_seed"),
            "dataset_fingerprint": target_mapping.get("dataset_fingerprint"),
        },
    }
    if is_capped:
        merged["slot_to_id"] = slot_to_id.to(torch.int64)
        merged["id_to_slot"] = id_to_slot.to(torch.int32)
    _validate_merged_seq_repr(merged)
    return merged


def _validate_seq_repr_partial_set(
    partials: Sequence[SeqReprPartial],
) -> None:
    seen_workers: set[int] = set()
    first_metadata = partials[0][0]
    first_payload = partials[0][1]
    for metadata, payload in partials:
        if metadata.artifact_name != "seq_repr":
            raise ValueError("all partials must be seq_repr artifacts")
        if metadata.worker_id in seen_workers:
            raise ValueError(f"duplicate seq_repr partial for worker {metadata.worker_id}")
        seen_workers.add(metadata.worker_id)
        if metadata.run_id != first_metadata.run_id:
            raise ValueError("seq_repr partial run_id mismatch")
        if metadata.config_hash != first_metadata.config_hash:
            raise ValueError("seq_repr partial config hash mismatch")
        for key in ["repr_mode", "repr_dim", "n_seqs"]:
            if payload[key] != first_payload[key]:
                raise ValueError(f"seq_repr partial {key} mismatch")
        validate_pass1_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_artifact_name="seq_repr",
            expected_config_hash=first_metadata.config_hash,
        )
        _validate_seq_repr_mapping_compatibility(first_payload, payload)


def _mapping_from_seq_repr_payload(payload: Dict[str, object]) -> dict[str, object]:
    n_seqs = int(payload["n_seqs"])
    n_stored = int(payload["n_stored"])
    if bool(payload["is_capped"]):
        return {
            "slot_to_id": payload["slot_to_id"],
            "id_to_slot": payload["id_to_slot"],
            "n_seqs": n_seqs,
            "n_stored": n_stored,
            "is_capped": True,
        }
    return {
        "slot_to_id": torch.arange(n_seqs + 1, dtype=torch.int64),
        "id_to_slot": torch.arange(n_seqs + 1, dtype=torch.int32),
        "n_seqs": n_seqs,
        "n_stored": n_stored,
        "is_capped": False,
    }


def _validate_seq_repr_mapping_compatibility(
    first_payload: Dict[str, object],
    payload: Dict[str, object],
) -> None:
    if bool(first_payload["is_capped"]) != bool(payload["is_capped"]):
        raise ValueError("seq_repr partial cap mode mismatch")
    if bool(first_payload["is_capped"]):
        if not torch.equal(first_payload["slot_to_id"], payload["slot_to_id"]):
            raise ValueError("seq_repr partial slot_to_id mismatch")
        if not torch.equal(first_payload["id_to_slot"], payload["id_to_slot"]):
            raise ValueError("seq_repr partial id_to_slot mismatch")


def _validate_merged_seq_repr(merged: Dict[str, object]) -> None:
    repr_buf = merged["repr_buf"]
    if not isinstance(repr_buf, torch.Tensor) or repr_buf.ndim != 2:
        raise ValueError("merged seq_repr repr_buf must be 2D")
    if not torch.isfinite(repr_buf.float()).all():
        raise ValueError("merged seq_repr repr_buf contains non-finite values")
    expected_shape = (int(merged["n_stored"]) + 1, int(merged["repr_dim"]))
    if tuple(repr_buf.shape) != expected_shape:
        raise ValueError("merged seq_repr repr_buf shape mismatch")
    if bool(merged["is_capped"]):
        validate_seq_repr_mapping(
            {
                "slot_to_id": merged["slot_to_id"],
                "id_to_slot": merged["id_to_slot"],
                "n_seqs": merged["n_seqs"],
                "n_stored": merged["n_stored"],
                "is_capped": True,
            }
        )


__all__ = [
    "load_and_merge_seq_repr_partials",
    "merge_seq_repr_partials",
]
