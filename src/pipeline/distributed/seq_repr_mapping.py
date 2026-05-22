"""Deterministic global slot mappings for distributed ``seq_repr`` stores."""

from __future__ import annotations

import hashlib
from typing import Optional

import torch

from .shard_table import ShardRecord


def shard_table_fingerprint(shard_table: list[ShardRecord]) -> str:
    """Return a stable fingerprint for sequence membership and shard ordering."""

    encoded = "|".join(
        f"{record.shard_index}:{record.shard_filename}:{record.sequence_count}:"
        f"{record.global_start_id}:{record.global_end_id}"
        for record in shard_table
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def derive_seq_repr_cap_seed(
    *,
    sampling_seed: int,
    dataset_fingerprint: str,
    cap_size: int,
    total_sequence_count: int,
) -> int:
    """Derive an artifact-specific seed without depending on run_id."""

    material = (
        f"{sampling_seed}|seq_repr|{dataset_fingerprint}|"
        f"{cap_size}|{total_sequence_count}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF


def build_seq_repr_cap_mapping(
    *,
    total_sequence_count: int,
    max_repr_seqs: Optional[int],
    sampling_seed: int,
    dataset_fingerprint: str,
) -> dict[str, torch.Tensor | int | bool | str]:
    """Build deterministic ``slot_to_id`` / ``id_to_slot`` tensors."""

    if total_sequence_count < 0:
        raise ValueError("total_sequence_count must be >= 0")
    if max_repr_seqs is not None and max_repr_seqs < 0:
        raise ValueError("max_repr_seqs must be >= 0 when provided")
    n_stored = (
        min(max_repr_seqs, total_sequence_count)
        if max_repr_seqs is not None
        else total_sequence_count
    )
    is_capped = max_repr_seqs is not None and max_repr_seqs < total_sequence_count

    if is_capped:
        seed = derive_seq_repr_cap_seed(
            sampling_seed=sampling_seed,
            dataset_fingerprint=dataset_fingerprint,
            cap_size=n_stored,
            total_sequence_count=total_sequence_count,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        kept = torch.randperm(total_sequence_count, generator=generator)[:n_stored]
        kept = kept.sort().values + 1
        slot_to_id = torch.zeros(n_stored + 1, dtype=torch.int64)
        slot_to_id[1:] = kept.to(torch.int64)
        id_to_slot = torch.zeros(total_sequence_count + 1, dtype=torch.int32)
        id_to_slot[kept] = torch.arange(1, n_stored + 1, dtype=torch.int32)
    else:
        seed = derive_seq_repr_cap_seed(
            sampling_seed=sampling_seed,
            dataset_fingerprint=dataset_fingerprint,
            cap_size=n_stored,
            total_sequence_count=total_sequence_count,
        )
        slot_to_id = torch.arange(total_sequence_count + 1, dtype=torch.int64)
        id_to_slot = torch.arange(total_sequence_count + 1, dtype=torch.int32)

    return {
        "slot_to_id": slot_to_id,
        "id_to_slot": id_to_slot,
        "n_seqs": total_sequence_count,
        "n_stored": n_stored,
        "is_capped": is_capped,
        "sampling_seed": sampling_seed,
        "derived_seed": seed,
        "dataset_fingerprint": dataset_fingerprint,
    }


def validate_seq_repr_mapping(mapping: dict[str, object]) -> None:
    slot_to_id = mapping.get("slot_to_id")
    id_to_slot = mapping.get("id_to_slot")
    n_seqs = mapping.get("n_seqs")
    n_stored = mapping.get("n_stored")
    if not isinstance(slot_to_id, torch.Tensor) or slot_to_id.dtype != torch.int64:
        raise ValueError("slot_to_id must be an int64 tensor")
    if not isinstance(id_to_slot, torch.Tensor) or id_to_slot.dtype != torch.int32:
        raise ValueError("id_to_slot must be an int32 tensor")
    if not isinstance(n_seqs, int) or not isinstance(n_stored, int):
        raise ValueError("n_seqs and n_stored must be integers")
    if slot_to_id.shape != (n_stored + 1,):
        raise ValueError("slot_to_id shape does not match n_stored")
    if id_to_slot.shape != (n_seqs + 1,):
        raise ValueError("id_to_slot shape does not match n_seqs")
    if int(slot_to_id[0].item()) != 0 or int(id_to_slot[0].item()) != 0:
        raise ValueError("seq_repr mappings must keep slot/id 0 as sentinel")
    selected = slot_to_id[1:]
    if selected.numel() and (
        int(selected.min().item()) < 1 or int(selected.max().item()) > n_seqs
    ):
        raise ValueError("slot_to_id contains sequence IDs out of range")
    if selected.unique().numel() != selected.numel():
        raise ValueError("slot_to_id contains duplicate sequence IDs")
    for slot, sequence_id in enumerate(selected.tolist(), start=1):
        if int(id_to_slot[sequence_id].item()) != slot:
            raise ValueError("id_to_slot does not invert slot_to_id")
