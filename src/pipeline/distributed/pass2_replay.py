"""Replay-list helpers for distributed pass-2 candidate dumps."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Mapping, Protocol, Sequence

import torch

from .assignments import partition_sequence_ids
from .manifest import DistributedRunManifest, WorkAssignments
from .shard_table import ShardRecord, contains_sequence_id


class TopContextLike(Protocol):
    def get_all_sequence_ids(self) -> list[int]:
        ...


@dataclass(frozen=True)
class Pass2ReplayList:
    sequence_ids: list[int]
    sequence_count: int
    sequence_hash: str


@dataclass(frozen=True)
class Pass2WorkerInput:
    worker_id: int
    sequence_ids: list[int]
    sequence_count: int
    sequence_id_min: int | None
    sequence_id_max: int | None
    replay_sequence_hash: str | None


def build_pass2_replay_list(
    top_ctx: TopContextLike | Mapping[str, object],
    *,
    shard_table: Sequence[ShardRecord] | None = None,
) -> Pass2ReplayList:
    """Build the deterministic global replay sequence list from merged top_ctx."""

    raw_sequence_ids = _extract_top_ctx_sequence_ids(top_ctx)
    sequence_ids = sorted({int(sequence_id) for sequence_id in raw_sequence_ids if int(sequence_id) != 0})
    if shard_table is not None:
        for sequence_id in sequence_ids:
            if not contains_sequence_id(shard_table, sequence_id):
                raise ValueError(f"replay sequence ID out of range: {sequence_id}")
    sequence_hash = hash_replay_sequence_ids(sequence_ids)
    return Pass2ReplayList(
        sequence_ids=sequence_ids,
        sequence_count=len(sequence_ids),
        sequence_hash=sequence_hash,
    )


def assign_pass2_replay_sequences(
    manifest: DistributedRunManifest,
    top_ctx: TopContextLike | Mapping[str, object],
) -> DistributedRunManifest:
    """
    Return a manifest copy with pass-2 replay assignments and hash/count metadata.

    The replay list is partitioned into contiguous chunks in global replay-list order.
    """

    replay = build_pass2_replay_list(top_ctx, shard_table=manifest.shard_table)
    pass2_assignments = partition_sequence_ids(
        replay.sequence_ids,
        manifest.worker_count,
        shard_table=manifest.shard_table,
    )
    work_assignment_data = manifest.work_assignments.model_dump()
    work_assignment_data.update(
        {
            "pass2_sequence_ids": pass2_assignments,
            "pass2_replay_sequence_count": replay.sequence_count,
            "pass2_replay_sequence_hash": replay.sequence_hash,
        }
    )
    work_assignments = WorkAssignments.model_validate(work_assignment_data)
    manifest_data = manifest.model_dump()
    manifest_data["work_assignments"] = work_assignments.model_dump()
    return DistributedRunManifest.model_validate(manifest_data)


def get_pass2_worker_input(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> Pass2WorkerInput:
    """Return validated pass-2 input metadata for one worker."""

    if worker_id < 0 or worker_id >= manifest.worker_count:
        raise ValueError("worker_id out of range")
    if not _trust_pass2_replay_assignments():
        validate_pass2_replay_assignments(manifest)
    sequence_ids = list(manifest.work_assignments.pass2_sequence_ids.get(str(worker_id), []))
    if sequence_ids and _trust_pass2_replay_assignments():
        sequence_id_min = sequence_ids[0]
        sequence_id_max = sequence_ids[-1]
    else:
        sequence_id_min = min(sequence_ids) if sequence_ids else None
        sequence_id_max = max(sequence_ids) if sequence_ids else None
    return Pass2WorkerInput(
        worker_id=worker_id,
        sequence_ids=sequence_ids,
        sequence_count=len(sequence_ids),
        sequence_id_min=sequence_id_min,
        sequence_id_max=sequence_id_max,
        replay_sequence_hash=manifest.work_assignments.pass2_replay_sequence_hash,
    )


def validate_pass2_replay_assignments(manifest: DistributedRunManifest) -> None:
    """Validate pass-2 assignments as contiguous chunks of one replay list."""

    if _trust_pass2_replay_assignments():
        return

    assignments = manifest.work_assignments.pass2_sequence_ids
    if set(assignments) - {str(worker_id) for worker_id in range(manifest.worker_count)}:
        raise ValueError("pass2 assignment worker key out of range")

    replay_sequence_ids: list[int] = []
    for worker_id in range(manifest.worker_count):
        sequence_ids = list(assignments.get(str(worker_id), []))
        replay_sequence_ids.extend(sequence_ids)

    if replay_sequence_ids != sorted(replay_sequence_ids):
        raise ValueError("pass2 replay assignments must preserve sorted replay-list order")
    if len(replay_sequence_ids) != len(set(replay_sequence_ids)):
        raise ValueError("pass2 replay assignments contain duplicate sequence IDs")
    for sequence_id in replay_sequence_ids:
        if not contains_sequence_id(manifest.shard_table, sequence_id):
            raise ValueError(f"pass2 replay sequence ID out of range: {sequence_id}")

    expected_count = manifest.work_assignments.pass2_replay_sequence_count
    if expected_count is not None and expected_count != len(replay_sequence_ids):
        raise ValueError("pass2 replay sequence count does not match assigned sequences")

    expected_hash = manifest.work_assignments.pass2_replay_sequence_hash
    actual_hash = hash_replay_sequence_ids(replay_sequence_ids)
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("pass2 replay sequence hash does not match assigned sequences")

    expected_assignments = partition_sequence_ids(
        replay_sequence_ids,
        manifest.worker_count,
        shard_table=manifest.shard_table,
    )
    if assignments != expected_assignments:
        raise ValueError("pass2 replay assignments must be contiguous worker chunks")


def hash_replay_sequence_ids(sequence_ids: Sequence[int]) -> str:
    """Hash a replay sequence list using a stable JSON representation."""

    payload = json.dumps(
        [int(sequence_id) for sequence_id in sequence_ids],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_top_ctx_sequence_ids(top_ctx: TopContextLike | Mapping[str, object]) -> list[int]:
    if hasattr(top_ctx, "get_all_sequence_ids"):
        return [int(sequence_id) for sequence_id in top_ctx.get_all_sequence_ids()]
    if "ctx_seq_idx" not in top_ctx:
        raise ValueError("top_ctx payload must contain ctx_seq_idx")
    ctx_seq_idx = top_ctx["ctx_seq_idx"]
    if not isinstance(ctx_seq_idx, torch.Tensor):
        raise TypeError("top_ctx ctx_seq_idx must be a torch.Tensor")
    return [int(sequence_id) for sequence_id in torch.unique(ctx_seq_idx).cpu().tolist()]


def _trust_pass2_replay_assignments() -> bool:
    return os.environ.get("TURING_TRUST_PASS2_REPLAY_ASSIGNMENTS") == "1"
