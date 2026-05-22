"""Pure deterministic assignment helpers for distributed pipeline work."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, TypeVar

from .manifest import WorkAssignments
from .shard_table import ShardRecord, contains_sequence_id


T = TypeVar("T")


def partition_pass1_shards(
    shard_table: Sequence[ShardRecord],
    worker_count: int,
    *,
    selected_shards: Iterable[int] | None = None,
) -> tuple[Dict[str, List[int]], Dict[str, int]]:
    """Balance whole shards by actual sequence count using deterministic greedy."""

    _validate_worker_count(worker_count)
    by_index = {record.shard_index: record for record in shard_table}
    selected = (
        list(selected_shards)
        if selected_shards is not None
        else [record.shard_index for record in shard_table]
    )
    _reject_duplicates(selected, "selected shard")
    for shard_id in selected:
        if shard_id not in by_index:
            raise ValueError(f"selected shard index out of range: {shard_id}")

    assignments = {str(worker_id): [] for worker_id in range(worker_count)}
    totals = {str(worker_id): 0 for worker_id in range(worker_count)}
    sorted_records = sorted(
        (by_index[shard_id] for shard_id in selected),
        key=lambda record: (-record.sequence_count, record.shard_index),
    )
    for record in sorted_records:
        worker_key = min(
            totals,
            key=lambda key: (totals[key], int(key)),
        )
        assignments[worker_key].append(record.shard_index)
        totals[worker_key] += record.sequence_count

    for shard_ids in assignments.values():
        shard_ids.sort()
    return assignments, totals


def partition_contiguous(items: Sequence[T], worker_count: int) -> Dict[str, List[T]]:
    """Split a list into contiguous chunks, distributing remainders first."""

    _validate_worker_count(worker_count)
    item_count = len(items)
    base, remainder = divmod(item_count, worker_count)
    assignments: Dict[str, List[T]] = {}
    cursor = 0
    for worker_id in range(worker_count):
        take = base + (1 if worker_id < remainder else 0)
        assignments[str(worker_id)] = list(items[cursor : cursor + take])
        cursor += take
    return assignments


def partition_sequence_ids(
    sequence_ids: Sequence[int],
    worker_count: int,
    *,
    shard_table: Sequence[ShardRecord] | None = None,
) -> Dict[str, List[int]]:
    """Partition global sequence IDs while preserving stable order per worker."""

    _reject_duplicates(sequence_ids, "sequence ID")
    if shard_table is not None:
        for sequence_id in sequence_ids:
            if not contains_sequence_id(shard_table, sequence_id):
                raise ValueError(f"sequence ID out of range: {sequence_id}")
    return partition_contiguous(sequence_ids, worker_count)


def partition_seed_ids(seed_ids: Sequence[int], worker_count: int) -> Dict[str, List[int]]:
    """Partition selected discovery seed/candidate IDs deterministically."""

    _reject_duplicates(seed_ids, "seed ID")
    return partition_contiguous(seed_ids, worker_count)


def build_work_assignments(
    shard_table: Sequence[ShardRecord],
    worker_count: int,
    *,
    pass2_sequence_ids: Sequence[int] | None = None,
    discovery_seed_ids: Sequence[int] | None = None,
    selected_shards: Iterable[int] | None = None,
) -> WorkAssignments:
    """Build a manifest-ready work assignment block from pure helpers."""

    pass1_shards, pass1_sequence_totals = partition_pass1_shards(
        shard_table,
        worker_count,
        selected_shards=selected_shards,
    )
    return WorkAssignments(
        pass1_shards=pass1_shards,
        pass1_sequence_totals=pass1_sequence_totals,
        pass2_sequence_ids=partition_sequence_ids(
            list(pass2_sequence_ids or []),
            worker_count,
            shard_table=shard_table,
        ),
        discovery_seed_ids=partition_seed_ids(
            list(discovery_seed_ids or []),
            worker_count,
        ),
    )


def _validate_worker_count(worker_count: int) -> None:
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")


def _reject_duplicates(values: Iterable[int], label: str) -> None:
    seen: set[int] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label}: {value}")
        seen.add(value)
