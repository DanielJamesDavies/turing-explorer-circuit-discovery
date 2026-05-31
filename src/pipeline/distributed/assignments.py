"""Pure deterministic assignment helpers for distributed pipeline work."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, TypeVar

from .manifest import DiscoveryCandidateAssignment, DiscoveryTaskAssignment, WorkAssignments
from .shard_table import ShardRecord, contains_sequence_id


T = TypeVar("T")
SEED_FREE_DISCOVERY_METHODS = {"cluster_contrast"}
DISCOVERY_SCHEDULING_CANDIDATE_CONTIGUOUS = "candidate_contiguous"
DISCOVERY_SCHEDULING_CANDIDATE_SHUFFLED = "candidate_shuffled"
DISCOVERY_SCHEDULING_METHOD_COST_GREEDY = "method_cost_greedy"


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


def partition_seed_ids(
    seed_ids: Sequence[int],
    worker_count: int,
    *,
    shuffle_seed: int | None = None,
) -> Dict[str, List[int]]:
    """Partition selected discovery seed/candidate IDs deterministically."""

    _reject_duplicates(seed_ids, "seed ID")
    partitioned_ids = list(seed_ids)
    if shuffle_seed is not None:
        random.Random(int(shuffle_seed)).shuffle(partitioned_ids)
    return partition_contiguous(partitioned_ids, worker_count)


def build_discovery_candidate_assignments(
    candidates: Sequence[Mapping[str, Any]],
    worker_count: int,
    *,
    methods: Sequence[str],
    shuffle_seed: int | None = None,
) -> tuple[Dict[str, List[int]], Dict[str, List[DiscoveryCandidateAssignment]]]:
    """Partition selected candidates and attach seed/task metadata per worker."""

    candidate_indices = list(range(len(candidates)))
    seed_assignments = partition_seed_ids(
        candidate_indices,
        worker_count,
        shuffle_seed=shuffle_seed,
    )
    method_list = [
        str(method)
        for method in methods
        if str(method) not in SEED_FREE_DISCOVERY_METHODS
    ]
    assignment_metadata: Dict[str, List[DiscoveryCandidateAssignment]] = {
        str(worker_id): []
        for worker_id in range(worker_count)
    }
    for worker_id, indices in seed_assignments.items():
        for candidate_index in indices:
            candidate = candidates[candidate_index]
            assignment_metadata[worker_id].append(
                DiscoveryCandidateAssignment(
                    candidate_index=candidate_index,
                    comp_idx=int(candidate["comp_idx"]),
                    latent_idx=int(candidate["latent_idx"]),
                    methods=method_list,
                    estimated_task_count=len(method_list),
                )
            )
    return seed_assignments, assignment_metadata


def assign_seed_free_method_owners(
    methods: Sequence[str],
    worker_count: int,
    *,
    owner_worker_id: int = 0,
) -> Dict[str, int]:
    """Assign seed-free discovery methods to one designated worker."""

    _validate_worker_count(worker_count)
    if owner_worker_id < 0 or owner_worker_id >= worker_count:
        raise ValueError("seed-free owner worker ID out of range")
    return {
        str(method): owner_worker_id
        for method in methods
        if str(method) in SEED_FREE_DISCOVERY_METHODS
    }


def build_discovery_task_assignments(
    candidates: Sequence[Mapping[str, Any]],
    worker_count: int,
    *,
    methods: Sequence[str],
    strategy: str = DISCOVERY_SCHEDULING_CANDIDATE_CONTIGUOUS,
    method_costs: Mapping[str, float] | None = None,
    seed_free_method_owners: Mapping[str, int] | None = None,
    shuffle_seed: int | None = None,
) -> tuple[Dict[str, List[DiscoveryTaskAssignment]], Dict[str, float]]:
    """Build deterministic discovery task schedules for reporting and future resume."""

    _validate_worker_count(worker_count)
    method_costs = method_costs or {}
    seed_free_method_owners = seed_free_method_owners or {}
    seed_methods = [
        str(method)
        for method in methods
        if str(method) not in SEED_FREE_DISCOVERY_METHODS
    ]
    tasks = _seed_based_discovery_tasks(candidates, seed_methods, method_costs)
    for method, owner in seed_free_method_owners.items():
        if owner < 0 or owner >= worker_count:
            raise ValueError("seed-free owner worker ID out of range")

    if strategy == DISCOVERY_SCHEDULING_CANDIDATE_CONTIGUOUS:
        assignments, totals = _candidate_contiguous_task_assignments(
            tasks,
            len(candidates),
            worker_count,
        )
    elif strategy == DISCOVERY_SCHEDULING_CANDIDATE_SHUFFLED:
        assignments, totals = _candidate_contiguous_task_assignments(
            tasks,
            len(candidates),
            worker_count,
            shuffle_seed=shuffle_seed,
        )
    elif strategy == DISCOVERY_SCHEDULING_METHOD_COST_GREEDY:
        assignments, totals = _greedy_task_assignments(tasks, worker_count)
    else:
        raise ValueError(f"unsupported discovery scheduling strategy: {strategy}")

    next_task_id = len(tasks)
    for method, owner in sorted(seed_free_method_owners.items()):
        cost = float(method_costs.get(method, 1.0))
        if cost < 0:
            raise ValueError("discovery method costs must be >= 0")
        assignments[str(owner)].append(
            DiscoveryTaskAssignment(
                task_id=next_task_id,
                method=method,
                estimated_cost=cost,
                seed_free=True,
            )
        )
        totals[str(owner)] += cost
        next_task_id += 1
    return assignments, totals


def build_discovery_scheduling_report(
    work_assignments: WorkAssignments,
    worker_count: int,
) -> Dict[str, Any]:
    """Summarize planned discovery task distribution by worker and method."""

    _validate_worker_count(worker_count)
    workers = []
    totals_by_method: Dict[str, int] = {}
    total_task_count = 0
    total_estimated_cost = 0.0
    for worker_id in range(worker_count):
        worker_key = str(worker_id)
        tasks = work_assignments.discovery_task_assignments.get(worker_key, [])
        methods: Dict[str, int] = {}
        for task in tasks:
            methods[task.method] = methods.get(task.method, 0) + 1
            totals_by_method[task.method] = totals_by_method.get(task.method, 0) + 1
        task_count = len(tasks)
        estimated_cost = float(work_assignments.discovery_worker_estimated_costs.get(worker_key, 0.0))
        total_task_count += task_count
        total_estimated_cost += estimated_cost
        workers.append(
            {
                "worker_id": worker_id,
                "candidate_count": len(work_assignments.discovery_seed_ids.get(worker_key, [])),
                "task_count": task_count,
                "estimated_cost": estimated_cost,
                "methods": methods,
                "seed_free_methods": sorted(
                    method
                    for method, owner in work_assignments.discovery_seed_free_method_owners.items()
                    if owner == worker_id
                ),
                "failed_task_ranges": work_assignments.discovery_failed_task_ranges.get(worker_key, []),
            }
        )
    return {
        "schema_version": 1,
        "scheduling_strategy": work_assignments.discovery_scheduling_strategy,
        "worker_count": worker_count,
        "total_task_count": total_task_count,
        "total_estimated_cost": total_estimated_cost,
        "methods": totals_by_method,
        "workers": workers,
    }


def select_discovery_resume_tasks(
    task_assignments: Mapping[str, Sequence[DiscoveryTaskAssignment]],
    failed_task_ranges: Mapping[str, Sequence[Sequence[int]]],
) -> Dict[str, List[DiscoveryTaskAssignment]]:
    """Select failed task IDs from a planned schedule for task-range resume."""

    selected: Dict[str, List[DiscoveryTaskAssignment]] = {}
    for worker_id, ranges in failed_task_ranges.items():
        task_ids: set[int] = set()
        for task_range in ranges:
            if len(task_range) != 2:
                raise ValueError("failed discovery task ranges must have start and end")
            start, end = int(task_range[0]), int(task_range[1])
            if start < 0 or end < start:
                raise ValueError("failed discovery task ranges must be non-negative and ordered")
            task_ids.update(range(start, end + 1))
        selected[worker_id] = [
            task
            for task in task_assignments.get(worker_id, [])
            if task.task_id in task_ids
        ]
    return selected


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


def _seed_based_discovery_tasks(
    candidates: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    method_costs: Mapping[str, float],
) -> List[DiscoveryTaskAssignment]:
    tasks: List[DiscoveryTaskAssignment] = []
    for candidate_index, candidate in enumerate(candidates):
        for method in methods:
            cost = float(method_costs.get(method, 1.0))
            if cost < 0:
                raise ValueError("discovery method costs must be >= 0")
            tasks.append(
                DiscoveryTaskAssignment(
                    task_id=len(tasks),
                    candidate_index=candidate_index,
                    comp_idx=int(candidate["comp_idx"]),
                    latent_idx=int(candidate["latent_idx"]),
                    method=method,
                    estimated_cost=cost,
                )
            )
    return tasks


def _candidate_contiguous_task_assignments(
    tasks: Sequence[DiscoveryTaskAssignment],
    candidate_count: int,
    worker_count: int,
    *,
    shuffle_seed: int | None = None,
) -> tuple[Dict[str, List[DiscoveryTaskAssignment]], Dict[str, float]]:
    candidate_assignments = partition_seed_ids(
        list(range(candidate_count)),
        worker_count,
        shuffle_seed=shuffle_seed,
    )
    candidate_to_worker = {
        candidate_index: worker_id
        for worker_id, candidate_indices in candidate_assignments.items()
        for candidate_index in candidate_indices
    }
    assignments: Dict[str, List[DiscoveryTaskAssignment]] = {
        str(worker_id): []
        for worker_id in range(worker_count)
    }
    totals = {str(worker_id): 0.0 for worker_id in range(worker_count)}
    for task in tasks:
        assert task.candidate_index is not None
        worker_id = candidate_to_worker[task.candidate_index]
        assignments[worker_id].append(task)
        totals[worker_id] += task.estimated_cost
    return assignments, totals


def _greedy_task_assignments(
    tasks: Sequence[DiscoveryTaskAssignment],
    worker_count: int,
) -> tuple[Dict[str, List[DiscoveryTaskAssignment]], Dict[str, float]]:
    assignments: Dict[str, List[DiscoveryTaskAssignment]] = {
        str(worker_id): []
        for worker_id in range(worker_count)
    }
    totals = {str(worker_id): 0.0 for worker_id in range(worker_count)}
    for task in sorted(tasks, key=lambda item: (-item.estimated_cost, item.task_id)):
        worker_id = min(totals, key=lambda key: (totals[key], int(key)))
        assignments[worker_id].append(task)
        totals[worker_id] += task.estimated_cost
    for worker_tasks in assignments.values():
        worker_tasks.sort(key=lambda task: task.task_id)
    return assignments, totals


def _validate_worker_count(worker_count: int) -> None:
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")


def _reject_duplicates(values: Iterable[int], label: str) -> None:
    seen: set[int] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label}: {value}")
        seen.add(value)
