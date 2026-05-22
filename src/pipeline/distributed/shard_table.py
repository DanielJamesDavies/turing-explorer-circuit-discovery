"""Canonical dataset shard table for distributed sequence ID assignment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class ShardRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_index: int
    shard_filename: str
    sequence_count: int
    global_start_id: int
    global_end_id: int
    shard_size_bytes: int
    shard_mtime_ns: int
    index_filename: str
    index_size_bytes: int
    index_mtime_ns: int

    @field_validator("shard_index", "sequence_count")
    @classmethod
    def non_negative_ints(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be >= 0")
        return value

    @field_validator("global_start_id", "global_end_id")
    @classmethod
    def positive_sequence_bounds(cls, value: int) -> int:
        if value < 1:
            raise ValueError("global sequence bounds must be >= 1")
        return value

    @field_validator("shard_size_bytes", "shard_mtime_ns", "index_size_bytes", "index_mtime_ns")
    @classmethod
    def non_negative_metadata(cls, value: int) -> int:
        if value < 0:
            raise ValueError("file metadata values must be >= 0")
        return value


def list_shard_files(dataset_path: str | Path, n_shards: int | None = None) -> List[str]:
    """List shard files in the same deterministic order as DataLoader."""

    if n_shards is not None and n_shards < 0:
        raise ValueError("n_shards must be >= 0 when provided")
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset path does not exist: {path}")
    shards = [item.name for item in path.iterdir() if item.name.endswith(".npy")]
    shards.sort(key=_shard_sort_key)
    return shards if n_shards is None else shards[:n_shards]


def build_shard_table(
    dataset_path: str | Path,
    n_shards: int | None = None,
    *,
    skip_first_token: bool = True,
) -> List[ShardRecord]:
    """Build the canonical full-dataset shard table with half-open ID ranges."""

    path = Path(dataset_path)
    records: List[ShardRecord] = []
    next_sequence_id = 1
    for shard_index, shard_filename in enumerate(list_shard_files(path, n_shards)):
        shard_path = path / shard_filename
        index_path = _index_path(path, shard_filename, skip_first_token)
        index = _load_or_build_index(shard_path, index_path, skip_first_token)
        sequence_count = int(len(index))
        global_start_id = next_sequence_id
        global_end_id = next_sequence_id + sequence_count
        next_sequence_id = global_end_id

        shard_stat = shard_path.stat()
        index_stat = index_path.stat()
        records.append(
            ShardRecord(
                shard_index=shard_index,
                shard_filename=shard_filename,
                sequence_count=sequence_count,
                global_start_id=global_start_id,
                global_end_id=global_end_id,
                shard_size_bytes=shard_stat.st_size,
                shard_mtime_ns=shard_stat.st_mtime_ns,
                index_filename=str(index_path.relative_to(path)),
                index_size_bytes=index_stat.st_size,
                index_mtime_ns=index_stat.st_mtime_ns,
            )
        )

    validate_shard_records(records)
    return records


def validate_shard_table(
    dataset_path: str | Path,
    shard_table: Sequence[ShardRecord],
    *,
    skip_first_token: bool = True,
) -> None:
    """Validate that on-disk shards and cached indices still match the table."""

    path = Path(dataset_path)
    shard_files = list_shard_files(path, len(shard_table))
    expected_filenames = [record.shard_filename for record in shard_table]
    if shard_files != expected_filenames:
        raise ValueError("shard files or shard order differ from shard table")

    validate_shard_records(shard_table)
    for record in shard_table:
        shard_path = path / record.shard_filename
        index_path = path / record.index_filename
        if not shard_path.exists():
            raise FileNotFoundError(f"missing shard file: {record.shard_filename}")
        if not index_path.exists():
            raise FileNotFoundError(f"missing shard index file: {record.index_filename}")

        shard_stat = shard_path.stat()
        index_stat = index_path.stat()
        if index_stat.st_mtime_ns < shard_stat.st_mtime_ns:
            raise ValueError(f"stale shard index for {record.shard_filename}")
        if shard_stat.st_size != record.shard_size_bytes:
            raise ValueError(f"shard file metadata changed for {record.shard_filename}")
        if shard_stat.st_mtime_ns != record.shard_mtime_ns:
            raise ValueError(f"shard file metadata changed for {record.shard_filename}")
        if index_stat.st_size != record.index_size_bytes:
            raise ValueError(f"shard index metadata changed for {record.shard_filename}")
        if index_stat.st_mtime_ns != record.index_mtime_ns:
            raise ValueError(f"shard index metadata changed for {record.shard_filename}")

        index = np.load(index_path)
        expected_index_path = _index_path(path, record.shard_filename, skip_first_token)
        if index_path != expected_index_path:
            raise ValueError(f"cached shard index path changed for {record.shard_filename}")
        if len(index) != record.sequence_count:
            raise ValueError(f"sequence count changed for {record.shard_filename}")


def validate_shard_records(shard_table: Sequence[ShardRecord]) -> None:
    """Validate that records form contiguous non-overlapping half-open ranges."""

    next_sequence_id = 1
    seen_shards: set[int] = set()
    for expected_index, record in enumerate(shard_table):
        if record.shard_index in seen_shards:
            raise ValueError("duplicated shard index in shard table")
        seen_shards.add(record.shard_index)
        if record.shard_index != expected_index:
            raise ValueError("shard indices must be contiguous and ordered")
        if record.global_start_id != next_sequence_id:
            raise ValueError("global sequence IDs must be contiguous")
        if record.global_end_id != record.global_start_id + record.sequence_count:
            raise ValueError("global sequence range does not match sequence count")
        next_sequence_id = record.global_end_id


def sequence_ids_for_shard(record: ShardRecord) -> List[int]:
    """Return the global sequence IDs for one shard record."""

    return list(range(record.global_start_id, record.global_end_id))


def sequence_ids_for_shards(
    shard_table: Sequence[ShardRecord],
    shard_indices: Iterable[int],
) -> List[int]:
    """Return global sequence IDs for assigned shards in caller-provided order."""

    by_index = {record.shard_index: record for record in shard_table}
    sequence_ids: List[int] = []
    for shard_index in shard_indices:
        if shard_index not in by_index:
            raise ValueError(f"assigned shard index out of range: {shard_index}")
        sequence_ids.extend(sequence_ids_for_shard(by_index[shard_index]))
    return sequence_ids


def contains_sequence_id(shard_table: Sequence[ShardRecord], sequence_id: int) -> bool:
    """Return true when a global sequence ID is covered by the shard table."""

    return any(
        record.global_start_id <= sequence_id < record.global_end_id
        for record in shard_table
    )


def _shard_sort_key(filename: str) -> int:
    stem = filename.split("_", 1)[1].split(".", 1)[0]
    return int(stem)


def _index_path(
    dataset_path: Path,
    shard_filename: str,
    skip_first_token: bool,
) -> Path:
    suffix = f"_sft{int(skip_first_token)}.idx.npy"
    return dataset_path / ".shard_indices" / f"{shard_filename}{suffix}"


def _load_or_build_index(
    shard_path: Path,
    index_path: Path,
    skip_first_token: bool,
) -> np.ndarray:
    if index_path.exists() and index_path.stat().st_mtime_ns >= shard_path.stat().st_mtime_ns:
        return np.load(index_path)
    index = _build_shard_index(shard_path, skip_first_token)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(index_path, index)
    return index


def _build_shard_index(shard_path: Path, skip_first_token: bool) -> np.ndarray:
    shard = np.load(shard_path, mmap_mode="r")
    if len(shard) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    sep_positions = np.where(shard == -1)[0]
    skip = 1 if skip_first_token else 0
    segment_starts = np.concatenate([[0], sep_positions + 1])
    segment_ends = np.concatenate([sep_positions, [len(shard)]])
    starts = segment_starts + skip
    ends = segment_ends
    valid = (ends - starts) > 1
    return np.stack([starts[valid], ends[valid]], axis=1).astype(np.int64)
