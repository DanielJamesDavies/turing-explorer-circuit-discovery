"""Seq-latent-index shard merge helpers for distributed pass 1."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Dict, Sequence

import torch


def merge_seq_latent_index_shards(
    worker_index_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    expected_shard_ids: Sequence[int],
    enabled: bool = True,
    shard_id_ranges: Dict[int, tuple[int, int]] | None = None,
) -> Dict[str, object]:
    """Copy worker seq_latent_index shard files into one canonical directory."""

    if not enabled:
        return {
            "enabled": False,
            "copied_shards": [],
            "duplicate_identical_shards": [],
            "output_dir": str(output_dir),
        }
    expected = sorted({int(shard_id) for shard_id in expected_shard_ids})
    if len(expected) != len(list(expected_shard_ids)):
        raise ValueError("expected seq_latent_index shard IDs must be unique")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    copied: list[int] = []
    duplicate_identical: list[int] = []
    seen_sources: dict[int, Path] = {}

    for worker_dir in worker_index_dirs:
        worker_path = Path(worker_dir)
        if not worker_path.exists():
            continue
        if not worker_path.is_dir():
            raise ValueError(f"seq_latent_index worker path is not a directory: {worker_path}")
        for source_path in sorted(worker_path.glob("shard_*.pt")):
            shard_id = _parse_seq_latent_index_shard_id(source_path)
            if shard_id not in expected:
                raise ValueError(f"unexpected seq_latent_index shard output: {shard_id}")
            _validate_seq_latent_index_shard_file(
                source_path,
                shard_id=shard_id,
                shard_id_ranges=shard_id_ranges,
            )
            destination = output_path / source_path.name
            if shard_id in seen_sources or destination.exists():
                if not destination.exists():
                    _copy_file_atomic(seen_sources[shard_id], destination)
                if not _seq_latent_index_files_equivalent(destination, source_path):
                    raise ValueError(
                        f"duplicate seq_latent_index shard differs: shard_{shard_id}.pt"
                    )
                duplicate_identical.append(shard_id)
                continue
            _copy_file_atomic(source_path, destination)
            seen_sources[shard_id] = source_path
            copied.append(shard_id)

    missing = [shard_id for shard_id in expected if not (output_path / f"shard_{shard_id}.pt").exists()]
    if missing:
        raise ValueError(f"missing seq_latent_index shard outputs: {missing}")

    return {
        "enabled": True,
        "copied_shards": copied,
        "duplicate_identical_shards": duplicate_identical,
        "expected_shards": expected,
        "output_dir": str(output_path),
    }


def _parse_seq_latent_index_shard_id(path: Path) -> int:
    stem = path.stem
    prefix = "shard_"
    if not stem.startswith(prefix):
        raise ValueError(f"invalid seq_latent_index shard filename: {path.name}")
    try:
        return int(stem[len(prefix) :])
    except ValueError as exc:
        raise ValueError(f"invalid seq_latent_index shard filename: {path.name}") from exc


def _validate_seq_latent_index_shard_file(
    path: Path,
    *,
    shard_id: int,
    shard_id_ranges: Dict[int, tuple[int, int]] | None = None,
) -> Dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"seq_latent_index shard_{shard_id}.pt must contain a dict")
    for component_id, tensor in payload.items():
        if not isinstance(component_id, int):
            raise ValueError("seq_latent_index component keys must be integers")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("seq_latent_index component values must be tensors")
        if tensor.dtype != torch.int32 or tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError("seq_latent_index tensors must have shape [N, 2] and dtype int32")
        if tensor.numel() == 0:
            continue
        sequence_ids = tensor[:, 0]
        latent_ids = tensor[:, 1]
        if int(sequence_ids.min()) < 1:
            raise ValueError("seq_latent_index sequence IDs must be positive")
        if int(latent_ids.min()) < 0:
            raise ValueError("seq_latent_index latent IDs must be non-negative")
        if shard_id_ranges is not None:
            if shard_id not in shard_id_ranges:
                raise ValueError(f"missing expected sequence range for shard {shard_id}")
            start_id, end_id = shard_id_ranges[shard_id]
            if int(sequence_ids.min()) < start_id or int(sequence_ids.max()) > end_id:
                raise ValueError("seq_latent_index sequence IDs outside shard range")
    return payload


def _seq_latent_index_files_equivalent(first: Path, second: Path) -> bool:
    if first.read_bytes() == second.read_bytes():
        return True
    first_payload = _validate_seq_latent_index_shard_file(first, shard_id=_parse_seq_latent_index_shard_id(first))
    second_payload = _validate_seq_latent_index_shard_file(second, shard_id=_parse_seq_latent_index_shard_id(second))
    if set(first_payload) != set(second_payload):
        return False
    return all(torch.equal(first_payload[key], second_payload[key]) for key in first_payload)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f"{destination.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    shutil.copyfile(source, tmp_path)
    tmp_path.replace(destination)


__all__ = ["merge_seq_latent_index_shards"]
