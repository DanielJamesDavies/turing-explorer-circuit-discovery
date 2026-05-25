#!/usr/bin/env python3
"""Count dataset sequences and pass-2 replay sequences.

Dataset shards are token streams with ``-1`` separators. This script counts the
same valid sequence rows as ``DataLoader``/``build_shard_table`` without loading
all shards into RAM. If ``top_ctx.pt`` is available, it also counts the unique
nonzero replay sequence IDs used by distributed pass 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count dataset shard sequences and optional pass-2 replay sequences."
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Dataset directory containing shard_*.npy files.",
    )
    parser.add_argument(
        "--top-ctx",
        default=None,
        help="Optional path to top_ctx.pt, usually outputs/<run_id>/top_ctx.pt.",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=None,
        help="Optional limit matching config data.n_shards. Defaults to all shards.",
    )
    parser.add_argument(
        "--include-first-token",
        action="store_true",
        help="Count sequences without DataLoader's default first-token skip.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help=(
            "Pass-2 candidates kept per replay sequence. Defaults to "
            "num_components * n_candidates_per_component."
        ),
    )
    parser.add_argument("--num-components", type=int, default=36)
    parser.add_argument("--n-candidates-per-component", type=int, default=16)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text summary.",
    )
    return parser.parse_args()


def list_shards(data_dir: Path, n_shards: int | None) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset directory does not exist: {data_dir}")
    shards = sorted(
        (path for path in data_dir.iterdir() if path.name.endswith(".npy")),
        key=_shard_sort_key,
    )
    return shards if n_shards is None else shards[:n_shards]


def count_shard_sequences(
    shard_path: Path,
    *,
    data_dir: Path,
    skip_first_token: bool,
) -> int:
    index_path = _index_path(data_dir, shard_path.name, skip_first_token)
    if index_path.exists() and index_path.stat().st_mtime_ns >= shard_path.stat().st_mtime_ns:
        return int(len(np.load(index_path, mmap_mode="r")))
    return int(len(build_shard_index(shard_path, skip_first_token=skip_first_token)))


def build_shard_index(shard_path: Path, *, skip_first_token: bool) -> np.ndarray:
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


def count_replay_sequences(top_ctx_path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(top_ctx_path, map_location="cpu", weights_only=False)
    ctx_seq_idx = _extract_ctx_seq_idx(payload)
    if not isinstance(ctx_seq_idx, torch.Tensor):
        raise TypeError("top_ctx ctx_seq_idx must be a torch.Tensor")
    nonzero = ctx_seq_idx[ctx_seq_idx != 0].to(torch.int64)
    unique = torch.unique(nonzero)
    total_entries = int(ctx_seq_idx.numel())
    nonzero_entries = int(nonzero.numel())
    unique_count = int(unique.numel())
    return {
        "path": str(top_ctx_path),
        "ctx_seq_idx_shape": [int(dim) for dim in ctx_seq_idx.shape],
        "ctx_seq_idx_entries": total_entries,
        "ctx_seq_idx_nonzero_entries": nonzero_entries,
        "replay_sequence_count": unique_count,
        "replay_sequence_min": int(unique.min().item()) if unique_count else None,
        "replay_sequence_max": int(unique.max().item()) if unique_count else None,
    }


def estimate_candidate_dump_bytes(replay_count: int | None, m: int) -> dict[str, Any]:
    if replay_count is None:
        return {
            "m": m,
            "bytes_per_replay_sequence": m * 8,
            "candidate_dump_bytes": None,
            "candidate_dump_gib": None,
            "validation_headroom_3x_gib": None,
        }
    total_bytes = replay_count * m * 8
    return {
        "m": m,
        "bytes_per_replay_sequence": m * 8,
        "candidate_dump_bytes": total_bytes,
        "candidate_dump_gib": total_bytes / (1024**3),
        "validation_headroom_3x_gib": (total_bytes * 3) / (1024**3),
    }


def _extract_ctx_seq_idx(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "ctx_seq_idx" in payload:
            return payload["ctx_seq_idx"]
        if "top_ctx" in payload:
            return _extract_ctx_seq_idx(payload["top_ctx"])
        if "payload" in payload:
            return _extract_ctx_seq_idx(payload["payload"])
    if hasattr(payload, "ctx_seq_idx"):
        return getattr(payload, "ctx_seq_idx")
    raise ValueError("top_ctx payload does not contain ctx_seq_idx")


def _shard_sort_key(path: Path) -> int:
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"shard filename must look like shard_<n>.npy: {path.name}") from error


def _index_path(data_dir: Path, shard_filename: str, skip_first_token: bool) -> Path:
    suffix = f"_sft{int(skip_first_token)}.idx.npy"
    return data_dir / ".shard_indices" / f"{shard_filename}{suffix}"


def format_gib(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} GiB"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data)
    skip_first_token = not args.include_first_token
    shards = list_shards(data_dir, args.n_shards)
    shard_counts: list[int] = []
    for shard in shards:
        shard_counts.append(
            count_shard_sequences(
                shard,
                data_dir=data_dir,
                skip_first_token=skip_first_token,
            )
        )

    total_sequences = int(sum(shard_counts))
    m = int(args.m) if args.m is not None else int(args.num_components * args.n_candidates_per_component)
    replay: dict[str, Any] | None = None
    if args.top_ctx is not None:
        replay = count_replay_sequences(Path(args.top_ctx))
    replay_count = replay["replay_sequence_count"] if replay is not None else None
    dump_estimate = estimate_candidate_dump_bytes(replay_count, m)
    result = {
        "data_dir": str(data_dir),
        "shard_count": len(shards),
        "skip_first_token": skip_first_token,
        "total_sequences": total_sequences,
        "min_sequences_per_shard": min(shard_counts) if shard_counts else 0,
        "max_sequences_per_shard": max(shard_counts) if shard_counts else 0,
        "mean_sequences_per_shard": (total_sequences / len(shard_counts)) if shard_counts else 0.0,
        "top_ctx": replay,
        "pass2_candidate_dump_estimate": dump_estimate,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print("Dataset")
    print(f"  data_dir: {result['data_dir']}")
    print(f"  shard_count: {result['shard_count']}")
    print(f"  skip_first_token: {str(skip_first_token).lower()}")
    print(f"  total_sequences: {total_sequences:,}")
    print(f"  min_sequences_per_shard: {result['min_sequences_per_shard']:,}")
    print(f"  max_sequences_per_shard: {result['max_sequences_per_shard']:,}")
    print(f"  mean_sequences_per_shard: {result['mean_sequences_per_shard']:,.2f}")
    print("")
    print("Pass 2 Replay")
    if replay is None:
        print("  top_ctx: not provided")
        print("  replay_sequence_count: unknown until top_ctx.pt exists")
    else:
        print(f"  top_ctx: {replay['path']}")
        print(f"  ctx_seq_idx_shape: {replay['ctx_seq_idx_shape']}")
        print(f"  ctx_seq_idx_nonzero_entries: {replay['ctx_seq_idx_nonzero_entries']:,}")
        print(f"  replay_sequence_count: {replay['replay_sequence_count']:,}")
        print(f"  replay_sequence_min: {replay['replay_sequence_min']}")
        print(f"  replay_sequence_max: {replay['replay_sequence_max']}")
    print("")
    print("Pass 2 Candidate Dump Estimate")
    print(f"  m: {dump_estimate['m']:,}")
    print(f"  bytes_per_replay_sequence: {dump_estimate['bytes_per_replay_sequence']:,}")
    print(f"  candidate_dump_size: {format_gib(dump_estimate['candidate_dump_gib'])}")
    print(f"  suggested_validation_headroom_3x: {format_gib(dump_estimate['validation_headroom_3x_gib'])}")


if __name__ == "__main__":
    main()
