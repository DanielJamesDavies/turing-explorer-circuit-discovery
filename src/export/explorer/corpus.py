"""Direct reader for the token corpus (data/shard_*.npy).

Layout (checked 2026-09-27): each shard is a flat int64 array of fixed-width
segments, SEG_SLOTS = 66 slots per sequence:
    slot 0      not a token (doc/offset id, can exceed the vocab); skipped
    slots 1-64  the 64 tokens of the sequence
    slot 65     -1 separator
Global sequence ids are 1-based in shard order, matching
DataLoader(skip_first_token=True) and the ids stored in top_ctx / mid_ctx
(where 0 marks an empty slot).
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List

import numpy as np

SEG_SLOTS = 66
SEQ_LEN = 64
TOKEN_SLICE = slice(1, 1 + SEQ_LEN)
VOCAB_SIZE = 32064  # Phi-3 tokenizer (model/tokenizer.py)


def shard_files(data_dir: str) -> List[str]:
    """Shard paths sorted by shard number, as DataLoader orders them."""
    names = [f for f in os.listdir(data_dir) if re.fullmatch(r"shard_\d+\.npy", f)]
    names.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
    return [os.path.join(data_dir, f) for f in names]


class Corpus:
    """Maps 1-based sequence ids to 64-token rows by arithmetic.

    `seqs_per_shard` is the number of sequences in each shard; preflight checks
    whether every shard is full (8192), in which case ids map with no table.
    """

    def __init__(self, data_dir: str, seqs_per_shard: List[int] | None = None):
        self.paths = shard_files(data_dir)
        if seqs_per_shard is None:
            seqs_per_shard = [self._count(p) for p in self.paths]
        self.counts = np.asarray(seqs_per_shard, dtype=np.int64)
        self.starts = np.concatenate([[1], 1 + np.cumsum(self.counts)[:-1]])  # first id per shard
        self.n_seq = int(self.counts.sum())
        self._mm: Dict[int, np.ndarray] = {}

    @staticmethod
    def _count(path: str) -> int:
        n = np.load(path, mmap_mode="r").shape[0]
        return n // SEG_SLOTS

    def _shard(self, i: int) -> np.ndarray:
        if i not in self._mm:
            self._mm[i] = np.load(self.paths[i], mmap_mode="r")
        return self._mm[i]

    def locate(self, seq_id: int) -> tuple[int, int]:
        if not 1 <= seq_id <= self.n_seq:
            raise IndexError(f"sequence id {seq_id} outside 1..{self.n_seq}")
        shard = int(np.searchsorted(self.starts, seq_id, side="right") - 1)
        return shard, int(seq_id - self.starts[shard])

    def get(self, seq_id: int) -> np.ndarray:
        shard, row = self.locate(seq_id)
        base = row * SEG_SLOTS
        return np.asarray(self._shard(shard)[base + TOKEN_SLICE.start: base + TOKEN_SLICE.stop])

    def get_many(self, seq_ids: Iterable[int]) -> np.ndarray:
        rows = [self.get(int(s)) for s in seq_ids]
        return np.stack(rows) if rows else np.zeros((0, SEQ_LEN), np.int64)
