"""Stage 2: the whole token corpus as one uint16 [n_seq, 64] array, row = seq_id - 1."""
from __future__ import annotations

import os
import time

import numpy as np

from export.explorer.bundle import Bundle
from export.explorer.corpus import SEG_SLOTS, SEQ_LEN, TOKEN_SLICE, VOCAB_SIZE, shard_files


def build(bundle: Bundle, data_dir: str) -> dict:
    t0 = time.time()
    paths = shard_files(data_dir)
    counts = [np.load(p, mmap_mode="r").shape[0] // SEG_SLOTS for p in paths]
    n_seq = int(sum(counts))
    out = os.path.join(bundle.tokens, "tokens.npy")
    tmp = out + ".tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(n_seq, SEQ_LEN))
    row = 0
    for i, (p, c) in enumerate(zip(paths, counts)):
        seg = np.asarray(np.load(p, mmap_mode="r")).reshape(-1, SEG_SLOTS)
        if not (seg[:, -1] == -1).all():
            raise ValueError(f"{p}: separator not at slot {SEG_SLOTS - 1}")
        tok = seg[:, TOKEN_SLICE]
        if int(tok.max()) >= VOCAB_SIZE or int(tok.min()) < 0:
            raise ValueError(f"{p}: token id outside [0, {VOCAB_SIZE})")
        arr[row:row + c] = tok.astype(np.uint16)
        row += c
        if i % 500 == 0:
            print(f"    tokens: shard {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    arr.flush()
    del arr
    os.replace(tmp, out)
    return dict(data=data_dir, n_seq=n_seq, n_shards=len(paths), seqs_per_shard_first=counts[0],
                seqs_last_shard=counts[-1], secs=round(time.time() - t0, 1))
