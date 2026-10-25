"""Corpus-wide index for the held-out audit (085): one pass over TuringLLM's data/shard_*.npy.

Per sequence id (1-based, positional over the numerically sorted shards: the DataLoader's rule, skip_first_token):
  src      int32   the segment's first element (the "source-file index" the loader skips)
  hash     uint64  069 dup_report's content hash of the 64 loaded tokens (same weights, so ids hash identically)
plus the per-id duplicate multiplicity (how many corpus windows share that hash).

Read-only on data/. Writes experiments/085-heldout-audit/out/corpus_{src,hash,mult}.npy.

  PYTHONPATH=src .venv/bin/python experiments/085-heldout-audit/corpus_index.py
"""
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
DATA = REPO / "data"
OUT = HERE / "out"
SEQ = 64


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = np.random.default_rng(0).integers(1, 2 ** 62, size=SEQ, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    shards = sorted((f for f in os.listdir(DATA) if f.startswith("shard_") and f.endswith(".npy")),
                    key=lambda f: int(f.split("_")[1].split(".")[0]))[:3030]          # config data.n_shards = 3030
    srcs, hashes, lens = [np.zeros(1, np.int32)], [np.zeros(1, np.uint64)], []
    t0 = time.time()
    for n, f in enumerate(shards):
        a = np.load(DATA / f)
        seps = np.flatnonzero(a == -1)
        seg0 = np.concatenate([[0], seps + 1])
        ends = np.concatenate([seps, [len(a)]])
        starts = seg0 + 1
        keep = (ends - starts) > 1
        seg0, starts, ends = seg0[keep], starts[keep], ends[keep]
        L = ends - starts
        lens.append(L)
        if not (L == SEQ).all():
            raise SystemExit("ragged shard %s (lengths %s)" % (f, np.unique(L)))
        rows = a[starts[:, None] + np.arange(SEQ)[None, :]]
        srcs.append(a[seg0].astype(np.int32))
        with np.errstate(over="ignore"):
            hashes.append((rows.astype(np.uint64) * w[None, :]).sum(axis=1, dtype=np.uint64))
        if n % 300 == 0:
            print("%d/%d shards %.0fs" % (n, len(shards), time.time() - t0), flush=True)
    src, h = np.concatenate(srcs), np.concatenate(hashes)
    u, inv, cnt = np.unique(h[1:], return_inverse=True, return_counts=True)
    mult = np.concatenate([[0], cnt[inv]]).astype(np.int32)
    np.save(OUT / "corpus_src.npy", src)
    np.save(OUT / "corpus_hash.npy", h)
    np.save(OUT / "corpus_mult.npy", mult)
    n = len(h) - 1
    print("sequences %d | unique contents %d | duplicate rate %.4f | distinct src %d | %.0fs"
          % (n, len(u), 1 - len(u) / n, len(np.unique(src[1:])), time.time() - t0))


if __name__ == "__main__":
    main()
