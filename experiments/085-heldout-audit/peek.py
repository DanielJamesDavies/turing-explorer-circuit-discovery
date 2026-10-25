"""Decode corpus windows / record rows for the audit README examples (085). Read-only.

  PYTHONPATH=src .venv/bin/python experiments/085-heldout-audit/peek.py src <src_index> [n]
  PYTHONPATH=src .venv/bin/python experiments/085-heldout-audit/peek.py rec <key> <pool S|M|C> <row> [<row> ...]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
CTX = REPO / "experiments" / "062-h100-protocol-v1" / "out_full" / "out" / "ctx"


def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", use_fast=True, local_files_only=True)


def window(i):
    """Tokens of corpus sequence id i (1-based over sorted shards, 8192 per shard, skip_first_token)."""
    import os
    shard, local = (i - 1) // 8192, (i - 1) % 8192
    idx = np.load(REPO / "data" / ".shard_indices" / ("shard_%d.npy_sft1.idx.npy" % shard))
    a = np.load(REPO / "data" / ("shard_%d.npy" % shard), mmap_mode="r")
    st, en = idx[local]
    return np.asarray(a[st:en]), int(a[st - 1])


def main():
    T = tok()
    if sys.argv[1] == "src":
        src = np.load(HERE / "out" / "corpus_src.npy", mmap_mode="r")
        mult = np.load(HERE / "out" / "corpus_mult.npy", mmap_mode="r")
        h = np.load(HERE / "out" / "corpus_hash.npy", mmap_mode="r")
        s = int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        ids = np.flatnonzero(np.asarray(src) == s)
        ids = ids[ids > 0]
        print("src %d: %d windows, %d with a twin" % (s, len(ids), int((np.asarray(mult)[ids] > 1).sum())))
        for i in ids[:n]:
            t, s0 = window(int(i))
            print("--- id %d (src %d, mult %d, hash %x)" % (i, s0, mult[i], h[i]))
            print(repr(T.decode(t.tolist()))[:400])
    else:
        import torch
        key, pool = sys.argv[2], sys.argv[3]
        rec = torch.load(CTX / ("%s.pt" % key), weights_only=False)[key]
        for r in map(int, sys.argv[4:]):
            if pool == "C":
                t, a, pk, i = rec["neg"][r], None, None, (rec["neg_ids"] or [None] * 999)[r]
            else:
                d = rec["strong" if pool == "S" else "mid"]
                t, a, pk, i = d["pos"][r], int(d["arg"][r]), float(d["peak"][r]), d["ids"][r]
            print("--- %s %s row %d id %s anchor %s peak %s" % (key, pool, r, i, a, pk))
            toks = t.tolist()
            if a is not None:
                print("prefix to anchor:", repr(T.decode(toks[:a + 1]))[:300], "| anchor token:", repr(T.decode([toks[a]])))
            print(repr(T.decode(toks))[:400])


if __name__ == "__main__":
    main()
