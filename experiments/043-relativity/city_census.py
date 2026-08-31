"""Census capital-city mentions in TuringLLM's training data.

Goal: find a city RARE enough that the model plausibly has no dedicated
latent (forcing composition) but COMMON enough to give ~64+ probe
windows. Counts anchored windows per city over a shard sample, using
the LAST sub-token of each city name (where the full identity exists).

  PYTHONPATH=src python experiments/043-relativity/city_census.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from model.tokenizer import Tokenizer

SEQ = 64
CITIES = ["Paris", "Tokyo", "Cairo", "Vienna", "Oslo", "Helsinki",
          "Canberra", "Nairobi", "Lima", "Havana", "Reykjavik",
          "Kathmandu", "Ottawa", "Wellington", "Ulaanbaatar"]


def main():
    tok = Tokenizer()
    anchors = {}
    for c in CITIES:
        # 29871 is the bare-space sentinel SentencePiece emits when a
        # leading-space string is encoded in isolation; real text fuses
        # the space into the word token, so requiring it never matches.
        ids = [i for i in tok.encode(" " + c) if i > 2 and i != 29871]
        anchors[c] = (ids, ids[-1])
        print("%-12s -> %s (anchor %d = %r)"
              % (c, ids, ids[-1], tok.decode([ids[-1]])))
    shards = sorted(Path("data").glob("*.npy"))[::10][:120]
    counts = {c: 0 for c in CITIES}
    for sp in shards:
        sh = np.asarray(np.load(sp, mmap_mode="r"))
        for c, (ids, last) in anchors.items():
            hits = np.where(sh == last)[0]
            if len(ids) > 1 and len(hits):
                # require the full sub-token sequence, not just the tail
                ok = np.zeros(len(hits), dtype=bool)
                for k, t in enumerate(reversed(ids)):
                    idx = hits - k
                    ok_k = (idx >= 0)
                    ok_k[ok_k] = sh[idx[ok_k]] == t
                    ok = ok_k if k == 0 else (ok & ok_k)
                hits = hits[ok]
            counts[c] += int(len(hits))
    print("\nanchored mentions in %d sampled shards (~%.0fM tokens):"
          % (len(shards), len(shards) * 540672 / 1e6))
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("  %-12s %6d   (x25 full-data estimate ~%d)"
              % (c, n, n * 3030 // len(shards)))


if __name__ == "__main__":
    main()
