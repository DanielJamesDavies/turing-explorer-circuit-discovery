"""Find training sequences containing relativity vocabulary.

Seed selection here is TOKEN-DRIVEN, not label-driven: we locate the
tokens we care about in TuringLLM's own training data, then (in
find_seeds.py) ask which latents fire hardest AT those positions. No
auto-interp is involved anywhere in the loop.

Shards are flat int64 token streams; -1 marks padding. Sequences are
fixed length SEQ (64) as used everywhere else in the project.

  PYTHONPATH=src python experiments/043-relativity/find_seqs.py [n_shards]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from model.tokenizer import Tokenizer          # noqa: E402

HERE = Path(__file__).parent
SEQ = 64
# 'relativ' is the shared stem of relativity/relativistic; 'Ein'+'stein'
# is the entity. We record the STEM position as the anchor because that
# is the token whose activation defines a "relativity" latent.
ANCHORS = {"relativ": 14215, "Ein": 2694, "spac_etime": 26325,
           "gravity": 20953, "quantum": 12101}


def main():
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    shards = sorted(Path("data").glob("*.npy"))
    step = max(1, len(shards) // n_shards)
    shards = shards[::step][:n_shards]
    tok = Tokenizer()
    want = {v: k for k, v in ANCHORS.items()}
    out, counts = [], {k: 0 for k in ANCHORS}
    for si, sp in enumerate(shards):
        # Segment exactly as DataLoader._build_shard_index does: sequences
        # live between -1 separators, and the first token of each is
        # skipped (skip_first_token=True). Reshaping the raw stream into
        # fixed blocks misaligns every sequence and yields out-of-vocab
        # ids, so mirror the loader instead of guessing.
        sh = np.asarray(np.load(sp, mmap_mode="r"))
        sep = np.where(sh == -1)[0]
        starts = np.concatenate([[0], sep + 1]) + 1
        ends = np.concatenate([sep, [len(sh)]])
        keep = (ends - starts) == SEQ
        starts, ends = starts[keep], ends[keep]
        if not len(starts):
            continue
        arr = np.stack([sh[a_:b_] for a_, b_ in zip(starts, ends)])
        for tid, name in want.items():
            rows, cols = np.nonzero(arr == tid)
            for r, c in zip(rows.tolist(), cols.tolist()):
                if c == 0:                      # anchor at BOS slot: skip
                    continue
                counts[name] += 1
                out.append({"shard": sp.name, "seq": int(r), "pos": int(c),
                            "anchor": name,
                            "tokens": arr[r].tolist()})
        if (si + 1) % 100 == 0:
            print("  %d/%d shards, %d hits" % (si + 1, len(shards), len(out)),
                  flush=True)
    with open(HERE / "relativity_seqs.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("\nhits by anchor:", counts)
    print("%d sequences -> relativity_seqs.jsonl" % len(out))
    for r in [x for x in out if x["anchor"] == "relativ"][:4]:
        txt = tok.decode([t for t in r["tokens"] if t >= 0])
        mark = tok.decode([r["tokens"][r["pos"]]])
        print("\n  [%s @%d = %r]  %s" % (r["anchor"], r["pos"], mark, txt[:240]))


if __name__ == "__main__":
    main()
