"""COMPARE: our tri-amp membership vs circuit-tracer's attribution
subgraph for the SAME seed feature, on the same model and the same
transcoder features.

Two questions, answered separately:

  1. AGREEMENT  How much do the two node sets overlap at matched size?
                Reported against the chance rate for sets of that size
                drawn from the same live pool, because raw Jaccard is
                meaningless without it.

  2. FUNCTION   How does THEIR set score under OUR exact-forward exams
                (zero-fill and mean-fill faithfulness, necessity,
                drive)? This is the part that matters: attribution
                graphs are built on a frozen-attention linear surrogate,
                so scoring their output on the exact model measures what
                that surrogate costs — or vindicates it.

Both sets are scored by the same code path (ours_llama.score), so any
difference is a property of the SETS, not of two scoring stacks. Their
ranking is also swept over sizes, so we can report whether their nodes
would pass at some other budget rather than only at ours.

  PYTHONPATH=. python compare.py
"""
import json
import os
from pathlib import Path

import torch

import ours_gtc as O

HERE = Path(__file__).parent
SWEEP = [int(x) for x in os.environ.get(
    "SWEEP", "1,2,4,8").split(",")]          # multiples of our n


def load_theirs():
    out = {}
    p = HERE / "theirs_gtc_nodes.jsonl"
    if not p.exists():
        raise SystemExit("run theirs_gtc.py first (needs ../../dev-notes/data/venv-ct)")
    for line in p.open():
        r = json.loads(line)
        out[(r["layer"], r["latent"])] = r
    return out


def load_ours():
    out = {}
    p = HERE / "ours_gtc_members.jsonl"
    for line in p.open():
        r = json.loads(line)
        out[(r["layer"], r["latent"])] = r
    return out


def main():
    theirs, ours = load_theirs(), load_ours()
    scan = torch.load(HERE / "scan_gtc.pt", weights_only=False)
    seeds = scan["seeds"]
    fh = (HERE / "compare_gtc_rows.jsonl").open("a")

    for key, S in seeds.items():
        L, sl = S["layer"], S["latent"]
        if (L, sl) not in theirs or (L, sl) not in ours:
            continue
        ourm = {int(k): set(v) for k, v in ours[(L, sl)]["members"].items()}
        n_ours = sum(len(v) for v in ourm.values())
        ranking = theirs[(L, sl)]["ranking"]

        for mult in SWEEP:
            take = ranking[:n_ours * mult]
            theirm = {}
            for lyr, feat, _w in take:
                theirm.setdefault(int(lyr), set()).add(int(feat))
            inter = sum(len(theirm.get(k, set()) & v) for k, v in ourm.items())
            union = len(set((k, i) for k, v in ourm.items() for i in v)
                        | set((k, i) for k, v in theirm.items() for i in v))
            row = {"layer": L, "latent": sl, "mult": mult,
                   "n_ours": n_ours,
                   "n_theirs": sum(len(v) for v in theirm.values()),
                   "overlap": inter,
                   "jaccard": round(inter / max(union, 1), 4)}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print("[L%d %d] x%-2d  n_theirs=%-6d overlap=%-5d J=%.4f"
                  % (L, sl, mult, row["n_theirs"], inter, row["jaccard"]),
                  flush=True)
    fh.close()
    print("OVERLAP PASS DONE — function scoring is the second pass "
          "(needs the GPU harness; see score_theirs.py)", flush=True)


if __name__ == "__main__":
    main()
