"""Pick a held-out test set for prompt changes: passing circuits in the given 0-based layers, none described yet.

    python experiments/065-circuit-describer/pick_holdout.py --layers 6,7,8,9,10 --per-layer 3 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys

from report import RESULTS, Bundle, key_of


def seen_keys() -> set[str]:
    keys = set()
    for path in RESULTS.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(json.loads(line)["key"])
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,7,8,9,10")
    ap.add_argument("--per-layer", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    b = Bundle()
    seen = seen_keys()
    rng = random.Random(a.seed)
    picked = []
    for layer in [int(x) for x in a.layers.split(",")]:
        pool = [key_of(g) for (g,) in b.explorer.execute(
            "SELECT seed_gid FROM circuit WHERE pass = 1 AND layer = ? ORDER BY cid", (layer,))]
        pool = [k for k in pool if k not in seen]
        picked += rng.sample(pool, a.per_layer)
    sys.stdout.write(" ".join(picked) + "\n")


if __name__ == "__main__":
    main()
