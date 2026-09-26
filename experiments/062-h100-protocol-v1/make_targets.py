"""Target lists for the protocol-v1 H100 runs (run once, locally; the lists are committed).

  targets_stage1.txt  stratified sample of the 15,046 production targets: PER_CELL per (layer, site kind) cell (36
                      cells), plus the 16 pilot targets of 055 / 059 so the pod reproduces the local results
  targets_sweep.txt   a stratified subset of stage 1 for the WCM-vs-unweighted sweep (Figure 4 / DAN-76 / DAN-77):
                      SWEEP_PER_CELL per cell, plus the 16 pilot targets

  PYTHONPATH=src python experiments/062-h100-protocol-v1/make_targets.py
  env: PER_CELL (40)  SWEEP_PER_CELL (5)  SEED (0)
"""
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EXP = HERE.parent
PER_CELL = int(os.environ.get("PER_CELL", 40))
SWEEP_PER_CELL = int(os.environ.get("SWEEP_PER_CELL", 5))
RNG = np.random.default_rng(int(os.environ.get("SEED", 0)))


def main():
    ct = pd.read_parquet(EXP / "049-circuit-graph" / "tables_full" / "circuits.parquet")
    ct["key"] = ct.seed_layer.astype(str) + "." + ct.seed_kind.astype(str) + "." + ct.seed_index.astype(str)
    prod = ct.drop_duplicates("key")[["key", "seed_layer", "seed_kind"]]
    pilot = [s for s in (EXP / "055-close-contrast" / "seeds.txt").read_text().split() if s]
    stage1, sweep = list(pilot), list(pilot)
    for (layer, kind), g in prod.groupby(["seed_layer", "seed_kind"]):
        keys = [k for k in g.key.tolist() if k not in pilot]
        pick = list(RNG.choice(keys, size=min(PER_CELL, len(keys)), replace=False))
        stage1 += pick
        sweep += pick[:SWEEP_PER_CELL]
    # interleave by layer so every shard sees every depth (targets[i::k] per GPU)
    order = lambda xs: sorted(xs, key=lambda k: (zlib.crc32(k.encode()), k))       # deterministic shuffle
    (HERE / "targets_stage1.txt").write_text("\n".join(order(stage1)) + "\n")
    (HERE / "targets_sweep.txt").write_text("\n".join(order(sweep)) + "\n")
    # the full production list (DAN-75): stage 1 first, so a full run launched into the same OUT skips what stage 1
    # already did (per-target files) and the remaining targets follow in the same deterministic shuffle
    rest = [k for k in prod.key.tolist() if k not in set(stage1)]
    (HERE / "targets_full.txt").write_text("\n".join(order(stage1) + order(rest)) + "\n")
    print("full: %d targets" % (len(stage1) + len(rest)))
    print("stage 1: %d targets (%d cells x %d + %d pilot); sweep: %d" % (len(stage1), prod.groupby(["seed_layer", "seed_kind"]).ngroups,
                                                                        PER_CELL, len(pilot), len(sweep)))


if __name__ == "__main__":
    main()
