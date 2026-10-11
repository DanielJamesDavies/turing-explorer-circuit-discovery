"""DAN-64 target lists (run once, locally; the lists are committed).

The frame is the 15k protocol-v1 run (062 out_full): targets that were fitted (status ok), are not vacuous on held-out
strongest, and whose context record and production circuit are on disk (the sweep copies both read-only). Layer-0
attention has no upstream SAE site and is not in the run, so there are 35 (layer, kind) cells, not 36.

  targets.txt          PER_CELL per (layer, kind) cell: 35 x 8 = 280
  targets_random.txt   the random-baseline subsample of targets.txt: RANDOM_PER_CELL per (depth band, kind), 9 x 5 = 45
  targets_smoke.txt    8 targets of targets.txt for the local smoke test, one per depth (layers 1-11) and kind in turn
  members.txt          the 062 tarball members (out/ctx/<key>.pt, out/main/circuits/<key>.pt) for the pod
                       (tar -T members.txt, README section 6)

  PYTHONPATH=src .venv/bin/python experiments/071-contrast-sweep/make_targets.py
  env: SRC_OUT (062 out_full/out)  PER_CELL (8)  RANDOM_PER_CELL (5)  SEED (20261009)
       SUFFIX (appended to every file name, e.g. "_4" for a 4-per-cell list; the smoke list is not rewritten when set)
"""
import glob
import json
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logic as L  # noqa: E402

SRC_OUT = Path(os.environ.get("SRC_OUT", str(HERE.parent / "062-h100-protocol-v1" / "out_full" / "out")))
PER_CELL = int(os.environ.get("PER_CELL", 8))
RANDOM_PER_CELL = int(os.environ.get("RANDOM_PER_CELL", 5))
SEED = int(os.environ.get("SEED", 20261009))
SUFFIX = os.environ.get("SUFFIX", "")
KIND_RANK = {"attn": 0, "mlp": 1, "resid": 2}
SMOKE_LAYERS = (1, 2, 4, 5, 7, 8, 10, 11)


def rows(pattern):
    out = []
    for f in sorted(glob.glob(str(SRC_OUT / "main" / pattern))):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def main():
    ok = {r["seed"] for r in rows("status.shard*.jsonl") if r.get("ok")}
    vac = {r["seed"] for r in rows("eval.shard*.jsonl") if r.get("held") == "strong" and r.get("vacuous_tk")}
    frame = sorted(k for k in ok - vac
                   if (SRC_OUT / "ctx" / ("%s.pt" % k)).exists() and (SRC_OUT / "main" / "circuits" / ("%s.pt" % k)).exists())
    print("frame: %d ok, %d vacuous, %d usable with ctx + circuit on disk" % (len(ok), len(vac & ok), len(frame)))
    rng = np.random.default_rng(SEED)
    cells = defaultdict(list)
    for k in frame:
        l, kind, _ = k.split(".")
        cells[(int(l), kind)].append(k)
    pick = []
    for cell in sorted(cells, key=lambda c: (c[0], KIND_RANK[c[1]])):
        ks = sorted(cells[cell])
        pick += sorted(rng.choice(ks, size=min(PER_CELL, len(ks)), replace=False).tolist())
    print("targets: %d cells x %d = %d" % (len(cells), PER_CELL, len(pick)))

    rcells = defaultdict(list)
    for k in pick:
        l, kind, _ = k.split(".")
        rcells[(L.band(int(l)), kind)].append(k)
    rnd = []
    for cell in sorted(rcells):
        ks = sorted(rcells[cell])
        rnd += sorted(rng.choice(ks, size=min(RANDOM_PER_CELL, len(ks)), replace=False).tolist())
    print("random subsample: %d cells x %d = %d" % (len(rcells), RANDOM_PER_CELL, len(rnd)))

    kinds = ("attn", "mlp", "resid")
    smoke = []
    for j, layer in enumerate(SMOKE_LAYERS):
        ks = sorted(k for k in pick if int(k.split(".")[0]) == layer and k.split(".")[1] == kinds[j % 3])
        smoke.append(ks[0])
    print("smoke: %s" % " ".join(smoke))

    order = lambda xs: sorted(xs, key=lambda k: (zlib.crc32(k.encode()), k))     # deterministic shuffle (062's)
    (HERE / ("targets%s.txt" % SUFFIX)).write_text("\n".join(order(pick)) + "\n")
    (HERE / ("targets_random%s.txt" % SUFFIX)).write_text("\n".join(order(rnd)) + "\n")
    if not SUFFIX:
        (HERE / "targets_smoke.txt").write_text("\n".join(smoke) + "\n")
    (HERE / ("members%s.txt" % SUFFIX)).write_text("\n".join(["out/ctx/%s.pt" % k for k in order(pick)]
                                                + ["out/main/circuits/%s.pt" % k for k in order(pick)]) + "\n")


if __name__ == "__main__":
    main()
