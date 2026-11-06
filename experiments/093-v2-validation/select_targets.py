"""093 targets: 72 TuringLLM targets from the 062 15k run, 8 per depth band (L0-4 / L5-7 / L8-11) x kind (attn / mlp /
resid), drawn at random within each cell (seed 20261026). Population: every target with a held-out-strongest eval row
in out_full/out/main (no error / skip), a non-vacuous denominator (vacuous_tk False), a context record and a production
circuit. Not enriched for failures: the draw ignores every score.

  python experiments/093-v2-validation/select_targets.py      -> targets.txt, targets.json (cells + production rows)
"""
import glob
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "062-h100-protocol-v1" / "out_full" / "out"
SEED = 20261026
PER_CELL = 8
BANDS = (("L0-4", 0, 4), ("L5-7", 5, 7), ("L8-11", 8, 11))
KINDS = ("attn", "mlp", "resid")
PROD_KEYS = ("n", "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "freeM_dense_tk",
             "freeN_dense_tk", "vacuous_tk", "thin", "arm")


def band_of(layer):
    return next(b for b, lo, hi in BANDS if lo <= layer <= hi)


def main():
    rows = {}
    for f in sorted(glob.glob(str(RUN / "main" / "eval.shard*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("held") == "strong" and "error" not in r and "skip" not in r:
                rows[r["seed"]] = r                                   # last row wins (merge.py's rule)
    pop = {}
    for key, r in rows.items():
        l, k, _ = key.split(".")
        if k not in KINDS or r.get("vacuous_tk") is not False:
            continue
        if not (RUN / "ctx" / ("%s.pt" % key)).exists() or not (RUN / "main" / "circuits" / ("%s.pt" % key)).exists():
            continue
        pop.setdefault((band_of(int(l)), k), []).append(key)
    rng = random.Random(SEED)
    cells, keys = {}, []
    for b, _, _ in BANDS:
        for k in KINDS:
            cell = sorted(pop[(b, k)])
            pick = rng.sample(cell, PER_CELL)
            cells["%s/%s" % (b, k)] = dict(population=len(cell), targets=pick)
            keys += pick
    prod = {key: {f: rows[key].get(f) for f in PROD_KEYS} for key in keys}
    (HERE / "targets.txt").write_text("\n".join(keys) + "\n")
    (HERE / "targets.json").write_text(json.dumps(dict(seed=SEED, per_cell=PER_CELL, bands=BANDS, cells=cells,
                                                       population=sum(len(v) for v in pop.values()),
                                                       n_vacuous_excluded=sum(1 for r in rows.values()
                                                                              if r.get("vacuous_tk")),
                                                       production=prod), indent=1))
    for c, v in cells.items():
        print("%-12s population %5d  ->  %s" % (c, v["population"], " ".join(v["targets"])))
    print("%d targets" % len(keys))


if __name__ == "__main__":
    main()
