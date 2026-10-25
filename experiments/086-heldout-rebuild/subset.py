"""FROZEN STRATIFIED SUBSET (DAN-157): the targets of the as-run re-score that gives the paired leakage effect.

The as-run records (062/out_full/out/ctx) are re-scored on the pod's hardware for a 1,000-target subset only (a full
as-run re-score is not needed: the paired difference ctx vs ctx_v2 on the same targets and the same machine is the
leakage effect; the ctx_v2 re-score covers everything). The subset is drawn once, here, and frozen:
  eligible   the 062 targets (targets_full.txt) with a main circuit: the circuit file exists and the run's last status
             row for the target is ok (fitted, accepted, scored). Rejected / skipped / errored targets are not drawn
  strata     band (L0-4 / L5-7 / L8-11) x kind (attn / mlp / resid): 9 cells, as even as possible: n // 9 per cell,
             the remainder one each to the first cells in sorted order; a cell with fewer eligible targets gives all of
             them and its shortfall is spread the same way over the cells that still have targets
  draw       random.Random(SEED).sample over each cell's sorted keys, cells in sorted order (deterministic)
  output     OUT_LIST (sorted keys, one per line) + OUT_LIST.manifest.json (seed, counts per cell, eligibility,
             md5 of the list and of the source list)

  PYTHONPATH=src python experiments/086-heldout-rebuild/subset.py      (no GPU; seconds)
  env: TARGETS (062 targets_full.txt)  MAIN_DIR (062/out_full/out/main)  N (1000)  SEED (20261024)
       OUT_LIST (086/targets_asrun1000.txt)
"""
import hashlib
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).parent
E062 = HERE.parent / "062-h100-protocol-v1"
TARGETS = Path(os.environ.get("TARGETS", str(E062 / "targets_full.txt")))
MAIN_DIR = Path(os.environ.get("MAIN_DIR", str(E062 / "out_full" / "out" / "main")))
N = int(os.environ.get("N", "1000"))
SEED = int(os.environ.get("SEED", "20261024"))
OUT_LIST = Path(os.environ.get("OUT_LIST", str(HERE / "targets_asrun1000.txt")))


def band_of(key):
    l = int(key.split(".")[0])
    return "L0-4" if l <= 4 else ("L5-7" if l <= 7 else "L8-11")


def cell_of(key):
    return (band_of(key), key.split(".")[1])


def allocate(sizes, n):
    """{cell: count}: as even as possible over the cells (sorted order), capped by each cell's size."""
    cells = sorted(sizes)
    out = {c: 0 for c in cells}
    left = min(int(n), sum(sizes.values()))
    while left > 0:
        open_ = [c for c in cells if out[c] < sizes[c]]
        share, rem = divmod(left, len(open_))
        for k, c in enumerate(open_):
            add = min(share + (1 if k < rem else 0), sizes[c] - out[c])
            out[c] += add
            left -= add
    return out


def stratified_subset(keys, n=N, seed=SEED):
    """(sorted subset, {cell: count}) of keys: n targets stratified by band x kind (module docstring)."""
    cells = {}
    for k in sorted(set(keys)):
        cells.setdefault(cell_of(k), []).append(k)
    alloc = allocate({c: len(v) for c, v in cells.items()}, n)
    rng = random.Random(int(seed))
    out = []
    for c in sorted(cells):
        out += rng.sample(cells[c], alloc[c])
    return sorted(out), {"%s/%s" % c: alloc[c] for c in sorted(cells)}


def eligible(targets, main_dir=MAIN_DIR):
    """Targets with a main circuit file whose last status row is ok."""
    status = {}
    for p in sorted(main_dir.glob("status.shard*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            status[r["seed"]] = r
    circ = main_dir / "circuits"
    return [k for k in targets if status.get(k, {}).get("ok") and (circ / ("%s.pt" % k)).exists()]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    targets = [t for t in TARGETS.read_text().split() if t]
    el = eligible(targets)
    if not el:
        raise SystemExit("no eligible targets under %s (status rows + circuits)" % MAIN_DIR)
    sub, counts = stratified_subset(el, N, SEED)
    OUT_LIST.write_text("\n".join(sub) + "\n", encoding="utf-8")
    man = dict(what="frozen stratified subset for the as-run re-score (paired leakage effect, DAN-157)",
               n=len(sub), seed=SEED, strata="band (L0-4 / L5-7 / L8-11) x kind", counts=counts,
               eligible="targets_full.txt with a main circuit file and an ok last status row (062 out_full/out/main)",
               n_targets=len(targets), n_eligible=len(el),
               eligible_per_cell={"%s/%s" % c: sum(cell_of(k) == c for k in el) for c in sorted({cell_of(k) for k in el})},
               source=str(TARGETS.name), source_md5=md5(TARGETS), list=OUT_LIST.name, md5=md5(OUT_LIST))
    mp = OUT_LIST.with_name(OUT_LIST.name + ".manifest.json")
    mp.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
