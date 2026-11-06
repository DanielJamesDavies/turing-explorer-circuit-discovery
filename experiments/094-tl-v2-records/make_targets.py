"""TuringLLM 8k TARGET LIST for the protocol-v2 refit (Daniel, 2026-10-25): a seeded, stratified subset of the 062 15k.

  .venv/bin/python experiments/094-tl-v2-records/make_targets.py            (CPU, seconds; writes next to this file)

Frame: the 062 run's 15,046 targets (062/targets_full.txt), 35 layer x kind cells (no 0.attn). Quota: 8,000 spread as
evenly as the cells allow (water-filling: every cell gets the same q, or all it has when it holds fewer; 1.attn holds
134); the cells that get q + 1 are chosen by the seeded RNG.

Must-includes (all inside the frame; checked), placed first and counted against their cell's quota:
  sweep190   the 190 lambda-sweep targets (062/out_full/out/sweep: every seed with an eval row; 062/targets_sweep.txt
             lists 191, 7.attn.17087 has no sweep row)
  nulls126   the DAN-181 null panel (084/targets_turingllm_nulls.txt)
  case6      the case-study appendix table (paper/main.tex tab. case studies, 063): 9.resid.20419 (the running case),
             10.resid.15497, 8.resid.31095, 8.attn.29991, 9.resid.7640, 9.resid.17596
  gallery9   the random gallery (072, first draw, seed 20261012)
  asrun1000  the frozen stratified as-run subset (086/targets_asrun1000.txt), taken whole if every cell's quota holds it
The rest of each cell: a seeded permutation of the cell's other targets (seed SEED, numpy default_rng).

Writes targets_tl8k.txt (sorted by layer, kind, latent), the subsets as targets_tl8k_<name>.txt (the stable-core 128:
seeded, stratified over the 35 cells, from the 8k minus nothing: it may overlap the must-includes; gate64: the T2
early verify gate, 64 spread over the 35 cells by its own seeded generator, since the sorted list's first 64 would
all be 0.mlp), and
targets_tl8k.manifest.json (counts per cell, the must-include overlaps, md5 of every list and of the inputs).
"""
import hashlib
import json
import os
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
E62 = REPO / "experiments" / "062-h100-protocol-v1"
SEED = int(os.environ.get("SEED", "20261026"))
N_TOTAL = int(os.environ.get("N_TOTAL", "8000"))
N_STABLE = int(os.environ.get("N_STABLE", "128"))
N_GATE = int(os.environ.get("N_GATE", "64"))
OUT = Path(os.environ.get("OUT_TARGETS", str(HERE)))
KIND_RANK = {"attn": 0, "mlp": 1, "resid": 2}

INPUTS = OrderedDict(
    full=E62 / "targets_full.txt",
    nulls126=REPO / "experiments" / "084-harder-nulls" / "targets_turingllm_nulls.txt",
    asrun1000=REPO / "experiments" / "086-heldout-rebuild" / "targets_asrun1000.txt",
)
SWEEP_DIR = E62 / "out_full" / "out" / "sweep"
CASE6 = ["9.resid.20419", "10.resid.15497", "8.resid.31095", "8.attn.29991", "9.resid.7640", "9.resid.17596"]
GALLERY9 = ["3.attn.9767", "0.mlp.12557", "0.resid.27712", "5.attn.10168", "5.mlp.15947", "7.resid.29770",
            "9.attn.38891", "8.mlp.2453", "9.resid.10786"]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def read(path):
    return [t for t in Path(path).read_text().split() if t]


def cell(key):
    l, k, _ = key.split(".")
    return "%s.%s" % (l, k)


def sort_key(key):
    l, k, i = key.split(".")
    return int(l), KIND_RANK[k], int(i)


def sweep_targets():
    """Every seed with a row in the 062 sweep's eval files (the 190 the lambda-sweep figure uses)."""
    s = set()
    for p in sorted(SWEEP_DIR.glob("eval.shard*.jsonl")):
        for line in open(p):
            try:
                s.add(json.loads(line)["seed"])
            except Exception:  # noqa: BLE001  (truncated line)
                continue
    return sorted(s, key=sort_key)


def quotas(sizes, total, rng):
    """Water-filling: the largest q with sum(min(size, q)) <= total, then +1 for seeded cells until the total is met."""
    cells = sorted(sizes, key=lambda c: (int(c.split(".")[0]), KIND_RANK[c.split(".")[1]]))
    if total > sum(sizes.values()):
        raise SystemExit("N_TOTAL %d exceeds the frame (%d)" % (total, sum(sizes.values())))
    q = 0
    while sum(min(sizes[c], q + 1) for c in cells) <= total:
        q += 1
    quota = {c: min(sizes[c], q) for c in cells}
    room = [c for c in cells if sizes[c] > q]
    extra = total - sum(quota.values())
    for c in rng.permutation(room)[:extra]:
        quota[str(c)] += 1
    return quota


def main():
    full = read(INPUTS["full"])
    fset = set(full)
    assert len(fset) == len(full), "duplicate keys in targets_full"
    subsets = OrderedDict(sweep190=sweep_targets(), nulls126=read(INPUTS["nulls126"]), case6=list(CASE6),
                          gallery9=list(GALLERY9), asrun1000=read(INPUTS["asrun1000"]))
    if len(subsets["sweep190"]) != 190:
        raise SystemExit("expected 190 sweep targets in %s, found %d" % (SWEEP_DIR, len(subsets["sweep190"])))
    for name, ks in subsets.items():
        bad = [k for k in ks if k not in fset]
        if bad:
            raise SystemExit("%s: %d targets outside the 15k frame (%s)" % (name, len(bad), bad[:5]))
    by_cell = {}
    for k in full:
        by_cell.setdefault(cell(k), []).append(k)
    sizes = {c: len(v) for c, v in by_cell.items()}
    rng = np.random.default_rng(SEED)
    quota = quotas(sizes, N_TOTAL, rng)
    must_core = set().union(*(set(subsets[n]) for n in ("sweep190", "nulls126", "case6", "gallery9")))
    core_count = Counter(cell(k) for k in must_core)
    with_asrun = must_core | set(subsets["asrun1000"])
    asrun_count = Counter(cell(k) for k in with_asrun)
    over_core = {c: (n, quota[c]) for c, n in core_count.items() if n > quota[c]}
    if over_core:
        raise SystemExit("must-includes exceed their cell quota: %s" % over_core)
    asrun_fits = all(n <= quota[c] for c, n in asrun_count.items())
    must = with_asrun if asrun_fits else must_core
    chosen = []
    for c in sorted(by_cell, key=lambda c: (int(c.split(".")[0]), KIND_RANK[c.split(".")[1]])):
        mine = sorted((k for k in by_cell[c] if k in must), key=sort_key)
        rest = sorted((k for k in by_cell[c] if k not in must), key=sort_key)
        perm = rng.permutation(len(rest))
        chosen += mine + [rest[j] for j in perm[:quota[c] - len(mine)]]
    chosen = sorted(set(chosen), key=sort_key)
    assert len(chosen) == N_TOTAL, len(chosen)
    cset = set(chosen)
    # stable core: N_STABLE stratified over the cells (round-robin over cells in seeded order, seeded pick in a cell)
    pools = {c: sorted((k for k in chosen if cell(k) == c), key=sort_key) for c in quota}
    order = [str(c) for c in rng.permutation(sorted(quota))]
    picks = {c: [pools[c][j] for j in rng.permutation(len(pools[c]))] for c in order}
    stable, r = [], 0
    while len(stable) < N_STABLE:
        for c in order:
            if r < len(picks[c]) and len(stable) < N_STABLE:
                stable.append(picks[c][r])
        r += 1
    subsets["stable128"] = sorted(stable, key=sort_key)
    # the T2 early gate (2026-10-26): N_GATE targets spread over every cell, from its OWN generator (the draws above,
    # hence the 8k list and every earlier subset, are unchanged): round-robin over the cells in a seeded order, a
    # seeded pick inside each cell (64 = one per cell + a second in 29 cells)
    grng = np.random.default_rng([SEED, N_GATE])
    gorder = [str(c) for c in grng.permutation(sorted(quota))]
    gpicks = {c: [pools[c][j] for j in grng.permutation(len(pools[c]))] for c in gorder}
    gate, r = [], 0
    while len(gate) < min(N_GATE, len(chosen)):
        for c in gorder:
            if r < len(gpicks[c]) and len(gate) < N_GATE:
                gate.append(gpicks[c][r])
        r += 1
    subsets["gate64"] = sorted(gate, key=sort_key)

    OUT.mkdir(parents=True, exist_ok=True)
    files = OrderedDict()
    p = OUT / "targets_tl8k.txt"
    p.write_text("\n".join(chosen) + "\n")
    files["targets_tl8k.txt"] = md5(p)
    for name, ks in subsets.items():
        if name == "asrun1000" and not asrun_fits:
            continue
        p = OUT / ("targets_tl8k_%s.txt" % name)
        p.write_text("\n".join(ks) + "\n")
        files[p.name] = md5(p)
    kept = Counter(cell(k) for k in chosen)
    man = OrderedDict(
        seed=SEED, n_total=len(chosen), n_frame=len(full), n_cells=len(quota),
        rule="water-filled equal quota per layer x kind cell; must-includes first; rest seeded permutation "
             "(numpy default_rng(%d)); stable core %d stratified round-robin" % (SEED, N_STABLE),
        quota_q=sorted(Counter(quota.values()).items()),          # [(quota, n cells)]
        quota=OrderedDict((c, quota[c]) for c in sorted(quota, key=lambda c: (int(c.split(".")[0]),
                                                                              KIND_RANK[c.split(".")[1]]))),
        cell_sizes=sizes, kept_per_cell=dict(kept),
        must_includes=OrderedDict((n, dict(n=len(ks), in_8k=sum(k in cset for k in ks)))
                                  for n, ks in subsets.items()),
        must_include_union=len(must), asrun1000_included=asrun_fits,
        overlaps=OrderedDict(("%s&%s" % (a, b), len(set(subsets[a]) & set(subsets[b])))
                             for i, a in enumerate(subsets) for b in list(subsets)[i + 1:]),
        inputs=OrderedDict((n, dict(path=str(p.relative_to(REPO)), md5=md5(p))) for n, p in INPUTS.items()),
        sweep_source=str(SWEEP_DIR.relative_to(REPO)), files=files)
    (OUT / "targets_tl8k.manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("seed", "n_total", "quota_q", "must_includes", "must_include_union",
                                          "asrun1000_included", "files")}, indent=1))
    print("quota:", " ".join("%s:%d" % kv for kv in man["quota"].items()))


if __name__ == "__main__":
    main()
