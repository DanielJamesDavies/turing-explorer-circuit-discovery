"""102 check panel: 72 NEW TuringLLM targets drawn exactly as exp 093 drew the dev panel (093 select_targets.py), with a
different seed and disjoint from 093 / 095's 72 (dev) and from 099 / 100's 18-target panel.

Population (093's rule, verbatim): every target with a held-out-strongest eval row in 062 out_full/out/main (no error /
skip; last row wins), kind in attn / mlp / resid, a non-vacuous denominator (vacuous_tk False), a context record in
062 out_full/out/ctx and a production circuit in out_full/out/main/circuits. Then the excluded keys are removed and 8
targets are drawn per depth band (L0-4 / L5-7 / L8-11) x kind cell with random.Random(SEED).sample over the sorted
cell. No score is used by the draw.

  python experiments/102-pass-improvement/select_check.py   -> check_targets.txt, check_targets.json
Refuses to overwrite an existing check_targets.json (the panel is frozen once drawn).
"""
import glob
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RUN = EXP / "062-h100-protocol-v1" / "out_full" / "out"
SEED = 20261031
PER_CELL = 8
BANDS = (("L0-4", 0, 4), ("L5-7", 5, 7), ("L8-11", 8, 11))
KINDS = ("attn", "mlp", "resid")


def band_of(layer):
    return next(b for b, lo, hi in BANDS if lo <= layer <= hi)


def main():
    out_json, out_txt = HERE / "check_targets.json", HERE / "check_targets.txt"
    if out_json.exists():
        raise SystemExit("check panel already frozen: %s" % out_json)
    dev = [k for k in (EXP / "093-v2-validation" / "targets.txt").read_text().split() if k]
    p99 = json.loads((EXP / "099-recursive-causal-scrubbing" / "panel.json").read_text())["targets"]
    excl = set(dev) | set(p99)
    assert len(dev) == 72 and len(set(p99)) == 18, (len(dev), len(p99))
    rows = {}
    for f in sorted(glob.glob(str(RUN / "main" / "eval.shard*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("held") == "strong" and "error" not in r and "skip" not in r:
                rows[r["seed"]] = r
    pop = {}
    for key, r in rows.items():
        l, k, _ = key.split(".")
        if k not in KINDS or r.get("vacuous_tk") is not False:
            continue
        if not (RUN / "ctx" / ("%s.pt" % key)).exists() or not (RUN / "main" / "circuits" / ("%s.pt" % key)).exists():
            continue
        if key in excl:
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
    assert not set(keys) & excl and len(set(keys)) == 72
    out_txt.write_text("\n".join(keys) + "\n")
    out_json.write_text(json.dumps(dict(seed=SEED, per_cell=PER_CELL, bands=BANDS, cells=cells,
                                        population=sum(len(v) for v in pop.values()),
                                        excluded=dict(dev_093=len(dev), panel_099=len(set(p99))),
                                        rule="093 select_targets population; minus 093 dev 72 and 099 panel 18; "
                                             "random.Random(%d).sample(sorted(cell), 8) per band x kind" % SEED),
                                   indent=1))
    for c, v in cells.items():
        print("%-12s population %5d  ->  %s" % (c, v["population"], " ".join(v["targets"])))
    print("%d targets" % len(keys))


if __name__ == "__main__":
    main()
