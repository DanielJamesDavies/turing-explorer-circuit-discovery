"""DAN-181 TuringLLM target panel (decided 2026-10-24): 126 targets = 14 per depth band x site kind cell, drawn from the
062 lambda-sweep targets (targets_sweep.txt; the panel DAN-159 / 083 uses), restricted to targets with a production
circuit. The first 7 of each cell (63, the extra-draw subsample: SUB_DRAWS draws) are the 63 targets of the DAN-17
sweep run (062/out_random, 7 per cell, seed 20261003), so every subsample target has stored DAN-17 circuits; the other
7 per cell are a seeded draw (nulls_core.PANEL_SEED) from the rest of the cell. CPU only, no model.

Written once and frozen (a differing existing list is never overwritten; --force overrides):
  experiments/084-harder-nulls/targets_turingllm_nulls.txt   the order harder_nulls.py takes (subsample first)
  experiments/084-harder-nulls/targets_turingllm_nulls.json  counts per cell, sub_n, DAN-17 pairing, input md5s

  PYTHONPATH=src python experiments/084-harder-nulls/make_panel.py
The public models' panels come from 082/make_targets_public.py (subset "nulls", 128 per model).
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
E62 = HERE.parent / "062-h100-protocol-v1"
sys.path.insert(0, str(HERE))
import nulls_core as NC  # noqa: E402

PER_CELL, SUB_PER_CELL = 14, 7
SWEEP = E62 / "targets_sweep.txt"
MAIN = E62 / "out_full" / "out" / "main" / "circuits"
PRIOR = E62 / "out_random" / "random.jsonl"                 # DAN-17 on the sweep panel: 63 targets x 2 draws
DAN17_DIRS = (E62 / "out_random", E62 / "out_random_scale")  # where a target's DAN-17 circuit (<key>.d0.pt) may live
LIST = HERE / "targets_turingllm_nulls.txt"
MANIFEST = HERE / "targets_turingllm_nulls.json"


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def has_circuit(key):
    """A production circuit exists and was accepted (a rejected fit is saved as None)."""
    import torch
    p = MAIN / ("%s.pt" % key)
    return p.exists() and torch.load(p, weights_only=False) is not None


def dan17_source(key):
    for d in DAN17_DIRS:
        if (d / "circuits" / ("%s.d0.pt" % key)).exists():
            return d.name
    return None


def build():
    sweep = [k for k in SWEEP.read_text().split() if k]
    keys = [k for k in sweep if has_circuit(k)]
    prior = NC.unique_in_order(json.loads(l)["seed"] for l in open(PRIOR) if l.strip())
    order, sub_n = NC.panel(keys, PER_CELL, SUB_PER_CELL, prior=prior, seed=NC.PANEL_SEED)
    cell = lambda k: "%s %s" % (NC.band(NC.parse(k)[0]), NC.parse(k)[1])
    src = {k: dan17_source(k) for k in order}
    man = dict(
        what="DAN-181 TuringLLM panel: %d per band x kind cell from the 062 sweep targets with a production circuit; "
             "the first %d per cell (positions < sub_n) are the DAN-17 sweep run's targets" % (PER_CELL, SUB_PER_CELL),
        per_cell=PER_CELL, sub_per_cell=SUB_PER_CELL, seed=NC.PANEL_SEED, n=len(order), sub_n=sub_n,
        sweep_listed=len(sweep), sweep_with_circuit=len(keys),
        sweep_without_circuit=sorted(set(sweep) - set(keys)),
        cells={c: n for c, n in sorted(Counter(cell(k) for k in order).items())},
        cells_available={c: n for c, n in sorted(Counter(cell(k) for k in keys).items())},
        prior_in_subsample=sum(1 for k in order[:sub_n] if k in set(prior)),
        dan17_stored={d.name: sum(1 for v in src.values() if v == d.name) for d in DAN17_DIRS},
        dan17_stored_subsample=sum(1 for k in order[:sub_n] if src[k]),
        dan17_refit=sorted(k for k, v in src.items() if v is None),
        inputs={p.name: md5(p) for p in (SWEEP, PRIOR)})
    man["dan17_refit_n"] = len(man["dan17_refit"])
    return order, man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="overwrite a frozen list that differs")
    a = ap.parse_args(argv)
    order, man = build()
    text = "\n".join(order) + "\n"
    if LIST.exists() and LIST.read_text() != text and not a.force:
        raise SystemExit("refusing to overwrite the frozen %s (it differs; --force overrides)" % LIST)
    with open(LIST, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(text)
    man["list_md5"] = md5(LIST)
    with open(MANIFEST, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print("panel: %d targets (sub_n %d); cells %s" % (man["n"], man["sub_n"], man["cells"]))
    print("DAN-17 circuits stored: %s (subsample %d/%d); refit in-process: %d" % (
        man["dan17_stored"], man["dan17_stored_subsample"], man["sub_n"], man["dan17_refit_n"]))
    print("wrote", LIST, MANIFEST)


if __name__ == "__main__":
    main()
