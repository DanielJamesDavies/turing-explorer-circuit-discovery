"""Arena A (A8b companion, 2026-11-05): Ge leaf-sum recovery jobs for Dunefsky's HELD-OUT circuits (run_dunefsky.py on
score_perprompt.py --emit-heldout-prompts), for symmetry with the training-prompt circuits (score_arena_a.py
--emit-ge-jobs). The jobs are built by score_arena_a's own emitter, unchanged (dunefsky_units + ge_jobs: their graph's
transcoder nodes at their own positions, on the circuit's own prompt). The only change is the identity: uid and job
get the "pp|" prefix that score_perprompt.py gives a held-out baseline row, so `score_perprompt.py --summary` joins
ge_recovery_tcz.py's output to those rows by uid. score_perprompt.py and score_arena_a.py are not changed.

No model and no GPU. .venv, repo root:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/bin/python -B experiments/107-true-baselines/arena_a/ge_jobs_heldout.py \
      --dunefsky-held <run_dunefsky.py out on the held prompts> --out <file.jsonl> [--keys k1,k2]
Then: ge_recovery_tcz.py --jobs <file.jsonl> --out <perprompt OUT>/ge_recovery.<name>.jsonl (ge107).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_arena_a as SA  # noqa: E402

PREFIX = "pp|"                  # = score_perprompt.py's held-out baseline uid ("pp|" + score_arena_a's uid)


def held_jobs(d_dir, keys=None):
    units = SA.dunefsky_units(d_dir, set(keys) if keys else None, "a2")
    jobs = SA.ge_jobs(units)
    for j in jobs:
        j["uid"] = PREFIX + j["uid"]
        j["job"] = PREFIX + j["job"]
        j["pset"] = "held"
    return units, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dunefsky-held", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", default="")
    a = ap.parse_args()
    os.environ["ARENA_A_SCOPE"] = "a2"
    sys.path.insert(0, str(SA.ROOT / "src"))
    import wcm_tcz as WT
    WT.register()
    keys = [k for k in a.keys.split(",") if k] or None
    units, jobs = held_jobs(a.dunefsky_held, keys)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    tmp.replace(out)
    print("held-out Dunefsky Ge jobs: %d (from %d circuits) -> %s" % (len(jobs), len(units), out), flush=True)


if __name__ == "__main__":
    main()
