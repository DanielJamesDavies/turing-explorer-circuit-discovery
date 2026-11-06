"""Experiment 081, v2 local run (2026-10-27): a DIAGNOSTIC re-score of fitted circuits on TRAINING items (GPU).

Why: IOI's and greater-than's held-out items are built from the other half of the name / noun lists (README,
Splits), so a held-out score mixes "does the circuit carry the task" with "does it carry names (nouns) it never saw".
This scores the same circuits, with the same A / C means (the whole training / contrast pools), on the first N_IN
training items: in-sample items, but from the training lists. It is not a held-out claim; it is the contrast that
separates the two readings.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -X utf8 experiments/081-task-circuits/v2_insample.py
  env: TASKS (ioi,greater_than)  ARMS (W,U)  N_IN (1000)  OUT (out_v2/<MODEL>)  EVAL_BS
Writes OUT/insample.jsonl (one row per task x arm x lambda; resumable).
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task_common as C  # noqa: E402

OUT = Path(os.environ.get("OUT", str(HERE / "out_v2" / C.MODEL)))
TASKS = [t for t in os.environ.get("TASKS", "ioi,greater_than").split(",") if t]
ARMS = [a for a in os.environ.get("ARMS", "W,U").split(",") if a]
N_IN = int(os.environ.get("N_IN", 1000))


def main():
    import engine as E
    from run import load_circuit, scalars
    p = OUT / "insample.jsonl"
    done = set()
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            done.add((r["task"], r["arm"], r["lam"]))
    eng = E.Engine(C.MODEL)
    fh = open(p, "a")
    for task in TASKS:
        T = C.load_task(task)
        tr = list(T["split"]["train"])
        view = dict(T)
        view["split"] = dict(train=tr, held=tr[:N_IN])          # score on training items; means over the whole pool
        B = eng.batch(view, C.load_contrast(task) if T["contrast"] == "generated" else None)
        for arm in ARMS:
            d = OUT / "circuits" / task / arm
            for cp in sorted(d.glob("*.json")) if d.exists() else []:
                lt = cp.stem
                if (task, arm, lt) in done or ".d" in lt:
                    continue
                alphas, _ = load_circuit(cp)
                t0 = time.time()
                row = dict(task=task, arm=arm, lam=lt, n_items=len(tr[:N_IN]), insample=True,
                           **scalars(eng.score(B, alphas)))
                row["t_score"] = round(time.time() - t0, 1)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print("%-14s %-2s lam %-7s n=%-6s train-items Z %.2f A %.2f C %.2f nec %.2f pass %s (%.0fs)" % (
                    task, arm, lt, row.get("n"), row.get("task_free0") or float("nan"),
                    row.get("task_freeM_topk") or float("nan"), row.get("task_freeN_topk") or float("nan"),
                    row.get("task_nec") or float("nan"), row.get("task_pass"), row["t_score"]), flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
