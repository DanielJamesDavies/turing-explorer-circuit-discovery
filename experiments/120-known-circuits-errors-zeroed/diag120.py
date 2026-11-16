"""120 diagnostic (GPU): 117's ORIGINAL (errors-live-fitted) circuits re-scored with 116's errpatch, no refit.

As 116 §1: if the error switch is wired correctly for these targets, the zeroed-scorer pass of the original circuits
should fall below the live pass. Jobs per circuit: text strong (zero + live; the live row also checks this process
reproduces 117's own row) and task strong folds 0-3 (zero only; live = 117's eval.jsonl, same GPU and code).
Records are 117's (corpus: 117's ctx via CTX_DIR; task: 117's task_ctx). Rows -> out/diag/eval.jsonl.

  wsl: MAX_MIN=40 PYTHONPATH=src .venv/bin/python -X utf8 -u experiments/120-known-circuits-errors-zeroed/diag120.py
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common120 as C  # noqa: E402

D = C.OUT / "diag"
D.mkdir(parents=True, exist_ok=True)
os.environ["OUT"] = str(D / "wcm_unused")
os.environ["CTX_DIR"] = str(C.E117 / "out" / "wcm" / "ctx")
import lib117 as L  # noqa: E402

L.setup_env()
import torch  # noqa: E402
import driver  # noqa: E402
import errpatch as EP  # noqa: E402

MAX_MIN = float(os.environ.get("MAX_MIN", "40"))


def main():
    t0 = time.time()
    R = driver.Runner()
    EP.install()
    keys = [t for t in (C.E117 / "out" / "targets_order.txt").read_text().split() if t]
    path = D / "eval.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                done.add((r["seed"], r["circuit"], r["data"], r["fold"], r["err_mode"]))
    fh = open(path, "a")
    for key in keys:
        if (time.time() - t0) / 60 > MAX_MIN:
            print("[120 diag] MAX_MIN reached (resumable)", flush=True)
            break
        rec_c = R.contexts(key)
        rec_t = torch.load(C.E117 / "out" / "wcm" / "task_ctx" / ("%s.pt" % key), weights_only=False)[key]
        for cname, sub in (("corpus", "main"), ("task", "task")):
            cp = C.E117 / "out" / "wcm" / sub / "circuits" / ("%s.pt" % key)
            if not cp.exists():
                continue
            c = torch.load(cp, weights_only=False)
            if c is None:
                continue
            jobs = []
            if rec_c.get("strong") is not None:
                jobs += [("text", 0, rec_c, m) for m in ("zero", "live")]
            if rec_t.get("strong") is not None:
                for k in range(4):
                    rf = L.fold(rec_t, "strong", k)
                    if rf is not None:
                        jobs.append(("task", k, rf, "zero"))
            for data, k, rec, m in jobs:
                if (key, cname, data, k, m) in done:
                    continue
                try:
                    EP.MODE["m"] = m
                    row = R.score(c, rec, "strong")
                finally:
                    EP.MODE["m"] = "live"
                row = {a: b for a, b in row.items() if isinstance(b, (int, float, str, bool)) or b is None}
                row.update(seed=key, circuit=cname, data=data, held="strong", fold=k, err_mode=m)
                fh.write(json.dumps(row) + "\n"); fh.flush()
        print("[120 diag] %s done (%.1f min)" % (key, (time.time() - t0) / 60), flush=True)
    print("EXIT 0", flush=True)


if __name__ == "__main__":
    main()
