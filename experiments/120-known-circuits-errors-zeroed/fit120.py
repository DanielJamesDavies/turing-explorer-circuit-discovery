"""120 stage 1 (GPU): 117's fits (fit117.main, unchanged) with the SAE errors zeroed in training AND scoring.

  - records: the SAME records as 117 (corpus: CTX_DIR = 117's out/wcm/ctx, read-only; task: copies of 117's
    out/wcm/task_ctx in out/wcm/task_ctx, so fit117 loads instead of rebuilding)
  - training: 116's install_train_error_ablation (common120.train_patch), every masked site trains on decode(code)
  - scoring: 116's errpatch. Runner.score is wrapped to score each job twice in-process: errors zeroed (the primary
    row, written by fit117 to out/wcm/dan215/eval.jsonl) and errors live (attached to the same row as "live")
Otherwise 117's recipe: WCM v4, lambda 1e-3, 400 steps, DEEP_BS 1, round-robin target order.

  wsl: BUDGET_MIN=75 PYTHONPATH=src .venv/bin/python -X utf8 -u experiments/120-known-circuits-errors-zeroed/fit120.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common120 as C  # noqa: E402

os.environ["OUT"] = str(C.WCM)
os.environ["CTX_DIR"] = str(C.E117 / "out" / "wcm" / "ctx")
os.environ.setdefault("TARGETS", str(C.E117 / "out" / "targets_order.txt"))
for k in ("task_ctx",):
    if not (C.WCM / k).is_dir():
        raise SystemExit("copy 117's out/wcm/%s into %s first" % (k, C.WCM / k))

import fit117  # noqa: E402  (117's L.setup_env() + 062 driver import, configured from the env above)
import driver  # noqa: E402
import errpatch as EP  # noqa: E402

install_train, TSTATS = C.train_patch()
install_train()
EP.install()
_orig_score = driver.Runner.score


def _scalar(r):
    return {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None}


def score(self, c, rec, held, alphas=None):
    try:
        EP.MODE["m"] = "live"
        live = _scalar(_orig_score(self, c, rec, held, alphas))
        EP.MODE["m"] = "zero"
        row = _orig_score(self, c, rec, held, alphas)
    finally:
        EP.MODE["m"] = "live"
    row.update(err_mode="zero", fit_err="zero", live=live, train_err_calls=TSTATS["calls"])
    return row


driver.Runner.score = score

if __name__ == "__main__":
    print("[120] train error patch installed; out %s; ctx %s" % (os.environ["OUT"], os.environ["CTX_DIR"]), flush=True)
    fit117.main()
    print("[120] train error-ablation calls %d" % TSTATS["calls"], flush=True)
    print("EXIT 0", flush=True)
