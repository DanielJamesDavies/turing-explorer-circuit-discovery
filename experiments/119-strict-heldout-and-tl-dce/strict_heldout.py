"""119 #3: 073 strict resample test on the PRODUCTION public circuits, with a choice of C donors.

073 strict_eval.main() calls score_target(G, key, pool) without c_held, so its C donors are the target's 48 TRAINING
contrast contexts: the donors the WCM v4 resample term trained on. This wrapper runs the same main() with
score_target bound to c_held=C_HELD (the 079 / 091 / 092 / 095 / 102 call pattern: SE.score_target(..., c_held=True)),
so C_HELD=1 gives C donors = the target's HELD-OUT contrast contexts. Everything else is 073's code unchanged:
targets (the chain's nulls list), A donors (48 strongest-train), R donors (the run's own random_pool.pt, copied from
the production strict output so R is identical), N_DRAWS 2, the anchor-null random variant.

  wsl: cd <repo>; PYTHONPATH=src HF_HUB_OFFLINE=1 MODEL=pythia-70m C_HELD=1 \
       .venv/bin/python -X utf8 experiments/119-strict-heldout-and-tl-dce/strict_heldout.py
  env: MODEL (pythia-70m | gpt2-small)  C_HELD (0 | 1)  LIMIT (optional smoke)
Writes out/strict_<model>_<train|held>/rows.jsonl (073's row schema).
"""
import functools
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL = os.environ["MODEL"]
C_HELD = os.environ.get("C_HELD", "0") == "1"
ARC = REPO / "experiments" / "109-pod-run-analysis" / "out" / "archives" / ("pub-%s-S10-results" % MODEL) \
    / "experiments" / "082-public-runs"
PROD_OUT = ARC / ("out_%s" % MODEL)
OUT73 = HERE / "out" / ("strict_%s_%s" % (MODEL, "held" if C_HELD else "train"))

# the chain's strict arm env (082 chain.sh arm ... strict: MODE=main TARGETS=nulls OUT73 RAND_DIR=anchor nulls)
os.environ.update(
    MODE="main", FAMILY=MODEL, DATA=str(REPO / "data_public" / MODEL), STORES=str(REPO / "outputs_public" / MODEL),
    OUT=str(PROD_OUT), CTX_DIR=str(PROD_OUT / "ctx"), WCM_VERSION="v4", SAE_PRELOAD="0",
    TARGETS=str(ARC / ("targets_%s_nulls.txt" % MODEL)), OUT73=str(OUT73),
    RAND_DIR=str(PROD_OUT / "harder_nulls" / "circuits" / "anchor"))
os.environ.setdefault("N_DRAWS", "2")
sys.path.insert(0, str(REPO / "experiments" / "073-strict-eval"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

import strict_eval as SE  # noqa: E402

OUT73.mkdir(parents=True, exist_ok=True)
pool_dst = OUT73 / "random_pool.pt"
if not pool_dst.exists():                      # the production run's own R pool, so R donors are identical
    torch.save(torch.load(PROD_OUT / "strict" / "random_pool.pt"), pool_dst)
SE.score_target = functools.partial(SE.score_target, c_held=C_HELD)
print("119 strict: model %s, C donors = %s contrast contexts" % (MODEL, "HELD-OUT" if C_HELD else "TRAINING"),
      flush=True)
SE.main()
