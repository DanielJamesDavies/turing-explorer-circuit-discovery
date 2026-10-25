"""Golden check, whole tree: TuringLLM fit + eval (strong, mid) + 056 specificity + alpha = 1 through 062's
driver.Runner, run once with HEAD's tree (git archive of src/ + 049 / 056 / 059 / 062, native .so copied) and once
with the working tree. Adapted from the session's scratchpad golden_harness.py: the only change is that the driver
and src come from TREE (PYTHONPATH must be TREE/src), so every edited module (src, 049, 056, 062) is compared,
not driver.py alone. Records are read from the real run (062/out_full/out/ctx), never written.

  TREE=<tree root> ALPHA1=old|new OUTH=<dir> KEYS=a,b STEPS_H=8 PYTHONPATH=<TREE>/src python golden_tree.py
Run from the real repo root (data / models / outputs paths), CPU (CUDA_VISIBLE_DEVICES=), with CTX_DIR, GROUP_SPLIT,
EVAL_SPLIT, FAMILY, SAE_PRELOAD unset.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

for v in ("CTX_DIR", "GROUP_SPLIT", "EVAL_SPLIT", "FAMILY", "SAE_PRELOAD", "DATA", "STORES"):
    if os.environ.get(v):
        raise SystemExit("%s is set (%r): the check is for the default TuringLLM path" % (v, os.environ[v]))
REPO = Path.cwd()
TREE = Path(os.environ["TREE"]).resolve()
E062 = TREE / "experiments" / "062-h100-protocol-v1"
RUN = REPO / "experiments" / "062-h100-protocol-v1" / "out_full" / "out"
ALPHA1 = os.environ["ALPHA1"]
OUTH = Path(os.environ["OUTH"]).resolve()
KEYS = os.environ["KEYS"].split(",")
STEPS_H = int(os.environ.get("STEPS_H", 8))
os.environ["OUT"] = str(OUTH / "driver_out")
os.environ.setdefault("MODE", "main")
sys.path.insert(0, str(E062))
import driver as D  # noqa: E402
import circuit.protocol_contexts as PC  # noqa: E402
import eval.wcm_eval as WE  # noqa: E402
print("tree", TREE, "| driver", D.__file__, "| src", PC.__file__, WE.__file__, "| LAM", D.LAM, "STEPS", D.STEPS,
      flush=True)
assert str(TREE) in D.__file__ and str(TREE) in PC.__file__, "modules not loaded from TREE"


def ones(c):
    out = defaultdict(dict)
    for n in c.nodes.values():
        if n.metadata.get("role") == "seed":
            continue
        f = n.metadata["feature_id"]
        out[(int(f.layer), str(f.kind))][int(f.index)] = 1.0
    return dict(out)


R = D.Runner()
M = R.method(D.GAMMA, D.LAM, True, steps=STEPS_H)
(OUTH / "circuits").mkdir(parents=True, exist_ok=True)
ftr, frow = open(OUTH / "train.jsonl", "w"), open(OUTH / "rows.jsonl", "w")
for key in KEYS:
    rec = torch.load(RUN / "ctx" / ("%s.pt" % key), weights_only=False)[key]
    arm, thin = R.train_arm(rec)
    c, t_fit = R.fit(M, rec, key, arm, OUTH / "circuits" / ("%s.pt" % key), train_fh=ftr)
    print(key, "arm", arm, "mid", rec.get("mid") is not None, "n", D.n_nodes(c), "fit %.0fs" % t_fit, flush=True)
    for held in ("strong", "mid"):
        if rec.get(held) is not None:
            frow.write(json.dumps(dict(R.score(c, rec, held), _what="eval_" + held)) + "\n")
    for r in R.spec(c, rec, arm):
        frow.write(json.dumps(dict(r, _what="spec")) + "\n")
    if ALPHA1 == "old":                                   # rescore_alpha1.py's call at HEAD
        R.H.patch_eval_contexts(R.G, rec, "strong")
        r = R.V.score_circuit(c, {}, override_alphas=ones(c), skip_roles=True)
    else:                                                 # rescore_alpha1.py's call now
        r = R.score(c, rec, "strong", alphas=ones(c))
    frow.write(json.dumps(dict(r, _what="alpha1")) + "\n")
    frow.flush()
print("DONE", flush=True)
