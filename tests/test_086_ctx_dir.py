"""DAN-157 CTX_DIR plumbing in experiments/062-h100-protocol-v1/driver.py (every 062 companion, 073, 083, 084 and the
086 re-score read records through driver.Runner.contexts): unset = unchanged; set = records read from there, never
built, and the explicit eval split forced (the scorer and 056 specificity read it). Subprocesses keep the env apart."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
E062 = REPO / "experiments" / "062-h100-protocol-v1"

PROBE = r"""
import json, os, sys
from types import SimpleNamespace
sys.path.insert(0, %(e062)r)
import driver
import torch
out = dict(ctx_dir=driver.CTX_DIR, eval_split=os.environ.get("EVAL_SPLIT"), group=driver.GROUP_SPLIT)
R = object.__new__(driver.Runner)
R.H = SimpleNamespace(parse=lambda k: (1, "mlp", 2))
R.ctx_dir = driver.Path(driver.CTX_DIR) if driver.CTX_DIR else driver.OUT / "ctx"
try:
    out["have"] = R.contexts("1.mlp.2")["marker"]
except Exception as e:
    out["have"] = type(e).__name__
try:
    R.contexts("1.mlp.3")
    out["missing"] = "built"
except Exception as e:
    out["missing"] = type(e).__name__
import amp_eval_pass_v2 as V
out["V"] = V.EVAL_SPLIT
print("JSON" + json.dumps(out))
"""


def run(env_extra, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CTX_DIR", "EVAL_SPLIT", "OUT", "FAMILY")}
    env.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="", OUT=str(tmp_path / "out"), **env_extra)
    p = subprocess.run([sys.executable, "-c", PROBE % dict(e062=str(E062))], env=env, capture_output=True, text=True,
                       cwd=str(REPO), timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
    return p, (json.loads(line[-1][4:]) if line else None)


def put(d, key, marker):
    d.mkdir(parents=True, exist_ok=True)
    torch.save({key: dict(marker=marker, strong=None)}, d / ("%s.pt" % key))


def test_ctx_dir_set(tmp_path):
    put(tmp_path / "v2", "1.mlp.2", "rebuilt")
    put(tmp_path / "out" / "ctx", "1.mlp.2", "original")
    p, out = run(dict(CTX_DIR=str(tmp_path / "v2")), tmp_path)
    assert out is not None, p.stderr[-2000:]
    assert out["eval_split"] == "explicit" and out["V"] == "explicit"
    assert out["have"] == "rebuilt"                       # read from CTX_DIR, not OUT/ctx
    assert out["missing"] == "FileNotFoundError"          # never built into a rebuilt directory
    assert out["group"] is False                          # TuringLLM: no group split


def test_ctx_dir_unset_is_unchanged(tmp_path):
    put(tmp_path / "out" / "ctx", "1.mlp.2", "original")
    p, out = run({}, tmp_path)
    assert out is not None, p.stderr[-2000:]
    assert out["ctx_dir"] is None and out["eval_split"] is None and out["V"] == "as_run"
    assert out["have"] == "original"
    assert out["missing"] != "FileNotFoundError"          # the old path would build the record (fails here: no G)


def test_ctx_dir_refuses_as_run_split(tmp_path):
    p, out = run(dict(CTX_DIR=str(tmp_path), EVAL_SPLIT="as_run"), tmp_path)
    assert p.returncode != 0 and "EVAL_SPLIT=explicit" in (p.stderr + p.stdout)


def test_rescore_defaults(tmp_path):
    code = ("import os, sys; sys.path.insert(0, %r); import rescore as RS; "
            "print('JSON' + __import__('json').dumps(dict(e=os.environ['EVAL_SPLIT'], out=RS.OUT_RS.name, "
            "ctx=RS.CTX_DIR.name, jobs=RS.JOBS)))" % str(REPO / "experiments" / "086-heldout-rebuild"))
    env = {k: v for k, v in os.environ.items() if k not in ("CTX_DIR", "EVAL_SPLIT", "OUT")}
    env.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=str(REPO), timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
    assert line, p.stderr[-2000:]
    out = json.loads(line[-1][4:])
    # DAN-17 random is no longer a default job (084's harder nulls score their references on ctx_v2 directly)
    assert out == dict(e="explicit", out="rescore_ctx_v2", ctx="ctx_v2", jobs=["main", "sweep", "alpha1"])
