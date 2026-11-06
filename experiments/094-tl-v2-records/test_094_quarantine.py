"""CPU tests for the T2 early gate + quarantine (094 quarantine.py, 082 chain.sh t2_records / tl_use), on fake verify
shards, plus a DRY_RUN of the chain showing every stage after T2 reading the quarantine-applied lists.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/094-tl-v2-records/test_094_quarantine.py -q
"""
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
REPO = HERE.parent.parent
spec = importlib.util.spec_from_file_location("e094_quarantine", HERE / "quarantine.py")
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)

KEYS = ["%d.mlp.%d" % (l, i) for l in range(3) for i in range(10)]           # 30 fake targets
GATE = KEYS[::5]                                                              # 6 gate targets


def write(p, keys):
    p.write_text("".join(k + "\n" for k in keys))
    return str(p)


def shards(d, rows, k=2):
    """rows: {key: True (ok) | False (failed) | None (missing record) | absent (no row)} over k shard files."""
    d.mkdir(parents=True, exist_ok=True)
    fhs = [open(d / ("verify.shard%d.jsonl" % i), "w") for i in range(k)]
    for j, (key, v) in enumerate(rows.items()):
        r = dict(seed=key, missing=True) if v is None else dict(seed=key, ok=bool(v), failures=[] if v else
                                                                ["strong 3: near vs an earlier activating row"])
        fhs[j % k].write(json.dumps(r) + "\n")
    for f in fhs:
        f.close()
    return str(d)


def test_early_gate_failure_stops(tmp_path, capsys):
    g = write(tmp_path / "gate.txt", GATE)
    ok = shards(tmp_path / "ok", {k: True for k in GATE})
    assert Q.main(["gate", "--verify", ok, "--gate", g]) == 0
    bad = shards(tmp_path / "bad", dict({k: True for k in GATE}, **{GATE[2]: False}))
    assert Q.main(["gate", "--verify", bad, "--gate", g]) == 1
    assert "GATE FAIL %s" % GATE[2] in capsys.readouterr().out
    missing = shards(tmp_path / "missing", {k: True for k in GATE[1:]})          # no row at all for GATE[0]
    assert Q.main(["gate", "--verify", missing, "--gate", g]) == 1
    norec = shards(tmp_path / "norec", dict({k: True for k in GATE}, **{GATE[4]: None}))
    assert Q.main(["gate", "--verify", norec, "--gate", g]) == 1


def test_late_failure_quarantines_and_continues(tmp_path, capsys):
    t = write(tmp_path / "all.txt", KEYS)
    g = write(tmp_path / "gate.txt", GATE)
    sweep = write(tmp_path / "sweep.txt", KEYS[:8])
    stable = write(tmp_path / "stable.txt", KEYS[20:26])
    late_fail, late_missing, late_norow = KEYS[3], KEYS[21], KEYS[29]           # none of them a gate target
    rows = {k: True for k in KEYS if k != late_norow}
    rows[late_fail], rows[late_missing] = False, None
    v = shards(tmp_path / "v", rows, k=3)
    out = tmp_path / "lists"
    rc = Q.main(["apply", "--verify", v, "--targets", t, "--gate", g, "--out", str(out),
                 "--subset", "sweep190=" + sweep, "--subset", "stable128=" + stable])
    text = capsys.readouterr().out
    assert rc == 0
    assert "3 of 30 targets quarantined" in text
    assert "MUST-INCLUDE target(s) of sweep190 quarantined: %s" % late_fail in text     # loud flag
    derived = (out / "targets_tl8k.txt").read_text().split()
    assert derived == [k for k in KEYS if k not in (late_fail, late_missing, late_norow)]   # list order kept
    assert (out / "targets_tl8k_sweep190.txt").read_text().split() == [k for k in KEYS[:8] if k != late_fail]
    assert late_missing not in (out / "targets_tl8k_stable128.txt").read_text().split()
    tsv = (out / "quarantine.tsv").read_text().splitlines()
    assert tsv[0] == "key\treason\tsubsets" and len(tsv) == 4
    assert any(l.startswith(late_fail + "\t") and "near" in l and "sweep190" in l for l in tsv)
    summ = json.loads((out / "quarantine.json").read_text())
    assert summ["n_quarantined"] == 3 and summ["must_include_hits"]["sweep190"] == [late_fail]
    assert summ["files"]["targets_tl8k.txt"]["n"] == 27


def test_gate_target_failing_full_verify_is_an_error(tmp_path):
    t = write(tmp_path / "all.txt", KEYS)
    g = write(tmp_path / "gate.txt", GATE)
    v = shards(tmp_path / "v", dict({k: True for k in KEYS}, **{GATE[1]: False}))
    assert Q.main(["apply", "--verify", v, "--targets", t, "--gate", g, "--out", str(tmp_path / "l")]) == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_dry_run_reads_the_derived_lists(tmp_path):
    env = dict(os.environ, DRY_RUN="1", TL_TRACK="1", TL_ROOT="/root/turing-tl", MODELS="pythia-70m",
               RESULTS=str(tmp_path / "results"), CKPT=str(tmp_path / "ckpt"))
    p = subprocess.run(["bash", str(REPO / "experiments" / "082-public-runs" / "chain.sh")], env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    out = p.stdout
    lists = "/root/turing-tl/experiments/082-public-runs/out_turingllm/lists/"
    t2 = out[out.index("turingllm.T2: START"):out.index("turingllm.T3: START")]
    assert "quarantine.py gate" in t2 and "quarantine.py apply" in t2
    assert t2.index("records_gate_verify") < t2.index("records_build on")            # the gate comes first
    assert "targets_tl8k_gate64.txt" in t2
    after = out[out.index("turingllm.T3: START"):]
    for name in ("targets_tl8k.txt", "targets_tl8k_sweep190.txt", "targets_tl8k_stable128.txt",
                 "targets_tl8k_nulls126.txt", "targets_tl8k_case6.txt"):
        assert lists + name in after, name
    assert "/094-tl-v2-records/targets_tl8k" not in after                          # no frozen list after T2
    assert "turingllm.T9: SKIPPED" in out                                           # T9 off by default
