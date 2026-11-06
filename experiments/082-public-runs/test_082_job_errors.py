"""CPU tests of the per-job error gate (2026-10-26): job_errors.py (snapshot / check / watch) and chain.sh's run_job
(the real function text, extracted from chain.sh and run in bash on fake jobs that write rows).

  CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest experiments/082-public-runs/test_082_job_errors.py -q
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("e082_job_errors", HERE / "job_errors.py")
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)
ERR = "ValueError: train_fill='sparse' ... needs free_amplitude=True"


def rows(path, flags):
    """Append one row per flag (True = error row)."""
    with open(path, "a") as fh:
        for i, e in enumerate(flags):
            fh.write(json.dumps(dict(seed="k%d" % i, **({"error": ERR} if e else {"ok": True}))) + "\n")


def test_all_error_job_fails(tmp_path, capsys):
    g = str(tmp_path / "eval.shard*.jsonl")
    snap = str(tmp_path / "s.snap")
    rows(tmp_path / "eval.shard0.jsonl", [False] * 5)                  # rows of an EARLIER job (the snapshot hides them)
    J.main(["snapshot", "--rows", g, "--snapshot", snap])
    rows(tmp_path / "eval.shard0.jsonl", [True, True])                 # fewer than 16 fits: all of them count
    rows(tmp_path / "eval.shard1.jsonl", [True])
    assert J.main(["check", "--rows", g, "--snapshot", snap, "--name", "pythia-70m.unweighted",
                   "--jsonl", str(tmp_path / "j.jsonl")]) == 1
    out = capsys.readouterr().out
    assert "JOB pythia-70m.unweighted: FAILED, the first 3 fits all errored (first: ValueError" in out
    rec = json.loads((tmp_path / "j.jsonl").read_text())
    assert rec["failed"] and rec["n"] == 3 and rec["n_err"] == 3


def test_first_16_all_error_fails_even_if_later_rows_pass(tmp_path):
    g = str(tmp_path / "r.shard*.jsonl")
    snap = str(tmp_path / "s.snap")
    J.main(["snapshot", "--rows", g, "--snapshot", snap])
    rows(tmp_path / "r.shard0.jsonl", [True] * 8 + [False] * 4)
    rows(tmp_path / "r.shard1.jsonl", [True] * 8 + [False] * 4)        # fit order: row j of every shard first
    assert J.summary(g, snap, 16)["failed"]


def test_partial_errors_continue_and_are_counted(tmp_path, capsys):
    g = str(tmp_path / "status.shard*.jsonl")
    snap = str(tmp_path / "s.snap")
    J.main(["snapshot", "--rows", g, "--snapshot", snap])
    rows(tmp_path / "status.shard0.jsonl", [True, False, True, False, False])
    assert J.main(["check", "--rows", g, "--snapshot", snap, "--name", "turingllm.main_c0"]) == 0
    assert "JOB turingllm.main_c0: 5 rows, 2 errors (first: ValueError" in capsys.readouterr().out


def test_rewritten_file_counts_from_start(tmp_path):
    p = tmp_path / "v.jsonl"
    rows(p, [False] * 10)
    snap = str(tmp_path / "s.snap")
    J.main(["snapshot", "--rows", str(p), "--snapshot", snap])
    p.write_text("")
    rows(p, [True, True])
    s = J.summary(str(p), snap, 16)
    assert s["n"] == 2 and s["failed"]


def test_watch_stops_an_all_error_launch(tmp_path):
    g = str(tmp_path / "eval.shard*.jsonl")
    snap = str(tmp_path / "s.snap")
    J.main(["snapshot", "--rows", g, "--snapshot", snap])
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    rows(tmp_path / "eval.shard0.jsonl", [True] * 16)
    t0 = time.time()
    rc = J.main(["watch", "--rows", g, "--snapshot", snap, "--pid", str(job.pid), "--name", "x", "--every", "0.2",
                 "--live", str(tmp_path / "x.live.json")])
    assert rc == 1 and job.wait(timeout=20) != 0 and time.time() - t0 < 20
    assert json.loads((tmp_path / "x.live.json").read_text())["state"] == "stopped"


# ----------------------------------------------------------------------------------------------- chain.sh run_job
def chain_functions():
    """say / run / split_rows / run_job, verbatim from chain.sh."""
    src = (HERE / "chain.sh").read_text()
    out = []
    for name in ("say", "run", "split_rows", "run_job"):
        m = re.search(r"^%s\(\) \{.*?^\}$" % name, src, re.S | re.M)
        assert m, name
        out.append(m.group(0))
    return "\n".join(out)


def bash_job(tmp_path, flags, tail_sleep):
    """run_job on a fake launch that appends rows (one per 0.1 s) then sleeps; returns (rc, events, seconds)."""
    fake = tmp_path / "fake.sh"
    lines = "".join("echo '%s' >> %s; sleep 0.1\n" % (json.dumps(dict(seed="k%d" % i, **({"error": ERR} if e else
                                                                                              {"ok": True}))),
                                                        tmp_path / "out" / "eval.shard0.jsonl")
                    for i, e in enumerate(flags))
    fake.write_text("mkdir -p %s\n%ssleep %s\n" % (tmp_path / "out", lines, tail_sleep))
    script = "\n".join(["set -uo pipefail", 'DRY_RUN=0 LOGS="%s" STATE="%s" E="%s" PY="%s" JOB_GATE=16 JOB_WATCH_EVERY=0.3'
                        % (tmp_path / "logs", tmp_path / "logs/state", HERE, sys.executable), chain_functions(),
                        'split_rows A=1 ROWS="%s/out/eval.shard*.jsonl" B=2' % tmp_path,
                        '[ "$JOB_ROWS" = "%s/out/eval.shard*.jsonl" ] && [ "${JOB_ARGS[*]}" = "A=1 B=2" ] || exit 9' % tmp_path,
                        'run_job "%s" pythia-70m.unweighted "$JOB_ROWS" bash "%s"' % (tmp_path / "logs/u.launch.log", fake),
                        'echo "rc=$?"'])
    t0 = time.time()
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)
    rc = int(re.search(r"rc=(\d+)", p.stdout).group(1)) if "rc=" in p.stdout else p.returncode
    ev = (tmp_path / "logs" / "chain.events").read_text() if (tmp_path / "logs" / "chain.events").exists() else ""
    return rc, ev, time.time() - t0, p


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("setsid") is None, reason="needs bash + setsid")
def test_run_job_all_error_fails_fast(tmp_path):
    rc, ev, secs, p = bash_job(tmp_path, [True] * 20, 60)
    assert rc == 1, p.stdout + p.stderr
    assert "JOB pythia-70m.unweighted: FAILED, the first 16 fits all errored (first: ValueError" in ev
    assert secs < 30                                                   # stopped by the watcher, not after the sleep 60
    rec = [json.loads(l) for l in (tmp_path / "logs" / "job_errors.jsonl").read_text().splitlines()]
    assert rec[-1]["failed"]


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("setsid") is None, reason="needs bash + setsid")
def test_run_job_partial_errors_continue_logged(tmp_path):
    rc, ev, secs, p = bash_job(tmp_path, [False, True, False, False, True] + [False] * 15, 0)
    assert rc == 0, p.stdout + p.stderr
    assert "JOB pythia-70m.unweighted: 20 rows, 2 errors (first: ValueError" in ev
    assert "2 errors" in (tmp_path / "logs" / "u.launch.log").read_text()
