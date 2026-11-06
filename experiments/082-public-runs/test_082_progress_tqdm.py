"""CPU test of progress_tqdm.py on a synthetic chain layout (the LIMIT smoke's paths): per-GPU bars from the shard
logs via the launcher's shard -> GPU mapping, the track bars, the quarantine line, FAILED / GIVING UP counts.

  CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest experiments/082-public-runs/test_082_progress_tqdm.py -q
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("e082_progress_tqdm", HERE / "progress_tqdm.py")
PT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PT)


def layout(tmp):
    work = tmp / "smoke_v2"
    logs = work / "logs"
    state = logs / "state"
    state.mkdir(parents=True)
    for k in ("turingllm.T0", "turingllm.T1", "turingllm.T2"):
        (state / (k + ".start")).write_text("x"); (state / (k + ".done")).write_text("x")
    (state / "turingllm.T3.start").write_text("x")
    (state / "pythia-70m.S6.start").write_text("x")
    (logs / "chain.events").write_text(
        "[10:00:00] turingllm.T2: START\n"
        "[10:00:01]   turingllm records_build on GPUs 2,3\n"
        "[10:20:00]   turingllm.T2: FAILED (rc 1)\n"
        "[10:30:00]   turingllm main_c0 on GPUs 2,3\n"
        "[10:31:00]   pythia-70m main on GPUs 4 (P=1)\n")
    tl = logs / "turingllm"
    (tl / "main_c0").mkdir(parents=True)
    (tl / "records_build.launch.log").write_text("build: all 2 shards returned (2 GPUs [2 3] x 1 per GPU)\n")
    (tl / "records_build").mkdir()
    (tl / "records_build" / "build.shard0.log").write_text("[10:01:00 fresh shard0]    1/4 a ok\n")   # finished launch
    (tl / "main_c0.launch.log").write_text("")
    # 4 shards on GPUs [2, 3]: shard i -> GPU [2, 3][i % 2]
    for i, (d, t) in enumerate([(10, 100), (20, 100), (30, 100), (40, 100)]):
        body = "".join("[10:%02d:00 shard%d]  %3d/%d 3.mlp.%d ok n=12\n" % (30 + j, i, j * d // 5, t, j)
                       for j in range(1, 6))
        (tl / "main_c0" / ("main.shard%d.log" % i)).write_text(body + ("shard %d GIVING UP\n" % i if i == 3 else ""))
    (logs / "pythia-70m" / "main").mkdir(parents=True)
    (logs / "pythia-70m" / "main" / "main.shard0.log").write_text(
        "[10:31:00 shard0] MODE sweep | 5 targets | order\n[10:31:01 shard0] arm W_0.001 (weighted=True)\n"
        "[10:32:00 shard0] W_0.001  0.mlp.1  ok n=3\n[10:33:00 shard0] W_0.001  0.mlp.2  ok n=4\n")
    v2 = work / "out_turingllm" / "out_v2" / "main"
    v2.mkdir(parents=True)
    (v2 / "status.shard0.jsonl").write_text("".join(json.dumps(dict(seed="3.mlp.%d" % j, ok=True)) + "\n"
                                                    for j in range(7)) + json.dumps(dict(seed="x", error="e")) + "\n")
    lists = work / "out_turingllm" / "lists"
    lists.mkdir()
    (lists / "targets_tl8k.txt").write_text("".join("3.mlp.%d\n" % j for j in range(50)))
    (lists / "quarantine.tsv").write_text("key\treason\tsubsets\n9.resid.1\tnear\tsweep190,gate64\n2.attn.5\tmissing\t\n")
    # per-job error accounting (job_errors.py): a failed job, a clean one, a running job with partial errors
    (logs / "job_errors.jsonl").write_text(
        json.dumps(dict(job="turingllm.sweep_U_0.001", n=2, n_err=2, failed=True, first_error="ValueError: boom",
                        time="2026-10-26 10:20:00")) + "\n" +
        json.dumps(dict(job="turingllm.sweep_W_0.001", n=2, n_err=0, failed=False, first_error=None,
                        time="2026-10-26 10:21:00")) + "\n")
    (state / "jobs").mkdir()
    (state / "jobs" / "turingllm.main_c0.live.json").write_text(json.dumps(dict(
        job="turingllm.main_c0", n=40, n_err=3, failed=False, first_error="RuntimeError: CUDA OOM", state="running",
        time="2026-10-26 10:40:00")))
    return work


def test_frame_on_synthetic_layout(tmp_path, capsys):
    work = layout(tmp_path)
    assert PT.main(["--once", "--limit", "--work", str(work), "--tl-root", str(tmp_path), "--models", "pythia-70m"]) == 0
    out = capsys.readouterr().out
    gl = {int(l.split()[1]): l for l in out.splitlines() if l.startswith("GPU ")}
    assert "TL:main_c0" in gl[2] and "40/200" in gl[2]          # shards 0 + 2: last lines 10/100 + 30/100
    assert "TL:main_c0" in gl[3] and "120/200" in gl[3]         # shards 1 + 3: 20 + 100 (shard 3 ended: GIVING UP)
    assert "pythia-70m:main" in gl[4] and "2/5" in gl[4]        # driver sweep rows after the header
    assert "idle" in gl[0] and "records_build" not in out.split("-- tracks")[0]   # a returned launch is not running
    assert "TuringLLM 8k main fits" in out and "7/50" in out
    assert "turingllm T3 main_c0" in out
    assert "quarantined (T2): 2: 9.resid.1 2.attn.5" in out and "MUST-INCLUDE: 9.resid.1" in out
    assert "FAILED 1" in out and "GIVING UP 1" in out
    assert "-- jobs: 3 checked, 2 with error rows" in out
    jl = [l for l in out.splitlines() if l.strip().startswith("!!")]
    assert "turingllm.sweep_U_0.001: 2 of 2 fits errored -- FAILED (early gate) [failed]" in jl[0]     # failed first
    assert "turingllm.main_c0: 3 of 40 fits errored [running] first: RuntimeError" in jl[1]
    assert not any("sweep_W_0.001" in l for l in jl)
    assert "it/s" not in out or "tg/s" in out or "s/tg" in out                   # tqdm's own rate field, unit tg


def test_progress_sh_lists_jobs_with_errors(tmp_path):
    import os
    import shutil
    import subprocess
    if shutil.which("bash") is None:
        return
    work = layout(tmp_path)
    p = subprocess.run(["bash", str(HERE / "progress.sh")], env=dict(os.environ, LIMIT="2", WORK=str(work)),
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    assert "-- jobs: 3 checked, 2 with error rows" in p.stdout
    assert "!! turingllm.sweep_U_0.001: 2 of 2 fits errored -- FAILED (early gate)" in p.stdout


def test_empty_layout_does_not_crash(tmp_path, capsys):
    assert PT.main(["--once", "--limit", "--work", str(tmp_path / "none"), "--tl-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "GPU 7  idle" in out and "no quarantine list yet" in out
