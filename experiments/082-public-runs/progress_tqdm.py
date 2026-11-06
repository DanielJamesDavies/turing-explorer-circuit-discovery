"""tqdm view of the 082 chain across the GPUs (Daniel, 2026-10-26). Read only: chain.events, the stage markers, the
per-shard logs launch.sh writes, the main status rows and T2's quarantine list. Nothing the jobs write is changed.

  python experiments/082-public-runs/progress_tqdm.py              # the pod: redraw every 60 s (Ctrl-C to stop)
  python experiments/082-public-runs/progress_tqdm.py --once       # one frame
  python experiments/082-public-runs/progress_tqdm.py --limit      # the local LIMIT smoke (WORK = 082/smoke_v2)
  options: --interval S (60)  --work DIR  --tl-root DIR (/root/turing-tl)  --models "pythia-70m gpt2-small"
           --gpus N (8)  --no-clear

What it draws (tqdm.format_meter: done/total, rate, ETA):
  GPU 0..N-1   the work of the launches running now (chain.events "<track> <name> on GPUs <list>" not yet matched
               by "all N shards returned" in its launch log), summed over the shards on that GPU: shard i runs on
               GPU list[i % K] (062/launch.sh). Per shard, done / total = the last "<n>/<total>" progress line of its
               log (driver, companions, 094 build / verify, 086 scan "| n/N done |"); driver sweep = rows after the
               "MODE sweep | N targets" header. Rate: the shard log's own timestamps, else the change between frames
  tracks       per track (turingllm, each public model): the current stage (markers) and its running launch, summed;
               plus the main fits (unique seeds in OUT/main/status.shard*.jsonl over the target list)
  jobs         every fitting job with error rows (job_errors.py: LOGS/job_errors.jsonl + the running jobs' live files),
               highlighted in red, an early-gate FAILED job first
  quarantine   T2's quarantined targets (lists/quarantine.tsv: count, the first IDs, must-include flags)
  failures     FAILED lines in chain.events, GIVING UP lines in the shard logs
No pass rate is printed here (the merge / report files give both, post-Top-K *_tk and pre-activation *_pre).
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
E = "experiments/082-public-runs"
D94 = "experiments/094-tl-v2-records"
TL_STAGES = ["T%d" % i for i in range(13)]      # T10-T12: the without-R arm and the donor-background test
PUB_STAGES = {"pythia-70m": "S1 S2 S3 S4 S5 S6".split(), "gpt2-small": "S1 S2 S3 S7 S8 S9".split()}
EVENT = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+(turingllm|pythia-70m|gpt2-small) (.+?) on GPUs ([\d,]+)")
STAMP = re.compile(r"^\[(\d\d):(\d\d)(?::(\d\d))?")
PROG = re.compile(r"^\s*(?:main\s+)?(\d+)/(\d+)\b")
SCAN = re.compile(r"\|\s*(\d+)/(\d+) done \|")
SWEEP_HEAD = re.compile(r"MODE sweep \| (\d+) targets")
SWEEP_ROW = re.compile(r"^\[[^\]]*\]\s+[WU]_\S+\s+\d+\.\w+\.\d+\s")


def paths(a):
    smoke = a.limit or bool(os.environ.get("LIMIT"))
    work = Path(a.work or os.environ.get("WORK") or (E + ("/smoke_v2" if smoke else "")))
    if not work.is_absolute():
        work = ROOT / work
    logs = Path(os.environ.get("LOGS") or work / "logs")
    tl_root = Path(a.tl_root or os.environ.get("TL_ROOT") or (ROOT if smoke else "/root/turing-tl"))
    tl_work = Path(os.environ.get("TL_WORK") or (work / "out_turingllm" if smoke else tl_root / E / "out_turingllm"))
    v2 = Path(os.environ.get("TL_V2_OUT") or (tl_work / "out_v2" if smoke else tl_root / D94 / "out_v2"))
    return dict(smoke=smoke, work=work, logs=logs, state=Path(os.environ.get("STATE") or logs / "state"),
                tl_root=tl_root, tl_work=tl_work, v2=v2, lists=tl_work / "lists")


def lines(p):
    try:
        with open(p, errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def secs(m):
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)


class Rates:
    """done-per-second from the change between frames (fallback when a log has no timestamps)."""

    def __init__(self):
        self.h = {}

    def rate(self, key, done):
        now = time.time()
        old = self.h.get(key)
        if old is None or done < old[1]:
            self.h[key] = (now, done)
            return None
        t0, d0 = old
        if now - t0 > 1800:                                   # keep a recent window
            self.h[key] = (now - 900, d0 + (done - d0) * 0.5)
        return (done - d0) / (now - t0) if now > t0 and done > d0 else None


def shard_progress(path, rates):
    """(done, total, rate) of one shard log; total None when its format is not known."""
    ls = lines(path)
    done = total = None
    pts = []
    sweep_n, sweep_rows, arms = None, 0, 0
    finished = False
    for ln in ls:
        m = SWEEP_HEAD.search(ln)
        if m:
            sweep_n, sweep_rows, arms = int(m.group(1)), 0, 0
            continue
        if sweep_n is not None and re.match(r"^\[[^\]]*\]\s+arm [WU]_", ln):
            arms += 1                                           # one launch may run several arms (the public pilot)
            continue
        if sweep_n is not None and SWEEP_ROW.match(ln):
            sweep_rows += 1
            done, total = sweep_rows, sweep_n * max(1, arms)
            st = STAMP.match(ln)
            if st:
                pts.append((secs(st), done))
            continue
        m = SCAN.search(ln)
        body = re.sub(r"^\[[^\]]*\]", "", ln)
        m = m or PROG.match(body)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            st = STAMP.match(ln)
            if st:
                pts.append((secs(st), done))
            finished = False
        elif re.match(r"^shard \d+ (finished|GIVING UP)", ln):
            finished = True
    if finished and total:
        done = total
    rate = None
    pts = [p for p in pts if total is None or p[1] <= total][-30:]
    if len(pts) >= 2:
        dt = pts[-1][0] - pts[0][0]
        if dt < 0:
            dt += 86400                                         # past midnight
        if dt > 0 and pts[-1][1] > pts[0][1]:
            rate = (pts[-1][1] - pts[0][1]) / dt
    if rate is None and done is not None:
        rate = rates.rate(str(path), done)
    return done, total, rate


def launches(P):
    """[(track, name, gpus, logdir, launch log)] that are running now, latest first per track."""
    evs, last_at, latest = defaultdict(list), {}, {}
    finished_at = -1
    for i, ln in enumerate(lines(P["logs"] / "chain.events")):
        m = EVENT.match(ln)
        if m:
            evs[(m.group(2), m.group(3))].append([int(g) for g in m.group(4).split(",") if g])
            last_at[(m.group(2), m.group(3))] = i
            latest[m.group(2)] = (m.group(2), m.group(3))
        elif "chain finished" in ln:
            finished_at = i
    out = []
    for (track, name), lst in evs.items():
        raw = name
        if last_at[(track, raw)] < finished_at:               # the chain ended after this launch began
            continue
        if name.startswith("dan159 "):                        # 083 run_pod.sh: logs in TL_WORK/dan159/logs/<stage>.*
            stage = name.split()[1]
            logdir, launch_log, pat = P["tl_work"] / "dan159" / "logs", P["logs"] / "turingllm" / ("dan159_%s.log" % stage), \
                "%s.shard*.log" % stage
        else:
            name = name.split(",")[0].split(" (")[0].strip()
            logdir = P["logs"] / track / name
            launch_log, pat = P["logs"] / track / ("%s.launch.log" % name), "*.shard*.log"
        returned = sum(1 for ln in lines(launch_log) if re.search(r": all \d+ shards returned", ln))
        if len(lst) <= returned:
            continue
        logs_ = sorted(glob.glob(str(logdir / pat)))
        superseded = latest.get(track) != (track, raw)
        if not logs_ and superseded:
            continue                                          # no shard logs (e.g. 083's table) and a later launch began
        ends = [next((l for l in reversed(lines(f)) if l.strip()), "") for f in logs_]
        if logs_ and superseded and all(re.match(r"^shard \d+ (finished|GIVING UP)", e) for e in ends):
            continue                                          # every shard ended and a later launch began
        out.append((track, name, lst[-1], logdir, pat))
    return out


def stage_of(P, track):
    st = TL_STAGES if track == "turingllm" else PUB_STAGES.get(track, [])
    key = (lambda s: "turingllm.%s" % s) if track == "turingllm" else (lambda s: "%s.%s" % (track, s))
    cur, last_done = None, None
    for s in st:
        if (P["state"] / ("%s.done" % key(s))).exists():
            last_done = s
        elif (P["state"] / ("%s.start" % key(s))).exists():
            cur = s
    return cur or (("after " + last_done) if last_done else "not started")


def main_rows(out_dir):
    seen = set()
    for p in glob.glob(str(out_dir / "main" / "status.shard*.jsonl")):
        for ln in lines(p):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if "error" not in r and "seed" in r:
                seen.add(r["seed"])
    return seen


def job_errors(P):
    """{job: record}: the chain's per-job error accounting (job_errors.py): the final records in LOGS/job_errors.jsonl,
    overridden by a running job's live file (STATE/jobs/<job>.live.json) when that is newer."""
    out = {}
    for ln in lines(P["logs"] / "job_errors.jsonl"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        out[r.get("job")] = dict(r, state="failed" if r.get("failed") else "done")
    for f in glob.glob(str(P["state"] / "jobs" / "*.live.json")):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        old = out.get(r.get("job"))
        if old is None or str(r.get("time", "")) > str(old.get("time", "")):
            out[r.get("job")] = r
    return out


def n_list(p):
    return len([t for t in " ".join(lines(p)).split() if t]) if Path(p).exists() else None


def meter(desc, done, total, rate, ncols):
    done = done or 0
    elapsed = done / rate if rate else 0.0
    return tqdm.format_meter(done, total or None, elapsed, ncols=ncols, prefix=desc.ljust(34)[:34], rate=rate,
                             unit="tg")


def frame(P, a, rates):
    ncols = min(150, max(90, os.get_terminal_size(sys.stdout.fileno()).columns if sys.stdout.isatty() else 140))
    out = ["== 082 chain | %s | %s%s" % (time.strftime("%F %T"), P["work"], " | SMOKE" if P["smoke"] else "")]
    running = launches(P)
    per_gpu = defaultdict(lambda: [0, 0, 0.0, set(), False])       # done, total, rate, labels, unknown total
    per_launch = {}
    for track, name, gpus, logdir, pat in running:
        k = len(gpus) or 1
        agg = [0, 0, 0.0, False]
        for f in sorted(glob.glob(str(logdir / pat))):
            m = re.search(r"\.shard(\d+)\.log$", f)
            if not m:
                continue
            d, t, r = shard_progress(f, rates)
            g = gpus[int(m.group(1)) % k] if gpus else None
            slot = per_gpu[g]
            slot[3].add("%s:%s" % ("TL" if track == "turingllm" else track, name))
            slot[0] += d or 0; slot[2] += r or 0.0
            if t is None:
                slot[4] = True; agg[3] = True
            else:
                slot[1] += t; agg[1] += t
            agg[0] += d or 0; agg[2] += r or 0.0
        per_launch[track] = (name, agg)
    out.append("-- GPUs (running launches; shard i -> GPU list[i % K])")
    for g in range(a.gpus):
        if g in per_gpu:
            d, t, r, labels, unk = per_gpu[g]
            out.append(meter("GPU %d %s" % (g, "+".join(sorted(labels))), d, None if unk and not t else t, r or None,
                             ncols))
        else:
            note = "idle"
            for mdl, dg in (("pythia-70m", 0), ("gpt2-small", 1)):
                if g == int(os.environ.get("DATA_GPU_%s" % mdl.replace("-", "_"), dg)) and \
                        (P["state"] / ("%s.S3.start" % mdl)).exists() and not (P["state"] / ("%s.S3.done" % mdl)).exists():
                    tail = (lines(P["logs"] / mdl / "dp_pass1.log") or [""])[-1].replace("\r", " ")[-70:]
                    note = "%s data pass: %s" % (mdl, tail)
            out.append("GPU %d  %s" % (g, note))
    out.append("-- tracks")
    models = a.models.split()
    for track in ["turingllm"] + models:
        stg = stage_of(P, track)
        if track in per_launch:
            name, (d, t, r, unk) = per_launch[track]
            out.append(meter("%s %s %s" % (track, stg, name), d, t or None, r or None, ncols))
        else:
            out.append("%s %s (no launch running)" % (track.ljust(34)[:34], stg))
        if track == "turingllm":
            lst = P["lists"] / "targets_tl8k.txt"
            if not lst.exists():
                lst = (P["tl_work"] / "targets_smoke.txt") if P["smoke"] else (P["tl_root"] / D94 / "targets_tl8k.txt")
            done, total = len(main_rows(P["v2"])), n_list(lst)
            out.append(meter("TuringLLM 8k main fits", done, total, rates.rate("tl.main", done), ncols))
        else:
            lst = P["work"] / ("targets_%s_main%s.txt" % (track, ".limit" if P["smoke"] else ""))
            od = P["work"] / ("out_%s" % track)
            if (od / "main").exists():
                done = len(main_rows(od))
                out.append(meter("%s main fits" % track, done, n_list(lst), rates.rate(track + ".main", done), ncols))
    jobs = job_errors(P)
    bad = [j for j in jobs.values() if j.get("n_err")]
    red = (lambda s: "\033[1;31m%s\033[0m" % s) if (sys.stdout.isatty() and not a.no_color) else (lambda s: s)
    out.append("-- jobs: %d checked, %d with error rows%s" % (len(jobs), len(bad), "" if bad else " (none)"))
    for j in sorted(bad, key=lambda j: (not j.get("failed"), j.get("job", ""))):
        out.append(red("   !! %s: %d of %d fits errored%s [%s]%s" % (
            j.get("job"), j.get("n_err", 0), j.get("n", 0), " -- FAILED (early gate)" if j.get("failed") else "",
            j.get("state", "done"), (" first: %s" % str(j.get("first_error"))[:110]) if j.get("first_error") else "")))
    q = P["lists"] / "quarantine.tsv"
    qrows = [ln.split("\t") for ln in lines(q)[1:] if ln.strip()]
    if q.exists():
        must = [r[0] for r in qrows if len(r) > 2 and any(s in r[2] for s in ("sweep190", "nulls126", "case6", "gallery9"))]
        out.append("-- quarantined (T2): %d%s%s" % (len(qrows), (": " + " ".join(r[0] for r in qrows[:8]) +
                                                                (" ..." if len(qrows) > 8 else "")) if qrows else "",
                                                   ("  | MUST-INCLUDE: " + " ".join(must)) if must else ""))
    else:
        out.append("-- quarantined (T2): (no quarantine list yet)")
    ev = lines(P["logs"] / "chain.events")
    failed = [ln for ln in ev if "FAILED" in ln]
    gu = 0
    for f in glob.glob(str(P["logs"] / "**" / "*.shard*.log"), recursive=True) + \
            glob.glob(str(P["tl_work"] / "dan159" / "logs" / "*.shard*.log")):
        gu += sum("GIVING UP" in ln for ln in lines(f))
    out.append("-- failures: FAILED %d%s | GIVING UP %d" % (len(failed), (" (last: %s)" % failed[-1][11:120]) if failed
                                                            else "", gu))
    if ev:
        out.append("-- last event: %s" % ev[-1][:140])
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--limit", action="store_true", help="the local LIMIT smoke layout (WORK = 082/smoke_v2)")
    ap.add_argument("--work")
    ap.add_argument("--tl-root")
    ap.add_argument("--models", default=os.environ.get("MODELS", "pythia-70m gpt2-small"))
    ap.add_argument("--gpus", type=int, default=int(os.environ.get("N_GPUS", "8")))
    ap.add_argument("--no-clear", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args(argv)
    P = paths(a)
    rates = Rates()
    while True:
        txt = frame(P, a, rates)
        if not a.once and not a.no_clear:
            sys.stdout.write("\033[2J\033[H")
        print(txt, flush=True)
        if a.once:
            return 0
        try:
            time.sleep(a.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
