"""Per-job error accounting for the 082 chain (Daniel, 2026-10-26; the records gate's philosophy, for every fitting job).

A JOB is one launch of the chain (a stage, sub-stage, main chunk or sweep arm). Its ROWS are the jsonl rows its
shards append to the given files during the launch (a snapshot of the line counts before the launch tells new rows
from old ones; a file that shrank, i.e. was rewritten, counts from its start). A row is an ERROR row when it has an
"error" field (driver main status / sweep eval rows, companions, 083, 084, 094 build rows all write one per failed
fit). Rows are taken in "fit order": row j of every file (sorted names, i.e. every shard) before row j + 1.

  EARLY GATE  the first GATE (16) rows of the job, or all of them when the job wrote fewer: if they ALL error, the job
              FAILS (exit 1). A launch is watched while it runs (watch): once GATE rows exist and all error, its
              process group is stopped, so a code bug costs minutes on the pod
  otherwise   errors stop nothing: the count and the first error message are reported (exit 0)

  python job_errors.py snapshot --rows "GLOB[,GLOB...]" --snapshot FILE
  python job_errors.py watch    --rows ... --snapshot FILE --pid PID --name JOB --live FILE [--gate 16] [--every 30]
  python job_errors.py check    --rows ... --snapshot FILE --name JOB [--gate 16] [--jsonl FILE] [--live FILE]
Output of check (one line, for chain.events): "JOB <name>: <n> rows, <e> errors[ (first: ...)]" or
"JOB <name>: FAILED, the first <k> fits all errored (first: ...)". --jsonl appends the record that progress_tqdm.py and
progress.sh read (job, n, n_err, first_error, failed, time).
"""
import argparse
import glob
import json
import os
import signal
import sys
import time


def files_of(rows):
    out = []
    for g in [x for x in rows.split(",") if x]:
        out += sorted(glob.glob(g))
    return list(dict.fromkeys(out))


def count_lines(p):
    try:
        with open(p, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def snapshot(rows, path):
    snap = {f: count_lines(f) for f in files_of(rows)}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(snap, fh)
    os.replace(path + ".tmp", path)
    return snap


def new_rows(rows, snap_path):
    """Rows appended since the snapshot, in fit order (row j of every file before row j + 1)."""
    try:
        snap = json.load(open(snap_path))
    except (OSError, ValueError):
        snap = {}
    per = []
    for f in files_of(rows):
        try:
            ls = open(f, errors="replace").read().splitlines()
        except OSError:
            continue
        start = snap.get(f, 0)
        if start > len(ls):                                  # rewritten ("w" mode): every row is new
            start = 0
        rs = []
        for ln in ls[start:]:
            try:
                rs.append(json.loads(ln))
            except ValueError:
                continue                                     # a truncated last line
        per.append(rs)
    out, j = [], 0
    while any(j < len(r) for r in per):
        out += [r[j] for r in per if j < len(r)]
        j += 1
    return out


def summary(rows, snap_path, gate):
    rs = new_rows(rows, snap_path)
    err = [r for r in rs if r.get("error")]
    head = rs[:gate]
    failed = bool(head) and all(r.get("error") for r in head)
    first = str(err[0]["error"])[:240] if err else None
    return dict(n=len(rs), n_err=len(err), first_error=first, failed=failed, gate=gate,
                complete_gate=len(rs) >= gate)


def line(name, s):
    if s["failed"]:
        return "JOB %s: FAILED, the first %d fits all errored (first: %s)" % (name, min(s["n"], s["gate"]),
                                                                              s["first_error"])
    return "JOB %s: %d rows, %d errors%s" % (name, s["n"], s["n_err"],
                                            (" (first: %s)" % s["first_error"]) if s["n_err"] else "")


def write_live(path, name, s, state):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(dict(s, job=name, state=state, time=time.strftime("%F %T")), fh)
    os.replace(path + ".tmp", path)


def stop_group(pid):
    """Stop the launch: its process group when it leads one (the chain starts it under setsid), else the process and
    its descendants (/proc). Never the caller's own group."""
    try:
        pg = os.getpgid(pid)
    except OSError:
        return
    if pg == pid and pg != os.getpgid(0):
        os.killpg(pg, signal.SIGTERM)
        return
    kids, todo = [], [pid]
    while todo:
        p = todo.pop()
        kids.append(p)
        for d in glob.glob("/proc/[0-9]*/stat"):
            try:
                f = open(d).read().split(")")[-1].split()
                if int(f[1]) == p:
                    todo.append(int(d.split("/")[2]))
            except (OSError, ValueError, IndexError):
                continue
    for p in reversed(kids):
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("snapshot", "watch", "check"))
    ap.add_argument("--rows", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--name", default="job")
    ap.add_argument("--gate", type=int, default=16)
    ap.add_argument("--pid", type=int)
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--jsonl")
    ap.add_argument("--live")
    a = ap.parse_args(argv)
    if a.mode == "snapshot":
        snapshot(a.rows, a.snapshot)
        return 0
    if a.mode == "watch":
        while a.pid and alive(a.pid):
            s = summary(a.rows, a.snapshot, a.gate)
            write_live(a.live, a.name, s, "running")
            if s["complete_gate"] and s["failed"]:
                print(line(a.name, s) + " -- stopping the launch", flush=True)
                write_live(a.live, a.name, s, "stopped")
                stop_group(a.pid)
                return 1
            time.sleep(a.every)
        return 0
    s = summary(a.rows, a.snapshot, a.gate)
    print(line(a.name, s), flush=True)
    write_live(a.live, a.name, s, "failed" if s["failed"] else "done")
    if a.jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(a.jsonl)), exist_ok=True)
        with open(a.jsonl, "a") as fh:
            fh.write(json.dumps(dict(s, job=a.name, time=time.strftime("%F %T"))) + "\n")
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
