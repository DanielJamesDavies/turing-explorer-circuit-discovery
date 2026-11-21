"""Exp 128: the morning checks of the arena A smoke (run_smoke_local.sh -> run_pod.sh from P4) against the 2026-11-17
run_pod.sh changes (P4x / P6x after P6, AA_DF_PAR, ORDER=interleaved, a2_skipped_L0.txt, the lambda grid).

  python -I check_smoke.py --work <SMOKE_WORK> --lambdas 1e-3,1.6e-2 --ext-branches 100 --ext-k 1 [--out smoke_check.md]

Checks (each PASS / FAIL with its evidence):
  stages      every run_pod.sh stage P4 P5 P6 P4x P6x P4h P6b P7 has a steps.tsv row, the LAST row of each has rc 0
  wcm         an ok status row for every target in every a1 / a2 x lambda arm; the driver logged "order interleaved"
  p4x         Dunefsky extended-knob circuits circuits/<tc key>/p<j>.b<ext>.i5.pt for every tc key, j < ext_k
  p6x         score/a2.s0/rows.jsonl holds rows of the extended units (uid ...|b<ext>|...); logs/P6x exist
  ge_rec_x    score/all/ge_recovery.x.s*.jsonl written, >= 1 row, 0 error rows
  df_par      the P4 / P4x Dunefsky logs <gpu log>.df<s>: how many processes ran and how many had work (>= 2 each;
              df_par starts them in the background and waits for all, so they run in parallel by construction)
  summary     score/all/summary.md and perprompt/all/summary.md exist and are not empty
  l0          a2_skipped_L0.txt exists (empty when no layer-0 tc key is in the run)
Exit 0 all PASS, 1 any FAIL.
"""
import argparse
import json
import re
from pathlib import Path

STAGES = ("P4", "P5", "P6", "P4x", "P6x", "P4h", "P6b", "P7")


def lines(p):
    return [x for x in p.read_text(errors="replace").splitlines() if x.strip()] if p.exists() else []


def jl(p):
    out = []
    for x in lines(p):
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--lambdas", default="1e-3,1.6e-2")
    ap.add_argument("--ext-branches", default="100")
    ap.add_argument("--ext-k", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    W = Path(a.work)
    res = []

    def check(name, ok, ev):
        res.append((name, bool(ok), ev))

    # stages
    last = {}
    for x in lines(W / "steps.tsv"):
        p = x.split("\t")
        if len(p) >= 4:
            last[p[0]] = (int(p[2]) - int(p[1]), p[3])
    miss = [s for s in STAGES if s not in last]
    bad = {s: v for s, v in last.items() if v[1] != "0"}
    check("stages", not miss and not bad, "last rows: %s%s%s" % (
        ", ".join("%s rc %s (%d s)" % (s, last[s][1], last[s][0]) for s in STAGES if s in last),
        "; MISSING %s" % " ".join(miss) if miss else "", "; FAILED %s" % bad if bad else ""))
    keys = lines(W / "targets.keys.txt")
    tc = lines(W / "targets_tc.keys.txt")
    # wcm
    ev, ok = [], True
    for scope, lst in (("a1", keys), ("a2", tc)):
        for lam in a.lambdas.split(","):
            rows = []
            for f in sorted((W / "wcm" / ("%s_l%s" % (scope, lam)) / "main").glob("status.shard*.jsonl")):
                rows += jl(f)
            got = {r.get("seed") for r in rows if r.get("ok") or r.get("skip")}
            n = len(set(lst) & got)
            ok &= n >= len(lst)
            ev.append("%s_l%s %d/%d" % (scope, lam, n, len(lst)))
    inter = [f for f in (W / "logs" / "P5").rglob("main.shard*.log") if "order interleaved" in f.read_text(errors="replace")]
    ok &= bool(inter)
    check("wcm", ok, "; ".join(ev) + "; 'order interleaved' in %d shard log(s)" % len(inter))
    # p4x
    want = [(k, j, b) for k in tc for j in range(a.ext_k) for b in a.ext_branches.split(",")]
    have = [w for w in want if (W / "dunefsky" / "circuits" / w[0] / ("p%d.b%s.i5.pt" % (w[1], w[2]))).exists()]
    check("p4x", want and len(have) == len(want), "%d / %d extended circuits (%s)" % (
        len(have), len(want), ", ".join("%s p%d b%s" % w for w in want)))
    # p6x
    rows = []
    for f in sorted((W / "score").glob("a2.s*/rows.jsonl")):
        rows += jl(f)
    ext = [r for r in rows if r.get("method") == "dunefsky" and any("|b%s|" % b in str(r.get("uid")) for b in a.ext_branches.split(","))]
    p6x_logs = sorted(str(p.relative_to(W)) for p in (W / "logs" / "P6x").glob("*.log")) if (W / "logs" / "P6x").is_dir() else []
    check("p6x", ext and p6x_logs, "%d score rows of extended units (%d units); logs %s" % (
        len(ext), len({r.get("uid") for r in ext}), p6x_logs))
    # ge recovery x
    gx = sorted((W / "score" / "all").glob("ge_recovery.x.s*.jsonl"))
    gr = [r for f in gx for r in jl(f)]
    ne = sum(1 for r in gr if r.get("status") == "error")
    newj = lines(W / "score" / "jobs_x" / "ge_jobs.new.jsonl")
    check("ge_rec_x", gx and gr and ne == 0, "%s: %d rows, %d errors (jobs_x/ge_jobs.new.jsonl: %d jobs)" % (
        [p.name for p in gx], len(gr), ne, len(newj)))
    # df_par
    ev, ok = [], True
    for stage in ("P4", "P4x"):
        logs = sorted((W / "logs" / stage).glob("gpu*.log.df*"))
        work, spans = 0, []
        for f in logs:
            txt = f.read_text(errors="replace")
            m = re.findall(r"dunefsky: (\d+) tc targets, (\d+) circuits to run", txt)
            n_c = sum(int(c) for _, c in m)
            work += n_c > 0
            spans.append("%s: %s" % (f.name, ", ".join("%s targets / %s circuits" % x for x in m) or "no start line"))
        ok &= len(logs) >= 2 and work >= 2
        ev.append("%s: %d processes, %d with work [%s]" % (stage, len(logs), work, "; ".join(spans)))
    check("df_par", ok, " | ".join(ev))
    # summaries
    s1, s2 = W / "score" / "all" / "summary.md", W / "perprompt" / "all" / "summary.md"
    check("summary", s1.exists() and s1.stat().st_size > 0 and s2.exists() and s2.stat().st_size > 0,
          "%s %s B, %s %s B" % (s1.name, s1.stat().st_size if s1.exists() else "MISSING",
                                "perprompt/" + s2.name, s2.stat().st_size if s2.exists() else "MISSING"))
    l0 = W / "a2_skipped_L0.txt"
    check("l0", l0.exists(), "a2_skipped_L0.txt %s (%d keys)" % ("exists" if l0.exists() else "MISSING", len(lines(l0))))
    L = ["# Exp 128 smoke check: %s" % W, ""] + ["- **%s** %s: %s" % ("PASS" if o else "FAIL", n, e) for n, o, e in res]
    txt = "\n".join(L) + "\n"
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    raise SystemExit(0 if all(o for _, o, _ in res) else 1)


if __name__ == "__main__":
    main()
