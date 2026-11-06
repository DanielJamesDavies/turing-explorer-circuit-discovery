"""Arena D pod stage (plan D8), P5: one summary of pod_stage_d.sh's WORK dir -> WORK/SUMMARY.md + WORK/summary.json.
CPU, repo .venv, reads files only (an empty or partial WORK dir is fine: what is missing is reported as missing).

  ./.venv/bin/python -B pod_summary_d.py WORK [--lams 3e-6,...] [--tasks simple,...] [--ag-k 40] [--out DIR]

Sections: gate (halves, fidelity verdicts, the weights / means cross-checks, HF refs), jobs (Caples grids, D2-WCM and
D1-WCM rows, AG prompts), scoring (score_d.py rows), timings (cmds.tsv summed per log = the D10 re-cost input).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def jl(path: Path):
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def jload(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def f3(x):
    return "-" if x is None else ("%.3f" % x if isinstance(x, (int, float)) else str(x))


def summarise(W: Path, lams, tasks, ag_k) -> dict:
    st = W / "state"
    S = dict(work=str(W), gate={}, jobs={}, scoring={}, timings={})
    g = S["gate"]
    for h in ("d2", "d1"):
        g["gate_" + h] = "PASS" if (st / ("gate_%s.ok" % h)).exists() else ("FAIL" if (st / ("gate_%s.fail" % h)).exists()
                                                                          else "not run")
    fid = W / "fidelity"
    for name in ("caples_A", "caples_fidelity"):
        j = jload(fid / ("%s.json" % name))
        g[name] = None if j is None else dict(passed=j.get("passed"), grids={k: v.get("passed") for k, v in
                                                                                 j.get("grids", {}).items()})
    for name in ("ag_fidelity_bf16_unsloth", "ag_fidelity_bf16_google", "ag_fidelity_fp32_google"):
        j = jload(fid / ("%s.json" % name))
        g[name] = None if j is None else {k: v.get("passed") for k, v in j.get("checks", {}).items()}
    w = jload(W / "checks" / "weights.json")
    g["weights_google_vs_unsloth"] = None if w is None else {k: w.get(k) for k in (
        "n_common", "n_bitwise_equal", "max_abs_diff", "all_equal")}
    m = jload(W / "checks" / "means_vs_local.json")
    g["means_pod_vs_local"] = None if m is None else dict(mean_tokens_equal=m.get("mean_tokens_equal"), layers=m.get(
        "layers"), ld=m.get("ld"))
    refs = W / "checks" / "hf_refs.txt"
    g["hf_refs"] = refs.read_text().splitlines() if refs.is_file() else None
    g["hard_stopped"] = (st / "HARD_STOPPED").exists()

    J = S["jobs"]
    runs = W / "caples_runs"
    J["caples_grids"] = {G: dict(done=(runs / G / "DONE").exists(),
                                 n_run_json=len(list((runs / G / "src" / "masks").glob("sva/*/logit_diff_*_run/*_run.json"))))
                         for G in ("A", "B", "C", "D", "dC", "dB", "dA") if (runs / G).exists()}
    keep = ("lam", "means", "substrate", "n_latents", "task_free0", "task_freeM_topk", "task_freeN_topk", "task_nec",
            "t_fit", "t_score")
    for fam_dir in ("wcm_d2", "wcm_d1"):
        for d in sorted((W / fam_dir).glob("*")) if (W / fam_dir).is_dir() else []:
            rows = jl(d / "eval.jsonl")
            J["%s/%s" % (fam_dir, d.name)] = dict(rows=len(rows), expected=len(lams),
                                                  table=[{k: r.get(k) for k in keep} for r in rows])
    ag = W / "ag_tasks"
    J["ag_tasks"] = {t: dict(n_prompts=len(list((ag / t / "held").glob("p*.json"))), expected=ag_k) for t in tasks}

    for d in sorted((W / "score_d").glob("*")) if (W / "score_d").is_dir() else []:
        rows = jl(d / "scores.jsonl")
        S["scoring"][d.name] = dict(rows=len(rows), theirs=(d / "theirs.json").exists())

    tm = defaultdict(lambda: dict(n=0, seconds=0, failed=0))
    tsv = W / "cmds.tsv"
    if tsv.is_file():
        for line in tsv.read_text().splitlines():
            p = line.split("\t")
            if len(p) < 4:
                continue
            try:
                t = tm[p[0]]
                t["n"] += 1
                t["seconds"] += int(p[2]) - int(p[1])
                t["failed"] += int(p[3] != "0")
            except ValueError:
                pass
    S["timings"] = dict(tm)
    S["gpu_seconds_total"] = sum(t["seconds"] for k, t in tm.items() if not k.startswith(("P0/", "P1/", "P5/")))
    return S


def markdown(S: dict) -> str:
    L = ["# Arena D pod stage summary", "", "WORK `%s`" % S["work"], "", "## Gate", ""]
    g = S["gate"]
    L += ["- gate D2: **%s**, gate D1: **%s**%s" % (g["gate_d2"], g["gate_d1"],
                                                    " (HARD STOP reached)" if g["hard_stopped"] else "")]
    for k in ("caples_A", "caples_fidelity", "ag_fidelity_bf16_unsloth", "ag_fidelity_bf16_google",
              "ag_fidelity_fp32_google", "weights_google_vs_unsloth"):
        L.append("- %s: %s" % (k, "missing" if g[k] is None else json.dumps(g[k], default=str)))
    L.append("- means pod vs local: %s" % ("missing" if g["means_pod_vs_local"] is None else
                                           "mean_tokens_equal %s" % g["means_pod_vs_local"]["mean_tokens_equal"]))
    if g["hf_refs"]:
        L += ["- HF refs:"] + ["  - `%s`" % x for x in g["hf_refs"]]
    L += ["", "## Jobs", ""]
    J = S["jobs"]
    for G, v in J["caples_grids"].items():
        L.append("- Caples grid %s: %s, %d run JSONs" % (G, "DONE" if v["done"] else "not done", v["n_run_json"]))
    for k, v in J.items():
        if not k.startswith("wcm_"):
            continue
        L.append("- %s: %d / %d lambdas" % (k, v["rows"], v["expected"]))
        for r in v["table"]:
            L.append("  - lam %s %s %s: %s latents | Z %s A %s C %s nec %s | fit %ss" % (
                r["lam"], r["substrate"], r["means"], r["n_latents"], f3(r["task_free0"]), f3(r["task_freeM_topk"]),
                f3(r["task_freeN_topk"]), f3(r["task_nec"]), r["t_fit"]))
    for t, v in J["ag_tasks"].items():
        L.append("- AG %s: %d / %d held prompts" % (t, v["n_prompts"], v["expected"]))
    L += ["", "## Scoring (score_d.py)", ""]
    L += ["- %s: %d rows, theirs %s" % (k, v["rows"], v["theirs"]) for k, v in S["scoring"].items()] or ["- none"]
    L += ["", "## Timings (cmds.tsv, per log)", "", "| log | commands | seconds | failed |", "|---|---|---|---|"]
    L += ["| %s | %d | %d | %d |" % (k, v["n"], v["seconds"], v["failed"]) for k, v in sorted(S["timings"].items())]
    L += ["", "GPU-process seconds outside P0 / P1 / P5: %d (%.2f h; not GPU-h: jobs run in parallel)"
          % (S["gpu_seconds_total"], S["gpu_seconds_total"] / 3600), ""]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("work")
    p.add_argument("--lams", default="3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    p.add_argument("--tasks", default="simple,within_rc,rc,nounpp")
    p.add_argument("--ag-k", type=int, default=40)
    p.add_argument("--out", default=None, help="default: WORK")
    a = p.parse_args(argv)
    W = Path(a.work)
    S = summarise(W, [x for x in a.lams.split(",") if x], [t for t in a.tasks.split(",") if t], a.ag_k)
    out = Path(a.out) if a.out else W
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=str) + "\n")
    md = markdown(S)
    (out / "SUMMARY.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
