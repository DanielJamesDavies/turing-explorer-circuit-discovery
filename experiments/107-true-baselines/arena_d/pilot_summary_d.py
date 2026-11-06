"""Arena D7: summary of run_pilot_d.sh's GPU night (CPU; reads files only, never fails on a missing one).

  python pilot_summary_d.py <pilot dir> [--spill-mib 15600] [--out <dir>]
Writes <out>/summary.json and <out>/SUMMARY.txt (default out = the pilot dir): per step exit code and wall time, peak
nvidia-smi memory per step and every sample above the spill line, the D6 / test-4 results, Caples' smoke and means
LDs, the WCM fits' timings per family (s / step at micro-batch 4 x 7 tokens), F1-F6, and the AG per-prompt timings.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _jsonl(p: Path):
    out = []
    for line in _read(p).splitlines():
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return out


def summarise(P: Path, spill: int) -> dict:
    S = dict(pilot=str(P), steps=[], mem={}, spill_samples=[], results={})
    for line in _read(P / "steps.tsv").splitlines():
        f = line.split("\t")
        if len(f) >= 4:
            S["steps"].append(dict(step=f[0], start=int(f[1]), end=int(f[2]), wall_s=int(f[2]) - int(f[1]),
                                   rc=int(f[3])))
    for line in _read(P / "gpu_mem.csv").splitlines():
        f = line.split(",")
        if len(f) < 3 or not f[2].strip().isdigit():
            continue
        st, mib = f[1], int(f[2])
        S["mem"][st] = max(S["mem"].get(st, 0), mib)
        if mib > spill:
            S["spill_samples"].append(dict(epoch=int(f[0]), step=st, mib=mib))
    L = P / "logs"
    R = S["results"]
    pyt = r"(?m)^(?:=+ )?(\d+ (?:passed|failed|skipped|error|deselected).*) in [0-9.]+s"
    R["d6"] = re.findall(r"RESULT D6 identity.*", _read(L / "d6.log"))
    R["d6_pytest"] = (re.findall(pyt, _read(L / "d6.log")) or [None])[-1]
    R["test4_real"] = re.findall(r"RESULT test 4 \(real.*", _read(L / "real_caples.log"))
    R["test4_pytest"] = (re.findall(pyt, _read(L / "real_caples.log")) or [None])[-1]
    m = re.search(r"\{\s*\"out\".*\}", _read(L / "caples_means.log"), re.S)
    R["caples_means"] = _json_text(m.group(0)) if m else None
    sm = _json(P / "caples_smoke" / "smoke_summary.json")
    R["caples_smoke"] = None if sm is None else {k: sm.get(k) for k in ("avg_model_ld", "avg_sae_ld",
                                                                          "avg_sae_mean_error_ld", "runs",
                                                                          "model_load_s", "total_s")}
    fam = {}
    for name in ("wcm_plt", "wcm_res"):
        for r in _jsonl(P / name / "eval.jsonl"):
            fam[name] = dict(family=r.get("family"), substrate=r.get("substrate"), means=r.get("means"),
                             data=r.get("data_mode"), steps=r.get("steps"), t_fit=r.get("t_fit"),
                             s_per_step=(r["t_fit"] / r["steps"] if r.get("t_fit") and r.get("steps") else None),
                             t_score=r.get("t_score"), peak_fit_mb=r.get("peak_fit_mb"),
                             peak_score_mb=r.get("peak_score_mb"), n_latents=r.get("n_latents"),
                             prompt_len=r.get("prompt_len"), n_train=r.get("n_train"), n_held=r.get("n_held"),
                             task_full=r.get("task_full"), Z=r.get("task_free0"), A=r.get("task_freeM_topk"),
                             C=r.get("task_freeN_topk"), nec=r.get("task_nec"))
    R["wcm_timings"] = fam
    fid = _json(P / "ag_fidelity_bf16.json")
    R["ag_fidelity"] = None if fid is None else {k: dict(passed=v.get("passed"), **{
        x: v[x] for x in ("feature_jaccard", "n_feature_nodes", "ref_feature_nodes", "logit_probs_ok", "identity",
                          "survive_06_099", "survive_08_098", "n_pinned", "record_files_checked",
                          "resident_gb_before_eager", "free_gb_before_eager", "lazy_vs_eager") if x in v})
        for k, v in fid.get("checks", {}).items()}
    if fid is not None:
        R["ag_fidelity_dallas"] = fid.get("dallas")
        R["ag_fidelity_model"] = {k: fid.get("model", {}).get(k) for k in ("transcoders_used", "transcoders_fallback",
                                                                           "load_s", "resident_gb")}
    ag = []
    for f in sorted((P / "ag_tasks").glob("*/held/p*.json")):
        r = _json(f) or {}
        ag.append(dict(task=r.get("task"), id=r.get("id"), n_pos=r.get("n_pos"), n_active=r.get("n_active"),
                       t_attribute_s=r.get("t_attribute_s"), t_prune_s=r.get("t_prune_s"), peak_mb=r.get("peak_mb")))
    R["ag_tasks"] = ag
    S["stopped_by_watchdog"] = (P / "STOPPED").exists()
    S["warnings"] = _read(P / "warnings.txt").splitlines()
    return S


def _json_text(t):
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        return t[:2000]


def text(S: dict, spill: int) -> str:
    out = ["arena D GPU pilot summary: %s" % S["pilot"], ""]
    out.append("steps (exit code, wall s, peak nvidia-smi MiB):")
    for s in S["steps"]:
        out.append("  %-12s rc %d  %6d s  peak %s MiB" % (s["step"], s["rc"], s["wall_s"], S["mem"].get(s["step"], "-")))
    if not S["steps"]:
        out.append("  (none)")
    out.append("spill line %d MiB: %d samples above%s" % (spill, len(S["spill_samples"]), (
        " (max %d MiB in %s)" % max((x["mib"], x["step"]) for x in S["spill_samples"])) if S["spill_samples"] else ""))
    out.append("watchdog stop: %s" % S["stopped_by_watchdog"])
    R = S["results"]
    out += ["", "D6 identity: %s | %s" % (R["d6_pytest"], "; ".join(R["d6"]) or "-"),
            "test 4 (real): %s | %s" % (R["test4_pytest"], "; ".join(R["test4_real"]) or "-"),
            "caples_means: %s" % (json.dumps(R["caples_means"], default=str)[:600] if R["caples_means"] else "-"),
            "caples_smoke: %s" % (json.dumps(R["caples_smoke"], default=str)[:600] if R["caples_smoke"] else "-"),
            "", "WCM tiny fits (micro-batch 4; s / step = t_fit / steps):"]
    for k, v in R["wcm_timings"].items():
        out.append("  %s: %s" % (k, json.dumps(v, default=str)))
    if not R["wcm_timings"]:
        out.append("  (none)")
    out += ["", "AG fidelity: %s" % (json.dumps(R["ag_fidelity"], default=str) if R["ag_fidelity"] else "-")]
    if R.get("ag_fidelity_dallas"):
        out.append("  Dallas: %s" % json.dumps({k: v for k, v in R["ag_fidelity_dallas"].items() if k != "tokens"}))
        out.append("  model: %s" % json.dumps(R.get("ag_fidelity_model")))
    out.append("AG task graphs: %d" % len(R["ag_tasks"]))
    for r in R["ag_tasks"]:
        out.append("  %s" % json.dumps(r))
    if S["warnings"]:
        out += ["", "warnings:"] + ["  " + w for w in S["warnings"]]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot")
    ap.add_argument("--spill-mib", type=int, default=15600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    P = Path(a.pilot)
    S = summarise(P, a.spill_mib)
    out = Path(a.out) if a.out else P
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=str) + "\n")
    t = text(S, a.spill_mib)
    (out / "SUMMARY.txt").write_text(t)
    print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
