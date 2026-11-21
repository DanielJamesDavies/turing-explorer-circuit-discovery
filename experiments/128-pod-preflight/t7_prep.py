"""Exp 128 t7_full prep (CPU, copies only): a one-target REF_RUN directory for the 083 suite (chain T7 = stages ref,
attr, fitted) from the 2026-11-07 production pod archives (WCM v4, TuringLLM), so T7 runs at production settings on
real production circuits without touching the archives or any production script.

  python -I t7_prep.py --key 3.attn.11307 --out experiments/128-pod-preflight/out/t7_full

What the 083 scripts read from REF_RUN (cmp_common.sweep_rows / circuit_path; also the 062 driver's OUT, whose
WCM_VERSION / WCM_R markers it checks): sweep/eval.*.jsonl and sweep/circuits/<arm>/<key>.pt. Written into OUT:
  ref/WCM_VERSION, ref/WCM_R           copied from the T3 archive's out_v2 (v4; R settings)
  ref/sweep/circuits/<arm>/<key>.pt    every arm of the T4 sweep archive that holds the key (8 W + 5 U)
  ref/sweep/eval.shard0.jsonl          the T4 sweep eval rows of the key (all shards)
  ref/main/circuits/<key>.pt, ref/main/eval.shard0.jsonl   the T3 production circuit + rows (for the report)
  targets.txt                          the key
  PREP.json                            sources, sizes, the production n of W_0.001
The record (CTX_DIR) is read in place from the T2 archive's ctx_fresh (the driver never writes a CTX_DIR).
Refuses to overwrite an existing ref/ (exit 1).
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARC = ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives"
T3 = ARC / "pub-turingllm-T3-main" / "root" / "turing-tl" / "experiments" / "094-tl-v2-records" / "out_v2"
T4 = ARC / "pub-turingllm-T4-sweep" / "root" / "turing-tl" / "experiments" / "094-tl-v2-records" / "out_v2"
CTX = ARC / "pub-turingllm-T2-records" / "root" / "turing-tl" / "experiments" / "082-public-runs" / "out_turingllm" / "ctx_fresh"
SWEEP190 = CTX.parent / "lists" / "targets_tl8k_sweep190.txt"


def key_rows(paths, key):
    out = []
    for p in paths:
        for line in p.read_text().splitlines():
            if ('"seed": "%s"' % key) in line:
                out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="3.attn.11307")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    O = Path(a.out)
    R = O / "ref"
    if R.exists():
        raise SystemExit("%s exists: not overwritten" % R)
    k = a.key
    if k not in SWEEP190.read_text().split():
        raise SystemExit("%s is not one of the 190 sweep targets (%s)" % (k, SWEEP190))
    for p in (T3 / "WCM_VERSION", T3 / "main" / "circuits" / ("%s.pt" % k), CTX / ("%s.pt" % k)):
        if not p.exists():
            raise SystemExit("missing %s" % p)
    if (T3 / "WCM_VERSION").read_text().strip() != "v4":
        raise SystemExit("T3 archive is not v4")
    tmp = O / "ref.partial"
    if tmp.exists():
        raise SystemExit("%s exists (an interrupted prep?): move it away" % tmp)
    (tmp / "sweep" / "circuits").mkdir(parents=True)
    (tmp / "main" / "circuits").mkdir(parents=True)
    for m in ("WCM_VERSION", "WCM_R", "LR_SCHEDULE"):
        if (T3 / m).exists():
            shutil.copy2(T3 / m, tmp / m)
    arms = []
    for d in sorted((T4 / "sweep" / "circuits").iterdir()):
        src = d / ("%s.pt" % k)
        if src.exists():
            (tmp / "sweep" / "circuits" / d.name).mkdir()
            shutil.copy2(src, tmp / "sweep" / "circuits" / d.name / src.name)
            arms.append(d.name)
    ev = key_rows(sorted((T4 / "sweep").glob("eval.*.jsonl")), k)
    (tmp / "sweep" / "eval.shard0.jsonl").write_text("\n".join(ev) + "\n")
    shutil.copy2(T3 / "main" / "circuits" / ("%s.pt" % k), tmp / "main" / "circuits" / ("%s.pt" % k))
    mev = key_rows(sorted((T3 / "main").glob("eval.*.jsonl")), k)
    (tmp / "main" / "eval.shard0.jsonl").write_text("\n".join(mev) + "\n")
    tmp.rename(R)
    (O / "targets.txt").write_text(k + "\n")
    n = {}
    for line in ev:
        r = json.loads(line)
        if "error" not in r:
            n[r.get("arm")] = r.get("n")
    info = dict(key=k, arms=arms, n_eval_rows=len(ev), n_by_arm=n, main_eval_rows=len(mev), ctx=str(CTX / ("%s.pt" % k)),
                sources=dict(T3=str(T3), T4=str(T4)), bytes=sum(p.stat().st_size for p in R.rglob("*") if p.is_file()))
    (O / "PREP.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    if len(arms) < 13 or not ev:
        raise SystemExit("only %d sweep arms / %d eval rows for %s" % (len(arms), len(ev), k))


if __name__ == "__main__":
    main()
