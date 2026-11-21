"""Arena D v4 quick check, step 3 (CPU; v4_check.sh, V4_CHECK.md): the production-path v4 fits vs lrtest's C1 (same
recipe through a wrapper) and C0 (v3) on the same family / task / lambda. Exit 1 on a provenance mismatch or a
tolerance miss.

  python v4_check_compare.py <v4_check out dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LRT = HERE / "lrtest" / "out" / "runs"
METRICS = (("Z", "task_free0"), ("A", "task_freeM_topk"), ("C", "task_freeN_topk"), ("nec", "task_nec"))
SIZE_TOL = 0.25         # |log size ratio| vs C1 within ~25% (C1 vs its R-seed replicate C1s2: 215 vs 232 = 8%)
MET_TOL = 0.05          # per metric vs C1 (C1 vs C1s2: <= 0.008)


def row(path: Path, lam: float = 1e-3):
    if not path.exists():
        return None
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if r.get("sfc_task") == "rc" and abs(float(r["lam"]) - lam) < 1e-12 and int(r["steps"]) == 400:
            return r
    return None


def main() -> int:
    out = Path(sys.argv[1])
    fails = []
    for fam in ("res", "plt"):
        v4 = row(out / fam / "eval.jsonl")
        c1, c0 = row(LRT / "C1" / fam / "eval.jsonl"), row(LRT / "C0" / fam / "eval.jsonl")
        if v4 is None:
            fails.append("%s: no v4 row" % fam)
            print("%s: NO v4 ROW in %s" % (fam, out / fam / "eval.jsonl"))
            continue
        circ = json.loads((out / fam / v4["circuit"]).read_text())
        f = circ["fit"]
        prov = dict(wcm=v4.get("wcm"), lr_peak=f.get("lr_peak"), warmup_steps=f.get("warmup_steps"),
                    weight_decay=f.get("weight_decay"), lr_sum=f.get("lr_sum"), schedule=f.get("lr_schedule"),
                    R=f.get("resample_weight"), rank=f.get("rank_weight"))
        ok_prov = (prov["wcm"] == "v4" and abs((prov["lr_peak"] or 0) - 0.1575) < 1e-9 and prov["warmup_steps"] == 40
                   and abs((prov["weight_decay"] or 0) - 0.05) < 1e-12 and abs((prov["lr_sum"] or 0) - 21.40) < 0.01
                   and prov["schedule"] == "geometric" and prov["R"] == 1.0 and not prov["rank"])
        print("%s provenance %s: %s" % (fam, "OK" if ok_prov else "MISMATCH", prov))
        if not ok_prov:
            fails.append("%s provenance" % fam)
        hdr = "  %-4s %7s " % ("arm", "n") + " ".join("%6s" % m for m, _ in METRICS) + "   t_fit  peak_MB"
        print(hdr)
        for name, r in (("v4", v4), ("C1", c1), ("C0", c0)):
            if r is None:
                print("  %-4s (no row)" % name)
                continue
            print("  %-4s %7d " % (name, r["n_latents"]) + " ".join(
                "%6s" % ("-" if r.get(k) is None else "%.3f" % r[k]) for _, k in METRICS)
                  + "  %6.0f  %7s" % (r.get("t_fit") or 0, r.get("peak_fit_mb")))
        if c1 is None:
            fails.append("%s: no lrtest C1 row to compare" % fam)
            continue
        import math
        dn = abs(math.log(max(v4["n_latents"], 1) / max(c1["n_latents"], 1)))
        dm = {m: (None if v4.get(k) is None or c1.get(k) is None else abs(v4[k] - c1[k])) for m, k in METRICS}
        same = v4["n_latents"] == c1["n_latents"] and all(d is not None and d < 1e-6 for d in dm.values())
        ok = dn <= math.log(1 + SIZE_TOL) and all(d is None or d <= MET_TOL for d in dm.values())
        print("  v4 vs C1: size ratio %.3f, |dmetric| %s -> %s%s" % (
            v4["n_latents"] / max(c1["n_latents"], 1), {m: None if d is None else round(d, 4) for m, d in dm.items()},
            "PASS" if ok else "FAIL", " (identical)" if same else ""))
        if not ok:
            fails.append("%s outside tolerance vs C1" % fam)
    print("v4_check_compare: %s" % ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
