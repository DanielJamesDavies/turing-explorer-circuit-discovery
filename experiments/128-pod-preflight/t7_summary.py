"""Exp 128 t7_full: one-page summary of the one-target 083 run (stages ref, attr, fitted) at production settings.

  python -I t7_summary.py --out083 experiments/128-pod-preflight/out/t7_full/dan159 --night-log .../night.log \
      --prep .../PREP.json --md .../T7_SUMMARY.md

Per stage: rows, error rows, and per row the label fields present (arm / ranking / size / source / alpha ...), n,
Z / A / C (free0_tk, freeM_topk_tk, freeN_topk_tk), nec (phi_sup_blind_tk); stage wall times from night.log
(START / END lines of t7_<stage>); the production W_0.001 size from PREP.json. table2.md (083's own table) sits beside.
"""
import argparse
import json
import re
from pathlib import Path

LABELS = ("arm", "ref_arm", "method", "source", "ranking", "size_arm", "size", "alpha", "mode", "fill", "label")
ZACN = (("Z", "free0_tk"), ("A", "freeM_topk_tk"), ("C", "freeN_topk_tk"), ("nec", "phi_sup_blind_tk"))


def f(x):
    return "%.3f" % x if isinstance(x, float) else ("-" if x is None else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out083", required=True)
    ap.add_argument("--night-log", default="")
    ap.add_argument("--prep", default="")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    O = Path(a.out083)
    L = ["# Exp 128 t7_full: 083 suite on one TuringLLM target, production settings (local RTX 5070 Ti)", ""]
    if a.prep and Path(a.prep).exists():
        p = json.loads(Path(a.prep).read_text())
        L.append("Target %s; production WCM sizes (T4 sweep, H100): %s." % (
            p["key"], ", ".join("%s %s" % (k, v) for k, v in sorted(p["n_by_arm"].items()))))
        L.append("")
    if a.night_log and Path(a.night_log).exists():
        t = {}
        for line in Path(a.night_log).read_text().splitlines():
            m = re.search(r"END (t7_\w+) rc (\d+) \((\d+) min\)", line)
            if m:
                t[m.group(1)] = "%s min (rc %s)" % (m.group(3), m.group(2))
        if t:
            L += ["Wall time per stage (incl. ~3 min start-up each): " + ", ".join("%s %s" % kv for kv in t.items()), ""]
    for s in ("ref", "attr", "fitted"):
        rows = []
        for fp in sorted((O / s).glob("rows*.jsonl")):
            for line in fp.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        err = [r for r in rows if "error" in r]
        L += ["## %s: %d rows, %d error rows" % (s, len(rows), len(err)), ""]
        if err:
            L.append("first error: %s" % str(err[0].get("error"))[:300])
        if rows:
            lab = [k for k in LABELS if any(k in r for r in rows)]
            L += ["| %s | n | Z | A | C | nec |" % " | ".join(lab), "|%s---|---|---|---|---|" % ("---|" * len(lab))]
            for r in rows:
                if "error" in r:
                    continue
                L.append("| %s | %s | %s |" % (" | ".join(f(r.get(k)) for k in lab), f(r.get("n")),
                                              " | ".join(f(r.get(c)) for _, c in ZACN)))
        L.append("")
    Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
