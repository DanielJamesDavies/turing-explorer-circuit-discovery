"""Tabulate every *_gemma_members.jsonl in this folder: members, held-out EF in
both frames (zero-fill / mean-fill), alpha=1 control, amplitude nulls.

  python experiments/052-gemma3-270m/collect_fits.py [substring filter]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
FILT = sys.argv[1] if len(sys.argv) > 1 else ""
rows = []
for p in sorted(HERE.glob("*_gemma_members.jsonl")):
    if FILT not in p.name:
        continue
    r = json.loads(open(p).readline())
    s = r.get("scores", {})
    z, m = s.get("zero", {}), s.get("mean", {})
    rows.append((p.name.replace("_gemma_members.jsonl", ""), r))
print("%-34s %6s %8s %5s %5s | %7s %7s %13s | %7s %7s %13s"
      % ("fit", "n", "lam", "lr?", "steps", "EF0", "EF0 a1", "EF0 nulls", "EFm", "EFm a1", "EFm nulls"))
for name, r in rows:
    s = r.get("scores", {})
    f = lambda d, k: ("%.3f" % d[k]) if d.get(k) is not None else "-"
    nl = lambda d: "/".join("%.2f" % x for x in d.get("EF_nulls", []) if x is not None) or "-"
    z, m = s.get("zero", {}), s.get("mean", {})
    print("%-34s %6d %8.0e %5s %5d | %7s %7s %13s | %7s %7s %13s"
          % (name, r["n_members"], r["lam"], r.get("lr", "-"), r["steps"],
             f(z, "EF"), f(z, "EF_a1"), nl(z), f(m, "EF"), f(m, "EF_a1"), nl(m)))
