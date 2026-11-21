"""133 (CPU): results.jsonl -> a markdown table (out/table.md) and stdout.

  .venv/bin/python experiments/133-ioi-wcm-sweep/make_table.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(HERE / "out" / "results.jsonl")]
rows = [r for r in rows if "error" not in r]
# last row per name wins
by = {}
for r in rows:
    by[r["name"]] = r


def f(v, d=2):
    if isinstance(v, bool) or v is None:
        return "-" if v is None else ("**yes**" if v else "no")
    if isinstance(v, (int, float)):
        return ("%%.%df" % d) % v
    return str(v)


def arm(r):
    if r.get("rescore"):
        return "rescore " + r["rescore"].split("circuits/ioi/")[-1].replace(".json", "").replace("experiments/081-task-circuits/", "")
    s = "%s %s lr %.3g %s" % (r.get("sites"), r.get("arm"), r.get("lr"), r.get("sched"))
    s += " R%s%s" % (f(r.get("R"), 1), " paired" if r.get("pair") else "")
    if r.get("off_w"):
        s += " off %s x%g" % (r.get("off_mode"), r["off_w"])
    if r.get("steps") and r.get("steps") != 400:
        s += " %d steps" % r["steps"]
    return s


hdr = ("| config | arm | lambda | n | dis Z | dis A | dis C | dis nec | dis acc | dis mp Z | dis pass | sn Z | sn A | "
       "sn C | sn nec | sn acc | sn mp Z | sn pass |")
lines = [hdr, "|" + "---|" * 18]
for name, r in by.items():
    lam = r.get("lam")
    lines.append("| %s | %s | %s | %s | %s |" % (
        name, arm(r), ("%g" % lam) if isinstance(lam, (int, float)) else "-", r.get("n"),
        " | ".join(f(r.get("%s_%s" % (p, k))) for p in ("dis", "sn") for k in (
            "free0", "freeM_topk", "freeN_topk", "nec", "acc_free0", "mp_free0", "pass"))))
txt = "\n".join(lines) + "\n"
(HERE / "out" / "table.md").write_text(txt)
print(txt)
