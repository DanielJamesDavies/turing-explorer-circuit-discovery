"""D1 assembly: per-notion tables from the bake-off rows.

Per D0.1's verdict the two driver notions are reported SEPARATELY:
intervention-drivers (phi-cf at small K, the headline) and pinned-drivers
(pin0_c). sup is a sanity gate. Kind and deep-band splits per the
evidence-standards discipline.

  python experiments/012-driver-bakeoff/summarise.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
rows = []
for f in sorted(HERE.glob("rows_s*.jsonl")):
    for line in f.open():
        if line.strip():
            rows.append(json.loads(line))

ARMS = ["A", "D", "C", "MI", "R", "H", "RAND"]
KS = [64, 256, 1024, 4096]
DEEP = {8, 9, 10, 11}


def med(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def fmt(v, nd=2):
    return ("%.*f" % (nd, v)) if v is not None else "--"


def sel(arm, K, layers=None, kinds=None):
    out = []
    for r in rows:
        if r["arm"] != arm or r["K"] != K:
            continue
        if layers is not None and r["layer"] not in layers:
            continue
        if kinds is not None and r["kind"] not in kinds:
            continue
        out.append(r)
    return out


lines = ["# D1 summary tables (medians; 11 seeds with data)\n"]

lines.append("## T1 — Intervention-drivers: median phi-cf by arm x K "
             "(headline: small K)\n")
lines.append("| arm | K=64 | K=256 | K=1024 | K=4096 | K=256 deep(L8+) |")
lines.append("|---|---|---|---|---|---|")
for arm in ARMS:
    vals = [fmt(med([r["cf"] for r in sel(arm, K)])) for K in KS]
    deep = fmt(med([r["cf"] for r in sel(arm, 256, layers=DEEP)]))
    lines.append("| %s | %s | %s |" % (arm, " | ".join(vals), deep))

lines.append("\n## T2 — Pinned-drivers: median pin0_collapsed by arm x K\n")
lines.append("| arm | K=64 | K=256 | K=1024 | K=4096 | K=4096 deep(L8+) |")
lines.append("|---|---|---|---|---|---|")
for arm in ARMS:
    vals = [fmt(med([r["pin0_c"] for r in sel(arm, K)])) for K in KS]
    deep = fmt(med([r["pin0_c"] for r in sel(arm, 4096, layers=DEEP)]))
    lines.append("| %s | %s | %s |" % (arm, " | ".join(vals), deep))

lines.append("\n## T3 — sup sanity (median at K=64) and kind split of cf@256\n")
lines.append("| arm | sup@64 | cf@256 resid | cf@256 mlp | cf@256 attn |")
lines.append("|---|---|---|---|---|")
for arm in ARMS:
    s64 = fmt(med([r["sup"] for r in sel(arm, 64)]))
    kr = fmt(med([r["cf"] for r in sel(arm, 256, kinds={"resid"})]))
    km = fmt(med([r["cf"] for r in sel(arm, 256, kinds={"mlp"})]))
    ka = fmt(med([r["cf"] for r in sel(arm, 256) if "attn" in r["kind"]]))
    lines.append("| %s | %s | %s | %s | %s |" % (arm, s64, kr, km, ka))

lines.append("\n## T4 — K* (pinned-driver size, pin0>=0.8 on train) per seed\n")
lines.append("| seed | layer | kind | K* |")
lines.append("|---|---|---|---|")
kstars = {}
for r in rows:
    if r.get("k_star") is not None:
        kstars[r["seed"]] = (r["layer"], r["kind"], r["k_star"])
for seed, (layer, kind, ks) in sorted(kstars.items(),
                                      key=lambda kv: kv[1][0]):
    lines.append("| %s | L%d | %s | %s |"
                 % (seed, layer, kind, "ceiling(-1)" if ks == -1 else ks))

lines.append("\n## T5 — Amplified direct-mass: alpha* and calibrated cf\n")
lines.append("| K | med alpha* | med cf (raw) | med cf_alpha | rescued "
             "(cf_a - cf) |")
lines.append("|---|---|---|---|---|")
for K in KS:
    cs = sel("C", K)
    a = med([r["alpha_star"] for r in cs])
    c_raw = med([r["cf"] for r in cs])
    c_a = med([r["cf_alpha"] for r in cs])
    lines.append("| %d | %s | %s | %s | %s |"
                 % (K, fmt(a), fmt(c_raw), fmt(c_a),
                    fmt((c_a - c_raw) if None not in (c_a, c_raw) else None)))

n_by_arm = defaultdict(int)
for r in rows:
    n_by_arm[r["arm"]] += 1
lines.append("\nrows: %d | per arm: %s"
             % (len(rows), dict(sorted(n_by_arm.items()))))

out = "\n".join(lines)
print(out)
(HERE / "summary_tables.md").write_text(out + "\n", encoding="utf-8")
