"""Exp 132: curves tables (CPU). Reads out/curves_<tag>.jsonl and writes out/curves.md (one table per file x method),
plus the best point per method (max worst-of-Z/A/C, then smaller n).
  .venv/bin/python experiments/132-ioi-ceiling-baselines/report.py
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LABEL = {
    "c081": "081 v4 circuits (W = WCM, U = unweighted), rescored here",
    "sfc100": "SFC node attribution, IG 10 steps, 100 train items (re-implementation), |IE| > T_N, scope 48",
    "sfc1600": "SFC as above with all 1,600 train items (our adaptation: more data), scope 48",
    "sfc100@am": "SFC (100 items), nodes restricted to attn+mlp, scope attn+mlp, resid LIVE (our adaptation)",
    "sfc1600@am": "SFC (1,600 items), attn+mlp scope (our adaptation)",
    "atp": "attribution patching top-k, |mean IE|, 1,600 items, scope 48",
    "atp@am": "attribution patching top-k, attn+mlp scope (resid LIVE)",
    "atpabs": "attribution patching top-k by mean |IE per item| (ours), scope 48",
    "atpabs@am": "attribution patching top-k by mean |IE per item|, attn+mlp scope",
    "adiff": "activation-difference top-k (mean |f_cf - f_clean|), scope 48",
    "adiff@am": "activation-difference top-k, attn+mlp scope",
}


def f(v):
    return "%.3f" % v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else "-"


def worst(r):
    vals = [r.get(k) for k in ("task_free0", "task_freeM_topk", "task_freeN_topk")]
    if any(v is None for v in vals):
        return -9.0
    return min(v if v <= 1 else max(0.0, 2.0 - v) for v in vals)        # 1 is best; above 1 counts down


def main():
    L = ["# Exp 132 stage 2: faithfulness vs size (081 scorer, IOI held-out)", ""]
    best = []
    for p in sorted(OUT.glob("curves_*.jsonl")):
        rows = [json.loads(l) for l in open(p)]
        by = defaultdict(list)
        for r in rows:
            by[r["method"]].append(r)
        L += ["## %s (%d held-out items per point)" % (p.stem, rows[0]["n_heldout"] if rows else 0), ""]
        for m, rs in by.items():
            rs.sort(key=lambda r: r["n"])
            L += ["### %s" % m, "", LABEL.get(m, ""), "",
                  "| knob | n (site, latent) | native | Z | A | C | nec | acc Z | pass |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
            for r in rs:
                L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["knob"], r["n"], r.get("native"), f(r["task_free0"]), f(r["task_freeM_topk"]),
                    f(r["task_freeN_topk"]), f(r["task_nec"]), f(r["task_acc_free0"]), "PASS" if r["task_pass"] else ""))
            b = max(rs, key=lambda r: (worst(r), -r["n"]))
            sp = [r for r in rs if r["task_pass"]]
            best.append((p.stem, m, b, min(sp, key=lambda r: r["n"]) if sp else None))
            L.append("")
    L += ["## Best point per method (max of min(Z, A, C), values above 1 counted as 2 - v) and smallest passing", "",
          "| file | method | best knob | n | Z | A | C | nec | acc | smallest passing n |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stem, m, b, sp in best:
        L.append("| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            stem, m, b["knob"], b["n"], f(b["task_free0"]), f(b["task_freeM_topk"]), f(b["task_freeN_topk"]),
            f(b["task_nec"]), f(b["task_acc_free0"]), sp["n"] if sp else "none"))
    (OUT / "curves.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-(len(best) + 4):]))


if __name__ == "__main__":
    main()
