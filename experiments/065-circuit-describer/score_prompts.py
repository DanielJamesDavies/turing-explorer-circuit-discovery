"""Score prompt versions against results/test_expectations.json (types written down before each run).

    python experiments/065-circuit-describer/score_prompts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
RUNS = {"v1": "_v1/descriptions.jsonl", "v2": "test_prompt_v2.jsonl", "v3": "test_prompt_v3.jsonl",
        "v4": "test_prompt_v4.jsonl"}


def load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("result"):
                out[r["key"]] = r   # last record per key wins
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exp = json.loads((RESULTS / "test_expectations.json").read_text(encoding="utf-8"))
    runs = {v: load(f) for v, f in RUNS.items()}
    for group in ("original14", "holdout15"):
        cases = exp[group]
        print(f"\n== {group}")
        print(f"{'circuit':16s} {'expect':12s} " + " ".join(f"{v:12s}" for v in runs))
        totals = {v: [0, 0, 0] for v in runs}   # exact, exact-or-alt, n
        for key, e in cases.items():
            cells = []
            for v, recs in runs.items():
                t = recs.get(key, {}).get("result", {}).get("type")
                if t is None:
                    cells.append(f"{'-':12s}")
                    continue
                exact, ok = t == e["expect"], t == e["expect"] or t in e["alt"]
                totals[v][0] += exact
                totals[v][1] += ok
                totals[v][2] += 1
                cells.append(f"{t[:10] + ('*' if exact else ('~' if ok else ' ')):12s}")
            print(f"{key:16s} {e['expect'][:12]:12s} " + " ".join(cells))
        print("exact / exact-or-alt: " + "  ".join(
            f"{v} {a}/{n} , {b}/{n}" for v, (a, b, n) in totals.items() if n))
    print("\n* = first choice, ~ = acceptable alternative")
    for v, recs in runs.items():
        outs = [r.get("output_tokens") or 0 for r in recs.values()]
        ins = [r.get("input_tokens") or 0 for r in recs.values()]
        if outs:
            print(f"{v}: mean input {sum(ins) / len(ins):.0f}, mean output {sum(outs) / len(outs):.0f} tokens")


FACET_RUNS = {"v5": "test_prompt_v5.jsonl"}


def facets():
    """Score trigger + condition runs against results/test_expectations_facets.json."""
    exp = json.loads((RESULTS / "test_expectations_facets.json").read_text(encoding="utf-8"))
    for v, name in FACET_RUNS.items():
        recs = load(name)
        if not recs:
            continue
        for group in ("original14", "holdout15"):
            print(f"\n== {v} facets, {group}")
            print(f"{'circuit':16s} {'expect':28s} {'got':28s} computed")
            tot = {"trig": [0, 0], "cond": [0, 0], "both": [0, 0], "n": 0}
            for key, e in exp[group].items():
                r = recs.get(key)
                if not r:
                    continue
                g = r["result"]
                t_ok = g["trigger"] == e["trigger"] or g["trigger"] in e.get("alt_trigger", [])
                c_ok = g["condition"] == e["condition"] or g["condition"] in e.get("alt_condition", [])
                t_ex, c_ex = g["trigger"] == e["trigger"], g["condition"] == e["condition"]
                tot["trig"][0] += t_ex; tot["trig"][1] += t_ok
                tot["cond"][0] += c_ex; tot["cond"][1] += c_ok
                tot["both"][0] += t_ex and c_ex; tot["both"][1] += t_ok and c_ok
                tot["n"] += 1
                mark = lambda ex, ok: "*" if ex else ("~" if ok else " ")
                print(f"{key:16s} {e['trigger'] + ' + ' + e['condition']:28s} "
                      f"{g['trigger'] + mark(t_ex, t_ok) + ' + ' + g['condition'] + mark(c_ex, c_ok):28s} "
                      f"{'yes' if r.get('trigger_computed') else ''} ({r.get('merged_consistency')})")
            n = tot["n"]
            print(f"of {n} (exact, exact-or-alt):  trigger {tot['trig'][0]}, {tot['trig'][1]}   "
                  f"condition {tot['cond'][0]}, {tot['cond'][1]}   both {tot['both'][0]}, {tot['both'][1]}")
        outs = [r.get("output_tokens") or 0 for r in recs.values()]
        ins = [r.get("input_tokens") or 0 for r in recs.values()]
        print(f"{v}: mean input {sum(ins) / len(ins):.0f}, mean output {sum(outs) / len(outs):.0f} tokens")


if __name__ == "__main__":
    if "--facets" in sys.argv:
        sys.stdout.reconfigure(encoding="utf-8")
        facets()
    else:
        main()
