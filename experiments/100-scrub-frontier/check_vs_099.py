"""Equivalence check: 100's re-scoring of 099's reused production circuits (arm wcm_1e-3) against 099's own outputs.

Same donors (indices), same per-draw anchor means for empty / fit (099 'wcm') / a1 ('wcm_a1') / pinned ('wcm_pinned'),
same production scores. Writes <run>/check_vs_099.json. No GPU.

  python experiments/100-scrub-frontier/check_vs_099.py --run experiments/100-scrub-frontier/out/<dir>
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
T99 = HERE.parent / "099-recursive-causal-scrubbing" / "out" / "panel18" / "targets"
MAP = {"fit": "wcm", "a1": "wcm_a1", "pinned": "wcm_pinned"}
PROD = ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "free0_pre", "freeM_topk_pre",
        "freeN_topk_pre", "a_pos_tk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    out = []
    for p in sorted((args.run / "targets").glob("*.wcm_1e-3.json")):
        new = json.loads(p.read_text())
        key = new["seed"]
        old = json.loads((T99 / (key + ".json")).read_text())
        row = next(json.loads(l) for l in open(args.run / "rows.jsonl") if json.loads(l)["seed"] == key
                   and json.loads(l)["arm"] == "wcm_1e-3")
        res = dict(seed=key)
        od = {(r["background"], r["draw"], r["variant"]): r for r in old["donor_results"]}
        res["donors_identical"] = all(e["donor_indices"] == od[(e["background"], e["draw"], "empty")]["donor_indices"]
                                      for e in new["empty"])
        diffs = {"empty": max(abs(e["mean"] - od[(e["background"], e["draw"], "empty")]["anchor"]["mean"])
                              for e in new["empty"])}
        for v, ov in MAP.items():
            diffs[v] = max(abs(r["mean"] - od[(r["background"], r["draw"], ov)]["anchor"]["mean"])
                           for r in new["variants"][v])
        res["max_abs_anchor_mean_diff"] = diffs
        res["prod_max_abs_diff"] = max(abs(row["prod"][k] - old["production"]["wcm"][k]) for k in PROD
                                       if row["prod"].get(k) is not None and old["production"]["wcm"].get(k) is not None)
        out.append(res)
        print(json.dumps(res))
    (args.run / "check_vs_099.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
