"""CASE STUDY ON THE PRODUCTION CIRCUIT: 063/case_study.py's four measurements (group shares, knock-outs vs
contribution-rank-matched random removals, group edges, probe sentences + year sweep), run unchanged on the
production (WCM v4) circuit and the fresh records.

Groups: by default the 063 role groups of 9.resid.20419, restricted to the members still present in the v4 circuit
(the missing ones are listed in the output), so the comparison asks "does the OLD readable core still carry the NEW
circuit?". GROUPS_JSON=<file> gives other groups ({"name": [members], ...}, "_story": [names], "_probes": {...},
"_years": 0|1) for another target.

  PYTHONPATH=src python -X utf8 experiments/110-case-studies-v4/case_v4.py
  env: KEY (9.resid.20419)  GROUPS_JSON  N_DRAWS (8)  RANK_WIN (15)  SKIP_KO=1  SKIP_EDGES=1
  -> out/case/<KEY>.json
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib110 as L  # noqa: E402

L.use_production()
os.environ.setdefault("KEY", "9.resid.20419")
import case_study as CS  # noqa: E402  (063; reads OUT / KEY from the environment set above)

KEY = CS.KEY
CS.MEMBERS = L.RESULTS / "members" / ("%s.csv" % KEY)


def main():
    present = {"%d.%s.%d" % (l, k, i) for l, k, i, _ in L.members(L.load_circuit(KEY))}
    if os.environ.get("GROUPS_JSON"):
        spec = json.loads(Path(os.environ["GROUPS_JSON"]).read_text(encoding="utf-8"))
        groups = {g: ms for g, ms in spec.items() if not g.startswith("_")}
        CS.STORY = spec["_story"]
        if "_probes" in spec:
            CS.PROBES = spec["_probes"]
        if not spec.get("_years", 0):
            CS.YEARS_BC, CS.YEARS_AD = [], []
    else:
        groups = CS.GROUPS
    missing = {g: [m for m in ms if m not in present] for g, ms in groups.items()}
    CS.GROUPS = {g: [m for m in ms if m in present] for g, ms in groups.items()}
    CS.GROUPS = {g: ms for g, ms in CS.GROUPS.items() if ms}
    CS.STORY = [g for g in CS.STORY if g in CS.GROUPS]
    t0 = time.time()
    C = CS.Case()
    out = dict(key=KEY, circuit="production v4 (094 out_v2/main)", records="fresh (ctx_fresh)",
               n_members=len(present), missing_from_v4=missing, groups=C.shares(), story=CS.STORY)
    out["probes"] = C.probes(); print("probes done %.0fs" % (time.time() - t0), flush=True)
    if not os.environ.get("SKIP_EDGES"):
        out["edges"] = C.edges(); print("edges done %.0fs" % (time.time() - t0), flush=True)
    if not os.environ.get("SKIP_KO"):
        out["knockouts"] = C.knockouts(); print("knock-outs done %.0fs" % (time.time() - t0), flush=True)
    d = L.RESULTS / "case"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("%s.json" % KEY)).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", d / ("%s.json" % KEY), flush=True)


if __name__ == "__main__":
    main()
