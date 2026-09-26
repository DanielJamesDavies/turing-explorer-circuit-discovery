"""FLATTEN the production shard stores into two tables for graph analysis.

  members.parquet  one row per (circuit, member node):
      cid, seed_layer, seed_kind, seed_index, layer, kind, index, role,
      amplitude, attribution, edge_weight
  circuits.parquet one row per circuit:
      cid, shard, seed_layer, seed_kind, seed_index, n_members,
      cf_faith, posctx_sup, ablation_sup, amp_* (amp_stats), target_loss,
      target_pre_act

  PYTHONPATH=src python experiments/049-circuit-graph/extract_tables.py [DATA_DIR]
"""
import glob
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
OUT = Path(os.environ.get("OUT", str(HERE / "tables")))
OUT.mkdir(parents=True, exist_ok=True)

mrows, crows = [], []
cid = 0
t0 = time.time()
for p in sorted(glob.glob(str(DATA / "discovered_circuits.shard*.pt"))):
    shard = int(p.rsplit("shard", 1)[1].split(".")[0])
    cs = torch.load(p, weights_only=False, map_location="cpu")
    for c in cs.values():
        md = c.metadata
        seed = next(n for n in c.nodes.values() if n.metadata.get("role") == "seed")
        sf = seed.metadata["feature_id"]
        # edge weights keyed by target node uuid (seed -> member)
        ew = {}
        edges = c.edges.values() if isinstance(c.edges, dict) else c.edges
        for e in edges:
            ew[e.target_uuid] = float((e.metadata or {}).get("weight", float("nan")))
            ew.setdefault(e.source_uuid, float("nan"))
        n_members = 0
        for n in c.nodes.values():
            if n is seed:
                continue
            f = n.metadata["feature_id"]
            mrows.append((cid, sf.layer, sf.kind, sf.index, f.layer, f.kind, f.index,
                          n.metadata.get("role", ""), float(n.metadata.get("amplitude", float("nan"))),
                          float(n.metadata.get("attribution_score", float("nan"))),
                          ew.get(n.uuid, float("nan"))))
            n_members += 1
        ev = md.get("evals") or {}
        am = md.get("amp_stats") or {}
        crows.append(dict(cid=cid, shard=shard, seed_layer=sf.layer, seed_kind=sf.kind, seed_index=sf.index,
                          n_members=n_members,
                          cf_faith=ev.get("counterfactual_faithfulness"), posctx_sup=ev.get("posctx_suppression_score"),
                          ablation_sup=ev.get("ablation_suppression_score"),
                          amp_median=am.get("median"), amp_p10=am.get("p10"), amp_p90=am.get("p90"), amp_max=am.get("max"),
                          amp_frac_elevated=am.get("frac_elevated"), amp_frac_reduced=am.get("frac_reduced"),
                          target_loss=md.get("target_loss"), target_pre_act=md.get("target_pre_act")))
        cid += 1
    print("shard %2d: %4d circuits | running total %5d circuits, %8d member rows | %.0fs"
          % (shard, len(cs), cid, len(mrows), time.time() - t0), flush=True)

members = pd.DataFrame(mrows, columns=["cid", "seed_layer", "seed_kind", "seed_index", "layer", "kind", "index",
                                       "role", "amplitude", "attribution", "edge_weight"])
for col in ("seed_kind", "kind", "role"):
    members[col] = members[col].astype("category")
circuits = pd.DataFrame(crows)
try:
    members.to_parquet(OUT / "members.parquet", index=False)
    circuits.to_parquet(OUT / "circuits.parquet", index=False)
    fmt = "parquet"
except Exception as e:  # no pyarrow
    members.to_pickle(OUT / "members.pkl")
    circuits.to_pickle(OUT / "circuits.pkl")
    fmt = "pickle (%s)" % e.__class__.__name__
print("\nwrote %s: %d circuits, %d member rows -> %s" % (fmt, len(circuits), len(members), OUT))
print(circuits.describe().T[["mean", "50%", "min", "max"]].to_string())
print("\nmembers by kind:", members["kind"].value_counts().to_dict())
print("roles:", members["role"].value_counts().to_dict())
