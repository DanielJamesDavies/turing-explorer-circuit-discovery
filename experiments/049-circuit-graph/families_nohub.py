"""FAMILIES WITHOUT THE INFRASTRUCTURE: the first pass (graph_analysis.py)
found 4 families whose cores were all L0 hub latents (one in 99% of
circuits). Here hubs (fan-out >= HUB_FRAC of circuits) are removed from
the membership matrix before overlap, so families are defined by the
SPECIFIC members circuits share, and Leiden runs at a finer resolution.
Also lists the near-duplicate pairs (jaccard >= 0.5 on full membership).

  python experiments/049-circuit-graph/families_nohub.py
Env: HUB_FRAC (0.05), LIFT_MIN (3), OVERLAP_MIN (4), RES (2.0), CORE_FRAC (0.5), TOP (30)
"""
import json
import os
from collections import Counter
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).parent
T = Path(os.environ.get("TABLES", str(HERE / "tables")))
OUT = Path(os.environ.get("OUT", str(HERE / "results")))
HUB_FRAC = float(os.environ.get("HUB_FRAC", 0.05))
LIFT_MIN = float(os.environ.get("LIFT_MIN", 3.0))
OVERLAP_MIN = int(os.environ.get("OVERLAP_MIN", 4))
RES = float(os.environ.get("RES", 2.0))
CORE_FRAC = float(os.environ.get("CORE_FRAC", 0.5))
TOP = int(os.environ.get("TOP", 30))

M = pd.read_parquet(T / "members.parquet")
C = pd.read_parquet(T / "circuits.parquet")
N = len(C)
M["lkey"] = M["layer"].astype(str) + "." + M["kind"].astype(str) + "." + M["index"].astype(str)
C["skey"] = C["seed_layer"].astype(str) + "." + C["seed_kind"].astype(str) + "." + C["seed_index"].astype(str)
lat_ids = {k: i for i, k in enumerate(pd.unique(M["lkey"]))}
lk = np.array(list(lat_ids.keys()))
M["lid"] = M["lkey"].map(lat_ids).astype(np.int64)
X = sp.csr_matrix((np.ones(len(M), dtype=np.float32), (M["cid"].values, M["lid"].values)), shape=(N, len(lat_ids)))
X.data[:] = 1.0
fanout = np.asarray(X.sum(0)).ravel()
fanin_full = np.asarray(X.sum(1)).ravel()

# ---- near-duplicates on FULL membership --------------------------------------
O = (X @ X.T).tocsr(); O.setdiag(0); O.eliminate_zeros(); Oc = O.tocoo()
jac = Oc.data / (fanin_full[Oc.row] + fanin_full[Oc.col] - Oc.data)
dup = pd.DataFrame({"a": Oc.row, "b": Oc.col, "overlap": Oc.data, "jaccard": jac})
dup = dup[(dup["a"] < dup["b"]) & (dup["jaccard"] >= 0.5)].sort_values("jaccard", ascending=False)
sk = C.set_index("cid")["skey"]
print("NEAR-DUPLICATE PAIRS (jaccard >= 0.5): %d" % len(dup))
dup["a"] = dup["a"].astype(int); dup["b"] = dup["b"].astype(int)
same_idx = sum(1 for _, r in dup.iterrows() if sk[int(r.a)].split(".")[2] == sk[int(r.b)].split(".")[2])
same_layer = sum(1 for _, r in dup.iterrows() if sk[int(r.a)].split(".")[0] == sk[int(r.b)].split(".")[0])
print("  same latent index (resid/mlp twins of one feature): %d | same layer: %d" % (same_idx, same_layer))
for _, r in dup.head(15).iterrows():
    print("    %-16s ~ %-16s | shared %4d | jaccard %.2f | sizes %d/%d" % (sk[int(r.a)], sk[int(r.b)], r.overlap, r.jaccard, fanin_full[int(r.a)], fanin_full[int(r.b)]))

# ---- hub-excluded families --------------------------------------------------
hub = fanout >= HUB_FRAC * N
print("\nHUBS excluded (fan-out >= %.0f%% of circuits): %d latents holding %.1f%% of all membership; by layer %s"
      % (100 * HUB_FRAC, hub.sum(), 100 * fanout[hub].sum() / fanout.sum(),
         dict(sorted(Counter(int(s.split(".")[0]) for s in lk[hub]).items()))))
Xs = X[:, ~hub].tocsr()
lks = lk[~hub]
fanin = np.asarray(Xs.sum(1)).ravel()
fo = np.asarray(Xs.sum(0)).ravel()
print("  members per circuit after exclusion: median %d (was %d) | circuits left with < 3 specific members: %d"
      % (np.median(fanin), np.median(fanin_full), (fanin < 3).sum()))
O = (Xs @ Xs.T).tocsr(); O.setdiag(0); O.eliminate_zeros(); Oc = O.tocoo()
E_tot = Xs.sum(); s2 = float((fo.astype(np.float64) ** 2).sum())
exp = fanin[Oc.row] * fanin[Oc.col] * s2 / (E_tot ** 2)
lift = Oc.data / np.maximum(exp, 1e-9)
pairs = pd.DataFrame({"a": Oc.row, "b": Oc.col, "overlap": Oc.data, "lift": lift})
pairs = pairs[pairs["a"] < pairs["b"]]
print("  pairs sharing >= 1 specific member: %d (%.1f%% of all pairs) | lift median %.2f p90 %.2f p99 %.2f"
      % (len(pairs), 100 * len(pairs) / (N * (N - 1) // 2), pairs["lift"].median(), pairs["lift"].quantile(0.9), pairs["lift"].quantile(0.99)))
strong = pairs[(pairs["lift"] >= LIFT_MIN) & (pairs["overlap"] >= OVERLAP_MIN)]
print("  strong pairs (lift >= %.0f, overlap >= %d): %d touching %d circuits"
      % (LIFT_MIN, OVERLAP_MIN, len(strong), len(set(strong["a"]) | set(strong["b"]))))
g = ig.Graph(n=N, edges=list(zip(strong["a"].astype(int), strong["b"].astype(int))))
g.es["weight"] = np.log(strong["lift"].values + 1.0).tolist()
part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=RES, seed=0)
fam = np.array(part.membership)
sizes = Counter(fam)
big = [f for f, n in sizes.most_common() if n >= 5]
cov = sum(sizes[f] for f in big)
print("\nFAMILIES (hub-excluded, resolution %.1f): %d of size >= 5 covering %d circuits (%.1f%%); size distribution: >=100: %d, 20-99: %d, 5-19: %d"
      % (RES, len(big), cov, 100 * cov / N, sum(sizes[f] >= 100 for f in big), sum(20 <= sizes[f] < 100 for f in big), sum(5 <= sizes[f] < 20 for f in big)))
C["family"] = fam
rows = []
for f in big[:TOP]:
    cids = C.loc[C["family"] == f, "cid"].values
    pres = np.asarray(Xs[cids].mean(0)).ravel()
    core = np.where(pres >= CORE_FRAC)[0]
    if len(core) < 3:                       # big heterogeneous families: fall back to the 8 most-shared latents
        core = np.argsort(-pres)[:8]
    core_l = sorted(lks[core], key=lambda s: (int(s.split(".")[0]), s))
    top_pres = [(str(lks[j]), round(float(pres[j]), 2)) for j in np.argsort(-pres)[:8]]
    seeds = C.loc[C["family"] == f]
    lay = seeds["seed_layer"].value_counts().sort_index().to_dict()
    kinds = seeds["seed_kind"].value_counts().to_dict()
    cl = dict(sorted(Counter(int(s.split(".")[0]) for s in core_l).items()))
    ck = Counter(s.split(".")[1] for s in core_l)
    rows.append(dict(family=int(f), n=int(len(cids)), seed_layers=lay, seed_kinds=kinds, n_core=int(len(core)),
                     core_layers=cl, core_kinds=dict(ck), core=core_l[:80], top_presence=top_pres, seeds_sample=seeds["skey"].head(10).tolist()))
    print("  fam %3d | %4d circuits | seeds L%s kinds %s | CORE %3d latents (layers %s, kinds %s) | e.g. %s"
          % (f, len(cids), lay, kinds, len(core), cl, dict(ck), " ".join("%s@%.2f" % t for t in top_pres[:5])))
json.dump(rows, open(OUT / "families_nohub.json", "w"), indent=1)
C[["cid", "skey", "seed_layer", "seed_kind", "n_members", "family"]].to_csv(OUT / "circuit_families_nohub.csv", index=False)
# family purity by seed layer/kind: are families "same seed layer" groups or cross-layer mechanisms?
pur = []
for f in big:
    s = C.loc[C["family"] == f]
    pur.append((s["seed_layer"].value_counts(normalize=True).iloc[0], s["seed_kind"].value_counts(normalize=True).iloc[0]))
pur = np.array(pur)
print("\n  family purity (share of the dominant seed layer / kind): layer median %.2f | kind median %.2f | families spanning >= 3 seed layers: %d"
      % (np.median(pur[:, 0]), np.median(pur[:, 1]), sum(1 for f in big if C.loc[C['family'] == f, 'seed_layer'].nunique() >= 3)))
print("->", OUT / "families_nohub.json")
