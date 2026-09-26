"""STABILITY (item 8 without arm two): split the circuits by shard parity
(even vs odd shards = two independent samples of seeds with the same
layer profile) and ask which graph statistics replicate:
  - fan-out per latent (Spearman across halves; hub-set Jaccard at top-K)
  - latent-latent co-membership lift for pairs (does a pair that co-occurs
    above chance in half A do so in half B?)
  - family cores: rerun the hub-excluded family pipeline on each half and
    measure core-set overlap (best-match Jaccard per core)

  python experiments/049-circuit-graph/stability.py
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

HERE = Path(__file__).parent
T = Path(os.environ.get("TABLES", str(HERE / "tables"))); R = Path(os.environ.get("OUT", str(HERE / "results")))
R.mkdir(parents=True, exist_ok=True)
HUB_FRAC, LIFT_MIN, OVERLAP_MIN, RES, CORE_FRAC = 0.05, 3.0, 4, 2.0, 0.5
M = pd.read_parquet(T / "members.parquet"); C = pd.read_parquet(T / "circuits.parquet")
M["lkey"] = M["layer"].astype(str) + "." + M["kind"].astype(str) + "." + M["index"].astype(str)
lat_ids = {k: i for i, k in enumerate(pd.unique(M["lkey"]))}; lk = np.array(list(lat_ids.keys()))
M["lid"] = M["lkey"].map(lat_ids).astype(np.int64)
half_of = dict(zip(C["cid"], C["shard"] % 2))
M["half"] = M["cid"].map(half_of)
N = len(C); L = len(lat_ids)
X = sp.csr_matrix((np.ones(len(M), dtype=np.float32), (M["cid"].values, M["lid"].values)), shape=(N, L)); X.data[:] = 1
halves = {h: np.array(sorted(C.loc[C["shard"] % 2 == h, "cid"])) for h in (0, 1)}
print("halves: %d / %d circuits" % (len(halves[0]), len(halves[1])))
report = {}

# ---- fan-out replication -------------------------------------------------------
fo = {h: np.asarray(X[halves[h]].sum(0)).ravel() for h in (0, 1)}
both = (fo[0] > 0) | (fo[1] > 0)
rho = spearmanr(fo[0][both], fo[1][both]).correlation
print("fan-out: Spearman across halves (latents present in either) %.3f" % rho)
for K in (50, 200, 1000):
    a = set(np.argsort(-fo[0])[:K]); b = set(np.argsort(-fo[1])[:K])
    print("  top-%d hub set Jaccard: %.2f" % (K, len(a & b) / len(a | b)))
report["fanout_spearman"] = float(rho)
report["hub_jaccard"] = {K: float(len(set(np.argsort(-fo[0])[:K]) & set(np.argsort(-fo[1])[:K])) / len(set(np.argsort(-fo[0])[:K]) | set(np.argsort(-fo[1])[:K]))) for K in (50, 200, 1000)}

# ---- pair co-membership lift replication (non-hub latents, pairs with >= 5 co-memberships in half 0)
hub = (fo[0] + fo[1]) >= HUB_FRAC * N
keep = np.where(~hub & ((fo[0] + fo[1]) >= 10))[0]
print("\npair lift replication on %d non-hub latents with >= 10 memberships" % len(keep))
def pair_lift(h):
    Xh = X[halves[h]][:, keep].tocsc()
    n = Xh.shape[0]; d = np.asarray(Xh.sum(0)).ravel()
    co = (Xh.T @ Xh).tocoo()
    m = (co.row < co.col) & (co.data >= 3)
    exp = d[co.row[m]] * d[co.col[m]] / n
    return pd.DataFrame({"i": co.row[m], "j": co.col[m], "co": co.data[m], "lift": co.data[m] / np.maximum(exp, 1e-9)})
p0, p1 = pair_lift(0), pair_lift(1)
j = p0.merge(p1, on=["i", "j"], suffixes=("_0", "_1"))
print("  pairs with >= 3 co-memberships in both halves: %d (of %d / %d) | Spearman(lift0, lift1) = %.3f"
      % (len(j), len(p0), len(p1), spearmanr(j["lift_0"], j["lift_1"]).correlation if len(j) > 2 else float("nan")))
s0 = p0[p0["lift"] >= 5];
rep = j[(j["lift_0"] >= 5)]
print("  strong pairs in half 0 (lift >= 5, co >= 3): %d | of those also present (co >= 3) in half 1: %d (%.1f%%) | with lift >= 5 in half 1: %d (%.1f%%)"
      % (len(s0), len(rep), 100 * len(rep) / max(1, len(s0)), (rep["lift_1"] >= 5).sum(), 100 * (rep["lift_1"] >= 5).sum() / max(1, len(s0))))
report["pair_lift_spearman"] = float(spearmanr(j["lift_0"], j["lift_1"]).correlation) if len(j) > 2 else None
report["strong_pair_replication"] = dict(n_strong0=int(len(s0)), present1=int(len(rep)), strong1=int((rep["lift_1"] >= 5).sum()))

# ---- family core replication ---------------------------------------------------
def families(h):
    cids = halves[h]; Xh = X[cids][:, ~hub].tocsr(); n = Xh.shape[0]
    fanin = np.asarray(Xh.sum(1)).ravel(); fo_ = np.asarray(Xh.sum(0)).ravel()
    O = (Xh @ Xh.T).tocsr(); O.setdiag(0); O.eliminate_zeros(); Oc = O.tocoo()
    exp = fanin[Oc.row] * fanin[Oc.col] * float((fo_.astype(np.float64) ** 2).sum()) / (Xh.sum() ** 2)
    lift = Oc.data / np.maximum(exp, 1e-9)
    m = (Oc.row < Oc.col) & (lift >= LIFT_MIN) & (Oc.data >= OVERLAP_MIN)
    g = ig.Graph(n=n, edges=list(zip(Oc.row[m].astype(int), Oc.col[m].astype(int)))); g.es["weight"] = np.log(lift[m] + 1).tolist()
    fam = np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=RES, seed=0).membership)
    cores = []
    lks = lk[~hub]
    for f, cnt in Counter(fam).most_common():
        if cnt < 10:
            break
        pres = np.asarray(Xh[np.where(fam == f)[0]].mean(0)).ravel()
        core = set(lks[np.where(pres >= CORE_FRAC)[0]])
        if len(core) >= 3:
            cores.append((cnt, core))
    return cores
c0, c1 = families(0), families(1)
print("\nfamily cores (>= 3 latents, families >= 10 circuits): half 0 has %d, half 1 has %d" % (len(c0), len(c1)))
best = []
for cnt, core in c0:
    bj = max((len(core & c) / len(core | c) for _, c in c1), default=0.0)
    best.append(bj)
if best:
    print("  best-match core Jaccard, half0 -> half1: median %.2f | frac >= 0.5: %.2f | frac == 0: %.2f | n=%d"
          % (np.median(best), np.mean(np.array(best) >= 0.5), np.mean(np.array(best) == 0), len(best)))
    for (cnt, core), bj in sorted(zip(c0, best), key=lambda x: -x[1])[:8]:
        print("    core (%d circuits) %s ... best Jaccard %.2f" % (cnt, " ".join(sorted(core)[:5]), bj))
report["core_replication"] = dict(n0=len(c0), n1=len(c1), best_jaccard=[float(b) for b in best])
json.dump(report, open(R / "stability_report.json", "w"), indent=1)
print("->", R / "stability_report.json")
