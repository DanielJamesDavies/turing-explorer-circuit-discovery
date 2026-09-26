"""CIRCUIT-LEVEL GRAPH over the production circuits (circuits as nodes).

Inputs: tables/members.parquet, tables/circuits.parquet (extract_tables.py).

1. FAN-OUT of a latent = number of circuits it is a member of; FAN-IN of a
   circuit = its member count. Distributions, hubs, hub composition.
2. SEED->SEED graph: edge A->B when A's seed latent is a member of B's
   circuit. Degrees, layer ordering, longest chains (DAG by layer).
3. FREQUENCY-CORRECTED OVERLAP between circuits: observed shared members
   vs the bipartite configuration-model expectation (bigger circuits and
   ubiquitous latents overlap by chance) -> lift. Density-artefact-safe.
4. FAMILIES: Leiden communities on the lift-thresholded circuit graph;
   per family, the shared CORE (members present in >= CORE_FRAC of the
   family) = candidate "organ".

  python experiments/049-circuit-graph/graph_analysis.py
Env: LIFT_MIN (3), OVERLAP_MIN (5), CORE_FRAC (0.5), TOP (40)
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).parent
T = Path(os.environ.get("TABLES", str(HERE / "tables")))
OUT = Path(os.environ.get("OUT", str(HERE / "results")))
OUT.mkdir(parents=True, exist_ok=True)
LIFT_MIN = float(os.environ.get("LIFT_MIN", 3.0))
OVERLAP_MIN = int(os.environ.get("OVERLAP_MIN", 5))
CORE_FRAC = float(os.environ.get("CORE_FRAC", 0.5))
TOP = int(os.environ.get("TOP", 40))

M = pd.read_parquet(T / "members.parquet")
C = pd.read_parquet(T / "circuits.parquet")
N = len(C)
print("circuits %d | member rows %d | seeds by layer %s" % (N, len(M), C["seed_layer"].value_counts().sort_index().to_dict()))
print("seeds by kind %s" % C["seed_kind"].value_counts().to_dict())
report = {}

# ---- latent ids ------------------------------------------------------------
M["lkey"] = M["layer"].astype(str) + "." + M["kind"].astype(str) + "." + M["index"].astype(str)
C["skey"] = C["seed_layer"].astype(str) + "." + C["seed_kind"].astype(str) + "." + C["seed_index"].astype(str)
lat_ids = {k: i for i, k in enumerate(pd.unique(M["lkey"]))}
M["lid"] = M["lkey"].map(lat_ids).astype(np.int64)
L = len(lat_ids)
X = sp.csr_matrix((np.ones(len(M), dtype=np.float32), (M["cid"].values, M["lid"].values)), shape=(N, L))
X.data[:] = 1.0  # duplicates -> binary
print("unique member latents %d (of a %d-latent dictionary sample)" % (L, 12 * 3 * 16384))

# ---- 1. fan-out / fan-in ---------------------------------------------------
fanout = np.asarray(X.sum(0)).ravel()
fanin = np.asarray(X.sum(1)).ravel()
lk = np.array(list(lat_ids.keys()))
fo = pd.DataFrame({"latent": lk, "fanout": fanout})
fo[["layer", "kind", "index"]] = fo["latent"].str.split(".", expand=True)
fo["layer"] = fo["layer"].astype(int)
fo["frac_of_circuits"] = fo["fanout"] / N
fo = fo.sort_values("fanout", ascending=False)
q = np.percentile(fanout, [50, 90, 99, 99.9])
print("\nFAN-OUT (circuits per member latent): median %d | p90 %d | p99 %d | p99.9 %d | max %d"
      % (q[0], q[1], q[2], q[3], fanout.max()))
print("  latents in exactly 1 circuit: %.1f%% | in >= 100 circuits: %d | in >= 10%% of circuits: %d"
      % (100 * (fanout == 1).mean(), (fanout >= 100).sum(), (fanout >= 0.1 * N).sum()))
print("  share of all membership held by the top 1%% of latents: %.1f%%"
      % (100 * np.sort(fanout)[::-1][: max(1, L // 100)].sum() / fanout.sum()))
print("\n  TOP %d fan-out latents (hubs):" % TOP)
for _, r in fo.head(TOP).iterrows():
    print("    %-16s in %5d circuits (%.1f%%)" % (r["latent"], r["fanout"], 100 * r["frac_of_circuits"]))
hub_kind = fo.head(200).groupby(["layer", "kind"]).size()
print("  top-200 hubs by (layer, kind):", {("L%d.%s" % k): int(v) for k, v in hub_kind.items()})
print("\nFAN-IN (members per circuit): median %d | p10 %d | p90 %d | max %d"
      % (np.median(fanin), np.percentile(fanin, 10), np.percentile(fanin, 90), fanin.max()))
print("  by seed layer:", C.groupby("seed_layer")["n_members"].median().astype(int).to_dict())
print("  by seed kind :", C.groupby("seed_kind")["n_members"].median().astype(int).to_dict())
fo.to_csv(OUT / "fanout_latents.csv", index=False)
report["fanout"] = dict(median=float(q[0]), p90=float(q[1]), p99=float(q[2]), p999=float(q[3]), max=int(fanout.max()),
                        frac_singletons=float((fanout == 1).mean()), n_ge100=int((fanout >= 100).sum()),
                        top1pct_share=float(np.sort(fanout)[::-1][: max(1, L // 100)].sum() / fanout.sum()))
report["fanin"] = dict(median=float(np.median(fanin)), p10=float(np.percentile(fanin, 10)), p90=float(np.percentile(fanin, 90)))

# ---- layer ordering of membership -------------------------------------------
M["dl"] = M["seed_layer"] - M["layer"]
print("\nLAYER ORDERING: member layer relative to seed layer (seed - member): "
      "same-layer %.1f%% | below %.1f%% | above %.1f%%"
      % (100 * (M["dl"] == 0).mean(), 100 * (M["dl"] > 0).mean(), 100 * (M["dl"] < 0).mean()))
print("  mean depth of members below seed: %.2f layers | share of members within 1 layer: %.1f%%"
      % (M.loc[M["dl"] > 0, "dl"].mean(), 100 * (M["dl"].abs() <= 1).mean()))

# ---- 2. seed -> seed graph ---------------------------------------------------
seed_of = dict(zip(C["skey"], C["cid"]))
Ms = M[M["lkey"].isin(seed_of)]
edges = pd.DataFrame({"src": Ms["lkey"].map(seed_of).values, "dst": Ms["cid"].values,
                      "amplitude": Ms["amplitude"].values, "attribution": Ms["attribution"].values})
edges = edges[edges["src"] != edges["dst"]].drop_duplicates(["src", "dst"])
print("\nSEED->SEED GRAPH: %d seeds are members of other seeds' circuits; %d edges (%.2f per circuit)"
      % (edges["src"].nunique(), len(edges), len(edges) / N))
outdeg = edges.groupby("src").size(); indeg = edges.groupby("dst").size()
print("  out-degree: median %d p90 %d max %d | in-degree: median %d p90 %d max %d | circuits with no seed-members: %d"
      % (outdeg.median(), outdeg.quantile(0.9), outdeg.max(), indeg.median(), indeg.quantile(0.9), indeg.max(),
         N - indeg.index.nunique()))
import networkx as nx
G = nx.DiGraph()
G.add_nodes_from(range(N))
G.add_edges_from(zip(edges["src"], edges["dst"]))
cyc = not nx.is_directed_acyclic_graph(G)
if cyc:
    # same-layer members create 2-cycles; break by keeping only strictly-lower-layer edges for the chain analysis
    lay = C.set_index("cid")["seed_layer"]
    e2 = edges[lay.loc[edges["src"]].values < lay.loc[edges["dst"]].values]
    Gd = nx.DiGraph(); Gd.add_nodes_from(range(N)); Gd.add_edges_from(zip(e2["src"], e2["dst"]))
    print("  graph has cycles (same-layer membership); chain analysis on the %d strictly-upward edges" % len(e2))
else:
    Gd = G
lp = nx.dag_longest_path(Gd)
depth = {}
for n in nx.topological_sort(Gd):
    depth[n] = max((depth[p] + 1 for p in Gd.predecessors(n)), default=0)
dv = np.array(list(depth.values()))
print("  longest chain: %d circuits deep: %s" % (len(lp), " -> ".join(C.loc[C["cid"].isin(lp)].set_index("cid").loc[lp, "skey"].tolist()[:12])))
print("  chain depth distribution: depth0 %.1f%% | depth1 %.1f%% | depth2 %.1f%% | depth>=3 %.1f%%"
      % (100 * (dv == 0).mean(), 100 * (dv == 1).mean(), 100 * (dv == 2).mean(), 100 * (dv >= 3).mean()))
edges.to_csv(OUT / "seed_seed_edges.csv", index=False)
report["seed_graph"] = dict(n_edges=int(len(edges)), longest_chain=int(len(lp)), cyclic=bool(cyc),
                            outdeg_max=int(outdeg.max()), indeg_max=int(indeg.max()))

# ---- 3. frequency-corrected overlap ----------------------------------------
O = (X @ X.T).tocsr()            # observed shared members
O.setdiag(0); O.eliminate_zeros()
E_tot = X.sum()
s2 = float((fanout.astype(np.float64) ** 2).sum())
# configuration-model expectation for a bipartite graph: E[O_AB] = d_A d_B * sum_l d_l^2 / E^2
Oc = O.tocoo()
exp = fanin[Oc.row] * fanin[Oc.col] * s2 / (E_tot ** 2)
lift = Oc.data / np.maximum(exp, 1e-9)
jac = Oc.data / (fanin[Oc.row] + fanin[Oc.col] - Oc.data)
pairs = pd.DataFrame({"a": Oc.row, "b": Oc.col, "overlap": Oc.data, "expected": exp, "lift": lift, "jaccard": jac})
pairs = pairs[pairs["a"] < pairs["b"]]
print("\nOVERLAP (pairs sharing >= 1 member): %d of %d possible pairs (%.1f%%)"
      % (len(pairs), N * (N - 1) // 2, 100 * len(pairs) / (N * (N - 1) // 2)))
print("  jaccard: median %.3f p90 %.3f p99 %.3f | lift: median %.2f p90 %.2f p99 %.2f"
      % (pairs["jaccard"].median(), pairs["jaccard"].quantile(0.9), pairs["jaccard"].quantile(0.99),
         pairs["lift"].median(), pairs["lift"].quantile(0.9), pairs["lift"].quantile(0.99)))
strong = pairs[(pairs["lift"] >= LIFT_MIN) & (pairs["overlap"] >= OVERLAP_MIN)]
print("  pairs with lift >= %.0f and overlap >= %d: %d (%.2f%% of sharing pairs) touching %d circuits"
      % (LIFT_MIN, OVERLAP_MIN, len(strong), 100 * len(strong) / max(1, len(pairs)), len(set(strong["a"]) | set(strong["b"]))))
hi_j = pairs[pairs["jaccard"] >= 0.5]
print("  pairs with jaccard >= 0.5 (near-duplicate circuits): %d" % len(hi_j))
# how much of raw overlap is density: correlation of overlap with size product
print("  corr(overlap, |A|*|B|) = %.3f  (the density artefact) | corr(lift, |A|*|B|) = %.3f"
      % (np.corrcoef(pairs["overlap"], fanin[pairs["a"]] * fanin[pairs["b"]])[0, 1],
         np.corrcoef(pairs["lift"], fanin[pairs["a"]] * fanin[pairs["b"]])[0, 1]))
report["overlap"] = dict(n_sharing_pairs=int(len(pairs)), jaccard_median=float(pairs["jaccard"].median()),
                         lift_median=float(pairs["lift"].median()), n_strong=int(len(strong)), n_near_dup=int(len(hi_j)))

# ---- 4. families (Leiden on the lift graph) ---------------------------------
import igraph as ig
import leidenalg as la
g = ig.Graph(n=N, edges=list(zip(strong["a"].astype(int), strong["b"].astype(int))))
g.es["weight"] = np.log(strong["lift"].values + 1.0).tolist()
part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=1.0, seed=0)
fam = np.array(part.membership)
sizes = Counter(fam)
big = [f for f, n in sizes.most_common() if n >= 5]
print("\nFAMILIES (Leiden on lift >= %.0f graph): %d communities of size >= 5 covering %d circuits; %d circuits isolated/tiny"
      % (LIFT_MIN, len(big), sum(sizes[f] for f in big), N - sum(sizes[f] for f in big)))
C["family"] = fam
fam_rows = []
for f in big[:TOP]:
    cids = C.loc[C["family"] == f, "cid"].values
    sub = X[cids]
    pres = np.asarray(sub.mean(0)).ravel()
    core = np.where(pres >= CORE_FRAC)[0]
    core_l = sorted(lk[core], key=lambda s: (int(s.split(".")[0]), s))
    seeds = C.loc[C["family"] == f]
    lay = seeds["seed_layer"].value_counts().sort_index().to_dict()
    kinds = seeds["seed_kind"].value_counts().to_dict()
    core_layers = Counter(int(s.split(".")[0]) for s in core_l)
    fam_rows.append(dict(family=int(f), n_circuits=int(len(cids)), seed_layers=lay, seed_kinds=kinds,
                         n_core=int(len(core)), core_layers=dict(sorted(core_layers.items())), core=core_l[:60]))
    print("  family %3d | %4d circuits | seeds by layer %s | kinds %s | CORE (>=%.0f%% of circuits): %d latents, layers %s"
          % (f, len(cids), lay, kinds, 100 * CORE_FRAC, len(core), dict(sorted(core_layers.items()))))
    print("      core sample:", " ".join(core_l[:12]))
json.dump(fam_rows, open(OUT / "families.json", "w"), indent=1)
C[["cid", "shard", "skey", "seed_layer", "seed_kind", "seed_index", "n_members", "family"]].to_csv(OUT / "circuit_families.csv", index=False)
report["families"] = dict(n_big=len(big), covered=int(sum(sizes[f] for f in big)),
                          sizes_top=[int(sizes[f]) for f in big[:20]])
json.dump(report, open(OUT / "graph_report.json", "w"), indent=1)
print("\n->", OUT)
