"""WHICH OF THE 15k TURINGLLM CIRCUITS ARE INTERESTING? An operational sort.

Hypothesis under test (Daniel, 2026-09-12): "TuringLLM doesn't have many
circuits we're interested in". Before accepting it, rank every circuit
on properties that separate a concept-level mechanism from a token
detector wired to hubs, and look at the top of the list.

Per circuit (all from tables_full / results_full / the stores, CPU only):
  abstract    1 - top_token_consistency of the SEED (post_analysis): the
              seed is not a single-token detector
  specific    1 - share of members that are hubs (fan-out >= 5% of circuits)
  composed    layer spread of the members (std) and layers spanned
  modulating  amplitude work: p90/median gain and fraction reduced (<1)
              - the tri-amp gains change inputs rather than pass them through
  depended-on log(1 + in-degree) in the seed->seed graph: other circuits
              use this seed as a member
  gates       posctx suppression >= 0.9 (members necessary) and >= 30 members

score = mean of z-scored components over gated circuits. Output:
  results_full/interesting_circuits.csv (all circuits, ranked)
  a printed top-N with everything known about each (family, labels if any)

  python experiments/049-circuit-graph/interesting_circuits.py
Env: TOP (40), DATA (data_full), TABLES, OUT
"""
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
DATA = Path(os.environ.get("DATA", str(HERE / "data_full")))
T = Path(os.environ.get("TABLES", str(HERE / "tables_full")))
R = Path(os.environ.get("OUT", str(HERE / "results_full")))
TOP = int(os.environ.get("TOP", 40))
KINDS = ["attn", "mlp", "resid"]

C = pd.read_parquet(T / "circuits.parquet")
M = pd.read_parquet(T / "members.parquet")
M["kind"] = M["kind"].astype(str)
C["skey"] = C["seed_layer"].astype(str) + "." + C["seed_kind"].astype(str) + "." + C["seed_index"].astype(str)
print("circuits %d | member rows %d" % (len(C), len(M)))

# ---- post-analysis fields from the stores (keyed by seed) -----------------------
pa = {}
for p in sorted(glob.glob(str(DATA / "discovered_circuits.shard*.pt"))):
    for c in torch.load(p, weights_only=False, map_location="cpu").values():
        md = c.metadata; x = md.get("post_analysis") or {}
        key = "%d.%s.%d" % (md["seed_comp"] // 3, KINDS[md["seed_comp"] % 3], md["seed_latent"])
        pa[key] = dict(tok_cons=x.get("top_token_consistency_pct", np.nan), layer_std=x.get("layer_std", np.nan),
                       layer_mean=x.get("layer_mean", np.nan), coact=x.get("coact_overlap_pct", np.nan))
P = pd.DataFrame.from_dict(pa, orient="index")
C = C.merge(P, left_on="skey", right_index=True, how="left")

# ---- hub share, layers spanned -------------------------------------------------------
fo = pd.read_csv(R / "fanout_latents.csv")
hubs = set(fo.loc[fo["fanout"] >= 0.05 * len(C), "latent"])
M["lkey"] = M["layer"].astype(str) + "." + M["kind"] + "." + M["index"].astype(str)
M["is_hub"] = M["lkey"].isin(hubs)
g = M.groupby("cid")
C = C.merge(pd.DataFrame({"hub_share": g["is_hub"].mean(), "n_layers": g["layer"].nunique(),
                          "n_specific": g["is_hub"].apply(lambda s: int((~s).sum()))}),
            left_on="cid", right_index=True, how="left")

# ---- graph in-degree, family ------------------------------------------------------------
E = pd.read_csv(R / "seed_seed_edges.csv")
indeg = E.groupby("dst").size()
C["in_degree"] = C["cid"].map(indeg).fillna(0).astype(int)
fam = pd.read_csv(R / "circuit_families_nohub.csv").set_index("cid")["family"]
C["family"] = C["cid"].map(fam)
labels = {}
for lp in (R / "labels_cofire.json", R / "labels.json"):
    if lp.exists():
        for r in json.load(open(lp)):
            labels.setdefault(r["latent"], "%s (%d%%)" % (r["peak_top"][0][0], 100 * r["peak_consistency"]))
C["label"] = C["skey"].map(labels)

# ---- components + score ------------------------------------------------------------------
gate = (C["posctx_sup"] >= 0.9) & (C["n_members"] >= 30)
S = C[gate].copy()
comp = pd.DataFrame({
    "abstract": 1 - S["tok_cons"].fillna(S["tok_cons"].median()) / 100.0,
    "specific": 1 - S["hub_share"],
    "composed": S["layer_std"].fillna(0),
    "modulating": (S["amp_p90"] / S["amp_median"] - 1).fillna(0) + S["amp_frac_reduced"].fillna(0),
    "depended_on": np.log1p(S["in_degree"]),
}, index=S.index)
# z-score WITHIN seed layer: every component (layers spanned, spread,
# modulation, abstractness) grows with depth, so a global z-score just
# ranks "deep". Within-layer, the score means "interesting relative to
# peers at the same depth" and the top slice covers the whole stack.
Z = comp.groupby(S["seed_layer"]).transform(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1.0))
S["score"] = Z.mean(axis=1)
for c in comp:
    S["z_" + c] = Z[c]
C = C.merge(S[["cid", "score"] + ["z_" + c for c in comp]], on="cid", how="left")
C = C.sort_values("score", ascending=False)
C.to_csv(R / "interesting_circuits.csv", index=False)

print("gated (sup >= 0.9, >= 30 members): %d of %d" % (int(gate.sum()), len(C)))
print("score distribution: p50 %.2f | p90 %.2f | p99 %.2f | max %.2f" % tuple(S["score"].quantile([0.5, 0.9, 0.99]).tolist() + [S["score"].max()]))
print("top-1%% by seed layer: %s | by kind: %s"
      % (S.nlargest(len(S) // 100, "score")["seed_layer"].value_counts().sort_index().to_dict(),
         S.nlargest(len(S) // 100, "score")["seed_kind"].value_counts().to_dict()))
def show(df, n):
    for _, r in df.head(n).iterrows():
        print("  %5.2f | %-14s | %4d [%4d] | %.2f | %2d/%.2f | %.2f %.2f | %3d | %5s | fam %-4s | %s"
              % (r["score"], r["skey"], r["n_members"], r["n_specific"], r["hub_share"], r["n_layers"], r["layer_std"],
                 r["amp_p90"] / r["amp_median"], r["amp_frac_reduced"], r["in_degree"],
                 ("%.0f%%" % r["tok_cons"]) if pd.notna(r["tok_cons"]) else "-", str(r["family"]) if pd.notna(r["family"]) else "-",
                 r["label"] if pd.notna(r["label"]) else ""))


print("\n(score | seed | members [specific] | hub share | layers spanned/std | amp p90/med, reduced | in-deg | tok-cons | family | label)")
for lo, hi in ((0, 3), (4, 7), (8, 11)):
    print("\nTOP %d, seed layers %d-%d:" % (TOP // 3, lo, hi))
    show(C[(C["seed_layer"] >= lo) & (C["seed_layer"] <= hi)], TOP // 3)
# stratified candidate list: top 33 per seed layer -> ~400 for labelling / amp eval
strat = C[C["score"].notna()].sort_values("score", ascending=False).groupby("seed_layer").head(33)
strat[["cid", "skey", "seed_layer", "seed_kind", "seed_index", "score"]].to_csv(R / "interesting_top400.csv", index=False)
print("\nstratified candidates: %d (33 per layer)" % len(strat))
print("\n->", R / "interesting_circuits.csv", "and interesting_top400.csv (for labelling / amp eval)")
