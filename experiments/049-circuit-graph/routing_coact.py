"""ROUTING / CO-ACTIVATION (item 6, CPU): do family cores fire TOGETHER on
real text? Uses the pipeline's top_coactivation store (per latent, its
top-128 PMI partners over the corpus). For each family core, the fraction
of core pairs that are mutual top-PMI partners, against random pairs of
latents drawn from the same sites. Also the same test for the hubs.

  PYTHONPATH=src python experiments/049-circuit-graph/routing_coact.py
"""
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
R = Path(os.environ.get("OUT", str(HERE / "results"))); T = Path(os.environ.get("TABLES", str(HERE / "tables")))
KINDS = ["attn", "mlp", "resid"]; NK = 3
tc = torch.load("outputs/top_coactivation.pt", weights_only=False, map_location="cpu", mmap=True)
TI = tc["top_indices"]; TV = tc["top_values"]      # [36, 40960, 128]
print("top_coactivation: mode %s | %d partners per latent" % (tc.get("mode"), TI.shape[-1]))
random.seed(0)


def comp(l, k):
    return l * NK + KINDS.index(k)


def partner_set(lkey):
    l, k, i = lkey.split("."); l, i = int(l), int(i)
    row = TI[comp(l, k), i].tolist(); vals = TV[comp(l, k), i].tolist()
    # partner index encodes (component, latent) as comp*40960 + idx ? -> detect
    return row, vals


# detect the partner encoding from the value range
mx = int(TI.max())
FLAT = mx >= 40960
print("partner ids are %s (max %d)" % ("flat comp*40960+idx" if FLAT else "latent idx within same component?", mx))


def pset(lkey):
    row, _ = partner_set(lkey)
    if FLAT:
        return {(r // 40960, r % 40960) for r in row if r >= 0}
    l, k, i = lkey.split("."); c = comp(int(l), k)
    return {(c, r) for r in row if r >= 0}


def key_to_ck(lkey):
    l, k, i = lkey.split("."); return (comp(int(l), k), int(i))


def mutual_frac(latents):
    ps = {x: pset(x) for x in latents}
    n = 0; m = 0; either = 0
    for a in range(len(latents)):
        for b in range(a + 1, len(latents)):
            ka, kb = key_to_ck(latents[a]), key_to_ck(latents[b])
            ab = kb in ps[latents[a]]; ba = ka in ps[latents[b]]
            n += 1; m += ab and ba; either += ab or ba
    return (m / n if n else float("nan")), (either / n if n else float("nan")), n


M = pd.read_parquet(T / "members.parquet")
M["lkey"] = M["layer"].astype(str) + "." + M["kind"].astype(str) + "." + M["index"].astype(str)
by_site = {s: g["lkey"].unique().tolist() for s, g in M.groupby(["layer", "kind"], observed=True)}


def random_like(latents):
    out = []
    for x in latents:
        l, k, _ = x.split("."); pool = by_site.get((int(l), k), [])
        out.append(random.choice(pool) if pool else x)
    return out


fams = json.load(open(R / "families_nohub.json"))
rows = []
print("\nFAMILY CORES: fraction of core pairs that are mutual top-128 PMI partners (either-direction in brackets) vs site-matched random")
for f in fams:
    core = f["core"][:25]
    if len(core) < 3:
        continue
    mf, ef, n = mutual_frac(core)
    rm = np.mean([mutual_frac(random_like(core))[0] for _ in range(5)])
    re = np.mean([mutual_frac(random_like(core))[1] for _ in range(5)])
    rows.append(dict(family=f["family"], n_circuits=f["n"], n_core=len(core), mutual=mf, either=ef, rand_mutual=rm, rand_either=re))
    print("  fam %3d | core %2d | mutual %.2f (rand %.2f) | either %.2f (rand %.2f)" % (f["family"], len(core), mf, rm, ef, re))
if rows:
    d = pd.DataFrame(rows)
    print("  across families: mutual median %.2f vs rand %.2f | either median %.2f vs rand %.2f"
          % (d["mutual"].median(), d["rand_mutual"].median(), d["either"].median(), d["rand_either"].median()))
    d.to_csv(R / "routing_coact_families.csv", index=False)

fo = pd.read_csv(R / "fanout_latents.csv")
hubs = fo["latent"].head(40).tolist()
mf, ef, n = mutual_frac(hubs)
rm = np.mean([mutual_frac(random_like(hubs))[0] for _ in range(3)])
print("\nTOP-40 HUBS: mutual PMI-partner fraction %.2f (site-matched random %.2f) | either %.2f" % (mf, rm, ef))
