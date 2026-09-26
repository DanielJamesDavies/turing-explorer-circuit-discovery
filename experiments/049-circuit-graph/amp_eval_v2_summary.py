"""SUMMARISE the amplitude-aware pass (rows v=3, amp_eval_v3*.jsonl): timing + ETA, metric medians by layer band,
the scope check (every upstream site ablated), role statistics, the split-ordering diagnostic.

Every ratio is reported on the ACTIVATION read (`_tk`, primary, protocol v1) and, as a diagnostic, the
PRE-ACTIVATION read (`_pre`). v1/v2 rows (mixed reads, member-site-only ablation; DAN-66/67) are not summarised.

  python experiments/049-circuit-graph/amp_eval_v2_summary.py "<glob of v3 jsonl>" [more globs ...]
  default glob: results_full/amp_eval_v3.shard*.jsonl
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
pats = sys.argv[1:] or [str(HERE / "results_full" / "amp_eval_v3.shard*.jsonl")]
rows = {}
n_skip = n_err = n_old = 0
for pat in pats:
    for p in sorted(glob.glob(pat)):
        for ln in open(p):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "skip" in r:
                n_skip += 1
            elif "error" in r:
                n_err += 1
            elif r.get("v") != 3:
                n_old += 1                       # v1/v2 rows: not comparable
            else:
                rows[r["seed"]] = r               # last row per seed wins
df = pd.DataFrame(list(rows.values()))
print("scored %d | skip rows %d | error rows %d | pre-v3 rows ignored %d" % (len(df), n_skip, n_err, n_old))
if df.empty:
    sys.exit("no v3 rows")
df["band"] = pd.cut(df["layer"], [-1, 3, 7, 11], labels=["L0-3", "L4-7", "L8-11"])
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)

# bounded (overshoot = error) variants, one read throughout: 1 - |x - a_pos| / |a_pos - e|
for name, e in (("free0", "e0"), ("free0_a1", "e0"), ("freeM_dense", "eM_dense"), ("freeM_topk", "eM_topk"),
                ("freeN_dense", "eN_dense"), ("freeN_topk", "eN_topk"), ("free0_perm", "e0"), ("free0_rand", "e0"),
                ("phi_pin_alpha_zero", "e0"), ("phi_pin_alpha_topk", "eM_topk")):
    for rd in ("tk", "pre"):
        raw, ek, ak = "%s_raw_%s" % (name, rd), "%s_%s" % (e, rd), "a_pos_" + rd
        if raw in df:
            den = (df[ak] - df[ek]).abs()
            df["%s_b_%s" % (name, rd)] = np.where(den > 1e-9, 1 - (df[raw] - df[ak]).abs() / den, np.nan)

print("\n=== TIMING (seconds per circuit) ===")
t = df.groupby("layer")["secs"].agg(["median", "mean", "max", "count"]).round(2)
t["peak_mem_mb"] = df.groupby("layer")["peak_mem_mb"].max()
print(t.to_string())
try:
    ct = pd.read_parquet(HERE / "tables_full" / "circuits.parquet")
    per_layer = ct.groupby("seed_layer").size()
    mean_by_layer = df.groupby("layer")["secs"].mean()
    # the first circuit of a process carries warm-up; use the median where it is lower
    use = np.minimum(mean_by_layer, df.groupby("layer")["secs"].median() * 1.15)
    tot = float((per_layer * use.reindex(per_layer.index).fillna(use.mean())).sum())
    print("projected scoring time for all %d circuits at these per-layer costs: %.1f h (single process)" % (int(per_layer.sum()), tot / 3600))
except Exception as e:
    print("(no ETA: %s)" % e)

print("\n=== SCOPE (DAN-67): every upstream site ablated in the circuit-only runs ===")
print("  all_sites_ablated: %d of %d circuits" % (int(df["all_sites_ablated"].fillna(False).sum()), len(df)))

print("\n=== METRIC MEDIANS BY LAYER BAND (activation read = primary; _pre = diagnostic) ===")
BASE = ["a_pos_tk", "e0_tk", "eM_dense_tk", "eM_topk_tk", "eN_dense_tk", "eN_topk_tk", "a_base_tk", "a_pos_pre", "e0_pre"]
RATIOS = ["free0", "free0_b", "free0_tr", "free0_a1", "free0_a1_b", "free0_perm", "free0_perm_b", "free0_rand", "free0_rand_b",
          "freeM_dense", "freeM_topk", "freeM_topk_b", "freeN_dense", "freeN_topk", "freeN_topk_b",
          "phi_sup_blind", "phi_sup_role", "phi_sup_alpha", "sup_activators_only", "release",
          "phi_cf_alpha_blind", "phi_cf_alpha_role", "phi_cf_a1_role",
          "phi_pin_alpha_zero", "phi_pin_alpha_zero_b", "phi_pin_alpha_topk", "phi_pin_alpha_topk_b", "phi_pin_alpha_dense"]
OTHER = ["n_members", "n_inhibitor", "inhibitor_mass_share", "alpha_median"]
METRICS = [m for m in BASE if m in df]
for rd in ("tk", "pre"):
    METRICS += [m + "_" + rd for m in RATIOS if m + "_" + rd in df]
METRICS += [m for m in OTHER if m in df]
tab = df.groupby("band", observed=True)[METRICS].median().T
tab["all"] = df[METRICS].median()
tab["n_valid"] = df[METRICS].notna().sum()
print(tab.round(3).to_string())
print("\ncircuits per band:", df.groupby("band", observed=True).size().to_dict())

print("\n=== VACUOUS DENOMINATORS ===")
for rd in ("tk", "pre"):
    print("  vacuous_%s (|a_pos - e0| < 5%% of a_pos): %.3f" % (rd, df["vacuous_" + rd].mean()))
for e in ("e0_tk", "eM_dense_tk", "eM_topk_tk", "eN_dense_tk", "eN_topk_tk"):
    print("  %-10s > 50%% of a_pos_tk: %.3f   (by band: %s)" % (e, (df[e] > 0.5 * df["a_pos_tk"]).mean(),
                                                            (df[e] > 0.5 * df["a_pos_tk"]).groupby(df["band"], observed=True).mean().round(2).to_dict()))
print("  e0_pre >= a_pos_pre (pre-activation ratios undefined under zero fill): %.3f" % (df["e0_pre"] >= df["a_pos_pre"]).mean())

print("\n=== NECESSITY: role-aware vs role-blind (activation read) ===")
allact = df[df["n_inhibitor"] == 0]
print("  circuits with no inhibitors: %d (there phi_sup_role == phi_sup_blind by construction)" % len(allact))
g = (df["phi_sup_role_tk"] - df["phi_sup_blind_tk"])
print("  role - blind: median %+.4f | p10 %+.4f | p90 %+.4f | |gap| > 0.05 in %.2f of circuits" % (g.median(), g.quantile(.1), g.quantile(.9), (g.abs() > 0.05).mean()))
ex = df.assign(gap=g).sort_values("gap")
cols = ["seed", "n_members", "n_inhibitor", "inhibitor_mass_share", "phi_sup_blind_tk", "phi_sup_role_tk", "sup_activators_only_tk", "release_tk", "gap"]
print("  largest gaps either way:")
print(pd.concat([ex.head(4), ex.tail(4)])[cols].round(3).to_string(index=False))

print("\n=== ROLES (grad x activation, train slice; the stored attribution is the unsigned gate probability) ===")
sh = df["n_inhibitor"] / df["n_members"].clip(lower=1)
print("  inhibitor share of members: median %.3f | p10 %.3f | p90 %.3f ; of |attribution| mass: median %.3f" % (sh.median(), sh.quantile(.1), sh.quantile(.9), df["inhibitor_mass_share"].median()))
print("  release (inhibitors -> 0), activation read: median %+.3f | > 0 (target rises) in %.2f | < -0.2 in %.2f of circuits"
      % (df["release_tk"].median(), (df["release_tk"] > 0).mean(), (df["release_tk"] < -0.2).mean()))

print("\n=== SPLIT ORDERING: natural a_pos on train (first 48) vs held-out (last 16), activation read ===")
full = df[df["n_pos"] == 64]
print("  n %d | median a_pos train %.3f | held-out %.3f | median ratio ho/train %.3f | held-out < train in %.3f of seeds"
      % (len(full), full["a_pos_tr_tk"].median(), full["a_pos_tk"].median(), (full["a_pos_tk"] / full["a_pos_tr_tk"]).median(), (full["a_pos_tk"] < full["a_pos_tr_tk"]).mean()))
bl = np.array([b for b in full["a_pos_blocks"] if len(b) == 4])
if len(bl):
    rel = bl / bl[:, :1]
    print("  a_pos per 16-context block relative to block 0 (median): %s" % np.round(np.median(rel, axis=0), 3).tolist())
    print("  by band, median ho/train: %s" % (full["a_pos_tk"] / full["a_pos_tr_tk"]).groupby(full["band"], observed=True).median().round(3).to_dict())
out = Path(glob.glob(pats[0])[0]).parent / "amp_eval_v3_summary.csv" if glob.glob(pats[0]) else None
if out is not None:
    df.drop(columns=["per_kind", "a_pos_blocks"], errors="ignore").to_csv(out, index=False)
    print("\n->", out)
