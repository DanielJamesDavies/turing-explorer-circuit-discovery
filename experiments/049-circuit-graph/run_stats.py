"""RUN STATISTICS for the production tri-amp circuits: every per-circuit
eval the pipeline stored, summarised overall and by seed layer / kind.

Stored per circuit (circuit.metadata):
  evals.counterfactual_faithfulness   in-run, BARE member set (amplitudes stripped):
                                      seed recovered on negctx when the members are
                                      restored, relative to the natural posctx level
  evals.posctx_suppression_score      in-run, bare: seed drop on posctx when the
                                      members are ablated (1 = fully suppressed)
  evals.ablation_suppression_score    (identical to posctx_suppression_score here)
  amp_stats                           tri-amp gains: n, median, p10, p90, max,
                                      frac_elevated (>1), frac_reduced (<1)
  target_pre_act / target_loss        the seed's natural pre-activation on its
                                      probes and the fit's final data loss
  post_analysis.*                     layer_mean/std/min/max of members,
                                      coact_overlap_pct (members that are also
                                      top-coactivation partners of the seed),
                                      edge_weight_gini, activity_mean/median,
                                      rarity_pct, top_token_consistency_pct
task_metrics.shard*.jsonl             seconds per seed, forward passes, phases

  python experiments/049-circuit-graph/run_stats.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
DATA = Path(os.environ.get("DATA", str(HERE / "data_full")))
OUT = Path(os.environ.get("OUT", str(HERE / "results_full")))
KINDS = ["attn", "mlp", "resid"]

rows = []
for p in sorted(glob.glob(str(DATA / "discovered_circuits.shard*.pt"))):
    for c in torch.load(p, weights_only=False, map_location="cpu").values():
        md = c.metadata; ev = md.get("evals") or {}; am = md.get("amp_stats") or {}; pa = md.get("post_analysis") or {}
        rows.append(dict(seed_layer=md["seed_comp"] // 3, seed_kind=KINDS[md["seed_comp"] % 3], n_members=md.get("n_supports", len(c.nodes) - 1),
                         cf_faith=ev.get("counterfactual_faithfulness"), sup=ev.get("posctx_suppression_score"),
                         amp_median=am.get("median"), amp_p90=am.get("p90"), amp_max=am.get("max"), frac_elev=am.get("frac_elevated"), frac_red=am.get("frac_reduced"),
                         pre_act=md.get("target_pre_act"), loss=md.get("target_loss"),
                         layer_mean=pa.get("layer_mean"), layer_std=pa.get("layer_std"), coact=pa.get("coact_overlap_pct"), gini=pa.get("edge_weight_gini"),
                         rarity=pa.get("rarity_pct"), tok_cons=pa.get("top_token_consistency_pct")))
df = pd.DataFrame(rows)
tm = []
for p in glob.glob(str(DATA / "task_metrics.shard*.jsonl")):
    for ln in open(p):
        r = json.loads(ln)
        if r["accepted_circuit_count"]:
            tm.append(dict(seed_layer=r["comp_idx"] // 3, seed_kind=KINDS[r["comp_idx"] % 3], secs=r["total_s"], fwd=r["forward_pass_count"],
                           peak_gb=r.get("peak_cuda_memory_bytes", 0) / 1e9))
tm = pd.DataFrame(tm)
df.to_csv(OUT / "run_stats_per_circuit.csv", index=False)
print("circuits: %d | seeds by kind %s" % (len(df), df["seed_kind"].value_counts().to_dict()))


def q(s):
    s = s.dropna()
    return "median %.3f | p10 %.3f | p90 %.3f | mean %.3f" % (s.median(), s.quantile(0.1), s.quantile(0.9), s.mean())


print("\n=== IN-RUN EVALS (bare member set, amplitudes stripped) ===")
print("counterfactual faithfulness (negctx restoration): " + q(df["cf_faith"]))
print("  frac >= 0.5: %.2f | >= 0.8: %.2f | <= 0: %.2f | > 1.5: %.2f" % ((df.cf_faith >= 0.5).mean(), (df.cf_faith >= 0.8).mean(), (df.cf_faith <= 0).mean(), (df.cf_faith > 1.5).mean()))
print("posctx suppression (ablate members -> seed drop): " + q(df["sup"]))
print("  frac >= 0.9: %.2f | >= 0.5: %.2f | < 0: %.2f" % ((df["sup"] >= 0.9).mean(), (df["sup"] >= 0.5).mean(), (df["sup"] < 0).mean()))
print("\nby seed layer (median cf_faith / median sup / median members / median amp / frac_elev):")
g = df.groupby("seed_layer")
for l, d in g:
    print("  L%-2d n=%4d | cf %.3f | sup %.3f | members %4d | amp median %.3f p90 %.2f | elev %.2f red %.2f | pre_act %.1f | member layer mean %.2f sd %.2f | coact %.1f%% | tokcons %.0f%%"
          % (l, len(d), d.cf_faith.median(), d["sup"].median(), d.n_members.median(), d.amp_median.median(), d.amp_p90.median(), d.frac_elev.median(),
             d.frac_red.median(), d.pre_act.median(), d.layer_mean.median(), d.layer_std.median(), d.coact.median(), d.tok_cons.median()))
print("\nby seed kind:")
for k, d in df.groupby("seed_kind"):
    print("  %-5s n=%4d | cf %.3f | sup %.3f | members %4d | amp median %.3f | elev %.2f | pre_act %.1f | coact %.1f%%"
          % (k, len(d), d.cf_faith.median(), d["sup"].median(), d.n_members.median(), d.amp_median.median(), d.frac_elev.median(), d.pre_act.median(), d.coact.median()))
print("\n=== AMPLITUDES (per-circuit stats) ===")
for c in ("amp_median", "amp_p90", "amp_max", "frac_elev", "frac_red"):
    print("  %-10s " % c + q(df[c]))
print("\n=== POST-ANALYSIS ===")
for c in ("layer_mean", "layer_std", "coact", "gini", "rarity", "tok_cons"):
    print("  %-10s " % c + q(df[c]))
print("\n=== CORRELATIONS (Spearman) ===")
cols = ["n_members", "cf_faith", "sup", "amp_median", "frac_elev", "pre_act", "seed_layer", "coact", "tok_cons"]
print(df[cols].corr(method="spearman").round(2).to_string())
print("\n=== COST ===")
print("seconds per circuit: " + q(tm["secs"]) + " | total GPU-hours (sum of seed seconds / 3600): %.1f" % (tm["secs"].sum() / 3600))
print("forward passes per seed: " + q(tm["fwd"]) + " | peak CUDA GB: median %.1f max %.1f" % (tm.peak_gb.median(), tm.peak_gb.max()))
print("by layer (median s / median forwards): " + " ".join("L%d %.0f/%.0f" % (l, d.secs.median(), d.fwd.median()) for l, d in tm.groupby("seed_layer")))
print("->", OUT / "run_stats_per_circuit.csv")
