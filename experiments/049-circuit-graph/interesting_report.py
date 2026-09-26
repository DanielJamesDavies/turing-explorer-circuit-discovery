"""JOIN the interestingness ranking with the activation-gated labels and
the amplitude-aware eval for the candidate set, and print what the top
of the list actually IS.

Per candidate: score | seed | held-out F0 with amps (bounded) | a=1 |
amp effect | permuted null | peak token (consistency) | second/third
tokens | two example windows.

Also: label-quality buckets. A candidate is
  lexical   peak consistency >= 60% (a token detector after all)
  concept   consistency < 30% but the top-3 tokens share an obvious theme
            (judged by eye from the printout — the script only buckets by
            consistency)
  mixed     in between

  python experiments/049-circuit-graph/interesting_report.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R = Path(os.environ.get("OUT", str(HERE / "results_full")))
TOP = int(os.environ.get("TOP", 60))

rank = pd.read_csv(R / "interesting_circuits.csv")
cand = pd.read_csv(R / "interesting_top400.csv")
labels = {r["latent"]: r for r in json.load(open(R / "labels_interesting.json"))}
ev = []
for ln in open(R / "amp_eval_subset.jsonl"):
    r = json.loads(ln)
    if "a_pos_ho" in r:
        den = abs(r["a_pos_ho"] - r["e0_ho"])
        r["F0_amp_b"] = 1 - abs(r["ampF0_ho"] - r["a_pos_ho"]) / den if den > 1e-9 else np.nan
        r["F0_a1_b"] = 1 - abs(r["F0a1_ho"] - r["a_pos_ho"]) / den if den > 1e-9 else np.nan
        r["F0_perm_b"] = 1 - abs(r["ampF0_perm"] - r["a_pos_ho"]) / den if den > 1e-9 else np.nan
        ev.append(r)
ev = pd.DataFrame(ev).set_index("seed")
df = cand.merge(rank[["skey", "n_members", "hub_share", "n_layers", "family", "tok_cons", "in_degree"]], on="skey", how="left")
df = df.merge(ev[["F0_amp_b", "F0_a1_b", "F0_perm_b", "cf_amp", "n"]], left_on="skey", right_index=True, how="left")
df["peak"] = df["skey"].map(lambda k: labels[k]["peak_top"][0][0] if k in labels else None)
df["cons"] = df["skey"].map(lambda k: labels[k]["peak_consistency"] if k in labels else np.nan)
df["top3"] = df["skey"].map(lambda k: ", ".join(repr(t) for t, _ in labels[k]["peak_top"][:3]) if k in labels else "")
df["ex"] = df["skey"].map(lambda k: " || ".join(w.replace("\n", "\\n")[-70:] for w in labels[k]["windows"][:2]) if k in labels else "")
df = df.sort_values("score", ascending=False)
df.to_csv(R / "interesting_report.csv", index=False)

ok = df.dropna(subset=["F0_amp_b"])
print("candidates %d | labelled %d | amp-evaluated %d" % (len(df), df["peak"].notna().sum(), len(ok)))
print("held-out F0 with amps (bounded): median %.3f | >= 0.8: %.2f | a=1 median %.3f | amp effect median %+.3f | permuted null median %.3f"
      % (ok["F0_amp_b"].median(), (ok["F0_amp_b"] >= 0.8).mean(), ok["F0_a1_b"].median(),
         (ok["F0_amp_b"] - ok["F0_a1_b"]).median(), ok["F0_perm_b"].median()))
lab = df.dropna(subset=["cons"])
print("label buckets by peak-token consistency: lexical (>=60%%) %d | mixed %d | non-lexical (<30%%) %d"
      % ((lab["cons"] >= 0.6).sum(), ((lab["cons"] >= 0.3) & (lab["cons"] < 0.6)).sum(), (lab["cons"] < 0.3).sum()))
print("  non-lexical by seed layer:", lab[lab["cons"] < 0.3]["seed_layer"].value_counts().sort_index().to_dict())
print("  F0_amp_b of non-lexical candidates: median %.3f | of lexical: median %.3f"
      % (ok[ok["cons"] < 0.3]["F0_amp_b"].median(), ok[ok["cons"] >= 0.6]["F0_amp_b"].median()))

print("\nTOP %d BY SCORE  (score | seed | n | F0amp | a=1 | null | peak (cons) | top-3 | examples)" % TOP)
for _, r in df.head(TOP).iterrows():
    print("  %4.2f | %-14s | %4d | %5s | %5s | %5s | %-14s (%3.0f%%) | %-32s | %s"
          % (r["score"], r["skey"], r["n_members"],
             "%.2f" % r["F0_amp_b"] if pd.notna(r["F0_amp_b"]) else "-", "%.2f" % r["F0_a1_b"] if pd.notna(r["F0_a1_b"]) else "-",
             "%.2f" % r["F0_perm_b"] if pd.notna(r["F0_perm_b"]) else "-",
             repr(r["peak"])[:14] if pd.notna(r["peak"]) else "-", 100 * r["cons"] if pd.notna(r["cons"]) else 0,
             r["top3"][:32], r["ex"][:110]))

print("\nNON-LEXICAL candidates with F0_amp >= 0.8 (the concept-circuit shortlist), by layer:")
sl = ok[(ok["cons"] < 0.3) & (ok["F0_amp_b"] >= 0.8)].sort_values(["seed_layer", "score"], ascending=[True, False])
for _, r in sl.iterrows():
    print("  L%-2d %-14s | n %4d | F0amp %.2f a1 %.2f | %-34s | %s"
          % (r["seed_layer"], r["skey"], r["n_members"], r["F0_amp_b"], r["F0_a1_b"], r["top3"][:34], r["ex"][:120]))
print("\n-> %s (%d shortlisted)" % (R / "interesting_report.csv", len(sl)))
