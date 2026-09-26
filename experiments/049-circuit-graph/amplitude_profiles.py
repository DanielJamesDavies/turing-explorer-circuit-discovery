"""AMPLITUDE PROFILES: how each latent's fitted gain behaves across every
circuit it belongs to (tri-amp only; the gains travel with the nodes).

Per latent: n circuits, median gain, IQR, frac > 1 (elevated), frac < 1
(reduced), and whether it flips (both elevated and reduced in different
circuits = the same latent used two ways). Plus gain vs attribution and
gain vs fan-out.

  python experiments/049-circuit-graph/amplitude_profiles.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
T = Path(os.environ.get("TABLES", str(HERE / "tables")))
OUT = Path(os.environ.get("OUT", str(HERE / "results")))
OUT.mkdir(parents=True, exist_ok=True)
MIN_N = int(os.environ.get("MIN_N", 10))

M = pd.read_parquet(T / "members.parquet")
M["lkey"] = M["layer"].astype(str) + "." + M["kind"].astype(str) + "." + M["index"].astype(str)
a = M["amplitude"]
print("member rows %d | amplitude: median %.3f | p10 %.3f | p90 %.3f | max %.2f | frac > 1: %.3f | frac < 0.5: %.3f | frac > 2: %.3f"
      % (len(M), a.median(), a.quantile(0.1), a.quantile(0.9), a.max(), (a > 1).mean(), (a < 0.5).mean(), (a > 2).mean()))
print("amplitude by member kind:", M.groupby("kind", observed=True)["amplitude"].median().round(3).to_dict())
print("amplitude by member layer (median):", M.groupby("layer")["amplitude"].median().round(3).to_dict())
M["dl"] = M["seed_layer"] - M["layer"]
print("amplitude by distance below seed (median):", M.groupby("dl")["amplitude"].median().round(3).to_dict())

g = M.groupby("lkey")["amplitude"]
prof = pd.DataFrame({"n": g.size(), "median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75),
                     "frac_elev": g.apply(lambda s: (s > 1.0).mean()), "frac_red": g.apply(lambda s: (s < 1.0).mean()),
                     "frac_strong_elev": g.apply(lambda s: (s > 1.5).mean()), "frac_strong_red": g.apply(lambda s: (s < 0.67).mean())})
prof["iqr"] = prof["q75"] - prof["q25"]
prof["flips"] = (prof["frac_strong_elev"] >= 0.2) & (prof["frac_strong_red"] >= 0.2)
prof = prof.reset_index()
prof[["layer", "kind", "index"]] = prof["lkey"].str.split(".", expand=True)
prof["layer"] = prof["layer"].astype(int)
P = prof[prof["n"] >= MIN_N]
print("\nlatents with >= %d memberships: %d" % (MIN_N, len(P)))
print("  consistently boosted (median > 1.5, frac_elev >= 0.8): %d" % ((P["median"] > 1.5) & (P["frac_elev"] >= 0.8)).sum())
print("  consistently reduced (median < 0.67, frac_red >= 0.8): %d" % ((P["median"] < 0.67) & (P["frac_red"] >= 0.8)).sum())
print("  flip latents (>= 20%% strongly up AND >= 20%% strongly down): %d (%.1f%%)" % (P["flips"].sum(), 100 * P["flips"].mean()))
print("  near-unity everywhere (iqr < 0.2 and |median-1| < 0.1): %d (%.1f%%)"
      % (((P["iqr"] < 0.2) & ((P["median"] - 1).abs() < 0.1)).sum(), 100 * ((P["iqr"] < 0.2) & ((P["median"] - 1).abs() < 0.1)).mean()))
print("\n  TOP 20 consistently boosted (by median, n >= %d):" % MIN_N)
for _, r in P.sort_values("median", ascending=False).head(20).iterrows():
    print("    %-16s n %5d | median %.2f | iqr %.2f | elev %.2f" % (r["lkey"], r["n"], r["median"], r["iqr"], r["frac_elev"]))
print("  TOP 20 consistently reduced:")
for _, r in P.sort_values("median").head(20).iterrows():
    print("    %-16s n %5d | median %.2f | iqr %.2f | red %.2f" % (r["lkey"], r["n"], r["median"], r["iqr"], r["frac_red"]))
print("  TOP 20 flip latents (largest iqr among flips):")
for _, r in P[P["flips"]].sort_values("iqr", ascending=False).head(20).iterrows():
    print("    %-16s n %5d | median %.2f | iqr %.2f | up %.2f down %.2f" % (r["lkey"], r["n"], r["median"], r["iqr"], r["frac_strong_elev"], r["frac_strong_red"]))

# gain vs fan-out and vs attribution
print("\n  corr(log fan-out, median gain) over latents with n>=%d: %.3f" % (MIN_N, np.corrcoef(np.log(P["n"]), P["median"])[0, 1]))
print("  corr(attribution, log amplitude) over member rows: %.3f"
      % np.corrcoef(M["attribution"].fillna(0), np.log(M["amplitude"].clip(1e-3)))[0, 1])
hub = P.sort_values("n", ascending=False).head(30)
print("  gain profile of the 30 highest fan-out latents: median-of-medians %.2f | frac with median>1.2: %.2f | frac flips: %.2f"
      % (hub["median"].median(), (hub["median"] > 1.2).mean(), hub["flips"].mean()))
prof.to_csv(OUT / "amplitude_profiles.csv", index=False)
json.dump(dict(n_rows=int(len(M)), amp_median=float(a.median()), frac_gt1=float((a > 1).mean()), frac_lt05=float((a < 0.5).mean()),
               n_profiled=int(len(P)), n_boosted=int(((P["median"] > 1.5) & (P["frac_elev"] >= 0.8)).sum()),
               n_reduced=int(((P["median"] < 0.67) & (P["frac_red"] >= 0.8)).sum()), n_flips=int(P["flips"].sum())),
          open(OUT / "amplitude_report.json", "w"), indent=1)
print("->", OUT / "amplitude_profiles.csv")
