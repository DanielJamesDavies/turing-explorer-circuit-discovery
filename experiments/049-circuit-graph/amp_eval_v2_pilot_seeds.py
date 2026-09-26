"""Pick the stratified pilot for amp_eval_pass_v2.py: 5 circuits per seed layer, kinds mixed.

Per layer: one random circuit of each kind present, one more at random, and the circuit with the highest
share of REDUCED gains (amp_frac_reduced) in a random draw of 200 — a proxy for inhibitor-heavy circuits
(the stored attribution is unsigned, so inhibitor counts are not known before the pass runs).

  python experiments/049-circuit-graph/amp_eval_v2_pilot_seeds.py  -> results_full/amp_eval_v2_pilot_seeds.txt
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ct = pd.read_parquet(HERE / "tables_full" / "circuits.parquet")
ct["skey"] = ct.seed_layer.astype(str) + "." + ct.seed_kind.astype(str) + "." + ct.seed_index.astype(str)
picks = []
for layer, d in ct.groupby("seed_layer"):
    chosen = []
    for kind, dk in d.groupby("seed_kind", observed=True):
        chosen.append(dk.sample(1, random_state=100 + int(layer)).iloc[0])
    rest = d[~d.skey.isin([r.skey for r in chosen])]
    draw = rest.sample(min(200, len(rest)), random_state=200 + int(layer))
    chosen.append(draw.sort_values("amp_frac_reduced", ascending=False).iloc[0])
    rest = rest[~rest.skey.isin([r.skey for r in chosen])]
    while len(chosen) < 5:
        r = rest.sample(1, random_state=300 + int(layer) + len(chosen)).iloc[0]
        chosen.append(r); rest = rest[rest.skey != r.skey]
    for r in chosen:
        picks.append(r.skey)
        print("L%-2d %-5s %-16s n_members %4d amp_frac_reduced %.3f" % (layer, r.seed_kind, r.skey, r.n_members, r.amp_frac_reduced))
out = HERE / "results_full" / "amp_eval_v2_pilot_seeds.txt"
out.write_text("\n".join(picks) + "\n")
print("\n%d seeds -> %s" % (len(picks), out))
