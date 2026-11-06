"""Adaptive λ, option C, applied: train c_predict's GBM (static features) on all 189 sweep targets and assign a λ to
each dev target; writes the 24-target dev subset and one target list per assigned λ (run_dev24.sh fits each list at
its λ). Dev = 093's 72 dev targets (disjoint from 102's check panel); the subset is every third target in sorted order.

Static features from the 15k run's main fit of each target (062 out_full/out/main: eval a_pos_pre, train
n_train_pos; out/targets.csv rank_clean). C was learned on protocol v1 (constant lr, no R); the λ grid is reused as is
under the current fit (geometric x3.15 keeps sum(lr) at constant's), an assumption this run tests.

  python experiments/107-true-baselines/adaptive_lambda/c_assign.py
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c_predict as CP  # noqa: E402

O62 = CP.O62
DEV = HERE.parents[1] / "093-v2-validation" / "targets.txt"


def main():
    df = CP.load()
    df["is_attn"] = (df.kind == "attn").astype(float)
    df["is_mlp"] = (df.kind == "mlp").astype(float)
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    model = GradientBoostingClassifier(random_state=0, max_depth=2, n_estimators=150).fit(df[CP.STATIC].values,
                                                                                         df.oracle.values)
    # --targets FILE --prefix NAME: assign another panel (e.g. 102's check panel) instead of the dev subset
    if "--targets" in sys.argv:
        dev = Path(sys.argv[sys.argv.index("--targets") + 1]).read_text().split()
        prefix = sys.argv[sys.argv.index("--prefix") + 1]
    else:
        dev, prefix = sorted(DEV.read_text().split())[::3], "dev24"
    ev = {r["seed"]: r for r in CP.jl(O62 / "main" / "eval.shard*.jsonl") if r.get("held") == "strong"}
    tr = {r["seed"]: r for r in CP.jl(O62 / "main" / "train.shard*.jsonl")}
    tg = pd.read_csv(O62 / "targets.csv").set_index("seed")
    rows = []
    for k in dev:
        layer, kind = int(k.split(".")[0]), k.split(".")[1]
        e, t = ev.get(k) or {}, tr.get(k) or {}
        rank = tg.loc[k, "rank_clean"] if k in tg.index else np.nan
        rows.append(dict(seed=k, layer=layer, is_attn=float(kind == "attn"), is_mlp=float(kind == "mlp"),
                         log_strength=math.log(max(float(e.get("a_pos_pre") or 1e-3), 1e-3)),
                         log_rank=math.log(float(rank)) if not pd.isna(rank) else df.log_rank.median(),
                         thin=float((t.get("n_train_pos") or 64) < 64), have_eval=k in ev, have_train=k in tr))
    X = pd.DataFrame(rows)
    pred = model.predict(X[CP.STATIC].values)
    X["lam"] = [CP.GRID[max(int(p), 0)] for p in pred]
    print(X[["seed", "layer", "log_strength", "log_rank", "thin", "have_eval", "have_train", "lam"]].to_string(index=False))
    if prefix == "dev24":
        (HERE / "targets_dev24.txt").write_text("\n".join(dev) + "\n")
    for lam, g in X.groupby("lam"):
        (HERE / ("targets_%s_C_%g.txt" % (prefix, lam))).write_text("\n".join(g.seed) + "\n")
    print("C assignment counts:", X.lam.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
