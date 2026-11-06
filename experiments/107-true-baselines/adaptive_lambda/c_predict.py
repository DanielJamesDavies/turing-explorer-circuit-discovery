"""Adaptive λ, option C (2026-11-02, Daniel: easy targets should reach small sizes, hard ones the smallest size that
still passes): PREDICT each target's λ from its properties, evaluated OFFLINE on the 15k run's λ sweep.

Data: 062 out_full/out/sweep, 190 targets, each fitted and scored at every W λ in (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)
(protocol v1: constant lr, no R; so this learns which targets need a lower / higher λ, and the absolute level must be
recalibrated for the current protocol). Per target the ORACLE λ* = the largest grid λ whose circuit passes (102's rule
on held-out strongest: Z / A / C in [0.75, 1.5], necessity >= 0.9, not vacuous); its size is the smallest passing size.

Features available before the real fit:
  static   layer, kind, clean strength (log a_pos_pre), clean Top-K rank (targets.csv rank_clean, log), thin (no
           mid-band pool: n_train_pos < 48 + 16)
  probe    the λ 2.5e-4 fit's training curve at step 100 (zero / floor / pos terms, log n_members, mean gate): what a
           100-step low-λ probe fit would show (25% of a fit's cost; the sweep's train rows hold every W λ but 1e-3,
           which ran in the main run)
Policy: predict λ*'s grid index (classifier), optionally `shift` steps more conservative; a target predicted "never
passes" gets the smallest λ. Every policy is scored on the sweep's ACTUAL outcome at the chosen λ (no new fits), 5-fold
CV repeated over seeds, against each fixed λ and the oracle.

  python experiments/107-true-baselines/adaptive_lambda/c_predict.py
"""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
O62 = HERE.parents[1] / "062-h100-protocol-v1" / "out_full" / "out"
GRID = (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)
BAND = (0.75, 1.5)


def jl(pattern):
    out = []
    for p in sorted(glob.glob(str(pattern))):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def passes(e):
    if e.get("vacuous_tk"):
        return None
    v = [e.get(k) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")]
    nec = e.get("phi_sup_blind_tk")
    return bool(all(isinstance(x, (int, float)) and BAND[0] <= x <= BAND[1] for x in v)
                and isinstance(nec, (int, float)) and nec >= 0.9)


def load():
    ev = [r for r in jl(O62 / "sweep" / "eval.shard*.jsonl") if r.get("held") == "strong" and str(r.get("arm", "")).startswith("W_")]
    tr = {(r["seed"], r["arm"]): r for r in jl(O62 / "sweep" / "train.shard*.jsonl")}
    st = {(r["seed"], r.get("arm")): r for r in jl(O62 / "sweep" / "status.shard*.jsonl") if r.get("ok")}
    tg = pd.read_csv(O62 / "targets.csv").set_index("seed")
    by = defaultdict(dict)
    for e in ev:
        lam = float(e["arm"].split("_")[1])
        if lam in GRID:
            by[e["seed"]][lam] = e
    rows = []
    for seed, d in by.items():
        if len(d) != len(GRID):
            continue
        p = {lam: passes(d[lam]) for lam in GRID}
        if any(v is None for v in p.values()):
            continue                                              # vacuous at some λ: not scorable
        size = {}
        for lam in GRID:
            s = st.get((seed, "W_%g" % lam)) or {}
            n = s.get("n") or d[lam].get("n")
            size[lam] = n
        passing = [i for i, lam in enumerate(GRID) if p[lam]]
        oracle = max(passing) if passing else -1
        layer, kind = int(seed.split(".")[0]), seed.split(".")[1]
        r0 = d[1e-3]
        t = tr.get((seed, "W_0.00025")) or {}           # the probe: the lowest-λ fit (λ 1e-3 lives in the main run)
        c = t.get("curve") or {}
        terms = c.get("terms") or {}
        f = dict(seed=seed, layer=layer, kind=kind, oracle=oracle,
                 log_strength=math.log(max(float(r0.get("a_pos_pre") or 1e-3), 1e-3)),
                 log_rank=math.log(float(tg.loc[seed, "rank_clean"])) if seed in tg.index and
                 not pd.isna(tg.loc[seed, "rank_clean"]) else np.nan,
                 thin=float((t.get("n_train_pos") or 64) < 64),
                 p_zero=(terms.get("zero") or [np.nan] * 101)[100], p_floor=(terms.get("floor") or [np.nan] * 101)[100],
                 p_pos=(terms.get("pos") or [np.nan] * 101)[100],
                 p_logn=math.log((c.get("n_members") or [1] * 101)[100] + 1),
                 p_mm=(c.get("mean_m") or [np.nan] * 101)[100])
        for lam in GRID:
            f["pass_%g" % lam], f["n_%g" % lam] = p[lam], size[lam]
            # the scorer's TRAINING-context Z read (no held-out data): what a ladder may select on
            z = d[lam].get("free0_tr_tk")
            f["ztr_%g" % lam] = bool(isinstance(z, (int, float)) and BAND[0] <= z <= BAND[1])
        rows.append(f)
    return pd.DataFrame(rows)


STATIC = ["layer", "is_attn", "is_mlp", "log_strength", "log_rank", "thin"]
PROBE = ["p_zero", "p_floor", "p_pos", "p_logn", "p_mm"]


def outcome(df, choice):
    """pass rate, median size of passing circuits, and the passing targets' size vs their oracle size."""
    ok, sizes, ratio = [], [], []
    for (_, r), i in zip(df.iterrows(), choice):
        i = max(int(i), 0)
        lam = GRID[i]
        ok.append(bool(r["pass_%g" % lam]))
        if r["pass_%g" % lam]:
            sizes.append(r["n_%g" % lam])
            if r["oracle"] >= 0:
                ratio.append(r["n_%g" % lam] / r["n_%g" % GRID[int(r["oracle"])]])
    return np.mean(ok), np.median(sizes) if sizes else np.nan, np.median(ratio) if ratio else np.nan


def main():
    df = load()
    df["is_attn"] = (df.kind == "attn").astype(float)
    df["is_mlp"] = (df.kind == "mlp").astype(float)
    print("feature coverage (non-missing share): " + ", ".join(
        "%s %.2f" % (c, df[c].notna().mean()) for c in STATIC + PROBE if c in df))
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)    # a column missing everywhere -> 0 (uninformative)
    print("targets %d | oracle λ* index counts (-1 = never passes; 0..4 = %s): %s"
          % (len(df), GRID, dict(sorted(df.oracle.value_counts().items()))))
    print("pass at each fixed λ: " + ", ".join("%g %.2f" % (lam, df["pass_%g" % lam].mean()) for lam in GRID))
    lines = ["| policy | pass rate | median size (passing) | size vs oracle (passing) |", "|---|---|---|---|"]
    for i, lam in enumerate(GRID):
        lines.append("| fixed λ %g | %.2f | %.0f | %.2f |" % ((lam,) + outcome(df, [i] * len(df))))
    lines.append("| oracle (largest passing λ) | %.2f | %.0f | %.2f |" % outcome(df, df.oracle.clip(lower=0).values))
    # option A as a descending LADDER selected on the training-context Z read only: try the largest λ first, step
    # down until Z on the training contexts is in the band (or the grid ends); cost = fits tried
    choice, cost = [], []
    for _, r in df.iterrows():
        i = len(GRID) - 1
        while i > 0 and not r["ztr_%g" % GRID[i]]:
            i -= 1
        choice.append(i)
        cost.append(len(GRID) - i)
    lines.append("| A: ladder down from 4e-3, select on training-context Z (mean %.1f fits) | %.2f | %.0f | %.2f |"
                 % ((np.mean(cost),) + outcome(df, choice)))
    # the same ladder starting at 1e-3 (production) and going DOWN only when Z_train fails, UP while it passes
    choice, cost = [], []
    for _, r in df.iterrows():
        i, n = 2, 1
        if r["ztr_%g" % GRID[i]]:
            while i < len(GRID) - 1 and r["ztr_%g" % GRID[i + 1]]:
                i, n = i + 1, n + 1
            n += int(i < len(GRID) - 1)                         # the failing step above that stopped the climb
        else:
            while i > 0 and not r["ztr_%g" % GRID[i]]:
                i, n = i - 1, n + 1
        choice.append(i)
        cost.append(n)
    lines.append("| A: ladder from 1e-3 up / down on training-context Z (mean %.1f fits) | %.2f | %.0f | %.2f |"
                 % ((np.mean(cost),) + outcome(df, choice)))
    y = df.oracle.values
    for feats_name, feats in (("static", STATIC), ("static+probe", STATIC + PROBE)):
        X = df[feats].values
        for model_name, mk in (("GBM", lambda s: GradientBoostingClassifier(random_state=s, max_depth=2, n_estimators=150)),
                               ("RF", lambda s: RandomForestClassifier(random_state=s, n_estimators=300, min_samples_leaf=3))):
            for shift in (0, 1):
                res = []
                for seed in range(5):
                    pred = np.zeros(len(df), dtype=int)
                    for trn, tst in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, np.clip(y, -1, 4)):
                        m = mk(seed).fit(X[trn], y[trn])
                        pred[tst] = m.predict(X[tst])
                    choice = np.where(pred < 0, 0, np.clip(pred - shift, 0, len(GRID) - 1))
                    res.append(outcome(df, choice))
                res = np.array(res)
                lines.append("| C: %s, %s, shift %d | %.2f ± %.2f | %.0f | %.2f |" % (
                    model_name, feats_name, shift, res[:, 0].mean(), res[:, 0].std(), np.nanmean(res[:, 1]),
                    np.nanmean(res[:, 2])))
    print("\n".join(lines))
    (HERE / "c_predict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
