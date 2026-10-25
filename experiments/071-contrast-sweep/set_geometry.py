"""Geometry of the contrast sets: how far each context sits from the target's activating contexts, and how varied
each set is (Daniel, 2026-10-11: "what are the distances of the random ones? The diversity?").

For every target with built sets in OUT (default the local smoke, out_smoke2), and every set (close, random,
distant, fused; fused split into its close / distant / random parts):
  sim        each context's max cosine similarity to the target's top-context store in pooled final-layer
             representation space: the close selector's own ranking score (1 = like an activating context)
  diversity  mean pairwise cosine similarity between the set's own contexts (lower = more varied)
Writes OUT/geometry/contexts.csv (one row per context), OUT/geometry/summary.csv and OUT/geometry/set_geometry.png.

  OUT=experiments/071-contrast-sweep/out_smoke2 PYTHONPATH=src python experiments/071-contrast-sweep/set_geometry.py
  env: PLOT=1 (figure only, from contexts.csv; no GPU)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUT", str(HERE / "out_smoke2")))
GEO = OUT / "geometry"
SETS = ("close_silent", "random_silent", "distant_silent", "fused_silent")
LABEL = {"close_silent": "close", "random_silent": "random", "distant_silent": "distant", "fused_silent": "fused"}


def measure():
    import pandas as pd
    sys.path.insert(0, str(HERE.parent / "062-h100-protocol-v1"))
    os.environ.setdefault("OUT", str(HERE.parent / "062-h100-protocol-v1" / "out_full" / "out"))
    import driver
    from circuit.protocol_contexts import make_selector
    from pipeline.component_index import component_idx
    R = driver.Runner()
    G = R.G
    sel = make_selector(G["inference"], G["bank"], G["M0"].probe_builder.loader)
    dev = sel._ranking_device()
    keys = sorted(p.stem for p in (OUT / "sets" / "close_silent").glob("*.pt"))
    per = GEO / "per_target"                      # one json per target, so a stopped run resumes where it left off
    per.mkdir(parents=True, exist_ok=True)
    for key in keys:
        if (per / ("%s.json" % key)).exists():
            continue
        rows, summ = [], []
        l, k, i = key.split(".")
        comp = component_idx(int(l), G["KINDS"].index(k), G["NK"])
        ref, _ = sel.topctx_reference_reprs(comp, int(i))
        refn = sel._normalized_reference_reps(ref, dev)
        for name in SETS:
            p = OUT / "sets" / name / ("%s.pt" % key)
            if not p.exists():
                continue
            st = torch.load(p, weights_only=False)
            ids = [int(x) for x in st["ids"]]
            parts = st.get("parts") or [LABEL[name]] * len(ids)
            reps = sel._get_repr_for_ids(ids).float()
            valid = reps.abs().sum(dim=1) > 0
            sims = torch.full((len(ids),), float("nan"))
            if bool(valid.any()):
                sims[valid] = sel._positive_set_max_similarity(reps[valid], refn, dev).float().cpu()
            v = torch.nn.functional.normalize(reps[valid], dim=1)
            cos = v @ v.T
            n = int(v.shape[0])
            div = float((cos.sum() - cos.diag().sum()) / (n * (n - 1))) if n > 1 else float("nan")
            for j, (cid, s) in enumerate(zip(ids, sims.tolist())):
                rows.append(dict(seed=key, set=LABEL[name], part=str(parts[j]).replace("_silent", ""),
                                 split="train" if j < st["n_train"] else "held", seq_id=cid, sim=s))
            s_ok = sims[~torch.isnan(sims)].numpy()
            summ.append(dict(seed=key, set=LABEL[name], n=len(ids), n_valid=n, sim_med=np.median(s_ok),
                             sim_p10=np.quantile(s_ok, .1), sim_p90=np.quantile(s_ok, .9), sim_min=s_ok.min(),
                             sim_max=s_ok.max(), sim_iqr=np.quantile(s_ok, .75) - np.quantile(s_ok, .25),
                             within_cos=div))
        tmp = per / ("%s.json.tmp" % key)
        tmp.write_text(json.dumps(dict(rows=rows, summ=summ), default=float))
        tmp.replace(per / ("%s.json" % key))
        print("%-16s done" % key, flush=True)
    rows, summ = [], []
    for p in sorted(per.glob("*.json")):
        d = json.loads(p.read_text())
        rows += d["rows"]
        summ += d["summ"]
    pd.DataFrame(rows).to_csv(GEO / "contexts.csv", index=False)
    pd.DataFrame(summ).to_csv(GEO / "summary.csv", index=False)


def report_and_plot():
    import matplotlib.pyplot as plt
    import pandas as pd
    from analysis import style as S
    c = pd.read_csv(GEO / "contexts.csv")
    s = pd.read_csv(GEO / "summary.csv")
    order = ["close", "random", "distant", "fused"]
    print("Similarity to the target's top contexts (pooled over %d targets; 1 = like an activating context)"
          % c.seed.nunique())
    q = c.groupby("set").sim.describe(percentiles=[.1, .25, .5, .75, .9]).loc[order]
    print(q[["count", "min", "10%", "25%", "50%", "75%", "90%", "max"]].round(3).to_string())
    print("\nfused, by part:")
    print(c[c.set == "fused"].groupby("part").sim.describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "90%"]]
          .round(3).to_string())
    print("\nPer target (median over targets): spread and within-set similarity (lower = more varied)")
    print(s.groupby("set")[["sim_iqr", "sim_p10", "sim_p90", "within_cos"]].median().loc[order].round(3).to_string())

    S.configure_matplotlib()
    colors = {"close": S.NEG_MODE_COLORS["close"], "random": S.NEG_MODE_COLORS["random"],
              "distant": S.NEG_MODE_COLORS["distant"], "fused": S.CATEGORICAL[3]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    rng = np.random.default_rng(0)
    for x, name in enumerate(order):
        d = c[c.set == name]
        if name == "fused":
            for part, off in (("close", -0.22), ("random", 0.0), ("distant", 0.22)):
                dp = d[d.part == part].sim.dropna().values
                a1.scatter(x + off + rng.uniform(-0.07, 0.07, len(dp)), dp, s=6, alpha=0.35,
                           color=S.tint(colors[part], 0.15), edgecolors="none")
        else:
            dv = d.sim.dropna().values
            a1.scatter(x + rng.uniform(-0.25, 0.25, len(dv)), dv, s=6, alpha=0.35, color=colors[name],
                       edgecolors="none")
        dv = d.sim.dropna().values
        for qq, w in ((np.quantile(dv, [.25, .75]), 3.0),):
            a1.plot([x + 0.33, x + 0.33], qq, color=S.INK, lw=w, solid_capstyle="round")
        a1.plot([x + 0.28, x + 0.38], [np.median(dv)] * 2, color=S.INK, lw=1.5)
    a1.set_xticks(range(len(order)), order)
    a1.set_ylabel("similarity to the target's top contexts")
    a1.set_title("Each contrast context (%d targets, 64 per set)" % c.seed.nunique())
    for x, name in enumerate(order):
        v = s[s.set == name].within_cos.values
        a2.scatter(np.full(len(v), x) + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=colors[name],
                   edgecolors="none", alpha=0.85)
        a2.plot([x - 0.25, x + 0.25], [np.median(v)] * 2, color=S.INK, lw=1.5)
    a2.set_xticks(range(len(order)), order)
    a2.set_ylabel("mean pairwise similarity within the set")
    a2.set_title("How varied each set is (lower = more varied)")
    fig.tight_layout()
    S.save_figure(fig, GEO / "set_geometry.png")
    print("\nfigure: %s" % (GEO / "set_geometry.png"))


if __name__ == "__main__":
    if not os.environ.get("PLOT"):
        measure()
    report_and_plot()
