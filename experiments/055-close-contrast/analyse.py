"""Analyse the 055 matrix: training contrast source (old store / close / random / distant) x evaluation contrast source
(close / random / distant / store), 16 pilot targets, v3 eval rows (activation read, held-out contexts).

  PYTHONPATH=src python experiments/055-close-contrast/analyse.py   -> printed tables + results/summary.md
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
ARMS = ["old", "store", "close", "random", "distant",    # store = local same-C refit, the noise floor for "old"
        "eq1", "eq2",                                    # close C with gamma_C = gamma_A = 1 at lambda 1e-3 / 2e-3
        "otall3", "otcut3", "otcut3e3", "otcut2",        # close C + off-target term (056): gamma_S 1e-3 all; cut at 1e-3 / 3e-3 / 1e-2
        "otrank2",                                       # otcut3e3 + Top-K hinge gamma_R 1e-2 (the provisional primary config)
        "rankonly"]                                      # Top-K hinge gamma_R 1e-2 alone, no off-target term
EVS = ["close", "random", "distant", "store"]
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]    # faithfulness under Z, A (sp), C (sp): the headline three
BAND = (0.8, 1.25)
# (field, label); every field reads the activation (post-Top-K) on held-out contexts
METRICS = [("free0_tk", "faith Z"), ("freeM_topk_tk", "faith A (sp)"), ("freeN_topk_tk", "faith C (sp)"),
           ("freeN_dense_tk", "faith C (dense)"), ("phi_sup_blind_tk", "necessity"), ("phi_cf_alpha_blind_tk", "suff. to induce"),
           ("phi_pin_alpha_topk_tk", "clamped A (sp)"), ("free0_rand_tk", "random baseline Z")]


def load_rows():
    rows = []
    for arm in ARMS:
        for ev in EVS:
            p = HERE / "results" / ("amp_eval_v3_%s_ev%s.jsonl" % (arm, ev))
            if p.exists():
                for l in open(p):
                    r = json.loads(l)
                    if "skip" not in r and "error" not in r:
                        rows.append(dict(r, arm=arm, ev=ev))
    return pd.DataFrame(rows)


def members(arm, seeds):
    """seed -> set of node names, for Jaccard between training arms."""
    import sys
    sys.path.insert(0, str(ROOT))
    import amp_eval_pass_v2 as V
    out = {}
    if arm == "old":
        ct = pd.read_parquet(ROOT / "tables_full" / "circuits.parquet")
        ct["skey"] = ct.seed_layer.astype(str) + "." + ct.seed_kind.astype(str) + "." + ct.seed_index.astype(str)
        paths = [ROOT / "data_full" / ("discovered_circuits.shard%d.pt" % i) for i in sorted(set(ct[ct.skey.isin(seeds)]["shard"]))]
    else:
        paths = [HERE / ("data_%s" % arm) / "discovered_circuits.shard0.pt"]
    for p in paths:
        if not Path(p).exists():
            continue
        for c in torch.load(p, weights_only=False, map_location="cpu").values():
            k = V.key_of(c)
            if k in seeds:
                s = V.seed_of(c)
                out[k] = {"%d.%s.%d" % (n.metadata["feature_id"].layer, n.metadata["feature_id"].kind, n.metadata["feature_id"].index)
                          for n in c.nodes.values() if n is not s}
    return out


def main():
    df = load_rows()
    seeds = [s for s in (HERE / "seeds.txt").read_text().split() if s]
    lines = ["# 055 contrast-source matrix (%d targets, activation read, held-out contexts)\n" % len(seeds)]

    def emit(s=""):
        print(s); lines.append(s)

    emit("rows per (arm, eval source):")
    emit(df.groupby(["arm", "ev"]).size().unstack().reindex(index=ARMS, columns=EVS).to_string())

    emit("\n## HEADLINE: faithfulness under each ablation method (freeN on close contrast contexts)")
    emit("band = share of targets with ALL THREE in [%.2f, %.2f] (illustrative; the pass rule is DAN-8)" % BAND)
    hd = df[df.ev == "close"]
    lo, hi = BAND
    ref = hd[hd.arm == "close"].set_index("seed")
    for arm in ARMS:
        a = hd[hd.arm == arm].set_index("seed")
        if a.empty:
            continue
        inb = ((a[HEAD] >= lo) & (a[HEAD] <= hi)).all(axis=1)
        cells = " | ".join("%s %.3f" % (lab, a[f].median()) for f, lab in zip(HEAD, ("free0", "freeM_topk", "freeN_topk")))
        dense = " | ".join("%s %.3f" % (f.replace("_tk", ""), a[f].median()) for f in ("freeM_dense_tk", "freeN_dense_tk"))
        common = a.index.intersection(ref.index)
        delta = " ".join("%+.3f" % (a.loc[common, f] - ref.loc[common, f]).median() for f in HEAD) if arm != "close" else "(reference)"
        emit("%-8s nodes %4d | %s | all three in band %2d/%d | dense: %s | vs close@0.25 %s"
             % (arm, a.n_members.median(), cells, int(inb.sum()), len(a), dense, delta))

    emit("\n## Circuit size (nodes), by training source")
    size = df[df.ev == "close"].groupby("arm").n_members.describe()[["count", "25%", "50%", "75%"]].reindex(ARMS)
    emit(size.to_string())

    for f, lab in METRICS:
        if f not in df:
            continue
        emit("\n## %s (median; rows = training source, columns = evaluation contrast source)" % lab)
        emit(df.pivot_table(index="arm", columns="ev", values=f, aggfunc="median").reindex(index=ARMS, columns=EVS).round(3).to_string())

    emit("\n## Evaluated on close contrast contexts, split by whether the target was on the store's fallback list")
    sub = df[df.ev == "close"]
    for f, lab in METRICS:
        if f in sub:
            t = sub.pivot_table(index="arm", columns="ctr_fallback", values=f, aggfunc="median").reindex(ARMS).round(3)
            t.columns = ["retrieved" if not c else "fallback" for c in t.columns]
            emit("%-18s %s" % (lab, t.to_dict("index")))

    emit("\n## Paired change vs the old circuits (evaluated on close; per target new - old, median and share improved)")
    base = sub[sub.arm == "old"].set_index("seed")
    for arm in [a for a in ARMS if a != "old"]:
        cur = sub[sub.arm == arm].set_index("seed")
        common = base.index.intersection(cur.index)
        parts = []
        for f, lab in METRICS:
            if f in cur:
                d = (cur.loc[common, f] - base.loc[common, f]).dropna()
                if len(d):
                    parts.append("%s %+.3f (%.0f%% up)" % (lab, d.median(), 100 * (d > 0).mean()))
        emit("%-8s n=%d | %s" % (arm, len(common), " | ".join(parts)))

    emit("\n## Node overlap between training sources (Jaccard, median over targets)")
    mem = {a: members(a, set(seeds)) for a in ARMS}
    for i, a in enumerate(ARMS):
        cells = []
        for b in ARMS:
            js = [len(mem[a][s] & mem[b][s]) / max(1, len(mem[a][s] | mem[b][s])) for s in seeds if s in mem[a] and s in mem[b]]
            cells.append("%s %.2f" % (b, float(np.median(js)) if js else float("nan")))
        emit("%-8s %s" % (a, "  ".join(cells)))

    (HERE / "results" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n->", HERE / "results" / "summary.md")


if __name__ == "__main__":
    main()
