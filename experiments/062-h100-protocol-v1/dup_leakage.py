"""Exact-duplicate leakage between training and held-out contexts in the 15k run (found by DAN-37's duplicate report).

About a quarter of TuringLLM's strongest stored contexts are exact duplicate token windows (corpus boilerplate), so a
held-out strongest context can be token-identical to one the circuit trained on. Per target, from the run's cached
context records (no GPU, nothing re-scored):

  leak_fit   held-out strongest rows identical to a row of the fit's training set (32 strongest + 16 mid, or arm A)
  leak_ref   held-out strongest rows identical to a row of the 48 strongest-train evaluation reference
  dup_held   held-out strongest rows identical to another held-out row

then the DAN-8 pass rate (held-out strongest) for targets with and without fit leakage, overall and within
(layer band, kind) cells, since duplicate-heavy latents are not a random subset.

  PYTHONPATH=src python experiments/062-h100-protocol-v1/dup_leakage.py        (writes out_dup_leakage/leakage.csv)
  REPORT=1 ... (summary only, from the csv)
"""
import glob
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
CTX = HERE / "out_full" / "out" / "ctx"
OUT = HERE / "out_dup_leakage"
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]


def rows_of(t):
    return [tuple(r) for r in t.tolist()]


def measure():
    import pandas as pd
    import torch
    out = []
    files = sorted(glob.glob(str(CTX / "*.pt")))
    for n, p in enumerate(files):
        key, rec = next(iter(torch.load(p, weights_only=False).items()))
        s = rec.get("strong")
        if s is None:
            continue
        pos = s["pos"]
        held = rows_of(pos[s["held"]])
        ref = set(rows_of(pos[s["train"]]))
        if rec.get("mid") is not None:
            fit = set(rows_of(pos[rec["D_train"]])) | set(rows_of(rec["mid"]["pos"][rec["B_mid"]]))
        else:
            fit = ref
        out.append(dict(seed=key, n_held=len(held), leak_fit=sum(h in fit for h in held),
                        leak_ref=sum(h in ref for h in held), dup_held=len(held) - len(set(held)),
                        dup_pool=len(rows_of(pos)) - len(set(rows_of(pos)))))
        if n % 2000 == 0:
            print("%d/%d" % (n, len(files)), flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(OUT / "leakage.csv", index=False)


def report():
    import pandas as pd
    d = pd.read_csv(OUT / "leakage.csv")
    ev = {}
    for f in glob.glob(str(HERE / "out_full" / "out" / "main" / "eval.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "error" not in r and r.get("held") == "strong":
                ev[r["seed"]] = r
    e = pd.DataFrame(list(ev.values()))
    e = e[~e["vacuous_tk"].fillna(False).astype(bool)]
    ok = e[HEAD].notna().all(axis=1)
    e["passed"] = ok & ((e[HEAD] >= 0.8) & (e[HEAD] <= 1.5)).all(axis=1) & (e["phi_sup_blind_tk"] >= 0.9)
    d = d.merge(e[["seed", "passed"]], on="seed")
    d["band"] = d.seed.map(lambda k: "L0-4" if int(k.split(".")[0]) <= 4 else ("L5-7" if int(k.split(".")[0]) <= 7 else "L8-11"))
    d["kind"] = d.seed.map(lambda k: k.split(".")[1])
    d["leaky"] = d.leak_fit > 0
    tot = d.n_held.sum()
    print("targets %d | held-out strongest rows %d" % (len(d), tot))
    print("rows identical to a fit-training row:      %6d (%.1f%%), targets with any: %d (%.1f%%)" % (
        d.leak_fit.sum(), 100 * d.leak_fit.sum() / tot, d.leaky.sum(), 100 * d.leaky.mean()))
    print("rows identical to an eval-reference row:   %6d (%.1f%%)" % (d.leak_ref.sum(), 100 * d.leak_ref.sum() / tot))
    print("rows duplicated within the held-out set:   %6d (%.1f%%)" % (d.dup_held.sum(), 100 * d.dup_held.sum() / tot))
    print("pools with any duplicate row: %.1f%%; duplicate rows in the 64-pools: %.1f%%" % (
        100 * (d.dup_pool > 0).mean(), 100 * d.dup_pool.sum() / (64 * len(d))))
    print("\nleaked held-out rows per leaky target: median %d, p90 %d, max %d" % (
        d[d.leaky].leak_fit.median(), d[d.leaky].leak_fit.quantile(.9), d[d.leaky].leak_fit.max()))
    print("\npass rate: leaky %.1f%% (n=%d) vs clean %.1f%% (n=%d)" % (
        100 * d[d.leaky].passed.mean(), d.leaky.sum(), 100 * d[~d.leaky].passed.mean(), (~d.leaky).sum()))
    g = d.groupby(["band", "kind", "leaky"]).passed.agg(["mean", "size"]).unstack("leaky")
    print("\nby cell (pass rate, n):")
    print((g.round(3)).to_string())
    # cell-standardised difference: weight each cell's (leaky - clean) by its leaky count
    w = []
    for (b, k), x in d.groupby(["band", "kind"]):
        if x.leaky.any() and (~x.leaky).any():
            w.append((x.leaky.sum(), x[x.leaky].passed.mean() - x[~x.leaky].passed.mean()))
    if w:
        num = sum(n * v for n, v in w); den = sum(n for n, _ in w)
        print("\ncell-standardised pass difference, leaky - clean: %+.1f points" % (100 * num / den))
    print("headline pass rate if leaky targets are dropped: %.1f%% (all: %.1f%%)" % (
        100 * d[~d.leaky].passed.mean(), 100 * d.passed.mean()))


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else (measure(), report())
