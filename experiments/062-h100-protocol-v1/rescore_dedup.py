"""Re-score 15k-run circuits on genuinely unseen held-out contexts (exact-duplicate leakage, dup_leakage.py).

25% of the run's held-out strongest contexts are token-identical to a context the circuit trained on (corpus
boilerplate; 92% of targets have at least one). Leaky targets pass LESS often than clean ones in every cell, but that
compares different latents. This re-scores the SAME circuits on their held-out strongest set with every row that is
identical to a fit-training row (or to an earlier held-out row) removed, against the unchanged 48 strongest-train
reference, so the two rows per target differ only in the leaked contexts. Nothing in out_full is written.

Targets: N_PER per (depth band, kind) cell from the leaky targets with at least MIN_CLEAN clean held-out rows, cells
interleaved so a partial run is still stratified. Resumable.

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/rescore_dedup.py
  env: N_PER (200)  MIN_CLEAN (4)  OUT_RD (default experiments/062-h100-protocol-v1/out_rescore_dedup)
       REPORT=1 (paired before / after summary; no GPU)
"""
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
os.environ["EVAL_SPLIT"] = "explicit"
sys.path.insert(0, str(HERE))
import driver  # noqa: E402

N_PER = int(os.environ.get("N_PER", 200))
MIN_CLEAN = int(os.environ.get("MIN_CLEAN", 4))
OUT_RD = Path(os.environ.get("OUT_RD", str(HERE / "out_rescore_dedup")))
LEAK = HERE / "out_dup_leakage" / "leakage.csv"
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
from merge import BAND, NEC_MIN  # noqa: E402  (the DAN-8 pass rule)


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def rows_of(t):
    return [tuple(r) for r in t.tolist()]


def clean_held(rec):
    """Indices (into the strongest pool) of held-out rows not identical to a fit-training row nor to an earlier
    held-out row."""
    s = rec["strong"]
    pos = s["pos"]
    if rec.get("mid") is not None:
        fit = set(rows_of(pos[rec["D_train"]])) | set(rows_of(rec["mid"]["pos"][rec["B_mid"]]))
    else:
        fit = set(rows_of(pos[s["train"]]))
    keep, seen = [], set()
    for j in list(s["held"]):
        r = tuple(pos[j].tolist())
        if r in fit or r in seen:
            continue
        seen.add(r)
        keep.append(int(j))
    return keep


def pick_targets():
    import pandas as pd
    d = pd.read_csv(LEAK)
    d = d[(d.leak_fit > 0) & (d.n_held - d.leak_fit - d.dup_held >= MIN_CLEAN)]
    cells = defaultdict(list)
    for k in d.seed:
        l, kind, _ = k.split(".")
        cells[(band(int(l)), kind)].append(k)
    rng = random.Random(20261009)
    picks = [rng.sample(sorted(cells[c]), min(N_PER, len(cells[c]))) for c in sorted(cells)]
    return [p[i] for i in range(N_PER) for p in picks if i < len(p)]


def probe_dedup(rec, keep, device):
    """48 strongest-train (the unchanged evaluation reference) then the clean held-out rows; n_train explicit."""
    import torch
    from circuit.probe_dataset import ProbeDataset
    s = rec["strong"]
    idx = torch.tensor([int(i) for i in s["train"]] + keep, dtype=torch.long)
    return ProbeDataset(pos_tokens=s["pos"][idx].to(device), target_tokens=s["tgt"][idx].to(device),
                        neg_tokens=rec["neg"].to(device), pos_argmax=s["arg"][idx].to(device),
                        metadata=dict(n_train=len(s["train"]), n=int(idx.numel())))


def main():
    import torch
    from circuit.protocol_contexts import FixedSelector
    OUT_RD.mkdir(parents=True, exist_ok=True)
    path = OUT_RD / "eval_dedup.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["seed"] for l in open(path) if "error" not in json.loads(l)}
    keys = pick_targets()
    print("rescore dedup: %d targets (%d per cell), %d done" % (len(keys), N_PER, len(done)), flush=True)
    R = driver.Runner()
    G = R.G
    fh = open(path, "a")
    t0 = time.time()
    for n_, key in enumerate(keys):
        if key in done:
            continue
        try:
            rec = R.contexts(key)
            keep = clean_held(rec)
            pd_ = probe_dedup(rec, keep, G["device"])
            sel = FixedSelector(rec["neg"].to(G["device"]))
            G["M0"].build_probe_dataset = lambda comp, i, _p=pd_: _p
            G["M0"]._neg_context_selector = lambda _s=sel: _s
            c = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
            row = R.V.score_circuit(c, {}, skip_roles=True)
            row.update(held="strong_dedup", n_held_orig=len(rec["strong"]["held"]), n_held_clean=len(keep))
        except Exception as e:  # noqa: BLE001
            row = dict(seed=key, held="strong_dedup", error="%s: %s" % (type(e).__name__, str(e)[:300]))
            traceback.print_exc()
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("%4d/%d %-16s held %s -> %s  %s  %.0fs" % (n_ + 1, len(keys), key, row.get("n_held_orig"),
                                                       row.get("n_held_clean"), row.get("error", "ok"), time.time() - t0),
              flush=True)
    print("DONE", flush=True)


def report():
    import glob
    import pandas as pd
    new = pd.DataFrame([json.loads(l) for l in open(OUT_RD / "eval_dedup.jsonl")])
    new = new[new.get("error", pd.Series(index=new.index, dtype=object)).isna()].drop_duplicates("seed", keep="last")
    ev = {}
    for f in glob.glob(str(HERE / "out_full" / "out" / "main" / "eval.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "error" not in r and r.get("held") == "strong":
                ev[r["seed"]] = r
    old = pd.DataFrame([ev[k] for k in new.seed if k in ev])

    def flag(d):
        d = d.set_index("seed")
        ok = d[HEAD].notna().all(axis=1)
        p = ok & ((d[HEAD] >= BAND[0]) & (d[HEAD] <= BAND[1])).all(axis=1) & (d["phi_sup_blind_tk"] >= NEC_MIN)
        return d.assign(passed=p, dev=(d[HEAD] - 1).abs().max(axis=1),
                        vac=d["vacuous_tk"].fillna(False).astype(bool))

    a, b = flag(old), flag(new)
    j = a[["passed", "dev", "vac"] + HEAD].join(b[["passed", "dev", "vac"] + HEAD + ["n_held_clean"]],
                                                lsuffix="_all", rsuffix="_clean", how="inner")
    j = j[~j.vac_all & ~j.vac_clean]
    j["band"] = [band(int(k.split(".")[0])) for k in j.index]
    j["kind"] = [k.split(".")[1] for k in j.index]
    print("paired targets: %d (clean held-out rows: median %d of 16)" % (len(j), j.n_held_clean.median()))
    print("pass rate: all held-out %.1f%%  ->  unseen held-out only %.1f%%" % (100 * j.passed_all.mean(),
                                                                               100 * j.passed_clean.mean()))
    print("flips: pass->fail %d, fail->pass %d" % (int((j.passed_all & ~j.passed_clean).sum()),
                                                  int((~j.passed_all & j.passed_clean).sum())))
    print("median worst-of-3: %.3f -> %.3f" % (j.dev_all.median(), j.dev_clean.median()))
    for h in HEAD:
        print("  %-15s median %.3f -> %.3f" % (h, j[h + "_all"].median(), j[h + "_clean"].median()))
    print("\nby cell (pass rate all -> clean, n):")
    g = j.groupby(["band", "kind"]).agg(all=("passed_all", "mean"), clean=("passed_clean", "mean"), n=("passed_all", "size"))
    print(g.round(3).to_string())


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else main()
