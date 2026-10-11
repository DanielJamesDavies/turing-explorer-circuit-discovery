"""Re-score the mid-band rows hit by the padding defect (protocol-v1.md §3.6), with the explicit eval split.

In the 15k run, 58 mid-band rows had fewer than 16 held-out mid-band contexts. The scorer's 3:1 split then moved
1-11 strongest-train contexts into the "held-out" slice (and the A mean / pins used fewer than 48 contexts). This
re-scores exactly those rows with EVAL_SPLIT=explicit (amp_eval_pass_v2: the train count comes from the probe), on the
run's own cached contexts and circuits. Nothing in out_full is written.

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/rescore_mid.py
  env: OUT_RM (default experiments/062-h100-protocol-v1/out_rescore_mid)  REPORT=1 (before / after summary; no GPU)
"""
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
os.environ["EVAL_SPLIT"] = "explicit"
sys.path.insert(0, str(HERE))
import driver  # noqa: E402

OUT_RM = Path(os.environ.get("OUT_RM", str(HERE / "out_rescore_mid")))
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]


def run_rows(held):
    """The run's eval rows for `held`, last row per target (merge.py keeps the last)."""
    out = {}
    for f in sorted(glob.glob(str(driver.OUT / "main" / "eval.shard*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "error" not in r and r.get("held") == held:
                out[r["seed"]] = r
    return out


def affected():
    """Mid-band rows whose probe was short (n_pos < 64): exactly the padded rows of §3.6."""
    return sorted(k for k, r in run_rows("mid").items() if r["n_pos"] < 64)


def main():
    import torch
    OUT_RM.mkdir(parents=True, exist_ok=True)
    path = OUT_RM / "eval_mid.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["seed"] for l in open(path) if "error" not in json.loads(l)}
    keys = affected()
    print("rescore mid-band: %d affected rows (%d done)" % (len(keys), len(done)), flush=True)
    R = driver.Runner()
    fh = open(path, "a")
    t0 = time.time()
    for n_, key in enumerate(keys):
        if key in done:
            continue
        try:
            rec = R.contexts(key)                                       # the run's cached record (read only)
            arm, thin = R.train_arm(rec)
            c = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
            row = R.score(c, rec, "mid")
            row.update(held="mid", arm=arm, thin=thin, eval_split="explicit",
                       n_held_mid=len(rec["mid"]["held"]), n_strong_train=len(rec["strong"]["train"]))
        except Exception as e:  # noqa: BLE001
            row = dict(seed=key, held="mid", error="%s: %s" % (type(e).__name__, str(e)[:300]))
            traceback.print_exc()
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("%3d/%d %-16s n_train %s n_ho %s  %s  %.0fs" % (n_ + 1, len(keys), key, row.get("n_train"),
                                                          row.get("n_ho"), row.get("error", "ok"), time.time() - t0),
              flush=True)
    print("DONE", flush=True)


def report():
    import numpy as np
    import pandas as pd
    new = {r["seed"]: r for r in map(json.loads, open(OUT_RM / "eval_mid.jsonl")) if "error" not in r}
    old = run_rows("mid")

    def summarise(rows, label):
        d = pd.DataFrame(list(rows.values()))
        d = d[~d["vacuous_tk"].fillna(False).astype(bool)]
        ok = d[HEAD].notna().all(axis=1)
        p = ok & ((d[HEAD] >= 0.8) & (d[HEAD] <= 1.5)).all(axis=1) & (d["phi_sup_blind_tk"] >= 0.9)
        dev = (d[HEAD] - 1).abs().max(axis=1)                     # as merge.py: worst over the defined scores
        print("%-34s rows %5d  pass %.2f%%  median worst-of-3 %.3f" % (label, len(d), 100 * p.mean(), dev.median()))
        return d.assign(passed=p, dev=dev).set_index("seed")

    keys = sorted(new)
    print("check: n_train now equals the strongest-train count on %d / %d rows; held-out = the mid-band held set on %d"
          % (sum(new[k]["n_train"] == new[k]["n_strong_train"] for k in keys), len(keys),
             sum(new[k]["n_ho"] == new[k]["n_held_mid"] for k in keys)))
    a = summarise({k: old[k] for k in keys}, "the %d rows, as run" % len(keys))
    b = summarise(new, "the %d rows, explicit split" % len(keys))
    j = a[["passed", "dev"]].join(b[["passed", "dev"]], lsuffix="_old", rsuffix="_new", how="inner")
    print("pass flips: %d fail->pass, %d pass->fail; median |dev change| %.3f" % (
        int((~j.passed_old & j.passed_new).sum()), int((j.passed_old & ~j.passed_new).sum()),
        float(np.abs(j.dev_new - j.dev_old).median())))
    summarise(old, "ALL mid-band rows, as run")
    summarise({**old, **new}, "ALL mid-band rows, 58 replaced")


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else main()
