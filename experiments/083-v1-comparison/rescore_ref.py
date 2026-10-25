"""Reference rows on the run's own scorer and contexts: re-score the stored 062 circuits (WCM at every price,
unweighted at every price) with driver.Runner.score, never refitting them.

Two reasons to run it: (1) the held-out sets were rebuilt (DAN-157: the driver's CTX_DIR, e.g. 062/out_full/out/ctx_v2)
and every reference row must be re-scored on the rebuilt sets (table2.py then uses REF=local); (2) same-machine
scoring, so the WCM / unweighted rows and the 083 rows share one GPU type (H100 and RTX refits differ, golden Jaccard
0.41-0.96; re-scoring a stored circuit is a smaller, scoring-only difference).

Circuits: 062/out_full/out/sweep/circuits/<arm>/ and the weak-penalty run directories (out_bands / out_depth /
out_lowlam). REF_ARMS: "W_all" / "U_all" (every price the target has; the default, for the DAN-77 curves), named arms
("W_0.001", "U_0.001", ...), or "U_matched" (the unweighted arm nearest W_0.001's size: a secondary note only, since
Daniel chose the same-price comparison).

  CUDA_VISIBLE_DEVICES=0 SHARD=0/1 PYTHONPATH=src python experiments/083-v1-comparison/rescore_ref.py
  env: REF_ARMS (W_all,U_all)  TARGETS  LIMIT  SHARD  OUT083  CTX_DIR (read by driver)
Rows keyed (seed, arm, ctx_dir).
"""
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cmp_common as C  # noqa: E402

REF_ARMS = [a for a in os.environ.get("REF_ARMS", "W_all,U_all").split(",") if a]


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), C.TAG, s), flush=True)


def arms_for(have):
    """Concrete sweep arms for a target, `have` = {arm: n}: [(arm on disk, label)]."""
    out = []
    for a in REF_ARMS:
        if a == "U_matched":
            u = C.nearest_size(have.get(C.REF_ARM), {k: v for k, v in have.items() if k.startswith("U_")})
            if u:
                out.append((u, "U_matched"))
        else:
            out += [(x, x) for x in C.resolve_sizes(have, [a])]
    return list({a: (a, lab) for a, lab in reversed(out)}.values())[::-1]     # one row per arm (first label wins)


def main():
    driver = C.import_driver()
    targets = C.shard_targets()
    sizes = C.sizes_by_arm(C.sweep_rows())
    d = C.OUT / "ref"
    d.mkdir(parents=True, exist_ok=True)
    rows_path = d / ("rows.%s.jsonl" % C.TAG)
    done = C.done_keys(rows_path, ("seed", "arm", "ctx_dir"))
    log("re-score 062 circuits | %d targets | arms %s" % (len(targets), REF_ARMS))
    R = driver.Runner()
    ctx_dir = str(R.ctx_dir)
    fh = open(rows_path, "a")
    for n_, key in enumerate(targets):
        rec = h = None
        for arm, label in arms_for(sizes.get(key, {})):
            if (key, arm, ctx_dir) in done:
                continue
            row = dict(seed=key, arm=arm, label=label, held="strong", rescored="local", shard=C.TAG)
            try:
                if rec is None:
                    rec = R.contexts(key)
                    h = C.training_contexts(R, rec)[4]
                path = C.circuit_path(arm, key)
                if path is None:
                    raise FileNotFoundError("no stored circuit for %s %s" % (arm, key))
                row.update(circuit=os.path.relpath(path, C.ROOT).replace(os.sep, "/"), **C.provenance(R, h))
                c = torch.load(path, weights_only=False)
                t = time.time()
                r = C.scalar_row(R.score(c, rec, "strong"))
                r.update(row, weighted=arm.startswith("W_"), lam=float(arm.split("_")[1]), t_score=round(time.time() - t, 1))
                row = r
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            C.write_row(fh, row)
            log("%4d/%d %-16s %-10s n=%s Z %s A %s C %s nec %s" % (
                n_ + 1, len(targets), key, arm, row.get("n"),
                *("%.2f" % row[c_] if isinstance(row.get(c_), float) else "-" for c_ in C.HEAD + [C.NEC])))
    log("DONE")


if __name__ == "__main__":
    main()
