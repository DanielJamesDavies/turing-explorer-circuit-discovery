"""115: independent check of the built held-out sets (CPU). Recomputes the groups from the raw files (data/text_paths.txt,
085 corpus_src.npy / corpus_hash.npy) without build_heldout.py's Groups class, and checks per target:
  - no new held-out row (nor any of its content twins) shares a document / topic with a training row (strong.train +
    mid.train, with their twins);
  - the new held-out set has the original's size;
  - every training-side field equals the production record's (strong.train, D_train, mid, B_mid, neg, neg_ids, first
    n0 rows of strong.*).
Prints one line per target and a decoded example (file paths) for the first target.

  python -X utf8 experiments/115-docdisjoint-rescore-pilot/check_build.py [GROUP=doc|topic ...]
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CTX_FRESH = (REPO / "experiments/109-pod-run-analysis/out/archives/pub-turingllm-T2-records/root/turing-tl/experiments/"
             "082-public-runs/out_turingllm/ctx_fresh")
C85 = REPO / "experiments/085-heldout-audit/out"


def main():
    groups = sys.argv[1:] or ["doc", "topic"]
    paths = [p for p in (REPO / "data/text_paths.txt").read_text().split("\n") if p]
    rel = [p.split("/dataset/", 1)[1] for p in paths]
    src = np.load(C85 / "corpus_src.npy")
    h = np.load(C85 / "corpus_hash.npy")
    order = np.argsort(h[1:], kind="stable"); sh = h[1:][order]

    def twins(i):
        lo, hi = np.searchsorted(sh, h[i]), np.searchsorted(sh, h[i], side="right")
        return order[lo:hi] + 1

    def grp(i, g):
        r = [rel[s] for s in src[twins(int(i))]]
        return set(r) if g == "doc" else {os.path.dirname(x) for x in r}

    bad = 0
    for g in groups:
        d = HERE / "out" / ("ctx_%s" % g)
        for f in sorted(d.glob("*.pt")):
            key = f.stem
            new = torch.load(f, weights_only=False)[key]
            old = torch.load(CTX_FRESH / f.name, weights_only=False)[key]
            s, o = new["strong"], old["strong"]
            tr = [o["ids"][j] for j in o["train"]] + ([old["mid"]["ids"][j] for j in old["mid"]["train"]]
                                                      if old.get("mid") is not None else [])
            forb = set().union(*[grp(i, g) for i in tr])
            held = [s["ids"][j] for j in s["held"]]
            hit = [i for i in held if grp(i, g) & forb]
            n0 = int(o["pos"].shape[0])
            same = (s["train"] == o["train"] and new["D_train"] == old["D_train"] and new.get("B_mid") == old.get("B_mid")
                    and torch.equal(new["neg"], old["neg"]) and new.get("neg_ids") == old.get("neg_ids")
                    and torch.equal(s["pos"][:n0], o["pos"]) and list(s["ids"][:n0]) == list(o["ids"])
                    and (old.get("mid") is None or torch.equal(new["mid"]["pos"], old["mid"]["pos"]))
                    and (old.get("mid") is None or new["mid"]["held"] == old["mid"]["held"]))
            size_ok = len(s["held"]) == len(o["held"])
            ok = not hit and same and size_ok and len(set(held)) == len(held)
            bad += not ok
            print("%-6s %-16s held %2d (orig %2d) train rows %3d forbidden %s groups %4d  intersect %d  training fields "
                  "unchanged %s  -> %s" % (g, key, len(s["held"]), len(o["held"]), len(tr), g, len(forb), len(hit), same,
                                           "OK" if ok else "FAIL"))
            if f == sorted(d.glob("*.pt"))[0]:
                print("   example new held rows: %s" % [(i, paths[src[i]].split("/dataset/")[1]) for i in held[-2:]])
                print("   example train rows:    %s" % [(i, paths[src[i]].split("/dataset/")[1]) for i in tr[:2]])
    print("FAILURES: %d" % bad)


if __name__ == "__main__":
    main()
