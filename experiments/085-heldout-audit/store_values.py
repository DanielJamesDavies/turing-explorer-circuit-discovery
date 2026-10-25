"""What the TuringLLM stores hold beyond each target's 64 strongest contexts, and how the split meets duplicates (085).

Per 15k target (targets_full.txt), from outputs/top_ctx.pt and outputs/mid_ctx.pt (read-only) + the corpus index:
  top store   slots filled, store mean-score (the store ranks by MEAN activation over the window, not by peak) at
              ranks 1 / 16 / 32 / 48 / 64, unique contents among the 64 (corpus hash), and the decay v64 / v1
  mid store   slots filled, mean-score range vs the top store's weakest (mid max / top v64)
  twins       for the subset records: every held-out strongest row with an exact twin in strongest-train, the
              distance between the two rows in the split's peak ranking (ties -> adjacent ranks)

  PYTHONPATH=src .venv/bin/python experiments/085-heldout-audit/store_values.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import audit  # noqa: E402

KINDS = ["attn", "mlp", "resid"]


def main():
    import pandas as pd
    keys = [k for k in (REPO / "experiments/062-h100-protocol-v1/targets_full.txt").read_text().split() if k]
    top = torch.load(REPO / "outputs" / "top_ctx.pt", map_location="cpu", weights_only=False)
    mid = torch.load(REPO / "outputs" / "mid_ctx.pt", map_location="cpu", weights_only=False)
    h = np.load(HERE / "out" / "corpus_hash.npy", mmap_mode="r")
    ti, tv = top["ctx_seq_idx"], top["ctx_seq_val"].float()
    mi, mv = mid["ctx_seq_idx"], mid["ctx_seq_val"].float()
    print("top store", tuple(ti.shape), "mid store", tuple(mi.shape), flush=True)
    rows = []
    for key in keys:
        l, k, i = key.split(".")
        c = int(l) * 3 + KINDS.index(k); i = int(i)
        ids, vals = ti[c, i].numpy().astype(np.int64), tv[c, i].numpy()
        ok = ids > 0
        order = np.argsort(-vals, kind="stable")
        ids, vals, ok = ids[order], vals[order], ok[order]
        v = vals[ok]
        hh = np.asarray(h)[ids[ok]]
        # position of the n-th unique content in the store ranking
        seen, first_unique = set(), []
        for r, x in enumerate(hh.tolist()):
            if x not in seen:
                seen.add(x); first_unique.append(r)
        mids, mvals = mi[c, i].numpy().astype(np.int64), mv[c, i].numpy()
        mok = mids > 0
        row = dict(key=key, layer=int(l), kind=k, band=audit.band(int(l)), top_filled=int(ok.sum()),
                   top_unique=len(seen), mid_filled=int(mok.sum()))
        for r in (1, 16, 32, 48, 64):
            row["v%d" % r] = float(v[r - 1]) if len(v) >= r else np.nan
        row["decay_v64_v1"] = row["v64"] / row["v1"] if len(v) >= 64 and row["v1"] > 0 else np.nan
        row["mid_max_over_v64"] = float(mvals[mok].max() / v[-1]) if mok.any() and len(v) else np.nan
        row["mid_med_over_v64"] = float(np.median(mvals[mok]) / v[-1]) if mok.any() and len(v) else np.nan
        row["mid_med_over_v1"] = float(np.median(mvals[mok]) / v[0]) if mok.any() and len(v) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "out" / "store_values.csv", index=False)
    print(df.describe().T.to_string())

    # ---- twins and the split (subset records)
    tw = []
    for f in audit.pick_subset(sorted(audit.CTX.glob("*.pt"))):
        key, rec = next(iter(torch.load(f, weights_only=False).items()))
        s = rec.get("strong")
        if s is None:
            continue
        pk = s["peak"].float()
        ranked = torch.argsort(pk, descending=True, stable=True).tolist()
        rank_of = {j: r for r, j in enumerate(ranked)}
        hs = audit.chash(s["pos"].long().numpy())
        tr = set(int(x) for x in s["train"])
        for j in s["held"]:
            j = int(j)
            twins = [t for t in tr if hs[t] == hs[j]]
            for t in twins:
                tw.append(dict(key=key, held=j, twin=t, rank_held=rank_of[j], rank_twin=rank_of[t],
                               dist=abs(rank_of[j] - rank_of[t]), same_peak=bool(pk[j] == pk[t])))
    tw = pd.DataFrame(tw)
    tw.to_csv(HERE / "out" / "twin_ranks_subset.csv", index=False)
    if len(tw):
        print("\ntwin pairs (held-out strongest row, strongest-train twin): %d" % len(tw))
        print("same peak: %.3f | rank distance: %s" % (tw.same_peak.mean(),
              tw.dist.value_counts().sort_index().head(8).to_dict()))


if __name__ == "__main__":
    main()
