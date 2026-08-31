"""CHEAP BATCH C: first-pass interchangeability (holiday question 4).

Our circuit and coact_amp's circuit overlap only 32-47% on Gemma, yet
both reconstruct the seed. Are the non-shared members SUBSTITUTES —
different latent indices carrying nearly the same decoder direction — or
genuinely different computation routes?

For every OUR-member absent from the coact set, find the best
W_dec-cosine match among coact members AT THE SAME LAYER (a substitute
must write the same direction from the same depth to be interchangeable
in place), and vice versa. High-cosine coverage says "the circuit is an
equivalence class over redundancy clusters"; low coverage says the two
sets genuinely use different machinery, which is the stronger
non-identifiability reading.

Baseline for calibration: the best cosine between OUR members and
RANDOM live latents of the same layer (what 'no relationship' looks
like in a 16k-dictionary — near-orthogonal, so ~0.0-0.1).

  python substitutes.py     (CPU; loads Gemma SAE decoders per layer)
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
G = HERE.parent / "037-gemmascope"
CACHE = Path.home() / "gemmascope"
TIER_L0 = {0: 30, 1: 50, 2: 95, 3: 95, 4: 85, 5: 114}   # tier 2, from logs


def wdec(layer):
    z = np.load(CACHE / ("layer_%d_w16k_l0_%d.npz" % (layer, TIER_L0[layer])))
    W = torch.tensor(z["W_dec"])                     # [16384, 2304]
    return F.normalize(W, dim=-1)


def main():
    tri, co = {}, {}
    for line in open(G / "ours_gemma_members_t2.jsonl"):
        r = json.loads(line)
        d = {int(k): sorted(int(i) for i in v)
             for k, v in r["members"].items()}
        (tri if r.get("arm") == "triamp400" else co)[
            (r["layer"], r["latent"])] = d

    Wn = {L: wdec(L) for L in TIER_L0}
    rng = random.Random(7)
    print("seed         overlap  ||  ours-only: n  sub>0.7  sub>0.5  "
          "med-best  ||  rand-ctrl med-best")
    agg = {"n": 0, "s7": 0, "s5": 0}
    for k in sorted(tri):
        if k not in co:
            continue
        ours = {(l, i) for l, v in tri[k].items() for i in v}
        theirs = {(l, i) for l, v in co[k].items() for i in v}
        inter = len(ours & theirs)
        only = sorted(ours - theirs)
        bests, rand_bests = [], []
        s7 = s5 = 0
        for l, i in only:
            cand = [j for (ll, j) in theirs if ll == l and j != i]
            if not cand:
                bests.append(0.0)
                continue
            cos = float((Wn[l][i] @ Wn[l][cand].T).max())
            bests.append(cos)
            s7 += cos > 0.7
            s5 += cos > 0.5
            rand = rng.sample(range(Wn[l].shape[0]), len(cand))
            rand_bests.append(float((Wn[l][i] @ Wn[l][rand].T).max()))
        med = sorted(bests)[len(bests) // 2] if bests else float("nan")
        rmed = (sorted(rand_bests)[len(rand_bests) // 2]
                if rand_bests else float("nan"))
        agg["n"] += len(only); agg["s7"] += s7; agg["s5"] += s5
        print("L%-2d %-8d %3d/%-3d  ||  %10d  %7d  %7d  %8.3f  ||  %8.3f"
              % (k[0], k[1], inter, len(ours), len(only), s7, s5, med, rmed))
    print("\nTOTAL ours-only members: %d | with a same-layer coact "
          "substitute at cos>0.7: %d (%.0f%%) | >0.5: %d (%.0f%%)"
          % (agg["n"], agg["s7"], 100 * agg["s7"] / max(agg["n"], 1),
             agg["s5"], 100 * agg["s5"] / max(agg["n"], 1)))


if __name__ == "__main__":
    main()
