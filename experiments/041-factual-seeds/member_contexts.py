"""What does each circuit member actually DO for this seed?

Auto-interp labels are generic topic tags ("chemistry terminology"),
which is why reading a circuit off labels teaches little. This instead
shows, for every member, the tokens it fires on INSIDE THE SEED'S OWN
positive windows -- i.e. the member's role in the contexts where the
seed is active. One forward pass per layer, all members of that layer
read at once.

  PYTHONPATH=. python member_contexts.py 8/2016 [MEMFILE=..._chem_...]
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
GEMMA = HERE.parent / "038-transcoder-compare-gemma"
sys.path.insert(0, str(GEMMA))
os.chdir(GEMMA)

import json
import ours_gtc as G

TOPK = int(os.environ.get("TOPK", 3))
CTX = int(os.environ.get("CTX", 7))
MEMFILE = os.environ.get("MEMFILE", "ours_gtc_chem_members.jsonl")

LABELS = {}
lf = HERE / "labels.jsonl"
if lf.exists():
    for line in open(lf):
        r = json.loads(line)
        LABELS[(r["layer"], r["feat"])] = r["label"]


def main():
    key = sys.argv[1]
    L0, S0 = (int(x) for x in key.split("/"))
    seeds = None
    for f in ("run_seeds.pt", "chem_seeds.pt", "factual_seeds.pt"):
        d = torch.load(HERE / f, weights_only=False)
        if key in d["seeds"]:
            seeds, toks = d["seeds"], d["tokens"]
            break
    assert seeds, "seed %s not found" % key
    pos = toks[seeds[key]["pos_windows"]]

    alphas = None
    for line in open(GEMMA / MEMFILE):
        r = json.loads(line)
        if (int(r["layer"]), int(r["latent"])) == (L0, S0) \
                and r["arm"] == "triamp400":
            alphas = {(int(l), int(f)): float(a)
                      for l, d in r["alphas"].items() for f, a in d.items()}
    assert alphas, "no triamp400 members for %s" % key

    by_layer = {}
    for (l, f), a in alphas.items():
        by_layer.setdefault(l, []).append((f, a))

    print("SEED %s | %s" % (key, seeds[key].get("label", "")))
    print("reading %d members inside the seed's own %d positive windows\n"
          % (len(alphas), len(pos)))

    rows = []
    for layer in sorted(by_layer):
        feats = [f for f, _ in by_layer[layer]]
        idx = torch.tensor(feats, dtype=torch.long, device=G.DEV)
        acts = []
        with torch.no_grad():
            for s0 in range(0, len(pos), G.BATCH):
                cap = {}
                hd = G.block(layer).pre_feedforward_layernorm \
                    .register_forward_hook(
                        lambda m, i, o: cap.__setitem__("f", G.features(layer, o)))
                G.model(pos[s0:s0 + G.BATCH].to(G.DEV))
                hd.remove()
                acts.append(cap["f"][:, :, idx].float().cpu())
        A = torch.cat(acts)                     # [W, T, nfeat]
        for j, (f, a) in enumerate(by_layer[layer]):
            flat = A[:, :, j].flatten()
            top = flat.topk(min(TOPK, flat.numel()))
            snips = []
            for v, p in zip(top.values.tolist(), top.indices.tolist()):
                if v <= 0:
                    continue
                w, t = divmod(p, A.shape[1])
                lo, hi = max(0, t - CTX), min(A.shape[1], t + CTX + 1)
                txt = "".join("[[%s]]" % G.tok.decode([int(pos[w, i])])
                              if i == t else G.tok.decode([int(pos[w, i])])
                              for i in range(lo, hi)).replace("\n", " ")
                snips.append(txt.strip())
            rows.append((a, layer, f, snips))

    for a, layer, f, snips in sorted(rows, reverse=True):
        print("a=%.2f  L%d/%d  %s" % (a, layer, f,
                                      LABELS.get((layer, f), "")[:64]))
        for s in snips:
            print("        ...%s..." % s)
        if not snips:
            print("        (silent in the seed's own windows)")
        print()


if __name__ == "__main__":
    main()
