"""SFC-STYLE ATTRIBUTION (external method E2) on the Llama transcoder
arena: rank upstream features by attribution patching, rooted at the
seed, on the EXACT model.

The method family is Sparse Feature Circuits (Marks et al. 2025):
score each feature by activation x gradient of the metric, here the
seed's pre-activation at its anchor -- the first-order effect of
ablating that feature. Unlike circuit-tracer this runs on the exact
forward pass (no frozen-attention surrogate, no error nodes), and it is
seed-rooted by construction, so it needs none of the budget/survival
machinery. It is the second external method of the comparison plan
(the home-bank arena has its own SFC replication in tab:matrix).

Mechanics: for each upstream layer, the read hook computes the natural
code c and injects a zero leaf delta so that the model output depends
on it: write hook adds (delta @ W_dec). d(seed_pre_anchor)/d(delta_i)
is then the exact gradient wrt feature i's activation at every
position; attribution = c * grad summed over positions, mean over the
48 TRAIN windows. Rank by |attribution|; export the full ranking plus
the size-matched head, same schema as the direct-edge export, so the
same scoring loaders apply.

  PYTHONPATH=. python sfc_gtc.py
"""
import json
import os
from pathlib import Path

import torch

import ours_llama as O

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUT", str(HERE / "sfc_nodes.jsonl")))
BATCH = int(os.environ.get("SFC_BATCH", 8))


def main():
    scan = torch.load(HERE / O.SCAN_NAME if hasattr(O, "SCAN_NAME") else
                      HERE / "scan_llama.pt", weights_only=False)
    toks, seeds = scan["tokens"], scan["seeds"]
    sizes = {}
    rows_path = HERE / (O.ROWS_NAME if hasattr(O, "ROWS_NAME") else "ours_llama_rows.jsonl")
    if rows_path.exists():
        for line in rows_path.open():
            r = json.loads(line)
            if r.get("arm") == "triamp400":
                sizes[(r["layer"], r["latent"])] = r["n"]
    done = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            done.add((r["layer"], r["latent"]))

    fh = OUT.open("a")
    for key in sorted(seeds):
        S = seeds[key]
        L, sl = S["layer"], S["latent"]
        if (L, sl) in done:
            continue
        UP = list(range(L))
        pos_tr = toks[S["pos_windows"]][:O.N_TRAIN]

        # anchors on the natural model
        cap = {}
        hd = O.block(L).mlp.register_forward_hook(
            lambda m, i, o: cap.__setitem__("f", O.features(L, i[0])))
        with torch.no_grad():
            O.model(pos_tr.to(O.DEV))
        hd.remove()
        nat = cap["f"][..., sl]
        nat[:, 0] = -float("inf")
        anchors = nat.argmax(dim=1)

        attr = {l: torch.zeros(O.D_TC, device=O.DEV) for l in UP}
        for s0 in range(0, pos_tr.shape[0], BATCH):
            tk = pos_tr[s0:s0 + BATCH]
            an = anchors[s0:s0 + BATCH]
            deltas, codes, handles = {}, {}, []

            def rw_hook(layer):
                # Llama transcoders read the MLP INPUT and write the MLP
                # OUTPUT: one hook does both (as the harness Runner does).
                def hook(mod, inp, out):
                    c = O.features(layer, inp[0]).detach()
                    d = torch.zeros_like(c, requires_grad=True)
                    codes[layer], deltas[layer] = c, d
                    return out + d.to(out.dtype) @ O.tc(layer)["W_dec"]
                return hook

            for l in UP:
                handles.append(O.block(l).mlp.register_forward_hook(rw_hook(l)))
            seedcap = {}
            handles.append(O.block(L).mlp.register_forward_hook(
                lambda m, i, o: seedcap.__setitem__(
                    "p", O.pre_acts(L, i[0])[..., sl])))
            O.model(tk.to(O.DEV))
            for h in handles:
                h.remove()
            B = tk.shape[0]
            rows = torch.arange(B, device=O.DEV)
            metric = seedcap["p"][rows, an.to(O.DEV)].float().sum()
            grads = torch.autograd.grad(metric, [deltas[l] for l in UP])
            for l, g in zip(UP, grads):
                attr[l] += (codes[l].float() * g.float()).sum(dim=(0, 1))

        ranked = sorted(((l, int(i), float(attr[l][i]))
                         for l in UP
                         for i in attr[l].abs().argsort(descending=True)[:8000].tolist()
                         if float(attr[l][i]) != 0.0),
                        key=lambda t: -abs(t[2]))[:20000]
        n_ref = sizes.get((L, sl))
        rec = {"layer": L, "latent": sl, "n_ranked": len(ranked), "n_ref": n_ref,
               "top_matched": [[l, f] for l, f, _ in ranked[:n_ref]] if n_ref else None,
               "ranking": [[l, f, round(w, 6)] for l, f, w in ranked]}
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print("[L%d %d] SFC ranking %d features | top-%s exported | top5 %s"
              % (L, sl, len(ranked), n_ref,
                 [(l, f, round(w, 3)) for l, f, w in ranked[:5]]), flush=True)
    fh.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
