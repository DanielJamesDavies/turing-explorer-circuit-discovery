"""Latent pilot: where does each target rank among its site's latents, CLEAN, on its own contexts? (Daniel 2026-11-02:
"where do they rank in their topctx and midctx topks?")

For every pilot target and each context pool of its record (062 out_full/out/ctx: `strong` = the 64 top contexts,
`mid` = the mid-band pool, None for thin targets), one clean forward; at the target's site and each context's anchor
(rec[pool]["arg"]) the full pre-activation vector (encoder, as attn_heads.py's margin check) gives the target's rank
(1 = largest), whether it is inside the site's Top-K (K = the bank's k), and its margin over the first excluded latent,
(pre_target - pre_(K+1)) / pre_target. Train and held-out contexts are reported together and separately (the scorer
reads the held-out strongest).

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src .venv/bin/python \
      experiments/107-true-baselines/latent_pilot/clean_ranks.py
"""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
D062 = HERE.parents[1] / "062-h100-protocol-v1"
os.environ.setdefault("OUT", str(D062 / "out_full" / "out"))
os.environ.setdefault("CTX_DIR", str(D062 / "out_full" / "out" / "ctx"))
os.environ.setdefault("MODE", "main")
sys.path.insert(0, str(D062))
import driver  # noqa: E402


def ranks_for(R, key, pool):
    G, V = R.G, R.V
    rec = R.contexts(key)
    P = rec.get(pool)
    if P is None:
        return None
    layer, kind, sl = key.split(".")[0], key.split(".")[1], int(key.split(".")[2])
    layer = int(layer)
    sae = G["bank"].saes[kind][layer]
    W = sae.encoder.weight.detach().float()
    b = sae._get_bias_eff().detach().float()
    K = int(G["K"])
    pt = P["pos"].to(G["device"])
    pa = P["arg"].to(G["device"]).long()
    cap = {}

    class Cap(V.AmpInjectPatcher):
        def transform(self, layer_idx, kind_, x):
            if (layer_idx, kind_) == self.seed_site:
                rr = torch.arange(x.shape[0], device=x.device)
                cap["x"] = x[rr, pa.clamp(0, x.shape[1] - 1)].float()
            return super().transform(layer_idx, kind_, x)

    p = Cap({}, (layer, kind), W[sl], b[sl], sl)
    with torch.no_grad():
        G["inference"].forward(pt, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
        pre = cap["x"] @ W.T + b                                         # [n, d_sae]
        tgt = pre[:, sl]
        rank = (pre > tgt.unsqueeze(1)).sum(1) + 1                       # 1 = the largest
        kth1 = pre.topk(K + 1, dim=-1).values[:, K]                      # the first latent left out of Top-K
        margin = (tgt - kth1) / tgt.clamp(min=1e-6)
    tr = set(P.get("train") or [])
    rows = []
    for i in range(int(pt.shape[0])):
        rows.append(dict(i=i, split="train" if i in tr else "held", rank=int(rank[i]), pre=float(tgt[i]),
                         in_topk=bool(rank[i] <= K), margin=float(margin[i])))
    return dict(K=K, rows=rows)


def summ(rows, K):
    if not rows:
        return "-"
    r = sorted(x["rank"] for x in rows)
    med = r[len(r) // 2]
    return "median rank %d (p10 %d, p90 %d) | in Top-K %d/%d | rank > K/2 %d/%d | median pre %.2f" % (
        med, r[len(r) // 10], r[(9 * len(r)) // 10], sum(x["in_topk"] for x in rows), len(rows),
        sum(x["rank"] > K // 2 for x in rows), len(rows), sorted(x["pre"] for x in rows)[len(rows) // 2])


def main():
    keys = [k for k in (HERE / "targets.txt").read_text().split() if k]
    R = driver.Runner()
    out = {}
    for key in keys:
        out[key] = {}
        for pool in ("strong", "mid"):
            res = ranks_for(R, key, pool)
            out[key][pool] = res
            if res is None:
                print("%-15s %-6s (no pool)" % (key, pool), flush=True)
                continue
            for sp in ("all", "train", "held"):
                rows = [x for x in res["rows"] if sp == "all" or x["split"] == sp]
                print("%-15s %-6s %-5s K %d | %s" % (key, pool, sp, res["K"], summ(rows, res["K"])), flush=True)
    (HERE / "clean_ranks.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
