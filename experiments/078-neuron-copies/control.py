"""078 calibration: what would a perfect single-neuron copy score, and what is chance?

  r ceiling  for N_CTRL random neurons per layer: a "latent" that is an exact sparse copy of the neuron, firing only
             on the neuron's top-q tokens with the neuron's value (q = the median firing rate of that layer's alive
             latents, from out/per_latent_L<L>.pt). Pearson r(copy, neuron) is the best r a true near-copy could get
             under the same sparsity; also r for a copy that is the neuron's whole positive part.
  cos chance max over neurons of |cos(random unit vector, centred W_out row)|, 4096 random directions.

  PYTHONPATH=src HF_HUB_OFFLINE=1 python experiments/078-neuron-copies/control.py   -> out/control.json
"""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "070-public-model-adapters"))
OUT = HERE / "out"
N_WIN = int(os.environ.get("N_WIN", 1600))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,1,2,3,4,5,6,7,8,9,11").split(",")]
N_CTRL = 512


def pearson(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp(min=1e-12)


def main():
    from transformers import GPT2LMHeadModel
    from common import load_tokenizer, wikitext_windows
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    toks = wikitext_windows(load_tokenizer("gpt2-small"), N_WIN, 64, skip=20000)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    g = torch.Generator().manual_seed(0)
    res = {}
    for L in LAYERS:
        pl = torch.load(OUT / ("per_latent_L%d.pt" % L))
        fires = pl["fires"].float()
        n_tok = toks.numel()
        q = float((fires[fires > 0] / n_tok).median())
        cap = {}
        h = model.transformer.h[L].mlp.c_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0]))
        hs = []
        idx = torch.randperm(3072, generator=g)[:N_CTRL].to(dev)
        with torch.no_grad():
            for s0 in range(0, toks.shape[0], 64):
                model(toks[s0:s0 + 64].to(dev))
                hs.append(cap["h"].reshape(-1, 3072)[:, idx].float())
        h.remove()
        H = torch.cat(hs)                                                # [n_tok, N_CTRL]
        thr = torch.quantile(H, 1 - q, dim=0)
        copy_q = torch.where(H >= thr, H, torch.zeros_like(H))
        copy_pos = H.clamp(min=0)
        r_q, r_pos = pearson(copy_q, H), pearson(copy_pos, H)
        W = model.transformer.h[L].mlp.c_proj.weight.detach().float()
        W = W - W.mean(1, keepdim=True); W = W / W.norm(dim=1, keepdim=True)
        rnd = torch.randn(4096, 768, generator=g).to(dev); rnd = rnd / rnd.norm(dim=1, keepdim=True)
        cos_chance = (rnd @ W.T).abs().max(1).values
        lay = json.loads((OUT / ("layer_%d.json" % L)).read_text())
        res[L] = dict(q_median_fire_rate=q, r_ceiling_sparse_median=float(r_q.median()),
                      r_ceiling_sparse_p10=float(r_q.quantile(0.1)), r_ceiling_positive_part_median=float(r_pos.median()),
                      cos_chance_median=float(cos_chance.median()), cos_chance_p99=float(cos_chance.quantile(0.99)),
                      observed_r_median=lay["r_median"], observed_cos_median=lay["cos_median"],
                      observed_cos_ge_0p7=lay["cos_ge_0p7"], observed_r_ge_0p8=lay["r_ge_0p8"])
        print("L%d q %.4f  r ceiling (sparse copy) med %.2f p10 %.2f | observed r med %.2f || cos chance med %.2f p99 %.2f"
              " | observed cos med %.2f" % (L, q, res[L]["r_ceiling_sparse_median"], res[L]["r_ceiling_sparse_p10"],
                                           lay["r_median"], res[L]["cos_chance_median"], res[L]["cos_chance_p99"],
                                           lay["cos_median"]), flush=True)
    (OUT / "control.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
