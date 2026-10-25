"""078 — are GPT-2 small MLP-site SAE latents near-copies of single MLP neurons? (prerequisite for neuron-basis WCM)

Cammarata 2026 ("Neural Networks Are Already Clean") reports that on vision layers 88% of SAE activity is near-copies
of single neurons. This measures the same thing for GPT-2 small's mlp-out dictionaries (jbloom OAI v5 32k, k = 32),
two ways, per layer:

  weights      max over neurons i of cos(decoder row j, centred W_out row i). The SAE reads the layer-normed MLP
               output (x - mean) / std, so W_out rows are centred over d_model before the cosine.
  activations  max over neurons i of Pearson r(latent code a_j(t), post-GELU neuron h_i(t)) over N_TOK wikitext tokens
               (no BOS, 64-token windows, as the dictionaries were trained). Latents that never fire are counted
               separately. Also r for the latent's best-matching-by-weight neuron.

Output: out/layer_<L>.json (distribution summaries) and out/per_latent_L<L>.pt (per-latent best neuron, r, cos).

  PYTHONPATH=src HF_HUB_OFFLINE=1 python experiments/078-neuron-copies/neuron_copies.py    env: N_WIN (1600)  LAYERS
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "070-public-model-adapters"))
OUT = HERE / "out"
N_WIN = int(os.environ.get("N_WIN", 1600))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,1,2,3,4,5,6,7,8,9,11").split(",")]
BATCH = 32


def main():
    from transformers import GPT2LMHeadModel
    from common import load_tokenizer, wikitext_windows
    from sae.public_bank import PublicSAEBank

    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer("gpt2-small")
    toks = wikitext_windows(tok, N_WIN, 64, skip=20000)                 # skip past 053/070's smoke windows
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    bank = PublicSAEBank(device=dev, family="gpt2-small", preload=False)
    print("078: %d windows (%d tokens), layers %s" % (toks.shape[0], toks.numel(), LAYERS), flush=True)

    for L in LAYERS:
        t0 = time.time()
        sae = bank.saes["mlp"][L]
        Wd = sae.decoder.weight.detach().float()                         # [d_model, d_sae] or [d_sae, d_model]
        if Wd.shape[0] == model.config.n_embd:
            Wd = Wd.T
        Wd = Wd / Wd.norm(dim=1, keepdim=True).clamp(min=1e-12)          # [d_sae, 768]
        Wout = model.transformer.h[L].mlp.c_proj.weight.detach().float()  # GPT-2 Conv1D: [3072, 768]
        Wout = Wout - Wout.mean(dim=1, keepdim=True)
        Wout = Wout / Wout.norm(dim=1, keepdim=True).clamp(min=1e-12)
        cos = Wd @ Wout.T                                                # [d_sae, 3072]
        cmax, cidx = cos.abs().max(dim=1)
        dsae, nn_ = int(Wd.shape[0]), int(Wout.shape[0])
        del cos

        # streaming moments for corr(a_j, h_i): sums of a, a^2, h, h^2, a h^T
        Sa = torch.zeros(dsae, device=dev, dtype=torch.float64); Saa = torch.zeros_like(Sa)
        Sh = torch.zeros(nn_, device=dev, dtype=torch.float64); Shh = torch.zeros_like(Sh)
        Sah = torch.zeros(dsae, nn_, device=dev, dtype=torch.float32)
        fires = torch.zeros(dsae, device=dev, dtype=torch.long)
        cap = {}

        def pre_hook(mod, args):
            cap["h"] = args[0].detach()

        def post_hook(mod, args, out):
            cap["o"] = out.detach()

        hs = [model.transformer.h[L].mlp.c_proj.register_forward_pre_hook(pre_hook),
              model.transformer.h[L].mlp.c_proj.register_forward_hook(post_hook)]
        n = 0
        with torch.no_grad():
            for s0 in range(0, toks.shape[0], BATCH):
                model(toks[s0:s0 + BATCH].to(dev))
                h = cap["h"].reshape(-1, nn_).float()
                x = bank.site_input(cap["o"].float(), "mlp", L)
                ta, ti = bank.encode(x, "mlp", L)
                a = torch.zeros(h.shape[0], dsae, device=dev, dtype=torch.float32)
                a.scatter_(1, ti.reshape(-1, ti.shape[-1]), ta.reshape(-1, ta.shape[-1]).float())
                Sa += a.sum(0).double(); Saa += (a * a).sum(0).double()
                Sh += h.sum(0).double(); Shh += (h * h).sum(0).double()
                Sah += a.T @ h
                fires += (a > 0).sum(0)
                n += h.shape[0]
        for hk in hs:
            hk.remove()
        ma, mh = Sa / n, Sh / n
        va = (Saa / n - ma ** 2).clamp(min=0); vh = (Shh / n - mh ** 2).clamp(min=0)
        alive = fires > 0
        R = torch.zeros(dsae, device=dev); Ridx = torch.zeros(dsae, device=dev, dtype=torch.long)
        Rw = torch.zeros(dsae, device=dev)
        for c0 in range(0, dsae, 4096):
            cov = Sah[c0:c0 + 4096].double() / n - ma[c0:c0 + 4096, None] * mh[None, :]
            den = (va[c0:c0 + 4096, None] * vh[None, :]).sqrt().clamp(min=1e-12)
            r = (cov / den).float()
            R[c0:c0 + 4096], Ridx[c0:c0 + 4096] = r.max(dim=1)
            Rw[c0:c0 + 4096] = r.gather(1, cidx[c0:c0 + 4096, None].to(dev)).squeeze(1)
        Ra = R[alive].cpu()
        ca = cmax[alive].cpu()

        def share(v, t):
            return float((v >= t).float().mean()) if v.numel() else None

        summ = dict(layer=L, d_sae=dsae, n_neurons=nn_, n_tokens=n, n_alive=int(alive.sum()),
                    frac_alive=float(alive.float().mean()),
                    r_median=float(Ra.median()), r_ge_0p9=share(Ra, 0.9), r_ge_0p8=share(Ra, 0.8),
                    r_ge_0p7=share(Ra, 0.7), r_ge_0p5=share(Ra, 0.5),
                    cos_median=float(ca.median()), cos_ge_0p9=share(ca, 0.9), cos_ge_0p7=share(ca, 0.7),
                    cos_ge_0p5=share(ca, 0.5),
                    weight_and_act_agree=float((Ridx[alive] == cidx[alive].to(dev)).float().mean()),
                    # activity-weighted: share of total latent activation mass carried by near-copies (r >= 0.8)
                    mass_share_r_ge_0p8=float((Sa[alive][R[alive] >= 0.8].sum() / Sa[alive].sum())),
                    neurons_used_as_best=int(torch.unique(Ridx[alive]).numel()),
                    secs=round(time.time() - t0, 1))
        (OUT / ("layer_%d.json" % L)).write_text(json.dumps(summ, indent=1))
        torch.save(dict(r=R.cpu(), r_idx=Ridx.cpu(), cos=cmax.cpu(), cos_idx=cidx.cpu(), r_at_cos=Rw.cpu(),
                        fires=fires.cpu(), mean_act=ma.float().cpu()), OUT / ("per_latent_L%d.pt" % L))
        print("[%s] L%d alive %d  r med %.2f  r>=0.8 %.1f%%  cos med %.2f  cos>=0.7 %.1f%%  mass(r>=0.8) %.1f%%  %.0fs"
              % (time.strftime("%H:%M"), L, summ["n_alive"], summ["r_median"], 100 * summ["r_ge_0p8"],
                 summ["cos_median"], 100 * summ["cos_ge_0p7"], 100 * summ["mass_share_r_ge_0p8"], summ["secs"]),
              flush=True)
        del Sah
        bank.release("mlp", L) if hasattr(bank, "release") else None
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
