"""078 (activation test, replaces Pearson r): firing-set overlap between each latent and its best neuron.

Pearson r is uninformative here (control.py: an exact sparse copy of a neuron scores only r ~ 0.2 at the latents'
firing rate). Instead, for each neuron i its TAIL set = tokens where h_i is in its top q (q in QS), and for each latent j
its FIRING set = tokens where a_j > 0. Per latent, the best neuron by Jaccard |F_j & T_i| / |F_j | T_i|, and for it:
  precision = |F_j & T_i| / |F_j|   (when the latent fires, the neuron is in its tail)
  recall    = |F_j & T_i| / |T_i|   (when the neuron is in its tail, the latent fires)
A near-copy has both high. Reported per layer: share of alive latents with precision >= 0.8 AND recall >= 0.5 at
some q, and the activation-mass-weighted share.

  PYTHONPATH=src HF_HUB_OFFLINE=1 python experiments/078-neuron-copies/copies_overlap.py   -> out/overlap_L<L>.json
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
QS = (0.01, 0.003, 0.001, 0.0003, 0.0001)     # tail sizes; each latent is scored at the q nearest its own rate
BATCH = 32


def main():
    from transformers import GPT2LMHeadModel
    from common import load_tokenizer, wikitext_windows
    from sae.public_bank import PublicSAEBank
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    toks = wikitext_windows(load_tokenizer("gpt2-small"), N_WIN, 64, skip=20000)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    bank = PublicSAEBank(device=dev, family="gpt2-small", preload=False)
    allres = {}
    for L in LAYERS:
        t0 = time.time()
        cap = {}
        hs = [model.transformer.h[L].mlp.c_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0])),
              model.transformer.h[L].mlp.c_proj.register_forward_hook(lambda m, a, o: cap.__setitem__("o", o))]
        H, TI, TA = [], [], []
        with torch.no_grad():
            for s0 in range(0, toks.shape[0], BATCH):
                model(toks[s0:s0 + BATCH].to(dev))
                H.append(cap["h"].reshape(-1, 3072).half())
                ta, ti = bank.encode(bank.site_input(cap["o"].float(), "mlp", L), "mlp", L)
                TA.append(ta.reshape(-1, ta.shape[-1]).float()); TI.append(ti.reshape(-1, ti.shape[-1]))
        for h in hs:
            h.remove()
        H = torch.cat(H)                                              # [N, 3072] fp16
        TA, TI = torch.cat(TA), torch.cat(TI)                         # [N, k]
        N, dsae = H.shape[0], int(bank.d_sae)
        fire_n = torch.zeros(dsae, device=dev).index_add_(0, TI[TA > 0], torch.ones(int((TA > 0).sum()), device=dev))
        mass = torch.zeros(dsae, device=dev).index_add_(0, TI[TA > 0], TA[TA > 0])
        alive = fire_n > 0
        best = {}
        for q in QS:
            thr = torch.quantile(H.float()[torch.randperm(N, device=dev)[:200000]], 1 - q, dim=0)
            C = torch.zeros(dsae, 3072, device=dev)
            tail_n = torch.zeros(3072, device=dev)
            for c0 in range(0, N, 8192):
                T = (H[c0:c0 + 8192].float() >= thr).float()          # [n, 3072]
                tail_n += T.sum(0)
                F_ = torch.zeros(T.shape[0], dsae, device=dev)
                F_.scatter_(1, TI[c0:c0 + 8192], (TA[c0:c0 + 8192] > 0).float())
                C += F_.T @ T
            jac = C / (fire_n[:, None] + tail_n[None, :] - C).clamp(min=1)
            jb, ib = jac.max(1)
            cb = C.gather(1, ib[:, None]).squeeze(1)
            prec = cb / fire_n.clamp(min=1)
            rec = cb / tail_n[ib].clamp(min=1)
            best[q] = (jb, prec, rec, ib)
            del C, jac
        copy = torch.zeros(dsae, dtype=torch.bool, device=dev)
        out = dict(layer=L, n_tokens=N, n_alive=int(alive.sum()))
        for q, (jb, prec, rec, ib) in best.items():
            c_q = (prec >= 0.8) & (rec >= 0.5)
            copy |= c_q
            out["q%g" % q] = dict(jaccard_median=float(jb[alive].median()), precision_median=float(prec[alive].median()),
                                  recall_median=float(rec[alive].median()),
                                  share_copy=float(c_q[alive].float().mean()),
                                  share_prec_ge_0p8=float((prec[alive] >= 0.8).float().mean()),
                                  mass_share_copy=float(mass[alive & c_q].sum() / mass[alive].sum()))
        out["share_copy_any_q"] = float(copy[alive].float().mean())
        out["mass_share_copy_any_q"] = float(mass[alive & copy].sum() / mass[alive].sum())
        # MATCHED: each latent scored at the tail size nearest (log scale) its own firing rate, so recall is not
        # bounded by a size mismatch. copy = precision >= 0.5 and recall >= 0.5 (Jaccard >= 1/3).
        rate = fire_n / N
        qs = torch.tensor(QS, device=dev)
        qi = (rate.clamp(min=1e-7).log()[:, None] - qs.log()[None, :]).abs().argmin(1)
        J = torch.stack([best[q][0] for q in QS], 1).gather(1, qi[:, None]).squeeze(1)
        P = torch.stack([best[q][1] for q in QS], 1).gather(1, qi[:, None]).squeeze(1)
        Rr = torch.stack([best[q][2] for q in QS], 1).gather(1, qi[:, None]).squeeze(1)
        cm = (P >= 0.5) & (Rr >= 0.5)
        out["matched"] = dict(jaccard_median=float(J[alive].median()), jaccard_p90=float(J[alive].quantile(0.9)),
                              precision_median=float(P[alive].median()), recall_median=float(Rr[alive].median()),
                              share_copy=float(cm[alive].float().mean()),
                              share_jac_ge_0p5=float((J[alive] >= 0.5).float().mean()),
                              mass_share_copy=float(mass[alive & cm].sum() / mass[alive].sum()),
                              rate_median=float(rate[alive].median()))
        torch.save(dict(J=J.cpu(), P=P.cpu(), R=Rr.cpu(), rate=rate.cpu(), mass=mass.cpu()), OUT / ("overlap_L%d.pt" % L))
        out["secs"] = round(time.time() - t0, 1)
        (OUT / ("overlap_L%d.json" % L)).write_text(json.dumps(out, indent=1))
        allres[L] = out
        m = out["matched"]
        print("[%s] L%d MATCHED: copy %.1f%% of latents (%.1f%% of mass), jac med %.2f p90 %.2f, prec med %.2f, "
              "rec med %.2f, rate med %.5f  %.0fs" % (time.strftime("%H:%M"), L, 100 * m["share_copy"],
                                                     100 * m["mass_share_copy"], m["jaccard_median"], m["jaccard_p90"],
                                                     m["precision_median"], m["recall_median"], m["rate_median"],
                                                     out["secs"]), flush=True)
        del H, TA, TI
        torch.cuda.empty_cache()
    (OUT / "overlap_all.json").write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
