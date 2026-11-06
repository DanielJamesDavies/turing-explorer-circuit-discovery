"""088 SAE PROPERTIES, step 1 (GPU): per-target SAE geometry, store statistics and a corpus Top-K cut.

For every target latent with a held-out-strongest eval row in the 062 full run (main/eval.shard*.jsonl, held ==
"strong"), at its own site (one SAE loaded at a time):
  encoder   enc_norm = |W_enc[i]|, b_enc = encoder.bias[i], b_eff = b_enc - W_enc[i] . b_dec (the bias the engine's
            pre-activation uses), b_eff_z = (b_eff) / std over corpus tokens of the target's pre-activation
  decoder   dec_norm = |W_dec[:, i]|
  enc_dec_cos  cosine of W_enc[i] and W_dec[:, i]
  rivals    max cosine to any OTHER latent's encoder row (enc_nn) / decoder column (dec_nn) at the site, and the counts
            with cosine > 0.3 / > 0.5 / > 0.7 (enc_n03, enc_n05, enc_n07, dec_n03, ...)
  site-relative percentiles of enc_norm / dec_norm / b_eff among all 40,960 latents of the site (*_pct)
  stores    latent_stats.pt: freq = active_count / (sum of the site's active_count / 128) (fraction of tokens in which
            the latent is in the Top-K, assuming every token has 128 active latents), mean_act = mean value when active
  corpus    N_SEQ windows of 64 tokens (shard SHARD_IDX, position 0 dropped): per token the 128th largest post-ReLU
            pre-activation at the site (the Top-K cut); site medians cut_med / cut_p10 / cut_p90; per target the
            fraction of sampled tokens it is in the Top-K (freq_sample), its pre-activation mean / std / max.

  wsl: cd <repo> && PYTHONPATH=src .venv/bin/python experiments/088-sae-properties/compute.py
  env: N_SEQ (512)  SHARD_IDX (2000)  OUT (experiments/088-sae-properties/out)
"""
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = Path(os.environ.get("OUT", str(HERE / "out")))
N_SEQ = int(os.environ.get("N_SEQ", "512"))
SHARD_IDX = int(os.environ.get("SHARD_IDX", "2000"))
EVAL_GLOB = str(REPO / "experiments/062-h100-protocol-v1/out_full/out/main/eval.shard*.jsonl")
KINDS = ["attn", "mlp", "resid"]
K = 128
dev = torch.device("cuda")


def log(s):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), s), flush=True)


def targets():
    keys = set()
    for p in glob.glob(EVAL_GLOB):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("held") == "strong" and "error" not in r:
                keys.add(r["seed"])
    by_site = {}
    for k in keys:
        l, kd, i = k.split(".")
        by_site.setdefault((int(l), kd), []).append(int(i))
    return by_site


def corpus_acts():
    """[N_SEQ, L, K, T-1, d] bf16 on CPU: the site activations for N_SEQ full-length windows."""
    from config import config
    from model.inference import Inference
    from data.loader import DataLoader
    loader = DataLoader(device=torch.device("cpu"))
    inf = Inference(device=dev)
    shard = np.load(os.path.join(config.data.dataset_path, loader.shard_files[SHARD_IDX]), mmap_mode="r")
    idx = loader._shard_indices[SHARD_IDX]
    seqs = [shard[s:e][:64].astype(np.int64) for s, e in idx if e - s >= 64][:N_SEQ]
    toks = torch.tensor(np.stack(seqs))
    log("corpus: %d windows from shard %d" % (len(seqs), SHARD_IDX))
    out = []
    for b in range(0, len(toks), 128):
        _, _, acts = inf.forward(toks[b:b + 128].to(dev), num_gen=1, tokenize_final=False)
        out.append(acts[:, :, :, 0, 1:, :].to(torch.bfloat16).cpu())   # [B, L, K, T-1, d], drop position 0
        del acts
    del inf
    torch.cuda.empty_cache()
    return torch.cat(out)


def pct_of(vals_all, vals_t):
    s = torch.sort(vals_all).values
    return (torch.searchsorted(s, vals_t.contiguous()).float() / len(s)).cpu().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_site = targets()
    log("targets: %d over %d sites" % (sum(len(v) for v in by_site.values()), len(by_site)))
    ls = torch.load(REPO / "outputs/latent_stats.pt", map_location="cpu", weights_only=False)
    acts = corpus_acts()
    log("corpus acts %s" % (tuple(acts.shape),))
    rows, site_rows = [], []
    for layer in range(12):
        for ki, kind in enumerate(KINDS):
            idx = sorted(by_site.get((layer, kind), []))
            if not idx:
                continue
            t0 = time.time()
            sd = torch.load(REPO / ("models/TuringLLM/SAE/sae-%s/sae_%s_layer_%d.pth" % (kind, kind, layer)),
                            map_location="cpu", weights_only=True)
            We = sd["encoder.weight"].to(dev)                  # [D, d]
            be = sd["encoder.bias"].to(dev)                    # [D]
            Wd = sd["decoder.weight"].to(dev).T.contiguous()   # [D, d] (columns of W_dec)
            bd = sd["decoder_bias"].to(dev)
            beff = be - We @ bd
            en, dn = We.norm(dim=1), Wd.norm(dim=1)
            ti = torch.tensor(idx, device=dev)
            Ehat, Dhat = We / en[:, None], Wd / dn[:, None]
            ce = Ehat[ti] @ Ehat.T                             # [n_t, D]
            cd = Dhat[ti] @ Dhat.T
            ar = torch.arange(len(idx), device=dev)
            ce[ar, ti] = -2.0; cd[ar, ti] = -2.0
            enc_dec_cos = (Ehat[ti] * Dhat[ti]).sum(1)
            # corpus: Top-K cut per token and the targets' pre-activations
            X = acts[:, layer, ki].reshape(-1, acts.shape[-1])
            cuts, tpre, tin = [], [], []
            for c in range(0, X.shape[0], 4096):
                x = X[c:c + 4096].to(dev).float()
                pre = x @ We.T + beff
                kth = torch.relu(pre).topk(K, dim=1).values[:, -1]
                cuts.append(kth)
                tp = pre[:, ti]
                tpre.append(tp)
                tin.append((torch.relu(tp) >= kth[:, None]) & (tp > 0))
                del pre
            cuts, tpre, tin = torch.cat(cuts), torch.cat(tpre), torch.cat(tin)
            comp = layer * 3 + ki
            ac = ls["active_count"][comp].double()
            ntok = ac.sum() / K
            site_rows.append(dict(layer=layer, kind=kind, cut_med=float(cuts.median()),
                                  cut_p10=float(cuts.quantile(0.1)), cut_p90=float(cuts.quantile(0.9)),
                                  enc_norm_med=float(en.median()), dec_norm_med=float(dn.median()),
                                  b_eff_med=float(beff.median()), n_tok_sample=int(cuts.numel()),
                                  n_tok_stats=float(ntok)))
            d = dict(
                key=["%d.%s.%d" % (layer, kind, i) for i in idx], layer=layer, kind=kind, latent=idx,
                enc_norm=en[ti].cpu().numpy(), dec_norm=dn[ti].cpu().numpy(),
                b_enc=be[ti].cpu().numpy(), b_eff=beff[ti].cpu().numpy(),
                enc_dec_cos=enc_dec_cos.cpu().numpy(),
                enc_nn=ce.max(1).values.cpu().numpy(), dec_nn=cd.max(1).values.cpu().numpy(),
                enc_n03=(ce > 0.3).sum(1).cpu().numpy(), dec_n03=(cd > 0.3).sum(1).cpu().numpy(),
                enc_n05=(ce > 0.5).sum(1).cpu().numpy(), enc_n07=(ce > 0.7).sum(1).cpu().numpy(),
                dec_n05=(cd > 0.5).sum(1).cpu().numpy(), dec_n07=(cd > 0.7).sum(1).cpu().numpy(),
                enc_norm_pct=pct_of(en, en[ti]), dec_norm_pct=pct_of(dn, dn[ti]), b_eff_pct=pct_of(beff, beff[ti]),
                freq=(ac[idx] / ntok).numpy(), mean_act=ls["mean"][comp][idx].numpy(),
                std_act=(ls["m2"][comp][idx] / (ac[idx] - 1).clamp(min=1)).sqrt().numpy(),
                freq_sample=tin.float().mean(0).cpu().numpy(),
                pre_mean=tpre.mean(0).cpu().numpy(), pre_std=tpre.std(0).cpu().numpy(),
                pre_max=tpre.max(0).values.cpu().numpy(),
                cut_med=float(cuts.median()),
            )
            rows.append(pd.DataFrame(d))
            log("L%d %s: %d targets, cut_med %.3f, %.1fs" % (layer, kind, len(idx), float(cuts.median()),
                                                            time.time() - t0))
            del We, be, Wd, bd, beff, Ehat, Dhat, ce, cd, sd, cuts, tpre, tin
            torch.cuda.empty_cache()
    pd.concat(rows).to_csv(OUT / "props.csv", index=False)
    pd.DataFrame(site_rows).to_csv(OUT / "sites.csv", index=False)
    log("wrote %s" % (OUT / "props.csv"))


if __name__ == "__main__":
    main()
