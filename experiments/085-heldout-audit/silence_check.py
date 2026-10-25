"""Forward-pass checks the stored records cannot answer on their own (085). Small: the 350-target audit subset.

Per target, one forward over its stored rows (same hook and batch size 16 as protocol_contexts._peak):
  contrast   max post-Top-K activation of the target over all 64 positions, per contrast row: the selector's
             silence test (<= 0). Also the target's best rank among the site's latents by pre-activation at any
             position (how close each "silent" context came to entering the Top-K; K = 128).
  activating peak / anchor re-computed for every strongest and mid-band row and compared with the stored ones
             (the split and every score depend on them), and the anchor read itself (> 0?).

Read-only. Writes out/silence_subset.csv.

  PYTHONPATH=src .venv/bin/python experiments/085-heldout-audit/silence_check.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import audit  # noqa: E402

BS = 16


def main():
    from config import config  # noqa: F401
    from hardware import detect_devices
    from model.inference import Inference
    from pipeline.component_index import component_idx
    from sae.bank import SAEBank
    from sae.dense import target_latent_activations
    import pandas as pd
    torch.set_float32_matmul_precision("high")
    dev = detect_devices()[0]
    inference = Inference(device=dev, compile=False)
    bank = SAEBank(devices=[dev], load_decoders=False, compile=False)
    kinds = list(bank.kinds)
    files = audit.pick_subset(sorted(audit.CTX.glob("*.pt")))
    rows = []
    t0 = time.time()
    for n, f in enumerate(files):
        key, rec = next(iter(torch.load(f, weights_only=False).items()))
        if rec.get("strong") is None:
            continue
        l, k, i = key.split(".")
        l, i = int(l), int(i)
        ki = kinds.index(k)
        sae = bank.saes[k][l]
        w = sae.encoder.weight[i].detach(); b = sae._get_bias_eff()[i].detach()

        def run(tokens):
            vals, args, rank_best = [], [], []

            def hook(layer_idx, acts):
                if layer_idx == l:
                    x = acts[ki]
                    ta, ti = bank.encode(x, k, layer_idx)
                    a = target_latent_activations(ta, ti, i).float()
                    vals.append(a.max(-1).values.cpu()); args.append(a.argmax(-1).cpu())
                    # rank of the target's pre-activation among the site's latents, best over positions
                    xx = x.to(sae.encoder.weight.device, sae.encoder.weight.dtype)
                    pre = torch.nn.functional.linear(xx, sae.encoder.weight, sae._get_bias_eff()).float()
                    tgt = pre[..., i:i + 1]
                    rk = (pre > tgt).sum(-1) + 1                              # 1 = top; Top-K means rank <= K
                    rank_best.append(rk.min(-1).values.cpu())
            inference.disable_compile() if hasattr(inference, "disable_compile") else None
            with torch.no_grad():
                for s0 in range(0, int(tokens.shape[0]), BS):
                    inference.forward(tokens[s0:s0 + BS].to(dev), activations_callback=hook, return_activations=False,
                                      tokenize_final=False)
            return torch.cat(vals), torch.cat(args), (torch.cat(rank_best) if rank_best else None)

        r = dict(key=key, layer=l, kind=k)
        v, a, rk = run(rec["neg"])
        r.update(ctr_n=int(v.numel()), ctr_not_silent=int((v > 0).sum()), ctr_max=float(v.max()))
        if rk is not None:
            r.update(ctr_rank_best_min=int(rk.min()), ctr_rank_best_med=float(rk.float().median()),
                     ctr_within_2k=int((rk <= 2 * int(bank.k)).sum()))
        for name in ("strong", "mid"):
            d = rec.get(name)
            if d is None:
                continue
            v, a, _ = run(d["pos"])
            pk = d["peak"].float()
            r["%s_peak_maxabsdiff" % name] = float((v - pk).abs().max())
            r["%s_peak_reldiff_med" % name] = float(((v - pk).abs() / pk.abs().clamp(min=1e-6)).median())
            r["%s_anchor_changed" % name] = int((a != d["arg"]).sum())
            r["%s_peak_le0_now" % name] = int((v <= 0).sum())
            ho = list(d["held"])
            r["%s_held_anchor_changed" % name] = int((a[ho] != d["arg"][ho]).sum())
        rows.append(r)
        if n % 25 == 0:
            print("%d/%d %s %.0fs" % (n, len(files), key, time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(HERE / "out" / "silence_subset.csv", index=False)
    print("wrote %d rows %.0fs" % (len(rows), time.time() - t0))


if __name__ == "__main__":
    main()
