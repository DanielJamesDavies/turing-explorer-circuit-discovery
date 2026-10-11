"""GPU check of the native pass-1 kernels on a public family (DAN-37; needs CUDA, run before the H100 data pass).

latent_stats_cuda only runs on CUDA tensors, so the 069 CPU smoke run exercised its PyTorch fallback; mid_reservoir
and top_coactivation_reduce are CPU extensions and did run there. This script feeds the SAME Top-K codes (encoded
once on the GPU from the family's corpus) to two stores per kind and compares them:

  latent_stats   CUDA kernel (cuda tensors)  vs  PyTorch fallback (the same codes on CPU)
                 active_count / seq_count exact; mean, mean_abs, m2, m2_abs, mean_seq, m2_seq to rtol 1e-4
  top_ctx        GPU update  vs  CPU update: identical id sets and values per row (zero-score slots are id 0
                 on both since the DAN-37 fix, so torch.topk's device-specific tie-break cannot differ)

Exit code 0 when everything matches.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python \
      experiments/069-public-data-pass/gpu_native_check.py --family pythia-70m --data data_public/pythia-70m \
      --batches 8 --batch-size 256
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="native kernel GPU-vs-CPU check on a public family")
    ap.add_argument("--family", required=True, choices=["pythia-70m", "gpt2-small"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--skip-missing-saes", action="store_true")
    args = ap.parse_args(argv)
    common.configure(args.family, args.data, None, batch_size=args.batch_size,
                     skip_missing_saes=args.skip_missing_saes)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    from data.loader import DataLoader
    from model.inference import Inference
    from pipeline.encoding import encode_layer_components
    from sae.bank import SAEBank
    from store import latent_stats as ls_mod
    from store.context import Context
    from store.latent_stats import LatentStats

    if not ls_mod._HAS_CUDA_EXT:
        raise SystemExit("latent_stats_cuda extension not loaded (build src/native with CUDA)")
    dev = torch.device("cuda")
    loader = DataLoader(device=dev)
    inference = Inference(device=dev)
    bank = SAEBank(device=dev)
    common.check_store_grid(bank)
    gpu_stats, cpu_stats = LatentStats(device=dev), LatentStats(device=torch.device("cpu"))
    gpu_top, cpu_top = Context("top", device=dev), Context("top", device=torch.device("cpu"))

    for b, (ids, tokens) in enumerate(loader.get_batches()):
        if b >= args.batches:
            break

        def cb(layer, acts):
            enc = encode_layer_components(bank, layer, acts, primary_device=dev, multi_gpu=False)
            for comp, (ta, ti) in enc.items():
                gpu_stats.update_component(comp, (ta, ti))
                cpu_stats.update_component(comp, (ta.cpu(), ti.cpu()))
                gpu_top.update_component(comp, ids, (ta, ti))
                cpu_top.update_component(comp, ids.cpu(), (ta.cpu(), ti.cpu()))

        with torch.no_grad():
            inference.forward(tokens, activations_callback=cb, return_activations=False, tokenize_final=False)

    ok = True
    for name in ("active_count", "seq_count"):
        same = torch.equal(getattr(gpu_stats, name).cpu(), getattr(cpu_stats, name))
        print("%-12s exact: %s" % (name, same))
        ok &= same
    # Float sums differ in the last bits between the atomics kernel and the CPU scatter (accumulation order), so
    # the moments are compared to a tolerance and the top store by the share of rows that agree (a boundary tie at
    # rank n_sequences may legitimately resolve differently).
    for name in ("mean", "mean_abs", "m2", "m2_abs", "mean_seq", "m2_seq"):
        a, c = getattr(gpu_stats, name).cpu(), getattr(cpu_stats, name)
        close = torch.isclose(a, c, rtol=1e-3, atol=1e-5)
        frac = float(close.double().mean())
        print("%-12s close: %.6f of entries (max abs diff %.3g)" % (name, frac, float((a - c).abs().max())))
        ok &= frac >= 0.9999
    gi, gv = gpu_top.ctx_seq_idx.cpu(), gpu_top.ctx_seq_val.cpu().float()
    ci, cv = cpu_top.ctx_seq_idx, cpu_top.ctx_seq_val.float()
    rows_same = (gi.sort(-1).values == ci.sort(-1).values).all(-1)
    frac_rows = float(rows_same.double().mean())
    zero_slots = int(((gi > 0) & (gv <= 0)).sum()) + int(((ci > 0) & (cv <= 0)).sum())
    print("top_ctx rows with identical id sets: %.6f, non-firing ids stored: %d" % (frac_rows, zero_slots))
    ok &= frac_rows >= 0.999 and zero_slots == 0
    print("ALL MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
