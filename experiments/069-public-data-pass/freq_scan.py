"""Per-latent sequence firing rates at a few sites over a larger corpus (DAN-37 corpus-size estimate).

The 4,096-sequence smoke run resolves a latent's sequence firing rate p only down to ~1e-3, so the corpus-size
estimate (estimate_corpus.py) is a lower bound for the rare tail. This scan encodes only --sites (cheap) over
--max-seqs sequences of a shard directory and counts, per latent, the sequences it fires in. With the mid-band
fraction f measured by the full pass (median per kind, from --out's latent_stats / mid_ctx), it reports, for a grid of
corpus sizes N, the share of latents whose top store (N p >= 64) and mid store (W + 64 / (p f) <= N) would be full.

  PYTHONPATH=src .venv/bin/python experiments/069-public-data-pass/freq_scan.py --family pythia-70m \
      --data data_public/pythia-70m-scan --out outputs_public/pythia-70m --sites 1.attn 2.mlp 4.resid
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GRID = [2.5e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2.5e7]


def main(argv=None):
    ap = argparse.ArgumentParser(description="sequence firing-rate scan at a few sites")
    ap.add_argument("--family", required=True, choices=["pythia-70m", "gpt2-small"])
    ap.add_argument("--data", required=True, help="shard directory to scan")
    ap.add_argument("--out", required=True, help="full-pass artefact root (for f) and report location")
    ap.add_argument("--sites", nargs="+", required=True, help="L.kind ...")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--real-warmup-seqs", type=int, default=100 * 512)
    ap.add_argument("--min-count", type=int, default=5)
    args = ap.parse_args(argv)
    common.configure(args.family, args.data, None, batch_size=args.batch_size)
    import json
    import torch
    from data.loader import DataLoader
    from model.inference import Inference
    from sae.bank import SAEBank

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(device=device)
    inference = Inference(device=device)
    bank = SAEBank(device=device, preload=False)          # public bank: loads only the scanned sites
    kinds = list(bank.kinds)
    sites = []
    for s in args.sites:
        layer, kind = s.split(".")
        sites.append((int(layer), kind))
    by_layer = {}
    for layer, kind in sites:
        by_layer.setdefault(layer, []).append(kind)
    counts = {s: torch.zeros(int(bank.d_sae), dtype=torch.long) for s in sites}
    n = 0
    t0 = time.perf_counter()
    for ids, tokens in loader.get_batches():
        if args.max_seqs is not None and n >= args.max_seqs:
            break

        def cb(layer, acts):
            for kind in by_layer.get(layer, []):
                ta, ti = bank.encode(acts[kinds.index(kind)], kind, layer)
                fired = torch.zeros(ta.shape[0], int(bank.d_sae), dtype=torch.bool, device=ta.device)
                fired.scatter_(1, ti.reshape(ta.shape[0], -1).long(), (ta.reshape(ta.shape[0], -1) > 0))
                counts[(layer, kind)] += fired.sum(0).cpu()

        with torch.no_grad():
            inference.forward(tokens, activations_callback=cb, return_activations=False, tokenize_final=False)
        n += int(tokens.shape[0])
        if (n // args.batch_size) % 20 == 0:
            print("  %d sequences (%.0f s)" % (n, time.perf_counter() - t0), flush=True)

    # mid-band fraction f per kind from the full pass (median over latents seen in >= 50 post-warm-up sequences)
    ls = torch.load(os.path.join(args.out, "latent_stats.pt"), map_location="cpu", weights_only=False)
    mid = torch.load(os.path.join(args.out, "mid_ctx.pt"), map_location="cpu", weights_only=False)
    man = common.load_json(os.path.join(args.out, "data_pass_manifest.json"))
    n_full = int(man["n_sequences"])
    warm = int(man["overrides"]["mid_warmup_batches"]) * int(man["overrides"]["batch_size"])
    post = ls["seq_count"].double() * max(0, n_full - warm) / n_full
    f_all = mid["reservoir_n"].double() / post.clamp(min=1)
    rows = {}
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    for (layer, kind), c in counts.items():
        comp = layer * len(kinds) + kinds.index(kind)
        ok = post[comp] >= 50
        f = float(f_all[comp][ok].median()) if ok.any() else float(f_all[post >= 50].median())
        p = c.double() / max(1, n)
        resolved = c >= args.min_count
        need_top = torch.where(resolved, 64.0 / p.clamp(min=1e-12), torch.full_like(p, float("inf")))
        need_mid = torch.where(resolved, args.real_warmup_seqs + 64.0 / (p * f).clamp(min=1e-12),
                               torch.full_like(p, float("inf")))
        need = torch.maximum(need_top, need_mid)
        rows["%d.%s" % (layer, kind)] = dict(
            f_band_median=round(f, 4), dead_frac=round(float((c == 0).double().mean()), 4),
            resolved_frac=round(float(resolved.double().mean()), 4),
            p_seq_quantiles=dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                     [float("%.3g" % v) for v in torch.quantile(p.float(), q).tolist()])),
            coverage={"%.1e" % g: dict(top_full=round(float((need_top <= g).double().mean()), 4),
                                       both_full=round(float((need <= g).double().mean()), 4)) for g in GRID})
    report = dict(family=args.family, scanned_sequences=n, data=os.path.abspath(args.data),
                  real_warmup_seqs=args.real_warmup_seqs, min_count=args.min_count,
                  note="coverage counts latents seen in < min_count scanned sequences as NOT full: a lower bound "
                       "valid up to N ~ 64 / (f * min_count / scanned_sequences)",
                  sites=rows, seconds=round(time.perf_counter() - t0, 1))
    path = os.path.join(args.out, "freq_scan.json")
    common.write_json(path, report)
    print(json.dumps(report, indent=2))
    print("report:", path)


if __name__ == "__main__":
    main()
