"""How many 64-token sequences a full-pool data pass needs (DAN-37), from measured firing statistics.

A target needs (protocol-v1 §3.1) 64 strongest contexts and 64 mid-band contexts. For a latent that fires in a
fraction p of sequences, and whose per-sequence score lands in the mid band mu + [0.5, 1.5] sigma for a fraction f of
its firing sequences (the reservoir only samples after W warm-up sequences):

    top store full   N >= 64 / p
    mid store full   N >= W + 64 / (p f)

p and f are read per latent from a finished pass (latent_stats.seq_count, mid_ctx.reservoir_n). This prints, for a
grid of corpus sizes, the fraction of latents (a uniform draw over every component, i.e. the stratified_random
frame) whose two stores would be full, for the public run (--out) and, as calibration, for TuringLLM's own stores
(--reference, read-only; its 24.8M-sequence corpus gave 90 % full mid stores).

A small corpus resolves p only down to ~1/N_small: latents seen in fewer than --min-count sequences are counted as
NOT full at every N, so the public coverage figures are lower bounds. f is taken per latent where the latent fired
in >= --f-min-count post-warm-up sequences, else the median f of its component.

  PYTHONPATH=src .venv/bin/python experiments/069-public-data-pass/estimate_corpus.py \
      --family pythia-70m --data data_public/pythia-70m --out outputs_public/pythia-70m --reference outputs
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GRID = [1e6, 2e6, 5e6, 1e7, 2e7, 2.5e7, 5e7, 1e8]


def _load(root: str):
    import torch
    ls = torch.load(os.path.join(root, "latent_stats.pt"), map_location="cpu", weights_only=False)
    mid = torch.load(os.path.join(root, "mid_ctx.pt"), map_location="cpu", weights_only=False, mmap=True)
    sr = torch.load(os.path.join(root, "seq_repr.pt"), map_location="cpu", weights_only=False, mmap=True)
    return ls, mid, int(sr["n_seqs"])


def coverage(ls, mid, n_seqs: int, warmup_seqs: int, real_warmup_seqs: int, min_count: int, f_min_count: int):
    import torch
    sc = ls["seq_count"].double()                                # [C, D] sequences in which the latent fired
    rn = mid["reservoir_n"].double()                             # in-band sequences seen after warm-up
    live = sc.sum(1) > 0                                         # a component with no firing at all = missing SAE site
    n_missing_comp = int((~live).sum())
    sc, rn = sc[live], rn[live]
    p = sc / n_seqs
    post = (sc * max(0, n_seqs - warmup_seqs) / n_seqs).clamp(min=1)
    f = rn / post
    ok_f = sc * (n_seqs - warmup_seqs) / n_seqs >= f_min_count
    f_comp = torch.stack([f[c][ok_f[c]].median() if ok_f[c].any() else torch.tensor(float("nan"))
                          for c in range(f.shape[0])])
    f_med = float(f[ok_f].median()) if ok_f.any() else float("nan")
    f_comp = torch.where(torch.isnan(f_comp), torch.full_like(f_comp, f_med), f_comp)
    f_use = torch.where(ok_f, f, f_comp.unsqueeze(1).expand_as(f)).clamp(min=1e-6)
    resolved = sc >= min_count
    need_top = torch.where(resolved, 64.0 / p.clamp(min=1e-12), torch.full_like(p, float("inf")))
    need_mid = torch.where(resolved, real_warmup_seqs + 64.0 / (p * f_use).clamp(min=1e-12),
                           torch.full_like(p, float("inf")))
    need = torch.maximum(need_top, need_mid)
    rows = {}
    for n in GRID:
        rows["%.1e" % n] = dict(top_full=round(float((need_top <= n).double().mean()), 4),
                                both_full=round(float((need <= n).double().mean()), 4))
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    fin = need[torch.isfinite(need)].float()
    return dict(n_seqs=n_seqs, latents=int(sc.numel()), components_excluded_no_firing=n_missing_comp,
                dead_frac=round(float((sc == 0).double().mean()), 4),
                resolved_frac=round(float(resolved.double().mean()), 4),
                p_seq_quantiles=dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                         [float("%.3g" % v) for v in torch.quantile(p.flatten().float(), q).tolist()])),
                f_band_median=round(f_med, 4),
                f_band_quantiles=dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                          [round(v, 4) for v in torch.quantile(f[ok_f].float(), q).tolist()]))
                if ok_f.any() else None,
                need_quantiles_resolved=dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                                 [float("%.3g" % v) for v in torch.quantile(fin, q).tolist()]))
                if fin.numel() else None,
                coverage=rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="corpus size needed for full top / mid stores")
    ap.add_argument("--family", required=True, choices=["pythia-70m", "gpt2-small"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None, help="TuringLLM artefact root (read-only), e.g. outputs")
    ap.add_argument("--real-warmup-seqs", type=int, default=100 * 512,
                    help="warm-up of the real run (warmup_batches x batch_size; config default 100 x 512)")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--f-min-count", type=int, default=50)
    args = ap.parse_args(argv)
    common.configure(args.family, args.data, args.out)
    import json

    man = common.load_json(os.path.join(args.out, "data_pass_manifest.json"))
    warm = int(man["overrides"]["mid_warmup_batches"]) * int(man["overrides"]["batch_size"])
    ls, mid, n = _load(args.out)
    report = dict(family=args.family, real_warmup_seqs=args.real_warmup_seqs,
                  public=coverage(ls, mid, n, warm, args.real_warmup_seqs, args.min_count, args.f_min_count))
    if args.reference:
        rls, rmid, rn = _load(args.reference)
        steps = rls.get("component_steps", {})
        rwarm = 100 * (rn // max(1, int(max(steps.values())))) if steps else 0   # 100 warm-up batches
        report["reference_turingllm"] = coverage(rls, rmid, rn, rwarm, rwarm, args.min_count, args.f_min_count)
        report["reference_turingllm"]["mid_full_measured"] = round(float((rmid["reservoir_fill"] >= 64).double().mean()), 4)
    path = os.path.join(args.out, "corpus_estimate.json")
    common.write_json(path, report)
    print(json.dumps(report, indent=2))
    print("report:", path)


if __name__ == "__main__":
    main()
