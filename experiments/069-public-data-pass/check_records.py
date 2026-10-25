"""Build protocol-v1 context records from a public family's artefacts and verify them independently (DAN-37).

For each target, `circuit.protocol_contexts.build_record` (the production builder, unchanged) is run on the stores
under --out, then every record is checked without the builder's own code paths:

  strong / mid   ids unique, disjoint, loaded tokens == the loader's sequence; the target fires in every context
                 (peak > 0, re-encoded here); mid ids carry a store score inside the final mu + [0.5, 1.5] sigma band
  contrast       verified silent: the target is outside its site's Top-K at every position of every contrast context
                 (fresh forward + bank.encode, and the margin to the k-th pre-activation); no contrast id is one of the
                 target's own top / mid ids
  split          D_train / B_mid / train / held sizes as protocol-v1 §3.2-3.3 specify for the pool sizes found

Targets: --targets L.kind.i ... or --n-auto N (seed 0: N latents whose top store holds >= 8 contexts, spread over the
sites, plus --n-random latents drawn without that filter to show the skip paths).

  PYTHONPATH=src .venv/bin/python experiments/069-public-data-pass/check_records.py \
      --family pythia-70m --data data_public/pythia-70m --out outputs_public/pythia-70m --n-auto 6
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def _comp(bank, layer: int, kind: str) -> int:
    return layer * len(bank.kinds) + list(bank.kinds).index(kind)


def _auto_targets(bank, top_ctx, mid_ctx, n_auto: int, n_random: int, seed: int = 0):
    import torch
    g = torch.Generator().manual_seed(seed)
    n_top = (top_ctx.ctx_seq_idx > 0).sum(-1).cpu()            # [C, D]
    C = int(n_top.shape[0])
    # mid pool as protocol_contexts.store_ids builds it: mid ids not already in the top row (on the CPU: with a GPU
    # visible the two stores can load on different devices)
    n_mid = torch.zeros_like(n_top)
    for c in range(C):
        top, mid = top_ctx.ctx_seq_idx[c].long().cpu(), mid_ctx.ctx_seq_idx[c].long().cpu()
        in_top = (mid[:, :, None] == top[:, None, :]).any(-1)
        n_mid[c] = ((mid > 0) & ~in_top).sum(-1)
    sites = [c for c in range(C) if int((n_top[c] >= 8).sum()) > 0]
    picks = []
    order = torch.randperm(len(sites), generator=g).tolist()
    for j in range(n_auto):
        c = sites[order[j % len(sites)]]
        ok = torch.nonzero((n_top[c] >= 8) & (n_mid[c] >= 8)).flatten()
        if ok.numel() == 0:
            ok = torch.nonzero(n_top[c] >= 8).flatten()
        picks.append((c, int(ok[torch.randint(ok.numel(), (1,), generator=g)].item()), "auto"))
    for _ in range(n_random):
        c = int(torch.randint(C, (1,), generator=g).item())
        picks.append((c, int(torch.randint(int(n_top.shape[1]), (1,), generator=g).item()), "random"))
    return picks


def _encode_target(inference, bank, layer, kind_idx, kind, latent, tokens, batch=16):
    """Per-context (max post-Top-K activation of the target, min over positions of [k-th pre-activation - target
    pre-activation]) from a fresh forward pass."""
    import torch
    from sae.dense import target_latent_activations
    acts, margins = [], []

    def hook(layer_idx, activations):
        if layer_idx != layer:
            return
        x = activations[kind_idx]
        ta, ti = bank.encode(x, kind, layer_idx)
        acts.append(target_latent_activations(ta, ti, latent).float().amax(-1).cpu())
        pre = torch.relu(bank.preact(x, kind, layer_idx).float())
        kth = pre.topk(int(bank.k), dim=-1).values[..., -1]
        margins.append((kth - pre[..., latent]).amin(-1).cpu())

    with torch.no_grad():
        for s0 in range(0, int(tokens.shape[0]), batch):
            inference.forward(tokens[s0:s0 + batch].to(inference.device), activations_callback=hook,
                              return_activations=False, tokenize_final=False)
    return torch.cat(acts), torch.cat(margins)


def check_record(rec, inference, bank, loader, comp, latent, top_ctx, mid_ctx, latent_stats, cfg):
    import torch
    from circuit.protocol_contexts import skip_reason, store_ids
    from pipeline.component_index import split_component_idx

    kinds = list(bank.kinds)
    layer, kind_idx = split_component_idx(comp, len(kinds))
    kind = kinds[kind_idx]
    top_ids, mid_ids = store_ids(top_ctx, mid_ctx, comp, latent)
    out = dict(target="%d.%s.%d" % (layer, kind, latent), n_top=rec["n_top"], n_mid=rec["n_mid"],
               skip=skip_reason(rec), fires_in_frac_seqs=None)
    n_seqs = int(sum(loader._shard_sequence_counts))
    out["fires_in_frac_seqs"] = round(int(latent_stats.seq_count[comp, latent]) / max(1, n_seqs), 5)
    checks = {}
    for pool in ("strong", "mid"):
        p = rec.get(pool)
        if p is None:
            out[pool] = None
            continue
        ids = p["ids"]
        same = all(torch.equal(p["pos"][i, :len(loader.get_sequence(s))].long(),
                               torch.as_tensor(loader.get_sequence(s)[:64]).long()) for i, s in enumerate(ids))
        acts, _m = _encode_target(inference, bank, layer, kind_idx, kind, latent, p["pos"])
        out[pool] = dict(n=len(ids), train=len(p["train"]), held=len(p["held"]),
                         peak_min=round(float(p["peak"].min()), 4), peak_max=round(float(p["peak"].max()), 4))
        checks["%s_ids_unique" % pool] = len(set(ids)) == len(ids)
        checks["%s_tokens_match_loader" % pool] = bool(same)
        checks["%s_target_fires_everywhere" % pool] = bool((acts > 0).all())
        checks["%s_peak_matches_reencode" % pool] = bool(torch.allclose(acts, p["peak"].float(), atol=1e-4))
    if rec.get("strong") is not None and rec.get("mid") is not None:
        checks["strong_mid_disjoint"] = not (set(rec["strong"]["ids"]) & set(rec["mid"]["ids"]))
        # mid store scores vs the FINAL band (the reservoir admitted them under the running band)
        mu = float(latent_stats.mean_seq[comp, latent])
        sd = float(latent_stats.std_seq(comp)[latent])
        lo, hi = mu + float(mid_ctx._band_low) * sd, mu + float(mid_ctx._band_high) * sd
        row_ids = mid_ctx.ctx_seq_idx[comp, latent].tolist()
        row_val = mid_ctx.ctx_seq_val[comp, latent].tolist()
        score = {int(s): float(v) for s, v in zip(row_ids, row_val) if int(s) > 0}
        inside = [lo <= score[s] <= hi for s in rec["mid"]["ids"] if s in score]
        out["mid_in_final_band_frac"] = round(sum(inside) / max(1, len(inside)), 3)
    if rec.get("D_train") is not None:
        out["D_train"] = len(rec["D_train"])
        out["B_mid"] = len(rec["B_mid"]) if rec.get("B_mid") is not None else None
    neg = rec.get("neg")
    if neg is not None:
        acts, margin = _encode_target(inference, bank, layer, kind_idx, kind, latent, neg)
        out["contrast"] = dict(n=int(neg.shape[0]), max_target_act=round(float(acts.max()), 6),
                               min_topk_margin=round(float(margin.min()), 6))
        checks["contrast_verified_silent"] = bool((acts <= 0).all())
        checks["contrast_outside_topk_margin"] = bool((margin >= 0).all())
        if rec.get("neg_ids") is not None:
            checks["contrast_disjoint_from_own_pools"] = not (set(rec["neg_ids"]) & (set(top_ids) | set(mid_ids)))
            checks["contrast_ids_unique"] = len(set(rec["neg_ids"])) == len(rec["neg_ids"])
        n = int(neg.shape[0])
        out["contrast"]["train_held"] = [n - int(round(n * cfg.holdout_frac)), int(round(n * cfg.holdout_frac))]
    out["checks"] = checks
    out["all_checks_pass"] = all(checks.values()) if checks else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="protocol-v1 records from public-family artefacts")
    ap.add_argument("--family", required=True, choices=["pythia-70m", "gpt2-small"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="*", default=None, help="L.kind.i ...")
    ap.add_argument("--n-auto", type=int, default=6)
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--skip-missing-saes", action="store_true")
    ap.add_argument("--report", default=None, help="JSON path (default <out>/records_check.json)")
    args = ap.parse_args(argv)

    cfg = common.configure(args.family, args.data, args.out, skip_missing_saes=args.skip_missing_saes)
    import torch
    from circuit.protocol_contexts import build_record, make_selector
    from data.loader import DataLoader
    from model.inference import Inference
    from sae.bank import SAEBank
    from store.context import mid_ctx, top_ctx
    from store.latent_stats import latent_stats

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(device=device)
    n_seqs = int(sum(loader._shard_sequence_counts))
    common.load_stores(os.path.abspath(args.out), n_seqs)
    latent_stats.load(os.path.join(args.out, "latent_stats.pt"))
    inference = Inference(device=device)
    bank = SAEBank(device=device)
    common.check_store_grid(bank)

    if args.targets:
        picks = []
        for t in args.targets:
            layer, kind, idx = t.split(".")
            picks.append((_comp(bank, int(layer), kind), int(idx), "given"))
    else:
        picks = _auto_targets(bank, top_ctx, mid_ctx, args.n_auto, args.n_random)

    v1 = cfg.discovery.context_v1
    rows = []
    for comp, latent, why in picks:
        t0 = time.perf_counter()
        selector = make_selector(inference, bank, loader)
        rec = build_record(inference, bank, loader, comp, latent, selector=selector, cfg=v1)
        row = check_record(rec, inference, bank, loader, comp, latent, top_ctx, mid_ctx, latent_stats, v1)
        row["picked"] = why
        row["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(row, flush=True)

    report = dict(family=args.family, out_root=os.path.abspath(args.out), n_sequences=n_seqs,
                  context_v1=v1.model_dump(), records=rows,
                  n_usable=sum(1 for r in rows if r["skip"] is None),
                  all_usable_pass=all(r["all_checks_pass"] for r in rows if r["skip"] is None))
    path = args.report or os.path.join(args.out, "records_check.json")
    common.write_json(path, report)
    print("records:", len(rows), "usable:", report["n_usable"], "all usable pass:", report["all_usable_pass"])
    print("report:", path)


if __name__ == "__main__":
    main()
