"""Duplicate-rate report for a family's context stores (DAN-37: "dedupe contexts by sequence").

Two levels:
  corpus   exact duplicate 64-token windows found while the corpus was cut (corpus_manifest.json: dropped at
           preparation unless --no-dedupe), and a re-count over the shards actually written (0 when deduped)
  stores   per latent row of the top store (64 strongest) and the mid-band reservoir: the fraction of stored contexts
           whose token content repeats an earlier context of the same row ("by content"), the fraction of rows with
           any repeat, and, for the protocol-v1 pools (strong = top row, mid = mid row minus top ids), mid contexts
           whose content is already in the strongest pool. Ids are unique per row by construction, so every repeat
           is the same text under two sequence ids: what Neuronpedia's 3-33 % duplicates are.

--reference-data / --reference-out run the same store measurement on TuringLLM's corpus and stores (read-only;
e.g. data and outputs): its corpus was not deduplicated, so this shows what dedupe removes on a real corpus.

Content identity = a 64-bit linear hash of the token ids (random odd weights, fixed seed); a collision between two
different 64-token windows has probability below 2^-47.

  PYTHONPATH=src .venv/bin/python experiments/069-public-data-pass/dup_report.py \
      --family pythia-70m --data data_public/pythia-70m --out outputs_public/pythia-70m \
      [--reference-data data --reference-out outputs]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SEQ = 64


def content_hashes(data_root: str, n_shards=None):
    """[n_seqs + 1] uint64: hash of each loaded sequence (id = 1-based position over the sorted shards, the
    loader's rule; first token of each segment skipped; sequences shorter than 64 are zero-padded as the loader
    pads them). Index 0 = sentinel 0. `n_shards` = the loader's data.n_shards cut (default: every shard)."""
    import numpy as np
    w = np.random.default_rng(0).integers(1, 2 ** 62, size=SEQ, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    shards = sorted((f for f in os.listdir(data_root) if f.startswith("shard_") and f.endswith(".npy")),
                    key=lambda f: int(f.split("_")[1].split(".")[0]))[:n_shards]
    out = [np.zeros(1, dtype=np.uint64)]
    for f in shards:
        a = np.load(os.path.join(data_root, f))
        seps = np.flatnonzero(a == -1)
        starts = np.concatenate([[0], seps + 1]) + 1             # +1: skip_first_token
        ends = np.concatenate([seps, [len(a)]])
        keep = (ends - starts) > 1
        starts, ends = starts[keep], ends[keep]
        lens = np.minimum(ends - starts, SEQ)
        if (lens == SEQ).all() and len(set((ends - starts).tolist())) == 1:
            rows = a[starts[:, None] + np.arange(SEQ)[None, :]]
        else:                                                   # ragged shard: pad like DataLoader._batch_to_tensor
            rows = np.zeros((len(starts), SEQ), dtype=np.int64)
            for r, (s, n) in enumerate(zip(starts, lens)):
                rows[r, :n] = a[s:s + n]
        with np.errstate(over="ignore"):
            out.append((rows.astype(np.uint64) * w[None, :]).sum(axis=1, dtype=np.uint64))
    return np.concatenate(out)


def row_repeats(ids, hashes):
    """ids [D, N] int (0 = empty). Returns (n_stored [D], n_repeats [D]): repeats by content within each row."""
    import numpy as np
    ids = ids.astype(np.int64)
    valid = ids > 0
    h = hashes[np.where(valid, ids, 0)]
    # invalid slots get distinct sentinel hashes so they never count as repeats
    filler = (np.arange(h.size, dtype=np.uint64) + np.uint64(1)).reshape(h.shape) * np.uint64(0x9E3779B97F4A7C15)
    h = np.where(valid, h, filler)
    hs = np.sort(h, axis=1)
    vs = np.take_along_axis(valid, np.argsort(h, axis=1), axis=1)
    rep = ((hs[:, 1:] == hs[:, :-1]) & vs[:, 1:] & vs[:, :-1]).sum(1)
    return valid.sum(1), rep


def cross_repeats(top_ids, mid_ids, hashes):
    """Mid contexts (mid row minus top ids) whose content is already in the same target's top row."""
    import numpy as np
    top = top_ids.astype(np.int64)
    mid = mid_ids.astype(np.int64)
    mid_valid = (mid > 0) & ~((mid[:, :, None] == top[:, None, :]).any(-1))
    th = np.where(top > 0, hashes[top], np.uint64(0))
    mh = hashes[np.where(mid_valid, mid, 0)]
    hit = (mh[:, :, None] == th[:, None, :]).any(-1) & mid_valid & (th > 0).any(-1, keepdims=True)
    return mid_valid.sum(1), hit.sum(1)


def summary(n_stored, n_rep, min_stored=1):
    import numpy as np
    m = n_stored >= min_stored
    tot, rep = int(n_stored[m].sum()), int(n_rep[m].sum())
    rate = n_rep[m] / np.maximum(n_stored[m], 1)
    q = (lambda p: round(float(np.quantile(rate, p)), 6)) if rate.size else (lambda p: 0.0)
    return dict(rows=int(m.sum()), contexts=tot, repeats=rep, rate=round(rep / tot, 6) if tot else 0.0,
                rows_with_repeat_frac=round(float((n_rep[m] > 0).mean()), 6) if rate.size else 0.0,
                row_rate_p50=q(0.5), row_rate_p90=q(0.9), row_rate_p99=q(0.99),
                row_rate_max=round(float(rate.max()), 6) if rate.size else 0.0)


def store_report(top_path: str, mid_path: str, hashes) -> dict:
    import numpy as np
    import torch
    top = torch.load(top_path, map_location="cpu", weights_only=False, mmap=True)["ctx_seq_idx"]
    mid = torch.load(mid_path, map_location="cpu", weights_only=False, mmap=True)["ctx_seq_idx"]
    acc = {k: ([], []) for k in ("top", "mid", "cross")}
    for c in range(int(top.shape[0])):
        t, m = top[c].numpy(), mid[c].numpy()
        for key, (s, r) in (("top", row_repeats(t, hashes)), ("mid", row_repeats(m, hashes)),
                            ("cross", cross_repeats(t, m, hashes))):
            acc[key][0].append(s)
            acc[key][1].append(r)
    cat = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in acc.items()}
    n = int(top.shape[-1])
    return dict(top_store=dict(all_rows=summary(*cat["top"]), full_rows=summary(*cat["top"], min_stored=n)),
                mid_store=dict(all_rows=summary(*cat["mid"]), rows_ge8=summary(*cat["mid"], min_stored=8)),
                mid_pool_content_in_strong_pool=summary(*cat["cross"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="duplicate-rate report (contexts deduplicated by sequence)")
    ap.add_argument("--family", required=True, choices=["pythia-70m", "gpt2-small"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-data", default=None, help="TuringLLM shard dir (read-only), e.g. data")
    ap.add_argument("--reference-out", default=None, help="TuringLLM store dir (read-only), e.g. outputs")
    ap.add_argument("--reference-n-shards", type=int, default=None,
                    help="the reference run's data.n_shards (config.yaml: 3030)")
    ap.add_argument("--report", default=None, help="JSON path (default <out>/duplicate_report.json)")
    args = ap.parse_args(argv)
    common.configure(args.family, args.data, args.out)
    import json

    import numpy as np
    hashes = content_hashes(os.path.abspath(args.data))
    n_seqs = len(hashes) - 1
    unique = int(np.unique(hashes[1:]).size)
    man_path = os.path.join(args.data, "corpus_manifest.json")
    man = common.load_json(man_path) if os.path.exists(man_path) else {}
    counts = man.get("counts", {})
    report = dict(
        family=args.family,
        corpus=dict(sequences=n_seqs, unique_contents=unique, duplicate_rate_in_shards=round(1 - unique / max(1, n_seqs), 6),
                    windows_cut=counts.get("windows_cut"),
                    duplicate_windows_dropped_at_prep=counts.get("duplicate_windows_dropped"),
                    duplicate_window_rate_at_prep=counts.get("duplicate_window_rate"),
                    deduped_at_prep=man.get("dedupe")),
        **store_report(os.path.join(args.out, "top_ctx.pt"), os.path.join(args.out, "mid_ctx.pt"), hashes),
        note="repeat = a stored context whose 64 tokens equal an earlier context of the same row; "
             "mid_pool_content_in_strong_pool = mid-band pool contexts (ids already deduplicated against the "
             "strongest pool) whose text is in the strongest pool.",
    )
    if args.reference_data and args.reference_out:
        ref_hashes = content_hashes(os.path.abspath(args.reference_data), args.reference_n_shards)
        ref_unique = int(np.unique(ref_hashes[1:]).size)
        report["reference_turingllm"] = dict(
            corpus=dict(sequences=len(ref_hashes) - 1, unique_contents=ref_unique,
                        duplicate_rate_in_shards=round(1 - ref_unique / max(1, len(ref_hashes) - 1), 6)),
            **store_report(os.path.join(args.reference_out, "top_ctx.pt"),
                           os.path.join(args.reference_out, "mid_ctx.pt"), ref_hashes))
    path = args.report or os.path.join(args.out, "duplicate_report.json")
    common.write_json(path, report)
    print(json.dumps(report, indent=2))
    print("report:", path)


if __name__ == "__main__":
    main()
