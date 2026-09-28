"""Stage 0: read-only preflight for the explorer converter.

Checks every assumption the later stages rely on and prints a report. Writes
nothing except the optional JSON report. Exit status is non-zero if any check
FAILs; WARN items need a decision but do not block.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch

from export.explorer import ids
from export.explorer.corpus import SEG_SLOTS, SEQ_LEN, VOCAB_SIZE, Corpus, shard_files

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


@dataclass
class Report:
    items: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, section: str, name: str, status: str, detail: Any = "") -> None:
        self.items.append(dict(section=section, name=name, status=status, detail=detail))
        print(f"  [{status}] {name}: {detail}", flush=True)

    def check(self, section: str, name: str, ok: bool, detail: Any = "", soft: bool = False) -> bool:
        self.add(section, name, PASS if ok else (WARN if soft else FAIL), detail)
        return ok

    @property
    def n_fail(self) -> int:
        return sum(i["status"] == FAIL for i in self.items)


def _load(path: str) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _read_jsonl(pattern: str) -> List[dict]:
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


# --------------------------------------------------------------------------- stores

def check_stores(rep: Report, outputs_dir: str, ctx: Dict[str, Any]) -> None:
    S = "stores"
    print(f"\n== stores ({outputs_dir})")
    shape = (ids.N_COMPONENTS, ids.D_SAE)

    stats = _load(os.path.join(outputs_dir, "latent_stats.pt"))
    active = stats["active_count"]
    rep.check(S, "latent_stats shape", tuple(active.shape) == shape, tuple(active.shape))
    dead = active == 0
    rep.add(S, "dead latents (active_count == 0)", INFO,
            f"{int(dead.sum())} of {ids.N_LATENTS} ({float(dead.float().mean()):.2%})")
    # TopK k = 128 everywhere, but zero activations are not counted as active, so
    # active_count.sum / k undercounts on a few components; the max is the token count.
    per_comp_tokens = (active.sum(dim=1) // 128).tolist()
    ctx["tokens_seen"] = int(max(per_comp_tokens))
    rep.add(S, "tokens seen (max over components of active_count.sum / k)", INFO,
            f"{ctx['tokens_seen']} ({sum(t < ctx['tokens_seen'] for t in per_comp_tokens)} components lower)")
    ctx["dead"] = dead
    ctx["active"] = active

    for name in ("top_ctx", "mid_ctx"):
        d = _load(os.path.join(outputs_dir, f"{name}.pt"))
        sid, val = d["ctx_seq_idx"], d["ctx_seq_val"]
        rep.check(S, f"{name} shape", tuple(sid.shape) == shape + (64,), f"{tuple(sid.shape)} ids {sid.dtype}, vals {val.dtype}")
        filled = sid > 0
        n_filled = filled.sum(dim=2)
        rep.add(S, f"{name} filled slots per latent", INFO,
                f"full(64) {int((n_filled == 64).sum())}, partial {int(((n_filled > 0) & (n_filled < 64)).sum())}, "
                f"empty {int((n_filled == 0).sum())}")
        rep.check(S, f"{name} empty slots (id 0) carry value 0",
                  bool((val[~filled] == 0).all()), f"{int((val[~filled] != 0).sum())} nonzero values at id 0", soft=True)
        live_empty = int((~dead & (n_filled == 0)).sum())
        if name == "top_ctx":
            rep.check(S, "top_ctx empty rows are dead latents", live_empty == 0, f"{live_empty} live latents with no top_ctx")
        else:
            rep.add(S, "live latents with an empty mid band", INFO, live_empty)
        ctx[f"{name}_ids"] = sid
        ctx[f"{name}_max_id"] = int(sid.max())
        del d, val

    co = _load(os.path.join(outputs_dir, "top_coactivation.pt"))
    cid, cval = co["top_indices"], co["top_values"]
    K = int(cid.shape[2])
    ctx["coact_k"] = K
    rep.check(S, "top_coactivation shape", tuple(cid.shape[:2]) == shape, f"{tuple(cid.shape)} mode={co.get('mode')}")
    rep.check(S, "coact ids are global gids (0 <= id < N_LATENTS)",
              int(cid.min()) >= 0 and int(cid.max()) < ids.N_LATENTS, f"range {int(cid.min())}..{int(cid.max())}")
    rep.add(S, "coact value range (PMI, clamped)", INFO, f"{float(cval.min()):.3g}..{float(cval.max()):.3g}")
    flat_ids = cid.reshape(ids.N_LATENTS, K)
    flat_val = cval.reshape(ids.N_LATENTS, K)
    dead_flat = dead.reshape(-1)
    if dead_flat.any():
        d_rows = flat_ids[dead_flat][:1000]
        d_vals = flat_val[dead_flat][:1000]
        rep.add(S, "coact rows of dead latents", INFO,
                f"ids all 0: {bool((d_rows == 0).all())}, values unique (sample): {torch.unique(d_vals)[:6].tolist()}")
    rng = random.Random(0)
    live = torch.nonzero(~dead_flat).squeeze(1).tolist()
    sample = rng.sample(live, min(2000, len(live)))
    self_hits = sum(int(g in set(flat_ids[g].tolist())) for g in sample)
    partner_live = float((~dead_flat[flat_ids[sample].long()]).float().mean())
    rep.add(S, "coact includes the latent itself (live sample)", INFO, f"{self_hits}/{len(sample)}")
    rep.check(S, "coact partners are live latents (sample)", partner_live > 0.99, f"{partner_live:.4f}", soft=True)
    sv = flat_val[sample]
    sorted_desc = float((sv[:, :-1] >= sv[:, 1:]).all(dim=1).float().mean())
    rep.add(S, "coact rows already sorted by value (desc), live sample", INFO,
            f"{sorted_desc:.1%} of rows (stage 1 sorts them)")
    floor = sv == float(cval.min())
    rep.add(S, "coact entries at the PMI floor (padding?), live sample", INFO,
            f"{float(floor.float().mean()):.1%} of entries; rows with any: {float(floor.any(dim=1).float().mean()):.1%}")
    zero_id = flat_ids[sample] == 0
    rep.add(S, "coact id 0 in live rows (0 = 0.attn.0 or padding)", INFO,
            f"{int(zero_id.sum())} entries, of which at PMI floor: {int((zero_id & floor).sum())}")
    del co, cid, cval, flat_ids, flat_val


# --------------------------------------------------------------------------- corpus

def check_corpus(rep: Report, data_dir: str, ctx: Dict[str, Any], sample_shards: int) -> None:
    S = "corpus"
    print(f"\n== corpus ({data_dir})")
    paths = shard_files(data_dir)
    rep.add(S, "shards", INFO, len(paths))
    scan = paths if sample_shards <= 0 else random.Random(0).sample(paths, min(sample_shards, len(paths)))
    counts, bad_layout, max_tok, min_tok, slot0_max = [], [], 0, 1 << 62, 0
    t0 = time.time()
    for i, p in enumerate(scan):
        a = np.load(p, mmap_mode="r")
        if a.shape[0] % SEG_SLOTS:
            bad_layout.append((os.path.basename(p), "size not a multiple of 66"))
            continue
        seg = np.asarray(a).reshape(-1, SEG_SLOTS)
        if not (seg[:, -1] == -1).all() or (seg[:, :-1] == -1).any():
            bad_layout.append((os.path.basename(p), "separator not at slot 65 only"))
        tok = seg[:, 1:1 + SEQ_LEN]
        max_tok, min_tok = max(max_tok, int(tok.max())), min(min_tok, int(tok.min()))
        slot0_max = max(slot0_max, int(seg[:, 0].max()))
        if i % 250 == 0:
            print(f"    scanned {i + 1}/{len(scan)} shards ({time.time() - t0:.0f}s)", flush=True)
    for p in paths:  # sizes are cheap to read for every shard
        counts.append(np.load(p, mmap_mode="r").shape[0] // SEG_SLOTS)
    full = collections.Counter(counts)
    scope = "all shards" if sample_shards <= 0 else f"{len(scan)} sampled shards"
    rep.check(S, f"segment layout 66 slots, separator last ({scope})", not bad_layout, bad_layout[:5] or "ok")
    rep.check(S, f"token ids in slots 1-64 < {VOCAB_SIZE} ({scope})", max_tok < VOCAB_SIZE and min_tok >= 0,
              f"min {min_tok}, max {max_tok}")
    rep.add(S, "slot 0 max (not a token)", INFO, slot0_max)
    rep.check(S, "every shard but the last holds 8192 sequences", set(counts[:-1]) == {8192},
              f"{dict(full)}; last shard {counts[-1]}")
    n_seq = int(sum(counts))
    ctx["n_seq"] = n_seq
    ctx["seqs_per_shard"] = counts
    rep.add(S, "total sequences", INFO, n_seq)
    for name in ("top_ctx", "mid_ctx"):
        if f"{name}_max_id" in ctx:
            rep.check(S, f"{name} max id <= total sequences", ctx[f"{name}_max_id"] <= n_seq,
                      f"{ctx[f'{name}_max_id']} <= {n_seq}")
    if "tokens_seen" in ctx:
        rep.check(S, "stores saw every sequence (tokens seen == sequences x 64)", ctx["tokens_seen"] == n_seq * SEQ_LEN,
                  f"{ctx['tokens_seen']} vs {n_seq * SEQ_LEN}")

    corpus = Corpus(data_dir, counts)
    ctx["corpus"] = corpus
    try:
        from data.loader import DataLoader
        loader = DataLoader(device=torch.device("cpu"), pin_memory=False)
        rng = random.Random(1)
        probe = [1, 8192, 8193, n_seq] + [rng.randint(1, n_seq) for _ in range(40)]
        mism = [s for s in probe if not np.array_equal(corpus.get(s), loader.get_sequence(s)[:SEQ_LEN])]
        rep.check(S, "Corpus.get(id) == DataLoader.get_sequence(id) (44 ids incl. shard edges)", not mism, mism[:5] or "ok")
    except Exception as e:  # noqa: BLE001 - report, don't crash the preflight
        rep.add(S, "DataLoader comparison", WARN, f"skipped: {e!r}")


# --------------------------------------------------------------------------- run

def check_run(rep: Report, run_dir: str, ctx: Dict[str, Any], n_token_targets: int) -> None:
    S = "run"
    print(f"\n== run ({run_dir})")
    main = os.path.join(run_dir, "main")
    status = _read_jsonl(os.path.join(main, "status.shard*.jsonl"))
    contexts = {r["seed"]: r for r in _read_jsonl(os.path.join(main, "contexts.shard*.jsonl"))}
    evals = _read_jsonl(os.path.join(main, "eval.shard*.jsonl"))
    specs = _read_jsonl(os.path.join(main, "spec.shard*.jsonl"))
    trains = {r["seed"]: r for r in _read_jsonl(os.path.join(main, "train.shard*.jsonl"))}
    circ_files = {os.path.basename(f)[:-3]: f for f in glob.glob(os.path.join(main, "circuits", "*.pt"))}
    ctx_files = {os.path.basename(f)[:-3]: f for f in glob.glob(os.path.join(run_dir, "ctx", "*.pt"))}

    targets = {r["seed"] for r in status}
    skipped = {r["seed"]: r.get("skip") for r in status if r.get("skip")}
    ok = targets - set(skipped)
    rep.add(S, "targets / circuits / skipped", INFO,
            f"{len(targets)} / {len(circ_files)} / {len(skipped)} {dict(collections.Counter(skipped.values()))}")
    kinds = collections.Counter(ids.parse_key(k)[1] for k in targets)
    rep.add(S, "targets by kind", INFO, dict(kinds))
    rep.check(S, "circuit files == non-skipped targets", set(circ_files) == ok,
              f"missing {sorted(ok - set(circ_files))[:5]}, extra {sorted(set(circ_files) - ok)[:5]}")
    rep.check(S, "context files for every target", targets <= set(ctx_files), f"missing {sorted(targets - set(ctx_files))[:5]}")
    rep.check(S, "contexts jsonl row per circuit", set(contexts) == set(circ_files), f"{len(contexts)} rows")
    rep.check(S, "train jsonl row per circuit", set(trains) == set(circ_files), f"{len(trains)} rows")
    ev_per = collections.Counter(r["seed"] for r in evals)
    ev_held = collections.Counter(r.get("held") for r in evals)
    rep.check(S, "eval rows per circuit in {1, 2}", set(ev_per.values()) <= {1, 2} and set(ev_per) == set(circ_files),
              f"{dict(collections.Counter(ev_per.values()))} held={dict(ev_held)}")
    sp_per = collections.Counter(r["seed"] for r in specs)
    rep.check(S, "spec rows per circuit == 3 (Z/A/C)", set(sp_per.values()) == {3} and set(sp_per) == set(circ_files),
              f"{dict(collections.Counter(sp_per.values()))} pi={dict(collections.Counter(r.get('pi') for r in specs))}")
    eval_comp_bad = [r["seed"] for r in evals
                     if r.get("comp_idx") is not None and r["comp_idx"] != ids.comp_of(*ids.parse_key(r["seed"])[:2])]
    rep.check(S, "eval comp_idx == layer*3 + kind_idx", not eval_comp_bad, eval_comp_bad[:5] or "ok")

    # circuits: kind order, star structure, member counts
    bad_meta, bad_seed, non_star, dup_nodes, missing_attr = [], [], [], [], []
    n_nodes, topo = [], collections.Counter()
    for key, f in sorted(circ_files.items()):
        c = _load(f)
        if isinstance(c, dict):
            c = next(iter(c.values()))
        if c is None:
            topo["None"] += 1
            continue
        layer, kind, latent = ids.parse_key(key)
        md = c.metadata
        if md.get("seed_comp") != ids.comp_of(layer, kind) or md.get("seed_latent") != latent:
            bad_meta.append(key)
        seeds = [n for n in c.nodes.values() if n.metadata.get("role") == "seed"]
        fid = seeds[0].feature_id if len(seeds) == 1 else None
        if fid is None or (fid.layer, fid.kind, fid.index) != (layer, kind, latent):
            bad_seed.append(key)
        seed_uuid = seeds[0].uuid if seeds else None
        if any(e.target_uuid != seed_uuid for e in c.edges) or len(c.edges) != len(c.nodes) - 1:
            non_star.append(key)
        fids = [(n.feature_id.layer, n.feature_id.kind, n.feature_id.index) for n in c.nodes.values()]
        if len(set(fids)) != len(fids):
            dup_nodes.append(key)
        if any("attribution_score" not in n.metadata or "amplitude" not in n.metadata
               for n in c.nodes.values() if n.metadata.get("role") != "seed"):
            missing_attr.append(key)
        n_nodes.append(len(c.nodes))
        topo["star" if key not in non_star else "other"] += 1
    rep.check(S, "circuit metadata seed_comp/seed_latent match key (kind order)", not bad_meta, bad_meta[:5] or "ok")
    rep.check(S, "exactly one seed node, matching key", not bad_seed, bad_seed[:5] or "ok")
    rep.check(S, "circuits are stars (member -> seed edges)", not non_star, f"{dict(topo)} {non_star[:5]}")
    rep.check(S, "no duplicate latents within a circuit", not dup_nodes, dup_nodes[:5] or "ok")
    rep.check(S, "members carry attribution_score and amplitude", not missing_attr, missing_attr[:5] or "ok")
    if n_nodes:
        a = np.asarray(n_nodes)
        rep.add(S, "nodes per circuit", INFO,
                f"min {a.min()}, median {int(np.median(a))}, max {a.max()}; members total {int((a - 1).sum())}")
        ctx["members"] = int((a - 1).sum())
        ctx["n_circuits"] = len(a)

    # contexts: store basis, roles, contrast, token round trip
    top_ids, mid_ids = ctx.get("top_ctx_ids"), ctx.get("mid_ctx_ids")
    corpus: Corpus | None = ctx.get("corpus")
    not_in_store, role_bad, neg_bad, neg_overlap, tok_bad, n_neg = [], [], [], [], [], []
    thin = 0
    keys = sorted(targets)
    tok_keys = set(random.Random(2).sample(keys, min(n_token_targets, len(keys))))
    for key in keys:
        rec = _load(ctx_files[key])[key]
        layer, kind, latent = ids.parse_key(key)
        comp = ids.comp_of(layer, kind)
        strong, mid = rec.get("strong"), rec.get("mid")
        if mid is None:
            thin += 1
        for pool, store in (("strong", top_ids), ("mid", mid_ids)):
            p = rec.get(pool)
            if p is None or store is None:
                continue
            row = set(store[comp, latent].tolist())
            if not {int(s) for s in p["ids"]} <= row:
                not_in_store.append(f"{key}:{pool}")
        neg_ids = rec.get("neg_ids")
        if rec.get("neg") is not None:
            n_neg.append(len(rec["neg"]))
            if neg_ids is None or len(neg_ids) != len(rec["neg"]):
                neg_bad.append(key)
            pos_ids = {int(s) for p in (strong, mid) if p is not None for s in p["ids"]}
            if neg_ids and pos_ids & {int(s) for s in neg_ids}:
                neg_overlap.append(key)
        row = contexts.get(key)
        if row is not None and strong is not None:
            s_ids = [int(x) for x in strong["ids"]]
            held_strong = {s_ids[int(j)] for j in strong["held"]}
            ok_roles = held_strong == {int(x) for x in row["held_strong_ids"]}
            if mid is not None:
                m_ids = [int(x) for x in mid["ids"]]
                ok_roles &= {m_ids[int(j)] for j in mid["held"]} == {int(x) for x in row["held_mid_ids"]}
            if neg_ids is not None:
                ok_roles &= [int(x) for x in row["contrast_train_ids"]] + [int(x) for x in row["contrast_held_ids"]] \
                    == [int(x) for x in neg_ids]
            if not ok_roles:
                role_bad.append(key)
        if key in tok_keys and corpus is not None:
            for pool in ("strong", "mid"):
                p = rec.get(pool)
                if p is not None and not np.array_equal(corpus.get_many(p["ids"]), p["pos"].numpy()):
                    tok_bad.append(f"{key}:{pool}")
            if rec.get("neg") is not None and neg_ids is not None \
                    and not np.array_equal(corpus.get_many(neg_ids), rec["neg"].numpy()):
                tok_bad.append(f"{key}:neg")
    rep.add(S, "thin targets (no mid pool)", INFO, thin)
    rep.add(S, "contrast contexts per target", INFO, dict(collections.Counter(n_neg)))
    rep.check(S, "run strong/mid ids come from the store top/mid rows (same id basis)",
              not not_in_store, f"{len(not_in_store)} pools outside store {not_in_store[:5]}")
    rep.check(S, "neg_ids present and row-aligned with neg tokens", not neg_bad, neg_bad[:5] or "ok")
    rep.check(S, "contrast ids disjoint from activating ids", not neg_overlap, neg_overlap[:5] or "ok")
    rep.check(S, "jsonl roles == ctx held indices; contrast_train+held == neg_ids order", not role_bad,
              f"{len(role_bad)} mismatched {role_bad[:5]}")
    rep.check(S, f"Corpus tokens == stored ctx tokens ({len(tok_keys)} targets, all pools)", not tok_bad,
              tok_bad[:5] or "ok")
    ctx["n_targets"] = len(targets)
    ctx["n_ctx_rows"] = sum(n_neg) + sum(64 for _ in keys) * 2


# --------------------------------------------------------------------------- sizes

def estimate_sizes(rep: Report, ctx: Dict[str, Any], full_targets: int) -> None:
    S = "sizes"
    print("\n== size estimate")
    gb = 1024 ** 3
    K = ctx.get("coact_k", 64)
    arrays = ids.N_LATENTS * (2 * 64 * (4 + 2) + K * (4 + 2))
    tokens = ctx.get("n_seq", 0) * SEQ_LEN * 2
    rep.add(S, "arrays/", INFO, f"{arrays / gb:.2f} GB (top+mid 64, coact {K})")
    rep.add(S, "tokens/", INFO, f"{tokens / gb:.2f} GB")
    if ctx.get("n_circuits"):
        scale = full_targets / max(ctx["n_targets"], 1)
        members = ctx["members"] * scale
        # member row ~40 B incl. index; graph blob ~ 12 B/node compressed; target_ctx row ~ 40 B
        sqlite = members * 40 + members * 12 + full_targets * 192 * 40
        rep.add(S, f"explorer.sqlite (scaled to {full_targets} targets)", INFO,
                f"~{sqlite / gb:.2f} GB ({members / 1e6:.1f}M members)")
        rep.add(S, "bundle total (approx.)", INFO, f"~{(arrays + tokens + sqlite) / gb:.1f} GB")


def run(args) -> int:
    rep = Report()
    ctx: Dict[str, Any] = {}
    t0 = time.time()
    if not args.skip_stores:
        check_stores(rep, args.outputs, ctx)
    check_corpus(rep, args.data, ctx, args.sample_shards)
    if args.run:
        check_run(rep, args.run, ctx, args.token_targets)
    estimate_sizes(rep, ctx, args.full_targets)
    n_warn = sum(i["status"] == WARN for i in rep.items)
    print(f"\npreflight: {rep.n_fail} FAIL, {n_warn} WARN, {len(rep.items)} items, {time.time() - t0:.0f}s")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(dict(items=rep.items, n_fail=rep.n_fail, n_warn=n_warn,
                           seqs_per_shard=ctx.get("seqs_per_shard")), fh, indent=1, default=str)
        print(f"report written to {args.report}")
    return 1 if rep.n_fail else 0
