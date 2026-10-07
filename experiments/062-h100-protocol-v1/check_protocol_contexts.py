"""DAN-78 equivalence check: src/circuit/protocol_contexts.py against the run's cached context records.

Per target (stratified over layers 0-11 x attn / mlp / resid from out/targets.csv, plus thin and skip targets):
  record     rebuilt with the src module == 059 pool_test.build run on this GPU (every field exact), and == the
             cached out/ctx/<key>.pt (tokens, anchors, splits, ids, D_train, B_mid exact; peak within PEAK_TOL, max
             |diff| reported; the contrast set may be a reordering of the same contexts, from near-tied similarity
             ranks on different hardware: reported per target with whether its train / held-out sets still agree)
  training   src training_probe(cached rec) == pool_test.probe(rec, pool_test.train_set(rec, arm), held) exactly
             (and from the rebuilt record up to that contrast reordering); the engine path (config
             context_protocol="v1", cache_dir = out/ctx) M.build_probe_dataset / M._floor_negatives give the same
  eval       src eval_contexts(cached rec) == the 059 patch_eval_contexts formula, and so does
             protocol_harness.patch_eval_contexts, for held = strong and mid

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/check_protocol_contexts.py
  env: N_PER_CELL (targets per layer x kind, default 1)  SCRATCH (where pool_test.build writes, default /tmp)
"""
import csv
import os
import random
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).parent
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
sys.path.insert(0, str(HERE))
import driver  # noqa: E402

N_PER_CELL = int(os.environ.get("N_PER_CELL", "1"))
SCRATCH = Path(os.environ.get("SCRATCH", tempfile.gettempdir()))
PEAK_TOL = 1e-3
# thin (no mid-band pool: full / 17 / 9 strongest contexts) and skip records (too few contexts, no contrast)
EXTRA = ["6.attn.10737", "2.attn.4114", "10.attn.20049", "8.attn.21571", "4.attn.15114"]


def pick_targets():
    rows = list(csv.DictReader(open(driver.OUT / "targets.csv")))
    rng = random.Random(78)
    out = []
    for layer in range(12):
        for kind in ("attn", "mlp", "resid"):
            cell = [r["seed"] for r in rows if int(r["layer"]) == layer and r["kind"] == kind]
            out += rng.sample(cell, min(N_PER_CELL, len(cell)))
    return out + [k for k in EXTRA if k not in out]


def neg_reorder(a, b):
    """Rows of a's contrast set out of place relative to b's, given the same id set; and whether the stratified
    train / held-out contrast SETS still agree. None when the sets differ."""
    ia, ib = a.get("neg_ids"), b.get("neg_ids")
    if ia is None or ib is None or sorted(ia) != sorted(ib):
        return None
    n_tr = len(ia) - int(round(len(ia) * 0.25))
    by_id = {s: r for s, r in zip(ib, b["neg"])}
    assert all(torch.equal(r, by_id[s]) for s, r in zip(ia, a["neg"])), "same ids, different tokens"
    return sum(x != y for x, y in zip(ia, ib)), set(ia[:n_tr]) == set(ib[:n_tr])


def eq_parts(a, b, where, peak_diffs, exact):
    """Assert record a == record b. exact=False (vs the H100 cache) allows peak within PEAK_TOL and a contrast set
    that is a reordering of the same contexts (returned as (rows out of place, train / held sets equal)); every
    other field must match exactly."""
    assert set(a) == set(b), (where, sorted(set(a) ^ set(b)))
    reorder = None
    for k in a:
        if k in ("strong", "mid"):
            if a[k] is None or b[k] is None:
                assert a[k] is None and b[k] is None, (where, k)
                continue
            pa, pb = a[k], b[k]
            assert set(pa) == set(pb), (where, k)
            for f in ("pos", "tgt", "arg"):
                assert torch.equal(pa[f].cpu(), pb[f].cpu()), (where, k, f)
            for f in ("train", "held", "ids"):
                assert list(pa[f]) == list(pb[f]), (where, k, f)
            d = float((pa["peak"].double() - pb["peak"].double()).abs().max())
            peak_diffs.append(d)
            assert (d == 0.0) if exact else (d <= PEAK_TOL), (where, k, "peak", d)
        elif k in ("neg", "neg_ids"):
            same = torch.equal(a["neg"], b["neg"]) and a.get("neg_ids") == b.get("neg_ids")
            if not same:
                assert not exact, (where, k)
                reorder = neg_reorder(a, b)
                assert reorder is not None, (where, "contrast id sets differ")
        else:
            assert a[k] == b[k], (where, k)
    return reorder


def eq_probe(a, b, where, neg=True):
    for f in ("pos_tokens", "target_tokens", "pos_argmax") + (("neg_tokens",) if neg else ()):
        x, y = getattr(a, f), getattr(b, f)
        assert x.device == y.device and x.dtype == y.dtype and torch.equal(x, y), (where, f)
    assert a.metadata == b.metadata, (where, a.metadata, b.metadata)


def main():
    from circuit import protocol_contexts as PC
    from config import config
    from pipeline.component_index import component_idx
    R = driver.Runner()
    G, P, H = R.G, R.P, R.H
    M0 = G["M0"]
    orig_pd, orig_sel = M0.build_probe_dataset, M0._neg_context_selector
    M = R.method(driver.GAMMA, driver.LAM, True)
    loader = M0.probe_builder.loader
    targets = pick_targets()
    print("checking %d targets" % len(targets), flush=True)
    d_cache, d_local = [], []
    n_train_ok = n_eval_ok = n_engine_ok = 0
    kinds_seen, layers_seen, skips, thin, reordered = set(), set(), [], [], []
    for key in targets:
        l, k, i = key.split("."); l, i = int(l), int(i)
        comp = component_idx(l, G["KINDS"].index(k), G["NK"])
        cached = torch.load(R.ctx_dir / ("%s.pt" % key), weights_only=False)[key]
        new = PC.build_record(G["inference"], G["bank"], loader, comp, i)
        ro = eq_parts(new, cached, key + " vs cache", d_cache, exact=False)
        if ro is not None:
            reordered.append("%s(%d rows, train/held sets %s)" % (key, ro[0], "same" if ro[1] else "DIFFER"))
        # the 059 builder on this GPU (M0 must hold its original selector)
        M0.build_probe_dataset, M0._neg_context_selector = orig_pd, orig_sel
        P.seeds = lambda _k=key: [_k]
        P.CTX = SCRATCH / "dan78_pool_test_ctx.pt"
        P.build(G)
        local = torch.load(P.CTX, weights_only=False)[key]
        eq_parts(new, local, key + " vs pool_test.build", d_local, exact=True)
        # the driver's builder (Runner.contexts), pointed at an empty scratch cache so it builds and writes there
        run_ctx, R.ctx_dir = R.ctx_dir, SCRATCH / "dan78_ctx"
        R.ctx_dir.mkdir(parents=True, exist_ok=True)
        (R.ctx_dir / ("%s.pt" % key)).unlink(missing_ok=True)
        try:
            eq_parts(R.contexts(key), local, key + " Runner.contexts vs pool_test.build", [], exact=True)
        finally:
            R.ctx_dir = run_ctx
        kinds_seen.add(k); layers_seen.add(l)
        if PC.skip_reason(cached):
            skips.append("%s:%s" % (key, PC.skip_reason(cached)))
            assert PC.training_probe(new, R.dev).pos_tokens.shape[0] == 0
            continue
        arm, is_thin = R.train_arm(cached)
        assert (arm, is_thin) == PC.training_arm(new)
        if is_thin:
            thin.append("%s(n_top %d)" % (key, cached["n_top"]))
        # training ProbeDataset: src vs the driver's construction on the cached record (exact); from the rebuilt
        # record the contrast rows match only up to the reordering reported above
        held = [P.pick(cached, "strong", cached["strong"]["held"])] + (
            [P.pick(cached, "mid", cached["mid"]["held"])] if cached["mid"] else [])
        ref = P.probe(cached, P.train_set(cached, arm), held, R.dev)
        eq_probe(PC.training_probe(cached, R.dev, arm), ref, key + " training (cached rec)")   # = Runner.fit's call
        eq_probe(PC.training_probe(new, R.dev), ref, key + " training (rebuilt rec)", neg=ro is None)
        n_train_ok += 1
        # the engine path: discovery method under context_protocol="v1", reading the run's cache
        config.discovery.context_protocol = "v1"
        config.discovery.context_v1.cache_dir = str(R.ctx_dir)
        try:
            pd_ = M.build_probe_dataset(comp, i)
            eq_probe(pd_, ref, key + " engine build_probe_dataset")
            assert M._floor_negatives(pd_, comp, i, None) is pd_.neg_tokens
        finally:
            config.discovery.context_protocol = "legacy"; config.discovery.context_v1.cache_dir = None
        n_engine_ok += 1
        # eval ProbeDataset + contrast: src vs the 059 formula and the harness
        for h in ("strong", "mid"):
            if cached.get(h) is None:
                continue
            ref_e = P.probe(cached, P.pick(cached, "strong", cached["strong"]["train"]),
                            [P.pick(cached, h, cached[h]["held"])], R.dev)
            ref_neg = P.FixedSelector(cached["neg"].to(R.dev)).select().tokens
            pd_e, sel = PC.eval_contexts(cached, h, R.dev)
            eq_probe(pd_e, ref_e, key + " eval " + h)
            assert torch.equal(sel.select().tokens, ref_neg)
            eq_probe(PC.eval_probe(new, h, R.dev), ref_e, key + " eval (rebuilt rec) " + h, neg=ro is None)
            pd_h = H.patch_eval_contexts(G, cached, h)
            eq_probe(pd_h, ref_e, key + " harness eval " + h)
            eq_probe(M0.build_probe_dataset(comp, i), ref_e, key + " harness M0 " + h)
            assert torch.equal(M0._neg_context_selector().select(comp, i, "close", 64, 16).tokens, ref_neg)
            n_eval_ok += 1
        M0.build_probe_dataset, M0._neg_context_selector = orig_pd, orig_sel
        print("ok %-15s arm %s%s | max |peak diff| vs cache %.3g%s" % (
            key, arm, " thin" if is_thin else "", max(d_cache[-2:] if cached["mid"] else d_cache[-1:]),
            "" if ro is None else " | contrast vs cache: %d rows reordered, train/held sets %s" % (
                ro[0], "same" if ro[1] else "DIFFER")), flush=True)
    print("\nRESULT: %d targets (layers %s, kinds %s) | records == cache (peak max |diff| %.3g over %d pools, %d exactly 0)"
          " | records == pool_test.build on this GPU (peak max |diff| %.3g) | training probes %d | engine path %d |"
          " eval probes %d | thin %s | skips %s | contrast reordered vs cache: %d %s" % (
              len(targets), sorted(layers_seen), sorted(kinds_seen), max(d_cache), len(d_cache),
              sum(1 for d in d_cache if d == 0.0), max(d_local), n_train_ok, n_engine_ok, n_eval_ok, thin, skips,
              len(reordered), reordered))


if __name__ == "__main__":
    main()
