"""Shared helpers for experiments that fit / score circuits under the protocol-v1 contexts of 059.

Reuses pool_test.py's context cache, probe construction and train sets without editing pool_test.py (a running
job re-imports it per stage). Everything goes through the same injection as 059: the discovery method's
build_probe_dataset / _floor_negatives and the eval-side M0 probe builder / contrast selector are replaced per target.
The protocol's own contexts now live in src/circuit/protocol_contexts.py (DAN-78); patch_eval_contexts reads them
from there, while fit_arm keeps pool_test's train sets (it serves 059's experimental arms too).

  fit_arm(G, ctx, targets, out_dir, train_arm="B", gamma=1.0, lam=2e-3, free_amp=True, steps=400)
  patch_eval_contexts(G, rec, held="strong")        -> M0 now returns strongest-train 48 + the chosen held-out 16
  eval_arm(G, V, ctx, targets, circuits, out_path, helds=("strong",), pool=None, tag={})
  load_pool()                                        -> the random-latent pool 059 uses (members of arms D/A/B/C)
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import pool_test as P  # noqa: E402


def parse(key):
    l, k, i = key.split(".")
    return int(l), k, int(i)


def configure(gamma=1.0, lam=2e-3, free_amp=True, steps=400):
    """The primary config (close C, rank-keep 3e-3, train-only ablation values) with the given weights / price."""
    from config import config
    disc = config.discovery
    lm = disc.learned_mask
    lm.mask_floor_source = "triple"; lm.free_amplitude = bool(free_amp); lm.l1_lambda = float(lam)
    lm.deep_site_threshold = 99
    lm.dual_floor_weight = float(gamma); lm.triple_floor_weight = float(gamma); lm.offtarget_weight = 0.0
    lm.rank_weight = 3e-3; lm.rank_mode = "keep"; lm.floors_train_only = True; lm.steps = int(steps)
    disc.floor_negctx_mode = "close"; disc.eval_batch_size = 64
    disc.probe_sequence_count = 128; disc.eval_sequence_count = 128
    return config


def fit_arm(G, ctx, targets, out_dir, train_arm="B", gamma=1.0, lam=2e-3, free_amp=True, steps=400, log=print):
    """Fit one arm on `targets` with 059's train set `train_arm`; resumable (skips fitted targets)."""
    from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
    configure(gamma, lam, free_amp, steps)
    M = _build_mode_method("ablation_gradient", "mask", G["inference"], G["bank"], G["avg_acts"], G["M0"].probe_builder)
    dev = G["device"]; KINDS = G["KINDS"]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "discovered_circuits.shard0.pt"
    found = torch.load(path, weights_only=False) if path.exists() else {}
    for key in targets:
        if key in found or key not in ctx:
            continue
        rec = ctx[key]
        tr = P.train_set(rec, train_arm)
        if tr is None:
            log("  %-15s skipped (no mid-band pool)" % key); continue
        held = [P.pick(rec, "strong", rec["strong"]["held"])] + ([P.pick(rec, "mid", rec["mid"]["held"])] if rec["mid"] else [])
        pdset = P.probe(rec, tr, held, dev)
        M.build_probe_dataset = lambda comp, i, _p=pdset: _p
        M._floor_negatives = lambda probe_data, comp, i, logger: probe_data.neg_tokens
        l, k, i = parse(key); comp = l * len(KINDS) + KINDS.index(k)
        ts = time.time()
        try:
            c = M.discover(comp, i)
        except Exception as e:  # noqa: BLE001
            log("  %-15s ERROR %s: %s" % (key, type(e).__name__, str(e)[:200])); continue
        if c is None:
            log("  %-15s rejected" % key); continue
        found[key] = c
        n = sum(1 for nd in c.nodes.values() if nd.metadata.get("role") != "seed")
        log("  %-15s %s  %5d nodes  %.0fs" % (key, out_dir.name, n, time.time() - ts))
        torch.save(found, path)
    return found


def patch_eval_contexts(G, rec, held="strong"):
    """Make G['M0'] hand every consumer (eval pass, 056, 057) the protocol contexts for this target: the 48
    strongest-train contexts (the common evaluation reference) followed by the chosen 16 held-out ones, and the
    stratified close contrast set. The contexts come from src (circuit/protocol_contexts.eval_contexts, DAN-78;
    identical to the earlier pool_test.probe construction, checked by 062/check_protocol_contexts.py)."""
    from circuit.protocol_contexts import eval_contexts
    M0 = G["M0"]
    pd_, sel = eval_contexts(rec, held, G["device"])
    M0.build_probe_dataset = lambda comp, i, _p=pd_: _p
    M0._neg_context_selector = lambda _s=sel: _s
    return pd_


def load_pool():
    pool = defaultdict(set)
    for arm in P.POOL_ARMS:
        p = P.HERE / ("data_%s" % arm) / "discovered_circuits.shard0.pt"
        if p.exists():
            for c in torch.load(p, weights_only=False, map_location="cpu").values():
                for n in c.nodes.values():
                    if n.metadata.get("role") != "seed":
                        f = n.metadata["feature_id"]; pool[(f.layer, f.kind)].add(f.index)
    return {s: np.array(sorted(v)) for s, v in pool.items()}


def eval_arm(G, V, ctx, targets, circuits, out_path, helds=("strong",), pool=None, tag=None, log=print):
    """Score circuits with the eval pass on the protocol contexts; resumable per (target, held)."""
    pool = load_pool() if pool is None else pool
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    arm = (tag or {}).get("arm")
    if out_path.exists():   # resume key includes the arm tag: several arms may share one output file
        done = {(r["seed"], r.get("held")) for r in map(json.loads, open(out_path))
                if "error" not in r and r.get("arm") == arm}
    with open(out_path, "a") as fh:
        for key in targets:
            c = circuits.get(key)
            if c is None or key not in ctx:
                continue
            rec = ctx[key]
            for h in helds:
                if (key, h) in done or rec.get(h) is None:
                    continue
                patch_eval_contexts(G, rec, h)
                try:
                    row = V.score_circuit(c, pool, skip_roles=True)
                    row.update(held=h, **(tag or {}))
                except Exception as e:  # noqa: BLE001
                    row = dict(seed=key, held=h, error="%s: %s" % (type(e).__name__, str(e)[:300]), **(tag or {}))
                fh.write(json.dumps(row) + "\n"); fh.flush()
                f = lambda x: "%.2f" % x if isinstance(x, float) else "-"
                log("  %-15s held %-6s n %s  free0 %s  fM %s  fN %s" % (key, h, row.get("n"), f(row.get("free0_tk")),
                                                                       f(row.get("freeM_topk_tk")), f(row.get("freeN_topk_tk"))))
