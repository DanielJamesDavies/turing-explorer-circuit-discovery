"""Same method, two endpoints — latent vs behavioural circuit size.

The SFC comparison ("their circuits are 67-69 nodes, ours are 2,000") is
confounded by model, SAE, data, method, metric and endpoint all differing
at once. This runs ONE method over TWO endpoints on the same model, SAEs,
data, positions, sites and hyperparameters, so only the endpoint moves:

  pos    loss = squared error of the SEED LATENT's pre-activation
                against its natural value            (internal endpoint)
  logit  loss = squared error of log p(target token) at the same anchor
                against the FULL MODEL's log-prob    (behavioural endpoint)

Both reproduce rather than maximise, both are masked over the same
upstream sites, both use the frozen house hyperparameters. Reported per
seed: node count, and BOTH faithfulness measures on BOTH circuits
(free0 for the latent endpoint, logit faithfulness for the behavioural
one), so the size-vs-faithfulness trade is visible in one table rather
than being read across papers.

Logit faithfulness follows the SFC form, m(C) restored against the empty
circuit and the full model:
    (m(C) - m(empty)) / m(full) - m(empty))
with m = mean log p(target) at the anchor position and the empty state
being every upstream latent mean-ablated (posctx means, SFC's own floor).

  PYTHONPATH=src python experiments/025-logit-endpoint/runner.py
"""
import gzip
import json
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.learned_mask import run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    CircuitOnlyPatcher, circuit_only_activation, collect_site_means,
    measure_seed_activation, upstream_sites)
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
import os
# comp_idx 8 = L2 resid (8 upstream sites), 26 = L8 resid (26 sites).
COMP_IDX = int(os.environ.get("COMP_IDX", 8))
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
N_SEEDS = int(os.environ.get("N_SEEDS", 4))
# A single lambda cannot compare the two endpoints: the house 1e-5 is
# calibrated for the seed's pre-activation, which is exquisitely sensitive
# to upstream latents, while the OUTPUT distribution is far more robust to
# the same perturbation. At 1e-5 the logit arm keeps ~everything
# (n=40,969, faith -0.03 — measured). Both arms therefore sweep lambda and
# the comparison is read off the size-vs-faithfulness CURVE.
LAMBDAS = [1e-5, 1e-3, 1e-2, 1e-1]
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, D_SAE), device=bank.device)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.floor_source = "posctx"
cfg = disc.learned_mask

LAYER, KI = split_component_idx(COMP_IDX, n_kinds)
KIND = bank.kinds[KI]
# Two arms, two NATIVE scopes — this is the point of the comparison, not a
# confound to be normalised away:
#   pos    upstream of the seed. A latent endpoint can only be driven from
#          upstream, so that IS its whole causal scope.
#   logit  EVERY site, every kind, whole model (SFC masks every submodule).
#          The seed's posctx sequences supply the data distribution only;
#          the seed itself is not the endpoint and its own site is masked
#          like any other (tap_seed=False in the engine).
UP = sorted(upstream_sites(bank, LAYER, KIND))
ALL_SITES = sorted((l, k) for l in range(bank.n_layer) for k in bank.kinds
                   if bank.saes[k][l] is not None)
SCOPE = {"pos": UP, "logit": ALL_SITES}

# Deterministic panel: same construction as the l2-crossover runner
# (rng 42 shuffle, first 32, sorted), so COMP_IDX=8 reproduces that
# panel's seeds exactly and deeper components get a comparable draw.
import random as _random
_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu",
                   weights_only=False)
_pool = [int(c["latent_idx"]) for c in _cand if int(c["comp_idx"]) == COMP_IDX]
_random.Random(42).shuffle(_pool)
SEEDS = sorted(_pool[:32])[:N_SEEDS]
del _cand
print("L%d %s | %d seeds | %d upstream sites | house recipe steps=%d lr=%s "
      "l1=%s binarize=%s" % (LAYER, KIND, len(SEEDS), len(UP), cfg.steps,
                             cfg.lr, cfg.l1_lambda, cfg.binarize), flush=True)

TAG = "" if COMP_IDX == 8 else "_c%d" % COMP_IDX
fh_out = (HERE / ("rows%s.jsonl" % TAG)).open("a")
mh = gzip.open(HERE / ("members%s.jsonl.gz" % TAG), "at")
for sl in SEEDS:
    t_seed = time.time()
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(COMP_IDX, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    tgt = pd_.target_tokens[:N_SEQ][torch.arange(pt.shape[0]), pa]
    a_pos = float(measure_seed_activation(inference, bank, pt, LAYER, KIND, sl,
                                          pa, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, UP, pt, LAYER,
                                         KIND, sl, pos_argmax=pa,
                                         batch_size=EVAL_BS))
    den_lat = a_pos - a_e0
    # ALL_SITES is a sorted list (the engine's `sites` wants order);
    # collect_site_anchors does set arithmetic on it, so pass a set.
    means_all = collect_site_means(inference, bank, pt, set(ALL_SITES))

    def logit_metric(keep, site_means=None, scope=None):
        # SCOPE MATTERS AND MUST MATCH WHAT THE MASK CONTROLS. An earlier
        # version passed in_scope=ALL_SITES, which mean-ablates every site
        # outside the circuit INCLUDING the whole stack above the seed —
        # sites neither arm masks. That measures "can 8 upstream sites
        # drive the output through a dead upper stack", not the endpoint
        # question. Default is now the masked scope (UP).
        scope = set(ALL_SITES) if scope is None else set(scope)
        tot, n = 0.0, int(pt.shape[0])
        inference.disable_compile()
        try:
            for s in range(0, n, EVAL_BS):
                tk = pt[s:s + EVAL_BS]
                p = CircuitOnlyPatcher(bank=bank, keep_indices=keep,
                                       in_scope=scope, seed_layer=-1,
                                       seed_kind="", seed_latent_idx=0,
                                       site_means=site_means)
                _, lg, _ = inference.forward(tk, patcher=p, all_logits=True,
                                             grad_enabled=False,
                                             return_activations=False,
                                             tokenize_final=False)
                b = torch.arange(tk.shape[0], device=device)
                lp = torch.log_softmax(
                    lg[b, pa[s:s + EVAL_BS].to(device)].float(), dim=-1)
                tot += float(lp[b, tgt[s:s + EVAL_BS].to(device)].sum())
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    def full_logit():
        tot, n = 0.0, int(pt.shape[0])
        inference.disable_compile()
        try:
            for s in range(0, n, EVAL_BS):
                tk = pt[s:s + EVAL_BS]
                _, lg, _ = inference.forward(tk, all_logits=True,
                                             grad_enabled=False,
                                             return_activations=False,
                                             tokenize_final=False)
                b = torch.arange(tk.shape[0], device=device)
                lp = torch.log_softmax(
                    lg[b, pa[s:s + EVAL_BS].to(device)].float(), dim=-1)
                tot += float(lp[b, tgt[s:s + EVAL_BS].to(device)].sum())
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    means_up = {s: v for s, v in means_all.items() if s in set(UP)}
    m_full = full_logit()
    # one empty-circuit baseline PER SCOPE — each arm is scored against the
    # ablation of its own scope, which is what its mask actually controls
    m_empty = {"pos": logit_metric({}, site_means=means_up, scope=set(UP)),
               "logit": logit_metric({}, site_means=means_all,
                                     scope=set(ALL_SITES))}
    print("\n[%d] a_pos %.3f | logit full %.3f | empty: UP %.3f (den %.3f) / "
          "ALL %.3f (den %.3f)"
          % (sl, a_pos, m_full, m_empty["pos"], m_full - m_empty["pos"],
             m_empty["logit"], m_full - m_empty["logit"]), flush=True)

    common = dict(seed_layer=LAYER, seed_kind=KIND,
                  seed_latent_idx=sl, pos_tokens=pt, pos_argmax=pa,
                  neg_tokens=nt, mask_floor_source=cfg.mask_floor_source,
                  dual_floor_weight=cfg.dual_floor_weight,
                  binarize=cfg.binarize, steps=cfg.steps, lr=cfg.lr,
                  l1_lambda=cfg.l1_lambda, keep_threshold=cfg.keep_threshold,
                  batch_size=disc.probe_batch_size,
                  holdout_frac=cfg.holdout_frac, theta_init=cfg.theta_init,
                  log_every=0, deep_site_threshold=cfg.deep_site_threshold,
                  deep_batch_size=cfg.deep_batch_size, optimizer=cfg.optimizer,
                  weight_decay=cfg.weight_decay, code_dtype=cfg.code_dtype,
                  lr_schedule=cfg.lr_schedule, lr_min_frac=cfg.lr_min_frac,
                  warmup_frac=cfg.warmup_frac)

    for arm, lam in [(a, l) for a in ("pos", "logit") for l in LAMBDAS]:
        t0 = time.time()
        kw = dict(common)
        kw["l1_lambda"] = lam
        kw["sites"] = SCOPE[arm]
        if arm == "logit":
            kw["target_tokens"] = tgt
            # dual floor is pos-only by construction; the engine raises
            # otherwise, so the behavioural arm runs the single zero floor.
            kw["mask_floor_source"] = "zero"
            kw.pop("dual_floor_weight", None)
        try:
            scores, prov = run_learned_mask(inference, bank, objective=arm, **kw)
        except Exception as e:
            print("  %-6s l1=%-7g FAILED %s: %s"
                  % (arm, lam, type(e).__name__, e), flush=True)
            continue
        sc = set(SCOPE[arm])
        mem = sorted({(f.layer, f.kind, int(f.index)) for f in scores
                      if (f.layer, f.kind) in sc})
        keep = {}
        for l, k, i in mem:
            keep.setdefault((l, k), set()).add(i)
        # free0 always reads the seed through its OWN upstream scope, so it
        # is comparable across arms even though the logit arm's membership
        # spans the whole model — its non-upstream members simply cannot
        # affect the seed.
        keep_up = {s: v for s, v in keep.items() if s in set(UP)}
        n_up = sum(len(v) for v in keep_up.values())
        a_c = float(circuit_only_activation(inference, bank, keep_up, UP, pt,
                                            LAYER, KIND, sl, pos_argmax=pa,
                                            batch_size=EVAL_BS))
        mns = means_up if arm == "pos" else means_all
        m_c = logit_metric(keep, site_means=mns, scope=sc)
        den_log = m_full - m_empty[arm]
        row = {"latent": sl, "arm": arm, "l1": lam, "n": len(mem),
               "n_sites": len(SCOPE[arm]), "n_upstream_members": n_up,
               "pct_scope": round(100.0 * len(mem) / (len(SCOPE[arm]) * D_SAE), 4),
               "free0": round((a_c - a_e0) / den_lat, 4) if abs(den_lat) > 1e-9 else None,
               "logit_faith": round((m_c - m_empty[arm]) / den_log, 4) if abs(den_log) > 1e-9 else None,
               "m_circuit": round(m_c, 4), "m_full": round(m_full, 4),
               "m_empty": round(m_empty[arm], 4), "den_log": round(den_log, 4),
               "holdout": prov.get("holdout_data_loss"),
               "secs": round(time.time() - t0, 1)}
        fh_out.write(json.dumps(row) + "\n"); fh_out.flush()
        mh.write(json.dumps({"latent": sl, "arm": arm, "l1": lam,
                             "members": [[l, k, i] for l, k, i in mem]}) + "\n")
        mh.flush()
        print("  %-6s l1=%-7g n=%-7d (%d sites, %.3f%% scope, %d upstream) "
              "free0=%-8s logit_faith=%-8s  %.0fs"
              % (arm, lam, len(mem), len(SCOPE[arm]), row["pct_scope"], n_up,
                 row["free0"], row["logit_faith"], row["secs"]), flush=True)
        torch.cuda.empty_cache()
    print("  (seed %.0fs)" % (time.time() - t_seed), flush=True)

fh_out.close(); mh.close()
print("ALL DONE", flush=True)
