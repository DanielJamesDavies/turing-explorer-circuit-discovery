"""How should lambda be set PER SEED? Three calibration rules, head to head.

The problem: one lambda does not fit all depths. Dual needs ~1e-4 at L2 but
~3e-6 at L10 - a 33x span. Site count explains only ~4x of that (8 sites vs
32), so the rest is per-latent influence: at depth more latents share the
work, so each one's DATA gradient is smaller, while the L1 gradient per
latent is lambda*m*(1-m), independent of depth. The same lambda therefore
out-competes a weaker data signal at depth. (Same family of failure as the
mean-normalised L1 whose per-latent gradient fell under Adam's eps.)

Three rules:
  A grad-scaled  lambda = c * quantile(|dL_data/dtheta|) measured at init.
                 One extra backward pass, no search. c is anchored ONCE on the
                 known-good deep point (L10 ~ 3e-6) and then transferred, so
                 the test is whether ONE c works across depth.
  B faith-target bisect lambda for the LARGEST lambda (smallest circuit)
                 reaching free0 >= FREE0_TARGET. Matches the house convention
                 (magnitude_prune already bisects to a free0 target).
                 Calibrates on free0 and is then judged on the NEUTRAL
                 k-sparse metrics, so it is not tuned on its own scoreboard.
  C size-match   bisect lambda until n ~ the zero-floor arm's n. Simplest to
                 compare against, but assumes zero-floor's size is the right
                 size, which nothing establishes.

B and C SHARE the probe runs: one lambda -> (n, free0) map per seed serves
both targets, so the cost is 3 probes + 3 finals + 1 reference per seed
rather than two independent searches.

  PYTHONPATH=src python experiments/010-lambda-calibration/runner.py
"""
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.floors import collect_site_means
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
import circuit.instrument.learned_mask as lm

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ, EVAL_BS = 64, 16
PROBE_LAMBDAS = (1e-4, 1e-5, 1e-6)
FREE0_TARGET = 0.95
GAMMA = 0.25
ANCHOR_SEED, ANCHOR_LAMBDA = 10, 3e-6      # the known-good deep knee
WANT = [int(x) for x in os.environ.get("SEED_IDS", "2,10").split(",")]
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, bank.d_sae), device=bank.device)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.magnitude_prune = False
disc.recurrence_prune = False
disc.floor_negctx_mode = "random"
ncs = disc.neg_context_selection
ncs.preact_filter = True
ncs.preact_select = "cleanest"


def _strat(cands, n, seed=42):
    by = defaultdict(list)
    for i, c in enumerate(cands):
        by[int(c["comp_idx"]) // n_kinds].append(i)
    rng = random.Random(seed)
    for k in by:
        rng.shuffle(by[k])
    out = []
    for rank in range(max(len(v) for v in by.values())):
        for k in sorted(by):
            if rank < len(by[k]):
                out.append(by[k][rank])
                if len(out) >= n:
                    return out
    return out


_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu", weights_only=False)
_idx = _strat(_cand, 16)
SEEDS = [(i, int(_cand[_idx[i]]["comp_idx"]), int(_cand[_idx[i]]["latent_idx"]))
         for i in WANT]


def grad_scale(up, layer, kind, sl, pt, pa, q=0.99):
    """Per-latent |dL_data/dtheta| at init, one forward+backward.

    Read at a HIGH quantile, not the median: most latents are irrelevant and
    contribute ~0 gradient, so the median tracks the dictionary size rather
    than the signal. The latents that matter are the tail.
    """
    sae = bank.saes[kind][layer]
    w = sae.encoder.weight[sl].detach()
    b = sae._get_bias_eff()[sl].detach()
    thetas = {s: torch.full((bank.d_sae,), 4.0, device=bank.device,
                            requires_grad=True) for s in up}
    p = lm.LearnedMaskPatcher(bank, thetas, layer, kind, w, b,
                              code_dtype=disc.learned_mask.code_dtype)
    pre = lm._forward_preact(inference, p, pt[:4], grad=True)
    idx = torch.arange(pre.shape[0], device=pre.device)
    vals = pre[idx, pa[:4].to(pre.device)]
    tgt = vals.detach()          # gradient scale only; target value irrelevant
    ((vals - (tgt * 0.5)) ** 2).mean().backward()
    g = torch.cat([t.grad.abs().flatten() for t in thetas.values()])
    out = {"q99": float(g.quantile(0.99)), "q999": float(g.quantile(0.999)),
           "median": float(g.median()), "max": float(g.max())}
    for t in thetas.values():
        t.grad = None
    return out


def run_dual(up, layer, kind, sl, pt, pa, nt, lam):
    t0 = time.perf_counter()
    scores, prov = lm.run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
        mask_floor_source="dual", dual_floor_weight=GAMMA,
        steps=400, lr=0.05, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=disc.learned_mask.deep_site_threshold,
        deep_batch_size=disc.learned_mask.deep_batch_size,
        optimizer="adamw", weight_decay=0.05,
        code_dtype=disc.learned_mask.code_dtype)
    return scores, prov, time.perf_counter() - t0


def run_zero(up, layer, kind, sl, pt, pa, lam=1e-4):
    scores, prov = lm.run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
        pos_tokens=pt, pos_argmax=pa,
        steps=400, lr=0.05, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=disc.learned_mask.deep_site_threshold,
        deep_batch_size=disc.learned_mask.deep_batch_size,
        optimizer="adamw", weight_decay=0.05,
        code_dtype=disc.learned_mask.code_dtype)
    return scores, prov


rows = []
state = {}
for seed_i, sc_idx, sl in SEEDS:
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))
    meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                              avg_acts, probe_builder)
    pd_ = meth.build_probe_dataset(sc_idx, sl)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    selector = meth._neg_context_selector()
    ref = selector.posctx_reference(pt, sc_idx, sl,
                                    batch_size=int(ncs.filter_batch_size))
    sel = selector.select(sc_idx, sl, "random", max_sequences=N_SEQ,
                          batch_size=4, candidate_pool_size=ncs.candidate_pool_size,
                          exact=False, non_activation_threshold=0.0,
                          preact_filter=True, preact_select="cleanest",
                          preact_max_frac=0.25, posctx_reference=ref,
                          selection_seed=int(ncs.selection_seed),
                          filter_batch_size=int(ncs.filter_batch_size),
                          load_window_size=int(ncs.load_window_size), logger=None)
    nt_eval = sel.tokens[:N_SEQ]
    nt_floor = pd_.neg_tokens[:N_SEQ]

    a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind, sl,
                                          pa, batch_size=EVAL_BS))
    means_up = collect_site_means(inference, bank, pt, set(up))
    means_neg = collect_site_means(inference, bank, nt_eval, set(up))

    def empty(sm=None, tk=False):
        return float(circuit_only_activation(
            inference, bank, {}, up, pt, layer, kind, sl, pos_argmax=pa,
            site_means=sm, batch_size=EVAL_BS, respect_topk=tk))

    a_e0, a_eMT = empty(), empty(means_up, True)
    a_eNT = empty(means_neg, True)

    # FACTORY, not a closure over the loop variables. A `def metrics(...)`
    # written inline captures `up`/`layer`/`kind`/`sl`/`a_pos`/anchors BY
    # REFERENCE to this loop's scope, so every stored copy would read the LAST
    # seed's values once the loop ends - the finals below run after it. That
    # silently evaluated seed 2's circuits against seed 10's sites and
    # anchors, giving free0 == 0.0 across the board while seed 10's rows were
    # correct only by accident of being last. Binding through parameters makes
    # each seed's evaluator independent.
    def make_metrics(up, layer, kind, sl, pt, pa, a_pos, a_e0, a_eNT, a_eMT,
                     means_up, means_neg):
        def metrics(scores):
            keep = {}
            for f in scores:
                keep.setdefault((f.layer, f.kind), set()).add(int(f.index))
            n = len(scores)

            def phi(a_e, sm=None, tk=False):
                a_c = float(circuit_only_activation(
                    inference, bank, keep, up, pt, layer, kind, sl,
                    pos_argmax=pa, site_means=sm, batch_size=EVAL_BS,
                    respect_topk=tk)) if n else a_e
                d = a_pos - a_e
                return round((a_c - a_e) / d, 4) if abs(d) > 1e-9 else None
            return n, phi(a_e0), phi(a_eNT, means_neg, True), phi(a_eMT, means_up, True)
        return metrics

    metrics = make_metrics(up, layer, kind, sl, pt, pa, a_pos, a_e0, a_eNT,
                           a_eMT, means_up, means_neg)

    print("\n=== seed %d  L%d %s  | %d sites | a_pos %.3f ==="
          % (seed_i, layer, kind, len(up), a_pos), flush=True)

    gs = grad_scale(up, layer, kind, sl, pt, pa)
    print("  per-latent |dL/dtheta| at init: q99 %.3e  q999 %.3e  median %.3e"
          % (gs["q99"], gs["q999"], gs["median"]), flush=True)

    zs, _ = run_zero(up, layer, kind, sl, pt, pa)
    zn, zf0, zfN, zfM = metrics(zs)
    print("  zero-floor reference: n=%s free0=%s freeN_tk=%s freeM_tk=%s"
          % (format(zn, ","), zf0, zfN, zfM), flush=True)

    probes = []
    for lam in PROBE_LAMBDAS:
        sc, pv, secs = run_dual(up, layer, kind, sl, pt, pa, nt_floor, lam)
        n, f0, fN, fM = metrics(sc)
        probes.append({"lam": lam, "n": n, "free0": f0})
        print("  probe lam=%.0e -> n=%-8s free0=%-8s freeN_tk=%-8s (%.0fs)"
              % (lam, format(n, ","), f0, fN, secs), flush=True)
        del sc
        torch.cuda.empty_cache()

    state[seed_i] = dict(up=up, layer=layer, kind=kind, sl=sl, pt=pt, pa=pa,
                         nt=nt_floor, gs=gs, probes=probes, metrics=metrics,
                         zero=(zn, zf0, zfN, zfM), sites=len(up), a_pos=a_pos)
    rows.append({"seed_i": seed_i, "layer": layer, "kind": kind,
                 "sites": len(up), "a_pos": round(a_pos, 4), "grad_scale": gs,
                 "zero_floor": {"n": zn, "free0": zf0, "freeN_topk": zfN,
                                "freeM_topk": zfM},
                 "probes": probes})

# ---- anchor rule A on the deep seed, then transfer -------------------------
anchor = state.get(ANCHOR_SEED)
C = ANCHOR_LAMBDA / anchor["gs"]["q99"] if anchor else None
print("\nrule A anchored on seed %d: c = %.4e / %.4e = %.4e"
      % (ANCHOR_SEED, ANCHOR_LAMBDA, anchor["gs"]["q99"], C), flush=True)


def interp(probes, key, target):
    """Log-lambda interpolation between the bracketing probes."""
    pts = sorted([(p["lam"], p[key]) for p in probes if p[key] is not None])
    for (l1, v1), (l2, v2) in zip(pts, pts[1:]):
        if (v1 - target) * (v2 - target) <= 0 and v1 != v2:
            f = (target - v1) / (v2 - v1)
            return math.exp(math.log(l1) + f * (math.log(l2) - math.log(l1)))
    return None


print("\n%-6s %-14s %-11s %9s %9s %10s %10s"
      % ("seed", "rule", "lambda", "n", "free0", "freeN_tk", "freeM_tk"), flush=True)
for seed_i, st in state.items():
    lam_a = C * st["gs"]["q99"]
    lam_b = interp(st["probes"], "free0", FREE0_TARGET)
    lam_c = interp(st["probes"], "n", st["zero"][0])
    # NOTE on rule B: the 3-probe log-interpolation is coarse. At L10 it
    # returned 1.696e-6 (free0 0.9928, n 106,414) when 3e-6 already reaches
    # free0 0.9514 with 68,870 nodes - i.e. it overshot the "SMALLEST circuit
    # reaching the target" it is supposed to find. Reported as-is; a refinement
    # step would need another run per seed.
    for rule, lam in (("A grad-scaled", lam_a), ("B faith>=%.2f" % FREE0_TARGET, lam_b),
                      ("C size-match", lam_c)):
        if lam is None:
            print("%-6d %-14s %-11s  (target not bracketed by probes)"
                  % (seed_i, rule, "-"), flush=True)
            rows.append({"seed_i": seed_i, "rule": rule, "lambda": None})
            continue
        sc, pv, secs = run_dual(st["up"], st["layer"], st["kind"], st["sl"],
                                st["pt"], st["pa"], st["nt"], lam)
        n, f0, fN, fM = st["metrics"](sc)
        print("%-6d %-14s %-11.3e %9s %9s %10s %10s"
              % (seed_i, rule, lam, format(n, ","), f0, fN, fM), flush=True)
        rows.append({"seed_i": seed_i, "rule": rule, "lambda": lam, "n": n,
                     "free0": f0, "freeN_topk": fN, "freeM_topk": fM,
                     "secs": round(secs, 1)})
        del sc
        torch.cuda.empty_cache()

(HERE / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("\nwrote rows.jsonl")
print("Rule A is judged on TRANSFER: c was fitted on seed %d only." % ANCHOR_SEED)
print("Rule B calibrates on free0 and is judged on the NEUTRAL k-sparse metrics.")
