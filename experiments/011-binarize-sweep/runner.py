"""SWEEP 0: training-time binarisation - none vs ste vs anneal.

Motivation (keep_threshold sweep, 2026-07-30): the soft mask converges to
genuinely FRACTIONAL members - 79% of L8's membership has m in (0.5, 0.9),
and 8%/25%/20% of membership (L2/L8/L10) sits within +-0.05 of the cut - and
any post-hoc binarisation is lossy, asymmetrically (raising the cut is
catastrophic, lowering it helps). ste/anneal make TRAINING see the binary
semantics the evals execute (the TopK-SAE property, adapted to a global
membership that still needs gradients for non-members).

PRE-REGISTERED PREDICTIONS:
  1. ste/anneal collapse the near-cut mass to ~0 (the gate is binary or
     near-binary by the end, so theta has no reason to sit at the boundary).
  2. At MATCHED n, ste/anneal beat none on the binary evals.
  If (2) fails, fractional membership is load-bearing structure - not just
  lasso shrinkage - and survival-pressure reporting becomes the primary
  framing rather than harder training.
  Risk watch: STE's biased gradient can oscillate members at the boundary -
  per-step membership flip counts are recorded.

Protocol per seed: none @ lambda=1e-5 (baseline, defines n_target) ->
per-mode probe @ 1e-5 -> per-mode run at
lambda = 1e-5 * (n_probe/n_target)^(1/0.759). The exponent is a soft-gate
quantity, so per-mode sizes are checked in the output rather than trusted;
if the probe already lands within 2% of target, it doubles as the final.

  PYTHONPATH=src python experiments/011-binarize-sweep/runner.py
"""
import json
import time
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
N_SEQ, EVAL_BS, LAMBDA, EXP = 64, 16, 1e-5, 0.759
TARGETS = [(8, 17043, "L2-resid"), (25, 4085, "L8-mlp"), (32, 36965, "L10-resid")]
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
lmcfg = disc.learned_mask

# ---------------------------------------------------------------------------
# Optimiser hook: final-m capture + per-step membership flip counts.
# ---------------------------------------------------------------------------
_orig_step = torch.optim.AdamW.step
cap = {"i": 0, "total": 0, "sites": None, "m": None, "prev": None, "flips": []}


def _step(self, *a, **kw):
    out = _orig_step(self, *a, **kw)
    cap["i"] += 1
    params = self.param_groups[0]["params"]
    hard = torch.cat([(torch.sigmoid(p.detach()) > 0.5).flatten()
                      for p in params])
    if cap["prev"] is not None:
        cap["flips"].append(int((hard != cap["prev"]).sum()))
    cap["prev"] = hard
    if cap["i"] == cap["total"]:
        cap["m"] = {site: torch.sigmoid(p.detach()).float().cpu()
                    for site, p in zip(cap["sites"], params)}
    return out


torch.optim.AdamW.step = _step


def train(up, layer, kind, sl, pt, pa, nt, lam, mode):
    cap["i"] = 0
    cap["total"] = int(lmcfg.steps)
    cap["sites"] = up
    cap["m"] = None
    cap["prev"] = None
    cap["flips"] = []
    t0 = time.perf_counter()
    scores, prov = lm.run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
        mask_floor_source=lmcfg.mask_floor_source,
        dual_floor_weight=lmcfg.dual_floor_weight,
        steps=lmcfg.steps, lr=lmcfg.lr, l1_lambda=lam,
        keep_threshold=lmcfg.keep_threshold,
        batch_size=4, holdout_frac=lmcfg.holdout_frac, log_every=0,
        deep_site_threshold=lmcfg.deep_site_threshold,
        deep_batch_size=lmcfg.deep_batch_size,
        optimizer=lmcfg.optimizer, weight_decay=lmcfg.weight_decay,
        code_dtype=lmcfg.code_dtype, binarize=mode)
    secs = time.perf_counter() - t0
    flips = cap["flips"]
    tail = flips[len(flips) // 2:]
    flip_stats = {"mean_tail": (sum(tail) / max(len(tail), 1)),
                  "max": max(flips) if flips else 0,
                  "last50_mean": (sum(flips[-50:]) / max(len(flips[-50:]), 1))}
    all_m = torch.cat([v for v in cap["m"].values()])
    near = int(((all_m >= 0.45) & (all_m < 0.55)).sum())
    kept = int((all_m > 0.5).sum())
    hist = {"0.45-0.55": near, "0.55-0.90": int(((all_m >= 0.55) & (all_m < 0.9)).sum()),
            "0.90-1.00": int((all_m >= 0.9).sum())}
    return scores, secs, flip_stats, near, kept, hist


OUT = HERE / "rows.jsonl"
fh = OUT.open("a")
for comp, latent, label in TARGETS:
    layer, ki = split_component_idx(comp, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))
    meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                              avg_acts, probe_builder)
    pd_ = meth.build_probe_dataset(comp, latent)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind,
                                          latent, pa, batch_size=EVAL_BS))
    means_up = collect_site_means(inference, bank, pt, set(up))
    means_neg = collect_site_means(inference, bank, nt, set(up))

    def empty(sm=None, tk=False):
        return float(circuit_only_activation(
            inference, bank, {}, up, pt, layer, kind, latent, pos_argmax=pa,
            site_means=sm, batch_size=EVAL_BS, respect_topk=tk))

    a_e0, a_eMT, a_eNT = empty(), empty(means_up, True), empty(means_neg, True)

    def metrics(scores):
        keep = {}
        for f in scores:
            keep.setdefault((f.layer, f.kind), set()).add(int(f.index))
        n = sum(len(v) for v in keep.values())

        def phi(a_e, sm=None, tk=False):
            a_c = float(circuit_only_activation(
                inference, bank, keep, up, pt, layer, kind, latent,
                pos_argmax=pa, site_means=sm, batch_size=EVAL_BS,
                respect_topk=tk)) if n else a_e
            d = a_pos - a_e
            return round((a_c - a_e) / d, 4) if abs(d) > 1e-9 else None
        return n, phi(a_e0), phi(a_eNT, means_neg, True), phi(a_eMT, means_up, True)

    print("\n=== %s (comp %d latent %d, %d sites) ===" % (label, comp, latent,
                                                          len(up)), flush=True)
    n_target = None
    for mode in ("none", "ste", "anneal"):
        # probe at the anchor lambda
        sc, secs, fst, near, kept, hist = train(up, layer, kind, latent,
                                                pt, pa, nt, LAMBDA, mode)
        n_p, f0, fN, fM = metrics(sc)
        del sc
        torch.cuda.empty_cache()
        if mode == "none":
            n_target = n_p
            lam_used, stage = LAMBDA, "baseline"
            n, f0f, fNf, fMf = n_p, f0, fN, fM
        else:
            lam_used = LAMBDA * (n_p / max(n_target, 1)) ** (1.0 / EXP)
            if 0.98 <= (n_p / max(n_target, 1)) <= 1.02:
                stage = "probe==matched"
                n, f0f, fNf, fMf = n_p, f0, fN, fM
            else:
                stage = "matched"
                sc, secs, fst, near, kept, hist = train(
                    up, layer, kind, latent, pt, pa, nt, lam_used, mode)
                n, f0f, fNf, fMf = metrics(sc)
                del sc
                torch.cuda.empty_cache()
        row = {"comp_idx": comp, "latent": latent, "label": label,
               "mode": mode, "stage": stage, "lambda": lam_used,
               "n_probe": n_p, "n": n, "n_target": n_target,
               "free0": f0f, "freeN_topk": fNf, "freeM_topk": fMf,
               "near_cut": near, "kept": kept,
               "near_cut_pct_of_kept": round(100.0 * near / max(kept, 1), 2),
               "m_hist_members": hist, "flips": fst, "secs": round(secs, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-7s %-15s lam=%.3e n=%-9s free0=%-8s freeN_tk=%-8s "
              "freeM_tk=%-8s | near-cut %s (%.1f%% of kept) | flips tail-mean "
              "%.0f last50 %.0f"
              % (mode, stage, lam_used, format(n, ","), f0f, fNf, fMf,
                 format(near, ","), row["near_cut_pct_of_kept"],
                 fst["mean_tail"], fst["last50_mean"]), flush=True)
fh.close()
torch.optim.AdamW.step = _orig_step
print("\nwrote rows.jsonl")
print("Predictions: near-cut collapses under ste/anneal; binary evals improve")
print("at matched n. If they LOSE at matched n, fractional membership is")
print("load-bearing structure, not shrinkage.")
