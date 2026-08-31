"""Probe-count sweep: is the learned mask data-limited?

The sparse-circuit generalisation penalty measured on 64 probes (~12% train ->
held-out) is the signature of a data-limited fit. Unlike lr, probe count has
no coupled budget: steps, lr, lambda and wd all stay fixed, so more probes
means strictly more distinct data behind the same optimisation.

The headline metric is the GAP (free0_train - free0_holdout), not free0
itself: if the mask is overfitting its probes, the gap should shrink as the
probe pool grows. Circuit size and calibration are reported alongside because
more data may also change what the optimiser considers necessary.

Note on epochs: batch and steps are fixed, so 400 steps x batch 4 = 1600
sequence-visits regardless. With 48 training probes that is ~33 passes over
the data; with 192 it is ~8. So this sweep trades repetition for coverage,
which is exactly the trade the overfitting hypothesis predicts should help.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/probe_sweep.py
"""
import json
import os
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.instrument.learned_mask import run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
SEEDS = {"L2": (8, 30122), "L8": (25, 10628), "L10": (32, 3021)}
TAG = os.environ.get("SEED_TAG", "L8")
SC_IDX, LATENT = SEEDS[TAG]
LAMBDA = float(os.environ.get("LAMBDA", 1e-4))
STEPS = int(os.environ.get("STEPS", 400))
LR = float(os.environ.get("LR", 0.05))
WD = float(os.environ.get("WD", 0.05))
COUNTS = [int(x) for x in os.environ.get("COUNTS", "32,64,128,256").split(",")]
EVAL_BS, NK = 16, 3
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * NK, bank.d_sae), device=bank.device)

_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_batch_size = 4
layer, ki = split_component_idx(SC_IDX, NK)
kind = bank.kinds[ki]
up = sorted(upstream_sites(bank, layer, kind))

# One wide build; each sweep point takes a prefix. Requesting the largest
# count once means every point draws from the SAME ordering, so a smaller
# point is a strict subset of a larger one.
from store.context import mid_ctx, neg_ctx, top_ctx
wide = probe_builder.build_for_latent(SC_IDX, LATENT, top_ctx, mid_ctx, neg_ctx,
                                      n_pos=max(COUNTS), n_neg=64)
PT_ALL, PA_ALL = wide.pos_tokens, wide.pos_argmax
print("[%s] probe pool available: %d (requested %d)"
      % (TAG, PT_ALL.shape[0], max(COUNTS)), flush=True)


def slice_free0(keep, tokens, anchors):
    """free0 with the slice's OWN anchors."""
    if tokens.shape[0] == 0 or not keep:
        return None
    ap = float(measure_seed_activation(inference, bank, tokens, layer, kind,
                                       LATENT, anchors, batch_size=EVAL_BS))
    ae = float(circuit_only_activation(inference, bank, {}, up, tokens, layer,
                                       kind, LATENT, pos_argmax=anchors,
                                       batch_size=EVAL_BS))
    ac = float(circuit_only_activation(inference, bank, keep, up, tokens, layer,
                                       kind, LATENT, pos_argmax=anchors,
                                       batch_size=EVAL_BS))
    d = ap - ae
    return round((ac - ae) / d, 4) if abs(d) > 1e-9 else None


fh = (HERE / "probe_rows.jsonl").open("a")
print("lambda %g steps %d lr %g wd %g (product %.2f)"
      % (LAMBDA, STEPS, LR, WD, STEPS * LR * WD), flush=True)
print("%-8s %6s %6s %10s %9s %9s %8s %8s %7s"
      % ("n_probe", "train", "hold", "n", "f0_train", "f0_hold", "GAP",
         "m_kept", "secs"), flush=True)
for n_probe in COUNTS:
    if n_probe > PT_ALL.shape[0]:
        print("%-8d  -- pool has only %d, skipped --" % (n_probe, PT_ALL.shape[0]),
              flush=True)
        continue
    pt, pa = PT_ALL[:n_probe], PA_ALL[:n_probe]
    t0 = time.time()
    scores, prov = run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa,
        steps=STEPS, lr=LR, l1_lambda=LAMBDA, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size,
        optimizer="adamw", weight_decay=WD,
        code_dtype=config.discovery.learned_mask.code_dtype)
    keep = {}
    for fid in scores:
        keep.setdefault((fid.layer, fid.kind), set()).add(int(fid.index))
    n_tr = int(prov["n_train_pos"])
    f_tr = slice_free0(keep, pt[:n_tr], pa[:n_tr])
    f_ho = slice_free0(keep, pt[n_tr:], pa[n_tr:])
    gap = (round(f_tr - f_ho, 4) if (f_tr is not None and f_ho is not None)
           else None)
    row = {"tag": TAG, "n_probe": n_probe, "lambda": LAMBDA, "steps": STEPS,
           "lr": LR, "weight_decay": WD, "n": len(scores),
           "n_train_pos": n_tr, "n_holdout_pos": int(prov["n_holdout_pos"]),
           "epochs_over_train": round(STEPS * 4 / max(n_tr, 1), 2),
           "free0_train": f_tr, "free0_holdout": f_ho, "gap": gap,
           "mean_m_kept": prov.get("mean_m_kept"),
           "holdout_data_loss": prov["holdout_data_loss"],
           "loss_final": prov["loss_final"],
           "secs": round(time.time() - t0, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-8d %6d %6d %10s %9s %9s %8s %8s %7.0f"
          % (n_probe, n_tr, int(prov["n_holdout_pos"]), format(len(scores), ","),
             f_tr, f_ho, gap, prov.get("mean_m_kept"), row["secs"]), flush=True)
    torch.cuda.empty_cache()
fh.close()
