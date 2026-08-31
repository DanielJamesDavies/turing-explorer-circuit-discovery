"""Lambda sweep: the learned mask's size/faithfulness Pareto curve on one seed.

lambda is a per-latent price (penalty is a sum), so sweeping it traces the
whole curve in independent runs. Each point: run the engine directly (skip
assembly — we want scores + provenance incl. the holdout loss), evaluate
free0 of the kept set, log everything.

Reference curve: attribution top-K free0 at the same seed (direct-drivers
run) — the one-shot ranking the mask must beat at matched size.

  SEED_TAG=L2 PYTHONPATH=src python experiments/008-learned-mask-spike/lambda_sweep.py
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
TAG = os.environ.get("SEED_TAG", "L2")
SC_IDX, LATENT = SEEDS[TAG]
LAMBDAS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
STEPS = int(os.environ.get("STEPS", 200))
LR = float(os.environ.get("LR", 0.1))
OPT = os.environ.get("OPT", "adam")
WD = float(os.environ.get("WD", 0.01 if os.environ.get("OPT") == "adamw" else 0.0))
N_SEQ, EVAL_BS, NK = 64, 16, 3
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
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.probe_batch_size = 4

layer, ki = split_component_idx(SC_IDX, NK)
kind = bank.kinds[ki]
up = upstream_sites(bank, layer, kind)

m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                        avg_acts, probe_builder)
pd_ = m0.build_probe_dataset(SC_IDX, LATENT)
pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]

a_pos = measure_seed_activation(inference, bank, pt, layer, kind, LATENT, pa,
                                batch_size=EVAL_BS)
a_e0 = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, LATENT,
                               pos_argmax=pa, batch_size=EVAL_BS)
den = float(a_pos) - float(a_e0)

fh = (HERE / "lambda_rows.jsonl").open("a")
print("[%s] a_pos %.4f | sweeping lambda over %s" % (TAG, a_pos, LAMBDAS), flush=True)
print("%-10s %10s %9s %9s %9s %12s %8s"
      % ("lambda", "n", "f0_all", "f0_train", "f0_hold", "holdout_loss", "secs"),
      flush=True)
for lam in LAMBDAS:
    t0 = time.time()
    scores, prov = run_learned_mask(
        inference, bank, objective="pos", sites=sorted(up),
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa,
        steps=STEPS, lr=LR, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size,
        optimizer=OPT, weight_decay=WD)
    keep = {}
    for fid in scores:
        keep.setdefault((fid.layer, fid.kind), set()).add(int(fid.index))
    n = len(scores)
    if n:
        a_c = circuit_only_activation(inference, bank, keep, up, pt, layer,
                                      kind, LATENT, pos_argmax=pa,
                                      batch_size=EVAL_BS)
        free0 = round((float(a_c) - float(a_e0)) / den, 4)
    else:
        free0 = 0.0

    # Per-slice free0 with each slice's OWN anchors. The all-probe number
    # above mixes the 75% the optimiser trained on with the 25% it did not,
    # and mixing an all-probe figure with a held-out one across runs is a
    # real error mode (it happened, 2026-07-25) — so every row now carries
    # both, and the split comes from provenance rather than being re-derived.
    def _slice_free0(tokens, anchors):
        if tokens.shape[0] == 0 or not keep:
            return None
        ap = float(measure_seed_activation(inference, bank, tokens, layer, kind,
                                           LATENT, anchors, batch_size=EVAL_BS))
        ae = float(circuit_only_activation(inference, bank, {}, up, tokens,
                                           layer, kind, LATENT,
                                           pos_argmax=anchors, batch_size=EVAL_BS))
        ac = float(circuit_only_activation(inference, bank, keep, up, tokens,
                                           layer, kind, LATENT,
                                           pos_argmax=anchors, batch_size=EVAL_BS))
        d = ap - ae
        return round((ac - ae) / d, 4) if abs(d) > 1e-9 else None

    n_tr = int(prov["n_train_pos"])
    free0_tr = _slice_free0(pt[:n_tr], pa[:n_tr])
    free0_ho = _slice_free0(pt[n_tr:], pa[n_tr:])
    secs = time.time() - t0
    row = {"tag": TAG, "seed": "%d/%d" % (SC_IDX, LATENT), "lambda": lam,
           "steps": STEPS, "lr": LR, "optimizer": OPT, "weight_decay": WD,
           "n": n, "free0_all_probes": free0,
           "free0_train": free0_tr, "free0_holdout": free0_ho,
           "n_train_pos": int(prov["n_train_pos"]),
           "n_holdout_pos": int(prov["n_holdout_pos"]),
           "loss_final": prov["loss_final"],
           "holdout_data_loss": prov["holdout_data_loss"],
           "mean_m_final": prov["mean_m_final"], "secs": round(secs, 1),
           "a_pos": round(float(a_pos), 4)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-10g %10s %9s %9s %9s %12s %8.0f"
          % (lam, format(n, ","), free0, free0_tr, free0_ho,
             ("%.4f" % prov["holdout_data_loss"]
              if prov["holdout_data_loss"] is not None else "—"), secs),
          flush=True)
    torch.cuda.empty_cache()
fh.close()
