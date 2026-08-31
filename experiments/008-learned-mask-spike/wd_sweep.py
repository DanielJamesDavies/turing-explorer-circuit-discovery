"""Weight-decay sweep at fixed lambda: is AdamW's edge real or a crossover?

The Adam-vs-AdamW comparison at 400/0.05 showed a monotone trend in
(free0_AdamW - free0_Adam) across lambda: -0.149, -0.043, +0.007, +0.002,
-0.002. The single positive sits where the trend crosses zero, so it may be
an artifact rather than a benefit.

Mechanism under test: decoupled decay pulls theta -> 0, i.e. m -> 0.5, the
keep threshold. That HURTS sparsification (it drags L1-suppressed latents
back over the line) but could HELP by countering sigmoid saturation — a
latent at theta=+8 has gradient ~3e-4 and is effectively frozen, so decay
keeps membership revisable late in training.

  interior optimum in wd  -> the anti-saturation effect is real, keep AdamW
  monotone decline from 0 -> the +0.007 was the crossover, Adam is the default

CONTROL: AdamW at weight_decay=0 IS Adam. The wd=0 row must reproduce the
existing Adam/400/0.05/lambda=1e-4 row (57,268 members, free0_all 0.9751); if
it does not, the two runs differ for some other reason and the whole
comparison is void.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/wd_sweep.py
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
STEPS, LR = 400, 0.05
WDS = [float(x) for x in os.environ.get("WDS", "0.0,0.003,0.01,0.03,0.1").split(",")]
CONTROL = {"n": 57268, "free0_all": 0.9751}     # Adam/400/0.05 @ lambda 1e-4
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

a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind, LATENT,
                                      pa, batch_size=EVAL_BS))
a_e0 = float(circuit_only_activation(inference, bank, {}, up, pt, layer, kind,
                                     LATENT, pos_argmax=pa, batch_size=EVAL_BS))
den = a_pos - a_e0


def slice_free0(keep, tokens, anchors):
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


fh = (HERE / "wd_rows.jsonl").open("a")
print("[%s] AdamW wd sweep | lambda %g steps %d lr %g" % (TAG, LAMBDA, STEPS, LR),
      flush=True)
print("%-8s %10s %9s %9s %9s %8s %8s %7s"
      % ("wd", "n", "f0_all", "f0_train", "f0_hold", "mean_m", "m_kept", "secs"),
      flush=True)
for wd in WDS:
    t0 = time.time()
    scores, prov = run_learned_mask(
        inference, bank, objective="pos", sites=sorted(up),
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa,
        steps=STEPS, lr=LR, l1_lambda=LAMBDA, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size,
        optimizer="adamw", weight_decay=wd)
    keep = {}
    for fid in scores:
        keep.setdefault((fid.layer, fid.kind), set()).add(int(fid.index))
    n = len(scores)
    a_c = float(circuit_only_activation(inference, bank, keep, up, pt, layer,
                                        kind, LATENT, pos_argmax=pa,
                                        batch_size=EVAL_BS)) if n else a_e0
    f_all = round((a_c - a_e0) / den, 4) if abs(den) > 1e-9 else None
    n_tr = int(prov["n_train_pos"])
    row = {"tag": TAG, "lambda": LAMBDA, "steps": STEPS, "lr": LR,
           "optimizer": "adamw", "weight_decay": wd, "n": n,
           "free0_all_probes": f_all,
           "free0_train": slice_free0(keep, pt[:n_tr], pa[:n_tr]),
           "free0_holdout": slice_free0(keep, pt[n_tr:], pa[n_tr:]),
           "mean_m_final": prov["mean_m_final"],
           "mean_m_kept": prov.get("mean_m_kept"),
           "min_m_kept": prov.get("min_m_kept"),
           "holdout_data_loss": prov["holdout_data_loss"],
           "secs": round(time.time() - t0, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-8g %10s %9s %9s %9s %8.4f %8s %7.0f"
          % (wd, format(n, ","), f_all, row["free0_train"], row["free0_holdout"],
             prov["mean_m_final"], prov.get("mean_m_kept"), row["secs"]),
          flush=True)
    if wd == 0.0:
        dn = abs(n - CONTROL["n"]) / CONTROL["n"]
        df = abs((f_all or 0) - CONTROL["free0_all"])
        print("   CONTROL (AdamW wd=0 == Adam): n differs %.2f%%, free0 differs "
              "%.4f — %s" % (100 * dn, df,
                             "consistent" if dn < 0.02 and df < 0.02
                             else "MISMATCH, comparison void"), flush=True)
    torch.cuda.empty_cache()
fh.close()
