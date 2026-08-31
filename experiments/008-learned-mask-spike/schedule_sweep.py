"""Does an lr SCHEDULE beat a constant lr at matched budget?

Both budgets scale with sum(lr) — sparsity is sum(lr)*lambda, decay is
sum(lr)*wd — so a cosine/linear decay to zero halves sum(lr) for the same
peak. Peak lr is therefore DOUBLED (0.05 -> 0.1) so every arm sees
sum(lr) = 20, identical lambda and wd budgets, and the only difference is
the SHAPE of the schedule.

Rationale for expecting a difference: membership is a threshold crossing at
m = 0.5, so with constant lr a latent oscillating near the boundary has its
inclusion decided by wherever the final step left it. Decay freezes
membership progressively instead.

lr_min_frac = 0: at lr = 0 the AdamW update is exactly zero (both the
gradient and the decay term scale with lr), so a zero floor costs nothing
and makes the budget match exact.

Constant is RE-RUN here rather than reused from earlier sweeps: the code has
changed since (bf16 codes, empty_cache), so old rows are not comparable.

  PYTHONPATH=src python experiments/008-learned-mask-spike/schedule_sweep.py
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
LAMBDA, STEPS = 1e-4, 400
# OPT=adam sets wd=0, removing AdamW's DECOUPLED decay. That decay is
# multiplicative — theta_final = theta_0 * prod(1 - lr_t*wd) — and by AM-GM
# that product depends on the SHAPE of {lr_t}, not just on sum(lr)*wd, so it
# is the prime suspect for why matched budgets did not preserve circuit size.
# If the size gap between schedules collapses under plain Adam, decoupled
# decay is the cause; if it persists, the data/L1 interaction is.
OPT = os.environ.get("OPT", "adamw")
WD = 0.05 if OPT == "adamw" else 0.0
# (label, lr_peak, schedule) — every arm gives sum(lr) = 20, so schedule
# SHAPE is the only variable. ARMS=warmup runs the mirror-image set: decay
# made circuits 11-28% BIGGER, and the explanation (pruning is a slow
# threshold crossing, so late lr is what shrinks a circuit) predicts warmup
# should prune HARDER than constant. Constant is repeated in both sets as the
# shared reference point.
_REF = dict(lr=0.05, lr_schedule="constant", lr_min_frac=0.0, warmup_frac=0.0,
            l1_lambda=LAMBDA, weight_decay=None)     # None -> WD from OPT
# peak 0.3 / floor 0.01 / 10% warmup then cosine: sum(lr) = 62.1, i.e. 3.11x
# the reference budget. Run it BOTH raw and budget-matched — otherwise "the
# schedule helped" is indistinguishable from "it got 3x more optimisation".
_WARMCOS = dict(lr=0.30, lr_schedule="cosine", lr_min_frac=0.01 / 0.30,
                warmup_frac=0.1)
_SETS = {
    "decay": [("constant", _REF),
              ("cosine",   dict(_REF, lr=0.10, lr_schedule="cosine")),
              ("linear",   dict(_REF, lr=0.10, lr_schedule="linear"))],
    "warmup": [("constant",  _REF),
               ("cosine_up", dict(_REF, lr=0.10, lr_schedule="cosine_up")),
               ("linear_up", dict(_REF, lr=0.10, lr_schedule="linear_up"))],
    # The decisive isolation: same HOT budget as warmcos_raw (sum(lr)=62.15)
    # but a FLAT schedule. warmcos_raw beat constant while warmcos_match (hot
    # shape, reference budget) lost to it, which points at budget rather than
    # shape as the active ingredient. If hotflat reproduces warmcos_raw, shape
    # is confirmed irrelevant and lr is just a budget dial.
    "hotflat": [("constant", _REF),
                ("hotflat", dict(_REF, lr=62.145 / 400))],
    # THE compression sweep. Independent variable is lambda, NOT lr: lr scales
    # the data gradient, the L1 term and the decay together, so it cannot say
    # which one compresses. The L1 term is the only sparsifier here (decoupled
    # decay pulls theta->0 i.e. m->0.5, regularising CONFIDENCE, which is why
    # wd moves m_kept and lambda moves n). At fixed hotflat lr, sum(lr) stays
    # 62.145 and the decay budget stays pinned at 3.11, so only sparsity moves.
    # Plot against the scale-free sparsity budget sum(lr)*lambda.
    "lam": [("lam=%.0e" % L, dict(_REF, lr=62.145 / 400, l1_lambda=L))
            for L in (1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3)],
    "hot": [("constant", _REF),
            # same schedule, two budget treatments
            ("warmcos_raw", dict(_REF, **_WARMCOS)),
            ("warmcos_match", dict(_REF, **_WARMCOS,
                                   l1_lambda=0.002 / 62.145,
                                   weight_decay=1.0 / 62.145))],
}
ARMS = _SETS[os.environ.get("ARMS", "decay")]
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
up = sorted(upstream_sites(bank, layer, kind))

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


fh = (HERE / "schedule_rows.jsonl").open("a")
print("[%s] %s set | steps %d | optimizer %s"
      % (TAG, os.environ.get("ARMS", "decay"), STEPS, OPT), flush=True)
print("%-14s %6s %6s %7s %6s %8s %9s %9s %9s %8s %6s"
      % ("arm", "lr_pk", "lr_min", "sum_lr", "wd", "lambda", "f0_all",
         "f0_train", "f0_hold", "m_kept", "secs"), flush=True)
for label, spec in ARMS:
    t0 = time.time()
    wd_arm = WD if spec["weight_decay"] is None else spec["weight_decay"]
    scores, prov = run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa,
        steps=STEPS, lr=spec["lr"], l1_lambda=spec["l1_lambda"],
        keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size,
        optimizer=OPT, weight_decay=wd_arm,
        code_dtype=config.discovery.learned_mask.code_dtype,
        lr_schedule=spec["lr_schedule"], lr_min_frac=spec["lr_min_frac"],
        warmup_frac=spec["warmup_frac"])
    keep = {}
    for fid in scores:
        keep.setdefault((fid.layer, fid.kind), set()).add(int(fid.index))
    n = len(scores)
    a_c = float(circuit_only_activation(inference, bank, keep, up, pt, layer,
                                        kind, LATENT, pos_argmax=pa,
                                        batch_size=EVAL_BS)) if n else a_e0
    f_all = round((a_c - a_e0) / den, 4) if abs(den) > 1e-9 else None
    n_tr = int(prov["n_train_pos"])
    row = {"tag": TAG, "arm": label, "optimizer": OPT,
           "schedule": spec["lr_schedule"], "lr_peak": spec["lr"],
           "lr_floor": prov["lr_floor"], "warmup_frac": spec["warmup_frac"],
           "warmup_steps": prov["warmup_steps"], "steps": STEPS,
           "lambda": spec["l1_lambda"], "weight_decay": wd_arm,
           "lr_sum": prov["lr_sum"],
           "decay_product": prov["decay_product"],
           "sparsity_product": prov["sparsity_product"],
           "n": n, "free0_all_probes": f_all,
           "free0_train": slice_free0(keep, pt[:n_tr], pa[:n_tr]),
           "free0_holdout": slice_free0(keep, pt[n_tr:], pa[n_tr:]),
           "mean_m_kept": prov.get("mean_m_kept"),
           "holdout_data_loss": prov["holdout_data_loss"],
           "loss_final": prov["loss_final"],
           "secs": round(time.time() - t0, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-14s %6g %6g %7.2f %6.4f %8.2e %9s %9s %9s %8s %6.0f"
          % (label, spec["lr"], prov["lr_floor"], prov["lr_sum"], wd_arm,
             spec["l1_lambda"], f_all, row["free0_train"],
             row["free0_holdout"], prov.get("mean_m_kept"), row["secs"]),
          flush=True)
    print("               n = %s" % format(n, ","), flush=True)
    torch.cuda.empty_cache()
fh.close()
