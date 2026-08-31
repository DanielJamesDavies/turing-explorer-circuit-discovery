"""Held-out free0 for abl-mask: does closure generalise to unseen probes?

The mask_negctx gate collapsed 0.44 -> 0.055 from train to held-out negatives.
abl-mask's holdout DATA LOSS tracked train closely, which I took as evidence
it generalises — but every free0 in the lambda sweeps was computed on ALL 64
probes, 48 of which the optimiser trained on. This measures free0 per slice.

  train    the probes the optimiser saw (provenance n_train_pos)
  holdout  the probes it did not (provenance n_holdout_pos)
  fresh    additional posctx sequences beyond the 64, if the store has them

Each slice gets its OWN anchors: a_pos(slice) and a_e0(slice), so free0 is
computed exactly as it would be if that slice were the whole probe set.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/abl_mask_heldout.py
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
LAMBDAS = [1e-3, 1e-4, 1e-5]
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

# Ask for MORE than N_SEQ so anything beyond the optimiser's probe set can
# serve as a fresh slice (the store may simply not have that many).
from store.context import mid_ctx, neg_ctx, top_ctx
pd_wide = probe_builder.build_for_latent(SC_IDX, LATENT, top_ctx, mid_ctx,
                                         neg_ctx, n_pos=128, n_neg=N_SEQ)
pt_all, pa_all = pd_wide.pos_tokens, pd_wide.pos_argmax
pt, pa = pt_all[:N_SEQ], pa_all[:N_SEQ]
pt_fresh, pa_fresh = pt_all[N_SEQ:], pa_all[N_SEQ:]
print("[%s] probe pool %d (optimiser sees %d, fresh %d)"
      % (TAG, pt_all.shape[0], pt.shape[0], pt_fresh.shape[0]), flush=True)


def free0_on(keep, tokens, anchors):
    """free0 with the slice's OWN anchors — as if it were the whole set."""
    if tokens.shape[0] == 0:
        return None, None, None
    a_pos = float(measure_seed_activation(inference, bank, tokens, layer, kind,
                                          LATENT, anchors, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, up, tokens, layer,
                                         kind, LATENT, pos_argmax=anchors,
                                         batch_size=EVAL_BS))
    a_c = float(circuit_only_activation(inference, bank, keep, up, tokens, layer,
                                        kind, LATENT, pos_argmax=anchors,
                                        batch_size=EVAL_BS))
    d = a_pos - a_e0
    return (round((a_c - a_e0) / d, 4) if abs(d) > 1e-9 else None,
            round(a_c, 4), round(a_pos, 4))


fh = (HERE / "abl_rows.jsonl").open("a")
print("%-9s %9s | %8s %8s | %8s %8s | %8s"
      % ("lambda", "n", "tr free0", "tr a_c", "ho free0", "ho a_c", "fr free0"),
      flush=True)
for lam in LAMBDAS:
    t0 = time.time()
    scores, prov = run_learned_mask(
        inference, bank, objective="pos", sites=sorted(up),
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa,
        steps=200, lr=0.1, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size)
    keep = {}
    for fid in scores:
        keep.setdefault((fid.layer, fid.kind), set()).add(int(fid.index))

    n_tr = int(prov["n_train_pos"])          # from provenance, never re-derived
    f_tr, ac_tr, ap_tr = free0_on(keep, pt[:n_tr], pa[:n_tr])
    f_ho, ac_ho, ap_ho = free0_on(keep, pt[n_tr:], pa[n_tr:])
    f_fr, ac_fr, ap_fr = free0_on(keep, pt_fresh, pa_fresh)

    row = {"tag": TAG, "lambda": lam, "n": len(scores),
           "n_train_pos": n_tr, "n_holdout_pos": int(prov["n_holdout_pos"]),
           "n_fresh_pos": int(pt_fresh.shape[0]),
           "free0_train": f_tr, "a_c_train": ac_tr, "a_pos_train": ap_tr,
           "free0_holdout": f_ho, "a_c_holdout": ac_ho, "a_pos_holdout": ap_ho,
           "free0_fresh": f_fr, "a_c_fresh": ac_fr, "a_pos_fresh": ap_fr,
           "holdout_data_loss": prov["holdout_data_loss"],
           "secs": round(time.time() - t0, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-9g %9s | %8s %8s | %8s %8s | %8s"
          % (lam, format(len(scores), ","), f_tr, ac_tr, f_ho, ac_ho,
             f_fr if f_fr is not None else "—"), flush=True)
    torch.cuda.empty_cache()
fh.close()
print("\nwrote %s" % (HERE / "abl_rows.jsonl"))
