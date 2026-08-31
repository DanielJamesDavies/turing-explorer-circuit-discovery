"""Do the existing post-hoc prunes shrink a MASK circuit further?

The mask already selects by optimisation, so rec2/mag are being asked to
improve on a set that was chosen jointly rather than by ranking. Two things
under test:

  1. can rec2 (fires in >= 2 sequences) and mag (free0 bisection) cut the
     mask's 82k L8 circuit without losing closure?
  2. does the mask's m value work as a RANKING for magnitude bisection?
     m is compressed into (0.5, 1.0] by the keep threshold, so its ordering
     may carry much less information than attribution scores do.

free0 is reported per probe slice (train / held-out), since the mask's own
optimisation saw only the train slice.

  SEED_TAG=L8 LAMBDA=1e-4 PYTHONPATH=src python experiments/008-learned-mask-spike/mask_prune_forks.py
"""
import json
import os
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.instrument.learned_mask import run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from circuit.types.feature_id import FeatureID
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.magnitude_prune import prune_by_magnitude_bisection
from eval.recurrence_prune import prune_by_sequence_recurrence
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from store.circuits import Circuit, CircuitNode

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
SEEDS = {"L2": (8, 30122), "L8": (25, 10628), "L10": (32, 3021)}
TAG = os.environ.get("SEED_TAG", "L8")
SC_IDX, LATENT = SEEDS[TAG]
LAMBDA = float(os.environ.get("LAMBDA", 1e-4))
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
nt = pd_.neg_tokens[:N_SEQ]

print("[%s] mask lambda %g" % (TAG, LAMBDA), flush=True)
t0 = time.time()
scores, prov = run_learned_mask(
    inference, bank, objective="pos", sites=sorted(up),
    seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
    pos_tokens=pt, pos_argmax=pa, steps=200, lr=0.1, l1_lambda=LAMBDA,
    keep_threshold=0.5, batch_size=4, holdout_frac=0.25, log_every=0,
    deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
    deep_batch_size=config.discovery.learned_mask.deep_batch_size)
n_tr = int(prov["n_train_pos"])
ms = torch.tensor(list(scores.values()))
print("mask: %s members in %.0fs | m range %.4f..%.4f (median %.4f)"
      % (format(len(scores), ","), time.time() - t0, float(ms.min()),
         float(ms.max()), float(ms.median())), flush=True)


def build_circuit():
    c = Circuit(name="mask")
    c.add_node(CircuitNode(metadata={
        "feature_id": FeatureID(layer, kind, LATENT), "role": "seed"}))
    for fid, m in scores.items():
        c.add_node(CircuitNode(metadata={
            "feature_id": fid, "role": "ablation_support",
            "attribution_score": float(m)}))
    return c


def keep_of(c):
    keep = {}
    for node in c.nodes.values():
        f = node.feature_id
        if node.metadata.get("role") == "seed" or f is None:
            continue
        if (f.layer, f.kind) in up:
            keep.setdefault((f.layer, f.kind), set()).add(int(f.index))
    return keep


def free0_on(keep, tokens, anchors):
    if tokens.shape[0] == 0 or not keep:
        return None
    a_pos = float(measure_seed_activation(inference, bank, tokens, layer, kind,
                                          LATENT, anchors, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, up, tokens, layer,
                                         kind, LATENT, pos_argmax=anchors,
                                         batch_size=EVAL_BS))
    a_c = float(circuit_only_activation(inference, bank, keep, up, tokens, layer,
                                        kind, LATENT, pos_argmax=anchors,
                                        batch_size=EVAL_BS))
    d = a_pos - a_e0
    return round((a_c - a_e0) / d, 4) if abs(d) > 1e-9 else None


FORKS = ["raw", "+rec2", "+mag", "+rec2+mag"]
fh = (HERE / "mask_prune_rows.jsonl").open("a")
print("\n%-12s %10s %8s | %10s %10s | %7s"
      % ("fork", "n", "kept%", "free0 tr", "free0 ho", "secs"), flush=True)
for fork in FORKS:
    t0 = time.time()
    c = build_circuit()
    # rec2 judges members on how many probe sequences they fire in; mag
    # bisects the |score| ranking against free0 — here that ranking is the
    # mask's m, which the keep threshold has compressed into (0.5, 1].
    if "rec2" in fork:
        prune_by_sequence_recurrence(inference, bank, c, pos_tokens=pt,
                                     neg_tokens=nt, min_sequences=2)
    if "mag" in fork:
        prune_by_magnitude_bisection(inference, bank, c, pos_tokens=pt,
                                     seed_layer=layer, seed_kind=kind,
                                     seed_latent_idx=LATENT, pos_argmax=pa,
                                     objective="free")
    keep = keep_of(c)
    n = sum(len(v) for v in keep.values())
    row = {"tag": TAG, "lambda": LAMBDA, "fork": fork, "n": n,
           "kept_pct": round(100.0 * n / max(len(scores), 1), 2),
           "free0_train": free0_on(keep, pt[:n_tr], pa[:n_tr]),
           "free0_holdout": free0_on(keep, pt[n_tr:], pa[n_tr:]),
           "n_mask": len(scores), "secs": round(time.time() - t0, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-12s %10s %7.1f%% | %10s %10s | %7.0f"
          % (fork, format(n, ","), row["kept_pct"], row["free0_train"],
             row["free0_holdout"], row["secs"]), flush=True)
    del c
    torch.cuda.empty_cache()
fh.close()
