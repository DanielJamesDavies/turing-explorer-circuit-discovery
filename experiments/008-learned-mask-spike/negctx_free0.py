"""free0 of the mask_negctx GATE sets, measured rather than asserted.

The gate latents are suppressive-role edits; evaluated as a free0 keep-set on
posctx (keep only them, zero everything else) the expectation from the
inhibitor-only knockouts is ~0 — but that was an analogy, not a measurement.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/negctx_free0.py
"""
import json
import os
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

m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                        avg_acts, probe_builder)
pd_ = m0.build_probe_dataset(SC_IDX, LATENT)
pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
nt = pd_.neg_tokens[:N_SEQ]

a_pos = measure_seed_activation(inference, bank, pt, layer, kind, LATENT, pa,
                                batch_size=EVAL_BS)
a_e0 = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, LATENT,
                               pos_argmax=pa, batch_size=EVAL_BS)
den = float(a_pos) - float(a_e0)

print("[%s] free0 of the mask_negctx gate sets (keep = gate, posctx)" % TAG,
      flush=True)
print("%-10s %10s %10s %12s" % ("lambda", "n_edits", "free0", "preact"), flush=True)
fh = (HERE / "negctx_free0_rows.jsonl").open("a")
for lam in LAMBDAS:
    scores, prov = run_learned_mask(
        inference, bank, objective="negctx", sites=sorted(up),
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt, target_act=float(a_pos),
        steps=200, lr=0.1, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size)
    keep = {}
    for fid in scores:
        keep.setdefault((fid.layer, fid.kind), set()).add(int(fid.index))
    n = len(scores)
    a_c = circuit_only_activation(inference, bank, keep, up, pt, layer, kind,
                                  LATENT, pos_argmax=pa, batch_size=EVAL_BS)
    p_c = circuit_only_activation(inference, bank, keep, up, pt, layer, kind,
                                  LATENT, pos_argmax=pa, batch_size=EVAL_BS,
                                  preact=True)
    free0 = round((float(a_c) - float(a_e0)) / den, 4) if abs(den) > 1e-9 else None
    row = {"tag": TAG, "lambda": lam, "n_edits": n, "free0": free0,
           "a_c": round(float(a_c), 4), "p_c": round(float(p_c), 4),
           "a_pos": round(float(a_pos), 4)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-10g %10s %10s %12.4f" % (lam, format(n, ","), free0, p_c), flush=True)
    torch.cuda.empty_cache()
fh.close()
