"""Lambda sweep for cf-mask_inject on one seed: the full learned heir of the
original counterfactual question, with its built-in decomposition.

Per point, from provenance (train negatives, pre-activation units):
  p_gate_only    recovery from the learned edits alone (deltas off)
  p_inject_only  recovery from the learned injection alone (mask natural)
  p_both         the joint intervention (the trained state)
plus n split by role and the recovery fractions vs (target - p_neg_nat).

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/inject_sweep.py
"""
import json
import os
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.instrument.learned_mask import LearnedMaskPatcher, run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import measure_seed_activation, upstream_sites
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
LAMBDAS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
N_SEQ, NK = 64, 3
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

sae = bank.saes[kind][layer]
w_seed = sae.encoder.weight[LATENT].detach()
b_seed = sae._get_bias_eff()[LATENT].detach()
a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind, LATENT,
                                      pa, batch_size=16))
_p = LearnedMaskPatcher(bank, {}, layer, kind, w_seed, b_seed)
inference.disable_compile()
try:
    inference.forward(nt, patcher=_p, grad_enabled=False,
                      return_activations=False, tokenize_final=False)
finally:
    inference.enable_compile()
pre_nat = _p.seed_pre.detach()
anchors = pre_nat.argmax(dim=-1)
p_neg_nat = float(pre_nat[torch.arange(pre_nat.shape[0], device=pre_nat.device),
                          anchors].mean())
gden = a_pos - p_neg_nat
print("[%s] target %.4f | natural neg preact %.4f" % (TAG, a_pos, p_neg_nat),
      flush=True)
print("%-9s %8s %8s | %9s %9s %9s | %8s %8s %8s"
      % ("lambda", "n_inj", "n_edit", "p_gate", "p_inject", "p_both",
         "rec_gate", "rec_inj", "rec_both"), flush=True)

fh = (HERE / "inject_rows.jsonl").open("a")
for lam in LAMBDAS:
    t0 = time.time()
    scores, prov = run_learned_mask(
        inference, bank, objective="inject", sites=sorted(up),
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt, target_act=a_pos,
        steps=200, lr=0.1, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size)
    n_inj = sum(1 for v in scores.values() if v > 0)
    n_edit = sum(1 for v in scores.values() if v < 0)
    rec = lambda p: round((p - p_neg_nat) / gden, 4) if abs(gden) > 1e-9 else None
    row = {"tag": TAG, "lambda": lam, "n_inject": n_inj, "n_edit": n_edit,
           "p_gate_only": prov["p_gate_only"],
           "p_inject_only": prov["p_inject_only"], "p_both": prov["p_both"],
           "rec_gate": rec(prov["p_gate_only"]),
           "rec_inject": rec(prov["p_inject_only"]),
           "rec_both": rec(prov["p_both"]),
           "holdout_data_loss": prov["holdout_data_loss"],
           "a_pos": round(a_pos, 4), "p_neg_nat": round(p_neg_nat, 4),
           "secs": round(time.time() - t0, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-9g %8s %8s | %9.3f %9.3f %9.3f | %8s %8s %8s"
          % (lam, format(n_inj, ","), format(n_edit, ","),
             prov["p_gate_only"], prov["p_inject_only"], prov["p_both"],
             row["rec_gate"], row["rec_inject"], row["rec_both"]), flush=True)
    torch.cuda.empty_cache()
fh.close()
