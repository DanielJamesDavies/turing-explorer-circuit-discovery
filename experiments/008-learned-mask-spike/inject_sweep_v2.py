"""mask_inject v2: sweep inject_lambda on its OWN scale, log concentration.

v1 shared one lambda between the two levers and found a diffuse
sub-threshold delta blanket that reached the target exactly with ZERO
selected latents, abandoning the gate entirely. v2 prices delta separately
and reports its concentration, so the row shows whether "recovery" came from
a sparse population or a blanket.

l1_lambda (the gate's price) is PINNED at the value mask_negctx used for its
best gate (1e-3, rec_gate 0.34); only inject_lambda moves. The interpretable
regime is where injection can no longer trivially reach the target.

  SEED_TAG=L8 EXCLUDE=0 PYTHONPATH=src python experiments/008-learned-mask-spike/inject_sweep_v2.py
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
EXCLUDE = int(os.environ.get("EXCLUDE", 0))
GATE_LAMBDA = 1e-3                      # pinned: mask_negctx's best-gate price
INJ_LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
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
print("[%s] target %.4f | natural neg %.4f | gate lambda %g | exclude %d site(s)"
      % (TAG, a_pos, p_neg_nat, GATE_LAMBDA, EXCLUDE), flush=True)
print("%-9s %7s %7s | %8s %8s | %8s %9s %8s | %8s"
      % ("inj_lam", "n_inj", "n_edit", "rec_gate", "rec_both",
         "d_sum", "d_top1%", "d_max", "secs"), flush=True)

fh = (HERE / "inject_v2_rows.jsonl").open("a")
for ilam in INJ_LAMBDAS:
    t0 = time.time()
    scores, prov = run_learned_mask(
        inference, bank, objective="inject", sites=sorted(up),
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt, target_act=a_pos,
        steps=200, lr=0.1, l1_lambda=GATE_LAMBDA, inject_lambda=ilam,
        inject_exclude_sites=EXCLUDE, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size)
    rec = lambda p: round((p - p_neg_nat) / gden, 4) if abs(gden) > 1e-9 else None
    row = {"tag": TAG, "gate_lambda": GATE_LAMBDA, "inject_lambda": ilam,
           "exclude_sites": EXCLUDE,
           "n_inject": sum(1 for v in scores.values() if v > 0),
           "n_edit": sum(1 for v in scores.values() if v < 0),
           "rec_gate": rec(prov["p_gate_only"]),
           "rec_inject": rec(prov["p_inject_only"]),
           "rec_both": rec(prov["p_both"]),
           "delta_sum": prov["delta_sum"],
           "delta_top1pct_share": prov["delta_top1pct_share"],
           "delta_max": prov["delta_max"],
           "n_delta_gt_0p5": prov["n_delta_gt_0p5"],
           "holdout_data_loss": prov["holdout_data_loss"],
           "a_pos": round(a_pos, 4), "p_neg_nat": round(p_neg_nat, 4),
           "secs": round(time.time() - t0, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-9g %7s %7s | %8s %8s | %8.2f %9s %8.3f | %8.0f"
          % (ilam, format(row["n_inject"], ","), format(row["n_edit"], ","),
             row["rec_gate"], row["rec_both"], row["delta_sum"],
             row["delta_top1pct_share"], row["delta_max"], row["secs"]),
          flush=True)
    torch.cuda.empty_cache()
fh.close()
