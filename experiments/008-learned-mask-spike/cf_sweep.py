"""Lambda sweeps for the cf-hosted mask modes (contrast + negctx), one seed.

Each objective gets the eval its loss actually targets, alongside free0:

  mask_contrast — selectivity: free0 on posctx (closure) AND the kept set's
    behaviour on negctx (circuit-only on neg tokens at the seed's would-be
    anchors). A selective circuit reconstructs firing on posctx and does NOT
    fire the seed on negctx (a_negfire ~= natural neg ~= 0).

  mask_negctx — gate opening ON THE NATURAL STREAM: the selected edits are
    zeroed out of an otherwise-complete keep set (keep-all is identity), so
    this is the ceteris-paribus knockout, learned. gate_recovery =
    (p_gate - p_neg_nat) / (target - p_neg_nat), measured in PRE-ACTIVATION
    (uncensored). The residual gap decomposes silence into
    suppression-gated vs drive-absent.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/cf_sweep.py
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
LAMBDAS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
STEPS, LR, BETA = 200, 0.1, 1.0
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

sae = bank.saes[kind][layer]
w_seed = sae.encoder.weight[LATENT].detach()
b_seed = sae._get_bias_eff()[LATENT].detach()

a_pos = measure_seed_activation(inference, bank, pt, layer, kind, LATENT, pa,
                                batch_size=EVAL_BS)
a_e0 = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, LATENT,
                               pos_argmax=pa, batch_size=EVAL_BS)
den = float(a_pos) - float(a_e0)

# natural negctx pre-act + would-be-firing anchors (the engine's own anchor)
_p = LearnedMaskPatcher(bank, {}, layer, kind, w_seed, b_seed)
inference.disable_compile()
try:
    inference.forward(nt, patcher=_p, grad_enabled=False,
                      return_activations=False, tokenize_final=False)
finally:
    inference.enable_compile()
neg_nat_pre = _p.seed_pre.detach()
neg_anchors = neg_nat_pre.argmax(dim=-1).cpu()
idxB = torch.arange(neg_nat_pre.shape[0], device=neg_nat_pre.device)
p_neg_nat = float(neg_nat_pre[idxB, neg_anchors.to(neg_nat_pre.device)].mean())
print("[%s] a_pos %.4f | natural negctx preact at anchors %.4f"
      % (TAG, a_pos, p_neg_nat), flush=True)

fh = (HERE / "cf_rows.jsonl").open("a")
for objective in ("contrast", "negctx"):
    print("\n=== cf-mask_%s ===" % objective, flush=True)
    if objective == "contrast":
        print("%-10s %10s %9s %10s %12s %8s"
              % ("lambda", "n", "free0", "negfire", "holdout", "secs"), flush=True)
    else:
        print("%-10s %10s %12s %11s %12s %8s"
              % ("lambda", "n_edits", "p_gate", "gate_rec", "holdout", "secs"), flush=True)
    for lam in LAMBDAS:
        t0 = time.time()
        scores, prov = run_learned_mask(
            inference, bank, objective=objective, sites=sorted(up),
            seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
            pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
            target_act=float(a_pos), steps=STEPS, lr=LR, l1_lambda=lam,
            beta=BETA, keep_threshold=0.5, batch_size=4, holdout_frac=0.25,
            log_every=0,
            deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
            deep_batch_size=config.discovery.learned_mask.deep_batch_size)
        n = len(scores)
        row = {"tag": TAG, "seed": "%d/%d" % (SC_IDX, LATENT),
               "objective": objective, "lambda": lam, "steps": STEPS, "lr": LR,
               "beta": BETA, "n": n,
               "holdout_data_loss": prov["holdout_data_loss"],
               "loss_final": prov["loss_final"],
               "a_pos": round(float(a_pos), 4),
               "p_neg_nat": round(p_neg_nat, 4)}
        if objective == "contrast":
            keep = {}
            for fid in scores:
                keep.setdefault((fid.layer, fid.kind), set()).add(int(fid.index))
            if n:
                a_c = circuit_only_activation(
                    inference, bank, keep, up, pt, layer, kind, LATENT,
                    pos_argmax=pa, batch_size=EVAL_BS)
                row["free0"] = round((float(a_c) - float(a_e0)) / den, 4)
                # selectivity: the kept set run circuit-only on NEGCTX at the
                # would-be anchors must NOT fire the seed
                row["a_negfire"] = round(float(circuit_only_activation(
                    inference, bank, keep, up, nt, layer, kind, LATENT,
                    pos_argmax=neg_anchors, batch_size=EVAL_BS)), 4)
            else:
                row["free0"], row["a_negfire"] = 0.0, 0.0
            print("%-10g %10s %9s %10s %12s %8.0f"
                  % (lam, format(n, ","), row["free0"], row["a_negfire"],
                     ("%.3f" % prov["holdout_data_loss"]
                      if prov["holdout_data_loss"] is not None else "—"),
                     time.time() - t0), flush=True)
        else:
            # gate opening on the NATURAL stream: keep everything except the
            # selected edits (keep-all is identity), pre-act at neg anchors
            sel = {}
            for fid in scores:
                sel.setdefault((fid.layer, fid.kind), set()).add(int(fid.index))
            keep_minus = {s: set(range(bank.d_sae)) - sel.get(s, set())
                          for s in up}
            p_gate = float(circuit_only_activation(
                inference, bank, keep_minus, up, nt, layer, kind, LATENT,
                pos_argmax=neg_anchors, batch_size=EVAL_BS, preact=True))
            gden = float(a_pos) - p_neg_nat
            row["n_edits"] = n
            row["p_gate"] = round(p_gate, 4)
            row["gate_recovery"] = (round((p_gate - p_neg_nat) / gden, 4)
                                    if abs(gden) > 1e-9 else None)
            print("%-10g %10s %12.4f %11s %12s %8.0f"
                  % (lam, format(n, ","), p_gate, row["gate_recovery"],
                     ("%.3f" % prov["holdout_data_loss"]
                      if prov["holdout_data_loss"] is not None else "—"),
                     time.time() - t0), flush=True)
        row["secs"] = round(time.time() - t0, 1)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        torch.cuda.empty_cache()
fh.close()
