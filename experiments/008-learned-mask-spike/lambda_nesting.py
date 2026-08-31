"""Is the lambda sweep NESTED, or does it jump between basins?

L8 lambda=1.6e-3 is an outlier: holdout DATA loss 5.06 against 1.88-2.34
everywhere else, and free0_hold 0.4953 against ~1.0. Its higher-lambda
neighbour (3.2e-3) is back on trend. So the run at 1.6e-3 optimised WORSE,
rather than 3.2e-3 finding something special.

If sparsification were a smooth trajectory, each higher-lambda circuit would
be close to a SUBSET of the one below it: raising the penalty should remove
members, not swap them. Containment = |A & B| / |B| with B the smaller
(higher-lambda) set answers that directly:

  containment ~ 1.0 everywhere  -> smooth nested pruning, 1.6e-3 just
                                   overshot along the same path
  containment dips at 1.6e-3    -> that run left the path into a different
                                   basin, i.e. a rugged landscape

Runs are bit-deterministic, so these reproduce the sweep rows exactly.

  SEED_TAG=L8 PYTHONPATH=src python .../lambda_nesting.py
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
from eval.ablation_faithfulness import upstream_sites
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
LAMBDAS = (1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3)
HOTFLAT = 62.145 / 400
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
config.discovery.probe_sequence_count = N_SEQ
config.discovery.eval_sequence_count = N_SEQ
config.discovery.probe_batch_size = 4

layer, ki = split_component_idx(SC_IDX, NK)
kind = bank.kinds[ki]
up = sorted(upstream_sites(bank, layer, kind))

m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                        avg_acts, probe_builder)
pd_ = m0.build_probe_dataset(SC_IDX, LATENT)
pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]

sets, loss = {}, {}
for lam in LAMBDAS:
    scores, prov = run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa,
        steps=400, lr=HOTFLAT, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size,
        optimizer="adamw", weight_decay=0.05,
        code_dtype=config.discovery.learned_mask.code_dtype)
    sets[lam] = {(f.layer, f.kind, int(f.index)) for f in scores}
    loss[lam] = prov["holdout_data_loss"]
    print("lambda %.1e  n = %-8s holdout_data_loss %.4f"
          % (lam, format(len(scores), ","), loss[lam]), flush=True)
    torch.cuda.empty_cache()

print("\nconsecutive pairs — containment = |A&B|/|B|, B = smaller (higher lambda)")
print("%-22s %9s %9s %11s %9s" % ("pair", "n_lo", "n_hi", "containment", "jaccard"))
out = []
for a, b in zip(LAMBDAS, LAMBDAS[1:]):
    A, B = sets[a], sets[b]
    small, big = (B, A) if len(B) <= len(A) else (A, B)
    cont = len(A & B) / max(len(small), 1)
    jac = len(A & B) / max(len(A | B), 1)
    out.append({"lo": a, "hi": b, "n_lo": len(A), "n_hi": len(B),
                "containment": cont, "jaccard": jac,
                "new_in_hi": len(B - A)})
    print("%-22s %9s %9s %11.4f %9.4f"
          % ("%.1e -> %.1e" % (a, b), format(len(A), ","), format(len(B), ","),
             cont, jac))

print("\nmembers present at high lambda but ABSENT at the lambda below")
print("(smooth pruning implies ~0; a basin jump implies many)")
for r in out:
    print("  %.1e -> %.1e : %s of %s  (%.1f%%)"
          % (r["lo"], r["hi"], format(r["new_in_hi"], ","),
             format(r["n_hi"], ","), 100 * r["new_in_hi"] / max(r["n_hi"], 1)))

(HERE / f"lambda_nesting_{TAG}.json").write_text(json.dumps(
    {"tag": TAG, "holdout_data_loss": {"%.1e" % k: v for k, v in loss.items()},
     "sizes": {"%.1e" % k: len(v) for k, v in sets.items()},
     "pairs": out}, indent=2))
