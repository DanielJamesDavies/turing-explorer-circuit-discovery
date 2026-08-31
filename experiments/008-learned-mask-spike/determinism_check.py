"""Is the ~17% membership churn caused by bf16, or is it run-to-run anyway?

dtype_check.py found fp32 vs stream(bf16) share only ~0.83 Jaccard, with
shared members' m differing by up to 0.34. That is far too large to be
rounding, so either (a) bf16 genuinely changes the answer, or (b) the mask
optimum is non-unique / chaotic and ANY two runs disagree that much
regardless of dtype. Those have opposite implications and the previous
script cannot distinguish them, because it never repeated a dtype.

This runs each dtype TWICE and prints the full pairwise Jaccard matrix.
Read it as:
  within-fp32 ~= within-stream ~= across  -> churn is run-to-run, not dtype
  within-* high, across low               -> bf16 really does change selection

Nothing here is seeded deliberately: the point is to measure the natural
variation the pipeline actually has, not to suppress it.

  PYTHONPATH=src python experiments/008-learned-mask-spike/determinism_check.py
"""
import itertools
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
# fp32 costs ~95s at L8 but ~633s at L10 (it spills past VRAM), so the
# repeat-control is run at L8 where four arms are affordable.
ARMS = [("fp32.a", "fp32"), ("fp32.b", "fp32"),
        ("strm.a", "stream"), ("strm.b", "stream")]
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
a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind, LATENT,
                                      pa, batch_size=EVAL_BS))
a_e0 = float(circuit_only_activation(inference, bank, {}, up, pt, layer, kind,
                                     LATENT, pos_argmax=pa, batch_size=EVAL_BS))
den = a_pos - a_e0

runs = {}
for label, dt in ARMS:
    t0 = time.time()
    scores, prov = run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa,
        steps=400, lr=0.05, l1_lambda=1e-4, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size,
        optimizer="adamw", weight_decay=0.05, code_dtype=dt)
    keep = {}
    for fid in scores:
        keep.setdefault((fid.layer, fid.kind), set()).add(int(fid.index))
    a_c = float(circuit_only_activation(inference, bank, keep, up, pt, layer,
                                        kind, LATENT, pos_argmax=pa,
                                        batch_size=EVAL_BS)) if scores else a_e0
    runs[label] = {"m": {(f.layer, f.kind, int(f.index)): float(v)
                         for f, v in scores.items()},
                   "n": len(scores),
                   "free0": round((a_c - a_e0) / den, 4) if abs(den) > 1e-9 else None,
                   "loss_final": prov["loss_final"],
                   "secs": round(time.time() - t0, 1)}
    print("%-8s n=%-9s free0=%-8s loss=%.6f  %.0fs"
          % (label, format(len(scores), ","), runs[label]["free0"],
             prov["loss_final"], runs[label]["secs"]), flush=True)
    torch.cuda.empty_cache()

print("\npairwise (jaccard | %flip of union | max |dm| on shared)")
pairs = {}
for x, y in itertools.combinations([lab for lab, _ in ARMS], 2):
    a, b = set(runs[x]["m"]), set(runs[y]["m"])
    both = a & b
    jac = len(both) / max(len(a | b), 1)
    dm = max((abs(runs[x]["m"][k] - runs[y]["m"][k]) for k in both), default=0.0)
    kind_ = ("within" if x[:4] == y[:4] else "ACROSS")
    pairs[f"{x}|{y}"] = {"jaccard": jac, "max_dm_shared": dm, "kind": kind_}
    print("  %-8s %-8s %-7s  %.4f   %5.2f%%   %.5f"
          % (x, y, kind_, jac, 100 * len(a ^ b) / max(len(a | b), 1), dm))

within = [v["jaccard"] for v in pairs.values() if v["kind"] == "within"]
across = [v["jaccard"] for v in pairs.values() if v["kind"] == "ACROSS"]
print("\nmean jaccard  within-dtype %.4f   across-dtype %.4f   gap %.4f"
      % (sum(within) / len(within), sum(across) / len(across),
         sum(within) / len(within) - sum(across) / len(across)))

(HERE / f"determinism_{TAG}.json").write_text(json.dumps(
    {"tag": TAG,
     "runs": {k: {kk: vv for kk, vv in v.items() if kk != "m"}
              for k, v in runs.items()},
     "pairs": pairs,
     "mean_jaccard_within": sum(within) / len(within),
     "mean_jaccard_across": sum(across) / len(across)}, indent=2))
