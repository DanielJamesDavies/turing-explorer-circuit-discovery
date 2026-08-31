"""Does code_dtype="stream" (bf16) change WHICH latents get selected?

code_dtype="stream" keeps the cached SAE codes in the model's native bf16
instead of promoting them to fp32, saving ~1 GB of peak VRAM. That is only
an acceptable default if it does not move the selection. Membership is a
threshold crossing at m = 0.5, so a latent sitting exactly on the boundary
can flip on pure numerical noise; the question is whether flips are RARE
AND MARGINAL (fine) or widespread (not fine).

Reports the symmetric difference of the two selections and, for any latent
that flips, how far its m sits from the 0.5 threshold.

  PYTHONPATH=src python experiments/008-learned-mask-spike/dtype_check.py
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

out = {}
for dt in ("fp32", "stream"):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
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
    out[dt] = {
        # run_learned_mask returns Dict[FeatureID, float]; for objective="pos"
        # the value IS the final m, so |value - 0.5| is the threshold margin.
        "members": {(f.layer, f.kind, int(f.index)): float(v)
                    for f, v in scores.items()},
        "n": len(scores),
        "free0": round((a_c - a_e0) / den, 4) if abs(den) > 1e-9 else None,
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "secs": round(time.time() - t0, 1)}
    print("%-7s n=%-9s free0=%-8s peak=%.2f GB  %.0fs"
          % (dt, format(len(scores), ","), out[dt]["free0"],
             out[dt]["peak_gb"], out[dt]["secs"]), flush=True)
    torch.cuda.empty_cache()

a, b = set(out["fp32"]["members"]), set(out["stream"]["members"])
only_a, only_b, both = a - b, b - a, a & b
# score IS the final m for the learned mask, so |score - 0.5| is the margin
margins = [abs(out["fp32"]["members"].get(k, out["stream"]["members"].get(k)) - 0.5)
           for k in (only_a | only_b)]
shared = [(out["fp32"]["members"][k], out["stream"]["members"][k]) for k in both]
max_dm = max((abs(x - y) for x, y in shared), default=0.0)

print("\nfp32-only %s | stream-only %s | shared %s"
      % (format(len(only_a), ","), format(len(only_b), ","), format(len(both), ",")))
print("jaccard                    %.4f" % (len(both) / max(len(a | b), 1)))
print("flipped as %% of union      %.2f%%" % (100 * len(only_a | only_b) / max(len(a | b), 1)))
if margins:
    margins.sort()
    print("flipped |m-0.5| median     %.4f   max %.4f" % (margins[len(margins)//2], margins[-1]))
print("max |m_fp32 - m_stream| on shared members  %.5f" % max_dm)
print("free0  fp32 %s   stream %s" % (out["fp32"]["free0"], out["stream"]["free0"]))

(HERE / f"dtype_check_{TAG}.json").write_text(json.dumps(
    {"tag": TAG,
     "fp32": {k: v for k, v in out["fp32"].items() if k != "members"},
     "stream": {k: v for k, v in out["stream"].items() if k != "members"},
     "n_fp32_only": len(only_a), "n_stream_only": len(only_b),
     "n_shared": len(both),
     "jaccard": len(both) / max(len(a | b), 1),
     "flipped_margin_median": (margins[len(margins)//2] if margins else None),
     "flipped_margin_max": (margins[-1] if margins else None),
     "max_delta_m_shared": max_dm}, indent=2))
