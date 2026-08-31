"""Where does the hot-budget pruning actually remove members?

The 3.11x budget takes L10 from 108,068 members to 26,450. A single average
per site hides the shape of that: pruning could be uniform, or it could be
concentrated in the shallow sites (far from the seed) or the deep ones (near
it). This dumps per-site membership for both arms side by side.

Runs are bit-deterministic, so these reproduce the sweep rows exactly.

  ARMS is not used here; both arms are hardcoded to match the sweep.
  SEED_TAG=L10 PYTHONPATH=src python .../per_site_counts.py
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
TAG = os.environ.get("SEED_TAG", "L10")
SC_IDX, LATENT = SEEDS[TAG]
N_SEQ, NK = 64, 3
# (label, lr) — flat schedule both, so lr alone sets the budget
ARMS = [("constant", 0.05), ("hotflat", 62.145 / 400)]
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

print("seed %s = layer %d %s latent %d | d_sae %s | %d upstream sites"
      % (TAG, layer, kind, LATENT, format(bank.d_sae, ","), len(up)), flush=True)

counts = {}
for label, lr in ARMS:
    scores, _ = run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa,
        steps=400, lr=lr, l1_lambda=1e-4, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size,
        optimizer="adamw", weight_decay=0.05,
        code_dtype=config.discovery.learned_mask.code_dtype)
    per = {}
    for fid in scores:
        per[(fid.layer, fid.kind)] = per.get((fid.layer, fid.kind), 0) + 1
    counts[label] = per
    print("%-9s total %s" % (label, format(len(scores), ",")), flush=True)
    torch.cuda.empty_cache()

print("\n%-14s %10s %10s %8s %8s %8s"
      % ("site", "constant", "hotflat", "kept%", "const%dsae", "hot%dsae"))
rows = []
for site in up:
    c = counts["constant"].get(site, 0)
    h = counts["hotflat"].get(site, 0)
    rows.append({"layer": site[0], "kind": site[1], "constant": c, "hotflat": h,
                 "kept_frac": (h / c if c else None),
                 "constant_frac_of_dsae": c / bank.d_sae,
                 "hotflat_frac_of_dsae": h / bank.d_sae})
    print("%-14s %10s %10s %8s %8.3f %8.3f"
          % ("L%d-%s" % (site[0], site[1]), format(c, ","), format(h, ","),
             ("%.3f" % (h / c)) if c else "-",
             c / bank.d_sae, h / bank.d_sae))

tc = sum(r["constant"] for r in rows); th = sum(r["hotflat"] for r in rows)
print("%-14s %10s %10s %8.3f" % ("TOTAL", format(tc, ","), format(th, ","),
                                 th / tc if tc else 0))
print("%-14s %10.1f %10.1f" % ("mean/site", tc / len(up), th / len(up)))
med = sorted(r["hotflat"] for r in rows)[len(rows) // 2]
print("%-14s %10s %10d" % ("median/site",
                           format(sorted(r["constant"] for r in rows)[len(rows) // 2], ","),
                           med))

(HERE / f"per_site_{TAG}.json").write_text(json.dumps(
    {"tag": TAG, "layer": layer, "kind": kind, "latent": LATENT,
     "d_sae": bank.d_sae, "n_sites": len(up), "rows": rows}, indent=2))
