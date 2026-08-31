"""Distribution of direct-effect edge weights onto the seed: how many members
have GENUINE direct edges, vs tail noise the top-K sweep dragged along?

Reports, per seed: quantiles of |w|, top-k mass shares, counts above relative
thresholds, and the participation ratio (sum w)^2 / sum w^2 — the standard
"effective number of contributors" (equals N for uniform weights, 1 for a
single dominant one).

  SEED_TAG=L10 PYTHONPATH=src python experiments/007-direct-drivers/edge_dist.py
"""
import os
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.sae_graph import SAEGraphInstrument
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
CIRC = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2/experiments/"
            "007-free0-cf-32seed/circuits")
SEEDS = {"L2": (8, 30122), "L8": (25, 10628), "L9": (27, 6859), "L10": (32, 3021)}
TAG = os.environ.get("SEED_TAG", "L10")
SC_IDX, LATENT = SEEDS[TAG]
N_SEQ, GRAD_B, NK = 64, 8, 3
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * NK, bank.d_sae), device=bank.device)
_apply_sweep_config(max_per_site=24)
config.discovery.probe_batch_size = 4

layer, ki = split_component_idx(SC_IDX, NK)
kind = bank.kinds[ki]
up = upstream_sites(bank, layer, kind)
m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                        avg_acts, probe_builder)
pd_ = m0.build_probe_dataset(SC_IDX, LATENT)
pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]

d = torch.load(CIRC / ("%d_%d" % (SC_IDX, LATENT)) / "abl-ig_mean_PA__rec2mag.pt",
               map_location="cpu", weights_only=False)
roles = [d["roles_legend"][i] for i in d["role"].tolist()]
members = [(l, d["kinds_legend"][k], i)
           for (l, k, i), r in zip(zip(d["layer"].tolist(), d["kind_idx"].tolist(),
                                       d["index"].tolist()), roles)
           if r != "seed" and (l, d["kinds_legend"][k]) in up]

sae = bank.saes[kind][layer]
w_seed = sae.encoder.weight[LATENT].detach()
b_seed = sae._get_bias_eff()[LATENT].detach()
instrument = SAEGraphInstrument(bank)
seed_pre = []
orig = instrument.transform

def tap(layer_idx, kd, x):
    if layer_idx == layer and kd == kind:
        seed_pre.append(x @ w_seed.to(x.device, x.dtype) + b_seed.to(x.device, x.dtype))
        return x
    return orig(layer_idx, kd, x)

instrument.transform = tap
inference.disable_compile()
try:
    inference.forward(pt[:GRAD_B], patcher=instrument, grad_enabled=True,
                      return_activations=False, tokenize_final=False)
finally:
    inference.enable_compile()
pre = seed_pre[0]
idx = torch.arange(min(GRAD_B, pa.shape[0]), device=pre.device)
metric = pre[idx, pa[:len(idx)].to(pre.device).clamp(0, pre.shape[1] - 1)].mean()
graph = instrument.graph
sites = [s for s in sorted(graph.activations) if s in up]
anchors = [graph.get_latents(*s)[0].act for s in sites]
grads = torch.autograd.grad(metric, anchors, allow_unused=True)

vals = []
for s, a, g in zip(sites, anchors, grads):
    if g is None:
        continue
    w = (g * a.detach()).sum(dim=1).mean(dim=0)
    mem_at = [i for (l, kd, i) in members if (l, kd) == s]
    if mem_at:
        vals.append(w[torch.tensor(mem_at, device=w.device)].abs().float().cpu())
w = torch.cat(vals)
w_sorted, _ = torch.sort(w, descending=True)
n = w.numel()
tot = float(w.sum()) or 1e-12
mx = float(w_sorted[0])

print("\n[%s] %s members with a direct-edge value" % (TAG, format(n, ",")))
print("max |w| = %.3e" % mx)
print("\nquantiles of |w|:")
for q in (0.5, 0.9, 0.99, 0.999):
    print("  p%-5g %.3e  (%.5f%% of max)" % (100 * q,
          float(torch.quantile(w, q)), 100 * float(torch.quantile(w, q)) / mx))
print("\ncumulative mass share of top-k:")
for k in (16, 64, 256, 1024, 4096, 16384):
    if k <= n:
        print("  top %-6d %6.1f%%" % (k, 100 * float(w_sorted[:k].sum()) / tot))
print("\ncount above relative threshold:")
for f in (0.1, 0.01, 0.001, 0.0001):
    print("  |w| > %g%% of max: %s members"
          % (100 * f, format(int((w > f * mx).sum()), ",")))
pr = float(w.sum()) ** 2 / float((w ** 2).sum())
print("\nparticipation ratio (effective # of direct contributors): %.0f" % pr)
