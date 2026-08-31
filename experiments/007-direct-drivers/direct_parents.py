"""How many DIRECT PARENTS does a deep seed actually have?

The residual passthrough gives every upstream latent a nonzero unmediated path
into the seed, so "direct parent" is a continuous weight, not a discrete edge.
This measures whether the direct-effect mass has a natural cutoff:

  1. concentration — members needed for 50/90/99% of total |direct| mass
  2. layer profile — where direct mass lives vs where MEMBERSHIP lives
     (the chain intuition: direct parents hug the seed, ancestors sit deep)
  3. direct-vs-attribution — rank agreement; a member with high attribution
     but ~zero direct weight is a MEDIATED ancestor
  4. sufficiency at the natural sets — keep the 50%/90%/99%-direct-mass sets,
     pinned and free, with size-matched random controls

Also SAVES the per-member direct weights (direct_weights_{tag}.pt) so this
never needs the GPU again.

  SEED_TAG=L10 PYTHONPATH=src python experiments/007-direct-drivers/direct_parents.py
"""
import json
import os
import random
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
CIRC = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2/experiments/"
            "007-free0-cf-32seed/circuits")
HERE = Path(__file__).parent
SEEDS = {"L2": (8, 30122), "L8": (25, 10628), "L9": (27, 6859), "L10": (32, 3021)}
TAG = os.environ.get("SEED_TAG", "L10")
SC_IDX, LATENT = SEEDS[TAG]
CIRC_FILE = "abl-ig_mean_PA__rec2mag.pt"
N_SEQ, EVAL_BS, GRAD_B, NK = 64, 16, 8, 3
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
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4

layer, ki = split_component_idx(SC_IDX, NK)
kind = bank.kinds[ki]
up = upstream_sites(bank, layer, kind)

m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                        avg_acts, probe_builder)
pd_ = m0.build_probe_dataset(SC_IDX, LATENT)
pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]

d = torch.load(CIRC / ("%d_%d" % (SC_IDX, LATENT)) / CIRC_FILE,
               map_location="cpu", weights_only=False)
roles = [d["roles_legend"][i] for i in d["role"].tolist()]
members = [((l, d["kinds_legend"][k], i), abs(float(s)))
           for (l, k, i, s), r in zip(zip(d["layer"].tolist(), d["kind_idx"].tolist(),
                                          d["index"].tolist(), d["score"].tolist()), roles)
           if r != "seed" and (l, d["kinds_legend"][k]) in up]
attr = dict(members)

# ---- direct weights (one forward + one backward) ------------------------
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
direct = {}
for s, a, g in zip(sites, anchors, grads):
    if g is None:
        continue
    w = (g * a.detach()).sum(dim=1).mean(dim=0)
    for (l, kd, i) in attr:
        if (l, kd) == s:
            direct[(l, kd, i)] = abs(float(w[i]))
del instrument, graph, anchors, grads, seed_pre
torch.cuda.empty_cache()
torch.save({"direct": direct, "attr": attr, "tag": TAG},
           HERE / ("direct_weights_%s.pt" % TAG))

n = len(direct)
ranked = sorted(direct.items(), key=lambda kv: -kv[1])
tot = sum(v for _, v in ranked) or 1e-12

# ---- 1. concentration ---------------------------------------------------
print("\n[%s] %s members | total |direct| mass %.4f" % (TAG, format(n, ","), tot))
cum, marks, out_marks = 0.0, [0.5, 0.9, 0.99], {}
mi = 0
for rank_i, (_, v) in enumerate(ranked, 1):
    cum += v
    while mi < len(marks) and cum >= marks[mi] * tot:
        out_marks[marks[mi]] = rank_i
        mi += 1
print("1. CONCENTRATION of direct-effect mass:")
for m in marks:
    r = out_marks.get(m, n)
    print("   %2.0f%% of direct mass -> top %8s members (%.2f%% of circuit)"
          % (100 * m, format(r, ","), 100.0 * r / n))

# ---- 2. layer profile ---------------------------------------------------
print("2. LAYER PROFILE (direct mass% vs membership% by distance-to-seed):")
buckets = [(0, 1), (2, 3), (4, 6), (7, 12)]
for lo, hi in buckets:
    dm = sum(v for (l, _, _), v in direct.items() if lo <= layer - l <= hi) / tot
    mm = sum(1 for (l, _, _) in direct if lo <= layer - l <= hi) / n
    print("   d=%d..%-2d   direct mass %5.1f%%   members %5.1f%%"
          % (lo, hi, 100 * dm, 100 * mm))

# ---- 3. direct vs attribution ------------------------------------------
attr_ranked = sorted(attr, key=lambda t: -attr[t])
top_att = set(attr_ranked[:out_marks.get(0.9, n)])
top_dir = set(t for t, _ in ranked[:out_marks.get(0.9, n)])
ovl = len(top_att & top_dir) / max(len(top_dir), 1)
print("3. top-(90%%-mass) direct set vs same-size attribution set: overlap %.2f" % ovl)

# ---- 4. sufficiency at natural sets ------------------------------------
a_pos = measure_seed_activation(inference, bank, pt, layer, kind, LATENT, pa,
                                batch_size=EVAL_BS)
a_e0 = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, LATENT,
                               pos_argmax=pa, batch_size=EVAL_BS)
den = float(a_pos) - float(a_e0)
_, pins = collect_site_anchors(inference, bank, pt, up, pa,
                               pin_position_specific=False)
rng = random.Random(42)
allm = list(attr)

def phi(keys, pinned):
    keep = {}
    for (l, kd, i) in keys:
        keep.setdefault((l, kd), set()).add(i)
    a_c = circuit_only_activation(inference, bank, keep, up, pt, layer, kind,
                                  LATENT, pos_argmax=pa, batch_size=EVAL_BS,
                                  pin_values=pins if pinned else None)
    return round((float(a_c) - float(a_e0)) / den, 4) if abs(den) > 1e-9 else None

print("4. SUFFICIENCY at natural direct-mass sets:")
print("   %-14s %9s | %8s %8s | %8s %8s"
      % ("set", "K", "pin", "pin_rnd", "free", "free_rnd"))
fh = (HERE / "parents_rows.jsonl").open("a")
for m in marks:
    K = out_marks.get(m, n)
    keys = [t for t, _ in ranked[:K]]
    rnd = rng.sample(allm, K)
    row = {"tag": TAG, "mass": m, "K": K,
           "pin": phi(keys, True), "pin_rand": phi(rnd, True),
           "free": phi(keys, False), "free_rand": phi(rnd, False)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("   %2.0f%%-mass %13s | %8s %8s | %8s %8s"
          % (100 * m, format(K, ","), row["pin"], row["pin_rand"],
             row["free"], row["free_rand"]))
fh.close()
