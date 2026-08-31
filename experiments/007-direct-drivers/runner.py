"""Direct drivers: are the seed's DIRECT-EFFECT edges a sufficient sub-circuit?

For a saved rec2+mag circuit (abl-ig_mean PA), compute each member's
direct-effect edge weight onto the seed — SFC's edge construction, restricted
to the one target we care about:

    w(u -> seed) = grad_{u, stop(M)}(seed_pre_at_probe) * u_natural

computed with SAEGraphInstrument: every site's feature code enters through a
DETACHED leaf anchor, so the backward from the seed's pre-activation reaches
each anchor only along paths free of other feature nodes — the gradient IS the
unmediated direct effect (u_baseline = 0, free0-coherent). One instrumented
forward + one backward, member->member edges never built (that all-pairs loop
is what OOMed in July; the seed-directed slice was always cheap).

Then the eval the question asks for: keep only the top-K members by |direct
edge|, zero-ablate every other latent (free0 semantics, live re-encode), and
sweep K. Controls at every K:
  attr   — top-K by |attribution| (the existing driver ranking)
  rand   — random K members (size-matched null)

  SEED_TAG=L10 PYTHONPATH=src python experiments/007-direct-drivers/runner.py
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
from eval.ablation_faithfulness import circuit_only_activation, upstream_sites
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
OUT = HERE / "rows.jsonl"
SEEDS = {"L2": (8, 30122), "L8": (25, 10628), "L9": (27, 6859), "L10": (32, 3021)}
TAG = os.environ.get("SEED_TAG", "L10")
SC_IDX, LATENT = SEEDS[TAG]
CIRC_FILE = "abl-ig_mean_PA__rec2mag.pt"
N_SEQ, EVAL_BS, GRAD_B, NK = 64, 16, 8, 3
KS = [16, 64, 256, 1024, 4096, 16384]
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

# ---- probe data (no discovery) -----------------------------------------
m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                        avg_acts, probe_builder)
pd_ = m0.build_probe_dataset(SC_IDX, LATENT)
pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]

# ---- saved circuit ------------------------------------------------------
d = torch.load(CIRC / ("%d_%d" % (SC_IDX, LATENT)) / CIRC_FILE,
               map_location="cpu", weights_only=False)
roles = [d["roles_legend"][i] for i in d["role"].tolist()]
members = [((l, d["kinds_legend"][k], i), abs(float(s)))
           for (l, k, i, s), r in zip(zip(d["layer"].tolist(), d["kind_idx"].tolist(),
                                          d["index"].tolist(), d["score"].tolist()), roles)
           if r != "seed" and (l, d["kinds_legend"][k]) in up]
attr = dict(members)
print("[%s] circuit %s: %s members in upstream scope"
      % (TAG, CIRC_FILE, format(len(members), ",")), flush=True)

# ---- direct-effect edges onto the seed (one forward + one backward) -----
sae = bank.saes[kind][layer]
w_seed = sae.encoder.weight[LATENT].detach()
b_seed = sae._get_bias_eff()[LATENT].detach()

t0 = time.time()
instrument = SAEGraphInstrument(bank)
seed_pre = []
orig = instrument.transform

def tap(layer_idx, kd, x):
    if layer_idx == layer and kd == kind:
        seed_pre.append(x @ w_seed.to(x.device, x.dtype) + b_seed.to(x.device, x.dtype))
        return x
    return orig(layer_idx, kd, x)

instrument.transform = tap
gb = pt[:GRAD_B]
inference.disable_compile()
try:
    inference.forward(gb, patcher=instrument, grad_enabled=True,
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
    w = (g * a.detach()).sum(dim=1).mean(dim=0)          # [d_sae] grad·natural
    for (l, kd, i), _ in members:
        if (l, kd) == s:
            direct[(l, kd, i)] = abs(float(w[i]))
print("direct edges computed for %s members in %.0fs"
      % (format(len(direct), ","), time.time() - t0), flush=True)
del instrument, graph, anchors, grads, seed_pre
torch.cuda.empty_cache()

# ---- sufficiency sweep --------------------------------------------------
a_pos = None
from eval.ablation_faithfulness import measure_seed_activation
a_pos = measure_seed_activation(inference, bank, pt, layer, kind, LATENT, pa,
                                batch_size=EVAL_BS)
a_e0 = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, LATENT,
                               pos_argmax=pa, batch_size=EVAL_BS)
den = float(a_pos) - float(a_e0)

by_direct = sorted(direct, key=lambda t: -direct[t])
by_attr = sorted(attr, key=lambda t: -attr[t])
rng = random.Random(42)
shuffled = list(attr); rng.shuffle(shuffled)

# Pins: the members' natural clean values (collapsed probe-position means) —
# "SET only these latents, silence everything else" is pinned + zero fill.
from eval.floors import collect_site_anchors
_, pins = collect_site_anchors(inference, bank, pt, up, pa,
                               pin_position_specific=False)

def phi_of(keys, pinned):
    keep = {}
    for (l, kd, i) in keys:
        keep.setdefault((l, kd), set()).add(i)
    a_c = circuit_only_activation(inference, bank, keep, up, pt, layer, kind,
                                  LATENT, pos_argmax=pa, batch_size=EVAL_BS,
                                  pin_values=pins if pinned else None)
    return round((float(a_c) - float(a_e0)) / den, 4) if abs(den) > 1e-9 else None

fh = OUT.open("a")
print("\n%-8s | %9s %9s %9s | %9s %9s %9s"
      % ("K", "free_dir", "free_att", "free_rnd", "pin_dir", "pin_att", "pin_rnd"),
      flush=True)
for K in [k for k in KS if k <= len(members)] + [len(members)]:
    row = {"seed": "%d/%d" % (SC_IDX, LATENT), "tag": TAG, "K": K,
           "free0_direct": phi_of(by_direct[:K], False),
           "free0_attr": phi_of(by_attr[:K], False),
           "free0_rand": phi_of(shuffled[:K], False),
           "pin0_direct": phi_of(by_direct[:K], True),
           "pin0_attr": phi_of(by_attr[:K], True),
           "pin0_rand": phi_of(shuffled[:K], True),
           "n_members": len(members), "grad_batch": GRAD_B}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-8d | %9s %9s %9s | %9s %9s %9s"
          % (K, row["free0_direct"], row["free0_attr"], row["free0_rand"],
             row["pin0_direct"], row["pin0_attr"], row["pin0_rand"]), flush=True)
fh.close()
print("\nwrote %s" % OUT)
