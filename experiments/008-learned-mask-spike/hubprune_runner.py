"""Hub prune: what happens when members firing >10% of tokens are removed?

The mass-by-rate analysis found ~0.5% of members (firing rate > 10% — the
quasi-always-on "hub" latents) carry ~10% of attribution mass, ~20x their
count share. They are also the members whose pinning/injection is most
off-natural. This removes them from the saved rec2+mag circuit and re-runs
the eval matrix, against two references:

  full      — the circuit as saved
  no-hub    — members with rate > HUB_RATE removed
  rand-ctl  — same NUMBER of members removed at random (seed 42)

Metrics: free0, freeM_dense, pinMC_dense (mean-fill pins — the overshoot
metric), and cf/cfa with details (injection overshoot). If the hub latents
drive pinned/injected overshoot, no-hub should pull pinMC and cf toward 1.0
with little free0 cost; if free0 craters instead, hubs are load-bearing.

  SEED_TAG=L10 PYTHONPATH=src python experiments/008-learned-mask-spike/runner.py
"""
import json
import os
import random
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from circuit.types.feature_id import FeatureID
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, collect_site_means, measure_seed_activation,
    upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from store.circuits import Circuit, CircuitNode

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
HUB_RATE = 0.10
N_SEQ, EVAL_BS, NK = 64, 16, 3
torch.set_float32_matmul_precision("high")

LS = torch.load(RUN_ROOT / "latent_stats.pt", map_location="cpu", weights_only=False)
RATE = LS["active_count"].float() / (6060 * 262144.0)

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
nt = pd_.neg_tokens[:N_SEQ]

d = torch.load(CIRC / ("%d_%d" % (SC_IDX, LATENT)) / CIRC_FILE,
               map_location="cpu", weights_only=False)
roles_l = [d["roles_legend"][i] for i in d["role"].tolist()]
members = []          # (triple, |score|, role, rate)
for (l, k, i, s), r in zip(zip(d["layer"].tolist(), d["kind_idx"].tolist(),
                               d["index"].tolist(), d["score"].tolist()), roles_l):
    if r == "seed" or (l, d["kinds_legend"][k]) not in up:
        continue
    members.append(((l, d["kinds_legend"][k], i), abs(float(s)), r,
                    float(RATE[l * NK + k, i])))

hubs = [m for m in members if m[3] > HUB_RATE]
tot_mass = sum(m[1] for m in members) or 1e-12
hub_mass = sum(m[1] for m in hubs)
rng = random.Random(42)
rand_removed = set(id(m) for m in rng.sample(members, len(hubs)))
print("[%s] members %s | hubs(rate>%.0f%%) %s (%.2f%% of members, %.1f%% of mass)"
      % (TAG, format(len(members), ","), 100 * HUB_RATE, format(len(hubs), ","),
         100.0 * len(hubs) / len(members), 100.0 * hub_mass / tot_mass), flush=True)

a_pos = measure_seed_activation(inference, bank, pt, layer, kind, LATENT, pa,
                                batch_size=EVAL_BS)
means_up = collect_site_means(inference, bank, pt, up)
a_e0 = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, LATENT,
                               pos_argmax=pa, batch_size=EVAL_BS)
a_eM = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, LATENT,
                               pos_argmax=pa, site_means=means_up, batch_size=EVAL_BS)
_, pins = collect_site_anchors(inference, bank, pt, up, pa,
                               pin_position_specific=False)

VARIANTS = {
    "full": members,
    "no-hub": [m for m in members if m[3] <= HUB_RATE],
    "rand-ctl": [m for m in members if id(m) not in rand_removed],
}

fh = OUT.open("a")
print("\n%-9s %9s | %8s %8s %8s | %8s %8s | %8s %8s"
      % ("variant", "n", "free0", "freeM", "pinMC", "cf", "cf_bnd", "cfa", "cfa_bnd"),
      flush=True)
for name, mm in VARIANTS.items():
    keep = {}
    circ = Circuit(name=name)
    circ.add_node(CircuitNode(metadata={
        "feature_id": FeatureID(layer, kind, LATENT), "role": "seed"}))
    for (t, sc, role, _r) in mm:
        keep.setdefault((t[0], t[1]), set()).add(t[2])
        circ.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(t[0], t[1], t[2]), "role": role,
            "attribution_score": sc}))

    def act(site_means=None, pin_values=None):
        return circuit_only_activation(
            inference, bank, keep, up, pt, layer, kind, LATENT, pos_argmax=pa,
            site_means=site_means, pin_values=pin_values, batch_size=EVAL_BS)

    den0 = float(a_pos) - float(a_e0)
    denM = float(a_pos) - float(a_eM)
    free0 = (float(act()) - float(a_e0)) / den0 if abs(den0) > 1e-9 else None
    freeM = ((float(act(site_means=means_up)) - float(a_eM)) / denM
             if abs(denM) > 1e-9 else None)
    pinMC = ((float(act(site_means=means_up, pin_values=pins)) - float(a_eM)) / denM
             if abs(denM) > 1e-9 else None)
    cfd = {}
    for mode, pref in (("legacy", "cf"), ("negctx_preact", "cfa")):
        try:
            _c, _s, det = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, circ, neg_tokens=nt, pos_tokens=pt,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
                pos_argmax=pa, circuit_layers={L for (L, _) in keep},
                anchor_mode=mode, return_details=True)
            cfd[pref] = round(float(_c), 4)
            cfd[pref + "_bnd"] = (round(det["cf_bounded"], 4)
                                  if det.get("cf_bounded") is not None else None)
        except Exception as exc:
            cfd[pref] = None; cfd[pref + "_bnd"] = None
    row = {"tag": TAG, "seed": "%d/%d" % (SC_IDX, LATENT), "variant": name,
           "n": len(mm), "hub_rate": HUB_RATE,
           "free0": round(free0, 4), "freeM": round(freeM, 4),
           "pinMC": round(pinMC, 4), **cfd,
           "a_pos": round(float(a_pos), 4), "a_e0": round(float(a_e0), 4),
           "a_eM": round(float(a_eM), 4)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-9s %9s | %8.4f %8.4f %8.4f | %8s %8s | %8s %8s"
          % (name, format(len(mm), ","), free0, freeM, pinMC,
             cfd.get("cf"), cfd.get("cf_bnd"), cfd.get("cfa"), cfd.get("cfa_bnd")),
          flush=True)
fh.close()
