"""Error-node mode smoke: do non-member SAE errors free-ride in our φ numbers?

Historically every φ preserves the SAE reconstruction error at EVERY upstream
site — error terms are invisible free members of every circuit. The new mode
(CircuitOnlyPatcher.keep_error_sites + collect_site_error_means) makes them
ablatable nodes, SFC-style. This measures, on L2 and L9 with a real
abl-ig_mean PA circuit:

  a_e0            empty circuit, errors preserved   (the historical anchor)
  a_e0_noerr      empty circuit, ALL errors zeroed  (denominator question:
                                                     did errors prop up empty?)
  free0           kept latents, errors preserved    (historical metric)
  free0_esites    errors kept ONLY at member sites, zeroed elsewhere
  free0_emean     errors kept ONLY at member sites, mean-filled elsewhere

If free0_esites ~= free0, non-member errors are inert at the latent endpoint
and the historical numbers stand as-is. If it drops, error nodes are
load-bearing members our circuits have been silently granted.

  PYTHONPATH=src python experiments/004-error-node-mode/smoke.py
"""
import json
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.floors import collect_site_error_means
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
OUT = HERE / "smoke_rows.jsonl"
SEEDS = [(8, 30122, "L2"), (27, 6859, "L9")]
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), D_SAE), device=bank.device)
n_kinds = len(bank.kinds)

_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = True
disc.position_aware_select = "abs_pctl"
disc.position_aware_threshold = 90.0
disc.floor_source = "posctx"
disc.magnitude_prune = False
disc.recurrence_prune = False
disc.min_faithfulness = -100.0
config.discovery.ablation_gradient.negative_roles = "include"

fh = OUT.open("a")
for sc_idx, sl, label in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    t0 = time.time()

    m = _build_mode_method("ablation_gradient", "ig_mean", inference, bank,
                           avg_acts, probe_builder)
    circ = m.discover(sc_idx, sl)
    if circ is None:
        print("[%s] no circuit" % seed_key, flush=True)
        continue
    pd_ = m.build_probe_dataset(sc_idx, sl)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]

    keep = {}
    for node in circ.nodes.values():
        f = node.feature_id
        if node.metadata.get("role") == "seed" or f is None:
            continue
        if (f.layer, f.kind) in up:
            keep.setdefault((f.layer, f.kind), set()).add(int(f.index))
    member_sites = set(keep)
    n_lat = sum(len(v) for v in keep.values())

    a_pos = measure_seed_activation(inference, bank, pt, layer, kind, sl, pa,
                                    batch_size=EVAL_BS)
    error_means = collect_site_error_means(inference, bank, pt, up)

    def act(k, es=None, em=None):
        return circuit_only_activation(
            inference, bank, k, up, pt, layer, kind, sl, pos_argmax=pa,
            batch_size=EVAL_BS, keep_error_sites=es, error_means=em)

    a_e0 = act({})
    a_e0_noerr = act({}, es=set())
    a_free = act(keep)
    a_free_esites = act(keep, es=member_sites)
    a_free_emean = act(keep, es=member_sites, em=error_means)
    # A raw PA union occupies EVERY upstream site, so member-site errors ==
    # all errors and the above trio degenerates. Two conditions that don't:
    a_free_noerr = act(keep, es=set())              # latents kept, ALL errors cut
    from eval.magnitude_prune import prune_by_magnitude_bisection
    from eval.recurrence_prune import prune_by_sequence_recurrence
    prune_by_sequence_recurrence(inference, bank, circ, pos_tokens=pt,
                                 neg_tokens=pd_.neg_tokens[:N_SEQ], min_sequences=2)
    prune_by_magnitude_bisection(inference, bank, circ, pos_tokens=pt,
                                 seed_layer=layer, seed_kind=kind,
                                 seed_latent_idx=sl, pos_argmax=pa, objective="free")
    keep_p = {}
    for node in circ.nodes.values():
        f = node.feature_id
        if node.metadata.get("role") == "seed" or f is None:
            continue
        if (f.layer, f.kind) in up:
            keep_p.setdefault((f.layer, f.kind), set()).add(int(f.index))
    psites = set(keep_p)
    n_lat_p = sum(len(v) for v in keep_p.values())
    a_free_p = act(keep_p)                          # pruned, errors preserved
    a_free_p_esites = act(keep_p, es=psites)        # errors only at pruned sites
    a_free_p_emean = act(keep_p, es=psites, em=error_means)

    def phi(a_c, a_e):
        den = float(a_pos) - float(a_e)
        return round((float(a_c) - float(a_e)) / den, 4) if abs(den) > 1e-9 else None

    row = {
        "seed": seed_key, "label": label, "layer": layer, "kind": kind,
        "arm": "abl-ig_mean PA", "up_nodes": n_lat,
        "n_member_sites": len(member_sites), "n_up_sites": len(up),
        "a_pos": round(float(a_pos), 4),
        "a_e0": round(float(a_e0), 4),
        "a_e0_noerr": round(float(a_e0_noerr), 4),
        "free0": phi(a_free, a_e0),
        "free0_esites": phi(a_free_esites, a_e0_noerr),
        "free0_emean": phi(a_free_emean, a_e0_noerr),
        "free0_noerr": phi(a_free_noerr, a_e0_noerr),
        "pruned_up_nodes": n_lat_p, "n_pruned_sites": len(psites),
        "free0_p": phi(a_free_p, a_e0),
        "free0_p_esites": phi(a_free_p_esites, a_e0_noerr),
        "free0_p_emean": phi(a_free_p_emean, a_e0_noerr),
        "ac_free": round(float(a_free), 4),
        "ac_free_esites": round(float(a_free_esites), 4),
        "ac_free_emean": round(float(a_free_emean), 4),
        "ac_free_noerr": round(float(a_free_noerr), 4),
        "ac_free_p": round(float(a_free_p), 4),
        "ac_free_p_esites": round(float(a_free_p_esites), 4),
        "ac_free_p_emean": round(float(a_free_p_emean), 4),
        "secs": round(time.time() - t0, 1),
    }
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("[%s] %s n=%d sites=%d/%d | a_pos %.3f a_e0 %.3f a_e0_noerr %.3f"
          % (seed_key, label, n_lat, len(member_sites), len(up), a_pos, a_e0,
             a_e0_noerr), flush=True)
    print("   raw    free0 %-7s esites %-7s emean %-7s NOERR %-7s"
          % (row["free0"], row["free0_esites"], row["free0_emean"],
             row["free0_noerr"]), flush=True)
    print("   pruned n=%d sites=%d/%d free0 %-7s esites %-7s emean %-7s (%.0fs)"
          % (n_lat_p, len(psites), len(up), row["free0_p"],
             row["free0_p_esites"], row["free0_p_emean"], row["secs"]), flush=True)
    del circ
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

fh.close()
print("wrote %s" % OUT)
