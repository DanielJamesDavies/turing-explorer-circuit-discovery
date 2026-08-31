"""A/B: uniform theta init vs probe-active init.

Uniform init at theta_init=4.0 spends the first ~80-100 steps marching every
probe-INACTIVE latent (zero data gradient under the zero floor; the vast
majority of n_sites * d_sae) across the keep threshold by L1 pressure alone -
the measured plateau where n = the entire dictionary at steps 25-50.

theta_init_mode="active" starts those latents at theta_lo=-4 instead. Under
the zero floor that deletes the burn-in without touching anything the data
term can see. Under the DUAL floor (tested here, since it is the current
method) inactive latents DO carry gradient - masking them injects the floor
value - so this is a genuine prior and the A/B must check final quality, not
just speed.

Reads per arm: n at checkpoints (does the plateau vanish?), final n, free0 /
freeN_topk / freeM_topk on fixed eval negatives, wall clock.

  PYTHONPATH=src python .../init_ab.py
"""
import json
import os
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.floors import collect_site_means
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
import circuit.instrument.learned_mask as lm

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ, EVAL_BS, GAMMA, LAMBDA = 64, 16, 0.25, 1e-5
CHECKPOINTS = (25, 50, 100, 200, 300, 400)
# one seed per component, the ones with stored trajectories for cross-checking
TARGETS = [(8, 17043), (25, 4085), (32, 36965)]
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, bank.d_sae), device=bank.device)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.magnitude_prune = False
disc.recurrence_prune = False

_orig_step = torch.optim.AdamW.step
state = {"i": 0, "snaps": {}}


def _step(self, *a, **kw):
    out = _orig_step(self, *a, **kw)
    state["i"] += 1
    if state["i"] in CHECKPOINTS:
        n = 0
        for g in self.param_groups:
            for p in g["params"]:
                n += int((torch.sigmoid(p.detach()) > 0.5).sum())
        state["snaps"][state["i"]] = n
    return out


torch.optim.AdamW.step = _step

OUT = HERE / "init_ab.jsonl"
fh = OUT.open("a")
for comp, latent in TARGETS:
    layer, ki = split_component_idx(comp, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))
    meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                              avg_acts, probe_builder)
    pd_ = meth.build_probe_dataset(comp, latent)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind,
                                          latent, pa, batch_size=EVAL_BS))
    means_up = collect_site_means(inference, bank, pt, set(up))
    means_neg = collect_site_means(inference, bank, nt, set(up))

    def empty(sm=None, tk=False):
        return float(circuit_only_activation(
            inference, bank, {}, up, pt, layer, kind, latent, pos_argmax=pa,
            site_means=sm, batch_size=EVAL_BS, respect_topk=tk))

    a_e0, a_eMT, a_eNT = empty(), empty(means_up, True), empty(means_neg, True)
    print("\n=== comp %d latent %d (L%d %s, %d sites) ==="
          % (comp, latent, layer, kind, len(up)), flush=True)

    for mode in ("uniform", "active"):
        state["i"] = 0; state["snaps"] = {}
        t0 = time.perf_counter()
        scores, prov = lm.run_learned_mask(
            inference, bank, objective="pos", sites=up,
            seed_layer=layer, seed_kind=kind, seed_latent_idx=latent,
            pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
            mask_floor_source="dual", dual_floor_weight=GAMMA,
            steps=400, lr=0.05, l1_lambda=LAMBDA, keep_threshold=0.5,
            batch_size=4, holdout_frac=0.25, log_every=0,
            deep_site_threshold=disc.learned_mask.deep_site_threshold,
            deep_batch_size=disc.learned_mask.deep_batch_size,
            optimizer="adamw", weight_decay=0.05,
            code_dtype=disc.learned_mask.code_dtype,
            theta_init_mode=mode)
        secs = time.perf_counter() - t0
        keep = {}
        for f in scores:
            keep.setdefault((f.layer, f.kind), set()).add(int(f.index))
        n = len(scores)

        def phi(a_e, sm=None, tk=False):
            a_c = float(circuit_only_activation(
                inference, bank, keep, up, pt, layer, kind, latent,
                pos_argmax=pa, site_means=sm, batch_size=EVAL_BS,
                respect_topk=tk)) if n else a_e
            d = a_pos - a_e
            return round((a_c - a_e) / d, 4) if abs(d) > 1e-9 else None

        row = {"comp_idx": comp, "latent": latent, "mode": mode,
               "snaps": dict(state["snaps"]), "n": n,
               "free0": phi(a_e0), "freeN_topk": phi(a_eNT, means_neg, True),
               "freeM_topk": phi(a_eMT, means_up, True),
               "secs": round(secs, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-8s n@25=%-9s n@100=%-9s n@400=%-9s | free0=%-8s "
              "freeN_tk=%-8s freeM_tk=%-8s | %.0fs"
              % (mode, format(row["snaps"].get(25, 0), ","),
                 format(row["snaps"].get(100, 0), ","), format(n, ","),
                 row["free0"], row["freeN_topk"], row["freeM_topk"], secs),
              flush=True)
        del scores
        torch.cuda.empty_cache()
fh.close()
torch.optim.AdamW.step = _orig_step
print("\nwrote init_ab.jsonl")
