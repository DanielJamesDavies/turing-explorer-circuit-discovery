"""What does stopping an anneal-400 run at step 200 actually give you?

Truncation keeps the 400-run's LOW per-step pressure (lambda 1e-5 - the
low-churn regime) but forfeits half the total pressure (circuit ~2x large)
and stops with the gate half-soft (T = 0.05^(200/399) ~ 0.22, so the
binary-aligned forward that motivates anneal has not happened yet).

One training run per seed; membership snapshotted at steps 200 and 400 from
the optimiser's own parameters and both evaluated. Bit-determinism makes the
step-200 snapshot IDENTICAL to a truncated run.

  PYTHONPATH=src python experiments/011-binarize-sweep/anneal_truncate.py
"""
import json
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
N_SEQ, EVAL_BS = 64, 16
CHECKPOINTS = (200, 400)
TARGETS = [(8, 17043, "L2-resid"), (25, 4085, "L8-mlp"), (32, 36965, "L10-resid")]
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
lmcfg = disc.learned_mask

_orig_step = torch.optim.AdamW.step
cap = {"i": 0, "sites": None, "snaps": {}, "flips": [], "prev": None}


def _step(self, *a, **kw):
    out = _orig_step(self, *a, **kw)
    cap["i"] += 1
    params = self.param_groups[0]["params"]
    hard = torch.cat([(torch.sigmoid(p.detach()) > 0.5).flatten()
                      for p in params])
    if cap["prev"] is not None:
        cap["flips"].append(int((hard != cap["prev"]).sum()))
    cap["prev"] = hard
    if cap["i"] in CHECKPOINTS:
        cap["snaps"][cap["i"]] = {
            site: (torch.sigmoid(p.detach()) > 0.5).nonzero(as_tuple=True)[0].cpu()
            for site, p in zip(cap["sites"], params)}
    return out


torch.optim.AdamW.step = _step

OUT = HERE / "anneal_truncate.jsonl"
fh = OUT.open("a")
for comp, latent, label in TARGETS:
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

    cap["i"] = 0; cap["sites"] = up; cap["snaps"] = {}
    cap["flips"] = []; cap["prev"] = None
    t0 = time.perf_counter()
    scores, _ = lm.run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=latent,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
        mask_floor_source=lmcfg.mask_floor_source,
        dual_floor_weight=lmcfg.dual_floor_weight,
        steps=400, lr=lmcfg.lr, l1_lambda=1e-5, keep_threshold=0.5,
        batch_size=4, holdout_frac=lmcfg.holdout_frac, log_every=0,
        deep_site_threshold=lmcfg.deep_site_threshold,
        deep_batch_size=lmcfg.deep_batch_size,
        optimizer=lmcfg.optimizer, weight_decay=lmcfg.weight_decay,
        code_dtype=lmcfg.code_dtype, binarize="anneal")
    secs = time.perf_counter() - t0
    del scores
    torch.cuda.empty_cache()

    print("\n=== %s (%.0fs training) ===" % (label, secs), flush=True)
    for step in CHECKPOINTS:
        keep = {site: set(idx.tolist()) for site, idx in cap["snaps"][step].items()
                if idx.numel()}
        n = sum(len(v) for v in keep.values())

        def phi(a_e, sm=None, tk=False):
            a_c = float(circuit_only_activation(
                inference, bank, keep, up, pt, layer, kind, latent,
                pos_argmax=pa, site_means=sm, batch_size=EVAL_BS,
                respect_topk=tk)) if n else a_e
            d = a_pos - a_e
            return round((a_c - a_e) / d, 4) if abs(d) > 1e-9 else None

        window = cap["flips"][step - 26:step - 1]
        row = {"comp_idx": comp, "latent": latent, "label": label,
               "stop_at": step, "n": n, "free0": phi(a_e0),
               "freeN_topk": phi(a_eNT, means_neg, True),
               "freeM_topk": phi(a_eMT, means_up, True),
               "flips_last25": (sum(window) / max(len(window), 1)),
               "temperature": round(0.05 ** (min(step, 400) / 399), 4)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  stop@%d  T=%.3f n=%-9s free0=%-8s freeN_tk=%-8s "
              "freeM_tk=%-8s | last25 flips %.0f/step"
              % (step, row["temperature"], format(n, ","), row["free0"],
                 row["freeN_topk"], row["freeM_topk"], row["flips_last25"]),
              flush=True)
fh.close()
torch.optim.AdamW.step = _orig_step
print("\ndone")
