"""Follow-up: 200-step anneal with a reach-then-hold schedule.

The compressed (200-step) anneal matched the 400-step metrics but ended with
2-2.7x the membership churn - the schedule spent all 200 steps DESCENDING and
hit the floor temperature only at the last step, leaving no time at final
sharpness to settle. anneal_reach_frac=0.7 reaches T=0.05 at step 140 and
holds it for the remaining 60.

Per seed: one run at the already-calibrated 200-step lambda (from
anneal_steps.jsonl's -cal arms), wd=0.10, reach=0.7. Judged against the
stored anneal-400 reference rows on metrics AND end-of-run flips.

  PYTHONPATH=src python experiments/011-binarize-sweep/anneal_hold.py
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
# lambda per seed = the calibrated 200-step values from anneal_steps -cal arms
TARGETS = [(8, 17043, "L2-resid", 2.24e-5),
           (25, 4085, "L8-mlp", 2.59e-5),
           (32, 36965, "L10-resid", 2.83e-5)]
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
cap = {"i": 0, "total": 0, "sites": None, "prev": None, "flips": []}


def _step(self, *a, **kw):
    out = _orig_step(self, *a, **kw)
    cap["i"] += 1
    params = self.param_groups[0]["params"]
    hard = torch.cat([(torch.sigmoid(p.detach()) > 0.5).flatten()
                      for p in params])
    if cap["prev"] is not None:
        cap["flips"].append(int((hard != cap["prev"]).sum()))
    cap["prev"] = hard
    return out


torch.optim.AdamW.step = _step

OUT = HERE / "anneal_steps.jsonl"
fh = OUT.open("a")
for comp, latent, label, lam in TARGETS:
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

    cap["i"] = 0; cap["total"] = 200; cap["sites"] = up
    cap["prev"] = None; cap["flips"] = []
    t0 = time.perf_counter()
    scores, _ = lm.run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=latent,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
        mask_floor_source=lmcfg.mask_floor_source,
        dual_floor_weight=lmcfg.dual_floor_weight,
        steps=200, lr=lmcfg.lr, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=lmcfg.holdout_frac, log_every=0,
        deep_site_threshold=lmcfg.deep_site_threshold,
        deep_batch_size=lmcfg.deep_batch_size,
        optimizer=lmcfg.optimizer, weight_decay=0.10,
        code_dtype=lmcfg.code_dtype, binarize="anneal",
        anneal_reach_frac=0.7)
    secs = time.perf_counter() - t0
    keep = {}
    for f in scores:
        keep.setdefault((f.layer, f.kind), set()).add(int(f.index))
    n = sum(len(v) for v in keep.values())
    del scores

    def phi(a_e, sm=None, tk=False):
        a_c = float(circuit_only_activation(
            inference, bank, keep, up, pt, layer, kind, latent, pos_argmax=pa,
            site_means=sm, batch_size=EVAL_BS, respect_topk=tk)) if n else a_e
        d = a_pos - a_e
        return round((a_c - a_e) / d, 4) if abs(d) > 1e-9 else None

    flips = list(cap["flips"])
    tail = flips[-25:] if flips else [0]
    row = {"comp_idx": comp, "latent": latent, "label": label,
           "arm": "anneal-200-hold", "steps": 200, "lambda": lam, "wd": 0.10,
           "anneal_reach_frac": 0.7, "n": n, "free0": phi(a_e0),
           "freeN_topk": phi(a_eNT, means_neg, True),
           "freeM_topk": phi(a_eMT, means_up, True), "flips": flips,
           "secs": round(secs, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-10s hold n=%-9s free0=%-8s freeN_tk=%-8s freeM_tk=%-8s | "
          "last25 flips %.0f/step | %.0fs"
          % (label, format(n, ","), row["free0"], row["freeN_topk"],
             row["freeM_topk"], sum(tail) / len(tail), secs), flush=True)
    torch.cuda.empty_cache()
fh.close()
torch.optim.AdamW.step = _orig_step
print("done")
