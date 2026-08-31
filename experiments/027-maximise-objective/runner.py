"""objective='maximise' on the L2 and L8 panels — a DRIVER object.

`pos` reproduces the seed's natural per-sequence level; `maximise` has no
target at all and drives it as high as the mask can, with L1 pricing each
latent kept. It is the unbounded limit of `raise`.

Because it is a driver objective, free0 is the WRONG headline eval — a
set that overshoots the seed reads badly there (recursive-map RESULT 6b:
free0 is a threshold test, and post-top-k censoring makes overshoot
invisible). So this reports:

  mult     achieved multiple of natural PRE-activation (what it optimised)
  free0    for reference only, expected poor
  cf/sup   the driver gates — inject into negative contexts (sufficiency)
           and ablate within an intact model (necessity)

Same seeds and lambdas as experiments/025-logit-endpoint, whose
`pos` rows are the matched comparison arm — no need to re-run those.

  COMP_IDX=8  PYTHONPATH=src python experiments/027-maximise-objective/runner.py
  COMP_IDX=26 ...
"""
import json
import os
import random
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.learned_mask import _at, _natural, run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from store.circuits import Circuit, CircuitNode

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP_IDX = int(os.environ.get("COMP_IDX", 8))
N_SEEDS = int(os.environ.get("N_SEEDS", 4))
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
LAMBDAS = [1e-5, 1e-3, 1e-2, 1e-1]
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, D_SAE), device=bank.device)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.floor_source = "posctx"
cfg = disc.learned_mask

LAYER, KI = split_component_idx(COMP_IDX, n_kinds)
KIND = bank.kinds[KI]
UP = sorted(upstream_sites(bank, LAYER, KIND))
SCOPE = len(UP) * D_SAE

_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu",
                   weights_only=False)
_pool = [int(c["latent_idx"]) for c in _cand if int(c["comp_idx"]) == COMP_IDX]
random.Random(42).shuffle(_pool)
SEEDS = sorted(_pool[:32])[:N_SEEDS]
del _cand
print("L%d %s | %d seeds | %d upstream sites | scope %d"
      % (LAYER, KIND, len(SEEDS), len(UP), SCOPE), flush=True)

TAG = "" if COMP_IDX == 8 else "_c%d" % COMP_IDX
fh = (HERE / ("rows%s.jsonl" % TAG)).open("a")
for sl in SEEDS:
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, pb)
    pd_ = m0.build_probe_dataset(COMP_IDX, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    sae = bank.saes[KIND][LAYER]
    w_seed = sae.encoder.weight[sl].detach()
    b_seed = sae._get_bias_eff()[sl].detach()
    nat_pre = float(_at(_natural(inference, bank, pt, LAYER, KIND,
                                 w_seed, b_seed), pa).mean())
    a_pos = float(measure_seed_activation(inference, bank, pt, LAYER, KIND, sl,
                                          pa, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, UP, pt, LAYER,
                                         KIND, sl, pos_argmax=pa,
                                         batch_size=EVAL_BS))
    den = a_pos - a_e0
    print("\n[%d] a_pos %.3f | natural pre-act %.3f" % (sl, a_pos, nat_pre),
          flush=True)

    for lam in LAMBDAS:
        t0 = time.time()
        try:
            scores, prov = run_learned_mask(
                inference, bank, objective="maximise", sites=UP,
                seed_layer=LAYER, seed_kind=KIND, seed_latent_idx=sl,
                pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
                # dual floor is pos-only by construction
                mask_floor_source="zero",
                binarize=cfg.binarize, steps=cfg.steps, lr=cfg.lr,
                l1_lambda=lam, keep_threshold=cfg.keep_threshold,
                batch_size=disc.probe_batch_size,
                holdout_frac=cfg.holdout_frac, theta_init=cfg.theta_init,
                log_every=0, deep_site_threshold=cfg.deep_site_threshold,
                deep_batch_size=cfg.deep_batch_size, optimizer=cfg.optimizer,
                weight_decay=cfg.weight_decay, code_dtype=cfg.code_dtype,
                lr_schedule=cfg.lr_schedule, lr_min_frac=cfg.lr_min_frac,
                warmup_frac=cfg.warmup_frac)
        except Exception as e:
            print("  l1=%-7g FAILED %s: %s" % (lam, type(e).__name__, e),
                  flush=True)
            continue
        mem = sorted({(f.layer, f.kind, int(f.index)) for f in scores
                      if (f.layer, f.kind) in set(UP)})
        keep = {}
        for l, k, i in mem:
            keep.setdefault((l, k), set()).add(i)
        c_pre = float(circuit_only_activation(
            inference, bank, keep, UP, pt, LAYER, KIND, sl, pos_argmax=pa,
            batch_size=EVAL_BS, preact=True))
        f0 = float(circuit_only_activation(
            inference, bank, keep, UP, pt, LAYER, KIND, sl, pos_argmax=pa,
            batch_size=EVAL_BS))
        circ = Circuit(name="max")
        for (l, kd), idx in ((s, i) for s, v in keep.items() for i in v):
            circ.add_node(CircuitNode(metadata={
                "layer_idx": l, "kind": kd, "latent_idx": idx,
                "role": "ablation_support"}))
        try:
            cf_v, sup_v = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, circ, neg_tokens=nt, pos_tokens=pt,
                seed_layer=LAYER, seed_kind=KIND, seed_latent_idx=sl,
                pos_argmax=pa, circuit_layers={l for (l, _) in keep})
            cf_v, sup_v = round(float(cf_v), 4), round(float(sup_v), 4)
        except Exception:
            cf_v = sup_v = None
        row = {"latent": sl, "layer": LAYER, "l1": lam, "n": len(mem),
               "pct_scope": round(100.0 * len(mem) / SCOPE, 4),
               "mult": round(c_pre / max(nat_pre, 1e-9), 3),
               "circuit_preact": round(c_pre, 3), "nat_preact": round(nat_pre, 3),
               "free0": round((f0 - a_e0) / den, 4) if abs(den) > 1e-9 else None,
               "cf": cf_v, "sup": sup_v,
               "holdout": prov.get("holdout_data_loss"),
               "secs": round(time.time() - t0, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  l1=%-7g n=%-7d mult=%-9.2f free0=%-8s cf=%-8s sup=%-8s %.0fs"
              % (lam, len(mem), row["mult"], row["free0"], cf_v, sup_v,
                 row["secs"]), flush=True)
        torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
