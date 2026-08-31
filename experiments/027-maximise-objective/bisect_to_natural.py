"""Prune each objective to free0 == 1 and compare SIZE at matched fidelity.

`pos` approaches natural from below — free0 falls monotonically with
lambda (0.985 -> 0.897 -> 0.399), so its free0~1 point is simply its
loosest lambda, ~2,400 nodes at L2.

`maximise` approaches from ABOVE: it overshoots (free0 2-8 at usable
lambda) and crosses 1.0 as sparsity pressure rises. Its crossing point
looked like a few hundred nodes on the coarse grid.

If the crossing really is smaller, then ranking latents by CONTRIBUTION
and cutting until you land on natural is a better selection principle
than penalising deviation in both directions — same fidelity, smaller
circuit. That is the claim this run tests.

Bisects log10(lambda) per seed per objective against free0 = 1.0.
Degenerate maximise solutions (mult >> 1 with free0 == 0, the
off-manifold exploit) read as free0 = 0 and so push the search back
towards lower lambda, which is the correct direction — the search
avoids that regime by construction.

  COMP_IDX=8 PYTHONPATH=src python experiments/027-maximise-objective/bisect_to_natural.py
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
STEPS_BISECT = int(os.environ.get("STEPS_BISECT", 7))
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
TARGET = 1.0
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
print("L%d %s | %d seeds | %d sites | target free0=%.2f | %d bisection steps"
      % (LAYER, KIND, len(SEEDS), len(UP), TARGET, STEPS_BISECT), flush=True)

TAG = "" if COMP_IDX == 8 else "_c%d" % COMP_IDX
fh = (HERE / ("bisect%s.jsonl" % TAG)).open("a")
for sl in SEEDS:
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, pb)
    pd_ = m0.build_probe_dataset(COMP_IDX, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    sae = bank.saes[KIND][LAYER]
    nat_pre = float(_at(_natural(inference, bank, pt, LAYER, KIND,
                                 sae.encoder.weight[sl].detach(),
                                 sae._get_bias_eff()[sl].detach()), pa).mean())
    a_pos = float(measure_seed_activation(inference, bank, pt, LAYER, KIND, sl,
                                          pa, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, UP, pt, LAYER,
                                         KIND, sl, pos_argmax=pa,
                                         batch_size=EVAL_BS))
    den = a_pos - a_e0
    print("\n[%d] a_pos %.3f" % (sl, a_pos), flush=True)

    def evaluate(objective, lam):
        kw = dict(sites=UP, seed_layer=LAYER, seed_kind=KIND,
                  seed_latent_idx=sl, pos_tokens=pt, pos_argmax=pa,
                  neg_tokens=nt, binarize=cfg.binarize, steps=cfg.steps,
                  lr=cfg.lr, l1_lambda=lam, keep_threshold=cfg.keep_threshold,
                  batch_size=disc.probe_batch_size,
                  holdout_frac=cfg.holdout_frac, theta_init=cfg.theta_init,
                  log_every=0, deep_site_threshold=cfg.deep_site_threshold,
                  deep_batch_size=cfg.deep_batch_size,
                  optimizer=cfg.optimizer, weight_decay=cfg.weight_decay,
                  code_dtype=cfg.code_dtype, lr_schedule=cfg.lr_schedule,
                  lr_min_frac=cfg.lr_min_frac, warmup_frac=cfg.warmup_frac)
        # dual floor is pos-only by construction; keep pos on the house
        # recipe and give maximise the single zero floor
        kw["mask_floor_source"] = (cfg.mask_floor_source if objective == "pos"
                                   else "zero")
        if objective == "pos":
            kw["dual_floor_weight"] = cfg.dual_floor_weight
        scores, _ = run_learned_mask(inference, bank, objective=objective, **kw)
        mem = sorted({(f.layer, f.kind, int(f.index)) for f in scores
                      if (f.layer, f.kind) in set(UP)})
        keep = {}
        for l, k, i in mem:
            keep.setdefault((l, k), set()).add(i)
        f0 = float(circuit_only_activation(
            inference, bank, keep, UP, pt, LAYER, KIND, sl, pos_argmax=pa,
            batch_size=EVAL_BS))
        pre = float(circuit_only_activation(
            inference, bank, keep, UP, pt, LAYER, KIND, sl, pos_argmax=pa,
            batch_size=EVAL_BS, preact=True))
        torch.cuda.empty_cache()
        return (mem, keep,
                (f0 - a_e0) / den if abs(den) > 1e-9 else 0.0,
                pre / max(nat_pre, 1e-9))

    for objective in ("pos", "maximise"):
        t0 = time.time()
        lo, hi = -6.0, -1.0          # log10 lambda; lo = loose, hi = tight
        best = None
        for it in range(STEPS_BISECT):
            mid = (lo + hi) / 2
            lam = 10 ** mid
            mem, keep, f0, mult = evaluate(objective, lam)
            if best is None or abs(f0 - TARGET) < abs(best[2] - TARGET):
                best = (lam, mem, f0, mult, keep)
            print("    %-9s it%d l1=%-9.2e n=%-7d free0=%-8.3f mult=%.2f"
                  % (objective, it, lam, len(mem), f0, mult), flush=True)
            if f0 > TARGET:
                lo = mid             # still above natural -> prune harder
            else:
                hi = mid
        lam, mem, f0, mult, keep = best
        circ = Circuit(name="b")
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
        row = {"latent": sl, "layer": LAYER, "objective": objective,
               "l1": lam, "n": len(mem),
               "pct_scope": round(100.0 * len(mem) / SCOPE, 4),
               "free0": round(f0, 4), "mult": round(mult, 3),
               "cf": cf_v, "sup": sup_v, "secs": round(time.time() - t0, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  => %-9s BEST l1=%.2e n=%-7d free0=%.3f cf=%s sup=%s (%.0fs)"
              % (objective, lam, len(mem), f0, cf_v, sup_v, row["secs"]),
              flush=True)

fh.close()
print("ALL DONE", flush=True)
