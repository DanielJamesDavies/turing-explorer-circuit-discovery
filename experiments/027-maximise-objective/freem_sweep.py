"""maximise vs pos on the FULL floor panel, including SFC's own metric.

freeM_dense is the SFC analogue: non-members at the task mean, members
RECOMPUTED from the degraded stream (so causal self-support is still
required), dense (no top-k constraint) — the file's docstring calls dense
mean-field ablation "SFC-standard".

    metric      floor        members        = SFC?
    free0       zero         recomputed     no (harsher floor)
    freeM_dense task mean    recomputed     YES
    freeM_topk  task mean    recomputed     no (we add the k-sparse constraint)
    pin0        zero         pinned clean   no (lenient on members)

Every metric is normalised against ITS OWN empty-circuit baseline, i.e.
(m(C) - m(empty)) / (m(full) - m(empty)) with the same floor throughout —
SFC's formula.

  COMP_IDX=8  PYTHONPATH=src python experiments/027-maximise-objective/freem_sweep.py
"""
import json
import os
import random
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.learned_mask import run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
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
HERE = Path(__file__).parent
COMP_IDX = int(os.environ.get("COMP_IDX", 8))
N_SEEDS = int(os.environ.get("N_SEEDS", 4))
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
LAMBDAS = [float(x) for x in os.environ.get(
    "LAMBDAS", "1e-5,1e-4,1e-3,3e-3,1e-2").split(",")]
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
print("L%d %s | %d seeds | %d sites | lambdas %s"
      % (LAYER, KIND, len(SEEDS), len(UP), LAMBDAS), flush=True)

TAG = "" if COMP_IDX == 8 else "_c%d" % COMP_IDX
fh = (HERE / ("freem%s.jsonl" % TAG)).open("a")
for sl in SEEDS:
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, pb)
    pd_ = m0.build_probe_dataset(COMP_IDX, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    a_pos = float(measure_seed_activation(inference, bank, pt, LAYER, KIND, sl,
                                          pa, batch_size=EVAL_BS))
    means, pins = collect_site_anchors(inference, bank, pt, set(UP), pa,
                                       pin_position_specific=False)

    def act(keep, **kw):
        return float(circuit_only_activation(
            inference, bank, keep, UP, pt, LAYER, KIND, sl, pos_argmax=pa,
            batch_size=EVAL_BS, **kw))

    # one empty-circuit baseline PER FLOOR — SFC's m(empty)
    e0 = act({})
    eM_d = act({}, site_means=means)
    eM_t = act({}, site_means=means, respect_topk=True)
    eP = act({}, pin_values=pins)
    print("\n[%d] a_pos %.3f | empty: zero %.3f meanD %.3f meanTK %.3f pin %.3f"
          % (sl, a_pos, e0, eM_d, eM_t, eP), flush=True)

    def norm(v, e):
        return round((v - e) / (a_pos - e), 4) if abs(a_pos - e) > 1e-9 else None

    # ARMS: (label, objective, training floor). maximise appears TWICE —
    # once on the zero floor it was originally run with, once on the
    # posctx MEAN floor that matches how freeM_dense evaluates. The floor
    # defines what "dropping a latent" means during training: under zero,
    # dropping a suppressor deletes it and the seed rises, so the objective
    # is rewarded for deletion; under a mean floor the suppressor stays at
    # its typical value and that strategy earns nothing. Without this arm,
    # objective and training-ablation-semantics are confounded.
    for label, objective, floor in (("pos", "pos", cfg.mask_floor_source),
                                    ("max/zero", "maximise", "zero"),
                                    ("max/mean", "maximise", "posctx")):
        for lam in LAMBDAS:
            t0 = time.time()
            kw = dict(sites=UP, seed_layer=LAYER, seed_kind=KIND,
                      seed_latent_idx=sl, pos_tokens=pt, pos_argmax=pa,
                      neg_tokens=nt, binarize=cfg.binarize, steps=cfg.steps,
                      lr=cfg.lr, l1_lambda=lam,
                      keep_threshold=cfg.keep_threshold,
                      batch_size=disc.probe_batch_size,
                      holdout_frac=cfg.holdout_frac, theta_init=cfg.theta_init,
                      log_every=0,
                      deep_site_threshold=cfg.deep_site_threshold,
                      deep_batch_size=cfg.deep_batch_size,
                      optimizer=cfg.optimizer, weight_decay=cfg.weight_decay,
                      code_dtype=cfg.code_dtype, lr_schedule=cfg.lr_schedule,
                      lr_min_frac=cfg.lr_min_frac, warmup_frac=cfg.warmup_frac)
            kw["mask_floor_source"] = floor
            if objective == "pos":
                kw["dual_floor_weight"] = cfg.dual_floor_weight
            try:
                scores, _ = run_learned_mask(inference, bank,
                                             objective=objective, **kw)
            except Exception as e:
                print("  %-9s l1=%-8g FAILED %s" % (label, lam,
                                                    type(e).__name__), flush=True)
                continue
            mem = sorted({(f.layer, f.kind, int(f.index)) for f in scores
                          if (f.layer, f.kind) in set(UP)})
            keep = {}
            for l, k, i in mem:
                keep.setdefault((l, k), set()).add(i)
            f0 = norm(act(keep), e0)
            fmd = norm(act(keep, site_means=means), eM_d)
            fmt = norm(act(keep, site_means=means, respect_topk=True), eM_t)
            p0 = norm(act(keep, pin_values=pins), eP)
            circ = Circuit(name="f")
            for (l, kd), idx in ((s, i) for s, v in keep.items() for i in v):
                circ.add_node(CircuitNode(metadata={
                    "layer_idx": l, "kind": kd, "latent_idx": idx,
                    "role": "ablation_support"}))
            try:
                cf_v, sup_v = evaluate_counterfactual_faithfulness(
                    inference, bank, avg_acts, circ, neg_tokens=nt,
                    pos_tokens=pt, seed_layer=LAYER, seed_kind=KIND,
                    seed_latent_idx=sl, pos_argmax=pa,
                    circuit_layers={l for (l, _) in keep})
                cf_v, sup_v = round(float(cf_v), 4), round(float(sup_v), 4)
            except Exception:
                cf_v = sup_v = None
            row = {"latent": sl, "layer": LAYER, "objective": label,
                   "train_floor": floor,
                   "l1": lam, "n": len(mem),
                   "pct_scope": round(100.0 * len(mem) / SCOPE, 4),
                   "free0": f0, "freeM_dense": fmd, "freeM_topk": fmt,
                   "pin0": p0, "cf": cf_v, "sup": sup_v,
                   "secs": round(time.time() - t0, 1)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("  %-9s l1=%-8g n=%-7d free0=%-8s freeMd=%-8s freeMt=%-8s "
                  "pin0=%-8s cf=%-7s sup=%-7s %.0fs"
                  % (label, lam, len(mem), f0, fmd, fmt, p0, cf_v, sup_v,
                     row["secs"]), flush=True)
            torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
