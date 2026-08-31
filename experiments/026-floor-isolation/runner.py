"""(1) Isolate the training floor for `pos`, and (2) archive membership.

EXPERIMENT 1. Every `pos` result in this project used mask_floor_source=
"dual", and the dual floor has never been isolated FOR POS. It matters
now because dual_floor_weight defaults to 0.25, so "dual" is weighted
4:1 toward the zero floor — it may be contributing almost nothing, or
its negctx quarter may be exactly what stops `pos` exploiting free0 the
way `maximise` does. Four floors, same objective, same everything else.

EXPERIMENT 2 (free from the same run). Membership is ARCHIVED per row.
We know the arms disagree on the metrics; we do not know whether they
disagree on the LATENTS. If pos and maximise at matched n share most of
their members, the objectives are arguing about weighting and margin; if
they share few, they are genuinely different objects. That determines
how much of this thread is about measurement versus mechanism.

Scored on the full panel — free0 / freeM_dense / freeM_topk / pin0 / cf /
sup — because the lesson of freem_sweep is that any single metric is
exploitable by whatever was trained on it.

  COMP_IDX=8 PYTHONPATH=src python experiments/026-floor-isolation/runner.py
"""
import gzip
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
    "LAMBDAS", "1e-5,1e-4,1e-3,3e-3").split(",")]
# (label, objective, training floor). ARMS_ONLY (comma-separated labels)
# runs a subset, so a newly added arm can be appended to an existing panel
# without re-running the arms already measured.
# 4th element = triple_floor_weight (the posctx term's gamma); None means
# "not a triple arm". weight 0 reproduces dual exactly, 0.25 is the house
# triple, so intermediate values interpolate between the two.
ALL_ARMS = [("pos/zero", "pos", "zero", None),
            ("pos/posctx", "pos", "posctx", None),
            ("pos/negctx", "pos", "negctx", None),
            ("pos/dual", "pos", "dual", None),
            ("pos/triple", "pos", "triple", 0.25),
            ("pos/tri.05", "pos", "triple", 0.05),
            ("pos/tri.10", "pos", "triple", 0.10),
            ("pos/tri.50", "pos", "triple", 0.50),
            ("pos/tri1.0", "pos", "triple", 1.00),
            ("pos/tri2.0", "pos", "triple", 2.00),
            ("pos/tri4.0", "pos", "triple", 4.00),
            ("pos/tri8.0", "pos", "triple", 8.00),
            ("amp/dual", "pos", "dual", None),
            ("amp/tri.10", "pos", "triple", 0.10),
            ("amp/tri2.0", "pos", "triple", 2.00),
            ("max/zero", "maximise", "zero", None),
            ("max/mean", "maximise", "posctx", None),
            ("max/triple", "maximise", "triple", 0.25)]
_only = os.environ.get("ARMS_ONLY")
ARMS = ([a for a in ALL_ARMS if a[0] in set(_only.split(","))] if _only
        else ALL_ARMS)
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
print("L%d %s | %d seeds | %d sites | %d arms | lambdas %s"
      % (LAYER, KIND, len(SEEDS), len(UP), len(ARMS), LAMBDAS), flush=True)

TAG = "" if COMP_IDX == 8 else "_c%d" % COMP_IDX
fh = (HERE / ("rows%s.jsonl" % TAG)).open("a")
mh = gzip.open(HERE / ("members%s.jsonl.gz" % TAG), "at")
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

    e0, eMd = act({}), act({}, site_means=means)
    eMt = act({}, site_means=means, respect_topk=True)
    eP = act({}, pin_values=pins)
    print("\n[%d] a_pos %.3f | empty: zero %.3f meanD %.3f meanTK %.3f pin %.3f"
          % (sl, a_pos, e0, eMd, eMt, eP), flush=True)

    def norm(v, e):
        return round((v - e) / (a_pos - e), 4) if abs(a_pos - e) > 1e-9 else None

    for label, objective, floor, tw in ARMS:
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
                      lr_min_frac=cfg.lr_min_frac, warmup_frac=cfg.warmup_frac,
                      mask_floor_source=floor)
            if floor in ("dual", "triple"):
                kw["dual_floor_weight"] = cfg.dual_floor_weight
            # "amp/" arms: free per-latent amplitude on top of the gate
            kw["free_amplitude"] = label.startswith("amp")
            if floor == "triple" and tw is not None:
                kw["triple_floor_weight"] = tw
            try:
                scores, prov = run_learned_mask(inference, bank,
                                                objective=objective, **kw)
            except Exception as e:
                print("  %-11s l1=%-8g FAILED %s: %s"
                      % (label, lam, type(e).__name__, e), flush=True)
                continue
            mem = sorted({(f.layer, f.kind, int(f.index)) for f in scores
                          if (f.layer, f.kind) in set(UP)})
            keep = {}
            for l, k, i in mem:
                keep.setdefault((l, k), set()).add(i)
            f0 = norm(act(keep), e0)
            fmd = norm(act(keep, site_means=means), eMd)
            fmt = norm(act(keep, site_means=means, respect_topk=True), eMt)
            p0 = norm(act(keep, pin_values=pins), eP)
            circ = Circuit(name="fi")
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
            # A large zero-floor empty baseline inflates free0's
            # denominator (a_pos - e0), which makes the [0.8, 1.25] band
            # map to an enormous range of raw activations — the criterion
            # goes VACUOUS rather than strict. L9 seed 1639 has e0 =
            # 14,248 against a_pos 21.1, so its free0 carries no
            # information. Flag it so ALL-PASS is not read as evidence.
            f0_vacuous = abs(a_pos - e0) > 5.0 * max(abs(a_pos), 1e-9)
            allpass = all(v is not None and 0.8 <= v <= 1.25
                          for v in (f0, fmd, fmt)) and (cf_v or 0) > 0.7
            allpass = allpass and not f0_vacuous
            row = {"latent": sl, "layer": LAYER, "arm": label,
                   "objective": objective, "train_floor": floor,
                   "triple_w": tw, "l1": lam,
                   "n": len(mem),
                   "pct_scope": round(100.0 * len(mem) / SCOPE, 4),
                   "free0": f0, "freeM_dense": fmd, "freeM_topk": fmt,
                   "pin0": p0, "cf": cf_v, "sup": sup_v, "all_pass": allpass,
                   "f0_vacuous": bool(f0_vacuous),
                   "amp_stats": prov.get("amp_stats"),
                   "secs": round(time.time() - t0, 1)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            mh.write(json.dumps({"latent": sl, "arm": label, "l1": lam,
                                 "n": len(mem),
                                 "members": [[l, k, i] for l, k, i in mem]})
                     + "\n")
            mh.flush()
            print("  %-11s l1=%-8g n=%-7d f0=%-8s fmd=%-8s fmt=%-8s pin0=%-8s "
                  "cf=%-7s%s %.0fs"
                  % (label, lam, len(mem), f0, fmd, fmt, p0, cf_v,
                     "  ALL-PASS" if allpass else "", row["secs"]), flush=True)
            torch.cuda.empty_cache()

fh.close(); mh.close()
print("ALL DONE", flush=True)
