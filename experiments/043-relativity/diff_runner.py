"""DIFFERENTIAL-SEED CIRCUITS: fit a tri-amp circuit against the
VIRTUAL direction w_A - w_B for two same-site latents A and B.

The reconstruction target is the DIFFERENCE signal, so composition the
two concepts share cancels by construction and the fitted members are
the differentia -- what makes A fire that B does not. Probes are A's
own probe dataset with anchors at A's argmax (documented asymmetry: the
mirror B-anchored fit is the reverse question).

Scoring is in the MEAN-FILL pre-activation frame through the canonical
evaluator (a virtual direction has no top-k slot, so post-top-k reads
do not apply): EF = (read(circuit) - read(empty)) / (natural - read(empty)),
plus a necessity clamp arm and N_NULL random same-size member sets.

  DIFF="29:3736-4523" PYTHONPATH=src python .../diff_runner.py
  env: STEPS (400), N_NULL (2), LAM (1e-3)
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "src")
from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from circuit.instrument.learned_mask import run_learned_mask
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, upstream_sites)
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/"
                "Runs/20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP, PAIR = os.environ["DIFF"].split(":")
COMP = int(COMP)
LAT_A, LAT_B = (int(x) for x in PAIR.split("-"))
STEPS = int(os.environ.get("STEPS", 400))
N_NULL = int(os.environ.get("N_NULL", 2))
LAM = float(os.environ.get("LAM", 1e-3))
N_SEQ, N_TRAIN, EVAL_BS = 64, 48, 16

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(),
               compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.floor_source = "posctx"
cfg = disc.learned_mask


def main():
    layer, ki = split_component_idx(COMP, n_kinds)
    kind = bank.kinds[ki]
    sae = bank.saes[kind][layer]
    w_v = (sae.encoder.weight[LAT_A] - sae.encoder.weight[LAT_B]).detach()
    b_v = (sae._get_bias_eff()[LAT_A] - sae._get_bias_eff()[LAT_B]).detach()

    avg = torch.zeros((bank.n_layer * n_kinds, bank.d_sae),
                      device=bank.device)
    m0 = _build_mode_method("counterfactual_gradient", "local", inference,
                            bank, avg, pb)
    pd_ = m0.build_probe_dataset(COMP, LAT_A)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TRAIN], pa[:N_TRAIN], nt[:N_TRAIN]
    pt_ho, pa_ho = pt[N_TRAIN:], pa[N_TRAIN:]
    UP = sorted(upstream_sites(bank, layer, kind))
    means_tr, _ = collect_site_anchors(inference, bank, pt_tr, set(UP),
                                       pa_tr, pin_position_specific=False)

    def read(al):
        keep = {st: set(d) for st, d in al.items() if d}
        scales = {}
        for st, d in al.items():
            if not d:
                continue
            v = torch.ones(bank.d_sae)
            for i, a in d.items():
                v[int(i)] = float(a)
            scales[st] = v
        return float(circuit_only_activation(
            inference, bank, keep, UP, pt_ho, layer, kind, LAT_A,
            pos_argmax=pa_ho, site_means=means_tr, keep_scales=scales,
            preact=True, seed_vector=(w_v, b_v), batch_size=EVAL_BS))

    nat = float(circuit_only_activation(
        inference, bank, {}, set(), pt_ho, layer, kind, LAT_A,
        pos_argmax=pa_ho, preact=True, seed_vector=(w_v, b_v),
        batch_size=EVAL_BS))
    eM = float(circuit_only_activation(
        inference, bank, {}, UP, pt_ho, layer, kind, LAT_A,
        pos_argmax=pa_ho, site_means=means_tr, preact=True,
        seed_vector=(w_v, b_v), batch_size=EVAL_BS))
    denom = nat - eM
    print("virtual seed c%d/%d-minus-%d | natural %.3f | eM %.3f | "
          "denom %.3f" % (COMP, LAT_A, LAT_B, nat, eM, denom), flush=True)
    assert abs(denom) > 1e-6, "vacuous differential"

    t0 = time.time()
    scores, prov = run_learned_mask(
        inference, bank, objective="pos", sites=UP, seed_layer=layer,
        seed_kind=kind, seed_latent_idx=LAT_A, seed_vector=(w_v, b_v),
        pos_tokens=pt_tr, pos_argmax=pa_tr, neg_tokens=nt_tr,
        mask_floor_source="triple", dual_floor_weight=cfg.dual_floor_weight,
        triple_floor_weight=0.05, free_amplitude=True, steps=STEPS,
        lr=cfg.lr, l1_lambda=LAM, keep_threshold=cfg.keep_threshold,
        batch_size=disc.probe_batch_size, holdout_frac=cfg.holdout_frac,
        log_every=0, deep_site_threshold=cfg.deep_site_threshold,
        deep_batch_size=cfg.deep_batch_size, optimizer=cfg.optimizer,
        weight_decay=cfg.weight_decay, code_dtype=cfg.code_dtype,
        binarize=cfg.binarize, theta_init=cfg.theta_init,
        lr_schedule=cfg.lr_schedule, lr_min_frac=cfg.lr_min_frac,
        warmup_frac=cfg.warmup_frac)
    ak = prov.get("amp_kept") or {}
    alphas = {}
    for k, d in ak.items():
        lyr, knd = k.split("/")
        alphas[(int(lyr), knd)] = {int(i): float(v) for i, v in d.items()}
    n = sum(len(d) for d in alphas.values())
    ef = (read(alphas) - eM) / denom
    print("diffamp%d n=%d EF=%.3f (%.0fs)"
          % (STEPS, n, ef, time.time() - t0), flush=True)

    import random
    rng = random.Random(5)
    live = {st: list(range(bank.d_sae)) for st in alphas}
    nulls = []
    for j in range(N_NULL):
        na = {st: {rng.randrange(bank.d_sae): 1.0 for _ in range(len(d))}
              for st, d in alphas.items()}
        nulls.append((read(na) - eM) / denom)
        print("  null%d EF=%.3f" % (j, nulls[-1]), flush=True)

    out = {"comp": COMP, "lat_a": LAT_A, "lat_b": LAT_B, "n": n,
           "EF": round(ef, 4), "natural": round(nat, 3),
           "eM": round(eM, 3), "nulls": [round(x, 4) for x in nulls],
           "alphas": {"%d/%s" % s: {str(i): round(a, 4)
                                    for i, a in d.items()}
                      for s, d in alphas.items()}}
    with (HERE / "diff_circuits.jsonl").open("a") as fh:
        fh.write(json.dumps(out) + "\n")
    print("-> diff_circuits.jsonl", flush=True)


if __name__ == "__main__":
    main()
