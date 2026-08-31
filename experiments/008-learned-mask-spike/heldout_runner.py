"""Held-out gate recovery: does the learned gate generalise to unseen negatives?

BLOCKING for the negctx/inject family. mask_negctx and mask_inject optimise
against a set of negative contexts; a learned method can memorise them in a
way one-shot attribution cannot. The v2 inject sweep made the risk concrete —
gate recovery 0.87 on TRAIN negatives with holdout data loss 10.9 — but that
holdout number is a loss, not a recovery, so it can't be compared with the
0.34 the mask_negctx sweep reported.

This measures the SAME quantity on both slices, externally:

  gate_recovery(slice) = (p_gate(slice) - p_nat(slice)) / (a_pos - p_nat(slice))

where p_gate is the ceteris-paribus knockout — every latent kept at its
natural value EXCEPT the learned edits, on the natural stream, measured in
PRE-ACTIVATION at each sequence's would-be-firing anchor. p_nat is that same
slice's untouched pre-activation. The optimiser's own split is read from
provenance (n_train_neg / n_holdout_neg) rather than re-derived.

A third slice, FRESH negatives retrieved independently (different selection
seed via neg_mode), tests generalisation beyond the probe set entirely — the
strongest version, since the holdout slice still comes from the same
retrieval.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/runner.py
"""
import json
import os
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.instrument.learned_mask import LearnedMaskPatcher, run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
SEEDS = {"L2": (8, 30122), "L8": (25, 10628), "L10": (32, 3021)}
TAG = os.environ.get("SEED_TAG", "L8")
SC_IDX, LATENT = SEEDS[TAG]
LAMBDAS = [1e-3, 1e-4, 1e-5]
N_SEQ, EVAL_BS, NK = 64, 16, 3
torch.set_float32_matmul_precision("high")

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
disc.probe_batch_size = 4

layer, ki = split_component_idx(SC_IDX, NK)
kind = bank.kinds[ki]
up = upstream_sites(bank, layer, kind)
sae = bank.saes[kind][layer]
w_seed = sae.encoder.weight[LATENT].detach()
b_seed = sae._get_bias_eff()[LATENT].detach()

m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                        avg_acts, probe_builder)
pd_ = m0.build_probe_dataset(SC_IDX, LATENT)
pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
nt = pd_.neg_tokens[:N_SEQ]
a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind, LATENT,
                                      pa, batch_size=EVAL_BS))

# FRESH negatives: an independent retrieval (random mode), disjoint from the
# probe dataset's stored close negatives wherever the pools differ.
fresh = None
try:
    cfmeth = _build_mode_method("counterfactual_gradient", "local", inference,
                                bank, avg_acts, probe_builder)
    cfmeth.neg_mode = "random"
    from observability.circuit_logger import CircuitLogger
    lg = CircuitLogger(SC_IDX, LATENT, "heldout_probe")
    fresh = cfmeth._get_neg_tokens(pd_, SC_IDX, LATENT, lg)
    if fresh is not None:
        fresh = fresh[:N_SEQ]
except Exception as exc:
    print("fresh-negative retrieval unavailable: %s: %s"
          % (type(exc).__name__, exc), flush=True)


def natural_pre(tokens):
    """(anchors, mean natural pre-activation at those anchors)."""
    p = LearnedMaskPatcher(bank, {}, layer, kind, w_seed, b_seed)
    inference.disable_compile()
    try:
        inference.forward(tokens, patcher=p, grad_enabled=False,
                          return_activations=False, tokenize_final=False)
    finally:
        inference.enable_compile()
    pre = p.seed_pre.detach()
    anc = pre.argmax(dim=-1)
    idx = torch.arange(pre.shape[0], device=pre.device)
    return anc.cpu(), float(pre[idx, anc].mean())


def gate_recovery(edits, tokens, anchors, p_nat):
    """Ceteris-paribus knockout: everything natural EXCEPT the learned edits."""
    keep = {s: set(range(bank.d_sae)) - edits.get(s, set()) for s in up}
    p_gate = float(circuit_only_activation(
        inference, bank, keep, up, tokens, layer, kind, LATENT,
        pos_argmax=anchors, batch_size=EVAL_BS, preact=True))
    d = a_pos - p_nat
    return p_gate, (round((p_gate - p_nat) / d, 4) if abs(d) > 1e-9 else None)


fh = (HERE / "rows.jsonl").open("a")
print("[%s] target %.4f | negatives %d" % (TAG, a_pos, nt.shape[0]), flush=True)
print("%-9s %8s | %19s | %19s | %19s"
      % ("lambda", "n_edit", "TRAIN  p_gate  rec", "HOLDOUT  p_gate  rec",
         "FRESH  p_gate  rec"), flush=True)

for lam in LAMBDAS:
    t0 = time.time()
    scores, prov = run_learned_mask(
        inference, bank, objective="negctx", sites=sorted(up),
        seed_layer=layer, seed_kind=kind, seed_latent_idx=LATENT,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt, target_act=a_pos,
        steps=200, lr=0.1, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=config.discovery.learned_mask.deep_site_threshold,
        deep_batch_size=config.discovery.learned_mask.deep_batch_size)
    edits = {}
    for fid in scores:
        edits.setdefault((fid.layer, fid.kind), set()).add(int(fid.index))

    # the optimiser's own split, read from provenance (never re-derived)
    n_tr = int(prov["n_train_neg"])
    slices = {"train": nt[:n_tr], "holdout": nt[n_tr:]}
    if fresh is not None and fresh.shape[0]:
        slices["fresh"] = fresh

    row = {"tag": TAG, "lambda": lam, "n_edit": len(scores),
           "n_train_neg": n_tr, "n_holdout_neg": int(prov["n_holdout_neg"]),
           "holdout_data_loss": prov["holdout_data_loss"],
           "a_pos": round(a_pos, 4), "secs": round(time.time() - t0, 1)}
    out = []
    for name, toks in slices.items():
        if toks is None or toks.shape[0] == 0:
            out.append("%19s" % "—")
            continue
        anc, p_nat = natural_pre(toks)
        p_gate, rec = gate_recovery(edits, toks, anc, p_nat)
        row["p_nat_%s" % name] = round(p_nat, 4)
        row["p_gate_%s" % name] = round(p_gate, 4)
        row["rec_%s" % name] = rec
        row["n_%s" % name] = int(toks.shape[0])
        out.append("%9.3f %9s" % (p_gate, rec))
    while len(out) < 3:
        out.append("%19s" % "—")
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-9g %8s | %19s | %19s | %19s"
          % (lam, format(len(scores), ","), out[0], out[1], out[2]), flush=True)
    torch.cuda.empty_cache()
fh.close()
print("\nwrote %s" % (HERE / "rows.jsonl"))
