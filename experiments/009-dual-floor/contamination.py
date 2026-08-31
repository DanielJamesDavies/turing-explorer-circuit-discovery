"""PHASE 1: how contaminated is each negctx mode, measured PRE-TOP-K?

A "negative" is normally accepted on its POST-TOP-K seed value, which
target_latent_activations pins to exactly 0 whenever the seed misses top-k. A
sequence where the seed very nearly fired is therefore indistinguishable from
one where it is silent. Daniel's hypothesis: "close" negatives are the ones
most likely to nearly fire, so close is the most contaminated mode — which
would explain why random negatives have repeatedly beaten close.

This tests the PREMISE directly, with no circuit involved: select negatives
the normal (unfiltered) way, then measure what their seed activation really
is pre-top-k, as a fraction of the seed's own posctx reference.

It also SIZES THE THRESHOLD. preact_max_frac defaults to 0.1, which is a
guess; the quantiles here say what it should actually be, per mode and per
depth. Run this before committing the number.

Reports per (seed, mode): the post-top-k value the current filter sees (~0 by
construction) against the pre-top-k truth, plus the share of negatives above
5 / 10 / 25% of the posctx reference.

  PYTHONPATH=src python experiments/009-dual-floor/contamination.py
"""
import json
import os
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ = 64
MODES = ("store", "close", "random", "distant")
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
disc.probe_batch_size = 4
disc.position_aware = False

# Same layer-stratified sampler (seed=42) as every frozen run, so these seeds
# are the same objects as indices 2/5/8/10 of the 16-seed matrix.
import random
from collections import defaultdict


def _layer_stratified_indices(candidates, sample_size, seed=42):
    by_layer = defaultdict(list)
    for index, cand in enumerate(candidates):
        by_layer[int(cand["comp_idx"]) // n_kinds].append(index)
    rng = random.Random(seed)
    for layer in by_layer:
        rng.shuffle(by_layer[layer])
    out = []
    for rank in range(max(len(v) for v in by_layer.values())):
        for layer in sorted(by_layer):
            if rank < len(by_layer[layer]):
                out.append(by_layer[layer][rank])
                if len(out) >= sample_size:
                    return out
    return out


_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu", weights_only=False)
_idx = _layer_stratified_indices(_cand, 16)
WANT = [int(x) for x in os.environ.get("SEED_IDS", "2,5,8,10").split(",")]
SEEDS = [(i, int(_cand[_idx[i]]["comp_idx"]), int(_cand[_idx[i]]["latent_idx"]))
         for i in WANT]

method = _build_mode_method("ablation_gradient", "mask", inference, bank,
                            avg_acts, probe_builder)
selector = method._neg_context_selector()
cfg = disc.neg_context_selection

rows = []
print("posctx_ref = the seed's MEDIAN pre-top-k activation on its positives.")
print("post_med   = what the CURRENT filter sees (0 == looks perfectly silent).")
print("pre_*      = the uncensored truth, as %% of posctx_ref.\n")
for seed_i, sc_idx, sl in SEEDS:
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    pd_ = method.probe_builder.build(sc_idx, sl) if hasattr(method.probe_builder, "build") \
        else method.build_probe_dataset(sc_idx, sl)
    pt = pd_.pos_tokens[:N_SEQ]
    if pt.shape[0] == 0:
        print("[seed %d] no positives — skip" % seed_i, flush=True)
        continue
    ref = selector.posctx_reference(pt, sc_idx, sl,
                                    batch_size=int(cfg.filter_batch_size))
    print("=== seed %d (%d/%d) L%d %s | posctx_ref %.4f ==="
          % (seed_i, sc_idx, sl, layer, kind, ref or 0.0), flush=True)
    print("%-9s %6s %9s %9s %9s %9s %8s %8s %8s"
          % ("mode", "n", "post_med", "pre_med", "pre_p90", "pre_max",
             ">5%", ">10%", ">25%"), flush=True)
    for mode in MODES:
        if mode == "store":
            neg = pd_.neg_tokens[:N_SEQ]
        else:
            sel = selector.select(
                sc_idx, sl, mode, max_sequences=N_SEQ,
                batch_size=max(1, int(disc.probe_batch_size)),
                candidate_pool_size=(method.distant_pool_size if mode == "distant"
                                     else cfg.candidate_pool_size),
                exact=bool(cfg.exact_negctx_ranking),
                non_activation_threshold=float(cfg.non_activation_threshold),
                preact_filter=False,          # measuring what gets through TODAY
                selection_seed=int(cfg.selection_seed),
                filter_batch_size=int(cfg.filter_batch_size),
                load_window_size=int(cfg.load_window_size),
                logger=None)
            neg = sel.tokens if sel is not None else torch.zeros(0, dtype=torch.long)
        if neg.shape[0] == 0:
            print("%-9s %6d   (none selected)" % (mode, 0), flush=True)
            continue
        post = selector.collect_seed_max_activations(
            neg, sc_idx, sl, batch_size=int(cfg.filter_batch_size))
        pre = selector.collect_seed_max_activations(
            neg, sc_idx, sl, batch_size=int(cfg.filter_batch_size), preact=True)
        frac = (pre / max(ref or 1e-9, 1e-9))
        row = {"seed_i": seed_i, "seed": "%d/%d" % (sc_idx, sl), "layer": layer,
               "kind": kind, "mode": mode, "n": int(neg.shape[0]),
               "posctx_ref": round(float(ref or 0.0), 4),
               "post_median": round(float(post.median()), 6),
               "post_max": round(float(post.max()), 6),
               "pre_median": round(float(pre.median()), 4),
               "pre_p90": round(float(pre.quantile(0.9)), 4),
               "pre_max": round(float(pre.max()), 4),
               "frac_median": round(float(frac.median()), 4),
               "frac_p90": round(float(frac.quantile(0.9)), 4),
               "frac_max": round(float(frac.max()), 4),
               "share_gt_05": round(float((frac > 0.05).float().mean()), 4),
               "share_gt_10": round(float((frac > 0.10).float().mean()), 4),
               "share_gt_25": round(float((frac > 0.25).float().mean()), 4)}
        rows.append(row)
        print("%-9s %6d %9.4f %8.1f%% %8.1f%% %8.1f%% %7.0f%% %7.0f%% %7.0f%%"
              % (mode, row["n"], row["post_median"], 100 * row["frac_median"],
                 100 * row["frac_p90"], 100 * row["frac_max"],
                 100 * row["share_gt_05"], 100 * row["share_gt_10"],
                 100 * row["share_gt_25"]), flush=True)
    print(flush=True)

(HERE / "contamination.jsonl").write_text(
    "\n".join(json.dumps(r) for r in rows) + "\n")
print("wrote contamination.jsonl (%d rows)" % len(rows))
