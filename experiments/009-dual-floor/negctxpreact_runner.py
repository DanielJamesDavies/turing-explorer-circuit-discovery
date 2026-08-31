"""PHASE 2: does a negctx mask floor help, and does pre-top-k filtering fix
"close"?

Two changes are under test together because they interact:

  mask_floor_source="negctx" — a fully masked latent lands on the negctx mean
    instead of 0, so the mask's m=0 state IS the state freeN measures against.
    With the zero floor the mask trains under free0's counterfactual and is
    therefore always scored on home turf.

  preact_filter — a negative is accepted on its PRE-top-k seed value rather
    than the post-top-k one, which is pinned to exactly 0 whenever the seed
    misses top-k. Phase 1 measured the damage: post-top-k reads 0.0000 in 15
    of 16 (seed, mode) cells while the median "negative" is actually driving
    the seed to 2.8% (L2) .. 27.8% (L10) of its posctx level, and CLOSE is
    more contaminated than RANDOM at 4/4 seeds (>25% share at L10: 75% vs
    33%). Prediction under test: filtering should help close MORE than random.

TWO CONTROLS THAT MAKE THIS READABLE

1. FIXED EVAL NEGATIVES. freeN/freeN_topk take their anchor from a negative
   set, so scoring each arm on ITS OWN negatives would grade every arm on a
   different exam. One clean set per seed (random + preact filter, the
   cleanest per Phase 1) anchors every arm; only the circuit varies.

2. MATCHED-NODE CONTROL. The same lambda need not mean the same thing under
   the two floors — masking to a mean removes less signal than masking to
   zero, so the L1 term buys a different amount of loss per unit m and the
   arms can land at different sizes. Since every fill-based metric is
   size-confounded, the zero-floor arm is ALSO evaluated truncated to the
   candidate's node count.

preact_max_frac=0.25, not the 0.1 default: Phase 1 showed 0.1 would reject
100% of close and 91% of random candidates at L10, leaving no negatives at
all. Scan/reject counts are logged so an infeasible seed fails loudly.

  SEED_IDX=n PYTHONPATH=src python experiments/009-dual-floor/runner.py
"""
import json
import os
import random
import time
from collections import defaultdict
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
from store.circuits import Circuit

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ, EVAL_BS = 64, 16
PREACT_FRAC = 0.25
WANT = [int(x) for x in os.environ.get("SEED_IDS", "2,5,8,10").split(",")]
SEL = int(os.environ["SEED_IDX"]) if os.environ.get("SEED_IDX") else None
OUT = HERE / ("rows_s%d.jsonl" % SEL if SEL is not None else "rows.jsonl")
torch.set_float32_matmul_precision("high")

# (label, mask_floor_source, floor_negctx_mode, preact_filter)
ARMS = [
    ("mask zero-floor",        "zero",   "store",  False),
    ("mask negctx/store",      "negctx", "store",  False),
    ("mask negctx/close",      "negctx", "close",  False),
    ("mask negctx/close+pre",  "negctx", "close",  True),
    ("mask negctx/random",     "negctx", "random", False),
    ("mask negctx/random+pre", "negctx", "random", True),
]
CANDIDATE = "mask negctx/close+pre"      # what the matched control is sized to
MATCHED = "mask zero-floor@matched"

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, bank.d_sae), device=bank.device)
disc = config.discovery


def base_state():
    _apply_sweep_config(max_per_site=24)
    disc.probe_sequence_count = N_SEQ
    disc.eval_sequence_count = N_SEQ
    disc.eval_batch_size = EVAL_BS
    disc.probe_batch_size = 4
    disc.position_aware = False          # attribution_mode="mask" is NPA
    disc.magnitude_prune = False
    disc.recurrence_prune = False
    disc.min_faithfulness = -100.0
    disc.floor_source = "posctx"          # the IG hops' knob; unused here
    lm = disc.learned_mask
    lm.steps, lm.lr, lm.l1_lambda = 400, 0.05, 1e-4
    lm.optimizer, lm.weight_decay = "adamw", 0.05
    lm.lr_schedule, lm.warmup_frac = "constant", 0.0
    lm.keep_threshold, lm.holdout_frac = 0.5, 0.25
    ncs = disc.neg_context_selection
    ncs.preact_max_frac = PREACT_FRAC
    ncs.preact_reference_stat = "median"


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
SEEDS = [(i, int(_cand[_idx[i]]["comp_idx"]), int(_cand[_idx[i]]["latent_idx"]))
         for i in WANT]

done = set()
if OUT.exists():
    for line in OUT.open():
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["arm"]))
fh = OUT.open("a")

todo = SEEDS if SEL is None else [s for s in SEEDS if s[0] == SEL]
for seed_i, sc_idx, sl in todo:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))
    base_state()
    m0 = _build_mode_method("ablation_gradient", "mask", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    if pd_.pos_tokens.shape[0] == 0:
        print("[%s] no positives — skip" % seed_key, flush=True); continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]

    # ---- fixed, clean eval negatives (see module docstring, control 1) -----
    selector = m0._neg_context_selector()
    ncs = disc.neg_context_selection
    ref = selector.posctx_reference(pt, sc_idx, sl,
                                    batch_size=int(ncs.filter_batch_size))
    eval_sel = selector.select(
        sc_idx, sl, "random", max_sequences=N_SEQ,
        batch_size=max(1, int(disc.probe_batch_size)),
        candidate_pool_size=ncs.candidate_pool_size,
        exact=bool(ncs.exact_negctx_ranking),
        non_activation_threshold=float(ncs.non_activation_threshold),
        preact_filter=True, preact_max_frac=PREACT_FRAC, posctx_reference=ref,
        selection_seed=int(ncs.selection_seed),
        filter_batch_size=int(ncs.filter_batch_size),
        load_window_size=int(ncs.load_window_size), logger=None)
    if eval_sel is None or eval_sel.tokens.shape[0] == 0:
        print("[%s] NO CLEAN EVAL NEGATIVES at frac=%.2f — skipping seed "
              "(a finding, not a crash: this seed may have none)"
              % (seed_key, PREACT_FRAC), flush=True)
        continue
    nt_eval = eval_sel.tokens[:N_SEQ]

    a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind, sl,
                                          pa, batch_size=EVAL_BS))
    means_up = collect_site_means(inference, bank, pt, set(up))
    means_neg = collect_site_means(inference, bank, nt_eval, set(up))

    def empty(site_means=None, respect_topk=False):
        return float(circuit_only_activation(
            inference, bank, {}, up, pt, layer, kind, sl, pos_argmax=pa,
            site_means=site_means, batch_size=EVAL_BS,
            respect_topk=respect_topk))

    a_e0, a_eM = empty(), empty(means_up)
    a_eMT = empty(means_up, True)
    a_eN, a_eNT = empty(means_neg), empty(means_neg, True)
    print("\n[%d %s] L%d %s | %d sites | a_pos %.4f ref %.4f | eval-neg %d "
          "(clean) | leak posctx %.0f%% negctx %.0f%%"
          % (seed_i, seed_key, layer, kind, len(up), a_pos, ref or 0.0,
             nt_eval.shape[0], 100 * a_eM / max(a_pos, 1e-9),
             100 * a_eN / max(a_pos, 1e-9)), flush=True)

    def phi(keep, a_e, site_means=None, respect_topk=False):
        if a_e is None:
            return None
        a_c = float(circuit_only_activation(
            inference, bank, keep, up, pt, layer, kind, sl, pos_argmax=pa,
            site_means=site_means, batch_size=EVAL_BS,
            respect_topk=respect_topk))
        den = a_pos - a_e
        return round((a_c - a_e) / den, 4) if abs(den) > 1e-9 else None

    def members_of(circ):
        out = []
        for node in circ.nodes.values():
            if node.metadata.get("role") == "seed":
                continue
            f = node.feature_id
            if f is None:
                continue
            sc = node.metadata.get("attribution_score")
            out.append((abs(float(sc or 0.0)), (f.layer, f.kind), int(f.index)))
        out.sort(key=lambda x: -x[0])
        return out

    def eval_row(label, circ, spec, secs):
        feats = members_of(circ)
        keep = {}
        for _, site, idx in feats:
            if site in up:
                keep.setdefault(site, set()).add(idx)
        n = sum(len(v) for v in keep.values())
        row = {"seed": seed_key, "seed_i": seed_i, "layer": layer, "kind": kind,
               "arm": label, "mask_floor_source": spec[0],
               "floor_negctx_mode": spec[1], "preact_filter": spec[2],
               "preact_max_frac": PREACT_FRAC, "posctx_ref": round(ref or 0.0, 4),
               "n_eval_neg": int(nt_eval.shape[0]),
               "up_nodes": n, "n_sites_up": len(up),
               "free0": phi(keep, a_e0),
               "freeM_dense": phi(keep, a_eM, means_up),
               "freeM_topk": phi(keep, a_eMT, means_up, True),
               "freeN": phi(keep, a_eN, means_neg),
               "freeN_topk": phi(keep, a_eNT, means_neg, True),
               "a_pos": round(a_pos, 4), "a_e0": round(a_e0, 4),
               "a_eM": round(a_eM, 4), "a_eMT": round(a_eMT, 4),
               "a_eN": round(a_eN, 4), "a_eNT": round(a_eNT, 4),
               "secs": round(secs, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-24s n=%8s free0=%-8s freeN=%-8s freeN_tk=%-8s freeM_tk=%-8s"
              % (label, format(n, ","), row["free0"], row["freeN"],
                 row["freeN_topk"], row["freeM_topk"]), flush=True)
        return n

    zero_circ, cand_n = None, None
    for label, floor_src, neg_mode, pre in ARMS:
        if (seed_key, label) in done and label != "mask zero-floor":
            print("  skip (done) %s" % label, flush=True); continue
        try:
            base_state()
            disc.learned_mask.mask_floor_source = floor_src
            disc.floor_negctx_mode = neg_mode
            disc.neg_context_selection.preact_filter = bool(pre)
            meth = _build_mode_method("ablation_gradient", "mask", inference,
                                      bank, avg_acts, probe_builder)
            t0 = time.time()
            circ = meth.discover(sc_idx, sl)
            if circ is None:
                print("  %-24s NO CIRCUIT" % label, flush=True); continue
            n = eval_row(label, circ, (floor_src, neg_mode, pre), time.time() - t0)
            if label == "mask zero-floor":
                zero_circ = circ
            elif label == CANDIDATE:
                cand_n = n
            if label != "mask zero-floor":
                del circ
        except Exception as exc:
            print("  %-24s FAILED %s: %s" % (label, type(exc).__name__, exc),
                  flush=True)
        finally:
            torch.cuda.empty_cache()

    # ---- matched-node control (control 2; no discovery) --------------------
    if zero_circ is not None and cand_n and (seed_key, MATCHED) not in done:
        try:
            ranked = members_of(zero_circ)[:cand_n]
            wanted = {(s, i) for _, s, i in ranked}
            cm = Circuit(name=zero_circ.name)
            for u, node in zero_circ.nodes.items():
                f = node.feature_id
                if node.metadata.get("role") == "seed" or (
                        f is not None and ((f.layer, f.kind), int(f.index)) in wanted):
                    cm.nodes[u] = node
            cm.metadata = dict(zero_circ.metadata)
            eval_row(MATCHED, cm, ("zero", "truncated", False), 0.0)
        except Exception as exc:
            print("  %-24s FAILED %s: %s" % (MATCHED, type(exc).__name__, exc),
                  flush=True)
    del zero_circ
    torch.cuda.empty_cache()

fh.close()
print("\nwrote %s" % OUT)
