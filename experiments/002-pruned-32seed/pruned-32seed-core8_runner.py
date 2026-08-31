"""32-seed PRUNED run: the top PA arms + two new arms, with magnitude-bisection
pruning ENABLED (relative floor: keep free0 within magnitude_prune_tolerance
0.05 of each raw circuit's own free0 — well-defined for every arm/depth).

Arms (7):
  cf-ig_mean PA                — both-sign mean-floor IG (the new closure top)
  abl-ig_zero PA               — NEW: zero-floor IG (0->natural, drive) with
                                 include roles = the free0-coherent
                                 "integrated activation gradient" (ig_zero)
  cf-ig_negctx PA              — the contrast-path IG
  abl-restoration PA abs_pctl  — rounds=sites
  cf-restoration PA abs_pctl   — NEW: rounds=sites, cf-hosted (= restoration
                                 with BOTH-sign role delivery)
  activation-gradient
  abl-local PA

Same widths/eval schema as the matrix; 32 layer-stratified seeds (first 16 =
the matrix sample). Rows -> pruned32_16x2.jsonl (crash-safe resume).
PYTHONPATH=src python pruned32_run.py   (repo root)
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method, _restore_sweep_config,
)
from analysis.circuits.gradient_method_neg_mode_grid_runner import _candidate_with_index
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, collect_site_means, measure_seed_activation, upstream_sites,
)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
N_SEEDS = 32
OUT = Path(__file__).parent / "pruned32_core8.jsonl"
EVAL_BS = 16
PA_PCTL = 90.0


def _layer_stratified_indices(candidates, sample_size, n_kinds, seed=42):
    by_layer = defaultdict(list)
    for index, cand in enumerate(candidates):
        by_layer[int(cand["comp_idx"]) // n_kinds].append(index)
    rng = random.Random(seed)
    for layer in by_layer:
        rng.shuffle(by_layer[layer])
    selected = []
    max_len = max(len(v) for v in by_layer.values())
    for rank in range(max_len):
        for layer in sorted(by_layer):
            idxs = by_layer[layer]
            if rank < len(idxs):
                selected.append(idxs[rank])
                if len(selected) >= sample_size:
                    return selected
    return selected


load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), bank.d_sae),
                       dtype=torch.float32, device=bank.device)
n_kinds = len(bank.kinds)

all_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu", weights_only=False)
idxs = _layer_stratified_indices(all_cand, N_SEEDS, n_kinds)
cands = [_candidate_with_index(all_cand[i], i) for i in idxs]
layers_hit = sorted({int(c["comp_idx"]) // n_kinds for c in cands})
print(f"sampled {len(cands)} seeds across layers {layers_hit} -> {OUT}", flush=True)

original = _apply_sweep_config(max_per_site=24)
disc = config.discovery
cf_cfg = config.discovery.counterfactual_gradient
ab_cfg = config.discovery.ablation_gradient
saved = (disc.probe_sequence_count, disc.eval_sequence_count, disc.eval_batch_size,
         disc.position_aware, disc.position_aware_select, disc.position_aware_threshold,
         disc.position_aware_top_n, disc.magnitude_prune, disc.floor_source,
         cf_cfg.max_neg_sequences, cf_cfg.neg_batch_size, cf_cfg.activator_signal,
         cf_cfg.ig_negctx_objective,
         ab_cfg.negative_roles,
         ab_cfg.restoration.rounds, ab_cfg.restoration.round_select,
         ab_cfg.restoration.round_abs_pctl,
         cf_cfg.restoration.rounds, cf_cfg.restoration.round_select,
         cf_cfg.restoration.round_abs_pctl)
disc.probe_sequence_count = 64
disc.eval_sequence_count = 64
disc.eval_batch_size = EVAL_BS
disc.position_aware_top_n = 96
disc.magnitude_prune = True          # THE run's knob: relative floor (tol 0.05)
cf_cfg.max_neg_sequences = 64
cf_cfg.neg_batch_size = 8


def pa_on():
    disc.position_aware = True
    disc.position_aware_select = "abs_pctl"
    disc.position_aware_threshold = PA_PCTL


def base_state():
    # All PA arms use include: post role-fix, PA free0/size are sign-invariant,
    # so include just gives honest inhibitor labels (and an honest cf split).
    pa_on()
    disc.floor_source = "posctx"
    ab_cfg.negative_roles = "include"
    cf_cfg.negative_roles = "include"
    cf_cfg.activator_signal = "gradient_x_posctx"
    cf_cfg.ig_negctx_objective = "gap"


# The 8-arm core: the baseline x schedule lattice, all PA, all abs_pctl@90,
# magnitude_prune on. (label, method, mode, extra_apply, restoration_cfg_or_None)
ARMS = [
    # single-point
    ("act-grad", "activation_gradient", "", None, None),                  # natural
    ("cf-local PA", "counterfactual_gradient", "local", None, None),      # negctx
    # path (integrated gradients)
    ("abl-ig_mean PA", "ablation_gradient", "ig_mean", None, None),       # mean floor, drive obj
    ("cf-ig_mean PA", "counterfactual_gradient", "ig_mean", None, None),  # mean floor, gap obj
    ("abl-ig_zero PA", "ablation_gradient", "ig_mean",                    # zero floor (integrated act-grad)
     lambda: setattr(disc, "floor_source", "zero"), None),
    ("cf-ig_negctx PA", "counterfactual_gradient", "ig_negctx", None, None),  # negctx path
    # iterated (restoration, rounds=sites)
    ("abl-restoration PA", "ablation_gradient", "restoration", None, ab_cfg),  # mean floor
    ("abl-restoration-zero PA", "ablation_gradient", "restoration",       # zero floor (act-restoration)
     lambda: setattr(disc, "floor_source", "zero"), ab_cfg),
]

t0 = time.time()
n_rows = 0
done = set()
if OUT.exists():
    for line in OUT.open():
        if line.strip():
            r = json.loads(line)
            done.add((r["comp"], r["latent"], r["arm"]))
    print(f"resuming: {len(done)} rows banked", flush=True)
try:
    with OUT.open("a") as fh:
        for si, cand in enumerate(cands):
            sc, sl = int(cand["comp_idx"]), int(cand["latent_idx"])
            seed_layer, ski = split_component_idx(sc, n_kinds)
            seed_kind = bank.kinds[ski]
            t_seed = time.time()
            try:
                m0 = _build_mode_method("counterfactual_gradient", "local",
                                        inference, bank, avg_acts, probe_builder)
                pd = m0.build_probe_dataset(sc, sl)
                if pd.pos_tokens.shape[0] == 0:
                    print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind}: no pos — skip", flush=True)
                    continue
                sites = upstream_sites(bank, seed_layer, seed_kind)
                if not sites:
                    print(f"[{si+1}/{len(cands)}] {sc}/{sl}: no upstream — skip", flush=True)
                    continue
                pt, pa, neg = pd.pos_tokens[:64], pd.pos_argmax[:64], pd.neg_tokens[:64]
                a_pos = measure_seed_activation(inference, bank, pt, seed_layer, seed_kind, sl, pa, batch_size=EVAL_BS)
                means = collect_site_means(inference, bank, pt, sites)
                _, pins_col = collect_site_anchors(inference, bank, pt, sites, pa, pin_position_specific=False)
                a_e0 = circuit_only_activation(inference, bank, {}, sites, pt, seed_layer, seed_kind, sl, pos_argmax=pa, batch_size=EVAL_BS)
                a_eM = circuit_only_activation(inference, bank, {}, sites, pt, seed_layer, seed_kind, sl, pos_argmax=pa, site_means=means, batch_size=EVAL_BS)
                a_eMT = circuit_only_activation(inference, bank, {}, sites, pt, seed_layer, seed_kind, sl, pos_argmax=pa, site_means=means, batch_size=EVAL_BS, respect_topk=True)
                if abs(a_pos - a_e0) < 1e-6 and abs(a_pos - a_eM) < 1e-6:
                    print(f"[{si+1}/{len(cands)}] {sc}/{sl}: degenerate — skip", flush=True)
                    continue
            except Exception as e:
                print(f"[{si+1}/{len(cands)}] {sc}/{sl} anchors FAILED: {type(e).__name__}: {e}", flush=True)
                continue

            rounds_ceiling = len(sites)

            def phi(keep, site_means=None, pin_values=None, respect_topk=False):
                a_e = (a_eMT if respect_topk else a_eM) if site_means is not None else a_e0
                den = a_pos - a_e
                if abs(den) < 1e-9:
                    return None
                a_c = circuit_only_activation(inference, bank, keep, sites, pt, seed_layer, seed_kind, sl,
                                              pos_argmax=pa, site_means=site_means, pin_values=pin_values,
                                              batch_size=EVAL_BS, respect_topk=respect_topk)
                return round(float((a_c - a_e) / den), 4)

            for label, method, mode, extra, resto_cfg in ARMS:
                if (sc, sl, label) in done:
                    continue
                base_state()
                if extra is not None:
                    extra()
                if resto_cfg is not None:
                    resto_cfg.restoration.rounds = rounds_ceiling
                    resto_cfg.restoration.round_select = "abs_pctl"
                    resto_cfg.restoration.round_abs_pctl = 95.0
                t_arm = time.time()
                try:
                    m = _build_mode_method(method, mode, inference, bank, avg_acts, probe_builder)
                    circuit = m.discover(sc, sl)
                except Exception as e:
                    print(f"    {label:<28} FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                secs = time.time() - t_arm
                if circuit is None:
                    print(f"    {label:<28} -> no circuit ({secs:.0f}s)", flush=True)
                    continue
                keep = {}
                for node in circuit.nodes.values():
                    if node.metadata.get("role") == "seed":
                        continue
                    fid = node.feature_id
                    if fid is not None:
                        keep.setdefault((fid.layer, fid.kind), set()).add(fid.index)
                n = sum(len(v) for v in keep.values())
                if n == 0:
                    print(f"    {label:<28} -> empty ({secs:.0f}s)", flush=True)
                    continue
                t_eval = time.time()
                try:
                    cf, sup = evaluate_counterfactual_faithfulness(
                        inference, bank, avg_acts, circuit, neg_tokens=neg, pos_tokens=pt,
                        seed_layer=seed_layer, seed_kind=seed_kind, seed_latent_idx=sl,
                        pos_argmax=pa, circuit_layers={L for (L, k) in keep})
                    cf, sup = round(float(cf), 4), round(float(sup), 4)
                except Exception:
                    cf = sup = None
                n_raw = int(circuit.metadata.get("n_members_pre_prune", n))
                rec = {"comp": sc, "latent": sl, "layer": seed_layer, "kind": seed_kind,
                       "arm": label, "size": n, "size_raw": n_raw,
                       "rounds_ceiling": rounds_ceiling, "pruned": True,
                       "free0": phi(keep),
                       "freeM_dense": phi(keep, site_means=means),
                       "freeM_topk": phi(keep, site_means=means, respect_topk=True),
                       "pinMC_dense": phi(keep, site_means=means, pin_values=pins_col),
                       "pinMC_topk": phi(keep, site_means=means, pin_values=pins_col, respect_topk=True),
                       "cf": cf, "sup": sup, "secs_discover": round(secs, 1),
                       "secs_evals": round(time.time() - t_eval, 1)}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_rows += 1
                print(f"    {label:<28} disc {secs:>5.0f}s n={n_raw:>7}->{n:<7} "
                      f"| free0={rec['free0']} cf={cf}", flush=True)
                del circuit, keep
                if torch.cuda.is_available() and torch.cuda.memory_reserved() > 14e9:
                    torch.cuda.empty_cache()
            el = time.time() - t0
            per_seed = el / (si + 1)
            print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind} | {time.time()-t_seed:.0f}s "
                  f"| rows={n_rows} | avg {per_seed:.0f}s/seed | ETA {per_seed*(len(cands)-si-1)/60:.0f} min", flush=True)
finally:
    (disc.probe_sequence_count, disc.eval_sequence_count, disc.eval_batch_size,
     disc.position_aware, disc.position_aware_select, disc.position_aware_threshold,
     disc.position_aware_top_n, disc.magnitude_prune, disc.floor_source,
     cf_cfg.max_neg_sequences, cf_cfg.neg_batch_size, cf_cfg.activator_signal,
     cf_cfg.ig_negctx_objective,
     ab_cfg.negative_roles,
     ab_cfg.restoration.rounds, ab_cfg.restoration.round_select,
     ab_cfg.restoration.round_abs_pctl,
     cf_cfg.restoration.rounds, cf_cfg.restoration.round_select,
     cf_cfg.restoration.round_abs_pctl) = saved
    _restore_sweep_config(original)
print(f"\ndone: {n_rows} rows in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
