"""16-seed run: the three winner arms x three negctx modes (close/random/distant).

KEY FACT (verified in code): all three winners DISCOVER ON POSCTX --
  abl-ig_mean PA        mean-floor IG, drive obj, posctx
  cf-ig_mean PA         mean-floor IG, gap obj, posctx (NOT negctx! _run_ig_mean_hop
                        runs on pos_tokens)
  abl-restoration PA    mean-ablated posctx, greedy rounds=sites, abs_pctl@95
So the circuit (hence size/free0/freeM/pinMC) is NEG-MODE-INVARIANT. neg_mode only
moves cf/sup, the one eval that consumes neg_tokens. We therefore discover each arm
ONCE per seed and evaluate cf/sup against all three mode-specific neg sets, built via
the discovery selector path (`_select_neg_context`) -- NOT build_probe_dataset, whose
neg_tokens are always the stored "close" hard-negatives regardless of neg_mode.

Config matches the pruned-32seed winners exactly (PA abs_pctl@90, magnitude_prune on,
posctx floor, include roles, seq/eval count 64). 16 layer-stratified seeds (= the
matrix sample). One row per (seed, arm, neg_mode); closure cols repeat across a seed's
three mode-rows by construction. Rows -> winners_negmode.jsonl (crash-safe resume).

PYTHONPATH=src python winners_negmode_runner.py   (repo root, via wsl + .venv)
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
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, collect_site_means, measure_seed_activation, upstream_sites,
)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
N_SEEDS = 16
OUT = Path(__file__).parent / "winners_negmode.jsonl"
EVAL_BS = 16
PA_PCTL = 90.0
MODES = ["close", "random", "distant"]


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
from circuit.probe_dataset import ProbeDatasetBuilder  # noqa: E402
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
         cf_cfg.ig_negctx_objective, cf_cfg.neg_mode,
         ab_cfg.negative_roles,
         ab_cfg.restoration.rounds, ab_cfg.restoration.round_select,
         ab_cfg.restoration.round_abs_pctl)
disc.probe_sequence_count = 64
disc.eval_sequence_count = 64
disc.eval_batch_size = EVAL_BS
disc.position_aware_top_n = 96
disc.magnitude_prune = True
cf_cfg.max_neg_sequences = 64
cf_cfg.neg_batch_size = 8


def pa_on():
    disc.position_aware = True
    disc.position_aware_select = "abs_pctl"
    disc.position_aware_threshold = PA_PCTL


def base_state():
    pa_on()
    disc.floor_source = "posctx"
    ab_cfg.negative_roles = "include"
    cf_cfg.negative_roles = "include"
    cf_cfg.activator_signal = "gradient_x_posctx"
    cf_cfg.ig_negctx_objective = "gap"


# (label, method, mode, restoration_cfg_or_None)
ARMS = [
    ("abl-ig_mean PA", "ablation_gradient", "ig_mean", None),
    ("cf-ig_mean PA", "counterfactual_gradient", "ig_mean", None),
    ("abl-restoration PA", "ablation_gradient", "restoration", ab_cfg),
]

t0 = time.time()
n_rows = 0
done = set()
if OUT.exists():
    for line in OUT.open():
        if line.strip():
            r = json.loads(line)
            done.add((r["comp"], r["latent"], r["arm"], r["neg_mode"]))
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
                pt, pa = pd.pos_tokens[:64], pd.pos_argmax[:64]
                a_pos = measure_seed_activation(inference, bank, pt, seed_layer, seed_kind, sl, pa, batch_size=EVAL_BS)
                means = collect_site_means(inference, bank, pt, sites)
                _, pins_col = collect_site_anchors(inference, bank, pt, sites, pa, pin_position_specific=False)
                a_e0 = circuit_only_activation(inference, bank, {}, sites, pt, seed_layer, seed_kind, sl, pos_argmax=pa, batch_size=EVAL_BS)
                a_eM = circuit_only_activation(inference, bank, {}, sites, pt, seed_layer, seed_kind, sl, pos_argmax=pa, site_means=means, batch_size=EVAL_BS)
                a_eMT = circuit_only_activation(inference, bank, {}, sites, pt, seed_layer, seed_kind, sl, pos_argmax=pa, site_means=means, batch_size=EVAL_BS, respect_topk=True)
                if abs(a_pos - a_e0) < 1e-6 and abs(a_pos - a_eM) < 1e-6:
                    print(f"[{si+1}/{len(cands)}] {sc}/{sl}: degenerate — skip", flush=True)
                    continue
                # Mode-specific neg sets via the discovery selector (NOT the stored
                # close-only build_probe_dataset path).
                neg_logger = CircuitLogger(sc, sl, "negbuild")
                neg_by_mode = {}
                for mode in MODES:
                    try:
                        sel = m0._select_neg_context(sc, sl, mode, 64, cf_cfg.neg_batch_size, neg_logger)
                        neg_by_mode[mode] = None if sel is None else sel.tokens[:64]
                    except Exception as e:
                        print(f"    negctx[{mode}] FAILED: {type(e).__name__}: {e}", flush=True)
                        neg_by_mode[mode] = None
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

            for label, method, mode, resto_cfg in ARMS:
                if all((sc, sl, label, nm) in done for nm in MODES):
                    continue
                base_state()
                if resto_cfg is not None:
                    resto_cfg.restoration.rounds = rounds_ceiling
                    resto_cfg.restoration.round_select = "abs_pctl"
                    resto_cfg.restoration.round_abs_pctl = 95.0
                t_arm = time.time()
                try:
                    m = _build_mode_method(method, mode, inference, bank, avg_acts, probe_builder)
                    circuit = m.discover(sc, sl)
                except Exception as e:
                    print(f"    {label:<22} FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                secs = time.time() - t_arm
                if circuit is None:
                    print(f"    {label:<22} -> no circuit ({secs:.0f}s)", flush=True)
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
                    print(f"    {label:<22} -> empty ({secs:.0f}s)", flush=True)
                    continue
                n_raw = int(circuit.metadata.get("n_members_pre_prune", n))
                closure = {
                    "size": n, "size_raw": n_raw,
                    "free0": phi(keep),
                    "freeM_dense": phi(keep, site_means=means),
                    "freeM_topk": phi(keep, site_means=means, respect_topk=True),
                    "pinMC_dense": phi(keep, site_means=means, pin_values=pins_col),
                    "pinMC_topk": phi(keep, site_means=means, pin_values=pins_col, respect_topk=True),
                }
                cf_by_mode = {}
                for nm in MODES:
                    if (sc, sl, label, nm) in done:
                        continue
                    neg = neg_by_mode.get(nm)
                    if neg is None or neg.shape[0] == 0:
                        cf = sup = None
                        t_eval = 0.0
                    else:
                        t_eval = time.time()
                        try:
                            cf, sup = evaluate_counterfactual_faithfulness(
                                inference, bank, avg_acts, circuit, neg_tokens=neg, pos_tokens=pt,
                                seed_layer=seed_layer, seed_kind=seed_kind, seed_latent_idx=sl,
                                pos_argmax=pa, circuit_layers={L for (L, k) in keep})
                            cf, sup = round(float(cf), 4), round(float(sup), 4)
                        except Exception:
                            cf = sup = None
                        t_eval = time.time() - t_eval
                    rec = {"comp": sc, "latent": sl, "layer": seed_layer, "kind": seed_kind,
                           "arm": label, "neg_mode": nm, "rounds_ceiling": rounds_ceiling,
                           "pruned": True, **closure, "cf": cf, "sup": sup,
                           "secs_discover": round(secs, 1), "secs_eval": round(t_eval, 1)}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_rows += 1
                    cf_by_mode[nm] = cf
                print(f"    {label:<22} disc {secs:>5.0f}s n={n_raw:>7}->{n:<7} "
                      f"| free0={closure['free0']} cf(c/r/d)="
                      f"{cf_by_mode.get('close')}/{cf_by_mode.get('random')}/{cf_by_mode.get('distant')}", flush=True)
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
     cf_cfg.ig_negctx_objective, cf_cfg.neg_mode,
     ab_cfg.negative_roles,
     ab_cfg.restoration.rounds, ab_cfg.restoration.round_select,
     ab_cfg.restoration.round_abs_pctl) = saved
    _restore_sweep_config(original)
print(f"\ndone: {n_rows} rows in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
