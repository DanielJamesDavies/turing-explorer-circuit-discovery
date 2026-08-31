"""Shared-closure-library diagnostic (PA structure, test 1 of 2).

Hypothesis: a PA circuit's low-|attr| CLOSURE TAIL is largely seed-GENERIC
(shared stream-maintenance latents), while its high-|attr| DRIVER HEAD is
seed-specific. If true, circuits factor into (per-seed skeleton) + (one
shared library), and the library is distinguished from mean ablation by
being LIVE (computes content-appropriate values) rather than constant.

Per seed (8 usable, layer-stratified, abl-ig_mean PA abs_pctl@90 + prune —
the winner arm):
  1. discover; split members into HEAD (top-K by |attribution_score|) and
     TAIL (rest), K in {512, 2048}.
  2. bank memberships; cross-seed pairwise Jaccard of heads vs tails.
  3. leave-one-out library L(-i) = tail latents present in >= LIB_MIN of the
     OTHER seeds' tails (transfer test — no self-circularity).
  4. evals: free0(full) | free0(skel) | free0(skel u L(-i)) |
     freeM_topk(skel)  <- the mean-ablation-vs-live-library comparison:
     if free0(skel u L) > freeM_topk(skel) from the HARSHER zero floor, the
     background computes; if not, the "library" is a mean scaffold in
     disguise.

Rows -> shared_library.jsonl; aggregate table printed at the end.
PYTHONPATH=src python shared_library_test.py   (repo root, wsl + .venv)
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
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
N_SAMPLE = 10          # sample 10 to land ~8 usable
HEAD_KS = (512, 2048)
LIB_MIN = 3            # tail latent in >= 3 of the other seeds' tails
OUT = Path(__file__).parent / "shared_library.jsonl"
EVAL_BS = 16


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
idxs = _layer_stratified_indices(all_cand, N_SAMPLE, n_kinds)
cands = [_candidate_with_index(all_cand[i], i) for i in idxs]
print(f"sampled {len(cands)} candidates -> {OUT}", flush=True)

original = _apply_sweep_config(max_per_site=24)
disc = config.discovery
saved = (disc.probe_sequence_count, disc.eval_sequence_count, disc.eval_batch_size,
         disc.position_aware, disc.position_aware_select, disc.position_aware_threshold,
         disc.position_aware_top_n, disc.magnitude_prune, disc.floor_source,
         config.discovery.ablation_gradient.negative_roles)
disc.probe_sequence_count = 64
disc.eval_sequence_count = 64
disc.eval_batch_size = EVAL_BS
disc.position_aware = True
disc.position_aware_select = "abs_pctl"
disc.position_aware_threshold = 90.0
disc.position_aware_top_n = 96
disc.magnitude_prune = True
disc.floor_source = "posctx"
config.discovery.ablation_gradient.negative_roles = "include"

# ---- pass 1: discover + bank memberships/anchors per seed -----------------
seeds = []   # list of dicts with membership + eval anchors
t0 = time.time()
try:
    for si, cand in enumerate(cands):
        sc, sl = int(cand["comp_idx"]), int(cand["latent_idx"])
        seed_layer, ski = split_component_idx(sc, n_kinds)
        seed_kind = bank.kinds[ski]
        try:
            m = _build_mode_method("ablation_gradient", "ig_mean",
                                   inference, bank, avg_acts, probe_builder)
            pd = m.build_probe_dataset(sc, sl)
            if pd.pos_tokens.shape[0] == 0:
                print(f"[{si+1}] {sc}/{sl} L{seed_layer}: no pos — skip", flush=True)
                continue
            sites = upstream_sites(bank, seed_layer, seed_kind)
            if not sites:
                continue
            t = time.time()
            circuit = m.discover(sc, sl)
            if circuit is None:
                print(f"[{si+1}] {sc}/{sl}: no circuit — skip", flush=True)
                continue
            members = []   # ((layer, kind, latent), |score|)
            for node in circuit.nodes.values():
                if node.metadata.get("role") == "seed":
                    continue
                fid = node.feature_id
                if fid is None:
                    continue
                score = abs(float(node.metadata.get("attribution_score") or 0.0))
                members.append(((fid.layer, fid.kind, fid.index), score))
            members.sort(key=lambda x: -x[1])
            pt, pa = pd.pos_tokens[:64], pd.pos_argmax[:64]
            a_pos = measure_seed_activation(inference, bank, pt, seed_layer, seed_kind, sl, pa, batch_size=EVAL_BS)
            means = collect_site_means(inference, bank, pt, sites)
            a_e0 = circuit_only_activation(inference, bank, {}, sites, pt, seed_layer, seed_kind, sl, pos_argmax=pa, batch_size=EVAL_BS)
            a_eMT = circuit_only_activation(inference, bank, {}, sites, pt, seed_layer, seed_kind, sl, pos_argmax=pa, site_means=means, batch_size=EVAL_BS, respect_topk=True)
            seeds.append(dict(sc=sc, sl=sl, layer=seed_layer, kind=seed_kind,
                              members=members, sites=sites, pt=pt, pa=pa,
                              a_pos=a_pos, means=means, a_e0=a_e0, a_eMT=a_eMT))
            del circuit
            print(f"[{si+1}] {sc}/{sl} L{seed_layer} {seed_kind}: {len(members)} members "
                  f"({time.time()-t:.0f}s)", flush=True)
            if torch.cuda.is_available() and torch.cuda.memory_reserved() > 14e9:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{si+1}] {sc}/{sl} FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"\n{len(seeds)} usable seeds in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- pass 2: overlap + library evals per head-K ----------------------
    with OUT.open("w") as fh:
        for K in HEAD_KS:
            heads = [set(t for t, _ in s["members"][:K]) for s in seeds]
            tails = [set(t for t, _ in s["members"][K:]) for s in seeds]

            def jacc(sets):
                vals = []
                for i in range(len(sets)):
                    for j in range(i + 1, len(sets)):
                        u = len(sets[i] | sets[j])
                        vals.append(len(sets[i] & sets[j]) / u if u else 0.0)
                return sum(vals) / len(vals) if vals else 0.0

            jh, jt = jacc(heads), jacc(tails)
            print(f"\n=== K={K} ===  mean pairwise Jaccard: heads {jh:.4f} | tails {jt:.4f}", flush=True)

            for i, s in enumerate(seeds):
                # leave-one-out library from OTHER seeds' tails
                counts = defaultdict(int)
                for j, tail in enumerate(tails):
                    if j == i:
                        continue
                    for trip in tail:
                        counts[trip] += 1
                lib = {t for t, c in counts.items() if c >= LIB_MIN}
                site_set = set(s["sites"])
                lib_here = {t for t in lib if (t[0], t[1]) in site_set}

                def keep_of(trips):
                    keep = {}
                    for (L, kind, idx) in trips:
                        keep.setdefault((L, kind), set()).add(idx)
                    return keep

                full = set(t for t, _ in s["members"])
                skel = set(t for t, _ in s["members"][:K])
                skel_lib = skel | lib_here

                def phi(trips, site_means=None, respect_topk=False):
                    a_e = s["a_eMT"] if respect_topk else s["a_e0"]
                    if site_means is None:
                        a_e = s["a_e0"]
                    den = s["a_pos"] - a_e
                    if abs(den) < 1e-9:
                        return None
                    a_c = circuit_only_activation(
                        inference, bank, keep_of(trips), s["sites"], s["pt"],
                        s["layer"], s["kind"], s["sl"], pos_argmax=s["pa"],
                        site_means=site_means, batch_size=EVAL_BS,
                        respect_topk=respect_topk)
                    return round(float((a_c - a_e) / den), 4)

                rec = {
                    "K": K, "comp": s["sc"], "latent": s["sl"], "layer": s["layer"],
                    "kind": s["kind"], "n_full": len(full), "n_skel": len(skel),
                    "n_lib_here": len(lib_here), "n_skel_lib": len(skel_lib),
                    "free0_full": phi(full),
                    "free0_skel": phi(skel),
                    "free0_skel_lib": phi(skel_lib),
                    "freeMtopk_skel": phi(skel, site_means=s["means"], respect_topk=True),
                    "jacc_heads": round(jh, 4), "jacc_tails": round(jt, 4),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"  L{s['layer']:>2} {s['kind']:<5} full={rec['n_full']:>7} "
                      f"skel={rec['n_skel']:>5} +lib={rec['n_skel_lib']:>7} | "
                      f"free0 full={rec['free0_full']} skel={rec['free0_skel']} "
                      f"skel+lib={rec['free0_skel_lib']} | freeMtopk(skel)={rec['freeMtopk_skel']}",
                      flush=True)
finally:
    (disc.probe_sequence_count, disc.eval_sequence_count, disc.eval_batch_size,
     disc.position_aware, disc.position_aware_select, disc.position_aware_threshold,
     disc.position_aware_top_n, disc.magnitude_prune, disc.floor_source,
     config.discovery.ablation_gradient.negative_roles) = saved
    _restore_sweep_config(original)
print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
