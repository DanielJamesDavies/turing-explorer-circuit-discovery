"""Does cutting the low-score tail of a VALIDATED set crater free0?

  4 seeds (L2/L8/L9/L10) x abl-ig_mean PA
      x {+rec2+mag, +rec2+mag+eff-p50, +rec2+mag+eff-p90} ONLY.

The direct probe of C3 ("closure is collective; individual contributions sit at
the measurement floor"): magnitude bisection keeps the smallest FUNCTIONAL
prefix of the |attribution| ranking; the chained effect-threshold forks then
cut the bottom 50% / 90% of that validated set BY ITS OWN |score| distribution
(threshold_mode="pctl"). Same ranking, so the delta is pure stopping rule, and
the two cut depths give a dose-response:

  * free0 craters  -> the validated set's low-score members are collectively
                      load-bearing (C3 holds, now measured on a pruned set).
  * free0 survives -> the tail was dispensable; C3's "individually-invisible
                      but needed" claim weakens and a cheap threshold suffices.

Percentile cuts replace the first attempt's absolute T=0.1 (Marks et al.'s
node default), which sits ~2.3x ABOVE our maximum member score (L2: max
0.043, p50 7e-5) and deletes the whole circuit — the scale-mismatch record is
archived in abs-t0.1-2026-07-23/. Percentiles are scale-free, so every seed's
cut lands inside its own distribution by construction. The resolved absolute
cut and the quantiles are still logged per row.

All forks descend from ONE discovery and ONE rec2+mag prune (clones of the
pruned circuit), so the comparison shares everything upstream — rows within a
seed are same-discovery and directly comparable; rerunning a seed rediscovers
(pooled_abs_threshold subsample noise), so never mix rows across launches.

Evaluation is the standard fixed-anchor matrix (free0/freeM/pinMC + cf + faith
+ anchors + raw a_c), same geometry as the negctx grid.

Per-seed process isolation; resume-safe. Launch via launch.sh (never inline
`for i in ...` through wsl bash -lc — the outer shell eats $i).

  SEED_IDX=0..3 PYTHONPATH=src python experiments/003-eff-prune-test/runner.py
"""
import json
import os
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    CircuitOnlyPatcher, circuit_only_activation, collect_site_means,
    measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.effect_prune import prune_by_effect_threshold
from eval.floors import collect_site_anchors
from eval.magnitude_prune import prune_by_magnitude_bisection
from eval.recurrence_prune import prune_by_sequence_recurrence
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from store.circuits import Circuit

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
SEEDS = [(8, 30122, "L2"), (25, 10628, "L8"), (27, 6859, "L9"), (32, 3021, "L10")]
SEED_SEL = int(os.environ["SEED_IDX"]) if os.environ.get("SEED_IDX") else None
OUT = HERE / ("rows_s%d.jsonl" % SEED_SEL if SEED_SEL is not None else "rows.jsonl")

# (fork suffix, pctl cut of the validated set's own |score| distribution)
EFF_FORKS = [("+eff-p50", 50.0), ("+eff-p90", 90.0)]
N_SEQ, EVAL_BS, D_SAE, PA_PCTL = 64, 16, 40960, 90.0
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), D_SAE), device=bank.device)
n_kinds = len(bank.kinds)
ALL_SITES = {(l, k) for l in range(bank.n_layer) for k in bank.kinds}

_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = True
disc.position_aware_select = "abs_pctl"
disc.position_aware_threshold = PA_PCTL
disc.floor_source = "posctx"
disc.magnitude_prune = False
disc.recurrence_prune = False
disc.min_faithfulness = -100.0
config.discovery.ablation_gradient.negative_roles = "include"

done = set()
if OUT.exists():
    for line in OUT.open():
        if line.strip():
            done.add((json.loads(line)["seed"], json.loads(line)["arm"]))

fh = OUT.open("a")
todo = SEEDS if SEED_SEL is None else [SEEDS[SEED_SEL]]

for sc_idx, sl, label in todo:
    seed_key = "%d/%d" % (sc_idx, sl)
    arms = ["abl-ig_mean PA +rec2+mag"] + [
        "abl-ig_mean PA +rec2+mag" + sfx for sfx, _ in EFF_FORKS]
    if all((seed_key, a) in done for a in arms):
        print("[%s] all arms done — skip" % seed_key, flush=True)
        continue
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)

    m = _build_mode_method("ablation_gradient", "ig_mean", inference, bank,
                           avg_acts, probe_builder)
    pd_ = m.build_probe_dataset(sc_idx, sl)
    if pd_.pos_tokens.shape[0] == 0:
        print("[%s] no positive contexts — skip" % seed_key, flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt_fix = pd_.neg_tokens[:N_SEQ]
    tgt = pd_.target_tokens[:N_SEQ][torch.arange(pt.shape[0]), pa]

    a_pos = measure_seed_activation(inference, bank, pt, layer, kind, sl, pa,
                                    batch_size=EVAL_BS)
    means_up = collect_site_means(inference, bank, pt, up)
    means_all = collect_site_means(inference, bank, pt, ALL_SITES)
    a_e0 = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, sl,
                                   pos_argmax=pa, batch_size=EVAL_BS)
    a_eM = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, sl,
                                   pos_argmax=pa, site_means=means_up, batch_size=EVAL_BS)
    a_eMT = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, sl,
                                    pos_argmax=pa, site_means=means_up,
                                    batch_size=EVAL_BS, respect_topk=True)
    _, pins = collect_site_anchors(inference, bank, pt, up, pa,
                                   pin_position_specific=False)

    def logit_metric(keep, site_means=None):
        tot, n = 0.0, int(pt.shape[0])
        inference.disable_compile()
        try:
            for s in range(0, n, EVAL_BS):
                tk = pt[s:s + EVAL_BS]
                p = CircuitOnlyPatcher(bank=bank, keep_indices=keep,
                                       in_scope=ALL_SITES, seed_layer=-1,
                                       seed_kind="", seed_latent_idx=0,
                                       site_means=site_means)
                _, lg, _ = inference.forward(tk, patcher=p, all_logits=True,
                                             grad_enabled=False,
                                             return_activations=False,
                                             tokenize_final=False)
                b = torch.arange(tk.shape[0], device=device)
                lp = torch.log_softmax(lg[b, pa[s:s + EVAL_BS].to(device)].float(), dim=-1)
                tot += float(lp[b, tgt[s:s + EVAL_BS].to(device)].sum())
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    def full_logit():
        tot, n = 0.0, int(pt.shape[0])
        inference.disable_compile()
        try:
            for s in range(0, n, EVAL_BS):
                tk = pt[s:s + EVAL_BS]
                _, lg, _ = inference.forward(tk, all_logits=True, grad_enabled=False,
                                             return_activations=False,
                                             tokenize_final=False)
                b = torch.arange(tk.shape[0], device=device)
                lp = torch.log_softmax(lg[b, pa[s:s + EVAL_BS].to(device)].float(), dim=-1)
                tot += float(lp[b, tgt[s:s + EVAL_BS].to(device)].sum())
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    m_full, m_e_d = full_logit(), logit_metric({}, site_means=means_all)
    print("\n[%s] %s L%d %s | %d upstream sites | a_pos %.4f a_eM %.4f"
          % (seed_key, label, layer, kind, len(up), a_pos, a_eM), flush=True)

    raw_ac = {}

    def phi(keep, a_e, site_means=None, pin_values=None, respect_topk=False, tag=None):
        a_c = circuit_only_activation(inference, bank, keep, up, pt, layer, kind,
                                      sl, pos_argmax=pa, site_means=site_means,
                                      pin_values=pin_values, batch_size=EVAL_BS,
                                      respect_topk=respect_topk)
        if tag is not None:
            raw_ac[tag] = round(float(a_c), 4)
        den = a_pos - a_e
        if abs(den) < 1e-9:
            return None
        return round(float((a_c - a_e) / den), 4)

    def members_of(circ):
        out = []
        for node in circ.nodes.values():
            if node.metadata.get("role") == "seed":
                continue
            f = node.feature_id
            if f is None:
                continue
            sc = node.metadata.get("effect_score")
            if sc is None:
                sc = node.metadata.get("attribution_score")
            if sc is None:
                sc = node.metadata.get("weight") or 0.0
            out.append((abs(float(sc)), (f.layer, f.kind), int(f.index)))
        out.sort(key=lambda x: -x[0])
        return out

    def keep_from(feats, sites=None):
        keep = {}
        for _, site, idx in feats:
            if sites is not None and site not in sites:
                continue
            keep.setdefault(site, set()).add(idx)
        return keep

    def trimmed(circ, sites):
        c = Circuit(name=circ.name)
        for u, n in circ.nodes.items():
            f = n.feature_id
            if n.metadata.get("role") == "seed" or (f is not None and (f.layer, f.kind) in sites):
                c.nodes[u] = n
        c.edges = [e for e in circ.edges if e.source_uuid in c.nodes and e.target_uuid in c.nodes]
        c.metadata = dict(circ.metadata)
        return c

    t0 = time.time()
    circ = m.discover(sc_idx, sl)
    secs_disc = time.time() - t0
    if circ is None:
        print("  NO CIRCUIT", flush=True)
        continue

    # ---- shared rec2+mag prune (both forks descend from this) --------------
    c1 = trimmed(circ, ALL_SITES)
    prune_by_sequence_recurrence(inference, bank, c1, pos_tokens=pt,
                                 neg_tokens=nt_fix, min_sequences=2)
    prune_by_magnitude_bisection(inference, bank, c1, pos_tokens=pt,
                                 seed_layer=layer, seed_kind=kind,
                                 seed_latent_idx=sl, pos_argmax=pa,
                                 objective="free")

    # ---- eff forks: percentile cuts on TOP of the validated set ------------
    # Each fork clones the SAME rec2+mag circuit, so the forks are siblings
    # (p90 is not applied after p50).
    fork_circuits = [(arms[0], c1, None, None)]
    for (sfx, pctl), arm_name in zip(EFF_FORKS, arms[1:]):
        ce = trimmed(c1, ALL_SITES)
        removed = prune_by_effect_threshold(ce, threshold=pctl,
                                            threshold_mode="pctl")
        fork_circuits.append((arm_name, ce, pctl, len(removed)))

    for arm, c, eff_pctl, eff_removed in fork_circuits:
        if (seed_key, arm) in done:
            print("  skip (done) %s" % arm, flush=True)
            continue
        feats = members_of(c)
        keep_lat = keep_from(feats, sites=up)
        keep_all = keep_from(feats)
        n_lat = sum(len(v) for v in keep_lat.values())
        try:
            cf_v, sup_v = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, trimmed(c, up),
                neg_tokens=nt_fix, pos_tokens=pt,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_argmax=pa, circuit_layers={L for (L, _) in keep_lat})
            cf_v, sup_v = round(float(cf_v), 4), round(float(sup_v), 4)
        except Exception:
            cf_v = sup_v = None
        mc_d = logit_metric(keep_all, site_means=means_all)

        raw_ac.clear()
        row = {
            "seed": seed_key, "label": label, "layer": layer, "kind": kind,
            "arm": arm, "up_nodes": n_lat, "n_sites_up": len(up),
            "n_member_sites": len(keep_lat),
            "free0": phi(keep_lat, a_e0, tag="free0"),
            "freeM_dense": phi(keep_lat, a_eM, site_means=means_up, tag="freeM_dense"),
            "freeM_topk": phi(keep_lat, a_eMT, site_means=means_up,
                              respect_topk=True, tag="freeM_topk"),
            "pinMC_dense": phi(keep_lat, a_eM, site_means=means_up,
                               pin_values=pins, tag="pinMC_dense"),
            "cf": cf_v, "sup": sup_v,
            "faith_dense": (round((mc_d - m_e_d) / (m_full - m_e_d), 4)
                            if abs(m_full - m_e_d) > 1e-9 else None),
            "secs_discover": round(secs_disc, 1),
        }
        if eff_pctl is not None:
            row["eff_pctl"] = eff_pctl
            row["eff_cut_abs"] = c.metadata.get("effect_prune_threshold")
            row["eff_removed"] = eff_removed
            row["eff_score_q"] = c.metadata.get("effect_prune_score_q")
        row["a_pos"] = round(float(a_pos), 4)
        row["a_e0"] = round(float(a_e0), 4)
        row["a_eM"] = round(float(a_eM), 4)
        row["a_eMT"] = round(float(a_eMT), 4)
        row.update({"ac_" + t: v for t, v in raw_ac.items()})
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-30s n=%7d free0=%-8s freeM=%-8s cf=%-7s faith=%-8s"
              % (arm, n_lat, row["free0"], row["freeM_dense"], row["cf"],
                 row["faith_dense"]), flush=True)
        if eff_pctl is not None:
            print("      p%g cut=%.3g removed %s of %s"
                  % (eff_pctl, row["eff_cut_abs"] or 0.0,
                     format(eff_removed, ","),
                     format(eff_removed + n_lat, ",")), flush=True)
    del circ, c1, fork_circuits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

fh.close()
print("\nwrote %s" % OUT)
