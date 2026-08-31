"""Grid: does the negctx floor help, and does negative hardness matter?

  4 seeds (L2/L8/L9/L10)  x  2 methods  x  4 discovery floors  = 32 discoveries

METHODS are the two DISTINCT floor consumers, deliberately not the two
top-scoring arms: ig_mean reaches the floor through
gradient_base._integrated_baseline_attribution, restoration through
instrument/restoration.run_restoration_selection. Together they exercise both
wired paths. cf-ig_mean is excluded because it is the same method as
abl-ig_mean (they agree to 0.006 on every metric across all four seeds — same
posctx floor, same IG path); sfc is excluded because it never calls
resolve_site_floors at all.

FLOORS: posctx (the in-run control) plus negctx under each negative-hardness
mode -- close, random, distant.

DESIGN. The discovery floor varies; the EVALUATION is held fixed. Every arm is
scored against identical anchors -- including a FIXED negctx eval floor taken
from the neg_ctx store regardless of which negatives the discovery used -- so
every column below is comparable across floors. Change both and the comparison
means nothing.

FULL EVAL MATRIX (no single metric decides this):
  free0        zero floor, live re-encode      -- floor-independent
  freeM_dense  posctx mean fill                -- the legacy/SFC-comparable one
  freeM_topk   posctx fill, k-sparse respected -- on-manifold variant
  freeN        negctx mean fill                -- shares free0's denominator
                                                  (a_eN==0) but a different
                                                  numerator: on-manifold fill
  pinMC_dense  posctx fill + pinned drivers    -- known unbounded, reached 2.05
  pinNC        negctx fill + pinned drivers    -- the same measure, cold floor
  cf / sup     counterfactual faithfulness on negctx + support
  faith_dense  logit-metric faithfulness over ALL sites
  up_nodes, pct_dict_up, secs_discover
  a_eN         the DISCOVERY floor's own a_empty (the leak measurement)

Anchors are logged per row (a_pos/a_e0/a_eM/a_eMT/a_eNfix + ac_*), so every
ratio is reconstructible -- see experiments/005-floor-diagnostic.

Per-seed process isolation (one OOM poisons the allocator); resume-safe.
Launch via launch.sh -- do NOT inline the loop through `wsl bash -lc`, where
$i is eaten by the outer shell and every seed silently runs as SEED_IDX="".

  SEED_IDX=0..3 PYTHONPATH=src python experiments/006-negctx-floor-grid/runner.py
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
from eval.floors import collect_site_anchors
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

METHODS = [("ig_mean", "ig_mean"), ("restoration", "restoration")]
# (label, floor_source, floor_negctx_mode)
FLOORS = [("posctx", "posctx", "store"),
          ("negctx-close", "negctx", "close"),
          ("negctx-random", "negctx", "random"),
          ("negctx-distant", "negctx", "distant")]

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


def base_state():
    """Reset every knob the grid varies, so arms cannot leak into each other."""
    _apply_sweep_config(max_per_site=24)
    disc = config.discovery
    disc.probe_sequence_count = N_SEQ
    disc.eval_sequence_count = N_SEQ
    disc.eval_batch_size = EVAL_BS
    disc.probe_batch_size = 4
    disc.position_aware = True
    disc.position_aware_select = "abs_pctl"
    disc.position_aware_threshold = PA_PCTL
    disc.magnitude_prune = False
    disc.recurrence_prune = False
    disc.min_faithfulness = -100.0
    disc.floor_source = "posctx"
    disc.floor_negctx_mode = "store"
    config.discovery.ablation_gradient.negative_roles = "include"
    return disc


done = set()
if OUT.exists():
    for line in OUT.open():
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["arm"]))

fh = OUT.open("a")
todo = SEEDS if SEED_SEL is None else [SEEDS[SEED_SEL]]

for sc_idx, sl, label in todo:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)

    base_state()
    m0 = _build_mode_method("ablation_gradient", "ig_mean", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    if pd_.pos_tokens.shape[0] == 0:
        print("[%s] no positive contexts — skip" % seed_key, flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt_fix = pd_.neg_tokens[:N_SEQ]          # FIXED eval negatives (neg_ctx store)
    tgt = pd_.target_tokens[:N_SEQ][torch.arange(pt.shape[0]), pa]

    # ---- FIXED evaluation anchors, shared by every arm of this seed --------
    a_pos = measure_seed_activation(inference, bank, pt, layer, kind, sl, pa,
                                    batch_size=EVAL_BS)
    means_up = collect_site_means(inference, bank, pt, up)
    means_neg = collect_site_means(inference, bank, nt_fix, up)
    means_all = collect_site_means(inference, bank, pt, ALL_SITES)
    a_e0 = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, sl,
                                   pos_argmax=pa, batch_size=EVAL_BS)
    a_eM = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, sl,
                                   pos_argmax=pa, site_means=means_up, batch_size=EVAL_BS)
    a_eMT = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, sl,
                                    pos_argmax=pa, site_means=means_up,
                                    batch_size=EVAL_BS, respect_topk=True)
    a_eNfix = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, sl,
                                      pos_argmax=pa, site_means=means_neg,
                                      batch_size=EVAL_BS)
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

    print("\n[%s] %s L%d %s | %d upstream sites" % (seed_key, label, layer, kind, len(up)),
          flush=True)
    print("    a_pos %.4f | a_e0 %.4f | a_eM %.4f (leak %.0f%%) | a_eNfix %.4f"
          % (a_pos, a_e0, a_eM, 100.0 * a_eM / max(a_pos, 1e-9), a_eNfix), flush=True)

    raw_ac = {}

    def phi(keep, a_e, site_means=None, pin_values=None, respect_topk=False, tag=None):
        """Anchor passed explicitly — three floors are in play here, so
        inferring it from `site_means is not None` (as the other runners do)
        would silently score the negctx fill against the posctx anchor."""
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

    for floor_label, floor_source, floor_mode in FLOORS:
        for meth_label, attr_mode in METHODS:
            arm = "abl-%s PA / %s" % (meth_label, floor_label)
            if (seed_key, arm) in done:
                print("  skip (done) %s" % arm, flush=True)
                continue
            disc = base_state()
            disc.floor_source = floor_source
            disc.floor_negctx_mode = floor_mode
            t0 = time.time()
            try:
                m = _build_mode_method("ablation_gradient", attr_mode, inference,
                                       bank, avg_acts, probe_builder)
                circ = m.discover(sc_idx, sl)
                secs = time.time() - t0
                if circ is None:
                    print("  %-36s NO CIRCUIT" % arm, flush=True)
                    continue

                feats = members_of(circ)
                keep_lat = keep_from(feats, sites=up)
                keep_all = keep_from(feats)
                n_lat = sum(len(v) for v in keep_lat.values())

                # This floor's OWN a_empty — the leak measurement. Uses exactly
                # the negatives the discovery used (m._floor_neg_tokens), not a
                # re-derivation, so it cannot drift from what was discovered.
                a_eN = None
                nt = getattr(m, "_floor_neg_tokens", None)
                if floor_source == "negctx" and nt is not None and nt.shape[0] > 0:
                    negm = collect_site_means(inference, bank, nt, up)
                    a_eN = round(float(circuit_only_activation(
                        inference, bank, {}, up, pt, layer, kind, sl, pos_argmax=pa,
                        site_means=negm, batch_size=EVAL_BS)), 4)

                try:
                    cf_v, sup_v = evaluate_counterfactual_faithfulness(
                        inference, bank, avg_acts, trimmed(circ, up),
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
                    "arm": arm, "method": meth_label, "floor": floor_label,
                    "floor_source": floor_source, "floor_negctx_mode": floor_mode,
                    "up_nodes": n_lat, "n_sites_up": len(up),
                    "pct_dict_up": round(100 * n_lat / max(len(up), 1) / D_SAE, 3),
                    "free0": phi(keep_lat, a_e0, tag="free0"),
                    "freeM_dense": phi(keep_lat, a_eM, site_means=means_up,
                                       tag="freeM_dense"),
                    "freeM_topk": phi(keep_lat, a_eMT, site_means=means_up,
                                      respect_topk=True, tag="freeM_topk"),
                    "freeN": phi(keep_lat, a_eNfix, site_means=means_neg, tag="freeN"),
                    "pinMC_dense": phi(keep_lat, a_eM, site_means=means_up,
                                       pin_values=pins, tag="pinMC_dense"),
                    "pinNC": phi(keep_lat, a_eNfix, site_means=means_neg,
                                 pin_values=pins, tag="pinNC"),
                    "cf": cf_v, "sup": sup_v,
                    "faith_dense": (round((mc_d - m_e_d) / (m_full - m_e_d), 4)
                                    if abs(m_full - m_e_d) > 1e-9 else None),
                    "a_eN": a_eN,
                    "n_floor_neg": int(nt.shape[0]) if nt is not None else None,
                    "secs_discover": round(secs, 1),
                }
                row["a_pos"] = round(float(a_pos), 4)
                row["a_e0"] = round(float(a_e0), 4)
                row["a_eM"] = round(float(a_eM), 4)
                row["a_eMT"] = round(float(a_eMT), 4)
                row["a_eNfix"] = round(float(a_eNfix), 4)
                row.update({"ac_" + t: v for t, v in raw_ac.items()})
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print("  %-36s n=%7d free0=%-7s freeN=%-7s freeM=%-7s pinMC=%-7s "
                      "pinNC=%-7s cf=%-7s faith=%-7s (%.0fs)"
                      % (arm, n_lat, row["free0"], row["freeN"], row["freeM_dense"],
                         row["pinMC_dense"], row["pinNC"], row["cf"],
                         row["faith_dense"], secs), flush=True)
                del circ
            except Exception as exc:
                print("  %-36s FAILED %s: %s" % (arm, type(exc).__name__, exc), flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

fh.close()
print("\nwrote %s" % OUT)
