"""D3.6 — Mask-refine over attribution support: 24-seed run.

Per seed:
  support source R = abl-restoration PA ranking (rounds=sites), the D1
  intervention-driver winner: from the D2.2 archive where it exists,
  discovered once and archived here otherwise.
  MF    house-recipe mask, FULL dictionary (baseline; config defaults —
        dual floor, lambda 1e-5, 400 steps, anneal)
  MS50  same recipe, support = R top-50,000
  MS10  same recipe, support = R top-10,000
Same lambda across arms — the controlled comparison is "same pressure,
different search space".

Metrics per arm: n members, free0 on the held-out split, cf/sup on the
frozen exam, holdout data loss, wall clock. Plus the ORIGINAL D3.6 gate
measurement on MF: containment of its members in R top-10k/50k; and
Jaccard(MF, MS*) — does restricting the space change the answer or just
the search? Member lists archived as members_{arm}_{seed}.jsonl.gz.

Panel: the 11 D1 seeds + 13 new layer-stratified picks (same rng-42
queue construction as D1; candidates without positives are skipped
deterministically).

  PYTHONPATH=src python experiments/018-maskrefine/runner.py
"""
import gzip
import json
import os
import random
import time
from collections import defaultdict
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
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from store.circuits import Circuit, CircuitNode

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
D22 = HERE.parent / "019-roles-drivers"
N_SEQ, N_TR, EVAL_BS, PA_PCTL = 64, 48, 16, 90.0
SUPPORT_NS = {"MS50": 50000, "MS10": 10000}
D_SAE = 40960
TARGET_SEEDS = 24
torch.set_float32_matmul_precision("high")

D1_PANEL = [(2, 19766), (8, 20333), (9, 38734), (13, 30053), (17, 38268),
            (20, 35678), (25, 10628), (26, 17432), (27, 6859), (29, 2753),
            (35, 6599)]

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
cf_cfg, ab_cfg = disc.counterfactual_gradient, disc.ablation_gradient
lm = disc.learned_mask

# ---- candidate queue (extends the D1 construction deterministically) ------
_all_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu",
                       weights_only=False)
_by_layer = defaultdict(list)
for _i, _c in enumerate(_all_cand):
    _by_layer[int(_c["comp_idx"]) // n_kinds].append(_i)
_rng = random.Random(42)
for _L in sorted(_by_layer):
    _rng.shuffle(_by_layer[_L])

def candidate_queue():
    """D1 panel first, then round-robin layers from each shuffled list's
    tail (D1 took [-1]); dedup."""
    seen = set(D1_PANEL)
    for s in D1_PANEL:
        yield s
    depth = 2
    while depth < 40:
        for L in sorted(_by_layer):
            lst = _by_layer[L]
            if depth <= len(lst):
                i = lst[-depth]
                s = (int(_all_cand[i]["comp_idx"]),
                     int(_all_cand[i]["latent_idx"]))
                if s not in seen:
                    seen.add(s)
                    yield s
        depth += 1


def base_state():
    disc.probe_sequence_count = N_TR
    disc.eval_sequence_count = N_TR
    disc.eval_batch_size = EVAL_BS
    disc.probe_batch_size = 4
    disc.position_aware = True
    disc.position_aware_select = "abs_pctl"
    disc.position_aware_threshold = PA_PCTL
    disc.floor_source = "posctx"
    disc.magnitude_prune = False
    disc.recurrence_prune = False
    disc.min_faithfulness = -100.0
    cf_cfg.max_neg_sequences = N_TR
    cf_cfg.neg_batch_size = 8
    cf_cfg.negative_roles = "include"
    ab_cfg.negative_roles = "include"
    cf_cfg.pruning_threshold = 0.0
    ab_cfg.pruning_threshold = 0.0
    for c in (ab_cfg, cf_cfg):
        c.restoration.round_select = "abs_pctl"
        c.restoration.round_abs_pctl = 95.0


KNOWN_ROLES = {"counterfactual_activator", "counterfactual_inhibitor",
               "ablation_support"}

OUT = HERE / "rows.jsonl"
fh = OUT.open("a")
done = set()
if OUT.exists():
    for line in open(OUT):
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["arm"]))

if os.environ.get("SMOKE"):
    TARGET_SEEDS = 1
    lm.steps = 25
    OUT = HERE / "rows_smoke.jsonl"
    fh.close(); fh = OUT.open("a")
    print("SMOKE MODE: 1 seed, 25 steps", flush=True)

n_seeds_run = 0
for sc_idx, sl in candidate_queue():
    if n_seeds_run >= TARGET_SEEDS:
        break
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    up_sorted = sorted(up)
    if not up:
        continue

    base_state()
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    if pd_.pos_tokens.shape[0] == 0:
        print("[%s] no positives — skip (not counted)" % seed_key, flush=True)
        continue

    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TR], pa[:N_TR], nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]
    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    # Degenerate-candidate guard (added after pass 1): a seed whose eval
    # anchors never fire (a_pos ~ 0) or with no negatives cannot be trained
    # (dual floor) or scored (den ~ 0) — skip WITHOUT consuming a slot so
    # the deterministic queue backfills a live seed instead.
    if a_pos_ev <= 0.05 or nt.shape[0] == 0:
        print("[%s] degenerate (a_pos %.3f, negs %d) — skip (not counted)"
              % (seed_key, a_pos_ev, int(nt.shape[0])), flush=True)
        continue
    n_seeds_run += 1
    if all((seed_key, a) in done for a in ("MF", "MS50", "MS10")):
        continue
    a_e0_ev = float(circuit_only_activation(inference, bank, {}, up, pt_ev,
                                            layer, kind, sl, pos_argmax=pa_ev,
                                            batch_size=EVAL_BS))
    den_ev = a_pos_ev - a_e0_ev

    # ---- R support ranking (archive or fresh) -----------------------------
    apath = D22 / ("ranking_R_%d_%d.jsonl.gz" % (sc_idx, sl))
    if not apath.exists():
        apath = HERE / ("ranking_R_%d_%d.jsonl.gz" % (sc_idx, sl))
    if not apath.exists():
        base_state()
        for c in (ab_cfg, cf_cfg):
            c.restoration.rounds = max(1, len(up))
        meth = _build_mode_method("ablation_gradient", "restoration", inference,
                                  bank, avg_acts, probe_builder)
        t0 = time.time()
        circ = meth.discover(sc_idx, sl)
        del meth
        if circ is None:
            print("[%s] R discovery empty — skip" % seed_key, flush=True)
            continue
        ents = []
        for node in circ.nodes.values():
            role = node.metadata.get("role")
            if role == "seed":
                continue
            f = node.feature_id
            if f is None or (f.layer, f.kind) not in up:
                continue
            sc = node.metadata.get("effect_score")
            if sc is None:
                sc = node.metadata.get("attribution_score") or 0.0
            rr = node.metadata.get("selected_round", 0)
            role_n = role if role in KNOWN_ROLES else "ablation_support"
            ents.append((rr, abs(float(sc)), (f.layer, f.kind), int(f.index),
                         role_n))
        ents.sort(key=lambda x: (x[0], -x[1]))
        del circ
        with gzip.open(apath, "wt", encoding="utf-8") as gz:
            for rr, s, (l, kd), idx, role in ents:
                gz.write(json.dumps([round(s, 6), l, kd, idx, role, rr]) + "\n")
        print("[%s] R support discovered: %d members in %.0fs"
              % (seed_key, len(ents), time.time() - t0), flush=True)
        torch.cuda.empty_cache()

    rank = []
    with gzip.open(apath, "rt", encoding="utf-8") as gz:
        for line in gz:
            s, l, kd, idx, role, rr = json.loads(line)
            rank.append(((l, kd), int(idx)))
            if len(rank) >= max(SUPPORT_NS.values()):
                break
    n_rank_head = len(rank)

    def support_of(n):
        sup = {}
        for (site, idx) in rank[:n]:
            sup.setdefault(site, []).append(idx)
        return {site: torch.tensor(sorted(set(ix)), dtype=torch.long)
                for site, ix in sup.items()}

    print("\n[%s] L%d %s — %d sites | a_pos %.3f | R head %d"
          % (seed_key, layer, kind, len(up), a_pos_ev, n_rank_head), flush=True)

    def keep_of(members):
        keep = {}
        for (site, idx), _m in members:
            keep.setdefault(site, set()).add(idx)
        return keep

    def phi0(members):
        if abs(den_ev) < 1e-9:
            return None
        a_c = float(circuit_only_activation(
            inference, bank, keep_of(members), up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS))
        return round((a_c - a_e0_ev) / den_ev, 4)

    def cf_eval(members):
        c = Circuit(name="d36")
        for (l, kd), idx in [m for m, _ in members]:
            c.add_node(CircuitNode(metadata={
                "layer_idx": l, "kind": kd, "latent_idx": idx,
                "role": "ablation_support"}))
        try:
            cf_v, sup_v = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, c, neg_tokens=nt_ev, pos_tokens=pt_ev,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_argmax=pa_ev,
                circuit_layers={l for ((l, _), _), _ in
                                [(m, 0) for m, _ in members]})
            return round(float(cf_v), 4), round(float(sup_v), 4)
        except Exception as exc:
            print("    cf_eval error: %s" % str(exc)[:80], flush=True)
            return None, None

    members_by_arm = {}
    for arm in ("MF", "MS50", "MS10"):
        if (seed_key, arm) in done:
            continue
        support = None if arm == "MF" else support_of(SUPPORT_NS[arm])
        t0 = time.time()
        try:
            scores, prov = run_learned_mask(
                inference, bank, objective="pos", sites=up_sorted,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_tokens=pt_tr, pos_argmax=pa_tr, neg_tokens=nt_tr,
                mask_floor_source=lm.mask_floor_source,
                dual_floor_weight=lm.dual_floor_weight,
                binarize=lm.binarize, steps=lm.steps, lr=lm.lr,
                l1_lambda=lm.l1_lambda, keep_threshold=lm.keep_threshold,
                batch_size=4, holdout_frac=lm.holdout_frac,
                theta_init=lm.theta_init, log_every=0,
                deep_site_threshold=lm.deep_site_threshold,
                deep_batch_size=lm.deep_batch_size,
                optimizer=lm.optimizer, weight_decay=lm.weight_decay,
                code_dtype=lm.code_dtype, lr_schedule=lm.lr_schedule,
                lr_min_frac=lm.lr_min_frac, warmup_frac=lm.warmup_frac,
                support=support)
        except Exception as exc:
            print("  %s ERROR %s: %s" % (arm, type(exc).__name__,
                                         str(exc)[:100]), flush=True)
            continue
        secs = round(time.time() - t0, 1)
        members = [(((f.layer, f.kind), f.index), float(m))
                   for f, m in scores.items()]
        members_by_arm[arm] = {m for m, _ in members}
        mpath = HERE / ("members_%s_%d_%d.jsonl.gz" % (arm, sc_idx, sl))
        with gzip.open(mpath, "wt", encoding="utf-8") as gz:
            for ((l, kd), idx), m in members:
                gz.write(json.dumps([l, kd, idx, round(m, 4)]) + "\n")

        t1 = time.time()
        f0 = phi0(members)
        cf_v, sup_v = cf_eval(members)
        row = {
            "seed": seed_key, "layer": layer, "kind": kind, "arm": arm,
            "support_n": prov.get("support_n"), "n": len(members),
            "pct_dict": round(100.0 * len(members) / (len(up) * D_SAE), 4),
            "free0": f0, "cf": cf_v, "sup": sup_v,
            "holdout_loss": prov.get("holdout_data_loss"),
            "loss_final": prov.get("loss_final"),
            "mean_m_kept": prov.get("mean_m_kept"),
            "secs_train": secs, "secs_eval": round(time.time() - t1, 1),
        }
        # containment / agreement columns
        if arm == "MF":
            mf = members_by_arm["MF"]
            for label, n in (("in_r10k", 10000), ("in_r50k", 50000)):
                topn = set(rank[:n])
                row[label] = (round(len([m for m in mf if m in topn])
                                    / max(len(mf), 1), 4) if mf else None)
        else:
            mf = members_by_arm.get("MF")
            if mf is not None:
                ms = members_by_arm[arm]
                inter = len(mf & ms)
                row["jaccard_vs_MF"] = round(
                    inter / max(len(mf | ms), 1), 4)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-4s n=%7d free0=%-8s cf=%-8s sup=%-8s ho=%-9s "
              "train %ss %s"
              % (arm, len(members), f0, cf_v, sup_v, row["holdout_loss"],
                 secs,
                 ("cont10k=%s cont50k=%s" % (row.get("in_r10k"),
                                             row.get("in_r50k")))
                 if arm == "MF" else "jacc=%s" % row.get("jaccard_vs_MF")),
              flush=True)
    torch.cuda.empty_cache()

print("ALL DONE (%d seeds)" % n_seeds_run, flush=True)
fh.close()
