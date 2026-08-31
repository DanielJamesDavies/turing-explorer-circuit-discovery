"""D2.2(a,b) — Roles for drivers: include vs exclude inhibitors at each K,
plus inhibitor fraction by K and depth. ALSO the rankings archive: D1 never
saved rankings, which blocks D4.3/D4.4/D3.6 — this run archives the full
signed rankings per (seed, arm) as gzipped jsonl.

Arms: the two working driver rankings from D1 —
  R  abl-restoration PA (rounds=sites; order = (round, -|score|))
  D  cf-ig_mean PA      (order = -|score|)
Both discovered ONCE per seed with negative_roles="include" so the ranking
CONTAINS inhibitors; exclusion is post-hoc and SIZE-MATCHED: the exclude
variant takes the top-K among non-inhibitors (evidence standard: matched
node counts), not a filtered-and-smaller set. Exam: frozen D0.2 protocol
(48/16 store split), phi-cf/phi-sup + free0/pin0_c per (arm, K, variant).
Recheck target: v1's "excluding inhibitors collapses phi-cf 0.57 -> 0.04".

  PYTHONPATH=src python experiments/019-roles-drivers/runner.py
"""
import gzip
import json
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from store.circuits import Circuit, CircuitNode

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ, N_TR, EVAL_BS, PA_PCTL = 64, 48, 16, 90.0
KS = (64, 256, 1024, 4096)
D_SAE = 40960
torch.set_float32_matmul_precision("high")

SEEDS = [(2, 19766), (8, 20333), (9, 38734), (13, 30053), (17, 38268),
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

KNOWN_ROLES = {"counterfactual_activator", "counterfactual_inhibitor",
               "ablation_support"}
INHIB = "counterfactual_inhibitor"


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


OUT = HERE / "rows.jsonl"
fh = OUT.open("a")
done = set()
if OUT.exists():
    for line in open(OUT):
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["arm"], r["variant"], r["K"]))

for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    dict_up = max(len(up), 1) * D_SAE

    base_state()
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    if pd_.pos_tokens.shape[0] == 0:
        print("[%s] no positives — skip" % seed_key, flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr = pt[:N_TR], pa[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]

    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    a_e0_ev = float(circuit_only_activation(inference, bank, {}, up, pt_ev, layer,
                                            kind, sl, pos_argmax=pa_ev,
                                            batch_size=EVAL_BS))
    den_ev = a_pos_ev - a_e0_ev
    _, pins_c = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                     pin_position_specific=False)

    print("\n[%s] L%d %s — %d sites | eval a_pos %.3f"
          % (seed_key, layer, kind, len(up), a_pos_ev), flush=True)

    def signed_members(circ, by_round=False):
        out = []
        for node in circ.nodes.values():
            role = node.metadata.get("role")
            if role == "seed":
                continue
            f = node.feature_id
            if f is None or (f.layer, f.kind) not in up:
                continue
            sc = node.metadata.get("effect_score")
            if sc is None:
                sc = node.metadata.get("attribution_score")
            if sc is None:
                sc = node.metadata.get("weight") or 0.0
            rr = node.metadata.get("selected_round", 0) if by_round else 0
            role_n = role if role in KNOWN_ROLES else "ablation_support"
            out.append((rr, abs(float(sc)), (f.layer, f.kind), int(f.index), role_n))
        out.sort(key=lambda x: (x[0], -x[1]))
        return [(s, site, idx, role, rr) for rr, s, site, idx, role in out]

    def keep_of(entries):
        keep = {}
        for _, site, idx, _, _ in entries:
            keep.setdefault(site, set()).add(idx)
        return keep

    def phi0(entries, pins=None):
        if abs(den_ev) < 1e-9:
            return None
        a_c = float(circuit_only_activation(
            inference, bank, keep_of(entries), up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, pin_values=pins))
        return round((a_c - a_e0_ev) / den_ev, 4)

    def cf_eval(entries):
        c = Circuit(name="d22")
        for _, (l, kd), idx, role, _ in entries:
            c.add_node(CircuitNode(metadata={
                "layer_idx": l, "kind": kd, "latent_idx": idx, "role": role}))
        try:
            cf_v, sup_v = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, c, neg_tokens=nt_ev, pos_tokens=pt_ev,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_argmax=pa_ev,
                circuit_layers={l for _, (l, _), _, _, _ in entries})
            return round(float(cf_v), 4), round(float(sup_v), 4)
        except Exception as exc:
            print("    cf_eval error: %s" % str(exc)[:80], flush=True)
            return None, None

    # ---- discover once per arm, ARCHIVE the ranking -----------------------
    for arm, method, mode, by_round in (
            ("R", "ablation_gradient", "restoration", True),
            ("D", "counterfactual_gradient", "ig_mean", False)):
        if all((seed_key, arm, v, K) in done
               for v in ("include", "exclude") for K in KS):
            continue
        base_state()
        if arm == "R":
            for c in (ab_cfg, cf_cfg):
                c.restoration.rounds = max(1, len(up))
                c.restoration.round_select = "abs_pctl"
                c.restoration.round_abs_pctl = 95.0
        try:
            meth = _build_mode_method(method, mode, inference, bank, avg_acts,
                                      probe_builder)
            t0 = time.time()
            circ = meth.discover(sc_idx, sl)
            secs = round(time.time() - t0, 1)
            del meth
            if circ is None:
                raise RuntimeError("no circuit")
            rank = signed_members(circ, by_round=by_round)
            del circ
        except Exception as exc:
            print("  %s DISCOVERY ERROR %s: %s"
                  % (arm, type(exc).__name__, str(exc)[:90]), flush=True)
            continue
        torch.cuda.empty_cache()

        apath = HERE / ("ranking_%s_%d_%d.jsonl.gz" % (arm, sc_idx, sl))
        with gzip.open(apath, "wt", encoding="utf-8") as gz:
            for s, (l, kd), idx, role, rr in rank:
                gz.write(json.dumps([round(s, 6), l, kd, idx, role, rr]) + "\n")
        n_inhib_total = sum(1 for e in rank if e[3] == INHIB)
        print("  %s discovered: %d members (%.1f%% inhib) in %ss -> %s"
              % (arm, len(rank), 100.0 * n_inhib_total / max(len(rank), 1),
                 secs, apath.name), flush=True)

        rank_act = [e for e in rank if e[3] != INHIB]
        for K in KS:
            for variant, entries in (("include", rank[:K]),
                                     ("exclude", rank_act[:K])):
                if (seed_key, arm, variant, K) in done or not entries:
                    continue
                t1 = time.time()
                cf_v, sup_v = cf_eval(entries)
                n_inhib = sum(1 for e in entries if e[3] == INHIB)
                row = {
                    "seed": seed_key, "layer": layer, "kind": kind,
                    "arm": arm, "variant": variant, "K": K,
                    "n": len(entries), "n_inhib": n_inhib,
                    "inhib_frac": round(n_inhib / max(len(entries), 1), 4),
                    "pct_dict": round(100.0 * len(entries) / dict_up, 4),
                    "free0": phi0(entries),
                    "pin0_c": phi0(entries, pins=pins_c),
                    "cf": cf_v, "sup": sup_v,
                    "imp_err": (round(abs(cf_v * den_ev + a_e0_ev - a_pos_ev)
                                      / max(a_pos_ev, 1e-9), 4)
                                if cf_v is not None else None),
                    "n_rank_total": len(rank),
                    "inhib_frac_total": round(n_inhib_total / max(len(rank), 1), 4),
                    "secs_disc": secs,
                    "secs_eval": round(time.time() - t1, 1),
                }
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print("  %s/%-7s K=%4d cf=%-7s sup=%-7s inhib%%=%-6s "
                      "pin0c=%-7s free0=%-7s"
                      % (arm, variant, K, cf_v, sup_v,
                         round(100 * row["inhib_frac"], 1),
                         row["pin0_c"], row["free0"]), flush=True)
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
