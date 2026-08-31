"""PREACT rerun: uncensored version of ablmask_minus_brakes.

The post-top-k read floors at exactly 0.000 once the seed drops
below its SAE cutoff, so the deep collapses were censored. This
repeats every measurement on the seed PRE-activation (continuous,
signed) — free0_pre uses pre-act anchors throughout, and the
brake-native raise is re-measured pre-act too.

Original docstring:
Does the abl-mask closure circuit NEED its hidden brakes?

The overlap result says 73% of learned brakes sit INSIDE the abl-mask
member set, delivered (role-blindly) as supports. If the closure is a
BALANCE set, deleting those members should make the circuit OVERSHOOT
the natural level — free0 > 1 — rather than merely degrade.

Arms per seed (no training; archived member lists only):
  MF            the D3.6 abl-mask circuit as-is
  MF-brakes     the same circuit with the learned brakes removed
  MF-random     control: same number of RANDOM members removed
Columns: free0 (signed, so overshoot is visible), the raw circuit-only
activation vs a_pos, freeN_topk, cf, sup.

  PYTHONPATH=src python experiments/022-inhibmask/ablmask_minus_brakes.py
"""
import gzip
import json
import random
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.discovery.counterfactual_gradient import SeedPreActCapture
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
D36 = HERE.parent / "018-maskrefine"
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
ARM = "lam0.001"
D_SAE = 40960
torch.set_float32_matmul_precision("high")

SEEDS = [(2, 19766), (8, 20333), (9, 38734), (13, 30053),
         (20, 35678), (26, 17432), (27, 6859), (35, 6599)]

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, bank.d_sae), device=bank.device)
_apply_sweep_config(max_per_site=24)
config.discovery.eval_batch_size = EVAL_BS

rows = []
for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    up_sorted = sorted(up)

    mf_path = D36 / ("members_MF_%d_%d.jsonl.gz" % (sc_idx, sl))
    br_path = HERE / ("members_v2_%s_%d_%d.jsonl.gz" % (ARM, sc_idx, sl))
    if not (mf_path.exists() and br_path.exists()):
        print("missing artefacts for %s" % seed_key, flush=True); continue
    mf = set()
    with gzip.open(mf_path, "rt", encoding="utf-8") as gz:
        for line in gz:
            l_, kd_, idx_, v_ = json.loads(line)
            mf.add(((l_, kd_), int(idx_)))
    brakes = set()
    with gzip.open(br_path, "rt", encoding="utf-8") as gz:
        for line in gz:
            l_, kd_, idx_, v_ = json.loads(line)
            brakes.add(((l_, kd_), int(idx_)))
    inside = mf & brakes
    if not inside:
        print("[%s] no brakes inside MF — skip" % seed_key, flush=True); continue

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    nt_tr = nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]
    a_pos = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                          sl, pa_ev, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, up, pt_ev, layer,
                                         kind, sl, pos_argmax=pa_ev,
                                         batch_size=EVAL_BS))
    den = a_pos - a_e0
    # ---- PRE-ACTIVATION anchors (uncensored) --------------------------
    sae0 = bank.saes[kind][layer]
    capp = SeedPreActCapture(layer, kind, sae0.encoder.weight[sl].detach(),
                             sae0._get_bias_eff()[sl].detach())
    inference.disable_compile()
    try:
        pcs = []
        with torch.no_grad():
            for s0 in range(0, int(pt_ev.shape[0]), EVAL_BS):
                capp.seed_pre_act = None
                inference.forward(pt_ev[s0:s0 + EVAL_BS], patcher=capp,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
                pcs.append(capp.seed_pre_act.detach())
    finally:
        inference.enable_compile()
    _pre = torch.cat(pcs, 0)
    _bi = torch.arange(_pre.shape[0], device=_pre.device)
    a_pos_pre = float(_pre[_bi, pa_ev[:_pre.shape[0]].to(_pre.device)
                           .clamp(0, _pre.shape[1] - 1)].mean())
    a_e0_pre = float(circuit_only_activation(
        inference, bank, {}, up, pt_ev, layer, kind, sl, pos_argmax=pa_ev,
        batch_size=EVAL_BS, preact=True))
    den_pre = a_pos_pre - a_e0_pre
    sae_ = bank.saes[kind][layer]
    cap = SeedPreActCapture(layer, kind, sae_.encoder.weight[sl].detach(),
                            sae_._get_bias_eff()[sl].detach())
    inference.disable_compile()
    try:
        ch = []
        with torch.no_grad():
            for s0 in range(0, int(nt_tr.shape[0]), EVAL_BS):
                cap.seed_pre_act = None
                inference.forward(nt_tr[s0:s0 + EVAL_BS], patcher=cap,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
                ch.append(cap.seed_pre_act.detach())
    finally:
        inference.enable_compile()
    na_tr = torch.cat(ch, 0).argmax(dim=1).cpu()
    _, neg_means = collect_site_anchors(inference, bank, nt_tr, up, na_tr,
                                        pin_position_specific=False)

    def score(mem, label):
        keep = {}
        for s, i in mem:
            keep.setdefault(s, set()).add(i)
        a_c = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS))
        a_c_pre = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, preact=True))
        a_n = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS,
            site_means=neg_means, respect_topk=True))
        c = Circuit(name="mf")
        for (l, kd), idx in mem:
            c.add_node(CircuitNode(metadata={
                "layer_idx": l, "kind": kd, "latent_idx": idx,
                "role": "ablation_support"}))
        try:
            cf_v, sup_v = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, c, neg_tokens=nt_ev, pos_tokens=pt_ev,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_argmax=pa_ev, circuit_layers={l for (l, _), _ in mem})
            cf_v, sup_v = round(float(cf_v), 4), round(float(sup_v), 4)
        except Exception:
            cf_v = sup_v = None
        return {"seed": seed_key, "layer": layer, "kind": kind, "arm": label,
                "n": len(mem), "a_circuit": round(a_c, 4),
                "a_pos": round(a_pos, 4),
                "free0": round((a_c - a_e0) / den, 4) if abs(den) > 1e-9 else None,
                "a_circuit_pre": round(a_c_pre, 4),
                "a_pos_pre": round(a_pos_pre, 4),
                "free0_pre": (round((a_c_pre - a_e0_pre) / den_pre, 4)
                              if abs(den_pre) > 1e-9 else None),
                "freeN_topk": round((a_n - a_e0) / den, 4) if abs(den) > 1e-9 else None,
                "cf": cf_v, "sup": sup_v}

    rng = random.Random(23)
    mf_l = sorted(mf)
    drop_rand = set(rng.sample(mf_l, min(len(inside), len(mf_l))))
    for mem, lab in ((mf, "MF"), (mf - inside, "MF-brakes"),
                     (mf - drop_rand, "MF-random")):
        r = score(mem, lab)
        r["n_removed"] = len(mf) - len(mem)
        rows.append(r)
        print("%-10s L%-2d %-5s %-10s n=%-6d (-%d) | POST a=%8.3f free0=%-8s "
              "| PRE a=%9.3f (nat %8.3f) free0_pre=%s"
              % (seed_key, layer, kind, lab, r["n"], r["n_removed"],
                 r["a_circuit"], r["free0"], r["a_circuit_pre"],
                 r["a_pos_pre"], r["free0_pre"]), flush=True)
    torch.cuda.empty_cache()

with (HERE / "ablmask_minus_brakes_preact.jsonl").open("w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print("ALL DONE", flush=True)
