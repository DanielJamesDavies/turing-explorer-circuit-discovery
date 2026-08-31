"""free0 of the driver sets used in the D3.7 gates (K = 8/16/64).

free0 keeps members at their NATURAL values and ablates everything else,
so it is a function of MEMBERSHIP ONLY — the injected value (uniform
alpha, capped ceiling, or learned delta) never enters. All D3.7 arms and
AMPC therefore share one free0 per (seed, K). Measured here rather than
asserted, with a size-matched random control and the pre-activation read
alongside the post-top-k one (the censoring caveat from
ablmask_minus_brakes_preact).

  PYTHONPATH=src python experiments/016-gainmask-gate/free0_of_driver_sets.py
"""
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
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
D1 = HERE.parent / "012-driver-bakeoff"
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
KS = (8, 16, 64, 1024)
D_SAE = 40960
torch.set_float32_matmul_precision("high")
SEEDS = [(13, 30053), (25, 10628), (26, 17432), (35, 6599)]

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

fh = (HERE / "free0_driver_sets.jsonl").open("w")
for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    up_sorted = sorted(up)

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    pt_ev, pa_ev = pt[N_TR:], pa[N_TR:]
    a_pos = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                          sl, pa_ev, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, up, pt_ev, layer,
                                         kind, sl, pos_argmax=pa_ev,
                                         batch_size=EVAL_BS))
    den = a_pos - a_e0
    # pre-act anchors (uncensored read)
    sae0 = bank.saes[kind][layer]
    cap = SeedPreActCapture(layer, kind, sae0.encoder.weight[sl].detach(),
                            sae0._get_bias_eff()[sl].detach())
    inference.disable_compile()
    try:
        ch = []
        with torch.no_grad():
            for s0 in range(0, int(pt_ev.shape[0]), EVAL_BS):
                cap.seed_pre_act = None
                inference.forward(pt_ev[s0:s0 + EVAL_BS], patcher=cap,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
                ch.append(cap.seed_pre_act.detach())
    finally:
        inference.enable_compile()
    pre = torch.cat(ch, 0)
    bi = torch.arange(pre.shape[0], device=pre.device)
    a_pos_pre = float(pre[bi, pa_ev[:pre.shape[0]].to(pre.device)
                          .clamp(0, pre.shape[1] - 1)].mean())

    dw = torch.load(D1 / ("direct_full_%d_%d.pt" % (sc_idx, sl)),
                    map_location="cpu", weights_only=False)["direct"]
    tri = []
    for s, w in dw.items():
        v, ix = torch.topk(w, k=min(2048, w.numel()))
        tri += [(float(a), s, int(i)) for a, i in zip(v, ix)]
    tri.sort(key=lambda x: -x[0])
    rank_c = [(s, i) for _, s, i in tri]
    rng = random.Random(5)
    print("\n[%s] L%d %s | a_pos %.3f (pre %.3f) | empty-circuit %.3f"
          % (seed_key, layer, kind, a_pos, a_pos_pre, a_e0), flush=True)

    for K in KS:
        for label, mem in (("direct_K", rank_c[:K]),
                           ("random_K", [(up_sorted[rng.randrange(len(up_sorted))],
                                          rng.randrange(D_SAE))
                                         for _ in range(K)])):
            keep = {}
            for s, i in mem:
                keep.setdefault(s, set()).add(i)
            a_c = float(circuit_only_activation(
                inference, bank, keep, up, pt_ev, layer, kind, sl,
                pos_argmax=pa_ev, batch_size=EVAL_BS))
            a_cp = float(circuit_only_activation(
                inference, bank, keep, up, pt_ev, layer, kind, sl,
                pos_argmax=pa_ev, batch_size=EVAL_BS, preact=True))
            row = {"seed": seed_key, "layer": layer, "kind": kind, "K": K,
                   "set": label, "n": len(mem),
                   "free0": round((a_c - a_e0) / den, 4) if abs(den) > 1e-9 else None,
                   "a_circuit": round(a_c, 4), "a_pos": round(a_pos, 4),
                   "a_circuit_pre": round(a_cp, 4), "a_pos_pre": round(a_pos_pre, 4)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("  K=%-5d %-9s free0=%-8s  post a=%8.3f/%8.3f  pre a=%9.3f/%9.3f"
                  % (K, label, row["free0"], a_c, a_pos, a_cp, a_pos_pre),
                  flush=True)
    torch.cuda.empty_cache()
print("ALL DONE", flush=True)
fh.close()
