"""EDGE AUDIT of a fitted tri-amp circuit: per-member causal weights.

The learned mask is the SEARCH (it cuts 1.5M upstream latents to a few
hundred members); this is the VERIFICATION of individual member->seed
edges, which set-level validation does not give:

  necessity share   full circuit, member i alone clamped to fill
                    -> nec_i = (B_full - B_drop_i) / (B_full - e0)
  sufficiency share member i alone restored, everything else at fill
                    -> suf_i = (B_only_i - e0)   / (B_full - e0)
  synergy           double knockouts over the top-K members:
                    syn_ij = [dB(i&j) - dB(i) - dB(j)] / (B_full - e0)
                    superadditive pairs (syn >> 0) are BINDINGS --
                    the fact-shaped structure; additive members are
                    associations.

alpha is a calibration coefficient, NOT causal weight; this measures
the causal weights directly. Zero-fill discipline throughout; all reads
on HELD-OUT probes at the seed argmax, THROUGH THE CANONICAL EVALUATOR
(circuit_only_activation + a new per-latent keep_scales argument) in
the PRE-ACTIVATION frame. The runner-style delta-injection patcher is
NOT used: the 2026-08-27 diagnostic showed it explodes numerically
(read 1.47e6 vs canonical 0.0) when many sites are zero-filled at once
-- a configuration the runner never evaluates but this audit does.

  COMP=35 LAT=13633 [TOPK_INT=20] [SMOKE=1] PYTHONPATH=src python \
      experiments/043-relativity/edge_audit.py
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, upstream_sites)
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP = int(os.environ["COMP"])
LAT = int(os.environ["LAT"])
TOPK_INT = int(os.environ.get("TOPK_INT", 20))
SMOKE = os.environ.get("SMOKE") == "1"
N_SEQ, N_TRAIN, EVAL_BS = 64, 48, 16

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(),
               compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.floor_source = "posctx"


def main():
    alphas = None
    for line in open(HERE / os.environ.get("MEMFILE", "members.jsonl")):
        r = json.loads(line)
        if (r["comp_idx"], r["latent"], r["arm"]) == (COMP, LAT, os.environ.get("AUDIT_ARM", "triamp400")):
            alphas = {}
            for site, d in r["alphas"].items():
                lyr, knd = site.split("/")
                alphas[(int(lyr), knd)] = {int(i): float(a)
                                           for i, a in d.items()}
    assert alphas, "no members for %d/%d" % (COMP, LAT)
    members = [(s, i) for s, d in alphas.items() for i in d]
    print("seed c%d/%d: %d members" % (COMP, LAT, len(members)), flush=True)

    layer, ki = split_component_idx(COMP, n_kinds)
    kind = bank.kinds[ki]
    avg_acts = torch.zeros((bank.n_layer * n_kinds, bank.d_sae),
                           device=bank.device)
    m0 = _build_mode_method("counterfactual_gradient", "local", inference,
                            bank, avg_acts, pb)
    pd_ = m0.build_probe_dataset(COMP, LAT)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    pt_tr, pa_tr = pt[:N_TRAIN], pa[:N_TRAIN]
    pt_ho, pa_ho = pt[N_TRAIN:], pa[N_TRAIN:]
    UP = sorted(upstream_sites(bank, layer, kind))
    # MEAN-FILL frame (the FMd side of the paper), because the zero-fill
    # audit regime is out-of-distribution: with every site zero-filled
    # the pre-activation explodes to ~1.6e6 (two independent patcher
    # implementations agree), and the historical e0 "0.000" is that
    # explosion CENSORED by the post-top-k read. Mean-filled non-members
    # keep the stream in-distribution, so single-member knockouts and
    # restores are meaningful. Means from TRAIN, reads on HELD-OUT.
    means_tr, _pins = collect_site_anchors(inference, bank, pt_tr,
                                           set(UP), pa_tr,
                                           pin_position_specific=False)

    a_pos_ho = float(circuit_only_activation(
        inference, bank, {}, set(), pt_ho, layer, kind, LAT,
        pos_argmax=pa_ho, preact=True, batch_size=EVAL_BS))
    e0_ho = float(circuit_only_activation(
        inference, bank, {}, UP, pt_ho, layer, kind, LAT,
        pos_argmax=pa_ho, site_means=means_tr, preact=True,
        batch_size=EVAL_BS))

    def read(al):
        keep = {st: set(d) for st, d in al.items() if d}
        scales = {}
        for st, d in al.items():
            if not d:
                continue
            v = torch.ones(bank.d_sae)
            for i, a in d.items():
                v[int(i)] = float(a)
            scales[st] = v
        return float(circuit_only_activation(
            inference, bank, keep, UP, pt_ho, layer, kind, LAT,
            pos_argmax=pa_ho, site_means=means_tr, keep_scales=scales,
            preact=True, batch_size=EVAL_BS))

    def drop(*rm):
        al = {s: dict(d) for s, d in alphas.items()}
        for s, i in rm:
            al[s].pop(i, None)
        return al

    def only(s, i):
        return {s2: ({i: alphas[s][i]} if s2 == s else {}) for s2 in alphas}

    B_full = read(alphas)
    denom = B_full - e0_ho
    print("a_pos_ho %.3f | e0 %.3f | B_full %.3f (F0 %.3f) | denom %.3f"
          % (a_pos_ho, e0_ho, B_full,
             (B_full - e0_ho) / max(a_pos_ho - e0_ho, 1e-9), denom),
          flush=True)
    assert abs(denom) > 1e-6, "vacuous audit"

    audit = members[:10] if SMOKE else members
    out = HERE / ("edge_audit_c%d_%d_%s.jsonl"
              % (COMP, LAT, os.environ.get("AUDIT_ARM", "triamp400")))
    fh = out.open("w")
    rows = []
    for k, (s, i) in enumerate(audit):
        nec = (B_full - read(drop((s, i)))) / denom
        suf = (read(only(s, i)) - e0_ho) / denom
        r = {"site": "%d/%s" % s, "latent": i, "alpha": alphas[s][i],
             "necessity": round(nec, 4), "sufficiency": round(suf, 4)}
        rows.append(r)
        fh.write(json.dumps(r) + "\n")
        fh.flush()
        if (k + 1) % 25 == 0:
            print("  %d/%d members audited" % (k + 1, len(audit)), flush=True)

    rows.sort(key=lambda r: -abs(r["necessity"]))
    top = rows[:TOPK_INT]
    print("\ntop members by |necessity| (edge weight):", flush=True)
    print("%-10s %-7s %6s %8s %8s" % ("site", "lat", "alpha", "nec", "suf"))
    for r in top:
        print("%-10s %-7d %6.2f %8.4f %8.4f"
              % (r["site"], r["latent"], r["alpha"], r["necessity"],
                 r["sufficiency"]))

    print("\ninteraction matrix (top %d): syn_ij = [dB(ij)-dB(i)-dB(j)]/denom"
          % len(top), flush=True)
    dB = {}
    for r in top:
        lyr, knd = r["site"].split("/")
        key = ((int(lyr), knd), r["latent"])
        dB[key] = B_full - read(drop(key))
    keys = list(dB)
    syn_rows = []
    for a_ in range(len(keys)):
        for b_ in range(a_ + 1, len(keys)):
            i, j = keys[a_], keys[b_]
            dij = B_full - read(drop(i, j))
            syn = (dij - dB[i] - dB[j]) / denom
            syn_rows.append({"i": "%d/%s-%d" % (i[0][0], i[0][1], i[1]),
                             "j": "%d/%s-%d" % (j[0][0], j[0][1], j[1]),
                             "syn": round(float(syn), 4)})
            fh.write(json.dumps(syn_rows[-1]) + "\n")
    fh.close()
    syn_rows.sort(key=lambda r: -abs(r["syn"]))
    print("largest |synergies|:")
    for r in syn_rows[:15]:
        print("  %-18s x %-18s syn %+0.4f" % (r["i"], r["j"], r["syn"]))
    print("\n-> %s" % out.name)


if __name__ == "__main__":
    main()
