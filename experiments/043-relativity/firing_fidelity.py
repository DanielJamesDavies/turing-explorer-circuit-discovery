"""FIRING FIDELITY: per held-out probe, does the seed FIRE (survive its
SAE's top-k) under circuit-only execution? The evaluation the margin
objective implies -- and a check the value-frame gate never made.

Read = canonical evaluator, post-top-k (censored) activation per
sequence: fired iff the read > 0. Mean-fill frame. Reports firing
rate and mean fired-value for: natural, and each requested arm's
circuit.

  COMP=29 LAT=3736 ARMS=triamp400,mrgamp400 MEMFILE=know_members.jsonl \
    PYTHONPATH=src python .../firing_fidelity.py
(falls back to members.jsonl for arms not found in MEMFILE)
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
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/"
                "Runs/20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP = int(os.environ["COMP"])
LAT = int(os.environ["LAT"])
ARMS = os.environ.get("ARMS", "triamp400,mrgamp400").split(",")
MEMFILE = os.environ.get("MEMFILE", "know_members.jsonl")
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


def load_alphas(arm):
    for fn in (MEMFILE, "members.jsonl"):
        p = HERE / fn
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if (r["comp_idx"], r["latent"], r["arm"]) == (COMP, LAT, arm):
                out = {}
                for site, d in r["alphas"].items():
                    lyr, knd = site.split("/")
                    out[(int(lyr), knd)] = {int(i): float(a)
                                            for i, a in d.items()}
                return out
    return None


def main():
    layer, ki = split_component_idx(COMP, n_kinds)
    kind = bank.kinds[ki]
    avg = torch.zeros((bank.n_layer * n_kinds, bank.d_sae),
                      device=bank.device)
    m0 = _build_mode_method("counterfactual_gradient", "local", inference,
                            bank, avg, pb)
    pd_ = m0.build_probe_dataset(COMP, LAT)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    pt_tr, pa_tr = pt[:N_TRAIN], pa[:N_TRAIN]
    pt_ho, pa_ho = pt[N_TRAIN:], pa[N_TRAIN:]
    UP = sorted(upstream_sites(bank, layer, kind))
    means_tr, _ = collect_site_anchors(inference, bank, pt_tr, set(UP),
                                       pa_tr, pin_position_specific=False)

    def perseq(al):
        keep = {st: set(d) for st, d in al.items() if d} if al else {}
        scales = {}
        if al:
            for st, d in al.items():
                if not d:
                    continue
                v = torch.ones(bank.d_sae)
                for i, a in d.items():
                    v[int(i)] = float(a)
                scales[st] = v
        vals = []
        for b in range(int(pt_ho.shape[0])):
            vals.append(float(circuit_only_activation(
                inference, bank, keep, UP if al else set(),
                pt_ho[b:b + 1], layer, kind, LAT,
                pos_argmax=pa_ho[b:b + 1],
                site_means=means_tr if al else None,
                keep_scales=scales if al else None)))
        return vals

    nat = perseq(None)
    n_nat = sum(1 for v in nat if v > 0)
    print("seed c%d/%d | natural: fires %d/%d, mean fired %.2f"
          % (COMP, LAT, n_nat, len(nat),
             sum(v for v in nat if v > 0) / max(n_nat, 1)))
    for arm in ARMS:
        al = load_alphas(arm)
        if al is None:
            print("  %-11s (no members found)" % arm)
            continue
        vs = perseq(al)
        n_f = sum(1 for v in vs if v > 0)
        match = sum(1 for a, b in zip(nat, vs)
                    if (a > 0) == (b > 0))
        print("  %-11s fires %2d/%d | firing-match with natural %d/%d | "
              "mean fired %.2f"
              % (arm, n_f, len(vs), match, len(vs),
                 sum(v for v in vs if v > 0) / max(n_f, 1)))


if __name__ == "__main__":
    main()
