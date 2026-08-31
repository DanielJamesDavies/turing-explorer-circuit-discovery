"""The activators-only question, measured UNCENSORED (seed pre-activation).

The post-top-k read used by free0 is floored at 0: a seed that falls out of its
SAE's top-k reads exactly 0.000 however far below the cutoff it sits. In the
free0-actonly run that censored 70 of 95 rows, so "are the inhibitors really
activators?" was unanswerable — a 0.000 is consistent with both "removing them
removed drive" and "removing them left the seed just below threshold".

Pre-activation (w_seed . x + b_seed) is continuous and signed, and is the SAME
quantity discovery optimises (the ig_mean "drive" objective), so this also
closes a discovery/evaluation metric mismatch.

Per (seed, arm), all in pre-activation units:
  p_pos     natural run (no intervention)          — the target
  p_e0      empty circuit, everything zeroed       — the floor
  p_all     all members (both signs)               — standard free0's state
  p_act     activators only, inhibitors zeroed     — Daniel's proposal
  p_inh     inhibitors only
and the normalised phi = (p - p_e0) / (p_pos - p_e0) for each, which is free0's
formula computed on an uncensored measurement.

DECISIVE READ: if p_act > p_all, the inhibitor members were suppressing (their
removal raises drive) — Daniel's model holds. If p_act < p_all, they were net
contributing drive at that seed.

  SEED_IDX=0..9 PYTHONPATH=src python experiments/008-learned-mask-spike/runner.py
"""
import json
import os
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import circuit_only_activation, upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
CIRC = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2/experiments/"
            "007-free0-cf-32seed/circuits")
HERE = Path(__file__).parent
SEED_SEL = int(os.environ["SEED_IDX"]) if os.environ.get("SEED_IDX") else None
OUT = HERE / ("rows_s%d.jsonl" % SEED_SEL if SEED_SEL is not None else "rows.jsonl")

ARMS = [("sfc-npa", "sfc-npa__own-threshold"),
        ("sfc-pa_union", "sfc-pa_union__own-threshold"),
        ("abl-ig_mean PA", "abl-ig_mean_PA__raw"),
        ("abl-ig_mean PA +rec2+mag", "abl-ig_mean_PA__rec2mag"),
        ("abl-restoration PA", "abl-restoration_PA__raw"),
        ("abl-restoration PA +rec2+mag", "abl-restoration_PA__rec2mag"),
        ("cf-ig_mean PA", "cf-ig_mean_PA__raw"),
        ("cf-ig_mean PA +rec2+mag", "cf-ig_mean_PA__rec2mag"),
        ("cf-ig_negctx PA", "cf-ig_negctx_PA__raw"),
        ("cf-ig_negctx PA +rec2+mag", "cf-ig_negctx_PA__rec2mag")]
N_SEQ, EVAL_BS, NK = 64, 16, 3
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * NK, bank.d_sae), device=bank.device)

_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4

seed_dirs = sorted(os.listdir(CIRC))
todo = seed_dirs if SEED_SEL is None else [seed_dirs[SEED_SEL]]
done = set()
if OUT.exists():
    for line in OUT.open():
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["arm"]))

fh = OUT.open("a")
for sd in todo:
    sc_idx, latent = (int(x) for x in sd.split("_"))
    seed_key = "%d/%d" % (sc_idx, latent)
    layer, ki = split_component_idx(sc_idx, NK)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, latent)
    if pd_.pos_tokens.shape[0] == 0:
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]

    def pre(keep):
        return float(circuit_only_activation(
            inference, bank, keep, up, pt, layer, kind, latent, pos_argmax=pa,
            batch_size=EVAL_BS, preact=True))

    # natural: keep every latent at every upstream site (keep-all is identity)
    keep_nat = {s: set(range(bank.d_sae)) for s in up}
    p_pos = pre(keep_nat)
    p_e0 = pre({})
    den = p_pos - p_e0
    print("\n[%s] L%d %s | p_pos %.4f  p_e0 %.4f  (den %.4f)"
          % (seed_key, layer, kind, p_pos, p_e0, den), flush=True)
    print("  %-30s %9s %9s %9s | %8s %8s"
          % ("arm", "p_all", "p_act", "p_inh", "phi_all", "phi_act"), flush=True)

    for arm, f in ARMS:
        if (seed_key, arm) in done:
            continue
        p = CIRC / sd / (f + ".pt")
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        roles = [d["roles_legend"][i] for i in d["role"].tolist()]
        k_all, k_act, k_inh = {}, {}, {}
        n_act = n_inh = 0
        for (l, k, i, s), r in zip(zip(d["layer"].tolist(), d["kind_idx"].tolist(),
                                       d["index"].tolist(), d["score"].tolist()), roles):
            kd = d["kinds_legend"][k]
            if r in ("seed", "residual") or (l, kd) not in up:
                continue
            k_all.setdefault((l, kd), set()).add(i)
            if ("inhibitor" in r) or (r == "attributed" and float(s) < 0):
                k_inh.setdefault((l, kd), set()).add(i); n_inh += 1
            else:
                k_act.setdefault((l, kd), set()).add(i); n_act += 1

        p_all, p_act, p_inh = pre(k_all), pre(k_act), pre(k_inh)
        phi = lambda v: round((v - p_e0) / den, 4) if abs(den) > 1e-9 else None
        row = {"seed": seed_key, "layer": layer, "kind": kind, "arm": arm,
               "n_act": n_act, "n_inh": n_inh,
               "p_pos": round(p_pos, 4), "p_e0": round(p_e0, 4),
               "p_all": round(p_all, 4), "p_act": round(p_act, 4),
               "p_inh": round(p_inh, 4),
               "phi_all": phi(p_all), "phi_act": phi(p_act), "phi_inh": phi(p_inh),
               "act_minus_all": round(p_act - p_all, 4)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-30s %9.4f %9.4f %9.4f | %8s %8s"
              % (arm, p_all, p_act, p_inh, row["phi_all"], row["phi_act"]), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
fh.close()
print("\nwrote %s" % OUT)
