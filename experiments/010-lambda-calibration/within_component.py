"""Is circuit size a COMPONENT property or a SEED property?

Every lambda measurement so far used ONE seed per component - L2-resid
(comp 8), L5-mlp (16), L8-mlp (25), L10-resid (32) - so component and seed are
PERFECTLY CONFOUNDED. "Calibrate lambda per component" was proposed as if its
assumption were mild; it is in fact the whole question, and nothing measured
so far bears on it.

The across-component spread of n at a fixed lambda=1e-5 is 23x:

    comp  8  n = 1,285
    comp 16  n = 9,194
    comp 25  n = 29,493
    comp 32  n = 26,577

This runs 5 seeds INSIDE one component, at that same fixed lambda, and looks
at the spread:

  * tight (say <=1.5x)  -> variation is component-level. Per-component
                           calibration works: ~3 seeds per component amortises
                           to ~1% overhead over a 16k-seed run.
  * wide (approaching 23x) -> variation is seed-level. Per-component
                           calibration is dead, and predicting lambda needs
                           SEED features, which is the interesting version of
                           the question anyway.

Two components, one shallow and one deep, in case the answer depends on depth.

Seed-level features are recorded alongside (firing rate, a_pos, posctx
pre-activation reference, upstream site count) so that if the spread IS wide,
we already have a first look at what correlates with it - for free, without
committing to the ~700-run sweep.

  PYTHONPATH=src python .../within_component.py
"""
import json
import os
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import measure_seed_activation, upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
import circuit.instrument.learned_mask as lm

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ, EVAL_BS, GAMMA = 64, 16, 0.25
LAMBDA = 1e-5                    # the same probe lambda used everywhere else
N_PER_COMPONENT = int(os.environ.get("N_PER_COMPONENT", "5"))
COMPONENTS = [int(x) for x in os.environ.get("COMPONENTS", "8,32").split(",")]
torch.set_float32_matmul_precision("high")

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
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.probe_batch_size = 4
disc.position_aware = False
disc.magnitude_prune = False
disc.recurrence_prune = False

_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu", weights_only=False)
LS = torch.load(RUN_ROOT / "latent_stats.pt", map_location="cpu", weights_only=False)
RATE = LS["active_count"].float() / (6060 * 262144.0)   # as in the hub-prune work

OUT = HERE / "within_component.jsonl"
done = set()
if OUT.exists():
    for line in OUT.open():
        if line.strip():
            r = json.loads(line)
            done.add((r["comp_idx"], r["latent"]))
fh = OUT.open("a")

for comp in COMPONENTS:
    layer, ki = split_component_idx(comp, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))
    # candidates.pt order is the pipeline's own ranking; take the first N in
    # that component that actually have positive contexts. Deterministic.
    pool = [int(c["latent_idx"]) for c in _cand if int(c["comp_idx"]) == comp]
    print("\n=== component %d  (L%d %s, %d sites) | %d candidates available ==="
          % (comp, layer, kind, len(up), len(pool)), flush=True)
    print("  %-9s %-10s %9s %10s %11s %10s"
          % ("latent", "n", "rate%", "a_pos", "posctx_ref", "secs"), flush=True)

    taken = 0
    for latent in pool:
        if taken >= N_PER_COMPONENT:
            break
        if (comp, latent) in done:
            taken += 1
            continue
        meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                                  avg_acts, probe_builder)
        pd_ = meth.build_probe_dataset(comp, latent)
        if pd_.pos_tokens.shape[0] == 0:
            continue                       # no positives; not a usable seed
        pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
        nt = pd_.neg_tokens[:N_SEQ]
        a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind,
                                              latent, pa, batch_size=EVAL_BS))
        ref = meth._neg_context_selector().posctx_reference(
            pt, comp, latent,
            batch_size=int(disc.neg_context_selection.filter_batch_size))
        t0 = time.perf_counter()
        scores, _ = lm.run_learned_mask(
            inference, bank, objective="pos", sites=up,
            seed_layer=layer, seed_kind=kind, seed_latent_idx=latent,
            pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
            mask_floor_source="dual", dual_floor_weight=GAMMA,
            steps=400, lr=0.05, l1_lambda=LAMBDA, keep_threshold=0.5,
            batch_size=4, holdout_frac=0.25, log_every=0,
            deep_site_threshold=disc.learned_mask.deep_site_threshold,
            deep_batch_size=disc.learned_mask.deep_batch_size,
            optimizer="adamw", weight_decay=0.05,
            code_dtype=disc.learned_mask.code_dtype)
        secs = time.perf_counter() - t0
        n = len(scores)
        del scores
        torch.cuda.empty_cache()
        rate = float(RATE[comp, latent])
        row = {"comp_idx": comp, "layer": layer, "kind": kind,
               "latent": latent, "sites": len(up), "lambda": LAMBDA, "n": n,
               "firing_rate": rate, "a_pos": round(a_pos, 4),
               "posctx_ref": round(float(ref or 0.0), 4),
               "secs": round(secs, 1)}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print("  %-9d %-10s %8.3f%% %10.3f %11.3f %10.0f"
              % (latent, format(n, ","), 100 * rate, a_pos, ref or 0.0, secs),
              flush=True)
        taken += 1

fh.close()

rows = [json.loads(l) for l in OUT.open() if l.strip()]
print("\n%-6s %-8s %10s %10s %10s %9s"
      % ("comp", "n_seeds", "min n", "median n", "max n", "spread"))
for comp in sorted({r["comp_idx"] for r in rows}):
    ns = sorted(r["n"] for r in rows if r["comp_idx"] == comp)
    if not ns:
        continue
    print("%-6d %-8d %10s %10s %10s %8.2fx"
          % (comp, len(ns), format(ns[0], ","), format(ns[len(ns) // 2], ","),
             format(ns[-1], ","), ns[-1] / max(ns[0], 1)))
print("\nFor reference, the ACROSS-component spread at this lambda is 23x")
print("(comp 8: 1,285 -> comp 25: 29,493). If within-component spread")
print("approaches that, size is a SEED property and per-component calibration")
print("cannot work.")
