"""Decisive floor diagnostic: is the mean-floor denominator the source of the
L9 sign flip and the L10 inflation?

Every mean-floor metric (freeM, pinMC) normalises by  den = a_pos - a_empty(F),
where a_empty(F) is the seed's activation with the circuit EMPTY and every
upstream latent pinned to floor F. The 4-seed validation run showed pinMC
uniformly NEGATIVE at L9 (-0.12..-0.17) and inflated at L10 (~2.0) across ALL
arms -- abl, cf, restoration, act-grad, sfc alike. Those arms share exactly one
input: the posctx floor (gradient_base.py:591 and both eval call sites pass
pos_tokens). So the hypothesis is that at L9 the posctx mean already drives the
seed HARDER than the natural run does, i.e.

    a_empty(posctx) > a_pos   ->   den < 0   ->   every positive numerator
                                                  reports as negative.

This measures den under five floors WITHOUT touching the repo: collect_site_means
already takes a `tokens` argument, so the proposed negctx floor is just passing
neg_tokens (which ProbeDataset carries for every seed) instead of pos_tokens.

Decisive outcome:
  * CONFIRMS the diagnosis if den(posctx) < 0 at L9 and small-positive at L10.
  * VALIDATES the negctx floor if den(negctx) is comfortably positive on both,
    i.e. a_empty(negctx) sits well below a_pos.
If den(posctx) is healthy on both seeds, the floor is NOT the problem and the
negctx design should be shelved.

No discovery is run -- these are anchors only, ~5 forward passes per seed.

  PYTHONPATH=src python experiments/005-floor-diagnostic/floor_check.py
"""
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
    circuit_only_activation, collect_site_means, measure_seed_activation,
    upstream_sites)
from eval.floors import collect_diverse_site_floors, collect_global_site_floors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
OUT = HERE / "floor_anchors.jsonl"

# The 4-seed validation run's seeds, so rows join against
# experiments/005-rebuild-validation-4seed/rows_s*.jsonl
SEEDS = [(8, 30122, "L2"), (25, 10628, "L8"), (27, 6859, "L9"), (32, 3021, "L10")]
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), D_SAE), device=bank.device)
n_kinds = len(bank.kinds)

# Match the validation run's probe/eval geometry exactly.
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = True
disc.position_aware_select = "abs_pctl"
disc.position_aware_threshold = 90.0
disc.floor_source = "posctx"
disc.magnitude_prune = False
disc.recurrence_prune = False
disc.min_faithfulness = -100.0

meth = _build_mode_method("counterfactual_gradient", "ig_mean",
                          inference, bank, avg_acts, probe_builder)

fh = OUT.open("a")
rows = []
for sc_idx, sl, label in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    t0 = time.time()

    try:
        pd_ = meth.build_probe_dataset(sc_idx, sl)
        if pd_.pos_tokens.shape[0] == 0 or pd_.neg_tokens.shape[0] == 0:
            print("[%s] %s empty probe set -- skipped" % (seed_key, label), flush=True)
            continue
        pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
        nt = pd_.neg_tokens[:N_SEQ]

        a_pos = measure_seed_activation(inference, bank, pt, layer, kind, sl, pa,
                                        batch_size=EVAL_BS)

        floors = {
            "zero":   None,                                             # a_e0 path
            "posctx": collect_site_means(inference, bank, pt, up),      # today's default
            "negctx": collect_site_means(inference, bank, nt, up),      # PROPOSED
            "global": collect_global_site_floors(inference, bank, loader, up),
            "diverse": collect_diverse_site_floors(inference, bank, loader, up),
        }

        row = {"seed": seed_key, "label": label, "layer": layer, "kind": kind,
               "n_up_sites": len(up), "a_pos": round(float(a_pos), 4)}
        print("\n[%s] %s %s  %d upstream sites  a_pos=%.4f"
              % (seed_key, label, kind, len(up), a_pos), flush=True)
        print("   %-8s %10s %10s %8s" % ("floor", "a_empty", "den", "sign"), flush=True)

        for name, means in floors.items():
            a_e = circuit_only_activation(inference, bank, {}, up, pt, layer, kind,
                                          sl, pos_argmax=pa, site_means=means,
                                          batch_size=EVAL_BS)
            den = float(a_pos) - float(a_e)
            row["a_empty_%s" % name] = round(float(a_e), 4)
            row["den_%s" % name] = round(den, 4)
            flag = "NEG!" if den < 0 else ("tiny" if abs(den) < 0.15 else "ok")
            print("   %-8s %10.4f %10.4f %8s" % (name, a_e, den, flag), flush=True)

        row["secs"] = round(time.time() - t0, 1)
        rows.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
    except Exception as exc:            # one seed's OOM must not kill the sweep
        print("[%s] %s FAILED: %s: %s"
              % (seed_key, label, type(exc).__name__, exc), flush=True)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

fh.close()

print("\n" + "=" * 72)
print("SUMMARY -- denominator (a_pos - a_empty) by floor")
print("%-5s %9s %10s %10s %10s %10s %10s"
      % ("seed", "a_pos", "zero", "posctx", "negctx", "global", "diverse"))
for r in rows:
    print("%-5s %9.4f %10.4f %10.4f %10.4f %10.4f %10.4f"
          % (r["label"], r["a_pos"], r["den_zero"], r["den_posctx"],
             r["den_negctx"], r["den_global"], r["den_diverse"]))
print("\nwrote %s" % OUT)
