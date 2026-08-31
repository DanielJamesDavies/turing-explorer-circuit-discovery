"""Centrepiece REFRESH collector (v2): 3 seeds per depth band, chosen
with the PANEL's seed selection (sorted pool head) so the weighted-
circuit markers from 029-panel land on the same seeds; tail
ladder points (16384/65536) added to smooth the free curve's final
rise. Rows -> curves2.jsonl; v1 single-seed data stays in curves2.jsonl.

  PYTHONPATH=src python experiments/028-control-vs-faithfulness-figure/collector2.py
"""
import json
import random
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method, _site_role_groups,
    _truncated_circuit_per_site)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, evaluate_ablation_faithfulness,
    measure_seed_activation, upstream_sites)
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
COMPS = [(10, "shallow"), (23, "mid"), (34, "deep")]   # L3 mlp, L7 resid, L11 mlp
M_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384,
            65536, 10**9]
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, D_SAE), device=bank.device)
# No discovery-side truncation: the curve cuts the union at EVAL time only.
_apply_sweep_config(max_per_site=200000)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.floor_source = "posctx"
# The July recipe: position-aware discovery, abs-p50 percentile selection.
disc.position_aware = True
disc.position_aware_select = "abs_pctl"
disc.position_aware_threshold = 50.0

_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu",
                   weights_only=False)


def seed_pool(comp_idx):
    pool = [int(c["latent_idx"]) for c in _cand if int(c["comp_idx"]) == comp_idx]
    random.Random(42).shuffle(pool)
    return pool


done = set()
if (HERE / "curves2.jsonl").exists():
    for line in (HERE / "curves2.jsonl").open():
        try:
            r = json.loads(line)
            if r.get("m") == "full":
                done.add((r["comp_idx"], r["latent"]))
        except Exception:
            pass
fh = (HERE / "curves2.jsonl").open("a")
for comp_idx, band in COMPS:
    layer, ki = split_component_idx(comp_idx, n_kinds)
    kind = bank.kinds[ki]
    # PANEL seed selection: sorted head of the shuffled pool
    for sl in sorted(seed_pool(comp_idx)[:32])[:3]:
      if (comp_idx, sl) in done:
        print("[%s] seed %d already collected" % (band, sl), flush=True)
        continue
      method = _build_mode_method("ablation_gradient", "ig_mean", inference,
                                  bank, avg_acts, pb)
      circuit = None
      if True:
        t0 = time.time()
        try:
            pd_ = method.build_probe_dataset(comp_idx, sl)
            if pd_ is None or int(pd_.pos_tokens.shape[0]) < 8:
                print("[%s] seed %d: thin probes, skipping" % (band, sl), flush=True)
                continue
            circuit = method.discover(comp_idx, sl)
        except Exception as e:
            print("[%s] seed %d: %s: %s" % (band, sl, type(e).__name__, e), flush=True)
            circuit = None
      if circuit is None or len(circuit.nodes) == 0:
        print("[%s] seed %d NO CIRCUIT — skipped" % (band, sl), flush=True)
        continue
      disc_secs = time.time() - t0
      pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
      print("\n[%s] L%d %s latent %d | %d nodes | discovery %.0fs"
            % (band, layer, kind, sl, len(circuit.nodes), disc_secs), flush=True)

      a_posctx = float(measure_seed_activation(inference, bank, pt, layer, kind,
                                               sl, pa, batch_size=EVAL_BS))
      in_scope = upstream_sites(bank, layer, kind)
      site_means, pin_values = collect_site_anchors(
          inference, bank, pt, in_scope, pa, pin_position_specific=True)
      a_empty_mean = float(circuit_only_activation(
          inference, bank, {}, in_scope, pt, layer, kind, sl, pos_argmax=pa,
          site_means=site_means, batch_size=EVAL_BS))
      a_empty_zero = float(circuit_only_activation(
          inference, bank, {}, in_scope, pt, layer, kind, sl, pos_argmax=pa,
          batch_size=EVAL_BS))
      print("  a_posctx %.3f | a_empty mean %.3f zero %.3f"
            % (a_posctx, a_empty_mean, a_empty_zero), flush=True)

      site_groups = _site_role_groups(circuit)
      prev_n = None
      for m in M_LADDER:
          sub, n_use = _truncated_circuit_per_site(circuit, site_groups, int(m))
          if n_use == prev_n:
              continue
          prev_n = n_use
          t0 = time.time()
          free_mean, _ = evaluate_ablation_faithfulness(
              inference, bank, avg_acts, sub, pos_tokens=pt, seed_layer=layer,
              seed_kind=kind, seed_latent_idx=sl, pos_argmax=pa,
              ablation="mean", site_means=site_means, a_posctx=a_posctx,
              a_empty=a_empty_mean)
          free_zero, _ = evaluate_ablation_faithfulness(
              inference, bank, avg_acts, sub, pos_tokens=pt, seed_layer=layer,
              seed_kind=kind, seed_latent_idx=sl, pos_argmax=pa,
              ablation="zero", a_posctx=a_posctx, a_empty=a_empty_zero)
          pinned_mean, _ = evaluate_ablation_faithfulness(
              inference, bank, avg_acts, sub, pos_tokens=pt, seed_layer=layer,
              seed_kind=kind, seed_latent_idx=sl, pos_argmax=pa,
              ablation="mean", site_means=site_means, pin_values=pin_values,
              a_posctx=a_posctx, a_empty=a_empty_mean)
          row = {"band": band, "comp_idx": comp_idx, "layer": layer,
                 "kind": kind, "latent": sl, "m": (m if m < 10**9 else "full"),
                 "n": n_use, "free_mean": round(float(free_mean), 4),
                 "free_zero": round(float(free_zero), 4),
                 "pinned_mean": round(float(pinned_mean), 4),
                 "secs": round(time.time() - t0, 1)}
          fh.write(json.dumps(row) + "\n"); fh.flush()
          print("  m=%-8s n=%-7d pinned=%-8s free_mean=%-8s free_zero=%-8s %.0fs"
                % (row["m"], n_use, row["pinned_mean"], row["free_mean"],
                   row["free_zero"], row["secs"]), flush=True)
      del circuit, site_groups, pin_values, site_means, method
      torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
