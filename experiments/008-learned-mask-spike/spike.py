"""Phase-0 spike: abl-mask end-to-end on L2.

Kill criterion from the plan: if the mask's circuit does not beat the
same-size attribution baseline on free0, stop and rethink. Reference points
(L2, saved circuits): abl-ig_mean rec2mag = 0.965 @ 15,590 members; the
direct-drivers K-sweep gave attribution top-K free0 of 0.008 @ 256 / 0.47 @
4,096.

  PYTHONPATH=src python experiments/008-learned-mask-spike/spike.py
"""
import json
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
SC_IDX, LATENT = 8, 30122          # L2 resid
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
disc.position_aware = False            # mask mode raises on PA by design
disc.magnitude_prune = False
disc.recurrence_prune = False
disc.min_faithfulness = -100.0
disc.learned_mask.steps = 150
disc.learned_mask.lr = 0.1
disc.learned_mask.l1_lambda = 0.01

layer, ki = split_component_idx(SC_IDX, NK)
kind = bank.kinds[ki]
up = upstream_sites(bank, layer, kind)

t0 = time.time()
m = _build_mode_method("ablation_gradient", "mask", inference, bank,
                       avg_acts, probe_builder)
circ = m.discover(SC_IDX, LATENT)
secs = time.time() - t0
if circ is None:
    raise SystemExit("NO CIRCUIT")

pd_ = m.build_probe_dataset(SC_IDX, LATENT)
pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]

keep = {}
for node in circ.nodes.values():
    f = node.feature_id
    if node.metadata.get("role") == "seed" or f is None:
        continue
    if (f.layer, f.kind) in up:
        keep.setdefault((f.layer, f.kind), set()).add(int(f.index))
n = sum(len(v) for v in keep.values())

a_pos = measure_seed_activation(inference, bank, pt, layer, kind, LATENT, pa,
                                batch_size=EVAL_BS)
a_e0 = circuit_only_activation(inference, bank, {}, up, pt, layer, kind, LATENT,
                               pos_argmax=pa, batch_size=EVAL_BS)
a_c = circuit_only_activation(inference, bank, keep, up, pt, layer, kind, LATENT,
                              pos_argmax=pa, batch_size=EVAL_BS)
den = float(a_pos) - float(a_e0)
free0 = (float(a_c) - float(a_e0)) / den if abs(den) > 1e-9 else None
pre_c = circuit_only_activation(inference, bank, keep, up, pt, layer, kind,
                                LATENT, pos_argmax=pa, batch_size=EVAL_BS,
                                preact=True)

row = {"seed": "%d/%d" % (SC_IDX, LATENT), "arm": "abl-mask",
       "n_members": n, "free0": round(free0, 4), "a_c": round(float(a_c), 4),
       "p_c": round(float(pre_c), 4), "a_pos": round(float(a_pos), 4),
       "secs_discover": round(secs, 1),
       "steps": disc.learned_mask.steps, "lr": disc.learned_mask.lr,
       "l1": disc.learned_mask.l1_lambda}
with (HERE / "rows.jsonl").open("a") as fh:
    fh.write(json.dumps(row) + "\n")
print("\nabl-mask L2: n=%s free0=%.4f (a_c %.3f / a_pos %.3f, preact %.3f) in %.0fs"
      % (format(n, ","), free0, a_c, a_pos, pre_c, secs))
print("references: abl-ig_mean rec2mag 0.965 @ 15,590 | attr top-4096 0.47")
