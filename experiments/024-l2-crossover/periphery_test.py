"""Is the private periphery load-bearing? (test A)

38% of a typical L2 circuit is nodes no other seed uses, and they are
the rarest latents in the panel (median corpus density 0.0017; 10% never
fire in a 64-sequence corpus sample at all). Two readings:

  passenger  — mask slack / free riders. Drop them, free0 holds.
  real       — genuine context-specific support. Drop them, free0 dies.

Scores every circuit five ways on the SAME probe set:
  full            all members
  minus_private   members appearing in >=2 of the 32 circuits
  private_only    members appearing in exactly 1
  core_only       members appearing in >=24
  rand_matched    random live latents, size-matched to minus_private
                  (so "minus_private holds up" cannot just mean "big set")

  PYTHONPATH=src python experiments/024-l2-crossover/periphery_test.py
"""
import collections
import gzip
import json
import random
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
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
COMP_IDX, N_SEQ, EVAL_BS, D_SAE = 8, 64, 16, 40960
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, D_SAE), device=bank.device)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.position_aware = False
disc.floor_source = "posctx"

LAYER, KI = split_component_idx(COMP_IDX, n_kinds)
KIND = bank.kinds[KI]
UP = upstream_sites(bank, LAYER, KIND)

mem = {}
with gzip.open(HERE / "members.jsonl.gz", "rt") as fh:
    for line in fh:
        if line.strip():
            r = json.loads(line)
            mem[r["latent"]] = {(a, b, c) for a, b, c in r["members"]}
freq = collections.Counter()
for m in mem.values():
    for x in m:
        freq[x] += 1
S = len(mem)
print("%d circuits loaded" % S, flush=True)

d = torch.load(HERE / "corpus_density.pt", weights_only=False)
pool = []
for s in sorted(d["counts"]):
    v = d["counts"][s]
    for i in (v > 0).nonzero(as_tuple=True)[0].tolist():
        pool.append((s[0], s[1], i))
rng = random.Random(5)
print("live pool %d" % len(pool), flush=True)

fh_out = (HERE / "periphery.jsonl").open("a")
for sl in sorted(mem):
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(COMP_IDX, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    a_pos = float(measure_seed_activation(inference, bank, pt, LAYER, KIND, sl,
                                          pa, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, UP, pt, LAYER,
                                         KIND, sl, pos_argmax=pa,
                                         batch_size=EVAL_BS))
    den = a_pos - a_e0

    def phi(members):
        if not members:
            return None
        keep = {}
        for l, k, i in members:
            keep.setdefault((l, k), set()).add(i)
        a = float(circuit_only_activation(inference, bank, keep, UP, pt, LAYER,
                                          KIND, sl, pos_argmax=pa,
                                          batch_size=EVAL_BS))
        return round((a - a_e0) / den, 4) if abs(den) > 1e-9 else None

    M = mem[sl]
    minus = {x for x in M if freq[x] >= 2}
    priv = {x for x in M if freq[x] == 1}
    core = {x for x in M if freq[x] >= 24}
    rnd = set(rng.sample(pool, min(len(minus), len(pool))))
    row = {"latent": sl, "n_full": len(M), "n_minus_private": len(minus),
           "n_private": len(priv), "n_core": len(core),
           "pct_private": round(100.0 * len(priv) / len(M), 1),
           "full": phi(M), "minus_private": phi(minus),
           "private_only": phi(priv), "core_only": phi(core),
           "rand_matched": phi(rnd)}
    fh_out.write(json.dumps(row) + "\n"); fh_out.flush()
    print("  %-6d n=%-5d priv=%-4.0f%% | full %-7s minus_priv %-7s "
          "priv_only %-7s core %-7s rand %-7s"
          % (sl, len(M), row["pct_private"], row["full"], row["minus_private"],
             row["private_only"], row["core_only"], row["rand_matched"]),
          flush=True)
    torch.cuda.empty_cache()

fh_out.close()
print("ALL DONE", flush=True)
