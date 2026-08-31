"""Does circuit SIZE just track how many contexts the mask sees? (test B)

The abl-mask is NPA and trained over probe_sequence_count sequences at
all positions, so its membership is a UNION over contexts. At any single
token position only 1,024 latents are active upstream (8 sites x top-k
128), yet circuits run ~2,000 nodes — about 31 latents per probe
context. If "circuit size" is really "per-context support x number of
contexts", n should scale with the probe count rather than converge.

Same seeds at probe_sequence_count 8 / 16 / 32 / 64. NOTE: this
deliberately varies a parameter pinned at 64 for production runs — it is
a diagnostic, not a recipe change.

Evaluation is held FIXED at a 64-sequence probe set for every arm, so
free0 is comparable across probe counts and only discovery input varies.
Also reports Jaccard against the same seed's 64-sequence circuit: if the
small-P circuits are subsets, the mask is accreting, not re-deciding.

  PYTHONPATH=src python experiments/024-l2-crossover/probe_scaling.py
"""
import gzip
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
    circuit_only_activation, measure_seed_activation, upstream_sites)
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP_IDX, EVAL_SEQ, EVAL_BS, D_SAE = 8, 64, 16, 40960
PROBE_COUNTS = [8, 16, 32, 64]
N_SEEDS = 6
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
cf_cfg, ab_cfg = disc.counterfactual_gradient, disc.ablation_gradient

LAYER, KI = split_component_idx(COMP_IDX, n_kinds)
KIND = bank.kinds[KI]
UP = upstream_sites(bank, LAYER, KIND)

ref = {}
with gzip.open(HERE / "members.jsonl.gz", "rt") as fh:
    for line in fh:
        if line.strip():
            r = json.loads(line)
            ref[r["latent"]] = {(a, b, c) for a, b, c in r["members"]}
SEEDS = sorted(ref)[:N_SEEDS]
print("scaling test on %d seeds x %s probe sequences"
      % (len(SEEDS), PROBE_COUNTS), flush=True)


def base_state(probe_n):
    disc.probe_sequence_count = probe_n
    disc.eval_sequence_count = probe_n
    disc.eval_batch_size = EVAL_BS
    disc.probe_batch_size = 4
    disc.position_aware = False
    disc.floor_source = "posctx"
    disc.magnitude_prune = False
    disc.recurrence_prune = False
    disc.min_faithfulness = -100.0
    cf_cfg.max_neg_sequences = probe_n
    cf_cfg.neg_batch_size = 8
    cf_cfg.negative_roles = "include"
    ab_cfg.negative_roles = "include"
    cf_cfg.pruning_threshold = 0.0
    ab_cfg.pruning_threshold = 0.0


fh_out = (HERE / "probe_scaling.jsonl").open("a")
for sl in SEEDS:
    # fixed 64-sequence evaluation set, built once per seed
    base_state(EVAL_SEQ)
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(COMP_IDX, sl)
    del m0
    pt, pa = pd_.pos_tokens[:EVAL_SEQ], pd_.pos_argmax[:EVAL_SEQ]
    a_pos = float(measure_seed_activation(inference, bank, pt, LAYER, KIND, sl,
                                          pa, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, UP, pt, LAYER,
                                         KIND, sl, pos_argmax=pa,
                                         batch_size=EVAL_BS))
    den = a_pos - a_e0
    print("\n[%d] a_pos %.3f" % (sl, a_pos), flush=True)

    for pn in PROBE_COUNTS:
        t0 = time.time()
        base_state(pn)
        meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                                  avg_acts, probe_builder)
        circ = meth.discover(COMP_IDX, sl)
        del meth
        torch.cuda.empty_cache()
        if circ is None:
            print("   P=%-3d no circuit" % pn, flush=True)
            continue
        M = set()
        for node in circ.nodes.values():
            if node.metadata.get("role") == "seed":
                continue
            f = node.feature_id
            if f is not None and (f.layer, f.kind) in UP:
                M.add((f.layer, f.kind, int(f.index)))
        keep = {}
        for l, k, i in M:
            keep.setdefault((l, k), set()).add(i)
        a = float(circuit_only_activation(inference, bank, keep, UP, pt, LAYER,
                                          KIND, sl, pos_argmax=pa,
                                          batch_size=EVAL_BS))
        R = ref.get(sl, set())
        row = {"latent": sl, "probe_n": pn, "n": len(M),
               "per_context": round(len(M) / pn, 1),
               "free0_on64": round((a - a_e0) / den, 4) if abs(den) > 1e-9 else None,
               "jaccard_vs_64ref": round(len(M & R) / max(len(M | R), 1), 4),
               "frac_of_ref": round(len(M & R) / max(len(R), 1), 4),
               "secs": round(time.time() - t0, 1)}
        fh_out.write(json.dumps(row) + "\n"); fh_out.flush()
        print("   P=%-3d n=%-6d n/P=%-7.1f free0(64)=%-7s  J vs ref %-6s  "
              "covers %.0f%% of ref  %.0fs"
              % (pn, len(M), row["per_context"], row["free0_on64"],
                 row["jaccard_vs_64ref"], 100 * row["frac_of_ref"], row["secs"]),
              flush=True)
        del circ
        torch.cuda.empty_cache()

fh_out.close()
print("ALL DONE", flush=True)
