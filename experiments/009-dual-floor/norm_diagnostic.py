"""Is dual's L10 collapse caused by MY normaliser rather than the model?

At L10 dual's size barely moves with lambda (33x change -> 1.9x size) or with
gamma, capping at ~5,100 nodes against zero-floor's 108,068. The L1 term is
not binding, so something else decides membership.

Suspect: the per-term normalisation.

    L = L_zero/||closed_zero|| + gamma * L_negctx/||closed_negctx||

Each term is divided by its OWN fully-closed-mask loss. If the negctx floor
already sits close to the target, ||closed_negctx|| is SMALL, so the factor
gamma/||closed_negctx|| becomes LARGE and the negctx term dominates. Phase 1
measured L10's negatives as the most contaminated of any seed (median
pre-top-k drive 21-38% of the posctx level), and a contaminated floor is by
construction closer to the target. That would make dual behave like
negctx-only at L10 -- which independently collapses there (free0 exactly
0.0, negative freeM_topk).

It would also explain the lambda-insensitivity: if one normalised term is
orders of magnitude larger than the other, lambda is negligible beside it.

The normalisers are computed BEFORE the optimisation loop, so steps=1 is
enough to read them -- no need to pay for 400 steps per seed.

Prediction if the hypothesis holds: ratio norm_zero/norm_floor near 1 at the
shallow seeds and LARGE at L10.

  PYTHONPATH=src python experiments/009-dual-floor/norm_diagnostic.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.instrument.learned_mask import run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ = 64
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
disc.floor_negctx_mode = "random"
ncs = disc.neg_context_selection
ncs.preact_filter = True
ncs.preact_select = "cleanest"
ncs.preact_max_frac = 0.25


def _strat(cands, n, seed=42):
    by = defaultdict(list)
    for i, c in enumerate(cands):
        by[int(c["comp_idx"]) // n_kinds].append(i)
    rng = random.Random(seed)
    for k in by:
        rng.shuffle(by[k])
    out = []
    for rank in range(max(len(v) for v in by.values())):
        for k in sorted(by):
            if rank < len(by[k]):
                out.append(by[k][rank])
                if len(out) >= n:
                    return out
    return out


_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu", weights_only=False)
_idx = _strat(_cand, 16)
SEEDS = [(i, int(_cand[_idx[i]]["comp_idx"]), int(_cand[_idx[i]]["latent_idx"]))
         for i in (2, 5, 8, 10)]

rows = []
print("%-6s %-10s %7s %13s %14s %10s" % (
    "seed", "site", "sites", "norm_zero", "norm_floor", "ratio"), flush=True)
for seed_i, sc_idx, sl in SEEDS:
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))
    meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                              avg_acts, probe_builder)
    pd_ = meth.build_probe_dataset(sc_idx, sl)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    try:
        # steps=1: the normalisers are computed before the loop, so this is
        # the whole cost of the question.
        _, prov = run_learned_mask(
            inference, bank, objective="pos", sites=up,
            seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
            pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
            mask_floor_source="dual", dual_floor_weight=0.25,
            steps=1, lr=0.05, l1_lambda=3e-6,
            # ABOVE sigmoid(theta_init=4.0)=0.982, so the member-selection
            # loop selects NOTHING and costs nothing. With steps=1 the mask
            # has not trained, so at the normal 0.5 threshold EVERY latent
            # qualifies and selection becomes a 1.3M-iteration Python loop
            # per seed at L10 - the opposite of cheap. The normalisers are
            # computed before the loop, so this changes nothing we read.
            keep_threshold=0.99,
            batch_size=4, holdout_frac=0.25, log_every=0,
            deep_site_threshold=disc.learned_mask.deep_site_threshold,
            deep_batch_size=disc.learned_mask.deep_batch_size,
            optimizer="adamw", weight_decay=0.05,
            code_dtype=disc.learned_mask.code_dtype)
        nz, nf = prov["dual_norm_zero"], prov["dual_norm_floor"]
        rows.append({"seed_i": seed_i, "seed": "%d/%d" % (sc_idx, sl),
                     "layer": layer, "kind": kind, "n_sites": len(up),
                     "norm_zero": nz, "norm_floor": nf,
                     "ratio_zero_over_floor": nz / max(nf, 1e-12)})
        print("%-6d %-10s %7d %13.5f %14.5f %10.2f" % (
            seed_i, "L%d-%s" % (layer, kind), len(up), nz, nf,
            nz / max(nf, 1e-12)), flush=True)
    except Exception as exc:
        print("%-6d FAILED %s: %s" % (seed_i, type(exc).__name__, exc), flush=True)
    torch.cuda.empty_cache()

(HERE / "norm_diagnostic.json").write_text(json.dumps(rows, indent=2))
print("\nratio >> 1 means the ZERO term is scaled up (negctx floor already near")
print("target, i.e. contaminated negatives); ratio ~1 means the two floors are")
print("comparably hard and gamma means what it says.")
