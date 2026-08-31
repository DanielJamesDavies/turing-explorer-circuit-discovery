"""Does lambda * n_sites hold constant across depth?

The L1 penalty is a SUM over sites x latents, so total sparsity pressure grows
linearly with the number of upstream sites while the data loss (one scalar at
one anchor position) does not. That predicts lambda_good ~ 1/n_sites.

Two clean points measured 2026-07-30 (gamma=0.25, post normaliser-fix):

    seed  sites  lambda_good   lambda*n_sites
    L2       8      ~8e-6          6.4e-5
    L10     32      ~3e-6          9.6e-5

Within 1.5x, against a raw lambda that moves 2.7x. But two points is a line,
not a law. This adds L5 (16 sites) and L8 (25 sites) - the intermediate
depths - measured with the CURRENT code. Earlier L5/L8 numbers came from the
pre-normaliser-fix sweep and are not trustworthy.

lambda_good is defined here as the lambda at which dual's node count matches
the zero-floor arm's, so the four points are comparable on one definition.
That is a SIZE-matching definition, chosen because it is cheap; it is not a
claim that zero-floor's size is the correct size.

Rejected alternatives, for the record:
  * grad-scaled lambda = c * q99(|dL/dtheta|) is ANTI-correlated: L2 has 20x
    LESS gradient than L10 but needs 2.7x MORE lambda (lambda ~ grad^-0.4).
  * bisect-to-faithfulness works but costs 3-5 full runs per seed to find
    lambda before measuring anything - hours on a 16-seed matrix.

CAUTION from our own history: we previously mean-normalised this penalty over
ALL latents (/327,680), which drove the per-latent gradient to ~5e-10, under
Adam's eps, and NOTHING pruned. Dividing by n_sites is 8-32x, far from that
floor - but it is the same knob, so the lambda scale shifts by ~n_sites and
must be re-anchored.

  PYTHONPATH=src python .../site_scaling.py
"""
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
import circuit.instrument.learned_mask as lm

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ, GAMMA = 64, 0.25
PROBES = (1e-5, 3e-6, 1e-6)
WANT = [int(x) for x in os.environ.get("SEED_IDS", "5,8").split(",")]
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
         for i in WANT]


def run(up, layer, kind, sl, pt, pa, nt, lam, dual):
    kw = dict(inference=inference, bank=bank, objective="pos", sites=up,
              seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
              pos_tokens=pt, pos_argmax=pa, steps=400, lr=0.05,
              l1_lambda=lam, keep_threshold=0.5, batch_size=4,
              holdout_frac=0.25, log_every=0,
              deep_site_threshold=disc.learned_mask.deep_site_threshold,
              deep_batch_size=disc.learned_mask.deep_batch_size,
              optimizer="adamw", weight_decay=0.05,
              code_dtype=disc.learned_mask.code_dtype)
    if dual:
        kw.update(neg_tokens=nt, mask_floor_source="dual",
                  dual_floor_weight=GAMMA)
    t0 = time.perf_counter()
    scores, _ = lm.run_learned_mask(**kw)
    return len(scores), time.perf_counter() - t0


rows = []
print("%-5s %-10s %6s %10s %12s %14s"
      % ("seed", "site", "sites", "n_zero", "lambda_good", "lam*n_sites"), flush=True)
for seed_i, sc_idx, sl in SEEDS:
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))
    meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                              avg_acts, probe_builder)
    pd_ = meth.build_probe_dataset(sc_idx, sl)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]

    n_zero, secs = run(up, layer, kind, sl, pt, pa, nt, 1e-4, dual=False)
    print("  seed %d L%d-%s: zero-floor n=%s (%.0fs)"
          % (seed_i, layer, kind, format(n_zero, ","), secs), flush=True)

    pts = []
    for lam in PROBES:
        n, secs = run(up, layer, kind, sl, pt, pa, nt, lam, dual=True)
        pts.append((lam, n))
        print("    dual lam=%.0e -> n=%-9s (%.0fs)"
              % (lam, format(n, ","), secs), flush=True)
        torch.cuda.empty_cache()

    # log-lambda interpolation to the zero-floor node count
    lam_good = None
    ordered = sorted(pts)
    for (l1, n1), (l2, n2) in zip(ordered, ordered[1:]):
        if (n1 - n_zero) * (n2 - n_zero) <= 0 and n1 != n2:
            f = (n_zero - n1) / (n2 - n1)
            lam_good = math.exp(math.log(l1) + f * (math.log(l2) - math.log(l1)))
            break
    prod = lam_good * len(up) if lam_good else None
    print("%-5d %-10s %6d %10s %12s %14s"
          % (seed_i, "L%d-%s" % (layer, kind), len(up), format(n_zero, ","),
             ("%.3e" % lam_good) if lam_good else "not bracketed",
             ("%.3e" % prod) if prod else "-"), flush=True)
    rows.append({"seed_i": seed_i, "layer": layer, "kind": kind,
                 "sites": len(up), "n_zero": n_zero,
                 "probes": [{"lam": l, "n": n} for l, n in pts],
                 "lambda_good": lam_good, "lambda_x_sites": prod})

(HERE / "site_scaling.jsonl").write_text(
    "\n".join(json.dumps(r) for r in rows) + "\n")
print("\nKnown points for comparison (same definition, gamma=0.25):")
print("  L2   8 sites  lambda_good ~8e-6   lam*n_sites 6.4e-5")
print("  L10 32 sites  lambda_good ~3e-6   lam*n_sites 9.6e-5")
print("If L5 (16) and L8 (25) land in the same 6e-5..1.3e-4 band, lambda is")
print("depth-invariant once the penalty is divided by n_sites.")
