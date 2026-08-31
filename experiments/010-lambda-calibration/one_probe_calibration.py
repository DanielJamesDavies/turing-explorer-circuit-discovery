"""One-probe power-law lambda calibration, tested END TO END.

Measured today across 4 seeds (gamma=0.25, post normaliser-fix), the node
count follows a power law in lambda:

    n ~ lambda^s      s = -0.759   (per-seed -0.656 .. -0.892)

so ONE probe run fixes the curve and the lambda hitting any target size is

    lambda_target = lambda_probe * (n_probe / n_target)^(1/0.759)

Cost: 1 run per seed, against 3-5 for bisect-to-faithfulness - and bisect
missed its own target in BOTH directions (0.9008 vs a 0.95 target at L2;
overshot 35% at L10).

WHY THIS IS THE HONEST TEST. The earlier 3.9-6.8% agreement was
lambda-vs-lambda, comparing a shared exponent against a per-seed exponent
fitted to the SAME probe curve - close to circular. Here the prediction is
run and the ACHIEVED node count is compared against the target. That is
out-of-sample.

Rejected on measurement, for the record:
  * lambda = c * q99(|dL/dtheta|)  - ANTI-correlated. L2 has 20x LESS
    gradient than L10 yet needs ~4x MORE lambda; transferring c from L10 gave
    L2 a 23,366-node circuit against a 1,673-node target (14x oversized).
  * lambda ~ 1/n_sites - the product spans 3.86x and is NON-MONOTONE, peaking
    at L5 (1.96e-4) against L2 5.65e-5 and L10 5.07e-5. It halves the
    variance of raw lambda (7.72x) but is not a law.

  PYTHONPATH=src python .../one_probe_calibration.py
"""
import json
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
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.floors import collect_site_means
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
PROBE_LAMBDA = 1e-5
EXPONENT = 0.759
# Extrapolation guard: beyond this the per-seed exponent spread (+-16%) starts
# to matter. All four calibration cases moved n by under 4x from the probe.
SAFE_RATIO = 5.0
WANT = [int(x) for x in os.environ.get("SEED_IDS", "2,5,8,10").split(",")]
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
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.magnitude_prune = False
disc.recurrence_prune = False


def calibrate_lambda(n_probe, n_target, lam_probe=PROBE_LAMBDA,
                     exponent=EXPONENT):
    """lambda hitting n_target, from a single (lam_probe -> n_probe) point."""
    ratio = n_probe / max(n_target, 1)
    lam = lam_probe * ratio ** (1.0 / exponent)
    far = not (1.0 / SAFE_RATIO <= ratio <= SAFE_RATIO)
    return lam, ratio, far


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


def make_metrics(up, layer, kind, sl, pt, pa, a_pos, a_e0, a_eNT, a_eMT,
                 means_up, means_neg):
    """Factory, not an inline closure: an inline def would capture the loop
    variables by reference and every stored copy would read the LAST seed's
    values (that bug silently scored seed 2 against seed 10's anchors)."""
    def metrics(scores):
        keep = {}
        for f in scores:
            keep.setdefault((f.layer, f.kind), set()).add(int(f.index))
        n = len(scores)

        def phi(a_e, sm=None, tk=False):
            a_c = float(circuit_only_activation(
                inference, bank, keep, up, pt, layer, kind, sl, pos_argmax=pa,
                site_means=sm, batch_size=EVAL_BS, respect_topk=tk)) if n else a_e
            d = a_pos - a_e
            return round((a_c - a_e) / d, 4) if abs(d) > 1e-9 else None
        return n, phi(a_e0), phi(a_eNT, means_neg, True), phi(a_eMT, means_up, True)
    return metrics


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
    return scores, time.perf_counter() - t0


# Append-and-flush PER SEED. Writing the whole file after the loop loses
# everything to an interruption - a session teardown killed this run after
# seed 2 and took a completed, correct result with it. Also lets a rerun
# skip seeds already done.
OUT = HERE / "one_probe_calibration.jsonl"
done = set()
if OUT.exists():
    for line in OUT.open():
        if line.strip():
            done.add(json.loads(line)["seed_i"])
fh = OUT.open("a")
rows = []
for seed_i, sc_idx, sl in SEEDS:
    if seed_i in done:
        print("skip (done) seed %d" % seed_i, flush=True)
        continue
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))
    meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                              avg_acts, probe_builder)
    pd_ = meth.build_probe_dataset(sc_idx, sl)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]

    a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind, sl,
                                          pa, batch_size=EVAL_BS))
    means_up = collect_site_means(inference, bank, pt, set(up))
    means_neg = collect_site_means(inference, bank, nt, set(up))

    def empty(sm=None, tk=False):
        return float(circuit_only_activation(
            inference, bank, {}, up, pt, layer, kind, sl, pos_argmax=pa,
            site_means=sm, batch_size=EVAL_BS, respect_topk=tk))

    a_e0, a_eMT, a_eNT = empty(), empty(means_up, True), empty(means_neg, True)
    metrics = make_metrics(up, layer, kind, sl, pt, pa, a_pos, a_e0, a_eNT,
                           a_eMT, means_up, means_neg)

    print("\n=== seed %d  L%d %s  | %d sites ===" % (seed_i, layer, kind, len(up)),
          flush=True)
    zs, secs = run(up, layer, kind, sl, pt, pa, nt, 1e-4, dual=False)
    zn, zf0, zfN, zfM = metrics(zs)
    del zs
    print("  zero-floor TARGET   n=%-9s free0=%-8s freeN_tk=%-8s freeM_tk=%-8s (%.0fs)"
          % (format(zn, ","), zf0, zfN, zfM, secs), flush=True)

    ps, secs = run(up, layer, kind, sl, pt, pa, nt, PROBE_LAMBDA, dual=True)
    pn = len(ps)
    del ps
    torch.cuda.empty_cache()
    lam, ratio, far = calibrate_lambda(pn, zn)
    print("  probe lam=%.0e       n=%-9s -> ratio %.2f -> lambda %.3e%s (%.0fs)"
          % (PROBE_LAMBDA, format(pn, ","), ratio, lam,
             "  [WIDE EXTRAPOLATION]" if far else "", secs), flush=True)

    fs, secs = run(up, layer, kind, sl, pt, pa, nt, lam, dual=True)
    fn, f0, fN, fM = metrics(fs)
    del fs
    torch.cuda.empty_cache()
    err = 100.0 * (fn - zn) / max(zn, 1)
    print("  CALIBRATED dual     n=%-9s free0=%-8s freeN_tk=%-8s freeM_tk=%-8s (%.0fs)"
          % (format(fn, ","), f0, fN, fM, secs), flush=True)
    print("  -> size error vs target: %+.1f%%   |  neutral metrics vs zero-floor: "
          "freeN_tk %+.4f  freeM_tk %+.4f"
          % (err, (fN or 0) - (zfN or 0), (fM or 0) - (zfM or 0)), flush=True)
    row = {"seed_i": seed_i, "layer": layer, "kind": kind,
           "sites": len(up), "n_target": zn, "n_probe": pn,
           "lambda": lam, "ratio": ratio, "wide_extrapolation": far,
           "n_achieved": fn, "size_err_pct": round(err, 2),
           "zero": {"free0": zf0, "freeN_topk": zfN, "freeM_topk": zfM},
           "dual": {"free0": f0, "freeN_topk": fN, "freeM_topk": fM}}
    rows.append(row)
    fh.write(json.dumps(row) + "\n")
    fh.flush()

fh.close()
print("\nwrote %s" % OUT)
print("Size error is the OUT-OF-SAMPLE test of the rule. The metric deltas")
print("are then a size-matched dual-vs-zero-floor comparison at every seed.")
