"""HYPOTHESIS TEST: gradient-weighted per-site sparsity pricing.

Claim under test: flat per-latent lambda is REGRESSIVE. The C3 dose-response
showed that at depth, members with individually tiny scores collectively
carry the whole reconstruction, and the L10 per-site breakdown showed flat
pricing gutting exactly the middle-layer sites where signal is most diffuse
(kept fraction 0.13-0.18 at L1-L5 vs 0.72 next to the seed). If those diffuse
tails are load-bearing, charging them the same per latent as concentrated
near-seed drivers prunes the wrong population.

Weighting: lambda_s = lambda * w_s with w_s proportional to the site's
per-latent gradient scale (q99 of |dL/dtheta| at init), normalised to
geometric mean 1 so the global lambda calibration carries over. Concentrated
sites (large per-latent gradients) get pricier; diffuse sites cheaper.

Prediction: at MATCHED size, weighted beats flat on reconstruction, most at
depth. Flat wins = site allocation was already data-optimal and the weights
just distort it.

Per seed: grad pass (cheap) -> flat run at lambda=1e-5 -> weighted run at the
same lambda (doubles as the probe) -> weighted run size-matched to flat via
the one-probe power law (exponent 0.759, measured max error 3.6%).
Per-site member counts are recorded for the mechanism check: did the
weighting actually shift members toward diffuse sites?

  PYTHONPATH=src python .../site_weights_ab.py
"""
import json
import math
import time
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
N_SEQ, EVAL_BS, GAMMA, LAMBDA, EXP = 64, 16, 0.25, 1e-5, 0.759
TARGETS = [(8, 17043, "L2-resid"), (25, 4085, "L8-mlp"), (32, 36965, "L10-resid")]
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


def site_grad_scales(up, layer, kind, sl, pt, pa):
    """Per-site gradient scale: median of the NONZERO |dL/dtheta|.

    v1 used q99 over ALL 40,960 latents and produced weight range
    0.000..241.747 (2.5e8x): only probe-active latents receive gradient, so
    any site whose active set on the probe slice is under ~1% of the
    dictionary has q99 EXACTLY ZERO. At L2 that hit the two near-seed sites
    (L2-attn, L2-mlp) - clamped to 1e-12, their latents became FREE and
    flooded (35,620 + 23,553 members) while the geometric-mean normalisation
    dragged every other site's weight up to ~240x, emptying them
    (weighted@matched kept ONE member outside the free sites). Two fixes:
    the scale is now over NONZERO gradients only, and gradients accumulate
    over 16 probes (4 micro-batches) rather than 4 for a stabler estimate.
    """
    sae = bank.saes[kind][layer]
    w = sae.encoder.weight[sl].detach()
    b = sae._get_bias_eff()[sl].detach()
    thetas = {s: torch.full((bank.d_sae,), 4.0, device=bank.device,
                            requires_grad=True) for s in up}
    p = lm.LearnedMaskPatcher(bank, thetas, layer, kind, w, b,
                              code_dtype=disc.learned_mask.code_dtype)
    for s0 in range(0, 16, 4):
        pre = lm._forward_preact(inference, p, pt[s0:s0 + 4], grad=True)
        idx = torch.arange(pre.shape[0], device=pre.device)
        vals = pre[idx, pa[s0:s0 + 4].to(pre.device)]
        ((vals - vals.detach() * 0.5) ** 2).mean().backward()
    scales = {}
    for s, t in thetas.items():
        g = t.grad.abs()
        nz = g[g > 0]
        scales[s] = float(nz.median()) if nz.numel() else float("nan")
        t.grad = None
    return scales


def run(up, layer, kind, sl, pt, pa, nt, lam, weights):
    t0 = time.perf_counter()
    scores, prov = lm.run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
        mask_floor_source="dual", dual_floor_weight=GAMMA,
        steps=400, lr=0.05, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=disc.learned_mask.deep_site_threshold,
        deep_batch_size=disc.learned_mask.deep_batch_size,
        optimizer="adamw", weight_decay=0.05,
        code_dtype=disc.learned_mask.code_dtype,
        site_lambda_weights=weights)
    return scores, time.perf_counter() - t0


OUT = HERE / "site_weights_ab.jsonl"
fh = OUT.open("a")
for comp, latent, label in TARGETS:
    layer, ki = split_component_idx(comp, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))
    meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                              avg_acts, probe_builder)
    pd_ = meth.build_probe_dataset(comp, latent)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    a_pos = float(measure_seed_activation(inference, bank, pt, layer, kind,
                                          latent, pa, batch_size=EVAL_BS))
    means_up = collect_site_means(inference, bank, pt, set(up))
    means_neg = collect_site_means(inference, bank, nt, set(up))

    def empty(sm=None, tk=False):
        return float(circuit_only_activation(
            inference, bank, {}, up, pt, layer, kind, latent, pos_argmax=pa,
            site_means=sm, batch_size=EVAL_BS, respect_topk=tk))

    a_e0, a_eMT, a_eNT = empty(), empty(means_up, True), empty(means_neg, True)

    scales = site_grad_scales(up, layer, kind, latent, pt, pa)
    if any(math.isnan(v) for v in scales.values()):
        raise RuntimeError("a site produced NO nonzero gradients - refusing "
                           "to price it; investigate rather than default")
    logs = [math.log(v) for v in scales.values()]
    gm = math.exp(sum(logs) / len(logs))
    # CLAMPED to [1/4, 4]: no site is ever free (weight 0 = unbounded
    # membership, the v1 flood) and none absurdly overpriced. The hypothesis
    # is about MODEST redistribution; if it only works at extreme ratios it
    # is not the mechanism we proposed.
    weights = {s: min(max(v / gm, 0.25), 4.0) for s, v in scales.items()}
    wmin, wmax = min(weights.values()), max(weights.values())
    print("\n=== %s (comp %d latent %d, %d sites) | weight range %.3f..%.3f "
          "(%.0fx) ===" % (label, comp, latent, len(up), wmin, wmax,
                           wmax / max(wmin, 1e-12)), flush=True)

    def metrics(scores):
        keep = {}
        for f in scores:
            keep.setdefault((f.layer, f.kind), set()).add(int(f.index))
        n = sum(len(v) for v in keep.values())
        per_site = {("L%d-%s" % s): len(v) for s, v in keep.items()}

        def phi(a_e, sm=None, tk=False):
            a_c = float(circuit_only_activation(
                inference, bank, keep, up, pt, layer, kind, latent,
                pos_argmax=pa, site_means=sm, batch_size=EVAL_BS,
                respect_topk=tk)) if n else a_e
            d = a_pos - a_e
            return round((a_c - a_e) / d, 4) if abs(d) > 1e-9 else None
        return n, per_site, phi(a_e0), phi(a_eNT, means_neg, True), \
            phi(a_eMT, means_up, True)

    arms = []
    fs, secs = run(up, layer, kind, latent, pt, pa, nt, LAMBDA, None)
    n_flat, ps_flat, f0, fN, fM = metrics(fs); del fs
    torch.cuda.empty_cache()
    arms.append(("flat", LAMBDA, n_flat, ps_flat, f0, fN, fM, secs))

    ws, secs = run(up, layer, kind, latent, pt, pa, nt, LAMBDA, weights)
    n_w, ps_w, f0, fN, fM = metrics(ws); del ws
    torch.cuda.empty_cache()
    arms.append(("weighted", LAMBDA, n_w, ps_w, f0, fN, fM, secs))

    # size-match weighted to the flat arm via the one-probe power law
    lam2 = LAMBDA * (n_w / max(n_flat, 1)) ** (1.0 / EXP)
    ms, secs = run(up, layer, kind, latent, pt, pa, nt, lam2, weights)
    n_m, ps_m, f0, fN, fM = metrics(ms); del ms
    torch.cuda.empty_cache()
    arms.append(("weighted@matched", lam2, n_m, ps_m, f0, fN, fM, secs))

    for name, lam, n, ps, f0, fN, fM, secs in arms:
        row = {"comp_idx": comp, "latent": latent, "label": label,
               "arm": name, "lambda": lam, "n": n, "per_site": ps,
               "free0": f0, "freeN_topk": fN, "freeM_topk": fM,
               "weights_range": [wmin, wmax], "secs": round(secs, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-18s lam=%.3e n=%-9s free0=%-8s freeN_tk=%-8s freeM_tk=%-8s"
              % (name, lam, format(n, ","), f0, fN, fM), flush=True)
fh.close()
print("\nwrote site_weights_ab.jsonl")
print("Judge weighted@matched vs flat: same size, only the PRICING differs.")
