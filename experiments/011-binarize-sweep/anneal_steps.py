"""Can a 200-step anneal match the 400-step anneal?

Anneal is the first mode with a natural endpoint - the gate freezes as T->0 -
so the question is whether compressing the whole schedule into half the steps
reaches the same place. Two knobs must move with the step count:

  lambda: sparsity pressure is steps*lr*lambda, so 200 steps at the same
     lambda is HALF the pressure. Arm A doubles lambda (2e-5) to hold the
     product; if its size misses the 400-step reference by >5%, arm B re-runs
     at a probe-corrected lambda (soft-gate exponent 0.759, checked not
     trusted).
  wd: the house rule is steps*lr*wd ~ 1.0 (m_kept calibration breaks
     silently otherwise), so the 200-step arms use wd=0.1.

The T schedule itself needs no adjustment - it is defined over `steps`, so a
200-step run anneals 1.0 -> 0.05 twice as fast by construction.

FULL flip trajectories are saved this time (Sweep 0 kept only summaries), so
the freeze curves can be compared directly: if the 400-step run's flips hit
~zero well before the end, the tail steps were provably idle.

  PYTHONPATH=src python experiments/011-binarize-sweep/anneal_steps.py
"""
import json
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
N_SEQ, EVAL_BS, EXP = 64, 16, 0.759
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
lmcfg = disc.learned_mask

_orig_step = torch.optim.AdamW.step
cap = {"i": 0, "total": 0, "sites": None, "m": None, "prev": None, "flips": []}


def _step(self, *a, **kw):
    out = _orig_step(self, *a, **kw)
    cap["i"] += 1
    params = self.param_groups[0]["params"]
    hard = torch.cat([(torch.sigmoid(p.detach()) > 0.5).flatten()
                      for p in params])
    if cap["prev"] is not None:
        cap["flips"].append(int((hard != cap["prev"]).sum()))
    cap["prev"] = hard
    if cap["i"] == cap["total"]:
        cap["m"] = {site: torch.sigmoid(p.detach()).float().cpu()
                    for site, p in zip(cap["sites"], params)}
    return out


torch.optim.AdamW.step = _step


def train(up, layer, kind, sl, pt, pa, nt, steps, lam, wd):
    cap["i"] = 0; cap["total"] = steps; cap["sites"] = up
    cap["m"] = None; cap["prev"] = None; cap["flips"] = []
    t0 = time.perf_counter()
    scores, _ = lm.run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
        mask_floor_source=lmcfg.mask_floor_source,
        dual_floor_weight=lmcfg.dual_floor_weight,
        steps=steps, lr=lmcfg.lr, l1_lambda=lam, keep_threshold=0.5,
        batch_size=4, holdout_frac=lmcfg.holdout_frac, log_every=0,
        deep_site_threshold=lmcfg.deep_site_threshold,
        deep_batch_size=lmcfg.deep_batch_size,
        optimizer=lmcfg.optimizer, weight_decay=wd,
        code_dtype=lmcfg.code_dtype, binarize="anneal")
    secs = time.perf_counter() - t0
    all_m = torch.cat([v for v in cap["m"].values()])
    near = int(((all_m >= 0.45) & (all_m < 0.55)).sum())
    return scores, secs, list(cap["flips"]), near


OUT = HERE / "anneal_steps.jsonl"
fh = OUT.open("a")
rows = []
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

    def metrics(scores):
        keep = {}
        for f in scores:
            keep.setdefault((f.layer, f.kind), set()).add(int(f.index))
        n = sum(len(v) for v in keep.values())

        def phi(a_e, sm=None, tk=False):
            a_c = float(circuit_only_activation(
                inference, bank, keep, up, pt, layer, kind, latent,
                pos_argmax=pa, site_means=sm, batch_size=EVAL_BS,
                respect_topk=tk)) if n else a_e
            d = a_pos - a_e
            return round((a_c - a_e) / d, 4) if abs(d) > 1e-9 else None
        return n, phi(a_e0), phi(a_eNT, means_neg, True), phi(a_eMT, means_up, True)

    print("\n=== %s (comp %d latent %d, %d sites) ===" % (label, comp, latent,
                                                          len(up)), flush=True)
    arms = [("anneal-400", 400, 1e-5, 0.05)]
    sc, secs, flips, near = train(up, layer, kind, latent, pt, pa, nt,
                                  400, 1e-5, 0.05)
    n_ref, f0, fN, fM = metrics(sc); del sc
    torch.cuda.empty_cache()
    results = [("anneal-400", 400, 1e-5, 0.05, n_ref, f0, fN, fM, flips, near, secs)]

    lam_a = 2e-5
    sc, secs, flips, near = train(up, layer, kind, latent, pt, pa, nt,
                                  200, lam_a, 0.10)
    n_a, f0, fN, fM = metrics(sc); del sc
    torch.cuda.empty_cache()
    results.append(("anneal-200-p", 200, lam_a, 0.10, n_a, f0, fN, fM, flips, near, secs))

    if not (0.95 <= n_a / max(n_ref, 1) <= 1.05):
        lam_b = lam_a * (n_a / max(n_ref, 1)) ** (1.0 / EXP)
        sc, secs, flips, near = train(up, layer, kind, latent, pt, pa, nt,
                                      200, lam_b, 0.10)
        n_b, f0, fN, fM = metrics(sc); del sc
        torch.cuda.empty_cache()
        results.append(("anneal-200-cal", 200, lam_b, 0.10, n_b, f0, fN, fM,
                        flips, near, secs))

    for name, steps, lam, wd, n, f0, fN, fM, flips, near, secs in results:
        row = {"comp_idx": comp, "latent": latent, "label": label, "arm": name,
               "steps": steps, "lambda": lam, "wd": wd, "n": n,
               "n_ref400": n_ref, "free0": f0, "freeN_topk": fN,
               "freeM_topk": fM, "near_cut": near, "flips": flips,
               "secs": round(secs, 1)}
        rows.append(row)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        tail = flips[-25:] if flips else [0]
        print("  %-15s lam=%.2e wd=%.2f n=%-9s free0=%-8s freeN_tk=%-8s "
              "freeM_tk=%-8s | last25 flips %.0f/step | %.0fs"
              % (name, lam, wd, format(n, ","), f0, fN, fM,
                 sum(tail) / len(tail), secs), flush=True)
fh.close()
torch.optim.AdamW.step = _orig_step

# ---- freeze curves ---------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"anneal-400": "#4477aa", "anneal-200-p": "#ee6677",
          "anneal-200-cal": "#228833"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
for (comp, latent, label), ax in zip(TARGETS, axes):
    for r in rows:
        if r["label"] != label:
            continue
        fl = r["flips"]
        xs = [i / max(len(fl) - 1, 1) for i in range(len(fl))]
        ax.plot(xs, fl, lw=1.0, color=COLORS.get(r["arm"], "grey"),
                label="%s (n=%s)" % (r["arm"], format(r["n"], ",")))
    ax.set_title(label)
    ax.set_xlabel("training progress (step / steps)")
    ax.set_yscale("symlog")
    ax.legend(fontsize=7)
axes[0].set_ylabel("membership flips per step")
fig.suptitle("Anneal freeze curves: 400 vs 200 steps (schedule compressed)")
fig.tight_layout()
fig.savefig(HERE / "anneal_steps_freeze.png", dpi=140)
plt.close(fig)
print("wrote anneal_steps_freeze.png", flush=True)
