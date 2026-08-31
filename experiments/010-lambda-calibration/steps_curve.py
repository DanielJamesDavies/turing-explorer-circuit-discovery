"""How do circuit size AND quality evolve with training length?

Motivation: the trajectory measurement showed n still falling 11-58% between
steps 300 and 400 - 400 steps is not convergence. This extends to 1000 steps
and, unlike the count-only probe, EVALS every checkpoint, so we can see
whether the shrinkage is refinement (metrics hold while n falls) or erosion
(metrics decay with n).

ONE run per seed, not five: lr is constant and the pipeline bit-deterministic,
so the optimiser state at step 400 of a 1000-step run IS the 400-step run's
final state. Membership is snapshotted at each checkpoint from the optimiser's
own parameters (params[i] corresponds to sites[i]: thetas is built as
{s: ... for s in sites} and params = list(thetas.values()) for the pos
objective - no deltas), then all snapshots are evaluated after the run.

Outputs steps_curve.jsonl plus four PNGs (n, free0, freeN_topk, freeM_topk vs
steps; one line per seed).

  PYTHONPATH=src python .../steps_curve.py
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
N_SEQ, EVAL_BS, GAMMA, LAMBDA = 64, 16, 0.25, 1e-5
CHECKPOINTS = (200, 400, 600, 800, 1000)
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

_orig_step = torch.optim.AdamW.step
cap = {"i": 0, "sites": None, "snaps": {}}


def _step(self, *a, **kw):
    out = _orig_step(self, *a, **kw)
    cap["i"] += 1
    if cap["i"] in CHECKPOINTS:
        snap = {}
        params = self.param_groups[0]["params"]
        for site, p in zip(cap["sites"], params):
            idx = (torch.sigmoid(p.detach()) > 0.5).nonzero(as_tuple=True)[0]
            snap[site] = idx.cpu()
        cap["snaps"][cap["i"]] = snap
    return out


torch.optim.AdamW.step = _step

OUT = HERE / "steps_curve.jsonl"
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

    cap["i"] = 0
    cap["sites"] = up
    cap["snaps"] = {}
    print("\n=== %s (comp %d latent %d, %d sites) | a_pos %.3f ==="
          % (label, comp, latent, len(up), a_pos), flush=True)
    t0 = time.perf_counter()
    scores, _ = lm.run_learned_mask(
        inference, bank, objective="pos", sites=up,
        seed_layer=layer, seed_kind=kind, seed_latent_idx=latent,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
        mask_floor_source="dual", dual_floor_weight=GAMMA,
        steps=max(CHECKPOINTS), lr=0.05, l1_lambda=LAMBDA, keep_threshold=0.5,
        batch_size=4, holdout_frac=0.25, log_every=0,
        deep_site_threshold=disc.learned_mask.deep_site_threshold,
        deep_batch_size=disc.learned_mask.deep_batch_size,
        optimizer="adamw", weight_decay=0.05,
        code_dtype=disc.learned_mask.code_dtype)
    train_secs = time.perf_counter() - t0
    del scores
    torch.cuda.empty_cache()
    print("  trained %d steps in %.0fs; evaluating %d checkpoints"
          % (max(CHECKPOINTS), train_secs, len(CHECKPOINTS)), flush=True)

    for step in CHECKPOINTS:
        snap = cap["snaps"][step]
        keep = {site: set(idx.tolist()) for site, idx in snap.items()
                if idx.numel()}
        n = sum(len(v) for v in keep.values())

        def phi(a_e, sm=None, tk=False):
            a_c = float(circuit_only_activation(
                inference, bank, keep, up, pt, layer, kind, latent,
                pos_argmax=pa, site_means=sm, batch_size=EVAL_BS,
                respect_topk=tk)) if n else a_e
            d = a_pos - a_e
            return round((a_c - a_e) / d, 4) if abs(d) > 1e-9 else None

        row = {"comp_idx": comp, "latent": latent, "label": label,
               "steps": step, "n": n,
               "free0": phi(a_e0), "freeN_topk": phi(a_eNT, means_neg, True),
               "freeM_topk": phi(a_eMT, means_up, True)}
        rows.append(row)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  s=%-5d n=%-9s free0=%-8s freeN_tk=%-8s freeM_tk=%-8s"
              % (step, format(n, ","), row["free0"], row["freeN_topk"],
                 row["freeM_topk"]), flush=True)
    cap["snaps"] = {}
    torch.cuda.empty_cache()
fh.close()
torch.optim.AdamW.step = _orig_step

# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"L2-resid": "#4477aa", "L8-mlp": "#ee6677", "L10-resid": "#228833"}
PLOTS = [("n", "members (log scale)", "steps_curve_n.png", True),
         ("free0", "free0", "steps_curve_free0.png", False),
         ("freeN_topk", "freeN_topk", "steps_curve_freeN_topk.png", False),
         ("freeM_topk", "freeM_topk", "steps_curve_freeM_topk.png", False)]
for key, ylabel, fname, logy in PLOTS:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for comp, latent, label in TARGETS:
        pts = [(r["steps"], r[key]) for r in rows
               if r["label"] == label and r[key] is not None]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                color=COLORS[label], label="%s (%d)" % (label, latent))
    if logy:
        ax.set_yscale("log")
    else:
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("training steps")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(CHECKPOINTS))
    ax.legend()
    ax.set_title("dual mask (gamma=0.25, lambda=1e-5) vs training length")
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=140)
    plt.close(fig)
    print("wrote", fname, flush=True)
