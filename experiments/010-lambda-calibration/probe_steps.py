"""Can the probe run fewer steps? The probe only needs a node COUNT.

The probe exists to measure n at a known lambda so the power law can solve for
the target. It does not need a GOOD circuit - only a countable one. If n at
100 steps predicts n at 400 by a stable factor, the probe costs a quarter.

MEASURED IN ONE RUN, NOT FOUR. lr_schedule is "constant" and nothing else in
the loop depends on the total step count, so a 100-step run's final state IS
the state at step 100 of a 400-step run (the pipeline is bit-deterministic).
Snapshotting member counts mid-run therefore gives exactly what separate runs
would, at a quarter of the cost.

The count is read from the OPTIMISER's parameters rather than from a captured
patcher: run_learned_mask builds several patchers (the training one, the
dual zero-floor twin, and the closed-mask probes used for the normaliser),
and the last one constructed is a fully-shut diagnostic patcher - capturing
"the patcher" would silently read m = 0 everywhere.

What matters is not that n is stable - it will not be, the mask is still
pruning - but whether n(s)/n(400) is CONSISTENT ACROSS SEEDS. A stable ratio
is a correction factor; a variable one means early stopping just adds noise.

  PYTHONPATH=src python .../probe_steps.py
"""
import json
import os
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
N_SEQ, GAMMA, LAMBDA = 64, 0.25, 1e-5
CHECKPOINTS = (25, 50, 100, 150, 200, 300, 400)
PER_COMPONENT = int(os.environ.get("PER_COMPONENT", "2"))
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

# reuse the seeds already measured, so n at step 400 can be cross-checked
prev = [json.loads(l) for l in (HERE / "within_component.jsonl").open() if l.strip()]
by_comp = defaultdict(list)
for r in prev:
    by_comp[r["comp_idx"]].append(r)
TARGETS = []
for comp in sorted(by_comp):
    TARGETS.extend(by_comp[comp][:PER_COMPONENT])

_orig_step = torch.optim.AdamW.step
state = {"i": 0, "snaps": {}}


def _step(self, *a, **kw):
    out = _orig_step(self, *a, **kw)
    state["i"] += 1
    if state["i"] in CHECKPOINTS:
        n = 0
        for g in self.param_groups:
            for p in g["params"]:
                n += int((torch.sigmoid(p.detach()) > 0.5).sum())
        state["snaps"][state["i"]] = n
    return out


torch.optim.AdamW.step = _step

OUT = HERE / "probe_steps.jsonl"
fh = OUT.open("a")
print("%-6s %-10s %9s" % ("comp", "site", "latent")
      + "".join("%9s" % ("n@%d" % c) for c in CHECKPOINTS)
      + "%9s %8s" % ("stored", "secs"), flush=True)
for t in TARGETS:
    comp, latent = t["comp_idx"], t["latent"]
    layer, ki = split_component_idx(comp, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))
    meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                              avg_acts, probe_builder)
    pd_ = meth.build_probe_dataset(comp, latent)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    state["i"] = 0; state["snaps"] = {}
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
    secs = time.perf_counter() - t0
    snaps = dict(state["snaps"])
    row = {"comp_idx": comp, "latent": latent, "layer": layer, "kind": kind,
           "sites": len(up), "snaps": snaps, "n_final": len(scores),
           "n_stored": t["n"], "secs": round(secs, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("%-6d %-10s %9d" % (comp, "L%d-%s" % (layer, kind), latent)
          + "".join("%9s" % format(snaps.get(c, 0), ",") for c in CHECKPOINTS)
          + "%9s %8.0f" % (format(t["n"], ","), secs), flush=True)
    del scores
    torch.cuda.empty_cache()
fh.close()
torch.optim.AdamW.step = _orig_step

rows = [json.loads(l) for l in OUT.open() if l.strip()]
print("\nRATIO n(step)/n(400) - the question is whether this is CONSISTENT")
print("%-6s %-9s" % ("comp", "latent")
      + "".join("%9s" % ("s=%d" % c) for c in CHECKPOINTS))
for r in rows:
    f = r["snaps"].get(str(max(CHECKPOINTS))) or r["snaps"].get(max(CHECKPOINTS)) \
        or r["n_final"]
    print("%-6d %-9d" % (r["comp_idx"], r["latent"])
          + "".join("%9.3f" % ((r["snaps"].get(str(c)) or r["snaps"].get(c) or 0)
                               / max(f, 1)) for c in CHECKPOINTS))
print("\nA tight column = a usable correction factor at that step count.")
print("A spread column = early stopping only adds noise; keep 400 steps.")
