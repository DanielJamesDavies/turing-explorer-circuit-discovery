"""abl-mask on 32 L2-resid seeds — is there a UNIVERSAL core? (2026-08-04)

Daniel's question: run the closure mask over many seeds in one layer and
look at the crossover — are there latents that appear in EVERY seed's
circuit?

Why L2 resid specifically: every seed at comp_idx 8 shares the identical
8-site upstream scope (327,680 slots), so membership sets live in one
common universe and overlap is directly comparable without any
renormalisation. It is also the cheapest layer to run 32 masks on.

The trap this run is built to avoid: abl-mask circuits are ~10^4 nodes
and only ~6% of L2's dictionary is ever LIVE (RESULT 6c of the
recursive-map sweep: 20,914/327,680). Two sets of 10^4 drawn from a live
pool of ~2x10^4 MUST overlap heavily by pigeonhole. So "shared nodes"
means nothing without:

  (a) a null: random sets of matched size drawn from each seed's own LIVE
      set (the only latents a mask could ever select), and
  (b) a density control: corpus activation frequency per latent, to test
      whether universal nodes are just the latents that fire on
      everything rather than anything seed-specific.

Both are collected here; the analysis script consumes them.

Recipe is the frozen house recipe copied from
experiments/015-prelim-matrix/runner.py (abl-mask arm: NPA by
design, posctx floor, no post-hoc pruning — the mask's membership IS the
object).

  PYTHONPATH=src python experiments/024-l2-crossover/runner.py
"""
import gzip
import json
import random
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
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP_IDX = 8                      # L2 resid
N_SEEDS = 32
N_SEQ, EVAL_BS, PA_PCTL = 64, 16, 90.0
D_SAE = 40960
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, D_SAE), device=bank.device)

_all_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu",
                       weights_only=False)
_pool = [int(c["latent_idx"]) for c in _all_cand if int(c["comp_idx"]) == COMP_IDX]
random.Random(42).shuffle(_pool)
SEEDS = sorted(_pool[:N_SEEDS])
LAYER, KI = split_component_idx(COMP_IDX, n_kinds)
KIND = bank.kinds[KI]
UP = upstream_sites(bank, LAYER, KIND)
UP_SORTED = sorted(UP)
SCOPE = len(UP_SORTED) * D_SAE
print("L%d %s | %d seeds from %d candidates | %d upstream sites | scope %d"
      % (LAYER, KIND, len(SEEDS), len(_pool), len(UP_SORTED), SCOPE), flush=True)

_apply_sweep_config(max_per_site=24)
disc = config.discovery
cf_cfg, ab_cfg = disc.counterfactual_gradient, disc.ablation_gradient


def base_state():
    """Frozen house recipe (prelim-matrix ARMS, abl-mask row)."""
    disc.probe_sequence_count = N_SEQ
    disc.eval_sequence_count = N_SEQ
    disc.eval_batch_size = EVAL_BS
    disc.probe_batch_size = 4
    disc.position_aware = False          # abl-mask is NPA by design
    disc.floor_source = "posctx"
    disc.magnitude_prune = False
    disc.recurrence_prune = False
    disc.min_faithfulness = -100.0
    cf_cfg.max_neg_sequences = 64
    cf_cfg.neg_batch_size = 8
    cf_cfg.negative_roles = "include"
    ab_cfg.negative_roles = "include"
    cf_cfg.pruning_threshold = 0.0
    ab_cfg.pruning_threshold = 0.0


class ActiveCounter:
    """Counts, per upstream site, how often each latent is non-zero."""

    def __init__(self, sites):
        self.sites = set(sites)
        self.counts = {s: torch.zeros(D_SAE, dtype=torch.float64) for s in sites}
        self.positions = 0

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        s = (layer_idx, kind)
        if s in self.sites:
            ta, ti = bank.encode(x, kind, layer_idx)
            # top-k slots can carry zero values when fewer than k latents are
            # active; count only genuinely firing ones
            flat = ti.reshape(-1).to(torch.long)[ta.reshape(-1) > 0]
            c = torch.bincount(flat, minlength=D_SAE).double().cpu()
            self.counts[s] += c
            if s == min(self.sites):
                self.positions += int(ti.shape[0] * ti.shape[1])
        return x


def count_active(tokens):
    ctr = ActiveCounter(UP_SORTED)
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                inference.forward(tokens[s0:s0 + EVAL_BS], patcher=ctr,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    return ctr


# ---- corpus activation density, once (shared: every seed has the same scope)
DENS = HERE / "corpus_density.pt"
if DENS.exists():
    _d = torch.load(DENS, weights_only=False)
    dens_counts, dens_positions = _d["counts"], _d["positions"]
    print("corpus density: loaded cache", flush=True)
else:
    _rng = random.Random(77)
    _ids = [_rng.randrange(1, len(loader)) for _ in range(64)]
    _corpus = probe_builder._load_all_ids(_ids, max_length=64).to(device)
    _c = count_active(_corpus)
    dens_counts, dens_positions = _c.counts, _c.positions
    torch.save({"counts": dens_counts, "positions": dens_positions}, DENS)
    live = sum(int((v > 0).sum()) for v in dens_counts.values())
    print("corpus density: %d positions, %d/%d latents ever active (%.2f%%)"
          % (dens_positions, live, SCOPE, 100.0 * live / SCOPE), flush=True)
    del _corpus, _c
    torch.cuda.empty_cache()

OUT = HERE / "rows.jsonl"
MEMB = HERE / "members.jsonl.gz"
done = set()
if OUT.exists():
    for line in OUT.open():
        if line.strip():
            done.add(json.loads(line)["latent"])
    print("resuming: %d seeds already done" % len(done), flush=True)

fh = OUT.open("a")
mh = gzip.open(MEMB, "at")
for si, sl in enumerate(SEEDS):
    if sl in done:
        continue
    t0 = time.time()
    base_state()
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(COMP_IDX, sl)
    del m0
    if pd_.pos_tokens.shape[0] == 0:
        print("[%d/%d] latent %d — no positive contexts, skip"
              % (si + 1, len(SEEDS), sl), flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    a_pos = float(measure_seed_activation(inference, bank, pt, LAYER, KIND, sl,
                                          pa, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, UP, pt, LAYER,
                                         KIND, sl, pos_argmax=pa,
                                         batch_size=EVAL_BS))
    # live set: what EVER fires at the upstream sites on this seed's probes
    lc = count_active(pt)
    live_per_site = {s: int((v > 0).sum()) for s, v in lc.counts.items()}
    n_live = sum(live_per_site.values())
    del lc
    torch.cuda.empty_cache()

    base_state()
    meth = _build_mode_method("ablation_gradient", "mask", inference, bank,
                              avg_acts, probe_builder)
    circ = meth.discover(COMP_IDX, sl)
    del meth
    torch.cuda.empty_cache()
    if circ is None:
        print("[%d/%d] latent %d — no circuit" % (si + 1, len(SEEDS), sl),
              flush=True)
        continue

    mem = []
    for node in circ.nodes.values():
        if node.metadata.get("role") == "seed":
            continue
        f = node.feature_id
        if f is None or (f.layer, f.kind) not in UP:
            continue
        mem.append((f.layer, f.kind, int(f.index)))
    mem = sorted(set(mem))
    keep = {}
    for l, k, i in mem:
        keep.setdefault((l, k), set()).add(i)
    den = a_pos - a_e0
    f0 = float(circuit_only_activation(inference, bank, keep, UP, pt, LAYER,
                                       KIND, sl, pos_argmax=pa,
                                       batch_size=EVAL_BS))
    row = {"latent": sl, "comp_idx": COMP_IDX, "layer": LAYER, "kind": KIND,
           "n_nodes": len(mem), "n_live": n_live,
           "live_per_site": {"%d/%s" % s: v for s, v in live_per_site.items()},
           "pct_scope": round(100.0 * len(mem) / SCOPE, 4),
           "pct_live": round(100.0 * len(mem) / max(n_live, 1), 2),
           "a_pos": round(a_pos, 4),
           "free0": round((f0 - a_e0) / den, 4) if abs(den) > 1e-9 else None,
           "secs": round(time.time() - t0, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    mh.write(json.dumps({"latent": sl, "members": [
        [l, k, i] for l, k, i in mem]}) + "\n"); mh.flush()
    print("[%2d/%d] latent %-6d n=%-7d (%.2f%% live) free0=%-7s %.0fs"
          % (si + 1, len(SEEDS), sl, len(mem), row["pct_live"], row["free0"],
             row["secs"]), flush=True)
    del circ
    torch.cuda.empty_cache()

fh.close(); mh.close()
print("ALL DONE", flush=True)
