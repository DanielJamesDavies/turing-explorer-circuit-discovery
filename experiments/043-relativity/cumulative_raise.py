"""CUMULATIVE RAISE CURVE: target-free inhibitor measurement.

Instead of fitting a mask against a raise target (which degenerates
when the target is unreachable, and which unconstrained would climb
into the measured +1.6e6 stream-destruction artifact), MEASURE the
best achievable raise directly: order the censused candidates by solo
effect, remove them cumulatively, and record the seed's held-out
pre-activation after each addition. Superadditivity shows up as the
curve exceeding the sum of solos; vandalism cannot be reported because
nothing is optimised -- every point is just a measurement.

Runs in the frame of the census file given (pos or neg).

  COMP=29 LAT=3736 CENSUS=sign_census_c29_3736.jsonl [CTX=pos] \
      MAXK=48 PYTHONPATH=src python .../cumulative_raise.py
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/"
                "Runs/20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP = int(os.environ["COMP"])
LAT = int(os.environ["LAT"])
CENSUS = os.environ["CENSUS"]
MAXK = int(os.environ.get("MAXK", 48))
N_SEQ, N_TRAIN, EVAL_BS = 64, 48, 16

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=True,
               compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4


class ZeroSet:
    def __init__(self, removals):
        self.by_site = {}
        for st, lat in removals:
            self.by_site.setdefault(st, []).append(lat)

    def __call__(self, model):
        return multi_patch(model, self.tf)

    def tf(self, layer_idx, kind, x):
        lats = self.by_site.get((layer_idx, kind))
        if not lats:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        code = dense.clone()
        code[..., lats] = 0.0
        out = bank.decode(code - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


class SeedTap:
    def __init__(self, seed_site, w, b, inner=None):
        self.seed_site, self.w, self.b = seed_site, w, b
        self.inner = inner
        self.pre = None

    def __call__(self, model):
        import contextlib

        @contextlib.contextmanager
        def cm():
            with (self.inner(model) if self.inner
                  else contextlib.nullcontext()):
                with multi_patch(model, self.tf):
                    yield
        return cm()

    def tf(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            w = self.w.to(device=x.device, dtype=x.dtype)
            b = self.b.to(device=x.device, dtype=x.dtype)
            self.pre = x @ w + b
        return x


def main():
    layer, ki = split_component_idx(COMP, n_kinds)
    kind = bank.kinds[ki]
    sae = bank.saes[kind][layer]
    w_seed = sae.encoder.weight[LAT].detach()
    b_seed = sae._get_bias_eff()[LAT].detach()
    avg = torch.zeros((bank.n_layer * n_kinds, bank.d_sae),
                      device=bank.device)
    m0 = _build_mode_method("counterfactual_gradient", "local", inference,
                            bank, avg, pb)
    pd_ = m0.build_probe_dataset(COMP, LAT)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    tokens_ho, anchors_ho = pt[N_TRAIN:], pa[N_TRAIN:]
    if os.environ.get("CTX") == "neg":
        nt = pd_.neg_tokens[:N_SEQ]
        tokens_ho = nt[N_TRAIN:]
        tap0 = SeedTap((layer, kind), w_seed, b_seed, None)
        chunks = []
        inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(tokens_ho.shape[0]), EVAL_BS):
                    tap0.pre = None
                    inference.forward(tokens_ho[s0:s0 + EVAL_BS],
                                      patcher=tap0, grad_enabled=False,
                                      return_activations=False,
                                      tokenize_final=False)
                    chunks.append(tap0.pre.detach())
        finally:
            inference.enable_compile()
        anchors_ho = torch.cat(chunks, 0).argmax(dim=1).cpu()

    def read(removals):
        tap = SeedTap((layer, kind), w_seed, b_seed,
                      ZeroSet(removals) if removals else None)
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(tokens_ho.shape[0]), EVAL_BS):
                    tap.pre = None
                    inference.forward(tokens_ho[s0:s0 + EVAL_BS],
                                      patcher=tap, grad_enabled=False,
                                      return_activations=False,
                                      tokenize_final=False)
                    pre = tap.pre
                    B = pre.shape[0]
                    rr = torch.arange(B, device=pre.device)
                    anc = anchors_ho[s0:s0 + B].to(pre.device).clamp(
                        0, pre.shape[1] - 1)
                    tot += float(pre[rr, anc].sum())
                    n += B
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    rows = [json.loads(l) for l in open(HERE / CENSUS)]
    # candidates with ANY positive solo effect, strongest first, then the
    # near-zero tail (sub-threshold candidates can still add collectively)
    rows.sort(key=lambda r: -r["delta"])
    cand = [r for r in rows if r["delta"] > 0][:MAXK]
    a_nat = read([])
    print("frame=%s natural %.3f | %d positive-solo candidates (of %d), "
          "cumulative removal:" % (os.environ.get("CTX", "pos"), a_nat,
                                   len(cand), len(rows)), flush=True)
    removals = []
    solo_sum = 0.0
    print("%-4s %10s %10s %12s" % ("k", "pre", "raise", "vs solo-sum"))
    for k, r in enumerate(cand, 1):
        lyr, knd = r["site"].split("/")
        removals.append(((int(lyr), knd), r["latent"]))
        solo_sum += r["delta"]
        a = read(removals)
        print("%-4d %10.3f %+10.3f %+12.3f"
              % (k, a, a - a_nat, (a - a_nat) - solo_sum), flush=True)
    print("\nfinal: raise %+0.3f (%.0f%% of natural) with k=%d removals"
          % (a - a_nat, 100 * (a - a_nat) / max(abs(a_nat), 1e-9),
             len(removals)))


if __name__ == "__main__":
    main()
