"""SIGN CENSUS: solo-remove each of the top-K active upstream latents
on a seed's pos contexts and measure the seed's pre-activation change.
The sign distribution of causal effects answers the fragility
hypothesis ("more inhibitors than activators?") directly, per seed,
before any mask is fitted -- and calibrates raise_gamma for the
inhibitor-mask campaign (weak total inhibition => gamma 2 unreachable).

Removal = zero the latent at its site at ALL positions (delta decode),
read the seed pre-act at its argmax on HELD-OUT probes.
  negative delta (seed falls)  -> the latent was an ACTIVATOR
  positive delta (seed rises)  -> the latent was an INHIBITOR

  COMP=29 LAT=3736 TOPK_ACT=200 PYTHONPATH=src python .../sign_census.py
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
from eval.ablation_faithfulness import upstream_sites
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
TOPK_ACT = int(os.environ.get("TOPK_ACT", 200))
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


class ZeroLatent:
    def __init__(self, site, lat):
        self.site, self.lat = site, lat

    def __call__(self, model):
        return multi_patch(model, self.tf)

    def tf(self, layer_idx, kind, x):
        if (layer_idx, kind) != self.site:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        code = dense.clone()
        code[..., self.lat] = 0.0
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
    pt_ho, pa_ho = pt[N_TRAIN:], pa[N_TRAIN:]
    UP = set(upstream_sites(bank, layer, kind))
    # CTX=neg: census on NEAR-MISS contexts where the seed is naturally
    # quiet -- the frame the fragility hypothesis actually lives in
    # ("most animals do not have trunks": what suppresses the seed
    # everywhere else?). Anchors = per-sequence argmax of the seed's
    # natural pre-activation on the negatives.
    if os.environ.get("CTX") == "neg":
        nt = pd_.neg_tokens[:N_SEQ]
        pt_ho = nt[N_TRAIN:]
        _tap = SeedTap((layer, kind), w_seed, b_seed, None)
        _chunks = []
        inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(pt_ho.shape[0]), EVAL_BS):
                    _tap.pre = None
                    inference.forward(pt_ho[s0:s0 + EVAL_BS], patcher=_tap,
                                      grad_enabled=False,
                                      return_activations=False,
                                      tokenize_final=False)
                    _chunks.append(_tap.pre.detach())
        finally:
            inference.enable_compile()
        pa_ho = torch.cat(_chunks, 0).argmax(dim=1).cpu()
        print("[CTX=neg] censusing on near-miss negatives", flush=True)

    # accumulate mean activation per upstream latent on held-out probes
    # simpler robust pass: accumulate mean activation per latent
    sums = {st: torch.zeros(bank.d_sae, device=device) for st in UP}

    def cb2(layer_idx, acts):
        with torch.no_grad():
            for ki_, kd in enumerate(bank.kinds):
                st = (layer_idx, kd)
                if st not in UP:
                    continue
                ta, ti = bank.encode(acts[ki_], kd, layer_idx)
                sums[st].index_add_(0, ti.reshape(-1).long(),
                                    ta.float().reshape(-1))

    with torch.no_grad():
        for s0 in range(0, int(pt_ho.shape[0]), EVAL_BS):
            inference.forward(pt_ho[s0:s0 + EVAL_BS], num_gen=1,
                              tokenize_final=False,
                              activations_callback=cb2,
                              return_activations=False)
    allv = []
    for st, v in sums.items():
        nz = (v > 0).nonzero(as_tuple=True)[0]
        for j in nz.tolist():
            allv.append((float(v[j]), st, j))
    allv.sort(reverse=True)
    cands = allv[:TOPK_ACT]
    print("seed c%d/%d: %d active upstream latents, censusing top %d"
          % (COMP, LAT, len(allv), len(cands)), flush=True)

    def read(patcher_inner):
        tap = SeedTap((layer, kind), w_seed, b_seed, patcher_inner)
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(pt_ho.shape[0]), EVAL_BS):
                    tap.pre = None
                    inference.forward(pt_ho[s0:s0 + EVAL_BS], patcher=tap,
                                      grad_enabled=False,
                                      return_activations=False,
                                      tokenize_final=False)
                    pre = tap.pre
                    B = pre.shape[0]
                    rr = torch.arange(B, device=pre.device)
                    anc = pa_ho[s0:s0 + B].to(pre.device).clamp(
                        0, pre.shape[1] - 1)
                    tot += float(pre[rr, anc].sum())
                    n += B
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    a_nat = read(None)
    print("natural pre %.3f" % a_nat, flush=True)
    rows = []
    out = HERE / ("sign_census_c%d_%d.jsonl" % (COMP, LAT))
    fh = out.open("w")
    for k, (asum, st, j) in enumerate(cands):
        d = read(ZeroLatent(st, j)) - a_nat
        rows.append({"site": "%d/%s" % st, "latent": j,
                     "act_sum": round(asum, 2), "delta": round(d, 4)})
        fh.write(json.dumps(rows[-1]) + "\n")
        if (k + 1) % 40 == 0:
            print("  %d/%d" % (k + 1, len(cands)), flush=True)
    fh.close()

    inh = [r for r in rows if r["delta"] > 0.01 * abs(a_nat)]
    act_ = [r for r in rows if r["delta"] < -0.01 * abs(a_nat)]
    neu = len(rows) - len(inh) - len(act_)
    print("\nCENSUS c%d/%d (threshold 1%% of natural %.2f):" % (COMP, LAT, a_nat))
    print("  ACTIVATORS %d | INHIBITORS %d | neutral %d"
          % (len(act_), len(inh), neu))
    tot_up = sum(r["delta"] for r in inh)
    tot_dn = sum(-r["delta"] for r in act_)
    print("  total inhibition available +%.2f (%.0f%% of natural) | "
          "total activation -%.2f" % (tot_up, 100 * tot_up / abs(a_nat),
                                      tot_dn))
    inh.sort(key=lambda r: -r["delta"])
    print("  strongest inhibitors:")
    for r in inh[:8]:
        print("    %-9s %-7d delta %+.3f" % (r["site"], r["latent"],
                                             r["delta"]))
    print("-> %s" % out.name)


if __name__ == "__main__":
    main()
