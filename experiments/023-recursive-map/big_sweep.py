"""Where do the free-family evals turn on? — the FULL-SCOPE ladder.

RESULT 5b left L8 flat at free0 0.000 to n=7,266 and L11 flat at 0.000
on everything to n=5,789. This pushes the same object to 100% of the
upstream dictionary (1.06M latents on L8, 1.43M on L11) to find where
each floor convention actually turns on — and, critically, where the
size-matched RANDOM control turns on, because a metric that only rises
once random rises too is measuring set size, not the circuit.

Uses the FLAT one-hop value-edge ranking, not recursive expansion:
RESULT 5c showed the two are indistinguishable on every metric at
matched size, and flat costs one backward pass total instead of one per
node, so it can be extended to arbitrary n. The full ranking over every
upstream latent is sorted ONCE and read off as prefixes.

Metrics: free0 (zero floor), freeM_topk (corpus-mean floor),
freeN_topk (negctx floor), pin0 (collapsed pins). Drive/necessity
columns are deliberately omitted — RESULT 5e showed cf_alpha and sup are
size-contaminated in exactly this regime, so they would be
uninterpretable here.

  PYTHONPATH=src python experiments/023-recursive-map/big_sweep.py
"""
import json
import random
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.floors import collect_site_anchors, collect_site_means
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ, N_TR, EVAL_BS, GRAD_B = 64, 48, 16, 8
D_SAE = 40960
# fractions of the upstream dictionary to probe
FRACS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
torch.set_float32_matmul_precision("high")
SEEDS = [(26, 17432), (35, 6599), (8, 20333)]

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, bank.d_sae), device=bank.device)
_apply_sweep_config(max_per_site=24)
config.discovery.eval_batch_size = EVAL_BS


class SeedPreActCapture:
    """Seed pre-activation on arbitrary tokens (for negctx argmax)."""

    def __init__(self, layer, kind, w_enc, b_enc):
        self.layer, self.kind = layer, kind
        self.w, self.b = w_enc, b_enc
        self.seed_pre_act = None

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == (self.layer, self.kind):
            self.seed_pre_act = (x.float() @ self.w.float()) + self.b.float()
        return x


fh = (HERE / "big_sweep.jsonl").open("a")
for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    up_sorted = sorted(up)
    scope = len(up_sorted) * D_SAE

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TR], pa[:N_TR], nt[:N_TR]
    pt_ev, pa_ev = pt[N_TR:], pa[N_TR:]
    a_pos = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                          sl, pa_ev, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, up, pt_ev, layer,
                                         kind, sl, pos_argmax=pa_ev,
                                         batch_size=EVAL_BS))
    den = a_pos - a_e0
    _, pins = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                   pin_position_specific=False)
    sae0 = bank.saes[kind][layer]
    cap = SeedPreActCapture(layer, kind, sae0.encoder.weight[sl].detach(),
                            sae0._get_bias_eff()[sl].detach())
    inference.disable_compile()
    try:
        ch = []
        with torch.no_grad():
            for s0 in range(0, int(nt_tr.shape[0]), EVAL_BS):
                cap.seed_pre_act = None
                inference.forward(nt_tr[s0:s0 + EVAL_BS], patcher=cap,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
                ch.append(cap.seed_pre_act.detach())
    finally:
        inference.enable_compile()
    na_tr = torch.cat(ch, 0).argmax(dim=1).cpu()
    _, neg_means = collect_site_anchors(inference, bank, nt_tr, up, na_tr,
                                        pin_position_specific=False)
    _rngm = random.Random(77)
    _ids = [_rngm.randrange(1, len(loader)) for _ in range(32)]
    _corpus = probe_builder._load_all_ids(_ids, max_length=64).to(pt_ev.device)
    mean_floor = collect_site_means(inference, bank, _corpus, up)
    print("\n[%s] L%d %s | %d sites | scope %d | a_pos %.3f | a_e0 %.3f"
          % (seed_key, layer, kind, len(up), scope, a_pos, a_e0), flush=True)

    # ---- one backward: the seed's full value-edge weight over ALL upstream
    instrument = SAEGraphInstrument(bank)
    inference.disable_compile()
    try:
        inference.forward(pt_tr[:GRAD_B], patcher=instrument, grad_enabled=True,
                          return_activations=False, tokenize_final=False)
    finally:
        inference.enable_compile()
    graph = instrument.graph
    avail = [s for s in up_sorted if s in graph.activations]
    anchors = [graph.get_latents(*s)[0].act for s in avail]
    _, conn, _ = graph.get_latents(layer, kind)
    v = conn.act[..., sl]
    B = min(v.shape[0], pa_tr.shape[0])
    rows = torch.arange(B, device=v.device)
    val = v[:B][rows, pa_tr[:B].to(v.device).clamp(0, v.shape[1] - 1)].mean()
    grads = torch.autograd.grad(val, anchors, allow_unused=True)
    W = torch.zeros(len(up_sorted), D_SAE)
    for s, a, g in zip(avail, anchors, grads):
        if g is None:
            continue
        W[up_sorted.index(s)] = (g * a.detach()).sum(dim=1).mean(dim=0).abs().float().cpu()
    instrument.release()
    del instrument, graph, anchors, grads
    torch.cuda.empty_cache()
    order = torch.argsort(W.flatten(), descending=True)
    nz = int((W.flatten() > 0).sum())
    print("  ranking built: %d/%d latents carry non-zero weight (%.2f%%)"
          % (nz, scope, 100.0 * nz / scope), flush=True)

    def keep_from_flat(idx):
        keep = {}
        site_of, lat_of = idx // D_SAE, idx % D_SAE
        for si in torch.unique(site_of).tolist():
            keep[up_sorted[si]] = set(lat_of[site_of == si].tolist())
        return keep

    def score(keep, tag, n, frac):
        f0 = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS))
        fm = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS,
            site_means=mean_floor, respect_topk=True))
        fn = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS,
            site_means=neg_means, respect_topk=True))
        p0 = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, pin_values=pins))
        nrm = lambda x: round((x - a_e0) / den, 4) if abs(den) > 1e-9 else None
        row = {"seed": seed_key, "layer": layer, "kind": kind, "set": tag,
               "n": int(n), "frac": frac, "scope": scope,
               "pct_dict": round(100.0 * n / scope, 3),
               "free0": nrm(f0), "freeM_topk": nrm(fm),
               "freeN_topk": nrm(fn), "pin0": nrm(p0)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("   %5.1f%% n=%-8d %-6s free0=%-8s freeM=%-8s freeN=%-8s pin0=%s"
              % (row["pct_dict"], n, tag, row["free0"], row["freeM_topk"],
                 row["freeN_topk"], row["pin0"]), flush=True)

    rng = torch.Generator().manual_seed(101)
    for frac in FRACS:
        n = max(1, int(round(frac * scope)))
        t0 = time.time()
        score(keep_from_flat(order[:n]), "flat", n, frac)
        score(keep_from_flat(torch.randperm(scope, generator=rng)[:n]),
              "random", n, frac)
        print("      (%.0fs)" % (time.time() - t0), flush=True)
    del W, order
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
