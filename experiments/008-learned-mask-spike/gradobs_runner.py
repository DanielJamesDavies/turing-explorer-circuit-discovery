"""Gradient/value observability, and the test behind the "learned circuit" idea.

Daniel's hypothesis: many members carry only MINOR negative gradients that do
not mean much individually; collectively they should not be cut. And — the
testable part — after REDUCING the inhibitors, their gradients may flip
POSITIVE, which is what would make an iterative/learned circuit (with a
learning rate) sensible rather than a one-shot sign split.

Measurements, all at the seed's probe positions:

  A. score distribution by sign (from the saved circuit) — are negatives
     systematically smaller than positives, or comparable?
  B. gradient at the NATURAL state: d(seed pre-act)/d(latent), per member,
     via SAEGraphInstrument's detached anchors (exactly the discovery signal).
  C. gradient at the INHIBITORS-ZEROED state — the iteration-2 gradient.
     THE KEY NUMBER: what fraction of members whose gradient was negative at
     the natural state have a POSITIVE gradient once the inhibitors are down?
     High -> the sign is state-dependent, a learned/iterative method is
     motivated. Low -> the sign is intrinsic, one-shot selection is right.
  D. gradient x value (the contribution), by sign, both states.

Writes per-member arrays to grads_{TAG}.pt for offline plotting.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/runner.py
"""
import json
import os
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
CIRC = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2/experiments/"
            "007-free0-cf-32seed/circuits")
HERE = Path(__file__).parent
SEEDS = {"L2": (8, 30122), "L5": (16, 32227), "L8": (25, 10628), "L10": (32, 3021)}
TAG = os.environ.get("SEED_TAG", "L8")
SC_IDX, LATENT = SEEDS[TAG]
ARM_FILE = "abl-ig_mean_PA__rec2mag"
GRAD_B, NK = 8, 3
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * NK, bank.d_sae), device=bank.device)

_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = 32
disc.eval_sequence_count = 32
disc.probe_batch_size = 4

layer, ki = split_component_idx(SC_IDX, NK)
kind = bank.kinds[ki]
up = upstream_sites(bank, layer, kind)

m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                        avg_acts, probe_builder)
pd_ = m0.build_probe_dataset(SC_IDX, LATENT)
pt, pa = pd_.pos_tokens[:GRAD_B], pd_.pos_argmax[:GRAD_B]

d = torch.load(CIRC / ("%d_%d" % (SC_IDX, LATENT)) / (ARM_FILE + ".pt"),
               map_location="cpu", weights_only=False)
roles = [d["roles_legend"][i] for i in d["role"].tolist()]
site_members = {}      # site -> (idx list, score list, is_inh list)
for (l, k, i, s), r in zip(zip(d["layer"].tolist(), d["kind_idx"].tolist(),
                               d["index"].tolist(), d["score"].tolist()), roles):
    kd = d["kinds_legend"][k]
    if r in ("seed", "residual") or (l, kd) not in up:
        continue
    rec = site_members.setdefault((l, kd), ([], [], []))
    rec[0].append(i); rec[1].append(float(s))
    rec[2].append(("inhibitor" in r) or (r == "attributed" and float(s) < 0))

all_scores = torch.tensor([s for v in site_members.values() for s in v[1]])
all_inh = torch.tensor([b for v in site_members.values() for b in v[2]])
print("[%s] %s | members %s (inh %s)"
      % (TAG, ARM_FILE, format(all_scores.numel(), ","),
         format(int(all_inh.sum()), ",")), flush=True)

pos, neg = all_scores[all_scores > 0], all_scores[all_scores < 0]
print("\nA. STORED SCORE distribution")
print("  %-10s %10s %12s %12s %12s %10s"
      % ("sign", "n", "|median|", "|p90|", "|max|", "sum|.|"))
for nm, t in (("positive", pos), ("negative", neg)):
    if t.numel():
        a = t.abs()
        print("  %-10s %10s %12.4e %12.4e %12.4e %10.3f"
              % (nm, format(t.numel(), ","), float(a.median()),
                 float(a.quantile(0.9)), float(a.max()), float(a.sum())))

w_seed = bank.saes[kind][layer].encoder.weight[LATENT].detach()
b_seed = bank.saes[kind][layer]._get_bias_eff()[LATENT].detach()


def grads_at(zero_inhibitors: bool):
    """d(seed pre-act)/d(latent) per member, optionally with inhibitors zeroed."""
    instrument = SAEGraphInstrument(bank)
    seed_pre = []
    orig = instrument.transform

    def tap(layer_idx, kd, x):
        if layer_idx == layer and kd == kind:
            seed_pre.append(x @ w_seed.to(x.device, x.dtype)
                            + b_seed.to(x.device, x.dtype))
            return x
        out = orig(layer_idx, kd, x)
        if zero_inhibitors and (layer_idx, kd) in site_members:
            idxs, _, inh = site_members[(layer_idx, kd)]
            sel = [i for i, b in zip(idxs, inh) if b]
            if sel:
                lat = instrument.graph.get_latents(layer_idx, kd)[0].act
                t = torch.tensor(sel, dtype=torch.long, device=lat.device)
                W = bank.saes[kd][layer_idx].decoder.weight.detach()
                contrib = lat[:, :, t].detach() @ W[:, t].T.to(lat.device, lat.dtype)
                out = out - contrib
        return out

    instrument.transform = tap
    inference.disable_compile()
    try:
        inference.forward(pt, patcher=instrument, grad_enabled=True,
                          return_activations=False, tokenize_final=False)
    finally:
        inference.enable_compile()
    pre = seed_pre[0]
    idx = torch.arange(min(GRAD_B, pa.shape[0]), device=pre.device)
    metric = pre[idx, pa[:len(idx)].to(pre.device).clamp(0, pre.shape[1] - 1)].mean()
    graph = instrument.graph
    sites = [s for s in sorted(graph.activations) if s in site_members]
    anchors = [graph.get_latents(*s)[0].act for s in sites]
    gs = torch.autograd.grad(metric, anchors, allow_unused=True)
    out_g, out_v = [], []
    for s, a, g in zip(sites, anchors, gs):
        idxs, _, _ = site_members[s]
        t = torch.tensor(idxs, dtype=torch.long)
        if g is None:
            out_g.append(torch.zeros(len(idxs))); out_v.append(torch.zeros(len(idxs)))
            continue
        gm = g.sum(dim=1).mean(dim=0).detach().float().cpu()
        vm = a.detach().sum(dim=1).mean(dim=0).float().cpu()
        out_g.append(gm[t]); out_v.append(vm[t])
    del instrument, graph, anchors, gs, seed_pre
    torch.cuda.empty_cache()
    return torch.cat(out_g), torch.cat(out_v)


g_nat, v_nat = grads_at(False)
g_abl, v_abl = grads_at(True)

print("\nB/C. GRADIENT d(seed pre-act)/d(latent) at probe positions")
print("  %-30s %10s %12s %12s %10s"
      % ("population", "n", "mean g", "|median g|", "frac g>0"))
for nm, mask in (("activator members", ~all_inh.bool()),
                 ("inhibitor members", all_inh.bool())):
    gn, ga = g_nat[mask], g_abl[mask]
    print("  %-30s %10s %12.4e %12.4e %9.1f%%"
          % (nm + " natural", format(int(mask.sum()), ","), float(gn.mean()),
             float(gn.abs().median()), 100 * float((gn > 0).float().mean())))
    print("  %-30s %10s %12.4e %12.4e %9.1f%%"
          % (nm + " inh-zeroed", "", float(ga.mean()),
             float(ga.abs().median()), 100 * float((ga > 0).float().mean())))

was_neg = g_nat < 0
flipped = was_neg & (g_abl > 0)
was_pos = g_nat > 0
flipped_dn = was_pos & (g_abl < 0)
print("\nTHE KEY NUMBER — gradient sign stability under intervention:")
print("  negative gradient at natural state : %s of %s"
      % (format(int(was_neg.sum()), ","), format(int(g_nat.numel()), ",")))
print("  of those, POSITIVE after zeroing inhibitors : %s (%.1f%%)"
      % (format(int(flipped.sum()), ","),
         100 * float(flipped.sum()) / max(int(was_neg.sum()), 1)))
print("  converse (positive -> negative) : %s of %s (%.1f%%)"
      % (format(int(flipped_dn.sum()), ","), format(int(was_pos.sum()), ","),
         100 * float(flipped_dn.sum()) / max(int(was_pos.sum()), 1)))
corr = float(torch.corrcoef(torch.stack([g_nat, g_abl]))[0, 1])
print("  corr(g_natural, g_ablated) = %.4f" % corr)

torch.save({"g_nat": g_nat, "g_abl": g_abl, "v_nat": v_nat, "v_abl": v_abl,
            "scores": all_scores, "is_inh": all_inh, "tag": TAG},
           HERE / ("grads_%s.pt" % TAG))
with (HERE / "rows.jsonl").open("a") as fh:
    fh.write(json.dumps({
        "tag": TAG, "seed": "%d/%d" % (SC_IDX, LATENT), "arm": ARM_FILE,
        "n_members": int(all_scores.numel()), "n_inh": int(all_inh.sum()),
        "frac_neg_grad_natural": round(float(was_neg.float().mean()), 4),
        "frac_flipped_pos": round(float(flipped.sum()) / max(int(was_neg.sum()), 1), 4),
        "frac_flipped_neg": round(float(flipped_dn.sum()) / max(int(was_pos.sum()), 1), 4),
        "corr_g": round(corr, 4)}) + "\n")
print("\nwrote grads_%s.pt" % TAG)
