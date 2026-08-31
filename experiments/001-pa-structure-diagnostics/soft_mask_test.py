"""DECISIVE recipe test — the per-position soft-mask functional eval.

Three per-(position, latent) keep rules, each swept over a threshold to trace
free0 vs UNION size (distinct latents kept at any position). free0 uses a
custom multiplicative patcher that reproduces production free0 exactly when
the mask is a broadcast binary keep-set: patched = live * mask; decode + error.

  union-mag   per-LATENT total |attr|; keep top-N -> broadcast to all positions
              (the current PA union / magnitude baseline).
  perpos-mag  per-(pos, latent) RAW |attr|; keep where > theta (position-aware
              magnitude — isolates "does per-position selection help").
  recipe      per-(pos, latent) LOW-RANK reconstruction (W.H, r=16) of the
              attribution; keep where > theta (isolates "does the recipe
              structure add value OVER raw per-position magnitude").

Decisive read (free0-vs-union-size curves):
  recipe > perpos-mag  -> recipes add functional value (denoise/generalize)
  recipe ~ perpos-mag  -> recipes are descriptive-only
  perpos ~ union-mag   -> position-awareness itself buys nothing here

Memory-safe: everything lives in the capped m_cols (<=24k) column space;
dense d_sae only materializes per-site-per-batch inside the patcher.
Rows -> soft_mask.jsonl.  PYTHONPATH=src python soft_mask_test.py
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method, _restore_sweep_config,
)
from analysis.circuits.gradient_method_neg_mode_grid_runner import _candidate_with_index
from circuit.instrument.ig_baseline import collect_natural_codes
from circuit.instrument.restoration import MaskedRestorationInstrument
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import measure_seed_activation, upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense, target_latent_activations

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
N_SEEDS = 10
TOPK_CAPTURE = 128
COL_CAP = 24_000
R_STAR = 16
NMF_ITERS = 120
CHUNK_B = 8
EVAL_BS = 16
TARGET_UNION = (1_000, 4_000, 16_000, 48_000)     # union-mag keep counts
QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)           # perpos / recipe thresholds
OUT = Path(__file__).parent / "soft_mask.jsonl"
Site = Tuple[int, str]
torch.set_float32_matmul_precision("high")


def _layer_stratified_indices(candidates, sample_size, n_kinds, seed=42):
    by_layer = defaultdict(list)
    for index, cand in enumerate(candidates):
        by_layer[int(cand["comp_idx"]) // n_kinds].append(index)
    rng = random.Random(seed)
    for layer in by_layer:
        rng.shuffle(by_layer[layer])
    selected, max_len = [], max(len(v) for v in by_layer.values())
    for rank in range(max_len):
        for layer in sorted(by_layer):
            idxs = by_layer[layer]
            if rank < len(idxs):
                selected.append(idxs[rank])
                if len(selected) >= sample_size:
                    return selected
    return selected


def nmf(V, r, iters=NMF_ITERS, seed=0, eps=1e-10):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, m = V.shape
    W = (torch.rand(n, r, generator=g) * 0.1 + 0.01).to(V.device)
    H = (torch.rand(r, m, generator=g) * 0.1 + 0.01).to(V.device)
    for _ in range(iters):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ (H @ H.T) + eps)
    return W, H


class SoftFree0Patcher:
    """free0 with a per-(pos, latent) mask carried in the m_cols column space.
    ``gate`` is [Bb, T, m_cols]; ``site_cols[(l,k)]`` = (gate_col_idx,
    latent_idx). Dense [Bb, T, d_sae] materializes for one site at a time."""

    def __init__(self, bank, in_scope, gate, site_cols, seed_layer, seed_kind,
                 seed_lat, pos_argmax):
        self.bank, self.in_scope = bank, in_scope
        self.gate, self.site_cols = gate, site_cols
        self.seed_layer, self.seed_kind, self.seed_lat = seed_layer, seed_kind, seed_lat
        self.pa = pos_argmax.detach().cpu()
        self.captured: Optional[float] = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        B, T, _ = x.shape
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            ta, ti = self.bank.encode(x, kind, layer_idx)
            sd = target_latent_activations(ta, ti, self.seed_lat)
            aB = min(B, self.pa.shape[0])
            pa = self.pa[:aB].to(x.device).clamp(0, T - 1)
            self.captured = sd[:aB][torch.arange(aB, device=x.device), pa].mean().item()
        if (layer_idx, kind) not in self.in_scope:
            return x
        ta, ti = self.bank.encode(x, kind, layer_idx)
        all_latents = sparse_topk_to_dense(ta, ti, self.bank.d_sae, dtype=x.dtype)
        error = x - self.bank.decode(all_latents, kind, layer_idx)
        gci, lat = self.site_cols[(layer_idx, kind)]
        m = torch.zeros(B, T, self.bank.d_sae, device=x.device, dtype=x.dtype)
        if gci.numel():
            m[:, :, lat] = self.gate[:B, :, gci].to(x.dtype)
        return self.bank.decode(all_latents * m, kind, layer_idx) + error


load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), bank.d_sae),
                       dtype=torch.float32, device=bank.device)
n_kinds = len(bank.kinds)
all_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu", weights_only=False)
cands = [_candidate_with_index(all_cand[i], i)
         for i in _layer_stratified_indices(all_cand, N_SEEDS, n_kinds)]
print(f"sampled {len(cands)} seeds -> {OUT}", flush=True)

original = _apply_sweep_config(max_per_site=24)
disc = config.discovery
saved = (disc.probe_sequence_count, disc.eval_sequence_count, disc.eval_batch_size)
disc.probe_sequence_count = 64
disc.eval_sequence_count = 64
disc.eval_batch_size = EVAL_BS


def seed_under_gate(gate, site_cols, in_scope, seed_layer, seed_kind, sl, pt, pa):
    vals = []
    for s in range(0, pt.shape[0], EVAL_BS):
        tk = pt[s:s + EVAL_BS]
        p = SoftFree0Patcher(bank, in_scope, gate[s:s + EVAL_BS], site_cols,
                             seed_layer, seed_kind, sl, pa[s:s + EVAL_BS])
        inference.disable_compile()
        try:
            inference.forward(tk, patcher=p, grad_enabled=False,
                              return_activations=False, tokenize_final=False)
        finally:
            inference.enable_compile()
        vals.append(p.captured * tk.shape[0])
    return sum(vals) / pt.shape[0]


t0 = time.time()
try:
    with OUT.open("a") as fh:
        for si, cand in enumerate(cands):
            sc, sl = int(cand["comp_idx"]), int(cand["latent_idx"])
            seed_layer, ski = split_component_idx(sc, n_kinds)
            seed_kind = bank.kinds[ski]
            t_seed = time.time()
            try:
                m0 = _build_mode_method("counterfactual_gradient", "local",
                                        inference, bank, avg_acts, probe_builder)
                pd = m0.build_probe_dataset(sc, sl)
                if pd.pos_tokens.shape[0] == 0:
                    print(f"[{si+1}] {sc}/{sl}: no pos — skip", flush=True); continue
                sites = sorted(upstream_sites(bank, seed_layer, seed_kind))
                if not sites:
                    continue
                pt, pa = pd.pos_tokens[:64], pd.pos_argmax[:64]
                B, T = pt.shape
                site_index = {s: i for i, s in enumerate(sites)}
                sae = bank.saes[seed_kind][seed_layer]
                w_seed = sae.encoder.weight[sl].detach()
                b_seed = sae._get_bias_eff()[sl].detach()
                _, residuals = collect_natural_codes(inference, bank, pt, set(sites))
                floors0 = {s: torch.zeros(bank.d_sae) for s in sites}
                masks1 = {s: torch.ones(bank.d_sae, dtype=torch.bool) for s in sites}

                # ---- CAPTURE: sparse COO (row=(b*T+t), col=site*d+lat) -------
                e_rows, e_cols, e_vals = [], [], []
                for start in range(0, B, CHUNK_B):
                    tk = pt[start:start + CHUNK_B]
                    res_chunk = {s: (r[start:start + CHUNK_B] if r.dim() == 3 else r)
                                 for s, r in residuals.items()}
                    inst = MaskedRestorationInstrument(
                        bank, set(sites), res_chunk, floors0, masks1,
                        seed_layer, seed_kind, w_seed, b_seed)
                    inference.disable_compile()
                    try:
                        inference.forward(tk, patcher=inst, grad_enabled=True,
                                          return_activations=False, tokenize_final=False)
                    finally:
                        inference.enable_compile()
                    pre = inst.seed_pre_act
                    Bc = tk.shape[0]
                    pac = pa[start:start + Bc].to(pre.device).clamp(0, pre.shape[1] - 1)
                    peak = pre[torch.arange(Bc, device=pre.device), pac]
                    order = sorted(inst.leaves)
                    grads = torch.autograd.grad(peak.mean(), [inst.leaves[s] for s in order],
                                                allow_unused=True)
                    rb = (torch.arange(Bc, device=device) + start).view(-1, 1) * T \
                        + torch.arange(T, device=device).view(1, -1)
                    for s, gr in zip(order, grads):
                        if gr is None:
                            continue
                        a = (gr.to(torch.float32) * inst.leaves[s].detach().to(torch.float32)).abs()
                        k = min(TOPK_CAPTURE, a.shape[-1])
                        vals, lats = a.topk(k, dim=-1)
                        cols = site_index[s] * bank.d_sae + lats
                        rows = rb.unsqueeze(-1).expand_as(cols)
                        nz = vals > 0
                        e_rows.append(rows[nz]); e_cols.append(cols[nz]); e_vals.append(vals[nz])
                    del inst, grads
                rows_f = torch.cat(e_rows); cols_f = torch.cat(e_cols); vals_f = torch.cat(e_vals)
                del e_rows, e_cols, e_vals
                n_slots = len(sites) * bank.d_sae

                mass_all = torch.zeros(n_slots, device=device)
                mass_all.index_add_(0, cols_f, vals_f)
                m_cols = min(COL_CAP, int((mass_all > 0).sum()))
                top_mass, top_cols = mass_all.topk(m_cols)          # global col ids
                col_map = torch.full((n_slots,), -1, dtype=torch.long, device=device)
                col_map[top_cols] = torch.arange(m_cols, device=device)
                keep = col_map[cols_f] >= 0
                rk, ck, vk = rows_f[keep], col_map[cols_f[keep]], vals_f[keep]
                del rows_f, cols_f, vals_f

                # dense per-(b,t) magnitude over kept columns [B*T, m_cols]
                Vraw = torch.zeros(B * T * m_cols, device=device)
                Vraw.index_add_(0, rk * m_cols + ck, vk)
                Vraw = Vraw.view(B * T, m_cols)
                # NMF on L1-normalized rows
                rowsum = Vraw.sum(dim=1, keepdim=True).clamp_min(1e-12)
                W, H = nmf(Vraw / rowsum, R_STAR, seed=0)
                R = (W @ H)                                          # [B*T, m_cols]
                perpos = Vraw                                       # raw magnitude
                del W, H

                # column -> (site, latent) and per-site kept-column groups
                col_site = (top_cols // bank.d_sae)
                col_lat = (top_cols % bank.d_sae)
                site_cols = {}
                for s in sites:
                    sel = (col_site == site_index[s]).nonzero(as_tuple=True)[0]
                    site_cols[s] = (sel, col_lat[sel])              # gate-col idx, latent idx

                a_pos = measure_seed_activation(inference, bank, pt, seed_layer,
                                                seed_kind, sl, pa, batch_size=EVAL_BS)
                zero_gate = torch.zeros(B, T, m_cols, device=device)
                a_e0 = seed_under_gate(zero_gate, site_cols, set(sites),
                                       seed_layer, seed_kind, sl, pt, pa)
                den = a_pos - a_e0

                def free0_of(gate):
                    if abs(den) < 1e-9:
                        return None
                    a_c = seed_under_gate(gate, site_cols, set(sites),
                                          seed_layer, seed_kind, sl, pt, pa)
                    return round(float((a_c - a_e0) / den), 4)

                def union_and_eff(gate):
                    pos = (gate > 0).reshape(-1, m_cols)
                    usize = int((pos.sum(dim=0) > 0).sum())          # distinct latents
                    eff = float(pos.sum()) / (B * T)                 # avg kept / position
                    return usize, round(eff, 1)

                results = {"union-mag": [], "perpos-mag": [], "recipe": []}
                # union-mag: top-N columns by mass, broadcast to all positions
                order_mass = torch.argsort(top_mass, descending=True)
                for N in TARGET_UNION:
                    keepc = order_mass[:min(N, m_cols)]
                    gate = torch.zeros(B, T, m_cols, device=device)
                    gate[:, :, keepc] = 1.0
                    u, e = union_and_eff(gate)
                    results["union-mag"].append((u, e, free0_of(gate)))
                    del gate
                # perpos / recipe: quantile thresholds on the per-(pos) score
                for name, score in (("perpos-mag", perpos.view(B, T, m_cols)),
                                    ("recipe", R.view(B, T, m_cols))):
                    pos = score[score > 0]
                    qs = torch.quantile(pos.float()[:2_000_000].to(device),
                                        torch.tensor(QUANTILES, device=device))
                    for thr in qs.tolist():
                        gate = (score > thr).float()
                        u, e = union_and_eff(gate)
                        results[name].append((u, e, free0_of(gate)))
                        del gate
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                rec = {"comp": sc, "latent": sl, "layer": seed_layer, "kind": seed_kind,
                       "m_cols": m_cols, "a_pos": round(float(a_pos), 4),
                       "a_e0": round(float(a_e0), 4), "results": results,
                       "secs": round(time.time() - t_seed, 1)}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind}:\n"
                      f"    union-mag  {results['union-mag']}\n"
                      f"    perpos-mag {results['perpos-mag']}\n"
                      f"    recipe     {results['recipe']}  ({rec['secs']:.0f}s)", flush=True)
                del Vraw, perpos, R, rk, ck, vk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"[{si+1}] {sc}/{sl} FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
finally:
    (disc.probe_sequence_count, disc.eval_sequence_count, disc.eval_batch_size) = saved
    _restore_sweep_config(original)
print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
