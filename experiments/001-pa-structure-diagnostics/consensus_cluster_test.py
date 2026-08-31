"""Consensus / co-association clustering (PA structure, test 8).

Directly implements "latents that stick together the MAJORITY of the time" and
diagnoses whether the junk-cluster problem is an NMF artifact (fixable) or
intrinsic (some load-bearing latents have no reproducible positional home).

Per seed (10), on the top COCLUST_N latents by mass:
  1. R_RUNS NMF fits (rank NMF_R), each on a random 80% row bootstrap + unique
     seed. Per run, each latent's cluster = argmax membership.
  2. CO-ASSOCIATION C[i,j] = fraction of runs latents i,j share a cluster
     (alignment-free; no cross-run cluster matching needed).
  3. HEADLINE DIAGNOSTIC: per-latent CONSENSUS STRENGTH = mean of its top-30
     co-associations. High => it has a stable group; low => it wanders.
     Distribution + fraction "stably homed" (>0.5) + mass weighting (do the
     HIGH-mass load-bearing latents get stable homes, or are they the residual?)
  4. COMMUNITIES from C with NO k: connected components at threshold 0.5.
     Report n communities >=5, fraction of latents in them, residual fraction
     and residual MASS fraction (the "junk that survives consensus").

Read:
  most latents stably homed, small/low-mass residual -> consensus SOLVES junk
  big load-bearing low-consensus residual -> junk is INTRINSIC (no method fixes)

Rows -> consensus_cluster.jsonl.  PYTHONPATH=src python consensus_cluster_test.py
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method, _restore_sweep_config
from analysis.circuits.gradient_method_neg_mode_grid_runner import _candidate_with_index
from circuit.instrument.ig_baseline import collect_natural_codes
from circuit.instrument.restoration import MaskedRestorationInstrument
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

def connected_components(A):
    """Label propagation: each node -> min node-index in its component.
    A: [n,n] bool symmetric (diagonal 0). Returns labels [n] (long)."""
    n = A.shape[0]
    labels = torch.arange(n, device=A.device)
    BIG = torch.full((n, n), n, device=A.device, dtype=torch.long)
    for _ in range(200):
        M = torch.where(A, labels.unsqueeze(0).expand(n, n), BIG)
        cand = torch.minimum(M.min(dim=1).values, labels)
        if torch.equal(cand, labels):
            break
        labels = cand
    return labels


RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
N_SEEDS = 10
TOPK_CAPTURE = 128
COCLUST_N = 4000          # top-mass latents to consensus-cluster
NMF_R = 32                # resolution of each base fit
R_RUNS = 25               # bootstrap + restart fits
TOP_NEIGH = 30            # neighbours for consensus-strength
BOOT_FRAC = 0.8
THR = 0.5                 # co-association -> community edge
NMF_ITERS = 100
CHUNK_B = 8
OUT = Path(__file__).parent / "consensus_cluster.jsonl"
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
    g = torch.Generator().manual_seed(seed)
    W = (torch.rand(V.shape[0], r, generator=g) * .1 + .01).to(V.device)
    H = (torch.rand(r, V.shape[1], generator=g) * .1 + .01).to(V.device)
    for _ in range(iters):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ (H @ H.T) + eps)
    return W, H


def pct(t, ps):
    return [round(float(torch.quantile(t, p)), 3) for p in ps]


load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), bank.d_sae), device=bank.device)
n_kinds = len(bank.kinds)
all_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu", weights_only=False)
cands = [_candidate_with_index(all_cand[i], i)
         for i in _layer_stratified_indices(all_cand, N_SEEDS, n_kinds)]
print(f"sampled {len(cands)} seeds -> {OUT}", flush=True)

original = _apply_sweep_config(max_per_site=24)
disc = config.discovery
saved = (disc.probe_sequence_count, disc.eval_sequence_count)
disc.probe_sequence_count = 64; disc.eval_sequence_count = 64

t0 = time.time()
try:
    with OUT.open("a") as fh:
        for si, cand in enumerate(cands):
            sc, sl = int(cand["comp_idx"]), int(cand["latent_idx"])
            seed_layer, ski = split_component_idx(sc, n_kinds)
            seed_kind = bank.kinds[ski]
            t_seed = time.time()
            try:
                m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank, avg_acts, probe_builder)
                pd = m0.build_probe_dataset(sc, sl)
                if pd.pos_tokens.shape[0] == 0:
                    print(f"[{si+1}] {sc}/{sl}: no pos — skip", flush=True); continue
                sites = sorted(upstream_sites(bank, seed_layer, seed_kind))
                if not sites:
                    continue
                pt, pa = pd.pos_tokens[:64], pd.pos_argmax[:64]
                B, T = pt.shape; sidx = {s: i for i, s in enumerate(sites)}
                sae = bank.saes[seed_kind][seed_layer]
                w_seed = sae.encoder.weight[sl].detach(); b_seed = sae._get_bias_eff()[sl].detach()
                _, res = collect_natural_codes(inference, bank, pt, set(sites))
                f0d = {s: torch.zeros(bank.d_sae) for s in sites}
                m1 = {s: torch.ones(bank.d_sae, dtype=torch.bool) for s in sites}
                er, ec, ev = [], [], []
                for st in range(0, B, CHUNK_B):
                    tk = pt[st:st + CHUNK_B]
                    rc = {s: (r[st:st + CHUNK_B] if r.dim() == 3 else r) for s, r in res.items()}
                    ins = MaskedRestorationInstrument(bank, set(sites), rc, f0d, m1, seed_layer, seed_kind, w_seed, b_seed)
                    inference.disable_compile()
                    try:
                        inference.forward(tk, patcher=ins, grad_enabled=True, return_activations=False, tokenize_final=False)
                    finally:
                        inference.enable_compile()
                    pre = ins.seed_pre_act; Bc = tk.shape[0]
                    pac = pa[st:st + Bc].to(pre.device).clamp(0, pre.shape[1] - 1)
                    peak = pre[torch.arange(Bc, device=pre.device), pac]
                    order = sorted(ins.leaves)
                    gr = torch.autograd.grad(peak.mean(), [ins.leaves[s] for s in order], allow_unused=True)
                    rb = (torch.arange(Bc, device=device) + st).view(-1, 1) * T + torch.arange(T, device=device).view(1, -1)
                    for s, g in zip(order, gr):
                        if g is None:
                            continue
                        a = (g.float() * ins.leaves[s].detach().float()).abs()
                        k = min(TOPK_CAPTURE, a.shape[-1]); v, la = a.topk(k, dim=-1)
                        co = sidx[s] * bank.d_sae + la; ro = rb.unsqueeze(-1).expand_as(co); nz = v > 0
                        er.append(ro[nz]); ec.append(co[nz]); ev.append(v[nz])
                    del ins, gr
                rf = torch.cat(er); cf = torch.cat(ec); vf = torch.cat(ev); nslot = len(sites) * bank.d_sae
                ma_all = torch.zeros(nslot, device=device); ma_all.index_add_(0, cf, vf)
                n = min(COCLUST_N, int((ma_all > 0).sum()))
                top_mass, top_cols = ma_all.topk(n)
                cm = torch.full((nslot,), -1, dtype=torch.long, device=device); cm[top_cols] = torch.arange(n, device=device)
                kp = cm[cf] >= 0; rk, ck, vk = rf[kp], cm[cf[kp]], vf[kp]
                V = torch.zeros(B * T * n, device=device); V.index_add_(0, rk * n + ck, vk); V = V.view(B * T, n)
                live = (V.sum(1) > 0).nonzero(as_tuple=True)[0]
                Vl = V[live]                                   # [n_live, n]

                # ---- R_RUNS bootstrap+restart NMF -> co-association ----------
                coassoc = torch.zeros(n, n, device=device)
                g = torch.Generator().manual_seed(0)
                for run in range(R_RUNS):
                    m_live = Vl.shape[0]
                    sub = torch.randperm(m_live, generator=g)[:int(BOOT_FRAC * m_live)].to(device)
                    Vb = Vl[sub]
                    Vb = Vb / Vb.sum(1, keepdim=True).clamp_min(1e-12)
                    _, H = nmf(Vb, NMF_R, seed=run)
                    lab = H.argmax(0)                          # [n] cluster per latent
                    coassoc += (lab.unsqueeze(0) == lab.unsqueeze(1)).float()
                coassoc /= R_RUNS
                coassoc.fill_diagonal_(0)

                # ---- consensus strength (top-30 co-assoc mean) --------------
                topv, _ = coassoc.topk(min(TOP_NEIGH, n - 1), dim=1)
                strength = topv.mean(dim=1)                    # [n]
                homed = strength > 0.5
                col_mass = top_mass                            # [n]
                mw = col_mass / col_mass.sum()
                hi = col_mass >= torch.quantile(col_mass, 0.75)
                lo = col_mass <= torch.quantile(col_mass, 0.25)

                # ---- communities: connected components at THR ---------------
                A = (coassoc > THR)
                A = A | A.T
                labels_t = connected_components(A)
                sizes = torch.bincount(labels_t)
                big = (sizes >= 5)
                in_big = big[labels_t]                         # [n] latent in a >=5 community
                comm = {
                    "n_comm_ge5": int(big.sum()),
                    "frac_in_comm": round(float(in_big.float().mean()), 4),
                    "residual_frac": round(float((~in_big).float().mean()), 4),
                    "residual_mass_frac": round(float(col_mass[~in_big].sum() / col_mass.sum()), 4),
                    "largest_comm_frac": round(float(sizes.max() / n), 4),
                }

                rec = {"comp": sc, "latent": sl, "layer": seed_layer, "kind": seed_kind,
                       "n_latents": n, "n_live_rows": int(Vl.shape[0]),
                       "strength_pctl_10_25_50_75_90": pct(strength, [.1, .25, .5, .75, .9]),
                       "frac_stably_homed": round(float(homed.float().mean()), 4),
                       "mass_weighted_strength": round(float((strength * mw).sum()), 4),
                       "strength_top25mass": round(float(strength[hi].mean()), 4),
                       "strength_bot25mass": round(float(strength[lo].mean()), 4),
                       **comm, "secs": round(time.time() - t_seed, 1)}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind} (n={n}): "
                      f"strength p50={rec['strength_pctl_10_25_50_75_90'][2]} "
                      f"homed={rec['frac_stably_homed']} | mass-wtd={rec['mass_weighted_strength']} "
                      f"(hi/lo mass {rec['strength_top25mass']}/{rec['strength_bot25mass']}) | "
                      f"comm>=5={rec['n_comm_ge5']} in-comm={rec['frac_in_comm']} "
                      f"resid={rec['residual_frac']} (mass {rec['residual_mass_frac']}) "
                      f"largest={rec['largest_comm_frac']} | {rec['secs']:.0f}s", flush=True)
                del V, Vl, coassoc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"[{si+1}] {sc}/{sl} FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
finally:
    (disc.probe_sequence_count, disc.eval_sequence_count) = saved
    _restore_sweep_config(original)
print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
