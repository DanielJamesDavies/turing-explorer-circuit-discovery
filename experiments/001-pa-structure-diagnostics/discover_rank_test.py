"""Discover the cluster count per seed (PA structure, test 7).

Two independent, unsupervised rank selectors over an NMF rank sweep
r in {4,8,16,24,32,48,64,96,128}, both from a split-half fit:
  STABILITY   fit A and B on disjoint sequence halves, greedy-match, count
              clusters with match cosine >= 0.9. n_stable(r) rises then
              plateaus/peaks -> the intrinsic stable-cluster count K_hat.
  HELD-OUT R2 fold-in B rows with H_A fixed; R2 on held-out B. Rises with r
              then plateaus/drops (rank overfitting) -> generalization count.
Tests the "one cluster per position (>=64)?" intuition directly: if the true
count is ~64 we'll see n_stable keep climbing to 64; if positions share
recipes (Test 2's low rank) it plateaus far below.

Reports per seed both curves, K_hat (argmax n_stable), heldout-R2 elbow, and
latents/cluster at K_hat. Capped at 24k top-mass latents (coverage ~0.9).
Rows -> discover_rank.jsonl.  PYTHONPATH=src python discover_rank_test.py
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

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
N_SEEDS = 10
TOPK_CAPTURE = 128
COL_CAP = 24_000
RANKS = (4, 8, 16, 24, 32, 48, 64, 96, 128)
NMF_ITERS = 120
JUNK_COS = 0.9
CHUNK_B = 8
OUT = Path(__file__).parent / "discover_rank.jsonl"
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


def foldin(Vb, H, iters=60, eps=1e-10, seed=7):
    g = torch.Generator().manual_seed(seed)
    Wb = (torch.rand(Vb.shape[0], H.shape[0], generator=g) * .1 + .01).to(Vb.device)
    HHt = H @ H.T
    for _ in range(iters):
        Wb *= (Vb @ H.T) / (Wb @ HHt + eps)
    return Wb


def n_stable(HA, HB, cos_thr=JUNK_COS):
    A = HA / HA.norm(dim=1, keepdim=True).clamp_min(1e-9)
    B = HB / HB.norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = (A @ B.T).clone(); r = C.shape[0]; cs = []
    for _ in range(r):
        idx = torch.argmax(C); i, j = int(idx // r), int(idx % r)
        cs.append(float(C[i, j])); C[i, :] = -1; C[:, j] = -1
    return sum(1 for c in cs if c >= cos_thr), cs


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
                rf = torch.cat(er); cf = torch.cat(ec); vf = torch.cat(ev); ns = len(sites) * bank.d_sae
                ma_all = torch.zeros(ns, device=device); ma_all.index_add_(0, cf, vf)
                mc = min(COL_CAP, int((ma_all > 0).sum())); tc = ma_all.topk(mc).indices
                cm = torch.full((ns,), -1, dtype=torch.long, device=device); cm[tc] = torch.arange(mc, device=device)
                kp = cm[cf] >= 0; rk, ck, vk = rf[kp], cm[cf[kp]], vf[kp]
                Vf = torch.zeros(B * T * mc, device=device); Vf.index_add_(0, rk * mc + ck, vk); Vf = Vf.view(B * T, mc)
                rsq = torch.arange(B * T, device=device) // T; hA = rsq % 2 == 0

                def nr(M):
                    return M / M.sum(dim=1, keepdim=True).clamp_min(1e-12)
                VA = nr(Vf[hA][Vf[hA].sum(1) > 0]); VB = nr(Vf[~hA][Vf[~hA].sum(1) > 0])
                VBnorm2 = (torch.linalg.norm(VB) ** 2).clamp_min(1e-12)

                stable_curve, r2_train, r2_held = {}, {}, {}
                for r in RANKS:
                    if r > VA.shape[0] or r > VB.shape[0]:
                        continue
                    WA, HA = nmf(VA, r, seed=0)
                    WB, HB = nmf(VB, r, seed=0)
                    nst, _ = n_stable(HA, HB)
                    stable_curve[r] = nst
                    r2_train[r] = round(float(1 - torch.linalg.norm(VA - WA @ HA) ** 2
                                              / (torch.linalg.norm(VA) ** 2).clamp_min(1e-12)), 4)
                    Wb = foldin(VB, HA)      # held-out: B rows on A's dictionary
                    r2_held[r] = round(float(1 - torch.linalg.norm(VB - Wb @ HA) ** 2 / VBnorm2), 4)
                    del WA, HA, WB, HB, Wb

                Khat = max(stable_curve, key=lambda r: stable_curve[r])
                # held-out elbow: last r before held-out R2 gain < 0.01
                held_r = sorted(r2_held)
                held_elbow = held_r[-1]
                for a, b in zip(held_r, held_r[1:]):
                    if r2_held[b] - r2_held[a] < 0.01:
                        held_elbow = a
                        break

                rec = {"comp": sc, "latent": sl, "layer": seed_layer, "kind": seed_kind, "m_cols": mc,
                       "n_stable_by_rank": stable_curve, "r2_train_by_rank": r2_train,
                       "r2_heldout_by_rank": r2_held, "K_hat_stability": Khat,
                       "max_stable": stable_curve[Khat], "K_heldout_elbow": held_elbow,
                       "lat_per_cluster_at_Khat": round(mc / max(stable_curve[Khat], 1), 1),
                       "secs": round(time.time() - t_seed, 1)}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind} (mc={mc}): "
                      f"n_stable {stable_curve} | K_hat={Khat} (max_stable={stable_curve[Khat]}) "
                      f"heldout-elbow={held_elbow} | lat/clus@Khat={rec['lat_per_cluster_at_Khat']} "
                      f"| {rec['secs']:.0f}s", flush=True)
                del Vf, VA, VB
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
