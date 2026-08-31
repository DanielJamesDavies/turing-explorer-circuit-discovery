"""HELD-OUT validated cohesion frontier (PA structure, test 13).

Test 12's frontier was measured on the SAME 20 consensus runs the core
extraction optimized against. Since extraction greedily drops members to
maximize MEASURED cohesion, it partly fits consensus noise (winner's curse) —
so that frontier is biased UP. This test quantifies and removes the bias.

  R_RUNS = 60 fits per site (3x test 12), split into two DISJOINT halves:
    C_A from runs 0..29   -> used for Louvain (gamma=4) AND core extraction
    C_B from runs 30..59  -> NEVER seen during extraction; used only to score
  For each bar: extract cores on C_A, then re-measure each core's cohesion on
  C_B. Coverage_insample = mass in cores clearing the bar on C_A (what test 12
  reported); coverage_heldout = mass in cores clearing it on C_B (honest).
  shrinkage = mean(coh_A) - mean(coh_B) over extracted cores = the bias.

Also reports a NULL: cohesion of size-matched RANDOM member sets on C_B, so we
can tell a real held-out cohesion from what any set of that size would score.

Rows -> heldout_frontier.jsonl.  PYTHONPATH=src python heldout_frontier_test.py
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
N_SEEDS = 8
TOPK_CAPTURE = 128
MIN_LAT, PERSITE_CAP = 50, 2000
MAX_R, MIN_R, R_DIV = 64, 8, 6
R_RUNS, BOOT_FRAC = 60, 0.8          # 3x test 12; split into 2 halves of 30
GAMMA = 4.0                           # the operating point chosen in test 12
BARS = (0.5, 0.6, 0.7, 0.8, 0.9)
MIN_CORE = 5
MAX_LEVELS, MAX_PASSES = 8, 12
NMF_ITERS, CHUNK_B = 100, 8
OUT = Path(__file__).parent / "heldout_frontier.jsonl"
torch.set_float32_matmul_precision("high")


def louvain(W, resolution=1.0, max_levels=MAX_LEVELS, max_passes=MAX_PASSES, seed=0):
    A = W.clone().to(torch.float64)
    n0 = A.shape[0]
    orig = torch.arange(n0)
    g = torch.Generator().manual_seed(seed)
    for _lvl in range(max_levels):
        n = A.shape[0]; k = A.sum(1); m2 = float(A.sum())
        if m2 <= 0:
            break
        comm = torch.arange(n); sigma = k.clone(); moved_any = False
        for _p in range(max_passes):
            moved = False
            for i in torch.randperm(n, generator=g).tolist():
                ci = int(comm[i]); ki = float(k[i])
                sigma[ci] -= ki
                row = A[i].clone(); row[i] = 0.0
                kin = torch.bincount(comm, weights=row, minlength=n)
                gains = kin - resolution * sigma * ki / m2
                best = int(torch.argmax(gains))
                if float(gains[best]) > float(gains[ci]) + 1e-12:
                    comm[i] = best; moved = True; moved_any = True
                sigma[int(comm[i])] += ki
            if not moved:
                break
        uniq, comm_new = torch.unique(comm, return_inverse=True)
        n_comm = int(uniq.numel())
        orig = comm_new[orig]
        if n_comm == n or not moved_any:
            break
        S = torch.zeros(n, n_comm, dtype=A.dtype)
        S[torch.arange(n), comm_new] = 1.0
        A = S.T @ A @ S
    return orig


def cohesion_of(C, idx):
    k = idx.numel()
    if k < 2:
        return 0.0
    sub = C[idx][:, idx]
    return float(sub.sum() / (k * (k - 1)))


def extract_core(C, idx, bar, min_core=MIN_CORE):
    cur = idx.clone()
    while cur.numel() >= min_core:
        k = cur.numel()
        sub = C[cur][:, cur]
        if float(sub.sum() / (k * (k - 1))) >= bar:
            return cur
        aff = sub.sum(dim=1) / (k - 1)
        drop = int(torch.argmin(aff))
        cur = torch.cat([cur[:drop], cur[drop + 1:]])
    return cur[:0]


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
print(f"sampled {len(cands)} seeds -> {OUT} (R_RUNS={R_RUNS}, gamma={GAMMA})", flush=True)

original = _apply_sweep_config(max_per_site=24)
disc = config.discovery
saved = (disc.probe_sequence_count, disc.eval_sequence_count)
disc.probe_sequence_count = 64; disc.eval_sequence_count = 64

t0 = time.time(); n_rows = 0
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
                srows = {s: [] for s in sites}; scols = {s: [] for s in sites}; svals = {s: [] for s in sites}
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
                        kk = min(TOPK_CAPTURE, a.shape[-1]); v, la = a.topk(kk, dim=-1)
                        ro = rb.unsqueeze(-1).expand_as(la); nz = v > 0
                        srows[s].append(ro[nz]); scols[s].append(la[nz]); svals[s].append(v[nz])
                    del ins, gr

                for s in sites:
                    if not srows[s]:
                        continue
                    rr = torch.cat(srows[s]); cc_ = torch.cat(scols[s]); vv = torch.cat(svals[s])
                    mass_full = torch.zeros(bank.d_sae, device=device); mass_full.index_add_(0, cc_, vv)
                    n_active = int((mass_full > 0).sum())
                    if n_active < MIN_LAT:
                        continue
                    ncap = min(PERSITE_CAP, n_active)
                    top_mass, top_lat = mass_full.topk(ncap)
                    lmap = torch.full((bank.d_sae,), -1, dtype=torch.long, device=device); lmap[top_lat] = torch.arange(ncap, device=device)
                    kp = lmap[cc_] >= 0
                    lr, lc, lv = rr[kp], lmap[cc_[kp]], vv[kp]
                    V = torch.zeros(B * T * ncap, device=device); V.index_add_(0, lr * ncap + lc, lv); V = V.view(B * T, ncap)
                    live = (V.sum(1) > 0).nonzero(as_tuple=True)[0]
                    Vl = V[live]
                    if Vl.shape[0] < 8:
                        del V; continue
                    r = min(MAX_R, max(MIN_R, ncap // R_DIV)); r = min(r, Vl.shape[0] - 1)
                    half = R_RUNS // 2
                    CA = torch.zeros(ncap, ncap, device=device)
                    CB = torch.zeros(ncap, ncap, device=device)
                    g = torch.Generator().manual_seed(sidx[s])
                    for run in range(R_RUNS):
                        sub = torch.randperm(Vl.shape[0], generator=g)[:max(2, int(BOOT_FRAC * Vl.shape[0]))].to(device)
                        Vb = Vl[sub]; Vb = Vb / Vb.sum(1, keepdim=True).clamp_min(1e-12)
                        _, H = nmf(Vb, r, seed=run)
                        lab = H.argmax(0)
                        same = (lab.unsqueeze(0) == lab.unsqueeze(1)).float()
                        if run < half:
                            CA += same
                        else:
                            CB += same
                    CA /= half; CB /= (R_RUNS - half)
                    CA.fill_diagonal_(0); CB.fill_diagonal_(0)
                    CAc = CA.cpu().to(torch.float64); CBc = CB.cpu().to(torch.float64)
                    mass_cpu = top_mass.cpu().to(torch.float64)
                    total_mass = float(mass_cpu.sum())

                    labels = louvain(CAc, resolution=GAMMA, seed=sidx[s])
                    groups = [(labels == c).nonzero(as_tuple=True)[0] for c in torch.unique(labels)]
                    groups = [gi for gi in groups if gi.numel() >= 2]
                    rng = torch.Generator().manual_seed(99)

                    for bar in BARS:
                        cores = [extract_core(CAc, gi, bar) for gi in groups]
                        cores = [ci for ci in cores if ci.numel() >= MIN_CORE]
                        if not cores:
                            continue
                        cohA = [cohesion_of(CAc, ci) for ci in cores]
                        cohB = [cohesion_of(CBc, ci) for ci in cores]
                        # size-matched random null on CB
                        nulls = []
                        for ci in cores:
                            ridx = torch.randperm(ncap, generator=rng)[:ci.numel()]
                            nulls.append(cohesion_of(CBc, ridx))
                        in_mass = sum(float(mass_cpu[ci].sum()) for ci, c in zip(cores, cohA) if c >= bar)
                        ho_mass = sum(float(mass_cpu[ci].sum()) for ci, c in zip(cores, cohB) if c >= bar)
                        in_lat = sum(int(ci.numel()) for ci, c in zip(cores, cohA) if c >= bar)
                        ho_lat = sum(int(ci.numel()) for ci, c in zip(cores, cohB) if c >= bar)
                        rec = {"comp": sc, "latent": sl, "seed_layer": seed_layer,
                               "site_layer": s[0], "site_kind": s[1], "site_n": ncap,
                               "bar": bar, "n_cores": len(cores),
                               "med_core_size": int(torch.tensor([float(c.numel()) for c in cores]).median()),
                               "mean_coh_A": round(float(torch.tensor(cohA).mean()), 4),
                               "mean_coh_B": round(float(torch.tensor(cohB).mean()), 4),
                               "shrinkage": round(float(torch.tensor(cohA).mean() - torch.tensor(cohB).mean()), 4),
                               "mean_coh_null": round(float(torch.tensor(nulls).mean()), 4),
                               "cov_mass_insample": round(in_mass / max(total_mass, 1e-9), 4),
                               "cov_mass_heldout": round(ho_mass / max(total_mass, 1e-9), 4),
                               "cov_lat_insample": round(in_lat / ncap, 4),
                               "cov_lat_heldout": round(ho_lat / ncap, 4)}
                        fh.write(json.dumps(rec) + "\n"); n_rows += 1
                    fh.flush()
                    del V, CA, CB, CAc, CBc
                print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind}: "
                      f"{n_rows} rows ({time.time()-t_seed:.0f}s)", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"[{si+1}] {sc}/{sl} FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
finally:
    (disc.probe_sequence_count, disc.eval_sequence_count) = saved
    _restore_sweep_config(original)
print(f"\ndone: {n_rows} rows in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
