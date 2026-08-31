"""Cohesion-coverage frontier (PA structure, test 12).

Cohesion is the primary metric, so make it a CONSTRAINT and let coverage be the
outcome: "at cohesion >= X, how much of the union can we cover?"

Two levers, one experiment, per (layer,kind) site:
  RESOLUTION SWEEP   Louvain at gamma in RESOLUTIONS. Higher gamma -> smaller,
                     tighter communities natively (counteracts modularity's
                     resolution limit, which merges tight groups into loose ones).
  CORE EXTRACTION    Louvain must assign EVERY latent, diluting clusters. For
                     each community, iteratively drop the member with the lowest
                     mean co-association to the rest until the cluster clears the
                     cohesion bar -> tight CORE + explicit leftovers. This gives
                     Louvain the "reject a member" ability it lacks.

Outputs, per (seed, site, gamma, bar): covered latents / covered mass, n cores,
median core size — i.e. the frontier. Also records the RAW (un-extracted)
coverage at each bar so we can separate "resolution helped" from
"core-extraction helped".

Rows -> cohesion_frontier.jsonl.  PYTHONPATH=src python cohesion_frontier_test.py
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
R_RUNS, BOOT_FRAC = 20, 0.8
RESOLUTIONS = (0.5, 1.0, 2.0, 4.0, 8.0)
BARS = (0.5, 0.6, 0.7, 0.8, 0.9)
MIN_CORE = 5              # a core must keep >= this many latents to count
MAX_LEVELS, MAX_PASSES = 8, 12
NMF_ITERS, CHUNK_B = 100, 8
OUT = Path(__file__).parent / "cohesion_frontier.jsonl"
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
    """mean pairwise co-association within member index tensor idx."""
    k = idx.numel()
    if k < 2:
        return 0.0
    sub = C[idx][:, idx]
    return float(sub.sum() / (k * (k - 1)))


def extract_core(C, idx, bar, min_core=MIN_CORE):
    """Iteratively drop the weakest member (lowest mean co-assoc to the rest)
    until the cluster's cohesion clears `bar`. Returns the surviving index
    tensor (possibly empty if it can never clear)."""
    cur = idx.clone()
    while cur.numel() >= min_core:
        k = cur.numel()
        sub = C[cur][:, cur]
        coh = float(sub.sum() / (k * (k - 1)))
        if coh >= bar:
            return cur
        # drop weakest member
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
print(f"sampled {len(cands)} seeds -> {OUT}", flush=True)

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
                    coassoc = torch.zeros(ncap, ncap, device=device)
                    g = torch.Generator().manual_seed(sidx[s])
                    for run in range(R_RUNS):
                        sub = torch.randperm(Vl.shape[0], generator=g)[:max(2, int(BOOT_FRAC * Vl.shape[0]))].to(device)
                        Vb = Vl[sub]; Vb = Vb / Vb.sum(1, keepdim=True).clamp_min(1e-12)
                        _, H = nmf(Vb, r, seed=run)
                        lab = H.argmax(0)
                        coassoc += (lab.unsqueeze(0) == lab.unsqueeze(1)).float()
                    coassoc /= R_RUNS
                    coassoc.fill_diagonal_(0)
                    Ccpu = coassoc.cpu().to(torch.float64)
                    total_mass = float(top_mass.sum())
                    mass_cpu = top_mass.cpu().to(torch.float64)

                    for gamma in RESOLUTIONS:
                        labels = louvain(Ccpu, resolution=gamma, seed=sidx[s])
                        groups = [(labels == c).nonzero(as_tuple=True)[0]
                                  for c in torch.unique(labels)]
                        groups = [gi for gi in groups if gi.numel() >= 2]
                        cohs = [cohesion_of(Ccpu, gi) for gi in groups]
                        for bar in BARS:
                            # RAW: whole communities that already clear the bar
                            raw_lat = sum(int(gi.numel()) for gi, c in zip(groups, cohs)
                                          if c >= bar and gi.numel() >= MIN_CORE)
                            raw_mass = sum(float(mass_cpu[gi].sum()) for gi, c in zip(groups, cohs)
                                           if c >= bar and gi.numel() >= MIN_CORE)
                            # CORE-EXTRACTED
                            cores = [extract_core(Ccpu, gi, bar) for gi in groups]
                            cores = [ci for ci in cores if ci.numel() >= MIN_CORE]
                            core_lat = sum(int(ci.numel()) for ci in cores)
                            core_mass = sum(float(mass_cpu[ci].sum()) for ci in cores)
                            med_sz = int(torch.tensor([float(ci.numel()) for ci in cores]).median()) if cores else 0
                            rec = {"comp": sc, "latent": sl, "seed_layer": seed_layer,
                                   "site_layer": s[0], "site_kind": s[1], "site_n": ncap,
                                   "gamma": gamma, "bar": bar,
                                   "n_comm": len(groups),
                                   "raw_cov_lat": round(raw_lat / ncap, 4),
                                   "raw_cov_mass": round(raw_mass / max(total_mass, 1e-9), 4),
                                   "core_n": len(cores), "core_med_size": med_sz,
                                   "core_cov_lat": round(core_lat / ncap, 4),
                                   "core_cov_mass": round(core_mass / max(total_mass, 1e-9), 4)}
                            fh.write(json.dumps(rec) + "\n"); n_rows += 1
                    fh.flush()
                    del V, coassoc, Ccpu
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
