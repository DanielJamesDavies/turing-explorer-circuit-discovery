"""Per-site consensus + LOUVAIN community detection (PA structure, test 10).

Same per-(layer,kind) co-association matrix as test 9, but extracts communities
by MODULARITY (Louvain) instead of thresholded connected components. CC is
single-linkage: a chain of moderate edges welds unrelated groups into one lump
(test 9: largest community = 62% of a site). Louvain asks instead "is this group
denser INSIDE than chance would predict", so chains get cut at sparse joints, it
uses the graded edge weights (not a 0.5 threshold), and needs no k.

Reports CC and Louvain side by side per site: n communities, largest-community
fraction, modularity Q, residual (communities < 5) frac + mass.

Rows -> louvain_persite.jsonl.  PYTHONPATH=src python louvain_persite_test.py
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
MIN_LAT = 50
PERSITE_CAP = 2500
MAX_R, MIN_R, R_DIV = 64, 8, 6
R_RUNS = 20
BOOT_FRAC = 0.8
THR = 0.5                 # CC threshold (for the comparison arm)
RESOLUTION = 1.0
MAX_LEVELS, MAX_PASSES = 8, 12
NMF_ITERS = 100
CHUNK_B = 8
OUT = Path(__file__).parent / "louvain_persite.jsonl"
torch.set_float32_matmul_precision("high")


def connected_components(A):
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


def modularity(A, comm, resolution=RESOLUTION):
    """Q = sum_c [ in_c/m2 - resolution*(tot_c/m2)^2 ] on weighted graph A."""
    m2 = A.sum()
    if m2 <= 0:
        return 0.0
    n_c = int(comm.max()) + 1
    k = A.sum(1)
    tot = torch.zeros(n_c, dtype=A.dtype).index_add_(0, comm, k)
    # in_c = sum of A[i,j] for i,j in c
    S = torch.zeros(A.shape[0], n_c, dtype=A.dtype)
    S[torch.arange(A.shape[0]), comm] = 1.0
    inc = (S.T @ A @ S).diagonal()
    return float((inc / m2 - resolution * (tot / m2) ** 2).sum())


def louvain(W, resolution=RESOLUTION, max_levels=MAX_LEVELS, max_passes=MAX_PASSES, seed=0):
    """Modularity-maximizing community detection (Louvain). W: [n,n] symmetric
    nonneg weights (torch CPU float64). Returns labels [n] over original nodes."""
    A = W.clone().to(torch.float64)
    n0 = A.shape[0]
    orig_to_cur = torch.arange(n0)
    g = torch.Generator().manual_seed(seed)
    for _level in range(max_levels):
        n = A.shape[0]
        k = A.sum(1)
        m2 = float(A.sum())
        if m2 <= 0:
            break
        comm = torch.arange(n)
        sigma_tot = k.clone()
        moved_any = False
        for _p in range(max_passes):
            moved = False
            for i in torch.randperm(n, generator=g).tolist():
                ci = int(comm[i])
                ki = float(k[i])
                sigma_tot[ci] -= ki
                row = A[i].clone()
                row[i] = 0.0
                k_i_in = torch.bincount(comm, weights=row, minlength=n)
                gains = k_i_in - resolution * sigma_tot * ki / m2
                best = int(torch.argmax(gains))
                if float(gains[best]) > float(gains[ci]) + 1e-12:
                    comm[i] = best
                    moved = True
                    moved_any = True
                sigma_tot[int(comm[i])] += ki
            if not moved:
                break
        uniq, comm_new = torch.unique(comm, return_inverse=True)
        n_comm = int(uniq.numel())
        orig_to_cur = comm_new[orig_to_cur]
        if n_comm == n or not moved_any:
            break
        # aggregate: B = S^T A S (keeps self-loops = internal weight)
        S = torch.zeros(n, n_comm, dtype=A.dtype)
        S[torch.arange(n), comm_new] = 1.0
        A = S.T @ A @ S
    return orig_to_cur


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


def comm_stats(labels, mass, n):
    sizes = torch.bincount(labels)
    big = sizes >= 5
    in_big = big[labels]
    return {
        "n_comm": int((sizes > 0).sum()),
        "n_comm_ge5": int(big.sum()),
        "largest_frac": float(sizes.max() / n),
        "frac_in_comm": float(in_big.float().mean()),
        "residual_frac": float((~in_big).float().mean()),
        "residual_mass": float(mass[~in_big].sum() / mass.sum().clamp_min(1e-9)),
    }


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

                cc_stats, lv_stats, qs = [], [], []
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
                    mass_c = top_mass
                    # CC arm
                    A = (coassoc > THR); A = A | A.T
                    cc_lab = connected_components(A)
                    cc_stats.append(comm_stats(cc_lab, mass_c, ncap))
                    # Louvain arm (CPU, weighted)
                    Wc = coassoc.cpu().to(torch.float64)
                    lv_lab = louvain(Wc, seed=sidx[s])
                    lv_stats.append(comm_stats(lv_lab.to(device), mass_c, ncap))
                    qs.append(modularity(Wc, lv_lab))
                    del V, coassoc, Wc

                if not lv_stats:
                    print(f"[{si+1}] {sc}/{sl}: no clusterable sites", flush=True); continue

                def med(lst, key):
                    return round(float(torch.tensor([d[key] for d in lst]).median()), 3)
                rec = {"comp": sc, "latent": sl, "layer": seed_layer, "kind": seed_kind,
                       "n_sites": len(lv_stats),
                       "cc_n_comm_ge5": med(cc_stats, "n_comm_ge5"),
                       "cc_largest_frac": med(cc_stats, "largest_frac"),
                       "cc_residual_frac": med(cc_stats, "residual_frac"),
                       "cc_residual_mass": med(cc_stats, "residual_mass"),
                       "lv_n_comm": med(lv_stats, "n_comm"),
                       "lv_n_comm_ge5": med(lv_stats, "n_comm_ge5"),
                       "lv_largest_frac": med(lv_stats, "largest_frac"),
                       "lv_frac_in_comm": med(lv_stats, "frac_in_comm"),
                       "lv_residual_frac": med(lv_stats, "residual_frac"),
                       "lv_residual_mass": med(lv_stats, "residual_mass"),
                       "modularity_Q": round(float(torch.tensor(qs).median()), 4),
                       "secs": round(time.time() - t_seed, 1)}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind} ({rec['n_sites']} sites): "
                      f"CC comm>=5={rec['cc_n_comm_ge5']} largest={rec['cc_largest_frac']} | "
                      f"LOUVAIN comm={rec['lv_n_comm']} (>=5: {rec['lv_n_comm_ge5']}) "
                      f"largest={rec['lv_largest_frac']} Q={rec['modularity_Q']} "
                      f"resid={rec['lv_residual_frac']} (mass {rec['lv_residual_mass']}) | {rec['secs']:.0f}s", flush=True)
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
