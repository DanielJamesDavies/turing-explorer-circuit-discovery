"""Per-(layer,kind) consensus clustering (PA structure, test 9).

Correction to test 8: cluster latents WITHIN EACH SAE SITE separately, so the
structure found is purely POSITIONAL (which latents at THIS site fire at the
same positions) — not conflated across layers/kinds. Rank capped at 64 (the
positional upper bound: one cluster per position), but discovered via
consensus, not forced.

Per seed (10), for each site with >= MIN_LAT active latents (capped to top
PERSITE_CAP by mass):
  - R_RUNS bootstrap+restart NMF (rank = min(64, n_s//R_DIV)); co-association;
    per-latent consensus strength (top-30 mean); frac stably homed (>0.5).
  - communities: connected components @0.5 (n>=5), residual frac + mass,
    largest-component frac (giant-blob check).
Aggregate across sites per seed (latent-weighted). Compare frac-homed to the
POOLED test-8 value (0.80) — does per-site clustering give cleaner homes?

Rows -> consensus_persite.jsonl.  PYTHONPATH=src python consensus_persite_test.py
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
MIN_LAT = 50              # min active latents to cluster a site
PERSITE_CAP = 3000        # top-mass latents per site
MAX_R = 64               # positional upper bound (one cluster per position)
MIN_R = 8
R_DIV = 6                # rank = min(MAX_R, n_s // R_DIV)
R_RUNS = 20
TOP_NEIGH = 30
BOOT_FRAC = 0.8
THR = 0.5
NMF_ITERS = 100
CHUNK_B = 8
OUT = Path(__file__).parent / "consensus_persite.jsonl"
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


def consensus_one_site(V, mass, gseed=0):
    """V: [rows, n_s] per-site positional matrix. Returns per-site metrics."""
    n = V.shape[1]
    live = (V.sum(1) > 0).nonzero(as_tuple=True)[0]
    Vl = V[live]
    if Vl.shape[0] < 8:
        return None
    r = min(MAX_R, max(MIN_R, n // R_DIV))
    r = min(r, Vl.shape[0] - 1)
    coassoc = torch.zeros(n, n, device=V.device)
    g = torch.Generator().manual_seed(gseed)
    for run in range(R_RUNS):
        m_live = Vl.shape[0]
        sub = torch.randperm(m_live, generator=g)[:max(2, int(BOOT_FRAC * m_live))].to(V.device)
        Vb = Vl[sub]
        Vb = Vb / Vb.sum(1, keepdim=True).clamp_min(1e-12)
        _, H = nmf(Vb, r, seed=run)
        lab = H.argmax(0)
        coassoc += (lab.unsqueeze(0) == lab.unsqueeze(1)).float()
    coassoc /= R_RUNS
    coassoc.fill_diagonal_(0)
    topv, _ = coassoc.topk(min(TOP_NEIGH, n - 1), dim=1)
    strength = topv.mean(dim=1)
    A = (coassoc > THR); A = A | A.T
    labels = connected_components(A)
    sizes = torch.bincount(labels)
    big = sizes >= 5
    in_big = big[labels]
    return {
        "n": n, "rank": r,
        "strength_med": float(strength.median()),
        "frac_homed": float((strength > 0.5).float().mean()),
        "n_comm_ge5": int(big.sum()),
        "frac_in_comm": float(in_big.float().mean()),
        "residual_frac": float((~in_big).float().mean()),
        "residual_mass_frac": float(mass[~in_big].sum() / mass.sum().clamp_min(1e-9)),
        "largest_comm_frac": float(sizes.max() / n),
        "strength": strength, "homed": strength > 0.5, "mass": mass,
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
                # per-site sparse COO: local row = b*T+t, col = latent index
                site_rows = {s: [] for s in sites}
                site_cols = {s: [] for s in sites}
                site_vals = {s: [] for s in sites}
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
                        ro = rb.unsqueeze(-1).expand_as(la); nz = v > 0
                        site_rows[s].append(ro[nz]); site_cols[s].append(la[nz]); site_vals[s].append(v[nz])
                    del ins, gr

                # cluster each site
                per_site = []
                all_strength, all_homed, all_mass = [], [], []
                for s in sites:
                    if not site_rows[s]:
                        continue
                    rr = torch.cat(site_rows[s]); cc = torch.cat(site_cols[s]); vv = torch.cat(site_vals[s])
                    mass_full = torch.zeros(bank.d_sae, device=device); mass_full.index_add_(0, cc, vv)
                    n_active = int((mass_full > 0).sum())
                    if n_active < MIN_LAT:
                        continue
                    ncap = min(PERSITE_CAP, n_active)
                    top_mass, top_lat = mass_full.topk(ncap)
                    lmap = torch.full((bank.d_sae,), -1, dtype=torch.long, device=device); lmap[top_lat] = torch.arange(ncap, device=device)
                    kp = lmap[cc] >= 0
                    lr, lc, lv = rr[kp], lmap[cc[kp]], vv[kp]
                    V = torch.zeros(B * T * ncap, device=device); V.index_add_(0, lr * ncap + lc, lv); V = V.view(B * T, ncap)
                    m = consensus_one_site(V, top_mass, gseed=sidx[s])
                    if m is None:
                        continue
                    per_site.append(m)
                    all_strength.append(m["strength"]); all_homed.append(m["homed"]); all_mass.append(m["mass"])
                    del V

                if not per_site:
                    print(f"[{si+1}] {sc}/{sl}: no clusterable sites — skip", flush=True); continue
                S = torch.cat(all_strength); Hm = torch.cat(all_homed); Ms = torch.cat(all_mass)

                def smed(key):
                    return round(float(torch.tensor([m[key] for m in per_site]).median()), 3)
                rec = {"comp": sc, "latent": sl, "layer": seed_layer, "kind": seed_kind,
                       "n_sites_clustered": len(per_site),
                       "median_n_per_site": int(torch.tensor([m["n"] for m in per_site]).median()),
                       "median_rank": int(torch.tensor([m["rank"] for m in per_site]).median()),
                       "frac_homed_overall": round(float(Hm.float().mean()), 4),
                       "frac_homed_massw": round(float((Hm.float() * Ms / Ms.sum()).sum()), 4),
                       "strength_med_overall": round(float(S.median()), 4),
                       "site_frac_in_comm_med": smed("frac_in_comm"),
                       "site_residual_frac_med": smed("residual_frac"),
                       "site_residual_mass_med": smed("residual_mass_frac"),
                       "site_largest_comm_med": smed("largest_comm_frac"),
                       "site_n_comm_med": smed("n_comm_ge5"),
                       "secs": round(time.time() - t_seed, 1)}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind}: "
                      f"{rec['n_sites_clustered']} sites (med n={rec['median_n_per_site']}, "
                      f"rank={rec['median_rank']}) | homed={rec['frac_homed_overall']} "
                      f"(massw {rec['frac_homed_massw']}) strength={rec['strength_med_overall']} | "
                      f"comm/site={rec['site_n_comm_med']} largest={rec['site_largest_comm_med']} "
                      f"resid={rec['site_residual_frac_med']} (mass {rec['site_residual_mass_med']}) "
                      f"| {rec['secs']:.0f}s", flush=True)
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
