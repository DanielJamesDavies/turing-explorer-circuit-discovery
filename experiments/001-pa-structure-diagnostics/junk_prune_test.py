"""Junk-cluster pruning test (PA structure, test 5).

If the unstable ("junk") NMF clusters are noise, their latents should be
prunable with little closure cost — and, to be a USEFUL signal (unlike the
position-gating that lost to magnitude in Test 3), prunable BETTER than
magnitude alone. Per seed (12):

  1. capture per-position |attr|; split-half NMF (r=16) fits A, B; greedy-match;
     JUNK cluster = cross-fit match cosine < 0.9. Assign each kept latent to
     its dominant cluster in fit A -> junk set = latents whose home is junk.
  2. free0 (production circuit_only_activation, position-independent union) for:
       full            all kept latents
       minus-junk      drop junk-cluster latents
       minus-magN      drop the N LOWEST-mass latents (N = |junk|) — the
                       magnitude-prune control: does cluster beat magnitude?
       minus-randN     drop N random latents — the weak control
       stable-only     == minus-junk (kept for clarity)
  Decisive:
    minus-junk ~ full         -> junk is prunable noise (validates structure)
    minus-junk > minus-magN   -> cluster identifies prunable latents BEYOND mass
    minus-junk >> minus-randN  -> not just "removing anything is fine"

Rows -> junk_prune.jsonl.  PYTHONPATH=src python junk_prune_test.py
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
from eval.ablation_faithfulness import circuit_only_activation, measure_seed_activation, upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
N_SEEDS = 12
TOPK_CAPTURE = 128
COL_CAP = 24_000
R = 16
NMF_ITERS = 150
JUNK_COS = 0.9
CHUNK_B = 8
EVAL_BS = 16
OUT = Path(__file__).parent / "junk_prune.jsonl"
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


def greedy(HA, HB):
    A = HA / HA.norm(dim=1, keepdim=True).clamp_min(1e-9)
    B = HB / HB.norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = (A @ B.T).clone(); r = C.shape[0]; cs = [0.] * r
    for _ in range(r):
        idx = torch.argmax(C); i, j = int(idx // r), int(idx % r)
        cs[i] = float(C[i, j]); C[i, :] = -1; C[:, j] = -1
    return cs


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
saved = (disc.probe_sequence_count, disc.eval_sequence_count, disc.eval_batch_size)
disc.probe_sequence_count = 64; disc.eval_sequence_count = 64; disc.eval_batch_size = EVAL_BS

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
                VA, VB = Vf[hA], Vf[~hA]; lA, lB = VA.sum(1) > 0, VB.sum(1) > 0
                WA, HA = nmf(nr(VA[lA]), R, seed=0); WB, HB = nmf(nr(VB[lB]), R, seed=0)
                cs = greedy(HA, HB)
                junk_clusters = [i for i, c in enumerate(cs) if c < JUNK_COS]
                n_stable = R - len(junk_clusters)
                # assign each kept column to dominant cluster (fit A membership)
                MA = HA / HA.sum(0, keepdim=True).clamp_min(1e-12)      # [R, mc]
                home = MA.argmax(0)                                     # [mc]
                col_mass = ma_all[tc]                                   # [mc]
                junk_mask = torch.zeros(mc, dtype=torch.bool, device=device)
                for jc in junk_clusters:
                    junk_mask |= (home == jc)
                n_junk = int(junk_mask.sum())
                junk_mass_frac = float(col_mass[junk_mask].sum() / col_mass.sum().clamp_min(1e-12))

                col_site = (tc // bank.d_sae); col_lat = (tc % bank.d_sae)

                def keep_from(mask):
                    kd = defaultdict(set)
                    idxs = mask.nonzero(as_tuple=True)[0]
                    for j in idxs.tolist():
                        kd[sites[int(col_site[j])]].add(int(col_lat[j]))
                    return {k: v for k, v in kd.items()}

                a_pos = measure_seed_activation(inference, bank, pt, seed_layer, seed_kind, sl, pa, batch_size=EVAL_BS)
                a_e0 = circuit_only_activation(inference, bank, {}, set(sites), pt, seed_layer, seed_kind, sl, pos_argmax=pa, batch_size=EVAL_BS)
                den = a_pos - a_e0

                def free0(mask):
                    if abs(den) < 1e-9:
                        return None
                    a_c = circuit_only_activation(inference, bank, keep_from(mask), set(sites), pt, seed_layer, seed_kind, sl, pos_argmax=pa, batch_size=EVAL_BS)
                    return round(float((a_c - a_e0) / den), 4)

                full = torch.ones(mc, dtype=torch.bool, device=device)
                minus_junk = ~junk_mask
                # magnitude control: drop the N=n_junk lowest-mass latents
                lowest = torch.argsort(col_mass)[:n_junk]
                minus_mag = full.clone(); minus_mag[lowest] = False
                # random control
                g = torch.Generator().manual_seed(11)
                randdrop = torch.randperm(mc, generator=g)[:n_junk].to(device)
                minus_rand = full.clone(); minus_rand[randdrop] = False

                rec = {"comp": sc, "latent": sl, "layer": seed_layer, "kind": seed_kind,
                       "m_cols": mc, "n_junk_clusters": len(junk_clusters), "n_stable_clusters": n_stable,
                       "n_junk_latents": n_junk, "junk_frac": round(n_junk / mc, 4),
                       "junk_mass_frac": round(junk_mass_frac, 4),
                       "match_cos_sorted": [round(x, 3) for x in sorted(cs, reverse=True)],
                       "free0_full": free0(full),
                       "free0_minus_junk": free0(minus_junk),
                       "free0_minus_magN": free0(minus_mag),
                       "free0_minus_randN": free0(minus_rand),
                       "secs": round(time.time() - t_seed, 1)}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind}: "
                      f"{len(junk_clusters)} junk clus, {n_junk} lat ({rec['junk_frac']*100:.0f}%, "
                      f"mass {rec['junk_mass_frac']*100:.0f}%) | free0 full={rec['free0_full']} "
                      f"-junk={rec['free0_minus_junk']} -magN={rec['free0_minus_magN']} "
                      f"-randN={rec['free0_minus_randN']} | {rec['secs']:.0f}s", flush=True)
                del Vf, VA, VB, WA, HA, HB
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
