"""Fuzzy latent-cluster EXPLAINER evals (PA structure, test 4).

The clusters are a DESCRIPTION of the circuit (never enforced on it): fuzzy
latent->cluster memberships from positional co-occurrence (NMF r=16 on
per-position |attr| rows). Held to the two standards a description must meet:

  EVAL 1  PREDICTIVE POWER (position completion). Fit clusters on half the
          sequences. On held-out rows: reveal half of each position's present
          latents, infer the position's cluster mixture (MU fold-in), predict
          the HIDDEN half. Pooled AUC vs two baselines: marginal frequency
          (cluster-free popularity) and shuffled-membership clusters.
          The claim is the LIFT over marginal.
  EVAL 2  STABILITY.
          (a) data:      fit on disjoint sequence halves, greedy-match
                         clusters (cosine), per-latent P(cluster|latent)
                         agreement + top-cluster persistence.
          (b) restart:   same data, different NMF seeds, same metrics.
          (c) fuzziness: per-latent membership entropy distribution
                         (median + fraction of "one-cluster citizens" < 0.5).
  BONUS   within-seed capture-set stability (Jaccard of the two halves'
          capture sets) — the control the Test-1 specificity claim owes:
          compare against the ~0.07 cross-seed tail Jaccard.

Works at ALL depths (presence prediction needs no closure). 16 seeds.
Rows -> cluster_explainer.jsonl; stable-cluster member dumps -> clusters/.
PYTHONPATH=src python cluster_explainer_test.py   (repo root, wsl + .venv)
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

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
from eval.ablation_faithfulness import upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
N_SEEDS = 16
TOPK_CAPTURE = 128
COL_CAP = 24_000
R = 16
NMF_ITERS = 150
FOLDIN_ITERS = 40
CHUNK_B = 8
N_NEG_SAMPLE = 512          # sampled absent latents per test row for AUC
MIN_PRESENT = 8             # rows with fewer present latents are skipped
OUT = Path(__file__).parent / "cluster_explainer.jsonl"
CLUST_DIR = Path(__file__).parent / "clusters"
CLUST_DIR.mkdir(exist_ok=True)
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


def fold_in(Vobs, H, iters=FOLDIN_ITERS, eps=1e-10, seed=7):
    """Infer row mixtures theta >= 0 for observed rows with H FIXED."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Th = (torch.rand(Vobs.shape[0], H.shape[0], generator=g) * 0.1 + 0.01).to(Vobs.device)
    HHt = H @ H.T
    for _ in range(iters):
        Th *= (Vobs @ H.T) / (Th @ HHt + eps)
    return Th


def pooled_auc(scores_pos, scores_neg):
    """AUC via rank statistic on pooled positive/negative score tensors."""
    s = torch.cat([scores_pos, scores_neg])
    labels = torch.cat([torch.ones_like(scores_pos), torch.zeros_like(scores_neg)])
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float32, device=s.device)
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return None
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def greedy_match(HA, HB):
    """Greedy max-cosine cluster matching. Returns perm: B-cluster for each A."""
    A = HA / HA.norm(dim=1, keepdim=True).clamp_min(1e-9)
    B = HB / HB.norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = (A @ B.T).clone()
    r = C.shape[0]
    perm = [-1] * r
    cos = [0.0] * r
    for _ in range(r):
        idx = torch.argmax(C)
        i, j = int(idx // r), int(idx % r)
        perm[i], cos[i] = j, float(C[i, j])
        C[i, :] = -1
        C[:, j] = -1
    return perm, cos


def membership(H):
    """P(cluster | latent): column-normalized H, [r, m]."""
    return H / H.sum(dim=0, keepdim=True).clamp_min(1e-12)


def stability_metrics(HA, HB, mass_A, mass_B):
    """Match clusters, then per-latent membership agreement on latents with
    mass in both fits. Returns (mean matched-cluster cosine, mean per-latent
    membership cosine, top-cluster persistence rate)."""
    perm, cos = greedy_match(HA, HB)
    HBp = HB[perm]
    MA, MB = membership(HA), membership(HBp)
    both = (mass_A > 0) & (mass_B > 0)
    if int(both.sum()) == 0:
        return sum(cos) / len(cos), None, None
    a, b = MA[:, both], MB[:, both]
    lat_cos = (a * b).sum(dim=0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-9)
    persist = (a.argmax(dim=0) == b.argmax(dim=0)).float().mean()
    return sum(cos) / len(cos), float(lat_cos.mean()), float(persist)


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
saved = (disc.probe_sequence_count, disc.eval_sequence_count)
disc.probe_sequence_count = 64
disc.eval_sequence_count = 64

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

                # ---- capture sparse per-position |attr| entries -------------
                e_rows, e_cols, e_vals = [], [], []
                for start in range(0, B, CHUNK_B):
                    tk = pt[start:start + CHUNK_B]
                    res_chunk = {s: (r_[start:start + CHUNK_B] if r_.dim() == 3 else r_)
                                 for s, r_ in residuals.items()}
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
                top_cols = mass_all.topk(m_cols).indices
                col_map = torch.full((n_slots,), -1, dtype=torch.long, device=device)
                col_map[top_cols] = torch.arange(m_cols, device=device)
                keep = col_map[cols_f] >= 0
                rk, ck, vk = rows_f[keep], col_map[cols_f[keep]], vals_f[keep]
                del rows_f, cols_f, vals_f

                Vfull = torch.zeros(B * T * m_cols, device=device)
                Vfull.index_add_(0, rk * m_cols + ck, vk)
                Vfull = Vfull.view(B * T, m_cols)
                row_seq = torch.arange(B * T, device=device) // T
                halfA = (row_seq % 2 == 0)          # even sequences
                halfB = ~halfA

                def norm_rows(M):
                    return M / M.sum(dim=1, keepdim=True).clamp_min(1e-12)

                VA, VB = norm_rows(Vfull[halfA]), norm_rows(Vfull[halfB])
                live_A = Vfull[halfA].sum(dim=1) > 0
                live_B = Vfull[halfB].sum(dim=1) > 0
                VA, VB = VA[live_A], VB[live_B]
                mass_A = Vfull[halfA].sum(dim=0)
                mass_B = Vfull[halfB].sum(dim=0)

                # within-seed capture-set stability (the Test-1 control)
                setA, setB = mass_A > 0, mass_B > 0
                jacc_within = float((setA & setB).sum()) / max(float((setA | setB).sum()), 1.0)

                # ---- fits ---------------------------------------------------
                WA, HA = nmf(VA, R, seed=0)
                WB, HB = nmf(VB, R, seed=0)
                _, HA2 = nmf(VA, R, seed=1)

                # ---- EVAL 2: stability -------------------------------------
                clus_cos_data, lat_cos_data, persist_data = stability_metrics(
                    HA, HB, mass_A, mass_B)
                clus_cos_rest, lat_cos_rest, persist_rest = stability_metrics(
                    HA, HA2, mass_A, mass_A)
                MA = membership(HA)
                present = mass_A > 0
                pm = MA[:, present]
                ent = (-(pm * (pm + 1e-12).log()).sum(dim=0) /
                       torch.log(torch.tensor(float(R)))).cpu()
                one_cluster = float((ent < 0.5).float().mean())

                # ---- EVAL 1: predictive (position completion, A->B) --------
                g = torch.Generator(device="cpu").manual_seed(3)
                Vtest = Vfull[halfB][live_B]                       # raw mass rows
                pres = Vtest > 0
                n_pres = pres.sum(dim=1)
                ok_rows = n_pres >= MIN_PRESENT
                Vtest, pres = Vtest[ok_rows], pres[ok_rows]
                reveal = torch.rand(pres.shape, generator=g).to(device) < 0.5
                obs = pres & reveal
                hid = pres & ~reveal
                Vobs = norm_rows(Vtest * obs.float())
                Hn = norm_rows(HA)                                  # P(latent|cluster)
                marg = (mass_A / mass_A.sum().clamp_min(1e-12))

                def auc_with(Hmat):
                    Th = fold_in(Vobs, Hmat)
                    Thn = norm_rows(Th)
                    S = Thn @ norm_rows(Hmat)                       # [rows, m] scores
                    pos_scores = S[hid]
                    # sampled absents per row
                    absent = ~pres & (mass_A > 0).unsqueeze(0)
                    ridx, cidx = absent.nonzero(as_tuple=True)
                    if ridx.numel() > pos_scores.numel() * 4:
                        sel = torch.randperm(ridx.numel(), generator=g)[
                            :pos_scores.numel() * 4].to(device)
                        ridx, cidx = ridx[sel], cidx[sel]
                    neg_scores = S[ridx, cidx]
                    return pooled_auc(pos_scores.flatten(), neg_scores)

                auc_cluster = auc_with(HA)
                # marginal baseline on the SAME pairs construction
                Smarg = marg.unsqueeze(0).expand(Vtest.shape[0], -1)
                pos_scores = Smarg[hid]
                absent = ~pres & (mass_A > 0).unsqueeze(0)
                ridx, cidx = absent.nonzero(as_tuple=True)
                if ridx.numel() > pos_scores.numel() * 4:
                    sel = torch.randperm(ridx.numel(), generator=g)[
                        :pos_scores.numel() * 4].to(device)
                    ridx, cidx = ridx[sel], cidx[sel]
                auc_marginal = pooled_auc(pos_scores.flatten(), Smarg[ridx, cidx])
                # shuffled-membership null: permute columns of HA
                permc = torch.randperm(m_cols, generator=g).to(device)
                auc_shuffled = auc_with(HA[:, permc])

                # ---- dump most-stable clusters' top members ----------------
                perm, cos = greedy_match(HA, HB)
                stable_order = sorted(range(R), key=lambda i: -cos[i])[:3]
                dumps = {}
                top_cols_cpu = top_cols.cpu()
                for kf in stable_order:
                    topj = torch.argsort(HA[kf], descending=True)[:8].cpu()
                    mem = []
                    for j in topj.tolist():
                        c = int(top_cols_cpu[j])
                        s_i, lat = divmod(c, bank.d_sae)
                        mem.append((sites[s_i][0], sites[s_i][1], lat))
                    dumps[kf] = {"match_cos": round(cos[kf], 3), "members": mem}
                torch.save(dumps, CLUST_DIR / f"clusters_{sc}_{sl}.pt")

                rec = {"comp": sc, "latent": sl, "layer": seed_layer, "kind": seed_kind,
                       "m_cols": m_cols, "n_test_rows": int(ok_rows.sum()),
                       "auc_cluster": round(auc_cluster, 4) if auc_cluster else None,
                       "auc_marginal": round(auc_marginal, 4) if auc_marginal else None,
                       "auc_shuffled": round(auc_shuffled, 4) if auc_shuffled else None,
                       "lift": round(auc_cluster - auc_marginal, 4)
                               if auc_cluster and auc_marginal else None,
                       "stab_data_cluster_cos": round(clus_cos_data, 4),
                       "stab_data_latent_cos": round(lat_cos_data, 4) if lat_cos_data else None,
                       "stab_data_persist": round(persist_data, 4) if persist_data else None,
                       "stab_restart_persist": round(persist_rest, 4) if persist_rest else None,
                       "entropy_median": round(float(ent.median()), 4),
                       "one_cluster_frac": round(one_cluster, 4),
                       "jacc_within_seed": round(jacc_within, 4),
                       "secs": round(time.time() - t_seed, 1)}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind}: "
                      f"AUC clus/marg/shuf {rec['auc_cluster']}/{rec['auc_marginal']}/"
                      f"{rec['auc_shuffled']} lift {rec['lift']} | "
                      f"persist data/restart {rec['stab_data_persist']}/"
                      f"{rec['stab_restart_persist']} | ent {rec['entropy_median']} "
                      f"1clu {rec['one_cluster_frac']} | jaccWithin {rec['jacc_within_seed']} "
                      f"| {rec['secs']:.0f}s", flush=True)
                del Vfull, VA, VB, WA, HA, HB, HA2, Vtest
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
