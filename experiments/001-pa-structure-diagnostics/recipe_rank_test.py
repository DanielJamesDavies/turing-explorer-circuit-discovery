"""Fuzzy-recipe / NMF rank test (PA structure, test 2 of 2) — go/no-go on the
recipes hypothesis: are per-position attributions LOW-RANK (r recipes deployed
at instance-specific positions) or heterogeneous (the union vindicated)?

Per seed (16 layer-stratified):
  CAPTURE   one act-grad-signal pass (grad x live value, drive objective) via
            MaskedRestorationInstrument with ALL-RESTORED masks (= live
            connected stream, gradient leaves at every site). Per (seq, pos,
            site) keep top-128 by |attr| -> sparse entries over
            (site, latent, sign) split-channel columns (both-sign finding:
            inhibitory ingredients stay distinct columns).
  MATRIX    rows = (seq, pos), L1-normalized (composition, not magnitude);
            columns capped to top-24k by total mass (coverage reported).
  NMF       GPU multiplicative updates, ranks {4, 8, 16, 32}, 80 iters;
            R^2 = 1 - ||V-WH||^2 / ||V||^2. NULL: same values, random column
            ids per entry (preserves row mass, destroys co-occurrence).
            Structure score = mean over ranks of (R2_real - R2_null).
  ENTROPY   at r=16: per-factor usage entropy over SEQUENCES (normalized) —
            low-entropy factors are document memorization, not position-types.
  FUNCTION  free0 of memberships at matched sizes {2k, 8k, 24k(all)}:
            recipe-ranked (max factor weight, r=16) vs magnitude-ranked
            (total |mass|) from the SAME capture. Union-collapsed keeps
            (per-position soft masks are phase 2).

Rows -> recipe_rank.jsonl; top-50 latents per factor (r=16) -> factors/.
PYTHONPATH=src python recipe_rank_test.py   (repo root, wsl + .venv)
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
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites,
)
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
N_SEEDS = 16
TOPK_CAPTURE = 128          # per (seq, pos, site)
COL_CAP = 24_000            # columns kept by total mass
RANKS = (4, 8, 16, 32)
R_STAR = 16                 # rank for entropy/function/factor dump
NMF_ITERS = 80
CHUNK_B = 8
EVAL_BS = 16
SIZES = (2_000, 8_000)
OUT = Path(__file__).parent / "recipe_rank.jsonl"
FACT_DIR = Path(__file__).parent / "factors"
FACT_DIR.mkdir(exist_ok=True)

torch.set_float32_matmul_precision("high")


def _layer_stratified_indices(candidates, sample_size, n_kinds, seed=42):
    by_layer = defaultdict(list)
    for index, cand in enumerate(candidates):
        by_layer[int(cand["comp_idx"]) // n_kinds].append(index)
    rng = random.Random(seed)
    for layer in by_layer:
        rng.shuffle(by_layer[layer])
    selected = []
    max_len = max(len(v) for v in by_layer.values())
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
    err = torch.linalg.norm(V - W @ H) ** 2
    R2 = float(1.0 - err / (torch.linalg.norm(V) ** 2))
    return W, H, R2


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
idxs = _layer_stratified_indices(all_cand, N_SEEDS, n_kinds)
cands = [_candidate_with_index(all_cand[i], i) for i in idxs]
print(f"sampled {len(cands)} seeds -> {OUT}", flush=True)

original = _apply_sweep_config(max_per_site=24)
disc = config.discovery
saved = (disc.probe_sequence_count, disc.eval_sequence_count, disc.eval_batch_size)
disc.probe_sequence_count = 64
disc.eval_sequence_count = 64
disc.eval_batch_size = EVAL_BS

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
                    print(f"[{si+1}] {sc}/{sl} L{seed_layer}: no pos — skip", flush=True)
                    continue
                sites = sorted(upstream_sites(bank, seed_layer, seed_kind))
                if not sites:
                    print(f"[{si+1}] {sc}/{sl}: no upstream — skip", flush=True)
                    continue
                pt, pa = pd.pos_tokens[:64], pd.pos_argmax[:64]
                B_total, T = pt.shape[0], pt.shape[1]
                n_rows_all = B_total * T

                sae = bank.saes[seed_kind][seed_layer]
                w_seed = sae.encoder.weight[sl].detach()
                b_seed = sae._get_bias_eff()[sl].detach()
                _, residuals = collect_natural_codes(inference, bank, pt, set(sites))
                floors0 = {s: torch.zeros(bank.d_sae) for s in sites}
                masks1 = {s: torch.ones(bank.d_sae, dtype=torch.bool) for s in sites}
                site_index = {s: i for i, s in enumerate(sites)}

                # ---- CAPTURE (vectorized): flat COO entry lists ------------
                ent_rows, ent_cols, ent_vals = [], [], []   # GPU flat tensors
                for start in range(0, B_total, CHUNK_B):
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
                    idx = torch.arange(Bc, device=pre.device)
                    pac = pa[start:start + Bc].to(pre.device).clamp(0, pre.shape[1] - 1)
                    peak = pre[idx, pac]
                    order = sorted(inst.leaves)
                    grads = torch.autograd.grad(peak.mean(), [inst.leaves[s] for s in order],
                                                allow_unused=True)
                    row_base = (torch.arange(Bc, device=device) + start).view(-1, 1) * T \
                        + torch.arange(T, device=device).view(1, -1)      # [Bc,T]
                    for s, gr in zip(order, grads):
                        if gr is None:
                            continue
                        attr = gr.to(torch.float32) * inst.leaves[s].detach().to(torch.float32)
                        k = min(TOPK_CAPTURE, attr.shape[-1])
                        vals, lats = attr.abs().topk(k, dim=-1)          # [Bc,T,k]
                        signs = (torch.gather(attr, -1, lats) < 0).long()
                        base = site_index[s] * bank.d_sae * 2
                        cols = base + lats * 2 + signs                   # [Bc,T,k]
                        rows = row_base.unsqueeze(-1).expand_as(cols)
                        nz = vals > 0
                        ent_rows.append(rows[nz])
                        ent_cols.append(cols[nz])
                        ent_vals.append(vals[nz])
                    del inst, grads
                rows_f = torch.cat(ent_rows)
                cols_f = torch.cat(ent_cols)
                vals_f = torch.cat(ent_vals)
                del ent_rows, ent_cols, ent_vals
                n_slots = len(sites) * bank.d_sae * 2

                # ---- column cap by total mass (vectorized) -----------------
                mass = torch.zeros(n_slots, device=device)
                mass.index_add_(0, cols_f, vals_f)
                total_mass = float(mass.sum())
                m_cols = min(COL_CAP, int((mass > 0).sum()))
                top_mass, kept_cols = mass.topk(m_cols)
                coverage = float(top_mass.sum()) / max(total_mass, 1e-12)
                col_map = torch.full((n_slots,), -1, dtype=torch.long, device=device)
                col_map[kept_cols] = torch.arange(m_cols, device=device)
                keep = col_map[cols_f] >= 0
                rows_k, cols_k, vals_k = rows_f[keep], col_map[cols_f[keep]], vals_f[keep]
                del rows_f, cols_f, vals_f

                def dense_from(rows_t, cols_t, vals_t):
                    Vd = torch.zeros(n_rows_all * m_cols, device=device)
                    Vd.index_add_(0, rows_t * m_cols + cols_t, vals_t)
                    Vd = Vd.view(n_rows_all, m_cols)
                    row_mass = Vd.sum(dim=1)
                    live = row_mass > 0
                    Vd = Vd[live] / row_mass[live].unsqueeze(1)
                    return Vd, live

                V, live_rows = dense_from(rows_k, cols_k, vals_k)
                n_rows = V.shape[0]
                row_seq = (torch.arange(n_rows_all, device=device) // T)[live_rows]

                # ---- NMF sweep: real vs null -------------------------------
                r2_real, r2_null = {}, {}
                W_star = H_star = None
                for r in RANKS:
                    W, H, R2 = nmf(V, r, seed=0)
                    r2_real[r] = round(R2, 4)
                    if r == R_STAR:
                        W_star, H_star = W.clone(), H.clone()
                    del W, H
                g = torch.Generator(device="cpu").manual_seed(1)
                cols_null = torch.randint(0, m_cols, (cols_k.numel(),), generator=g).to(device)
                Vn, _ = dense_from(rows_k, cols_null, vals_k)
                for r in RANKS:
                    _, _, R2 = nmf(Vn, r, seed=0)
                    r2_null[r] = round(R2, 4)
                del Vn
                structure = round(sum(r2_real[r] - r2_null[r] for r in RANKS) / len(RANKS), 4)

                # ---- factor sequence-entropy at r* -------------------------
                usage = torch.zeros(R_STAR, B_total, device=device)
                usage.index_add_(1, row_seq, W_star.T.to(device))
                p = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-12)
                ent = (-(p * (p + 1e-12).log()).sum(dim=1)
                       / torch.log(torch.tensor(float(B_total)))).cpu()

                # ---- functional: recipe- vs magnitude-ranked free0 ---------
                a_pos = measure_seed_activation(inference, bank, pt, seed_layer,
                                                seed_kind, sl, pa, batch_size=EVAL_BS)
                a_e0 = circuit_only_activation(inference, bank, {}, set(sites), pt,
                                               seed_layer, seed_kind, sl, pos_argmax=pa,
                                               batch_size=EVAL_BS)
                den = a_pos - a_e0
                kept_cols_cpu = kept_cols.cpu().tolist()

                def col_latent(ci):
                    c = kept_cols_cpu[ci]
                    s_i, rem = divmod(c, bank.d_sae * 2)
                    return (sites[s_i][0], sites[s_i][1], rem // 2)

                def phi_of(triples):
                    if abs(den) < 1e-9:
                        return None
                    kd = {}
                    for (L, kind, lat) in triples:
                        kd.setdefault((L, kind), set()).add(lat)
                    a_c = circuit_only_activation(inference, bank, kd, set(sites), pt,
                                                  seed_layer, seed_kind, sl, pos_argmax=pa,
                                                  batch_size=EVAL_BS)
                    return round(float((a_c - a_e0) / den), 4)

                func = {}
                rankings = {"mag": top_mass, "recipe": H_star.max(dim=0).values}
                for label, ranking in rankings.items():
                    order = torch.argsort(ranking, descending=True).cpu().tolist()
                    seen, uniq = set(), []
                    for ci in order:
                        trip = col_latent(ci)
                        if trip not in seen:
                            seen.add(trip)
                            uniq.append(trip)
                    for S in SIZES:
                        func[f"free0_{label}_{S}"] = phi_of(uniq[:S])
                all_trips = {col_latent(ci) for ci in range(m_cols)}
                func["free0_capture_union"] = phi_of(all_trips)
                func["n_capture_union"] = len(all_trips)

                tops = {}
                for kf in range(R_STAR):
                    top = torch.argsort(H_star[kf], descending=True)[:50].cpu().tolist()
                    tops[kf] = [col_latent(ci) + ("neg" if kept_cols_cpu[ci] % 2 else "pos",)
                                for ci in top]
                torch.save(tops, FACT_DIR / f"factors_{sc}_{sl}.pt")

                rec = {"comp": sc, "latent": sl, "layer": seed_layer, "kind": seed_kind,
                       "n_rows": n_rows, "n_cols": m_cols, "coverage": round(coverage, 4),
                       "r2_real": r2_real, "r2_null": r2_null, "structure": structure,
                       "entropy_median": round(float(ent.median()), 4),
                       "entropy_min": round(float(ent.min()), 4),
                       **func, "secs": round(time.time() - t_seed, 1)}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[{si+1}/{len(cands)}] {sc}/{sl} L{seed_layer} {seed_kind} | "
                      f"struct={structure} r2_real={r2_real} | cov={rec['coverage']} | "
                      f"ent med/min {rec['entropy_median']}/{rec['entropy_min']} | "
                      f"free0 mag/rec@8k {func.get('free0_mag_8000')}/"
                      f"{func.get('free0_recipe_8000')} | {rec['secs']:.0f}s", flush=True)
                del V, W_star, H_star, rows_k, cols_k, vals_k
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
