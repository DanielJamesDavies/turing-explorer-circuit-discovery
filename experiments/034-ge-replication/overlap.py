"""TRIAMP-VS-ATTRIBUTION-RANKING OVERLAP (2026-08-11, analysis only —
NOT yet in the paper, per Daniel).

Question: how much of the weighted circuit sits outside the first-order
attribution top-n at matched size, and are those outside members the
ones carrying the ablated-regime reconstruction?

Per seed (the 22 matrix seeds):
1. Ungated all-site attribution (as ge_metric.py).
2. triamp400 refit (deterministic; returns members WITH alphas).
3. topn = top |attr| set at n = |triamp|.
4. Overlap stats: fraction of triamp inside topn; attr-mass share and
   median rank percentile of the outside members.
5. Held-out zero-fill scoring (panel semantics) of three circuits:
   triamp_full, triamp_inside (members in topn, own alphas),
   triamp_outside (members not in topn, own alphas).
   If inside-only collapses relative to full, the outside-ranking
   members carry the reconstruction.

  PYTHONPATH=src .venv/bin/python experiments/034-ge-replication/overlap.py
"""
import json
import os
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.learned_mask import run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
PANEL_ROWS = HERE.parent / "029-panel" / "rows.jsonl"
SMOKE = os.environ.get("SMOKE") == "1"
N_SEQ, N_TRAIN, EVAL_BS, D_SAE = 64, 48, 16, 40960

torch.set_float32_matmul_precision("high")
load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, D_SAE), device=bank.device)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.floor_source = "posctx"
cfg = disc.learned_mask

SEEDS = []
for line in PANEL_ROWS.open():
    r = json.loads(line)
    if r.get("arm") == "triamp400":
        SEEDS.append((r["comp_idx"], r["band"], r["latent"], r["n"]))
if SMOKE:
    SEEDS = SEEDS[:1]
print("seeds: %d" % len(SEEDS), flush=True)

ROWS_PATH = HERE / "rows_overlap.jsonl"
done = set()
if ROWS_PATH.exists():
    for line in ROWS_PATH.open():
        try:
            r = json.loads(line)
            done.add((r["comp_idx"], r["latent"]))
        except Exception:
            pass
fh = ROWS_PATH.open("a")


class AllSiteAttrInstrument:
    def __init__(self, sites, seed_site, w_seed, b_seed):
        self.sites = set(sites)
        self.seed_site = seed_site
        self.w_seed, self.b_seed = w_seed, b_seed
        self.seed_pre = None
        self.dense = {}

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        site = (layer_idx, kind)
        if site == self.seed_site:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        if site not in self.sites:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        dense.retain_grad()
        self.dense[site] = dense
        recon = bank.decode(dense, kind, layer_idx, add_bias=False)
        recon_ref = bank.decode(dense.detach(), kind, layer_idx, add_bias=False)
        err = (x - recon_ref).detach()
        return (recon + err).to(x.dtype)


class AmpCircuitPatcher:
    def __init__(self, alphas, seed_site, w_seed, b_seed):
        self.alphas = alphas
        self.seed_site = seed_site
        self.w_seed, self.b_seed = w_seed, b_seed
        self.seed_pre = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        al = self.alphas.get((layer_idx, kind))
        if al is None:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        code = torch.zeros_like(dense)
        if al:
            idx = torch.tensor(sorted(al), device=dense.device, dtype=torch.long)
            av = torch.tensor([al[int(i)] for i in idx], device=dense.device,
                              dtype=dense.dtype)
            code[..., idx] = dense[..., idx] * av
        out = bank.decode(code - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


def read(patcher, tokens, anchors):
    tot, n = 0.0, 0
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]
                patcher.seed_pre = None
                inference.forward(tk, patcher=patcher, grad_enabled=False,
                                  return_activations=False,
                                  tokenize_final=False)
                pre = patcher.seed_pre
                B = pre.shape[0]
                rr = torch.arange(B, device=pre.device)
                anc = anchors[s0:s0 + B].to(pre.device).clamp(
                    0, pre.shape[1] - 1)
                tot += float(torch.relu(pre[rr, anc]).sum()); n += B
    finally:
        inference.enable_compile()
    return tot / max(n, 1)


for comp_idx, band, sl, n_panel in SEEDS:
    if (comp_idx, sl) in done:
        print("[%s %d] done, skipping" % (band, sl), flush=True)
        continue
    layer, ki = split_component_idx(comp_idx, n_kinds)
    kind = bank.kinds[ki]
    m0 = _build_mode_method("counterfactual_gradient", "local", inference,
                            bank, avg_acts, pb)
    try:
        pd_ = m0.build_probe_dataset(comp_idx, sl)
    except Exception as e:
        print("[%s %d] probes FAILED: %s" % (band, sl, e), flush=True)
        del m0
        continue
    del m0
    if pd_ is None or int(pd_.pos_tokens.shape[0]) < N_SEQ:
        print("[%s %d] thin probes, skipping" % (band, sl), flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TRAIN], pa[:N_TRAIN], nt[:N_TRAIN]
    pt_ho, pa_ho = pt[N_TRAIN:], pa[N_TRAIN:]
    sae = bank.saes[kind][layer]
    w_seed = sae.encoder.weight[sl].detach()
    b_seed = sae._get_bias_eff()[sl].detach()
    UP = sorted(upstream_sites(bank, layer, kind))
    seed_site = (layer, kind)
    disc_bs = 2 if len(UP) >= 25 else 4
    a_pos_ho = float(measure_seed_activation(inference, bank, pt_ho, layer,
                                             kind, sl, pa_ho, batch_size=EVAL_BS))
    e0_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho,
                                          layer, kind, sl, pos_argmax=pa_ho,
                                          batch_size=EVAL_BS))

    # 1. ungated attribution
    attrs = {s: torch.zeros(D_SAE) for s in UP}
    inference.disable_compile()
    try:
        for s0 in range(0, int(pt_tr.shape[0]), disc_bs):
            tk = pt_tr[s0:s0 + disc_bs]
            paw = pa_tr[s0:s0 + disc_bs]
            inst = AllSiteAttrInstrument(UP, seed_site, w_seed, b_seed)
            with torch.enable_grad():
                inference.forward(tk, patcher=inst, grad_enabled=True,
                                  return_activations=False,
                                  tokenize_final=False)
                pre = inst.seed_pre
                B = pre.shape[0]
                rr = torch.arange(B, device=pre.device)
                anc = paw.to(pre.device).clamp(0, pre.shape[1] - 1)
                metric = pre[rr, anc].mean()
                metric.backward()
            for s, d in inst.dense.items():
                if d.grad is not None:
                    attrs[s] += (d.detach() * d.grad).sum(dim=(0, 1)).float().cpu()
            del inst
    finally:
        inference.enable_compile()
        torch.cuda.empty_cache()

    # 2. triamp refit with alphas
    triple_w = 0.10 if band.startswith(("L2", "L3", "L5")) else 0.05
    kw = dict(sites=UP, seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
              pos_tokens=pt_tr, pos_argmax=pa_tr, neg_tokens=nt_tr,
              mask_floor_source="triple", dual_floor_weight=cfg.dual_floor_weight,
              triple_floor_weight=triple_w, free_amplitude=True,
              steps=int(cfg.steps), lr=cfg.lr, keep_threshold=cfg.keep_threshold,
              batch_size=disc.probe_batch_size, holdout_frac=cfg.holdout_frac,
              log_every=0, deep_site_threshold=cfg.deep_site_threshold,
              deep_batch_size=cfg.deep_batch_size, optimizer=cfg.optimizer,
              weight_decay=cfg.weight_decay, code_dtype=cfg.code_dtype,
              lr_schedule=cfg.lr_schedule, lr_min_frac=cfg.lr_min_frac,
              warmup_frac=cfg.warmup_frac, l1_lambda=1e-3,
              binarize=cfg.binarize, theta_init=cfg.theta_init)
    scores, prov = run_learned_mask(inference, bank, objective="pos", **kw)
    ak = prov.get("amp_kept") or {}
    alphas = {}
    for k, dd in ak.items():
        lyr, knd = k.split("/")
        alphas[(int(lyr), knd)] = {int(i): float(v) for i, v in dd.items()}
    n_tri = sum(len(d) for d in alphas.values())
    if n_tri == 0:
        print("[%s %d] refit returned empty membership, skipping"
              % (band, sl), flush=True)
        continue

    # 3. topn at matched size + rank structure
    flat = []
    for (l, kd), vec in attrs.items():
        a = vec.abs()
        nz = a.nonzero(as_tuple=True)[0]
        flat.extend((float(a[i]), l, kd, int(i)) for i in nz.tolist())
    flat.sort(reverse=True)
    topn = {(l, kd, i) for _, l, kd, i in flat[:n_tri]}
    rank_of = {(l, kd, i): r for r, (_, l, kd, i) in enumerate(flat)}
    n_ranked = max(len(flat), 1)
    tri_members = {(l, kd, i) for (l, kd), d in alphas.items() for i in d}
    inside = tri_members & topn
    outside = tri_members - topn
    out_ranks = [rank_of.get(m, n_ranked) / n_ranked for m in outside]
    out_ranks.sort()
    tot_abs = sum(a for a, *_ in flat) or 1.0
    mass_in = sum(a for a, l, kd, i in flat if (l, kd, i) in inside)
    mass_out = sum(a for a, l, kd, i in flat if (l, kd, i) in outside)

    # 4. score full / inside / outside under held-out zero fill
    def sub_alphas(members):
        out = {}
        for (l, kd), d in alphas.items():
            keep = {i: a for i, a in d.items() if (l, kd, i) in members}
            if keep:
                out[(l, kd)] = keep
        return out

    def f0(al):
        aw = read(AmpCircuitPatcher(al, seed_site, w_seed, b_seed), pt_ho, pa_ho)
        return (round((aw - e0_ho) / (a_pos_ho - e0_ho), 4)
                if abs(a_pos_ho - e0_ho) > 1e-9 else None)

    row = {"comp_idx": comp_idx, "band": band, "latent": sl,
           "n_tri": n_tri, "n_panel": n_panel,
           "frac_inside_topn": round(len(inside) / n_tri, 4),
           "n_inside": len(inside), "n_outside": len(outside),
           "mass_share_inside": round(mass_in / tot_abs, 4),
           "mass_share_outside": round(mass_out / tot_abs, 4),
           "outside_rank_pctl_med": (round(out_ranks[len(out_ranks) // 2], 4)
                                     if out_ranks else None),
           "F0_full": f0(alphas),
           "F0_inside": f0(sub_alphas(inside)),
           "F0_outside": f0(sub_alphas(outside)),
           "a_pos_ho": round(a_pos_ho, 3)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("  [%s %d] n=%d inside=%.2f | F0 full=%s inside=%s outside=%s"
          % (band, sl, n_tri, row["frac_inside_topn"], row["F0_full"],
             row["F0_inside"], row["F0_outside"]), flush=True)
    torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
