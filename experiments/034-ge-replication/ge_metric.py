"""GE-METRIC EVALUATOR (2026-08-11) — score memberships under the
intervention-free faithfulness surrogate of Ge et al. 2024.

Their eval (Theorem 3.1): in the linearised graph, the root's value
equals the sum of leaf attribution scores, so circuit faithfulness =
attribution recovery of the kept subgraph, no intervention. Exact in
their substitute model; on our stack the first-order analogue is
ATTRIBUTION-MASS RECOVERY: with attr(v) = a_v * d(seed preact)/d a_v at
the clean linearisation (their attribution, ungated), report

    recovery_signed = sum_{v in C} attr(v) / sum_{v} attr(v)
    recovery_abs    = sum_{v in C} |attr(v)| / sum_{v} |attr(v)|

for each membership C. One forward+backward per probe chunk scores all
sites at once (single linearisation = their standard-attribution frame).

Memberships scored per seed (the 22 matrix seeds):
  ge_hier   — from members_ge_hier_<comp>_<latent>.json (the replication run)
  topn_attr — top n_target latents by |attr| (Ge's "standard attribution"
              comparison arm, at matched size)
  triamp400 — the weighted circuit's membership, recovered by
              deterministic refit with the exact panel recipe (the
              pipeline is bit-deterministic at fixed settings; refit n is
              logged against the panel row's n as a check)

Run AFTER the replication run releases the GPU:
  PYTHONPATH=src .venv/bin/python experiments/034-ge-replication/ge_metric.py
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
from eval.ablation_faithfulness import upstream_sites
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
N_SEQ, N_TRAIN, D_SAE = 64, 48, 40960
DISC_BS = 8

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

ROWS_PATH = HERE / "rows_gemetric.jsonl"
done = set()
if ROWS_PATH.exists():
    for line in ROWS_PATH.open():
        try:
            r = json.loads(line)
            done.add((r["comp_idx"], r["latent"], r["membership"]))
        except Exception:
            pass
fh = ROWS_PATH.open("a")


class AllSiteAttrInstrument:
    """Ungated single-linearisation attribution: rewrite every upstream
    site's stream as decode(code) + detached error (errors detached, as
    Ge prescribes), retain grad on every site's dense code, tap the seed
    pre-activation."""

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


def all_site_attrs(UP, seed_site, w_seed, b_seed, pt_tr, pa_tr):
    attrs = {s: torch.zeros(D_SAE) for s in UP}
    inference.disable_compile()
    try:
        for s0 in range(0, int(pt_tr.shape[0]), DISC_BS):
            tk = pt_tr[s0:s0 + DISC_BS]
            pa = pa_tr[s0:s0 + DISC_BS]
            inst = AllSiteAttrInstrument(UP, seed_site, w_seed, b_seed)
            with torch.enable_grad():
                inference.forward(tk, patcher=inst, grad_enabled=True,
                                  return_activations=False,
                                  tokenize_final=False)
                pre = inst.seed_pre
                B = pre.shape[0]
                rr = torch.arange(B, device=pre.device)
                anc = pa.to(pre.device).clamp(0, pre.shape[1] - 1)
                metric = pre[rr, anc].mean()
                metric.backward()
            for s, d in inst.dense.items():
                if d.grad is not None:
                    attrs[s] += (d.detach() * d.grad).sum(dim=(0, 1)).float().cpu()
            del inst
    finally:
        inference.enable_compile()
        torch.cuda.empty_cache()
    return attrs


def refit_triamp(comp_idx, band, sl, layer, kind, UP, pt_tr, pa_tr, nt_tr):
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
    members = set()
    for k, dd in ak.items():
        lyr, knd = k.split("/")
        for i in dd:
            members.add((int(lyr), knd, int(i)))
    if not members:
        for fid in scores:
            members.add((fid.layer, fid.kind, int(fid.index)))
    return members


def recovery(attrs, members):
    tot_s, tot_a, kept_s, kept_a = 0.0, 0.0, 0.0, 0.0
    for (l, kd), vec in attrs.items():
        tot_s += float(vec.sum()); tot_a += float(vec.abs().sum())
        idx = [i for (ll, kk, i) in members if (ll, kk) == (l, kd)]
        if idx:
            t = torch.tensor(idx, dtype=torch.long)
            kept_s += float(vec[t].sum()); kept_a += float(vec[t].abs().sum())
    return (kept_s / tot_s if abs(tot_s) > 1e-12 else None,
            kept_a / tot_a if tot_a > 1e-12 else None)


for comp_idx, band, sl, n_target in SEEDS:
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
    sae = bank.saes[kind][layer]
    w_seed = sae.encoder.weight[sl].detach()
    b_seed = sae._get_bias_eff()[sl].detach()
    UP = sorted(upstream_sites(bank, layer, kind))
    seed_site = (layer, kind)

    attrs = all_site_attrs(UP, seed_site, w_seed, b_seed, pt_tr, pa_tr)

    memberships = {}
    mj = HERE / ("members_ge_hier_%d_%d.json" % (comp_idx, sl))
    if mj.exists():
        memberships["ge_hier"] = {(l, kd, int(i))
                                  for l, kd, i in json.loads(mj.read_text())}
    flat = []
    for (l, kd), vec in attrs.items():
        a = vec.abs()
        nz = a.nonzero(as_tuple=True)[0]
        flat.extend((float(a[i]), l, kd, int(i)) for i in nz.tolist())
    flat.sort(reverse=True)
    memberships["topn_attr"] = {(l, kd, i) for _, l, kd, i in flat[:n_target]}
    try:
        memberships["triamp400"] = refit_triamp(comp_idx, band, sl, layer,
                                                kind, UP, pt_tr, pa_tr, nt_tr)
    except Exception as e:
        print("[%s %d] triamp refit failed: %s" % (band, sl, e), flush=True)

    for name, members in memberships.items():
        if (comp_idx, sl, name) in done:
            continue
        rs, ra = recovery(attrs, members)
        row = {"comp_idx": comp_idx, "band": band, "latent": sl,
               "membership": name, "n": len(members),
               "n_panel": n_target if name == "triamp400" else None,
               "recovery_signed": round(rs, 4) if rs is not None else None,
               "recovery_abs": round(ra, 4) if ra is not None else None}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  [%s %d] %-10s n=%-6d rec_signed=%-8s rec_abs=%-8s"
              % (band, sl, name, len(members), row["recovery_signed"],
                 row["recovery_abs"]), flush=True)
    torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
