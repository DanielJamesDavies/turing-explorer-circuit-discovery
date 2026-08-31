"""The no-zero floor control (Daniel, 2026-08-06): tri-amp trained under
posctx + negctx ONLY ("pn" floor — negctx promoted to the primary 1.0
slot, posctx at the triple weight, NO zero term). The question: with
free amplitudes, does the zero term still earn its place, or can
alpha->0 substitute for it?

Prediction from R1 (negctx-only masks learn the delta from baseline,
free0 = 0 at depth): ampF0 collapses while ampFMd holds. If instead
ampF0 survives, the zero term is redundant under amp semantics and the
floor story needs revising.

Same seeds, same decision-cell weights, same scoring as the joint400
rows in amp_stepsweep_c8/c29.jsonl — rows directly comparable.

  COMP_IDX=8  PYTHONPATH=src python experiments/026-floor-isolation/amp_pnfloor.py
  COMP_IDX=29 ...
"""
import json
import os
import random
import time
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
from eval.floors import collect_site_anchors, collect_site_means
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
COMP_IDX = int(os.environ.get("COMP_IDX", 8))
N_SEEDS = int(os.environ.get("N_SEEDS", 3))

N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
# the decision cell per depth (smallest reliable weighted circuits)
CELL = (("pn", 0.10, 1e-3) if COMP_IDX == 8
        else ("pn", 0.05, 1e-3))
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

LAYER, KI = split_component_idx(COMP_IDX, n_kinds)
KIND = bank.kinds[KI]
UP = sorted(upstream_sites(bank, LAYER, KIND))

_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu",
                   weights_only=False)
_pool = [int(c["latent_idx"]) for c in _cand if int(c["comp_idx"]) == COMP_IDX]
random.Random(42).shuffle(_pool)
SEEDS = sorted(_pool[:32])
del _cand
if COMP_IDX == 29 and 1639 in SEEDS:
    SEEDS = [s for s in SEEDS if s != 1639]
SEEDS = SEEDS[:N_SEEDS]
print("L%d %s | seeds %s | cell %s" % (LAYER, KIND, SEEDS, (CELL,)), flush=True)


class AmpCircuitPatcher:
    """Kept latents at alpha * live value, everything else at floor
    (zero when floors is None). Verbatim semantics from amp_null.py."""

    def __init__(self, alphas, floors, seed_site, w_seed, b_seed):
        self.alphas, self.floors = alphas, floors or {}
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
        fl = self.floors.get((layer_idx, kind))
        code = (fl.to(device=dense.device, dtype=dense.dtype)
                .expand_as(dense).clone() if fl is not None
                else torch.zeros_like(dense))
        if al:
            idx = torch.tensor(sorted(al), device=dense.device, dtype=torch.long)
            av = torch.tensor([al[int(i)] for i in idx], device=dense.device,
                              dtype=dense.dtype)
            code[..., idx] = dense[..., idx] * av
        out = bank.decode(code - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


class AmpInjectPatcher:
    """cf under amp semantics: kept latents SET to alpha_i * pin_i in the
    otherwise-LIVE stream. Verbatim semantics from amp_cfsup.py."""

    def __init__(self, inject, seed_site, w_seed, b_seed):
        self.inject = inject          # {site: {idx: value}} value = alpha*pin
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
        inj = self.inject.get((layer_idx, kind))
        if not inj:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        code = dense.clone()
        idx = torch.tensor(sorted(inj), device=dense.device, dtype=torch.long)
        vals = torch.tensor([inj[int(i)] for i in idx], device=dense.device,
                            dtype=dense.dtype)
        code[..., idx] = vals
        out = bank.decode(code - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


fh = (HERE / ("amp_pnfloor_c%d.jsonl" % COMP_IDX)).open("a")
for sl in SEEDS:
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, pb)
    pd_ = m0.build_probe_dataset(COMP_IDX, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    sae = bank.saes[KIND][LAYER]
    w_seed = sae.encoder.weight[sl].detach()
    b_seed = sae._get_bias_eff()[sl].detach()
    a_pos = float(measure_seed_activation(inference, bank, pt, LAYER, KIND, sl,
                                          pa, batch_size=EVAL_BS))
    means = collect_site_means(inference, bank, pt, set(UP))
    e0 = float(circuit_only_activation(inference, bank, {}, UP, pt, LAYER,
                                       KIND, sl, pos_argmax=pa,
                                       batch_size=EVAL_BS))
    eMd = float(circuit_only_activation(inference, bank, {}, UP, pt, LAYER,
                                        KIND, sl, pos_argmax=pa,
                                        site_means=means, batch_size=EVAL_BS))
    _, pins = collect_site_anchors(inference, bank, pt, set(UP), pa,
                                   pin_position_specific=False)

    # negctx anchors + baseline (seed pre-act argmax on unmodified negctx)
    p0 = AmpInjectPatcher({}, (LAYER, KIND), w_seed, b_seed)
    pre_chunks = []
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(nt.shape[0]), EVAL_BS):
                p0.seed_pre = None
                inference.forward(nt[s0:s0 + EVAL_BS], patcher=p0,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
                pre_chunks.append(p0.seed_pre.detach())
    finally:
        inference.enable_compile()
    neg_pre = torch.cat(pre_chunks, 0)
    na = neg_pre.argmax(dim=1).cpu()
    rows_i = torch.arange(neg_pre.shape[0], device=neg_pre.device)
    a_base = float(torch.relu(neg_pre[rows_i, na.to(neg_pre.device)]).mean())
    den_cf = a_pos - a_base
    print("\n[%d] a_pos %.3f | e0 %.3f | a_base %.3f" % (sl, a_pos, e0, a_base),
          flush=True)

    def amp_act(alphas, floors):
        p = AmpCircuitPatcher(alphas, floors, (LAYER, KIND), w_seed, b_seed)
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(pt.shape[0]), EVAL_BS):
                tk = pt[s0:s0 + EVAL_BS]
                p.seed_pre = None
                inference.forward(tk, patcher=p, grad_enabled=False,
                                  return_activations=False, tokenize_final=False)
                pre = p.seed_pre
                B = min(pre.shape[0], pa[s0:s0 + EVAL_BS].shape[0])
                rows = torch.arange(B, device=pre.device)
                anc = pa[s0:s0 + EVAL_BS][:B].to(pre.device).clamp(0, pre.shape[1] - 1)
                tot += float(torch.relu(pre[:B][rows, anc]).sum()); n += B
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    def amp_cf(alphas):
        inject = {}
        for s, d in alphas.items():
            pv = pins.get(s)
            if pv is None:
                continue
            inject[s] = {i: float(a * float(pv[i])) for i, a in d.items()}
        p = AmpInjectPatcher(inject, (LAYER, KIND), w_seed, b_seed)
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(nt.shape[0]), EVAL_BS):
                tk = nt[s0:s0 + EVAL_BS]
                p.seed_pre = None
                inference.forward(tk, patcher=p, grad_enabled=False,
                                  return_activations=False, tokenize_final=False)
                pre = p.seed_pre
                B = pre.shape[0]
                rr = torch.arange(B, device=pre.device)
                anc = na[s0:s0 + B].to(pre.device).clamp(0, pre.shape[1] - 1)
                tot += float(torch.relu(pre[rr, anc]).sum()); n += B
        finally:
            inference.enable_compile()
        return ((tot / max(n, 1)) - a_base) / den_cf if abs(den_cf) > 1e-9 else None

    def discover(members=None, steps=None):
        floor, tw, lam = CELL
        kw = dict(sites=UP, seed_layer=LAYER, seed_kind=KIND,
                  seed_latent_idx=sl, pos_tokens=pt, pos_argmax=pa,
                  neg_tokens=nt, mask_floor_source=floor,
                  dual_floor_weight=cfg.dual_floor_weight,
                  triple_floor_weight=tw, free_amplitude=True,
                  steps=int(steps or cfg.steps), lr=cfg.lr,
                  keep_threshold=cfg.keep_threshold,
                  batch_size=disc.probe_batch_size,
                  holdout_frac=cfg.holdout_frac, log_every=0,
                  deep_site_threshold=cfg.deep_site_threshold,
                  deep_batch_size=cfg.deep_batch_size,
                  optimizer=cfg.optimizer, weight_decay=cfg.weight_decay,
                  code_dtype=cfg.code_dtype, lr_schedule=cfg.lr_schedule,
                  lr_min_frac=cfg.lr_min_frac, warmup_frac=cfg.warmup_frac)
        if members is None:
            kw.update(l1_lambda=lam, binarize=cfg.binarize,
                      theta_init=cfg.theta_init)
        else:
            support = {}
            for s, i in members:
                support.setdefault(s, []).append(i)
            kw.update(l1_lambda=0.0, binarize="none", theta_init=40.0,
                      support={s: torch.tensor(v, dtype=torch.long)
                               for s, v in support.items()})
        _, prov = run_learned_mask(inference, bank, objective="pos", **kw)
        ak = prov.get("amp_kept") or {}
        alphas = {}
        for k, d in ak.items():
            lyr, knd = k.split("/")
            alphas[(int(lyr), knd)] = {int(i): float(v) for i, v in d.items()}
        return alphas, prov.get("amp_stats") or {}

    def score(alphas, st, tag, secs):
        aw0 = amp_act(alphas, None)
        awM = amp_act(alphas, means)
        cfa = amp_cf(alphas)
        row = {"latent": sl, "arm": tag,
               "n": sum(len(d) for d in alphas.values()),
               "ampF0": round((aw0 - e0) / (a_pos - e0), 4),
               "ampFMd": (round((awM - eMd) / (a_pos - eMd), 4)
                          if abs(a_pos - eMd) > 1e-9 else None),
               "cf_amp": round(cfa, 4) if cfa is not None else None,
               "alpha_med": st.get("median"), "alpha_p90": st.get("p90"),
               "alpha_max": st.get("max"), "secs": round(secs, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-11s n=%-6d ampF0=%-8s ampFMd=%-8s cf_amp=%-8s a_p90=%s"
              % (tag, row["n"], row["ampF0"], row["ampFMd"], row["cf_amp"],
                 row["alpha_p90"]), flush=True)

    t0 = time.time()
    a1, st1 = discover()
    score(a1, st1, "pn%d" % cfg.steps, time.time() - t0)
    torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
