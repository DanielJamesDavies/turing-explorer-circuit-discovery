"""cf and sup for tri-amp weighted circuits — the last unvalidated panel.

sup needs no new semantics (necessity is removal; deletion does not
involve the coefficients), so it is computed with the standard evaluator
on the membership set.

cf gets THREE readings per circuit, the comparison Daniel asked for:

  cf_bare   the STANDARD eval, unchanged: evaluate_counterfactual_
            faithfulness on the membership (its own pin/alpha-bisection
            machinery, coefficients ignored). "How does the set do under
            the normal exam?"
  cf_amp    inject alpha_i * pin_i into the LIVE negctx stream — the
            amplitudes exactly as trained, NO refit on the negatives.
            Held-out by construction: the alphas were fitted to
            reproduce the seed on POSITIVE contexts under ablated
            backgrounds; driving it in NEGATIVE contexts through an
            intact stream is a task the fit never saw.
  null      the same cf_amp for RANDOM same-size sets with their own
            identically-fitted alphas — extends the R16 null to the
            drive axis (reconstruction and drive could dissociate).

cf_amp normalisation matches the cf convention: (a_intervened - a_base)
/ (a_pos - a_base), a_base = seed on unmodified negctx, read at the
would-be-firing anchor (negctx pre-act argmax).

  COMP_IDX=8  PYTHONPATH=src python experiments/026-floor-isolation/amp_cfsup.py
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
    measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense, target_latent_activations
from store.circuits import Circuit, CircuitNode

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP_IDX = int(os.environ.get("COMP_IDX", 8))
N_SEEDS = int(os.environ.get("N_SEEDS", 3))
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
# the decision cell per depth (smallest reliable weighted circuits)
CELL = (("triple", 0.10, 1e-3) if COMP_IDX == 8
        else ("triple", 0.05, 1e-3))
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
if COMP_IDX == 29 and 1639 in SEEDS:
    SEEDS = [s for s in SEEDS if s != 1639]
SEEDS = SEEDS[:N_SEEDS]
print("L%d %s | seeds %s | cell %s" % (LAYER, KIND, SEEDS, (CELL,)), flush=True)


class AmpInjectPatcher:
    """cf under amp semantics: kept latents SET to alpha_i * pin_i in the
    otherwise-LIVE stream (injection into an intact context, the cf
    convention — nothing else is floored or removed). Captures the seed's
    pre-activation for the anchored read."""

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


fh = (HERE / ("amp_cfsup_c%d.jsonl" % COMP_IDX)).open("a")
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
    a_base = float(torch.relu(
        neg_pre[rows_i, na.to(neg_pre.device)]).mean())
    den_cf = a_pos - a_base
    print("\n[%d] a_pos %.3f | a_base(negctx) %.3f" % (sl, a_pos, a_base),
          flush=True)

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

    def discover(members=None):
        """Tri-amp discovery (members=None) or amplitude-only fit on a
        fixed support (the null)."""
        floor, tw, lam = CELL
        kw = dict(sites=UP, seed_layer=LAYER, seed_kind=KIND,
                  seed_latent_idx=sl, pos_tokens=pt, pos_argmax=pa,
                  neg_tokens=nt, mask_floor_source=floor,
                  dual_floor_weight=cfg.dual_floor_weight,
                  triple_floor_weight=tw, free_amplitude=True,
                  steps=cfg.steps, lr=cfg.lr,
                  keep_threshold=cfg.keep_threshold,
                  batch_size=disc.probe_batch_size,
                  holdout_frac=cfg.holdout_frac, log_every=0,
                  deep_site_threshold=cfg.deep_site_threshold,
                  deep_batch_size=cfg.deep_batch_size,
                  optimizer=cfg.optimizer, weight_decay=cfg.weight_decay,
                  code_dtype=cfg.code_dtype, lr_schedule=cfg.lr_schedule,
                  lr_min_frac=cfg.lr_min_frac, warmup_frac=cfg.warmup_frac)
        if members is None:
            kw.update(l1_lambda=CELL[2], binarize=cfg.binarize,
                      theta_init=cfg.theta_init)
        else:
            support = {}
            for s, i in members:
                support.setdefault(s, []).append(i)
            kw.update(l1_lambda=0.0, binarize="none", theta_init=40.0,
                      support={s: torch.tensor(v, dtype=torch.long)
                               for s, v in support.items()})
        scores, prov = run_learned_mask(inference, bank, objective="pos", **kw)
        ak = prov.get("amp_kept") or {}
        alphas = {}
        for k, d in ak.items():
            lyr, knd = k.split("/")
            alphas[(int(lyr), knd)] = {int(i): float(v) for i, v in d.items()}
        return alphas

    t0 = time.time()
    alphas = discover()
    mem = [(s, i) for s, d in alphas.items() for i in d]
    n_mem = len(mem)

    # standard cf + sup on the membership (Daniel's "normal cf eval")
    circ = Circuit(name="amp")
    for (l, kd), idx in ((s, i) for s, d in alphas.items() for i in d):
        circ.add_node(CircuitNode(metadata={
            "layer_idx": l, "kind": kd, "latent_idx": idx,
            "role": "ablation_support"}))
    try:
        cf_bare, sup_v = evaluate_counterfactual_faithfulness(
            inference, bank, avg_acts, circ, neg_tokens=nt, pos_tokens=pt,
            seed_layer=LAYER, seed_kind=KIND, seed_latent_idx=sl,
            pos_argmax=pa, circuit_layers={l for (l, _), _ in
                                           ((s, i) for s, d in alphas.items()
                                            for i in d)})
        cf_bare, sup_v = round(float(cf_bare), 4), round(float(sup_v), 4)
    except Exception as e:
        print("  standard cf/sup FAILED: %s" % e, flush=True)
        cf_bare = sup_v = None

    cfa = amp_cf(alphas)
    row = {"latent": sl, "set": "discovered", "n": n_mem,
           "cf_bare": cf_bare, "cf_amp": round(cfa, 4) if cfa is not None else None,
           "sup": sup_v, "secs": round(time.time() - t0, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("  discovered n=%-6d cf_bare=%-8s cf_amp=%-8s sup=%-8s %.0fs"
          % (n_mem, cf_bare, row["cf_amp"], sup_v, row["secs"]), flush=True)

    # drive-axis null: random same-size sets, alphas fitted identically
    live = [(s, int(i)) for s, pv in pins.items()
            for i in (pv > 0).nonzero(as_tuple=True)[0].tolist()]
    rng = random.Random(2000 + sl)
    for draw in range(2):
        t0 = time.time()
        members = rng.sample(live, min(n_mem, len(live)))
        r_alphas = discover(members=members)
        r_cfa = amp_cf(r_alphas)
        r = {"latent": sl, "set": "random%d" % draw,
             "n": sum(len(d) for d in r_alphas.values()),
             "cf_amp": round(r_cfa, 4) if r_cfa is not None else None,
             "secs": round(time.time() - t0, 1)}
        fh.write(json.dumps(r) + "\n"); fh.flush()
        print("  random%d   n=%-6d cf_amp=%-8s %.0fs"
              % (draw, r["n"], r["cf_amp"], r["secs"]), flush=True)
    torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
