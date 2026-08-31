"""GE-STYLE HIERARCHICAL ATTRIBUTION — replication arm (2026-08-11).

Replicates the discovery method of Ge et al. 2024 (arXiv:2405.13868,
"Automatically Identifying Local and Global Circuits with Linear
Computation Graphs") adapted to our stack, rooted at an internal seed
latent, and scores the result under the SAME held-out protocol and
scoring code as the definitive matrix (029-panel), so the row is
directly comparable to tab:matrix.

Fidelity notes (state these in the paper):
- Ge's core: attr(v,t) = a_v * d a_t / d a_v, with gradient flow STOPPED
  during the backward at any node with attribution below a threshold tau
  (hierarchical attribution), errors detached as uninterpretable, target
  = an intermediate SAE feature's activation, prompts = contexts where
  the feature fires, tau swept for sparsity.
- Our adaptation: (1) SAE-site granularity — sites are processed in
  reverse causal order (downstream -> upstream); at each site the
  metric's gradient is taken with all previously-processed (more
  downstream) sites' below-threshold latents detached, which is the
  site-level discretisation of Ge's during-backward gating. (2) The
  stream is rewritten at every upstream site as decode(code) + detached
  error, so gradients flow only through SAE codes and error terms are
  detached exactly as Ge prescribes. (3) The root is the seed's encoder
  pre-activation at the probe anchors (their target is the feature's
  activation; ours must survive Top-K censoring). (4) Scores are
  aggregated over the seed's 48 train probes (Ge attributes per prompt).
  (5) Gating uses |attr| >= tau, keeping negative contributors (their
  bracket case study retains negative contributions). (6) tau is
  bisected per seed to land the membership at the seed's
  weighted-circuit size (the matched-size comparison of tab:matrix).
- No optimisation, no intervention during discovery: pure rank-and-gate
  attribution, as published. Evaluation (held-out free0/freeM/cf/sup) is
  ours — that is the point of the comparison.

Resumable: rows keyed (comp_idx, latent, arm). SMOKE=1 runs one seed
with 3 bisection steps.

  PYTHONPATH=src .venv/bin/python experiments/034-ge-replication/runner.py
"""
import json
import os
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense
from store.circuits import Circuit, CircuitNode
from analysis.circuits.gradient_size_sweep_runner import _build_mode_method

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
PANEL_ROWS = HERE.parent / "029-panel" / "rows.jsonl"
SMOKE = os.environ.get("SMOKE") == "1"
N_SEQ, N_TRAIN, EVAL_BS, D_SAE = 64, 48, 16, 40960
DISC_BS = 4              # discovery forward batch; depth-aware override below.
                         # Dropped 8->4 mid-run after WDDM spill on deep seeds,
                         # then 2 for seeds with >=25 upstream sites (L11 still
                         # spilled at 4). Result-identical at any value: equal
                         # chunk sizes -> uniform attr rescale -> same ranking,
                         # and tau re-bisects to the same size target.
BISECT_STEPS = 3 if SMOKE else 7
SIZE_TOL = 0.10          # accept membership within +-10% of target size

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

# ---- seeds: the 22 matrix seeds with their weighted-circuit target sizes
SEEDS = []
for line in PANEL_ROWS.open():
    r = json.loads(line)
    if r.get("arm") == "triamp400":
        SEEDS.append((r["comp_idx"], r["band"], r["latent"], r["n"]))
if SMOKE:
    SEEDS = SEEDS[:1]
print("seeds: %d (matrix panel, triamp400 target sizes)" % len(SEEDS), flush=True)

ROWS_PATH = HERE / "rows.jsonl"
done = set()
if ROWS_PATH.exists():
    for line in ROWS_PATH.open():
        try:
            r = json.loads(line)
            done.add((r["comp_idx"], r["latent"], r["arm"]))
        except Exception:
            pass
fh = ROWS_PATH.open("a")


# ============================ scoring (verbatim panel semantics) =============
class AmpCircuitPatcher:
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
    def __init__(self, inject, seed_site, w_seed, b_seed):
        self.inject = inject
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


# ============================ Ge-style hierarchical discovery ================
class GeHierInstrument:
    """Reverse-causal-sweep hierarchical attribution instrument.

    Every upstream site's stream is rewritten as decode(code) + detached
    error, so all gradient flows through SAE codes (errors detached, as
    Ge prescribes). Latents pruned at already-processed downstream sites
    have their code coordinates detached, gating gradient flow exactly
    as Ge's during-backward stop. The current score site's code retains
    grad for the attribution readout.
    """

    def __init__(self, sites, keep_masks, score_site, seed_site, w_seed, b_seed):
        self.sites = set(sites)
        self.keep_masks = keep_masks      # {site: float mask [d_sae]} 1=keep
        self.score_site = score_site
        self.seed_site = seed_site
        self.w_seed, self.b_seed = w_seed, b_seed
        self.seed_pre = None
        self.scored = None                # dense_g at score_site (retains grad)

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
        keep = self.keep_masks.get(site)
        if keep is None:
            dense_g = dense
        else:
            k = keep.to(device=dense.device, dtype=dense.dtype)
            dense_g = dense * k + dense.detach() * (1.0 - k)
        if site == self.score_site:
            dense_g.retain_grad()
            self.scored = dense_g
        recon = bank.decode(dense_g, kind, layer_idx, add_bias=False)
        recon_ref = bank.decode(dense.detach(), kind, layer_idx, add_bias=False)
        err = (x - recon_ref).detach()
        return (recon + err).to(x.dtype)


def site_order(sites):
    """Downstream -> upstream: later layers first; within a layer,
    later-in-causal-order kinds first (bank.kinds is causal order)."""
    return sorted(sites, key=lambda s: (s[0], bank.kinds.index(s[1])),
                  reverse=True)


def hier_run(tau, UP, seed_site, w_seed, b_seed, pt_tr, pa_tr):
    """One full hierarchical sweep at threshold tau.
    Returns keep_masks (float [d_sae] per site) and membership size."""
    order = site_order(UP)
    keep_masks = {}
    inference.disable_compile()
    try:
        for site in order:
            attr = torch.zeros(D_SAE, device=device)
            for s0 in range(0, int(pt_tr.shape[0]), DISC_BS):
                tk = pt_tr[s0:s0 + DISC_BS]
                pa = pa_tr[s0:s0 + DISC_BS]
                inst = GeHierInstrument(UP, keep_masks, site, seed_site,
                                        w_seed, b_seed)
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
                if inst.scored is not None and inst.scored.grad is not None:
                    g = inst.scored.grad
                    attr += (inst.scored.detach() * g).sum(dim=(0, 1)).float()
                del inst
            kept = (attr.abs() >= tau).float().cpu()
            keep_masks[site] = kept
        n = int(sum(int(m.sum()) for m in keep_masks.values()))
        return keep_masks, n
    finally:
        inference.enable_compile()
        torch.cuda.empty_cache()


def discover_at_size(n_target, UP, seed_site, w_seed, b_seed, pt_tr, pa_tr,
                     log):
    """Bisect tau (log-space) to land total membership near n_target."""
    # bracket: tau_lo keeps ~everything active, tau_hi keeps ~nothing
    masks, n_all = hier_run(0.0, UP, seed_site, w_seed, b_seed, pt_tr, pa_tr)
    log("  tau=0: n=%d (all-active reference)" % n_all)
    if n_all <= n_target:
        return masks, n_all, 0.0
    lo, hi = 1e-8, 1e2
    best = (masks, n_all, 0.0)
    for it in range(BISECT_STEPS):
        mid = (lo * hi) ** 0.5
        masks, n = hier_run(mid, UP, seed_site, w_seed, b_seed, pt_tr, pa_tr)
        log("  bisect %d: tau=%.3e n=%d (target %d)" % (it, mid, n, n_target))
        if abs(n - n_target) < abs(best[1] - n_target):
            best = (masks, n, mid)
        if abs(n - n_target) <= SIZE_TOL * n_target:
            return masks, n, mid
        if n > n_target:
            lo = mid
        else:
            hi = mid
    return best


# ============================ main loop ======================================
for comp_idx, band, sl, n_target in SEEDS:
    arm = "ge_hier@n"
    if (comp_idx, sl, arm) in done:
        print("[%s %d] done, skipping" % (band, sl), flush=True)
        continue
    layer, ki = split_component_idx(comp_idx, n_kinds)
    kind = bank.kinds[ki]
    m0 = _build_mode_method("counterfactual_gradient", "local", inference,
                            bank, avg_acts, pb)
    try:
        pd_ = m0.build_probe_dataset(comp_idx, sl)
    except Exception as e:
        print("[%s %d] probes FAILED %s: %s" % (band, sl, type(e).__name__, e),
              flush=True)
        del m0
        continue
    del m0
    if pd_ is None or int(pd_.pos_tokens.shape[0]) < N_SEQ:
        print("[%s %d] thin probes, skipping" % (band, sl), flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TRAIN], pa[:N_TRAIN], nt[:N_TRAIN]
    pt_ho, pa_ho, nt_ho = pt[N_TRAIN:], pa[N_TRAIN:], nt[N_TRAIN:]

    sae = bank.saes[kind][layer]
    w_seed = sae.encoder.weight[sl].detach()
    b_seed = sae._get_bias_eff()[sl].detach()
    UP = sorted(upstream_sites(bank, layer, kind))
    DISC_BS = 2 if len(UP) >= 25 else 4
    print("[%s %d] %d upstream sites -> disc batch %d"
          % (band, sl, len(UP), DISC_BS), flush=True)
    seed_site = (layer, kind)
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer,
                                             kind, sl, pa_tr, batch_size=EVAL_BS))
    a_pos_ho = float(measure_seed_activation(inference, bank, pt_ho, layer,
                                             kind, sl, pa_ho, batch_size=EVAL_BS))
    if a_pos_ho < 0.05:
        print("[%s %d] held-out a_pos too small, skipping" % (band, sl),
              flush=True)
        continue
    means_tr, pins_tr = collect_site_anchors(inference, bank, pt_tr, set(UP),
                                             pa_tr, pin_position_specific=False)
    e0_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho,
                                          layer, kind, sl, pos_argmax=pa_ho,
                                          batch_size=EVAL_BS))
    eMd_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho,
                                           layer, kind, sl, pos_argmax=pa_ho,
                                           site_means=means_tr,
                                           batch_size=EVAL_BS))
    p0 = AmpInjectPatcher({}, seed_site, w_seed, b_seed)
    chunks = []
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(nt_ho.shape[0]), EVAL_BS):
                p0.seed_pre = None
                inference.forward(nt_ho[s0:s0 + EVAL_BS], patcher=p0,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
                chunks.append(p0.seed_pre.detach())
    finally:
        inference.enable_compile()
    neg_pre = torch.cat(chunks, 0)
    na_ho = neg_pre.argmax(dim=1).cpu()
    a_base = float(torch.relu(
        neg_pre[torch.arange(neg_pre.shape[0], device=neg_pre.device),
                na_ho.to(neg_pre.device)]).mean())
    print("[%s %d] a_pos tr %.3f ho %.3f | e0_ho %.3f | target n %d"
          % (band, sl, a_pos_tr, a_pos_ho, e0_ho, n_target), flush=True)

    def read(patcher, tokens, anchors):
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
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

    t0 = time.time()
    masks, n_mem, tau = discover_at_size(
        n_target, UP, seed_site, w_seed, b_seed, pt_tr, pa_tr,
        lambda m: print(m, flush=True))
    disc_secs = time.time() - t0

    alphas = {}
    for site, m in masks.items():
        idx = m.nonzero(as_tuple=True)[0].tolist()
        if idx:
            alphas[site] = {int(i): 1.0 for i in idx}
    (HERE / ("members_ge_hier_%d_%d.json" % (comp_idx, sl))).write_text(
        json.dumps([[l, kd, int(i)] for (l, kd), d in alphas.items()
                    for i in d]))

    aw0 = read(AmpCircuitPatcher(alphas, None, seed_site, w_seed, b_seed),
               pt_ho, pa_ho)
    awM = read(AmpCircuitPatcher(alphas, means_tr, seed_site, w_seed, b_seed),
               pt_ho, pa_ho)
    inject = {s: {i: float(pins_tr[s][i]) for i in d}
              for s, d in alphas.items() if s in pins_tr}
    cfa = read(AmpInjectPatcher(inject, seed_site, w_seed, b_seed), nt_ho, na_ho)
    circ = Circuit(name="ge_hier")
    for (l, kd), dd in alphas.items():
        for i in dd:
            circ.add_node(CircuitNode(metadata={
                "layer_idx": l, "kind": kd, "latent_idx": int(i),
                "role": "ablation_support"}))
    try:
        cf_bare, sup_v = evaluate_counterfactual_faithfulness(
            inference, bank, avg_acts, circ, neg_tokens=nt_ho,
            pos_tokens=pt_ho, seed_layer=layer, seed_kind=kind,
            seed_latent_idx=sl, pos_argmax=pa_ho,
            circuit_layers={l for (l, _) in alphas})
        cf_bare, sup_v = round(float(cf_bare), 4), round(float(sup_v), 4)
    except Exception as e:
        print("  cf_bare/sup failed: %s" % e, flush=True)
        cf_bare = sup_v = None
    row = {"comp_idx": comp_idx, "band": band, "latent": sl, "arm": arm,
           "n": n_mem, "n_target": n_target, "tau": tau,
           "ampF0": (round((aw0 - e0_ho) / (a_pos_ho - e0_ho), 4)
                     if abs(a_pos_ho - e0_ho) > 1e-9 else None),
           "ampFMd": (round((awM - eMd_ho) / (a_pos_ho - eMd_ho), 4)
                      if abs(a_pos_ho - eMd_ho) > 1e-9 else None),
           "cf_amp": (round((cfa - a_base) / (a_pos_ho - a_base), 4)
                      if abs(a_pos_ho - a_base) > 1e-9 else None),
           "cf_bare": cf_bare, "sup": sup_v,
           "a_pos_ho": round(a_pos_ho, 3), "disc_secs": round(disc_secs, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("  %-10s n=%-6d F0=%-8s FMd=%-8s cf_bare=%-8s sup=%-8s (%.0fs)"
          % (arm, n_mem, row["ampF0"], row["ampFMd"], cf_bare, sup_v,
             disc_secs), flush=True)
    torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
