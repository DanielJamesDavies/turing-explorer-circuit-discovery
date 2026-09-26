"""AMPLITUDE-AWARE EVALUATION PASS over the production circuits (rows v=3).

v3 (2026-09-22, DAN-66 / DAN-67 / DAN-65 audit) fixes two scoring bugs of v1
and v2 and is NOT row-comparable with them (output: amp_eval_v3.*.jsonl):
  DAN-67  circuit-only runs go ONLY through the engine's CircuitOnlyPatcher,
          which ablates every upstream site outside the circuit. v1's delta
          patcher (AmpCircuitPatcher, deleted) returned `x` untouched at sites
          holding no member, while the empty baseline ablated every site, so
          numerator and denominator ablated different scopes (47% of pilot
          circuits affected). Each row records `all_sites_ablated`.
  DAN-66  every ratio uses ONE read of the target in all its terms. v1/v2's
          unsuffixed ratios read the intervened run as relu(pre-activation)
          but a_pos and the empty baselines post-Top-K, so the empty circuit
          scored median 0.14 instead of 0. v3 writes each ratio twice:
            `<name>_tk`   ACTIVATION read (post-Top-K) — PRIMARY: every pass/fail
                          number in the paper (protocol v1, DAN-7).
            `<name>_pre`  PRE-ACTIVATION read, relu(w.x + b) — DIAGNOSTIC
                          (graded reconstruction below the firing threshold).
          A ratio is None when its baseline is not below a_pos under that read
          (e.g. zero fill drives the pre-activation baseline off-distribution).
  necessity uses the protocol-v1 form (a_pos - a_{M\\C}) / (a_pos - a_empty).
v1's fields are replaced by engine-path equivalents: F0_amp -> free0,
F0_amp_tr -> free0_tr, F0_a1 -> free0_a1, FMd_amp -> freeM_dense,
cf_amp -> phi_cf_alpha_blind, F0_perm -> free0_perm, F0_rand -> free0_rand.
The pipeline's own evaluate_counterfactual_faithfulness (cf_bare_anch,
sup_anch) is no longer called.

The measurement family (the Table-1 family of
experiments/053-gpt2-topk/fit_latent_seed.py on the TuringLLM engine):

  floors     freeM_topk / freeN_topk / freeN_dense (+ engine-path free0 and
             freeM_dense), each normalised by ITS OWN empty-circuit baseline;
             all five baselines recorded (e0, eM_dense, eM_topk, eN_dense,
             eN_topk).
  roles      activator / inhibitor per member by grad x activation on the
             TRAIN slice (the stored attribution_score is the learned-mask gate
             probability sigmoid(theta) in (0.5, 1): unsigned, unusable).
  phi_sup    role-aware, blind, and alpha-aware; inhibitor RELEASE test and
             activators-only suppression.
  phi_cf     role-aware at fitted alpha and at alpha = 1, on held-out
             negatives, same anchor as v1's cf_amp (argmax of the seed's
             natural pre-activation on each negative sequence).
  phi_pin    members clamped to alpha x their CLEAN position-wise values, under
             zero / dense-posmean / topk-posmean fill.
  split      natural a_pos per 16-sequence block of the stored contexts.

READS. Raw measurements are stored under both reads of the target at the
anchor: `<name>_raw_pre` / `<name>_pre` = mean relu(w_seed.x + b_seed) and
`<name>_raw_tk` / `<name>_tk` = mean post-Top-K activation. Ratios never mix
them (see DAN-66 above).

PATCHERS. Circuit-only and pinned measurements go through the engine's
CircuitOnlyPatcher (keep_scales = fitted gains, respect_topk = the engine's
_respect_topk_fill); set-to-value / set-to-zero edits of the live stream go
through the engine's CounterfactualInterventionPatcher. Both are subclassed
ONLY to tap the seed read (and, for CircuitOnlyPatcher, to record which sites
it ablated). The one extra transform (phi_sup_alpha: members scaled in the
live stream) mirrors the engine form decode(patched) + (x - decode(natural))
— SAE error preserved, no cap/clamp.

  SEED_SHARD=0/16 PYTHONPATH=src python experiments/049-circuit-graph/amp_eval_pass_v2.py
      -> results_full/amp_eval_v3.shard<i>.jsonl   (resumable: scored seeds are skipped)
  SHARDS=0,1,2,...   several store shards in ONE process (one output file per shard; saves the ~3 min start-up)
  SMOKE=1            first 6 circuits -> amp_eval_v3_smoke.shard<i>.jsonl
  SEEDS=<a.b.c,d.e.f | file: one key per line, or csv with an `skey` column>
      TAG=<name>     -> amp_eval_v3_<TAG>.jsonl   (seeds looked up across all shards)
  PART=j/m           take items[j::m] of the selection (output gets .part<j>of<m>)
  DATA=<dir of discovered_circuits.shard*.pt>  OUT=<results dir>  TABLES=<parquet dir>
  CTR_SOURCE=store|close|random   contrast contexts for the C means and sufficiency to induce. "store" (default)
                 reads the neg_ctx kNN store, which gives 38% of targets one shared fallback list (DAN-64); "close" and
                 "random" re-select through NegContextSelector (activating contexts forwarded when missing from
                 seq_repr, own activating contexts excluded, candidates kept only if the target stays out of Top-K).
"""
import contextlib
import glob
import io
import json
import os
import random
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
DATA = Path(os.environ.get("DATA", str(HERE / "data_full")))
OUT = Path(os.environ.get("OUT", str(HERE / "results_full")))
TABLES = Path(os.environ.get("TABLES", str(HERE / "tables_full")))
SMOKE = os.environ.get("SMOKE") == "1"
SHARD_I, SHARD_K = (int(x) for x in os.environ.get("SEED_SHARD", "0/16").split("/"))
PART_J, PART_M = (int(x) for x in os.environ.get("PART", "0/1").split("/"))
SEEDS = os.environ.get("SEEDS")
TAG = os.environ.get("TAG", "subset")
CTR_SOURCE = os.environ.get("CTR_SOURCE", "store")
N_SEQ, EVAL_BS, ROLE_BS = 64, 16, 8
HOLDOUT_FRAC = 0.25          # engine split (learned_mask.split): n_hold = round(n * 0.25), train = the first n - n_hold
MIN_POS = 8

G = {}                       # engine handles, filled by setup()


def setup():
    """Load model, SAE bank and probe builder exactly as v1 did (same config overrides)."""
    if G:
        return G
    from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
    from circuit.probe_dataset import ProbeDatasetBuilder
    from config import config
    from data.loader import DataLoader
    from hardware import detect_devices, is_fast_memory, should_compile
    from model.inference import Inference
    from pipeline.discovery_artifacts import load_discovery_artifacts
    from sae.bank import SAEBank

    torch.set_float32_matmul_precision("high")
    load_discovery_artifacts("outputs", candidates_path="outputs/candidates.pt")
    devices = detect_devices(); device = devices[0]
    loader = DataLoader(device=device, pin_memory=is_fast_memory())
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
    pb = ProbeDatasetBuilder(inference, bank, loader)
    kinds = list(bank.kinds)
    avg_acts = torch.zeros((bank.n_layer * len(kinds), bank.d_sae), device=bank.device)
    disc = config.discovery
    disc.probe_sequence_count = N_SEQ; disc.eval_sequence_count = N_SEQ; disc.eval_batch_size = EVAL_BS
    disc.probe_batch_size = 4; disc.position_aware = False
    M0 = _build_mode_method("counterfactual_gradient", "local", inference, bank, avg_acts, pb)
    G.update(inference=inference, bank=bank, M0=M0, avg_acts=avg_acts, KINDS=kinds, NK=len(kinds),
             D=bank.d_sae, K=int(bank.k), device=device)
    return G


# --------------------------------------------------------------------------- patchers
def _tap(p, x, site):
    """Seed read, both conventions, from the (already edited) stream at the seed site."""
    from sae.dense import target_latent_activations
    bank = G["bank"]
    w = p.w_seed.to(device=x.device, dtype=x.dtype); b = p.b_seed.to(device=x.device, dtype=x.dtype)
    pre = x @ w + b
    with torch.no_grad():
        ta, ti = bank.encode(x.detach(), site[1], site[0])
        p.tap_tk = target_latent_activations(ta, ti, p.seed_idx).detach()
    return pre


# v1's AmpCircuitPatcher was DELETED in v3 (DAN-67): it left every upstream site that held no
# circuit member live (`if al is None: return x`). All circuit-only runs use the engine's
# CircuitOnlyPatcher (TapCO below), which ablates every in-scope site.


class AmpInjectPatcher:
    """Members SET to given values in the otherwise-live stream (empty inject = the natural run); seed tapped.
    Used for the natural-stream reads only; interventions go through the engine's patchers."""

    def __init__(self, inject, seed_site, w_seed, b_seed, seed_idx):
        self.inject = inject; self.seed_site = seed_site
        self.w_seed, self.b_seed, self.seed_idx = w_seed, b_seed, seed_idx
        self.seed_pre = None; self.tap_tk = None

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        from sae.dense import sparse_topk_to_dense
        bank, D = G["bank"], G["D"]
        if (layer_idx, kind) == self.seed_site:
            self.seed_pre = _tap(self, x, self.seed_site)
            return x
        inj = self.inject.get((layer_idx, kind))
        if not inj:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        code = dense.clone()
        idx = torch.tensor(sorted(inj), device=dense.device, dtype=torch.long)
        code[..., idx] = torch.tensor([inj[int(i)] for i in idx], device=dense.device, dtype=dense.dtype)
        return x + bank.decode(code - dense, kind, layer_idx, add_bias=False).to(x.dtype)


class LiveScalePatcher:
    """NEW (phi_sup_alpha): members scaled by a per-member factor in the otherwise-live stream.
    Engine form: decode(patched) + (x - decode(natural)) — SAE error preserved."""

    def __init__(self, scales, seed_site, w_seed, b_seed, seed_idx):
        self.scales = scales; self.seed_site = seed_site
        self.w_seed, self.b_seed, self.seed_idx = w_seed, b_seed, seed_idx
        self.seed_pre = None; self.tap_tk = None

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        from sae.dense import sparse_topk_to_dense
        bank, D = G["bank"], G["D"]
        if (layer_idx, kind) == self.seed_site:
            self.seed_pre = _tap(self, x, self.seed_site)
            return x
        sc = self.scales.get((layer_idx, kind))
        if sc is None:
            return x
        idx, fac = sc
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        error = x - bank.decode(dense, kind, layer_idx)
        patched = dense.clone()
        patched[..., idx] = dense[..., idx] * fac.to(dense.dtype)
        return bank.decode(patched, kind, layer_idx) + error


def _engine_classes():
    """Engine patchers subclassed ONLY to tap the seed read; the edit is the engine's own transform."""
    if "TapCO" in G:
        return G["TapCO"], G["TapCF"]
    from eval.ablation_faithfulness import CircuitOnlyPatcher
    from eval.counterfactual_faithfulness import CounterfactualInterventionPatcher

    class TapCO(CircuitOnlyPatcher):
        def __init__(self, *a, w_seed=None, b_seed=None, **k):
            super().__init__(*a, **k)
            self.w_seed, self.b_seed, self.seed_idx = w_seed, b_seed, self.seed_latent_idx
            self.seed_pre = None; self.tap_tk = None
            self.edited = set()          # sites this patcher ablated (DAN-67 audit: must equal the upstream scope)

        def transform(self, layer_idx, kind, x):
            if layer_idx == self.seed_layer and kind == self.seed_kind:
                self.seed_pre = _tap(self, x, (layer_idx, kind))
            if (layer_idx, kind) in self.in_scope:
                self.edited.add((layer_idx, kind))
            return super().transform(layer_idx, kind, x)

    class TapCF(CounterfactualInterventionPatcher):
        def __init__(self, *a, w_seed=None, b_seed=None, **k):
            super().__init__(*a, **k)
            self.w_seed, self.b_seed, self.seed_idx = w_seed, b_seed, self.seed_latent_idx
            self.seed_pre = None; self.tap_tk = None

        def transform(self, layer_idx, kind, x):
            if layer_idx == self.seed_layer and kind == self.seed_kind:
                self.seed_pre = _tap(self, x, (layer_idx, kind))
            return super().transform(layer_idx, kind, x)

    G["TapCO"], G["TapCF"] = TapCO, TapCF
    return TapCO, TapCF


class LazyPins:
    """Position-specific pins [B, T, d_sae] for the engine's pinned mode, materialised per site on
    demand from the captured member columns (a full dense clean stream per site would be ~6 GB)."""

    def __init__(self, cap, idx, shape):
        self.cap, self.idx, self.shape = cap, idx, shape

    def __getitem__(self, site):
        z = torch.zeros(*self.shape, G["D"], device=self.cap[site].device, dtype=torch.float32)
        z[..., self.idx[site]] = self.cap[site]
        return z


class RolePatcher:
    """grad x activation: add a zero perturbation z at every member site of the NATURAL stream;
    d seed / d w_i (w_i = 1 a scaling weight on member i) = sum_{b,t} c_i[b,t] * <d seed / d x[b,t], W_dec[:, i]>."""

    def __init__(self, msets, seed_site, w_seed, b_seed):
        self.msets, self.seed_site, self.w_seed, self.b_seed = msets, seed_site, w_seed, b_seed
        self.seed_pre = None; self.z = {}; self.codes = {}

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        from sae.dense import sparse_topk_to_dense
        site = (layer_idx, kind)
        if site == self.seed_site:
            self.seed_pre = x @ self.w_seed.to(device=x.device, dtype=x.dtype) + self.b_seed.to(device=x.device, dtype=x.dtype)
            return x
        idx = self.msets.get(site)
        if idx is None:
            return x
        with torch.no_grad():
            ta, ti = G["bank"].encode(x.detach(), kind, layer_idx)
            self.codes[site] = sparse_topk_to_dense(ta, ti, G["D"], dtype=torch.float32)[..., idx].clone()
        z = torch.zeros_like(x, requires_grad=True)
        self.z[site] = z
        return x + z


# --------------------------------------------------------------------------- reads
def read(make_patcher, tokens, anchors, per_seq=False):
    """Run `tokens` in EVAL_BS chunks under make_patcher(start, stop) -> patcher; seed read at `anchors`.
    Returns {"pre": mean relu(pre), "tk": mean post-Top-K}, or the per-sequence tensors."""
    inference = G["inference"]
    pres, tks = [], []
    edited = set()
    inference.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]; B = int(tk.shape[0])
                p = make_patcher(s0, s0 + B)
                inference.forward(tk, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
                pre = p.seed_pre; rr = torch.arange(B, device=pre.device)
                anc = anchors[s0:s0 + B].to(pre.device).clamp(0, pre.shape[1] - 1)
                pres.append(torch.relu(pre[rr, anc]).float()); tks.append(p.tap_tk[rr, anc].float())
                edited |= getattr(p, "edited", set())
    finally:
        inference.enable_compile()
    pres, tks = torch.cat(pres), torch.cat(tks)
    if per_seq:
        return pres, tks
    return {"pre": float(pres.mean()), "tk": float(tks.mean()), "edited": edited}


def frac(v, lo, hi):
    return (v - lo) / (hi - lo) if abs(hi - lo) > 1e-9 else None


def split_n(n):
    n_hold = int(round(n * HOLDOUT_FRAC))
    return max(1, n - n_hold)


def seed_of(c):
    return next(n for n in c.nodes.values() if n.metadata.get("role") == "seed")


def key_of(c):
    sf = seed_of(c).metadata["feature_id"]
    return "%d.%s.%d" % (sf.layer, sf.kind, sf.index)


# --------------------------------------------------------------------------- one circuit
def score_circuit(c, live_pool, override_alphas=None, skip_roles=False):
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_anchors, collect_site_means
    from pipeline.component_index import component_idx as comp_of
    from sae.dense import sparse_topk_to_dense
    inference, bank, M0, KINDS, NK, D, K = (G[k] for k in ("inference", "bank", "M0", "KINDS", "NK", "D", "K"))
    TapCO, TapCF = _engine_classes()
    ts = time.time()
    md = c.metadata
    seed = seed_of(c)
    sf = seed.metadata["feature_id"]; layer, kind, sl = sf.layer, sf.kind, sf.index
    key = "%d.%s.%d" % (layer, kind, sl)
    rng = random.Random(zlib.crc32(key.encode())); nrng = np.random.RandomState(zlib.crc32(key.encode()) % (2 ** 31))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if override_alphas is not None:
        alphas = override_alphas
    else:
        alphas = defaultdict(dict)
        for n in c.nodes.values():
            if n is seed:
                continue
            f = n.metadata["feature_id"]; alphas[(f.layer, f.kind)][int(f.index)] = float(n.metadata.get("amplitude", 1.0))
        alphas = dict(alphas)
    n_mem = sum(len(d) for d in alphas.values())
    comp = comp_of(layer, KINDS.index(kind), NK)
    pd_ = M0.build_probe_dataset(comp, sl)
    n_pos = 0 if pd_ is None else int(pd_.pos_tokens.shape[0])
    if n_pos < MIN_POS:
        return dict(seed=key, skip="thin_probes", n=n_mem, n_pos=n_pos)
    pt, pa, nt = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ], pd_.neg_tokens[:N_SEQ]
    ctr_fallback = bool(neg_ctx_store_fallback(comp, sl))
    if CTR_SOURCE != "store":
        sel = M0._neg_context_selector().select(comp, sl, CTR_SOURCE, max_sequences=N_SEQ, batch_size=EVAL_BS,
                                                exact=False, non_activation_threshold=0.0,
                                                filter_batch_size=32, load_window_size=256)
        nt = sel.tokens[:N_SEQ] if sel is not None else nt[:0]
    n_pos = int(pt.shape[0]); n_neg = int(nt.shape[0])
    if n_neg < 2:
        return dict(seed=key, skip="no_negatives", n=n_mem, n_pos=n_pos, n_neg=n_neg)
    n_tr = split_n(n_pos); n_ntr = split_n(n_neg)
    pt_tr, pa_tr, pt_ho, pa_ho = pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:]
    nt_tr, nt_ho = nt[:n_ntr], nt[n_ntr:]
    sae = bank.saes[kind][layer]; w_seed = sae.encoder.weight[sl].detach(); b_seed = sae._get_bias_eff()[sl].detach()
    UP = sorted(upstream_sites(bank, layer, kind)); UPS = set(UP); site = (layer, kind)
    SK = dict(w_seed=w_seed, b_seed=b_seed)
    dev = pt.device

    msets = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in alphas.items() if d}
    amps_of = {s: torch.tensor([alphas[s][int(i)] for i in v.tolist()], device=dev, dtype=torch.float32) for s, v in msets.items()}
    keep = {s: set(d) for s, d in alphas.items() if d}
    keep_tensors = dict(msets)
    scales = {}
    for s, v in msets.items():
        sv = torch.ones(D, device=dev, dtype=torch.float32); sv[v] = amps_of[s]; scales[s] = sv

    def co(keep_, tokens, anchors, means=None, topk=False, use_scales=True, pins=None, scales_=None):
        """Engine circuit-only run (CircuitOnlyPatcher): members at alpha x live (or pinned) value; EVERY other
        latent at every upstream site ablated per fill (zero when means is None). scales_ overrides the fitted gains."""
        kt = {s: keep_tensors[s] for s in keep_} if keep_ is keep else {
            s: torch.tensor(sorted(i), device=dev, dtype=torch.long) for s, i in keep_.items() if i}
        if pins is not None:
            assert int(tokens.shape[0]) <= EVAL_BS, "position-specific pins are captured for one chunk"
        ks = scales_ if scales_ is not None else (scales if (use_scales and keep_) else None)
        return read(lambda a, b: TapCO(bank=bank, keep_indices=keep_, in_scope=UPS, seed_layer=layer, seed_kind=kind,
                                       seed_latent_idx=sl, pos_argmax=anchors[a:b], site_means=means, pin_values=pins,
                                       respect_topk=topk, topk=K, keep_tensors=kt, keep_scales=ks, **SK), tokens, anchors)

    def cfp(targets, zero, tokens, anchors):
        """Engine live-stream edit (CounterfactualInterventionPatcher): `targets` SET to values, `zero` SET to 0."""
        return read(lambda a, b: TapCF(bank=bank, activator_targets=targets, inhibitor_indices=zero, seed_layer=layer,
                                       seed_kind=kind, seed_latent_idx=sl, pos_argmax=anchors[a:b], **SK), tokens, anchors)

    # ---- natural stream: a_pos on held-out and train, both reads; per-block a_pos for the split-ordering diagnostic
    nat_pre, nat_tk = read(lambda a, b: AmpInjectPatcher({}, site, w_seed, b_seed, sl), pt, pa, per_seq=True)
    a_pos = {"pre": float(nat_pre[n_tr:].mean()), "tk": float(nat_tk[n_tr:].mean())}
    a_pos_tr = {"pre": float(nat_pre[:n_tr].mean()), "tk": float(nat_tk[:n_tr].mean())}
    a_pos_blocks = [round(float(nat_tk[i:i + 16].mean()), 4) for i in range(0, n_pos, 16)]

    # ---- anchors: posctx means + collapsed pins (TRAIN positives), negctx means (TRAIN negatives)
    means_tr, pins_tr = collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)
    means_neg = collect_site_means(inference, bank, nt_tr, UPS) if UP else {}

    # ---- empty-circuit baselines (engine: every upstream site ablated), both reads
    E = {"e0": co({}, pt_ho, pa_ho), "eM_dense": co({}, pt_ho, pa_ho, means=means_tr),
         "eM_topk": co({}, pt_ho, pa_ho, means=means_tr, topk=True),
         "eN_dense": co({}, pt_ho, pa_ho, means=means_neg), "eN_topk": co({}, pt_ho, pa_ho, means=means_neg, topk=True)}
    e0_tr = co({}, pt_tr, pa_tr)

    # ---- negatives: natural run, would-be-firing anchors (argmax of the seed pre-activation), baseline
    neg_full = []
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(nt_ho.shape[0]), EVAL_BS):
                p0 = AmpInjectPatcher({}, site, w_seed, b_seed, sl)
                inference.forward(nt_ho[s0:s0 + EVAL_BS], patcher=p0, grad_enabled=False, return_activations=False, tokenize_final=False)
                neg_full.append((p0.seed_pre.detach(), p0.tap_tk))
    finally:
        inference.enable_compile()
    neg_pre = torch.cat([a for a, _ in neg_full], 0); neg_tk = torch.cat([b for _, b in neg_full], 0)
    na_ho = neg_pre.argmax(dim=1).cpu(); rr = torch.arange(neg_pre.shape[0], device=neg_pre.device)
    A_BASE = {"pre": float(torch.relu(neg_pre[rr, na_ho.to(neg_pre.device)]).float().mean()),
              "tk": float(neg_tk[rr, na_ho.to(neg_tk.device)].float().mean())}
    pins_tr = {s: pins_tr[s].cpu() for s in alphas if s in pins_tr}      # same values; avoids one GPU sync per member below
    inject = {s: {i: float(a * float(pins_tr[s][i])) for i, a in d.items()} for s, d in alphas.items() if s in pins_tr}

    row = dict(seed=key, layer=layer, kind=kind, n=n_mem,
               amp_median=(md.get("amp_stats") or {}).get("median"), cf_bare_legacy=(md.get("evals") or {}).get("counterfactual_faithfulness"))

    def put(name, r):
        row[name + "_pre"] = r["pre"]; row[name + "_tk"] = r["tk"]

    def ratio(name, r, e, ap=None):
        """Faithfulness form (x - e) / (a_pos - e), ONE read per ratio (DAN-66): `_tk` activation (primary),
        `_pre` pre-activation (diagnostic). None when the baseline is not below a_pos under that read."""
        ap = a_pos if ap is None else ap
        for rd in ("tk", "pre"):
            row[name + "_" + rd] = frac(r[rd], e[rd], ap[rd]) if e[rd] < ap[rd] else None

    def ind_ratio(name, r):
        """Sufficiency to induce (A-_int - A-) / (A+ - A-) on contrast contexts, one read per ratio."""
        for rd in ("tk", "pre"):
            row[name + "_" + rd] = frac(r[rd], A_BASE[rd], a_pos[rd]) if A_BASE[rd] < a_pos[rd] else None

    put("a_pos", a_pos); put("a_pos_tr", a_pos_tr); put("a_base", A_BASE); put("e0_tr", e0_tr)
    for k_, r in E.items():
        put(k_, r)
    for rd in ("tk", "pre"):         # vacuity relative to a_pos (v1's rule, relative to e0, never fired: e0_tk is usually 0)
        row["vacuous_" + rd] = bool(abs(a_pos[rd] - E["e0"][rd]) < 0.05 * abs(a_pos[rd]))

    # 1. faithfulness, engine path, fitted gains; every upstream site outside the circuit ablated (DAN-67)
    runs = {"free0": (None, False, "e0"), "freeM_dense": (means_tr, False, "eM_dense"), "freeM_topk": (means_tr, True, "eM_topk"),
            "freeN_dense": (means_neg, False, "eN_dense"), "freeN_topk": (means_neg, True, "eN_topk")}
    for name, (mm, tk_, ek) in runs.items():
        r = co(keep, pt_ho, pa_ho, means=mm, topk=tk_)
        put(name + "_raw", r); ratio(name, r, E[ek])
        if name == "free0":
            row["all_sites_ablated"] = bool(r["edited"] == UPS)
    r = co(keep, pt_tr, pa_tr)
    put("free0_tr_raw", r); ratio("free0_tr", r, e0_tr, a_pos_tr)
    r = co(keep, pt_ho, pa_ho, use_scales=False)
    put("free0_a1_raw", r); ratio("free0_a1", r, E["e0"])

    # 1b. coefficient and selection checks (v1's nulls, now through the engine path; per-seed RNG so a resume reproduces them)
    perm_scales = {}
    for s, v in msets.items():
        vals = amps_of[s].tolist(); rng.shuffle(vals)
        sv = torch.ones(D, device=dev, dtype=torch.float32); sv[v] = torch.tensor(vals, device=dev, dtype=torch.float32)
        perm_scales[s] = sv
    r = co(keep, pt_ho, pa_ho, scales_=perm_scales) if msets else E["e0"]
    put("free0_perm_raw", r); ratio("free0_perm", r, E["e0"])
    rnd = {}
    for s, d in alphas.items():
        pool = live_pool.get(s)
        if pool is None or len(pool) == 0:
            continue
        n_ = min(len(d), len(pool)); rnd[s] = {int(i) for i in nrng.choice(pool, n_, replace=False)}
    r = co(rnd, pt_ho, pa_ho, use_scales=False) if rnd else E["e0"]
    put("free0_rand_raw", r); ratio("free0_rand", r, E["e0"])       # random set at alpha = 1 (NOT the paper's baseline; DAN-17/46)

    # 1c. sufficiency to induce, role-blind: every member SET to alpha x its train pin on held-out contrast contexts
    r = cfp(inject, {}, nt_ho, na_ho) if inject else {"pre": A_BASE["pre"], "tk": A_BASE["tk"]}
    put("phi_cf_alpha_blind_raw", r); ind_ratio("phi_cf_alpha_blind", r)

    # 2. roles: grad x activation on the TRAIN slice, natural stream
    t_roles = time.time()
    attr = {s: torch.zeros(len(v), device=dev, dtype=torch.float32) for s, v in msets.items()}
    if msets and not skip_roles:
        wdec = {s: bank.saes[s[1]][s[0]].decoder.weight.detach()[:, v.to(bank.saes[s[1]][s[0]].decoder.weight.device)].to(device=dev, dtype=torch.float32)
                for s, v in msets.items()}
        inference.disable_compile()
        try:
            for s0 in range(0, n_tr, ROLE_BS):
                tkb, anb = pt_tr[s0:s0 + ROLE_BS], pa_tr[s0:s0 + ROLE_BS]
                rp = RolePatcher(msets, site, w_seed, b_seed)
                inference.forward(tkb, patcher=rp, grad_enabled=True, return_activations=False, tokenize_final=False)
                pre = rp.seed_pre; B = pre.shape[0]
                anc = anb.to(pre.device).clamp(0, pre.shape[1] - 1)
                tgt = pre[torch.arange(B, device=pre.device), anc].float().sum()
                sites_ = list(rp.z)
                gs = torch.autograd.grad(tgt, [rp.z[s] for s in sites_], allow_unused=True)
                for s, g in zip(sites_, gs):
                    if g is not None:
                        attr[s] += ((g.detach().float() @ wdec[s]) * rp.codes[s]).sum(dim=(0, 1))
                del rp, gs, tgt, pre
        finally:
            inference.enable_compile()
        del wdec
    G["last"] = dict(key=key, attr=attr, msets=msets)       # side channel for amp_eval_v2_checks.py (not used by the pass)
    is_act = {s: (v >= 0) for s, v in attr.items()}
    secs_roles = time.time() - t_roles
    all_attr = torch.cat([v for v in attr.values()]) if attr else torch.zeros(0)
    n_act = int(sum(int(v.sum()) for v in is_act.values())); n_inh = n_mem - n_act
    mass = float(all_attr.abs().sum()) if n_mem else 0.0
    row.update(n_activator=n_act, n_inhibitor=n_inh,
               inhibitor_mass_share=(float(all_attr[all_attr < 0].abs().sum()) / mass if mass > 0 else None),
               n_attr_zero=int((all_attr == 0).sum()) if n_mem else 0)
    acts = {s: [int(i) for i, a in zip(msets[s].tolist(), is_act[s].tolist()) if a] for s in msets}
    inhs = {s: [int(i) for i, a in zip(msets[s].tolist(), is_act[s].tolist()) if not a] for s in msets}
    acts = {s: v for s, v in acts.items() if v}; inhs = {s: v for s, v in inhs.items() if v}

    def sup_put(name, r):
        """Necessity, protocol v1 (DAN-7): (a_pos - a_{M\\C}) / (a_pos - a_empty), a_empty = the zero-ablation empty circuit
        E['e0'], one read per ratio. `_ctr_<read>` = the retired contrast-baseline form (a_pos - x) / (a_pos - A-), for comparison."""
        put(name + "_raw", r)
        for rd in ("tk", "pre"):
            ap, e, b = a_pos[rd], E["e0"][rd], A_BASE[rd]
            row[name + "_" + rd] = (ap - r[rd]) / (ap - e) if ap - e > 1e-9 else None
            row[name + "_ctr_" + rd] = (ap - r[rd]) / (ap - b) if ap - b > 1e-9 else None

    # 3. phi_sup: blind (all members -> 0), role-aware (activators -> 0, inhibitors -> negctx mean), alpha-aware
    zero_all = {s: set(d) for s, d in alphas.items() if d}
    r_blind = cfp({}, zero_all, pt_ho, pa_ho); sup_put("phi_sup_blind", r_blind)
    if n_inh:
        inh_t = {s: {i: float(x) for i, x in zip(v, means_neg[s][torch.tensor(v, device=means_neg[s].device)].cpu().tolist())} for s, v in inhs.items()}
        r_role = cfp(inh_t, {s: set(v) for s, v in acts.items()}, pt_ho, pa_ho)
        r_acto = cfp({}, {s: set(v) for s, v in acts.items()}, pt_ho, pa_ho)
        r_rel = cfp({}, {s: set(v) for s, v in inhs.items()}, pt_ho, pa_ho)
    else:
        r_role = r_acto = r_blind; r_rel = a_pos
    sup_put("phi_sup_role", r_role); sup_put("sup_activators_only", r_acto)
    put("release_raw", r_rel)
    for rd in ("tk", "pre"):         # relative change of the target when every attributed inhibitor is set to 0
        row["release_" + rd] = (r_rel[rd] / a_pos[rd] - 1) if abs(a_pos[rd]) > 1e-9 else None
    sc = {s: (v, (1.0 - amps_of[s]).clamp(min=0)) for s, v in msets.items()}
    sup_put("phi_sup_alpha", read(lambda a, b: LiveScalePatcher(sc, site, w_seed, b_seed, sl), pt_ho, pa_ho))

    # 4. phi_cf, role-aware, held-out negatives, v1's anchor (argmax of the natural seed pre-activation per negative)
    for name, use_a in (("phi_cf_alpha_role", True), ("phi_cf_a1_role", False)):
        tg = {s: {i: float((alphas[s][i] if use_a else 1.0) * float(pins_tr[s][i])) for i in v} for s, v in acts.items()}
        r = cfp(tg, {s: set(v) for s, v in inhs.items()}, nt_ho, na_ho)
        put(name + "_raw", r); ind_ratio(name, r)

    # 5. phi_pin_alpha: members CLAMPED to alpha x their clean position-wise value (unedited run, same batch)
    if msets and int(pt_ho.shape[0]) <= EVAL_BS:
        cap = {}
        k2i = {k: i for i, k in enumerate(KINDS)}

        def hook(layer_idx, activations):
            for kd in KINDS:
                s = (layer_idx, kd)
                if s in msets:
                    ta, ti = bank.encode(activations[k2i[kd]], kd, layer_idx)
                    cap[s] = sparse_topk_to_dense(ta, ti, D, dtype=torch.float32)[..., msets[s]].clone()

        inference.disable_compile()
        try:
            with torch.no_grad():
                inference.forward(pt_ho, activations_callback=hook, return_activations=False, tokenize_final=False)
        finally:
            inference.enable_compile()
        pins = LazyPins(cap, msets, tuple(pt_ho.shape[:2]))
        for name, mm, tk_, ek in (("phi_pin_alpha_zero", None, False, "e0"), ("phi_pin_alpha_topk", means_tr, True, "eM_topk"),
                                  ("phi_pin_alpha_dense", means_tr, False, "eM_dense")):
            r = co(keep, pt_ho, pa_ho, means=mm, topk=tk_, pins=pins)
            put(name + "_raw", r); ratio(name, r, E[ek])
        del cap, pins

    # 7/8. bookkeeping
    amps = np.array([a for d in alphas.values() for a in d.values()]) if n_mem else np.array([1.0])
    per_kind = {k: int(sum(len(d) for s, d in alphas.items() if s[1] == k)) for k in KINDS}
    row.update(comp_idx=int(comp), latent_idx=int(sl), n_members=n_mem, per_kind=per_kind, n_upstream_sites=len(UP),
               n_member_sites=len(msets), alpha_median=float(np.median(amps)), alpha_p10=float(np.percentile(amps, 10)),
               alpha_p90=float(np.percentile(amps, 90)),
               n_alpha_off=int((amps < 0.1).sum()) if n_mem else 0,   # members at alpha ~0: zero-ablated under a mean fill
               n_alpha_suppress=int((amps < 0.4).sum()) if n_mem else 0,   # members held well below their live value
               frac_alpha_suppress=float((amps < 0.4).mean()) if n_mem else 0.0,
               n_pos=n_pos, n_neg=n_neg, n_train=n_tr, n_ho=n_pos - n_tr,
               n_neg_train=n_ntr, n_neg_ho=n_neg - n_ntr, thin=bool(n_pos < N_SEQ or n_neg < N_SEQ),
               a_pos_blocks=a_pos_blocks, ctr_source=CTR_SOURCE, ctr_fallback=ctr_fallback,
               peak_mem_mb=(round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None),
               secs_roles=round(secs_roles, 1), secs=round(time.time() - ts, 1), v=3)
    return row


# --------------------------------------------------------------------------- selection + main
def parse_seeds(spec):
    p = Path(spec)
    try:
        is_file = ("," not in spec) and p.exists()
    except OSError:
        is_file = False
    if is_file:
        if p.suffix == ".csv":
            import pandas as pd
            return list(pd.read_csv(p)["skey"])
        return [ln.strip() for ln in open(p) if ln.strip()]
    return [s.strip() for s in spec.split(",") if s.strip()]


def neg_ctx_store_fallback(comp, sl):
    """True when the store row is the shared zero-similarity fallback list (no activating context in seq_repr)."""
    from store.context import neg_ctx
    ids, vals = neg_ctx.ctx_seq_idx[comp, sl], neg_ctx.ctx_seq_val[comp, sl]
    return bool((ids > 0).any()) and float(vals.float().abs().sum()) == 0.0


def load_items(shard_i=None):
    shard_i = SHARD_I if shard_i is None else shard_i
    if SEEDS:
        want = parse_seeds(SEEDS); wset = set(want)
        shards = None
        try:
            import pandas as pd
            ct = pd.read_parquet(TABLES / "circuits.parquet")
            ct["skey"] = ct.seed_layer.astype(str) + "." + ct.seed_kind.astype(str) + "." + ct.seed_index.astype(str)
            shards = sorted(set(ct[ct.skey.isin(wset)]["shard"]))
        except Exception:
            pass
        paths = (sorted(glob.glob(str(DATA / "discovered_circuits.shard*.pt"))) if shards is None
                 else [str(DATA / ("discovered_circuits.shard%d.pt" % i)) for i in shards])
        found = {}
        for sp in paths:
            for c in torch.load(sp, weights_only=False, map_location="cpu").values():
                k = key_of(c)
                if k in wset:
                    found[k] = c
        items = [found[k] for k in want if k in found]
        name = "amp_eval_v3_%s" % TAG
        src = "seeds:%d/%d found" % (len(items), len(want))
    else:
        sp = DATA / ("discovered_circuits.shard%d.pt" % shard_i)
        items = list(torch.load(sp, weights_only=False, map_location="cpu").values())
        name = "amp_eval_v3%s.shard%d" % ("_smoke" if SMOKE else "", shard_i)
        src = sp.name
    pool = defaultdict(set)                    # live pool per site for the random-set null (v1: from the loaded items' members)
    for c in items:
        for n in c.nodes.values():
            if n.metadata.get("role") != "seed":
                f = n.metadata["feature_id"]; pool[(f.layer, f.kind)].add(f.index)
    pool = {s: np.array(sorted(v)) for s, v in pool.items()}
    if SMOKE:
        items = items[:6]
    if PART_M > 1:
        items = items[PART_J::PART_M]; name += ".part%dof%d" % (PART_J, PART_M)
    return items, pool, OUT / (name + ".jsonl"), src


def main():
    shards = os.environ.get("SHARDS")
    if shards and not SEEDS:                   # several store shards in ONE process (engine start-up is ~3 min)
        for i in (int(x) for x in shards.split(",") if x.strip()):
            run(i)
    else:
        run(None)


def run(shard_i):
    items, pool, out_path, src = load_items(shard_i)
    done = set()
    if out_path.exists():
        for ln in open(out_path):
            try:
                r = json.loads(ln)
                if "error" not in r:
                    done.add(r["seed"])
            except Exception:
                pass
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "a")
    print("v3 | %s | %d circuits | %d already done -> %s" % (src, len(items), len(done), out_path.name), flush=True)
    t0 = time.time(); n_done = 0; n_fail = 0
    f_ = lambda x: "%.2f" % x if isinstance(x, (int, float)) and x == x else "-"
    for c in items:
        key = key_of(c)
        if key in done:
            continue
        try:
            row = score_circuit(c, pool)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            if "skip" in row:
                continue
            n_done += 1
            if n_done % 10 == 1 or SMOKE or SEEDS:
                print("[%4d] %-16s n=%4d (inh %d) | a_pos %.2f e0 %.2f | F0 %s a1 %s | fM_tk %s fN_tk %s | sup role %s blind %s a %s | cf_a %s | pin0 %s pinTk %s | rel %s | sites %s | %.1fs (%.1f s/seed avg)"
                      % (n_done, key, row["n"], row["n_inhibitor"], row["a_pos_tk"], row["e0_tk"],
                         *[f_(row.get(k + "_tk")) for k in ("free0", "free0_a1", "freeM_topk", "freeN_topk", "phi_sup_role", "phi_sup_blind", "phi_sup_alpha",
                                                            "phi_cf_alpha_role", "phi_pin_alpha_zero", "phi_pin_alpha_topk", "release")],
                         "ok" if row.get("all_sites_ablated") else "MISSING", row["secs"], (time.time() - t0) / max(n_done, 1)), flush=True)
        except Exception as e:
            n_fail += 1
            fh.write(json.dumps(dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))) + "\n"); fh.flush()
            print("[FAIL] %s: %s %s" % (key, type(e).__name__, str(e)[:200]), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if n_fail >= 5 and n_done == 0:
                sys.exit("aborting: 5 failures before any success")
    fh.close()
    print("DONE %s: %d scored, %d failed, %.0fs" % (out_path.name, n_done, n_fail, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
