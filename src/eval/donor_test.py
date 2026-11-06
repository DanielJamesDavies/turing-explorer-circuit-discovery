"""Donor-background (resample) test, protocol v2 §12.3 (Daniel, 2026-10-31): a robustness diagnostic reported for every
circuit alongside the production pass rule. It is not recursive causal scrubbing and does not replace the rule.

The test is 099's whole-donor background test as run in experiment 100 (``experiments/100-scrub-frontier/run.py``,
itself "EXACTLY 099 run_baseline"), moved into src with an explicit interface:

  bases      the 16 held-out strongest activating contexts + the 16 held-out contrast contexts (eval_probe(rec,
             "strong"): pos_tokens[n_train:], neg_tokens[n_neg_train:]); the target is read at the activating bases'
             anchors (the silent bases are run in the same batch, as 099 / 100 did, and only feed silent_firing)
  ablation   every latent outside the circuit, at every upstream site, takes a donor context's natural sparse code at
             the same position (073's ResamplePatcher, ported verbatim below): members at alpha x their live value, the
             donor's non-member active entries fill the remaining Top-K budget largest first, the current stream's SAE
             error kept live
  pools      A = the 48 strongest-train activating contexts, C = the 16 held-out contrast contexts (never a training
             donor), R = a fixed random-corpus pool filtered against every record (exact / near rule, build_random_pool)
  draws      DRAWS (8) paired draws per pool, one donor per base per draw: per draw randperm(pool)[:n_act] for the
             activating bases then randperm(pool)[:n_silent] for the silent ones, from ONE CPU torch generator seeded
             SEED + 1000 * layer + latent (099's random_seed 20261028), pools in the order A, C, R (draw_selections:
             the exact sequence of 099 / 100, so the donors are identical); the empty circuit runs under the same donors
  recovery   per pool (pooled): (mean over draws of the circuit's mean anchor read - the same for the empty circuit) /
             (natural mean anchor read - empty), None ("undefined") when the denominator is <= 1e-9 (an undefined score
             fails); read = post-Top-K activation (``*_pre`` = relu(w.x + b), recorded)
  pass       resample_pass = the production pass (eval.wcm_eval.passes on the held-out strongest row: Z, A, C in BAND,
             necessity >= NEC_MIN) AND donor recovery A and C (variant "fit") each in BAND. A subset of the production
             passes; vacuous rows are excluded from both denominators by the reader (wcm_eval.is_vacuous, protocol
             §12.4: the row's prod carries vacuous_v2 and the reads it is computed from).
  reported   R recovery; variants "a1" (members at alpha = 1) and "pinned" (members at their clean natural value on the
  with it    base x fitted alpha, 099's diagnostic of missing upstream dependencies); the per-pool empty leak; the split
             of every (draw, base) pair by whether the donor fires at the base's anchor (recovery_split); the self-donor
             identity of the empty circuit.

Edge case, not reached by 100's panel: a pool smaller than a base half (e.g. a record whose contrast held-out set has
fewer rows than its activating held-out set) is drawn as consecutive permutations (without replacement as far as the
pool allows, then the next permutation); 100 raised "donor set too small" there. The row flags it (pool_short).

Library: ``donor_target(G, key, rec, circuits, pool_tokens)`` scores several circuits of one target with the shared
work (base codes, pool codes, empty draws) done once; ``build_random_pool`` / ``record_rows`` / ``filter_pool`` build
the R pool; ``pooled`` / ``resample_pass`` / ``draw_selections`` are the pure pieces (CPU-tested,
tests/eval/test_donor_test.py).

CLI (a 062 companion: run by 062/launch.sh as SCRIPT, SHARD=i/k; imports the 062 driver for its Runner, records and
production scorer), env:
  OUT          the driver OUT holding the circuits (main/circuits/<key>.pt; sweep/circuits/<arm>/<key>.pt) and their
               production rows (main/eval.shard*.jsonl held=strong; sweep/eval.shard*.jsonl by arm). CTX_DIR, FAMILY,
               DATA, STORES, WCM_VERSION as the driver (WCM_VERSION must match OUT's marker)
  DONOR_ARMS   comma list [main]: "main", a sweep arm name (W_0.001, U_0.001, ...), or label=<path with {key}> for
               circuits elsewhere (their production row is re-scored with Runner.score)
  DONOR_TARGETS  the target list (required). Targets in [i::k] for SHARD=i/k, then 062's interleaved order
  OUT_DONOR    [OUT/donor] rows.jsonl / rows.shard<i>.jsonl (one per target x arm, resumable: a (seed, arm) row without
               "error" is done) and raw[.shard<i>].jsonl (per-base anchor reads, the donors drawn)
  DONOR_POOL   [OUT_DONOR/random_pool.pt] the R pool; built when missing (under a file lock, so one shard builds it and
               the others wait): 073's random_pool on the run's own corpus (POOL_SEED 20261016, 512 draws, >= 60
               tokens), filtered against every record in DONOR_POOL_RECORDS [the Runner's CTX_DIR]
  DONOR_DRAWS [8]  DONOR_VARIANTS [fit,a1,pinned]  LIMIT  DONOR_RESCORE=1 (re-score production instead of reading rows)
  REPORT=1     summary of OUT_DONOR's rows per arm (no GPU)

  wsl (repo root): OUT=<run> CTX_DIR=<records> WCM_VERSION=v2cr DONOR_TARGETS=<list> PYTHONPATH=src HF_HUB_OFFLINE=1 \
      CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u src/eval/donor_test.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

VERSION = "donor-test/v1 (protocol v2 §12.3; 099 run_baseline / 100 run.py whole-donor background)"
SEED = 20261028                  # 099's random_seed: generator SEED + 1000 * layer + latent
DRAWS = 8
POOLS = ("A", "C", "R")
VARIANTS = ("fit", "a1", "pinned")
BATCH = 16                       # 073 run_full / capture_codes batch
POOL_N = 512                     # 073 random_pool
POOL_SEED = 20261016
POOL_MIN_TOKENS = 60
UNDEFINED_EPS = 1e-9

Site = Tuple[int, str]


# ----------------------------------------------------------------------------------------------- pure pieces
def pooled(xs: Sequence[float], es: Sequence[float], natural: float) -> Optional[float]:
    """099 final_report / 100: (mean(out) - mean(empty)) / (natural - mean(empty)) over draws; None when undefined."""
    if not xs or not es:
        return None
    x, e = sum(xs) / len(xs), sum(es) / len(es)
    return (x - e) / (natural - e) if natural - e > UNDEFINED_EPS else None


def in_band(v: Optional[float], band: Tuple[float, float] = None) -> bool:
    from eval.wcm_eval import BAND
    lo, hi = band or BAND
    return v is not None and v == v and lo <= v <= hi


def resample_pass(prod_row: Mapping[str, Any], recovery_fit: Mapping[str, Optional[float]],
                  band: Tuple[float, float] = None) -> bool:
    """The production pass (wcm_eval.passes, post-Top-K read) AND pooled donor recovery A and C each in BAND."""
    from eval.wcm_eval import BAND, passes
    b = band or BAND
    return bool(passes(prod_row, band=b) and in_band(recovery_fit.get("A"), b) and in_band(recovery_fit.get("C"), b))


def _perm_take(n_pool: int, n: int, g: torch.Generator) -> Tuple[torch.Tensor, bool]:
    """randperm(n_pool)[:n]; a pool smaller than n takes consecutive permutations (flagged short)."""
    parts, got = [], 0
    while True:
        p = torch.randperm(n_pool, generator=g)[:n - got]
        parts.append(p)
        got += int(p.shape[0])
        if got >= n or n_pool == 0:
            break
    return torch.cat(parts), len(parts) > 1


def draw_selections(layer: int, latent: int, pool_sizes: Mapping[str, int], n_act: int, n_silent: int,
                    draws: int = DRAWS, seed: int = SEED, pools: Sequence[str] = POOLS):
    """{pool: [selection [n_act + n_silent] per draw]}, short {pool: bool}: 099 / 100's exact generator sequence (one
    CPU generator, pools in order, per draw the activating half then the silent half)."""
    g = torch.Generator().manual_seed(seed + layer * 1000 + latent)
    out, short = {}, {}
    for ds in pools:
        n = int(pool_sizes[ds])
        if n <= 0:
            raise RuntimeError("donor pool %s is empty" % ds)
        sels, sh = [], False
        for _ in range(draws):
            a, s1 = _perm_take(n, n_act, g)
            b, s2 = _perm_take(n, n_silent, g)
            sels.append(torch.cat([a, b]))
            sh = sh or s1 or s2
        out[ds], short[ds] = sels, sh
    return out, short


def split_recovery(out_ab: torch.Tensor, empty_ab: torch.Tensor, natural_b: torch.Tensor, mask: torch.Tensor):
    """Pooled recovery over the (draw, base) pairs selected by mask [draws, n_act]: (mean out - mean empty) /
    (mean natural - mean empty), natural repeated per draw; None when no pair or undefined."""
    if not bool(mask.any()):
        return None
    nat = natural_b.unsqueeze(0).expand_as(out_ab)
    x, e, a = float(out_ab[mask].double().mean()), float(empty_ab[mask].double().mean()), float(nat[mask].double().mean())
    return (x - e) / (a - e) if a - e > UNDEFINED_EPS else None


# ----------------------------------------------------------------------------------------------- 073 patcher + helpers
class ResamplePatcher:
    """073 strict_eval.ResamplePatcher (the used path, verbatim): circuit-only run with a resample fill. Members at
    alpha x live value (or at pins x alpha); every other latent at every in-scope site from the donor's natural sparse
    code at the same position (non-member donor entries, largest first, up to the Top-K budget left after the active
    members). Error term of the current stream preserved. Taps the target read."""

    def __init__(self, bank, in_scope, seed_site, w_seed, b_seed, seed_idx, keep_t, scales, donor, K, pins=None):
        self.bank, self.in_scope, self.seed_site = bank, in_scope, seed_site
        self.w_seed, self.b_seed, self.seed_idx = w_seed, b_seed, seed_idx
        self.keep_t, self.scales, self.donor, self.K = keep_t, scales, donor, int(K)
        self.pins = pins
        self.seed_pre = None
        self.tap_tk = None

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        from eval.wcm_eval import _tap
        from sae.dense import sparse_topk_to_dense
        site = (layer_idx, kind)
        if site == self.seed_site:
            self.seed_pre = _tap(self, self.bank, x, site)
        if site not in self.in_scope:
            return x
        bank = self.bank
        ta, ti = bank.encode(x, kind, layer_idx)
        D = int(bank.d_sae)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        error = x - bank.decode(dense, kind, layer_idx)
        patched = torch.zeros_like(dense)
        kt = self.keep_t.get(site)
        if kt is not None:
            if self.pins is not None:
                vals = self.pins[site][:x.shape[0]].to(dense.dtype)
            else:
                vals = dense[:, :, kt]
            sv = self.scales.get(site)
            if sv is not None:
                vals = vals * sv.to(vals.dtype)
            patched[:, :, kt] = vals
        dta, dti = self.donor[site]
        B = int(x.shape[0])
        dta, dti = dta[:B].to(dense.dtype), dti[:B]
        valid = dta > 0
        if kt is not None:
            valid = valid & ~torch.isin(dti, kt)
        budget = (self.K - (patched != 0).sum(-1)).clamp(min=0)                     # [B, T]
        order = dta.masked_fill(~valid, float("-inf")).argsort(-1, descending=True)
        dta_s, dti_s, val_s = dta.gather(-1, order), dti.gather(-1, order), valid.gather(-1, order)
        rank = val_s.long().cumsum(-1) - 1
        use = val_s & (rank < budget.unsqueeze(-1))
        patched.scatter_add_(-1, dti_s, dta_s * use.to(dta_s.dtype))
        return bank.decode(patched, kind, layer_idx) + error


def run_full(inference, make, tokens, pre=False):
    """073: target read (post-Top-K) at every position [n, T]; make(start, stop) builds the patcher per batch.
    pre=True also returns relu(w.x + b) (JumpReLU dictionaries: the threshold's activation): (tk, pre)."""
    from sae.dense import seed_activation
    out, outp = [], []
    inference.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(tokens.shape[0]), BATCH):
                tk = tokens[s0:s0 + BATCH]
                p = make(s0, s0 + int(tk.shape[0]))
                inference.forward(tk, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
                out.append(p.tap_tk.float())
                outp.append(seed_activation(p.seed_pre.float(), getattr(p, "seed_thr", None)))
    finally:
        inference.enable_compile()
    return (torch.cat(out), torch.cat(outp)) if pre else torch.cat(out)


def capture_codes(inference, bank, tokens, sites):
    """073: natural sparse codes {site: (acts [n, T, k], idx [n, T, k])} at the given sites."""
    kinds = list(bank.kinds)
    k2i = {k: i for i, k in enumerate(kinds)}
    parts = {s: ([], []) for s in sites}

    def hook(layer_idx, acts):
        for kd in kinds:
            s = (layer_idx, kd)
            if s in parts:
                ta, ti = bank.encode(acts[k2i[kd]], kd, layer_idx)
                parts[s][0].append(ta.detach().float()); parts[s][1].append(ti.detach())

    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), BATCH):
                inference.forward(tokens[s0:s0 + BATCH], activations_callback=hook, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    return {s: (torch.cat(a), torch.cat(i)) for s, (a, i) in parts.items()}


def at(tk, anchors):
    rr = torch.arange(tk.shape[0], device=tk.device)
    return tk[rr, anchors.to(tk.device).clamp(0, tk.shape[1] - 1)]


class BasePins:
    """100's LazyPins: members at their clean natural post-Top-K value on the base context, built per batch and site
    on demand from the bases' captured codes (identical values; memory O(batch x T x d_sae))."""

    def __init__(self, base_codes, kt, d_sae, a=0, z=None):
        self.base_codes, self.kt, self.D, self.a, self.z = base_codes, kt, int(d_sae), a, z

    def slice(self, a, z):
        return BasePins(self.base_codes, self.kt, self.D, a, z)

    def __getitem__(self, s):
        from sae.dense import sparse_topk_to_dense
        acts, ind = self.base_codes[s]
        dense = sparse_topk_to_dense(acts[self.a:self.z], ind[self.a:self.z], self.D, dtype=torch.float32)
        return dense[:, :, self.kt[s]]


def keep_and_scales(circuit, device):
    """({site: member index tensor (sorted)}, {site: fitted alpha per member}) as 099 / 100 build them."""
    from eval.wcm_eval import circuit_alphas
    al = circuit_alphas(circuit)
    kt = {s: torch.tensor(sorted(v), device=device) for s, v in al.items() if v}
    sc = {s: torch.tensor([al[s][int(i)] for i in kt[s].tolist()], device=device) for s in kt}
    return kt, sc


def n_members(circuit) -> int:
    return sum(1 for nd in circuit.nodes.values() if nd.metadata.get("role") != "seed")


# ----------------------------------------------------------------------------------------------- one target
@torch.no_grad()
def donor_target(G: Mapping[str, Any], key: str, rec: Mapping[str, Any], circuits: Mapping[str, Any],
                 pool_tokens: torch.Tensor, variants: Sequence[str] = VARIANTS, draws: int = DRAWS,
                 seed: int = SEED, sync=None) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """The donor test for every circuit in ``circuits`` ({arm: circuit}) of target ``key``, sharing the bases, the
    pools' codes and the empty draws. Returns ({arm: row}, {arm: raw}) (rows without the production fields)."""
    from circuit.protocol_contexts import eval_probe
    from eval.ablation_faithfulness import upstream_sites
    from eval.wcm_eval import NaturalTap, contrast_split, site_encoder
    inf, bank, dev = G["inference"], G["bank"], G["device"]
    sync = sync or (lambda: torch.cuda.synchronize() if torch.cuda.is_available() else None)
    t_all = time.time()
    layer, kind, idx = key.split(".")
    layer, idx = int(layer), int(idx)
    site = (layer, kind)
    W, b = site_encoder(bank, layer, kind)
    sites = sorted(upstream_sites(bank, layer, kind))
    K, D = int(bank.k), int(bank.d_sae)

    def taps(tokens):
        return run_full(inf, lambda a, z: NaturalTap(bank, site, W[idx], b[idx], idx), tokens.to(dev), pre=True)

    probe = eval_probe(rec, "strong", dev)
    ntr = int(probe.metadata["n_train"])
    nneg = contrast_split(len(probe.neg_tokens), probe.metadata)
    base, anchors = probe.pos_tokens[ntr:], probe.pos_argmax[ntr:]
    silent = probe.neg_tokens[nneg:]
    nb, ns = int(len(base)), int(len(silent))
    bases = torch.cat([base, silent])
    anchors = torch.cat([anchors, torch.zeros(ns, dtype=anchors.dtype, device=dev)])
    sync(); t0 = time.time()
    natural, natural_pre = taps(bases)
    natural_anchors = at(natural[:nb], anchors[:nb])
    natural_pre_anchors = at(natural_pre[:nb], anchors[:nb])
    nat_mean, nat_pre_mean = float(natural_anchors.mean()), float(natural_pre_anchors.mean())
    base_codes = capture_codes(inf, bank, bases, sites)
    specs = {}
    for arm, c in circuits.items():
        kt, sc = keep_and_scales(c, dev)
        for v in variants:
            if v == "fit":
                specs[(arm, v)] = (kt, sc, None)
            elif v == "a1":
                specs[(arm, v)] = (kt, {}, None)
            elif v == "pinned":
                specs[(arm, v)] = (kt, sc, BasePins(base_codes, kt, D))
            else:
                raise ValueError("unknown variant %r" % v)

    def execute(donors, kt, sc, pins=None):
        return run_full(inf, lambda a, z: ResamplePatcher(
            bank, set(sites), site, W[idx], b[idx], idx, kt, sc,
            {s: (v[a:z], i[a:z]) for s, (v, i) in donors.items()}, K,
            pins=pins.slice(a, z) if pins is not None else None), bases, pre=True)

    def anchor_reads(o, p):
        return at(o[:nb], anchors[:nb]), at(p[:nb], anchors[:nb]), (
            float((o[nb:] > 0).float().mean()) if ns else None)

    donor_sets = {"A": probe.pos_tokens[:ntr], "C": silent, "R": pool_tokens.to(dev)}
    sels, short = draw_selections(layer, idx, {k: len(v) for k, v in donor_sets.items()}, nb, ns, draws, seed)
    sync(); t_shared = time.time() - t0
    t_var = defaultdict(float)
    E = {ds: [] for ds in POOLS}                                   # per draw: (tk [nb], pre [nb], silent_firing)
    X = defaultdict(lambda: {ds: [] for ds in POOLS})
    fires, donor_idx, donor_act = {ds: [] for ds in POOLS}, {ds: [] for ds in POOLS}, {ds: [] for ds in POOLS}
    for ds, tokens in donor_sets.items():
        sync(); t0 = time.time()
        codes = capture_codes(inf, bank, tokens, sites)
        donor_nat, _ = taps(tokens)                                # the target read on the donor contexts themselves
        sync(); t_shared += time.time() - t0
        for draw in range(draws):
            selection = sels[ds][draw].to(dev)
            donor = {s: (v[selection], i[selection]) for s, (v, i) in codes.items()}
            sel_b = selection[:nb]
            d_at = donor_nat[sel_b, anchors[:nb].clamp(0, donor_nat.shape[1] - 1)]
            fires[ds].append((d_at > 0).cpu())
            donor_idx[ds].append(selection.cpu().tolist())
            donor_act[ds].append([round(float(v), 6) for v in d_at.tolist()])
            sync(); t0 = time.time()
            E[ds].append(anchor_reads(*execute(donor, {}, {})))
            sync(); t_shared += time.time() - t0
            for (arm, var), (kt, sc, pins) in specs.items():
                sync(); t0 = time.time()
                X[(arm, var)][ds].append(anchor_reads(*execute(donor, kt, sc, pins)))
                sync(); t_var[(arm, var)] += time.time() - t0
            del donor
        del codes, donor_nat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # self donor, empty circuit: an identity up to the decode / error round trip (099 / 100)
    sync(); t0 = time.time()
    o, _ = execute(base_codes, {}, {})
    self_empty = dict(max_abs=float((o - natural).abs().max()),
                      anchor_ratio=(float(at(o[:nb], anchors[:nb]).mean()) / nat_mean if nat_mean > UNDEFINED_EPS else None))
    sync(); t_shared += time.time() - t0
    del base_codes

    def means(lst, j=0):
        return [float(t[j].mean()) for t in lst]

    e_tk = {ds: means(E[ds]) for ds in POOLS}
    e_pre = {ds: means(E[ds], 1) for ds in POOLS}
    rows, raws = {}, {}
    n_arms = len(circuits)
    for arm, c in circuits.items():
        rec_v, rec_pre, rec_split, mean_v, silent_v = {}, {}, {}, {}, {}
        for var in variants:
            xs = X[(arm, var)]
            rec_v[var] = {ds: pooled(means(xs[ds]), e_tk[ds], nat_mean) for ds in POOLS}
            rec_pre[var] = {ds: pooled(means(xs[ds], 1), e_pre[ds], nat_pre_mean) for ds in POOLS}
            mean_v[var] = {ds: means(xs[ds]) for ds in POOLS}
            silent_v[var] = {ds: [t[2] for t in xs[ds]] for ds in POOLS}
            rec_split[var] = {}
            for ds in POOLS:
                o_ab = torch.stack([t[0].cpu() for t in xs[ds]])
                e_ab = torch.stack([t[0].cpu() for t in E[ds]])
                f_ab = torch.stack(fires[ds])
                nat_b = natural_anchors.cpu()
                rec_split[var][ds] = dict(donor_fires=split_recovery(o_ab, e_ab, nat_b, f_ab),
                                          donor_silent=split_recovery(o_ab, e_ab, nat_b, ~f_ab))
        row = dict(
            seed=key, arm=arm, layer=layer, kind=kind, n=n_members(c), n_base=nb, n_silent=ns, draws=draws,
            donor_seed=seed + layer * 1000 + idx, pool_sizes={k: int(len(v)) for k, v in donor_sets.items()},
            pool_short={k: bool(v) for k, v in short.items()},
            natural_mean=nat_mean, natural_pre_mean=nat_pre_mean,
            recovery=rec_v, recovery_pre=rec_pre, recovery_split=rec_split,
            donor_fires_frac={ds: float(torch.stack(fires[ds]).float().mean()) for ds in POOLS},
            empty_leak={ds: (sum(e_tk[ds]) / len(e_tk[ds]) / nat_mean if nat_mean > UNDEFINED_EPS else None)
                        for ds in POOLS},
            empty_means=e_tk, means=mean_v,
            silent_firing=dict(empty={ds: [t[2] for t in E[ds]] for ds in POOLS}, **silent_v),
            self_empty=self_empty,
            t_variant={v: round(t_var[(arm, v)], 2) for v in variants}, t_shared_target=round(t_shared, 2),
            n_arms_sharing=n_arms, t_target_total=round(time.time() - t_all, 2), donor_test=VERSION)
        rows[arm] = row
        raws[arm] = dict(seed=key, arm=arm, anchors=anchors[:nb].cpu().tolist(),
                         natural=[round(float(v), 6) for v in natural_anchors.tolist()],
                         donor_indices=donor_idx, donor_act_at_anchor=donor_act,
                         empty={ds: [[round(float(v), 6) for v in t[0].tolist()] for t in E[ds]] for ds in POOLS},
                         variants={var: {ds: [[round(float(v), 6) for v in t[0].tolist()] for t in X[(arm, var)][ds]]
                                         for ds in POOLS} for var in variants})
    return rows, raws


# ----------------------------------------------------------------------------------------------- the R pool
def record_rows(rec: Mapping[str, Any]) -> torch.Tensor:
    """Every token row of a context record that a donor test or a fit can see: the strongest and mid-band pools (all
    rows), the contrast rows and a rebuilt contrast held-out set (086 neg_held_v2). 099 filtered against strongest,
    mid-band and contrast (its records have no neg_held_v2)."""
    parts = [rec[p]["pos"] for p in ("strong", "mid") if rec.get(p) is not None]
    parts += [rec[k] for k in ("neg", "neg_held_v2") if isinstance(rec.get(k), torch.Tensor)]
    return torch.cat([torch.as_tensor(p).long().cpu() for p in parts]) if parts else torch.zeros((0, 64), dtype=torch.long)


def filter_pool(pool: torch.Tensor, record_row_blocks: Iterable[torch.Tensor]) -> List[int]:
    """Indices of pool rows that are neither exact nor near (heldout_rule: shared run >= 32 tokens, or >= 50% of the
    pool row's 8-grams in the record row, or >= 50% of the record row's 8-grams in the pool row) to ANY record row.
    The same rule as 099 run_baseline (pair_stats against every row), evaluated only on the (pool row, record row)
    pairs that share an 8-gram hash: every reject condition implies a shared 8-gram (an exact copy shares all, a
    32-token run 25, either 50% share at least one), so the prefilter is lossless and the result is identical."""
    from circuit import heldout_rule as HR
    pool = torch.as_tensor(pool).long().cpu()
    pg = HR._grams(pool)                                                     # [P, L-7]
    owners = defaultdict(set)
    for j, row in enumerate(pg.tolist()):
        for h in row:
            owners[h].add(j)
    uniq = torch.tensor(sorted(owners), dtype=torch.int64)
    cand = defaultdict(list)                                                 # pool row -> candidate record rows
    for O in record_row_blocks:
        O = torch.as_tensor(O).long().cpu()
        if O.numel() == 0:
            continue
        og = HR._grams(O)
        hit = torch.isin(og, uniq)
        for r in torch.nonzero(hit.any(-1)).flatten().tolist():
            js = set()
            for h in og[r][hit[r]].tolist():
                js |= owners[h]
            for j in js:
                cand[j].append(O[r])
    keep = []
    for j in range(int(pool.shape[0])):
        if j in cand:
            C = torch.unique(torch.stack(cand[j]), dim=0)
            ex, _, lcr, c8 = HR.pair_stats(pool[j][None], C)
            _, _, _, c8r = HR.pair_stats(C, pool[j][None])
            if ex.any() or HR.near_of(lcr, c8).any() or (c8r >= HR.NEAR_C8).any():
                continue
        keep.append(j)
    return keep


def random_pool_073(loader, n: int = POOL_N, seed: int = POOL_SEED, min_tokens: int = POOL_MIN_TOKENS):
    """073 strict_eval.random_pool, verbatim: n random corpus ids over the loader's shards, rows with >= min_tokens
    non-zero tokens. Returns (tokens [<= n, 64] cpu, ids)."""
    from circuit.protocol_contexts import _load
    ranges = [(a, b) for a, b in loader.shard_id_ranges if a >= 0]
    rng = random.Random(seed)
    ids = []
    while len(ids) < n:
        a, b = rng.choice(ranges)
        ids.append(rng.randint(a, b))
    pos, _, got = _load(loader, ids)
    keep = [i for i in range(pos.shape[0]) if int((pos[i] != 0).sum()) >= min_tokens]
    return pos[keep][:n].cpu(), [got[i] for i in keep][:n]


def _key_of(p: Path) -> str:
    return p.name[:-3] if p.name.endswith(".pt") else p.name


def records_digest(paths: Sequence[Path]) -> str:
    return hashlib.sha1("\n".join(sorted(_key_of(p) for p in paths)).encode()).hexdigest()


def iter_record_rows(paths: Sequence[Path], log=None):
    for n, p in enumerate(paths):
        d = torch.load(p, weights_only=False, map_location="cpu")
        for rec in d.values():
            if isinstance(rec, dict):
                yield record_rows(rec)
        if log and (n + 1) % 1000 == 0:
            log("  pool filter: %d/%d records read" % (n + 1, len(paths)))


def build_random_pool(loader, records_dir: Path, log=print) -> dict:
    """The R pool: 073's random pool on the run's own corpus, filtered against every record in records_dir."""
    t0 = time.time()
    tokens, ids = random_pool_073(loader)
    paths = sorted(Path(records_dir).glob("*.pt"))
    keep = filter_pool(tokens, iter_record_rows(paths, log))
    if len(keep) < 2 * BATCH:
        raise RuntimeError("only %d independent random donors after filtering" % len(keep))
    log("donor pool: %d of %d random corpus contexts kept after the exact / near filter against %d records (%.0fs)"
        % (len(keep), len(tokens), len(paths), time.time() - t0))
    return dict(tokens=tokens[keep], ids=[ids[j] for j in keep], retained_indices=keep, n_drawn=int(len(tokens)),
                source="073 random_pool (POOL_SEED %d, %d draws, >= %d tokens)" % (POOL_SEED, POOL_N, POOL_MIN_TOKENS),
                rule="099 exact / near (heldout_rule pair_stats; 8-gram prefilter, lossless)",
                records_dir=str(records_dir), n_records=len(paths), records_sha1=records_digest(paths),
                version=VERSION)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_or_build_pool(path: Path, loader, records_dir: Path, log=print) -> Tuple[torch.Tensor, dict]:
    """The pool at path; built (once, under an exclusive file lock: concurrent shards wait) when missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(str(path) + ".lock", "a")
    try:
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        except ImportError:                                                  # Windows: no lock (one process)
            pass
        if not path.exists():
            d = build_random_pool(loader, records_dir, log)
            tmp = path.with_name(path.name + ".%d.tmp" % os.getpid())
            torch.save(d, tmp)
            os.replace(tmp, path)
    finally:
        lock.close()
    d = torch.load(path, weights_only=False, map_location="cpu")
    info = {k: v for k, v in d.items() if k not in ("tokens", "ids", "retained_indices")}
    info["sha256"] = file_sha256(path)
    info["n"] = int(d["tokens"].shape[0])
    if records_dir is not None and d.get("records_sha1") is not None:
        paths = sorted(Path(records_dir).glob("*.pt"))
        if records_digest(paths) != d["records_sha1"]:
            log("WARNING donor pool %s was filtered against %s records (sha1 %s); %s now holds %d (sha1 %s)" % (
                path, d.get("n_records"), d["records_sha1"], records_dir, len(paths), records_digest(paths)))
            info["records_changed"] = True
    return d["tokens"].cpu(), info


# ----------------------------------------------------------------------------------------------- CLI (062 companion)
PROD_FIELDS = ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "vacuous_tk", "vacuous_v2", "e0_tk",
               "eM_topk_tk", "eN_topk_tk", "free0_pre",
               "freeM_topk_pre", "freeN_topk_pre", "phi_sup_blind_pre", "vacuous_pre", "a_pos_tk", "a_pos_pre",
               "free0_raw_pre", "phi_sup_blind_raw_pre", "wcm_version", "n")


def parse_arms(spec: str, out: Path) -> Dict[str, dict]:
    """DONOR_ARMS -> {arm: {path: template with {key}, rows: production rows source or None}}."""
    arms = {}
    for a in (x.strip() for x in spec.split(",")):
        if not a:
            continue
        if "=" in a:
            label, tmpl = a.split("=", 1)
            arms[label] = dict(path=tmpl, rows=None)
        elif a == "main":
            arms[a] = dict(path=str(out / "main" / "circuits" / "{key}.pt"), rows=("main", None))
        else:
            arms[a] = dict(path=str(out / "sweep" / "circuits" / a / "{key}.pt"), rows=("sweep", a))
    if not arms:
        raise SystemExit("DONOR_ARMS is empty")
    return arms


def production_rows(out: Path, arms: Mapping[str, dict], keys: set) -> Dict[Tuple[str, str], dict]:
    """{(seed, arm): the last non-error held-out strongest production row} from the driver's eval rows."""
    found = {}
    want = {a: d["rows"] for a, d in arms.items() if d["rows"] is not None}
    for mode in sorted({m for m, _ in want.values()}):
        for p in sorted((out / mode).glob("eval.shard*.jsonl")) + [out / mode / "eval.jsonl"]:
            if not p.exists():
                continue
            for line in open(p):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "error" in r or r.get("skip") or r.get("seed") not in keys or r.get("held", "strong") != "strong":
                    continue
                for a, (m, sweep_arm) in want.items():
                    if m == mode and (sweep_arm is None or r.get("arm") == sweep_arm):
                        found[(r["seed"], a)] = r
    return found


def summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, dict]:
    """Per arm: production / resample pass counts over the non-vacuous rows, median recovery per variant / pool."""
    import statistics
    from eval.wcm_eval import is_vacuous
    by = defaultdict(list)
    for r in rows:
        if "error" not in r and not r.get("skip"):
            by[r["arm"]].append(r)
    out = {}
    for arm, rs in sorted(by.items()):
        sc = [r for r in rs if not is_vacuous(r.get("prod", {}))]
        med = {}
        for var in VARIANTS:
            for ds in POOLS:
                vs = [r["recovery"].get(var, {}).get(ds) for r in sc if var in r.get("recovery", {})]
                vs = [v for v in vs if v is not None]
                if vs:
                    med["%s_%s" % (var, ds)] = round(statistics.median(vs), 4)
        out[arm] = dict(n=len(rs), n_scored=len(sc), prod_pass=sum(bool(r["pass_prod"]) for r in sc),
                        resample_pass=sum(bool(r["resample_pass"]) for r in sc),
                        undefined_A_C=sum(1 for r in sc if r["recovery"]["fit"].get("A") is None
                                          or r["recovery"]["fit"].get("C") is None),
                        median_recovery=med,
                        median_secs=(statistics.median(r["t_target_total"] / max(r["n_arms_sharing"], 1) for r in rs)
                                     if rs else None))
    return out


def _read_rows(d: Path, stem: str) -> List[dict]:
    rows = []
    for p in [d / ("%s.jsonl" % stem)] + sorted(d.glob("%s.shard*.jsonl" % stem)):
        if p.exists():
            for line in open(p):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def _clean(v):
    if isinstance(v, float):
        return v if v == v and abs(v) != float("inf") else None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def main():
    # run as a script, sys.path[0] is src/eval, whose sae/ package shadows src/sae: put src first instead
    here = Path(__file__).resolve().parent
    sys.path[:] = [str(here.parent)] + [p for p in sys.path if p and Path(p).resolve() != here]
    out = Path(os.environ.get("OUT", "")) if os.environ.get("OUT") else None
    if out is None:
        raise SystemExit("OUT (the driver run holding the circuits) is required")
    out_donor = Path(os.environ.get("OUT_DONOR") or (out / "donor"))
    if os.environ.get("REPORT"):
        print(json.dumps(summary(_read_rows(out_donor, "rows")), indent=1))
        return
    if not os.environ.get("DONOR_TARGETS"):
        raise SystemExit("DONOR_TARGETS (the target list) is required")
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "experiments" / "062-h100-protocol-v1"))
    import driver  # noqa: E402  (reads OUT / CTX_DIR / FAMILY / WCM_VERSION at import)
    from eval.wcm_eval import passes

    arms = parse_arms(os.environ.get("DONOR_ARMS", "main"), out)
    variants = tuple(v for v in os.environ.get("DONOR_VARIANTS", ",".join(VARIANTS)).split(",") if v)
    if "fit" not in variants:
        raise SystemExit("DONOR_VARIANTS must include fit (the resample pass reads it)")
    draws = int(os.environ.get("DONOR_DRAWS", DRAWS))
    limit = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
    rescore = os.environ.get("DONOR_RESCORE") == "1"
    out_donor.mkdir(parents=True, exist_ok=True)
    rows_path, raw_path = driver.shard_jsonl(out_donor, "rows"), driver.shard_jsonl(out_donor, "raw")
    done = {(r.get("seed"), r.get("arm")) for r in _read_rows(out_donor, "rows") if "error" not in r}
    keys = [k for k in Path(os.environ["DONOR_TARGETS"]).read_text().split() if k][driver.SHARD_I::driver.SHARD_K]
    keys = driver.interleave(keys, driver.SLOT)
    if limit:
        keys = keys[:limit]
    todo = [(k, [a for a in arms if (k, a) not in done]) for k in keys]
    todo = [(k, a) for k, a in todo if a]

    def log(s):
        print("[%s %s] %s" % (time.strftime("%H:%M:%S"), driver.TAG, s), flush=True)
    log("donor test: %d targets (%d with work) x arms %s | variants %s | %d draws | out %s | WCM %s" % (
        len(keys), len(todo), list(arms), list(variants), draws, out_donor, driver.WCM_VERSION))
    if not todo:
        log("DONE (nothing to do)")
        return
    R = driver.Runner()
    G = R.G
    pool_path = Path(os.environ.get("DONOR_POOL") or (out_donor / "random_pool.pt"))
    pool_records = Path(os.environ.get("DONOR_POOL_RECORDS") or R.ctx_dir)
    pool, pool_info = load_or_build_pool(pool_path, G["M0"].probe_builder.loader, pool_records, log)
    log("donor pool %s: %d contexts, sha256 %s" % (pool_path, pool_info["n"], pool_info["sha256"][:12]))
    prod = {} if rescore else production_rows(out, arms, {k for k, _ in todo})
    fh, fr = open(rows_path, "a"), open(raw_path, "a")
    t_run = time.time()
    for n_, (key, todo_arms) in enumerate(todo):
        t0 = time.time()
        circuits, errs, skipped = {}, {}, set()
        try:
            rec = R.contexts(key)
            if rec.get("strong") is None:
                for a in todo_arms:
                    fh.write(json.dumps(dict(seed=key, arm=a, skip="no_strong_pool")) + "\n")
                fh.flush()
                continue
            for a in todo_arms:
                p = Path(arms[a]["path"].format(key=key))
                if not p.exists():
                    errs[a] = "FileNotFoundError: no circuit %s" % p
                    continue
                c = torch.load(p, weights_only=False)
                if c is None:
                    fh.write(json.dumps(dict(seed=key, arm=a, skip="rejected", circuit=str(p))) + "\n")
                    skipped.add(a)                                 # done (skip row): no error row for it below
                    continue
                circuits[a] = (c, p)
            fh.flush()
            if circuits:
                rows, raws = donor_target(G, key, rec, {a: c for a, (c, _) in circuits.items()}, pool,
                                          variants=variants, draws=draws)
                for a, (c, p) in circuits.items():
                    pr = prod.get((key, a))
                    src = "eval_rows"
                    if pr is None:
                        pr, src = R.score(c, rec, "strong"), "rescored"
                    rows[a].update(prod={k: pr.get(k) for k in PROD_FIELDS if k in pr}, prod_source=src,
                                   pass_prod=bool(passes(pr)), resample_pass=resample_pass(pr, rows[a]["recovery"]["fit"]),
                                   circuit=str(p), wcm_version=pr.get("wcm_version", driver.WCM_VERSION),
                                   run_wcm_version=driver.WCM_VERSION, pool_sha256=pool_info["sha256"],
                                   pool_n=pool_info["n"], shard=driver.TAG)
                for a in circuits:                     # written only once every arm of the target is complete
                    fr.write(json.dumps(_clean(raws[a]), allow_nan=False) + "\n")
                    fh.write(json.dumps(_clean(rows[a]), allow_nan=False) + "\n")
                fr.flush(); fh.flush()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            for a in todo_arms:
                if a not in errs and a not in skipped:
                    errs[a] = "%s: %s" % (type(e).__name__, str(e)[:300])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for a, msg in errs.items():
            fh.write(json.dumps(dict(seed=key, arm=a, error=msg, shard=driver.TAG)) + "\n")
        fh.flush()
        el = time.time() - t_run
        g = lambda v: "-" if v is None else "%.3f" % v  # noqa: E731
        fit = {a: rows[a]["recovery"]["fit"] for a in circuits} if circuits and not errs else {}
        log("%d/%d %-16s %s %.1fs | %s | ~%.0f min left" % (
            n_ + 1, len(todo), key, "error" if errs else "ok", time.time() - t0,
            " ".join("%s A %s C %s R %s" % (a, g(v.get("A")), g(v.get("C")), g(v.get("R"))) for a, v in fit.items()),
            el / (n_ + 1) * (len(todo) - n_ - 1) / 60))
    log("DONE %.0fs" % (time.time() - t_run))


if __name__ == "__main__":
    main()
