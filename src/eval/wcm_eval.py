"""Protocol-v1 circuit scorer (DAN-73): the evaluation every protocol-v1 paper number is computed with.

This is the 062 run's scorer, ``experiments/049-circuit-graph/amp_eval_pass_v2.py: score_circuit(c, {},
skip_roles=True)``, moved into src with an explicit interface: the caller passes the inference engine, the SAE bank,
the circuit, the eval ProbeDataset and the contrast tokens, instead of module globals and a patched M0. Rows carry
the same fields with the same names and the same numbers (experiments/067-src-scorer-equivalence checks this on
the run's own circuits), so merge.py / pass_rule.py-style analysis reads them unchanged. Spec: paper/protocol-v1.md
§5 (evaluation) and §6 (pass rule); the pass rule, vacuity and graded companion are at the bottom of this module,
the specificity diagnostic and near-threshold stratum in eval/wcm_specificity.py.

Contexts. ``probe.pos_tokens`` = the reference (train) contexts followed by the held-out ones, ``pos_argmax`` their
anchors; protocol v1: the 48 strongest-train contexts then the 16 held-out strongest or mid-band ones
(circuit/protocol_contexts.eval_probe). Contrast rows: first split_n(n) train, the rest held out.

  split="explicit"  (default, every new run) the train count is ``probe.metadata["n_train"]``, so the held-out slice
                    is exactly the held-out set.
  split="as_run"    the 15k run's 3:1 rule, split_n(n_pos). On a short probe (mid-band with fewer held-out contexts
                    than the padding expects) it moves strongest-train contexts into the held-out slice
                    (protocol-v1.md §3.6, 58 mid-band rows). Kept only to reproduce the run.

Reads. Every measurement is taken under both reads of the target at the anchor and every ratio uses ONE read in all
its terms (DAN-66): ``<name>_tk`` the post-Top-K activation (PRIMARY: every pass / fail number) and ``<name>_pre``
relu(w.x + b) (diagnostic). Raw reads are ``<name>_raw_tk`` / ``<name>_raw_pre``.

Circuit-only runs go through the engine's CircuitOnlyPatcher with every upstream site in scope (DAN-67): members at
alpha x their re-encoded live value (or pinned), every other latent at the fill: zero, dense mean, or the
sparsity-preserving fill whose budget is the dictionary's Top-K k (``bank.k``, per model; 128 on TuringLLM). Live-
stream edits (necessity, sufficiency to induce) go through CounterfactualInterventionPatcher. Both are subclassed
only to tap the target read.

Scores (protocol-v1.md §5.2): free0 / freeM_topk / freeN_topk are Z / A / C (the pass rule), with freeM_dense,
freeN_dense, free0_tr, free0_a1 and free0_perm recorded; phi_sup_blind is necessity (a_pos - a_{M\\C}) /
(a_pos - e0); phi_cf_alpha_blind is sufficiency to induce on held-out contrast contexts; phi_sup_alpha and
phi_pin_alpha_{zero,topk,dense} are recorded diagnostics.

Out of scope (DAN-72, protocol v1 runs with skip_roles=True): role attribution, role-aware necessity and inhibitor
release are not computed. Their fields are still written, with the values the run wrote: every member counts as an
activator (n_activator = n, n_inhibitor = 0), phi_sup_role = sup_activators_only = phi_sup_blind, release = 0, and
phi_cf_alpha_role / phi_cf_a1_role are role-blind (every member set at alpha / 1 x its pin). Misleading names kept
for compatibility: ``free0_rand`` is the EMPTY circuit (the run passed an empty live pool, so no random set is
drawn; the paper's random baseline is DAN-17), and ``thin`` here means n_pos < n_seq or n_neg < n_seq (the 062
driver overwrote it with "no mid-band pool").

  from circuit.protocol_contexts import eval_probe
  row = score_circuit(inference, bank, circuit, eval_probe(rec, "strong", device))
  row.update(held="strong")
  passes(row), is_vacuous(row)
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import time
import zlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set, Tuple

import numpy as np
import torch

from eval.ablation_faithfulness import CircuitOnlyPatcher, upstream_sites
from eval.counterfactual_faithfulness import CounterfactualInterventionPatcher
from eval.floors import collect_site_anchors, collect_site_means
from model.hooks import multi_patch
from pipeline.component_index import component_idx
from sae.dense import sparse_topk_to_dense, target_latent_activations

Site = Tuple[int, str]
Alphas = Dict[Site, Dict[int, float]]

READS = ("tk", "pre")            # tk = post-Top-K activation (primary), pre = relu(w.x + b) (diagnostic)
SPLITS = ("explicit", "as_run")
N_SEQ = 64                       # activating / contrast rows read from the probe (the run's cap)
EVAL_BATCH = 16                  # sequences per forward; also the largest held-out set phi_pin is measured on
MIN_POS = 8                      # fewer activating contexts: skip "thin_probes"
HOLDOUT_FRAC = 0.25              # the engine split rule: n_hold = round(n * 0.25), train = the first n - n_hold
VACUOUS_REL = 0.05               # vacuous denominator: |a_pos - e0| < 0.05 |a_pos|

# DAN-8 pass rule (adopted 2026-09-27): Z, A, C all in BAND and necessity >= NEC_MIN, held-out strongest
BAND = (0.8, 1.5)
NEC_MIN = 0.9
FAITH = ("free0", "freeM_topk", "freeN_topk")           # Z, A (sparsity-preserving), C (sparsity-preserving)
NECESSITY = "phi_sup_blind"
INDUCE = "phi_cf_alpha_blind"
TIMING_FIELDS = ("secs", "secs_roles", "peak_mem_mb")   # machine-dependent row fields


# ----------------------------------------------------------------------------------------------- small maths
def frac(v: float, lo: float, hi: float) -> Optional[float]:
    """(v - lo) / (hi - lo); None when the denominator is (numerically) zero."""
    return (v - lo) / (hi - lo) if abs(hi - lo) > 1e-9 else None


def ratio(x: float, e: float, a_pos: float) -> Optional[float]:
    """The faithfulness form (x - e) / (a_pos - e); None unless the baseline e is below a_pos."""
    return frac(x, e, a_pos) if e < a_pos else None


def necessity(x: float, a_pos: float, e: float) -> Optional[float]:
    """Necessity, protocol v1 (DAN-7): (a_pos - x) / (a_pos - e), x = the target with the members removed, e = the
    empty circuit; None unless a_pos - e > 1e-9."""
    return (a_pos - x) / (a_pos - e) if a_pos - e > 1e-9 else None


def vacuous(a_pos: float, e0: float, rel: float = VACUOUS_REL) -> bool:
    """A vacuous denominator: the empty circuit already sits within rel x a_pos of a_pos (relative to a_pos; v1's
    rule relative to e0 never fired, e0_tk is usually 0)."""
    return bool(abs(a_pos - e0) < rel * abs(a_pos))


def split_n(n: int, holdout_frac: float = HOLDOUT_FRAC) -> int:
    """Train count of the engine's 3:1 rule: n - round(n * holdout_frac), at least 1 (Python's round: half to even)."""
    n_hold = int(round(n * holdout_frac))
    return max(1, n - n_hold)


def eval_split(n_pos: int, metadata: Optional[Mapping[str, Any]], split: str = "explicit",
               holdout_frac: float = HOLDOUT_FRAC) -> int:
    """The activating train count. "explicit": ``metadata["n_train"]`` (set by protocol_contexts.probe_from), which
    must leave both a train and a held-out slice. "as_run": split_n(n_pos)."""
    if split == "as_run":
        return split_n(n_pos, holdout_frac)
    if split != "explicit":
        raise ValueError("split must be one of %s, got %r" % (SPLITS, split))
    n_tr = (metadata or {}).get("n_train")
    if not n_tr:
        raise ValueError("split='explicit' needs probe.metadata['n_train'] (protocol_contexts.probe_from sets it)")
    n_tr = int(n_tr)
    if not 0 < n_tr < n_pos:
        raise ValueError("explicit split needs both a train and a held-out slice (n_train %d, n %d)" % (n_tr, n_pos))
    return n_tr


# ----------------------------------------------------------------------------------------------- circuit helpers
def seed_of(circuit: Any) -> Any:
    return next(n for n in circuit.nodes.values() if n.metadata.get("role") == "seed")


def circuit_key(circuit: Any) -> str:
    """The target's key "<layer>.<kind>.<latent>"."""
    sf = seed_of(circuit).metadata["feature_id"]
    return "%d.%s.%d" % (sf.layer, sf.kind, sf.index)


def circuit_alphas(circuit: Any) -> Alphas:
    """{site: {latent: alpha}} over the members in node order (alpha = the node's fitted ``amplitude``, 1 when it
    has none: unweighted circuits). The order seeds free0_perm's permutation, so it must stay the node order."""
    seed = seed_of(circuit)
    alphas: Dict[Site, Dict[int, float]] = defaultdict(dict)
    for n in circuit.nodes.values():
        if n is seed:
            continue
        f = n.metadata["feature_id"]
        alphas[(f.layer, f.kind)][int(f.index)] = float(n.metadata.get("amplitude", 1.0))
    return dict(alphas)


def site_encoder(bank: Any, layer: int, kind: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The site's encoder (W [d_sae, d_in], b_eff [d_sae]); row i gives latent i's pre-activation w.x + b."""
    sae = bank.saes[kind][layer]
    return sae.encoder.weight.detach(), sae._get_bias_eff().detach()


# ----------------------------------------------------------------------------------------------- patchers
def _tap(p: Any, bank: Any, x: torch.Tensor, site: Site) -> torch.Tensor:
    """Target read, both conventions, from the (already edited) stream at the target site: returns the
    pre-activation [B, T] and stores the post-Top-K activation on p.tap_tk."""
    w = p.w_seed.to(device=x.device, dtype=x.dtype)
    b = p.b_seed.to(device=x.device, dtype=x.dtype)
    pre = x @ w + b
    with torch.no_grad():
        ta, ti = bank.encode(x.detach(), site[1], site[0])
        p.tap_tk = target_latent_activations(ta, ti, p.seed_idx).detach()
    return pre


class NaturalTap:
    """The unedited stream; the target read is tapped at its site (the run's AmpInjectPatcher with nothing to
    inject)."""

    def __init__(self, bank: Any, seed_site: Site, w_seed: torch.Tensor, b_seed: torch.Tensor, seed_idx: int):
        self.bank, self.seed_site = bank, seed_site
        self.w_seed, self.b_seed, self.seed_idx = w_seed, b_seed, seed_idx
        self.seed_pre: Optional[torch.Tensor] = None
        self.tap_tk: Optional[torch.Tensor] = None

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if (layer_idx, kind) == self.seed_site:
            self.seed_pre = _tap(self, self.bank, x, self.seed_site)
        return x


class LiveScalePatcher(NaturalTap):
    """Members scaled by a per-member factor in the otherwise-live stream (phi_sup_alpha). Engine form:
    decode(patched) + (x - decode(natural)), SAE error preserved, no cap or clamp."""

    def __init__(self, bank: Any, scales: Dict[Site, Tuple[torch.Tensor, torch.Tensor]], seed_site: Site,
                 w_seed: torch.Tensor, b_seed: torch.Tensor, seed_idx: int):
        super().__init__(bank, seed_site, w_seed, b_seed, seed_idx)
        self.scales = scales

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if (layer_idx, kind) == self.seed_site:
            self.seed_pre = _tap(self, self.bank, x, self.seed_site)
            return x
        sc = self.scales.get((layer_idx, kind))
        if sc is None:
            return x
        idx, fac = sc
        bank = self.bank
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        error = x - bank.decode(dense, kind, layer_idx)
        patched = dense.clone()
        patched[..., idx] = dense[..., idx] * fac.to(dense.dtype)
        return bank.decode(patched, kind, layer_idx) + error


class TapCircuitOnly(CircuitOnlyPatcher):
    """The engine's CircuitOnlyPatcher, tapping the target read and recording which sites it ablated (DAN-67 scope
    check: must equal the upstream set)."""

    def __init__(self, *a, w_seed: torch.Tensor = None, b_seed: torch.Tensor = None, **k):
        super().__init__(*a, **k)
        self.w_seed, self.b_seed, self.seed_idx = w_seed, b_seed, self.seed_latent_idx
        self.seed_pre: Optional[torch.Tensor] = None
        self.tap_tk: Optional[torch.Tensor] = None
        self.edited: Set[Site] = set()

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            self.seed_pre = _tap(self, self.bank, x, (layer_idx, kind))
        if (layer_idx, kind) in self.in_scope:
            self.edited.add((layer_idx, kind))
        return super().transform(layer_idx, kind, x)


class TapIntervention(CounterfactualInterventionPatcher):
    """The engine's CounterfactualInterventionPatcher (named latents SET to values or 0 in the live stream),
    tapping the target read."""

    def __init__(self, *a, w_seed: torch.Tensor = None, b_seed: torch.Tensor = None, **k):
        super().__init__(*a, **k)
        self.w_seed, self.b_seed, self.seed_idx = w_seed, b_seed, self.seed_latent_idx
        self.seed_pre: Optional[torch.Tensor] = None
        self.tap_tk: Optional[torch.Tensor] = None

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            self.seed_pre = _tap(self, self.bank, x, (layer_idx, kind))
        return super().transform(layer_idx, kind, x)


class LazyPins:
    """Position-specific pins [B, T, d_sae] for the engine's pinned mode, materialised per site on demand from the
    captured member columns (a dense clean stream per site would be ~6 GB on TuringLLM)."""

    def __init__(self, cap: Dict[Site, torch.Tensor], idx: Dict[Site, torch.Tensor], shape: Tuple[int, int],
                 d_sae: int):
        self.cap, self.idx, self.shape, self.d_sae = cap, idx, shape, int(d_sae)

    def __getitem__(self, site: Site) -> torch.Tensor:
        z = torch.zeros(*self.shape, self.d_sae, device=self.cap[site].device, dtype=torch.float32)
        z[..., self.idx[site]] = self.cap[site]
        return z


# ----------------------------------------------------------------------------------------------- reads
def read_target(inference: Any, make_patcher: Callable[[int, int], Any], tokens: torch.Tensor,
                anchors: torch.Tensor, batch_size: int = EVAL_BATCH, per_seq: bool = False):
    """Run ``tokens`` in batch_size chunks under make_patcher(start, stop); target read at ``anchors``.
    Returns {"pre": mean relu(pre), "tk": mean post-Top-K, "edited": sites ablated}, or the per-sequence tensors."""
    pres, tks = [], []
    edited: Set[Site] = set()
    inference.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(tokens.shape[0]), batch_size):
                tk = tokens[s0:s0 + batch_size]
                B = int(tk.shape[0])
                p = make_patcher(s0, s0 + B)
                inference.forward(tk, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
                pre = p.seed_pre
                rr = torch.arange(B, device=pre.device)
                anc = anchors[s0:s0 + B].to(pre.device).clamp(0, pre.shape[1] - 1)
                pres.append(torch.relu(pre[rr, anc]).float())
                tks.append(p.tap_tk[rr, anc].float())
                edited |= getattr(p, "edited", set())
    finally:
        inference.enable_compile()
    pres, tks = torch.cat(pres), torch.cat(tks)
    if per_seq:
        return pres, tks
    return {"pre": float(pres.mean()), "tk": float(tks.mean()), "edited": edited}


# ----------------------------------------------------------------------------------------------- one circuit
@torch.no_grad()
def score_circuit(
    inference: Any,
    bank: Any,
    circuit: Any,
    probe: Any,
    contrast_tokens: Optional[torch.Tensor] = None,
    *,
    split: str = "explicit",
    batch_size: int = EVAL_BATCH,
    n_seq: int = N_SEQ,
    min_pos: int = MIN_POS,
    topk: Optional[int] = None,
    alphas: Optional[Alphas] = None,
    seed_vector: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ctr_source: str = "close",
    ctr_fallback: Optional[bool] = None,
    holdout_frac: float = HOLDOUT_FRAC,
) -> Dict[str, Any]:
    """Score one circuit on one eval set; returns the protocol-v1 row (see the module docstring), or a skip row
    {seed, skip: "thin_probes" | "no_negatives", ...}.

    probe            ProbeDataset: reference (train) contexts then held-out ones, with anchors; None = no contexts
    contrast_tokens  contrast contexts (train first, held out last); None = probe.neg_tokens
    split            "explicit" (probe.metadata["n_train"]) or "as_run" (the 15k run's 3:1 rule; §3.6 defect)
    batch_size       sequences per forward (the run: 16). phi_pin is measured only when the held-out set fits one batch
    n_seq            cap on activating and contrast rows (the run: 64)
    topk             the sparsity-preserving fill budget; None = the dictionary's own k (``bank.k``)
    alphas           {site: {latent: alpha}} overriding the circuit's fitted coefficients (e.g. alpha = 1 rescoring)
    seed_vector      (w, b) of the target's pre-activation; None = row ``latent`` of the site's encoder
    ctr_source, ctr_fallback   provenance labels written to the row (the run: "close", and whether the target's
                     neg_ctx store row is the shared fallback list; None when not known)
    """
    ts = time.time()
    md = circuit.metadata
    seed = seed_of(circuit)
    sf = seed.metadata["feature_id"]
    layer, kind, sl = sf.layer, sf.kind, sf.index
    key = "%d.%s.%d" % (layer, kind, sl)
    kinds = list(bank.kinds)
    D = int(bank.d_sae)
    K = int(bank.k) if topk is None else int(topk)
    rng = random.Random(zlib.crc32(key.encode()))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    alphas = circuit_alphas(circuit) if alphas is None else alphas
    n_mem = sum(len(d) for d in alphas.values())
    comp = component_idx(layer, kinds.index(kind), len(kinds))
    n_pos = 0 if probe is None else int(probe.pos_tokens.shape[0])
    if n_pos < min_pos:
        return dict(seed=key, skip="thin_probes", n=n_mem, n_pos=n_pos)
    pt, pa = probe.pos_tokens[:n_seq], probe.pos_argmax[:n_seq]
    nt = (probe.neg_tokens if contrast_tokens is None else contrast_tokens)[:n_seq]
    n_pos = int(pt.shape[0])
    n_neg = int(nt.shape[0])
    if n_neg < 2:
        return dict(seed=key, skip="no_negatives", n=n_mem, n_pos=n_pos, n_neg=n_neg)
    n_tr = eval_split(n_pos, probe.metadata, split, holdout_frac)
    n_ntr = split_n(n_neg, holdout_frac)
    pt_tr, pa_tr, pt_ho, pa_ho = pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:]
    nt_tr, nt_ho = nt[:n_ntr], nt[n_ntr:]
    if seed_vector is None:
        W, b = site_encoder(bank, layer, kind)
        w_seed, b_seed = W[sl], b[sl]
    else:
        w_seed, b_seed = seed_vector
    UP = sorted(upstream_sites(bank, layer, kind))
    UPS = set(UP)
    site = (layer, kind)
    SK = dict(w_seed=w_seed, b_seed=b_seed)
    dev = pt.device

    msets = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in alphas.items() if d}
    amps_of = {s: torch.tensor([alphas[s][int(i)] for i in v.tolist()], device=dev, dtype=torch.float32)
               for s, v in msets.items()}
    keep = {s: set(d) for s, d in alphas.items() if d}
    scales = {}
    for s, v in msets.items():
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        sv[v] = amps_of[s]
        scales[s] = sv

    def co(keep_, tokens, anchors, means=None, fill_topk=False, use_scales=True, pins=None, scales_=None):
        """Circuit-only run: members at alpha x live (or pinned) value; every other latent at every upstream site at
        the fill (zero when means is None; sparsity-preserving with budget K when fill_topk). scales_ overrides the
        fitted gains."""
        kt = {s: msets[s] for s in keep_} if keep_ is keep else {
            s: torch.tensor(sorted(i), device=dev, dtype=torch.long) for s, i in keep_.items() if i}
        if pins is not None:
            assert int(tokens.shape[0]) <= batch_size, "position-specific pins are captured for one batch"
        ks = scales_ if scales_ is not None else (scales if (use_scales and keep_) else None)
        return read_target(inference, lambda a, b: TapCircuitOnly(
            bank=bank, keep_indices=keep_, in_scope=UPS, seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
            pos_argmax=anchors[a:b], site_means=means, pin_values=pins, respect_topk=fill_topk, topk=K,
            keep_tensors=kt, keep_scales=ks, **SK), tokens, anchors, batch_size)

    def cfp(targets, zero, tokens, anchors):
        """Live-stream edit: ``targets`` SET to values, ``zero`` SET to 0, everything else re-encoded live."""
        return read_target(inference, lambda a, b: TapIntervention(
            bank=bank, activator_targets=targets, inhibitor_indices=zero, seed_layer=layer, seed_kind=kind,
            seed_latent_idx=sl, pos_argmax=anchors[a:b], **SK), tokens, anchors, batch_size)

    # ---- natural stream: a_pos on held-out and train, both reads; per-block a_pos (split-ordering diagnostic)
    nat_pre, nat_tk = read_target(inference, lambda a, b: NaturalTap(bank, site, w_seed, b_seed, sl), pt, pa,
                                  batch_size, per_seq=True)
    a_pos = {"pre": float(nat_pre[n_tr:].mean()), "tk": float(nat_tk[n_tr:].mean())}
    a_pos_tr = {"pre": float(nat_pre[:n_tr].mean()), "tk": float(nat_tk[:n_tr].mean())}
    a_pos_blocks = [round(float(nat_tk[i:i + 16].mean()), 4) for i in range(0, n_pos, 16)]

    # ---- ablation values: A means + collapsed pins (train activating rows), C means (train contrast rows)
    means_tr, pins_tr = collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)
    means_neg = collect_site_means(inference, bank, nt_tr, UPS) if UP else {}

    # ---- empty-circuit baselines (every upstream site ablated), both reads
    E = {"e0": co({}, pt_ho, pa_ho), "eM_dense": co({}, pt_ho, pa_ho, means=means_tr),
         "eM_topk": co({}, pt_ho, pa_ho, means=means_tr, fill_topk=True),
         "eN_dense": co({}, pt_ho, pa_ho, means=means_neg),
         "eN_topk": co({}, pt_ho, pa_ho, means=means_neg, fill_topk=True)}
    e0_tr = co({}, pt_tr, pa_tr)

    # ---- held-out contrast: natural run, would-be-firing anchors (argmax of the target pre-activation), A-
    neg_full = []
    inference.disable_compile()
    try:
        for s0 in range(0, int(nt_ho.shape[0]), batch_size):
            p0 = NaturalTap(bank, site, w_seed, b_seed, sl)
            inference.forward(nt_ho[s0:s0 + batch_size], patcher=p0, grad_enabled=False, return_activations=False,
                              tokenize_final=False)
            neg_full.append((p0.seed_pre.detach(), p0.tap_tk))
    finally:
        inference.enable_compile()
    neg_pre = torch.cat([a for a, _ in neg_full], 0)
    neg_tk = torch.cat([b for _, b in neg_full], 0)
    na_ho = neg_pre.argmax(dim=1).cpu()
    rr = torch.arange(neg_pre.shape[0], device=neg_pre.device)
    A_BASE = {"pre": float(torch.relu(neg_pre[rr, na_ho.to(neg_pre.device)]).float().mean()),
              "tk": float(neg_tk[rr, na_ho.to(neg_tk.device)].float().mean())}
    pins_tr = {s: pins_tr[s].cpu() for s in alphas if s in pins_tr}
    inject = {s: {i: float(a * float(pins_tr[s][i])) for i, a in d.items()} for s, d in alphas.items() if s in pins_tr}

    row: Dict[str, Any] = dict(seed=key, layer=layer, kind=kind, n=n_mem,
                               amp_median=(md.get("amp_stats") or {}).get("median"),
                               cf_bare_legacy=(md.get("evals") or {}).get("counterfactual_faithfulness"))

    def put(name, r):
        row[name + "_pre"] = r["pre"]
        row[name + "_tk"] = r["tk"]

    def put_ratio(name, r, e, ap=None):
        ap = a_pos if ap is None else ap
        for rd in ("tk", "pre"):
            row[name + "_" + rd] = ratio(r[rd], e[rd], ap[rd])

    def put_induce(name, r):
        """Sufficiency to induce (A-_int - A-) / (A+ - A-) on held-out contrast contexts."""
        for rd in ("tk", "pre"):
            row[name + "_" + rd] = ratio(r[rd], A_BASE[rd], a_pos[rd])

    def put_necessity(name, r):
        """Necessity against the zero-ablation empty circuit e0; ``_ctr_<read>`` = the retired contrast-baseline
        form (a_pos - x) / (a_pos - A-), recorded for comparison."""
        put(name + "_raw", r)
        for rd in ("tk", "pre"):
            row[name + "_" + rd] = necessity(r[rd], a_pos[rd], E["e0"][rd])
            row[name + "_ctr_" + rd] = necessity(r[rd], a_pos[rd], A_BASE[rd])

    put("a_pos", a_pos); put("a_pos_tr", a_pos_tr); put("a_base", A_BASE); put("e0_tr", e0_tr)
    for k_, r in E.items():
        put(k_, r)
    for rd in ("tk", "pre"):
        row["vacuous_" + rd] = vacuous(a_pos[rd], E["e0"][rd])

    # 1. faithfulness at the fitted gains: Z, dense / sparsity-preserving A and C
    runs = {"free0": (None, False, "e0"), "freeM_dense": (means_tr, False, "eM_dense"),
            "freeM_topk": (means_tr, True, "eM_topk"), "freeN_dense": (means_neg, False, "eN_dense"),
            "freeN_topk": (means_neg, True, "eN_topk")}
    for name, (mm, tk_, ek) in runs.items():
        r = co(keep, pt_ho, pa_ho, means=mm, fill_topk=tk_)
        put(name + "_raw", r); put_ratio(name, r, E[ek])
        if name == "free0":
            row["all_sites_ablated"] = bool(r["edited"] == UPS)
    r = co(keep, pt_tr, pa_tr)
    put("free0_tr_raw", r); put_ratio("free0_tr", r, e0_tr, a_pos_tr)
    r = co(keep, pt_ho, pa_ho, use_scales=False)
    put("free0_a1_raw", r); put_ratio("free0_a1", r, E["e0"])

    # 1b. coefficient check: alphas permuted within each site (per-target RNG, so a rerun reproduces it)
    perm_scales = {}
    for s, v in msets.items():
        vals = amps_of[s].tolist()
        rng.shuffle(vals)
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        sv[v] = torch.tensor(vals, device=dev, dtype=torch.float32)
        perm_scales[s] = sv
    r = co(keep, pt_ho, pa_ho, scales_=perm_scales) if msets else E["e0"]
    put("free0_perm_raw", r); put_ratio("free0_perm", r, E["e0"])
    # free0_rand: the EMPTY circuit (the run's empty live pool draws no random set); kept for row compatibility
    put("free0_rand_raw", E["e0"]); put_ratio("free0_rand", E["e0"], E["e0"])

    # 1c. sufficiency to induce, role-blind: every member SET to alpha x its train pin on held-out contrast contexts
    r = cfp(inject, {}, nt_ho, na_ho) if inject else {"pre": A_BASE["pre"], "tk": A_BASE["tk"]}
    put("phi_cf_alpha_blind_raw", r); put_induce("phi_cf_alpha_blind", r)

    # 2. roles: not computed under protocol v1 (DAN-72); every member counts as an activator
    row.update(n_activator=n_mem, n_inhibitor=0, inhibitor_mass_share=None, n_attr_zero=n_mem)

    # 3. necessity: every member SET to 0 in the live stream; alpha-aware: members scaled by max(0, 1 - alpha)
    zero_all = {s: set(d) for s, d in alphas.items() if d}
    r_blind = cfp({}, zero_all, pt_ho, pa_ho)
    put_necessity("phi_sup_blind", r_blind)
    put_necessity("phi_sup_role", r_blind)              # role-aware == blind without roles (as run)
    put_necessity("sup_activators_only", r_blind)
    put("release_raw", a_pos)                           # no inhibitors: nothing released (as run)
    for rd in ("tk", "pre"):
        row["release_" + rd] = (a_pos[rd] / a_pos[rd] - 1) if abs(a_pos[rd]) > 1e-9 else None
    sc = {s: (v, (1.0 - amps_of[s]).clamp(min=0)) for s, v in msets.items()}
    put_necessity("phi_sup_alpha", read_target(
        inference, lambda a, b: LiveScalePatcher(bank, sc, site, w_seed, b_seed, sl), pt_ho, pa_ho, batch_size))

    # 4. sufficiency to induce at alpha and at alpha = 1, every member as an activator (role-blind, as run)
    for name, use_a in (("phi_cf_alpha_role", True), ("phi_cf_a1_role", False)):
        tg = {s: {i: float((alphas[s][i] if use_a else 1.0) * float(pins_tr[s][i])) for i in v.tolist()}
              for s, v in msets.items()}
        r = cfp(tg, {}, nt_ho, na_ho)
        put(name + "_raw", r); put_induce(name, r)

    # 5. phi_pin_alpha: members CLAMPED to alpha x their clean position-wise value (unedited run, same batch)
    if msets and int(pt_ho.shape[0]) <= batch_size:
        cap = {}
        k2i = {k: i for i, k in enumerate(kinds)}

        def hook(layer_idx, activations):
            for kd in kinds:
                s = (layer_idx, kd)
                if s in msets:
                    ta, ti = bank.encode(activations[k2i[kd]], kd, layer_idx)
                    cap[s] = sparse_topk_to_dense(ta, ti, D, dtype=torch.float32)[..., msets[s]].clone()

        inference.disable_compile()
        try:
            inference.forward(pt_ho, activations_callback=hook, return_activations=False, tokenize_final=False)
        finally:
            inference.enable_compile()
        pins = LazyPins(cap, msets, tuple(pt_ho.shape[:2]), D)
        for name, mm, tk_, ek in (("phi_pin_alpha_zero", None, False, "e0"),
                                  ("phi_pin_alpha_topk", means_tr, True, "eM_topk"),
                                  ("phi_pin_alpha_dense", means_tr, False, "eM_dense")):
            r = co(keep, pt_ho, pa_ho, means=mm, fill_topk=tk_, pins=pins)
            put(name + "_raw", r); put_ratio(name, r, E[ek])
        del cap, pins

    # bookkeeping
    amps = np.array([a for d in alphas.values() for a in d.values()]) if n_mem else np.array([1.0])
    per_kind = {k: int(sum(len(d) for s, d in alphas.items() if s[1] == k)) for k in kinds}
    row.update(comp_idx=int(comp), latent_idx=int(sl), n_members=n_mem, per_kind=per_kind, n_upstream_sites=len(UP),
               n_member_sites=len(msets), alpha_median=float(np.median(amps)),
               alpha_p10=float(np.percentile(amps, 10)), alpha_p90=float(np.percentile(amps, 90)),
               n_alpha_off=int((amps < 0.1).sum()) if n_mem else 0,         # alpha ~0: zero-ablated under a mean fill
               n_alpha_suppress=int((amps < 0.4).sum()) if n_mem else 0,    # held well below the live value
               frac_alpha_suppress=float((amps < 0.4).mean()) if n_mem else 0.0,
               n_pos=n_pos, n_neg=n_neg, n_train=n_tr, n_ho=n_pos - n_tr,
               n_neg_train=n_ntr, n_neg_ho=n_neg - n_ntr, thin=bool(n_pos < n_seq or n_neg < n_seq),
               a_pos_blocks=a_pos_blocks, ctr_source=ctr_source, ctr_fallback=ctr_fallback,
               peak_mem_mb=(round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None),
               secs_roles=0.0, secs=round(time.time() - ts, 1), v=3)
    return row


def neg_ctx_store_fallback(neg_ctx: Any, comp: int, latent: int) -> bool:
    """True when the target's neg_ctx store row is the shared zero-similarity fallback list (DAN-64): the run's
    ``ctr_fallback`` provenance field."""
    ids, vals = neg_ctx.ctx_seq_idx[comp, latent], neg_ctx.ctx_seq_val[comp, latent]
    return bool((ids > 0).any()) and float(vals.float().abs().sum()) == 0.0


# ----------------------------------------------------------------------------------------------- pass rule
def _num(v: Any) -> Optional[float]:
    """A row value as a float; None for missing, None or NaN (pandas reads a JSON null as NaN)."""
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


def faith_fields(read: str = "tk") -> Tuple[str, str, str]:
    """The Z, A, C score fields under one read."""
    if read not in READS:
        raise ValueError("read must be one of %s, got %r" % (READS, read))
    return tuple("%s_%s" % (f, read) for f in FAITH)


def is_vacuous(row: Mapping[str, Any], read: str = "tk") -> bool:
    """The row's vacuous-denominator flag (Z denominator only); a missing flag counts as not vacuous. An undefined
    A or C ratio is NOT vacuous: it fails the pass rule."""
    return bool(row.get("vacuous_" + read) or False)


def passes(row: Mapping[str, Any], read: str = "tk", band: Tuple[float, float] = BAND, nec_min: float = NEC_MIN,
           necessity_read: str = "tk") -> bool:
    """The DAN-8 pass rule: Z, A, C all in ``band`` AND necessity >= nec_min; a missing score fails. Apply to rows
    with vacuous denominators excluded. ``necessity_read`` stays "tk" when ``read`` is "pre", as pass_rule.py's
    near-threshold pre-read companion ran it (necessity was always read post-Top-K)."""
    for f in faith_fields(read):
        v = _num(row.get(f))
        if v is None or not band[0] <= v <= band[1]:
            return False
    nec = _num(row.get("%s_%s" % (NECESSITY, necessity_read)))
    return nec is not None and nec >= nec_min


def worst_deviation(row: Mapping[str, Any], read: str = "tk", skipna: bool = True) -> Optional[float]:
    """The graded companion: max over Z, A, C of |faith - 1|. skipna=True (merge.py) ignores missing scores;
    skipna=False (rescore_mid.py) returns None when any is missing."""
    vals = [_num(row.get(f)) for f in faith_fields(read)]
    if not skipna and any(v is None for v in vals):
        return None
    vals = [abs(v - 1) for v in vals if v is not None]
    return max(vals) if vals else None


def pass_summary(rows: Iterable[Mapping[str, Any]], read: str = "tk", **rule: Any) -> Dict[str, Any]:
    """Pass rate over the non-vacuous rows (vacuous ones dropped from the denominator and counted)."""
    rows = list(rows)
    vac = [is_vacuous(r, read) for r in rows]
    scored = [r for r, v in zip(rows, vac) if not v]
    n_pass = sum(passes(r, read, **rule) for r in scored)
    return dict(n=len(rows), n_vacuous=int(sum(vac)), n_scored=len(scored), n_pass=int(n_pass),
                pass_rate=(n_pass / len(scored)) if scored else None)


__all__ = ["score_circuit", "read_target", "eval_split", "split_n", "frac", "ratio", "necessity", "vacuous",
           "circuit_alphas", "circuit_key", "seed_of", "site_encoder", "neg_ctx_store_fallback", "passes",
           "is_vacuous", "worst_deviation", "pass_summary", "faith_fields", "BAND", "NEC_MIN", "FAITH", "NECESSITY",
           "INDUCE", "READS", "SPLITS", "TIMING_FIELDS"]
