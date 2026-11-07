"""Protocol-v1 circuit scorer (DAN-73): the evaluation every protocol-v1 paper number is computed with.

This is the 062 run's scorer, ``experiments/049-circuit-graph/amp_eval_pass_v2.py: score_circuit(c, {},
skip_roles=True)``, moved into src with an explicit interface: the caller passes the inference engine, the SAE bank,
the circuit, the eval ProbeDataset and the contrast tokens, instead of module globals and a patched M0. Rows carry
the same fields with the same names and the same numbers (experiments/067-src-scorer-equivalence checks this on
the run's own circuits), so merge.py / pass_rule.py-style analysis reads them unchanged. Spec: paper/protocol.md
§5 (evaluation) and §6 (pass rule); the pass rule, vacuity and graded companion are at the bottom of this module,
the specificity diagnostic and near-threshold stratum in eval/wcm_specificity.py.

Contexts. ``probe.pos_tokens`` = the reference (train) contexts followed by the held-out ones, ``pos_argmax`` their
anchors; protocol v1: the 48 strongest-train contexts then the 16 held-out strongest or mid-band ones
(circuit/protocol_contexts.eval_probe). Contrast rows: first split_n(n) train, the rest held out; a probe from a
record whose contrast held-out set was rebuilt (086) carries metadata["n_neg_train"], used instead (contrast_split).

  split="explicit"  (default, every new run) the train count is ``probe.metadata["n_train"]``, so the held-out slice
                    is exactly the held-out set.
  split="as_run"    the 15k run's 3:1 rule, split_n(n_pos). On a short probe (mid-band with fewer held-out contexts
                    than the padding expects) it moves strongest-train contexts into the held-out slice
                    (protocol.md §3.6, 58 mid-band rows). Kept only to reproduce the run.

Reads. Every measurement is taken under both reads of the target at the anchor and every ratio uses ONE read in all
its terms (DAN-66): ``<name>_tk`` the post-Top-K activation (PRIMARY: every pass / fail number) and ``<name>_pre``
relu(w.x + b) (diagnostic). Raw reads are ``<name>_raw_tk`` / ``<name>_raw_pre``.

Circuit-only runs go through the engine's CircuitOnlyPatcher with every upstream site in scope (DAN-67): members at
alpha x their re-encoded live value (or pinned), every other latent at the fill: zero, dense mean, or the
sparsity-preserving fill whose budget is the dictionary's Top-K k (``bank.k``, per model; 128 on TuringLLM). Live-
stream edits (necessity, sufficiency to induce) go through CounterfactualInterventionPatcher. Both are subclassed
only to tap the target read.

Scores (protocol.md §5.2): free0 / freeM_topk / freeN_topk are Z / A / C (the pass rule), with freeM_dense,
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
from sae.dense import seed_activation, seed_threshold, sparse_topk_to_dense, target_latent_activations

Site = Tuple[int, str]
Alphas = Dict[Site, Dict[int, float]]

READS = ("tk", "pre")            # tk = post-Top-K activation (primary), pre = relu(w.x + b) (diagnostic)
SPLITS = ("explicit", "as_run")
N_SEQ = 64                       # activating / contrast rows read from the probe (the run's cap)
EVAL_BATCH = 16                  # sequences per forward; also the largest held-out set phi_pin is measured on
MIN_POS = 8                      # fewer activating contexts: skip "thin_probes"
HOLDOUT_FRAC = 0.25              # the engine split rule: n_hold = round(n * 0.25), train = the first n - n_hold
VACUOUS_REL = 0.05               # vacuous denominator: |a_pos - e0| < 0.05 |a_pos|
VACUOUS_BASELINES = ("e0", "eM_topk", "eN_topk")        # the pass rule's denominators: Z and necessity, A, C (§12.4)
VACUOUS_RULES = ("v2", "v1")

# DAN-8 pass rule (adopted 2026-09-27; floor 0.8 -> 0.75 on 2026-10-31, before the production run): Z, A, C all in
# BAND and necessity >= NEC_MIN, held-out strongest
BAND = (0.75, 1.5)
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


def vacuous_v2(a_pos: float, baselines: Iterable[float], rel: float = VACUOUS_REL) -> bool:
    """Protocol §12.4 (exp 105): vacuous when a_pos <= 0 or any pass-rule baseline e exceeds (1 - rel) a_pos (signed).
    A function of the empty-circuit reads only, so of the target and its contexts, never of the circuit."""
    return bool(a_pos <= 0 or any(a_pos - e < rel * a_pos for e in baselines))


def split_n(n: int, holdout_frac: float = HOLDOUT_FRAC) -> int:
    """Train count of the engine's 3:1 rule: n - round(n * holdout_frac), at least 1 (Python's round: half to even)."""
    n_hold = int(round(n * holdout_frac))
    return max(1, n - n_hold)


def contrast_split(n_neg: int, metadata: Optional[Mapping[str, Any]], holdout_frac: float = HOLDOUT_FRAC,
                   own: bool = True) -> int:
    """The contrast-train count. A probe built from a record whose contrast held-out set was rebuilt (086;
    protocol_contexts.eval_contrast) carries metadata["n_neg_train"] (the run's 48 contrast-train rows, then a held-out
    set that may hold fewer than 16): that count, in either split mode. Otherwise (every as-run record, and contrast
    rows that are not the probe's own, own=False) split_n(n_neg), unchanged."""
    v = (metadata or {}).get("n_neg_train") if own else None
    if v is None:
        return split_n(n_neg, holdout_frac)
    v = int(v)
    if not 0 < v < n_neg:
        raise ValueError("n_neg_train %d needs a contrast train and held-out slice (n_neg %d)" % (v, n_neg))
    return v


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
    if not hasattr(p, "seed_thr"):                         # the "pre" read's activation (JumpReLU threshold / ReLU)
        p.seed_thr = seed_threshold(bank, site[0], site[1], p.seed_idx)
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
                pres.append(seed_activation(pre[rr, anc], getattr(p, "seed_thr", None)).float())
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
    n_ntr = contrast_split(n_neg, probe.metadata, holdout_frac, own=contrast_tokens is None)
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
    A_BASE = {"pre": float(seed_activation(neg_pre[rr, na_ho.to(neg_pre.device)],
                                           getattr(p0, "seed_thr", None)).float().mean()),
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
    row["vacuous_v2"] = vacuous_v2(a_pos["tk"], [E[k]["tk"] for k in VACUOUS_BASELINES])   # §12.4, both reads

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


# ----------------------------------------------------------------------------------------------- output endpoint
# Experiment 080 (2026-10-18): the same circuit-only runs as score_circuit, read at the model's OUTPUT instead of at the
# target latent. m = mean over the held-out contexts of log p(the actual next token x_{t+1} | x_<=t) at each context's
# target anchor t. Additive: nothing above changes.
OUT_FAITH = ("out_free0", "out_freeM_topk", "out_freeN_topk")   # Z, A (sparsity-preserving), C (sparsity-preserving)
OUT_NECESSITY = "out_phi_sup_blind"
OUT_VACUOUS_NATS = 0.1           # vacuous output denominator: |m(full) - m(empty, Z)| < 0.1 nats
OUT_SCOPES = ("upstream", "all")


def output_vacuous(m_full: float, m_e0: float, nats: float = OUT_VACUOUS_NATS) -> bool:
    """A vacuous output denominator: ablating the scope (zero) moves the mean log-prob by less than `nats`. Absolute,
    in nats: a log-prob has no natural zero to be relative to (the latent rule, vacuous(), is relative to a_pos)."""
    return bool(abs(m_full - m_e0) < nats)


def next_tokens_at(probe: Any, anchors: torch.Tensor, n: int) -> torch.Tensor:
    """The actual next token at each context's anchor: probe.target_tokens [N, T] (shifted, target_tokens[i, t] is
    token t + 1) gathered at the anchors, or a [N] tensor used as is."""
    tt = probe.target_tokens[:n]
    if tt.dim() == 1:
        return tt
    rr = torch.arange(int(tt.shape[0]), device=tt.device)
    return tt[rr, anchors[:n].to(tt.device).clamp(0, tt.shape[1] - 1)]


def scope_sites(bank: Any, layer: int, kind: str, scope: str = "upstream") -> Set[Site]:
    """The sites a circuit's ablation covers. "upstream": the sites causally upstream of the target's site (the latent
    circuit's scope; the target's site and every later site run live). "all": every site of the model."""
    if scope == "upstream":
        return set(upstream_sites(bank, layer, kind))
    if scope == "all":
        return {(l, k) for l in range(int(bank.n_layer)) for k in bank.kinds if bank.saes[k][l] is not None}
    raise ValueError("scope must be one of %s, got %r" % (OUT_SCOPES, scope))


class EditTrackingCircuitOnly(CircuitOnlyPatcher):
    """The engine's CircuitOnlyPatcher, recording which sites it ablated (the scope check: must equal the scope).
    Built with seed_layer=-1, so it reads no latent."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.edited: Set[Site] = set()

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if (layer_idx, kind) in self.in_scope:
            self.edited.add((layer_idx, kind))
        return super().transform(layer_idx, kind, x)


def read_logprob(inference: Any, make_patcher: Optional[Callable[[int, int], Any]], tokens: torch.Tensor,
                 anchors: torch.Tensor, next_tokens: torch.Tensor, batch_size: int = EVAL_BATCH,
                 per_seq: bool = False):
    """Run ``tokens`` in batch_size chunks under make_patcher(start, stop) (None = the unedited model) and read
    log p(next_tokens) at ``anchors`` from the output logits. Returns {"m": mean log-prob, "edited": sites ablated},
    or the per-sequence tensor."""
    vals = []
    edited: Set[Site] = set()
    inference.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(tokens.shape[0]), batch_size):
                tk = tokens[s0:s0 + batch_size]
                B = int(tk.shape[0])
                p = make_patcher(s0, s0 + B) if make_patcher is not None else None
                out = inference.forward(tk, patcher=p, grad_enabled=False, all_logits=True, return_activations=False,
                                        tokenize_final=False)
                lg = out[1] if isinstance(out, (tuple, list)) else out
                rr = torch.arange(B, device=lg.device)
                anc = anchors[s0:s0 + B].to(lg.device).clamp(0, lg.shape[1] - 1)
                lp = torch.log_softmax(lg[rr, anc].float(), dim=-1)
                vals.append(lp[rr, next_tokens[s0:s0 + B].to(lg.device).long()].float())
                edited |= getattr(p, "edited", set())
    finally:
        inference.enable_compile()
    vals = torch.cat(vals)
    if per_seq:
        return vals
    return {"m": float(vals.mean()), "edited": edited}


@torch.no_grad()
def score_output(
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
    scope: str = "upstream",
    holdout_frac: float = HOLDOUT_FRAC,
) -> Dict[str, Any]:
    """Score one circuit on the OUTPUT endpoint (experiment 080); same contexts, split, ablation values, fills and
    patchers as score_circuit, read as m = mean log p(actual next token) at the held-out anchors.

      out_<pi>         output faithfulness (m(C) - m(empty)) / (m(full) - m(empty)), unclipped, for pi in Z (zero fill),
                       A (sparsity-preserving mean over the reference contexts), C (sparsity-preserving mean over the
                       training contrast contexts): out_free0 / out_freeM_topk / out_freeN_topk. Members at alpha x
                       their live value, every other latent at every in-scope site at the fill, SAE errors preserved;
                       sites outside the scope (scope="upstream": the target's site and everything after it) run live.
                       None unless m(empty) < m(full) (the latent scorer's guard).
      out_free0_a1     Z with every member at alpha = 1
      out_phi_sup_blind  output necessity (m(full) - m(full without the members)) / (m(full) - m(empty, Z)): the
                       members SET to 0 in the otherwise-live model
      out_vacuous      |m(full) - m(empty, Z)| < OUT_VACUOUS_NATS (exclude from pass rates, as is_vacuous)
    Raw means are out_full / out_e0 / out_eM_topk / out_eN_topk and out_<name>_raw. Members at sites outside the scope
    are dropped (counted in out_n_out_of_scope). Returns a skip row {seed, skip, ...} like score_circuit."""
    ts = time.time()
    seed = seed_of(circuit)
    sf = seed.metadata["feature_id"]
    layer, kind = sf.layer, sf.kind
    key = "%d.%s.%d" % (layer, kind, sf.index)
    K = int(bank.k) if topk is None else int(topk)
    UPS = scope_sites(bank, layer, kind, scope)
    alphas_all = circuit_alphas(circuit) if alphas is None else alphas
    alphas = {s: d for s, d in alphas_all.items() if s in UPS and d}
    n_mem = sum(len(d) for d in alphas.values())
    n_oos = sum(len(d) for s, d in alphas_all.items() if s not in UPS)
    n_pos = 0 if probe is None else int(probe.pos_tokens.shape[0])
    if n_pos < min_pos:
        return dict(seed=key, skip="thin_probes", n=n_mem, n_pos=n_pos)
    pt, pa = probe.pos_tokens[:n_seq], probe.pos_argmax[:n_seq]
    nt = (probe.neg_tokens if contrast_tokens is None else contrast_tokens)[:n_seq]
    n_pos, n_neg = int(pt.shape[0]), int(nt.shape[0])
    if n_neg < 2:
        return dict(seed=key, skip="no_negatives", n=n_mem, n_pos=n_pos, n_neg=n_neg)
    nx = next_tokens_at(probe, pa, n_pos)
    n_tr = eval_split(n_pos, probe.metadata, split, holdout_frac)
    n_ntr = contrast_split(n_neg, probe.metadata, holdout_frac, own=contrast_tokens is None)
    pt_tr, pa_tr, pt_ho, pa_ho, nx_ho = pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:], nx[n_tr:]
    nt_tr = nt[:n_ntr]
    dev = pt.device
    D = int(bank.d_sae)
    msets = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in alphas.items()}
    keep = {s: set(d) for s, d in alphas.items()}
    scales = {}
    for s, v in msets.items():
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        sv[v] = torch.tensor([alphas[s][int(i)] for i in v.tolist()], device=dev, dtype=torch.float32)
        scales[s] = sv

    def co(keep_, means=None, fill_topk=False, use_scales=True):
        """Circuit-only run on the held-out contexts, read at the output (see score_circuit's co)."""
        kt = {s: msets[s] for s in keep_}
        ks = scales if (use_scales and keep_) else None
        return read_logprob(inference, lambda a, b: EditTrackingCircuitOnly(
            bank=bank, keep_indices=keep_, in_scope=UPS, seed_layer=-1, seed_kind="", seed_latent_idx=0,
            pos_argmax=pa_ho[a:b], site_means=means, respect_topk=fill_topk, topk=K, keep_tensors=kt,
            keep_scales=ks), pt_ho, pa_ho, nx_ho, batch_size)

    # ablation values exactly as score_circuit: A means over the reference (train) activating rows, C means over the
    # training contrast rows
    means_tr, _ = collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)
    means_neg = collect_site_means(inference, bank, nt_tr, UPS) if UPS else {}
    full = read_logprob(inference, None, pt_ho, pa_ho, nx_ho, batch_size)["m"]
    e0_run = co({})
    E = {"e0": e0_run["m"], "eM_topk": co({}, means_tr, True)["m"], "eN_topk": co({}, means_neg, True)["m"]}

    row: Dict[str, Any] = dict(seed=key, layer=layer, kind=kind, n=n_mem, out_scope=scope, out_full=full,
                               out_den=full - E["e0"], out_vacuous=output_vacuous(full, E["e0"]))
    for k_, v in E.items():
        row["out_" + k_] = v
    for name, mm, tk_, ek in (("out_free0", None, False, "e0"), ("out_freeM_topk", means_tr, True, "eM_topk"),
                              ("out_freeN_topk", means_neg, True, "eN_topk")):
        r = co(keep, mm, tk_) if keep else {"m": E[ek], "edited": e0_run["edited"]}
        row[name + "_raw"] = r["m"]
        row[name] = ratio(r["m"], E[ek], full)
        if name == "out_free0":
            row["out_all_sites_ablated"] = bool(r["edited"] == UPS)
    r = co(keep, use_scales=False)["m"] if keep else E["e0"]
    row["out_free0_a1_raw"] = r
    row["out_free0_a1"] = ratio(r, E["e0"], full)
    # necessity: every member SET to 0 in the otherwise-live model (CounterfactualInterventionPatcher, SAE error kept)
    if keep:
        r = read_logprob(inference, lambda a, b: CounterfactualInterventionPatcher(
            bank=bank, activator_targets={}, inhibitor_indices=keep, seed_layer=-1, seed_kind="",
            seed_latent_idx=0, pos_argmax=pa_ho[a:b]), pt_ho, pa_ho, nx_ho, batch_size)["m"]
    else:
        r = full
    row["out_phi_sup_blind_raw"] = r
    row["out_phi_sup_blind"] = necessity(r, full, E["e0"])
    row.update(out_n_out_of_scope=n_oos, out_n_scope_sites=len(UPS), out_n_train=n_tr, out_n_ho=n_pos - n_tr,
               out_n_neg_train=n_ntr, out_secs=round(time.time() - ts, 1), out_v=1)
    return row


def passes_output(row: Mapping[str, Any], band: Tuple[float, float] = BAND, nec_min: float = NEC_MIN) -> bool:
    """The DAN-8 rule on the output endpoint: output Z, A, C all in ``band`` AND output necessity >= nec_min; a missing
    score fails. Apply to rows with out_vacuous excluded."""
    for f in OUT_FAITH:
        v = _num(row.get(f))
        if v is None or not band[0] <= v <= band[1]:
            return False
    nec = _num(row.get(OUT_NECESSITY))
    return nec is not None and nec >= nec_min


# ----------------------------------------------------------------------------------------------- task endpoint
# Experiment 081 (2026-10-19): the output-endpoint scorer for TASK circuits (greater-than, agreement, IOI, ...), whose
# endpoint is a task METRIC read at each prompt's anchor rather than log p of one next token, and whose circuits have
# no target latent (every SAE site is in scope by default). Same patchers, fills and ratio forms as score_output;
# additive: nothing above changes.
#   metric kinds (higher = more correct; good / bad are [N] token ids or [N, K] sets padded with -1)
#     logit_diff     logit(good) - logit(bad) (= log p(good) - log p(bad); SFC / IOI / agreement)
#     logprob        log p(good) (bad unused)
#     set_logmass    log P(good set) - log P(bad set) (the engine's logit_metric="set" endpoint)
#     set_probdiff   P(good set) - P(bad set) (greater-than's probability difference, Hanna et al. 2023)
#     seq_diff       (2026-10-21) MULTI-TOKEN answers, teacher-forced: rows come in (good, bad) pairs, each row =
#                    prompt + its answer[:-1], good = the row's answer tokens [N, L] (-1 padded), anchor = the prompt's
#                    last position; per pair log p(good seq) - log p(bad seq), log p(seq) = sum_k log p(a_k | prompt,
#                    a_<k) (the engine's logit_metric="seq"). Values, accuracies and ratios are per PAIR (item).
TASK_METRICS = ("logit_diff", "logprob", "set_logmass", "set_probdiff", "seq_diff")
TASK_FAITH = ("free0", "freeM_topk", "freeN_topk")              # Z, A (sparsity-preserving), C (sparsity-preserving)
TASK_NECESSITY = "nec"


def _token_sets(tok: torch.Tensor) -> torch.Tensor:
    """[N] ids -> [N, 1]; [N, K] sets (padded with -1) unchanged."""
    tok = torch.as_tensor(tok).long()
    return tok[:, None] if tok.dim() == 1 else tok


def _logmass(lp: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    t = _token_sets(tok).to(lp.device)
    g = lp.gather(1, t.clamp(min=0))
    return torch.logsumexp(g.masked_fill(t < 0, float("-inf")), dim=1)


def task_metric(lp: torch.Tensor, kind: str, good: torch.Tensor, bad: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-row task metric [N] from log-probs at the anchor, lp [N, V] (log_softmax of the anchor logits)."""
    if kind not in TASK_METRICS:
        raise ValueError("metric kind must be one of %s, got %r" % (TASK_METRICS, kind))
    if kind == "seq_diff":
        raise ValueError("seq_diff is read from teacher-forced answer positions, not from anchor log-probs: use "
                         "read_rows(..., answers) and pair_diff")
    lp = lp.float()
    if kind == "logprob":
        return _logmass(lp, good)
    if bad is None:
        raise ValueError("metric %r needs the bad (contrast) tokens" % kind)
    if kind in ("logit_diff", "set_logmass"):
        return _logmass(lp, good) - _logmass(lp, bad)
    return _logmass(lp, good).exp() - _logmass(lp, bad).exp()          # set_probdiff


def task_correct(lp: torch.Tensor, kind: str, good: torch.Tensor, bad: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-row correctness [N] bool: the metric is > 0 for the contrastive kinds; for "logprob", the argmax token is in
    the good set."""
    if kind == "logprob":
        g = _token_sets(good).to(lp.device)
        return (g == lp.argmax(dim=1, keepdim=True)).any(dim=1)
    return task_metric(lp, kind, good, bad) > 0


def task_top1(lp: torch.Tensor, good: torch.Tensor) -> torch.Tensor:
    """Per-row: the model's argmax token is in the good set."""
    g = _token_sets(good).to(lp.device)
    return (g == lp.argmax(dim=1, keepdim=True)).any(dim=1)


def answer_logprob(logits: torch.Tensor, anchors: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    """Teacher-forced log p(answer) per row: sum_k log p(a_k | prompt, a_<k) read at positions anchor + k of rows
    whose tokens are prompt + answer[:-1]; answers [B, L] padded with -1 -> [B] (float32)."""
    B, T = int(logits.shape[0]), int(logits.shape[1])
    a = torch.as_tensor(answers)[:B].to(logits.device).long()
    L = int(a.shape[1])
    pos = anchors[:B].to(logits.device).long()[:, None] + torch.arange(L, device=logits.device)[None, :]
    ok = (a >= 0) & (pos < T)
    lg = logits[torch.arange(B, device=logits.device)[:, None], pos.clamp(max=T - 1)].float()
    lp = torch.log_softmax(lg, dim=-1).gather(2, a.clamp(min=0)[..., None])[..., 0]
    return (lp * ok).sum(dim=1)


def pair_diff(v: torch.Tensor) -> torch.Tensor:
    """Per-row values of (good, bad) row pairs -> per-pair good - bad."""
    if int(v.shape[0]) % 2:
        raise ValueError("pair_diff needs (good, bad) row pairs (an even number of rows)")
    return v[0::2] - v[1::2]


def read_rows(inference: Any, make_patcher: Optional[Callable[[int, int], Any]], tokens: torch.Tensor,
              anchors: torch.Tensor, answers: Optional[torch.Tensor] = None,
              batch_size: int = EVAL_BATCH) -> Tuple[torch.Tensor, Optional[torch.Tensor], Set[Site]]:
    """read_anchor_logprobs plus, when ``answers`` [N, L] are given, each row's teacher-forced answer log-prob [N]
    from the SAME forward (answer_logprob). Returns (anchor log-probs [N, V], answer log-probs or None, edited)."""
    out, seqs = [], []
    edited: Set[Site] = set()
    inference.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(tokens.shape[0]), batch_size):
                tk = tokens[s0:s0 + batch_size]
                B = int(tk.shape[0])
                p = make_patcher(s0, s0 + B) if make_patcher is not None else None
                o = inference.forward(tk, patcher=p, grad_enabled=False, all_logits=True, return_activations=False,
                                      tokenize_final=False)
                lg = o[1] if isinstance(o, (tuple, list)) else o
                rr = torch.arange(B, device=lg.device)
                anc = anchors[s0:s0 + B].to(lg.device).clamp(0, lg.shape[1] - 1)
                out.append(torch.log_softmax(lg[rr, anc].float(), dim=-1).cpu())
                if answers is not None:
                    seqs.append(answer_logprob(lg, anchors[s0:s0 + B], answers[s0:s0 + B]).cpu())
                edited |= getattr(p, "edited", set())
    finally:
        inference.enable_compile()
    return torch.cat(out, 0), (torch.cat(seqs, 0) if answers is not None else None), edited


def read_anchor_logprobs(inference: Any, make_patcher: Optional[Callable[[int, int], Any]], tokens: torch.Tensor,
                         anchors: torch.Tensor, batch_size: int = EVAL_BATCH) -> Tuple[torch.Tensor, Set[Site]]:
    """Run ``tokens`` in batch_size chunks under make_patcher(start, stop) (None = the unedited model); return the
    log-probs at each row's anchor [N, V] (float32, CPU) and the sites the patchers ablated."""
    out = []
    edited: Set[Site] = set()
    inference.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(tokens.shape[0]), batch_size):
                tk = tokens[s0:s0 + batch_size]
                B = int(tk.shape[0])
                p = make_patcher(s0, s0 + B) if make_patcher is not None else None
                o = inference.forward(tk, patcher=p, grad_enabled=False, all_logits=True, return_activations=False,
                                      tokenize_final=False)
                lg = o[1] if isinstance(o, (tuple, list)) else o
                rr = torch.arange(B, device=lg.device)
                anc = anchors[s0:s0 + B].to(lg.device).clamp(0, lg.shape[1] - 1)
                out.append(torch.log_softmax(lg[rr, anc].float(), dim=-1).cpu())
                edited |= getattr(p, "edited", set())
    finally:
        inference.enable_compile()
    return torch.cat(out, 0), edited


@torch.no_grad()
def score_task(
    inference: Any,
    bank: Any,
    alphas: Alphas,
    tokens: torch.Tensor,
    anchors: torch.Tensor,
    n_train: int,
    cf_tokens: torch.Tensor,
    metrics: Mapping[str, Tuple[str, torch.Tensor, Optional[torch.Tensor]]],
    *,
    valid: Optional[torch.Tensor] = None,
    cf_valid: Optional[torch.Tensor] = None,
    sites: Optional[Set[Site]] = None,
    batch_size: int = EVAL_BATCH,
    topk: Optional[int] = None,
    vacuous_abs: Optional[Mapping[str, float]] = None,
    means_A: Optional[Dict[Site, torch.Tensor]] = None,
    means_C: Optional[Dict[Site, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Score one task circuit (experiment 081) on the held-out prompts, ALWAYS with its coefficients applied.

      alphas     {site: {latent: alpha}} (alpha 1 everywhere for an unweighted circuit)
      tokens     [N, T] right-padded prompts, the first n_train rows the training (reference) prompts, the rest held
                 out; anchors [N] = each prompt's prediction position (its last real token)
      cf_tokens  the counterfactual prompts of the TRAINING rows: the C ablation value is their mean
      metrics    {name: (kind, good [N] or [N, K], bad or None)} over all N rows (sliced to the held-out rows here);
                 every name's fields are written with that prefix
      valid, cf_valid   [N, T] / [N_cf, T] masks of real tokens: the A and C means average over them only (None = every
                 position)
      sites      the ablation scope (default every SAE site: the output is the endpoint, as in SFC)
      vacuous_abs  {name: threshold}: <name>_vacuous when |m(full) - m(empty, Z)| is below it (default
                 OUT_VACUOUS_NATS, 0.1; give probability-unit metrics their own)
      means_A, means_C  {site: [d_sae]} precomputed A / C ablation values (2026-10-21: 081 averages them over a whole
                 training / contrast pool once per task); None (default) = collected here from the first n_train rows
                 / cf_tokens, unchanged
    Each frame's [n_ho, V] log-probs are reduced to per-item values as soon as they are read, so memory holds one
    frame, not nine, for large held-out sets; the numbers are unchanged.
    Per metric name m: m_full, m_e0, m_eM_topk, m_eN_topk, m_den, m_vacuous; faithfulness m_free0 / m_freeM_topk /
    m_freeN_topk ((m(C) - m(empty)) / (m(full) - m(empty)), unclipped, None unless m(empty) < m(full)) with _raw
    means; m_free0_a1 (Z with every member at alpha = 1); m_nec = (m(full) - m(members set to 0, live)) /
    (m(full) - m(empty, Z)); accuracies m_acc_<frame> (share of held-out prompts with metric > 0) for full, e0 and the
    three circuit runs. Top level: n, per_site, n_ho, n_train, n_cf, all_sites_ablated, secs."""
    ts = time.time()
    K = int(bank.k) if topk is None else int(topk)
    if sites is None:
        sites = scope_sites(bank, 0, list(bank.kinds)[0], "all")
    UPS = set(sites)
    alphas = {s: d for s, d in alphas.items() if d}
    n_oos = sum(len(d) for s, d in alphas.items() if s not in UPS)
    alphas = {s: d for s, d in alphas.items() if s in UPS}
    n_mem = sum(len(d) for d in alphas.values())
    N = int(tokens.shape[0])
    if not 0 < int(n_train) < N:
        raise ValueError("score_task needs both training and held-out rows (n_train %d, N %d)" % (n_train, N))
    if int(cf_tokens.shape[0]) < 1:
        raise ValueError("score_task needs counterfactual prompts for the C ablation")
    n_tr = int(n_train)
    pt_tr, pa_tr, pt_ho, pa_ho = tokens[:n_tr], anchors[:n_tr], tokens[n_tr:], anchors[n_tr:]
    # multi-token answers (seq_diff): every seq metric reads the same teacher-forced rows
    seq_ans = [torch.as_tensor(g).long() for k_, g, _ in metrics.values() if k_ == "seq_diff"]
    if seq_ans:
        if n_tr % 2 or N % 2:
            raise ValueError("seq_diff metrics need (good, bad) row pairs: even n_train and N (got %d, %d)"
                             % (n_tr, N))
        if any(not torch.equal(a, seq_ans[0]) for a in seq_ans[1:]):
            raise ValueError("every seq_diff metric must use the same answer rows")
    ans_ho = seq_ans[0][n_tr:] if seq_ans else None
    dev = tokens.device
    D = int(bank.d_sae)
    msets = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in alphas.items()}
    keep = {s: set(d) for s, d in alphas.items()}
    scales = {}
    for s, v in msets.items():
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        sv[v] = torch.tensor([alphas[s][int(i)] for i in v.tolist()], device=dev, dtype=torch.float32)
        scales[s] = sv

    sets_ho = {m: (k_, _token_sets(g)[n_tr:], None if b is None else _token_sets(b)[n_tr:])
               for m, (k_, g, b) in metrics.items()}

    def reduce(lp_, sq_):
        """One frame's per-item (metric values, correctness, top-1 or None) per metric name; the [n_ho, V] log-probs
        are dropped after this."""
        out = {}
        for m, (k_, g_ho, b_ho) in sets_ho.items():
            if k_ == "seq_diff":
                d = pair_diff(sq_)
                out[m] = (d, d > 0, None)
            else:
                out[m] = (task_metric(lp_, k_, g_ho, b_ho), task_correct(lp_, k_, g_ho, b_ho), task_top1(lp_, g_ho))
        return out

    def rd(make):
        """One frame on the held-out rows, reduced (see reduce), and the sites edited."""
        lp_, sq_, ed_ = read_rows(inference, make, pt_ho, pa_ho, ans_ho, batch_size)
        return reduce(lp_, sq_), ed_

    def co(keep_, means=None, fill_topk=False, use_scales=True):
        kt = {s: msets[s] for s in keep_}
        ks = scales if (use_scales and keep_) else None
        return rd(lambda a, b: EditTrackingCircuitOnly(
            bank=bank, keep_indices=keep_, in_scope=UPS, seed_layer=-1, seed_kind="", seed_latent_idx=0,
            pos_argmax=pa_ho[a:b], site_means=means, respect_topk=fill_topk, topk=K, keep_tensors=kt,
            keep_scales=ks))

    # ablation values: A = mean over the training (reference) prompts, C = mean over the counterfactual prompts, each
    # over the real-token positions only when masks are given
    va = {} if valid is None else dict(valid=valid[:n_tr])
    vc = {} if cf_valid is None else dict(valid=cf_valid)
    means_tr = collect_site_means(inference, bank, pt_tr, UPS, **va) if means_A is None else means_A
    means_cf = collect_site_means(inference, bank, cf_tokens, UPS, **vc) if means_C is None else means_C
    for nm_, mm_ in (("means_A", means_tr), ("means_C", means_cf)):
        if UPS - set(mm_):
            raise ValueError("%s lacks sites %s" % (nm_, sorted(UPS - set(mm_))))
    lp = {"full": rd(None)[0]}
    lp["e0"], edited_e0 = co({})
    lp["eM_topk"] = co({}, means_tr, True)[0]
    lp["eN_topk"] = co({}, means_cf, True)[0]
    edited = edited_e0
    for name, mm, tk_ in (("free0", None, False), ("freeM_topk", means_tr, True), ("freeN_topk", means_cf, True)):
        if keep:
            lp[name], ed = co(keep, mm, tk_)
            if name == "free0":
                edited = ed
        else:
            lp[name] = lp[{"free0": "e0", "freeM_topk": "eM_topk", "freeN_topk": "eN_topk"}[name]]
    lp["free0_a1"] = co(keep, use_scales=False)[0] if keep else lp["e0"]
    if keep:
        lp["nec"] = rd(lambda a, b: CounterfactualInterventionPatcher(
            bank=bank, activator_targets={}, inhibitor_indices=keep, seed_layer=-1, seed_kind="", seed_latent_idx=0,
            pos_argmax=pa_ho[a:b]))[0]
    else:
        lp["nec"] = lp["full"]

    per_kind: Dict[str, int] = defaultdict(int)
    for s, d in alphas.items():
        per_kind[s[1]] += len(d)
    amps = np.array([a for d in alphas.values() for a in d.values()]) if n_mem else np.array([1.0])
    row: Dict[str, Any] = dict(n=n_mem, n_out_of_scope=n_oos, n_sites_scope=len(UPS), n_member_sites=len(msets),
                               per_kind=dict(per_kind), n_train=n_tr, n_ho=N - n_tr, n_cf=int(cf_tokens.shape[0]),
                               all_sites_ablated=bool(edited == UPS), alpha_median=float(np.median(amps)),
                               alpha_p10=float(np.percentile(amps, 10)), alpha_p90=float(np.percentile(amps, 90)))
    for mname, (kind, g_ho, b_ho) in sets_ho.items():
        vc_ = {f: v[mname] for f, v in lp.items()}
        val = {f: float(v[0].mean()) for f, v in vc_.items()}
        acc = {f: float(v[1].float().mean()) for f, v in vc_.items()}
        full, e0 = val["full"], val["e0"]
        thr = (vacuous_abs or {}).get(mname, OUT_VACUOUS_NATS)
        row.update({"%s_kind" % mname: kind, "%s_full" % mname: full, "%s_e0" % mname: e0,
                    "%s_eM_topk" % mname: val["eM_topk"], "%s_eN_topk" % mname: val["eN_topk"],
                    "%s_den" % mname: full - e0, "%s_vacuous" % mname: bool(abs(full - e0) < thr)})
        for f, ek in (("free0", "e0"), ("freeM_topk", "eM_topk"), ("freeN_topk", "eN_topk")):
            row["%s_%s_raw" % (mname, f)] = val[f]
            row["%s_%s" % (mname, f)] = ratio(val[f], val[ek], full)
        row["%s_free0_a1_raw" % mname] = val["free0_a1"]
        row["%s_free0_a1" % mname] = ratio(val["free0_a1"], e0, full)
        row["%s_nec_raw" % mname] = val["nec"]
        row["%s_nec" % mname] = necessity(val["nec"], full, e0)
        for f in ("full", "e0", "free0", "freeM_topk", "freeN_topk", "free0_a1"):
            row["%s_acc_%s" % (mname, f)] = acc[f]
        if kind == "seq_diff":                    # no single next token to be top-1 for a multi-token answer
            row["%s_top1_full" % mname] = row["%s_top1_free0" % mname] = None
            row["%s_n_items" % mname] = int(g_ho.shape[0]) // 2
        else:
            row["%s_top1_full" % mname] = float(vc_["full"][2].float().mean())
            row["%s_top1_free0" % mname] = float(vc_["free0"][2].float().mean())
        row["%s_pass" % mname] = passes_task(row, mname)
    row["secs"] = round(time.time() - ts, 1)
    return row


def passes_task(row: Mapping[str, Any], prefix: str = "task", band: Tuple[float, float] = BAND,
                nec_min: float = NEC_MIN) -> bool:
    """The DAN-8 rule on a task metric: <prefix>_free0 / _freeM_topk / _freeN_topk all in ``band`` AND <prefix>_nec
    >= nec_min; a missing score fails. Apply to rows with <prefix>_vacuous excluded."""
    for f in TASK_FAITH:
        v = _num(row.get("%s_%s" % (prefix, f)))
        if v is None or not band[0] <= v <= band[1]:
            return False
    nec = _num(row.get("%s_%s" % (prefix, TASK_NECESSITY)))
    return nec is not None and nec >= nec_min


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


def is_vacuous(row: Mapping[str, Any], read: str = "tk", rule: str = "v2") -> bool:
    """Whether the row is dropped from pass-rate denominators.

    rule="v2" (protocol §12.4, the default): vacuous_v2 on the post-Top-K reads, one exclusion set for both reads. It
    is recomputed from the stored a_pos_tk / e0_tk / eM_topk_tk / eN_topk_tk when the row has no vacuous_v2 field, so
    rows scored before §12.4 are re-classified without rescoring; a row without those reads falls back to its
    vacuous_tk flag. A term still undefined on a non-vacuous row fails the pass rule.
    rule="v1": the stored vacuous_<read> flag (Z denominator only, two-sided); a missing flag counts as not vacuous."""
    if rule not in VACUOUS_RULES:
        raise ValueError("rule must be one of %s, got %r" % (VACUOUS_RULES, rule))
    if rule == "v1":
        return bool(row.get("vacuous_" + read) or False)
    if row.get("vacuous_v2") is not None:
        return bool(row["vacuous_v2"])
    ap = _num(row.get("a_pos_tk"))
    es = [_num(row.get(k + "_tk")) for k in VACUOUS_BASELINES]
    if ap is None or any(e is None for e in es):
        return bool(row.get("vacuous_tk") or False)
    return vacuous_v2(ap, es)


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


def pass_summary(rows: Iterable[Mapping[str, Any]], read: str = "tk", vacuous_rule: str = "v2",
                 **rule: Any) -> Dict[str, Any]:
    """Pass rate over the non-vacuous rows (vacuous ones dropped from the denominator and counted)."""
    rows = list(rows)
    vac = [is_vacuous(r, read, vacuous_rule) for r in rows]
    scored = [r for r, v in zip(rows, vac) if not v]
    n_pass = sum(passes(r, read, **rule) for r in scored)
    return dict(n=len(rows), n_vacuous=int(sum(vac)), n_scored=len(scored), n_pass=int(n_pass),
                pass_rate=(n_pass / len(scored)) if scored else None)


__all__ = ["score_circuit", "read_target", "eval_split", "split_n", "frac", "ratio", "necessity", "vacuous",
           "vacuous_v2", "VACUOUS_BASELINES",
           "circuit_alphas", "circuit_key", "seed_of", "site_encoder", "neg_ctx_store_fallback", "passes",
           "is_vacuous", "worst_deviation", "pass_summary", "faith_fields", "BAND", "NEC_MIN", "FAITH", "NECESSITY",
           "INDUCE", "READS", "SPLITS", "TIMING_FIELDS",
           # output endpoint (experiment 080)
           "score_output", "passes_output", "read_logprob", "output_vacuous", "next_tokens_at", "scope_sites",
           "OUT_FAITH", "OUT_NECESSITY", "OUT_VACUOUS_NATS", "OUT_SCOPES",
           # task endpoint (experiment 081)
           "score_task", "passes_task", "task_metric", "task_correct", "task_top1", "read_anchor_logprobs",
           "read_rows", "answer_logprob", "pair_diff",
           "TASK_METRICS", "TASK_FAITH", "TASK_NECESSITY"]
