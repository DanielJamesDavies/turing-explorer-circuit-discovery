"""Position-aware allowed-set selection for circuit discovery.

The classic gradient attribution collapses the token-position axis (sums
grad*value over positions) before selecting top-k latents, yielding one fixed
membership set applied at every position. For non-templatic data this is the
wrong granularity: the SAE is k-sparse PER POSITION, so a fixed circuit must
cover the union of each position's top-k (thousands) and deep seeds — whose
peak reads a long causal prefix of differently-coded positions — collapse to
near-zero circuit-only sufficiency.

This module keeps the position axis: one total-effect gradient pass (SFC
pass-through gradients, so credit reaches every upstream latent at every
position) attributes the seed's peak pre-activation to each (position, latent);
the allowed set is then the UNION over the seed's causal prefix of each
position's top-N latents by |attribution|. The circuit stays a flat membership
set (roles by attribution sign) — only the selection rule changes — so the
downstream assembly and the (free, zero-ablation) evaluation are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import torch

from circuit.types.feature_id import FeatureID
from model.hooks import multi_patch
from sae.dense import sparse_topk_to_dense

Site = Tuple[int, str]

SELECT_MODES = ("top_n", "abs", "relative", "mass", "abs_pctl")

# Cap on |attr| values sampled per tensor when resolving an "abs_pctl"
# threshold — bounds the quantile's memory without biasing it meaningfully.
_PCTL_SAMPLE_CAP = 200_000


def pooled_abs_threshold(attrs, pctl: float) -> float:
    """The ``abs_pctl`` admission threshold: the ``pctl``-th percentile of the
    POOLED nonzero |attr| distribution across the given attribution tensors
    (all sites of one attribution pass, and both role signals for cf).

    Pooling across sites is deliberate — it reproduces the validated
    experiment (thresh64): one global cut per attribution pass, so admission
    follows signal strength ACROSS sites rather than each site keeping its own
    top slice. Self-calibrating: "p90" means the same thing on every seed even
    though raw attribution scales differ by orders of magnitude."""

    samples = []
    for attr in attrs:
        vals = attr.abs()
        nz = vals[vals > 0]
        if nz.numel() > _PCTL_SAMPLE_CAP:
            idx = torch.randint(0, nz.numel(), (_PCTL_SAMPLE_CAP,), device=nz.device)
            nz = nz.flatten()[idx]
        if nz.numel():
            # Stay on the attribution's device: the quantile runs where the
            # data lives (profiled 2026-07-18: per-call .cpu() + CPU quantile
            # was ~480ms x hundreds of calls — 30% of a deep restoration arm).
            samples.append(nz.detach().to(torch.float32))
    if not samples:
        return float("inf")  # nothing attributed -> admit nothing
    device = samples[0].device
    pooled = torch.cat([s.to(device) for s in samples])
    return float(torch.quantile(pooled, pctl / 100.0))


def resolve_role_delivery(supports, negatives, *, position_aware: bool,
                          include_negatives: bool):
    """Role delivery under the position-aware both-sign semantics (2026-07-19).

    A position-aware allowed set is a STREAM-RECONSTRUCTION membership: a
    negative-attribution latent still carries content in the residual stream at
    its position, so it must stay a member. Therefore ``negative_roles`` cannot
    *remove* negatives from a PA circuit — it only chooses whether they are
    labelled as a distinct inhibitor role:

      include            -> (supports, negatives-as-inhibitors)
      exclude + PA       -> (supports + negatives folded in as |score|, {})
      exclude + NPA      -> (supports, {})   [classic: negatives genuinely dropped]

    So for PA the union of the two returned dicts' keys is IDENTICAL between
    include and exclude — only the role split differs, and free0/size are
    unchanged by the toggle. For NPA (seed-driving selection, not stream
    reconstruction) exclude removes negatives, the long-standing behaviour that
    the cf-local φcf finding depends on."""
    if include_negatives:
        return dict(supports), dict(negatives)
    if position_aware:
        folded = dict(supports)
        for fid, score in negatives.items():
            folded[fid] = abs(float(score))  # kept as a member, labelled support
        return folded, {}
    return dict(supports), {}


def _selection_mask(block_abs: torch.Tensor, select: str, threshold: float, n: int) -> torch.Tensor:
    """Per-position membership mask over a causal-prefix block of |attribution|.

    ``block_abs`` is ``[prefix, d_sae]`` (non-negative). Returns a boolean mask
    of the same shape marking the latents kept at each position. The rules:

    - ``top_n``:    fixed count — the ``n`` largest |attr| per position.
    - ``abs``:      global absolute cut — ``|attr| >= threshold``.
    - ``relative``: scale-free — ``|attr| >= threshold * max|attr|`` at that position.
    - ``mass``:     cumulative — the smallest set covering ``threshold`` of the
                    position's total |attr| mass (always keeps at least the top one).

    ``abs_pctl`` never reaches here: it is an ``abs`` cut whose threshold is the
    ``threshold``-th percentile of the pass's pooled nonzero |attr|, resolved by
    the attribution frontends (``pooled_abs_threshold`` /
    ``PositionAwareSpec.resolved_for``) before selection.

    The ``relative`` and ``mass`` rules normalise by a per-position quantity
    (row max / row total). A *dead* position — one the seed does not attribute
    to, so its whole row is ~0 — would otherwise divide by ~0 and select the
    entire dictionary, which the union then spreads across the site. Such
    positions are guarded to select **nothing** (a position the seed doesn't
    read must not contribute latents). ``abs`` needs no guard: a global cut
    already selects nothing at a dead position.
    """
    if select == "abs_pctl":
        raise ValueError(
            "select='abs_pctl' must be resolved to an 'abs' threshold by the "
            "attribution frontend (pooled_abs_threshold / resolved_for) before "
            "selection — it needs the pass's pooled |attr| distribution."
        )
    if select == "top_n":
        n = min(n, block_abs.shape[-1])
        mask = torch.zeros_like(block_abs, dtype=torch.bool)
        if n > 0:
            _, idx = block_abs.topk(n, dim=-1)
            mask.scatter_(-1, idx, True)
        return mask
    if select == "abs":
        return block_abs >= threshold
    # `relative` / `mass`: a position is "live" only if its total |attr| is a
    # non-negligible fraction of the most-attributed position's — dead rows
    # (total ~0) contribute nothing rather than the whole dictionary.
    row_total = block_abs.sum(dim=-1, keepdim=True)
    live = row_total > 1e-6 * row_total.amax().clamp_min(1e-12)
    if select == "relative":
        rowmax = block_abs.amax(dim=-1, keepdim=True)
        return (block_abs >= (threshold * rowmax)) & live
    if select == "mass":
        order = block_abs.argsort(dim=-1, descending=True)
        sorted_abs = torch.gather(block_abs, -1, order)
        total = sorted_abs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # Exclusive cumulative fraction: keep an element while the mass *before*
        # it is still below threshold — this keeps the smallest set reaching the
        # threshold (including the crossing element) and always keeps the top one.
        frac_excl = (sorted_abs.cumsum(dim=-1) - sorted_abs) / total
        keep_sorted = frac_excl < threshold
        mask = torch.zeros_like(block_abs, dtype=torch.bool)
        mask.scatter_(-1, order, keep_sorted)
        return mask & live
    raise ValueError(f"unknown position_aware select mode: {select!r}")


@dataclass
class PositionAwareSpec:
    """Carries the seed's per-sequence anchor positions plus the selection rule,
    so any attribution frontend (activation-gradient / abl / cf) can swap its
    ``.sum(dim=(0, 1))`` position-collapse for the position-aware union."""

    peaks: torch.Tensor          # [B] seed anchor position per sequence
    top_n: int = 64
    select: str = "top_n"
    threshold: float = 0.0       # abs: raw cut; abs_pctl: PERCENTILE (0-100)
    position_weight: bool = False
    scope: str = "aggregate"

    def select_from(self, attr: torch.Tensor) -> Dict[int, float]:
        return select_position_aware(
            attr, self.peaks, top_n=self.top_n, select=self.select,
            threshold=self.threshold, position_weight=self.position_weight,
            scope=self.scope,
        )

    def resolved_for(self, attrs) -> "PositionAwareSpec":
        """For ``select="abs_pctl"``: resolve this pass's admission threshold
        from the pooled nonzero |attr| distribution of ``attrs`` (every site's
        attribution tensor for the pass) and return the equivalent ``abs``
        spec. Any other select mode returns self unchanged. Callers pool
        ACROSS sites, once per attribution pass."""
        if self.select != "abs_pctl":
            return self
        return PositionAwareSpec(
            peaks=self.peaks, top_n=self.top_n, select="abs",
            threshold=pooled_abs_threshold(attrs, self.threshold),
            position_weight=self.position_weight, scope=self.scope,
        )


def select_position_aware(
    attr: torch.Tensor,
    peaks: torch.Tensor,
    *,
    top_n: int,
    select: str = "top_n",
    threshold: float = 0.0,
    position_weight: bool = False,
    scope: str = "aggregate",
) -> Dict[int, float]:
    """Position-aware selection backend, shared by every attribution frontend.

    ``attr`` is ONE site's per-position attribution ``[B, T, d_sae]`` — whatever
    the method computed *before* it would have collapsed positions with
    ``.sum(dim=(0, 1))``. ``peaks`` is ``[B]``, the seed's anchor position per
    sequence (its firing peak on posctx, or its would-be-firing argmax on negctx).

    Returns ``{latent -> signed score}``: for each sequence, each position in the
    seed's causal prefix (<= peak) contributes its selected latents; the union is
    taken across positions and sequences, keeping each latent's largest-|score|.

    This is the ONLY thing that differs between a classic and a position-aware
    method: classic sums the position axis away, position-aware unions over it.
    """
    if attr.dim() != 3:
        raise ValueError(f"attr must be [B, T, d_sae], got {tuple(attr.shape)}")
    B = min(attr.shape[0], peaks.shape[0])
    selected: Dict[int, float] = {}
    b_range = range(1) if scope == "per_instance" else range(B)
    for b in b_range:
        prefix = int(peaks[b].item()) + 1
        block = attr[b, :prefix]  # [prefix, d_sae]
        if position_weight:
            strength = block.abs().sum(dim=-1, keepdim=True)
            block = block * (strength / strength.amax().clamp_min(1e-12))
        mask = _selection_mask(block.abs(), select, threshold, top_n)
        rows, lats = mask.nonzero(as_tuple=True)
        vals = block[rows, lats]
        for lat, val in zip(lats.tolist(), vals.tolist()):
            if lat not in selected or abs(val) > abs(selected[lat]):
                selected[lat] = val
    return selected


def select_position_aware_values(
    attr: torch.Tensor,
    peaks: torch.Tensor,
    *,
    top_n: int,
    select: str = "top_n",
    threshold: float = 0.0,
    position_weight: bool = False,
    scope: str = "aggregate",
) -> torch.Tensor:
    """Tensor twin of ``select_position_aware``: same per-position selection and
    largest-|score| union, returned as a ``[d_sae]`` tensor on ``attr``'s device
    (signed value of each selected latent; 0 = unselected). Built for hot loops
    (restoration rounds) that would otherwise pay a Python dict merge per
    chunk — convert to a dict ONCE after all chunks are merged.

    Tie behavior matches the dict path: within a block, ``abs().argmax`` takes
    the first position with the maximal |score| (the dict kept the first
    encountered under a strict ``>``); across sequences/chunks the strict ``>``
    keeps the earlier value. A selected-but-zero-valued entry is
    indistinguishable from unselected — identical to the zero-admission gate
    the restoration loop applies to the dict."""
    if attr.dim() != 3:
        raise ValueError(f"attr must be [B, T, d_sae], got {tuple(attr.shape)}")
    B = min(attr.shape[0], peaks.shape[0])
    out = torch.zeros(attr.shape[-1], dtype=torch.float32, device=attr.device)
    b_range = range(1) if scope == "per_instance" else range(B)
    for b in b_range:
        prefix = int(peaks[b].item()) + 1
        block = attr[b, :prefix]  # [prefix, d_sae]
        if position_weight:
            strength = block.abs().sum(dim=-1, keepdim=True)
            block = block * (strength / strength.amax().clamp_min(1e-12))
        mask = _selection_mask(block.abs(), select, threshold, top_n)
        vals = torch.where(mask, block, torch.zeros((), dtype=block.dtype,
                                                    device=block.device))
        idx = vals.abs().argmax(dim=0)  # first maximal position per latent
        cand = vals.gather(0, idx.unsqueeze(0)).squeeze(0).to(torch.float32)
        out = torch.where(cand.abs() > out.abs(), cand, out)
    return out


class _PositionAttrInstrument:
    """Taps every upstream site as a differentiable leaf with an identity
    pass-through, and captures the seed's per-position pre-activation."""

    def __init__(self, bank: Any, sites: Set[Site], seed_layer: int, seed_kind: str,
                 w_seed: torch.Tensor, b_seed: torch.Tensor) -> None:
        self.bank = bank
        self.sites = sites
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.w_seed = w_seed
        self.b_seed = b_seed
        self.leaves: Dict[Site, torch.Tensor] = {}
        self.naturals: Dict[Site, torch.Tensor] = {}
        self.seed_pre: Optional[torch.Tensor] = None

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b  # [B, T]
            return x
        site = (layer_idx, kind)
        if site not in self.sites:
            return x
        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(top_acts, top_indices, self.bank.d_sae, dtype=x.dtype)
        self.naturals[site] = dense.detach()
        leaf = dense.detach().requires_grad_(True)
        self.leaves[site] = leaf
        residual = (x - self.bank.decode(dense, kind, layer_idx)).detach()
        # decode(leaf) is differentiable; (x - x.detach()) carries the identity
        # gradient so credit flows through the stream to upstream leaves.
        return self.bank.decode(leaf, kind, layer_idx) + residual + (x - x.detach())


def position_aware_membership(
    inference: Any,
    bank: Any,
    *,
    tokens: torch.Tensor,
    sites: Set[Site],
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    pos_argmax: torch.Tensor,
    top_n: int,
    select: str = "top_n",
    threshold: float = 0.0,
    position_weight: bool = False,
    scope: str = "aggregate",
    negative_roles: bool = True,
    batch_size: Optional[int] = None,
) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
    """One total-effect gradient pass; allowed set = union over each seed's
    causal prefix (positions <= peak) of that position's selected latents by
    |attribution| (attribution = grad(seed peak) * natural value).

    ``batch_size``: sequences per grad-enabled forward (VRAM bound). ``tokens``
    may carry more; chunks are merged by max-|score| union — the same rule the
    selection applies across sequences within one pass, so chunking does not
    change the membership. None (default) = single pass (historical
    behaviour). The objective is a SUM over sequences, so per-sequence
    attributions are batch-independent and need no reweighting.

    ``select`` chooses the per-position rule (see :func:`_selection_mask`):
    ``"top_n"`` keeps a fixed ``top_n`` per position; ``"abs"`` / ``"relative"``
    / ``"mass"`` keep a variable count governed by ``threshold``, to shrink the
    union when the attribution is peaked.

    ``position_weight``: scale each (position, site) block's attributions by that
    block's normalised strength (Σ|attr| over latents, /max over positions), so
    latents at weakly-read positions score lower and fall below the threshold /
    get pruned first. A gradient proxy for attention routing — the attn-mediated
    "which positions does the seed read" signal — applied as a soft down-weight
    rather than a hard position drop. Most effective with ``select="abs"`` and
    the downstream magnitude prune (a per-position scalar can't reorder a fixed
    per-position top-N).

    ``scope``: ``"aggregate"`` unions the membership over the whole probe batch
    (one circuit for the seed); ``"per_instance"`` builds it from a single
    sequence (b=0) — the per-input "meal"-sized circuit for a specific example.

    Returns (supports, inhibitors). ALL selected latents are members either way
    (the allowed set is a STREAM-RECONSTRUCTION set — a negative-attribution
    latent still contributes to the residual stream at its position, so zeroing
    it corrupts free execution): ``negative_roles`` does NOT gate membership,
    only the role split. With ``negative_roles=True`` (include) the negatives
    come back as ``inhibitors``; with False (exclude) they are folded into
    ``supports`` as |score|. So the union of the two dicts is identical either
    way — free0/size are unchanged by the toggle (see resolve_role_delivery)."""

    if not sites:
        return {}, {}
    if select not in SELECT_MODES:
        raise ValueError(f"select must be one of {SELECT_MODES}, got {select!r}")
    if scope not in ("aggregate", "per_instance"):
        raise ValueError(f"scope must be 'aggregate' or 'per_instance', got {scope!r}")
    sae = bank.saes[seed_kind][seed_layer]
    w_seed = sae.encoder.weight[seed_latent_idx].detach()
    b_seed = sae._get_bias_eff()[seed_latent_idx].detach()

    B_total = int(tokens.shape[0])
    bs = B_total if batch_size is None else max(1, int(batch_size))
    if scope == "per_instance":
        # Per-instance membership reads only sequence b=0 — one chunk suffices.
        B_total = min(B_total, bs)

    members: Dict[FeatureID, float] = {}
    inference.disable_compile()
    try:
        for start in range(0, B_total, bs):
            tokens_chunk = tokens[start:start + bs]
            argmax_chunk = pos_argmax[start:start + bs]

            ins = _PositionAttrInstrument(bank, sites, seed_layer, seed_kind, w_seed, b_seed)
            inference.forward(tokens_chunk, patcher=ins, grad_enabled=True,
                              return_activations=False, tokenize_final=False)
            if ins.seed_pre is None:
                raise RuntimeError("seed pre-activation was not captured")

            pre = ins.seed_pre
            B = min(pre.shape[0], argmax_chunk.shape[0])
            peaks = argmax_chunk[:B].to(pre.device).clamp(0, pre.shape[1] - 1)
            # SUM over sequences: each sequence's gradient is independent of
            # the batch it rode in with, so chunked == single-pass exactly.
            objective = pre[:B][torch.arange(B, device=pre.device), peaks].sum()

            sites_order = sorted(ins.leaves)
            grads = torch.autograd.grad(objective, [ins.leaves[s] for s in sites_order],
                                        allow_unused=True)

            # grad x natural per site — the grads tuple already holds all sites
            # simultaneously, so materialising the attrs adds one elementwise
            # product per site, not a new memory regime.
            attrs = {
                site: (grad.detach().to(torch.float32)
                       * ins.naturals[site].to(torch.float32))  # [B, T, d_sae]
                for site, grad in zip(sites_order, grads)
                if grad is not None
            }
            # abs_pctl: one pooled cut per pass across ALL sites, so admission
            # follows signal strength across sites (the validated thresh64
            # protocol) rather than each site keeping its own slice.
            chunk_select, chunk_threshold = select, threshold
            if select == "abs_pctl":
                chunk_select = "abs"
                chunk_threshold = pooled_abs_threshold(attrs.values(), threshold)

            for site, attr in attrs.items():
                layer, kind = site
                # Per-position selection over each sequence's causal prefix, by
                # |attribution| — both signs are members (stream reconstruction).
                # The rule (top_n / abs / relative / mass) is set by `select`.
                selected = select_position_aware(
                    attr, peaks, top_n=top_n, select=chunk_select, threshold=chunk_threshold,
                    position_weight=position_weight, scope=scope,
                )
                for lat, val in selected.items():
                    # Keep the SIGNED value (merged by |val| across chunks /
                    # sequences) so the role split below can label inhibitors;
                    # membership is by magnitude, sign carries the role.
                    fid = FeatureID(layer=layer, kind=kind, index=lat)
                    if fid not in members or abs(val) > abs(members[fid]):
                        members[fid] = val
            del ins
    finally:
        inference.enable_compile()
    supports = {fid: v for fid, v in members.items() if v >= 0}
    negatives = {fid: v for fid, v in members.items() if v < 0}
    return resolve_role_delivery(supports, negatives, position_aware=True,
                                 include_negatives=negative_roles)


__all__ = [
    "position_aware_membership",
    "select_position_aware",
    "select_position_aware_values",
    "resolve_role_delivery",
    "PositionAwareSpec",
    "SELECT_MODES",
]
