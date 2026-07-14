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

from typing import Any, Dict, Optional, Set, Tuple

import torch

from circuit.types.feature_id import FeatureID
from model.hooks import multi_patch
from sae.dense import sparse_topk_to_dense

Site = Tuple[int, str]

SELECT_MODES = ("top_n", "abs", "relative", "mass")


def _selection_mask(block_abs: torch.Tensor, select: str, threshold: float, n: int) -> torch.Tensor:
    """Per-position membership mask over a causal-prefix block of |attribution|.

    ``block_abs`` is ``[prefix, d_sae]`` (non-negative). Returns a boolean mask
    of the same shape marking the latents kept at each position. The rules:

    - ``top_n``:    fixed count — the ``n`` largest |attr| per position.
    - ``abs``:      global absolute cut — ``|attr| >= threshold``.
    - ``relative``: scale-free — ``|attr| >= threshold * max|attr|`` at that position.
    - ``mass``:     cumulative — the smallest set covering ``threshold`` of the
                    position's total |attr| mass (always keeps at least the top one).

    The ``relative`` and ``mass`` rules normalise by a per-position quantity
    (row max / row total). A *dead* position — one the seed does not attribute
    to, so its whole row is ~0 — would otherwise divide by ~0 and select the
    entire dictionary, which the union then spreads across the site. Such
    positions are guarded to select **nothing** (a position the seed doesn't
    read must not contribute latents). ``abs`` needs no guard: a global cut
    already selects nothing at a dead position.
    """
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
) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
    """One total-effect gradient pass; allowed set = union over each seed's
    causal prefix (positions <= peak) of that position's selected latents by
    |attribution| (attribution = grad(seed peak) * natural value).

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

    Returns (members, {}) — ALL selected latents are members (kept / allowed to
    fire), because the allowed set is a STREAM-RECONSTRUCTION set: a latent with
    negative attribution to the seed still contributes to the residual stream at
    its position, so zeroing it corrupts free execution. Sign is preserved in
    the value (activator/inhibitor is interpretive metadata, not a keep filter),
    so `negative_roles` does not gate membership here."""

    if not sites:
        return {}, {}
    if select not in SELECT_MODES:
        raise ValueError(f"select must be one of {SELECT_MODES}, got {select!r}")
    if scope not in ("aggregate", "per_instance"):
        raise ValueError(f"scope must be 'aggregate' or 'per_instance', got {scope!r}")
    sae = bank.saes[seed_kind][seed_layer]
    w_seed = sae.encoder.weight[seed_latent_idx].detach()
    b_seed = sae._get_bias_eff()[seed_latent_idx].detach()

    ins = _PositionAttrInstrument(bank, sites, seed_layer, seed_kind, w_seed, b_seed)
    inference.disable_compile()
    try:
        inference.forward(tokens, patcher=ins, grad_enabled=True,
                          return_activations=False, tokenize_final=False)
    finally:
        inference.enable_compile()
    if ins.seed_pre is None:
        raise RuntimeError("seed pre-activation was not captured")

    pre = ins.seed_pre
    B = min(pre.shape[0], pos_argmax.shape[0])
    peaks = pos_argmax[:B].to(pre.device).clamp(0, pre.shape[1] - 1)
    objective = pre[:B][torch.arange(B, device=pre.device), peaks].sum()

    sites_order = sorted(ins.leaves)
    grads = torch.autograd.grad(objective, [ins.leaves[s] for s in sites_order], allow_unused=True)

    members: Dict[FeatureID, float] = {}
    for site, grad in zip(sites_order, grads):
        if grad is None:
            continue
        layer, kind = site
        attr = (grad.detach().to(torch.float32) * ins.naturals[site].to(torch.float32))  # [B, T, d_sae]
        # Per-position selection over each sequence's causal prefix, by
        # |attribution| — both signs are members (stream reconstruction). The
        # rule (top_n / abs / relative / mass) is set by `select`.
        selected: Dict[int, float] = {}  # latent -> signed score (max |attr|)
        b_range = range(1) if scope == "per_instance" else range(B)
        for b in b_range:
            prefix = int(peaks[b].item()) + 1
            block = attr[b, :prefix]  # [prefix, d_sae]
            if position_weight:
                # Down-weight latents by their position's read-strength (Σ|attr|
                # over latents at that position), normalised across positions.
                strength = block.abs().sum(dim=-1, keepdim=True)  # [prefix, 1]
                block = block * (strength / strength.amax().clamp_min(1e-12))
            mask = _selection_mask(block.abs(), select, threshold, top_n)  # [prefix, d_sae]
            rows, lats = mask.nonzero(as_tuple=True)
            vals = block[rows, lats]
            for lat, val in zip(lats.tolist(), vals.tolist()):
                if lat not in selected or abs(val) > abs(selected[lat]):
                    selected[lat] = val
        for lat, val in selected.items():
            # Value is the |attribution| magnitude so every member is a
            # positive-scored support (the assembly drops negative-scored
            # ones); membership, not sign, is what the allowed set needs.
            members[FeatureID(layer=layer, kind=kind, index=lat)] = abs(val)
    return members, {}


__all__ = ["position_aware_membership"]
