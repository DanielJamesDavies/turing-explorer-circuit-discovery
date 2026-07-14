"""Global magnitude prune with bisection search on free-φ (allowed-set sufficiency).

Unlike leave-one-out minimality (``eval/minimality.py``) — which costs one forward
pass per node per round and is blind to redundant coverage (any single member of a
highly redundant union looks individually removable) — this prune ranks every
non-seed member by ``|attribution_score|`` and keeps the smallest top-K prefix whose
circuit-only (free) activation still meets a sufficiency target. The crossing K is
found by binary search over K, so the whole prune is ~log2(N) forward passes rather
than O(N); it scales to the 10^4–10^5-node position-aware allowed sets where LOO is
intractable.

Method-agnostic: it operates on any assembled circuit with scored member nodes plus a
seed node, so it applies uniformly to counterfactual-gradient, ablation-gradient, and
(fused) hybrid-gradient circuits.

Caveat: bisection assumes free-φ is monotone non-decreasing in K (more of the
magnitude-ranked members ⇒ at least as sufficient). This holds approximately for a
magnitude-sorted keep set. The *result* is always validated — the kept K provably
meets the target — so non-monotonicity can only make K slightly larger than the true
knee, never violate the target.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from eval.ablation_faithfulness import (
    circuit_only_activation,
    measure_seed_activation,
    upstream_sites,
)


@torch.no_grad()
def prune_by_magnitude_bisection(
    inference: Any,
    sae_bank: Any,
    circuit: Any,
    *,
    pos_tokens: torch.Tensor,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    pos_argmax: Optional[torch.Tensor] = None,
    tolerance: float = 0.05,
    target: float = 0.0,
    min_keep: int = 1,
    objective: str = "free",
    logger: Any = None,
) -> List[str]:
    """Prune ``circuit`` in place to the smallest magnitude-ranked prefix that
    holds φ ≥ floor, where floor = ``target`` if ``target > 0`` else
    ``base_φ - tolerance`` (base_φ = full-circuit φ). Returns removed uuids.

    ``objective`` picks which sufficiency the prune preserves — the two ends of the
    drivers-vs-closure decomposition:
      * ``"free"`` (default): free-φ, kept latents re-encode live → a self-contained
        (CLOSED) circuit. Large for deep seeds (closure is distributed).
      * ``"pinned"``: pinned-φ, kept latents clamped to their clean position-specific
        values → the causal DRIVERS (node selection). Compact — the drivers are
        concentrated, so this prunes far harder than "free"."""

    if objective not in ("free", "pinned"):
        raise ValueError(f"objective must be 'free' or 'pinned', got {objective!r}")

    members: List[Tuple[float, str, Any]] = []
    for uuid, node in circuit.nodes.items():
        if node.metadata.get("role") == "seed":
            continue
        fid = node.feature_id
        if fid is None:
            continue
        score = abs(float(node.metadata.get("attribution_score") or 0.0))
        members.append((score, uuid, fid))

    n_total = len(members)
    if n_total <= min_keep:
        return []
    members.sort(key=lambda m: m[0], reverse=True)  # strongest |attr| first

    in_scope = upstream_sites(sae_bank, seed_layer, seed_kind)

    def keep_of(k: int) -> Dict[Tuple[int, str], Set[int]]:
        d: Dict[Tuple[int, str], Set[int]] = {}
        for _, _, fid in members[:k]:
            d.setdefault((fid.layer, fid.kind), set()).add(fid.index)
        return d

    a_posctx = measure_seed_activation(
        inference, sae_bank, pos_tokens, seed_layer, seed_kind, seed_latent_idx, pos_argmax
    )
    a_empty = circuit_only_activation(
        inference, sae_bank, {}, in_scope, pos_tokens,
        seed_layer, seed_kind, seed_latent_idx, pos_argmax=pos_argmax,
    )
    denom = a_posctx - a_empty
    if abs(denom) < 1e-9:
        return []  # degenerate: nothing to normalise against

    # For the "pinned" (drivers) objective, clamp kept latents to their clean
    # position-specific values; collected once (circuit-independent anchor).
    pin_values = None
    if objective == "pinned":
        from eval.floors import collect_site_anchors
        _, pin_values = collect_site_anchors(
            inference, sae_bank, pos_tokens, in_scope, pos_argmax,
            pin_position_specific=True,
        )

    cache: Dict[int, float] = {}

    def phi(k: int) -> float:
        if k not in cache:
            a_c = circuit_only_activation(
                inference, sae_bank, keep_of(k), in_scope, pos_tokens,
                seed_layer, seed_kind, seed_latent_idx, pos_argmax=pos_argmax,
                pin_values=pin_values,
            )
            cache[k] = (a_c - a_empty) / denom
        return cache[k]

    base_phi = phi(n_total)
    floor = target if target > 0 else base_phi - tolerance
    if base_phi < floor:
        # Even the full circuit is below an absolute target — keep everything.
        return []

    # Smallest K in [min_keep, n_total] with phi(K) >= floor. phi(n_total) = base_phi
    # >= floor by construction, so a crossing always exists.
    lo, hi, best = min_keep, n_total, n_total
    while lo <= hi:
        mid = (lo + hi) // 2
        if phi(mid) >= floor:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    removed = [uuid for _, uuid, _ in members[best:]]
    if not removed:
        return []
    rm = set(removed)
    for uuid in removed:
        circuit.nodes.pop(uuid, None)
    circuit.edges = [
        e for e in circuit.edges
        if e.source_uuid not in rm and e.target_uuid not in rm
    ]

    if logger is not None:
        logger.stage(
            "after magnitude prune",
            len(circuit.nodes), len(circuit.edges),
            note=(f"kept top-{best}/{n_total} by |attr|, removed {len(removed)} | "
                  f"{objective}-φ {base_phi:.3f}→{phi(best):.3f} (floor {floor:.3f})"),
        )
    return removed


__all__ = ["prune_by_magnitude_bisection"]
