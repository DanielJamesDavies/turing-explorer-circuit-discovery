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
    pin_values: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    logger: Any = None,
) -> List[str]:
    """Prune ``circuit`` in place to the smallest magnitude-ranked prefix that
    holds φ ≥ floor, where floor = ``target`` if ``target > 0`` else
    ``base_φ - tolerance`` (base_φ = full-circuit φ). Returns removed uuids.

    ``objective`` picks which sufficiency the prune preserves:
      * ``"free"`` (default): free0-φ, kept latents re-encode live against the
        ZERO-ablation floor → a self-contained (CLOSED) circuit.
      * ``"free_mean_dense"``: free-φ against the DENSE MEAN floor — i.e. the
        SFC-standard faithfulness metric (freeM_dense). Use this when the target
        metric is freeM_dense: bisecting on "free" optimises free0 and lets
        freeM_dense collapse under compression (the dense-fill off-manifold
        artefact), so the two must be matched. Prunes against exactly what is
        reported.
      * ``"free_mean_topk"``: same but with the on-manifold k-sparse fill
        (freeM_topk) — the honest fill that keeps the stream in the model's
        natural k-sparse regime.
      * ``"pinned"``: pinned-φ, kept latents clamped to their clean position-
        specific values → the causal DRIVERS. Prunes hardest.

    Monotonicity: bisection assumes φ non-decreasing in K. This holds
    approximately for free0/pinned. The mean-floor objectives are LESS monotone
    (a low-mass member can lower freeM_dense via the off-manifold fill), so the
    found K may sit slightly above the true knee — but the RESULT is always
    validated to meet the floor, so the target is never violated."""

    valid = ("free", "free_mean_dense", "free_mean_topk", "pinned")
    if objective not in valid:
        raise ValueError(f"objective must be one of {valid}, got {objective!r}")
    # mean-floor objectives evaluate free-φ against the site means; the dense vs
    # topk split is the fill regime.
    use_means = objective in ("free_mean_dense", "free_mean_topk")
    respect_topk = objective == "free_mean_topk"

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

    # Mean-floor objectives need the per-site means as the ablation floor; the
    # empty circuit and every phi(k) are then measured against that same floor,
    # so the bisected quantity IS freeM_dense/freeM_topk.
    site_means = None
    if use_means:
        from eval.floors import collect_site_means
        site_means = collect_site_means(inference, sae_bank, pos_tokens, in_scope)

    a_empty = circuit_only_activation(
        inference, sae_bank, {}, in_scope, pos_tokens,
        seed_layer, seed_kind, seed_latent_idx, pos_argmax=pos_argmax,
        site_means=site_means, respect_topk=respect_topk,
    )
    denom = a_posctx - a_empty
    if abs(denom) < 1e-9:
        return []  # degenerate: nothing to normalise against

    # For the "pinned" (drivers) objective, clamp kept latents to their clean
    # position-specific values; collected once (circuit-independent anchor).
    # Position-specific pins are [B, T, d_sae] PER SITE (~671 MB each at 64x64
    # with d_sae=40960) and depend only on (tokens, sites) — not on the circuit.
    # Callers sweeping several pruned variants of one circuit should collect them
    # once and pass them in; otherwise every pinned invocation rebuilds the same
    # multi-gigabyte structure.
    if objective == "pinned" and pin_values is None:
        from eval.floors import collect_site_anchors
        _, pin_values = collect_site_anchors(
            inference, sae_bank, pos_tokens, in_scope, pos_argmax,
            pin_position_specific=True,
        )
    elif objective != "pinned":
        pin_values = None

    cache: Dict[int, float] = {}

    def phi(k: int) -> float:
        if k not in cache:
            a_c = circuit_only_activation(
                inference, sae_bank, keep_of(k), in_scope, pos_tokens,
                seed_layer, seed_kind, seed_latent_idx, pos_argmax=pos_argmax,
                site_means=site_means, pin_values=pin_values,
                respect_topk=respect_topk,
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

    # Record the pre-prune member count + φ so callers can report the
    # compression ratio without a second (unpruned) run.
    circuit.metadata["n_members_pre_prune"] = n_total
    circuit.metadata["prune_phi_base"] = float(base_phi)
    circuit.metadata["prune_phi_kept"] = float(phi(best))
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
