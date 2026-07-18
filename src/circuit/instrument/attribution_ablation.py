"""Ablation-gradient scoring helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from circuit.types.feature_id import FeatureID
from pipeline.component_index import component_idx, get_all_upstream_components

from .position_aware import PositionAwareSpec
from .sae_graph import FeatureGraph


def compute_latent_ablation_scores(
    graph: FeatureGraph,
    target_scalar: torch.Tensor,
    seed_comp_idx: int,
    n_kinds: int,
    kinds: List[str],
    top_k_supports: int,
    min_active_count: int,
    active_count: torch.Tensor,
    top_k_scope: str = "global",
    position_aware: Optional["PositionAwareSpec"] = None,
) -> Dict[FeatureID, float]:
    """
    Score active upstream latents by first-order benefit from ablating them.

    ``target_scalar`` should be a positive-context seed-loss objective, e.g.
    seed pre-activation squared at the seed's peak positions. For an active
    latent f with gradient g = d(loss)/df, ablating f to zero has first-order
    change ``-f*g``. Therefore ``f*g > 0`` means ablation should reduce loss.

    position_aware:
        When given, the position axis is UNIONED over the seed's causal prefix
        instead of collapsed with .sum(dim=(0, 1)), and top_k_scope/top_k_* are
        bypassed — the same swap the cf scorer makes. Two semantic notes:

        * Membership keeps BOTH signs at |score| (allowed-set / stream-
          reconstruction semantics, matching activation_gradient), unlike the
          classic path's positive-support filter — a negatively-scored latent
          still carries stream content at its position, so zeroing it corrupts
          free execution.
        * The per-position signal is ``acts x grad(preact^2)`` =
          ``2*preact x (acts x grad(preact))`` — activation_gradient's
          attribution times a positive per-(sequence, position) drive weight.
          Under per-position "top_n" the rescaling cannot reorder a row, so
          membership is expected to match activation_gradient's; under
          "abs"/"abs_pctl" the pooled cut favours hard-firing positions — the
          drive-weighted variant this branch exists to expose.
    """

    if target_scalar.grad_fn is None:
        return {}

    allowed_components = set(
        get_all_upstream_components(
            seed_comp_idx,
            n_kinds,
            kinds,
            include_same_layer=True,
        )
    )
    upstream_pairs: List[Tuple[int, str]] = []
    for layer_p, kind_p in graph.activations:
        if kind_p not in kinds:
            continue
        comp_p = component_idx(layer_p, kinds.index(kind_p), n_kinds)
        if comp_p in allowed_components:
            upstream_pairs.append((layer_p, kind_p))

    if not upstream_pairs:
        return {}

    anchors: List[torch.Tensor] = []
    for layer_p, kind_p in upstream_pairs:
        for acts_grad, _, _ in graph.activations[(layer_p, kind_p)]:
            if acts_grad.act is not None:
                anchors.append(acts_grad.act)
            if acts_grad.res is not None:
                anchors.append(acts_grad.res)

    if not anchors:
        return {}

    grads = torch.autograd.grad(target_scalar, anchors, retain_graph=False, allow_unused=True)
    all_supports: List[Tuple[FeatureID, float]] = []
    # Position-aware selection is deferred until every site's gradients are in
    # hand, so an "abs_pctl" spec can resolve ONE pooled admission threshold
    # across all sites (mirrors the cf scorer). Entries hold references into
    # `grads`/the graph — no extra tensor residency.
    pa_entries: List[Tuple[int, str, torch.Tensor, torch.Tensor,
                           Optional[torch.Tensor]]] = []
    grad_iter = iter(grads)

    for layer_p, kind_p in upstream_pairs:
        kind_idx_p = kinds.index(kind_p)
        comp_p = component_idx(layer_p, kind_idx_p, n_kinds)
        count_mask: Optional[torch.Tensor] = None
        if active_count is not None:
            count_mask = (active_count[comp_p] >= min_active_count).to(graph.device)

        for acts_grad, _, _ in graph.activations[(layer_p, kind_p)]:
            grad_act = next(grad_iter) if acts_grad.act is not None else None
            _grad_res = next(grad_iter) if acts_grad.res is not None else None
            if grad_act is None or acts_grad.act is None:
                continue

            if position_aware is not None:
                pa_entries.append((layer_p, kind_p, acts_grad.act, grad_act, count_mask))
                continue

            scores = (acts_grad.act * grad_act).sum(dim=(0, 1))
            if count_mask is not None:
                scores = scores * count_mask.to(scores.device)

            pos_nz = (scores > 0).nonzero(as_tuple=False).squeeze(1)
            if pos_nz.numel() == 0:
                continue
            pos_vals = scores[pos_nz]
            if top_k_scope == "layer_kind":
                k = min(top_k_supports, pos_nz.numel())
                topk_vals, topk_local = pos_vals.topk(k)
                sel_idx = pos_nz[topk_local]
            else:
                sel_idx, topk_vals = pos_nz, pos_vals
            for idx_int, score in zip(sel_idx.cpu().tolist(), topk_vals.cpu().tolist()):
                all_supports.append((FeatureID(layer=layer_p, kind=kind_p, index=idx_int), float(score)))

    if position_aware is not None and pa_entries:
        def _attr_stream():
            # Streamed so each acts x grad product is sampled then freed.
            for (_, _, acts, grad_act, _) in pa_entries:
                yield acts * grad_act

        spec = position_aware.resolved_for(_attr_stream())
        for layer_p, kind_p, acts, grad_act, count_mask in pa_entries:
            for latent, score in spec.select_from(acts * grad_act).items():
                if count_mask is not None and not bool(count_mask[latent]):
                    continue
                # Both signs are members at |score| (allowed-set semantics —
                # see docstring); the union already replaced the ranking, so
                # no top-k truncation applies below.
                all_supports.append(
                    (FeatureID(layer=layer_p, kind=kind_p, index=latent), abs(float(score)))
                )
        return {fid: score for fid, score in all_supports}

    if top_k_scope == "global":
        all_supports.sort(key=lambda item: item[1], reverse=True)
        all_supports = all_supports[:top_k_supports]

    return {fid: score for fid, score in all_supports}


__all__ = ["compute_latent_ablation_scores"]
