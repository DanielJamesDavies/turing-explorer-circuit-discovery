"""Counterfactual attribution scoring helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from circuit.types.feature_id import FeatureID

from .sae_graph import FeatureGraph


def compute_latent_counterfactual_scores(
    graph: FeatureGraph,
    target_scalar: torch.Tensor,
    seed_layer: int,
    n_kinds: int,
    kinds: List[str],
    top_k_activators: int,
    top_k_inhibitors: int,
    min_active_count: int,
    active_count: torch.Tensor,
    top_k_scope: str = "global",
) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
    """
    Single-backward-pass counterfactual scoring for negctx sequences.

    Unlike compute_latent_upstream_scores (which scores only direct predecessor
    components of one target node), this function scores ALL upstream latents
    across ALL layers <= seed_layer in one backward pass.  The gradient flows
    through the full upstream computation graph automatically.

    target_scalar must be constructed by the caller (e.g. MSE loss against the
    seed's encoder pre-activation via SeedProjectionInstrument) so that it is
    non-zero even when the seed latent has zero SAE activation on negctx.

    Two score types are extracted from the same set of gradients:

    - activator_scores:  raw gradient sum(dim=(0,1)) — NOT scaled by activation.
      Positive gradient means "this latent would push the seed pre-activation
      upward if it were active."  Finds absent activators regardless of whether
      they are currently firing on negctx.

    - inhibitor_scores:  (acts * grad).sum(dim=(0,1)) where the product is
      negative.  The latent is currently active AND causally suppressing the
      seed's pre-activation.

    Args:
        graph:            FeatureGraph from a grad-enabled SAEGraphInstrument forward.
        target_scalar:    Differentiable scalar loss (e.g. -MSE of seed pre-activation).
        seed_layer:       Only layers <= seed_layer are scored.
        n_kinds:          Number of SAE kinds (e.g. 3 for attn/mlp/resid).
        kinds:            Ordered list of kind strings (e.g. ["attn", "mlp", "resid"]).
        top_k_activators: Max absent activators to return per scope unit.
        top_k_inhibitors: Max present inhibitors to return per scope unit.
        min_active_count: Latents below this global firing count are excluded.
        active_count:     [n_components, d_sae] tensor of lifetime firing counts.
        top_k_scope:      "global"     — single ranked list across all (layer, kind) pairs,
                                        sliced to top_k_activators / top_k_inhibitors.
                          "layer_kind" — top-K applied independently per (layer, kind) pair;
                                        results from all pairs are merged without re-ranking.

    Returns:
        (activator_scores, inhibitor_scores) — two FeatureID → float dicts.
    """
    from pipeline.component_index import component_idx as _component_idx

    _empty: Tuple[Dict[FeatureID, float], Dict[FeatureID, float]] = ({}, {})

    if target_scalar.grad_fn is None:
        return _empty

    # All upstream (layer, kind) pairs present in the graph at or below seed_layer
    upstream_pairs: List[Tuple[int, str]] = [
        (layer_p, kind_p)
        for (layer_p, kind_p) in graph.activations
        if layer_p <= seed_layer
    ]

    if not upstream_pairs:
        return _empty

    # Collect leaf anchors in a stable iteration order
    anchors: List[torch.Tensor] = []
    for layer_p, kind_p in upstream_pairs:
        for acts_grad, _, _ in graph.activations[(layer_p, kind_p)]:
            if acts_grad.act is not None:
                anchors.append(acts_grad.act)
            if acts_grad.res is not None:
                anchors.append(acts_grad.res)

    if not anchors:
        return _empty

    grads = torch.autograd.grad(target_scalar, anchors, retain_graph=False, allow_unused=True)

    all_activators: List[Tuple[FeatureID, float]] = []
    all_inhibitors: List[Tuple[FeatureID, float]] = []

    grad_iter = iter(grads)
    for layer_p, kind_p in upstream_pairs:
        kind_idx_p = kinds.index(kind_p)
        c_idx_p = _component_idx(layer_p, kind_idx_p, n_kinds)

        for acts_grad, _, _ in graph.activations[(layer_p, kind_p)]:
            grad_act = next(grad_iter) if acts_grad.act is not None else None
            _grad_res = next(grad_iter) if acts_grad.res is not None else None  # consumed, unused

            if grad_act is None:
                continue

            # Build active_count mask on the correct device
            count_mask: Optional[torch.Tensor] = None
            if active_count is not None:
                count_mask = (active_count[c_idx_p] >= min_active_count).to(grad_act.device)

            # --- Activator scores: raw gradient, NOT scaled by activation ---
            # Finds absent activators: high positive gradient even when acts ≈ 0.
            scores_act = grad_act.sum(dim=(0, 1))  # [d_sae]
            if count_mask is not None:
                scores_act = scores_act * count_mask

            pos_nz = (scores_act > 0).nonzero(as_tuple=False).squeeze(1)
            if pos_nz.numel() > 0:
                pos_vals = scores_act[pos_nz]
                if top_k_scope == "layer_kind":
                    k = min(top_k_activators, pos_nz.numel())
                    topk_vals, topk_local = pos_vals.topk(k)
                    sel_idx = pos_nz[topk_local]
                else:
                    sel_idx, topk_vals = pos_nz, pos_vals
                for idx_int, score in zip(sel_idx.cpu().tolist(), topk_vals.cpu().tolist()):
                    all_activators.append((FeatureID(layer=layer_p, kind=kind_p, index=idx_int), score))

            # --- Inhibitor scores: acts * grad where product < 0 ---
            # Finds present inhibitors: active latents causally suppressing the seed.
            scores_inh = (acts_grad.act * grad_act).sum(dim=(0, 1))  # [d_sae]
            if count_mask is not None:
                scores_inh = scores_inh * count_mask

            neg_nz = (scores_inh < 0).nonzero(as_tuple=False).squeeze(1)
            if neg_nz.numel() > 0:
                neg_vals = scores_inh[neg_nz]
                if top_k_scope == "layer_kind":
                    k = min(top_k_inhibitors, neg_nz.numel())
                    # most-negative first: topk on negated values
                    topk_vals, topk_local = (-neg_vals).topk(k)
                    topk_vals = -topk_vals
                    sel_idx = neg_nz[topk_local]
                else:
                    sel_idx, topk_vals = neg_nz, neg_vals
                for idx_int, score in zip(sel_idx.cpu().tolist(), topk_vals.cpu().tolist()):
                    all_inhibitors.append((FeatureID(layer=layer_p, kind=kind_p, index=idx_int), score))

    if top_k_scope == "global":
        # Single ranked list across all (layer, kind) pairs
        all_activators.sort(key=lambda x: x[1], reverse=True)
        all_inhibitors.sort(key=lambda x: x[1])
        all_activators = all_activators[:top_k_activators]
        all_inhibitors = all_inhibitors[:top_k_inhibitors]

    activator_dict = {fid: s for fid, s in all_activators}
    inhibitor_dict = {fid: s for fid, s in all_inhibitors}

    return activator_dict, inhibitor_dict


__all__ = ["compute_latent_counterfactual_scores"]
