"""Counterfactual attribution scoring helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from circuit.types.feature_id import FeatureID

from .position_aware import PositionAwareSpec
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
    position_aware: Optional["PositionAwareSpec"] = None,
    posctx_values: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
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

    - activator_scores:  the gradient, NOT scaled by the latent's negctx
      activation.  Positive gradient means "this latent would push the seed
      pre-activation upward if it were active."  Finds absent activators
      regardless of whether they are currently firing on negctx.  Supplying
      ``posctx_values`` scales this by the latent's posctx target instead
      (see that argument).

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
        position_aware:   When given (a PositionAwareSpec carrying the seed's per-sequence
                          anchor positions — here the negctx pre-activation argmax, i.e. the
                          "would-be firing" position — plus the selection rule), the position
                          axis is UNIONED over the seed's causal prefix instead of collapsed
                          with .sum(dim=(0, 1)), and top_k_scope/top_k_* are bypassed. This is
                          the only difference between classic and position-aware cf: identical
                          instrument, contrast objective and role semantics.
        posctx_values:    Optional {(layer, kind) -> [d_sae]} of each latent's posctx target
                          value. When given, the ACTIVATOR signal becomes grad x posctx_value —
                          the first-order effect of the intervention counterfactual
                          faithfulness performs (it injects each activator at its posctx value)
                          — instead of the bare gradient, which is a per-unit sensitivity the
                          eval never uses. Latents with no posctx value then score 0 rather
                          than ranking on a sensitivity they cannot cash in, which restores
                          k-sparsity to an otherwise dictionary-dense signal. Selected by
                          ``config.discovery.counterfactual_gradient.activator_signal``.
                          Must cover every scored site (see body). Inhibitors are unaffected:
                          acts x grad is already an effect, not a sensitivity.

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
    # Position-aware selection is deferred until every site's gradients are in
    # hand, so an "abs_pctl" spec can resolve ONE pooled admission threshold
    # across all sites and both role signals (the validated thresh64 protocol).
    # Entries hold references into `grads`/the graph — no extra tensor residency.
    pa_entries: List[Tuple[int, str, torch.Tensor, Optional[torch.Tensor],
                           Optional[torch.Tensor], Optional[torch.Tensor]]] = []

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

            # --- Activator signal: gradient, or gradient x posctx value ------
            # First-order IE of the intervention we ACTUALLY perform: the eval
            # injects each activator at its posctx value, so the effect is
            # grad x (posctx_value - negctx_value) ~= grad x posctx_value for an
            # absent activator. Ranking by the bare gradient ranks per-unit
            # SENSITIVITY instead — a quantity the eval never uses, and one that
            # is dense over the whole dictionary (no k-sparse structure to bound
            # the position-aware union).
            #
            # Every site reached here carries a gradient and is therefore
            # upstream of the seed, so posctx_values must cover it. A missing
            # entry would silently mix scaled and unscaled scores in one
            # ranking, so fail loudly rather than degrade.
            pv: Optional[torch.Tensor] = None
            if posctx_values is not None:
                if (layer_p, kind_p) not in posctx_values:
                    raise KeyError(
                        f"posctx_values is missing scored site ({layer_p}, {kind_p!r}); "
                        f"has {sorted(posctx_values)}. Mixing scaled and unscaled "
                        f"activator scores in one ranking is never correct."
                    )
                pv = posctx_values[(layer_p, kind_p)].to(grad_act.device, grad_act.dtype)

            # --- Position-aware branch --------------------------------------
            # The classic path collapses the position axis (.sum(dim=(0, 1)))
            # before ranking. When a PositionAwareSpec is supplied we keep that
            # axis instead: each position in the seed's causal prefix selects its
            # own top-N and the union is taken. Same gradients, same contrast
            # objective, same activator/inhibitor sign semantics — only the
            # reduction over positions changes.
            if position_aware is not None:
                pa_entries.append((layer_p, kind_p, grad_act, acts_grad.act, count_mask, pv))
                continue

            # --- Activator scores: NOT scaled by the negctx activation ------
            # Finds absent activators: high positive gradient even when acts ≈ 0.
            # `pv` (when set) is position-independent, so scaling after the
            # collapse is identical to scaling before it, and cheaper.
            scores_act = grad_act.sum(dim=(0, 1))  # [d_sae]
            if pv is not None:
                scores_act = scores_act * pv
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

    if position_aware is not None and pa_entries:
        def _act_attr(grad_act: torch.Tensor, pv: Optional[torch.Tensor]) -> torch.Tensor:
            return grad_act if pv is None else grad_act * pv.view(1, 1, -1)

        def _attr_stream():
            # Streamed so each attr tensor is sampled then freed — resolution
            # never holds more than one product at a time.
            for (_, _, grad_act, acts, _, pv) in pa_entries:
                yield _act_attr(grad_act, pv)
                if acts is not None:
                    yield acts * grad_act

        spec = position_aware.resolved_for(_attr_stream())
        for layer_p, kind_p, grad_act, acts, count_mask, pv in pa_entries:
            for latent, score in spec.select_from(_act_attr(grad_act, pv)).items():
                if count_mask is not None and not bool(count_mask[latent]):
                    continue
                if score > 0:  # absent activator: would push the seed up
                    all_activators.append(
                        (FeatureID(layer=layer_p, kind=kind_p, index=latent), float(score))
                    )
            if acts is not None:
                for latent, score in spec.select_from(acts * grad_act).items():
                    if count_mask is not None and not bool(count_mask[latent]):
                        continue
                    if score < 0:  # present inhibitor: active and suppressing
                        all_inhibitors.append(
                            (FeatureID(layer=layer_p, kind=kind_p, index=latent), float(score))
                        )

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
