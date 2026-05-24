"""Latent upstream attribution scoring."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from circuit.types.feature_id import FeatureID

from .attribution_types import UpstreamScores
from .sae_graph import FeatureGraph


def compute_latent_upstream_scores(
    graph: FeatureGraph,
    target_layer: int,
    target_kind: str,
    target_latent_idx: int,
    pos_argmax: torch.Tensor,
    predecessor_comp_indices: List[int],
    n_kinds: int,
    kinds: List[str],
    top_k: int,
    min_active_count: int,
    active_count: torch.Tensor,
    absent_inhibitor_top_k: int = 0,
    absent_inhibitor_threshold: float = 0.01,
) -> UpstreamScores:
    """
    Vectorised predecessor scoring for node discovery.

    Computes ∂(target_latent)/∂(predecessor_latents) for all latents in the
    predecessor components in a single backward pass, extracting two score types:

    - attribution (acts * grad): present latents driving or suppressing the target.
      Top-K selected globally by |score|.
    - absent_gradient (raw grad where acts ≈ 0): inactive latents that would suppress
      the target if they fired (strongly negative gradient).  Top-K selected by most-
      negative gradient.  Only computed when absent_inhibitor_top_k > 0.

    Both score types share the same backward pass — no extra cost.
    """
    from pipeline.component_index import split_component_idx, component_idx

    _empty = UpstreamScores()

    # 1. Connected acts for target
    try:
        _, target_acts_connected, _ = graph.get_latents(target_layer, target_kind)
    except (KeyError, IndexError):
        return _empty

    if target_acts_connected.act is None:
        return _empty

    B = target_acts_connected.act.shape[0]
    batch_idx = torch.arange(B, device=graph.device)

    # 2. Scalar target
    pos_argmax = pos_argmax.to(target_acts_connected.act.device)
    target_scalar = target_acts_connected.act[batch_idx, pos_argmax, target_latent_idx].sum()
    if target_scalar.grad_fn is None:
        return _empty

    # Move active_count to the graph device once so per-anchor slices are already on GPU
    active_count_gpu = active_count.to(graph.device) if active_count is not None else None

    # 3. Collect predecessor anchors
    anchors = []
    predecessor_pairs: List[Tuple[int, str]] = []
    for comp_idx_p in predecessor_comp_indices:
        layer_p, kind_idx_p = split_component_idx(comp_idx_p, n_kinds)
        kind_p = kinds[kind_idx_p]
        predecessor_pairs.append((layer_p, kind_p))

        if (layer_p, kind_p) in graph.activations:
            for acts_grad, _, _ in graph.activations[(layer_p, kind_p)]:
                if acts_grad.act is not None:
                    anchors.append(acts_grad.act)
                if acts_grad.res is not None:
                    anchors.append(acts_grad.res)

    if not anchors:
        return _empty

    # 4. Single backward pass — grads shared by both score types.
    # retain_graph=False: each chunk owns a fresh SAEGraphInstrument, so the graph
    # is never reused after this call.  Freeing it during backward reduces peak
    # memory and eliminates the cudaMalloc/cudaFree pressure from keeping it alive.
    grads = torch.autograd.grad(target_scalar, anchors, retain_graph=False, allow_unused=True)

    # 5. Score per component — attribution and absent_gradient extracted in one loop
    grad_iter = iter(grads)
    all_attribution: List[Tuple[FeatureID, float]] = []
    all_absent: List[Tuple[FeatureID, float]] = []
    compute_absent = absent_inhibitor_top_k > 0

    for layer_p, kind_p in predecessor_pairs:
        if (layer_p, kind_p) not in graph.activations:
            continue

        for acts_grad, _, _ in graph.activations[(layer_p, kind_p)]:
            grad_act = next(grad_iter) if acts_grad.act is not None else None
            grad_res = next(grad_iter) if acts_grad.res is not None else None  # noqa: F841 (consumed, unused for scoring)

            if grad_act is None:
                continue

            # Build active_count mask once per component (active_count_gpu already on device)
            count_mask: Optional[torch.Tensor] = None
            if active_count_gpu is not None:
                kind_idx_p = kinds.index(kind_p)
                c_idx_p = component_idx(layer_p, kind_idx_p, n_kinds)
                count_mask = active_count_gpu[c_idx_p] >= min_active_count

            # --- Attribution scores: acts * grad, summed over [B, T] ---
            attr_act = acts_grad.act * grad_act          # [B, T, d_sae]
            attr_scores = attr_act.sum(dim=(0, 1))       # [d_sae]
            if count_mask is not None:
                attr_scores = attr_scores * count_mask

            nonzero = (attr_scores != 0).nonzero(as_tuple=False).squeeze(1)
            if nonzero.numel() > 0:
                nz_scores = attr_scores[nonzero]
                for idx_int, score in zip(nonzero.cpu().tolist(), nz_scores.cpu().tolist()):
                    all_attribution.append((FeatureID(layer=layer_p, kind=kind_p, index=idx_int), score))

            # --- Absent-inhibitor scores: raw grad where acts ≈ 0 ---
            if compute_absent:
                absent_mask = acts_grad.act.abs() < 1e-6  # [B, T, d_sae]
                absent_scores = (grad_act * absent_mask.to(grad_act.dtype)).sum(dim=(0, 1))  # [d_sae]
                if count_mask is not None:
                    absent_scores = absent_scores * count_mask

                # Only keep latents with strongly negative gradient (would suppress if active)
                candidate_absent = (absent_scores < -absent_inhibitor_threshold).nonzero(as_tuple=False).squeeze(1)
                if candidate_absent.numel() > 0:
                    absent_vals = absent_scores[candidate_absent]
                    for idx_int, score in zip(candidate_absent.cpu().tolist(), absent_vals.cpu().tolist()):
                        all_absent.append((FeatureID(layer=layer_p, kind=kind_p, index=idx_int), score))

    # 6. Global top-K selection
    all_attribution.sort(key=lambda x: abs(x[1]), reverse=True)
    attribution_dict = {fid: score for fid, score in all_attribution[:top_k]}

    all_absent.sort(key=lambda x: x[1])  # most-negative first
    absent_dict = {fid: score for fid, score in all_absent[:absent_inhibitor_top_k]}

    return UpstreamScores(attribution=attribution_dict, absent_gradient=absent_dict)


__all__ = ["compute_latent_upstream_scores"]
