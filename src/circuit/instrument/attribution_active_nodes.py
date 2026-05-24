"""Active feature-node collection for attribution baselines."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from circuit.types.feature_id import FeatureID

from .sae_graph import FeatureGraph


def collect_active_feature_nodes(
    graph: FeatureGraph,
    kinds: List[str],
    n_kinds: int,
    min_active_count: int = 1,
    active_count: Optional[torch.Tensor] = None,
    max_feature_nodes: int = 0,
) -> List[FeatureID]:
    """
    Returns a deduplicated list of FeatureIDs for every latent that fired in at least
    one (layer, kind) of the forward pass, ranked by peak activation magnitude and
    optionally capped at max_feature_nodes.

    Scoring: each candidate feature is scored by its maximum activation value across
    all batch elements and token positions.  When max_feature_nodes > 0, only the
    top-N scoring features are returned.

    Note: when logit_top_k > 0, compute_direct_effects_matrix passes
    max_feature_nodes=0 here (unlimited) and instead applies the cap *after*
    logit backward passes, using logit-influence ranking (the more principled
    selection criterion from the original circuit tracer).  This function's
    max_feature_nodes cap is used only when logit_top_k == 0 (no logit targets).

    Ordering of returned list: (layer asc, kind_idx asc, latent_index asc) — stable
    across calls on the same graph so matrix row/column indices are deterministic.

    Args:
        graph:             FeatureGraph populated by SAEGraphInstrument.
        kinds:             Ordered list of kind strings (e.g. ["attn", "mlp", "resid"]).
        n_kinds:           Number of kinds (len(kinds)).
        min_active_count:  Latents whose global lifetime firing count is below this
                           threshold are excluded.  Ignored when active_count is None.
        active_count:      Optional [n_components, d_sae] CPU/GPU tensor of lifetime
                           firing counts (from latent_stats.active_count).
        max_feature_nodes: If > 0, keep only the top-N nodes by peak activation score.
                           0 means unlimited.  Primarily used as fallback when
                           logit_top_k == 0 (see note above).

    Returns:
        List[FeatureID] — deduplicated, sorted, one entry per active latent.
    """
    from pipeline.component_index import component_idx as _comp_idx

    # Collect (peak_activation, layer, kind, idx) tuples so we can rank and cap.
    candidates: List[Tuple[float, int, str, int]] = []
    seen: set = set()

    for (layer, kind), steps in graph.activations.items():
        kind_idx = kinds.index(kind)
        cidx = _comp_idx(layer, kind_idx, n_kinds)

        for _state_grad, state_connected, top_indices in steps:
            # state_connected.act is a dense [B, T, d_sae] tensor (zeros at inactive slots).
            # Peak activation for each latent = max over batch and sequence dimensions.
            if state_connected.act is not None:
                # Detach to avoid keeping the computation graph alive.
                peak_acts = state_connected.act.detach().abs().amax(dim=(0, 1))  # [d_sae]
                peak_acts_cpu = peak_acts.cpu()
            else:
                peak_acts_cpu = None

            unique_indices = top_indices.reshape(-1).unique().cpu().tolist()
            for raw_idx in unique_indices:
                idx = int(raw_idx)
                key = (layer, kind, idx)
                if key in seen:
                    continue
                if active_count is not None and min_active_count > 1:
                    count = (
                        active_count[cidx, idx].item()
                        if active_count.dim() == 2
                        else active_count[cidx * active_count.shape[-1] + idx].item()
                    )
                    if count < min_active_count:
                        continue
                seen.add(key)
                peak = float(peak_acts_cpu[idx].item()) if peak_acts_cpu is not None else 0.0
                candidates.append((peak, layer, kind, idx))

    # Rank by peak activation descending, cap if requested.
    candidates.sort(key=lambda c: -c[0])
    if max_feature_nodes > 0:
        candidates = candidates[:max_feature_nodes]

    # Re-sort by canonical (layer, kind_idx, index) order for stable matrix indices.
    nodes = [FeatureID(layer=layer, kind=kind, index=idx) for _, layer, kind, idx in candidates]
    nodes.sort(key=lambda f: (f.layer, kinds.index(f.kind), f.index))
    return nodes


__all__ = ["collect_active_feature_nodes"]
