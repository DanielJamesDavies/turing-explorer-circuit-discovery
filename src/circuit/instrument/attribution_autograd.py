"""Autograd anchor collection and edge scoring helpers for attribution."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from circuit.types.feature_id import FeatureID

from .sae_graph import FeatureGraph


def _upstream_anchors_for_target(
    graph: FeatureGraph,
    tgt_layer: int,
    tgt_kind: str,
    kinds: List[str],
    n_kinds: int,
) -> Tuple[List[torch.Tensor], List[Tuple[int, str, bool]]]:
    """
    Collects leaf anchor tensors for all (layer, kind) pairs causally upstream of
    (tgt_layer, tgt_kind), respecting the per-layer ordering defined by
    get_all_upstream_components (attn → mlp → resid within a layer, plus all
    strictly preceding layers).

    Returns:
        anchors:     Tensors to pass to autograd.grad — act then res per component.
        anchor_meta: Parallel (layer, kind, is_res) list for each tensor.
    """
    from pipeline.component_index import get_all_upstream_components, split_component_idx
    from pipeline.component_index import component_idx as _comp_idx_fn

    tgt_comp = _comp_idx_fn(tgt_layer, kinds.index(tgt_kind), n_kinds)
    upstream_comps = get_all_upstream_components(tgt_comp, n_kinds, kinds, include_same_layer=True)
    upstream_set: set = set()
    for c in upstream_comps:
        l_p, ki_p = split_component_idx(c, n_kinds)
        upstream_set.add((l_p, kinds[ki_p]))

    anchors: List[torch.Tensor] = []
    anchor_meta: List[Tuple[int, str, bool]] = []
    for (l_p, k_p), steps in graph.activations.items():
        if (l_p, k_p) not in upstream_set:
            continue
        for acts_grad, _, _ in steps:
            if acts_grad.act is not None:
                anchors.append(acts_grad.act)
                anchor_meta.append((l_p, k_p, False))
            if acts_grad.res is not None:
                anchors.append(acts_grad.res)
                anchor_meta.append((l_p, k_p, True))
    return anchors, anchor_meta


def _all_feature_anchors_with_meta(
    graph: FeatureGraph,
) -> Tuple[List[torch.Tensor], List[Tuple[int, str, bool]]]:
    """
    Collects ALL leaf anchor tensors from the graph.

    Used for logit target backward passes, where every upstream feature can
    influence the final output logits (no causal masking needed).

    Returns:
        anchors:     All leaf tensors for autograd.grad.
        anchor_meta: Parallel (layer, kind, is_res) list.
    """
    anchors: List[torch.Tensor] = []
    anchor_meta: List[Tuple[int, str, bool]] = []
    for (l_p, k_p), steps in graph.activations.items():
        for acts_grad, _, _ in steps:
            if acts_grad.act is not None:
                anchors.append(acts_grad.act)
                anchor_meta.append((l_p, k_p, False))
            if acts_grad.res is not None:
                anchors.append(acts_grad.res)
                anchor_meta.append((l_p, k_p, True))
    return anchors, anchor_meta


def _score_grads_into_adj(
    grads: Tuple[Optional[torch.Tensor], ...],
    anchor_meta: List[Tuple[int, str, bool]],
    graph: FeatureGraph,
    feature_nodes: List[FeatureID],
    node_to_idx: Dict[FeatureID, int],
    tgt_idx: int,
    adj: Dict[Tuple[int, int], float],
) -> None:
    """
    Extracts direct-effect scores for each source node and accumulates them
    into adj[(src_idx, tgt_idx)].

    Two scoring paths:
        Feature nodes (is_res=False):
            score = sum_{B,T}( f_grad.act[..., src.index] * grad_act[..., src.index] )
        Error nodes (is_res=True):
            score = sum_{B,T,d_model}( res_anchor * grad_res )
            The sentinel FeatureID(layer=l, kind=k+"_err", index=0) must already be
            present in node_to_idx; silently skipped if absent (error nodes disabled).
    """
    lk_to_grad: Dict[Tuple[int, str], torch.Tensor] = {}
    for (l_p, k_p, is_res), g in zip(anchor_meta, grads):
        if g is None:
            continue
        if is_res:
            # Score the reconstruction-error node for (l_p, k_p) → tgt_idx.
            # Silently skip if error sentinels are not registered in node_to_idx
            # (include_error_nodes=False or node list not yet established).
            error_fid = FeatureID(layer=l_p, kind=k_p + "_err", index=0)
            error_src_idx = node_to_idx.get(error_fid)
            if error_src_idx is None or error_src_idx == tgt_idx:
                continue
            try:
                acts_grad, _, _ = graph.get_latents(l_p, k_p)
            except (KeyError, IndexError):
                continue
            if acts_grad.res is None:
                continue
            score = float((acts_grad.res * g).sum().item())
            if score == 0.0:
                continue
            edge_key = (error_src_idx, tgt_idx)
            adj[edge_key] = adj.get(edge_key, 0.0) + score
        else:
            lk_to_grad[(l_p, k_p)] = g  # [B, T, d_sae]

    for src_node in feature_nodes:
        g = lk_to_grad.get((src_node.layer, src_node.kind))
        if g is None:
            continue
        try:
            acts_grad, _, _ = graph.get_latents(src_node.layer, src_node.kind)
        except (KeyError, IndexError):
            continue
        if acts_grad.act is None:
            continue

        score = float(
            (acts_grad.act[..., src_node.index] * g[..., src_node.index]).sum().item()
        )
        if score == 0.0:
            continue

        src_idx = node_to_idx.get(src_node)
        if src_idx is None or src_idx == tgt_idx:
            continue

        edge_key = (src_idx, tgt_idx)
        adj[edge_key] = adj.get(edge_key, 0.0) + score


def _score_token_grads_into_adj(
    grad_emb: torch.Tensor,
    emb_anchor: torch.Tensor,
    token_nodes: List[FeatureID],
    node_to_idx: Dict[FeatureID, int],
    tgt_idx: int,
    adj: Dict[Tuple[int, int], float],
) -> None:
    """
    Scores each input-token position's direct contribution to `tgt_idx` and
    accumulates results into `adj`.

    Attribution score for position p (first-order Taylor):
        score[p] = Σ_{b,d}  emb_anchor[b,p,d] * grad_emb[b,p,d]

    This matches the token-attribution formula used in the original circuit
    tracer for `embed_nodes`.  Summing over the batch dimension gives the
    mean-over-probe-sequences contribution.

    Args:
        grad_emb:    Gradient tensor of shape [B, T, d_model] — the result of
                     torch.autograd.grad(scalar, [emb_anchor])[0].
        emb_anchor:  Detached leaf embedding tensor [B, T, d_model] captured by
                     SAEGraphInstrumentWithEmbedding.
        token_nodes: List of FeatureID(layer=-2, kind="token", index=p) for each
                     position p in range(T).
        node_to_idx: Mapping from FeatureID to its column/row index in adj.
        tgt_idx:     Row index of the target node in adj.
        adj:         Sparse adjacency dict updated in-place.
    """
    # [T]: sum the activation×gradient product over batch and model-dim
    scores = (emb_anchor * grad_emb).sum(dim=(0, 2))  # [T]
    for p, tok_fid in enumerate(token_nodes):
        score = float(scores[p].item())
        if score == 0.0:
            continue
        src_idx = node_to_idx.get(tok_fid)
        if src_idx is None or src_idx == tgt_idx:
            continue
        adj[(src_idx, tgt_idx)] = adj.get((src_idx, tgt_idx), 0.0) + score


__all__ = [
    "_all_feature_anchors_with_meta",
    "_score_grads_into_adj",
    "_score_token_grads_into_adj",
    "_upstream_anchors_for_target",
]
