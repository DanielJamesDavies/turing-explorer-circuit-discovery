"""Logit-level attribution helpers."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from circuit.types.feature_id import FeatureID
from circuit.types.sparse_act import SparseAct

from .sae_graph import FeatureGraph


def compute_logit_attribution(
    graph: FeatureGraph,
    logits: torch.Tensor,
    pos_argmax: torch.Tensor,
    target_tokens: torch.Tensor,
) -> Dict[FeatureID, float]:
    """
    Pass 1 — Logit-based attribution.

    Runs a single backward pass from the target token logit at each sequence's
    peak activation position to all leaf anchors (top_acts_grad) in the graph.

    Cross-layer gradient flow is maintained by the identity passthrough term
    (x - x.detach()) in each SAEGraphInstrument hook.  This gives each leaf
    anchor the full downstream gradient (Jacobian = I w.r.t. x), rather than
    the lossy error-complement projection of the old approach.

    Args:
        graph:         FeatureGraph populated by SAEGraphInstrument.
        logits:        [B, T, vocab] — requires all_logits=True in Inference.forward.
        pos_argmax:    [B] — token position of peak seed activation per sequence.
        target_tokens: [B, T] — ground-truth next tokens (probe_data.target_tokens).

    Returns:
        Dict mapping FeatureID → attribution score (activation * gradient).
    """
    B = logits.shape[0]
    batch_idx = torch.arange(B, device=logits.device)

    # Target: logit of the ground-truth next token at the seed's peak position
    target_token_ids = target_tokens[batch_idx, pos_argmax.to(target_tokens.device)].to(logits.device)
    target_scalar = logits[batch_idx, pos_argmax, target_token_ids].sum()

    anchors = graph.all_anchors()
    if not anchors:
        return {}

    grads = torch.autograd.grad(target_scalar, anchors, retain_graph=True, allow_unused=True)

    # Build a flat map from anchor tensor id → (layer, kind, grad)
    anchor_info: List[Tuple[Tuple[int, str], SparseAct, SparseAct]] = []
    anchor_iter = iter(grads)
    for (layer, kind), steps in graph.activations.items():
        for acts_grad, _, _ in steps:
            grad_act = next(anchor_iter)
            grad_res = None
            if acts_grad.res is not None:
                grad_res = next(anchor_iter)

            if grad_act is not None or grad_res is not None:
                grad = SparseAct(act=grad_act, res=grad_res)
                anchor_info.append(((layer, kind), acts_grad, grad))

    attributions: Dict[FeatureID, float] = {}
    for (layer, kind), acts_grad, grad in anchor_info:
        _, _, indices = graph.get_latents(layer, kind, step=0)

        # Attribution score = activation * gradient
        # acts_grad and grad are SparseAct objects with dense [B, T, d_sae] act tensors
        attr = acts_grad * grad  # SparseAct
        attr_act = attr.act      # [B, T, d_sae] or None if grad had no act component

        if attr_act is None:
            continue

        unique_idx = indices.unique()  # [K] GPU
        scores_vec = attr_act[:, :, unique_idx].sum(dim=(0, 1))  # [K] — one GPU op
        for l_idx, score in zip(unique_idx.cpu().tolist(), scores_vec.cpu().tolist()):
            if score != 0.0:
                fid = FeatureID(layer=layer, kind=kind, index=l_idx)
                attributions[fid] = attributions.get(fid, 0.0) + score

    return attributions


__all__ = ["compute_logit_attribution"]
