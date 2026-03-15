import torch
from typing import Dict, Tuple, List, Optional
from .sae_graph import FeatureGraph


def compute_logit_attribution(
    graph: FeatureGraph,
    logits: torch.Tensor,
    pos_argmax: torch.Tensor,
    target_tokens: torch.Tensor,
) -> Dict[Tuple[int, str, int], float]:
    """
    Pass 1 — Logit-based attribution.

    Runs a single backward pass from the target token logit at each sequence's
    peak activation position to all leaf anchors (top_acts_grad) in the graph.

    The gradient path exists because logits depend on the final residual stream,
    which is connected back through each layer's error term to earlier anchors:
        logits → x_final → ... → x_L (via error terms) → x_L-1 → top_acts_grad_L-1

    Args:
        graph:         FeatureGraph populated by SAEGraphInstrument.
        logits:        [B, T, vocab] — requires all_logits=True in Inference.forward.
        pos_argmax:    [B] — token position of peak seed activation per sequence.
        target_tokens: [B, T] — ground-truth next tokens (probe_data.target_tokens).

    Returns:
        Dict mapping (layer, kind, latent_idx) → attribution score (activation * gradient).
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
    anchor_info: List[Tuple[Tuple[int, str], torch.Tensor, torch.Tensor]] = []
    anchor_iter = iter(grads)
    for (layer, kind), steps in graph.activations.items():
        for acts_grad, _, _ in steps:
            grad = next(anchor_iter)
            if grad is not None:
                anchor_info.append(((layer, kind), acts_grad, grad))

    attributions: Dict[Tuple[int, str, int], float] = {}
    for (layer, kind), acts_grad, grad in anchor_info:
        _, _, indices = graph.get_latents(layer, kind, step=0)
        attr_tensor = acts_grad.data * grad.data  # [B, T, K]

        unique_idx = indices.unique()
        for idx in unique_idx:
            l_idx = int(idx.item())
            mask = (indices == idx)
            score = attr_tensor[mask].sum().item()
            if score != 0.0:
                key = (layer, kind, l_idx)
                attributions[key] = attributions.get(key, 0.0) + score

    return attributions


def compute_feature_attribution(
    graph: FeatureGraph,
    target_layer: int,
    target_kind: str,
    target_latent_idx: int,
    pos_argmax: torch.Tensor,
    candidate_nodes: Optional[List[Tuple[int, str, int]]] = None,
) -> Dict[Tuple[int, str, int], float]:
    """
    Pass 2 — Feature-to-feature attribution.

    Uses top_acts_connected (the original encoder output, still in the computation
    graph) as the backward target rather than the detached leaf anchor.  This means
    gradients can flow through the error-term path to earlier layers' leaf anchors:

        top_acts_connected_B → encode(x_B_in) → x_B_in
            = reconstruction_A + error_A
            → reconstruction_A → decode(scatter(top_acts_grad_A)) → top_acts_grad_A ✓

    Args:
        graph:             FeatureGraph from SAEGraphInstrument.
        target_layer/kind/latent_idx: The downstream feature (node B).
        pos_argmax:        [B] peak positions for the probe sequences.
        candidate_nodes:   Upstream (layer, kind, latent_idx) tuples to evaluate.
                           If None, all upstream anchors are scored.

    Returns:
        Dict mapping (layer, kind, latent_idx) → attribution score.
    """
    _, target_acts_connected, target_indices = graph.get_latents(target_layer, target_kind, step=0)
    B, T, K = target_acts_connected.shape

    batch_indices = torch.arange(B, device=graph.device)
    vals_at_pos = target_indices[batch_indices, pos_argmax]  # [B, K]
    matches = (vals_at_pos == target_latent_idx)             # [B, K]

    if not matches.any():
        return {}

    # Backward target: connected acts at the target feature's peak positions
    acts_at_pos = target_acts_connected[batch_indices, pos_argmax]  # [B, K]
    target_sum = acts_at_pos[matches].sum()

    # Guard: if the encoder output has no grad_fn, the custom kernel path was taken
    # and gradients cannot flow.  Return empty rather than crashing.
    if target_sum.grad_fn is None:
        return {}

    # Collect upstream leaf anchors only
    anchors: List[torch.Tensor] = []
    anchor_keys: List[Tuple[int, str]] = []
    for (layer, kind), steps_data in graph.activations.items():
        if layer > target_layer:
            continue
        for acts_grad, _, _ in steps_data:
            anchors.append(acts_grad)
            anchor_keys.append((layer, kind))

    if not anchors:
        return {}

    grads = torch.autograd.grad(target_sum, anchors, retain_graph=True, allow_unused=True)

    key_to_grad: Dict[Tuple[int, str], torch.Tensor] = {}
    for key, grad in zip(anchor_keys, grads):
        if grad is not None:
            key_to_grad[key] = grad

    attributions: Dict[Tuple[int, str, int], float] = {}

    if candidate_nodes is not None:
        by_layer_kind: Dict[Tuple[int, str], List[int]] = {}
        for l, k, i in candidate_nodes:
            by_layer_kind.setdefault((l, k), []).append(i)

        for (layer, kind), latent_indices in by_layer_kind.items():
            if (layer, kind) not in key_to_grad:
                continue
            acts_grad, _, indices = graph.get_latents(layer, kind, step=0)
            grad = key_to_grad[(layer, kind)]
            attr_tensor = acts_grad.data * grad.data  # [B, T, K]

            for latent_idx in latent_indices:
                mask = (indices == latent_idx)
                score = attr_tensor[mask].sum().item()
                if score != 0.0:
                    attributions[(layer, kind, latent_idx)] = score
    else:
        for (layer, kind), grad in key_to_grad.items():
            acts_grad, _, indices = graph.get_latents(layer, kind, step=0)
            attr_tensor = acts_grad.data * grad.data

            unique_indices = indices.unique()
            for idx in unique_indices:
                l_idx = int(idx.item())
                score = attr_tensor[indices == idx].sum().item()
                if score != 0.0:
                    key = (layer, kind, l_idx)
                    attributions[key] = attributions.get(key, 0.0) + score

    return attributions


def compute_attribution(
    graph: FeatureGraph,
    target_layer: int,
    target_kind: str,
    target_latent_idx: int,
    pos_argmax: torch.Tensor,
    candidate_nodes: Optional[List[Tuple[int, str, int]]] = None,
) -> Dict[Tuple[int, str, int], float]:
    """
    Legacy attribution — backward from the detached leaf anchor of the target feature.

    NOTE: This only produces non-zero gradients for same-layer candidates. Cross-layer
    attribution is broken because top_acts_grad is a detached leaf with no path to
    earlier layers. Use compute_logit_attribution + compute_feature_attribution instead.

    Kept for backward compatibility with TopCoactivationDiscovery.
    """
    target_acts_grad, _, target_indices = graph.get_latents(target_layer, target_kind, step=0)
    B, T, K = target_acts_grad.shape

    batch_indices = torch.arange(B, device=graph.device)
    vals_at_pos = target_indices[batch_indices, pos_argmax]
    matches = (vals_at_pos == target_latent_idx)

    if not matches.any():
        return {}

    acts_at_pos = target_acts_grad[batch_indices, pos_argmax]
    target_sum = acts_at_pos[matches].sum()

    anchors: List[torch.Tensor] = []
    anchor_keys: List[Tuple[int, str]] = []
    for (layer, kind), steps_data in graph.activations.items():
        if layer > target_layer:
            continue
        for acts_grad, _, _ in steps_data:
            anchors.append(acts_grad)
            anchor_keys.append((layer, kind))

    grads = torch.autograd.grad(target_sum, anchors, retain_graph=True, allow_unused=True)

    key_to_grad: Dict[Tuple[int, str], torch.Tensor] = {}
    for key, grad in zip(anchor_keys, grads):
        if grad is not None:
            key_to_grad[key] = grad

    attributions: Dict[Tuple[int, str, int], float] = {}

    if candidate_nodes is not None:
        by_layer_kind: Dict[Tuple[int, str], List[int]] = {}
        for l, k, i in candidate_nodes:
            by_layer_kind.setdefault((l, k), []).append(i)

        for (layer, kind), latent_indices in by_layer_kind.items():
            if (layer, kind) not in key_to_grad:
                continue
            acts_grad, _, indices = graph.get_latents(layer, kind, step=0)
            grad = key_to_grad[(layer, kind)]
            attr_tensor = acts_grad.data * grad.data

            for latent_idx in latent_indices:
                mask = (indices == latent_idx)
                score = attr_tensor[mask].sum().item()
                if score != 0.0:
                    attributions[(layer, kind, latent_idx)] = score
    else:
        for (layer, kind), grad in key_to_grad.items():
            acts_grad, _, indices = graph.get_latents(layer, kind, step=0)
            attr_tensor = acts_grad.data * grad.data

            unique_indices = indices.unique()
            for idx in unique_indices:
                l_idx = int(idx.item())
                score = attr_tensor[indices == idx].sum().item()
                if score != 0.0:
                    key = (layer, kind, l_idx)
                    attributions[key] = attributions.get(key, 0.0) + score

    return attributions
