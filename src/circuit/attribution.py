import torch
from typing import Dict, Tuple, List, Optional
from .sae_graph import FeatureGraph
from .sparse_act import SparseAct
from .feature_id import FeatureID


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
        attr_act = attr.act      # [B, T, d_sae]

        unique_idx = indices.unique()
        for idx in unique_idx:
            l_idx = int(idx.item())
            # Sum over batch and sequence for this specific latent
            score = attr_act[..., l_idx].sum().item()
            if score != 0.0:
                fid = FeatureID(layer=layer, kind=kind, index=l_idx)
                attributions[fid] = attributions.get(fid, 0.0) + score

    return attributions


def compute_feature_attribution(
    graph: FeatureGraph,
    target_layer: int,
    target_kind: str,
    target_latent_idx: int,
    pos_argmax: torch.Tensor,
    candidate_nodes: Optional[List[FeatureID]] = None,
) -> Dict[FeatureID, float]:
    """
    Pass 2 — Feature-to-feature attribution.

    Uses top_acts_connected (the original encoder output, still in the computation
    graph) as the backward target rather than the detached leaf anchor.  Gradients
    flow cross-layer via the identity passthrough (x - x.detach()) at each hook:

        top_acts_connected_B → encode(x_B) → x_B
            → (x_A - x_A.detach()) → x_A (identity Jacobian)
            → leaf anchors at layer A (f_grad_A, res_anchor_A)

    Args:
        graph:             FeatureGraph from SAEGraphInstrument.
        target_layer/kind/latent_idx: The downstream feature (node B).
        pos_argmax:        [B] peak positions for the probe sequences.
        candidate_nodes:   Upstream FeatureID objects to evaluate.
                           If None, all upstream anchors are scored.

    Returns:
        Dict mapping FeatureID → attribution score.
    """
    _, target_acts_connected, target_indices = graph.get_latents(target_layer, target_kind, step=0)
    B, T, K = target_acts_connected.shape

    batch_indices = torch.arange(B, device=graph.device)
    vals_at_pos = target_indices[batch_indices, pos_argmax]  # [B, K]
    matches = (vals_at_pos == target_latent_idx)             # [B, K]

    if not matches.any():
        return {}

    # Backward target: connected acts at the target feature's peak positions
    # We index the dense [B, T, d_sae] tensor directly by the target_latent_idx
    target_sum = target_acts_connected.act[batch_indices, pos_argmax, target_latent_idx].sum()

    # Guard: if the encoder output has no grad_fn, the custom kernel path was taken
    # and gradients cannot flow.  Return empty rather than crashing.
    if target_sum.grad_fn is None:
        return {}

    # Collect upstream leaf anchors only
    anchors: List[torch.Tensor] = []
    for (layer, kind), steps_data in graph.activations.items():
        if layer > target_layer:
            continue
        for acts_grad, _, _ in steps_data:
            if acts_grad.act is not None:
                anchors.append(acts_grad.act)
            if acts_grad.res is not None:
                anchors.append(acts_grad.res)

    if not anchors:
        return {}

    grads = torch.autograd.grad(target_sum, anchors, retain_graph=True, allow_unused=True)

    # Better way: Re-collect grads into key_to_grad
    key_to_grad: Dict[Tuple[int, str], SparseAct] = {}
    anchor_iter = iter(grads)
    for (layer, kind), steps_data in graph.activations.items():
        if layer > target_layer:
            continue
        for acts_grad, _, _ in steps_data:
            grad_act = next(anchor_iter) if acts_grad.act is not None else None
            grad_res = next(anchor_iter) if acts_grad.res is not None else None
            
            if grad_act is not None or grad_res is not None:
                key_to_grad[(layer, kind)] = SparseAct(act=grad_act, res=grad_res)

    attributions: Dict[FeatureID, float] = {}

    if candidate_nodes is not None:
        by_layer_kind: Dict[Tuple[int, str], List[int]] = {}
        for fid in candidate_nodes:
            by_layer_kind.setdefault((fid.layer, fid.kind), []).append(fid.index)

        for (layer, kind), latent_indices in by_layer_kind.items():
            if (layer, kind) not in key_to_grad:
                continue
            acts_grad, _, _ = graph.get_latents(layer, kind, step=0)
            grad = key_to_grad[(layer, kind)]
            attr = acts_grad * grad
            attr_act = attr.act  # [B, T, d_sae]

            for latent_idx in latent_indices:
                score = attr_act[..., latent_idx].sum().item()
                if score != 0.0:
                    attributions[FeatureID(layer, kind, latent_idx)] = score
    else:
        for (layer, kind), grad in key_to_grad.items():
            acts_grad, _, indices = graph.get_latents(layer, kind, step=0)
            attr = acts_grad * grad
            attr_act = attr.act

            unique_indices = indices.unique()
            for idx in unique_indices:
                l_idx = int(idx.item())
                score = attr_act[..., l_idx].sum().item()
                if score != 0.0:
                    fid = FeatureID(layer, kind, l_idx)
                    attributions[fid] = attributions.get(fid, 0.0) + score

    return attributions


def compute_feature_gradient(
    graph: FeatureGraph,
    target_layer: int,
    target_kind: str,
    target_latent_idx: int,
    pos_argmax: torch.Tensor,
    candidate_nodes: List[FeatureID],
) -> Dict[FeatureID, float]:
    """
    Returns the raw gradient d(TargetAct)/d(CandidateAct) rather than Act * Grad.
    This allows identifying inhibitors that are not active in the current context
    but would have a strong negative effect if they were.
    """
    try:
        _, target_acts_connected, _ = graph.get_latents(target_layer, target_kind, step=0)
    except (KeyError, IndexError):
        return {}
        
    B, T, K = target_acts_connected.shape
    batch_indices = torch.arange(B, device=graph.device)
    
    # Backward target: connected acts at the target feature's peak positions
    # Use max() to ensure we pick the peak even if pos_argmax is slightly off
    target_scalar = target_acts_connected.act[batch_indices, pos_argmax, target_latent_idx].sum()

    if target_scalar.grad_fn is None:
        return {}

    # Collect relevant upstream leaf anchors
    anchors = []
    anchor_meta = [] # (layer, kind, is_res)
    
    for (layer, kind), steps_data in graph.activations.items():
        if layer > target_layer:
            continue
        for acts_grad, _, _ in steps_data:
            if acts_grad.act is not None:
                anchor_meta.append((layer, kind, False))
                anchors.append(acts_grad.act)
            if acts_grad.res is not None:
                anchor_meta.append((layer, kind, True))
                anchors.append(acts_grad.res)

    if not anchors:
        return {}

    grads = torch.autograd.grad(target_scalar, anchors, retain_graph=True, allow_unused=True)
    
    # Map back to FeatureID
    layer_kind_to_grad = {}
    for (layer, kind, is_res), g in zip(anchor_meta, grads):
        if not is_res and g is not None:
            layer_kind_to_grad[(layer, kind)] = g

    gradients = {}
    for fid in candidate_nodes:
        g = layer_kind_to_grad.get((fid.layer, fid.kind))
        if g is not None:
            # Sum gradient across batch and time for this latent
            val = g[..., fid.index].sum().item()
            if val != 0.0:
                gradients[fid] = val
                
    return gradients
