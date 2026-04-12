import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from .sae_graph import FeatureGraph
from circuit.types.sparse_act import SparseAct
from circuit.types.feature_id import FeatureID


@dataclass
class UpstreamScores:
    """
    Scores produced by a single backward pass in compute_latent_upstream_scores.

    attribution:     {FeatureID: acts * grad} for latents that are active on the
                     current context.  Positive score = activator; negative = present
                     inhibitor.  Top-K selected by |score| across all predecessors.

    absent_gradient: {FeatureID: raw grad} for latents that are *inactive* (acts ≈ 0)
                     but have a strongly negative gradient — i.e. they would suppress the
                     target if they fired.  Top-K selected by most-negative gradient.
                     Empty when absent_inhibitor_top_k == 0.
    """
    attribution: Dict[FeatureID, float] = field(default_factory=dict)
    absent_gradient: Dict[FeatureID, float] = field(default_factory=dict)


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
    if target_acts_connected.act is None:
        return {}

    B, _T, _K = target_acts_connected.shape

    batch_indices = torch.arange(B, device=graph.device)
    vals_at_pos = target_indices[batch_indices, pos_argmax]  # [B, K]
    matches = (vals_at_pos == target_latent_idx)             # [B, K]

    if not matches.any():
        return {}

    # Backward target: connected acts at the target feature's peak positions
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
            attr_act = attr.act  # [B, T, d_sae] or None

            if attr_act is None:
                continue

            for latent_idx in latent_indices:
                score = attr_act[..., latent_idx].sum().item()
                if score != 0.0:
                    attributions[FeatureID(layer, kind, latent_idx)] = score
    else:
        for (layer, kind), grad in key_to_grad.items():
            acts_grad, _, indices = graph.get_latents(layer, kind, step=0)
            attr = acts_grad * grad
            attr_act = attr.act

            if attr_act is None:
                continue

            unique_indices = indices.unique()  # [K] GPU
            scores_vec = attr_act[:, :, unique_indices].sum(dim=(0, 1))  # [K]
            for l_idx, score in zip(unique_indices.cpu().tolist(), scores_vec.cpu().tolist()):
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

    if target_acts_connected.act is None:
        return {}

    B = target_acts_connected.act.shape[0]
    batch_indices = torch.arange(B, device=graph.device)
    
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

    grads = torch.autograd.grad(target_scalar, anchors, retain_graph=False, allow_unused=True)

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
        top_k_activators: Max absent activators to return (ranked by raw gradient desc).
        top_k_inhibitors: Max present inhibitors to return (ranked by |inhibitor score| desc).
        min_active_count: Latents below this global firing count are excluded.
        active_count:     [n_components, d_sae] tensor of lifetime firing counts.

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
                for idx_int, score in zip(pos_nz.cpu().tolist(), pos_vals.cpu().tolist()):
                    all_activators.append((FeatureID(layer=layer_p, kind=kind_p, index=idx_int), score))

            # --- Inhibitor scores: acts * grad where product < 0 ---
            # Finds present inhibitors: active latents causally suppressing the seed.
            scores_inh = (acts_grad.act * grad_act).sum(dim=(0, 1))  # [d_sae]
            if count_mask is not None:
                scores_inh = scores_inh * count_mask

            neg_nz = (scores_inh < 0).nonzero(as_tuple=False).squeeze(1)
            if neg_nz.numel() > 0:
                neg_vals = scores_inh[neg_nz]
                for idx_int, score in zip(neg_nz.cpu().tolist(), neg_vals.cpu().tolist()):
                    all_inhibitors.append((FeatureID(layer=layer_p, kind=kind_p, index=idx_int), score))

    # Global top-K: activators ranked by raw gradient descending (most positive first)
    all_activators.sort(key=lambda x: x[1], reverse=True)
    activator_dict = {fid: s for fid, s in all_activators[:top_k_activators]}

    # Global top-K: inhibitors ranked by score ascending (most negative first)
    all_inhibitors.sort(key=lambda x: x[1])
    inhibitor_dict = {fid: s for fid, s in all_inhibitors[:top_k_inhibitors]}

    return activator_dict, inhibitor_dict


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
