import gc
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional
from .sae_graph import FeatureGraph, SAEGraphInstrument, SAEGraphInstrumentWithEmbedding
from .ct_influence import _compute_partial_neumann_influence
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


def _find_logit_targets(
    inference: Any,
    tokens: torch.Tensor,
    logit_top_k: int,
    desired_logit_prob: float = 0.95,
) -> Tuple[List[int], torch.Tensor]:
    """
    Selects logit target tokens by cumulative softmax probability, matching
    Anthropic's two-stage logit-target selection from attribute_transformerlens.py.

    Two-stage selection:
        Stage 1 — Pool cap: restrict candidates to the top-`logit_top_k` tokens by
                  softmax probability using torch.topk (O(vocab) without a full sort).
                  This guarantees that no low-rank token is ever selected, even if the
                  cumulative threshold hasn't been reached within the pool.
        Stage 2 — Cumulative cutoff: within the pool, keep the shortest prefix whose
                  cumulative probability >= desired_logit_prob.

    The returned probabilities are renormalised to sum to 1.0 over the selected set
    so they can be used directly as logit root weights in the Neumann series.

    Returns:
        token_ids:  Vocabulary indices of the selected targets (descending prob order).
        probs:      Renormalised float32 tensor [K] used for influence root weighting.
    """
    _, logits, _ = inference.forward(
        tokens,
        patcher=None,
        grad_enabled=False,
        return_activations=False,
        tokenize_final=False,
        all_logits=True,
    )
    if logits is None:
        return [], torch.zeros(0, dtype=torch.float32)

    last_logits = logits[:, -1, :].float() if logits.dim() == 3 else logits.float()
    mean_logits = last_logits.mean(dim=0)

    probs = torch.softmax(mean_logits, dim=0)

    # Stage 1: restrict to the top-logit_top_k pool (matches topk(max_n_logits) in original)
    pool_k = min(logit_top_k, probs.shape[0])
    top_p, top_idx = probs.topk(pool_k)

    # Stage 2: cumulative-prob cutoff within the pool
    cumsum = top_p.cumsum(dim=0)
    n_needed = int((cumsum < desired_logit_prob).sum().item()) + 1
    n = max(1, min(n_needed, pool_k))

    selected_idx = top_idx[:n]
    selected_probs = top_p[:n]

    # Renormalise so weights sum to 1.0 for Neumann series seeding
    prob_sum = selected_probs.sum()
    if prob_sum > 0:
        selected_probs = selected_probs / prob_sum

    return selected_idx.cpu().tolist(), selected_probs.cpu()


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


def _compute_partial_one_hop_influence(
    partial_adj: Dict[Tuple[int, int], float],
    n_feature_nodes: int,
    n_error_nodes: int,
    n_logit_nodes: int,
    logit_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a sparse one-hop logit-influence score for each feature node.

    Formula (row-normalised):
        influence[i] = Σ_j  lw[j] * |adj[(i, j)]| / Σ_k |adj[(k, j)]|

    where j ranges over logit-target node indices and lw[j] is the softmax
    probability weight for logit j.

    Node index layout expected in partial_adj:
        0 .. n_feature_nodes-1             — feature nodes
        n_feature_nodes .. +n_error_nodes  — error nodes
        n_feature_nodes+n_error_nodes ..   — logit sentinel nodes (n_logit_nodes of them)

    Args:
        partial_adj:     Sparse dict (src_idx, tgt_idx) → score.  Can be the full
                         adj or a ranking-only subset (partial_adj_ranking from Pass A).
        n_feature_nodes: Number of feature nodes at the head of the index space.
        n_error_nodes:   Number of error sentinel nodes (follows features).
        n_logit_nodes:   Number of logit sentinel nodes (follows error nodes).
        logit_probs:     Float32 tensor [n_logit_nodes] of softmax probabilities.
                         Falls back to uniform 1/K if shape doesn't match.

    Returns:
        Float32 CPU tensor of shape [n_feature_nodes].  Higher = more directly
        influential on the logit targets.
    """
    logit_start = n_feature_nodes + n_error_nodes

    # Logit root weights (same logic as compute_ct_influence)
    lw_dict: Dict[int, float] = {}
    if n_logit_nodes > 0:
        if logit_probs.shape[0] == n_logit_nodes:
            for j in range(n_logit_nodes):
                lw_dict[logit_start + j] = float(logit_probs[j])
        else:
            for j in range(n_logit_nodes):
                lw_dict[logit_start + j] = 1.0 / n_logit_nodes

    # L1 norm of each logit target's incoming column (for row-normalisation)
    logit_row_abs_sum: Dict[int, float] = {}
    for (s, t), v in partial_adj.items():
        if t in lw_dict:
            logit_row_abs_sum[t] = logit_row_abs_sum.get(t, 0.0) + abs(v)

    # Accumulate one-hop influence per feature node
    feature_inf_dict: Dict[int, float] = {}
    for (s, t), v in partial_adj.items():
        lw_t = lw_dict.get(t, 0.0)
        if lw_t == 0.0:
            continue
        row_sum = logit_row_abs_sum.get(t, 0.0)
        if row_sum < 1e-12:
            continue
        if 0 <= s < n_feature_nodes:
            feature_inf_dict[s] = (
                feature_inf_dict.get(s, 0.0) + lw_t * abs(v) / row_sum
            )

    # Dense result tensor (at most n_feature_nodes ≤ max_feature_nodes entries)
    result = torch.zeros(n_feature_nodes, dtype=torch.float32)
    for feat_idx, score in feature_inf_dict.items():
        result[feat_idx] = score
    return result


def compute_direct_effects_matrix(
    tokens: torch.Tensor,
    inference: Any,
    bank: Any,
    logit_top_k: int,
    probe_batch_size: int,
    kinds: List[str],
    n_kinds: int,
    min_active_count: int = 1,
    active_count: Optional[torch.Tensor] = None,
    max_feature_nodes: int = 0,
    stop_error_grad: bool = False,
    target_chunk_size: int = 8,  # deprecated — kept for call-site compatibility only
    desired_logit_prob: float = 0.95,
    include_error_nodes: bool = True,
    online_ranking_interval: int = 4,
    feature_batch_size: int = 32,
    include_token_nodes: bool = False,
) -> Tuple[Dict[Tuple[int, int], float], List[FeatureID], torch.Tensor]:
    """
    Builds a sparse prompt-local direct-effects adjacency matrix over active SAE
    latents (attn/mlp/resid across all layers), plus logit sentinel nodes.

    Matches the spirit of Anthropic's Attribution Graphs method: one no-grad
    discovery pass to collect and rank active nodes, then ONE retained-graph
    grad-enabled forward pass per sequence.  All target backward passes run
    against that single graph — eliminating the O(N/chunk) forward-pass overhead
    of the old chunked implementation.

    adj[(src_idx, tgt_idx)] =
        mean over sequences of sum_{B,T}( f_grad_src[..., src.index]
                                          * grad_src[..., src.index] )

    Node layout in all_nodes:
        [0 : N_feat]                        — feature nodes (FeatureID with layer >= 0)
        [N_feat : N_feat+E]                 — error nodes (kind=k+"_err", one per active
                                              (layer,kind)); omitted when include_error_nodes=False.
        [N_feat+E : N_feat+E+T]             — token sentinel nodes (layer=-2, kind="token",
                                              index=position); omitted when include_token_nodes=False.
        [N_feat+E+T :]                      — logit sentinel nodes (layer=-1, kind="logit").
        Logit nodes are always last so logit_start = N - logit_top_k is stable.

    Memory strategy for 16 GB VRAM:
        • probe_batch_size=1 keeps each retained graph small (≈ 1× forward pass).
        • max_feature_nodes caps how many target backward passes are run.
        • torch.cuda.empty_cache() + gc.collect() once per sequence after freeing
          the retained graph.

    The caller is responsible for:
        • calling inference.disable_compile() before and enable_compile() after.
        • using bank.pin_decoders() context to keep decoder weights resident.

    Args:
        tokens:            [N, T] probe sequences.
        inference:         Inference instance (compile already disabled).
        bank:              SAEBank (decoders already pinned).
        logit_top_k:       Number of logit sentinel nodes (0 = no logit targets).
        probe_batch_size:  Sequences per forward/backward pass.
        kinds:             Ordered SAE kind names, e.g. ["attn", "mlp", "resid"].
        n_kinds:           len(kinds).
        min_active_count:  Exclude latents with global firing count below this.
        active_count:      Optional [n_components, d_sae] lifetime-count tensor.
        max_feature_nodes:    Cap on feature nodes after logit-influence ranking. 0 = unlimited.
                              When logit_top_k > 0, all active features are first collected
                              (no early cap), logit backward passes run against the full set,
                              then the top-N by one-hop logit influence are kept.
                              When logit_top_k == 0, falls back to peak-activation ranking.
        stop_error_grad:      If True, error-term gradients are zeroed in SAEGraphInstrument.
        target_chunk_size:    Deprecated. No longer used. Retained for call-site compat.
        desired_logit_prob:   Cumulative softmax probability used to select logit targets.
                              Tokens are added in descending probability order until the
                              running sum >= desired_logit_prob (capped at logit_top_k).
        include_error_nodes:  If True (default), add one error sentinel node per active
                              (layer, kind) pair representing the SAE reconstruction error.
                              Error nodes appear as sources in adj (they influence downstream
                              features/logits) but are never backward-pass targets.
                              Set False to omit them entirely (equivalent to stop_error_grad
                              but also removes sentinels from all_nodes).
        include_token_nodes:  If True, add T token sentinel nodes (one per input position)
                              to all_nodes.  Each backward pass also differentiates w.r.t.
                              the input embedding captured by SAEGraphInstrumentWithEmbedding,
                              giving per-position token→feature attribution.  Default False.
        online_ranking_interval: Re-rank remaining unvisited features every N cycles
                              of Step 7's backward-pass loop.  Each cycle processes
                              `feature_batch_size` features; re-ranking re-sorts the
                              queue by current partial influence so features that gain
                              importance via feature→feature→logit paths are promoted
                              earlier.  Default 4 matches the original circuit-tracer
                              `update_interval=4`.  Set to 0 for one-shot ordering
                              (original Pass-A ranking only, no online updates).
        feature_batch_size:   Number of features processed per online-ranking cycle.
                              Only active when online_ranking_interval > 0.  Default 32
                              matches the original circuit-tracer `batch_size=32`.
                              Effective features per ranking update = interval × batch
                              = 4 × 32 = 128 (matching original queue size).

    Returns:
        (adj, all_nodes, logit_probs) — sparse dict, ordered node list, and float32
        tensor [K] of softmax probabilities for the K selected logit target tokens.
        Returns ({}, [], zeros(0)) if no active latents are found on the probe sequences.
    """
    n_token_batches_total = (tokens.shape[0] + probe_batch_size - 1) // probe_batch_size
    adj: Dict[Tuple[int, int], float] = {}
    n_token_batches = 0
    all_nodes: List[FeatureID] = []
    logit_probs: torch.Tensor = torch.zeros(0, dtype=torch.float32)

    for token_batch in tokens.split(probe_batch_size):
        B = token_batch.shape[0]
        batch_num = n_token_batches + 1

        # ------------------------------------------------------------------
        # Step 1: Node discovery — no-grad forward for this batch
        # ------------------------------------------------------------------
        disc = SAEGraphInstrument(bank, stop_error_grad=stop_error_grad)
        with torch.no_grad():
            inference.forward(
                token_batch,
                patcher=disc,
                grad_enabled=False,
                return_activations=False,
                tokenize_final=False,
                all_logits=False,
            )
        # Collect ALL active features (no cap here when logit_top_k > 0 — selection
        # happens post-logit-ranking in Step 4).  Fall back to activation-magnitude
        # cap only when logit_top_k == 0 (no logit targets to rank by).
        discovery_cap = 0 if logit_top_k > 0 else max_feature_nodes
        feature_nodes_all = collect_active_feature_nodes(
            disc.graph, kinds, n_kinds, min_active_count, active_count,
            max_feature_nodes=discovery_cap,
        )
        # Collect error sentinels — one per (layer, kind) pair the SAE processed.
        # Must happen before del disc while graph.activations is still populated.
        error_nodes_all: List[FeatureID] = (
            [
                FeatureID(layer=l, kind=k + "_err", index=0)
                for (l, k) in sorted(disc.graph.activations.keys())
            ]
            if include_error_nodes else []
        )
        del disc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if not feature_nodes_all:
            print(
                f"[CTBaseline]   Batch {batch_num}/{n_token_batches_total}: "
                "no active features found, skipping.",
                flush=True,
            )
            n_token_batches += 1
            continue

        # ------------------------------------------------------------------
        # Step 2: Logit sentinel token IDs (cumulative probability selection)
        # ------------------------------------------------------------------
        logit_node_ids: List[int] = []
        batch_logit_probs: torch.Tensor = torch.zeros(0, dtype=torch.float32)
        if logit_top_k > 0:
            logit_node_ids, batch_logit_probs = _find_logit_targets(
                inference, token_batch, logit_top_k, desired_logit_prob
            )
        # Keep probabilities from the first batch (representative for the circuit)
        if logit_probs.shape[0] == 0 and batch_logit_probs.shape[0] > 0:
            logit_probs = batch_logit_probs

        logit_nodes = [FeatureID(layer=-1, kind="logit", index=tid) for tid in logit_node_ids]
        n_logit_nodes_batch = len(logit_nodes)

        # ------------------------------------------------------------------
        # Step 3: Single retained-graph forward pass
        # ------------------------------------------------------------------
        # all_logits=True so the logits tensor is available for logit-target
        # backward passes without a second forward pass.
        if include_token_nodes:
            # SAEGraphInstrumentWithEmbedding captures the first-layer residual
            # stream as a detached leaf anchor for per-position attribution.
            instrument: SAEGraphInstrument = SAEGraphInstrumentWithEmbedding(
                bank,
                stop_error_grad=stop_error_grad,
                first_layer=0,
                first_kind=kinds[0],
            )
        else:
            instrument = SAEGraphInstrument(bank, stop_error_grad=stop_error_grad)
        try:
            _, logits, _ = inference.forward(
                token_batch,
                patcher=instrument,
                grad_enabled=True,
                return_activations=False,
                tokenize_final=False,
                all_logits=True,
            )

            # Build token-node list (T positions) when enabled and anchor was captured.
            T = token_batch.shape[1]
            emb_anchor: Optional[torch.Tensor] = None
            token_nodes: List[FeatureID] = []
            if (include_token_nodes
                    and isinstance(instrument, SAEGraphInstrumentWithEmbedding)
                    and instrument.emb_anchor is not None):
                emb_anchor = instrument.emb_anchor
                token_nodes = [
                    FeatureID(layer=-2, kind="token", index=p) for p in range(T)
                ]

            # Pre-cache upstream anchor groups: one lookup per (layer, kind) key.
            upstream_cache: Dict[Tuple[int, str], Tuple[List[torch.Tensor], List]] = {}
            for (l, k) in instrument.graph.activations.keys():
                ups_anchors, ups_meta = _upstream_anchors_for_target(
                    instrument.graph, l, k, kinds, n_kinds
                )
                if ups_anchors:
                    upstream_cache[(l, k)] = (ups_anchors, ups_meta)

            # All anchors for logit targets (every layer is upstream of logit output).
            all_anchors, all_anchor_meta = _all_feature_anchors_with_meta(instrument.graph)

            # ------------------------------------------------------------------
            # Step 4: Logit ranking passes (Pass A) — determine feature priority
            # ------------------------------------------------------------------
            # Run logit backward passes against the FULL feature set to score each
            # feature's direct influence on every logit target.  Results go into
            # partial_adj_ranking (used only for ranking; discarded afterwards).
            # All passes retain the graph since feature passes will follow.
            feature_nodes: List[FeatureID] = feature_nodes_all   # default: no cap
            n_feature_nodes_all = len(feature_nodes_all)

            if (logit_top_k > 0
                    and logit_node_ids
                    and logits is not None
                    and logits.grad_fn is not None):
                last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
                # Layout: features | error_nodes | logit_nodes
                # (mirrors final all_nodes so error sentinel indices are stable)
                all_nodes_full = feature_nodes_all + error_nodes_all + logit_nodes
                nti_full: Dict[FeatureID, int] = {n: i for i, n in enumerate(all_nodes_full)}
                N_full = len(all_nodes_full)
                partial_adj_ranking: Dict[Tuple[int, int], float] = {}

                for j, (logit_node, token_id) in enumerate(
                    zip(logit_nodes, logit_node_ids)
                ):
                    logit_tgt_full = nti_full.get(logit_node)
                    if logit_tgt_full is None or token_id >= last_logits.shape[-1]:
                        continue
                    scalar = (last_logits[:, token_id] - last_logits.mean(dim=-1)).sum()
                    if scalar.grad_fn is None:
                        continue
                    grads = torch.autograd.grad(
                        scalar, all_anchors, retain_graph=True, allow_unused=True
                    )
                    _score_grads_into_adj(
                        grads, all_anchor_meta, instrument.graph,
                        feature_nodes_all, nti_full, logit_tgt_full, partial_adj_ranking,
                    )

                # ------------------------------------------------------------------
                # Step 5: Rank features by one-hop logit influence; apply cap
                # ------------------------------------------------------------------
                # Computed SPARSELY to avoid a dense N_full×N_full matrix.
                # partial_adj_ranking has only K logit-target columns (K ≤ logit_top_k),
                # so we compute row-normalised one-hop influence directly from the dict.
                #
                # one_hop[i] = Σ_j  lw[j] * |adj[(i,j)]| / row_abs_sum[j]
                #
                # where j ranges over logit targets and row_abs_sum[j] is the L1 norm
                # of column j (= row j of A_norm since A is transposed before storage).
                if max_feature_nodes > 0 and max_feature_nodes < n_feature_nodes_all:
                    n_error_nodes_all = len(error_nodes_all)
                    feature_inf_tensor = _compute_partial_one_hop_influence(
                        partial_adj_ranking,
                        n_feature_nodes_all,
                        n_error_nodes_all,
                        n_logit_nodes_batch,
                        batch_logit_probs,
                    )
                    cap = min(max_feature_nodes, n_feature_nodes_all)
                    top_sorted = sorted(feature_inf_tensor.topk(cap).indices.tolist())
                    feature_nodes = [feature_nodes_all[i] for i in top_sorted]
                    print(
                        f"[CTBaseline]     Logit-influence ranking: "
                        f"{len(feature_nodes)}/{n_feature_nodes_all} features selected",
                        flush=True,
                    )

            # Canonical node list: features | error_nodes | token_nodes | logit_sentinels.
            # Established on the first batch; later batches score into the same indices.
            if not all_nodes:
                all_nodes = feature_nodes + error_nodes_all + token_nodes + logit_nodes

            node_to_idx: Dict[FeatureID, int] = {n: i for i, n in enumerate(all_nodes)}
            n_feature_nodes = len(feature_nodes)
            n_error_nodes = len(error_nodes_all)
            n_total_targets = n_feature_nodes + n_logit_nodes_batch

            n_token_nodes = len(token_nodes)
            print(
                f"[CTBaseline]   Batch {batch_num}/{n_token_batches_total} | "
                f"{n_feature_nodes} feature + {n_error_nodes} error"
                + (f" + {n_token_nodes} token" if n_token_nodes else "")
                + f" + {n_logit_nodes_batch} logit nodes"
                f" → {n_logit_nodes_batch + n_feature_nodes} backward passes",
                flush=True,
            )

            # ------------------------------------------------------------------
            # Step 6: Logit-target backward passes — Pass B (final, into adj)
            # ------------------------------------------------------------------
            # The retained graph is still alive from Pass A.  All logit passes
            # retain the graph since feature passes come after.
            # When include_token_nodes=True, emb_anchor is appended to the anchor
            # list so token attribution is computed alongside feature attribution.
            all_anchors_b = all_anchors + ([emb_anchor] if emb_anchor is not None else [])

            if logit_node_ids and logits is not None and logits.grad_fn is not None:
                last_logits = logits[:, -1, :] if logits.dim() == 3 else logits

                for j, (logit_node, token_id) in enumerate(
                    zip(logit_nodes, logit_node_ids)
                ):
                    logit_tgt_idx = node_to_idx.get(logit_node)
                    if logit_tgt_idx is None or token_id >= last_logits.shape[-1]:
                        continue
                    # Demean: attribute relative to mean logit, matching the
                    # original circuit tracer's use of W_U[i] - W_U.mean(0).
                    scalar = (last_logits[:, token_id] - last_logits.mean(dim=-1)).sum()
                    if scalar.grad_fn is None:
                        continue
                    grads = torch.autograd.grad(
                        scalar, all_anchors_b,
                        retain_graph=True,  # always retain: feature passes follow
                        allow_unused=True,
                    )
                    if emb_anchor is not None:
                        _score_grads_into_adj(
                            grads[:-1], all_anchor_meta, instrument.graph,
                            feature_nodes, node_to_idx, logit_tgt_idx, adj,
                        )
                        if grads[-1] is not None:
                            _score_token_grads_into_adj(
                                grads[-1], emb_anchor, token_nodes,
                                node_to_idx, logit_tgt_idx, adj,
                            )
                    else:
                        _score_grads_into_adj(
                            grads, all_anchor_meta, instrument.graph,
                            feature_nodes, node_to_idx, logit_tgt_idx, adj,
                        )

                print(
                    f"[CTBaseline]     {n_logit_nodes_batch} logit targets scored"
                    f" | edges so far: {len(adj)}",
                    flush=True,
                )

            # ------------------------------------------------------------------
            # Step 7: Feature-target backward passes (selected features only)
            # ------------------------------------------------------------------
            # Helper to run a single feature backward pass.  Returns True if a
            # torch.autograd.grad call was made, False if the feature was skipped.
            def _run_one_feature_pass(
                tgt_node: FeatureID, retain: bool
            ) -> bool:
                tgt_idx = node_to_idx.get(tgt_node)
                if tgt_idx is None:
                    return False
                try:
                    _, tgt_connected, _ = instrument.graph.get_latents(
                        tgt_node.layer, tgt_node.kind
                    )
                except (KeyError, IndexError):
                    return False
                if tgt_connected.act is None:
                    return False
                target_acts = tgt_connected.act[..., tgt_node.index]  # [B, T]
                pos_argmax = target_acts.argmax(dim=-1)                # [B]
                b_idx = torch.arange(B, device=target_acts.device)
                scalar = target_acts[b_idx, pos_argmax].sum()
                if scalar.grad_fn is None:
                    return False

                cached = upstream_cache.get((tgt_node.layer, tgt_node.kind))
                feat_anchors: List[torch.Tensor] = []
                feat_meta: List = []
                if cached is not None:
                    feat_anchors, feat_meta = cached

                # Append emb_anchor so token attribution is computed alongside
                # feature attribution in a single autograd.grad call.
                pass_anchors = feat_anchors + ([emb_anchor] if emb_anchor is not None else [])
                if not pass_anchors:
                    return False

                grads = torch.autograd.grad(
                    scalar, pass_anchors, retain_graph=retain, allow_unused=True
                )
                if feat_meta:
                    _score_grads_into_adj(
                        grads[:len(feat_anchors)], feat_meta, instrument.graph,
                        feature_nodes, node_to_idx, tgt_idx, adj,
                    )
                if emb_anchor is not None and grads[-1] is not None:
                    _score_token_grads_into_adj(
                        grads[-1], emb_anchor, token_nodes,
                        node_to_idx, tgt_idx, adj,
                    )
                return True

            use_online = (
                online_ranking_interval > 0
                and feature_batch_size > 0
                and n_feature_nodes > 1
            )

            if use_online:
                # Online path: re-rank unvisited features every
                # online_ranking_interval cycles so features promoted by
                # feature→feature→logit paths get processed earlier.
                unvisited: List[int] = list(range(n_feature_nodes))
                n_features_done = 0
                cycle = 0

                while unvisited:
                    if cycle % online_ranking_interval == 0:
                        # Full Neumann series on the growing partial adj, matching the
                        # original's compute_partial_influences (multi-hop ranking so
                        # feature A is promoted when it strongly drives already-processed
                        # feature B which in turn drives the logit).
                        n_nodes_total = len(all_nodes)
                        logit_start = n_feature_nodes + n_error_nodes + n_token_nodes
                        lw = torch.zeros(n_nodes_total, dtype=torch.float32)
                        if batch_logit_probs.shape[0] == n_logit_nodes_batch:
                            lw[logit_start:logit_start + n_logit_nodes_batch] = (
                                batch_logit_probs.float()
                            )
                        elif n_logit_nodes_batch > 0:
                            lw[logit_start:logit_start + n_logit_nodes_batch] = (
                                1.0 / n_logit_nodes_batch
                            )
                        full_inf = _compute_partial_neumann_influence(
                            adj, n_nodes_total, lw, max_iter=128
                        )
                        inf_vec = full_inf[:n_feature_nodes]
                        unvisited.sort(key=lambda fi: -float(inf_vec[fi]))

                    batch_fi = unvisited[:feature_batch_size]
                    unvisited = unvisited[feature_batch_size:]

                    for k, local_i in enumerate(batch_fi):
                        # retain=False only on the very last feature slot;
                        # use slot count (not backward count) — safe to
                        # over-retain; del instrument cleans up anyway.
                        slot = n_features_done + k
                        retain = (slot + 1) < n_feature_nodes
                        _run_one_feature_pass(feature_nodes[local_i], retain)

                    n_features_done += len(batch_fi)
                    cycle += 1
                    print(
                        f"[CTBaseline]     Online cycle {cycle}:"
                        f" {n_features_done}/{n_feature_nodes} features"
                        f" | edges: {len(adj)}",
                        flush=True,
                    )
            else:
                # One-shot sequential path (default)
                for i, tgt_node in enumerate(feature_nodes):
                    retain = (n_logit_nodes_batch + i) < n_total_targets - 1
                    _run_one_feature_pass(tgt_node, retain)

                    if (i + 1) % 256 == 0 or i == n_feature_nodes - 1:
                        print(
                            f"[CTBaseline]     {n_logit_nodes_batch + i + 1}"
                            f"/{n_total_targets} targets done | edges: {len(adj)}",
                            flush=True,
                        )

        finally:
            # ------------------------------------------------------------------
            # Step 6: Free the retained graph (runs even if a backward raises)
            # ------------------------------------------------------------------
            del instrument
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        n_token_batches += 1
        print(
            f"[CTBaseline]   Batch {n_token_batches}/{n_token_batches_total} done"
            f" | total edges: {len(adj)}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Step 7: Average scores across token batches
    # ------------------------------------------------------------------
    if n_token_batches > 1:
        adj = {k: v / n_token_batches for k, v in adj.items()}

    return adj, all_nodes, logit_probs


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
