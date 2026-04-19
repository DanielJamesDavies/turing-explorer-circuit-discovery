"""
ct_influence.py — Influence propagation and graph pruning for the circuit-tracer baseline.

Two public functions:

    compute_ct_influence  — Power-iteration Neumann series from logit root nodes.
    prune_ct_graph        — Threshold by influence + edge score, then iteratively
                            remove dead (disconnected) nodes.

Both operate on CPU tensors and the sparse adj dict produced by
compute_direct_effects_matrix (attribution.py).  They are deliberately kept
separate from the forward/backward machinery so they can be tested in isolation.
"""

import logging
import torch
from typing import Dict, List, Optional, Set, Tuple

from circuit.types.feature_id import FeatureID

logger = logging.getLogger(__name__)


def compute_ct_influence(
    adj: Dict[Tuple[int, int], float],
    all_nodes: List[FeatureID],
    logit_top_k: int,
    max_iter: int = 1000,
    logit_probabilities: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Propagates influence backward from logit root nodes through the direct-effects
    adjacency matrix using a truncated Neumann series:

        influence = logit_weights @ (A_norm + A_norm² + A_norm³ + …)

    This is the SAE analogue of the influence computation in Anthropic's Attribution
    Graphs paper (transformer-circuits.pub/2025/attribution-graphs/methods.html §3.3).

    The adjacency matrix A uses the convention A[tgt, src] = direct effect of src
    on tgt (matching compute_direct_effects_matrix).  Before propagation it is:
        1. Made non-negative via abs().
        2. Row-normalised so each row sums to ≤ 1 (stochastic mixing weights).
           Row-normalisation epsilon is 1e-10 (matching the original circuit-tracer).

    Logit root nodes (all_nodes[-logit_top_k:]) seed the propagation with their
    actual softmax probabilities when provided, or uniform 1/K otherwise.  Their
    influence in the output is whatever the series produces — typically 0, since no
    edges point *into* logit nodes from features (they are pure sources).

    Convergence: iterate until `current` is exactly zero (finite DAG property).
    `max_iter` is a safety cap; if reached, a RuntimeError is raised (matching the
    original circuit-tracer) — the caller should catch it and use a fallback.

    Args:
        adj:                  Sparse dict keyed by (src_idx, tgt_idx) → raw score.
                              Produced by compute_direct_effects_matrix.
        all_nodes:            Ordered node list (same ordering as matrix indices).
                              all_nodes[-logit_top_k:] must be logit sentinel nodes.
        logit_top_k:          Number of logit sentinel nodes at the tail of all_nodes.
                              If 0, all influence is zero (no roots → no propagation).
        max_iter:             Safety cap on power-iteration steps.  Default 1000 matches
                              the original circuit-tracer.  RuntimeError is raised if the
                              series has not converged by this iteration.
        logit_probabilities:  Optional float32 tensor of shape [logit_top_k] with the
                              softmax probability of each logit target (from
                              _find_logit_targets).  When provided, seeds root weights
                              by probability rather than uniform 1/K.

    Returns:
        influence: float32 CPU tensor of shape [N] where N = len(all_nodes).
                   Larger values indicate stronger multi-hop causal influence on
                   the logit outputs.  Logit nodes are typically 0 (no in-edges
                   from features); prune_ct_graph always keeps them regardless.

    Raises:
        RuntimeError: if the Neumann series has not converged after `max_iter`
                      iterations, indicating a graph with strong cycles from
                      probe-sequence averaging.  Callers should catch this and
                      apply a fallback (e.g. use zeros or skip the batch).
    """
    N = len(all_nodes)
    if N == 0 or logit_top_k <= 0:
        return torch.zeros(N, dtype=torch.float32)

    logit_start = N - logit_top_k

    # ── Build dense adjacency matrix A on CPU ────────────────────────────────
    # adj uses (src_idx, tgt_idx) keys; A[tgt, src] convention.
    A = torch.zeros(N, N, dtype=torch.float32)
    for (src_idx, tgt_idx), score in adj.items():
        if 0 <= src_idx < N and 0 <= tgt_idx < N:
            A[tgt_idx, src_idx] = float(score)

    # ── Absolute-value row-normalise ─────────────────────────────────────────
    # Taking abs converts inhibitory edges to positive mixing weights so that
    # inhibitors contribute to upstream influence (matching circuit-tracer §3.3).
    # Epsilon 1e-10 matches the original's clamp(min=1e-10).
    A_abs = A.abs()
    row_sums = A_abs.sum(dim=1, keepdim=True).clamp(min=1e-10)
    A_norm = A_abs / row_sums

    # ── Logit root weights ────────────────────────────────────────────────────
    # Use actual softmax probabilities when available; fall back to uniform.
    logit_weights = torch.zeros(N, dtype=torch.float32)
    if logit_probabilities is not None and logit_probabilities.shape[0] == logit_top_k:
        logit_weights[logit_start:] = logit_probabilities.float()
    else:
        logit_weights[logit_start:] = 1.0 / logit_top_k

    # ── Truncated Neumann series: logit_weights @ (A + A² + A³ + …) ─────────
    # Iterate until current is exactly zero (correct for finite DAGs) or until
    # max_iter is reached.  Raises RuntimeError on non-convergence (matching the
    # original circuit-tracer) so the caller can apply graceful degradation.
    current = logit_weights @ A_norm   # [N] — one-hop influence
    influence = current.clone()

    iterations = 1
    while current.any():
        if iterations >= max_iter:
            raise RuntimeError(
                f"compute_ct_influence: series did not converge after {max_iter} "
                "iterations (graph may contain strong cycles from probe-sequence "
                "averaging).  Catch this RuntimeError and apply a fallback."
            )
        current = current @ A_norm
        influence += current
        iterations += 1

    return influence


def _find_threshold(scores: torch.Tensor, fraction: float) -> float:
    """
    Returns the minimum score cutpoint such that nodes/edges with score >= cutpoint
    collectively account for at least `fraction` of the total score mass.

    Direct port of find_threshold from Anthropic's circuit-tracer graph.py.

    Example: fraction=0.8 keeps the smallest set of top-scoring nodes that together
    hold 80% of total influence.
    """
    if scores.numel() == 0:
        return 0.0
    total = scores.sum()
    if total <= 0:
        return 0.0
    sorted_scores = scores.sort(descending=True).values
    cumsum = sorted_scores.cumsum(0) / total
    idx = int(torch.searchsorted(cumsum.contiguous(), torch.tensor(fraction)).item())
    idx = min(idx, len(cumsum) - 1)
    return float(sorted_scores[idx])


def _compute_partial_neumann_influence(
    adj: Dict[Tuple[int, int], float],
    n_nodes: int,
    logit_weights: torch.Tensor,
    max_iter: int = 128,
) -> torch.Tensor:
    """
    Computes a full Neumann-series influence vector from a partial (growing) adj dict.

    Matches compute_partial_influences from Anthropic's circuit-tracer
    attribute_transformerlens.py: builds a dense [n_nodes, n_nodes] matrix from the
    current adj, row-normalises it, then runs the truncated Neumann series seeded by
    logit_weights — giving multi-hop influence rather than the one-hop approximation.

    This is used inside the online feature-ranking loop (Step 7 of
    compute_direct_effects_matrix) so that features which drive other already-processed
    features (and hence transitively drive the logit outputs) are ranked ahead of
    features with only weak direct logit connections.

    Max matrix size: (max_feature_nodes + n_error + n_token + n_logit)².
    With max_feature_nodes=2048, n_error≈72, n_logit≈8: N≈2130 → ~18 MB per call.
    Typically called ~ceil(max_feature_nodes / (interval × batch)) ≈ 16 times per
    batch — well within CPU memory budget.

    Args:
        adj:           Sparse dict (src_idx, tgt_idx) → raw score from
                       compute_direct_effects_matrix (growing during Step 7).
        n_nodes:       Total number of nodes: len(all_nodes).  Must cover the full
                       index range used in adj (features + error + token + logit).
        logit_weights: [n_nodes] root seed weights; non-zero only at logit indices.
        max_iter:      Safety cap on the Neumann series.  Default 128 matches the
                       original circuit-tracer's compute_partial_influences cap.

    Returns:
        Float32 CPU tensor of shape [n_nodes].  Slice [:n_feature_nodes] to get
        per-feature influence for re-ranking.
    """
    A = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    for (src, tgt), v in adj.items():
        if 0 <= src < n_nodes and 0 <= tgt < n_nodes:
            A[tgt, src] = float(v)

    A_abs = A.abs()
    row_sums = A_abs.sum(dim=1, keepdim=True).clamp(min=1e-12)
    A_norm = A_abs / row_sums

    current = logit_weights @ A_norm    # [n_nodes] — one-hop from logit roots
    inf_vec = current.clone()
    for _ in range(max_iter):
        if not current.any():
            break
        current = current @ A_norm
        inf_vec += current

    return inf_vec


def _compute_edge_influence(
    A_pruned: torch.Tensor,
    logit_weights: torch.Tensor,
    max_iter: int = 1000,
) -> torch.Tensor:
    """
    Scores each directed edge by influence flowing through the PRUNED subgraph.

    Matches compute_edge_influence from Anthropic's circuit-tracer graph.py:
        1. Re-normalise A_pruned (abs + row-normalise) on the pruned graph so that
           surviving edges are re-weighted after node removal.
        2. Run the full Neumann series on the pruned graph → pruned_influence.
        3. edge_score[tgt, src] = A_norm_p[tgt, src]
                                   * (pruned_influence[tgt] + logit_weights[tgt])

    Using the PRUNED matrix (dropped-node rows/cols zeroed) rather than the
    full pre-pruning matrix ensures that surviving edges are re-weighted and that
    edge scores reflect the actual pruned subgraph topology — a target node that
    lost 8 of its 10 incoming edges now distributes 100% of its row weight among
    the 2 survivors.

    Args:
        A_pruned:      [N, N] dense float32; rows/cols of removed nodes are zeroed.
                       A_pruned[tgt, src] convention (matching build in prune_ct_graph).
        logit_weights: [N] root seed weights (softmax probs for logit nodes, 0 elsewhere).
        max_iter:      Safety cap on the Neumann series (should converge in ≤ N steps
                       for a DAG; weak cycles from probe averaging may need more).

    Returns:
        Float32 tensor of shape [N, N]: edge_scores[tgt, src].
    """
    A_abs = A_pruned.abs()
    row_sums = A_abs.sum(dim=1, keepdim=True).clamp(min=1e-12)
    A_norm_p = A_abs / row_sums                 # re-normalised on pruned subgraph

    # Full Neumann series on the pruned graph (matches compute_influence internally)
    current = logit_weights @ A_norm_p          # [N] — one-hop from logit roots
    pruned_inf = current.clone()
    for _ in range(max_iter):
        if not current.any():
            break
        current = current @ A_norm_p
        pruned_inf += current

    # Add root weights so logit nodes receive positive combined weight
    # (mirrors original graph.py line 213: pruned_influence += logit_weights)
    pruned_inf = pruned_inf + logit_weights

    return A_norm_p * pruned_inf[:, None]       # [N, N] — broadcast over src axis


def prune_ct_graph(
    adj: Dict[Tuple[int, int], float],
    all_nodes: List[FeatureID],
    influence: torch.Tensor,
    node_threshold: float,
    edge_threshold: float,
    logit_top_k: int,
    logit_probabilities: Optional[torch.Tensor] = None,
    max_iter: int = 1000,
) -> Tuple[Dict[Tuple[int, int], float], List[FeatureID]]:
    """
    Prunes the direct-effects graph using scale-invariant fraction-based thresholds,
    matching Anthropic's circuit-tracer prune_graph algorithm.

    Algorithm:
        1. Node pruning — find the score cutpoint that covers `node_threshold`
           fraction of total feature-node influence (via _find_threshold).  Keep
           feature nodes above the cutpoint; always keep logit sentinel nodes.
        2. Edge pruning — build a pruned copy of the dense adjacency matrix (dropped
           nodes zeroed), re-normalise it, re-run the full Neumann series on the pruned
           subgraph, then score each edge by that re-computed influence.  This matches
           the original circuit-tracer's compute_edge_influence(pruned_matrix, …) which
           ensures surviving edges are re-weighted after node removal.
        3. Dead-node removal — iteratively remove nodes with missing connectivity:
             • Feature nodes: need both an outgoing and an incoming surviving edge.
             • Error nodes (kind ending "_err"): source-only — only need an outgoing
               edge (nothing writes into them; matches original's treatment).
             • Token nodes (kind == "token"): source-only — same rule as error nodes.
             • Logit nodes: always kept regardless.

    `node_threshold` and `edge_threshold` are **fractions** in [0, 1], not absolute
    values.  0.8 / 0.98 (defaults) keeps the smallest node/edge sets that together
    account for 80% / 98% of the respective influence mass.

    Args:
        adj:                  Sparse dict (src_idx, tgt_idx) → score.
        all_nodes:            Ordered node list matching adj indices.
        influence:            Per-node scores from compute_ct_influence [N].
        node_threshold:       Fraction of total node-influence to retain (0–1).
        edge_threshold:       Fraction of total edge-influence to retain (0–1).
        logit_top_k:          Number of logit sentinel nodes at tail of all_nodes.
        logit_probabilities:  Optional [logit_top_k] softmax probs for logit roots;
                              used when computing edge influence weights.  Falls back
                              to uniform 1/K if None or wrong shape.
        max_iter:             Safety cap on the Neumann series run inside
                              _compute_edge_influence on the pruned subgraph.
                              Default 1000 matches the original circuit-tracer.

    Returns:
        (pruned_adj, kept_nodes):
            pruned_adj  — sub-dict of adj restricted to kept nodes and edges.
            kept_nodes  — List[FeatureID] in original index order, logit nodes last.
    """
    N = len(all_nodes)
    if N == 0:
        return {}, []

    logit_start = N - logit_top_k

    # ── Build dense adjacency matrix A on CPU ────────────────────────────────
    # A[tgt, src] convention.  Kept in full form so we can zero out dropped nodes.
    A = torch.zeros(N, N, dtype=torch.float32)
    for (src_idx, tgt_idx), score in adj.items():
        if 0 <= src_idx < N and 0 <= tgt_idx < N:
            A[tgt_idx, src_idx] = float(score)

    # ── Logit root weights (mirrors compute_ct_influence) ─────────────────────
    logit_weights = torch.zeros(N, dtype=torch.float32)
    if logit_top_k > 0:
        if logit_probabilities is not None and logit_probabilities.shape[0] == logit_top_k:
            logit_weights[logit_start:] = logit_probabilities.float()
        else:
            logit_weights[logit_start:] = 1.0 / logit_top_k

    # ── Step 1: Node pruning (fraction-based) ─────────────────────────────────
    # Apply threshold only to feature-node influence scores; logit nodes are exempt.
    feature_influence = influence[:logit_start].clamp(min=0.0)
    node_cutoff = _find_threshold(feature_influence, node_threshold)

    kept_set: Set[int] = set()
    for i in range(N):
        if i >= logit_start:
            kept_set.add(i)                        # logit nodes always kept
        elif float(influence[i]) >= node_cutoff:
            kept_set.add(i)

    # ── Step 2: Edge pruning on the PRUNED subgraph ───────────────────────────
    # Build a pruned copy of A (zero out dropped-node rows and columns), then
    # pass it to _compute_edge_influence which re-normalises and re-runs the
    # full Neumann series on the pruned topology.  This matches the original
    # circuit-tracer's compute_edge_influence(pruned_matrix, logit_weights).
    A_pruned = A.clone()
    for i in range(N):
        if i not in kept_set:
            A_pruned[i, :] = 0.0
            A_pruned[:, i] = 0.0

    edge_scores = _compute_edge_influence(A_pruned, logit_weights, max_iter)

    kept_list = sorted(kept_set)
    kept_tensor = torch.tensor(kept_list, dtype=torch.long)
    sub_edge_scores = edge_scores[kept_tensor][:, kept_tensor].clamp(min=0.0)
    edge_cutoff = _find_threshold(sub_edge_scores.flatten(), edge_threshold)

    pruned_adj: Dict[Tuple[int, int], float] = {}
    for (s, t), v in adj.items():
        if s in kept_set and t in kept_set:
            if float(edge_scores[t, s]) >= edge_cutoff:
                pruned_adj[(s, t)] = v

    # ── Step 3: Iterative dead-node removal ───────────────────────────────────
    # Matches the original circuit tracer's directed connectivity rule:
    #   • Feature nodes need both an outgoing AND an incoming surviving edge.
    #   • Error nodes (kind ending "_err") are source-only — only an outgoing edge
    #     is required (nothing writes into them; matches original's treatment).
    #   • Token nodes (kind == "token") are also source-only — they represent input
    #     embeddings which are never downstream of any other SAE feature.
    #   • Logit nodes are exempt (pure sinks seeding influence: no outgoing required).
    #
    # source_only_indices: nodes that only need an outgoing edge to survive.
    source_only_indices: Set[int] = {
        i for i, n in enumerate(all_nodes)
        if n.kind.endswith("_err") or n.kind == "token"
    }

    changed = True
    while changed:
        changed = False
        nodes_with_out: Set[int] = set()   # nodes appearing as src in some edge
        nodes_with_in: Set[int] = set()    # nodes appearing as tgt in some edge
        for s, t in pruned_adj:
            nodes_with_out.add(s)
            nodes_with_in.add(t)

        dead: Set[int] = {
            i for i in kept_set
            if i < logit_start
            and (
                i not in nodes_with_out                      # all non-logit nodes need outgoing
                or (i not in source_only_indices             # feature nodes (not source-only)
                    and i not in nodes_with_in)              #   also need at least one incoming
            )
        }
        if dead:
            kept_set -= dead
            pruned_adj = {
                (s, t): v
                for (s, t), v in pruned_adj.items()
                if s not in dead and t not in dead
            }
            changed = True

    kept_nodes: List[FeatureID] = [all_nodes[i] for i in sorted(kept_set)]
    return pruned_adj, kept_nodes
