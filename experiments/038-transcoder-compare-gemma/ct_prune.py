"""circuit-tracer pruning, called as THEIR code, plus the one labelled
adaptation this comparison needs.

Pinned library: decoderesearch/circuit-tracer @ 8f1e2438 (2026-07-17),
clone verified unmodified (git status clean) -- see ct_faithfulness.py.

Two entry points:

  prune_published(graph)
      Their `prune_graph` verbatim, their defaults (node 0.8, edge 0.98),
      rooted at the logit nodes as published. Nothing of ours runs.

  prune_rooted(graph, root_rows, root_weights=None)
      Their `prune_graph` BODY with exactly one substitution: the root
      weight vector. As published, the root is the logit nodes weighted
      by their probabilities; here it is the given rows (our seed's
      rows). Every computation -- normalisation, multi-hop influence,
      cumulative thresholds, edge scoring, iterative cleanup -- is their
      function, called unchanged. The two analogues of "logits are
      always kept" and "logits are exempt from the needs-outgoing-edges
      rule" are applied to the root rows, since a root is a sink by
      definition. With root = the logit nodes and root_weights = the
      logit probabilities this MUST reproduce prune_graph exactly;
      `identity_check` asserts that, and ct_faithfulness.py runs it.

And the reading of a pruned graph that both arms share:

  seed_circuit(graph, result, L, sl)
      The surviving upstream feature nodes that are ANCESTORS of the
      seed in the pruned edge graph (a path of kept edges leads from
      them to a seed row). "Upstream of the seed in their pruned graph"
      is this set; for the rooted arm every survivor is an ancestor by
      construction, so the same function serves both.
"""
import torch

from circuit_tracer.graph import (PruneResult, compute_edge_influence,
                                  compute_node_influence, find_threshold,
                                  prune_graph)

NODE_T, EDGE_T = 0.8, 0.98


def prune_published(graph, node_threshold=NODE_T, edge_threshold=EDGE_T):
    return prune_graph(graph, node_threshold, edge_threshold)


def prune_rooted(graph, root_rows, root_weights=None,
                 node_threshold=NODE_T, edge_threshold=EDGE_T):
    adj = graph.adjacency_matrix
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_targets)
    n_features = len(graph.selected_features)
    root_rows = torch.as_tensor(root_rows, device=adj.device, dtype=torch.long)

    # --- the substitution: root weights on the given rows ---
    w = torch.zeros(adj.shape[0], device=adj.device, dtype=adj.dtype)
    w[root_rows] = (1.0 if root_weights is None
                    else torch.as_tensor(root_weights, device=adj.device,
                                         dtype=adj.dtype))
    # --- from here, their body, their functions ---
    node_influence = compute_node_influence(adj, w)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    node_mask[-n_logits - n_tokens:] = True          # theirs: keep tokens+logits
    node_mask[root_rows] = True                       # analogue: keep the root

    pruned = adj.clone()
    pruned[~node_mask] = 0
    pruned[:, ~node_mask] = 0
    edge_scores = compute_edge_influence(pruned, w)
    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)

    # "feature and error nodes need an outgoing edge" -- exempting the
    # root rows exactly as their code exempts the logit rows (a sink).
    sink = torch.zeros_like(node_mask)
    sink[-n_logits - n_tokens:] = True
    sink[root_rows] = True
    old = node_mask.clone()
    has_out = edge_mask[:, :-n_logits - n_tokens].any(0)
    node_mask[:-n_logits - n_tokens] &= (has_out | sink[:-n_logits - n_tokens])
    node_mask[:n_features] &= edge_mask[:n_features].any(1)
    while not torch.all(node_mask == old):
        old[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        has_out = edge_mask[:, :-n_logits - n_tokens].any(0)
        node_mask[:-n_logits - n_tokens] &= (has_out | sink[:-n_logits - n_tokens])
        node_mask[:n_features] &= edge_mask[:n_features].any(1)

    sorted_scores, sorted_idx = torch.sort(node_influence, descending=True)
    cum = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    final = torch.zeros_like(node_influence)
    final[sorted_idx] = cum
    return PruneResult(node_mask, edge_mask, final)


def prune_pinned(graph, pin_rows, node_threshold=NODE_T, edge_threshold=EDGE_T):
    """Daniel's minimal blocker (2026-08-23): their pruning EXACTLY as
    published -- logit root, their thresholds, their cleanup -- with ONE
    change: the given rows are pinned, using the same mechanism their
    code applies to logit/token nodes (forced into node_mask and exempt
    from the needs-edges cleanup). Node AND edge retention remain
    logit-weighted, so this asks: does keeping the seed visible, without
    re-rooting anything, leave it with surviving parents? A pinned seed
    with no kept incoming edges on a window is a real outcome and is
    reported as an empty circuit for that window."""
    adj = graph.adjacency_matrix
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_targets)
    n_features = len(graph.selected_features)
    pin_rows = torch.as_tensor(pin_rows, device=adj.device, dtype=torch.long)

    w = torch.zeros(adj.shape[0], device=adj.device, dtype=adj.dtype)
    w[-n_logits:] = graph.logit_probabilities            # their root, unchanged
    node_influence = compute_node_influence(adj, w)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    node_mask[-n_logits - n_tokens:] = True
    node_mask[pin_rows] = True                            # the one change

    pruned = adj.clone()
    pruned[~node_mask] = 0
    pruned[:, ~node_mask] = 0
    edge_scores = compute_edge_influence(pruned, w)
    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)

    sink = torch.zeros_like(node_mask)
    sink[-n_logits - n_tokens:] = True
    sink[pin_rows] = True                                 # pin: exempt from cleanup
    old = node_mask.clone()
    has_out = edge_mask[:, :-n_logits - n_tokens].any(0)
    node_mask[:-n_logits - n_tokens] &= (has_out | sink[:-n_logits - n_tokens])
    node_mask[:n_features] &= (edge_mask[:n_features].any(1) | sink[:n_features])
    while not torch.all(node_mask == old):
        old[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        has_out = edge_mask[:, :-n_logits - n_tokens].any(0)
        node_mask[:-n_logits - n_tokens] &= (has_out | sink[:-n_logits - n_tokens])
        node_mask[:n_features] &= (edge_mask[:n_features].any(1) | sink[:n_features])

    sorted_scores, sorted_idx = torch.sort(node_influence, descending=True)
    cum = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    final = torch.zeros_like(node_influence)
    final[sorted_idx] = cum
    return PruneResult(node_mask, edge_mask, final)


def identity_check(graph, node_threshold=NODE_T, edge_threshold=EDGE_T):
    """prune_rooted with the LOGIT root must equal prune_graph exactly."""
    n = graph.adjacency_matrix.shape[0]
    n_logits = len(graph.logit_targets)
    rows = torch.arange(n - n_logits, n)
    a = prune_graph(graph, node_threshold, edge_threshold)
    b = prune_rooted(graph, rows, graph.logit_probabilities,
                     node_threshold, edge_threshold)
    return (bool(torch.equal(a.node_mask, b.node_mask))
            and bool(torch.equal(a.edge_mask, b.edge_mask)))


def seed_rows(graph, L, sl):
    af = graph.active_features.to(graph.adjacency_matrix.device)
    sel = graph.selected_features.to(af.device).flatten()
    af_sel = af[sel]
    return ((af_sel[:, 0] == L) & (af_sel[:, 2] == sl)).nonzero(as_tuple=True)[0], af_sel


def seed_circuit(graph, result, L, sl):
    """Surviving upstream feature nodes with a kept-edge path to a seed
    row. Returns (seed_survived, [(layer, feature), ...] deduplicated
    over positions, n_ancestor_nodes)."""
    rows, af_sel = seed_rows(graph, L, sl)
    n_features = len(graph.selected_features)
    nm, em = result.node_mask, result.edge_mask
    alive_seed = rows[nm[rows]]
    if not len(alive_seed):
        return False, [], 0
    # ancestors: BFS over kept edges, target<-source (row = target)
    reach = torch.zeros(nm.shape[0], dtype=torch.bool, device=nm.device)
    frontier = alive_seed
    while len(frontier):
        srcs = em[frontier].any(0) & ~reach
        srcs[frontier] = False
        reach |= srcs
        frontier = srcs.nonzero(as_tuple=True)[0]
    feat = reach[:n_features] & (af_sel[:, 0] < L)
    idx = feat.nonzero(as_tuple=True)[0]
    members = sorted({(int(af_sel[i, 0]), int(af_sel[i, 2])) for i in idx.tolist()})
    return True, members, int(idx.numel())
