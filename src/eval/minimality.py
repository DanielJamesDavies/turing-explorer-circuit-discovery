import torch
from typing import Dict, Any, List, Optional, Set
from store.circuits import Circuit
from eval.faithfulness import evaluate_faithfulness
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness

@torch.no_grad()
def evaluate_minimality(
    inference: Any,
    sae_bank: Any,
    avg_acts: torch.Tensor,
    circuit: Circuit,
    tokens: torch.Tensor,
    pos_argmax: Optional[torch.Tensor] = None,
    max_layer: Optional[int] = None,
    circuit_layers: Optional[Set[int]] = None,
) -> Dict[str, float]:
    """
    Checks for "dead weight" in a circuit. 
    A circuit is minimal if removing any single node significantly reduces faithfulness.
    
    Logic: Perform "Leave-One-Out" (LOO) ablation for every node in the circuit.
    Returns a dictionary mapping node UUIDs to their importance (faithfulness drop when removed).
    
    Args:
        inference: The Inference instance (model.inference.Inference).
        sae_bank: The SAEBank containing the models.
        avg_acts: Tensor of average activations per latent.
        circuit: The Circuit object to evaluate.
        tokens: The input tokens tensor [batch, seq_len].
        pos_argmax: The position where each sequence peaks for the seed latent.
        max_layer: Optional layer limit for patching.
        
    Returns:
        Dict[str, float]: A mapping from node UUIDs to their importance (faithfulness drop).
    """
    # 1. Base faithfulness with the complete circuit
    base_faithfulness = evaluate_faithfulness(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax, max_layer=max_layer, circuit_layers=circuit_layers)

    # 2. Leave-One-Out Ablation for each node
    node_importance = {}
    original_nodes = circuit.nodes

    for node_uuid in original_nodes:
        modified_nodes = {k: v for k, v in original_nodes.items() if k != node_uuid}
        circuit.nodes = modified_nodes

        # Calculate faithfulness without this node
        new_faithfulness = evaluate_faithfulness(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax, max_layer=max_layer, circuit_layers=circuit_layers)
        
        # Importance = how much faithfulness drops when this node is missing
        node_importance[node_uuid] = base_faithfulness - new_faithfulness
        
        # Restore circuit nodes
        circuit.nodes = original_nodes
        
    return node_importance

@torch.no_grad()
def prune_non_minimal_nodes(
    inference: Any,
    sae_bank: Any,
    avg_acts: torch.Tensor,
    circuit: Circuit,
    tokens: torch.Tensor,
    pos_argmax: Optional[torch.Tensor] = None,
    threshold: float = 0.01,
    max_layer: Optional[int] = None,
    circuit_layers: Optional[Set[int]] = None,
) -> List[str]:
    """
    Identifies and removes nodes that contribute less than a threshold to faithfulness.
    
    Logic: This implementation uses **Iterative Pruning**. In each step, it identifies
    the single least important node (the one whose removal causes the smallest drop in
    faithfulness). If that drop is below the threshold, the node is removed and the
    entire circuit is re-evaluated.
    
    This avoids the "redundancy problem" where two identical nodes might both be
    seen as unimportant and removed simultaneously, destroying the circuit.
    
    Args:
        inference: The Inference instance (model.inference.Inference).
        sae_bank: The SAEBank containing the models.
        avg_acts: Tensor of average activations per latent.
        circuit: The Circuit object to evaluate.
        tokens: The input tokens tensor [batch, seq_len].
        pos_argmax: The position where each sequence peaks for the seed latent.
        threshold: The faithfulness drop threshold to consider a node minimal.
        max_layer: Optional layer limit for patching.
        
    Returns:
        List[str]: A list of removed node UUIDs.
    """
    removed_nodes = []
    
    while True:
        # 1. Evaluate current importance for all nodes
        n_eval = len(circuit.nodes)
        node_importance = evaluate_minimality(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax, max_layer=max_layer, circuit_layers=circuit_layers)
        
        # 2. Filter to find pruning candidates (excluding seed)
        candidates = []
        for node_uuid, importance in node_importance.items():
            node = circuit.nodes.get(node_uuid)
            if node and node.metadata.get("role") == "seed":
                continue
            candidates.append((node_uuid, importance))
            
        if not candidates:
            break
            
        # 3. Find the single least important node
        # We sort by importance ascending
        candidates.sort(key=lambda x: x[1])
        least_node_uuid, least_importance = candidates[0]
        
        # 4. Prune if below threshold
        if least_importance < threshold:
            # Remove the node
            circuit.nodes.pop(least_node_uuid)
            # Remove all associated edges
            circuit.edges = [
                e for e in circuit.edges 
                if e.source_uuid != least_node_uuid and e.target_uuid != least_node_uuid
            ]
            removed_nodes.append(least_node_uuid)
            
            # (Optional) Log progress
            # print(f"    [minimality] Pruned node {least_node_uuid} (importance {least_importance:.4f})")
        else:
            # Even the least important node is above threshold
            break
            
    return removed_nodes


@torch.no_grad()
def prune_non_minimal_nodes_cf(
    inference: Any,
    sae_bank: Any,
    avg_acts: torch.Tensor,
    circuit: Circuit,
    neg_tokens: torch.Tensor,
    pos_tokens: torch.Tensor,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    pos_argmax: Optional[torch.Tensor] = None,
    threshold: float = 0.05,
    circuit_layers: Optional[Set[int]] = None,
    max_candidates_per_iter: int = 32,
    max_iterations: int = 50,
) -> List[str]:
    """
    Counterfactual-faithfulness variant of iterative minimality pruning.

    Identical logic to ``prune_non_minimal_nodes`` but uses
    ``evaluate_counterfactual_faithfulness`` (cf_faith score) as the LOO
    signal instead of logit-level faithfulness.  This keeps the pruning
    objective aligned with how ``CounterfactualGradientDiscovery`` scores
    circuits: nodes are retained only if removing them meaningfully reduces
    the ability to activate the seed on contrast sequences.

    To keep the cost tractable for large circuits, each iteration only
    evaluates the ``max_candidates_per_iter`` weakest nodes (ranked by
    their stored ``attribution_score``, ascending).  Nodes with no score
    are treated as having score 0 and placed at the front of the queue.
    The loop is also capped at ``max_iterations`` rounds.

    Args:
        neg_tokens:              Contrast token sequences [B_neg, T].
        pos_tokens:              Positive-context token sequences [B_pos, T].
        seed_layer:              Layer index of the seed latent.
        seed_kind:               Kind string of the seed latent (e.g. "mlp").
        seed_latent_idx:         Index of the seed latent within its SAE.
        threshold:               Remove a node if its absence drops cf_faith
                                 by less than this value.
        circuit_layers:          Layers at which to apply CF interventions.
        max_candidates_per_iter: Max LOO evals per iteration (default 32).
                                 Prevents O(N²) cost on large circuits.
        max_iterations:          Hard cap on pruning rounds (default 50).

    Returns:
        List of removed node UUIDs.
    """
    removed_nodes: List[str] = []

    for _iter in range(max_iterations):
        base_cf_faith, _ = evaluate_counterfactual_faithfulness(
            inference, sae_bank, avg_acts, circuit,
            neg_tokens=neg_tokens,
            pos_tokens=pos_tokens,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_latent_idx=seed_latent_idx,
            pos_argmax=pos_argmax,
            circuit_layers=circuit_layers,
        )

        # Collect non-seed candidates, sorted by attribution_score ascending
        # (weakest nodes first — most likely to be prunable).
        candidates: List[tuple] = []
        for node_uuid, node in circuit.nodes.items():
            if node.metadata.get("role") == "seed":
                continue
            score = float(node.metadata.get("attribution_score") or 0.0)
            candidates.append((score, node_uuid))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        eval_candidates = [uuid for _, uuid in candidates[:max_candidates_per_iter]]

        # LOO: find the node among eval_candidates whose removal costs least
        loo_scores: Dict[str, float] = {}
        original_nodes = circuit.nodes
        for node_uuid in eval_candidates:
            circuit.nodes = {k: v for k, v in original_nodes.items() if k != node_uuid}
            loo_cf, _ = evaluate_counterfactual_faithfulness(
                inference, sae_bank, avg_acts, circuit,
                neg_tokens=neg_tokens,
                pos_tokens=pos_tokens,
                seed_layer=seed_layer,
                seed_kind=seed_kind,
                seed_latent_idx=seed_latent_idx,
                pos_argmax=pos_argmax,
                circuit_layers=circuit_layers,
            )
            loo_scores[node_uuid] = base_cf_faith - loo_cf
            circuit.nodes = original_nodes

        least_uuid = min(loo_scores, key=lambda u: loo_scores[u])
        least_drop = loo_scores[least_uuid]

        if least_drop < threshold:
            circuit.nodes.pop(least_uuid)
            circuit.edges = [
                e for e in circuit.edges
                if e.source_uuid != least_uuid and e.target_uuid != least_uuid
            ]
            removed_nodes.append(least_uuid)
        else:
            # The weakest candidate among the evaluated set is above threshold —
            # remaining nodes are strong enough to keep.
            break

    return removed_nodes


@torch.no_grad()
def prune_non_minimal_nodes_suppression(
    inference: Any,
    sae_bank: Any,
    avg_acts: torch.Tensor,
    circuit: Circuit,
    neg_tokens: torch.Tensor,
    pos_tokens: torch.Tensor,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    pos_argmax: Optional[torch.Tensor] = None,
    threshold: float = 0.05,
    circuit_layers: Optional[Set[int]] = None,
    max_candidates_per_iter: int = 32,
    max_iterations: int = 50,
) -> List[str]:
    """
    Iterative minimality pruning using positive-context suppression as the LOO signal.

    This is the suppression-oriented dual of ``prune_non_minimal_nodes_cf``:
    nodes are retained only if removing them meaningfully reduces the circuit's
    ability to suppress the seed on positive contexts.
    """
    removed_nodes: List[str] = []

    for _iter in range(max_iterations):
        _, base_sup = evaluate_counterfactual_faithfulness(
            inference, sae_bank, avg_acts, circuit,
            neg_tokens=neg_tokens,
            pos_tokens=pos_tokens,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_latent_idx=seed_latent_idx,
            pos_argmax=pos_argmax,
            circuit_layers=circuit_layers,
        )

        candidates: List[tuple] = []
        for node_uuid, node in circuit.nodes.items():
            if node.metadata.get("role") == "seed":
                continue
            score = float(node.metadata.get("attribution_score") or 0.0)
            candidates.append((score, node_uuid))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        eval_candidates = [uuid for _, uuid in candidates[:max_candidates_per_iter]]

        loo_scores: Dict[str, float] = {}
        original_nodes = circuit.nodes
        for node_uuid in eval_candidates:
            circuit.nodes = {k: v for k, v in original_nodes.items() if k != node_uuid}
            _, loo_sup = evaluate_counterfactual_faithfulness(
                inference, sae_bank, avg_acts, circuit,
                neg_tokens=neg_tokens,
                pos_tokens=pos_tokens,
                seed_layer=seed_layer,
                seed_kind=seed_kind,
                seed_latent_idx=seed_latent_idx,
                pos_argmax=pos_argmax,
                circuit_layers=circuit_layers,
            )
            loo_scores[node_uuid] = base_sup - loo_sup
            circuit.nodes = original_nodes

        least_uuid = min(loo_scores, key=lambda u: loo_scores[u])
        least_drop = loo_scores[least_uuid]

        if least_drop < threshold:
            circuit.nodes.pop(least_uuid)
            circuit.edges = [
                e for e in circuit.edges
                if e.source_uuid != least_uuid and e.target_uuid != least_uuid
            ]
            removed_nodes.append(least_uuid)
        else:
            break

    return removed_nodes
