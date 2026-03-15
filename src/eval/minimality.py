import torch
from typing import Dict, Any, List, Optional
from store.circuits import Circuit
from eval.faithfulness import evaluate_faithfulness

@torch.no_grad()
def evaluate_minimality(
    inference: Any, 
    sae_bank: Any, 
    avg_acts: torch.Tensor, 
    circuit: Circuit, 
    tokens: torch.Tensor,
    pos_argmax: Optional[torch.Tensor] = None
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
        
    Returns:
        Dict[str, float]: A mapping from node UUIDs to their importance (faithfulness drop).
    """
    # 1. Base faithfulness with the complete circuit
    base_faithfulness = evaluate_faithfulness(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax)
    
    # 2. Leave-One-Out Ablation for each node
    node_importance = {}
    original_nodes = circuit.nodes
    
    for node_uuid in original_nodes:
        # Temporarily remove node
        # Create a temporary node list without the current node
        modified_nodes = {k: v for k, v in original_nodes.items() if k != node_uuid}
        circuit.nodes = modified_nodes
        
        # Calculate faithfulness without this node
        new_faithfulness = evaluate_faithfulness(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax)
        
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
    threshold: float = 0.01
) -> List[str]:
    """
    Identifies and removes nodes that contribute less than a threshold to faithfulness.
    
    Args:
        inference: The Inference instance (model.inference.Inference).
        sae_bank: The SAEBank containing the models.
        avg_acts: Tensor of average activations per latent.
        circuit: The Circuit object to evaluate.
        tokens: The input tokens tensor [batch, seq_len].
        pos_argmax: The position where each sequence peaks for the seed latent.
        threshold: The faithfulness drop threshold to consider a node minimal.
        
    Returns:
        List[str]: A list of removed node UUIDs.
    """
    node_importance = evaluate_minimality(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax)
    removed_nodes = []
    
    for node_uuid, importance in node_importance.items():
        # Never prune the seed node if it's marked in metadata
        node = circuit.nodes.get(node_uuid)
        if node and node.metadata.get("role") == "seed":
            continue

        if importance < threshold:
            # 1. Remove the node
            circuit.nodes.pop(node_uuid)
            # 2. Remove all associated edges
            circuit.edges = [
                e for e in circuit.edges 
                if e.source_uuid != node_uuid and e.target_uuid != node_uuid
            ]
            removed_nodes.append(node_uuid)
            
    return removed_nodes
