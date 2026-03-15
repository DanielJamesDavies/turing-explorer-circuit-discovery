import torch
from typing import Any, Optional
from store.circuits import Circuit
from circuit.patcher import CircuitPatcher
from eval.faithfulness import _calculate_faithfulness_score

@torch.no_grad()
def evaluate_completeness(
    inference: Any, 
    sae_bank: Any, 
    avg_acts: torch.Tensor, 
    circuit: Circuit, 
    tokens: torch.Tensor,
    pos_argmax: Optional[torch.Tensor] = None
) -> float:
    """
    Measures if the circuit is complete by checking the performance of its complement.
    A circuit is complete if the rest of the model (the complement) cannot perform the task.
    
    Completeness = 1.0 - Faithfulness(Complement)
    
    Args:
        inference: The Inference instance.
        sae_bank: The SAEBank containing the models.
        avg_acts: Tensor of average activations per latent.
        circuit: The Circuit object to evaluate.
        tokens: The input tokens tensor [batch, seq_len].
        pos_argmax: The position where each sequence peaks for the seed latent.
        
    Returns:
        float: The completeness score (0 to 1 typical range, 1.0 is ideal).
    """
    # Use all_logits=True if pos_argmax is provided
    use_all = pos_argmax is not None

    # 1. Full Model Pass (Original Logits)
    _, original_logits, _ = inference.forward(
        tokens, 
        num_gen=1, 
        tokenize_final=False, 
        return_activations=False,
        all_logits=use_all
    )
    
    # 2. Complement Pass (Inverted Intervention)
    # We ablate ONLY the circuit nodes and keep everything else live
    complement_patcher = CircuitPatcher(sae_bank, circuit, avg_acts, inverse=True, pos_argmax=pos_argmax)
    _, complement_logits, _ = inference.forward(
        tokens,
        num_gen=1,
        tokenize_final=False,
        patcher=complement_patcher,
        return_activations=False,
        all_logits=use_all
    )

    # 3. Baseline Pass (Total Ablation)
    baseline_patcher = CircuitPatcher(sae_bank, None, avg_acts, pos_argmax=pos_argmax)
    _, baseline_logits, _ = inference.forward(
        tokens,
        num_gen=1,
        tokenize_final=False,
        patcher=baseline_patcher,
        return_activations=False,
        all_logits=use_all
    )
    
    # 4. Calculate Faithfulness of the Complement
    f_complement = _calculate_faithfulness_score(original_logits, complement_logits, baseline_logits, pos_argmax)
    
    # 5. Completeness is the inverse of the complement's faithfulness
    # (i.e. how much the model's performance was DESTROYED by removing only the circuit)
    return 1.0 - f_complement
