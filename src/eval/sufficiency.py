import torch
import torch.nn.functional as F
from typing import Optional, Any
from store.circuits import Circuit
from circuit.patcher import CircuitPatcher

@torch.no_grad()
def evaluate_sufficiency(
    inference: Any, 
    sae_bank: Any, 
    avg_acts: torch.Tensor, 
    circuit: Circuit, 
    tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    pos_argmax: Optional[torch.Tensor] = None
) -> float:
    """
    Measures if the circuit captures the "full story" for a specific prompt.
    Comparing the log-probability of target tokens in the circuit-model vs the full model.
    
    Sufficiency = exp(log_prob_circuit - log_prob_original)
    
    Args:
        inference: The Inference instance (model.inference.Inference).
        sae_bank: The SAEBank containing the models.
        avg_acts: Tensor of average activations per latent.
        circuit: The Circuit object to evaluate.
        tokens: The input tokens tensor [batch, seq_len].
        target_tokens: The expected next token(s) tensor [batch, seq_len].
        pos_argmax: The position where each sequence peaks for the seed latent.
        
    Returns:
        float: The sufficiency score (0 to 1 typical range).
    """
    # Use all_logits=True if pos_argmax is provided
    use_all = pos_argmax is not None

    # 1. Full Model Pass (Original Logprobs)
    # Inference.forward returns (tokens, logits, activations)
    _, original_logits, _ = inference.forward(
        tokens, 
        num_gen=1, 
        tokenize_final=False, 
        return_activations=False,
        all_logits=use_all
    )

    batch_indices = torch.arange(original_logits.size(0), device=original_logits.device)
    
    # Extract logprobs at specific position
    if pos_argmax is not None:
        # Extract logits at pos_argmax [B, V]
        orig_logits_at_pos = original_logits[batch_indices, pos_argmax]
        # target_tokens matches tokens [B, T], so target for pos t is tokens[t+1]
        # BUT sufficiency.py is passed target_tokens explicitly.
        # Assuming target_tokens is already shifted tokens[1:] or similar.
        # Wait, let's check how target_tokens is prepared in ProbeDataset.
        target_tokens_at_pos = target_tokens[batch_indices, pos_argmax]
    else:
        orig_logits_at_pos = original_logits[:, -1, :]
        target_tokens_at_pos = target_tokens[:, -1]

    log_probs_orig = F.log_softmax(orig_logits_at_pos, dim=-1)
    target_logprobs_orig = log_probs_orig.gather(-1, target_tokens_at_pos.unsqueeze(-1)).squeeze(-1)
    
    # 2. Circuit Pass (Intervened Logprobs)
    patcher = CircuitPatcher(sae_bank, circuit, avg_acts, pos_argmax=pos_argmax)
    _, circuit_logits, _ = inference.forward(
        tokens,
        num_gen=1,
        tokenize_final=False,
        patcher=patcher,
        return_activations=False,
        all_logits=use_all
    )

    if pos_argmax is not None:
        circ_logits_at_pos = circuit_logits[batch_indices, pos_argmax]
    else:
        circ_logits_at_pos = circuit_logits[:, -1, :]

    log_probs_circ = F.log_softmax(circ_logits_at_pos, dim=-1)
    target_logprobs_circ = log_probs_circ.gather(-1, target_tokens_at_pos.unsqueeze(-1)).squeeze(-1)
    
    # Sufficiency is how much of the original log-probability we recover
    sufficiency = torch.exp(target_logprobs_circ - target_logprobs_orig)
    
    return float(sufficiency.mean().item())
