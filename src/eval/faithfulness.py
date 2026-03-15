import torch
import torch.nn.functional as F
from typing import Optional, Any, Iterable
from store.circuits import Circuit
from circuit.patcher import CircuitPatcher

def _calculate_faithfulness_score(
    original_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    pos_argmax: Optional[torch.Tensor] = None
) -> float:
    """
    Calculates the normalized faithfulness score from logit tensors.
    Logic: 1 - (MSE(intervened, original) / MSE(baseline, original))
    """
    # Logits are [batch, seq_len, vocab_size] (if seq_len > 1) or [batch, 1, vocab_size]
    # Extract logits at specific positions if pos_argmax is provided
    if pos_argmax is not None:
        batch_indices = torch.arange(original_logits.size(0), device=original_logits.device)
        orig = original_logits[batch_indices, pos_argmax]
        interv = intervened_logits[batch_indices, pos_argmax]
        base = baseline_logits[batch_indices, pos_argmax]
    else:
        # Fallback to the last token position
        orig = original_logits[:, -1, :]
        interv = intervened_logits[:, -1, :]
        base = baseline_logits[:, -1, :]
    
    # Calculate MSEs relative to original logits
    mse_intervened = F.mse_loss(interv, orig)
    mse_base = F.mse_loss(base, orig)
    
    # Faithfulness calculation
    if mse_base < 1e-9:
        return 1.0 if mse_intervened < 1e-9 else 0.0
        
    score = 1.0 - (mse_intervened / mse_base)
    return float(score.item())

@torch.no_grad()
def evaluate_faithfulness(
    inference: Any, 
    sae_bank: Any, 
    avg_acts: torch.Tensor, 
    circuit: Optional[Circuit], 
    tokens: torch.Tensor,
    pos_argmax: Optional[torch.Tensor] = None
) -> float:
    """
    Calculates the faithfulness of a circuit on a specific sequence.
    
    Faithfulness = 1 - (MSE(L_circuit, L_original) / MSE(L_baseline, L_original))
    
    Args:
        inference: The Inference instance (model.inference.Inference).
        sae_bank: The SAEBank containing the models.
        avg_acts: Tensor of average activations per latent [n_layers, d_sae].
        circuit: The Circuit object to evaluate (if None, evaluates baseline).
        tokens: The input tokens tensor [batch, seq_len].
        pos_argmax: The position where each sequence peaks for the seed latent.
        
    Returns:
        float: The faithfulness score (0 to 1 typical range).
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
    
    # 2. Circuit Pass (Intervened Logits)
    patcher = CircuitPatcher(sae_bank, circuit, avg_acts, pos_argmax=pos_argmax)
    _, circuit_logits, _ = inference.forward(
        tokens,
        num_gen=1,
        tokenize_final=False,
        patcher=patcher,
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
    
    return _calculate_faithfulness_score(original_logits, circuit_logits, baseline_logits, pos_argmax)


@torch.no_grad()
def evaluate_kind_local_faithfulness(
    inference: Any,
    sae_bank: Any,
    avg_acts: torch.Tensor,
    circuit: Optional[Circuit],
    tokens: torch.Tensor,
    target_kinds: Iterable[str],
    pos_argmax: Optional[torch.Tensor] = None,
) -> float:
    """
    Calculates faithfulness restricted to selected SAE kinds.

    Only target_kinds are patched. All other kinds pass through unchanged.
    This yields a "kind-local" score: how well the circuit recovers the
    contribution of specific kinds (e.g. only MLP latents).
    """
    use_all = pos_argmax is not None
    kinds = tuple(target_kinds)

    _, original_logits, _ = inference.forward(
        tokens,
        num_gen=1,
        tokenize_final=False,
        return_activations=False,
        all_logits=use_all,
    )

    patcher = CircuitPatcher(
        sae_bank, circuit, avg_acts, pos_argmax=pos_argmax, patch_kinds=kinds
    )
    _, circuit_logits, _ = inference.forward(
        tokens,
        num_gen=1,
        tokenize_final=False,
        patcher=patcher,
        return_activations=False,
        all_logits=use_all,
    )

    baseline_patcher = CircuitPatcher(
        sae_bank, None, avg_acts, pos_argmax=pos_argmax, patch_kinds=kinds
    )
    _, baseline_logits, _ = inference.forward(
        tokens,
        num_gen=1,
        tokenize_final=False,
        patcher=baseline_patcher,
        return_activations=False,
        all_logits=use_all,
    )

    return _calculate_faithfulness_score(
        original_logits, circuit_logits, baseline_logits, pos_argmax
    )
