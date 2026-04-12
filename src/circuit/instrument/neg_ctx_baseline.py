import torch
from typing import Any, Tuple


def compute_neg_ctx_means(
    inference: Any,
    sae_bank: Any,
    neg_tokens: torch.Tensor,
    max_neg: int = 8,
) -> torch.Tensor:
    """
    Compute per-latent mean activations over negative-context sequences.

    Returns a ``[n_components, d_sae]`` tensor suitable for use as ``avg_acts``
    in ``CircuitPatcher``.  Each entry is the mean activation of that latent
    over the supplied ``neg_tokens``, averaged across all token positions and
    all firing occurrences.  Latents that never fire in the neg sequences
    receive a value of 0.

    This produces a principled ablation baseline for faithfulness evaluation:
    ablated features are replaced with their typical *background* level rather
    than zero.  The model therefore still processes the probe sequence in a
    realistic way, and the faithfulness denominator measures only the
    concept-specific elevation present in positive contexts — not the total
    difference between the model and a fully-zeroed state.

    If ``neg_tokens`` is empty or ``max_neg == 0``, a zero tensor is returned
    (equivalent to the old zero-ablation baseline).

    Args:
        inference:   ``Inference`` instance for running the model.
        sae_bank:    ``SAEBank`` containing the encoders.
        neg_tokens:  ``[N_neg, seq_len]`` hard-negative token sequences.
        max_neg:     Maximum number of neg sequences to use (default 8).

    Returns:
        ``[n_components, d_sae]`` float32 tensor on ``sae_bank.device``.
    """
    n_components = sae_bank.n_layer * len(sae_bank.kinds)
    d_sae = sae_bank.d_sae

    if neg_tokens.shape[0] == 0 or max_neg <= 0:
        return torch.zeros(
            (n_components, d_sae), dtype=torch.float32, device=sae_bank.device
        )

    n_neg = min(max_neg, neg_tokens.shape[0])
    neg_tokens = neg_tokens[:n_neg]

    # Accumulate on CPU to avoid device-synchronisation overhead per token.
    sum_acts   = torch.zeros((n_components, d_sae), dtype=torch.float32)
    hit_counts = torch.zeros((n_components, d_sae), dtype=torch.float32)

    def _callback(layer_idx: int, activations: Tuple[torch.Tensor, ...]) -> None:
        for kind_idx, kind in enumerate(sae_bank.kinds):
            top_acts, top_indices = sae_bank.encode(
                activations[kind_idx], kind, layer_idx
            )
            comp_idx  = layer_idx * len(sae_bank.kinds) + kind_idx
            flat_idx  = top_indices.long().reshape(-1).cpu()
            flat_acts = top_acts.float().reshape(-1).cpu()
            ones      = torch.ones(flat_acts.shape[0], dtype=torch.float32)
            sum_acts[comp_idx].scatter_add_(0, flat_idx, flat_acts)
            hit_counts[comp_idx].scatter_add_(0, flat_idx, ones)

    inference.forward(
        neg_tokens,
        activations_callback=_callback,
        return_activations=False,
        tokenize_final=False,
    )

    # Mean over firing occurrences; latents that never fired stay at 0.
    neg_avg = sum_acts / hit_counts.clamp(min=1.0)
    neg_avg[hit_counts == 0] = 0.0
    return neg_avg.to(sae_bank.device)
