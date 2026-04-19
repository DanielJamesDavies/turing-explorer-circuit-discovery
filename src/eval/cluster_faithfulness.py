"""
Log-prob faithfulness evaluation for ClusterContrastDiscovery circuits.

Identical to the Sparse Feature Circuits reference formula:

  faithfulness = (F(C) - F(∅)) / (F(M) - F(∅))

where F is the mean log-probability of each sequence's predicted token
(argmax of the full model's output — no external labels required).

  faithfulness — evaluated on positive sequences (in-cluster, near centroid):
    answer_token[i] = argmax(full_model(pos[i]))  for each positive sequence
    F(M) = mean log_prob(answer_token) under full model
    F(C) = mean log_prob(answer_token) under circuit-patched model
    F(∅) = mean log_prob(answer_token) under fully-ablated model

  specificity — evaluated on negative sequences (out-of-cluster, near centroid):
    cluster_token   = argmax(mean full_model logits across all positive sequences)
    F_pos           = mean log_prob(cluster_token) on positives under full model
    F_neg           = mean log_prob(cluster_token) on negatives under full model
    F_neg_injected  = mean log_prob(cluster_token) after injecting activators / suppressing inhibitors
    specificity = (F_neg_injected - F_neg) / (F_pos - F_neg)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any, Dict, Set, Tuple

from model.hooks import multi_patch
from store.circuits import Circuit


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return [B, vocab] last-token logits regardless of input shape."""
    if logits.dim() == 3:
        return logits[:, -1, :]
    return logits   # already [B, vocab]


def _mean_log_prob(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    """
    Mean log-probability of token_ids[i] under logits[i].

    logits:    [B, vocab]
    token_ids: [B] long
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)   # [B, vocab]
    return log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1).mean().item()


def _zero_avg_acts(bank: Any) -> torch.Tensor:
    """Zero avg_acts for CircuitPatcher (zero-ablation baseline, same as SFC patch=None)."""
    n_components = bank.n_layer * len(bank.kinds)
    return torch.zeros(n_components, bank.d_sae, dtype=torch.float32)


class _ClusterInjectionPatcher:
    """
    Injects cluster_activator latents to their mean positive-sequence values
    and suppresses cluster_inhibitor latents to zero, preserving the SAE error term.
    """

    def __init__(
        self,
        bank: Any,
        activator_targets: Dict[Tuple[int, str], Dict[int, float]],
        inhibitor_indices: Dict[Tuple[int, str], Set[int]],
    ) -> None:
        self.bank = bank
        self.activator_targets = activator_targets
        self.inhibitor_indices = inhibitor_indices

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        has_act = (layer_idx, kind) in self.activator_targets
        has_inh = (layer_idx, kind) in self.inhibitor_indices
        if not has_act and not has_inh:
            return x

        B, T, _ = x.shape
        target_dtype = x.dtype

        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        all_latents = torch.zeros(B, T, self.bank.d_sae, device=x.device, dtype=target_dtype)
        all_latents.scatter_(-1, top_indices.long(), top_acts.to(target_dtype))

        full_recon = self.bank.decode(all_latents, kind, layer_idx)
        error = x - full_recon

        patched = all_latents.clone()
        if has_act:
            for idx, val in self.activator_targets[(layer_idx, kind)].items():
                patched[:, :, idx] = float(val)
        if has_inh:
            for idx in self.inhibitor_indices[(layer_idx, kind)]:
                patched[:, :, idx] = 0.0

        return self.bank.decode(patched, kind, layer_idx) + error


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_cluster_faithfulness(
    inference: Any,
    bank: Any,
    circuit: Circuit,
    pos_tokens: torch.Tensor,    # [N_pos, T] — in-cluster, near centroid
    neg_tokens: torch.Tensor,    # [N_neg, T] — out-of-cluster, near centroid
    eval_position: str = "last", # currently only "last" is used (SFC convention)
    batch_size: int = 8,
) -> dict:
    """
    Evaluate a ClusterContrastDiscovery circuit using the SFC faithfulness formula.

    Returns a dict with keys:
      "faithfulness"         — (F(C) - F(∅)) / (F(M) - F(∅)) on positives
      "specificity"          — (F_neg_injected - F_neg) / (F_pos - F_neg)
      "f_M"                  — log_prob under full model (positives)
      "f_C"                  — log_prob under circuit-patched model (positives)
      "f_empty"              — log_prob under fully-ablated model (positives)
      "f_neg"                — log_prob of cluster token on negatives (full model)
      "f_neg_injected"       — log_prob of cluster token on negatives (injected)
    """
    from circuit.instrument.patcher import CircuitPatcher

    avg_acts    = _zero_avg_acts(bank)
    kinds       = bank.kinds
    kind_to_idx = {k: i for i, k in enumerate(kinds)}
    N_pos, N_neg = pos_tokens.shape[0], neg_tokens.shape[0]

    def _batched(tokens: torch.Tensor, patcher=None, callback=None) -> torch.Tensor:
        """Run no-grad batched forward passes, return concatenated last-token logits [N, vocab]."""
        out = []
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start : start + batch_size]
            kwargs: dict = dict(return_activations=False, tokenize_final=False)
            if patcher is not None:
                kwargs["patcher"] = patcher
            if callback is not None:
                kwargs["activations_callback"] = callback
            inference.disable_compile()
            try:
                _, logits, _ = inference.forward(batch, **kwargs)
            finally:
                inference.enable_compile()
            out.append(_last_token_logits(logits.float()).detach())
        return torch.cat(out, dim=0)   # [N, vocab]

    # ── Parse circuit nodes ────────────────────────────────────────────────────
    activator_fids: Dict[Tuple[int, str], list] = {}
    inhibitor_fids: Dict[Tuple[int, str], Set[int]] = {}

    for node in circuit.nodes.values():
        fid  = node.feature_id
        role = node.metadata.get("role", "")
        if fid is None:
            continue
        key = (fid.layer, fid.kind)
        if role == "cluster_activator":
            activator_fids.setdefault(key, []).append(fid.index)
        elif role == "cluster_inhibitor":
            inhibitor_fids.setdefault(key, set()).add(fid.index)

    # ── Pass 1: full model on positives ───────────────────────────────────────
    # Collect activator means via hook; accumulate across batches then average.
    act_sums:   Dict[Tuple[int, str], Dict[int, float]] = {}
    act_counts: Dict[Tuple[int, str], Dict[int, int]]   = {}

    def _collect_hook(layer_idx: int, activations: tuple) -> None:
        for kind in kinds:
            key = (layer_idx, kind)
            if key not in activator_fids:
                continue
            act = activations[kind_to_idx[kind]]
            ta, ti = bank.encode(act, kind, layer_idx)
            sums   = act_sums.setdefault(key, {})
            counts = act_counts.setdefault(key, {})
            for lat_idx in activator_fids[key]:
                is_t    = (ti == lat_idx)
                t_dense = torch.where(is_t, ta, torch.zeros_like(ta)).sum(-1)  # [B, T]
                val = t_dense[:, -1].sum().item()   # sum over batch (averaged later)
                n   = t_dense.shape[0]
                sums[lat_idx]   = sums.get(lat_idx, 0.0) + val
                counts[lat_idx] = counts.get(lat_idx, 0)  + n

    pos_full_last = _batched(pos_tokens, callback=_collect_hook)   # [N_pos, vocab]

    # Finalise activator target values
    activator_targets: Dict[Tuple[int, str], Dict[int, float]] = {}
    for key, sums in act_sums.items():
        activator_targets[key] = {
            idx: sums[idx] / act_counts[key][idx]
            for idx in sums
        }

    # Per-sequence answer token and cluster representative token
    answer_tokens = pos_full_last.argmax(dim=-1)                    # [N_pos]
    cluster_token = pos_full_last.mean(dim=0).argmax().unsqueeze(0) # [1]
    f_M = _mean_log_prob(pos_full_last, answer_tokens)

    # ── Pass 2: circuit-patched on positives ──────────────────────────────────
    patcher_circuit = CircuitPatcher(bank, circuit, avg_acts)
    pos_circuit_last = _batched(pos_tokens, patcher=patcher_circuit)
    f_C = _mean_log_prob(pos_circuit_last, answer_tokens)

    # ── Pass 3: fully ablated on positives ────────────────────────────────────
    patcher_empty = CircuitPatcher(bank, None, avg_acts)
    pos_empty_last = _batched(pos_tokens, patcher=patcher_empty)
    f_empty = _mean_log_prob(pos_empty_last, answer_tokens)

    # ── Faithfulness ──────────────────────────────────────────────────────────
    denom = f_M - f_empty
    if abs(denom) < 1e-8:
        faithfulness = 1.0 if abs(f_C - f_M) < 1e-8 else 0.0
    else:
        faithfulness = float((f_C - f_empty) / denom)

    # ── Pass 4: full model on negatives ───────────────────────────────────────
    neg_full_last = _batched(neg_tokens)                             # [N_neg, vocab]
    cluster_token_neg = cluster_token.expand(N_neg)
    f_neg         = _mean_log_prob(neg_full_last, cluster_token_neg)
    f_pos_cluster = _mean_log_prob(pos_full_last, cluster_token.expand(N_pos))

    # ── Pass 5: injection on negatives ────────────────────────────────────────
    patcher_inject = _ClusterInjectionPatcher(bank, activator_targets, inhibitor_fids)
    neg_injected_last = _batched(neg_tokens, patcher=patcher_inject)
    f_neg_injected = _mean_log_prob(neg_injected_last, cluster_token_neg)

    # ── Specificity ───────────────────────────────────────────────────────────
    denom_spec = f_pos_cluster - f_neg
    if abs(denom_spec) < 1e-8:
        specificity = 1.0 if abs(f_neg_injected - f_pos_cluster) < 1e-8 else 0.0
    else:
        specificity = float((f_neg_injected - f_neg) / denom_spec)

    return {
        "faithfulness":    faithfulness,
        "specificity":     specificity,
        "f_M":             f_M,
        "f_C":             f_C,
        "f_empty":         f_empty,
        "f_neg":           f_neg,
        "f_neg_injected":  f_neg_injected,
    }
