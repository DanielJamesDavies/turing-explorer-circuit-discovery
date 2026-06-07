"""
Posctx node presence and circuit sufficiency evaluation.

Two complementary measurements are made in two no-grad forward passes:

─── Pass 1: Natural posctx (presence detection + a_posctx) ────────────────────
Runs the model normally on posctx sequences.

  Per-node presence:
    node_presence_pct_activators : % of activator nodes that fired on ≥1 sequence
    node_presence_rate_mean      : mean firing rate for activators
    node_absence_pct_inhibitors  : % of inhibitor nodes silent on all sequences
    node_inhibitor_rate_mean     : mean firing rate for inhibitors (expected ≈ 0)

  Seed baseline:
    a_posctx = seed's mean activation across [B, T] (for the sufficiency ratio)

─── Pass 2: Circuit-isolated posctx (sufficiency) ─────────────────────────────
At every (layer, kind) the SAE activations are masked so only activator latents
survive; everything else — including inhibitors — is zeroed before decoding.
The SAE reconstruction error (x − decode(natural)) is preserved to avoid
introducing artificial drift.

  a_circuit_only = seed's mean activation under circuit isolation

  posctx_circuit_sufficiency = a_circuit_only / a_posctx

  Near 1.0 → the discovered activators alone are sufficient to drive the seed
              on posctx; there are no important upstream drivers outside the
              circuit.
  Near 0.0 → the seed's posctx firing relies heavily on latents not captured
              by the circuit.

Keys for a role group are omitted entirely when no nodes of that role exist.
"""

import sys
import torch
from typing import Any, Dict, List, Optional, Set, Tuple

from store.circuits import Circuit
from model.hooks import multi_patch

_INELIGIBLE_KINDS: Set[str] = {"logit", "token", "error"}


# ── Circuit-isolation patcher ──────────────────────────────────────────────────

class _CircuitSufficiencyPatcher:
    """
    A forward-pass hook for the circuit-isolation pass.

    At every (layer, kind):
      1. Encode the residual stream into SAE sparse activations.
      2. Build a dense latent tensor from top-k.
      3. Compute error = x - decode(natural_latents)  (always preserved).
      4. Zero out ALL latents except those in ``activator_map`` at this location.
      5. Return decode(masked_latents) + error.

    At (seed_layer, seed_kind):
      Encode x (which already reflects all upstream masking), capture the
      seed latent's mean activation, and return x UNCHANGED so the residual
      stream continues naturally from the measurement point.
    """

    def __init__(
        self,
        bank: Any,
        activator_map: Dict[Tuple[int, str], List[int]],
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
    ) -> None:
        self.bank = bank
        self.activator_map = activator_map
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.seed_latent_idx = seed_latent_idx
        self.captured_activation: Optional[float] = None

    def __call__(self, model: Any) -> Any:
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        target_dtype = x.dtype

        # ── Capture seed activation (before any modification at this layer) ─
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
            is_seed = (top_indices == self.seed_latent_idx)
            seed_dense = torch.where(is_seed, top_acts, torch.zeros_like(top_acts)).sum(-1)
            self.captured_activation = seed_dense.mean().item()
            # Return x unchanged — the seed's layer is only measured, not masked
            return x

        # ── Circuit-isolation masking ──────────────────────────────────────
        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        all_latents = torch.zeros(B, T, self.bank.d_sae, device=x.device, dtype=target_dtype)
        all_latents.scatter_(-1, top_indices.long(), top_acts.to(target_dtype))

        # Preserve the reconstruction error from the unmodified natural encoding
        full_recon = self.bank.decode(all_latents, kind, layer_idx)
        error = x - full_recon

        # Build keep-mask: 1 only for circuit activators at this (layer, kind)
        keep_mask = torch.zeros(self.bank.d_sae, device=x.device, dtype=target_dtype)
        for latent_idx in self.activator_map.get((layer_idx, kind), []):
            keep_mask[latent_idx] = 1.0

        masked_latents = all_latents * keep_mask  # broadcast [B, T, d_sae]
        return self.bank.decode(masked_latents, kind, layer_idx) + error


# ── Public evaluation function ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_node_presence(
    inference: Any,
    sae_bank: Any,
    circuit: Circuit,
    pos_tokens: torch.Tensor,  # [B, T]
) -> Dict[str, float]:
    """
    Two-pass evaluation of posctx node presence and circuit sufficiency.

    Pass 1 — natural forward:
        Records per-node firing rates and the seed's baseline activation.

    Pass 2 — circuit-isolated forward:
        Masks SAE latents at every (layer, kind) to keep only activators;
        measures what fraction of the seed's natural activation survives.

    Args:
        inference:   Inference instance.
        sae_bank:    SAEBank for encoding/decoding.
        circuit:     Accepted circuit with ``role`` metadata on nodes.
        pos_tokens:  ``[B, T]`` positive-context token tensor.

    Returns:
        Dict with up to five float keys (see module docstring).
        Returns ``{}`` if there are no eligible activator/inhibitor nodes or
        if ``pos_tokens`` is empty.
    """
    if pos_tokens.shape[0] == 0:
        return {}

    B = pos_tokens.shape[0]
    kinds = sae_bank.kinds
    kind_to_idx: Dict[str, int] = {k: i for i, k in enumerate(kinds)}

    # ── Collect nodes by role ─────────────────────────────────────────────
    activator_map: Dict[Tuple[int, str], List[int]] = {}
    inhibitor_map: Dict[Tuple[int, str], List[int]] = {}
    seed_fid = None

    for node in circuit.nodes.values():
        fid = node.feature_id
        if fid is None:
            continue
        role = node.metadata.get("role", "")
        if role == "seed":
            seed_fid = fid
            continue
        if fid.kind in _INELIGIBLE_KINDS:
            continue
        key = (fid.layer, fid.kind)
        if role in ("counterfactual_activator", "ablation_support"):
            activator_map.setdefault(key, []).append(fid.index)
        elif role == "counterfactual_inhibitor":
            inhibitor_map.setdefault(key, []).append(fid.index)

    if not activator_map and not inhibitor_map:
        return {}

    n_act = sum(len(v) for v in activator_map.values())
    n_inh = sum(len(v) for v in inhibitor_map.values())
    print(
        f"  [NodePresence] {n_act} activators, {n_inh} inhibitors | "
        f"B={B} posctx sequences"
    )
    sys.stdout.flush()

    all_node_keys: Set[Tuple[int, str]] = set(activator_map) | set(inhibitor_map)
    fired_seqs: Dict[Tuple[int, str, int], int] = {}
    a_posctx_buf: List[float] = []

    # ── Pass 1: natural forward ───────────────────────────────────────────
    def presence_hook(layer_idx: int, activations: tuple) -> None:
        # Capture seed baseline activation
        if seed_fid is not None and layer_idx == seed_fid.layer and seed_fid.kind in kind_to_idx:
            act = activations[kind_to_idx[seed_fid.kind]]
            ta, ti = sae_bank.encode(act, seed_fid.kind, layer_idx)
            is_s = (ti == seed_fid.index)
            s_dense = torch.where(is_s, ta, torch.zeros_like(ta)).sum(-1)  # [B, T]
            a_posctx_buf.append(s_dense.mean().item())

        # Per-node presence detection
        for kind in kinds:
            key = (layer_idx, kind)
            if key not in all_node_keys:
                continue
            act = activations[kind_to_idx[kind]]
            top_acts, top_indices = sae_bank.encode(act, kind, layer_idx)
            node_list = activator_map.get(key, []) + inhibitor_map.get(key, [])
            for latent_idx in node_list:
                is_present = (top_indices == latent_idx)
                fired_anywhere = (top_acts * is_present).sum(dim=(1, 2)) > 0
                fired_seqs[(layer_idx, kind, latent_idx)] = int(fired_anywhere.sum().item())

    inference.disable_compile()
    try:
        inference.forward(
            pos_tokens,
            activations_callback=presence_hook,
            return_activations=False,
            tokenize_final=False,
        )
    finally:
        inference.enable_compile()

    a_posctx = a_posctx_buf[0] if a_posctx_buf else 0.0

    # ── Aggregate presence results ────────────────────────────────────────
    result: Dict[str, float] = {}

    if activator_map:
        rates: List[float] = []
        for key, latents in activator_map.items():
            layer_idx, kind = key
            for latent_idx in latents:
                n = fired_seqs.get((layer_idx, kind, latent_idx), 0)
                rates.append(n / B)
        mean_rate = sum(rates) / len(rates)
        presence_pct = 100.0 * sum(r > 0 for r in rates) / len(rates)
        result["node_presence_pct_activators"] = round(presence_pct, 4)
        result["node_presence_rate_mean"] = round(mean_rate, 4)
        print(
            f"  [NodePresence] activators: presence={presence_pct:.1f}%  "
            f"mean_rate={mean_rate:.3f}"
        )
        sys.stdout.flush()

    if inhibitor_map:
        inh_rates: List[float] = []
        for key, latents in inhibitor_map.items():
            layer_idx, kind = key
            for latent_idx in latents:
                n = fired_seqs.get((layer_idx, kind, latent_idx), 0)
                inh_rates.append(n / B)
        mean_inh_rate = sum(inh_rates) / len(inh_rates)
        absence_pct = 100.0 * sum(r == 0.0 for r in inh_rates) / len(inh_rates)
        result["node_absence_pct_inhibitors"] = round(absence_pct, 4)
        result["node_inhibitor_rate_mean"] = round(mean_inh_rate, 4)
        print(
            f"  [NodePresence] inhibitors: absence={absence_pct:.1f}%  "
            f"mean_rate={mean_inh_rate:.3f}"
        )
        sys.stdout.flush()

    # ── Pass 2: circuit-isolated forward ─────────────────────────────────
    if seed_fid is None or abs(a_posctx) < 1e-9 or not activator_map:
        return result

    patcher = _CircuitSufficiencyPatcher(
        bank=sae_bank,
        activator_map=activator_map,
        seed_layer=seed_fid.layer,
        seed_kind=seed_fid.kind,
        seed_latent_idx=seed_fid.index,
    )

    inference.disable_compile()
    try:
        inference.forward(
            pos_tokens,
            patcher=patcher,
            return_activations=False,
            tokenize_final=False,
        )
    finally:
        inference.enable_compile()

    a_circuit_only = patcher.captured_activation or 0.0
    sufficiency = a_circuit_only / a_posctx

    print(
        f"  [NodePresence] circuit sufficiency: a_posctx={a_posctx:.4f}  "
        f"a_circuit_only={a_circuit_only:.4f}  ratio={sufficiency:.4f}"
    )
    sys.stdout.flush()

    result["posctx_circuit_sufficiency"] = round(sufficiency, 4)
    return result
