"""
Ablation (circuit-only) faithfulness: the SFC-style faithfulness notion
(Marks et al., 2025) transported to the seed endpoint.

Asks: "with everything EXCEPT the discovered circuit mean-ablated upstream,
does the seed still fire on its positive contexts?"

Protocol, aligned with sparse feature circuits:

  - Non-circuit latents at each upstream site are set to their MEAN value
    over the evaluation distribution (the seed's positive probe batch), not
    zero. Zero ablation destroys the residual-stream reconstruction at every
    layer and reduces the model to SAE error terms; SFC explicitly
    mean-ablates ("set to their average value over data from D") and notes
    that deleting residual-stream signal "severely disrupts the model".
  - SAE reconstruction error is always preserved (error nodes stay in place).
  - The score is normalised against the EMPTY circuit, as in SFC:

        score = (a_circuit_only − a_empty) / (a_posctx − a_empty)

    where a_empty is the seed's activation with everything upstream
    mean-ablated. Near 1.0: the discovered nodes recover the seed's
    activation relative to the mean-ablation floor.

"Upstream" means every (layer, kind) site that feeds the seed's site: all
kinds at strictly lower layers, plus earlier kinds within the seed's own
layer (attn -> mlp -> resid). Sites where the circuit has no nodes are fully
mean-ablated. The seed's own site is never touched (it is the measurement
point).

``ablation="zero"`` retains the harsher zero-ablation variant for
comparison; its scores conflate circuit quality with the severity of the
off-distribution shift.

Pinned mode (``pin_values`` given): kept latents are forced to their
clean-run probe-position values (phi_cf injection semantics) instead of
being re-encoded from the ablated stream. Free-vs-pinned decomposes the
in-situ failure: pinned isolates node selection quality, free additionally
requires causal closure (kept latents' own upstream chains surviving).

The anchors (site means, a_empty, a_posctx) are circuit-independent, so
callers sweeping many circuits per seed should compute them once via
``collect_site_means`` / ``measure_seed_activation`` and pass them in; the
per-circuit cost is then a single forward pass.
"""

import sys
import torch
from typing import Any, Dict, Optional, Set, Tuple

from store.circuits import Circuit
from model.hooks import multi_patch
from sae.dense import sparse_topk_to_dense, target_latent_activations


class CircuitOnlyPatcher:
    """
    A forward-pass hook that ablates every non-circuit latent at each
    upstream (layer, kind) site — to its distribution mean (``site_means``
    given) or to zero — preserving SAE reconstruction error, and captures
    the seed latent's response at the probe position.
    """

    def __init__(
        self,
        bank: Any,
        keep_indices: Dict[Tuple[int, str], Set[int]],
        in_scope: Set[Tuple[int, str]],
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
        pos_argmax: Optional[torch.Tensor] = None,
        site_means: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
        pin_values: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    ) -> None:
        self.bank = bank
        self.keep_indices = keep_indices
        self.in_scope = in_scope
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.seed_latent_idx = seed_latent_idx
        self.pos_argmax = pos_argmax.detach().cpu() if pos_argmax is not None else None
        self.site_means = site_means
        self.pin_values = pin_values
        self.captured_activation: Optional[float] = None

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        target_dtype = x.dtype

        # ── Capture seed latent activation ────────────────────────────────
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
            seed_dense = target_latent_activations(top_acts, top_indices, self.seed_latent_idx)  # [B, T]
            if self.pos_argmax is not None:
                actual_B = min(B, self.pos_argmax.shape[0])
                pa = self.pos_argmax[:actual_B].to(x.device).clamp(0, T - 1)
                val = seed_dense[:actual_B][torch.arange(actual_B, device=x.device), pa].mean().item()
            else:
                val = seed_dense.mean().item()
            self.captured_activation = val

        # ── Circuit-only ablation ─────────────────────────────────────────
        if (layer_idx, kind) not in self.in_scope:
            return x

        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        all_latents = sparse_topk_to_dense(top_acts, top_indices, self.bank.d_sae, dtype=target_dtype)

        # Preserve the SAE error term from the unmodified encoding
        full_recon = self.bank.decode(all_latents, kind, layer_idx)
        error = x - full_recon

        if self.site_means is not None:
            mean_vector = self.site_means[(layer_idx, kind)].to(
                device=all_latents.device, dtype=all_latents.dtype
            )
            patched = mean_vector.expand_as(all_latents).clone()
        else:
            patched = torch.zeros_like(all_latents)
        keep = self.keep_indices.get((layer_idx, kind))
        if keep:
            keep_tensor = torch.tensor(sorted(keep), device=all_latents.device, dtype=torch.long)
            if self.pin_values is not None:
                pins = self.pin_values[(layer_idx, kind)].to(
                    device=all_latents.device, dtype=all_latents.dtype
                )
                patched[:, :, keep_tensor] = pins[keep_tensor]
            else:
                patched[:, :, keep_tensor] = all_latents[:, :, keep_tensor]

        return self.bank.decode(patched, kind, layer_idx) + error


@torch.no_grad()
def measure_seed_activation(
    inference: Any,
    sae_bank: Any,
    tokens: torch.Tensor,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    argmax: Optional[torch.Tensor] = None,
) -> float:
    """Seed latent's mean activation at probe positions, no intervention."""

    kind_to_idx = {k: i for i, k in enumerate(sae_bank.kinds)}
    values = []

    def hook(layer_idx: int, activations: tuple) -> None:
        if layer_idx != seed_layer:
            return
        act = activations[kind_to_idx[seed_kind]]
        ta, ti = sae_bank.encode(act, seed_kind, layer_idx)
        s_dense = target_latent_activations(ta, ti, seed_latent_idx)  # [B, T]
        Bx = s_dense.shape[0]
        if argmax is not None:
            actual_B = min(Bx, argmax.shape[0])
            pa = argmax[:actual_B].to(s_dense.device).clamp(0, s_dense.shape[1] - 1)
            val = s_dense[:actual_B][torch.arange(actual_B, device=s_dense.device), pa].mean().item()
        else:
            val = s_dense.mean().item()
        values.append(val)

    inference.disable_compile()
    try:
        inference.forward(
            tokens,
            activations_callback=hook,
            return_activations=False,
            tokenize_final=False,
        )
    finally:
        inference.enable_compile()
    return float(values[0]) if values else 0.0


# Floors and site anchors moved to eval.floors (method-agnostic home for
# every mean-ablation consumer); re-exported here for compatibility.
from eval.floors import (  # noqa: F401  (re-exports)
    _GLOBAL_FLOOR_CACHE,
    _farthest_point_sample,
    collect_diverse_site_floors,
    collect_global_site_floors,
    collect_site_anchors,
    collect_site_means,
    resolve_site_floors,
)


def upstream_sites(sae_bank: Any, seed_layer: int, seed_kind: str) -> Set[Tuple[int, str]]:
    """Every (layer, kind) site feeding the seed's site: all kinds at lower
    layers plus earlier kinds within the seed's layer (attn -> mlp -> resid)."""

    kinds = list(sae_bank.kinds)
    seed_kind_idx = kinds.index(seed_kind)
    sites: Set[Tuple[int, str]] = set()
    for layer in range(seed_layer):
        for kind in kinds:
            sites.add((layer, kind))
    for kind_idx in range(seed_kind_idx):
        sites.add((seed_layer, kinds[kind_idx]))
    return sites


@torch.no_grad()
def circuit_only_activation(
    inference: Any,
    sae_bank: Any,
    keep_indices: Dict[Tuple[int, str], Set[int]],
    in_scope: Set[Tuple[int, str]],
    pos_tokens: torch.Tensor,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    pos_argmax: Optional[torch.Tensor] = None,
    site_means: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    pin_values: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
) -> float:
    """One forward pass with everything outside ``keep_indices`` ablated."""

    patcher = CircuitOnlyPatcher(
        bank=sae_bank,
        keep_indices=keep_indices,
        in_scope=in_scope,
        seed_layer=seed_layer,
        seed_kind=seed_kind,
        seed_latent_idx=seed_latent_idx,
        pos_argmax=pos_argmax,
        site_means=site_means,
        pin_values=pin_values,
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
    return float(patcher.captured_activation or 0.0)


@torch.no_grad()
def evaluate_ablation_faithfulness(
    inference: Any,
    sae_bank: Any,
    avg_acts: torch.Tensor,  # kept for API symmetry; not used internally
    circuit: Circuit,
    pos_tokens: torch.Tensor,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    pos_argmax: Optional[torch.Tensor] = None,
    ablation: str = "mean",
    site_means: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    pin_values: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    a_posctx: Optional[float] = None,
    a_empty: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Circuit-only sufficiency on positive contexts, SFC-normalised.

    Returns:
        (ablation_faithfulness, a_circuit_only)

        ablation_faithfulness:
            (a_circuit_only - a_empty) / (a_posctx - a_empty)
            Near 1.0 → the circuit recovers the seed's activation relative
            to the mean-ablation (empty-circuit) floor.

    The anchors are circuit-independent: pass site_means (from
    ``collect_site_means``), a_posctx (from ``measure_seed_activation``) and
    a_empty (from ``circuit_only_activation`` with empty keep_indices) to
    evaluate each circuit with a single forward pass.
    """

    if ablation not in ("mean", "zero"):
        raise ValueError(f"ablation must be 'mean' or 'zero', got {ablation!r}")

    keep_indices: Dict[Tuple[int, str], Set[int]] = {}
    for node in circuit.nodes.values():
        fid = node.feature_id
        if fid is None or node.metadata.get("role") == "seed":
            continue
        key = (fid.layer, fid.kind)
        if key == (seed_layer, seed_kind):
            continue
        keep_indices.setdefault(key, set()).add(fid.index)

    in_scope = upstream_sites(sae_bank, seed_layer, seed_kind)

    if ablation == "mean" and site_means is None:
        site_means = collect_site_means(inference, sae_bank, pos_tokens, in_scope)
    if ablation == "zero":
        site_means = None

    if a_posctx is None:
        a_posctx = measure_seed_activation(
            inference, sae_bank, pos_tokens, seed_layer, seed_kind, seed_latent_idx, pos_argmax
        )
    if a_empty is None:
        a_empty = circuit_only_activation(
            inference,
            sae_bank,
            {},
            in_scope,
            pos_tokens,
            seed_layer,
            seed_kind,
            seed_latent_idx,
            pos_argmax,
            site_means,
        )

    a_circuit_only = circuit_only_activation(
        inference,
        sae_bank,
        keep_indices,
        in_scope,
        pos_tokens,
        seed_layer,
        seed_kind,
        seed_latent_idx,
        pos_argmax,
        site_means,
        pin_values,
    )

    denom = a_posctx - a_empty
    if abs(denom) < 1e-9:
        score = 1.0 if abs(a_circuit_only - a_posctx) < 1e-9 else 0.0
    else:
        score = (a_circuit_only - a_empty) / denom
    variant = f"{ablation}+pinned" if pin_values is not None else ablation
    print(
        f"  [AblFaithfulness/{variant}] a_posctx: {a_posctx:.4f} | a_empty: {a_empty:.4f} | "
        f"a_circuit_only: {a_circuit_only:.4f} | abl={score:.4f}"
    )
    sys.stdout.flush()
    return float(score), float(a_circuit_only)
