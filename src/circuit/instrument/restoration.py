"""Masked-restoration instrument and scorer for iterative selection.

State semantics: at each upstream site the code is

    c = (1 - m) * floor  +  m * f_connected(x)

where m is a binary per-latent mask of restored latents. Unrestored latents
sit at the mean-ablation floor as GRADIENT LEAVES (their score = how much
restoring them would help); restored latents carry their LIVE encoder
values, connected to the stream — free-execution semantics, matching the
free ablation-faithfulness eval. Connected restoration is load-bearing: if
restored latents were pinned to cached constants, gradients to their
parents would vanish (the severed-gradient failure mode) and iteration
could never recruit chains.

Implementation of c keeping both gradient roles:

    leaf = ((1 - m) * floor + m * f_conn).detach().requires_grad_(True)
    c    = leaf + m * (f_conn - f_conn.detach())

so d(c)/d(leaf) = I (every latent's restoration gain is readable from
leaf.grad) and d(c)/d(f_conn) = m (parent gradients flow only through
restored latents). Output = decode(c) + cached clean residual + identity
passthrough (x - x.detach()); SFC-style pass-through gradients as in
SAEGraphInstrument.
"""

from __future__ import annotations

import sys

from typing import Any, Dict, Optional, Tuple

import torch

from model.hooks import multi_patch
from sae.dense import sparse_topk_to_dense

Site = Tuple[int, str]


class MaskedRestorationInstrument:
    def __init__(
        self,
        bank: Any,
        substitute_sites: set[Site],
        residuals: Dict[Site, torch.Tensor],
        site_floors: Dict[Site, torch.Tensor],
        masks: Dict[Site, torch.Tensor],  # bool [d_sae] per site
        seed_layer: int,
        seed_kind: str,
        w_seed: torch.Tensor,
        b_seed: torch.Tensor,
    ) -> None:
        self.bank = bank
        self.substitute_sites = substitute_sites
        self.residuals = residuals
        self.site_floors = site_floors
        self.masks = masks
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.w_seed = w_seed
        self.b_seed = b_seed
        self.leaves: Dict[Site, torch.Tensor] = {}
        self.seed_pre_act: Optional[torch.Tensor] = None

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre_act = x @ w + b
            return x

        site = (layer_idx, kind)
        if site not in self.substitute_sites:
            return x

        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        f_conn = sparse_topk_to_dense(top_acts, top_indices, self.bank.d_sae, dtype=x.dtype)
        floor = self.site_floors[site].to(device=x.device, dtype=x.dtype)
        mask = self.masks[site].to(device=x.device, dtype=x.dtype)

        leaf = ((1.0 - mask) * floor + mask * f_conn).detach().requires_grad_(True)
        self.leaves[site] = leaf
        code = leaf + mask * (f_conn - f_conn.detach())

        residual = self.residuals[site].to(device=x.device, dtype=x.dtype)
        return self.bank.decode(code, kind, layer_idx) + residual + (x - x.detach())


def restoration_scores(
    inference: Any,
    bank: Any,
    *,
    tokens: torch.Tensor,
    substitute_sites: set[Site],
    residuals: Dict[Site, torch.Tensor],
    site_floors: Dict[Site, torch.Tensor],
    natural_dense: Dict[Site, torch.Tensor],  # [d_sae] natural probe values
    masks: Dict[Site, torch.Tensor],
    seed_layer: int,
    seed_kind: str,
    w_seed: torch.Tensor,
    b_seed: torch.Tensor,
    pos_argmax: torch.Tensor,
    target_act: float,
) -> Tuple[Dict[Site, torch.Tensor], float]:
    """One grad pass at the current restored state.

    Returns (scores per site [d_sae], restored-state metric). Score of an
    unrestored latent = -d(gap loss)/d(leaf) * (natural - floor): predicted
    gain from restoring it. Restored latents are zeroed in the output.
    """

    instrument = MaskedRestorationInstrument(
        bank, substitute_sites, residuals, site_floors, masks,
        seed_layer, seed_kind, w_seed, b_seed,
    )
    inference.disable_compile()
    try:
        inference.forward(
            tokens,
            patcher=instrument,
            grad_enabled=True,
            return_activations=False,
            tokenize_final=False,
        )
    finally:
        inference.enable_compile()
    if instrument.seed_pre_act is None:
        raise RuntimeError("seed pre-activation was not captured")

    pre = instrument.seed_pre_act
    B = min(pre.shape[0], pos_argmax.shape[0])
    idx = torch.arange(B, device=pre.device)
    pa = pos_argmax[:B].to(pre.device).clamp(0, pre.shape[1] - 1)
    peak = pre[:B][idx, pa]
    metric = -((peak - target_act) ** 2).mean()

    sites_order = sorted(instrument.leaves)
    leaves = [instrument.leaves[site] for site in sites_order]
    grads = torch.autograd.grad(metric, leaves, allow_unused=True)

    scores: Dict[Site, torch.Tensor] = {}
    for site, grad in zip(sites_order, grads):
        if grad is None:
            scores[site] = torch.zeros(bank.d_sae, dtype=torch.float32)
            continue
        floor = site_floors[site].to(torch.float32).cpu()
        natural = natural_dense[site].to(torch.float32).cpu()
        delta = natural - floor
        per_latent = grad.to(torch.float32).sum(dim=(0, 1)).cpu() * delta
        per_latent = per_latent * (~masks[site].cpu().bool()).float()  # unrestored only
        scores[site] = per_latent
    return scores, float(metric.detach().item())


def run_restoration_selection(
    inference: Any,
    bank: Any,
    *,
    tokens: torch.Tensor,
    pos_argmax: torch.Tensor,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    target_act: float,
    rounds: int,
    per_round_k: int,
    certificate_tol: float,
    allow_negative: bool = True,
    loader: Any = None,
):
    """Full restoration-mode selection for one seed.

    Gathers anchors (floors/pins/residuals — same collectors as the
    ablation-faithfulness eval, so discovery and evaluation share one floor
    definition), runs the iterative selector with a restoration_scores
    closure, and returns (positives, negatives, provenance) with
    FeatureID-keyed score dicts shaped like the one-shot extractors'.
    """

    from circuit.discovery.iterative_selection import run_iterative_selection
    from circuit.instrument.ig_baseline import collect_natural_codes
    from circuit.types.feature_id import FeatureID
    from eval.ablation_faithfulness import (
        collect_site_anchors,
        resolve_site_floors,
        upstream_sites,
    )

    sites = upstream_sites(bank, seed_layer, seed_kind)
    if not sites:
        return {}, {}, None
    floors, pins = collect_site_anchors(inference, bank, tokens, sites, pos_argmax)
    # Shared floor knob: pins (natural probe values) always stay posctx.
    floors = resolve_site_floors(inference, bank, sites, posctx_means=floors, loader=loader)
    _, residuals = collect_natural_codes(inference, bank, tokens, sites)

    sae = bank.saes[seed_kind][seed_layer]
    w_seed = sae.encoder.weight[seed_latent_idx].detach()
    b_seed = sae._get_bias_eff()[seed_latent_idx].detach()
    masks = {site: torch.zeros(bank.d_sae, dtype=torch.bool) for site in sites}

    def score_fn(current_masks):
        return restoration_scores(
            inference, bank,
            tokens=tokens,
            substitute_sites=sites,
            residuals=residuals,
            site_floors=floors,
            natural_dense=pins,
            masks=current_masks,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            w_seed=w_seed,
            b_seed=b_seed,
            pos_argmax=pos_argmax,
            target_act=target_act,
        )

    # Relative certificate: the gap metric is -(gap)^2, so stopping within
    # tol of the target activation means gap^2 <= (tol * target)^2. An
    # absolute tolerance would be ~free for strong seeds and near-impossible
    # to trip meaningfully for weak ones.
    absolute_tol = (certificate_tol * max(abs(target_act), 1e-6)) ** 2
    result = run_iterative_selection(
        score_fn,
        masks=masks,
        rounds=rounds,
        per_round_k=per_round_k,
        certificate_tol=absolute_tol,
        target_metric=0.0,  # gap objective: loss reaches 0 when the seed is restored
        allow_negative=allow_negative,
    )
    print(
        f"  [Restoration] rounds_used={result.rounds_used} "
        f"stopped_early={result.stopped_early} "
        f"metric {result.metric_trajectory[0]:.4f} -> {result.metric_trajectory[-1]:.4f} "
        f"| selected {len(result.positives)}+{len(result.negatives)}"
    )
    sys.stdout.flush()
    positives = {
        FeatureID(site[0], site[1], latent): value
        for (site, latent), value in result.positives.items()
    }
    negatives = {
        FeatureID(site[0], site[1], latent): value
        for (site, latent), value in result.negatives.items()
    }
    return positives, negatives, result


def stamp_restoration_provenance(circuit: Any, result: Any) -> None:
    """Record per-node selection rounds and loop stats on the circuit."""

    if result is None:
        return
    round_by_fid = {
        (site[0], site[1], latent): round_index
        for (site, latent), round_index in result.round_of.items()
    }
    for node in circuit.nodes.values():
        fid = node.feature_id
        if fid is None:
            continue
        round_index = round_by_fid.get((fid.layer, fid.kind, fid.index))
        if round_index is not None:
            node.metadata["selected_round"] = round_index
    circuit.metadata.update(
        {
            "restoration_rounds_used": result.rounds_used,
            "restoration_stopped_early": result.stopped_early,
            "restoration_metric_trajectory": list(result.metric_trajectory),
        }
    )


__all__ = [
    "MaskedRestorationInstrument",
    "restoration_scores",
    "run_restoration_selection",
    "stamp_restoration_provenance",
]
