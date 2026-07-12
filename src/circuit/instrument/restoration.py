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
        natural_dense: Optional[Dict[Site, torch.Tensor]] = None,
        alpha: float = 0.0,
    ) -> None:
        # alpha interpolates the UNRESTORED base from the floor (alpha=0,
        # classic restoration) toward the natural probe values (alpha=1) —
        # the per-round IG path of attribution_mode="ig_restoration".
        # Restored latents stay connected at every alpha.
        if alpha != 0.0 and natural_dense is None:
            raise ValueError("alpha != 0 requires natural_dense (the alpha=1 endpoint)")
        self.bank = bank
        self.substitute_sites = substitute_sites
        self.residuals = residuals
        self.site_floors = site_floors
        self.masks = masks
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.w_seed = w_seed
        self.b_seed = b_seed
        self.natural_dense = natural_dense
        self.alpha = alpha
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

        if self.alpha == 0.0:
            base = floor  # byte-identical to classic restoration
        else:
            natural = self.natural_dense[site].to(device=x.device, dtype=x.dtype)
            base = floor + self.alpha * (natural - floor)

        leaf = ((1.0 - mask) * base + mask * f_conn).detach().requires_grad_(True)
        self.leaves[site] = leaf
        code = leaf + mask * (f_conn - f_conn.detach())

        residual = self.residuals[site].to(device=x.device, dtype=x.dtype)
        return self.bank.decode(code, kind, layer_idx) + residual + (x - x.detach())


def _restoration_grad_pass(
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
    alpha: float = 0.0,
) -> Tuple[Dict[Site, torch.Tensor], float]:
    """One grad pass at interpolation point alpha (0 = restored state).

    Returns (scores per site [d_sae], pass metric -(peak - target)^2.mean()).
    Score of an unrestored latent = -d(gap loss)/d(leaf) * (natural - floor):
    the IG integrand at this alpha. Restored latents are zeroed in the output.
    """

    instrument = MaskedRestorationInstrument(
        bank, substitute_sites, residuals, site_floors, masks,
        seed_layer, seed_kind, w_seed, b_seed,
        natural_dense=natural_dense, alpha=alpha,
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
    """One grad pass at the current restored state (alpha=0).

    Returns (scores per site [d_sae], restored-state metric). Score of an
    unrestored latent = -d(gap loss)/d(leaf) * (natural - floor): predicted
    gain from restoring it. Restored latents are zeroed in the output.
    """

    return _restoration_grad_pass(
        inference, bank,
        tokens=tokens,
        substitute_sites=substitute_sites,
        residuals=residuals,
        site_floors=site_floors,
        natural_dense=natural_dense,
        masks=masks,
        seed_layer=seed_layer,
        seed_kind=seed_kind,
        w_seed=w_seed,
        b_seed=b_seed,
        pos_argmax=pos_argmax,
        target_act=target_act,
        alpha=0.0,
    )


def ig_restoration_scores(
    inference: Any,
    bank: Any,
    *,
    tokens: torch.Tensor,
    substitute_sites: set[Site],
    residuals: Dict[Site, torch.Tensor],
    site_floors: Dict[Site, torch.Tensor],
    natural_dense: Dict[Site, torch.Tensor],
    masks: Dict[Site, torch.Tensor],
    seed_layer: int,
    seed_kind: str,
    w_seed: torch.Tensor,
    b_seed: torch.Tensor,
    pos_argmax: torch.Tensor,
    target_act: float,
    ig_steps: int = 4,
) -> Tuple[Dict[Site, torch.Tensor], float]:
    """Round scorer for attribution_mode="ig_restoration": mean of the
    restoration integrand over alpha in {i/ig_steps, i=0..ig_steps-1}
    (left-Riemann, matching integrated_baseline_scores). Restored latents
    stay connected at every alpha; the mask is fixed within a round so all
    alpha samples score the same unrestored set. The returned metric is the
    alpha=0 pass (the current restored-state metric), so the loop's
    trajectory/certificate semantics match the point scorer exactly —
    ig_steps=1 degenerates to restoration_scores.
    """

    if ig_steps < 1:
        raise ValueError(f"ig_steps must be >= 1, got {ig_steps}")
    scores: Dict[Site, torch.Tensor] = {}
    metric = 0.0
    for step in range(ig_steps):
        alpha = step / ig_steps
        pass_scores, pass_metric = _restoration_grad_pass(
            inference, bank,
            tokens=tokens,
            substitute_sites=substitute_sites,
            residuals=residuals,
            site_floors=site_floors,
            natural_dense=natural_dense,
            masks=masks,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            w_seed=w_seed,
            b_seed=b_seed,
            pos_argmax=pos_argmax,
            target_act=target_act,
            alpha=alpha,
        )
        if step == 0:
            metric = pass_metric
        for site, per_latent in pass_scores.items():
            contribution = per_latent / ig_steps
            if site in scores:
                scores[site] += contribution
            else:
                scores[site] = contribution
    return scores, metric


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
    scorer: str = "point",
    ig_steps: int = 4,
    final_ig_polish: bool = False,
    polish_ig_steps: int = 10,
):
    """Full restoration-mode selection for one seed.

    Gathers anchors (floors/pins/residuals — same collectors as the
    ablation-faithfulness eval, so discovery and evaluation share one floor
    definition), runs the iterative selector with the requested per-round
    scorer ("point" = single grad pass at the restored state, classic
    restoration; "ig" = per-round integrated gradients, ig_restoration),
    and returns (positives, negatives, provenance) with FeatureID-keyed
    score dicts shaped like the one-shot extractors'.

    final_ig_polish: after the loop, one integrated_baseline_scores pass
    over the full site set re-scores the selected candidates against the
    same complete circuit (greedy rounds score each pick against a
    different mask state, so raw loop scores are not mutually rankable).
    Ranking only — the returned positives/negatives keep their loop scores,
    so membership and threshold checks are unchanged; polished scores ride
    on result.polish_scores and are applied at provenance-stamping time.
    """

    from circuit.discovery.iterative_selection import run_iterative_selection
    from circuit.instrument.ig_baseline import (
        collect_natural_codes,
        integrated_baseline_scores,
    )
    from circuit.types.feature_id import FeatureID
    from eval.ablation_faithfulness import (
        collect_site_anchors,
        resolve_site_floors,
        upstream_sites,
    )

    if scorer not in ("point", "ig"):
        raise ValueError(f"scorer must be 'point' or 'ig', got {scorer!r}")

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
        common = dict(
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
        if scorer == "ig":
            return ig_restoration_scores(inference, bank, ig_steps=ig_steps, **common)
        return restoration_scores(inference, bank, **common)

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
        f"  [Restoration/{scorer}] rounds_used={result.rounds_used} "
        f"stopped_early={result.stopped_early} "
        f"metric {result.metric_trajectory[0]:.4f} -> {result.metric_trajectory[-1]:.4f} "
        f"| selected {len(result.positives)}+{len(result.negatives)}"
    )
    sys.stdout.flush()

    if final_ig_polish and result.round_of:
        # Consistent full-circuit re-scores: one IG pass floor->natural over
        # the same sites and floors the loop used (objective the loop
        # optimised). Membership is untouched.
        polish_by_site, _, _ = integrated_baseline_scores(
            inference, bank,
            tokens=tokens,
            substitute_sites=sites,
            site_floors=floors,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            w_seed=w_seed,
            b_seed=b_seed,
            pos_argmax=pos_argmax,
            objective="gap",
            target_act=target_act,
            ig_steps=polish_ig_steps,
        )
        result.polish_scores = {
            (site, latent): float(polish_by_site[site][latent])
            for (site, latent) in result.round_of
        }
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
    """Record per-node selection rounds and loop stats on the circuit.

    When the result carries polish_scores (final_ig_polish), the polished
    full-circuit IG scores replace each node's ranking signals — both
    metadata attribution_score and the node->seed edge weight, since size
    truncation ranks by |edge weight| first — with the original loop score
    preserved under selection_score."""

    if result is None:
        return
    round_by_fid = {
        (site[0], site[1], latent): round_index
        for (site, latent), round_index in result.round_of.items()
    }
    polish_scores = getattr(result, "polish_scores", None)
    polish_by_fid = {
        (site[0], site[1], latent): value
        for (site, latent), value in (polish_scores or {}).items()
    }
    polished_uuids = {}
    for node in circuit.nodes.values():
        fid = node.feature_id
        if fid is None:
            continue
        key = (fid.layer, fid.kind, fid.index)
        round_index = round_by_fid.get(key)
        if round_index is not None:
            node.metadata["selected_round"] = round_index
        polished = polish_by_fid.get(key)
        if polished is not None:
            node.metadata["selection_score"] = node.metadata.get("attribution_score")
            node.metadata["attribution_score"] = polished
            polished_uuids[node.uuid] = polished
    if polished_uuids:
        for edge in circuit.edges:
            if edge.source_uuid in polished_uuids:
                edge.metadata["selection_weight"] = edge.weight
                edge.metadata["weight"] = polished_uuids[edge.source_uuid]
    circuit.metadata.update(
        {
            "restoration_rounds_used": result.rounds_used,
            "restoration_stopped_early": result.stopped_early,
            "restoration_metric_trajectory": list(result.metric_trajectory),
            "restoration_ig_polished": polish_scores is not None,
        }
    )


__all__ = [
    "MaskedRestorationInstrument",
    "restoration_scores",
    "ig_restoration_scores",
    "run_restoration_selection",
    "stamp_restoration_provenance",
]
