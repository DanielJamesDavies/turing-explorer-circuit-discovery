"""Integrated-gradients attribution from the mean-ablation baseline.

Implements the "ig_baseline" attribution mode shared by the counterfactual-
and ablation-gradient discovery methods. The recipe follows Sparse Feature
Circuits (Marks et al., 2025): score each upstream latent with the
integrated-gradients estimator of its indirect effect (their IE_ig), taken
along the straight-line path in joint latent space between a patch baseline
and the clean state. Here the patch baseline is the mean-ablation floor used
by the ablation-faithfulness eval, so selection linearises exactly the
circuit-only counterfactual that evaluation measures:

    s_g = (1/N) * sum_alpha  d(metric)/d(a_g) |_{a = floor + alpha (nat - floor)}
                             * (a_g_nat - a_g_floor)

summed over batch and positions. Gradients are total derivatives carried
through substituted sites by identity passthroughs (SFC's pass-through
gradients), so every node is credited with its full mediated effect. As a
consequence the completeness identity holds per graph CUT, not for the sum
over all layers: the final cut before the seed sums to
metric(natural) - metric(floor), while upstream nodes add flow-through
credit on top. The logged certificate compares the all-site total against
the endpoint gap; totals above the target indicate mediated (multi-hop)
structure, not error.

As in SFC, natural per-site codes and reconstruction residuals are cached
from one clean pass and the interpolation replaces every upstream site's
output jointly; residuals are held at their clean values (error nodes stay
in place, mirroring both SFC's error-node treatment and the eval's
error-preservation).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

from circuit.types.feature_id import FeatureID
from model.hooks import multi_patch
from sae.dense import sparse_topk_to_dense

Site = Tuple[int, str]


class InterpolatedCodeInstrument:
    """Forward-pass hook that replaces each upstream site's output with
    decode(anchor(f_alpha)) + cached clean residual, where

        f_alpha = floor + alpha * (f_natural - floor)

    and the interpolated code is a gradient leaf. The seed's encoder
    pre-activation is captured from the arriving stream at the seed site
    (which is never substituted)."""

    def __init__(
        self,
        bank: Any,
        substitute_sites: set[Site],
        natural_codes: Dict[Site, Tuple[torch.Tensor, torch.Tensor]],
        residuals: Dict[Site, torch.Tensor],
        site_floors: Dict[Site, torch.Tensor],
        alpha: float,
        seed_layer: int,
        seed_kind: str,
        w_seed: torch.Tensor,
        b_seed: torch.Tensor,
    ) -> None:
        self.bank = bank
        self.substitute_sites = substitute_sites
        self.natural_codes = natural_codes
        self.residuals = residuals
        self.site_floors = site_floors
        self.alpha = float(alpha)
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.w_seed = w_seed
        self.b_seed = b_seed
        self.anchors: Dict[Site, torch.Tensor] = {}
        self.deltas: Dict[Site, torch.Tensor] = {}
        self.seed_pre_act: Optional[torch.Tensor] = None

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre_act = x @ w + b  # [B, T]
            return x

        site = (layer_idx, kind)
        if site not in self.substitute_sites:
            return x

        top_acts, top_indices = self.natural_codes[site]
        f_natural = sparse_topk_to_dense(top_acts, top_indices, self.bank.d_sae, dtype=torch.float32)
        floor = self.site_floors[site].to(device=f_natural.device, dtype=f_natural.dtype)
        delta = f_natural - floor
        f_alpha = (floor + self.alpha * delta).to(dtype=x.dtype)

        anchor = f_alpha.detach().requires_grad_(True)
        self.anchors[site] = anchor
        self.deltas[site] = delta

        residual = self.residuals[site].to(device=x.device, dtype=x.dtype)
        # (x - x.detach()) is numerically zero but carries an identity
        # gradient path through the site (SFC's "pass-through gradients";
        # same trick as SAEGraphInstrument). Without it, gradients die at the
        # first substituted site and every deeper anchor silently receives
        # None — collapsing total-effect attribution into direct-edge
        # attribution (the severed-gradient bug caught by
        # TestChainGradientFlow).
        return self.bank.decode(anchor, kind, layer_idx) + residual + (x - x.detach())


@torch.no_grad()
def collect_natural_codes(
    inference: Any,
    bank: Any,
    tokens: torch.Tensor,
    sites: set[Site],
) -> Tuple[Dict[Site, Tuple[torch.Tensor, torch.Tensor]], Dict[Site, torch.Tensor]]:
    """One clean pass caching, per site, the natural sparse code (top_acts,
    top_indices) and the reconstruction residual x - decode(dense)."""

    kind_to_idx = {k: i for i, k in enumerate(bank.kinds)}
    codes: Dict[Site, Tuple[torch.Tensor, torch.Tensor]] = {}
    residuals: Dict[Site, torch.Tensor] = {}

    def hook(layer_idx: int, activations: tuple) -> None:
        for kind in bank.kinds:
            if (layer_idx, kind) not in sites:
                continue
            act = activations[kind_to_idx[kind]]
            top_acts, top_indices = bank.encode(act, kind, layer_idx)
            dense = sparse_topk_to_dense(top_acts, top_indices, bank.d_sae, dtype=act.dtype)
            residuals[(layer_idx, kind)] = (act - bank.decode(dense, kind, layer_idx)).detach()
            codes[(layer_idx, kind)] = (top_acts.detach(), top_indices.detach())

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

    missing = sites - set(codes)
    if missing:
        raise RuntimeError(f"natural codes missing for sites: {sorted(missing)}")
    return codes, residuals


def integrated_baseline_scores(
    inference: Any,
    bank: Any,
    *,
    tokens: torch.Tensor,
    substitute_sites: set[Site],
    site_floors: Dict[Site, torch.Tensor],
    seed_layer: int,
    seed_kind: str,
    w_seed: torch.Tensor,
    b_seed: torch.Tensor,
    pos_argmax: torch.Tensor,
    objective: str,
    target_act: float = 0.0,
    ig_steps: int = 10,
) -> Tuple[Dict[Site, torch.Tensor], float, float]:
    """IG attribution of every upstream latent along floor -> natural.

    objective:
        "gap"   — metric = -(preact_peak - target_act)^2, mean over probes
                  (counterfactual gradient: closing the seed's activation gap)
        "drive" — metric = preact_peak, mean over probes
                  (ablation gradient: the seed's drive itself)

    Returns (scores_by_site [d_sae], metric_floor, metric_natural); the sum
    of all scores approximates metric_natural - metric_floor (IG
    completeness), which callers should log as the attribution certificate.
    """

    if objective not in ("gap", "drive"):
        raise ValueError(f"objective must be 'gap' or 'drive', got {objective!r}")

    natural_codes, residuals = collect_natural_codes(inference, bank, tokens, substitute_sites)

    def metric_from(pre_act: torch.Tensor) -> torch.Tensor:
        B = min(pre_act.shape[0], pos_argmax.shape[0])
        batch_idx = torch.arange(B, device=pre_act.device)
        pa = pos_argmax[:B].to(pre_act.device).clamp(0, pre_act.shape[1] - 1)
        peak = pre_act[:B][batch_idx, pa]
        if objective == "gap":
            return -((peak - target_act) ** 2).mean()
        return peak.mean()

    scores: Dict[Site, torch.Tensor] = {
        site: torch.zeros(bank.d_sae, dtype=torch.float64) for site in substitute_sites
    }
    metric_floor = 0.0
    metric_natural = 0.0

    inference.disable_compile()
    try:
        # N interpolation steps alpha in {0, 1/N, ..., (N-1)/N}, plus one
        # no-grad endpoint pass at alpha = 1 for the completeness log.
        for step in range(ig_steps + 1):
            alpha = step / ig_steps
            grad_enabled = step < ig_steps
            instrument = InterpolatedCodeInstrument(
                bank,
                substitute_sites,
                natural_codes,
                residuals,
                site_floors,
                alpha,
                seed_layer,
                seed_kind,
                w_seed,
                b_seed,
            )
            inference.forward(
                tokens,
                patcher=instrument,
                grad_enabled=grad_enabled,
                return_activations=False,
                tokenize_final=False,
            )
            if instrument.seed_pre_act is None:
                raise RuntimeError("seed pre-activation was not captured")
            metric = metric_from(instrument.seed_pre_act)
            if step == 0:
                metric_floor = float(metric.detach().item())
            if step == ig_steps:
                metric_natural = float(metric.detach().item())
                break

            sites_order = sorted(instrument.anchors)
            anchor_list = [instrument.anchors[site] for site in sites_order]
            grads = torch.autograd.grad(metric, anchor_list, retain_graph=False, allow_unused=True)
            for site, grad in zip(sites_order, grads):
                if grad is None:
                    continue
                contribution = (grad.to(torch.float64) * instrument.deltas[site].to(torch.float64)).sum(dim=(0, 1))
                # Accumulators live on CPU; contributions come from the GPU.
                scores[site] += (contribution / ig_steps).detach().cpu()
            del instrument
    finally:
        inference.enable_compile()

    total = float(sum(score.sum().item() for score in scores.values()))
    print(
        f"  [IGBaseline/{objective}] metric floor: {metric_floor:.4f} -> natural: {metric_natural:.4f} | "
        f"score total: {total:.4f} (completeness target {metric_natural - metric_floor:.4f})"
    )
    sys.stdout.flush()
    return scores, metric_floor, metric_natural


def extract_signed_roles(
    scores_by_site: Dict[Site, torch.Tensor],
    *,
    kinds: List[str],
    n_kinds: int,
    top_k_positive: int,
    top_k_negative: int,
    min_active_count: int,
    active_count: Optional[torch.Tensor],
    top_k_scope: str = "global",
) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
    """Split IG scores into positive-role and negative-role candidate dicts
    with the same scope semantics as the local attribution extractors
    (global ranked list, or per-(layer, kind) top-k)."""

    from pipeline.component_index import component_idx as _component_idx

    positives: List[Tuple[FeatureID, float]] = []
    negatives: List[Tuple[FeatureID, float]] = []
    for (layer, kind), site_scores in scores_by_site.items():
        values = site_scores.to(torch.float32)
        if active_count is not None:
            c_idx = _component_idx(layer, kinds.index(kind), n_kinds)
            mask = (active_count[c_idx] >= min_active_count).to(values.device)
            values = values * mask

        pos_nz = (values > 0).nonzero(as_tuple=False).squeeze(1)
        if pos_nz.numel() > 0:
            pos_vals = values[pos_nz]
            if top_k_scope == "layer_kind":
                k = min(top_k_positive, pos_nz.numel())
                top_vals, top_local = pos_vals.topk(k)
                pos_nz, pos_vals = pos_nz[top_local], top_vals
            for idx, score in zip(pos_nz.cpu().tolist(), pos_vals.cpu().tolist()):
                positives.append((FeatureID(layer=layer, kind=kind, index=idx), score))

        neg_nz = (values < 0).nonzero(as_tuple=False).squeeze(1)
        if neg_nz.numel() > 0:
            neg_vals = values[neg_nz]
            if top_k_scope == "layer_kind":
                k = min(top_k_negative, neg_nz.numel())
                top_vals, top_local = (-neg_vals).topk(k)
                neg_nz, neg_vals = neg_nz[top_local], -top_vals
            for idx, score in zip(neg_nz.cpu().tolist(), neg_vals.cpu().tolist()):
                negatives.append((FeatureID(layer=layer, kind=kind, index=idx), score))

    if top_k_scope == "global":
        positives.sort(key=lambda item: item[1], reverse=True)
        negatives.sort(key=lambda item: item[1])
        positives = positives[:top_k_positive]
        negatives = negatives[:top_k_negative]

    return dict(positives), dict(negatives)


__all__ = [
    "InterpolatedCodeInstrument",
    "collect_natural_codes",
    "integrated_baseline_scores",
    "extract_signed_roles",
]
