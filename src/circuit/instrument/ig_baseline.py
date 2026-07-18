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

from .position_aware import PositionAwareSpec

Site = Tuple[int, str]


class InterpolatedCodeInstrument:
    """Forward-pass hook that replaces each upstream site's output with
    decode(anchor(f_alpha)) + cached clean residual, interpolating each site's
    latents between two endpoints: the natural code cached from the tokens'
    own clean pass (sparse, per-position) and a dense position-independent
    vector (``site_floors``). ``path`` sets the direction:

        "to_natural"   (default) — f_alpha = floor + alpha * (f_natural - floor)
                       The original ig_baseline path: mean-ablation floor at
                       alpha=0, the clean state at alpha=1.
        "from_natural" — f_alpha = f_natural + alpha * (target - f_natural)
                       The contrastive path: the tokens' own (negctx) state at
                       alpha=0, the injected target at alpha=1. Here
                       ``site_floors`` carries the posctx TARGET values, not a
                       floor — the same [d_sae] vector per site that the
                       counterfactual-faithfulness eval injects, so alpha=1
                       reproduces the eval's intervened state (with the
                       negctx residuals held in place).

    The interpolated code is a gradient leaf, and ``deltas`` is always
    end - start, so grad x delta needs no sign handling by direction. The
    seed's encoder pre-activation is captured from the arriving stream at the
    seed site (which is never substituted)."""

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
        path: str = "to_natural",
    ) -> None:
        if path not in ("to_natural", "from_natural"):
            raise ValueError(f"path must be 'to_natural' or 'from_natural', got {path!r}")
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
        self.path = path
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
        dense = self.site_floors[site].to(device=f_natural.device, dtype=f_natural.dtype)
        # The dense [d_sae] endpoint broadcasts over [B, T, d_sae]. delta is
        # always end - start, so grad x delta is direction-agnostic downstream.
        start, end = (dense, f_natural) if self.path == "to_natural" else (f_natural, dense)
        delta = end - start
        f_alpha = (start + self.alpha * delta).to(dtype=x.dtype)

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
    position_aware: Optional["PositionAwareSpec"] = None,
    path: str = "to_natural",
    batch_size: Optional[int] = None,
) -> Tuple[Dict[Site, torch.Tensor], float, float]:
    """IG attribution of every upstream latent along a straight latent-space path.

    objective:
        "gap"   — metric = -(preact_peak - target_act)^2, mean over probes
                  (counterfactual gradient: closing the seed's activation gap)
        "drive" — metric = preact_peak, mean over probes
                  (ablation gradient: the seed's drive itself)

    path:
        "to_natural"   (default) — floor -> the tokens' clean state: the
                       original ig_baseline recipe. ``site_floors`` is the
                       mean-ablation floor.
        "from_natural" — the tokens' clean state -> a dense target: the
                       contrastive_ig recipe, run on NEGCTX tokens with
                       ``site_floors`` carrying the posctx target values the
                       counterfactual-faithfulness eval injects. alpha=1 then
                       reproduces the eval's intervened negctx state.

    batch_size:
        Sequences per grad-enabled forward (VRAM bound). ``tokens`` may carry
        more sequences than this; they are processed in chunks and merged with
        B_chunk/B_total weights, which reproduces the single-pass result
        exactly (per-sequence-mean metrics). None (default) = one pass over
        all of ``tokens`` — the historical behaviour.

    position_aware:
        When given, the position axis is UNIONED over the seed's causal prefix
        instead of collapsed with .sum(dim=(0, 1)): the per-position IG
        attribution is accumulated across steps, then each prefix position
        selects its own top-N and the union is taken. The returned
        scores_by_site keeps its [d_sae] shape but is SPARSE — only union
        members carry a score, everything else is 0 — so callers take the
        nonzeros as membership rather than applying a top-m truncation.
        Same instrument, same floor baseline, same objective; only the
        reduction over positions changes. Note: this accumulates a
        [B, T, d_sae] tensor per site on CPU (memory ~ B*T*d_sae*4 bytes per
        site), so it is materially heavier than the classic collapsed path.

    Returns (scores_by_site [d_sae], metric_start, metric_end) — the metric at
    alpha=0 and alpha=1 respectively (for "to_natural" these are the historical
    metric_floor / metric_natural). The sum of all scores approximates
    metric_end - metric_start (IG completeness), which callers should log as
    the attribution certificate. (The completeness certificate only holds on
    the classic reduction — the position-aware path selects a subset, so its
    scores do not sum to the gap.)
    """

    if objective not in ("gap", "drive"):
        raise ValueError(f"objective must be 'gap' or 'drive', got {objective!r}")
    if path not in ("to_natural", "from_natural"):
        raise ValueError(f"path must be 'to_natural' or 'from_natural', got {path!r}")

    # Sequence count vs batch size are separate ideas: `tokens` carries the
    # sequences that INFORM the attribution; `batch_size` bounds how many go
    # through one grad-enabled forward (VRAM). Chunks are merged with weights
    # B_chunk / B_total, which makes the merged result identical to a single
    # pass over all sequences (metrics are per-sequence means; gradients of a
    # chunk mean carry 1/B_chunk, so the reweighting converts them to the
    # global mean). None = single pass (historical behaviour).
    B_total = int(tokens.shape[0])
    bs = B_total if batch_size is None else max(1, int(batch_size))

    def run_chunk(tokens_chunk, argmax_chunk):
        natural_codes, residuals = collect_natural_codes(
            inference, bank, tokens_chunk, substitute_sites
        )

        def metric_from(pre_act: torch.Tensor) -> torch.Tensor:
            B = min(pre_act.shape[0], argmax_chunk.shape[0])
            batch_idx = torch.arange(B, device=pre_act.device)
            pa = argmax_chunk[:B].to(pre_act.device).clamp(0, pre_act.shape[1] - 1)
            peak = pre_act[:B][batch_idx, pa]
            if objective == "gap":
                return -((peak - target_act) ** 2).mean()
            return peak.mean()

        c_scores: Dict[Site, torch.Tensor] = {
            site: torch.zeros(bank.d_sae, dtype=torch.float64) for site in substitute_sites
        }
        # Position-aware: accumulate the full [B, T, d_sae] IG attribution per
        # site (CPU, float32) so the position axis survives to the selection.
        c_pa_accum: Dict[Site, torch.Tensor] = {}
        m_start = 0.0
        m_end = 0.0

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
                path=path,
            )
            inference.forward(
                tokens_chunk,
                patcher=instrument,
                grad_enabled=grad_enabled,
                return_activations=False,
                tokenize_final=False,
            )
            if instrument.seed_pre_act is None:
                raise RuntimeError("seed pre-activation was not captured")
            metric = metric_from(instrument.seed_pre_act)
            if step == 0:
                m_start = float(metric.detach().item())
            if step == ig_steps:
                m_end = float(metric.detach().item())
                break

            sites_order = sorted(instrument.anchors)
            anchor_list = [instrument.anchors[site] for site in sites_order]
            grads = torch.autograd.grad(metric, anchor_list, retain_graph=False, allow_unused=True)
            for site, grad in zip(sites_order, grads):
                if grad is None:
                    continue
                # fp32 product, not fp64. The terms come from a bf16 model
                # (~1e-2 relative noise), so a double-precision multiply
                # preserves noise; and the PA path immediately casts to fp32
                # anyway. The [B, T, d_sae] fp64 product was the arm's largest
                # transient (168MB/site) — a fossil from the collapsed-only
                # era, when scores were [d_sae] and fp64 was free. The
                # certificate keeps its integrity where it matters: the
                # collapsed accumulators below stay fp64 (in-place += promotes
                # the fp32 chunk sums), which is where the
                # millions-of-cancelling-terms accumulation actually happens.
                per_pos = grad.to(torch.float32) * instrument.deltas[site].to(torch.float32)
                if position_aware is not None:
                    # Keep the position axis: accumulate [B, T, d_sae] ON GPU
                    # and transfer once per chunk (below). Per-alpha .cpu()
                    # transfers were ~10x the bytes and dominated the pass —
                    # profiled at ~97s of .cpu() tottime on a 23-site seed
                    # (1,840 transfers x 84MB ~= 154GB over PCIe).
                    contrib = (per_pos / ig_steps).detach().to(torch.float32)
                    if site in c_pa_accum:
                        c_pa_accum[site] += contrib
                    else:
                        c_pa_accum[site] = contrib
                # Collapsed accumulators live on CPU; contributions are small.
                c_scores[site] += (per_pos.sum(dim=(0, 1)) / ig_steps).detach().cpu()
            del instrument
        # One transfer per site per chunk; selection stays CPU-side.
        return c_scores, {s: t.cpu() for s, t in c_pa_accum.items()}, m_start, m_end

    scores: Dict[Site, torch.Tensor] = {
        site: torch.zeros(bank.d_sae, dtype=torch.float64) for site in substitute_sites
    }
    # Position-aware members merged across chunks by max-|score| (the same rule
    # select_position_aware uses across positions and sequences within a chunk).
    pa_members: Dict[Site, Dict[int, float]] = {site: {} for site in substitute_sites}
    metric_floor = 0.0
    metric_natural = 0.0

    inference.disable_compile()
    try:
        for start in range(0, B_total, bs):
            tokens_chunk = tokens[start:start + bs]
            argmax_chunk = pos_argmax[start:start + bs]
            w = tokens_chunk.shape[0] / B_total
            c_scores, c_pa_accum, m_start, m_end = run_chunk(tokens_chunk, argmax_chunk)
            metric_floor += w * m_start
            metric_natural += w * m_end
            for site in substitute_sites:
                scores[site] += w * c_scores[site]
            if position_aware is not None:
                # The chunk-mean gradient carries 1/B_chunk; rescale to the
                # global mean so thresholds and cross-chunk max-|score| are
                # on one scale, then merge unions. abs_pctl resolves its cut
                # from the pooled |attr| of ALL sites in this pass.
                scaled = {site: accum * w for site, accum in c_pa_accum.items()}
                spec = position_aware.resolved_for(scaled.values())
                for site, attr in scaled.items():
                    merged = pa_members[site]
                    for latent, value in spec.select_from(attr).items():
                        if latent not in merged or abs(value) > abs(merged[latent]):
                            merged[latent] = value
    finally:
        inference.enable_compile()

    total = float(sum(score.sum().item() for score in scores.values()))
    ends = ("floor", "natural") if path == "to_natural" else ("natural", "target")
    n_chunks = (B_total + bs - 1) // bs if B_total else 0
    print(
        f"  [IGBaseline/{objective}] metric {ends[0]}: {metric_floor:.4f} -> {ends[1]}: {metric_natural:.4f} | "
        f"score total: {total:.4f} (completeness target {metric_natural - metric_floor:.4f})"
        + (f" | {B_total} seqs in {n_chunks} chunks" if n_chunks > 1 else "")
    )
    sys.stdout.flush()

    if position_aware is not None:
        # Union over the seed's causal prefix instead of the collapsed ranking:
        # each prefix position selects its own top-N; scores go sparse so callers
        # read membership off the nonzeros (no top-m truncation).
        n_members = 0
        for site in substitute_sites:
            sparse = torch.zeros(bank.d_sae, dtype=torch.float64)
            for latent, value in pa_members[site].items():
                sparse[latent] = value
                n_members += 1
            scores[site] = sparse
        print(f"  [IGBaseline/{objective}] position-aware union: {n_members} members "
              f"across {len(substitute_sites)} sites")
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
