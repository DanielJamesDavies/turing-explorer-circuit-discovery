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

from dataclasses import replace as dataclass_replace
from typing import Any, Dict, List, Optional, Tuple

import torch

from model.hooks import multi_patch
from sae.dense import sparse_topk_to_dense

from .position_aware import (
    PositionAwareSpec,
    pooled_abs_threshold,
    select_position_aware_values,
)

Site = Tuple[int, str]
Candidate = Tuple[Site, int]


class MaskedRestorationInstrument:
    """Two state semantics, selected by ``mode``:

    - ``"floor_restore"`` (classic restoration, default): unrestored latents
      sit at the mean-ablation floor as constants, restored latents carry
      their LIVE connected encoder values. Runs on posctx.

    - ``"target_inject"`` (restoration_negctx): the COMPLEMENT masking, run
      on negctx tokens. Unrestored latents carry their live connected values
      on the (modified) negctx stream — the on-manifold contrast state, the
      same alpha=0 the ig_negctx path uses — while restored latents are
      PINNED to their posctx target values from ``inject_targets`` (broadcast
      [d_sae] constants: exactly the counterfactual-faithfulness eval's
      injection, so the loop's final state IS the eval's intervened state).
      Severing restored latents from their parents is the matched semantics
      here, not a failure mode: the eval's intervention clamps them too.
      Chain recruitment happens DOWNSTREAM instead — injecting an early
      activator shifts the stream, downstream unrestored latents re-encode
      toward their posctx-like values through the connected (1-m) term, and
      the next round's gradient sees the moved state. Per-position deltas
      (target - live value) land in ``self.deltas`` for the scorer; alpha
      interpolation is not supported (point scorer only).
    """

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
        mode: str = "floor_restore",
        inject_targets: Optional[Dict[Site, torch.Tensor]] = None,
    ) -> None:
        # alpha interpolates the UNRESTORED base from the floor (alpha=0,
        # classic restoration) toward the natural probe values (alpha=1) —
        # the per-round IG path of attribution_mode="ig_restoration".
        # Restored latents stay connected at every alpha.
        if mode not in ("floor_restore", "target_inject"):
            raise ValueError(f"mode must be 'floor_restore' or 'target_inject', got {mode!r}")
        if mode == "target_inject":
            if inject_targets is None:
                raise ValueError("target_inject mode requires inject_targets")
            if alpha != 0.0:
                raise ValueError("target_inject mode supports the point scorer only (alpha=0)")
        elif alpha != 0.0 and natural_dense is None:
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
        self.mode = mode
        # target_inject: the PER-LATENT value each restored latent is pinned
        # to — posctx target for restored activators, 0 for restored
        # inhibitors (mirroring the cf eval's "inject activators at posctx,
        # suppress inhibitors to 0"). The driver mutates this across rounds as
        # latents are selected with a role. Unrestored latents ignore it.
        self.inject_targets = inject_targets
        self.leaves: Dict[Site, torch.Tensor] = {}
        self.live: Dict[Site, torch.Tensor] = {}  # target_inject: f_conn [B,T,d_sae]
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
        mask = self.masks[site].to(device=x.device, dtype=x.dtype)

        if self.mode == "target_inject":
            # Unrestored = live connected negctx values; restored = pinned to
            # their per-latent role value (posctx for activators, 0 for
            # inhibitors). d(code)/d(leaf) = I (every latent scoreable),
            # d(code)/d(f_conn) = (1-m) (parents reachable through the live,
            # unrestored latents only — the injected ones are clamped, as in
            # the eval). ``live`` is stashed so the scorer can weigh BOTH the
            # activator move (posctx - live) and the inhibitor move (0 - live)
            # per candidate and keep only the helping one.
            target = self.inject_targets[site].to(device=x.device, dtype=x.dtype)
            leaf = ((1.0 - mask) * f_conn + mask * target).detach().requires_grad_(True)
            self.leaves[site] = leaf
            self.live[site] = f_conn.detach()
            code = leaf + (1.0 - mask) * (f_conn - f_conn.detach())
        else:
            floor = self.site_floors[site].to(device=x.device, dtype=x.dtype)
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
    keep_positions: bool = False,
    factor_by_site: Optional[Dict[Site, torch.Tensor]] = None,
    mode: str = "floor_restore",
    inject_targets: Optional[Dict[Site, torch.Tensor]] = None,
    posctx_targets: Optional[Dict[Site, torch.Tensor]] = None,
    inject_mode: str = "both_sign",
    objective: str = "gap",
) -> Tuple[Dict[Site, torch.Tensor], Optional[Dict[Site, torch.Tensor]], float]:
    """One grad pass at interpolation point alpha (0 = restored state).

    Returns (scores per site [d_sae], per-position scores or None, pass
    metric -(peak - target)^2.mean()). Score of an unrestored latent =
    -d(gap loss)/d(leaf) * (natural - floor): the IG integrand at this alpha.
    Restored latents are zeroed in both outputs.

    ``keep_positions``: also return the UNCOLLAPSED per-position scores
    ``[B, T, d_sae]`` per site (CPU float32) — the position-aware round
    selection reads these instead of the ``.sum(dim=(0, 1))`` collapse.

    ``mode="target_inject"`` (restoration_negctx): tokens are negctx, restored
    latents pinned to their per-latent role value in ``inject_targets``
    (posctx for activators, 0 for inhibitors). Each UNRESTORED candidate is
    scored by the benefit of its best HELPING move, weighing both:
        b_act = sum_pos g * (posctx - live)   (inject at posctx: raise)
        b_inh = sum_pos g * (0      - live)   (inject at 0: remove)
    Role = activator if b_act >= b_inh else inhibitor; the returned score is
    the winning benefit signed by role (+ activator / - inhibitor), and
    latents whose best benefit is <= 0 (no helping move — clamping them to
    EITHER value pushes the seed away from firing) score 0 and are never
    selected. This is the fix for injecting seed-suppressing latents: nothing
    anti-closure is ever admitted, and the loop's certified state (activators
    at posctx, inhibitors at 0) IS the cf eval's Score-1 intervention.
    ``posctx_targets`` supplies the activator-move endpoint; ``objective``
    selects the backward scalar ("gap" | "drive"); the metric stays the gap
    form so the certificate is objective-invariant.
    """

    instrument = MaskedRestorationInstrument(
        bank, substitute_sites, residuals, site_floors, masks,
        seed_layer, seed_kind, w_seed, b_seed,
        natural_dense=natural_dense, alpha=alpha,
        mode=mode, inject_targets=inject_targets,
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
    if mode == "target_inject" and objective == "drive":
        backward_scalar = peak.mean()
    else:
        backward_scalar = metric

    sites_order = sorted(instrument.leaves)
    leaves = [instrument.leaves[site] for site in sites_order]
    grads = torch.autograd.grad(backward_scalar, leaves, allow_unused=True)

    scores: Dict[Site, torch.Tensor] = {}
    pos_scores: Optional[Dict[Site, torch.Tensor]] = {} if keep_positions else None
    for site, grad in zip(sites_order, grads):
        if grad is None:
            scores[site] = torch.zeros(bank.d_sae, dtype=torch.float32)
            continue
        if mode == "target_inject":
            # Three modes (see restoration_negctx_mode). All compute on the GPU
            # and move only the collapsed [d_sae] vector off. The returned score
            # SIGN encodes the delivered role (>=0 activator / <0 inhibitor),
            # which the driver reads to set each restored latent's pin value.
            g = grad.to(torch.float32)                              # [B,T,d_sae]
            live = instrument.live[site].to(g.device, torch.float32)  # [B,T,d_sae]
            posctx = posctx_targets[site].to(g.device, torch.float32)  # [d_sae]
            unrestored = (~masks[site].to(g.device).bool()).to(torch.float32)  # [d_sae]
            effect = g * (posctx.view(1, 1, -1) - live)            # posctx-injection effect
            if inject_mode == "posctx":
                # Original: sign & magnitude from the signed posctx effect
                # (roles by that sign; the driver pins EVERYONE to posctx).
                per_pos = effect * unrestored.view(1, 1, -1)
            else:
                g_sum = g.sum(dim=(0, 1))                           # [d_sae]
                gl_sum = (g * live).sum(dim=(0, 1))                 # sum_pos g*live
                b_act = g_sum * posctx - gl_sum                     # sum g*(posctx-live)
                b_inh = -gl_sum                                     # sum g*(0-live)
                role_act = b_act >= b_inh                           # [d_sae] bool
                sign_role = torch.where(role_act, torch.ones_like(posctx),
                                        -torch.ones_like(posctx))   # [d_sae]
                if inject_mode == "directional":
                    # Helping moves only: score = winning benefit, signed by
                    # role; latents that help in neither direction score 0.
                    helping = torch.maximum(b_act, b_inh) > 0
                    zeros = torch.zeros_like(posctx)
                    delta_role = torch.where(role_act, posctx, zeros)
                    sel = (sign_role * unrestored * helping.to(g.dtype)).view(1, 1, -1)
                    per_pos = g * (delta_role.view(1, 1, -1) - live) * sel
                else:  # "both_sign"
                    # Both-sign membership by |posctx effect| (as "posctx", for
                    # free0); sign = role-directional (for the inject value).
                    per_pos = effect.abs() * (sign_role * unrestored).view(1, 1, -1)
            scores[site] = per_pos.sum(dim=(0, 1)).cpu()
            if keep_positions:
                pos_scores[site] = per_pos  # GPU [B, T, d_sae]
            continue
        if keep_positions:
            # GPU-RESIDENT path (position-aware round selection). The
            # per-position integrand [B, T, d_sae] stays on the GPU through
            # accumulation / quantile / selection instead of being moved to the
            # CPU here. Profiled 2026-07-18: the old grad.cpu() moved ~42MB x
            # n_sites PER grad pass (~645GB PCIe over a deep arm) and forced the
            # whole PA round pipeline onto the CPU. Collapsed `scores` are
            # derived from the same GPU tensor and moved as a small [d_sae]
            # vector. PA arms are already nondeterministic at the abs_pctl
            # boundary (unseeded subsample), so the GPU-vs-CPU reduction-order
            # jitter this introduces is within their existing noise; the non-PA
            # branch below is left byte-identical to preserve determinism for
            # the arms whose selection reads the collapsed scores.
            g = grad.to(torch.float32)
            # factor = (natural - floor) * unrestored is CONSTANT across a
            # round's alpha x chunk grad passes (delta is constant all
            # discovery; the mask is fixed within a round). _round_scores
            # precomputes it once per round and passes it here — the fallback
            # below keeps this callable standalone. Profiled 2026-07-18: the
            # per-pass reconversion was ~30s of Tensor.to() self-time on a deep
            # arm. Result is bit-identical to the fallback (same ops, same
            # device/dtype) so it stays within the PA arm's existing noise.
            if factor_by_site is not None:
                factor = factor_by_site[site].to(g.device)
            else:
                delta = (natural_dense[site].to(g.device, torch.float32)
                         - site_floors[site].to(g.device, torch.float32))
                unrestored = (~masks[site].to(g.device).bool()).to(torch.float32)
                factor = delta * unrestored  # [d_sae], unrestored only
            scores[site] = (g.sum(dim=(0, 1)) * factor).cpu()
            pos_scores[site] = g * factor.view(1, 1, -1)  # GPU [B, T, d_sae]
        else:
            # Exact original CPU path — preserves bit-for-bit determinism for
            # non-position-aware restoration arms (their round_select_fn reads
            # these collapsed scores directly).
            floor = site_floors[site].to(torch.float32).cpu()
            natural = natural_dense[site].to(torch.float32).cpu()
            delta = natural - floor
            unrestored = (~masks[site].cpu().bool()).float()
            grad_cpu = grad.to(torch.float32).cpu()
            per_latent = grad_cpu.sum(dim=(0, 1)) * delta
            per_latent = per_latent * unrestored  # unrestored only
            scores[site] = per_latent
    return scores, pos_scores, float(metric.detach().item())


def _round_scores(
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
    alphas: List[float],
    batch_size: Optional[int] = None,
    position_select: Optional[PositionAwareSpec] = None,
    mode: str = "floor_restore",
    inject_targets: Optional[Dict[Site, torch.Tensor]] = None,
    posctx_targets: Optional[Dict[Site, torch.Tensor]] = None,
    inject_mode: str = "both_sign",
    objective: str = "gap",
) -> Tuple[Dict[Site, torch.Tensor], float, Optional[Dict[Candidate, float]]]:
    """One ROUND of restoration scoring: every alpha sample over every
    sequence chunk, merged.

    Sequence count vs batch size: ``tokens`` carries the sequences that
    inform the round; ``batch_size`` bounds one grad pass (VRAM). Chunks
    merge with B_chunk/B_total weights (the metric is a per-sequence mean),
    alpha samples average (left-Riemann) — chunk-outer / alpha-inner, so the
    per-position accumulator never holds more than one chunk.

    ``position_select``: PA-restoration. Per chunk, the per-position
    integrand is accumulated across alphas, then each prefix position selects
    (per-position top-N, or the pooled abs-percentile resolved from THIS
    round's distribution at THIS round's linearisation state — adaptive
    admission) and the union is merged across chunks by max-|score|. Returned
    third element maps (site, latent) -> signed score; None when off. The
    collapsed scores are always returned (trajectory/certificate semantics
    unchanged); the metric is the alpha=0 restored-state metric.
    """

    B_total = int(tokens.shape[0])
    bs = B_total if batch_size is None else max(1, int(batch_size))
    n_alpha = len(alphas)

    scores: Dict[Site, torch.Tensor] = {
        site: torch.zeros(bank.d_sae, dtype=torch.float32) for site in substitute_sites
    }
    metric = 0.0
    # PA selection stays tensor-shaped ([d_sae] running max-|score| per site,
    # 0 = unselected) until every chunk is merged; the Candidate dict is built
    # ONCE at the end. (Profiled 2026-07-18: per-chunk dict merges + a CPU
    # quantile per chunk were ~49% of a deep depth-scaled arm.)
    pa_running: Optional[Dict[Site, torch.Tensor]] = {} if position_select is not None else None

    # Precompute the per-site GPU factor (natural - floor) * unrestored ONCE for
    # this round — it is constant across every alpha x chunk grad pass (delta is
    # constant all discovery; the mask is fixed within a round). Passing it into
    # each grad pass removes the dominant Tensor.to() self-time hotspot. Only the
    # position-aware (GPU-resident) path consumes it; the non-PA path keeps its
    # exact CPU computation for determinism.
    factor_by_site: Optional[Dict[Site, torch.Tensor]] = None
    if position_select is not None and mode != "target_inject":
        # target_inject's factor is the PER-POSITION live delta, computed
        # inside each grad pass from the instrument — nothing to hoist.
        factor_by_site = {}
        for site in substitute_sites:
            # Compute on the anchors' own device; the grad pass moves the factor
            # to the grad's device (a no-op in production, where both are the
            # model device). Keeps this independent of bank.device (mock-safe).
            natural = natural_dense[site].to(torch.float32)
            floor = site_floors[site].to(natural.device, torch.float32)
            unrestored = (~masks[site].to(natural.device).bool()).to(torch.float32)
            factor_by_site[site] = (natural - floor) * unrestored

    for start in range(0, B_total, bs):
        tokens_chunk = tokens[start:start + bs]
        argmax_chunk = pos_argmax[start:start + bs]
        w = tokens_chunk.shape[0] / B_total
        # Residuals are cached per sequence ([B, T, d_model]) from one clean
        # pass over ALL of `tokens` — slice them to the chunk so each
        # sequence rides with its own reconstruction error. (Floors, pins and
        # masks are [d_sae] and broadcast.)
        residuals_chunk = {
            site: (r[start:start + bs] if r.dim() == 3 else r)
            for site, r in residuals.items()
        }
        pos_accum: Dict[Site, torch.Tensor] = {}
        for step, alpha in enumerate(alphas):
            c_scores, c_pos, c_metric = _restoration_grad_pass(
                inference, bank,
                tokens=tokens_chunk,
                substitute_sites=substitute_sites,
                residuals=residuals_chunk,
                site_floors=site_floors,
                natural_dense=natural_dense,
                masks=masks,
                seed_layer=seed_layer,
                seed_kind=seed_kind,
                w_seed=w_seed,
                b_seed=b_seed,
                pos_argmax=argmax_chunk,
                target_act=target_act,
                alpha=alpha,
                keep_positions=position_select is not None,
                factor_by_site=factor_by_site,
                mode=mode,
                inject_targets=inject_targets,
                posctx_targets=posctx_targets,
                inject_mode=inject_mode,
                objective=objective,
            )
            if step == 0:
                metric += w * c_metric
            for site, per_latent in c_scores.items():
                scores[site] += (w / n_alpha) * per_latent
            if c_pos is not None:
                for site, tensor in c_pos.items():
                    contrib = (w / n_alpha) * tensor
                    if site in pos_accum:
                        pos_accum[site] += contrib
                    else:
                        pos_accum[site] = contrib
        if position_select is not None and pos_accum:
            # Anchor at this chunk's peaks; abs_pctl resolves against this
            # round's own pooled distribution (re-resolved every round at the
            # moving linearisation point — adaptive admission).
            spec = dataclass_replace(position_select, peaks=argmax_chunk)
            spec = spec.resolved_for(pos_accum.values())
            for site, attr in pos_accum.items():
                vals = select_position_aware_values(
                    attr, spec.peaks, top_n=spec.top_n, select=spec.select,
                    threshold=spec.threshold, position_weight=spec.position_weight,
                    scope=spec.scope,
                )
                current = pa_running.get(site)
                pa_running[site] = vals if current is None else torch.where(
                    vals.abs() > current.abs(), vals, current
                )
    pa_selected: Optional[Dict[Candidate, float]] = None
    if pa_running is not None:
        # One dict conversion after all chunks. Zero = unselected OR selected
        # at zero attribution — both are "no evidence", matching the classic
        # loop's torch.nonzero admission gate.
        pa_selected = {}
        for site, vals in pa_running.items():
            nz = vals.nonzero(as_tuple=True)[0]
            for latent, value in zip(nz.tolist(), vals[nz].tolist()):
                pa_selected[(site, int(latent))] = float(value)
    return scores, metric, pa_selected


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

    scores, metric, _ = _round_scores(
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
        alphas=[0.0],
    )
    return scores, metric


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
    scores, metric, _ = _round_scores(
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
        alphas=[step / ig_steps for step in range(ig_steps)],
    )
    return scores, metric


def _make_round_select_fn(
    position_select: Optional[PositionAwareSpec],
    last_pa: Dict[str, Optional[Dict[Candidate, float]]],
    round_select: str,
    round_abs_pctl: float,
):
    """The three-branch round-admission rule, shared by both restoration
    drivers (floor_restore and target_inject): PA union side-channel /
    pooled abs-percentile over collapsed scores / None (classic global
    top-per_round_k, handled by the loop itself)."""

    if position_select is not None:
        def round_select_fn(scores):  # noqa: ARG001 — admission comes from the PA union
            selected = last_pa["selected"] or {}
            return [(value, candidate) for candidate, value in selected.items()]
        return round_select_fn
    if round_select == "abs_pctl":
        def round_select_fn(scores):
            # One pooled cut over the round's collapsed nonzero |score| across
            # sites — re-resolved each round at the new linearisation state.
            threshold = pooled_abs_threshold(
                [s.view(1, 1, -1) for s in scores.values()], round_abs_pctl
            )
            admitted = []
            for site, site_scores in scores.items():
                keep = (site_scores.abs() >= threshold) & (site_scores != 0)
                for latent in keep.nonzero(as_tuple=False).squeeze(1).tolist():
                    admitted.append((float(site_scores[latent]), (site, int(latent))))
            return admitted
        return round_select_fn
    return None


def run_negctx_restoration_selection(
    inference: Any,
    bank: Any,
    *,
    neg_tokens: torch.Tensor,
    neg_anchor: torch.Tensor,
    inject_targets: Dict[Site, torch.Tensor],
    sites: set[Site],
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    target_act: float,
    rounds: int,
    per_round_k: int,
    certificate_tol: float,
    allow_negative: bool = True,
    objective: str = "gap",
    inject_mode: str = "both_sign",
    round_select: str = "top_k",
    round_abs_pctl: float = 95.0,
    position_aware: bool = False,
    batch_size: Optional[int] = None,
):
    """Restoration-mode selection along the negctx -> posctx-target
    trajectory (attribution_mode="restoration_negctx", cf-only).

    The restoration loop transplanted onto ig_negctx's path: runs on
    ``neg_tokens``, each round linearising at the CURRENT injected state. Each
    unrestored candidate is scored by the benefit of its best ROLE-DIRECTIONAL
    move — inject at posctx (raise, activator) or inject at 0 (remove,
    inhibitor) — whichever helps the seed fire; latents that help in NEITHER
    direction are never selected (this is the fix: the old version pinned
    every selected latent to posctx, injecting seed-SUPPRESSING latents and
    certifying a state the eval never scores). Restored activators are pinned
    to their posctx target, restored inhibitors to 0 — EXACTLY the cf eval's
    Score-1 intervention ("inject activators at posctx, suppress inhibitors to
    0"), so the certificate closing means cf-faithfulness ~= 1 by construction.
    Each round re-linearises at the moved state, so admission adaptively
    chains toward still-starved latents.

    ``inject_targets`` are the posctx targets (from ``_posctx_targets``); they
    supply the activator-move endpoint and are copied into a mutable per-latent
    ``inject_values`` (posctx for restored activators, 0 for restored
    inhibitors) the instrument pins to. ``sites``/``neg_anchor`` are
    caller-supplied; residuals are cached here from one clean pass on
    ``neg_tokens``. ``batch_size`` should carry the caller's deep-site neg
    microbatch (_ig_negctx_batch). final_ig_polish is not supported.
    """

    from circuit.discovery.iterative_selection import run_iterative_selection
    from circuit.instrument.ig_baseline import collect_natural_codes
    from circuit.types.feature_id import FeatureID

    if round_select not in ("top_k", "abs_pctl"):
        raise ValueError(f"round_select must be 'top_k' or 'abs_pctl', got {round_select!r}")
    if objective not in ("gap", "drive"):
        raise ValueError(f"objective must be 'gap' or 'drive', got {objective!r}")
    if inject_mode not in ("posctx", "directional", "both_sign"):
        raise ValueError(
            f"inject_mode must be 'posctx' | 'directional' | 'both_sign', got {inject_mode!r}"
        )
    if not sites:
        return {}, {}, None

    _, residuals = collect_natural_codes(inference, bank, neg_tokens, sites)

    sae = bank.saes[seed_kind][seed_layer]
    w_seed = sae.encoder.weight[seed_latent_idx].detach()
    b_seed = sae._get_bias_eff()[seed_latent_idx].detach()
    masks = {site: torch.zeros(bank.d_sae, dtype=torch.bool) for site in sites}
    # Per-latent pin value for RESTORED latents: posctx (activators) or 0
    # (inhibitors), mutated by the role-recording admission below as latents
    # are selected. Unrestored latents ignore it (they stay live). Init to 0.
    inject_values = {
        site: torch.zeros(bank.d_sae, dtype=torch.float32) for site in sites
    }

    position_select: Optional[PositionAwareSpec] = None
    if position_aware:
        position_select = PositionAwareSpec(
            peaks=neg_anchor,
            top_n=per_round_k,
            select="top_n" if round_select == "top_k" else "abs_pctl",
            threshold=round_abs_pctl,
        )

    last_pa: Dict[str, Optional[Dict[Candidate, float]]] = {"selected": None}

    def score_fn(current_masks):
        scores, metric, pa_sel = _round_scores(
            inference, bank,
            tokens=neg_tokens,
            substitute_sites=sites,
            residuals=residuals,
            site_floors={},       # unused in target_inject
            natural_dense={},     # unused in target_inject
            masks=current_masks,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            w_seed=w_seed,
            b_seed=b_seed,
            pos_argmax=neg_anchor,
            target_act=target_act,
            alphas=[0.0],
            batch_size=batch_size,
            position_select=position_select,
            mode="target_inject",
            inject_targets=inject_values,     # per-latent role pins
            posctx_targets=inject_targets,    # activator-move endpoint
            inject_mode=inject_mode,
            objective=objective,
        )
        last_pa["selected"] = pa_sel
        return scores, metric

    base_round_select_fn = _make_round_select_fn(
        position_select, last_pa, round_select, round_abs_pctl
    )

    def round_select_fn(scores):
        """Wrap the shared admission to also stamp each admitted latent's pin
        value into inject_values. In "posctx" mode EVERY restored latent pins
        to its posctx target (the original behaviour). In "directional" /
        "both_sign" the score sign is the role: >= 0 activator (pin posctx),
        < 0 inhibitor (pin 0) — mirroring the cf eval. Stamping non-selected
        admits is harmless (only masked latents' pins are read). For top_k
        non-PA (no shared fn) we admit the global top-per_round_k by |score|
        here so pins are recorded on the same path the loop selects from."""
        if base_round_select_fn is not None:
            admitted = base_round_select_fn(scores)
        else:
            flat = []
            for site, site_scores in scores.items():
                nz = (site_scores != 0).nonzero(as_tuple=False).squeeze(1)
                for latent in nz.tolist():
                    flat.append((float(site_scores[latent]), (site, int(latent))))
            flat.sort(key=lambda item: abs(item[0]), reverse=True)
            admitted = flat[:per_round_k]
        for value, (site, latent) in admitted:
            if inject_mode == "posctx":
                inject_values[site][latent] = float(inject_targets[site][latent])
            else:
                inject_values[site][latent] = (
                    float(inject_targets[site][latent]) if value >= 0 else 0.0
                )
        return admitted

    # Same relative certificate as floor_restore: the gap metric is -(gap)^2,
    # so stopping within tol of the target activation means
    # gap^2 <= (tol * target)^2.
    absolute_tol = (certificate_tol * max(abs(target_act), 1e-6)) ** 2
    result = run_iterative_selection(
        score_fn,
        masks=masks,
        rounds=rounds,
        per_round_k=per_round_k,
        certificate_tol=absolute_tol,
        target_metric=0.0,
        allow_negative=allow_negative,
        round_select_fn=round_select_fn,
    )
    sel_desc = round_select + ("+PA" if position_aware else "")
    print(
        f"  [RestorationNegctx/{inject_mode}/{objective}/{sel_desc}] rounds_used={result.rounds_used} "
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
    neg_tokens: Optional[torch.Tensor] = None,
    scorer: str = "point",
    ig_steps: int = 4,
    final_ig_polish: bool = False,
    polish_ig_steps: int = 10,
    round_select: str = "top_k",
    round_abs_pctl: float = 95.0,
    position_aware: bool = False,
    batch_size: Optional[int] = None,
):
    """Full restoration-mode selection for one seed.

    Gathers anchors (floors/pins/residuals — same collectors as the
    ablation-faithfulness eval, so discovery and evaluation share one floor
    definition), runs the iterative selector with the requested per-round
    scorer ("point" = single grad pass at the restored state, classic
    restoration; "ig" = per-round integrated gradients, ig_restoration),
    and returns (positives, negatives, provenance) with FeatureID-keyed
    score dicts shaped like the one-shot extractors'.

    Round admission (round_select x position_aware — four cells):
      - "top_k",    PA off: classic global top-per_round_k by |score| (default,
                    bit-identical to the historical behaviour).
      - "abs_pctl", PA off: admit every latent whose collapsed round |score|
                    clears the round_abs_pctl percentile of the round's pooled
                    nonzero |score| across sites (variable count per round).
      - "top_k",    PA on: per-position top-per_round_k union over the seed's
                    causal prefix (budget grows with live positions — prefer
                    abs_pctl for PA).
      - "abs_pctl", PA on: per-position pooled-percentile union. The
                    percentile re-resolves EVERY round at the moving
                    linearisation state: positions whose chains are already
                    restored stop dominating and admission shifts to
                    still-starved positions — greedy positional coverage.

    ``batch_size``: sequences per grad pass; ``tokens`` may carry more
    (chunk-merged; see _round_scores). None = single pass (historical).

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
    if round_select not in ("top_k", "abs_pctl"):
        raise ValueError(f"round_select must be 'top_k' or 'abs_pctl', got {round_select!r}")

    sites = upstream_sites(bank, seed_layer, seed_kind)
    if not sites:
        return {}, {}, None
    floors, pins = collect_site_anchors(inference, bank, tokens, sites, pos_argmax)
    # Shared floor knob: pins (natural probe values) always stay posctx.
    floors = resolve_site_floors(inference, bank, sites, posctx_means=floors,
                                 loader=loader, neg_tokens=neg_tokens)
    _, residuals = collect_natural_codes(inference, bank, tokens, sites)

    sae = bank.saes[seed_kind][seed_layer]
    w_seed = sae.encoder.weight[seed_latent_idx].detach()
    b_seed = sae._get_bias_eff()[seed_latent_idx].detach()
    masks = {site: torch.zeros(bank.d_sae, dtype=torch.bool) for site in sites}

    # PA-restoration: per-round positional selection. The spec's rule mirrors
    # round_select (top_k -> per-position top-per_round_k; abs_pctl -> pooled
    # percentile, resolved per round in _round_scores). Peaks are stamped per
    # chunk inside _round_scores.
    position_select: Optional[PositionAwareSpec] = None
    if position_aware:
        position_select = PositionAwareSpec(
            peaks=pos_argmax,
            top_n=per_round_k,
            select="top_n" if round_select == "top_k" else "abs_pctl",
            threshold=round_abs_pctl,
        )

    alphas = [step / ig_steps for step in range(ig_steps)] if scorer == "ig" else [0.0]
    # Side-channel: _round_scores returns the round's PA union; the
    # round_select_fn below reads the latest one.
    last_pa: Dict[str, Optional[Dict[Candidate, float]]] = {"selected": None}

    def score_fn(current_masks):
        scores, metric, pa_sel = _round_scores(
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
            alphas=alphas,
            batch_size=batch_size,
            position_select=position_select,
        )
        last_pa["selected"] = pa_sel
        return scores, metric

    round_select_fn = _make_round_select_fn(
        position_select, last_pa, round_select, round_abs_pctl
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
        round_select_fn=round_select_fn,
    )
    sel_desc = round_select + ("+PA" if position_aware else "")
    print(
        f"  [Restoration/{scorer}/{sel_desc}] rounds_used={result.rounds_used} "
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
    "run_negctx_restoration_selection",
    "run_restoration_selection",
    "stamp_restoration_provenance",
]
