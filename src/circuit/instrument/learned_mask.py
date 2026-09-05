"""Learned continuous-mask circuit discovery (the loss-following mode family).

TRI-AMP MASK (named 2026-08-05): the recommended configuration is
    objective="pos" + mask_floor_source="triple" + free_amplitude=True
i.e. reproduce the seed's natural per-sequence pre-activation under THREE
ablation semantics at once (zero + negctx + posctx — a latent must earn
its place under all of them, which is what stops the mask specialising to
any single evaluation floor), with a learned per-latent amplitude on top
of the gate (off / natural / elevated — free range). The object it
produces is a WEIGHTED CIRCUIT: a membership set PLUS a coefficient
vector, and the two must be reported together — a 99-node weighted
circuit is not comparable to a 99-node bare set. Evidence:
dev-notes/data/floor-isolation-2026-08-05/README.md R1-R16, including
the random-set null that validates the amplitude semantics (R16).

Motivated by the 2026-07-24 gradient-observability findings: instantaneous
gradient signs are state-dependent to near-independence (corr(g_natural,
g_intervened) = 0.05 at L8), and the activator/inhibitor populations are
indistinguishable by value — so no single gradient, and no one-shot sign
split, is a reliable membership criterion. This mode never asks either
question: it optimises a soft membership m in (0,1)^N directly against a
reconstruction loss and lets the loss decide what stays.

Three objectives, one engine (mode names follow the ig_mean/ig_negctx
precedent — the _negctx suffix marks the counterfactual-distribution worker):

  "pos"      (abl-mask)          L = mse(pre_pos(m), pre_pos(natural)) + l1.mean(m)
             The sparsest set reproducing natural firing.
  "contrast" (cf-mask_contrast)  + beta * mse(pre_neg(m), pre_neg(natural))
             ... that also keeps the seed silent on negctx (selectivity).
  "negctx"   (cf-mask_negctx)    L = mse(pre_neg(m), target_pos) + l1.mean(1 - m)
             The minimal EDIT to the natural negctx stream that fires the
             seed: a gate-opening search over present-on-negctx latents.
             Sparsity flips to the edit (1 - m) because the reference state
             is the natural stream, not the empty one. Reported residual gap
             measures suppression-gated vs drive-absent silence.
  "raise"    (inhibitor-mask)   L = mse(pre_pos(m), gamma*a_pos) + l1.sum(1 - m)
             The MIRROR of "pos": instead of "keep the seed at natural, pay
             for every latent you KEEP" (smallest sufficient support), it is
             "push the seed ABOVE natural, pay for every latent you SILENCE"
             (smallest effective BRAKE set). Runs on POSCTX, where the seed
             already fires, and asks which latents are holding it back even
             there — the population D2.2 found carries necessity on attn
             seeds. The L1-on-edits is what creates the tension: a latent
             with no effect costs lambda to remove and buys nothing, so it
             is evicted; an ACTIVATOR would lower the seed if removed, so it
             is penalised twice and never recruited. Only genuine
             suppressors survive. Target is gamma * target_act with
             gamma = raise_gamma > 1. Scores are negative (inhibitors), as
             for "negctx". Distinct from "negctx", which asks the same
             question on SILENT contexts (a different population).
  "pin"      (pin-mask, D3.1)     L = mse(pre_pos(m; PINNED), natural) + l1.mean(m)
             The DRIVER twin of "pos". Identical loss shape, but a KEPT
             latent is clamped to its clean POSCTX PIN value rather than
             left at its live value, and a dropped latent goes to the
             floor — i.e. the forward is the PINNED counterfactual the
             pin0 metric measures, made differentiable. Because pinned
             members do not re-encode, the state a member sees is fixed
             by the pins, so training optimises the (P1) pinned-driver
             objective directly instead of the (P2) free-execution one.
             Members are delivered as supports, as for "pos".
  "inject"   (cf-mask_inject)    value' = m*value + delta, delta = softplus(psi) >= 0
             L = mse(pre_neg(m, delta), target_pos) + l1.(sum(1-m) + sum(delta))
             The FULL learned heir of the original counterfactual question
             ("what would make this seed fire here?"): the down-only mask can
             only remove suppression (present inhibitors), the additive delta
             can create drive from latents ABSENT on negctx (absent
             activators) — C1's two roles, learned jointly, each under its
             own sparsity price. Provenance carries the decomposition
             (p_gate_only / p_inject_only / p_both). delta is injected
             position-uniformly, inheriting the cf-injection limitation.

             SPARSITY ECONOMICS (v2, after the 2026-07-24 L8 sweep). v1 shared
             one lambda between the two levers and produced a DEGENERATE
             solution: a diffuse blanket of sub-threshold deltas across the
             whole dictionary reached the target exactly (rec_inject ~= 1.0)
             with ZERO latents above keep_threshold, and the gate was
             abandoned (rec_gate 0.34 -> 0.01). Diffuse additive steering can
             synthesise any direction in the decoder span, so an unpriced
             delta makes the objective near-vacuous. v2 therefore:
               * prices delta on its OWN scale (inject_lambda, activation
                 units — the mask's lambda is unitless and orders of
                 magnitude too weak for deltas);
               * logs delta CONCENTRATION (mass, top-1% share, counts above
                 thresholds) so diffuseness is visible in the row rather than
                 inferred three experiments later;
               * can exclude the N sites nearest the seed from delta
                 (inject_exclude_sites) — injection into the resid site
                 directly below the seed is trivially expressive, so forcing
                 drive to originate further upstream makes the selected
                 latents mean something.
             The interpretable regime is where injection CANNOT fully reach
             the target: the achievable-recovery-vs-sparsity curve is the
             measurement, not the perfect-recovery point.

Design rules carried over from the week's findings:
  * losses target the seed PRE-activation (the post-top-k read is censored to
    exactly 0 below the cutoff — zero gradient exactly where deep seeds live);
  * "reproduce, don't maximise": every target is a natural level, so the
    optimiser cannot win by super-stimulus amplification;
  * held-out probe split: optimisation uses the train slice only, and the
    final data loss is also reported on the held-out slice (a learned method
    can overfit probes in a way one-shot attribution cannot).

The masked transform is the free-execution semantics made differentiable:
decode(m * dense) + error, error preserved from the unmodified encoding.
m = 1 is exactly the identity (decode(all) + error == x).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from circuit.types.feature_id import FeatureID
from observability.phases import phase as _phase
from model.hooks import multi_patch
from sae.dense import sparse_topk_to_dense

Site = Tuple[int, str]

# SIGNED AMPLITUDE (2026-08-30): alpha = psi raw instead of
# softplus(psi), so the fit can assign NEGATIVE amplitudes --
# members that push the seed down (brakes discovered inside the
# standard fit). Per-call flag set by run_learned_mask; the patcher
# and penalties read it through _amp_of().
_SIGNED_AMP = [False]
# FIRING MARGIN (2026-08-30): when set, holds
# (W_enc_site, b_eff_site, k) and the seed tap reports the seed's
# MARGIN over the site's k-th largest pre-activation instead of the
# raw pre-activation.
_MARGIN = [None]


def _amp_of(psi):
    if _SIGNED_AMP[0]:
        return psi
    return torch.nn.functional.softplus(psi)


OBJECTIVES = ("pos", "contrast", "negctx", "inject", "raise", "pin", "logit",
              "maximise")

MASK_FLOOR_SOURCES = ("zero", "posctx", "negctx", "dual", "triple", "pn")
# The floors that need negative contexts to build. EXPORTED because callers
# must gate their negative RETRIEVAL on the same set: gradient_base skips the
# retrieval for non-"store" modes unless something will actually read it, and
# hardcoding one member there (it said != "negctx") silently starved "dual" —
# every dual arm on a close/random mode died on a missing floor. One tuple,
# imported, so a new floor cannot drift out of sync again.
FLOORS_NEEDING_NEGATIVES = ("negctx", "dual", "triple", "pn")


class LearnedMaskPatcher:
    """Differentiable masked forward: at every masked site the dense code is
    multiplied by sigmoid(theta) before decode (error term preserved); at the
    seed's site the pre-activation (w.x + b) is captured and x passes
    untouched. thetas are shared across forwards — the optimiser owns them."""

    def __init__(self, bank: Any, thetas: Dict[Site, torch.Tensor],
                 seed_layer: int, seed_kind: str,
                 w_seed: torch.Tensor, b_seed: torch.Tensor,
                 deltas: Optional[Dict[Site, torch.Tensor]] = None,
                 amps: Optional[Dict[Site, torch.Tensor]] = None,
                 code_dtype: str = "stream",
                 floors: Optional[Dict[Site, torch.Tensor]] = None,
                 pins: Optional[Dict[Site, torch.Tensor]] = None,
                 binarize: str = "none",
                 bin_threshold: float = 0.5,
                 tap_seed: bool = True) -> None:
        # tap_seed=False removes the seed site's special case entirely: it is
        # masked like any other site and no pre-activation is captured. Used
        # by objective='logit', whose endpoint is the OUTPUT, so there is no
        # seed to tap and no reason to exempt the seed's own site from
        # membership. Every pre-activation objective needs tap_seed=True.
        self.tap_seed = bool(tap_seed)
        self.bank = bank
        self.thetas = thetas
        self.deltas = deltas or {}
        # FREE AMPLITUDE (2026-08-05): per-latent multiplicative scale
        # alpha = softplus(psi), giving each KEPT latent free range — 0 via
        # the gate, its natural value at alpha=1, elevated above it — rather
        # than gating between two fixed endpoints. code becomes
        # m*alpha*dense + (1-m)*floor. None == historical gate-only.
        self.amps = amps or {}
        # Per-site floor [d_sae]: the value a FULLY masked latent takes.
        # None == the zero floor (code * m), the historical behaviour.
        self.floors = floors or {}
        # D3.1: per-site [d_sae] clean posctx values. When present the
        # transform runs the PINNED counterfactual instead of the scaled one.
        self.pins = pins or {}
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.w_seed = w_seed
        self.b_seed = b_seed
        # Dense-code dtype. "stream" matches CircuitOnlyPatcher (x.dtype);
        # "fp32" forces float32 as the original implementation did. The dense
        # codes and their retained backward graph are the largest tensors in
        # the optimisation, so fp32 roughly DOUBLES the mask's footprint —
        # measured at L10 (32 sites): 15.1GB peak against a 6.9GB baseline,
        # which spilled into WDDM shared memory. Parameters stay fp32 either
        # way; this only affects activations, i.e. ordinary mixed precision.
        self.code_dtype = code_dtype
        # Gate discretisation - the TopK-SAE lesson (its top-k lives INSIDE
        # the training forward, so it has no soft/hard gap by construction)
        # adapted to a GLOBAL membership that still needs gradients for
        # non-members:
        #   "none"   - sigmoid(theta), historical soft gate. Converged masks
        #              are genuinely fractional (79% of L8 members at
        #              m in (0.5,0.9)) and ANY post-hoc cut is lossy.
        #   "ste"    - forward uses the HARD mask (m > bin_threshold),
        #              backward flows through the soft sigmoid (straight-
        #              through estimator). Training forward == eval semantics
        #              exactly; non-members keep their gradient path.
        #   "anneal" - sigmoid(theta / T) with T -> 0 over training
        #              (continuous sparsification): the gate BECOMES binary.
        #              self.temperature is set per step by the training loop.
        self.binarize = binarize
        self.bin_threshold = float(bin_threshold)
        self.temperature = 1.0
        self.seed_pre: Optional[torch.Tensor] = None
        # constant per-site tensors cast once per (site, device, dtype) —
        # the per-call .to() copies were ~700 kernel launches per step.
        self._const_cache: Dict[Any, torch.Tensor] = {}

    def _const(self, tag: str, site: Site, t: torch.Tensor,
               like: torch.Tensor) -> torch.Tensor:
        key = (tag, site, like.device, like.dtype)
        c = self._const_cache.get(key)
        if c is None:
            c = t.to(device=like.device, dtype=like.dtype)
            self._const_cache[key] = c
        return c

    def release(self) -> None:
        """Drop the graph-connected seed capture. Bounded to one forward's
        graph (each _forward_preact overwrites it), so this is hygiene, not
        the multi-pass ratchet fix the discovery instruments needed
        (vram-ledger 2026-07-31)."""
        self.seed_pre = None

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if (self.tap_seed and layer_idx == self.seed_layer
                and kind == self.seed_kind):
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            _pre = x @ w + b
            if _MARGIN[0] is not None:
                _We, _be, _k = _MARGIN[0]
                _full = x.to(_We.dtype) @ _We.T + _be
                _tau = torch.topk(_full, _k, dim=-1).values[..., -1]
                _pre = _pre - _tau.to(_pre.dtype)
            self.seed_pre = _pre
            return x
        theta = self.thetas.get((layer_idx, kind))
        psi = self.deltas.get((layer_idx, kind))
        if theta is None and psi is None:
            return x
        ta, ti = self.bank.encode(x, kind, layer_idx)
        code_dt = torch.float32 if self.code_dtype == "fp32" else x.dtype
        dense = sparse_topk_to_dense(ta, ti, self.bank.d_sae, dtype=code_dt)
        # ONE decode, not two. The masked-stream semantics are
        #     out = decode(code) + (x - decode(dense))
        # and decode is affine with a shared bias, so that is exactly
        #     out = x + (code - dense) @ W_dec.T
        # We therefore build the DIFFERENCE (code - dense) directly and pass it
        # through the decoder once with add_bias=False. This drops a full
        # [B*T, d_sae] @ [d_sae, d_model] matmul AND its backward at every
        # masked site of every forward — and backward is ~70% of the training
        # loop (measured: 66.7ms vs 27.0ms forward, steady state, L10).
        #
        # It is also MORE accurate, not a trade: the old form subtracted two
        # nearly-equal decoded tensors (at init m = sigmoid(4) = 0.982, so
        # code ~ dense), which is catastrophic cancellation. Measured against
        # an fp64 reference at L5-resid, max abs error at init: fp32 4.54e-2 ->
        # 3.56e-4 (128x better), bf16 6.00e-2 -> 7.34e-3 (8x better).
        # Equivalence is ALGEBRAIC, not bitwise.
        delta_code = None
        if theta is not None:
            if self.binarize == "anneal":
                m = torch.sigmoid(theta / max(self.temperature, 1e-4))
            else:
                m = torch.sigmoid(theta)
            m = m.to(device=dense.device, dtype=dense.dtype)
            if self.binarize == "ste":
                hard = (m > self.bin_threshold).to(m.dtype)
                m = hard + m - m.detach()
            pin = self.pins.get((layer_idx, kind))
            floor = self.floors.get((layer_idx, kind))
            amp = self.amps.get((layer_idx, kind))
            if amp is not None and pin is None:
                # code = m*alpha*dense + (1-m)*floor, so
                # delta = dense*(m*alpha - 1) + (1-m)*floor.
                # At alpha=1 this reduces exactly to the gate-only forms
                # below (algebraic identity, both floor branches).
                alpha = _amp_of(amp).to(
                    device=dense.device, dtype=dense.dtype)
                delta_code = dense * (m * alpha - 1.0)
                if floor is not None:
                    delta_code = delta_code + (1.0 - m) * self._const(
                        "floor", (layer_idx, kind), floor, dense)
            elif pin is not None:
                # PINNED execution (D3.1): kept -> clean pin value, dropped
                # -> floor (or 0). code = m*pin + (1-m)*floor, so the delta
                # against the live dense code is code - dense.
                p = self._const("pin", (layer_idx, kind), pin, dense)
                base = (self._const("floor", (layer_idx, kind), floor, dense)
                        if floor is not None else torch.zeros_like(p))
                delta_code = (m * p + (1.0 - m) * base) - dense
            elif floor is None:
                # zero floor: code - dense == -dense * (1 - m)
                delta_code = -dense * (1.0 - m)
            else:
                # MEAN-ABLATION MASK: a fully masked latent lands on the floor
                # instead of 0, so m=0 reproduces the EVAL's empty-circuit
                # state rather than a state no eval measures. m=1 is still
                # exactly identity (delta_code == 0 there), so the
                # decode(all)+error invariant is untouched.
                # code - dense == (floor - dense) * (1 - m)
                f = self._const("floor", (layer_idx, kind), floor, dense)
                delta_code = (f - dense) * (1.0 - m)
        if psi is not None:
            # additive injection, always >= 0; position-uniform (broadcast
            # over [B, T]) — the same semantics as the cf injection eval.
            # It enters `code` additively, so it enters the difference as-is.
            delta = torch.nn.functional.softplus(psi).to(device=dense.device,
                                                         dtype=dense.dtype)
            delta_code = delta if delta_code is None else delta_code + delta
        if delta_code is None:
            return x
        out = self.bank.decode(delta_code, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


def _forward_preact(inference: Any, patcher: LearnedMaskPatcher,
                    tokens: torch.Tensor, grad: bool) -> torch.Tensor:
    patcher.seed_pre = None
    # Never run a patcher forward through the compiled model: each distinct
    # patcher closure fails dynamo's guards and adds a recompile variant to
    # module-level caches. This matches the eager-patcher policy every other
    # discovery path follows (restoration wraps each instrument forward the
    # same way). NOTE: this is hygiene/consistency, NOT the fix for the
    # 5.9GB post-discovery residue — that was parameter .grad accumulation
    # from loss.backward(), fixed by the backbone freeze in run_learned_mask.
    # Restoring the PRIOR state (rather than enable_compile unconditionally)
    # keeps this a no-op inside the training loop's existing disable window.
    was_compiled = getattr(inference, "_compiled", False)
    if was_compiled:
        inference.disable_compile()
    try:
        inference.forward(tokens, patcher=patcher, grad_enabled=grad,
                          return_activations=False, tokenize_final=False)
    finally:
        if was_compiled:
            inference.enable_compile()
    if patcher.seed_pre is None:
        raise RuntimeError("learned_mask: seed pre-activation was not captured")
    return patcher.seed_pre


def _forward_logits(inference: Any, patcher: LearnedMaskPatcher,
                    tokens: torch.Tensor, grad: bool) -> torch.Tensor:
    """Masked forward returning OUTPUT LOGITS [B, T, V].

    The behavioural counterpart of _forward_preact: the endpoint is the
    model's next-token distribution rather than the seed latent's
    pre-activation. Note this runs the FULL depth of the model — the
    pre-act path can stop at the seed's layer, this one cannot — so a
    logit-objective step is strictly more expensive than a pos one on the
    same sites. Same no-compile policy, same reasoning as _forward_preact.
    """
    was_compiled = getattr(inference, "_compiled", False)
    if was_compiled:
        inference.disable_compile()
    try:
        out = inference.forward(tokens, patcher=patcher, grad_enabled=grad,
                                all_logits=True, return_activations=False,
                                tokenize_final=False)
    finally:
        if was_compiled:
            inference.enable_compile()
    logits = out[1] if isinstance(out, (tuple, list)) else out
    if logits is None:
        raise RuntimeError("learned_mask[logit]: forward returned no logits "
                           "(all_logits=True not honoured?)")
    return logits


def _target_logprob(logits: torch.Tensor, anchors: torch.Tensor,
                    targets: torch.Tensor) -> torch.Tensor:
    """log p(target token) at each sequence's anchor position -> [B]."""
    B = min(int(logits.shape[0]), int(anchors.shape[0]), int(targets.shape[0]))
    idx = torch.arange(B, device=logits.device)
    pa = anchors[:B].to(logits.device).clamp(0, logits.shape[1] - 1)
    lp = torch.log_softmax(logits[:B][idx, pa].float(), dim=-1)
    return lp[idx, targets[:B].to(logits.device).long()]


def _at(pre: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    B = min(pre.shape[0], anchors.shape[0])
    idx = torch.arange(B, device=pre.device)
    pa = anchors[:B].to(pre.device).clamp(0, pre.shape[1] - 1)
    return pre[:B][idx, pa]


@torch.no_grad()
def _natural_logprob(inference: Any, bank: Any, tokens: torch.Tensor,
                     anchors: torch.Tensor, targets: torch.Tensor,
                     seed_layer: int, seed_kind: str,
                     w_seed: torch.Tensor, b_seed: torch.Tensor,
                     code_dtype: str = "stream",
                     batch: int = 8) -> torch.Tensor:
    """Full-model log p(target) at each anchor -> [B]. The logit
    objective's REPRODUCTION target, exactly as _natural is for pos."""
    p = LearnedMaskPatcher(bank, {}, seed_layer, seed_kind, w_seed, b_seed,
                           code_dtype=code_dtype)
    out = []
    for s in range(0, int(tokens.shape[0]), batch):
        lg = _forward_logits(inference, p, tokens[s:s + batch], grad=False)
        out.append(_target_logprob(lg, anchors[s:s + batch],
                                   targets[s:s + batch]).detach())
    return torch.cat(out, dim=0) if out else torch.zeros(0)


@torch.no_grad()
def _natural(inference: Any, bank: Any, tokens: torch.Tensor,
             seed_layer: int, seed_kind: str,
             w_seed: torch.Tensor, b_seed: torch.Tensor,
             code_dtype: str = "stream") -> torch.Tensor:
    """Natural (m=1 == untouched) seed pre-activation [B, T]."""
    p = LearnedMaskPatcher(bank, {}, seed_layer, seed_kind, w_seed, b_seed,
                           code_dtype=code_dtype)
    return _forward_preact(inference, p, tokens, grad=False).detach()


def _frozen_backbone_params(inference: Any, bank: Any) -> List[torch.Tensor]:
    """Every model/SAE parameter that currently requires grad. Duck-typed so
    test doubles without .model / .saes simply contribute nothing."""
    params: List[torch.Tensor] = []
    model = getattr(inference, "model", None)
    if model is not None and hasattr(model, "parameters"):
        params.extend(p for p in model.parameters() if p.requires_grad)
    for kind in getattr(bank, "kinds", ()):
        for sae in getattr(bank, "saes", {}).get(kind, ()):
            if sae is not None and hasattr(sae, "parameters"):
                params.extend(p for p in sae.parameters() if p.requires_grad)
    return params


def run_learned_mask(inference: Any, bank: Any, **kwargs
                     ) -> Tuple[Dict[FeatureID, float], Dict[str, Any]]:
    """Freeze the model/SAE backbone around the mask optimisation.

    The engine trains theta with ``loss.backward()`` — the only discovery
    path that does (everything else uses ``torch.autograd.grad`` on explicit
    anchors). ``backward()`` accumulates a ``.grad`` the size of EVERY
    requires-grad parameter it can reach: model + SAE encoders + the
    materialised decoders ≈ 6GB that then lives on the parameters for the
    rest of the process (measured 2026-08-01: a deep-seed mask discovery
    rested 5.95GB above other methods; memory-history snapshot attributed
    the blocks to mm_mat2_backward / EmbeddingBackward0 — weight grads).
    Freezing skips those weight-grad kernels entirely (theta gradients still
    flow THROUGH frozen modules), cutting both the residue and the in-run
    peak; the zero_grad clears anything accumulated by earlier callers."""
    params = _frozen_backbone_params(inference, bank)
    for p in params:
        p.requires_grad_(False)
    try:
        return _run_learned_mask_impl(inference, bank, **kwargs)
    finally:
        for p in params:
            p.requires_grad_(True)
            p.grad = None


def _run_learned_mask_impl(
    inference: Any,
    bank: Any,
    *,
    objective: str,
    sites: List[Site],
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    seed_vector: "Optional[Tuple[torch.Tensor, torch.Tensor]]" = None,
    signed_amplitude: bool = False,
    margin_topk: "Optional[int]" = None,
    pos_tokens: torch.Tensor,
    pos_argmax: torch.Tensor,
    neg_tokens: Optional[torch.Tensor] = None,
    target_tokens: Optional[torch.Tensor] = None,
    target_act: Optional[float] = None,
    steps: int = 200,
    lr: float = 0.05,
    l1_lambda: float = 1e-3,
    beta: float = 1.0,
    member_penalty: "Optional[Dict[Site, torch.Tensor]]" = None,
    member_penalty_weight: float = 0.0,
    neg_suppress_weight: float = 0.0,
    inject_lambda: Optional[float] = None,
    inject_exclude_sites: int = 0,
    keep_threshold: float = 0.5,
    batch_size: int = 4,
    holdout_frac: float = 0.25,
    theta_init: float = 4.0,
    theta_init_mode: str = "uniform",
    theta_lo: float = -4.0,
    site_lambda_weights: Optional[Dict[Site, float]] = None,
    log_every: int = 50,
    deep_site_threshold: Optional[int] = None,
    deep_batch_size: Optional[int] = None,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    code_dtype: str = "stream",
    lr_schedule: str = "constant",
    lr_min_frac: float = 0.05,
    warmup_frac: float = 0.0,
    mask_floor_source: str = "zero",
    dual_floor_weight: float = 1.0,
    # gamma on the posctx term of mask_floor_source="triple" and "pn";
    # ignored otherwise. Defaults to dual_floor_weight's house value so the
    # third semantics enters at the same price as the second.
    triple_floor_weight: float = 0.25,
    # FREE AMPLITUDE (Daniel, 2026-08-05): every latent gets a learned
    # multiplicative scale alpha = softplus(psi) on top of the gate, so the
    # mask has free range — off (gate), natural (alpha=1), or elevated
    # (alpha>1). psi initialises at softplus^-1(1) so training starts
    # EXACTLY at the historical gate-only behaviour. The promise is
    # compensation: one latent at 2x can replace two redundant latents,
    # which no pure gate can express. Bounded via the objective's target
    # (unlike `maximise`) — amplitudes that overshoot the target are pushed
    # back down by the same squared error that prices undershoot.
    free_amplitude: bool = False,
    # optional price on |alpha - 1| for MEMBERS (weighted by the gate), so
    # amplitudes deviate from natural only when the data term earns it.
    # 0.0 = truly free range.
    amp_l1: float = 0.0,
    binarize: str = "none",
    anneal_reach_frac: float = 1.0,
    support: Optional[Dict[Site, torch.Tensor]] = None,
    raise_gamma: float = 1.5,
    pin_values: Optional[Dict[Site, torch.Tensor]] = None,
    scale_normalize: bool = False,
    delta_init: Optional[Dict[Site, Dict[int, float]]] = None,
    suppress_weight: float = 0.0,
    step_hook: Optional[Any] = None,
    logger: Any = None,
) -> Tuple[Dict[FeatureID, float], Dict[str, Any]]:
    """Optimise the mask and return (scores, provenance).

    cf-mask v2 knobs (2026-08-02, all default-off so existing behaviour is
    bit-identical):

    ``scale_normalize`` (negctx/inject only): divide the data term by
    target_act**2 (relative squared error — the dual floor's bounded-
    normaliser lesson) and price delta per unit of TARGET (delta_sum /
    target_act). Seed scales span ~100x on the panel, so without this one
    lambda_inj default is simultaneously crushing on weak seeds and
    toothless on strong ones — the diagnosed root cause of the MI arm's
    dead-at-defaults verdict.

    ``delta_init``: {site: {latent: value}} warm start in ACTIVATION units
    (e.g. AMPC's alpha* x posctx pins) — psi is set to softplus^-1(value)
    so training begins AT a working intervention and refines, instead of
    having to find drive from scratch (the empty-circuit cliff).

    ``suppress_weight`` (inject only): adds a posctx term driving the seed
    to ZERO through the same gates/deltas (D3.2's dual-intervention second
    half). Weighted like the main term; scale-normalised when
    scale_normalize is set.

    ``support`` (D3.6, 2026-08-01): restrict the SEARCH SPACE to a candidate
    set per site — a bool mask [d_sae] or a LongTensor of latent indices.
    Latents outside the support are hard-excluded: theta clamped to -12
    (m ~= 6e-6, i.e. the floor) at init and re-clamped after every
    optimizer step, so neither the data gradient nor AdamW's decoupled
    decay can move them; they are unreachable by selection. Sites absent
    from the dict are excluded entirely. This is NOT theta_init_mode
    ("active" changes the START but still trains the full space); support
    changes the space itself — the mask becomes a pruner/refiner over an
    attribution-proposed candidate set, and its recall is capped at the
    support's. Applies to thetas only (the inject objective's deltas are
    not restricted).

    Scores: "pos"/"contrast" -> {fid: m} for kept members (m > keep_threshold,
    all positive). "negctx" -> {fid: -(1 - m)} for edited members
    ((1 - m) > keep_threshold, negative — they are delivered as inhibitors:
    the latents whose PRESENCE holds the seed off).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if objective in ("contrast", "negctx", "inject"):
        if neg_tokens is None or int(neg_tokens.shape[0]) == 0:
            raise ValueError(f"objective={objective!r} requires neg_tokens")
    if objective == "logit" and target_tokens is None:
        raise ValueError("objective='logit' requires target_tokens (the token "
                         "to predict at each sequence's pos_argmax): the "
                         "endpoint is BEHAVIOURAL, not the seed latent")
    if free_amplitude and objective == "pin":
        raise ValueError("free_amplitude is not defined for objective='pin': "
                         "pinned execution fixes kept latents to clean values, "
                         "which contradicts a learned amplitude on them")
    if objective == "pin" and pin_values is None:
        raise ValueError("objective='pin' requires pin_values "
                         "(per-site [d_sae] clean posctx anchors) — "
                         "the pinned counterfactual is undefined "
                         "without them")
    if objective == "raise" and target_act is None:
        raise ValueError("objective='raise' requires target_act "
                         "(the seed's natural posctx level; the "
                         "objective targets raise_gamma * it)")
    _SIGNED_AMP[0] = bool(signed_amplitude)
    if margin_topk is not None:
        _sae_m = bank.saes[seed_kind][seed_layer]
        _MARGIN[0] = (_sae_m.encoder.weight.detach(),
                      _sae_m._get_bias_eff().detach(), int(margin_topk))
    else:
        _MARGIN[0] = None
    if objective == "raise" and float(raise_gamma) <= 1.0:
        raise ValueError("raise_gamma must be > 1: the objective is to push "
                         f"the seed ABOVE natural, got {raise_gamma}")
    if objective in ("negctx", "inject") and target_act is None:
        raise ValueError(
            f"objective={objective!r} requires target_act (posctx level)")
    if binarize not in ("none", "ste", "anneal"):
        raise ValueError(
            f"binarize must be 'none', 'ste' or 'anneal', got {binarize!r}")
    if binarize == "anneal" and abs(float(keep_threshold) - 0.5) > 1e-9:
        raise ValueError(
            "binarize='anneal' hardens the gate at theta=0, i.e. m=0.5 - a "
            f"keep_threshold of {keep_threshold} would select a DIFFERENT set "
            "than the one training converged to. Use 0.5, or 'ste' which "
            "binarises at keep_threshold itself.")

    # ------------------------------------------------------------------
    # Mask floor: what a fully masked (m=0) latent becomes.
    #
    # WHY THIS EXISTS. With the zero floor the mask's training counterfactual
    # is exactly free0's, so the mask is always evaluated on home turf and a
    # mask-vs-mean-floor-method comparison is confounded. Setting the floor to
    # a mean makes m=0 reproduce the mean-ablated state that freeM/freeN
    # measure against, so the two families can be scored on a metric neither
    # one owns.
    #
    # WHY negctx RATHER THAN posctx. Measured on this run's seeds, the posctx
    # empty-circuit fill already reaches 23.0% / 29.9% of a_pos at L8 / L9
    # (0% at L0-L6) — the posctx floor CREDITS ITSELF at depth. The negctx
    # fill measures 0.0000 at every seed. A posctx-floored mask would be
    # trained toward, and scored on, the leaking counterfactual.
    #
    # Deliberately NOT config.discovery.floor_source: that knob is shared with
    # the ig hops, so driving the mask through it would silently change the
    # other arms in the same run. Same underlying collect_site_means, so the
    # floors are bit-identical to the eval anchors.
    if mask_floor_source not in MASK_FLOOR_SOURCES:
        raise ValueError(
            f"mask_floor_source must be one of {MASK_FLOOR_SOURCES}, "
            f"got {mask_floor_source!r} — 'global'/'diverse' need a data "
            "loader the mask engine does not take; add one before offering "
            "them rather than silently falling back to another floor")
    # "dual" = zero AND negctx, both scored every step. Allowed for the
    # objectives whose loss lives on the POSITIVES (pos, maximise); barred for
    # negctx/contrast/inject, which already put their loss on the negatives,
    # so a second negctx-floored term there is not the same question.
    #
    # maximise was added to this list 2026-08-05 after the freeM panel showed
    # a single-zero-floor maximise EXPLOITS that floor: it raises the seed by
    # DELETING SUPPRESSORS, which only works when "deleted" means set-to-zero.
    # Its free0/freeM_dense ratio was 10.7x median against pos's 1.17x. Under
    # the dual floor a latent must earn its place under BOTH ablation
    # semantics, and deletion buys nothing under the negctx floor — which is
    # exactly the pressure that should suppress the exploit. The original
    # restriction was a whitelist of one written before this objective
    # existed, not a judgement about it.
    #
    # "triple" = zero AND negctx AND posctx, all three scored every step.
    # Added 2026-08-05 on Daniel's suggestion. Note the third term trains on
    # the floor the design comment above deliberately excluded, so it must be
    # used knowingly: (a) posctx CREDITS ITSELF at depth (comment records
    # 23%/29.9% of a_pos at L8/L9 — though measured on the 8 seeds of
    # floor-isolation-2026-08-05 the leak is 0.0% on 7 and 3.5% on 1, so the
    # exclusion may deserve re-examination), and (b) training on posctx makes
    # freeM_dense IN-SAMPLE, costing the held-out metric that currently
    # validates pos/dual. Expect a weak third gradient too: a dropped latent
    # lands on its posctx mean, which on positive contexts is close to its
    # actual value, so the intervention is gentle by construction.
    # "pn" = negctx AND posctx, NO zero term. Added 2026-08-06 (Daniel's
    # question: with free amplitudes, does the zero floor still earn its
    # place, or can alpha->0 substitute for it?). The ablation is the direct
    # test of the R1 finding that the zero term is what forces the set to
    # carry the signal rather than the delta-from-baseline. Weighting:
    # negctx PROMOTES to the primary 1.0 slot that zero held (this is the
    # mode one would actually ship if zero were unnecessary — comparable
    # overall loss scale to dual/triple); posctx keeps triple_floor_weight;
    # dual_floor_weight is unused.
    DUAL_OK = ("pos", "maximise")
    triple_floor = mask_floor_source == "triple"
    pos_floor = mask_floor_source in ("triple", "pn")
    zero_term = mask_floor_source in ("dual", "triple")
    dual_floor = mask_floor_source in ("dual", "triple", "pn")
    if dual_floor and objective not in DUAL_OK:
        raise ValueError(
            f"mask_floor_source='dual' is defined for {DUAL_OK}; "
            f"got {objective!r}. The negctx/contrast/inject objectives already "
            "carry a negative-context term, so composing them with a dual "
            "floor is a different experiment — do it deliberately, not by "
            "accident.")
    floors: Optional[Dict[Site, torch.Tensor]] = None
    if mask_floor_source != "zero":
        from eval.floors import collect_site_means
        if mask_floor_source in FLOORS_NEEDING_NEGATIVES:
            if neg_tokens is None or int(neg_tokens.shape[0]) == 0:
                raise ValueError(
                    "mask_floor_source='negctx' requires neg_tokens, but none "
                    "were supplied. Refusing to fall back: a run labelled "
                    f"{mask_floor_source!r} that silently used posctx is worse "
                    "than a visible failure.")
            floor_tokens = neg_tokens
        else:
            floor_tokens = pos_tokens
        floors = collect_site_means(inference, bank, floor_tokens, set(sites))
        if logger is not None:
            logger.note(f"learned_mask: {mask_floor_source} floor over "
                        f"{int(floor_tokens.shape[0])} sequences, "
                        f"{len(floors)} sites")
    # triple's/pn's posctx floor: collected separately because the shared
    # `floors` slot is already carrying negctx.
    floors_pos: Optional[Dict[Site, torch.Tensor]] = None
    if pos_floor:
        from eval.floors import collect_site_means as _csm
        floors_pos = _csm(inference, bank, pos_tokens, set(sites))
        if logger is not None:
            logger.note(f"learned_mask: {mask_floor_source} posctx floor over "
                        f"{int(pos_tokens.shape[0])} sequences")

    # Depth-adaptive VRAM guard (the sites x per-site-tensors law): every
    # backward holds dense-code graphs at ALL masked sites simultaneously,
    # so deep seeds shrink the MICRO-batch — but keep the effective batch via
    # GRADIENT ACCUMULATION (micro-chunks backwarded separately, one
    # opt.step()). The optimisation regime is therefore identical across
    # depths; only the peak VRAM changes. Without this, deep seeds either
    # spill into WDDM shared memory at PCIe speed (measured, L10 at batch 4)
    # or pay batch-2 gradient noise (the pre-accumulation version).
    micro_bs = max(1, int(batch_size))
    if (deep_site_threshold is not None and deep_batch_size is not None
            and len(sites) > int(deep_site_threshold)):
        micro_bs = max(1, int(deep_batch_size))
        if logger is not None:
            logger.note(f"learned_mask: {len(sites)} sites > "
                        f"{deep_site_threshold} — micro-batch "
                        f"{batch_size} -> {micro_bs} with gradient "
                        f"accumulation (effective batch unchanged)")
    accum = max(1, -(-int(batch_size) // micro_bs))   # ceil division

    sae = bank.saes[seed_kind][seed_layer]
    w_seed = sae.encoder.weight[seed_latent_idx].detach()
    b_seed = sae._get_bias_eff()[seed_latent_idx].detach()
    # DIFFERENTIAL SEED (2026-08-28): seed_vector=(w, b) overrides
    # the encoder-row derivation, so a circuit can be fitted against
    # a VIRTUAL direction -- e.g. w_A - w_B, whose reconstruction
    # target is the difference signal between two same-site latents.
    # Topic-shared composition cancels by construction; the circuit
    # contains the differentia. Everything downstream (patchers,
    # floors, scoring taps) consumes (w_seed, b_seed) verbatim.
    if seed_vector is not None:
        w_seed = seed_vector[0].detach()
        b_seed = seed_vector[1].detach()
    device = getattr(bank, "device", pos_tokens.device)

    # theta_init_mode="active": probe-INACTIVE latents start at theta_lo
    # (m ~ 0.02) instead of theta_init (m ~ 0.98).
    #
    # Under the ZERO floor this is close to exact rather than a heuristic: the
    # transform is delta_code = -dense*(1-m), so a latent that never fires on
    # any probe sequence contributes ZERO for every m - its data gradient is
    # identically zero and only the L1 penalty moves it. Uniform init at +4
    # therefore spends the first ~theta_init/lr steps marching millions of
    # informationless thetas across the 0.5 threshold: the measured 80-100
    # step burn-in during which the "circuit" is the entire dictionary
    # (327,680 / 1,024,000 / 1,310,720 members at steps 25-50, exactly
    # n_sites * d_sae). Down-initialising them deletes that burn-in without
    # touching anything the data term can see.
    #
    # Under MEAN floors (negctx/dual) masked inactive latents inject the floor
    # value, so there this is a genuine PRIOR, not a no-op - the data gradient
    # can still pull any of them back up, but the starting point biases
    # membership and must be A/B-checked rather than assumed.
    if theta_init_mode not in ("uniform", "active"):
        raise ValueError("theta_init_mode must be 'uniform' or 'active', "
                         f"got {theta_init_mode!r}")
    if theta_init_mode == "active":
        # One no-grad pass over the probes per site: union of post-top-k
        # active latents. Uses the same encode the training forward uses.
        active_sets: Dict[Site, torch.Tensor] = {}

        class _ActiveCapture:
            def __init__(self):
                self.seen: Dict[Site, torch.Tensor] = {
                    st: torch.zeros(bank.d_sae, dtype=torch.bool,
                                    device=device) for st in sites}

            def __call__(self, model):
                return multi_patch(model, self.transform)

            def transform(self, layer_idx, kind, x):
                if (layer_idx, kind) in self.seen:
                    _, ti = bank.encode(x, kind, layer_idx)
                    self.seen[(layer_idx, kind)].index_fill_(
                        0, ti.reshape(-1).to(device), True)
                return x

        cap = _ActiveCapture()
        # Same compiled-model guard as _forward_preact (see its comment).
        _was = getattr(inference, "_compiled", False)
        if _was:
            inference.disable_compile()
        try:
            with torch.no_grad():
                inference.forward(pos_tokens, patcher=cap, grad_enabled=False,
                                  return_activations=False, tokenize_final=False)
        finally:
            if _was:
                inference.enable_compile()
        active_sets = cap.seen
        thetas = {}
        for st in sites:
            t = torch.full((bank.d_sae,), float(theta_lo), device=device)
            t[active_sets[st]] = float(theta_init)
            t.requires_grad_(True)
            thetas[st] = t
        if logger is not None:
            n_act = sum(int(v.sum()) for v in active_sets.values())
            logger.note(f"learned_mask: active-init {n_act:,} of "
                        f"{len(sites) * bank.d_sae:,} latents at theta_init="
                        f"{theta_init}, rest at theta_lo={theta_lo}")
    else:
        thetas = {
            s: torch.full((bank.d_sae,), float(theta_init), device=device,
                          requires_grad=True)
            for s in sites
        }

    # ---- D3.6 support restriction --------------------------------------
    SUPPORT_EXCL = -12.0          # sigmoid(-12) ~= 6e-6: at the floor
    support_excl_masks: Dict[Site, torch.Tensor] = {}
    if support is not None:
        n_in = 0
        for st in sites:
            spec = support.get(st)
            keep = torch.zeros(bank.d_sae, dtype=torch.bool, device=device)
            if spec is not None:
                spec = spec.to(device)
                if spec.dtype == torch.bool:
                    keep = spec.clone()
                else:
                    keep[spec.long()] = True
            support_excl_masks[st] = ~keep
            n_in += int(keep.sum())
            with torch.no_grad():
                thetas[st].masked_fill_(support_excl_masks[st], SUPPORT_EXCL)
        if logger is not None:
            logger.note(f"learned_mask: support restricted to {n_in:,} of "
                        f"{len(sites) * bank.d_sae:,} latents "
                        f"({100.0 * n_in / max(len(sites) * bank.d_sae, 1):.2f}%)")
    # "inject": an additive delta = softplus(psi) per latent, psi init -4
    # (delta ~= 0.018, small but with live gradient) — absent-on-negctx
    # latents are unreachable by the multiplicative mask, delta is their
    # only handle.
    deltas: Dict[Site, torch.Tensor] = {}
    if objective == "inject":
        # sites are ordered shallow -> deep, so the LAST entries are the
        # seed-adjacent ones; excluding them forces injected drive to
        # originate further upstream.
        n_ex = min(max(0, int(inject_exclude_sites)), len(sites))
        delta_sites = sites[:len(sites) - n_ex]
        deltas = {s: torch.full((bank.d_sae,), -4.0, device=device,
                                requires_grad=True) for s in delta_sites}
        if delta_init:
            # Warm start: psi = softplus^-1(v), so softplus(psi) == v at
            # step 0 — training begins AT the supplied intervention.
            # (math is imported at module level — a local import here would
            # shadow it for the whole function scope and break lr_at.)
            n_warm_init = 0
            with torch.no_grad():
                for st, vals in delta_init.items():
                    if st not in deltas:
                        continue           # excluded or non-upstream site
                    for idx, v in vals.items():
                        v = float(v)
                        if v <= 0:
                            continue
                        psi = v if v > 20 else math.log(math.expm1(v))
                        deltas[st][int(idx)] = psi
                        n_warm_init += 1
            if logger is not None:
                logger.note(f"learned_mask[inject]: warm-started {n_warm_init} "
                            f"deltas from delta_init")
        if n_ex and logger is not None:
            logger.note(f"learned_mask[inject]: delta excluded from the "
                        f"{n_ex} seed-adjacent site(s); {len(delta_sites)} "
                        f"of {len(sites)} sites injectable"
                        + (" — NO injectable sites left, this run is "
                           "gate-only (equivalent to mask_negctx)"
                           if not delta_sites else ""))
    # free amplitude: psi init at softplus^-1(1) = ln(e - 1), so alpha = 1
    # exactly and step 0 reproduces the gate-only mask bit-for-bit.
    amps: Dict[Site, torch.Tensor] = {}
    if free_amplitude:
        _psi1 = (1.0 if signed_amplitude
                 else math.log(math.expm1(1.0)))
        amps = {s: torch.full((bank.d_sae,), _psi1, device=device,
                              requires_grad=True) for s in sites}
    params = list(thetas.values()) + list(deltas.values()) + list(amps.values())
    inj_lambda = float(l1_lambda if inject_lambda is None else inject_lambda)
    # scale_normalize: relative error + target-relative delta pricing (see
    # docstring). Bounded by construction — a normaliser must never be able
    # to annihilate its own term.
    if scale_normalize and objective in ("negctx", "inject", "raise"):
        if target_act is None:
            raise ValueError("scale_normalize requires target_act")
        sn_data = 1.0 / max(float(target_act) ** 2, 1e-6)
        sn_delta = 1.0 / max(float(target_act), 1e-6)
    else:
        sn_data, sn_delta = 1.0, 1.0

    # Learning-rate schedule. Membership here is a THRESHOLD CROSSING, so late
    # in training a latent oscillating across m = 0.5 has its inclusion decided
    # by wherever the last step left it; decaying lr progressively freezes
    # membership instead. Cosine tapers smoothly (step decay would re-freeze at
    # arbitrary points). lr_min_frac is offered for schedules that want late
    # refinement, but a zero floor is safe: the AdamW update is
    # theta -= lr*grad + lr*wd*theta, so BOTH terms vanish with lr and the tail
    # steps simply become no-ops rather than decay drifting m without evidence.
    # The "_up" (warmup) variants are the mirror images. They exist because
    # DECAY MEASURABLY HURT: at matched sum(lr) it produced 11-28% BIGGER
    # circuits. Pruning a latent is a slow threshold crossing (Adam's step is
    # ~lr*sign, so theta must be walked from +theta_init across zero), which
    # means what governs size is not the lr integral but how much lr survives
    # LATE, while marginal latents are still mid-crossing. Decay starves
    # exactly those steps; warmup feeds them.
    _SCHEDULES = ("constant", "cosine", "linear", "cosine_up", "linear_up")
    if lr_schedule not in _SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {_SCHEDULES}, "
                         f"got {lr_schedule!r}")
    lr_floor = float(lr) * float(lr_min_frac)
    # warmup_frac ramps lr_floor -> lr over the first fraction of steps, THEN
    # the decay shape runs over what remains. This is the conventional recipe
    # the earlier decay arms did NOT test: those started at peak on step 0 and
    # fell to exactly zero, spending half their lr budget in the first quarter
    # of training. Only meaningful for the decaying schedules — a ramp into a
    # ramp is not a schedule anyone would ship.
    if not 0.0 <= float(warmup_frac) < 1.0:
        raise ValueError(f"warmup_frac must be in [0, 1), got {warmup_frac}")
    if warmup_frac and lr_schedule not in ("cosine", "linear"):
        raise ValueError("warmup_frac only applies to 'cosine' or 'linear', "
                         f"got lr_schedule={lr_schedule!r}")
    n_warm = int(round(float(warmup_frac) * int(steps)))

    def lr_at(step: int) -> float:
        if lr_schedule == "constant":
            return float(lr)
        if step < n_warm:
            # +1 so the ramp REACHES peak on the last warmup step rather than
            # one step short of it.
            return lr_floor + (float(lr) - lr_floor) * (step + 1) / n_warm
        frac = (step - n_warm) / max(int(steps) - n_warm - 1, 1)
        if lr_schedule == "linear":
            shape = 1.0 - frac
        elif lr_schedule == "linear_up":
            shape = frac
        elif lr_schedule == "cosine_up":
            shape = 0.5 * (1.0 - math.cos(math.pi * frac))
        else:                                   # cosine
            shape = 0.5 * (1.0 + math.cos(math.pi * frac))
        return lr_floor + (float(lr) - lr_floor) * shape

    # BOTH budgets scale with the lr INTEGRAL, not with lr: sparsity is
    # sum(lr)*l1_lambda and decay is sum(lr)*weight_decay. A decaying schedule
    # halves sum(lr) for the same peak, so peak lr must be ~2x the calibrated
    # constant lr to keep the budgets (and therefore the calibration) fixed.
    lr_sum = sum(lr_at(t) for t in range(int(steps)))
    decay_product = lr_sum * float(weight_decay)
    sparsity_product = lr_sum * float(l1_lambda)
    if logger is not None:
        logger.note(f"learned_mask[{lr_schedule}]: sum(lr) = {lr_sum:.2f} | "
                    f"decay budget {decay_product:.3f} (calibrated ~1.0, "
                    f"m_kept target ~0.75) | sparsity budget "
                    f"{sparsity_product:.2e}")
    if optimizer not in ("adam", "adamw"):
        raise ValueError(f"optimizer must be 'adam' or 'adamw', got {optimizer!r}")
    if optimizer == "adamw":
        # NOTE: decoupled decay pulls theta toward 0 == m toward 0.5 — the
        # keep-threshold boundary, NOT sparsity. It regularises confidence in
        # both directions; the L1 term remains the only sparsifier.
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(params, lr=lr)

    def split(tokens, anchors):
        n = int(tokens.shape[0])
        n_hold = int(round(n * holdout_frac))
        n_train = max(1, n - n_hold)
        return (tokens[:n_train], anchors[:n_train],
                tokens[n_train:], anchors[n_train:])

    # ---- targets from the NATURAL stream (reproduce, don't maximise) -------
    if objective == "logit":
        # No seed tap: the seed's own site is masked like any other, so no
        # pre-activation is captured and the pre-act target is undefined.
        # The logit branch below sets the real (log-prob) target.
        pos_tgt_all = torch.zeros(int(pos_tokens.shape[0]), device=device)
    else:
        pos_nat = _natural(inference, bank, pos_tokens, seed_layer, seed_kind,
                           w_seed, b_seed, code_dtype=code_dtype)
        pos_tgt_all = _at(pos_nat, pos_argmax)
    if (objective in ("contrast", "negctx", "inject")
            or neg_suppress_weight > 0):
        neg_nat = _natural(inference, bank, neg_tokens, seed_layer, seed_kind,
                           w_seed, b_seed, code_dtype=code_dtype)
        # would-be-firing anchor per negctx sequence (pre-act argmax) — the
        # same anchor the anchored cf eval and ig_negctx use.
        neg_anchors = neg_nat.argmax(dim=-1).cpu()
        neg_tgt_all = _at(neg_nat, neg_anchors)

    pt_tr, pa_tr, pt_ho, pa_ho = split(pos_tokens, pos_argmax)
    ptgt_tr = pos_tgt_all[:pt_tr.shape[0]]
    ptgt_ho = pos_tgt_all[pt_tr.shape[0]:]
    logit_tok = logit_tok_tr = logit_tok_ho = None
    if objective == "logit":
        # accept [B] (already gathered at the anchor) or [B, T] (gather here)
        tt = target_tokens
        if tt.dim() > 1:
            bidx = torch.arange(min(tt.shape[0], pos_argmax.shape[0]))
            tt = tt[bidx, pos_argmax[:tt.shape[0]].clamp(0, tt.shape[1] - 1)]
        logit_tok = tt[:pos_tokens.shape[0]]
        # target = the FULL model's log-prob, so the mask REPRODUCES the
        # behaviour rather than maximising it — the same "reproduce, don't
        # maximise" contract every other objective follows.
        lp_nat_all = _natural_logprob(inference, bank, pos_tokens, pos_argmax,
                                      logit_tok, seed_layer, seed_kind,
                                      w_seed, b_seed, code_dtype=code_dtype,
                                      batch=max(1, int(micro_bs)))
        ptgt_tr = lp_nat_all[:pt_tr.shape[0]]
        ptgt_ho = lp_nat_all[pt_tr.shape[0]:]
        logit_tok_tr = logit_tok[:pt_tr.shape[0]]
        logit_tok_ho = logit_tok[pt_tr.shape[0]:]
    if (objective in ("contrast", "negctx", "inject")
            or neg_suppress_weight > 0):
        nt_tr, na_tr, nt_ho, na_ho = split(neg_tokens, neg_anchors)
        ntgt_tr = neg_tgt_all[:nt_tr.shape[0]]
        ntgt_ho = neg_tgt_all[nt_tr.shape[0]:]

    # MAXIMISE normaliser. The data term is LINEAR in the seed's
    # pre-activation, so it needs a scale or lambda's meaning changes with
    # depth (a_pos ~3 at L2, ~19 at L8). One scalar — the mean natural
    # target — not a per-sequence ratio: per-sequence division blows up
    # wherever a target is near zero, and the per-sequence spread is small
    # anyway (measured cv 0.054-0.403 over L2/L8 seeds, since probe
    # sequences are selected as POSITIVE contexts and so sample the top of
    # the seed's range). After dividing, loss -1.0 means "the seed sits at
    # its natural level on average" and -2.0 means "twice natural".
    max_scale = 1.0
    if objective == "maximise":
        max_scale = max(float(pos_tgt_all.abs().mean()), 1e-3)
    _pins = pin_values if objective == "pin" else None
    _tap = objective != "logit"
    patcher = LearnedMaskPatcher(bank, thetas, seed_layer, seed_kind,
                                 w_seed, b_seed, deltas=deltas, amps=amps,
                                 code_dtype=code_dtype, floors=floors,
                                 pins=_pins,
                                 binarize=binarize,
                                 bin_threshold=keep_threshold,
                                 tap_seed=_tap)
    # DUAL FLOOR: a second patcher over the SAME thetas, differing only in the
    # floor, so one mask is scored under both ablation semantics and gradients
    # from both accumulate into the same parameters.
    #
    # Why: a single-floor mask learns whatever its own floor rewards. Measured
    # on L2/L5/L8 — the negctx-floored mask reaches freeN 0.66-1.06 while its
    # free0 is EXACTLY 0.0 at L5 and L8 (the post-top-k signature: the members
    # alone cannot get the seed into top-k). It learns the DELTA from the
    # negative baseline, never a sufficient set. The zero-floored mask has the
    # mirror problem: sufficient, but never asked what distinguishes firing
    # from a near-identical non-firing context.
    patcher_zero = (
        LearnedMaskPatcher(bank, thetas, seed_layer, seed_kind, w_seed, b_seed,
                           deltas=deltas, amps=amps, code_dtype=code_dtype,
                           floors=None,
                           binarize=binarize, bin_threshold=keep_threshold)
        if zero_term else None)
    # triple's/pn's posctx patcher: same thetas again, posctx floor.
    patcher_pos = (
        LearnedMaskPatcher(bank, thetas, seed_layer, seed_kind, w_seed, b_seed,
                           deltas=deltas, amps=amps, code_dtype=code_dtype,
                           floors=floors_pos, binarize=binarize,
                           bin_threshold=keep_threshold)
        if pos_floor else None)

    def mask_mean() -> torch.Tensor:
        return torch.stack([torch.sigmoid(t).mean() for t in thetas.values()]).mean()

    # Per-site sparsity weights: lambda_s = l1_lambda * w_s. None = flat
    # pricing (historical). Motivation for non-flat: the C3 dose-response
    # showed members with individually tiny scores COLLECTIVELY carry deep
    # reconstruction, and flat lambda charges those diffuse-signal sites the
    # same per latent as concentrated near-seed sites - so it prunes exactly
    # the population C3 showed is load-bearing. Weights are normalised by the
    # caller; the engine just applies them.
    if site_lambda_weights is not None:
        _missing = [st for st in sites if st not in site_lambda_weights]
        if _missing:
            raise ValueError(
                f"site_lambda_weights missing {len(_missing)} sites, e.g. "
                f"{_missing[:3]} - refusing to default missing sites to 1.0")

    def mask_sum() -> torch.Tensor:
        if site_lambda_weights is not None:
            return torch.stack(
                [float(site_lambda_weights[st]) * torch.sigmoid(t).sum()
                 for st, t in thetas.items()]).sum()
        # Penalty in PER-LATENT units (sum, not mean): mean-normalising over
        # ~3e5 latents makes the per-latent L1 gradient ~lambda*sigma'/N ~ 5e-10,
        # BELOW Adam's eps (1e-8) — the epsilon floor then swallows the
        # sparsity pressure and nothing prunes (measured: the L2 spike kept
        # all 327,680 latents). With sum, every latent feels lambda*sigma'(theta)
        # regardless of dictionary size, so lambda is a per-latent price.
        return torch.stack([torch.sigmoid(t).sum() for t in thetas.values()]).sum()

    def edit_sum() -> torch.Tensor:
        return torch.stack(
            [(1.0 - torch.sigmoid(t)).sum() for t in thetas.values()]).sum()

    def delta_sum() -> torch.Tensor:
        # deltas can be legitimately empty (every site excluded from
        # injection — a gate-only inject run); torch.stack rejects an empty
        # list, so return a real zero that still participates in autograd.
        if not deltas:
            return torch.zeros((), device=device)
        return torch.stack(
            [torch.nn.functional.softplus(q).sum() for q in deltas.values()]).sum()

    def data_loss(tokens, anchors, targets, micro_index, pat=None) -> torch.Tensor:
        pat = pat if pat is not None else patcher
        s = (micro_index * micro_bs) % max(int(tokens.shape[0]), 1)
        tk, an = tokens[s:s + micro_bs], anchors[s:s + micro_bs]
        tg = targets[s:s + micro_bs].to(device)
        if tk.shape[0] == 0:
            s = 0
            tk, an, tg = tokens[:micro_bs], anchors[:micro_bs], targets[:micro_bs].to(device)
        if objective == "logit":
            # BEHAVIOURAL endpoint: squared error of the masked forward's
            # log p(target) against the full model's. Same gates, same
            # sites, same positions as `pos` — only the endpoint moves.
            tokid = logit_tok_tr[s:s + tk.shape[0]]
            lg = _forward_logits(inference, pat, tk, grad=True)
            lp = _target_logprob(lg, an, tokid)
            return ((lp - tg[:lp.shape[0]].to(lp.device, lp.dtype)) ** 2).mean()
        pre = _forward_preact(inference, pat, tk, grad=True)
        vals = _at(pre, an)
        if objective == "maximise":
            # No target: drive the seed as high as the mask can, and let the
            # L1 term set the price of each latent kept. UNBOUNDED by
            # construction — there is no fixed point, so the member set is
            # whatever survives "value gained per latent > lambda". This is
            # a DRIVER objective (cf. `raise`, which is the bounded form
            # targeting raise_gamma * natural); it does not reproduce the
            # natural state and should not be read as a closure object.
            return -(vals.mean() / max_scale)
        if objective == "raise":
            tgt = torch.full_like(vals, float(raise_gamma) * float(target_act))
        elif objective in ("negctx", "inject"):
            tgt = torch.full_like(vals, float(target_act))
        else:
            tgt = tg[:vals.shape[0]].to(vals.device, vals.dtype)
        return ((vals - tgt) ** 2).mean()

    # TERM SCALING. Both dual terms measure the SAME quantity - squared error
    # of the seed's pre-activation against its natural value - on the SAME
    # target. They are therefore already in the same units, and one SHARED,
    # TARGET-SCALED normaliser is all that is needed: dividing by
    # mean(target^2) makes each a relative squared error, so gamma means
    # exactly "how much does the negctx counterfactual count relative to the
    # zero one".
    #
    # This previously divided each term by ITS OWN fully-closed-mask loss,
    # which is UNBOUNDED and produced a silent failure. Measured
    # norm_zero/norm_floor: L2 137.7, L5 1.08, L8 1.19, and L10 **1.9e8** -
    # at L10 the zero floor's closed state drives the seed's pre-activation to
    # ~1.8e5 (zeroing all 32 upstream sites is far off-manifold; the same
    # explosion that makes the pre-act empty floor blow up while free0's
    # post-top-k read still shows a clean 0.0000), giving a normaliser of
    # 3.3e10 against the negctx floor's 176. The zero term was divided into
    # oblivion, dual silently became negctx-only, and it reproduced
    # negctx-only's exact failure signature (free0 == 0.0, negative
    # freeM_topk). A normaliser must never be able to annihilate its own term.
    #
    # The closed-mask losses are still MEASURED and reported: they are a
    # genuinely useful off-manifold diagnostic (they are what exposed this).
    # They simply no longer set the scale.
    norm_zero = norm_floor = 1.0
    dual_norm = 1.0
    if dual_floor:
        with torch.no_grad():
            shut = {s: torch.full((bank.d_sae,), -40.0, device=device)
                    for s in sites}                      # sigmoid(-40) == 0
            for tag, fl in (("zero", None), ("floor", floors)):
                p_shut = LearnedMaskPatcher(bank, shut, seed_layer, seed_kind,
                                            w_seed, b_seed,
                                            code_dtype=code_dtype, floors=fl)
                v = _at(_forward_preact(inference, p_shut, pt_tr, grad=False),
                        pa_tr)
                closed_loss = float(((v - ptgt_tr.to(v.device, v.dtype)) ** 2).mean())
                if tag == "zero":
                    norm_zero = max(closed_loss, 1e-6)
                else:
                    norm_floor = max(closed_loss, 1e-6)
            # bounded by construction: the natural target's own scale
            dual_norm = max(float((ptgt_tr.to(torch.float32) ** 2).mean()), 1e-6)
        if logger is not None:
            logger.note(f"learned_mask[dual]: scale mean(target^2) "
                        f"{dual_norm:.4f} | gamma {dual_floor_weight} | "
                        f"closed-mask DIAGNOSTIC zero {norm_zero:.4g} / floor "
                        f"{norm_floor:.4g} (ratio "
                        f"{norm_zero / max(norm_floor, 1e-12):.4g}x — large "
                        f"means the zero floor's empty state is far "
                        f"off-manifold, NOT that the term is down-weighted)")

    losses: List[float] = []
    inference.disable_compile()
    try:
        for step in range(int(steps)):
            if binarize == "anneal":
                # geometric 1.0 -> 0.05, reaching the floor at
                # anneal_reach_frac of the run and HOLDING it after. 1.0 =
                # descend the whole run (the floor arrives only at the last
                # step). Compressed schedules measurably need the hold: at
                # 200 steps with reach=1.0 the boundary population is still
                # churning when the run ends (end-of-run flips 2-2.7x the
                # 400-step run despite matched metrics) - the mask gets no
                # time AT final sharpness to settle.
                prog = step / max(int(steps) - 1, 1)
                eff = min(prog / max(float(anneal_reach_frac), 1e-6), 1.0)
                t_now = 1.0 * (0.05 ** eff)
                patcher.temperature = t_now
                if patcher_zero is not None:
                    patcher_zero.temperature = t_now
                if patcher_pos is not None:
                    patcher_pos.temperature = t_now
            if lr_schedule != "constant":
                lr_now = lr_at(step)
                for group in opt.param_groups:
                    group["lr"] = lr_now
            opt.zero_grad()
            # Gradient accumulation: each micro-chunk's scaled loss is
            # backwarded on its own (freeing that chunk's graph before the
            # next forward), so peak VRAM is one micro-chunk while the
            # STEP's gradient equals the full effective-batch gradient.
            step_total = 0.0
            for j in range(accum):
                mi = step * accum + j
                # Every loss term is backwarded SEPARATELY (grad(a+b) =
                # grad(a) + grad(b)): summing contrast's two terms first kept
                # the posctx forward's graph alive while the negctx forward
                # built its own — two full all-site graphs at once, which
                # doubled peak VRAM and spilled into WDDM shared memory on L8
                # (measured). Per-term backward holds one graph at a time.
                if objective in ("pos", "maximise") and dual_floor:
                    # Same data, same anchors, same target — two ABLATION
                    # SEMANTICS. A latent must earn its place under both.
                    # `pos` divides by dual_norm = mean(target^2) to make its
                    # squared errors relative; `maximise` is LINEAR and its
                    # data_loss already divides by max_scale, so it needs no
                    # second normaliser — applying dual_norm there would
                    # rescale by the target's magnitude a second time.
                    dn = dual_norm if objective == "pos" else 1.0
                    if patcher_zero is not None:
                        terms = [(pt_tr, pa_tr, ptgt_tr, 1.0 / dn,
                                  patcher_zero),
                                 (pt_tr, pa_tr, ptgt_tr,
                                  float(dual_floor_weight) / dn, patcher)]
                    else:
                        # "pn": negctx promotes to the primary slot.
                        terms = [(pt_tr, pa_tr, ptgt_tr, 1.0 / dn, patcher)]
                    if patcher_pos is not None:
                        terms.append((pt_tr, pa_tr, ptgt_tr,
                                      float(triple_floor_weight) / dn,
                                      patcher_pos))
                elif objective == "pos":
                    terms = [(pt_tr, pa_tr, ptgt_tr, 1.0, None)]
                elif objective in ("pin", "logit", "maximise"):
                    terms = [(pt_tr, pa_tr, ptgt_tr, 1.0, None)]
                elif objective == "raise":
                    # posctx data; the target inside data_loss is
                    # gamma * target_act, so ptgt_tr is unused.
                    terms = [(pt_tr, pa_tr, ptgt_tr, sn_data, None)]
                elif objective == "contrast":
                    terms = [(pt_tr, pa_tr, ptgt_tr, 1.0, None),
                             (nt_tr, na_tr, ntgt_tr, beta, None)]
                else:  # negctx/inject — the loss lives on the negatives.
                    terms = [(nt_tr, na_tr, ntgt_tr, sn_data, None)]
                    if objective == "inject" and suppress_weight > 0:
                        # dual-intervention second half: the SAME gates and
                        # deltas must also silence the seed on posctx.
                        terms.append((pt_tr, pa_tr,
                                      torch.zeros_like(ptgt_tr),
                                      float(suppress_weight) * sn_data, None))
                # NEG-SUPPRESS (2026-08-28 knowledge-circuit work): the
                # circuit must also reproduce the seed's natural silence on
                # hard negatives — contrast's second term, made composable
                # with the pos objective and the triple floor. Members that
                # merely restate the seed's surface form fire on the hard
                # negatives too and are priced out here.
                if neg_suppress_weight > 0 and objective in ("pos",
                                                             "maximise"):
                    # Normalised like the primary pos term (divide by
                    # dual_norm): the raw-mse version outbids sparsity and
                    # the optimiser keeps ~18k members to make the negative
                    # stream near-natural (measured, 40-step smoke).
                    _dn = dual_norm if objective == "pos" else 1.0
                    terms.append((nt_tr, na_tr, ntgt_tr,
                                  float(neg_suppress_weight) / _dn, None))
                for tokens_t, anchors_t, targets_t, w, pat_t in terms:
                    with _phase("fit.fwd"):
                        part = w * data_loss(tokens_t, anchors_t, targets_t,
                                             mi, pat_t) / accum
                    with _phase("fit.bwd"):
                        part.backward()
                    step_total += float(part.detach())
            if objective in ("negctx", "raise"):
                penalty = l1_lambda * edit_sum()
            elif objective == "pin":
                penalty = l1_lambda * mask_sum()
            elif objective == "inject":
                # Two levers, two units, two prices: edits are unitless
                # (1 - m in [0, 1]), deltas are activation magnitudes. Sharing
                # one lambda let diffuse injection outbid the gate entirely
                # (v1 degeneracy) — inject_lambda is swept on its own scale.
                penalty = (l1_lambda * edit_sum()
                           + inj_lambda * sn_delta * delta_sum())
            else:
                penalty = l1_lambda * mask_sum()
            if member_penalty is not None and member_penalty_weight > 0:
                # ECHO PENALTY (2026-08-28): per-latent price scaled by a
                # precomputed vector — in the knowledge-circuit runs this is
                # corr(a_latent, a_seed)^2 over the probe stream, so members
                # that are mere copies of the seed's own signal pay extra.
                # Generic: any {site: [d_sae]} vector prices membership.
                _mp = [(torch.sigmoid(thetas[s])
                        * member_penalty[s].to(device=thetas[s].device,
                                               dtype=thetas[s].dtype)).sum()
                       for s in thetas if s in member_penalty]
                if _mp:
                    penalty = penalty + (float(member_penalty_weight)
                                         * torch.stack(_mp).sum())
            if amps:
                # LEAK GUARD (found by test, 2026-08-05): L1 prices the gate
                # m but the signal is m*alpha, so the optimiser can push m
                # BELOW the membership threshold (cheap) and inflate alpha to
                # compensate — loss ~0, membership empty, nothing priced.
                # Charging (1-m)*|alpha-1| at the same l1_lambda makes the
                # sub-threshold-gate route strictly more expensive than
                # honest membership (m~0.4, alpha~2.5 costs ~1.3*lambda vs a
                # member's ~1.0*lambda), so amplitude can only be used by
                # latents that PAY the membership price. Members' amplitudes
                # stay free unless amp_l1 > 0.
                off_member = torch.stack([
                    ((1.0 - torch.sigmoid(thetas[s]))
                     * (_amp_of(a) - 1.0).abs()).sum()
                    for s, a in amps.items()]).sum()
                penalty = penalty + l1_lambda * off_member
                if amp_l1 > 0:
                    penalty = penalty + amp_l1 * torch.stack([
                        (torch.sigmoid(thetas[s])
                         * (_amp_of(a) - 1.0).abs()).sum()
                        for s, a in amps.items()]).sum()
            with _phase("fit.penalty_bwd"):
                penalty.backward()
            # step_hook (diagnostics only): called AFTER backward with the
            # live gradients and BEFORE opt.step() applies them, then again
            # after the step via `post`. Never mutates state; a None hook is
            # a single branch per step.
            grads_now = None
            if step_hook is not None:
                grads_now = {
                    "theta": {s: (t.grad.detach().clone() if t.grad is not None
                                  else None) for s, t in thetas.items()},
                    "psi": {s: (d.grad.detach().clone() if d.grad is not None
                                else None) for s, d in deltas.items()},
                }
            with _phase("fit.opt"):
                opt.step()
            if support_excl_masks:
                # Re-clamp excluded latents: Adam momentum and AdamW's
                # decoupled decay both move parameters with zero data
                # gradient, so the exclusion must be re-imposed each step.
                with torch.no_grad():
                    for st, excl in support_excl_masks.items():
                        thetas[st].masked_fill_(excl, SUPPORT_EXCL)
            losses.append(step_total + float(penalty.detach()))
            if step_hook is not None:
                step_hook(step, {
                    "thetas": thetas, "deltas": deltas, "grads": grads_now,
                    "data_loss": step_total,
                    "penalty": float(penalty.detach()),
                    "temperature": float(getattr(patcher, "temperature", 1.0)),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "keep_threshold": float(keep_threshold),
                    "patcher": patcher,
                })
            if logger is not None and log_every and step % int(log_every) == 0:
                logger.note(f"learned_mask[{objective}] step {step} "
                            f"loss {losses[-1]:.5f} mean_m {float(mask_mean().detach()):.4f}")

        # Release the optimisation graph's cached blocks before the eval
        # phase allocates: measured 2.5GB of reserved memory recovered at L10
        # (15,664 -> 13,114 MB), which is the difference between fitting in
        # dedicated VRAM and paging to system RAM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # held-out data loss at the final mask (no grad)
        with torch.no_grad():
            if objective == "logit" and pt_ho.shape[0]:
                lg = _forward_logits(inference, patcher, pt_ho, grad=False)
                lp = _target_logprob(lg, pa_ho, logit_tok_ho)
                ho = float(((lp - ptgt_ho[:lp.shape[0]].to(lp.device, lp.dtype))
                            ** 2).mean())
            elif objective == "maximise" and pt_ho.shape[0]:
                pre = _forward_preact(inference, patcher, pt_ho, grad=False)
                # -1.0 == natural on average; more negative == amplified
                ho = float(-(_at(pre, pa_ho).mean() / max_scale))
            elif objective in ("pos", "pin") and pt_ho.shape[0]:
                pre = _forward_preact(inference, patcher, pt_ho, grad=False)
                ho = float(((_at(pre, pa_ho) - ptgt_ho.to(device)) ** 2).mean())
            elif objective == "contrast" and pt_ho.shape[0] and nt_ho.shape[0]:
                pre_p = _forward_preact(inference, patcher, pt_ho, grad=False)
                pre_n = _forward_preact(inference, patcher, nt_ho, grad=False)
                ho = float(((_at(pre_p, pa_ho) - ptgt_ho.to(device)) ** 2).mean()
                           + beta * ((_at(pre_n, na_ho) - ntgt_ho.to(device)) ** 2).mean())
            elif objective == "raise" and pt_ho.shape[0]:
                pre = _forward_preact(inference, patcher, pt_ho, grad=False)
                v = _at(pre, pa_ho)
                ho = float(((v - float(raise_gamma) * float(target_act)) ** 2).mean())
            elif objective in ("negctx", "inject") and nt_ho.shape[0]:
                pre = _forward_preact(inference, patcher, nt_ho, grad=False)
                v = _at(pre, na_ho)
                ho = float(((v - float(target_act)) ** 2).mean())
            else:
                ho = None
    finally:
        inference.enable_compile()

    # ---- selection ----------------------------------------------------------
    scores: Dict[FeatureID, float] = {}
    kept_m: List[float] = []
    # free amplitude: converged alpha per KEPT member, plus summary stats.
    _amp_stats = None
    _amp_kept: Optional[Dict[str, Dict[int, float]]] = None
    if amps:
        _amp_kept = {}
        _all_alpha: List[float] = []
        with torch.no_grad():
            for (layer, kind), theta in thetas.items():
                m = torch.sigmoid(theta)
                a_here = amps.get((layer, kind))
                if a_here is None:
                    continue
                alpha = _amp_of(a_here)
                idx = (m > keep_threshold).nonzero(as_tuple=True)[0]
                vals = alpha[idx].tolist()
                _amp_kept["%d/%s" % (layer, kind)] = {
                    int(i): round(v, 4) for i, v in zip(idx.tolist(), vals)}
                _all_alpha.extend(vals)
        if _all_alpha:
            t = torch.tensor(_all_alpha)
            _amp_stats = {
                "n": t.numel(),
                "median": round(float(t.median()), 4),
                "p10": round(float(t.quantile(0.10)), 4),
                "p90": round(float(t.quantile(0.90)), 4),
                "max": round(float(t.max()), 4),
                "frac_elevated": round(float((t > 1.1).float().mean()), 4),
                "frac_reduced": round(float((t < 0.9).float().mean()), 4),
            }
    with torch.no_grad():
        # DEVICE->HOST IN BULK. `m` lives on the GPU, so the obvious
        # `float(m[i])` inside the loop costs a full device sync PER ELEMENT.
        # At L10's ~108k members that is ~216k syncs: measured 13.52s against
        # 0.20s for the batched form, a 66x difference, in a loop that does no
        # arithmetic. Gather the selected values ONCE per site with .tolist()
        # and iterate over plain Python floats.
        for (layer, kind), theta in thetas.items():
            m = torch.sigmoid(theta)
            if objective == "inject":
                # gate half: edits, delivered as inhibitors (negative)
                edit = 1.0 - m
                idx = (edit > keep_threshold).nonzero(as_tuple=True)[0]
                ids = idx.tolist()
                edits = edit[idx].tolist()
                ms = m[idx].tolist()
                for i, e_i, m_i in zip(ids, edits, ms):
                    scores[FeatureID(layer, kind, i)] = -e_i
                    kept_m.append(m_i)
                # injection half: delta in ACTIVATION units, delivered as
                # activators (positive). keep_threshold is reused across two
                # unit systems — a documented v1 simplification.
                psi_here = deltas.get((layer, kind))
                if psi_here is None:
                    continue           # site excluded from injection
                delta = torch.nn.functional.softplus(psi_here)
                jdx = (delta > keep_threshold).nonzero(as_tuple=True)[0]
                for i, d_i in zip(jdx.tolist(), delta[jdx].tolist()):
                    fid = FeatureID(layer, kind, i)
                    scores[fid] = max(scores.get(fid, 0.0), d_i)
                continue
            if objective in ("negctx", "raise"):
                edit = 1.0 - m
                idx = (edit > keep_threshold).nonzero(as_tuple=True)[0]
                ids = idx.tolist()
                edits = edit[idx].tolist()
                ms = m[idx].tolist()
                for i, e_i, m_i in zip(ids, edits, ms):
                    scores[FeatureID(layer, kind, i)] = -e_i
                    kept_m.append(m_i)
            else:
                idx = (m > keep_threshold).nonzero(as_tuple=True)[0]
                ids = idx.tolist()
                ms = m[idx].tolist()
                for i, m_i in zip(ids, ms):
                    scores[FeatureID(layer, kind, i)] = m_i
                    kept_m.append(m_i)

    # "inject": decompose the recovery — gate-only (deltas off), inject-only
    # (mask at natural), both — three no-grad forwards on the train negatives.
    decomp = {}
    if objective == "inject":
        with torch.no_grad():
            def _decomp_p(th, de):
                pr = _forward_preact(
                    inference,
                    LearnedMaskPatcher(bank, th, seed_layer, seed_kind,
                                       w_seed, b_seed, deltas=de,
                                       code_dtype=code_dtype, floors=floors),
                    nt_tr, grad=False)
                return round(float(_at(pr, na_tr).mean()), 4)
            decomp = {"p_both": _decomp_p(thetas, deltas),
                      "p_gate_only": _decomp_p(thetas, {}),
                      "p_inject_only": _decomp_p({}, deltas)}

    # Delta concentration: a diffuse blanket and a sparse population can
    # reach the same target, so the ROW must distinguish them.
    delta_stats = {}
    if objective == "inject":
        # Always emit the full key set for inject runs, zero-filled when no
        # site is injectable — a missing key is indistinguishable from an
        # unlogged one downstream, and this is the diagnostic that would
        # explain WHY a run found nothing.
        delta_stats = {
            "delta_sum": 0.0, "delta_top1pct_share": None, "delta_max": 0.0,
            "n_delta_gt_0p1": 0, "n_delta_gt_0p5": 0, "n_delta_gt_1p0": 0,
            "n_delta_sites": len(deltas),
        }
        if deltas:
            with torch.no_grad():
                flat = torch.cat([torch.nn.functional.softplus(q).flatten()
                                  for q in deltas.values()])
                total = float(flat.sum())
                k = max(1, flat.numel() // 100)
                delta_stats.update({
                    "delta_sum": round(total, 4),
                    "delta_top1pct_share": (round(float(flat.topk(k).values.sum()) / total, 4)
                                            if total > 0 else None),
                    "delta_max": round(float(flat.max()), 4),
                    "n_delta_gt_0p1": int((flat > 0.1).sum()),
                    "n_delta_gt_0p5": int((flat > 0.5).sum()),
                    "n_delta_gt_1p0": int((flat > 1.0).sum()),
                })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    provenance = {
        "objective": objective,
        "support_n": (sum(int((~m).sum()) for m in support_excl_masks.values())
                      if support_excl_masks else None),
        "scale_normalize": bool(scale_normalize),
        "raise_gamma": (float(raise_gamma) if objective == "raise" else None),
        "n_pin_sites": (len(pin_values) if pin_values else None),
        "suppress_weight": float(suppress_weight),
        "n_delta_init": (sum(len(v) for v in delta_init.values())
                         if delta_init else None),
        "code_dtype": code_dtype,
        "inject_lambda": inj_lambda if objective == "inject" else None,
        "inject_exclude_sites": int(inject_exclude_sites) if objective == "inject" else None,
        **decomp,
        **delta_stats,
        "optimizer": optimizer,
        "weight_decay": float(weight_decay),
        "lr_schedule": lr_schedule,
        "lr_min_frac": float(lr_min_frac),
        "warmup_frac": float(warmup_frac),
        "theta_init_mode": theta_init_mode,
        "binarize": binarize,
        "anneal_reach_frac": float(anneal_reach_frac) if binarize == "anneal" else None,
        "site_lambda_weighted": site_lambda_weights is not None,
        "mask_floor_source": mask_floor_source,
        "mask_floor_sites": len(floors) if floors else 0,
        "dual_floor_weight": (float(dual_floor_weight) if zero_term
                              else None),
        "triple_floor_weight": (float(triple_floor_weight) if pos_floor
                                else None),
        "free_amplitude": bool(free_amplitude),
        "amp_l1": float(amp_l1) if free_amplitude else None,
        "amp_stats": _amp_stats,
        "amp_kept": _amp_kept,
        "dual_norm_shared": round(dual_norm, 6) if dual_floor else None,
        # diagnostics only — these no longer scale the loss
        "dual_norm_zero": float("%.6g" % norm_zero) if dual_floor else None,
        "dual_norm_floor": float("%.6g" % norm_floor) if dual_floor else None,
        "warmup_steps": int(n_warm),
        "lr_floor": float("%.6g" % lr_floor),
        "lr_peak": float(lr),
        "lr_sum": round(lr_sum, 4),
        # endpoints, not just the integral: decay and warmup share an lr_sum
        # but differ in DIRECTION, and direction is what moved circuit size.
        "lr_first": float("%.6g" % lr_at(0)),
        "lr_last": float("%.6g" % lr_at(int(steps) - 1)),
        "decay_product": round(decay_product, 4),
        "sparsity_product": float("%.4g" % sparsity_product),
        "batch_size_used": int(batch_size),      # effective batch (unchanged by guard)
        "micro_batch": int(micro_bs),
        "accum_chunks": int(accum),
        "steps": int(steps),
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "holdout_data_loss": ho,
        "n_kept": len(scores),
        "mean_m_final": float(mask_mean().detach()),
        # Mean m among KEPT members — the statistic that quantifies the
        # soft/hard gap. Training scales each member's contribution by its m;
        # evaluation uses binary membership, so members contribute their FULL
        # natural value. The shortfall (1 - mean_m_kept) is how much more the
        # evaluated circuit delivers than the trained one. mean_m_final is
        # dominated by the ~94% pruned toward 0 and cannot show this.
        "mean_m_kept": (round(sum(kept_m) / len(kept_m), 4) if kept_m else None),
        "min_m_kept": (round(min(kept_m), 4) if kept_m else None),
        # The probe split, exposed so callers can run their OWN evaluation on
        # exactly the sequences the optimiser never saw. A learned method can
        # memorise its probes in a way one-shot attribution cannot (measured:
        # mask_negctx gate recovery 0.87 train vs holdout loss 10.9), so
        # every recovery number must state which slice it came from — and a
        # caller re-deriving the split by hand would eventually drift from it.
        "n_train_pos": int(pt_tr.shape[0]),
        "n_holdout_pos": int(pt_ho.shape[0]),
        "n_train_neg": (int(nt_tr.shape[0])
                        if objective in ("contrast", "negctx", "inject") else None),
        "n_holdout_neg": (int(nt_ho.shape[0])
                          if objective in ("contrast", "negctx", "inject") else None),
    }
    if logger is not None:
        logger.note(f"learned_mask[{objective}]: kept {len(scores)} "
                    f"(loss {provenance['loss_initial']:.5f} -> "
                    f"{provenance['loss_final']:.5f}, holdout {ho})")
    patcher.release()
    if patcher_zero is not None:
        patcher_zero.release()
    if patcher_pos is not None:
        patcher_pos.release()
    return scores, provenance


__all__ = ["LearnedMaskPatcher", "run_learned_mask", "OBJECTIVES"]
