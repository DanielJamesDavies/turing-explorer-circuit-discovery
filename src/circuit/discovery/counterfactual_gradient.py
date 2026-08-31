import torch
import gc
from typing import Optional, Any, Tuple, Dict, Set, cast

from .gradient_base import DiscoveryContext, GradientDiscoveryBase, HopResult
from config import config
from store.circuits import Circuit
from store.latent_stats import latent_stats
from eval.minimality import prune_non_minimal_nodes_cf
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.instrument.attribution import compute_latent_counterfactual_scores
from circuit.instrument.ig_baseline import extract_signed_roles, integrated_baseline_scores
from circuit.types.feature_id import FeatureID
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx
from sae.dense import target_latent_activations


class SeedProjectionInstrument(SAEGraphInstrument):
    """
    SAEGraphInstrument subclass that captures the seed latent's encoder pre-activation
    during the forward pass.

    On contrast sequences the seed latent is typically absent from top-k, so
    f_connected[..., seed_latent_idx] is identically zero and backpropagating
    through it produces no gradient signal.  Instead we compute the seed's linear
    pre-activation directly from x — the natural SAE input at that (layer, kind):

        seed_pre_act = x @ W_enc[seed_latent_idx] + b_eff[seed_latent_idx]  -- [B, T]

    For an MLP SAE x is resid_pre + attn_out; for an attn SAE it is resid_pre;
    for a resid SAE it is the previous layer's resid_post.  Using x directly keeps
    the logic kind-agnostic and means seed_pre_act is non-zero even when the seed
    never appears in top-k.  It remains fully differentiable w.r.t. all upstream
    leaf anchors via the identity passthrough (x - x.detach()) that
    SAEGraphInstrument injects at each layer.
    """

    def __init__(
        self,
        bank: Any,
        seed_layer: int,
        seed_kind: str,
        w_seed: torch.Tensor,
        b_seed: torch.Tensor,
    ):
        super().__init__(bank)
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.w_seed = w_seed  # [d_model] — encoder row for the seed latent
        self.b_seed = b_seed  # scalar — effective encoder bias for the seed latent
        self.seed_pre_act: Optional[torch.Tensor] = None  # populated during forward: [B, T]

    def release(self) -> None:
        # seed_pre_act is graph-connected — it alone keeps the whole backward
        # graph (and every retained activation) alive.
        super().release()
        self.seed_pre_act = None

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        result = super().transform(layer_idx, kind, x)
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            # Capture AFTER super() so the identity passthrough (x - x.detach()) for
            # this layer is already in the computation graph.  x itself carries
            # gradients to upstream leaf anchors via previous layers' passthroughs.
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre_act = x @ w + b  # [B, T]
        return result


class SeedPreActCapture:
    """Capture-only patcher: computes the seed's encoder pre-activation
    ``x @ w_seed + b`` at the seed site and leaves the stream untouched
    everywhere.

    Exists because anchor-finding once reused SeedProjectionInstrument — a
    full attribution-graph instrument that materializes TWO dense
    ``[B, T, d_sae]`` copies at EVERY upstream site as a side effect. At the
    full 64-sequence width on a deep seed that was ~15.4GB (profiled: the
    single largest allocation of the whole contrastive arm, setting the
    allocator high-water for everything after it) to compute 64 argmaxes.

    Numerical note: the graph instrument's stream transform is an identity
    only up to float rounding (``decode(d) + x - decode(d)`` reassociates);
    this patcher reads the raw stream, so an argmax on a near-exact tie can
    in principle differ. Below run-to-run jitter in practice.
    """

    def __init__(self, seed_layer: int, seed_kind: str,
                 w_seed: torch.Tensor, b_seed: torch.Tensor) -> None:
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.w_seed = w_seed
        self.b_seed = b_seed
        self.seed_pre_act: Optional[torch.Tensor] = None  # [B, T]

    def __call__(self, model: Any):
        from model.hooks import multi_patch

        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre_act = x @ w + b
        return x


class CounterfactualGradientDiscovery(GradientDiscoveryBase):
    """
    Discovers circuit nodes by running gradient attribution on contrast sequences —
    inputs where the seed latent is inactive.

    The contrast sequence source is controlled by ``neg_mode`` (config):

    - ``"close"``   — hard negatives from neg_ctx: semantically similar to posctx
                      but with the seed absent (original behaviour).
    - ``"random"``  — random real corpus sequences from the saved global neg_ctx set.
    - ``"distant"`` — corpus sequences most distant from posctx in SAE latent space
                      at the seed's layer, filtered to non-activating sequences.
                      (Implemented in Phase 3.)

    Two node types are discovered regardless of mode:
    - counterfactual_activator: upstream latents with large positive raw gradient
      (they would cause the seed to fire if active).
    - counterfactual_inhibitor: upstream latents with negative acts×gradient
      (they are active and causally suppressing the seed).

    Evaluation always runs on posctx, testing whether the discovered nodes explain
    the seed's activation on its own context.
    """

    method_name = "counterfactual_gradient"
    circuit_name_prefix = "CounterfactualGrad"
    positive_role = "counterfactual_activator"
    empty_reject_message = "no activators or inhibitors passed threshold"

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        top_k_activators: Optional[int] = None,
        top_k_inhibitors: Optional[int] = None,
        top_k_scope: Optional[str] = None,
        activator_threshold: Optional[float] = None,
        inhibitor_threshold: Optional[float] = None,
        min_active_count: Optional[int] = None,
        max_neg_sequences: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
        min_faithfulness: Optional[float] = None,
        attribution_mode: Optional[str] = None,
        ig_steps: Optional[int] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.counterfactual_gradient
        self._init_shared_knobs(
            cfg,
            attribution_mode=attribution_mode,
            ig_steps=ig_steps,
            min_active_count=min_active_count,
            max_neg_sequences=max_neg_sequences,
            pruning_threshold=pruning_threshold,
            top_k_scope=top_k_scope,
        )
        self.activator_signal = cast(str, cfg.activator_signal)
        self.ig_negctx_objective = cast(str, cfg.ig_negctx_objective)
        self.restoration_negctx_mode = cast(str, cfg.restoration_negctx_mode)
        self.ig_negctx_deep_site_threshold = cast(int, cfg.ig_negctx_deep_site_threshold)
        self.ig_negctx_deep_neg_batch = cast(int, cfg.ig_negctx_deep_neg_batch)
        self.top_k_activators = (
            top_k_activators if top_k_activators is not None
            else cast(int, cfg.top_k_activators)
        )
        self.top_k_inhibitors = (
            top_k_inhibitors if top_k_inhibitors is not None
            else cast(int, cfg.top_k_inhibitors)
        )
        self.activator_threshold = (
            activator_threshold if activator_threshold is not None
            else cast(float, cfg.activator_threshold)
        )
        self.inhibitor_threshold = (
            inhibitor_threshold if inhibitor_threshold is not None
            else cast(float, cfg.inhibitor_threshold)
        )
        self.neg_batch_size = cast(int, cfg.neg_batch_size)
        self.min_faithfulness = (
            min_faithfulness if min_faithfulness is not None
            else cast(float, cfg.min_faithfulness)
        )

    # ------------------------------------------------------------------
    # Pipeline hooks (the template itself lives in GradientDiscoveryBase)
    # ------------------------------------------------------------------

    def _prepare(self, ctx: DiscoveryContext, logger: CircuitLogger) -> bool:
        """Source the contrast sequences (per neg_mode) and the seed's posctx
        target activation, which scales the admission thresholds."""
        neg_tokens = self._get_neg_tokens(
            ctx.probe_data,
            ctx.seed_comp_idx, ctx.seed_latent_idx,
            logger,
        )
        if neg_tokens is None:
            return False
        ctx.neg_tokens = neg_tokens

        target_act_pos = self._get_posctx_activation(
            ctx.seed_comp_idx, ctx.seed_latent_idx,
            ctx.pos_tokens_probe, ctx.pos_argmax_probe,
        )
        # Scale thresholds by target_act_pos so focal seeds (lower a_posctx) are not
        # disproportionately penalised. Gradient scores ≈ 2·a_posctx·(alignment), so
        # an absolute threshold is ~4× stricter for a seed with a_posctx=1 vs a_posctx=4.
        act_scale = max(target_act_pos, 0.1)
        ctx.target_act_pos = target_act_pos
        ctx.effective_activator_threshold = self.activator_threshold * act_scale
        ctx.effective_inhibitor_threshold = self.inhibitor_threshold * act_scale
        logger.note(
            f"target_act_pos: {target_act_pos:.4f} | "
            f"effective thresholds — activator: {ctx.effective_activator_threshold:.4f}, "
            f"inhibitor: {ctx.effective_inhibitor_threshold:.4f}"
        )
        return True

    def _run_attribution_hop(self, ctx: DiscoveryContext, logger: CircuitLogger) -> HopResult:
        # "local" runs the contrast-sequence gradient hop at the live negctx
        # input; "ig_mean" attributes along the mean-ablation-floor ->
        # natural-posctx path instead (SFC-style), in which case negctx is used
        # only by the evaluation step.
        # `position_aware` is a MODIFIER on whichever attribution runs below (it
        # swaps that method's position-collapse for a union over the seed's causal
        # prefix) — not a method of its own. (The baseline-free posctx
        # grad x natural attribution is its own method, ActivationGradientDiscovery,
        # not a counterfactual mode: it runs on posctx and cannot find absent
        # activators, so it never answered cf's question.)
        if self.attribution_mode == "ig_mean":
            activator_scores, inhibitor_scores = self._run_ig_mean_hop(
                ctx.seed_comp_idx, ctx.seed_latent_idx,
                ctx.pos_tokens_probe, ctx.pos_argmax_probe,
                ctx.target_act_pos, logger,
            )
            pass_label = "ig_mean grad pass"
        elif self.attribution_mode == "ig_negctx":
            activator_scores, inhibitor_scores = self._run_ig_negctx_hop(
                ctx.seed_comp_idx, ctx.seed_latent_idx, ctx.neg_tokens,
                ctx.pos_tokens_probe, ctx.pos_argmax_probe, ctx.target_act_pos, logger,
            )
            pass_label = f"ig_negctx/{self.ig_negctx_objective} grad pass"
        elif self.attribution_mode == "restoration_negctx":
            activator_scores, inhibitor_scores = self._run_restoration_negctx_hop(
                ctx.seed_comp_idx, ctx.seed_latent_idx, ctx.neg_tokens,
                ctx.pos_tokens_probe, ctx.pos_argmax_probe, ctx.target_act_pos, logger,
            )
            pass_label = f"restoration_negctx/{self.ig_negctx_objective} selection"
        elif self.attribution_mode in ("mask_contrast", "mask_negctx",
                                       "mask_inject"):
            activator_scores, inhibitor_scores = self._run_mask_hop(
                ctx.seed_comp_idx, ctx.seed_latent_idx, ctx.neg_tokens,
                ctx.pos_tokens_probe, ctx.pos_argmax_probe, ctx.target_act_pos, logger,
            )
            pass_label = f"{self.attribution_mode} optimisation"
        elif self.attribution_mode in ("restoration", "ig_restoration"):
            # Full probe_sequence_count: the round scorer chunks internally
            # at probe_batch_size (see _round_scores).
            activator_scores, inhibitor_scores = self._run_restoration_hop(
                ctx.seed_comp_idx, ctx.seed_latent_idx,
                ctx.pos_tokens_probe, ctx.pos_argmax_probe,
                ctx.target_act_pos, logger,
            )
            pass_label = f"{self.attribution_mode} selection"
        else:
            activator_scores, inhibitor_scores = self._run_contrast_hop(
                ctx.seed_comp_idx, ctx.seed_latent_idx, ctx.neg_tokens,
                ctx.target_act_pos, logger,
                pos_tokens=ctx.pos_tokens_probe, pos_argmax=ctx.pos_argmax_probe,
            )
            pass_label = f"{self.neg_mode} grad pass"
            if self.activator_signal == "gradient_x_posctx":
                pass_label += " (grad x posctx)"
        if self.position_aware:
            pass_label += " (position-aware union)"
        # Role semantics, applied ONCE for every cf mode: PA keeps both signs
        # (stream-reconstruction membership — exclude only unlabels the
        # inhibitors), NPA-exclude genuinely drops them (the activator-only
        # ablation study; the cf-local φcf collapse finding lives here).
        from circuit.instrument.position_aware import resolve_role_delivery
        activator_scores, inhibitor_scores = resolve_role_delivery(
            activator_scores, inhibitor_scores,
            position_aware=self.position_aware,
            include_negatives=self.negative_roles == "include",
        )
        logger.stage(
            pass_label,
            1, 0,
            note=(
                f"{len(activator_scores)} absent activators, "
                f"{len(inhibitor_scores)} present inhibitors before thresholding"
            ),
        )
        return HopResult(positives=activator_scores, negatives=inhibitor_scores)

    def _pre_assembly(self, ctx: DiscoveryContext, hop: HopResult) -> None:
        """Churn-fix #1: ONE vectorized active_count gather for every hop
        candidate, replacing a per-member ``active_count[comp, latent] <
        min_active_count`` tensor comparison inside the admission loop (~one
        host-device sync per member — the dominant assembly cost on
        position-aware circuits). Semantics identical: a candidate passes iff
        NOT (active_count < min_active_count)."""
        fids = list(hop.positives)
        fids.extend(f for f in hop.negatives if f not in hop.positives)
        if not fids:
            self._active_ok = frozenset()
            return
        comps = []
        latents = []
        for fid in fids:
            comp, latent = fid.to_component_id(ctx.n_kinds, ctx.kinds)
            comps.append(comp)
            latents.append(latent)
        # Module-global latent_stats so tests can patch
        # counterfactual_gradient.latent_stats.
        active_count = latent_stats.active_count
        idx_comp = torch.as_tensor(comps, dtype=torch.long, device=active_count.device)
        idx_lat = torch.as_tensor(latents, dtype=torch.long, device=active_count.device)
        passes = ~(active_count[idx_comp, idx_lat] < self.min_active_count)
        self._active_ok = frozenset(
            fid for fid, ok in zip(fids, passes.cpu().tolist()) if ok
        )

    def _admit_positive(self, ctx: DiscoveryContext, fid: FeatureID, score: float) -> bool:
        if score < ctx.effective_activator_threshold:
            return False
        return fid in self._active_ok

    def _admit_negative(self, ctx: DiscoveryContext, fid: FeatureID, score: float) -> bool:
        if abs(score) < ctx.effective_inhibitor_threshold:
            return False
        return fid in self._active_ok

    def _assembly_note(self, n_pos: int, n_neg: int) -> str:
        return f"{n_pos} activators, {n_neg} inhibitors after thresholding"

    def _eval_neg_tokens(
        self, ctx: DiscoveryContext, logger: CircuitLogger
    ) -> Optional[torch.Tensor]:
        # Evaluation reuses the SAME contrast sequences discovery attributed on.
        return ctx.neg_tokens

    def _log_assembly_complete(self, circuit: Circuit, circuit_layers: Set[int]) -> None:
        print(
            f"[CounterfactualGrad] Discovery complete: {len(circuit.nodes)} nodes, "
            f"{len(circuit.edges)} edges | circuit_layers={sorted(circuit_layers)}"
        )

    def _call_loo_prune(
        self,
        ctx: DiscoveryContext,
        circuit: Circuit,
        neg_tokens_eval: torch.Tensor,
        circuit_layers: Set[int],
    ) -> None:
        # Minimality pruning — uses cf_faith as the leave-one-out signal.
        # Referenced from THIS module so tests can patch
        # counterfactual_gradient.prune_non_minimal_nodes_cf.
        prune_non_minimal_nodes_cf(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            neg_tokens=neg_tokens_eval,
            pos_tokens=ctx.pos_tokens_eval,
            seed_layer=ctx.seed_layer,
            seed_kind=ctx.seed_kind,
            seed_latent_idx=ctx.seed_latent_idx,
            pos_argmax=ctx.pos_argmax_eval,
            threshold=self.pruning_threshold,
            circuit_layers=circuit_layers,
        )

    def _run_faithfulness_eval(
        self,
        ctx: DiscoveryContext,
        circuit: Circuit,
        neg_tokens_eval: torch.Tensor,
        circuit_layers: Set[int],
    ) -> Tuple[float, float]:
        # Referenced from THIS module so tests can patch
        # counterfactual_gradient.evaluate_counterfactual_faithfulness.
        return evaluate_counterfactual_faithfulness(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            neg_tokens=neg_tokens_eval,
            pos_tokens=ctx.pos_tokens_eval,
            seed_layer=ctx.seed_layer,
            seed_kind=ctx.seed_kind,
            seed_latent_idx=ctx.seed_latent_idx,
            pos_argmax=ctx.pos_argmax_eval,
            circuit_layers=circuit_layers,
        )

    def _accept(self, cf_faith: float, sup_score: float) -> Optional[str]:
        # Mask modes bypass the threshold — see GradientDiscoveryBase._accept.
        if getattr(self, "attribution_mode", "") in (
                "mask", "mask_contrast", "mask_negctx", "mask_inject"):
            return None
        if cf_faith < self.min_faithfulness:
            return (
                f"counterfactual_faithfulness {cf_faith:.4f} < "
                f"min_faithfulness {self.min_faithfulness}"
            )
        return None

    def _extra_metadata(
        self,
        ctx: DiscoveryContext,
        hop: HopResult,
        n_pos: int,
        n_neg: int,
        sup_score: float,
    ) -> Dict[str, Any]:
        return {"n_activators": n_pos, "n_inhibitors": n_neg}

    def _run_mask_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        neg_tokens: torch.Tensor,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        target_act_pos: float,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
        """The cf-hosted learned-mask modes.

        (Negctx-needing floors skip seeds with no negatives — see the
        matching guard in ablation_gradient._run_mask_hop.)

        mask_contrast: reconstruction on posctx PLUS beta-weighted silence on
        negctx — selectivity, not just drive. All kept members are supports
        (m >= 0; the mask never asks for a sign).
        mask_negctx: pure gate-opening on negctx — the minimal EDIT to the
        natural negctx stream that fires the seed at the posctx level. Kept
        members carry negative scores (-(1 - m)) and are delivered as
        INHIBITORS: latents whose presence holds the seed off, i.e. exactly
        the set the cf eval suppresses on negctx.
        """
        if self.position_aware:
            raise ValueError(
                f"attribution_mode={self.attribution_mode!r} does not support "
                "position_aware yet — failing loudly rather than silently "
                "ignoring the flag."
            )
        from circuit.instrument.learned_mask import run_learned_mask
        from eval.ablation_faithfulness import upstream_sites

        n_kinds = len(self.sae_bank.kinds)
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = self.sae_bank.kinds[seed_kind_idx]
        sites = sorted(upstream_sites(self.sae_bank, seed_layer, seed_kind))
        if not sites:
            logger.note("mask: seed has no upstream sites")
            return {}, {}
        cfg = config.discovery.learned_mask
        from circuit.instrument.learned_mask import FLOORS_NEEDING_NEGATIVES
        if (cfg.mask_floor_source in FLOORS_NEEDING_NEGATIVES
                and (neg_tokens is None or int(neg_tokens.shape[0]) == 0)):
            logger.note("mask: floor %r needs negctx but seed has none — "
                        "skipped" % cfg.mask_floor_source)
            return {}, {}
        objective = {"mask_contrast": "contrast", "mask_negctx": "negctx",
                     "mask_inject": "inject"}[self.attribution_mode]
        scores, prov = run_learned_mask(
            self.inference, self.sae_bank,
            objective=objective, sites=sites,
            seed_layer=seed_layer, seed_kind=seed_kind,
            seed_latent_idx=seed_latent_idx,
            pos_tokens=pos_tokens, pos_argmax=pos_argmax,
            neg_tokens=neg_tokens, target_act=target_act_pos,
            steps=cfg.steps, lr=cfg.lr, l1_lambda=cfg.l1_lambda, beta=cfg.beta,
            inject_lambda=cfg.inject_lambda,
            inject_exclude_sites=cfg.inject_exclude_sites,
            keep_threshold=cfg.keep_threshold,
            batch_size=self.probe_batch_size,
            holdout_frac=cfg.holdout_frac, theta_init=cfg.theta_init,
            log_every=cfg.log_every,
            deep_site_threshold=cfg.deep_site_threshold,
            deep_batch_size=cfg.deep_batch_size,
            optimizer=cfg.optimizer, weight_decay=cfg.weight_decay,
            code_dtype=cfg.code_dtype,
            lr_schedule=cfg.lr_schedule, lr_min_frac=cfg.lr_min_frac,
            warmup_frac=cfg.warmup_frac,
            mask_floor_source=cfg.mask_floor_source,
            dual_floor_weight=cfg.dual_floor_weight,
            triple_floor_weight=cfg.triple_floor_weight,
            free_amplitude=cfg.free_amplitude,
            amp_l1=cfg.amp_l1,
            signed_amplitude=cfg.signed_amplitude,
            neg_suppress_weight=cfg.neg_suppress_weight,
            margin_topk=cfg.margin_topk,
            binarize=cfg.binarize,
            logger=logger,
        )
        self._stash_amplitudes(prov)
        if objective == "negctx":
            return {}, scores          # all edits, delivered as inhibitors
        if objective == "inject":
            # both C1 roles at once: split by sign — positive deltas are the
            # learned absent activators, negative edits the learned present
            # inhibitors.
            acts = {f: v for f, v in scores.items() if v > 0}
            inhs = {f: v for f, v in scores.items() if v < 0}
            return acts, inhs
        return scores, {}              # all supports

    def _get_neg_tokens(
        self,
        probe_data: Any,
        seed_comp_idx: int,
        seed_latent_idx: int,
        logger: CircuitLogger,
    ) -> Optional[torch.Tensor]:
        """
        Returns the contrast token batch ``[N, 64]`` for the gradient attribution pass,
        according to ``self.neg_mode``:

        - ``"close"``   — closest non-activating sequences from global neg_ctx.
        - ``"random"``  — random real sequences from global neg_ctx.
        - ``"distant"`` — most distant non-activating sequences from global neg_ctx.
        """
        del probe_data
        selection = self._select_neg_context(
            seed_comp_idx,
            seed_latent_idx,
            self.neg_mode,
            self.max_neg_sequences,
            self.neg_batch_size,
            logger,
        )
        if selection is None:
            return None
        self._last_neg_selection_metadata = selection.metadata
        return selection.tokens

    def _select_neg_context(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        mode: str,
        max_neg_sequences: int,
        batch_size: int,
        logger: CircuitLogger,
    ):
        cfg = config.discovery.neg_context_selection
        candidate_pool_size = (
            self.distant_pool_size if mode == "distant" else cfg.candidate_pool_size
        )
        return self._neg_context_selector().select(
            seed_comp_idx,
            seed_latent_idx,
            mode,
            max_sequences=max_neg_sequences,
            batch_size=batch_size,
            candidate_pool_size=candidate_pool_size,
            exact=bool(cfg.exact_negctx_ranking),
            non_activation_threshold=float(cfg.non_activation_threshold),
            preact_filter=bool(cfg.preact_filter),
            preact_select=str(cfg.preact_select),
            preact_max_frac=float(cfg.preact_max_frac),
            posctx_reference=self._posctx_preact_reference,
            selection_seed=int(cfg.selection_seed),
            filter_batch_size=int(cfg.filter_batch_size),
            load_window_size=int(cfg.load_window_size),
            logger=logger,
        )

    def _get_posctx_activation(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
    ) -> float:
        """
        Runs a no-grad forward on pos_tokens and returns the seed latent's mean SAE
        activation at the pos_argmax positions — used as target_act_pos for the MSE loss.
        """
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]

        captured: list = []

        def capture_hook(layer_idx: int, activations: tuple) -> None:
            if layer_idx != seed_layer:
                return
            act = activations[seed_kind_idx]
            top_acts, top_indices = self.sae_bank.encode(act, seed_kind, layer_idx)
            target_acts = target_latent_activations(top_acts, top_indices, seed_latent_idx)  # [B, T]
            B = target_acts.shape[0]
            batch_idx = torch.arange(B, device=target_acts.device)
            pa = pos_argmax[:B].to(target_acts.device).clamp(0, target_acts.shape[1] - 1)
            val = target_acts[batch_idx, pa].mean().item()
            captured.append(val)

        self.inference.disable_compile()
        try:
            with torch.no_grad():
                self.inference.forward(
                    pos_tokens,
                    activations_callback=capture_hook,
                    return_activations=False,
                    tokenize_final=False,
                )
        finally:
            self.inference.enable_compile()

        return float(captured[0]) if captured else 0.0

    def _run_restoration_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        target_act_pos: float,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
        """The shared restoration engine under cf's profile: the gap target is
        the posctx target activation already measured by _prepare; signed roles
        returned directly. See RestorationConfig."""

        n_kinds = len(self.sae_bank.kinds)
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = self.sae_bank.kinds[seed_kind_idx]

        positives, negatives, result = self._restoration_selection(
            seed_layer, seed_kind, seed_latent_idx, pos_tokens, pos_argmax, target_act_pos
        )
        if result is None:
            logger.note("restoration: seed has no upstream sites")
        else:
            logger.note(
                f"restoration: rounds_used={result.rounds_used} "
                f"stopped_early={result.stopped_early} "
                f"metric {result.metric_trajectory[0]:.4f} -> {result.metric_trajectory[-1]:.4f}"
            )
        return positives, negatives

    def _run_ig_mean_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        target_act_pos: float,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
        """The shared IG-from-floor engine under cf's profile: "gap" objective
        against the posctx target (candidate scores linearise the circuit-only
        counterfactual), activator/inhibitor budgets, signed roles returned
        directly."""

        n_kinds = len(self.sae_bank.kinds)
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = self.sae_bank.kinds[seed_kind_idx]

        result = self._integrated_baseline_attribution(
            seed_layer, seed_kind, seed_latent_idx, pos_tokens, pos_argmax,
            objective="gap",
            target_act=target_act_pos,
            top_k_positive=self.top_k_activators,
            top_k_negative=self.top_k_inhibitors,
            metric_label="metric",
            logger=logger,
        )
        if result is None:
            return {}, {}
        positives, negatives, _metric_floor, _metric_natural = result
        return positives, negatives

    def _ig_negctx_batch(self, n_sites: int) -> int:
        """Effective neg microbatch for the contrastive path integral.

        ig_negctx's per-site residency (leaf + grad + fp32 delta + fp32
        per-position accumulator) is batch-proportional and held for ALL
        upstream sites at once, so deep seeds cross the card: measured peak
        ~= 7G + sites x 252MB at B=8 (fits at L7's 23 sites, pages at L10's
        29). Above the threshold the microbatch drops to
        ig_negctx_deep_neg_batch — halving every per-site tensor at the
        cost of more chunks — while shallow seeds keep the configured
        neg_batch_size and pay no extra chunk overhead."""
        if n_sites <= self.ig_negctx_deep_site_threshold:
            return self.neg_batch_size
        return max(1, min(self.neg_batch_size, self.ig_negctx_deep_neg_batch))

    def _negctx_anchor(
        self,
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
        neg_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """The seed's would-be-firing position per negctx sequence ``[B]`` —
        the pre-activation argmax, the same anchor the local contrast hop
        uses. One no-grad forward on neg_tokens with a CAPTURE-ONLY patcher:
        the stream is untouched and no site is densified (the previous
        graph-instrument reuse built ~15.4GB of dense codes at full width to
        answer this; see SeedPreActCapture)."""
        sae = self.sae_bank.saes[seed_kind][seed_layer]
        w_seed = sae.encoder.weight[seed_latent_idx].detach()
        b_seed = sae._get_bias_eff()[seed_latent_idx].detach()
        instrument = SeedPreActCapture(seed_layer, seed_kind, w_seed, b_seed)
        self.inference.disable_compile()
        try:
            with torch.no_grad():
                self.inference.forward(
                    neg_tokens,
                    patcher=instrument,
                    grad_enabled=False,
                    return_activations=False,
                    tokenize_final=False,
                )
        finally:
            self.inference.enable_compile()
        if instrument.seed_pre_act is None:
            raise RuntimeError("seed pre-activation was not captured on negctx")
        return instrument.seed_pre_act.argmax(dim=-1)

    def _run_ig_negctx_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        neg_tokens: torch.Tensor,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        target_act_pos: float,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
        """Integrated gradients along the LATENT-SPACE path from the negctx
        state (alpha=0) to the posctx target (alpha=1), run on negctx tokens.

        The exact estimator of what the "local" contrast hop linearises at a
        single point: instead of one gradient at the live negctx input, the
        gradient is averaged along the straight path that slides every
        upstream latent from its negctx value to the posctx value the
        counterfactual-faithfulness eval injects — so alpha=1 IS the eval's
        intervened state (negctx residuals held in place), and by IG
        completeness the attributions sum to the seed's actual change under
        that intervention (logged as the certificate by
        integrated_baseline_scores). The contrast lives in the path's
        endpoints rather than an MSE loss; the metric along the path is
        selected by ``ig_negctx_objective`` ("drive" | "gap"). See
        dev-notes/contrastive-ig-for-position-aware-cf.md.

        cf-only by construction: ablation gradient has no contrast input to
        anchor such a path. Costs ig_steps+1 forwards + ig_steps backwards on
        neg_tokens, plus one clean pass each on pos_tokens (targets) and
        neg_tokens (anchor).
        """
        from eval.ablation_faithfulness import upstream_sites

        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]

        sites = upstream_sites(self.sae_bank, seed_layer, seed_kind)
        if not sites:
            logger.note("ig_negctx: seed has no upstream sites")
            return {}, {}
        neg_batch = self._ig_negctx_batch(len(sites))
        if neg_batch != self.neg_batch_size:
            logger.note(
                f"ig_negctx: {len(sites)} sites > "
                f"{self.ig_negctx_deep_site_threshold} — neg microbatch "
                f"{self.neg_batch_size} -> {neg_batch} (per-site residency "
                f"~252MB x sites at B=8 crosses a 16GB card near 29 sites)"
            )

        targets = self._posctx_targets(seed_layer, seed_kind, pos_tokens, pos_argmax)
        neg_anchor = self._negctx_anchor(seed_layer, seed_kind, seed_latent_idx, neg_tokens)

        sae = self.sae_bank.saes[seed_kind][seed_layer]
        w_seed = sae.encoder.weight[seed_latent_idx].detach()
        b_seed = sae._get_bias_eff()[seed_latent_idx].detach()

        scores_by_site, metric_neg, metric_injected = integrated_baseline_scores(
            self.inference,
            self.sae_bank,
            tokens=neg_tokens,
            substitute_sites=sites,
            # The dense endpoint: posctx targets, NOT a floor — path below
            # flips the direction so these sit at alpha=1.
            site_floors=targets,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            w_seed=w_seed,
            b_seed=b_seed,
            pos_argmax=neg_anchor,
            position_aware=self._position_aware_spec(neg_anchor),
            objective=self.ig_negctx_objective,
            target_act=target_act_pos,
            ig_steps=self.ig_steps,
            path="from_natural",
            batch_size=neg_batch,
        )
        logger.note(
            f"ig_negctx/{self.ig_negctx_objective}: metric negctx "
            f"{metric_neg:.4f} -> injected {metric_injected:.4f} over {len(sites)} sites, "
            f"{self.ig_steps} steps"
            + (" (position-aware union)" if self.position_aware else "")
        )
        # Position-aware scores are already the union (same as the ig hop).
        # Extract BOTH signs; _run_attribution_hop applies the role semantics
        # (resolve_role_delivery) centrally for every cf mode.
        no_trunc = int(self.sae_bank.d_sae)
        top_pos = no_trunc if self.position_aware else self.top_k_activators
        top_neg = no_trunc if self.position_aware else self.top_k_inhibitors
        return extract_signed_roles(
            scores_by_site,
            kinds=list(kinds),
            n_kinds=n_kinds,
            top_k_positive=top_pos,
            top_k_negative=top_neg,
            min_active_count=self.min_active_count,
            active_count=latent_stats.active_count,
            top_k_scope=self.top_k_scope,
        )

    def _run_restoration_negctx_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        neg_tokens: torch.Tensor,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        target_act_pos: float,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
        """The restoration loop on ig_negctx's trajectory: greedy iterated
        selection on negctx tokens, restored latents pinned to their posctx
        targets (the cf eval's injection), each round re-linearising
        grad x (target - live value) at the current injected state. The
        certificate closing means the selected set makes the seed fire on
        negctx under injection. Shares ig_negctx's anchors (posctx targets,
        negctx would-be-firing positions) and its deep-site neg microbatch;
        the loop knobs come from RestorationConfig, the backward objective
        from ig_negctx_objective. See run_negctx_restoration_selection.
        """

        from circuit.instrument.restoration import run_negctx_restoration_selection
        from eval.ablation_faithfulness import upstream_sites

        n_kinds = len(self.sae_bank.kinds)
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = self.sae_bank.kinds[seed_kind_idx]

        sites = upstream_sites(self.sae_bank, seed_layer, seed_kind)
        if not sites:
            logger.note("restoration_negctx: seed has no upstream sites")
            return {}, {}
        neg_batch = self._ig_negctx_batch(len(sites))
        if neg_batch != self.neg_batch_size:
            logger.note(
                f"restoration_negctx: {len(sites)} sites > "
                f"{self.ig_negctx_deep_site_threshold} — neg microbatch "
                f"{self.neg_batch_size} -> {neg_batch} (per-site residency "
                f"matches ig_negctx's)"
            )

        targets = self._posctx_targets(seed_layer, seed_kind, pos_tokens, pos_argmax)
        neg_anchor = self._negctx_anchor(seed_layer, seed_kind, seed_latent_idx, neg_tokens)

        positives, negatives, result = run_negctx_restoration_selection(
            self.inference,
            self.sae_bank,
            neg_tokens=neg_tokens,
            neg_anchor=neg_anchor,
            inject_targets=targets,
            sites=sites,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_latent_idx=seed_latent_idx,
            target_act=target_act_pos,
            rounds=self.restoration_rounds,
            per_round_k=self.restoration_per_round_k,
            certificate_tol=self.restoration_certificate_tol,
            # Same gate as _restoration_selection: PA keeps both signs in the
            # loop (role split applied after, via resolve_role_delivery).
            allow_negative=self.negative_roles == "include" or self.position_aware,
            objective=self.ig_negctx_objective,
            inject_mode=self.restoration_negctx_mode,
            round_select=self.restoration_round_select,
            round_abs_pctl=self.restoration_round_abs_pctl,
            position_aware=self.position_aware,
            batch_size=neg_batch,
        )
        self._last_restoration = result
        if result is not None:
            logger.note(
                f"restoration_negctx/{self.ig_negctx_objective}: "
                f"rounds_used={result.rounds_used} "
                f"stopped_early={result.stopped_early} "
                f"metric {result.metric_trajectory[0]:.4f} -> {result.metric_trajectory[-1]:.4f}"
            )
        return positives, negatives

    def _collect_posctx_values(
        self,
        seed_layer: int,
        seed_kind: str,
        pos_tokens: Optional[torch.Tensor],
        pos_argmax: Optional[torch.Tensor],
        logger: CircuitLogger,
    ) -> Optional[Dict[Tuple[int, str], torch.Tensor]]:
        """Per-site posctx target values ``[d_sae]`` for the
        ``activator_signal="gradient_x_posctx"`` ranking, or None when the raw
        gradient (the classic signal) is selected.

        These are `collect_site_anchors`' collapsed pins — the mean dense latent
        value at the probe positions — which is exactly the value
        `evaluate_counterfactual_faithfulness` injects for each activator in its
        Pass 3. Scaling the gradient by them makes discovery rank the
        intervention the eval performs rather than the seed's per-unit
        sensitivity to it. Costs one extra clean forward pass per seed.
        """
        if self.activator_signal != "gradient_x_posctx":
            return None
        if pos_tokens is None:
            raise ValueError(
                "activator_signal='gradient_x_posctx' needs pos_tokens to collect "
                "the posctx target values"
            )
        pins = self._posctx_targets(seed_layer, seed_kind, pos_tokens, pos_argmax)
        logger.note(
            f"activator_signal=gradient_x_posctx: posctx targets over {len(pins)} sites"
        )
        return pins

    def _posctx_targets(
        self,
        seed_layer: int,
        seed_kind: str,
        pos_tokens: torch.Tensor,
        pos_argmax: Optional[torch.Tensor],
    ) -> Dict[Tuple[int, str], torch.Tensor]:
        """Per-site collapsed posctx pins ``[d_sae]`` — the mean dense latent
        value at the probe positions, i.e. exactly what
        `evaluate_counterfactual_faithfulness` injects per activator in Pass 3.
        Shared by the gradient_x_posctx activator signal (as the ranking's
        scale) and by ig_negctx (as the path's alpha=1 endpoint)."""
        from eval.ablation_faithfulness import upstream_sites
        from eval.floors import collect_site_anchors

        sites = upstream_sites(self.sae_bank, seed_layer, seed_kind)
        if not sites:
            return {}
        _, pins = collect_site_anchors(
            self.inference, self.sae_bank, pos_tokens, sites, pos_argmax,
            pin_position_specific=False,
        )
        return pins

    def _run_contrast_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        neg_tokens: torch.Tensor,
        target_act_pos: float,
        logger: CircuitLogger,
        pos_tokens: Optional[torch.Tensor] = None,
        pos_argmax: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
        """
        Runs grad-enabled forward passes on the contrast sequences using
        SeedProjectionInstrument, then calls compute_latent_counterfactual_scores
        to extract absent activators and present inhibitors.

        ``pos_tokens``/``pos_argmax`` are only read when ``activator_signal`` is
        "gradient_x_posctx", to collect the posctx targets the activator scores
        are scaled by; the gradients themselves are always taken on negctx.

        When ``neg_tokens`` exceeds ``neg_batch_size`` the sequences are split into
        microbatches; scores are averaged across all batches so the result is
        comparable to a single-batch run.  Setting ``neg_batch_size`` equal to (or
        greater than) ``max_neg_sequences`` restores the original single-pass behaviour.

        Works identically for all neg_mode values — the only difference is which
        tokens are passed in.
        """
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]

        # Build encoder direction vectors once — reused across microbatches
        sae = self.sae_bank.saes[seed_kind][seed_layer]
        w_seed = sae.encoder.weight[seed_latent_idx].detach()  # [d_model]
        b_seed = sae._get_bias_eff()[seed_latent_idx].detach()  # scalar

        # Collected once (not per microbatch): the same targets for every batch.
        posctx_values = self._collect_posctx_values(
            seed_layer, seed_kind, pos_tokens, pos_argmax, logger
        )

        all_act_scores: Dict[FeatureID, list] = {}
        all_inh_scores: Dict[FeatureID, list] = {}
        n_valid_batches = 0
        total_seqs = neg_tokens.shape[0]

        self.inference.disable_compile()
        try:
            for batch_start in range(0, total_seqs, self.neg_batch_size):
                batch = neg_tokens[batch_start : batch_start + self.neg_batch_size]
                instrument = SeedProjectionInstrument(
                    self.sae_bank, seed_layer, seed_kind, w_seed, b_seed
                )
                try:
                    self.inference.forward(
                        batch,
                        patcher=instrument,
                        grad_enabled=True,
                        return_activations=False,
                        tokenize_final=False,
                    )

                    if instrument.seed_pre_act is None:
                        logger.note(
                            f"SeedProjectionInstrument: seed_pre_act is None "
                            f"(batch offset {batch_start})"
                        )
                        continue

                    B = instrument.seed_pre_act.shape[0]
                    batch_idx = torch.arange(B, device=instrument.seed_pre_act.device)
                    pos_argmax_neg = instrument.seed_pre_act.argmax(dim=-1)  # [B]

                    pre_act_at_peak = instrument.seed_pre_act[batch_idx, pos_argmax_neg]  # [B]
                    target_tensor = torch.tensor(
                        target_act_pos,
                        device=pre_act_at_peak.device,
                        dtype=pre_act_at_peak.dtype,
                    )
                    # MSE loss: measures how far contrast pre-activation is from posctx target.
                    # target_scalar = -loss so gradients point toward increasing pre_act.
                    loss = ((pre_act_at_peak - target_tensor) ** 2).mean()
                    target_scalar = -loss

                    if abs(target_scalar.item()) < 1e-8:
                        logger.note(
                            f"near-zero target_scalar ({self.neg_mode}, "
                            f"batch offset {batch_start}) — skipping"
                        )
                        continue

                    # Log MSE details on the first valid batch only
                    if n_valid_batches == 0:
                        logger.note(
                            f"{self.neg_mode} MSE loss: {loss.item():.4f} | "
                            f"target_act_pos: {target_act_pos:.4f} | "
                            f"{self.neg_mode} pre_act mean: {pre_act_at_peak.mean().item():.4f}"
                        )

                    batch_act, batch_inh = compute_latent_counterfactual_scores(
                        graph=instrument.graph,
                        target_scalar=target_scalar,
                        seed_layer=seed_layer,
                        n_kinds=n_kinds,
                        kinds=kinds,
                        top_k_activators=self.top_k_activators,
                        top_k_inhibitors=self.top_k_inhibitors,
                        min_active_count=self.min_active_count,
                        active_count=latent_stats.active_count,
                        top_k_scope=self.top_k_scope,
                        # Position-aware cf: same contrast objective and gradients,
                        # but the upstream position axis is unioned over the seed's
                        # causal prefix instead of summed away. The anchor is the
                        # seed's would-be-firing position on negctx (pre-act argmax),
                        # which this hop already computes.
                        position_aware=self._position_aware_spec(pos_argmax_neg),
                        # None unless activator_signal="gradient_x_posctx", in which
                        # case activator scores become grad x posctx target.
                        posctx_values=posctx_values,
                    )

                    for fid, score in batch_act.items():
                        all_act_scores.setdefault(fid, []).append(score)
                    for fid, score in batch_inh.items():
                        all_inh_scores.setdefault(fid, []).append(score)
                    n_valid_batches += 1

                finally:
                    instrument.release()   # deterministic teardown (vram-ledger 2026-07-31)
                    del instrument
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        finally:
            self.inference.enable_compile()

        if n_valid_batches == 0:
            logger.note(f"{self.neg_mode} contrast hop: no valid microbatches")
            return {}, {}

        if n_valid_batches > 1:
            logger.note(
                f"{self.neg_mode} contrast hop: {n_valid_batches} microbatches "
                f"× ≤{self.neg_batch_size} seqs ({total_seqs} total)"
            )

        # Average scores across microbatches so scale is comparable to a single-pass run
        activator_scores = {fid: sum(s) / len(s) for fid, s in all_act_scores.items()}
        inhibitor_scores = {fid: sum(s) / len(s) for fid, s in all_inh_scores.items()}
        return activator_scores, inhibitor_scores
