import gc
from typing import Any, Dict, Optional, Tuple

import torch

from .counterfactual_gradient import SeedProjectionInstrument
from .gradient_base import GradientDiscoveryBase
from circuit.instrument.attribution import compute_latent_ablation_scores
from circuit.types.feature_id import FeatureID
from config import config
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx


class AblationGradientDiscovery(GradientDiscoveryBase):
    """
    Discovers support circuits by asking which active upstream latents should be
    ablated to suppress a seed latent on its positive contexts.

    Pipeline stages (probe -> assembly -> pruning -> evals -> acceptance) live
    in GradientDiscoveryBase, whose default hooks implement exactly this
    method's rules (the posctx-support profile); this class provides the
    attribution hop: the mode dispatch across local / ig_mean /
    restoration / ig_restoration.
    """

    method_name = "ablation_gradient"
    circuit_name_prefix = "AblationGrad"

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        top_k_supports: Optional[int] = None,
        top_k_scope: Optional[str] = None,
        support_threshold: Optional[float] = None,
        min_active_count: Optional[int] = None,
        max_neg_sequences: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
        min_suppression_score: Optional[float] = None,
        attribution_mode: Optional[str] = None,
        ig_steps: Optional[int] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        self._init_support_profile(
            config.discovery.ablation_gradient,
            top_k_supports=top_k_supports,
            support_threshold=support_threshold,
            min_suppression_score=min_suppression_score,
            attribution_mode=attribution_mode,
            ig_steps=ig_steps,
            min_active_count=min_active_count,
            max_neg_sequences=max_neg_sequences,
            pruning_threshold=pruning_threshold,
            top_k_scope=top_k_scope,
        )

    def _run_ablation_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], float, float]:
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]

        # Reset per seed: assembly delivers whatever the mode's hop stashes.
        self._pending_inhibitors = {}

        # `position_aware` is a MODIFIER on whichever attribution runs below (it
        # swaps that method's position-collapse for a union over the seed's causal
        # prefix) — not a method of its own. The baseline-free posctx
        # grad x natural attribution is its own top-level method now
        # (ActivationGradientDiscovery), not a mode here.
        if self.attribution_mode == "ig_mean":
            return self._run_ig_mean_hop(
                seed_layer, seed_kind, seed_latent_idx, pos_tokens, pos_argmax, logger
            )
        if self.attribution_mode == "mask":
            return self._run_mask_hop(
                seed_layer, seed_kind, seed_latent_idx, pos_tokens, pos_argmax, logger
            )
        if self.attribution_mode in ("restoration", "ig_restoration"):
            # Full probe_sequence_count: the round scorer chunks internally
            # at probe_batch_size (see restoration._round_scores).
            return self._run_restoration_hop(
                seed_layer, seed_kind, seed_latent_idx, pos_tokens, pos_argmax, logger,
            )
        sae = self.sae_bank.saes[seed_kind][seed_layer]
        w_seed = sae.encoder.weight[seed_latent_idx].detach()
        b_seed = sae._get_bias_eff()[seed_latent_idx].detach()

        # Local mode, microbatched over the probe sequences (sequence count vs
        # batch size — the same contract as cf's contrast hop): each chunk of
        # probe_batch_size runs its own grad-enabled pass; classic scores are
        # averaged across chunks, position-aware memberships merged by
        # max-|score| (the union's own rule).
        if self.negative_roles == "include" and not self.position_aware:
            logger.note(
                "negative_roles=include not yet supported in local mode "
                "(attribution util selects supports only); proceeding supports-only"
            )
        B_total = int(pos_tokens.shape[0])
        bs = max(1, int(self.probe_batch_size))
        all_scores: Dict[FeatureID, list] = {}
        loss_total = 0.0
        pre_act_total = 0.0
        n_seqs_done = 0
        was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            for start in range(0, B_total, bs):
                tokens_chunk = pos_tokens[start:start + bs]
                argmax_chunk = pos_argmax[start:start + bs]
                instrument = SeedProjectionInstrument(
                    self.sae_bank, seed_layer, seed_kind, w_seed, b_seed
                )
                try:
                    self.inference.forward(
                        tokens_chunk,
                        patcher=instrument,
                        grad_enabled=True,
                        return_activations=False,
                        tokenize_final=False,
                    )
                    if instrument.seed_pre_act is None:
                        logger.note(
                            f"SeedProjectionInstrument: seed_pre_act is None "
                            f"(chunk offset {start})"
                        )
                        continue

                    B = min(instrument.seed_pre_act.shape[0], argmax_chunk.shape[0])
                    batch_idx = torch.arange(B, device=instrument.seed_pre_act.device)
                    pa = argmax_chunk[:B].to(instrument.seed_pre_act.device).clamp(
                        0, instrument.seed_pre_act.shape[1] - 1
                    )
                    pre_act_at_peak = instrument.seed_pre_act[:B][batch_idx, pa]
                    loss = (pre_act_at_peak ** 2).mean()
                    chunk_loss = float(loss.detach().item())
                    loss_total += chunk_loss * B
                    pre_act_total += float(pre_act_at_peak.detach().mean().item()) * B
                    n_seqs_done += B
                    if abs(chunk_loss) < 1e-8:
                        logger.note(
                            f"near-zero positive-context ablation loss "
                            f"(chunk offset {start}) — skipping"
                        )
                        continue

                    chunk_scores = compute_latent_ablation_scores(
                        graph=instrument.graph,
                        target_scalar=loss,
                        seed_comp_idx=seed_comp_idx,
                        n_kinds=n_kinds,
                        kinds=kinds,
                        top_k_supports=self.top_k_supports,
                        min_active_count=self.min_active_count,
                        active_count=self._active_count(),
                        top_k_scope=self.top_k_scope,
                        # Position-aware abl-local: union the position axis over
                        # the seed's causal prefix instead of .sum(dim=(0, 1)).
                        # The anchor is the seed's firing peak on posctx. Per
                        # position the signal is activation_gradient's
                        # attribution x a positive drive weight, so top_n
                        # membership should match that method's; abs/abs_pctl
                        # expose the drive-weighted variant.
                        position_aware=self._position_aware_spec(argmax_chunk),
                    )
                    for fid, score in chunk_scores.items():
                        all_scores.setdefault(fid, []).append(score)
                finally:
                    del instrument
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        finally:
            if was_compiled:
                self.inference.enable_compile()

        if n_seqs_done == 0:
            return {}, 0.0, 0.0
        target_loss = loss_total / n_seqs_done
        target_pre_act = pre_act_total / n_seqs_done
        if self.position_aware:
            # Union across chunks by MAGNITUDE (signed scores now), then split
            # roles: both signs stay members; include labels the negatives as
            # inhibitors, exclude folds them into supports (resolve_role_delivery
            # — PA membership is sign-invariant, only the label changes).
            from circuit.instrument.position_aware import resolve_role_delivery
            signed = {fid: max(vals, key=abs) for fid, vals in all_scores.items()}
            supports, self._pending_inhibitors = resolve_role_delivery(
                {f: v for f, v in signed.items() if v >= 0},
                {f: v for f, v in signed.items() if v < 0},
                position_aware=True, include_negatives=self.negative_roles == "include",
            )
            scores = supports
        else:
            # Classic: average scores across chunks (cf contrast-hop contract).
            scores = {fid: sum(vals) / len(vals) for fid, vals in all_scores.items()}
        return scores, target_loss, target_pre_act

    def _run_ig_mean_hop(
        self,
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], float, float]:
        """The shared IG-from-floor engine under abl's profile: "drive"
        objective (the seed's pre-activation at probe positions), supports
        budget, negatives stashed for assembly."""

        result = self._integrated_baseline_attribution(
            seed_layer, seed_kind, seed_latent_idx, pos_tokens, pos_argmax,
            objective="drive",
            target_act=0.0,
            top_k_positive=self.top_k_supports,
            top_k_negative=self.top_k_inhibitors,
            metric_label="drive",
            logger=logger,
        )
        if result is None:
            return {}, 0.0, 0.0
        supports, negatives, metric_floor, metric_natural = result
        from circuit.instrument.position_aware import resolve_role_delivery
        supports, self._pending_inhibitors = resolve_role_delivery(
            supports, negatives, position_aware=self.position_aware,
            include_negatives=self.negative_roles == "include",
        )
        return supports, metric_floor, metric_natural

    def _run_mask_hop(
        self,
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], float, float]:
        """abl-mask: the learned continuous mask under the posctx objective —
        the sparsest soft membership whose masked stream reproduces the seed's
        natural pre-activation. Selection happens inside the engine (threshold
        on converged m), as with restoration; scores are m values, all
        positive, so role delivery is trivially supports-only.

        The shared floor_source stays inert here (the loss anchors at the
        natural state). The mask's own mask_floor_source instead sets what a
        FULLY MASKED latent becomes: "zero" keeps the training counterfactual
        identical to free0's, "negctx" makes m=0 reproduce the mean-ablated
        state freeN measures against — which is what lets the mask be compared
        with mean-floor methods on a metric neither of them owns."""

        if self.position_aware:
            raise ValueError(
                "attribution_mode='mask' does not support position_aware yet: "
                "a flat mask IS a flat membership, and the position-indexed "
                "mask is a planned variant — failing loudly rather than "
                "silently ignoring the flag."
            )
        from circuit.instrument.learned_mask import run_learned_mask
        from eval.ablation_faithfulness import upstream_sites

        sites = sorted(upstream_sites(self.sae_bank, seed_layer, seed_kind))
        if not sites:
            logger.note("mask: seed has no upstream sites")
            return {}, 0.0, 0.0
        cfg = config.discovery.learned_mask
        scores, prov = run_learned_mask(
            self.inference, self.sae_bank,
            objective="pos", sites=sites,
            seed_layer=seed_layer, seed_kind=seed_kind,
            seed_latent_idx=seed_latent_idx,
            pos_tokens=pos_tokens, pos_argmax=pos_argmax,
            # Only read when mask_floor_source="negctx"; supplied always so the
            # engine can raise on a missing floor rather than substitute one.
            neg_tokens=self._floor_neg_tokens,
            mask_floor_source=cfg.mask_floor_source,
            dual_floor_weight=cfg.dual_floor_weight,
            steps=cfg.steps, lr=cfg.lr, l1_lambda=cfg.l1_lambda,
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
            logger=logger,
        )
        self._pending_inhibitors = {}
        return scores, float(prov.get("loss_initial") or 0.0), float(
            prov.get("loss_final") or 0.0)

    def _run_restoration_hop(
        self,
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], float, float]:
        """The shared restoration engine under abl's profile: the gap target is
        the seed's clean posctx activation (measured in one pass); positives
        are the supports, negatives stashed for assembly."""

        from eval.ablation_faithfulness import measure_seed_activation

        target_act = measure_seed_activation(
            self.inference, self.sae_bank, pos_tokens,
            seed_layer, seed_kind, seed_latent_idx, pos_argmax,
        )
        positives, negatives, result = self._restoration_selection(
            seed_layer, seed_kind, seed_latent_idx, pos_tokens, pos_argmax, target_act
        )
        from circuit.instrument.position_aware import resolve_role_delivery
        positives, self._pending_inhibitors = resolve_role_delivery(
            positives, negatives, position_aware=self.position_aware,
            include_negatives=self.negative_roles == "include",
        )
        if result is None:
            logger.note("restoration: seed has no upstream sites")
            return {}, 0.0, 0.0
        logger.note(
            f"restoration({self.negative_roles}): "
            f"rounds_used={result.rounds_used} stopped_early={result.stopped_early} "
            f"{len(positives)} supports, {len(negatives)} inhibitors"
        )
        return positives, result.metric_trajectory[0], result.metric_trajectory[-1]
