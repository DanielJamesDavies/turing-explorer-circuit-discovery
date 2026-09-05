"""Shared gradient-discovery pipeline (template method).

Every gradient discovery method (counterfactual, ablation, activation) runs the
same 15-stage pipeline: seed resolution -> probe dataset -> sequence slicing ->
circuit + seed node -> ATTRIBUTION HOP -> assembly -> empty-reject ->
restoration provenance -> eval negatives -> LOO minimality prune -> magnitude
prune -> cf/suppression eval -> acceptance gate -> metadata -> accept. Only the
attribution hop — what tokens run forward, what objective is backpropagated,
where the gradient is linearised — is genuinely method-specific; the rest
differs by small pluggable rules (admission thresholds, role vocabulary, prune
function, acceptance gate). This base owns the template; subclasses provide the
hooks.

Default hook implementations follow the POSCTX-SUPPORT PROFILE shared by
ablation gradient and activation gradient (supports admitted by a flat
threshold, suppression acceptance gate, selector-sourced eval negatives).
Counterfactual gradient overrides the profile (signed roles, act-scaled
thresholds, faithfulness gate, discovery negatives reused for eval).

`position_aware` is a MODIFIER on whichever attribution runs in the hop (it
swaps that method's position-collapse for a union over the seed's causal
prefix) — not a method of its own. The baseline-free posctx grad x natural
attribution is its own top-level method (ActivationGradientDiscovery), not a
mode: it runs on posctx and cannot find absent activators, so it never answered
cf's question.

Test-patchability contract: the faithfulness eval and the LOO prune are called
through one-line hooks so each subclass module references ITS OWN module-level
import — tests patch e.g. ``circuit.discovery.counterfactual_gradient.
evaluate_counterfactual_faithfulness`` and must keep working.
"""

import gc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

import torch

from .base import DiscoveryMethod
from circuit.types.feature_id import FeatureID
from config import config
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.magnitude_prune import prune_by_magnitude_bisection
from eval.minimality import prune_non_minimal_nodes_suppression
from eval.recurrence_prune import prune_by_sequence_recurrence
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx
from store.circuits import Circuit, CircuitNode
from observability.phases import phase as _phase
from utils.neg_context_selector import NegContextSelector


@dataclass
class DiscoveryContext:
    """Per-seed state threaded through the template's stages and hooks."""

    seed_comp_idx: int
    seed_latent_idx: int
    n_kinds: int
    kinds: List[str]
    seed_layer: int
    seed_kind: str
    seed_fid: FeatureID
    probe_data: Any
    pos_tokens_probe: torch.Tensor
    pos_argmax_probe: torch.Tensor
    pos_tokens_eval: torch.Tensor
    pos_argmax_eval: torch.Tensor
    # cf-only fields, populated by its _prepare:
    neg_tokens: Optional[torch.Tensor] = None
    target_act_pos: float = 0.0
    effective_activator_threshold: float = 0.0
    effective_inhibitor_threshold: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HopResult:
    """What an attribution hop delivers to assembly."""

    positives: Dict[FeatureID, float]
    negatives: Dict[FeatureID, float]
    target_loss: float = 0.0
    target_pre_act: float = 0.0


class GradientDiscoveryBase(DiscoveryMethod):
    """Template for gradient discovery; see module docstring for the stages."""

    method_name = "gradient_base"
    circuit_name_prefix = "GradientBase"
    positive_role = "ablation_support"
    negative_role = "counterfactual_inhibitor"
    empty_reject_message = "no supports passed threshold"

    # ------------------------------------------------------------------
    # Shared config plumbing
    # ------------------------------------------------------------------

    def _init_shared_knobs(
        self,
        cfg: Any,
        *,
        attribution_mode: Optional[str] = None,
        ig_steps: Optional[int] = None,
        min_active_count: Optional[int] = None,
        max_neg_sequences: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
        top_k_scope: Optional[str] = None,
    ) -> None:
        """The knob block both method configs share field-for-field (each read
        from that method's OWN config object), plus the discovery-level
        position-aware / magnitude-prune / sequence-width blocks."""

        self.attribution_mode = (
            attribution_mode if attribution_mode is not None
            else cast(str, cfg.attribution_mode)
        )
        self.ig_steps = ig_steps if ig_steps is not None else cast(int, cfg.ig_steps)
        self.restoration_rounds = cast(int, cfg.restoration.rounds)
        self.restoration_per_round_k = cast(int, cfg.restoration.per_round_k)
        self.restoration_round_select = cast(str, cfg.restoration.round_select)
        self.restoration_round_abs_pctl = cast(float, cfg.restoration.round_abs_pctl)
        self.restoration_certificate_tol = cast(float, cfg.restoration.certificate_tol)
        self.restoration_ig_steps = cast(int, cfg.restoration.ig_steps)
        self.restoration_final_ig_polish = cast(bool, cfg.restoration.final_ig_polish)
        self.restoration_grad_batch_size = cast(
            Optional[int], cfg.restoration.grad_batch_size
        )
        self.negative_roles = cast(str, cfg.negative_roles)
        self.neg_mode = cast(str, cfg.neg_mode)
        self.distant_pool_size = cast(int, cfg.distant_pool_size)
        self.min_active_count = (
            min_active_count if min_active_count is not None
            else cast(int, cfg.min_active_count)
        )
        self.max_neg_sequences = (
            max_neg_sequences if max_neg_sequences is not None
            else cast(int, cfg.max_neg_sequences)
        )
        self.pruning_threshold = (
            pruning_threshold if pruning_threshold is not None
            else cast(float, cfg.pruning_threshold)
        )
        self.top_k_scope = (
            top_k_scope if top_k_scope is not None else cast(str, cfg.top_k_scope)
        )
        self.position_aware = cast(bool, config.discovery.position_aware)
        self.position_aware_top_n = cast(int, config.discovery.position_aware_top_n)
        self.position_aware_select = cast(str, config.discovery.position_aware_select)
        self.position_aware_threshold = cast(float, config.discovery.position_aware_threshold)
        self.position_aware_position_weight = cast(bool, config.discovery.position_aware_position_weight)
        self.position_aware_scope = cast(str, config.discovery.position_aware_scope)
        self.magnitude_prune = cast(bool, config.discovery.magnitude_prune)
        self.magnitude_prune_tolerance = cast(float, config.discovery.magnitude_prune_tolerance)
        self.magnitude_prune_target = cast(float, config.discovery.magnitude_prune_target)
        self.magnitude_prune_min_keep = cast(int, config.discovery.magnitude_prune_min_keep)
        self.magnitude_prune_objective = cast(str, config.discovery.magnitude_prune_objective)
        # Cross-sequence recurrence prune — runs BEFORE the magnitude prune, so
        # magnitude bisection operates on the already-derecurred membership.
        self.recurrence_prune = cast(bool, config.discovery.recurrence_prune)
        self.recurrence_prune_min_sequences = cast(
            int, config.discovery.recurrence_prune_min_sequences)
        self.recurrence_prune_min_keep = cast(
            int, config.discovery.recurrence_prune_min_keep)
        # Set per seed in _discover; only floor_source="negctx" reads it.
        self._floor_neg_tokens: Optional[torch.Tensor] = None
        # The seed's typical PRE-TOP-K posctx activation, the scale that
        # preact_filter's negative-rejection bar is a fraction of. Measured
        # ONCE per seed here rather than at each selector call site: it
        # costs a forward pass over the positives and every call site must
        # agree on the number or the same candidate could pass one filter
        # and fail another.
        self._posctx_preact_reference: Optional[float] = None
        # Sequence COUNT vs batch SIZE (see DiscoveryConfig): counts set how
        # many pos sequences inform discovery / evals; batch sizes bound one
        # forward pass, with chunked merging above them.
        self.probe_batch_size = cast(int, config.discovery.probe_batch_size)
        self.probe_sequence_count = cast(int, config.discovery.probe_sequence_count)
        self.eval_sequence_count = cast(int, config.discovery.eval_sequence_count)
        self.eval_batch_size = cast(int, config.discovery.eval_batch_size)
        self._last_restoration = None
        self._pending_inhibitors: Dict[FeatureID, float] = {}
        # Tri-amp: fitted per-latent amplitudes from the mask engine's
        # provenance (amp_kept). Stashed by the mask hops, attached to
        # node metadata in _assemble — without this the stored circuit
        # keeps the membership but loses the alpha vector, which is half
        # the tri-amp object (the 044-behaviours keep_scales lesson).
        self._pending_amplitudes: Dict[FeatureID, float] = {}
        self._pending_amp_stats: dict[str, Any] = {}
        self._last_neg_selection_metadata: dict[str, Any] = {}

    def _init_support_profile(
        self,
        cfg: Any,
        *,
        top_k_supports: Optional[int] = None,
        support_threshold: Optional[float] = None,
        min_suppression_score: Optional[float] = None,
        **shared_overrides: Any,
    ) -> None:
        """Shared knobs + the posctx-support extras (ablation / activation)."""

        self._init_shared_knobs(cfg, **shared_overrides)
        self.top_k_supports = (
            top_k_supports if top_k_supports is not None else cast(int, cfg.top_k_supports)
        )
        self.support_threshold = (
            support_threshold if support_threshold is not None
            else cast(float, cfg.support_threshold)
        )
        self.min_suppression_score = (
            min_suppression_score if min_suppression_score is not None
            else cast(float, cfg.min_suppression_score)
        )
        self.top_k_inhibitors = cast(int, cfg.top_k_inhibitors)

    # ------------------------------------------------------------------
    # The template
    # ------------------------------------------------------------------

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, self.method_name)
        # Per-seed state: never let one seed's amplitudes leak into the next.
        self._pending_amplitudes = {}
        self._pending_amp_stats = {}
        try:
            return self._discover(seed_comp_idx, seed_latent_idx, logger)
        finally:
            logger.save()
            # Coarse-boundary safety net for any instrument cycle that slipped
            # past the per-pass release() calls. gc.collect costs 50-500ms so
            # it runs once per discovery, never in the grad-pass hot loops
            # (vram-ledger 2026-07-31).
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _discover(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        logger: CircuitLogger,
    ) -> Optional[Circuit]:
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]
        seed_fid = FeatureID(seed_layer, seed_kind, seed_latent_idx)

        with _phase("seed.probes"):
            probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts)")
            return None

        # Negatives for floor_source="negctx", read by _integrated_baseline_
        # attribution and _restoration_selection. Set here, beside the probe
        # dataset they derive from, so they cannot go stale relative to the
        # seed. Left None outside discover(), where the negctx branch raises
        # rather than silently substituting another floor.
        _ncs = config.discovery.neg_context_selection
        self._posctx_preact_reference = (
            self._neg_context_selector().posctx_reference(
                probe_data.pos_tokens[: self.probe_sequence_count],
                seed_comp_idx, seed_latent_idx,
                batch_size=int(_ncs.filter_batch_size),
                stat=str(_ncs.preact_reference_stat))
            if bool(_ncs.preact_filter) else None)
        self._floor_neg_tokens = self._floor_negatives(
            probe_data, seed_comp_idx, seed_latent_idx, logger)

        ctx = DiscoveryContext(
            seed_comp_idx=seed_comp_idx,
            seed_latent_idx=seed_latent_idx,
            n_kinds=n_kinds,
            kinds=list(kinds),
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_fid=seed_fid,
            probe_data=probe_data,
            pos_tokens_probe=probe_data.pos_tokens[: self.probe_sequence_count],
            pos_argmax_probe=probe_data.pos_argmax[: self.probe_sequence_count],
            pos_tokens_eval=probe_data.pos_tokens[: self.eval_sequence_count],
            pos_argmax_eval=probe_data.pos_argmax[: self.eval_sequence_count],
        )
        logger.header(
            seed_layer,
            seed_kind,
            seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        if not self._prepare(ctx, logger):
            return None

        circuit = Circuit(name=f"{self.circuit_name_prefix}_S{seed_comp_idx}_{seed_latent_idx}")
        seed_node = CircuitNode(metadata={"feature_id": seed_fid, "role": "seed"})
        circuit.add_node(seed_node)
        fid_to_uuid: Dict[FeatureID, str] = {seed_fid: seed_node.uuid}

        with _phase("seed.fit"):
            hop = self._run_attribution_hop(ctx, logger)
        with _phase("seed.assemble"):
            self._pre_assembly(ctx, hop)
            n_pos = self._assemble(
                circuit, fid_to_uuid, seed_node.uuid, hop.positives,
                self.positive_role, lambda fid, s: self._admit_positive(ctx, fid, s),
            )
            n_neg = self._assemble(
                circuit, fid_to_uuid, seed_node.uuid, hop.negatives,
                self.negative_role, lambda fid, s: self._admit_negative(ctx, fid, s),
            )
        logger.stage(
            "circuit assembly",
            len(circuit.nodes),
            len(circuit.edges),
            note=self._assembly_note(n_pos, n_neg),
        )
        if len(circuit.nodes) <= 1:
            logger.reject(self.empty_reject_message)
            return None

        if (
            self.attribution_mode in ("restoration", "ig_restoration", "restoration_negctx")
            and self._last_restoration is not None
        ):
            from circuit.instrument.restoration import stamp_restoration_provenance

            stamp_restoration_provenance(circuit, self._last_restoration)

        with _phase("seed.eval_negs"):
            neg_tokens_eval = self._eval_neg_tokens(ctx, logger)
        if neg_tokens_eval is None:
            return None

        circuit_layers: Set[int] = {
            node.feature_id.layer
            for node in circuit.nodes.values()
            if node.feature_id is not None
        }
        self._log_assembly_complete(circuit, circuit_layers)

        if self.pruning_threshold > 0:
            n_before = len(circuit.nodes)
            with _phase("seed.prune_loo"):
                self._call_loo_prune(ctx, circuit, neg_tokens_eval, circuit_layers)
            circuit_layers = {
                node.feature_id.layer
                for node in circuit.nodes.values()
                if node.feature_id is not None
            }
            logger.stage(
                "after pruning",
                len(circuit.nodes),
                len(circuit.edges),
                note=f"removed {n_before - len(circuit.nodes)} nodes",
            )

        # Cross-sequence recurrence prune (optional) — drops per-input scaffolding
        # that PA's union-over-positions admits from a single probe sequence.
        # Runs BEFORE the magnitude prune so that bisection searches a smaller
        # ranked set; needs one forward pass, no sufficiency search.
        if self.recurrence_prune:
          with _phase("seed.prune_recurrence"):
            prune_by_sequence_recurrence(
                self.inference, self.sae_bank, circuit,
                pos_tokens=ctx.pos_tokens_eval,
                neg_tokens=neg_tokens_eval,
                min_sequences=self.recurrence_prune_min_sequences,
                min_keep=self.recurrence_prune_min_keep,
                logger=logger,
            )
            circuit_layers = {
                node.feature_id.layer
                for node in circuit.nodes.values()
                if node.feature_id is not None
            }

        # Global magnitude prune (optional) — scalable free-φ bisection, for the
        # large position-aware allowed sets that LOO minimality cannot touch.
        if self.magnitude_prune:
          with _phase("seed.prune_magnitude"):
            prune_by_magnitude_bisection(
                self.inference, self.sae_bank, circuit,
                pos_tokens=ctx.pos_tokens_eval,
                seed_layer=seed_layer, seed_kind=seed_kind, seed_latent_idx=seed_latent_idx,
                pos_argmax=ctx.pos_argmax_eval,
                tolerance=self.magnitude_prune_tolerance,
                target=self.magnitude_prune_target,
                min_keep=self.magnitude_prune_min_keep,
                objective=self.magnitude_prune_objective,
                logger=logger,
            )
            circuit_layers = {
                node.feature_id.layer
                for node in circuit.nodes.values()
                if node.feature_id is not None
            }

        with _phase("seed.cf_eval"):
            cf_faith, sup_score = self._run_faithfulness_eval(
                ctx, circuit, neg_tokens_eval, circuit_layers
            )
        logger.note(
            f"counterfactual_faithfulness: {cf_faith:.4f} | "
            f"posctx_suppression_score: {sup_score:.4f}"
        )

        reject_reason = self._accept(cf_faith, sup_score)
        if reject_reason is not None:
            logger.reject(reject_reason)
            return None

        circuit.metadata.update(
            {
                "counterfactual_faithfulness": cf_faith,
                "posctx_suppression_score": sup_score,
                "seed_comp": seed_comp_idx,
                "seed_latent": seed_latent_idx,
                "n_nodes": len(circuit.nodes),
                "n_edges": len(circuit.edges),
                "discovery_method": self.method_name,
                "neg_mode": self.neg_mode,
                "neg_selection": dict(self._last_neg_selection_metadata),
                **self._extra_metadata(ctx, hop, n_pos, n_neg, sup_score),
            }
        )
        if self._pending_amp_stats:
            circuit.metadata["amp_stats"] = dict(self._pending_amp_stats)
        logger.nodes(list(circuit.nodes.values()))
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit

    # ------------------------------------------------------------------
    # Assembly (shared mechanics)
    # ------------------------------------------------------------------

    def _stash_amplitudes(self, prov: Any) -> None:
        """Convert the mask engine's amp_kept provenance ({"layer/kind":
        {latent: alpha}}) into FeatureID-keyed amplitudes for _assemble.
        Empty when free_amplitude is off — nodes then carry no amplitude
        field and stored circuits are byte-identical to before."""
        ak = (prov or {}).get("amp_kept") or {}
        amps: Dict[FeatureID, float] = {}
        for site_key, d in ak.items():
            lyr, knd = site_key.split("/")
            for i, a in d.items():
                amps[FeatureID(layer=int(lyr), kind=knd,
                               index=int(i))] = float(a)
        self._pending_amplitudes = amps
        self._pending_amp_stats = dict((prov or {}).get("amp_stats") or {})

    def _assemble(
        self,
        circuit: Circuit,
        fid_to_uuid: Dict[FeatureID, str],
        seed_uuid: str,
        scores: Dict[FeatureID, float],
        role: str,
        admit: Callable[[FeatureID, float], bool],
    ) -> int:
        n_added = 0
        for upstream_fid, score in scores.items():
            if not admit(upstream_fid, score):
                continue
            if upstream_fid not in fid_to_uuid:
                meta = {
                    "feature_id": upstream_fid,
                    "role": role,
                    "attribution_score": score,
                }
                amp = self._pending_amplitudes.get(upstream_fid)
                if amp is not None:
                    meta["amplitude"] = amp
                node = CircuitNode(metadata=meta)
                circuit.add_node(node)
                fid_to_uuid[upstream_fid] = node.uuid
            circuit.add_edge(fid_to_uuid[upstream_fid], seed_uuid, weight=score)
            n_added += 1
        return n_added

    # ------------------------------------------------------------------
    # Hooks — defaults implement the posctx-support profile (abl / act)
    # ------------------------------------------------------------------

    def _prepare(self, ctx: DiscoveryContext, logger: CircuitLogger) -> bool:
        """Method-specific pre-hop stage (cf: source contrast sequences and the
        posctx target). Return False to reject the seed."""
        return True

    def _pre_assembly(self, ctx: DiscoveryContext, hop: HopResult) -> None:
        """Called once between the hop and assembly — the place to precompute
        vectorized admission state (one gather instead of per-member syncs)."""
        return None

    def _run_attribution_hop(self, ctx: DiscoveryContext, logger: CircuitLogger) -> HopResult:
        support_scores, target_loss, target_pre_act = self._run_ablation_hop(
            ctx.seed_comp_idx,
            ctx.seed_latent_idx,
            ctx.pos_tokens_probe,
            ctx.pos_argmax_probe,
            logger,
        )
        logger.stage(
            "positive ablation grad pass",
            1,
            0,
            note=(
                f"{len(support_scores)} support candidates | "
                f"loss={target_loss:.4f} pre_act={target_pre_act:.4f}"
            ),
        )
        return HopResult(
            positives=support_scores,
            negatives=dict(self._pending_inhibitors),
            target_loss=target_loss,
            target_pre_act=target_pre_act,
        )

    def _run_ablation_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], float, float]:
        raise NotImplementedError

    def _admit_positive(self, ctx: DiscoveryContext, fid: FeatureID, score: float) -> bool:
        return not (score < self.support_threshold)

    def _admit_negative(self, ctx: DiscoveryContext, fid: FeatureID, score: float) -> bool:
        # Support-profile negatives are restoration-include deliveries: the
        # loop already spent budget restoring them, so they ship unfiltered.
        return True

    def _assembly_note(self, n_pos: int, n_neg: int) -> str:
        return f"{n_pos} supports, {n_neg} inhibitors after thresholding"

    _MASK_MODES = ("mask", "mask_contrast", "mask_negctx", "mask_inject")

    def _eval_neg_tokens(
        self, ctx: DiscoveryContext, logger: CircuitLogger
    ) -> Optional[torch.Tensor]:
        # Learned-mask modes: the stored per-seed negctx (the KNN store the
        # floor already read, a free slice) serves the cf/sup eval too.
        # The selector re-retrieval (exact ranking + preact filter over a
        # candidate pool) cost 58% of per-seed time (phase profile,
        # 2026-09-05) to feed an eval whose thresholds mask modes bypass
        # and whose bare-set numbers are re-done amp-aware post hoc.
        # Eval is in-sample w.r.t. the floor's negatives, matching the
        # pipeline's in-sample positive eval.
        if getattr(self, "attribution_mode", "") in self._MASK_MODES:
            nt = getattr(ctx.probe_data, "neg_tokens", None)
            if nt is not None and int(nt.shape[0]) > 0:
                self._last_neg_selection_metadata = {
                    "source": "stored_negctx_slice",
                    "n": int(min(nt.shape[0], self.max_neg_sequences))}
                return nt[: max(1, int(self.max_neg_sequences))]
            logger.reject("mask mode: seed has no stored negctx for eval")
            return None
        return self._get_eval_neg_tokens(
            ctx.seed_comp_idx, ctx.seed_latent_idx, ctx.probe_data.neg_tokens, logger
        )

    def _log_assembly_complete(self, circuit: Circuit, circuit_layers: Set[int]) -> None:
        return None

    def _call_loo_prune(
        self,
        ctx: DiscoveryContext,
        circuit: Circuit,
        neg_tokens_eval: torch.Tensor,
        circuit_layers: Set[int],
    ) -> None:
        prune_non_minimal_nodes_suppression(
            self.inference,
            self.sae_bank,
            self.avg_acts,
            circuit,
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
        return evaluate_counterfactual_faithfulness(
            self.inference,
            self.sae_bank,
            self.avg_acts,
            circuit,
            neg_tokens=neg_tokens_eval,
            pos_tokens=ctx.pos_tokens_eval,
            seed_layer=ctx.seed_layer,
            seed_kind=ctx.seed_kind,
            seed_latent_idx=ctx.seed_latent_idx,
            pos_argmax=ctx.pos_argmax_eval,
            circuit_layers=circuit_layers,
        )

    def _accept(self, cf_faith: float, sup_score: float) -> Optional[str]:
        # Learned-mask modes: never threshold-reject. The gate metrics are
        # measured on the BARE member set (amplitudes stripped), which
        # systematically undervalues tri-amp circuits — the keep_scales
        # lesson. Mask circuits are stored unconditionally and judged
        # post-hoc with amp-aware evals.
        if getattr(self, "attribution_mode", "") in (
                "mask", "mask_contrast", "mask_negctx", "mask_inject"):
            return None
        if sup_score < self.min_suppression_score:
            return (
                f"posctx_suppression_score {sup_score:.4f} < "
                f"min_suppression_score {self.min_suppression_score}"
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
        return {
            "ablation_suppression_score": sup_score,
            "n_supports": n_pos,
            "target_loss": hop.target_loss,
            "target_pre_act": hop.target_pre_act,
        }

    # ------------------------------------------------------------------
    # Shared attribution engines (phase 2): the ig_mean and restoration
    # hops of cf and abl are the same engine call with a per-method profile
    # (objective/target, role budgets, log label). Each method's hop keeps its
    # own return shape, target sourcing, and log lines around these.
    # ------------------------------------------------------------------

    def _integrated_baseline_attribution(
        self,
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        *,
        objective: str,
        target_act: float,
        top_k_positive: int,
        top_k_negative: int,
        metric_label: str,
        logger: CircuitLogger,
    ) -> Optional[Tuple[Dict[FeatureID, float], Dict[FeatureID, float], float, float]]:
        """SFC-style integrated-gradients attribution (Marks et al. 2025) along
        the mean-ablation-floor -> natural path: floors, seed projection,
        `integrated_baseline_scores`, signed-role extraction. Returns
        ``(positives, negatives, metric_floor, metric_natural)``, or None when
        the seed has no upstream sites. ``target_act`` is read only by the
        "gap" objective (inert under "drive")."""

        from circuit.instrument.ig_baseline import extract_signed_roles, integrated_baseline_scores
        from eval.ablation_faithfulness import (
            collect_site_means,
            resolve_site_floors,
            upstream_sites,
        )

        kinds = self.sae_bank.kinds
        n_kinds = len(kinds)
        sites = upstream_sites(self.sae_bank, seed_layer, seed_kind)
        if not sites:
            logger.note("ig_mean: seed has no upstream sites")
            return None
        site_floors = collect_site_means(self.inference, self.sae_bank, pos_tokens, sites)
        site_floors = resolve_site_floors(
            self.inference, self.sae_bank, sites,
            posctx_means=site_floors, loader=self.probe_builder.loader,
            neg_tokens=self._floor_neg_tokens,
        )

        sae = self.sae_bank.saes[seed_kind][seed_layer]
        w_seed = sae.encoder.weight[seed_latent_idx].detach()
        b_seed = sae._get_bias_eff()[seed_latent_idx].detach()

        scores_by_site, metric_floor, metric_natural = integrated_baseline_scores(
            self.inference,
            self.sae_bank,
            tokens=pos_tokens,
            substitute_sites=sites,
            site_floors=site_floors,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            w_seed=w_seed,
            b_seed=b_seed,
            pos_argmax=pos_argmax,
            objective=objective,
            target_act=target_act,
            ig_steps=self.ig_steps,
            # Position-aware: union the upstream position axis over the seed's
            # causal prefix instead of summing it away. Scores come back sparse.
            position_aware=self._position_aware_spec(pos_argmax),
            batch_size=self.probe_batch_size,
        )
        logger.note(
            f"ig_mean: {metric_label} floor {metric_floor:.4f} -> natural {metric_natural:.4f} "
            f"over {len(sites)} sites, {self.ig_steps} steps"
            + (" (position-aware union)" if self.position_aware else "")
        )
        # Position-aware scores are already the union — the per-position selection
        # replaced the ranking, so don't re-truncate with top-k here.
        no_trunc = int(self.sae_bank.d_sae)
        top_pos = no_trunc if self.position_aware else top_k_positive
        top_neg = no_trunc if self.position_aware else top_k_negative
        # Extract BOTH signs unconditionally; the caller applies the role
        # semantics via resolve_role_delivery (PA keeps both, NPA-exclude drops
        # negatives). The positives are independent of top_k_negative, so NPA
        # behaviour is unchanged.
        positives, negatives = extract_signed_roles(
            scores_by_site,
            kinds=list(kinds),
            n_kinds=n_kinds,
            top_k_positive=top_pos,
            top_k_negative=top_neg,
            min_active_count=self.min_active_count,
            active_count=self._active_count(),
            top_k_scope=self.top_k_scope,
        )
        return positives, negatives, metric_floor, metric_natural

    def _restoration_selection(
        self,
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        target_act: float,
    ):
        """The shared `run_restoration_selection` engine call (iterative greedy
        restoration from the mean-ablation floor); stashes the result for the
        template's provenance stamp. Target sourcing, negative-role delivery,
        and logging stay per-method."""

        from circuit.instrument.restoration import run_restoration_selection

        positives, negatives, result = run_restoration_selection(
            self.inference,
            self.sae_bank,
            tokens=pos_tokens,
            pos_argmax=pos_argmax,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_latent_idx=seed_latent_idx,
            target_act=target_act,
            rounds=self.restoration_rounds,
            per_round_k=self.restoration_per_round_k,
            certificate_tol=self.restoration_certificate_tol,
            # PA restoration keeps both signs (stream reconstruction), so
            # negatives must participate in the greedy loop even under exclude;
            # the role split is applied after (resolve_role_delivery). NPA keeps
            # the classic gate (exclude never restores negatives).
            allow_negative=self.negative_roles == "include" or self.position_aware,
            loader=self.probe_builder.loader,
            neg_tokens=self._floor_neg_tokens,
            scorer="ig" if self.attribution_mode == "ig_restoration" else "point",
            ig_steps=self.restoration_ig_steps,
            final_ig_polish=self.restoration_final_ig_polish,
            polish_ig_steps=self.ig_steps,
            round_select=self.restoration_round_select,
            round_abs_pctl=self.restoration_round_abs_pctl,
            position_aware=self.position_aware,
            batch_size=self.restoration_grad_batch_size or self.probe_batch_size,
        )
        self._last_restoration = result
        return positives, negatives, result

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _position_aware_spec(self, peaks: torch.Tensor):
        """PositionAwareSpec for this run, or None when position-awareness is off
        (in which case the attribution keeps its classic .sum(dim=(0,1))
        position-collapse). ``peaks`` is the seed's per-sequence anchor position
        for whichever input this attribution runs on."""
        if not self.position_aware:
            return None
        from circuit.instrument.position_aware import PositionAwareSpec

        return PositionAwareSpec(
            peaks=peaks,
            top_n=self.position_aware_top_n,
            select=self.position_aware_select,
            threshold=self.position_aware_threshold,
            position_weight=self.position_aware_position_weight,
            scope=self.position_aware_scope,
        )

    def _active_count(self) -> torch.Tensor:
        from store.latent_stats import latent_stats

        return latent_stats.active_count

    def _get_eval_neg_tokens(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        stored_neg_tokens: torch.Tensor,
        logger: CircuitLogger,
    ) -> Optional[torch.Tensor]:
        del stored_neg_tokens
        max_neg = max(1, int(self.max_neg_sequences))
        cfg = config.discovery.neg_context_selection
        candidate_pool_size = (
            self.distant_pool_size if self.neg_mode == "distant" else cfg.candidate_pool_size
        )
        selection = self._neg_context_selector().select(
            seed_comp_idx,
            seed_latent_idx,
            self.neg_mode,
            max_sequences=max_neg,
            batch_size=max(1, int(config.discovery.probe_batch_size)),
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
        if selection is None:
            logger.reject(f"neg_mode={self.neg_mode}: no eval negative sequences available")
            return None
        self._last_neg_selection_metadata = selection.metadata
        return selection.tokens

    def _floor_negatives(
        self,
        probe_data: Any,
        seed_comp_idx: int,
        seed_latent_idx: int,
        logger: CircuitLogger,
    ) -> Optional[torch.Tensor]:
        """The negatives defining floor_source="negctx".

        "store" (default) reuses the probe dataset's negatives — the per-latent
        neg_ctx KNN store, i.e. the nearest non-activating sequences — so it is
        a free slice and works for every method, abl included (ProbeDataset
        always carries negatives; ctx.neg_tokens is cf-only and would not).

        "close"/"random"/"distant" re-retrieve through the shared selector, so
        the FLOOR's negative hardness can be varied independently of
        self.neg_mode, which governs ig_negctx / phi_cf and never the floor.
        Those modes cost a retrieval, so they are skipped entirely unless a
        negctx floor will actually read them.
        """
        mode = str(config.discovery.floor_negctx_mode)
        if mode == "store":
            return probe_data.neg_tokens[: self.probe_sequence_count]
        # The learned mask has its OWN negctx floor knob (deliberately not the
        # shared floor_source, which would move the ig hops too), so the
        # retrieval must also run when only the mask wants negatives —
        # otherwise a negctx-floored mask fails on a None it cannot explain.
        from circuit.instrument.learned_mask import FLOORS_NEEDING_NEGATIVES
        if (str(config.discovery.floor_source) != "negctx"
                and str(config.discovery.learned_mask.mask_floor_source)
                not in FLOORS_NEEDING_NEGATIVES):
            return None
        cfg = config.discovery.neg_context_selection
        selection = self._neg_context_selector().select(
            seed_comp_idx,
            seed_latent_idx,
            mode,
            max_sequences=max(1, int(self.probe_sequence_count)),
            batch_size=max(1, int(config.discovery.probe_batch_size)),
            candidate_pool_size=(
                self.distant_pool_size if mode == "distant" else cfg.candidate_pool_size
            ),
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
        if selection is None:
            # resolve_site_floors raises on None rather than substituting a
            # different floor, which is the intended loud failure.
            logger.note(f"floor_negctx_mode={mode}: no negatives selected")
            return None
        return selection.tokens

    def _neg_context_selector(self) -> NegContextSelector:
        from store.context import mid_ctx, neg_ctx, top_ctx
        from store.seq_repr import seq_repr

        if seq_repr is None:
            raise RuntimeError("seq_repr must be loaded before negative-context selection")

        return NegContextSelector(
            self.inference,
            self.sae_bank,
            self.probe_builder.loader,
            neg_ctx,
            seq_repr,
            top_ctx,
            mid_ctx,
        )


__all__ = ["DiscoveryContext", "GradientDiscoveryBase", "HopResult"]
