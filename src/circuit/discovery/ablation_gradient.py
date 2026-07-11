import gc
from typing import Any, Dict, Optional, Set, cast

import torch

from .base import DiscoveryMethod
from .counterfactual_gradient import SeedProjectionInstrument
from circuit.instrument.attribution import compute_latent_ablation_scores
from circuit.instrument.ig_baseline import extract_signed_roles, integrated_baseline_scores
from circuit.types.feature_id import FeatureID
from config import config
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.minimality import prune_non_minimal_nodes_suppression
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx
from store.circuits import Circuit, CircuitNode
from utils.neg_context_selector import NegContextSelector


class AblationGradientDiscovery(DiscoveryMethod):
    """
    Discovers support circuits by asking which active upstream latents should be
    ablated to suppress a seed latent on its positive contexts.
    """

    method_name = "ablation_gradient"

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
        cfg = config.discovery.ablation_gradient
        self.attribution_mode = (
            attribution_mode if attribution_mode is not None
            else cast(str, cfg.attribution_mode)
        )
        self.ig_steps = ig_steps if ig_steps is not None else cast(int, cfg.ig_steps)
        self.neg_mode = cast(str, cfg.neg_mode)
        self.distant_pool_size = cast(int, cfg.distant_pool_size)
        self.top_k_supports = top_k_supports if top_k_supports is not None else cast(int, cfg.top_k_supports)
        self.top_k_scope = top_k_scope if top_k_scope is not None else cast(str, cfg.top_k_scope)
        self.support_threshold = (
            support_threshold if support_threshold is not None else cast(float, cfg.support_threshold)
        )
        self.min_active_count = min_active_count if min_active_count is not None else cast(int, cfg.min_active_count)
        self.max_neg_sequences = (
            max_neg_sequences if max_neg_sequences is not None else cast(int, cfg.max_neg_sequences)
        )
        self.pruning_threshold = (
            pruning_threshold if pruning_threshold is not None else cast(float, cfg.pruning_threshold)
        )
        self.min_suppression_score = (
            min_suppression_score
            if min_suppression_score is not None
            else cast(float, cfg.min_suppression_score)
        )
        self.probe_batch_size = cast(int, config.discovery.probe_batch_size)
        self._last_neg_selection_metadata: dict[str, Any] = {}

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, self.method_name)
        try:
            return self._discover(seed_comp_idx, seed_latent_idx, logger)
        finally:
            logger.save()

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

        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts)")
            return None

        pos_tokens_eval = probe_data.pos_tokens[: self.probe_batch_size]
        pos_argmax_eval = probe_data.pos_argmax[: self.probe_batch_size]
        logger.header(
            seed_layer,
            seed_kind,
            seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        circuit = Circuit(name=f"AblationGrad_S{seed_comp_idx}_{seed_latent_idx}")
        seed_node = CircuitNode(metadata={"feature_id": seed_fid, "role": "seed"})
        circuit.add_node(seed_node)
        fid_to_uuid: Dict[FeatureID, str] = {seed_fid: seed_node.uuid}

        support_scores, target_loss, target_pre_act = self._run_ablation_hop(
            seed_comp_idx,
            seed_latent_idx,
            pos_tokens_eval,
            pos_argmax_eval,
            logger,
        )
        logger.stage(
            "positive ablation grad pass",
            1,
            0,
            note=f"{len(support_scores)} support candidates | loss={target_loss:.4f} pre_act={target_pre_act:.4f}",
        )

        n_supports = 0
        for upstream_fid, score in support_scores.items():
            if score < self.support_threshold:
                continue
            if upstream_fid not in fid_to_uuid:
                node = CircuitNode(
                    metadata={
                        "feature_id": upstream_fid,
                        "role": "ablation_support",
                        "attribution_score": score,
                    }
                )
                circuit.add_node(node)
                fid_to_uuid[upstream_fid] = node.uuid
            circuit.add_edge(fid_to_uuid[upstream_fid], seed_node.uuid, weight=score)
            n_supports += 1

        logger.stage(
            "circuit assembly",
            len(circuit.nodes),
            len(circuit.edges),
            note=f"{n_supports} supports after thresholding",
        )
        if len(circuit.nodes) <= 1:
            logger.reject("no supports passed threshold")
            return None

        neg_tokens_eval = self._get_eval_neg_tokens(
            seed_comp_idx,
            seed_latent_idx,
            probe_data.neg_tokens,
            logger,
        )
        if neg_tokens_eval is None:
            return None
        circuit_layers: Set[int] = {
            node.feature_id.layer
            for node in circuit.nodes.values()
            if node.feature_id is not None
        }

        if self.pruning_threshold > 0:
            n_before = len(circuit.nodes)
            prune_non_minimal_nodes_suppression(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                neg_tokens=neg_tokens_eval,
                pos_tokens=pos_tokens_eval,
                seed_layer=seed_layer,
                seed_kind=seed_kind,
                seed_latent_idx=seed_latent_idx,
                pos_argmax=pos_argmax_eval,
                threshold=self.pruning_threshold,
                circuit_layers=circuit_layers,
            )
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

        cf_faith, sup_score = evaluate_counterfactual_faithfulness(
            self.inference,
            self.sae_bank,
            self.avg_acts,
            circuit,
            neg_tokens=neg_tokens_eval,
            pos_tokens=pos_tokens_eval,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_latent_idx=seed_latent_idx,
            pos_argmax=pos_argmax_eval,
            circuit_layers=circuit_layers,
        )
        logger.note(
            f"counterfactual_faithfulness: {cf_faith:.4f} | "
            f"posctx_suppression_score: {sup_score:.4f}"
        )

        if sup_score < self.min_suppression_score:
            logger.reject(
                f"posctx_suppression_score {sup_score:.4f} < min_suppression_score {self.min_suppression_score}"
            )
            return None

        circuit.metadata.update(
            {
                "counterfactual_faithfulness": cf_faith,
                "posctx_suppression_score": sup_score,
                "ablation_suppression_score": sup_score,
                "seed_comp": seed_comp_idx,
                "seed_latent": seed_latent_idx,
                "n_nodes": len(circuit.nodes),
                "n_edges": len(circuit.edges),
                "n_supports": n_supports,
                "discovery_method": self.method_name,
                "neg_mode": self.neg_mode,
                "neg_selection": dict(self._last_neg_selection_metadata),
                "target_loss": target_loss,
                "target_pre_act": target_pre_act,
            }
        )
        logger.nodes(list(circuit.nodes.values()))
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit

    def _run_ablation_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        logger: CircuitLogger,
    ) -> tuple[Dict[FeatureID, float], float, float]:
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]
        sae = self.sae_bank.saes[seed_kind][seed_layer]
        w_seed = sae.encoder.weight[seed_latent_idx].detach()
        b_seed = sae._get_bias_eff()[seed_latent_idx].detach()

        if self.attribution_mode == "ig_baseline":
            return self._run_ig_baseline_hop(
                seed_layer, seed_kind, w_seed, b_seed, pos_tokens, pos_argmax, logger
            )

        instrument = SeedProjectionInstrument(self.sae_bank, seed_layer, seed_kind, w_seed, b_seed)
        was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            self.inference.forward(
                pos_tokens,
                patcher=instrument,
                grad_enabled=True,
                return_activations=False,
                tokenize_final=False,
            )
            if instrument.seed_pre_act is None:
                logger.note("SeedProjectionInstrument: seed_pre_act is None")
                return {}, 0.0, 0.0

            B = min(instrument.seed_pre_act.shape[0], pos_argmax.shape[0])
            batch_idx = torch.arange(B, device=instrument.seed_pre_act.device)
            pa = pos_argmax[:B].to(instrument.seed_pre_act.device).clamp(0, instrument.seed_pre_act.shape[1] - 1)
            pre_act_at_peak = instrument.seed_pre_act[:B][batch_idx, pa]
            loss = (pre_act_at_peak ** 2).mean()
            target_loss = float(loss.detach().item())
            target_pre_act = float(pre_act_at_peak.detach().mean().item())
            if abs(target_loss) < 1e-8:
                logger.note("near-zero positive-context ablation loss — skipping")
                return {}, target_loss, target_pre_act

            scores = compute_latent_ablation_scores(
                graph=instrument.graph,
                target_scalar=loss,
                seed_comp_idx=seed_comp_idx,
                n_kinds=n_kinds,
                kinds=kinds,
                top_k_supports=self.top_k_supports,
                min_active_count=self.min_active_count,
                active_count=self._active_count(),
                top_k_scope=self.top_k_scope,
            )
            return scores, target_loss, target_pre_act
        finally:
            del instrument
            if was_compiled:
                self.inference.enable_compile()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _run_ig_baseline_hop(
        self,
        seed_layer: int,
        seed_kind: str,
        w_seed: torch.Tensor,
        b_seed: torch.Tensor,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        logger: CircuitLogger,
    ) -> tuple[Dict[FeatureID, float], float, float]:
        """SFC-style integrated-gradients attribution (Marks et al. 2025)
        along the mean-ablation-floor -> natural path, with the seed's drive
        (pre-activation at probe positions) as the metric. Positive IG
        contributions are the support candidates."""

        from eval.ablation_faithfulness import collect_site_means, upstream_sites

        kinds = self.sae_bank.kinds
        n_kinds = len(kinds)
        sites = upstream_sites(self.sae_bank, seed_layer, seed_kind)
        if not sites:
            logger.note("ig_baseline: seed has no upstream sites")
            return {}, 0.0, 0.0
        site_floors = collect_site_means(self.inference, self.sae_bank, pos_tokens, sites)

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
            objective="drive",
            ig_steps=self.ig_steps,
        )
        logger.note(
            f"ig_baseline: drive floor {metric_floor:.4f} -> natural {metric_natural:.4f} "
            f"over {len(sites)} sites, {self.ig_steps} steps"
        )
        supports, _ = extract_signed_roles(
            scores_by_site,
            kinds=list(kinds),
            n_kinds=n_kinds,
            top_k_positive=self.top_k_supports,
            top_k_negative=0,
            min_active_count=self.min_active_count,
            active_count=self._active_count(),
            top_k_scope=self.top_k_scope,
        )
        return supports, metric_floor, metric_natural

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
