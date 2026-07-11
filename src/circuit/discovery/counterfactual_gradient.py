import torch
import gc
from typing import Optional, Any, Tuple, Dict, Set, cast

from .base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
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
from utils.neg_context_selector import NegContextSelector


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


class CounterfactualGradientDiscovery(DiscoveryMethod):
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

        self.attribution_mode = (
            attribution_mode if attribution_mode is not None
            else cast(str, cfg.attribution_mode)
        )
        self.ig_steps = ig_steps if ig_steps is not None else cast(int, cfg.ig_steps)
        self.neg_mode = cast(str, cfg.neg_mode)
        self.distant_pool_size = cast(int, cfg.distant_pool_size)
        self.top_k_activators = (
            top_k_activators if top_k_activators is not None
            else cast(int, cfg.top_k_activators)
        )
        self.top_k_inhibitors = (
            top_k_inhibitors if top_k_inhibitors is not None
            else cast(int, cfg.top_k_inhibitors)
        )
        self.top_k_scope = (
            top_k_scope if top_k_scope is not None
            else cast(str, cfg.top_k_scope)
        )
        self.activator_threshold = (
            activator_threshold if activator_threshold is not None
            else cast(float, cfg.activator_threshold)
        )
        self.inhibitor_threshold = (
            inhibitor_threshold if inhibitor_threshold is not None
            else cast(float, cfg.inhibitor_threshold)
        )
        self.min_active_count = (
            min_active_count if min_active_count is not None
            else cast(int, cfg.min_active_count)
        )
        self.max_neg_sequences = (
            max_neg_sequences if max_neg_sequences is not None
            else cast(int, cfg.max_neg_sequences)
        )
        self.neg_batch_size = cast(int, cfg.neg_batch_size)
        self.pruning_threshold = (
            pruning_threshold if pruning_threshold is not None
            else cast(float, cfg.pruning_threshold)
        )
        self.min_faithfulness = (
            min_faithfulness if min_faithfulness is not None
            else cast(float, cfg.min_faithfulness)
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

        circuit = Circuit(name=f"CounterfactualGrad_S{seed_comp_idx}_{seed_latent_idx}")

        # 1. Build probe dataset
        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts)")
            return None

        logger.header(
            seed_layer, seed_kind, seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        # 2. Posctx eval slice — used for seed target activation and circuit evaluation.
        pos_tokens_eval = probe_data.pos_tokens[:self.probe_batch_size]
        pos_argmax_eval = probe_data.pos_argmax[:self.probe_batch_size]

        # 3. Source the contrast sequences according to neg_mode
        neg_tokens = self._get_neg_tokens(
            probe_data,
            seed_comp_idx, seed_latent_idx,
            logger,
        )
        if neg_tokens is None:
            return None

        # 4. Seed node
        seed_node = CircuitNode(metadata={"feature_id": seed_fid, "role": "seed"})
        circuit.add_node(seed_node)
        fid_to_uuid: Dict[FeatureID, str] = {seed_fid: seed_node.uuid}

        # 5. Get the seed's mean activation on posctx — used as the MSE target
        target_act_pos = self._get_posctx_activation(
            seed_comp_idx, seed_latent_idx, pos_tokens_eval, pos_argmax_eval
        )
        # Scale thresholds by target_act_pos so focal seeds (lower a_posctx) are not
        # disproportionately penalised. Gradient scores ≈ 2·a_posctx·(alignment), so
        # an absolute threshold is ~4× stricter for a seed with a_posctx=1 vs a_posctx=4.
        act_scale = max(target_act_pos, 0.1)
        effective_activator_threshold = self.activator_threshold * act_scale
        effective_inhibitor_threshold = self.inhibitor_threshold * act_scale
        logger.note(
            f"target_act_pos: {target_act_pos:.4f} | "
            f"effective thresholds — activator: {effective_activator_threshold:.4f}, "
            f"inhibitor: {effective_inhibitor_threshold:.4f}"
        )

        # 6. Attribution pass. "local" runs the contrast-sequence gradient
        # hop at the live negctx input; "ig_baseline" attributes along the
        # mean-ablation-floor -> natural-posctx path instead (SFC-style),
        # in which case negctx is used only by the evaluation step.
        if self.attribution_mode == "ig_baseline":
            activator_scores, inhibitor_scores = self._run_ig_baseline_hop(
                seed_comp_idx, seed_latent_idx, pos_tokens_eval, pos_argmax_eval,
                target_act_pos, logger,
            )
            pass_label = "ig_baseline grad pass"
        else:
            activator_scores, inhibitor_scores = self._run_contrast_hop(
                seed_comp_idx, seed_latent_idx, neg_tokens, target_act_pos, logger
            )
            pass_label = f"{self.neg_mode} grad pass"
        logger.stage(
            pass_label,
            1, 0,
            note=(
                f"{len(activator_scores)} absent activators, "
                f"{len(inhibitor_scores)} present inhibitors before thresholding"
            ),
        )

        # 7. Add absent activators
        n_activators = 0
        for upstream_fid, score in activator_scores.items():
            if score < effective_activator_threshold:
                continue
            upstream_comp, upstream_latent = upstream_fid.to_component_id(n_kinds, kinds)
            if latent_stats.active_count[upstream_comp, upstream_latent] < self.min_active_count:
                continue
            if upstream_fid not in fid_to_uuid:
                node = CircuitNode(metadata={
                    "feature_id": upstream_fid,
                    "role": "counterfactual_activator",
                    "attribution_score": score,
                })
                circuit.add_node(node)
                fid_to_uuid[upstream_fid] = node.uuid
            circuit.add_edge(fid_to_uuid[upstream_fid], seed_node.uuid, weight=score)
            n_activators += 1

        # 8. Add present inhibitors
        n_inhibitors = 0
        for upstream_fid, score in inhibitor_scores.items():
            if abs(score) < effective_inhibitor_threshold:
                continue
            upstream_comp, upstream_latent = upstream_fid.to_component_id(n_kinds, kinds)
            if latent_stats.active_count[upstream_comp, upstream_latent] < self.min_active_count:
                continue
            if upstream_fid not in fid_to_uuid:
                node = CircuitNode(metadata={
                    "feature_id": upstream_fid,
                    "role": "counterfactual_inhibitor",
                    "attribution_score": score,
                })
                circuit.add_node(node)
                fid_to_uuid[upstream_fid] = node.uuid
            circuit.add_edge(fid_to_uuid[upstream_fid], seed_node.uuid, weight=score)
            n_inhibitors += 1

        logger.stage(
            "circuit assembly",
            len(circuit.nodes), len(circuit.edges),
            note=f"{n_activators} activators, {n_inhibitors} inhibitors after thresholding",
        )

        if len(circuit.nodes) <= 1:
            logger.reject("no activators or inhibitors passed threshold")
            return None

        # 9. Evaluation — runs on posctx, layer-bounded to seed_layer
        circuit_layers: Set[int] = {
            node.feature_id.layer
            for node in circuit.nodes.values()
            if node.feature_id is not None
        }
        print(
            f"[CounterfactualGrad] Discovery complete: {len(circuit.nodes)} nodes, "
            f"{len(circuit.edges)} edges | circuit_layers={sorted(circuit_layers)}"
        )

        # Minimality pruning (optional) — uses cf_faith as the leave-one-out signal
        if self.pruning_threshold > 0:
            n_before = len(circuit.nodes)
            prune_non_minimal_nodes_cf(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                neg_tokens=neg_tokens,
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
                "after pruning", len(circuit.nodes), len(circuit.edges),
                note=f"removed {n_before - len(circuit.nodes)} nodes",
            )

        cf_faith, sup_score = evaluate_counterfactual_faithfulness(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            neg_tokens=neg_tokens,
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

        if cf_faith < self.min_faithfulness:
            logger.reject(
                f"counterfactual_faithfulness {cf_faith:.4f} < min_faithfulness {self.min_faithfulness}"
            )
            return None

        circuit.metadata.update({
            "counterfactual_faithfulness": cf_faith,
            "posctx_suppression_score": sup_score,
            "seed_comp": seed_comp_idx,
            "seed_latent": seed_latent_idx,
            "n_nodes": len(circuit.nodes),
            "n_edges": len(circuit.edges),
            "n_activators": n_activators,
            "n_inhibitors": n_inhibitors,
            "discovery_method": self.method_name,
            "neg_mode": self.neg_mode,
            "neg_selection": dict(self._last_neg_selection_metadata),
        })
        logger.nodes(list(circuit.nodes.values()))
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit

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
            selection_seed=int(cfg.selection_seed),
            filter_batch_size=int(cfg.filter_batch_size),
            load_window_size=int(cfg.load_window_size),
            logger=logger,
        )

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

    def _run_ig_baseline_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        target_act_pos: float,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
        """SFC-style integrated-gradients attribution (Marks et al. 2025)
        along the mean-ablation-floor -> natural-posctx path, so candidate
        scores linearise the circuit-only counterfactual. Positive IG
        contributions are activator candidates, negative are inhibitors."""

        from eval.ablation_faithfulness import collect_site_means, upstream_sites

        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]

        sites = upstream_sites(self.sae_bank, seed_layer, seed_kind)
        if not sites:
            logger.note("ig_baseline: seed has no upstream sites")
            return {}, {}
        site_floors = collect_site_means(self.inference, self.sae_bank, pos_tokens, sites)

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
            objective="gap",
            target_act=target_act_pos,
            ig_steps=self.ig_steps,
        )
        logger.note(
            f"ig_baseline: metric floor {metric_floor:.4f} -> natural {metric_natural:.4f} "
            f"over {len(sites)} sites, {self.ig_steps} steps"
        )
        return extract_signed_roles(
            scores_by_site,
            kinds=list(kinds),
            n_kinds=n_kinds,
            top_k_positive=self.top_k_activators,
            top_k_negative=self.top_k_inhibitors,
            min_active_count=self.min_active_count,
            active_count=latent_stats.active_count,
            top_k_scope=self.top_k_scope,
        )

    def _run_contrast_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        neg_tokens: torch.Tensor,
        target_act_pos: float,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
        """
        Runs grad-enabled forward passes on the contrast sequences using
        SeedProjectionInstrument, then calls compute_latent_counterfactual_scores
        to extract absent activators and present inhibitors.

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
                    )

                    for fid, score in batch_act.items():
                        all_act_scores.setdefault(fid, []).append(score)
                    for fid, score in batch_inh.items():
                        all_inh_scores.setdefault(fid, []).append(score)
                    n_valid_batches += 1

                finally:
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
