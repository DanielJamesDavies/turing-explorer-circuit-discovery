import torch
import gc
from typing import Optional, Any, Tuple, Dict, Set, cast

from .base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
from store.latent_stats import latent_stats
from eval.faithfulness import evaluate_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes
from eval.upstream_faithfulness import evaluate_upstream_faithfulness
from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.instrument.attribution import compute_latent_counterfactual_scores
from circuit.types.feature_id import FeatureID
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx


class SeedProjectionInstrument(SAEGraphInstrument):
    """
    SAEGraphInstrument subclass that captures the seed latent's encoder pre-activation
    during the forward pass.

    On negctx sequences the seed SAE latent is typically absent from top-k, so
    f_connected[..., seed_latent_idx] is identically zero and a backward through
    it produces no gradient signal.  Instead we project the pre-SAE residual
    stream x directly onto the seed's encoder direction:

        seed_pre_act = x @ W_enc[seed_latent_idx] + b_eff[seed_latent_idx]  -- [B, T]

    This is non-zero even when the seed never appears in top-k, and is fully
    differentiable w.r.t. all upstream leaf anchors via the identity passthrough
    (x - x.detach()) that SAEGraphInstrument adds to each layer's output.
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
    Discovers circuit nodes by running gradient attribution on negctx sequences —
    inputs semantically similar to the seed's context but where the seed is inactive.

    Two node types are discovered:
    - counterfactual_activator: upstream latents with large positive raw gradient
      on negctx (they would cause the seed to fire if active — their absence
      explains why the seed is suppressed on these sequences).
    - counterfactual_inhibitor: upstream latents with negative acts×gradient on
      negctx (they are active and causally suppressing the seed).

    Discovery runs one grad-enabled forward pass on negctx.  Evaluation runs on
    posctx (the standard probe dataset), testing whether the discovered nodes
    explain the seed's activation on its own context.
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
        activator_threshold: Optional[float] = None,
        inhibitor_threshold: Optional[float] = None,
        min_active_count: Optional[int] = None,
        max_neg_sequences: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
        min_faithfulness: Optional[float] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.counterfactual_gradient

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
        self.min_faithfulness = (
            min_faithfulness if min_faithfulness is not None
            else cast(float, cfg.min_faithfulness)
        )
        self.probe_batch_size = cast(int, config.discovery.probe_batch_size)

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

        # 2. Require negctx sequences
        neg_tokens = probe_data.neg_tokens[:self.max_neg_sequences]
        if neg_tokens.shape[0] == 0:
            logger.reject("no negctx sequences available")
            return None

        # 3. Seed node
        seed_node = CircuitNode(metadata={"feature_id": seed_fid, "role": "seed"})
        circuit.add_node(seed_node)
        fid_to_uuid: Dict[FeatureID, str] = {seed_fid: seed_node.uuid}

        # 4. Get the seed's mean activation on posctx — used as the MSE target
        pos_tokens_eval = probe_data.pos_tokens[:self.probe_batch_size]
        pos_argmax_eval = probe_data.pos_argmax[:self.probe_batch_size]
        target_act_pos = self._get_posctx_activation(
            seed_comp_idx, seed_latent_idx, pos_tokens_eval, pos_argmax_eval
        )
        logger.note(f"target_act_pos (mean seed act on posctx): {target_act_pos:.4f}")

        # 5. Negctx gradient pass
        activator_scores, inhibitor_scores = self._run_negctx_hop(
            seed_comp_idx, seed_latent_idx, neg_tokens, target_act_pos, logger
        )
        logger.stage(
            "negctx grad pass",
            1, 0,
            note=(
                f"{len(activator_scores)} absent activators, "
                f"{len(inhibitor_scores)} present inhibitors before thresholding"
            ),
        )

        # 6. Add absent activators
        n_activators = 0
        for upstream_fid, score in activator_scores.items():
            if score < self.activator_threshold:
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

        # 7. Add present inhibitors
        n_inhibitors = 0
        for upstream_fid, score in inhibitor_scores.items():
            if abs(score) < self.inhibitor_threshold:
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

        # 8. Evaluation — runs on posctx, layer-bounded to seed_layer
        circuit_layers: Set[int] = {
            node.feature_id.layer
            for node in circuit.nodes.values()
            if node.feature_id is not None
        }
        print(
            f"[CounterfactualGrad] Discovery complete: {len(circuit.nodes)} nodes, "
            f"{len(circuit.edges)} edges | circuit_layers={sorted(circuit_layers)}"
        )

        # Minimality pruning (optional)
        if self.pruning_threshold > 0:
            n_before = len(circuit.nodes)
            prune_non_minimal_nodes(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                pos_tokens_eval, pos_argmax=pos_argmax_eval,
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

        up_faith = evaluate_upstream_faithfulness(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            seed_layer, seed_kind, seed_latent_idx,
            pos_tokens_eval, pos_argmax=pos_argmax_eval,
            circuit_layers=circuit_layers,
        )

        final_f = evaluate_faithfulness(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            pos_tokens_eval, pos_argmax=pos_argmax_eval,
            circuit_layers=circuit_layers,
        )

        final_s = evaluate_sufficiency(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            pos_tokens_eval, probe_data.target_tokens[:self.probe_batch_size],
            pos_argmax=pos_argmax_eval,
            circuit_layers=circuit_layers,
        )

        final_c = evaluate_completeness(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            pos_tokens_eval, pos_argmax=pos_argmax_eval,
            circuit_layers=circuit_layers,
        )

        logger.eval(final_f, final_s, final_c)
        logger.note(f"upstream_faithfulness: {up_faith:.4f}")

        if up_faith < self.min_faithfulness:
            logger.reject(
                f"upstream_faithfulness {up_faith:.4f} < min_faithfulness {self.min_faithfulness}"
            )
            return None

        circuit.metadata.update({
            "faithfulness": final_f,
            "sufficiency": final_s,
            "completeness": final_c,
            "upstream_faithfulness": up_faith,
            "seed_comp": seed_comp_idx,
            "seed_latent": seed_latent_idx,
            "n_nodes": len(circuit.nodes),
            "n_edges": len(circuit.edges),
            "n_activators": n_activators,
            "n_inhibitors": n_inhibitors,
            "discovery_method": self.method_name,
        })
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit

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
            is_target = (top_indices == seed_latent_idx)
            target_acts = torch.where(is_target, top_acts, torch.zeros_like(top_acts)).sum(dim=-1)  # [B, T]
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

    def _run_negctx_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        neg_tokens: torch.Tensor,
        target_act_pos: float,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]:
        """
        Runs a single grad-enabled forward pass on negctx sequences using
        SeedProjectionInstrument, then calls compute_latent_counterfactual_scores
        to extract absent activators and present inhibitors in one backward pass.
        """
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]

        # Build encoder direction vectors for the seed latent
        sae = self.sae_bank.saes[seed_kind][seed_layer]
        w_seed = sae.encoder.weight[seed_latent_idx].detach()  # [d_model]
        b_seed = sae._get_bias_eff()[seed_latent_idx].detach()  # scalar

        instrument = SeedProjectionInstrument(
            self.sae_bank, seed_layer, seed_kind, w_seed, b_seed
        )

        self.inference.disable_compile()
        try:
            self.inference.forward(
                neg_tokens,
                patcher=instrument,
                grad_enabled=True,
                return_activations=False,
                tokenize_final=False,
            )

            if instrument.seed_pre_act is None:
                logger.note("SeedProjectionInstrument: seed_pre_act is None after forward")
                return {}, {}

            B = instrument.seed_pre_act.shape[0]
            batch_idx = torch.arange(B, device=instrument.seed_pre_act.device)
            pos_argmax_neg = instrument.seed_pre_act.argmax(dim=-1)  # [B]

            pre_act_at_peak = instrument.seed_pre_act[batch_idx, pos_argmax_neg]  # [B]
            target_tensor = torch.tensor(
                target_act_pos,
                device=pre_act_at_peak.device,
                dtype=pre_act_at_peak.dtype,
            )
            # MSE loss: measures how far negctx pre-activation is from the posctx target.
            # target_scalar = -loss so that gradients point toward increasing pre_act.
            loss = ((pre_act_at_peak - target_tensor) ** 2).mean()
            target_scalar = -loss

            if abs(target_scalar.item()) < 1e-8:
                logger.note("near-zero target_scalar on negctx — no gradient signal")
                return {}, {}

            logger.note(
                f"negctx MSE loss: {loss.item():.4f} | "
                f"target_act_pos: {target_act_pos:.4f} | "
                f"negctx pre_act mean: {pre_act_at_peak.mean().item():.4f}"
            )

            activator_scores, inhibitor_scores = compute_latent_counterfactual_scores(
                graph=instrument.graph,
                target_scalar=target_scalar,
                seed_layer=seed_layer,
                n_kinds=n_kinds,
                kinds=kinds,
                top_k_activators=self.top_k_activators,
                top_k_inhibitors=self.top_k_inhibitors,
                min_active_count=self.min_active_count,
                active_count=latent_stats.active_count,
            )

            return activator_scores, inhibitor_scores

        finally:
            del instrument
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.inference.enable_compile()
