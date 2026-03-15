import torch
from typing import Optional, Any, Dict, List, Tuple, cast

from .base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
from circuit.sae_graph import SAEGraphInstrument
from circuit.circuit_logger import CircuitLogger
from eval.faithfulness import evaluate_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes
from pipeline.component_index import split_component_idx


class SFCAttributionPatching(DiscoveryMethod):
    """
    Sparse Feature Circuits (Marks et al. 2024) style circuit discovery.

    Implements attribution patching with a clean vs. baseline comparison:

      Node attribution:  effect = delta × gradient
        where gradient = d(metric_clean) / d(feature_activation)   [single backward]
        and   delta    = baseline_activation - clean_activation

      Edge attribution:  score = Jacobian(B←A) × delta_A
        where Jacobian is computed via compute_feature_attribution and
        weighted by the upstream delta rather than raw activation.

    Baseline modes (patch_mode in config):
      "mean_neg"  — average feature activations over neg_ctx sequences; more
                    realistic than zero ablation because it uses actual data
                    distributions.  Falls back to zero if no neg_ctx available.
      "zero"      — zero ablation (equivalent to SFC paper's patch=None path).

    The instrument uses stop_error_grad=True so gradients only propagate back
    through the SAE reconstruction path, giving clean SAE-mediated edge weights
    (matching the SFC paper's convention of zeroing residual.grad).
    """

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        node_threshold: Optional[float] = None,
        edge_threshold: Optional[float] = None,
        patch_mode: Optional[str] = None,
        max_neg: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
        probe_batch_size: Optional[int] = None,
        min_faithfulness: Optional[float] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.sfc_attribution_patching
        self.node_threshold = node_threshold if node_threshold is not None else cast(float, cfg.node_threshold or 0.01)
        self.edge_threshold = edge_threshold if edge_threshold is not None else cast(float, cfg.edge_threshold or 0.001)
        self.patch_mode = patch_mode or cast(str, cfg.patch_mode or "mean_neg")
        self.max_neg = max_neg if max_neg is not None else cast(int, cfg.max_neg or 8)
        self.pruning_threshold = pruning_threshold if pruning_threshold is not None else cast(float, cfg.pruning_threshold or 0.0)
        self.probe_batch_size = probe_batch_size or cast(int, config.discovery.probe_batch_size or 8)
        self.min_faithfulness = min_faithfulness or cast(float, config.discovery.min_faithfulness or 0.3)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """SFC attribution patching — full pipeline (Phases 2–4)."""
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "sfc_attribution_patching")
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
        # 1. Probe dataset
        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts found)")
            return None

        n_probe = min(self.probe_batch_size, probe_data.pos_tokens.shape[0])
        probe_tokens = probe_data.pos_tokens[:n_probe]
        probe_argmax = probe_data.pos_argmax[:n_probe]
        probe_targets = probe_data.target_tokens[:n_probe]

        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, len(self.sae_bank.kinds))
        seed_kind = self.sae_bank.kinds[seed_kind_idx]
        seed_key = (seed_layer, seed_kind, seed_latent_idx)

        logger.header(
            seed_layer, seed_kind, seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )
        logger.note(f"probe batch: {n_probe} sequences  patch_mode={self.patch_mode}")

        # 2. Clean forward pass — stop_error_grad=True ensures gradients travel
        #    only through SAE reconstruction paths, matching the SFC paper.
        self.inference.disable_compile()
        instrument = SAEGraphInstrument(self.sae_bank, stop_error_grad=True)
        _, logits, _ = self.inference.forward(
            probe_tokens,
            patcher=instrument,
            grad_enabled=True,
            return_activations=False,
            all_logits=True,
            tokenize_final=False,
        )
        self.inference.enable_compile()

        if logits is None:
            del instrument
            logger.reject("forward pass returned no logits")
            return None

        logger.note("clean forward pass complete")

        # 3. Single backward from the target-token logit at the seed's peak position.
        B = probe_tokens.shape[0]
        batch_idx = torch.arange(B, device=logits.device)
        target_ids = probe_targets[batch_idx, probe_argmax.to(probe_targets.device)].to(logits.device)
        metric_clean = logits[batch_idx, probe_argmax, target_ids].sum()

        all_anchors = instrument.graph.all_anchors()
        if not all_anchors:
            del instrument
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.reject("no gradient anchors found in computation graph")
            return None

        grads_list = torch.autograd.grad(
            metric_clean, all_anchors, retain_graph=True, allow_unused=True
        )

        # 4. Map each gradient back to its (layer, kind) anchor.
        grads_map: Dict[Tuple[int, str], torch.Tensor] = {}
        grad_iter = iter(grads_list)
        for (layer, kind), steps in instrument.graph.activations.items():
            for acts_grad, _, _ in steps:
                g = next(grad_iter)
                grads_map[(layer, kind)] = (
                    g if g is not None else torch.zeros_like(acts_grad.data)
                )

        # 5. Compute baseline (mean over neg_ctx sequences, or zero ablation).
        n_neg_used = min(self.max_neg, probe_data.neg_tokens.shape[0])
        baseline = self._compute_baseline(probe_data.neg_tokens)
        logger.note(
            f"baseline computed  patch_mode={self.patch_mode}  "
            f"n_neg_used={n_neg_used if self.patch_mode == 'mean_neg' else 0}"
        )

        # 6. Node attribution: effect = delta × gradient.
        node_effects: Dict[Tuple[int, str, int], float] = {}

        for (layer, kind), steps in instrument.graph.activations.items():
            for acts_grad, _, top_indices in steps:
                acts = acts_grad.data
                g    = grads_map[(layer, kind)]
                base = baseline.get((layer, kind))

                dev = acts.device
                b_idx = torch.arange(B, device=dev)
                pos   = probe_argmax.to(dev)

                acts_at_pos = acts[b_idx, pos]
                g_at_pos    = g[b_idx, pos]
                idx_at_pos  = top_indices[b_idx, pos].long()

                if base is not None:
                    baseline_at_pos = base[idx_at_pos].to(dev)
                else:
                    baseline_at_pos = torch.zeros_like(acts_at_pos)

                delta  = baseline_at_pos - acts_at_pos
                effect = delta * g_at_pos

                d_sae = self.sae_bank.d_sae
                effect_dense = torch.zeros(d_sae, dtype=torch.float32)
                effect_dense.scatter_add_(
                    0,
                    idx_at_pos.cpu().reshape(-1),
                    effect.cpu().float().reshape(-1),
                )

                for l_idx in effect_dense.nonzero(as_tuple=True)[0].tolist():
                    key = (layer, kind, l_idx)
                    node_effects[key] = node_effects.get(key, 0.0) + float(effect_dense[l_idx])

        if seed_key not in node_effects:
            node_effects[seed_key] = 0.0

        n_candidates = len(node_effects)

        # 7. Build circuit nodes — seed is unconditional; others need |effect| ≥ threshold.
        circuit = Circuit(name=f"SFCAttrPatch_S{seed_comp_idx}_{seed_latent_idx}")
        node_id_map: Dict[Tuple[int, str, int], str] = {}

        for key, score in node_effects.items():
            if key != seed_key and abs(score) < self.node_threshold:
                continue
            layer, kind, latent_idx = key
            role = "seed" if key == seed_key else "attributed"
            node = CircuitNode(metadata={
                "layer_idx": layer,
                "latent_idx": latent_idx,
                "kind": kind,
                "role": role,
                "effect_score": score,
            })
            circuit.add_node(node)
            node_id_map[key] = node.uuid

        logger.stage(
            "node attribution", len(circuit.nodes), 0,
            note=(
                f"{n_candidates} candidates, {len(circuit.nodes)} passed "
                f"threshold={self.node_threshold}"
            ),
        )

        if len(circuit.nodes) <= 1:
            del instrument
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.reject(
                f"only seed passed node threshold ({self.node_threshold})"
            )
            return None

        # --- Phase 3: Edge attribution ---
        kind_order = ["attn", "mlp", "resid"]
        included_keys_list = list(node_id_map.keys())

        for b_layer, b_kind, b_latent in included_keys_list:
            if (b_layer, b_kind) not in instrument.graph.activations:
                continue

            upstream_keys = [
                k for k in included_keys_list
                if k[0] < b_layer
                or (k[0] == b_layer and kind_order.index(k[1]) < kind_order.index(b_kind))
            ]
            if not upstream_keys:
                continue

            _, target_acts_connected, target_indices = instrument.graph.get_latents(b_layer, b_kind, step=0)
            dev   = instrument.graph.device
            b_idx = torch.arange(B, device=dev)
            pos   = probe_argmax.to(dev)

            vals_at_pos = target_indices[b_idx, pos]
            matches     = (vals_at_pos == b_latent)
            if not matches.any():
                continue

            target_sum = target_acts_connected[b_idx, pos][matches].sum()
            if target_sum.grad_fn is None:
                continue

            upstream_by_lk: Dict[Tuple[int, str], List[int]] = {}
            for a_layer, a_kind, a_latent in upstream_keys:
                if (a_layer, a_kind) in instrument.graph.activations:
                    upstream_by_lk.setdefault((a_layer, a_kind), []).append(a_latent)
            if not upstream_by_lk:
                continue

            anchor_tensors: List[torch.Tensor] = []
            anchor_layer_kinds: List[Tuple[int, str]] = []
            for lk in upstream_by_lk:
                for acts_grad_a, _, _ in instrument.graph.activations[lk]:
                    anchor_tensors.append(acts_grad_a)
                    anchor_layer_kinds.append(lk)

            try:
                edge_grads = torch.autograd.grad(
                    target_sum, anchor_tensors, retain_graph=True, allow_unused=True
                )
            except RuntimeError:
                continue

            for (a_layer, a_kind), grad in zip(anchor_layer_kinds, edge_grads):
                if grad is None:
                    continue

                acts_grad_a, _, top_indices_a = instrument.graph.get_latents(a_layer, a_kind, step=0)
                dev_a   = acts_grad_a.device
                b_idx_a = torch.arange(B, device=dev_a)
                pos_a   = probe_argmax.to(dev_a)

                acts_a  = acts_grad_a.data[b_idx_a, pos_a]
                grad_a  = grad[b_idx_a, pos_a]
                idx_a   = top_indices_a[b_idx_a, pos_a].long()

                base_a = baseline.get((a_layer, a_kind))
                if base_a is not None:
                    baseline_a = base_a[idx_a].to(dev_a)
                else:
                    baseline_a = torch.zeros_like(acts_a)

                delta_a = baseline_a - acts_a
                attr    = delta_a * grad_a

                for a_latent in upstream_by_lk[(a_layer, a_kind)]:
                    if (a_layer, a_kind, a_latent) not in node_id_map:
                        continue
                    mask = (idx_a == a_latent)
                    if not mask.any():
                        continue
                    score = float(attr[mask].sum().item())
                    if abs(score) < self.edge_threshold:
                        continue
                    circuit.add_edge(
                        node_id_map[(a_layer, a_kind, a_latent)],
                        node_id_map[(b_layer, b_kind, b_latent)],
                        weight=score,
                    )

        logger.stage(
            "edge attribution", len(circuit.nodes), len(circuit.edges),
            note=f"threshold={self.edge_threshold}",
        )

        del instrument
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(circuit.nodes) <= 1:
            logger.reject("circuit collapsed to ≤1 node after edge pass")
            return None

        # --- Phase 4: Pruning, evaluation, faithfulness gate ---
        # Zero ablation is used for evaluation: non-circuit features are replaced
        # with 0 rather than neg_avg.  This avoids spurious signal injection when
        # neg_avg > actual for a feature (which causes complement MSE > baseline MSE
        # and completeness > 1, making scores uninterpretable).
        # Note: neg_avg is still used above for the delta computation (node/edge
        # attribution), where it represents the meaningful counterfactual baseline.

        if self.pruning_threshold > 0:
            n_before = len(circuit.nodes)
            prune_non_minimal_nodes(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
                threshold=self.pruning_threshold,
            )
            logger.stage(
                "after pruning", len(circuit.nodes), len(circuit.edges),
                note=f"removed {n_before - len(circuit.nodes)} nodes",
            )

        final_f = evaluate_faithfulness(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
        )
        final_s = evaluate_sufficiency(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            probe_data.pos_tokens, probe_data.target_tokens,
            pos_argmax=probe_data.pos_argmax,
        )
        final_c = evaluate_completeness(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
        )

        logger.eval(final_f, final_s, final_c)

        if final_f < self.min_faithfulness:
            logger.reject(
                f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}"
            )
            return None

        circuit.metadata.update({
            "faithfulness": final_f,
            "sufficiency": final_s,
            "completeness": final_c,
            "seed_comp": seed_comp_idx,
            "seed_latent": seed_latent_idx,
            "n_nodes": len(circuit.nodes),
            "n_edges": len(circuit.edges),
            "discovery_method": "sfc_attribution_patching",
            "patch_mode": self.patch_mode,
            "node_threshold": self.node_threshold,
            "edge_threshold": self.edge_threshold,
        })
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_baseline(
        self,
        neg_tokens: torch.Tensor,
    ) -> Dict[Tuple[int, str], torch.Tensor]:
        """
        Returns a per-latent mean activation baseline for the delta computation.

        patch_mode == "mean_neg":
            Runs neg_tokens through the model (no grad) and averages each SAE
            feature's activation over all sequences and all token positions.
            Falls back to zero-ablation if neg_tokens is empty.

        patch_mode == "zero":
            Returns an empty dict.  Callers treat a missing key as zero baseline,
            so delta = 0 - clean_activation = -clean_activation.
        """
        if self.patch_mode == "zero" or neg_tokens.shape[0] == 0:
            return {}

        n_neg = min(self.max_neg, neg_tokens.shape[0])
        neg_tokens = neg_tokens[:n_neg]

        sum_acts: Dict[Tuple[int, str], torch.Tensor] = {}
        hit_counts: Dict[Tuple[int, str], torch.Tensor] = {}

        def callback(layer_idx: int, activations: Tuple[torch.Tensor, ...]) -> None:
            for kind_idx, kind in enumerate(self.sae_bank.kinds):
                act = activations[kind_idx]
                top_acts, top_indices = self.sae_bank.encode(act, kind, layer_idx)
                key = (layer_idx, kind)
                d_sae = self.sae_bank.d_sae
                device = act.device

                if key not in sum_acts:
                    sum_acts[key] = torch.zeros(d_sae, device=device, dtype=torch.float32)
                    hit_counts[key] = torch.zeros(d_sae, device=device, dtype=torch.float32)

                flat_idx  = top_indices.long().reshape(-1)
                flat_acts = top_acts.float().reshape(-1)
                sum_acts[key].scatter_add_(0, flat_idx, flat_acts)
                hit_counts[key].scatter_add_(0, flat_idx, torch.ones_like(flat_acts))

        self.inference.forward(
            neg_tokens,
            activations_callback=callback,
            return_activations=False,
            tokenize_final=False,
        )

        return {
            key: sum_acts[key] / hit_counts[key].clamp(min=1.0)
            for key in sum_acts
        }
