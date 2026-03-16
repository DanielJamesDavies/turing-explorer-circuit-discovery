import torch
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple, List, cast

from .base import DiscoveryMethod
from ..sparse_act import SparseAct
from config import config
from store.circuits import Circuit, CircuitNode
from circuit.sae_graph import SAEGraphInstrument
from circuit.circuit_logger import CircuitLogger
from eval.faithfulness import evaluate_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes
from pipeline.component_index import split_component_idx
from model.hooks import multi_patch, multi_stop_grad


@dataclass
class TopKState:
    """
    Memory-efficient sparse SAE state. Stores only the k active features per token
    instead of a full [B, T, d_sae] dense tensor — reducing persistent state memory
    by ~320× (k=128 vs d_sae=40960).
    """
    vals: torch.Tensor  # [B, T, k] float32 — top-k activation values
    idx:  torch.Tensor  # [B, T, k] long    — top-k feature indices
    res:  torch.Tensor  # [B, T, d_model]   — SAE reconstruction error

    @property
    def device(self) -> torch.device:
        return self.vals.device

    def zeros_like(self) -> 'TopKState':
        """Returns a zeroed state with the same index structure."""
        return TopKState(
            vals=torch.zeros_like(self.vals),
            idx=self.idx.clone(),
            res=torch.zeros_like(self.res),
        )


class IGPatcher:
    """
    Patches the forward pass with interpolated SAE states for Integrated Gradients.

    Key optimisation: uses small [B, T, k] leaf tensors (not [B, T, d_sae]) for
    autograd, reducing the gradient-tracking footprint by ~320×. The full [B, T, d_sae]
    dense tensor is built ephemerally inside each forward pass via scatter_add_ and is
    freed immediately after backward — it never accumulates in memory.

    scatter_add_ ensures correct interpolation for features active in both clean and
    patch top-k sets:
      dense[b, t, i] = (1-α)*clean_val_i + α*patch_val_i   for i in both
      dense[b, t, i] = (1-α)*clean_val_i                    for i in clean only
      dense[b, t, i] =          α*patch_val_i               for i in patch only
    """

    def __init__(
        self,
        bank: Any,
        clean_states: Dict[Tuple[int, str], TopKState],
        patch_states: Dict[Tuple[int, str], TopKState],
        alpha: float,
        d_sae: int,
    ):
        self.bank = bank
        self.clean_states = clean_states
        self.patch_states = patch_states
        self.alpha = alpha
        self.d_sae = d_sae
        # Populated by __call__ → transform; read after backward to collect grads
        self.leaves: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __call__(self, model: Any):
        self.leaves.clear()

        def transform(layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
            lk = (layer_idx, kind)
            clean = self.clean_states.get(lk)
            patch = self.patch_states.get(lk)
            if clean is None or patch is None:
                return x

            B, T, _ = x.shape
            dtype = x.dtype
            dev = x.device

            # Small leaf tensors — [B, T, k] not [B, T, d_sae]
            c_leaf = ((1 - self.alpha) * clean.vals.to(dev)).detach().float().requires_grad_(True)
            p_leaf = (self.alpha       * patch.vals.to(dev)).detach().float().requires_grad_(True)
            r_interp = ((1 - self.alpha) * clean.res.to(dev) + self.alpha * patch.res.to(dev))
            r_leaf = r_interp.detach().to(dtype).requires_grad_(True)

            # Ephemeral dense tensor — only exists in the forward graph, freed after backward
            dense = torch.zeros(B, T, self.d_sae, device=dev, dtype=dtype)
            dense.scatter_add_(-1, clean.idx.to(dev), c_leaf.to(dtype))
            dense.scatter_add_(-1, patch.idx.to(dev), p_leaf.to(dtype))

            self.leaves[lk] = (c_leaf, p_leaf, r_leaf)
            return self.bank.decode(dense, kind, layer_idx) + r_leaf

        return multi_patch(model, transform)


class SFCAttributionPatching(DiscoveryMethod):
    """
    Sparse Feature Circuits (Marks et al. 2024) style circuit discovery.
    1-to-1 replica of the feature-circuits algorithm with memory-efficient
    sparse state representation.

    Implements:
      - Integrated Gradients (IG) for node attribution, using [B,T,k] leaf tensors
      - JVP-based edge attribution with intermediate stop-grads
      - Joint attribution to SAE features and reconstruction error (residual)
      - Aggregation across all token positions and batch
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
        ig_steps: Optional[int] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.sfc_attribution_patching
        self.node_threshold    = node_threshold    if node_threshold    is not None else cast(float, cfg.node_threshold    or 0.1)
        self.edge_threshold    = edge_threshold    if edge_threshold    is not None else cast(float, cfg.edge_threshold    or 0.01)
        self.patch_mode        = patch_mode        or cast(str,   cfg.patch_mode        or "mean_neg")
        self.max_neg           = max_neg           if max_neg           is not None else cast(int,   cfg.max_neg           or 8)
        self.pruning_threshold = pruning_threshold if pruning_threshold is not None else cast(float, cfg.pruning_threshold or 0.0)
        self.probe_batch_size  = probe_batch_size  or cast(int,   config.discovery.probe_batch_size  or 8)
        self.min_faithfulness  = min_faithfulness  or cast(float, config.discovery.min_faithfulness  or 0.3)
        self.ig_steps          = ig_steps          if ig_steps          is not None else cast(int,   getattr(cfg, "ig_steps", 10))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
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

        n_probe       = min(self.probe_batch_size, probe_data.pos_tokens.shape[0])
        probe_tokens  = probe_data.pos_tokens[:n_probe]
        probe_argmax  = probe_data.pos_argmax[:n_probe]
        probe_targets = probe_data.target_tokens[:n_probe]

        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, len(self.sae_bank.kinds))
        seed_kind = self.sae_bank.kinds[seed_kind_idx]
        seed_key  = (seed_layer, seed_kind, seed_latent_idx)

        logger.header(seed_layer, seed_kind, seed_latent_idx,
                      probe_data.pos_tokens.shape[0], probe_data.neg_tokens.shape[0])
        logger.note(f"probe batch: {n_probe} sequences  ig_steps={self.ig_steps}")

        # 2. Collect clean and patch states (sparse TopKState — ~320× less memory than dense)
        clean_states = self._get_all_states(probe_tokens)

        n_neg = probe_data.neg_tokens.shape[0]
        if self.patch_mode == "zero" or n_neg == 0:
            patch_states = {lk: s.zeros_like() for lk, s in clean_states.items()}
        else:
            neg_tokens = probe_data.neg_tokens[:min(self.max_neg, n_neg)]
            if neg_tokens.shape[0] >= n_probe:
                patch_states = self._get_all_states(neg_tokens[:n_probe])
            else:
                # Fewer neg sequences than probe — fall back to zero ablation
                patch_states = {lk: s.zeros_like() for lk, s in clean_states.items()}

        # 3. Node attribution via Integrated Gradients (IG)
        #    Returns aggregated [d_sae] node scores and sparse intermediate data
        node_scores, res_scores, sparse_data = self._pe_ig(
            clean_states, patch_states, probe_tokens, probe_argmax, probe_targets
        )
        logger.note(f"IG complete  {self.ig_steps} steps  patch_mode={self.patch_mode}")

        # 4. Build circuit nodes
        circuit = Circuit(name=f"SFCAttrPatch_S{seed_comp_idx}_{seed_latent_idx}")
        node_id_map: Dict[Tuple[int, str, int], str] = {}
        resid_node_id_map: Dict[Tuple[int, str], str] = {}
        # Pre-build active latent index lists for O(1) upstream lookup in edge loop
        active_latents: Dict[Tuple[int, str], List[int]] = {}

        for (layer, kind), scores in node_scores.items():
            for l_idx in scores.abs().nonzero(as_tuple=True)[0].tolist():
                score = float(scores[l_idx].item())
                key   = (layer, kind, l_idx)
                if key != seed_key and abs(score) < self.node_threshold:
                    continue
                role = "seed" if key == seed_key else "attributed"
                node = CircuitNode(metadata={
                    "layer_idx": layer, "latent_idx": l_idx, "kind": kind,
                    "role": role, "effect_score": score,
                })
                circuit.add_node(node)
                node_id_map[key] = node.uuid
                active_latents.setdefault((layer, kind), []).append(l_idx)

            res_score = res_scores.get((layer, kind), 0.0)
            if abs(res_score) >= self.node_threshold:
                node = CircuitNode(metadata={
                    "layer_idx": layer, "kind": kind,
                    "role": "residual", "effect_score": res_score,
                })
                circuit.add_node(node)
                resid_node_id_map[(layer, kind)] = node.uuid

        logger.stage("node attribution", len(circuit.nodes), 0,
                     note=f"threshold={self.node_threshold}")

        if len(circuit.nodes) <= 1:
            logger.reject(f"only seed passed node threshold ({self.node_threshold})")
            return None

        # 5. Edge attribution via JVP with intermediate stop-grads
        #    Materialise full [B,T,d_sae] grads/deltas only for submodules with active nodes
        active_submods = sorted(
            set(active_latents.keys()) | set(resid_node_id_map.keys()),
            key=lambda x: (x[0], ["attn", "mlp", "resid"].index(x[1]))
        )

        grads_mat:  Dict[Tuple[int, str], SparseAct] = {}
        deltas_mat: Dict[Tuple[int, str], SparseAct] = {}
        for lk in active_submods:
            if lk in sparse_data:
                grads_mat[lk], deltas_mat[lk] = self._materialize_grad_delta(lk, sparse_data)

        # Single instrumented forward pass for the JVP graph
        self.inference.disable_compile()
        instrument = SAEGraphInstrument(self.sae_bank, stop_error_grad=True)
        self.inference.forward(probe_tokens, patcher=instrument,
                               grad_enabled=True, return_activations=False)
        self.inference.enable_compile()

        for downstream in reversed(active_submods):
            d_layer, d_kind = downstream
            if downstream not in grads_mat:
                continue
            if (d_layer, d_kind) not in instrument.graph.activations:
                continue

            _, d_state_conn = instrument.graph.get_latents(d_layer, d_kind)
            # [B, T, d_sae+1] — sparse (mostly zeros); backprop from specific target index
            backprop_vec = (grads_mat[downstream] @ d_state_conn).to_tensor()

            # Build target list: (feature_index_in_backprop_vec, circuit_uuid)
            targets_list: List[Tuple[int, str]] = [
                (l_idx, node_id_map[(d_layer, d_kind, l_idx)])
                for l_idx in active_latents.get(downstream, [])
            ]
            if downstream in resid_node_id_map:
                targets_list.append((self.sae_bank.d_sae, resid_node_id_map[downstream]))

            if not targets_list:
                continue

            for target_idx, target_uuid in targets_list:
                for upstream in active_submods[:active_submods.index(downstream)]:
                    u_layer, u_kind = upstream
                    if upstream not in deltas_mat:
                        continue

                    # Intermediate stop-grads: same-layer submods between upstream and downstream
                    stop_grads = [
                        mid for mid in active_submods[
                            active_submods.index(upstream) + 1 : active_submods.index(downstream)
                        ]
                        if mid[0] == d_layer
                    ]

                    with multi_stop_grad(self.inference.model, stop_grads):
                        self.inference.model.zero_grad()
                        backprop_vec[:, :, target_idx].sum().backward(retain_graph=True)

                        u_state_grad, _ = instrument.graph.get_latents(u_layer, u_kind)
                        attr      = u_state_grad.grad @ deltas_mat[upstream]
                        score_act = attr.sum(dim=1).mean(dim=0)

                        # Feature → target edges  (only iterate over known active indices)
                        if score_act.act is not None:
                            for u_latent in active_latents.get((u_layer, u_kind), []):
                                score = float(score_act.act[u_latent].item())
                                if abs(score) >= self.edge_threshold:
                                    circuit.add_edge(
                                        node_id_map[(u_layer, u_kind, u_latent)],
                                        target_uuid, weight=score,
                                    )

                        # Residual → target edge
                        if (u_layer, u_kind) in resid_node_id_map and score_act.resc is not None:
                            score = float(score_act.resc.item())
                            if abs(score) >= self.edge_threshold:
                                circuit.add_edge(
                                    resid_node_id_map[(u_layer, u_kind)],
                                    target_uuid, weight=score,
                                )

        del instrument
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.stage("edge attribution", len(circuit.nodes), len(circuit.edges),
                     note=f"threshold={self.edge_threshold}")

        if len(circuit.nodes) <= 1:
            logger.reject("circuit collapsed to ≤1 node after edge pass")
            return None

        # 6. Pruning, evaluation, faithfulness gate
        if self.pruning_threshold > 0:
            n_before = len(circuit.nodes)
            prune_non_minimal_nodes(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_tokens, pos_argmax=probe_argmax,
                threshold=self.pruning_threshold,
            )
            logger.stage("after pruning", len(circuit.nodes), len(circuit.edges),
                         note=f"removed {n_before - len(circuit.nodes)} nodes")

        final_f = evaluate_faithfulness(self.inference, self.sae_bank, self.avg_acts, circuit,
                                        probe_tokens, pos_argmax=probe_argmax)
        final_s = evaluate_sufficiency(self.inference, self.sae_bank, self.avg_acts, circuit,
                                       probe_tokens, probe_targets, pos_argmax=probe_argmax)
        final_c = evaluate_completeness(self.inference, self.sae_bank, self.avg_acts, circuit,
                                        probe_tokens, pos_argmax=probe_argmax)
        logger.eval(final_f, final_s, final_c)

        if final_f < self.min_faithfulness:
            logger.reject(f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}")
            return None

        circuit.metadata.update({
            "faithfulness": final_f, "sufficiency": final_s, "completeness": final_c,
            "seed_comp": seed_comp_idx, "seed_latent": seed_latent_idx,
            "n_nodes": len(circuit.nodes), "n_edges": len(circuit.edges),
            "discovery_method": "sfc_attribution_patching",
            "patch_mode": self.patch_mode,
            "node_threshold": self.node_threshold, "edge_threshold": self.edge_threshold,
        })
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_all_states(self, tokens: torch.Tensor) -> Dict[Tuple[int, str], TopKState]:
        """
        Runs a no-grad forward pass and collects SAE states as TopKState objects.
        Stores only (top_k_vals, top_k_idx, residual) — ~320× less memory than a
        full dense [B, T, d_sae] SparseAct.
        """
        states: Dict[Tuple[int, str], TopKState] = {}
        d_sae = self.sae_bank.d_sae

        def callback(layer_idx: int, activations: Tuple[torch.Tensor, ...]) -> None:
            for kind_idx, kind in enumerate(self.sae_bank.kinds):
                act = activations[kind_idx]
                top_acts, top_idx = self.sae_bank.encode(act, kind, layer_idx)
                B, T, _ = act.shape
                dtype = act.dtype
                # Ephemeral dense tensor just to compute residual, then freed
                dense = torch.zeros(B, T, d_sae, device=act.device, dtype=dtype)
                dense.scatter_(-1, top_idx.long(), top_acts.to(dtype))
                x_hat = self.sae_bank.decode(dense, kind, layer_idx)
                states[(layer_idx, kind)] = TopKState(
                    vals=top_acts.detach().float(),
                    idx=top_idx.detach().long(),
                    res=(act - x_hat).detach().float(),
                )

        self.inference.forward(tokens, activations_callback=callback, return_activations=False)
        return states

    def _pe_ig(
        self,
        clean_states: Dict[Tuple[int, str], TopKState],
        patch_states:  Dict[Tuple[int, str], TopKState],
        tokens:  torch.Tensor,
        argmax:  torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[
        Dict[Tuple[int, str], torch.Tensor],  # node_scores: [d_sae] per lk
        Dict[Tuple[int, str], float],          # res_scores: scalar per lk
        Dict[Tuple[int, str], Any],            # sparse_data for later edge materialisation
    ]:
        """
        Integrated Gradients node attribution using sparse [B,T,k] leaf tensors.

        Peak VRAM during this function:
          - Autograd graph: 36 × ephemeral [B,T,d_sae] decoder intermediates (~5.8 GB)
            These are freed immediately after each step's backward.
          - Leaf tensors: 36 × [B,T,k] × 3 (~150 MB total)
          No dense [B,T,d_sae] tensors accumulate in memory across steps.
        """
        B         = tokens.shape[0]
        batch_idx = torch.arange(B, device=tokens.device)
        d_sae     = self.sae_bank.d_sae

        # Sparse gradient accumulators — [B, T, k] instead of [B, T, d_sae]
        grad_sum_c: Dict[Tuple[int, str], torch.Tensor] = {
            lk: torch.zeros_like(s.vals) for lk, s in clean_states.items()
        }
        grad_sum_p: Dict[Tuple[int, str], torch.Tensor] = {
            lk: torch.zeros_like(patch_states[lk].vals) for lk in clean_states
        }
        grad_sum_r: Dict[Tuple[int, str], torch.Tensor] = {
            lk: torch.zeros_like(s.res) for lk, s in clean_states.items()
        }

        self.inference.disable_compile()
        for step in range(self.ig_steps):
            alpha   = step / self.ig_steps
            patcher = IGPatcher(self.sae_bank, clean_states, patch_states, alpha, d_sae)
            _, logits, _ = self.inference.forward(
                tokens, patcher=patcher, grad_enabled=True,
                return_activations=False, all_logits=True,
            )
            target_ids = targets[batch_idx, argmax.to(targets.device)].to(logits.device)
            logits[batch_idx, argmax, target_ids].sum().backward()  # No retain_graph

            for lk, (c_leaf, p_leaf, r_leaf) in patcher.leaves.items():
                if c_leaf.grad is not None:
                    grad_sum_c[lk].add_(c_leaf.grad.float())
                if p_leaf.grad is not None:
                    grad_sum_p[lk].add_(p_leaf.grad.float())
                if r_leaf.grad is not None:
                    grad_sum_r[lk].add_(r_leaf.grad.float())
        self.inference.enable_compile()

        node_scores: Dict[Tuple[int, str], torch.Tensor] = {}
        res_scores:  Dict[Tuple[int, str], float]         = {}
        sparse_data: Dict[Tuple[int, str], Any]           = {}

        for lk in clean_states:
            clean = clean_states[lk]
            patch = patch_states[lk]

            mean_g_c = grad_sum_c[lk] / self.ig_steps  # [B, T, k]
            mean_g_p = grad_sum_p[lk] / self.ig_steps  # [B, T, k]
            mean_g_r = grad_sum_r[lk] / self.ig_steps  # [B, T, d_model]

            # Effect = mean_grad * delta, scattered into [d_sae] (no full tensor kept)
            # Contributions from clean positions (delta component: -clean_vals)
            # Contributions from patch positions (delta component: +patch_vals)
            # For features in both: scatter_add_ sums them = mean_grad*(patch-clean) ✓
            node_score = torch.zeros(d_sae, device=clean.device)
            node_score.scatter_add_(
                0,
                clean.idx.reshape(-1),
                (mean_g_c * (-clean.vals)).reshape(-1).float(),
            )
            node_score.scatter_add_(
                0,
                patch.idx.reshape(-1),
                (mean_g_p * patch.vals).reshape(-1).float(),
            )
            node_scores[lk] = node_score / B  # mean over batch

            delta_res = patch.res.float() - clean.res.float()
            res_scores[lk] = float(
                (mean_g_r * delta_res).sum(dim=-1).sum(dim=1).mean(dim=0).item()
            )

            # Store only sparse ingredients — full [B,T,d_sae] built lazily in
            # _materialize_grad_delta only for submodules that have active nodes
            sparse_data[lk] = (mean_g_c, mean_g_p, mean_g_r,
                                clean.idx, patch.idx,
                                clean.vals, patch.vals, delta_res)

        return node_scores, res_scores, sparse_data

    def _materialize_grad_delta(
        self,
        lk: Tuple[int, str],
        sparse_data: Dict[Tuple[int, str], Any],
    ) -> Tuple[SparseAct, SparseAct]:
        """
        Materialise full [B, T, d_sae] SparseActs for one submodule on demand.
        Only called for submodules that have nodes passing the threshold.
        """
        mean_g_c, mean_g_p, mean_g_r, c_idx, p_idx, c_vals, p_vals, delta_res = sparse_data[lk]
        B, T, k = mean_g_c.shape
        d_sae   = self.sae_bank.d_sae

        # Mean gradient tensor: scatter clean then patch (duplicates get same value — safe)
        mean_g_act = torch.zeros(B, T, d_sae, device=c_idx.device, dtype=torch.float32)
        mean_g_act.scatter_(-1, c_idx, mean_g_c)
        mean_g_act.scatter_(-1, p_idx, mean_g_p)

        # Delta tensor: scatter_add_ correctly handles features in both sets
        delta_act = torch.zeros(B, T, d_sae, device=c_idx.device, dtype=torch.float32)
        delta_act.scatter_add_(-1, c_idx, -c_vals.float())
        delta_act.scatter_add_(-1, p_idx,  p_vals.float())

        return SparseAct(act=mean_g_act, res=mean_g_r), SparseAct(act=delta_act, res=delta_res)
