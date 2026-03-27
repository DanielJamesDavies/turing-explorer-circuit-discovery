import gc
import torch
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple, List, cast

from .base import DiscoveryMethod
from ..sparse_act import SparseAct
from config import config
from store.circuits import Circuit, CircuitNode
from circuit.sae_graph import SAEGraphInstrument
from circuit.feature_id import FeatureID
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

    def to_sparse_act(self, d_sae: int) -> SparseAct:
        """Expands sparse [B, T, k] representation to dense [B, T, d_sae] SparseAct."""
        B, T, _ = self.vals.shape
        act = torch.zeros(B, T, d_sae, device=self.device, dtype=torch.float32)
        act.scatter_(-1, self.idx, self.vals)
        return SparseAct(act=act, res=self.res.clone())


class SingleSubmodPatcher:
    """
    Patches only a single (layer, kind) submodule with interpolated SAE features
    for per-submodule Integrated Gradients, matching Marks et al. 2024.

    All other submodules see the original clean activation (passthrough), so the
    full model gradient path is naturally intact without the (x - x.detach()) trick.
    """

    def __init__(
        self,
        bank: Any,
        target_lk: Tuple[int, str],
        f_act: torch.Tensor,
        f_res: torch.Tensor,
    ):
        self.bank = bank
        self.target_lk = target_lk
        self.f_act = f_act  # [B, T, d_sae] — detached leaf with requires_grad
        self.f_res = f_res  # [B, T, d_model] — detached leaf with requires_grad

    def __call__(self, model: Any):
        def transform(layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
            if (layer_idx, kind) != self.target_lk:
                return x
            dtype = x.dtype
            return self.bank.decode(self.f_act.to(dtype), kind, layer_idx) + self.f_res.to(dtype)
        return multi_patch(model, transform)


def _vram_audit(label: str) -> None:
    """Prints allocated VRAM and the 10 largest live CUDA tensors."""
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    resv  = torch.cuda.memory_reserved()  / 1024**3
    print(f"\n[VRAM] {label}: allocated={alloc:.2f}GB  reserved={resv:.2f}GB")
    tensors = [
        (t.element_size() * t.nelement(), t.shape, t.dtype)
        for t in gc.get_objects()
        if isinstance(t, torch.Tensor) and t.is_cuda
    ]
    tensors.sort(key=lambda x: x[0], reverse=True)
    for size, shape, dtype in tensors[:10]:
        print(f"  {size/1024**3:.3f}GB  {dtype}  {list(shape)}")


class SFCAttributionPatching(DiscoveryMethod):
    """
    Sparse Feature Circuits (Marks et al. 2024) style circuit discovery.
    """
    method_name = "sfc_attribution_patching"

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
        self.min_faithfulness = (
            min_faithfulness
            if min_faithfulness is not None
            else cast(float, config.discovery.min_faithfulness or 0.3)
        )
        self.ig_steps          = ig_steps          if ig_steps          is not None else cast(int,   getattr(cfg, "ig_steps", 10))

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

        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]
        seed_fid = FeatureID(seed_layer, seed_kind, seed_latent_idx)

        logger.header(seed_layer, seed_kind, seed_latent_idx,
                      probe_data.pos_tokens.shape[0], probe_data.neg_tokens.shape[0])
        logger.note(f"probe batch: {n_probe} sequences  ig_steps={self.ig_steps}")

        # 2. Collect clean and patch states
        clean_states = self._get_all_states(probe_tokens)

        n_neg = probe_data.neg_tokens.shape[0]
        if self.patch_mode == "zero" or n_neg == 0:
            patch_states = {lk: s.zeros_like() for lk, s in clean_states.items()}
        else:
            neg_tokens = probe_data.neg_tokens[:min(self.max_neg, n_neg)]
            if neg_tokens.shape[0] >= n_probe:
                patch_states = self._get_all_states(neg_tokens[:n_probe])
            else:
                patch_states = {lk: s.zeros_like() for lk, s in clean_states.items()}

        # 3. Node attribution via IG
        effects, deltas_all, grads_all = self._pe_ig(
            clean_states, patch_states, probe_tokens, probe_argmax, probe_targets
        )
        logger.note(f"IG complete  {self.ig_steps} steps  patch_mode={self.patch_mode}")
        _vram_audit("after _pe_ig")

        # 4. Build circuit nodes
        circuit = Circuit(name=f"SFCAttrPatch_S{seed_comp_idx}_{seed_latent_idx}")
        node_id_map: Dict[FeatureID, str] = {}
        resid_node_id_map: Dict[Tuple[int, str], str] = {}
        active_latents: Dict[Tuple[int, str], List[int]] = {}

        for (layer, kind), effect in effects.items():
            agg = effect.sum(dim=1).mean(dim=0)
            scores = agg.act
            if scores is None:
                continue

            for l_idx in scores.abs().nonzero(as_tuple=True)[0].tolist():
                score = float(scores[l_idx].item())
                fid = FeatureID(layer, kind, l_idx)
                if fid != seed_fid and abs(score) < self.node_threshold:
                    continue
                role = "seed" if fid == seed_fid else "attributed"
                node = CircuitNode(metadata={
                    "feature_id": fid,
                    "role": role, "effect_score": score,
                })
                circuit.add_node(node)
                node_id_map[fid] = node.uuid
                active_latents.setdefault((layer, kind), []).append(l_idx)

            res_score = float(agg.resc.item()) if agg.resc is not None else 0.0
            if abs(res_score) >= self.node_threshold:
                node = CircuitNode(metadata={
                    "layer_idx": layer, "kind": kind,
                    "role": "residual", "effect_score": res_score,
                })
                circuit.add_node(node)
                resid_node_id_map[(layer, kind)] = node.uuid

        del effects
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _vram_audit("after del effects")

        logger.stage("node attribution", len(circuit.nodes), 0,
                     note=f"threshold={self.node_threshold}")

        if len(circuit.nodes) <= 1:
            logger.reject(f"only seed passed node threshold ({self.node_threshold})")
            return None

        # 5. Edge attribution via JVP — 6 edge types per layer (Marks et al. 2024)
        #    For each layer, working backward:
        #      MR: mlp  → resid                (no stop-grad)
        #      AR: attn → resid                (stop mlp)
        #      AM: attn → mlp                  (no stop-grad)
        #      RM: prev_resid → mlp            (stop attn)
        #      RA: prev_resid → attn           (no stop-grad)
        #      RR: prev_resid → resid          (stop mlp + attn)

        self.inference.disable_compile()
        self.inference.enable_grad_checkpointing()
        instrument = SAEGraphInstrument(self.sae_bank, stop_error_grad=True)
        self.inference.forward(probe_tokens, patcher=instrument,
                               grad_enabled=True, return_activations=False)
        self.inference.disable_grad_checkpointing()
        self.inference.enable_compile()
        _vram_audit("after SAE graph forward")

        n_layers = self.sae_bank.n_layer

        for layer in reversed(range(n_layers)):
            resid_lk = (layer, "resid")
            mlp_lk   = (layer, "mlp")
            attn_lk  = (layer, "attn")

            self._add_jvp_edges(
                instrument, grads_all, deltas_all, circuit,
                node_id_map, resid_node_id_map, active_latents,
                upstream=mlp_lk, downstream=resid_lk, stop_grads=[],
            )
            self._add_jvp_edges(
                instrument, grads_all, deltas_all, circuit,
                node_id_map, resid_node_id_map, active_latents,
                upstream=attn_lk, downstream=resid_lk, stop_grads=[mlp_lk],
            )
            self._add_jvp_edges(
                instrument, grads_all, deltas_all, circuit,
                node_id_map, resid_node_id_map, active_latents,
                upstream=attn_lk, downstream=mlp_lk, stop_grads=[],
            )

            if layer > 0:
                prev_resid_lk = (layer - 1, "resid")
                self._add_jvp_edges(
                    instrument, grads_all, deltas_all, circuit,
                    node_id_map, resid_node_id_map, active_latents,
                    upstream=prev_resid_lk, downstream=mlp_lk, stop_grads=[attn_lk],
                )
                self._add_jvp_edges(
                    instrument, grads_all, deltas_all, circuit,
                    node_id_map, resid_node_id_map, active_latents,
                    upstream=prev_resid_lk, downstream=attn_lk, stop_grads=[],
                )
                self._add_jvp_edges(
                    instrument, grads_all, deltas_all, circuit,
                    node_id_map, resid_node_id_map, active_latents,
                    upstream=prev_resid_lk, downstream=resid_lk, stop_grads=[mlp_lk, attn_lk],
                )

        del instrument
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.stage("edge attribution", len(circuit.nodes), len(circuit.edges),
                     note=f"threshold={self.edge_threshold}")

        if len(circuit.nodes) <= 1:
            logger.reject("circuit collapsed to ≤1 node after edge pass")
            return None

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

    def _add_jvp_edges(
        self,
        instrument: SAEGraphInstrument,
        grads: Dict[Tuple[int, str], SparseAct],
        deltas: Dict[Tuple[int, str], SparseAct],
        circuit: Circuit,
        node_id_map: Dict[FeatureID, str],
        resid_node_id_map: Dict[Tuple[int, str], str],
        active_latents: Dict[Tuple[int, str], List[int]],
        upstream: Tuple[int, str],
        downstream: Tuple[int, str],
        stop_grads: List[Tuple[int, str]],
    ) -> None:
        """
        Computes JVP edge attributions for a single (upstream → downstream) pair.

        For each active downstream feature, backpropagates through the graph (stopping
        gradients at `stop_grads` submodules), reads accumulated gradients at the upstream
        leaf anchor, and scores edges as grad @ delta.
        """
        if downstream not in grads or upstream not in deltas:
            return
        if downstream not in instrument.graph.activations:
            return
        if upstream not in instrument.graph.activations:
            return

        d_layer, d_kind = downstream
        u_layer, u_kind = upstream

        _, d_state_conn, _ = instrument.graph.get_latents(d_layer, d_kind)
        backprop_vec = (grads[downstream] @ d_state_conn).to_tensor()

        targets: List[Tuple[int, str]] = []
        for l_idx in active_latents.get(downstream, []):
            fid = FeatureID(d_layer, d_kind, l_idx)
            if fid in node_id_map:
                targets.append((l_idx, node_id_map[fid]))
        if downstream in resid_node_id_map:
            targets.append((self.sae_bank.d_sae, resid_node_id_map[downstream]))

        if not targets:
            return

        for target_idx, target_uuid in targets:
            with multi_stop_grad(self.inference.model, stop_grads):
                self.inference.model.zero_grad()
                instrument.graph.zero_grad()
                backprop_vec[:, :, target_idx].sum().backward(retain_graph=True)

                u_state_grad, _, _ = instrument.graph.get_latents(u_layer, u_kind)
                attr = u_state_grad.grad @ deltas[upstream]
                score_act = attr.sum(dim=1).mean(dim=0)

                if score_act.act is not None:
                    for u_latent in active_latents.get(upstream, []):
                        score = float(score_act.act[u_latent].item())
                        if abs(score) >= self.edge_threshold:
                            fid = FeatureID(u_layer, u_kind, u_latent)
                            if fid in node_id_map:
                                circuit.add_edge(
                                    node_id_map[fid],
                                    target_uuid, weight=score,
                                )

                if upstream in resid_node_id_map and score_act.resc is not None:
                    score = float(score_act.resc.item())
                    if abs(score) >= self.edge_threshold:
                        circuit.add_edge(
                            resid_node_id_map[upstream],
                            target_uuid, weight=score,
                        )

    def _get_all_states(self, tokens: torch.Tensor) -> Dict[Tuple[int, str], TopKState]:
        states: Dict[Tuple[int, str], TopKState] = {}
        d_sae = self.sae_bank.d_sae

        def callback(layer_idx: int, activations: Tuple[torch.Tensor, ...]) -> None:
            for kind_idx, kind in enumerate(self.sae_bank.kinds):
                act = activations[kind_idx]
                top_acts, top_idx = self.sae_bank.encode(act, kind, layer_idx)
                B, T, _ = act.shape
                dtype = act.dtype
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
        patch_states: Dict[Tuple[int, str], TopKState],
        tokens: torch.Tensor,
        argmax: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[
        Dict[Tuple[int, str], SparseAct],
        Dict[Tuple[int, str], SparseAct],
        Dict[Tuple[int, str], SparseAct],
    ]:
        """
        Per-submodule Integrated Gradients, matching Marks et al. 2024.

        For each (layer, kind) independently: interpolate that one submodule's
        SAE state between clean and patch across `ig_steps` points, holding all
        other submodules at their clean activations.  Backward from the logit
        metric at each step, then average gradients.

        Returns (effects, deltas, grads) where each is a dict keyed by (layer, kind):
            effects[lk] = grad @ delta  (SparseAct with .act and .resc)
            deltas[lk]  = patch - clean  (SparseAct with .act and .res)
            grads[lk]   = mean IG gradient (SparseAct with .act and .res)
        """
        B = tokens.shape[0]
        batch_idx = torch.arange(B, device=tokens.device)
        d_sae = self.sae_bank.d_sae

        effects: Dict[Tuple[int, str], SparseAct] = {}
        deltas:  Dict[Tuple[int, str], SparseAct] = {}
        grads:   Dict[Tuple[int, str], SparseAct] = {}

        self.inference.disable_compile()
        self.inference.enable_grad_checkpointing()
        for lk in clean_states:
            clean_sa = clean_states[lk].to_sparse_act(d_sae)
            patch_sa = patch_states[lk].to_sparse_act(d_sae)

            assert clean_sa.act is not None and patch_sa.act is not None
            assert clean_sa.res is not None and patch_sa.res is not None

            # Running sums avoid accumulating ig_steps tensors simultaneously.
            act_grad_sum = torch.zeros_like(clean_sa.act)
            res_grad_sum = torch.zeros_like(clean_sa.res)

            for step in range(self.ig_steps):
                alpha = step / self.ig_steps
                f_act = ((1 - alpha) * clean_sa.act + alpha * patch_sa.act).detach().requires_grad_(True)
                f_res = ((1 - alpha) * clean_sa.res + alpha * patch_sa.res).detach().requires_grad_(True)

                patcher = SingleSubmodPatcher(self.sae_bank, lk, f_act, f_res)
                _, logits, _ = self.inference.forward(
                    tokens, patcher=patcher, grad_enabled=True,
                    return_activations=False, all_logits=True,
                )
                target_ids = targets[batch_idx, argmax.to(targets.device)].to(logits.device)
                logits[batch_idx, argmax, target_ids].sum().backward()

                act_grad_sum += f_act.grad.detach() if f_act.grad is not None else torch.zeros_like(f_act)
                res_grad_sum += f_res.grad.detach() if f_res.grad is not None else torch.zeros_like(f_res)
                del f_act, f_res

            # Cast to bfloat16 before storing: reduces grads/deltas from float32 to
            # bfloat16, cutting the combined dict footprint by ~2× (2.88GB → 1.44GB each).
            mean_grad = SparseAct(
                act=(act_grad_sum / self.ig_steps).bfloat16(),
                res=(res_grad_sum / self.ig_steps).bfloat16(),
            )
            del act_grad_sum, res_grad_sum

            delta = SparseAct(
                act=(patch_sa.act - clean_sa.act).detach().bfloat16(),
                res=(patch_sa.res - clean_sa.res).detach().bfloat16(),
            )
            effect = mean_grad @ delta

            effects[lk] = SparseAct(
                act=effect.act.bfloat16() if effect.act is not None else None,
                resc=effect.resc.bfloat16() if effect.resc is not None else None,
            )
            deltas[lk] = delta
            grads[lk] = mean_grad

            del clean_sa, patch_sa
        self.inference.disable_grad_checkpointing()
        self.inference.enable_compile()

        return effects, deltas, grads

