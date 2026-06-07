import torch
from typing import Dict, Any, List, Tuple, Optional, Union
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense
from model.hooks import multi_patch
from circuit.types.sparse_act import SparseAct
from circuit.types.feature_id import FeatureID

class FeatureGraph:
    """
    Stores grad-anchors and graph-connected activations as SparseAct objects.
    
    Each entry stores a pair:
      - state_grad:      detached leaf SparseAct with requires_grad=True
      - state_connected: original SparseAct, connected to the computation graph
    """
    def __init__(self, device: torch.device):
        self.device = device
        # (layer, kind) -> List of (state_grad, state_connected, top_indices)
        self.activations: Dict[Tuple[int, str], List[Tuple[SparseAct, SparseAct, torch.Tensor]]] = {}

    def add(
        self,
        layer_idx: int,
        kind: str,
        state_grad: Union[SparseAct, torch.Tensor],
        state_connected: Union[SparseAct, torch.Tensor],
        top_indices: torch.Tensor,
    ):
        if isinstance(state_grad, torch.Tensor):
            state_grad = SparseAct(act=state_grad)
        if isinstance(state_connected, torch.Tensor):
            state_connected = SparseAct(act=state_connected)
            
        key = (layer_idx, kind)
        if key not in self.activations:
            self.activations[key] = []
        self.activations[key].append((state_grad, state_connected, top_indices))

    def get_latents(self, layer_idx: int, kind: str, step: int = 0) -> Tuple[SparseAct, SparseAct, torch.Tensor]:
        """Returns (state_grad, state_connected, top_indices) for the given layer/kind."""
        return self.activations[(layer_idx, kind)][step]

    def get_latents_by_id(self, feature_id: FeatureID, step: int = 0) -> Tuple[SparseAct, SparseAct, torch.Tensor]:
        """Returns (state_grad, state_connected, top_indices) for the given FeatureID."""
        return self.get_latents(feature_id.layer, feature_id.kind, step)

    def all_anchors(self) -> List[torch.Tensor]:
        """Returns all leaf anchor tensors (act and res) that require grad."""
        anchors = []
        for steps in self.activations.values():
            for state_grad, _, _ in steps:
                if state_grad.act.requires_grad:
                    anchors.append(state_grad.act)
                if state_grad.res is not None and state_grad.res.requires_grad:
                    anchors.append(state_grad.res)
                elif state_grad.resc is not None and state_grad.resc.requires_grad:
                    anchors.append(state_grad.resc)
        return anchors

    def zero_grad(self):
        """Zeros accumulated gradients on all leaf anchors."""
        for steps in self.activations.values():
            for state_grad, _, _ in steps:
                if state_grad.act is not None and state_grad.act.grad is not None:
                    state_grad.act.grad.zero_()
                if state_grad.res is not None and state_grad.res.grad is not None:
                    state_grad.res.grad.zero_()
                if state_grad.resc is not None and state_grad.resc.grad is not None:
                    state_grad.resc.grad.zero_()


class SAEGraphInstrument:
    """
    Instruments the forward pass to capture SAE features and residual error term
    with gradients enabled, matching the Sparse Feature Circuits (Marks et al. 2024)
    design.

    For each (layer, kind):
      - Encodes x into SAE feature activations f.
      - Computes reconstruction error (residual) = x - decode(f).
      - Replaces x with decode(f_grad) + res_grad + (x - x.detach()), where
        f_grad and res_grad are detached leaf anchors that capture gradients.
      - The (x - x.detach()) term is numerically zero but provides an identity
        gradient path to x, allowing cross-layer gradient flow without the
        lossy error-complement projection of the old (residual - residual.detach())
        approach.
      - If stop_error_grad=True, gradients through res_grad are zeroed in backward,
        ensuring causal attribution only flows through the SAE features.
    """
    def __init__(self, bank: SAEBank, stop_error_grad: bool = False):
        self.bank = bank
        self.stop_error_grad = stop_error_grad
        self.graph = FeatureGraph(bank.device)
        self.logits: Optional[torch.Tensor] = None

    def __call__(self, model: Any):
        """Hook entry point for Inference.forward(patcher=instrument)."""
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        # 1. Encode — top_acts is connected to x through the encoder
        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        B, T, _ = x.shape
        d_sae = self.bank.d_sae
        target_dtype = x.dtype

        # Construct full sparse feature tensor (needed for joint feature+residual attribution)
        f = sparse_topk_to_dense(top_acts, top_indices, d_sae, dtype=target_dtype)

        # 2. Decode and compute residual
        #    Use the graph-connected f so the encoder path stays differentiable
        x_hat_connected = self.bank.decode(f, kind, layer_idx)
        residual = x - x_hat_connected

        # 3. Create leaf anchors (detached, requires_grad=True) for attribution
        f_grad = f.detach().requires_grad_(True)
        res_anchor = residual.detach().requires_grad_(True)
        
        state_grad = SparseAct(act=f_grad, res=res_anchor)
        state_connected = SparseAct(act=f, res=residual)
        
        # 4. Store both in the feature graph (along with original top_indices)
        self.graph.add(layer_idx, kind, state_grad, state_connected, top_indices)

        # 5. Reconstruct from the f_grad anchor (so d(downstream)/d(f_grad) is well-defined)
        reconstruction = self.bank.decode(f_grad, kind, layer_idx)
        
        # 6. Zero residual gradient if requested (forces attribution through features only)
        if self.stop_error_grad:
            res_anchor.register_hook(lambda grad: torch.zeros_like(grad))

        # 7. Identity passthrough: (x - x.detach()) is numerically zero but carries
        #    gradient d/dx = I back to x, maintaining cross-layer connectivity.
        #    This replaces the old (residual - residual.detach()) which projected
        #    gradients onto the error complement (I - W_dec @ W_enc), losing signal.
        return reconstruction + res_anchor + (x - x.detach())


class SAEGraphInstrumentWithEmbedding(SAEGraphInstrument):
    """
    Thin subclass of SAEGraphInstrument that captures the input embedding
    (residual stream at layer 0, before the first SAE hook) as a detached leaf
    tensor, enabling gradient-based token attribution.

    On the first call to transform(layer=first_layer, kind=first_kind, x), a leaf
    tensor `emb_anchor` is created from `x.detach().requires_grad_(True)`.  The
    term `(emb_anchor - emb_anchor.detach())` is numerically zero but provides a
    clean gradient tap: callers can include `emb_anchor` in torch.autograd.grad
    anchor lists to obtain ∂(scalar)/∂(emb_anchor) — i.e. the Jacobian of any
    downstream scalar w.r.t. the embedding at every (batch, position, dimension).

    Token attribution score for position p:
        score[p] = Σ_{b,d}  emb_anchor[b,p,d] * grad_emb[b,p,d]

    Args:
        bank:             SAEBank passed through to the parent class.
        stop_error_grad:  Passed through to the parent class.
        first_layer:      Layer index of the first SAE hook (typically 0).
        first_kind:       Kind string of the first SAE hook (e.g. "attn").
    """

    def __init__(
        self,
        bank: "SAEBank",
        stop_error_grad: bool = False,
        first_layer: int = 0,
        first_kind: str = "attn",
    ):
        super().__init__(bank, stop_error_grad)
        self._first_lk: Tuple[int, str] = (first_layer, first_kind)
        self.emb_anchor: Optional[torch.Tensor] = None  # [B, T, d_model], leaf

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        out = super().transform(layer_idx, kind, x)
        if (layer_idx, kind) == self._first_lk and self.emb_anchor is None:
            # Capture embedding as a detached leaf so callers can attribute to it.
            # Adding (emb_anchor - emb_anchor.detach()) is numerically zero but
            # routes ∂out/∂emb_anchor = 1, enabling per-position attribution.
            self.emb_anchor = x.detach().requires_grad_(True)
            out = out + (self.emb_anchor - self.emb_anchor.detach())
        return out
