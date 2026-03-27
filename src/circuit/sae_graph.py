import torch
from typing import Dict, Any, List, Tuple, Optional, Union
from sae.bank import SAEBank
from model.hooks import multi_patch
from .sparse_act import SparseAct
from .feature_id import FeatureID

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
        f = torch.zeros(B, T, d_sae, device=x.device, dtype=target_dtype)
        f.scatter_(dim=-1, index=top_indices.long(), src=top_acts.to(target_dtype))

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
