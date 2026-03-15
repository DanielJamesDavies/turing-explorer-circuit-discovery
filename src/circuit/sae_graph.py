import torch
from typing import Dict, Any, List, Tuple, Optional
from sae.bank import SAEBank
from model.hooks import multi_patch

class FeatureGraph:
    """
    Stores grad-anchors and graph-connected activations from an instrumented forward pass.

    Each entry stores a 3-tuple:
      - top_acts_grad:      detached leaf tensor with requires_grad=True (for gradient accumulation)
      - top_acts_connected: original top_acts, still connected to the computation graph via the
                            encoder output; enables cross-layer feature-to-feature attribution
      - top_indices:        top-K latent indices [B, T, K]
    """
    def __init__(self, device: torch.device):
        self.device = device
        # (layer, kind) -> List of (top_acts_grad, top_acts_connected, top_indices)
        self.activations: Dict[Tuple[int, str], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}

    def add(
        self,
        layer_idx: int,
        kind: str,
        top_acts_grad: torch.Tensor,
        top_acts_connected: torch.Tensor,
        top_indices: torch.Tensor,
    ):
        key = (layer_idx, kind)
        if key not in self.activations:
            self.activations[key] = []
        self.activations[key].append((top_acts_grad, top_acts_connected, top_indices))

    def get_latents(self, layer_idx: int, kind: str, step: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (top_acts_grad, top_acts_connected, top_indices) for the given layer/kind."""
        return self.activations[(layer_idx, kind)][step]

    def all_anchors(self) -> List[torch.Tensor]:
        """Returns all leaf anchor tensors (top_acts_grad) across every layer/kind."""
        return [
            acts_grad
            for steps in self.activations.values()
            for acts_grad, _, _ in steps
        ]


class SAEGraphInstrument:
    """
    Instruments the forward pass to capture SAE features with gradients enabled.

    For each (layer, kind):
      - Encodes the activation x into SAE latent space.
      - Stores top_acts_grad  (detached leaf)  — leaf anchor for gradient attribution.
      - Stores top_acts_connected (not detached) — connected to x, enabling cross-layer
        feature-to-feature gradients via the error-term path.
      - Replaces x with SAE reconstruction (from leaf anchor) + error term, preserving
        information the SAE cannot explain while keeping the graph connected.

    Args:
        bank:            The SAEBank to encode/decode activations.
        stop_error_grad: If True, detach the error term before returning, so gradients
                         can only propagate back through the SAE reconstruction path
                         (i.e. through top_acts_grad).  This mirrors the feature-circuits
                         design where ``residual.grad`` is zeroed after each submodule,
                         ensuring edge weights reflect only SAE-mediated causal influence
                         rather than the "bypass" error path.
                         Set to False (default) to allow cross-layer gradient flow through
                         the error term, which is needed for Pass-1 logit attribution.
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

        # 2. Leaf anchor for gradient collection (detached from x)
        top_acts_grad = top_acts.detach().requires_grad_(True)

        # 3. Store both the leaf anchor and the graph-connected version
        self.graph.add(layer_idx, kind, top_acts_grad, top_acts, top_indices.detach())

        # 4. Decode using the leaf anchor (so d(downstream)/d(top_acts_grad) is well-defined)
        B, T, _ = x.shape
        target_dtype = x.dtype
        sparse_latents = torch.zeros(B, T, self.bank.d_sae, device=x.device, dtype=target_dtype)
        sparse_latents.scatter_(dim=-1, index=top_indices.long(), src=top_acts_grad.to(target_dtype))
        reconstruction = self.bank.decode(sparse_latents, kind, layer_idx)

        # 5. Error term: x minus the full SAE reconstruction (using the graph-connected
        #    top_acts so the encoder path stays differentiable for Pass-2 attribution).
        full_sparse = torch.zeros(B, T, self.bank.d_sae, device=x.device, dtype=target_dtype)
        full_sparse.scatter_(dim=-1, index=top_indices.long(), src=top_acts.to(target_dtype))
        full_recon = self.bank.decode(full_sparse, kind, layer_idx)
        error = x - full_recon

        # When stop_error_grad=True, detach the error so gradients can only flow back
        # through the reconstruction path (top_acts_grad → decoder).  This matches the
        # feature-circuits convention of zeroing residual.grad after each submodule,
        # giving edge weights that reflect purely SAE-mediated influence.
        if self.stop_error_grad:
            error = error.detach()

        return reconstruction + error
