import torch
import sys
from typing import Optional, Any, Set
from store.circuits import Circuit
from circuit.instrument.patcher import CircuitPatcher
from sae.dense import sparse_topk_to_dense, target_latent_activations

class SeedActivationCapturePatcher(CircuitPatcher):
    """
    Subclass of CircuitPatcher that captures the activation of a specific
    target latent (the seed) at its peak position during the forward pass.
    """
    def __init__(
        self, 
        bank, 
        circuit, 
        avg_acts, 
        seed_layer: int, 
        seed_kind: str, 
        seed_latent_idx: int, 
        pos_argmax: Optional[torch.Tensor] = None,
        patch_pos_selective: bool = False,
        **kwargs
    ):
        # We only pass pos_argmax to super if we want position-selective patching.
        # For upstream faithfulness, we usually want global patching.
        super().__init__(
            bank, circuit, avg_acts, 
            pos_argmax=pos_argmax if patch_pos_selective else None, 
            **kwargs
        )
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.seed_latent_idx = seed_latent_idx
        self.seed_pos_argmax = pos_argmax # Peak position for seed capture
        self.captured_activation: Optional[float] = None

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        # print(f"      [CapturePatcher] Layer {layer_idx} {kind} | x.shape {list(x.shape)}")
        # sys.stdout.flush()

        # 1. Early return if layer is outside the patching scope
        if self.max_layer is not None and layer_idx > self.max_layer:
            return x
        if self.circuit_layers is not None and layer_idx not in self.circuit_layers:
            return x
        if self.patch_kinds is not None and kind not in self.patch_kinds:
            return x
        if (layer_idx, kind) not in self.circuit_masks:
            # print(f"      [CapturePatcher] Warning: (layer {layer_idx}, kind {kind}) not in circuit_masks")
            # sys.stdout.flush()
            return x

        B, T, _ = x.shape
        target_dtype = x.dtype

        # 2. Encode once — reuse top_acts/top_indices for both capture and patching
        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)

        # 3. Capture seed latent activation before patching alters the residual stream
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            seed_acts_dense = target_latent_activations(top_acts, top_indices, self.seed_latent_idx)  # [B, T]
            
            # Debug: check if seed is active in top-k
            n_active = (seed_acts_dense > 0).sum().item()
            print(f"      [Capture] Layer {layer_idx} {kind} | Seed {self.seed_latent_idx} | Active in {n_active}/{B*T} positions")
            sys.stdout.flush()

            if self.seed_pos_argmax is not None:
                batch_indices = torch.arange(B, device=x.device)
                probe_pos = self.seed_pos_argmax.to(x.device)
                
                # Check if probe_pos is within bounds
                if probe_pos.max() >= T:
                    print(f"      [Capture] Error: probe_pos {probe_pos.tolist()} out of bounds for T={T}")
                    val = 0.0
                else:
                    val = seed_acts_dense[batch_indices, probe_pos].mean().item()
                
                self.captured_activation = val
                print(f"      [Capture] Mean at seed_pos_argmax: {val:.4f}")
                sys.stdout.flush()
            else:
                val = seed_acts_dense.mean().item()
                self.captured_activation = val
                print(f"      [Capture] Global mean: {val:.4f}")
                sys.stdout.flush()

        # 4. Apply patching (same logic as CircuitPatcher.transform, no second encode)
        all_latents = sparse_topk_to_dense(top_acts, top_indices, self.bank.d_sae, dtype=target_dtype)
        full_recon = self.bank.decode(all_latents, kind, layer_idx)
        error = x - full_recon

        circuit_mask = self.circuit_masks[(layer_idx, kind)]
        is_in_circuit = circuit_mask[top_indices.long()]
        if not self.inverse:
            live_acts = torch.where(is_in_circuit, top_acts, torch.zeros_like(top_acts))
        else:
            live_acts = torch.where(~is_in_circuit, top_acts, torch.zeros_like(top_acts))

        circuit_latents = sparse_topk_to_dense(live_acts, top_indices, self.bank.d_sae, dtype=target_dtype)
        circuit_recon = self.bank.decode(circuit_latents, kind, layer_idx)

        bg = self.background_tensors[(layer_idx, kind)].to(x.device, dtype=target_dtype)
        patched = circuit_recon + bg + error

        if self.pos_argmax is not None:
            # Position-selective patching (from base class pos_argmax)
            probe_pos = self.pos_argmax.to(x.device)
            if probe_pos.max() < T:
                is_probe = torch.zeros(B, T, 1, dtype=torch.bool, device=x.device)
                is_probe[torch.arange(B, device=x.device), probe_pos] = True
                return torch.where(is_probe, patched, x)
            else:
                return patched

        return patched

@torch.no_grad()
def evaluate_upstream_faithfulness(
    inference: Any,
    sae_bank: Any,
    avg_acts: torch.Tensor,
    circuit: Circuit,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    tokens: torch.Tensor,
    pos_argmax: Optional[torch.Tensor] = None,
    circuit_layers: Optional[Set[int]] = None,
) -> float:
    """
    Measures how well the circuit's upstream nodes explain the seed latent's activation.

    upstream_faithfulness = (a_circuit - a_ablated) / (a_full - a_ablated)

    where a_* is the seed latent's activation under different patching regimes.

    Args:
        circuit_layers: If provided, the patcher intervenes only at these specific layer
                        indices. Layers not in the set run completely naturally. Prefer
                        this over max_layer when the circuit only occupies a sparse subset
                        of layers (e.g. GradientUpstreamDiscovery with depth < seed_layer).
                        When None, falls back to max_layer=seed_layer (ablate all layers
                        up to and including the seed layer).
    """
    print(f"  [UpstreamFaithfulness] Starting eval for seed L{seed_layer} {seed_kind} idx {seed_latent_idx}")
    if circuit_layers is not None:
        print(f"  [UpstreamFaithfulness] Layer-restricted ablation: {sorted(circuit_layers)}")
    sys.stdout.flush()

    # When circuit_layers is provided, restrict ablation to those specific layers.
    # When not provided, fall back to max_layer=seed_layer for backward compatibility.
    _max_layer = seed_layer if circuit_layers is None else None

    # 1. Full model — patcher is a mathematical identity (full_circuit=True).
    #    CapturePatcher still reads seed activation via pos_argmax internally.
    full_patcher = SeedActivationCapturePatcher(
        sae_bank, None, avg_acts,
        seed_layer=seed_layer, seed_kind=seed_kind, seed_latent_idx=seed_latent_idx,
        pos_argmax=pos_argmax, full_circuit=True,
        max_layer=_max_layer, circuit_layers=circuit_layers,
    )
    inference.forward(tokens, patcher=full_patcher, return_activations=False)
    a_full = full_patcher.captured_activation

    if a_full is None:
        print("  [UpstreamFaithfulness] Error: a_full is None")
        sys.stdout.flush()
        return 0.0

    # 2. Circuit only — ablate non-circuit latents at the relevant layers.
    circuit_patcher = SeedActivationCapturePatcher(
        sae_bank, circuit, avg_acts,
        seed_layer=seed_layer, seed_kind=seed_kind, seed_latent_idx=seed_latent_idx,
        pos_argmax=pos_argmax,
        max_layer=_max_layer, circuit_layers=circuit_layers,
    )
    inference.forward(tokens, patcher=circuit_patcher, return_activations=False)
    a_circuit = circuit_patcher.captured_activation or 0.0

    # 3. All ablated — ablate everything at the relevant layers.
    ablated_patcher = SeedActivationCapturePatcher(
        sae_bank, None, avg_acts,
        seed_layer=seed_layer, seed_kind=seed_kind, seed_latent_idx=seed_latent_idx,
        pos_argmax=pos_argmax,
        max_layer=_max_layer, circuit_layers=circuit_layers,
    )
    inference.forward(tokens, patcher=ablated_patcher, return_activations=False)
    a_ablated = ablated_patcher.captured_activation or 0.0

    # Calculate score
    denom = a_full - a_ablated
    
    # Debug
    print(f"  [UpstreamFaithfulness] a_full: {a_full:.4f} | a_circuit: {a_circuit:.4f} | a_ablated: {a_ablated:.4f} | denom: {denom:.4f}")
    sys.stdout.flush()
    
    if abs(denom) < 1e-9:
        score = 1.0 if abs(a_circuit - a_full) < 1e-9 else 0.0
        print(f"  [UpstreamFaithfulness] Small denom: {score:.4f}")
        sys.stdout.flush()
        return float(score)
        
    score = (a_circuit - a_ablated) / denom
    print(f"  [UpstreamFaithfulness] Score: {score:.4f}")
    sys.stdout.flush()
    return float(score)
