import torch
from typing import Dict, Any, Optional, Set, Iterable
from sae.bank import SAEBank
from store.circuits import Circuit
from model.hooks import multi_patch
from pipeline.component_index import component_idx

class CircuitPatcher:
    """
    A patcher that intervenes on activations using a circuit definition.

    For non-circuit latents, replaces their activations with background (average)
    values while preserving the SAE error term (x - SAE_reconstruct(x)). This
    follows the Sparse Feature Circuits approach: the error term captures structure
    the SAE can't explain and is always preserved during intervention.

    Pre-computes background contributions per (layer, kind) at init time.

    Derivation of the patched output
    ---------------------------------
    Let  W = decoder weight,  b = decoder_bias,  e = top-k acts,  a = avg acts.

        full_recon  = W @ e + b
        error       = x - full_recon = x - W @ e - b

        circuit_recon = W @ circuit_e + b          (from bank.decode)
        bg            = W @ avg_non_circuit         (linear part only — no b)

        result = circuit_recon + bg + error
               = (W @ circuit_e + b) + (W @ avg) + (x - W @ e - b)
               = x + W @ (circuit_e + avg - e)      ✓  (exactly one b cancels)

    Storing bg WITHOUT the decoder_bias avoids a double-counting that would
    otherwise inject +b into the residual stream at every SAE layer, causing
    the circuit and baseline passes to accumulate an identical spurious bias
    that drives faithfulness scores toward 0.

    Position-selective mode (pos_argmax is not None)
    ------------------------------------------------
    When pos_argmax is supplied, the intervention is applied ONLY at each
    sequence's probe position (pos_argmax[b]).  All other token positions are
    passed through unchanged.  This prevents the patcher from disrupting the
    model's normal computation at positions that the circuit was never designed
    to explain, and makes faithfulness measure:

        "Given normal context up to pos_argmax, do the circuit features at
         that position alone recover the model's output there?"
    """
    def __init__(
        self,
        bank: SAEBank,
        circuit: Optional[Circuit],
        avg_acts: torch.Tensor,
        inverse: bool = False,
        pos_argmax: Optional[torch.Tensor] = None,
        patch_kinds: Optional[Iterable[str]] = None,
        full_circuit: bool = False,
    ):
        """
        Args:
            bank:         The SAEBank containing the models.
            circuit:      The Circuit definition (if None, all nodes are ablated).
            avg_acts:     Tensor [n_components, d_sae] where n_components = n_layers * len(bank.kinds).
            inverse:      If True, ablates only the circuit nodes and keeps everything else live.
                          If False (default), keeps only the circuit nodes live and ablates everything else.
            pos_argmax:   Optional [B] int tensor.  When provided, the intervention is applied
                          only at position pos_argmax[b] for each batch item b; all other
                          positions are returned unchanged (position-selective mode).
            patch_kinds:  Optional iterable of SAE kinds to patch (e.g. {"mlp"}).
                          Kinds not listed are passed through unchanged. If None,
                          all kinds are patched (default, backward-compatible behavior).
            full_circuit: If True, all latents are treated as circuit members.
                          When inverse=False the background is zero and the patcher
                          becomes a mathematical identity (recon + error = x).
                          When inverse=True (complement pass) the background is
                          W @ avg_acts, matching the circuit=None baseline patcher —
                          i.e. the complement of all-features is the empty circuit.
        """
        self.bank = bank
        self.circuit = circuit
        self.avg_acts = avg_acts
        self.inverse = inverse
        self.full_circuit = full_circuit
        # Keep on CPU; moved to target device lazily inside transform().
        self.pos_argmax = pos_argmax.detach().cpu() if pos_argmax is not None else None
        self.patch_kinds: Optional[Set[str]] = set(patch_kinds) if patch_kinds is not None else None
        
        self.background_tensors: Dict[tuple, torch.Tensor] = {}
        self.circuit_masks: Dict[tuple, torch.Tensor] = {}
        
        for l in range(bank.n_layer):
            for k_idx, kind in enumerate(bank.kinds):
                target_device = bank.layer_device_map[l]

                if full_circuit:
                    self.circuit_masks[(l, kind)] = torch.ones(
                        bank.d_sae, dtype=torch.bool, device=target_device
                    )
                    sae = bank.saes[kind][l]
                    if not inverse:
                        # mask is all-True → bg_latents[mask] = 0 → bg = 0.
                        # Patcher is a mathematical identity: recon + 0 + error = x.
                        self.background_tensors[(l, kind)] = torch.zeros(
                            sae.decoder_bias.shape, device=target_device, dtype=sae.decoder_bias.dtype
                        )
                    else:
                        # mask is all-True → ~mask is all-False → bg_latents unchanged.
                        # Complement of all-features = empty circuit, so bg = W @ avg_acts
                        # (same as the circuit=None baseline patcher).
                        with torch.no_grad():
                            ci = component_idx(l, k_idx, len(bank.kinds))
                            bg_latents = avg_acts[ci].to("cpu")
                            bg_tensor = bank.decode(bg_latents.view(1, 1, -1), kind, l)
                            b_dec = sae.decoder_bias.to(bg_tensor.device, dtype=bg_tensor.dtype)
                            self.background_tensors[(l, kind)] = (
                                (bg_tensor - b_dec).detach().squeeze(0).squeeze(0)
                            )
                    continue

                mask = torch.zeros(bank.d_sae, dtype=torch.bool, device="cpu")
                if circuit is not None:
                    for node in circuit.nodes.values():
                        fid = node.feature_id
                        if fid and fid.layer == l and fid.kind == kind:
                            mask[fid.index] = True
                
                self.circuit_masks[(l, kind)] = mask.to(target_device)
                
                with torch.no_grad():
                    comp_idx = component_idx(l, k_idx, len(bank.kinds))
                    bg_latents = avg_acts[comp_idx].clone().to("cpu")

                    if not self.inverse:
                        bg_latents[mask] = 0.0
                    else:
                        bg_latents[~mask] = 0.0

                    # Decode and immediately subtract decoder_bias so that the stored
                    # tensor is the *linear* part only: W @ avg_latents (no bias).
                    # This prevents decoder_bias from being double-counted in transform().
                    sae = bank.saes[kind][l]
                    bg_tensor = bank.decode(bg_latents.view(1, 1, -1), kind, l)
                    b_dec = sae.decoder_bias.to(bg_tensor.device, dtype=bg_tensor.dtype)
                    self.background_tensors[(l, kind)] = (bg_tensor - b_dec).detach().squeeze(0).squeeze(0)

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if self.patch_kinds is not None and kind not in self.patch_kinds:
            return x

        if (layer_idx, kind) not in self.circuit_masks:
            return x

        B, T, _ = x.shape
        target_dtype = x.dtype

        # 1. Encode into SAE latent space
        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)

        # 2. Full SAE reconstruction to compute error term
        all_latents = torch.zeros(B, T, self.bank.d_sae, device=x.device, dtype=target_dtype)
        all_latents.scatter_(dim=-1, index=top_indices.long(), src=top_acts.to(target_dtype))
        full_recon = self.bank.decode(all_latents, kind, layer_idx)
        error = x - full_recon

        # 3. Keep only circuit (or non-circuit if inverse) features
        mask = self.circuit_masks[(layer_idx, kind)]
        is_in_circuit = mask[top_indices.long()]
        
        if not self.inverse:
            live_acts = torch.where(is_in_circuit, top_acts, torch.zeros_like(top_acts))
        else:
            live_acts = torch.where(~is_in_circuit, top_acts, torch.zeros_like(top_acts))
        
        # 4. Decode the filtered features
        circuit_latents = torch.zeros(B, T, self.bank.d_sae, device=x.device, dtype=target_dtype)
        circuit_latents.scatter_(dim=-1, index=top_indices.long(), src=live_acts.to(target_dtype))
        circuit_recon = self.bank.decode(circuit_latents, kind, layer_idx)
        
        # 5. circuit features + background (linear part only) + preserved error term
        # bg is stored without decoder_bias (see __init__), so circuit_recon's single
        # decoder_bias is the only one in the sum — no double-counting.
        bg = self.background_tensors[(layer_idx, kind)].to(x.device, dtype=target_dtype)
        patched = circuit_recon + bg + error

        # Position-selective mode: only apply the intervention at each sequence's
        # probe position; return x unchanged at all other positions.
        if self.pos_argmax is not None:
            probe_pos = self.pos_argmax.to(x.device)
            is_probe = torch.zeros(B, T, 1, dtype=torch.bool, device=x.device)
            is_probe[torch.arange(B, device=x.device), probe_pos] = True
            return torch.where(is_probe, patched, x)

        return patched
