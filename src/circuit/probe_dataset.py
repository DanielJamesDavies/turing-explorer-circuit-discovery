import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from store.context import Context
from data.loader import DataLoader
from sae.bank import SAEBank
from pipeline.component_index import split_component_idx

@dataclass
class ProbeDataset:
    """
    A high-contrast dataset for evaluating a circuit around a seed latent.
    """
    pos_tokens: torch.Tensor  # [N_pos, seq_len] (typically seq_len=64)
    target_tokens: torch.Tensor # [N_pos, seq_len] (shifted: target_tokens[i, t] is token at t+1)
    neg_tokens: torch.Tensor  # [N_neg, seq_len]
    pos_argmax: torch.Tensor  # [N_pos] token position of peak activation for the seed latent
    metadata: Dict[str, Any]

class ProbeDatasetBuilder:
    def __init__(self, inference: Any, bank: SAEBank, loader: DataLoader):
        """
        Args:
            inference: The Inference instance to run the model.
            bank: The SAEBank to encode activations.
            loader: The DataLoader to fetch sequences.
        """
        self.inference = inference
        self.bank = bank
        self.loader = loader

    def build_for_latent(
        self, 
        comp_idx: int, 
        latent_idx: int, 
        top_ctx: Context,
        mid_ctx: Context,
        neg_ctx: Context,
        n_pos: int = 64, 
        n_neg: int = 64
    ) -> ProbeDataset:
        """
        Builds a ProbeDataset for a specific latent by gathering tokens and finding argmax.
        """
        # 1. Gather Positive IDs (Top + Mid)
        top_ids = top_ctx.ctx_seq_idx[comp_idx, latent_idx].tolist()
        mid_ids = mid_ctx.ctx_seq_idx[comp_idx, latent_idx].tolist()
        
        # Deduplicate and filter sentinel 0s
        pos_ids = []
        seen = set()
        for sid in (top_ids + mid_ids):
            sid_int = int(sid)
            if sid_int > 0 and sid_int not in seen:
                pos_ids.append(sid_int)
                seen.add(sid_int)
        pos_ids = pos_ids[:n_pos]

        # 2. Gather Negative IDs (Hard Negatives)
        neg_ids = neg_ctx.ctx_seq_idx[comp_idx, latent_idx].tolist()
        neg_ids = [int(sid) for sid in neg_ids if int(sid) > 0][:n_neg]

        # 3. Load Tokens (65 tokens to have targets for all 64 positions)
        all_pos_tokens = self._load_all_ids(pos_ids, max_length=65)
        neg_tokens = self._load_all_ids(neg_ids, max_length=64)

        # Prepare pos_tokens (input) and target_tokens (expected next tokens)
        pos_tokens = all_pos_tokens[:, :64]
        target_tokens = all_pos_tokens[:, 1:65]
        
        # Pad target_tokens if necessary (if any sequence was < 65 tokens)
        if target_tokens.shape[1] < pos_tokens.shape[1]:
            pad_len = pos_tokens.shape[1] - target_tokens.shape[1]
            padding = torch.zeros((target_tokens.shape[0], pad_len), 
                                  dtype=torch.long, device=target_tokens.device)
            target_tokens = torch.cat([target_tokens, padding], dim=1)

        # 4. Find pos_argmax for positive sequences
        pos_argmax = self._calculate_argmax(comp_idx, latent_idx, pos_tokens)

        return ProbeDataset(
            pos_tokens=pos_tokens,
            target_tokens=target_tokens,
            neg_tokens=neg_tokens,
            pos_argmax=pos_argmax,
            metadata={
                "comp_idx": comp_idx,
                "latent_idx": latent_idx,
                "n_pos": len(pos_ids),
                "n_neg": len(neg_ids)
            }
        )

    def _load_all_ids(self, ids: List[int], max_length: int = 64) -> torch.Tensor:
        """Helper to load a list of sequence IDs into a single tensor."""
        if not ids:
            return torch.zeros((0, max_length), dtype=torch.long, device=self.bank.device)
        
        # DataLoader handles padding and device placement
        all_batches = list(self.loader.get_batches_by_ids(ids, max_length=max_length))
        if not all_batches:
            return torch.zeros((0, max_length), dtype=torch.long, device=self.bank.device)
            
        return torch.cat([batch_tokens for _, batch_tokens in all_batches], dim=0)

    def _calculate_argmax(self, comp_idx: int, latent_idx: int, tokens: torch.Tensor) -> torch.Tensor:
        """
        Runs the model and SAE encoder to find the token position of peak activation.
        """
        if tokens.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=tokens.device)

        target_layer, kind_idx = split_component_idx(comp_idx, len(self.bank.kinds))
        kind = self.bank.kinds[kind_idx]
        
        # We'll store the argmaxes here
        all_argmax = []

        def capture_hook(layer_idx: int, activations: Tuple[torch.Tensor, ...]):
            if layer_idx == target_layer:
                # 1. Encode activations [B, T, d_model] -> [B, T, K]
                act = activations[kind_idx]
                top_acts, top_indices = self.bank.encode(act, kind, layer_idx)
                
                # 2. Reconstruct sparse/dense slice for the target latent
                # We want to find which T had the highest activation for latent_idx
                # top_indices is [B, T, K], top_acts is [B, T, K]
                
                # Find if latent_idx is in the top-K for each (batch, token)
                # is_target: [B, T, K] bool
                is_target = (top_indices == latent_idx)
                
                # Extract activation values where it is the target, else 0
                # target_acts: [B, T]
                target_acts = torch.where(is_target, top_acts, torch.zeros_like(top_acts)).sum(dim=-1)
                
                # 3. Find argmax over T
                # batch_argmax: [B]
                batch_argmax = target_acts.argmax(dim=-1)
                all_argmax.append(batch_argmax.cpu())

        # Run forward pass with the hook
        # We don't need to generate, just one step
        self.inference.forward(
            tokens, 
            num_gen=1, 
            tokenize_final=False, 
            activations_callback=capture_hook,
            return_activations=False
        )

        if not all_argmax:
            # Fallback if hook wasn't called (shouldn't happen)
            return torch.zeros(tokens.shape[0], dtype=torch.long, device="cpu")

        return torch.cat(all_argmax, dim=0)
