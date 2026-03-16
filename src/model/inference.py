import __main__
from typing import cast
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from contextlib import nullcontext

from config import config
from .turingllm import TuringLLM, TuringLLMConfig
from .hooks import capture_activations


def _fuse_mlp_projections(state_dict: dict) -> dict:
    """
    Remaps a checkpoint saved with split up_proj_swish / up_proj weights onto
    the fused gate_up_proj layout used by the current MLP definition.

    Old layout (per block):
        mlp.up_proj_swish.weight  [hidden_size, n_embd]  — gate branch
        mlp.up_proj_swish.bias    [hidden_size]
        mlp.up_proj.weight        [hidden_size, n_embd]  — value branch
        mlp.up_proj.bias          [hidden_size]

    New layout:
        mlp.gate_up_proj.weight   [hidden_size * 2, n_embd]  — gate first, value second
        mlp.gate_up_proj.bias     [hidden_size * 2]
    """
    new_sd: dict = {}
    consumed: set = set()

    for key, val in state_dict.items():
        if key in consumed:
            continue
        if ".mlp.up_proj_swish." in key:
            partner = key.replace(".mlp.up_proj_swish.", ".mlp.up_proj.")
            fused   = key.replace(".mlp.up_proj_swish.", ".mlp.gate_up_proj.")
            if partner in state_dict:
                new_sd[fused] = torch.cat([val, state_dict[partner]], dim=0)
                consumed.add(partner)
            else:
                new_sd[key] = val
        else:
            new_sd[key] = val

    return new_sd


class Inference:
    def __init__(self, device: torch.device, compile: bool = False):
        self.model = TuringLLM()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        setattr(__main__, "TuringLLMConfig", TuringLLMConfig)

        raw = torch.load(cast(str, config.weights.model_path), map_location="cpu", weights_only=False)["model"]
        state_dict = _fuse_mlp_projections(raw)
        self.model.load_state_dict(state_dict)
        target_dtype = None
        if self.device.type == "cuda":
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if target_dtype is not None:
            self.model.to(self.device, dtype=target_dtype).eval()
        else:
            self.model.to(self.device).eval()

        self._compiled = False
        self._eager_forwards = [block.forward for block in self.model.transformer.h]
        if compile and self.device.type != "cpu":
            self.enable_compile()

    def enable_compile(self):
        if self._compiled:
            return
        # Allow enough recompile slots for every (kind, layer) combination that
        # hooks and patchers will produce (3 kinds × n_layer, plus headroom).
        import torch._dynamo
        torch._dynamo.config.recompile_limit = max(64, len(self.model.transformer.h) * 3 * 4)
        for block in self.model.transformer.h:
            block.forward = torch.compile(block.forward, mode="default")
        self._compiled = True

    def disable_compile(self):
        if not self._compiled:
            return
        for block, eager_fn in zip(self.model.transformer.h, self._eager_forwards):
            block.forward = eager_fn
        self._compiled = False

    def forward(self, tokens: torch.Tensor, num_gen: int = 1, tokenize_final: bool = True, activations_callback=None, patcher=None, return_activations: bool = True, all_logits: bool = False, grad_enabled: bool = False):
        sample_rng = torch.Generator(device=self.device).manual_seed(12)
        activations = []
        logits = None

        sdpa_ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if self.device.type == "cuda" else nullcontext()
        grad_ctx = torch.enable_grad() if grad_enabled else torch.no_grad()

        for i in range(num_gen):
            with capture_activations(self.model, callback=activations_callback, capture=return_activations) as acts, (patcher(self.model) if patcher else nullcontext()):
                with grad_ctx, sdpa_ctx:
                    logits, _ = self.model(tokens, return_all_logits=(all_logits and i == 0))
            
            if return_activations:
                # If generating multiple tokens, only keep the last token's activations 
                # to maintain consistent tensor shapes for stacking.
                step_activations = acts.tensor
                if step_activations is not None and num_gen > 1:
                    step_activations = step_activations[:, :, :, -1:, :]
                activations.append(step_activations)
            
            if i < num_gen - 1 or tokenize_final:
                probs = F.softmax(logits[:, -1, :], dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=12, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                tokens = torch.cat((tokens, xcol), dim=1)

        # [G, B, L, K, T, N] -> [B, L, K, G, T, N]
        # G=Generations, B=Batch, L=Layer, K=Kind, T=Token, N=Neuron
        return tokens, logits, torch.stack(activations).permute(1, 2, 3, 0, 4, 5) if return_activations else None

