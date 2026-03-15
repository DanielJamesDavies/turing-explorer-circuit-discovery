from typing import Dict, Tuple, cast

import torch
from tqdm import tqdm

from .runtime import get_runtime
from .encoding import encode_layer_components
from sae.async_encode import PendingEncode
from store.context import mid_ctx, top_ctx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx


def _update_stores(
    mid_ctx_warmup: int,
    current_batch_last_latents: Dict[int, torch.Tensor],
    comp_idx: int,
    sequence_ids: torch.Tensor,
    latents: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    with torch.no_grad():
        latent_stats.update_component(comp_idx, latents)
        top_ctx.update_component(comp_idx, sequence_ids, latents)

        if latent_stats.component_steps[comp_idx] >= mid_ctx_warmup:
            mid_ctx.update_component(
                comp_idx,
                sequence_ids,
                latents,
                latent_stats.mean_seq[comp_idx],
                latent_stats.std_seq(comp_idx),
            )
        current_batch_last_latents[comp_idx] = latents[1][:, -1, :].detach()


def run_first_pass() -> None:
    runtime = get_runtime()
    print("--- First Pass: Latent Stats & Context ---")
    assert runtime.bank is not None
    assert runtime.seq_repr is not None
    assert runtime.loader is not None
    assert runtime.model is not None

    last_layer_idx = runtime.bank.n_layer - 1
    resid_kind_idx = runtime.bank.kinds.index("resid")
    pending_encodes: list[tuple[PendingEncode, torch.Tensor]] = []
    current_batch_last_latents: Dict[int, torch.Tensor] = {}

    def activations_callback(
        layer_idx: int,
        sequence_ids: torch.Tensor,
        activations: Tuple[torch.Tensor, ...],
    ) -> None:
        assert runtime.bank is not None
        assert runtime.seq_repr is not None

        if layer_idx == last_layer_idx:
            with torch.no_grad():
                runtime.seq_repr.update(sequence_ids, activations[resid_kind_idx])

        encoded = encode_layer_components(
            runtime.bank,
            layer_idx,
            activations,
            primary_device=runtime.device,
            multi_gpu=runtime.multi_gpu,
        )
        if runtime.multi_gpu:
            pending_encodes.append((cast(PendingEncode, encoded), sequence_ids))
            return

        for comp_idx, latents in cast(Dict[int, Tuple[torch.Tensor, torch.Tensor]], encoded).items():
            _update_stores(runtime.mid_ctx_warmup, current_batch_last_latents, comp_idx, sequence_ids, latents)

    for batch_ids, batch_tokens in tqdm(runtime.loader.get_batches(), total=len(runtime.loader), desc="Latent Stats & Ctx"):
        current_batch_last_latents.clear()
        pending_encodes.clear()
        tokens = cast(torch.Tensor, batch_tokens)

        _tokens, last_logits, _ = runtime.model.forward(
            tokens,
            num_gen=1,
            tokenize_final=False,
            activations_callback=lambda layer_idx, acts: activations_callback(layer_idx, batch_ids, acts),
            return_activations=False,
        )

        if runtime.multi_gpu:
            with torch.no_grad():
                for pending, seq_ids in pending_encodes:
                    results = pending.synchronize()
                    for comp_idx, latents in results.items():
                        _update_stores(runtime.mid_ctx_warmup, current_batch_last_latents, comp_idx, seq_ids, latents)

        if last_logits is not None:
            probs = torch.softmax(last_logits[:, -1, :], dim=-1)
            logit_ctx.update(current_batch_last_latents, probs)

    runtime.seq_repr.print_stats()
    print("")
