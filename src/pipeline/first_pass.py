from typing import Dict, Optional, Sequence, Tuple, cast

import torch
from tqdm import tqdm

from config import config
from .runtime import get_runtime
from .encoding import encode_layer_components
from .seq_latent_index import SeqLatentIndexAccumulator
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


def run_first_pass(
    *,
    assigned_shard_ids: Optional[Sequence[int]] = None,
    seq_latent_index_output_dir: str = "outputs/seq_latent_index",
) -> None:
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
    deferred_sae_encode = config.first_pass.sae_encode_mode == "deferred"
    if deferred_sae_encode:
        print("  SAE encode mode: deferred")

    seq_idx_cfg = config.latents.seq_latent_index
    accumulator: Optional[SeqLatentIndexAccumulator] = None
    if seq_idx_cfg.enabled:
        accumulator = SeqLatentIndexAccumulator(
            shard_id_ranges=runtime.loader.shard_id_ranges,
            top_k_per_component=seq_idx_cfg.top_k_per_component,
            output_dir=seq_latent_index_output_dir,
        )
        print(f"  seq_latent_index enabled: top_k_per_component={seq_idx_cfg.top_k_per_component}")

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

        if deferred_sae_encode:
            return

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
            if accumulator is not None:
                accumulator.update(comp_idx, sequence_ids, latents)

    def encode_deferred_activations(
        sequence_ids: torch.Tensor,
        deferred_layers: list[tuple[int, Tuple[torch.Tensor, ...]]],
    ) -> None:
        assert runtime.bank is not None

        for layer_idx, activations in deferred_layers:
            encoded = encode_layer_components(
                runtime.bank,
                layer_idx,
                activations,
                primary_device=runtime.device,
                multi_gpu=runtime.multi_gpu,
            )
            if runtime.multi_gpu:
                pending_encodes.append((cast(PendingEncode, encoded), sequence_ids))
            else:
                for comp_idx, latents in cast(Dict[int, Tuple[torch.Tensor, torch.Tensor]], encoded).items():
                    _update_stores(runtime.mid_ctx_warmup, current_batch_last_latents, comp_idx, sequence_ids, latents)
                    if accumulator is not None:
                        accumulator.update(comp_idx, sequence_ids, latents)

        if runtime.multi_gpu:
            with torch.no_grad():
                for pending, seq_ids in pending_encodes:
                    results = pending.synchronize()
                    for comp_idx, latents in results.items():
                        _update_stores(runtime.mid_ctx_warmup, current_batch_last_latents, comp_idx, seq_ids, latents)
                        if accumulator is not None:
                            accumulator.update(comp_idx, seq_ids, latents)

    if assigned_shard_ids is None:
        batch_iter = runtime.loader.get_batches()
        total_batches = len(runtime.loader)
    else:
        shard_ids = list(assigned_shard_ids)
        batch_iter = runtime.loader.get_batches_for_shards(shard_ids)
        total_batches = runtime.loader.num_batches_for_shards(shard_ids)

    for batch_ids, batch_tokens in tqdm(batch_iter, total=total_batches, desc="Latent Stats & Ctx"):
        current_batch_last_latents.clear()
        pending_encodes.clear()
        deferred_layers: list[tuple[int, Tuple[torch.Tensor, ...]]] = []
        tokens = cast(torch.Tensor, batch_tokens)

        def batch_activations_callback(
            layer_idx: int,
            acts: Tuple[torch.Tensor, ...],
        ) -> None:
            if deferred_sae_encode:
                # Hold raw model activations only for the current batch. Detach so
                # the post-forward SAE work cannot retain autograd graph state.
                deferred_layers.append(
                    (layer_idx, tuple(act.detach() for act in acts))
                )
            activations_callback(layer_idx, batch_ids, acts)

        _tokens, last_logits, _ = runtime.model.forward(
            tokens,
            num_gen=1,
            tokenize_final=False,
            activations_callback=batch_activations_callback,
            return_activations=False,
        )

        if deferred_sae_encode:
            encode_deferred_activations(batch_ids, deferred_layers)
            deferred_layers.clear()
        elif runtime.multi_gpu:
            with torch.no_grad():
                for pending, seq_ids in pending_encodes:
                    results = pending.synchronize()
                    for comp_idx, latents in results.items():
                        _update_stores(runtime.mid_ctx_warmup, current_batch_last_latents, comp_idx, seq_ids, latents)
                        if accumulator is not None:
                            accumulator.update(comp_idx, seq_ids, latents)

        if last_logits is not None:
            probs = torch.softmax(last_logits[:, -1, :], dim=-1)
            logit_ctx.update(current_batch_last_latents, probs)

        if accumulator is not None:
            accumulator.on_batch_complete(int(batch_ids.max().item()))

    if accumulator is not None:
        accumulator.flush_all()
        print("  seq_latent_index: all shards written")

    runtime.seq_repr.print_stats()
    print("")
