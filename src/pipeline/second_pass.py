import math
from typing import Dict, Tuple, cast

import torch
from tqdm import tqdm

from .runtime import get_runtime
from .encoding import encode_layer_components
from sae.async_encode import PendingEncode
from store.context import top_ctx
from store.latent_stats import latent_stats
from store.top_coactivation import top_coactivation


def run_second_pass() -> None:
    runtime = get_runtime()
    print("--- Second Pass: Top Co-Activation ---")
    assert runtime.loader is not None
    assert runtime.model is not None
    assert runtime.bank is not None

    top_coactivation.set_device(runtime.device)
    top_coactivation.set_frequency_factors(latent_stats.active_count.to(runtime.device))

    top_ctx_sequence_ids = top_ctx.get_all_sequence_ids()
    seq_offsets, seq_targets_global = top_ctx.get_sequence_to_latents_csr(device=runtime.cpu_device)
    top_coactivation.prepare_dump(top_ctx_sequence_ids)

    current_batch_latents: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    pending_coact: list[PendingEncode] = []

    def coact_callback(layer_idx: int, activations: Tuple[torch.Tensor, ...]) -> None:
        assert runtime.bank is not None
        encoded = encode_layer_components(
            runtime.bank,
            layer_idx,
            activations,
            primary_device=runtime.device,
            multi_gpu=runtime.multi_gpu,
        )
        if runtime.multi_gpu:
            pending_coact.append(cast(PendingEncode, encoded))
            return

        results = cast(Dict[int, Tuple[torch.Tensor, torch.Tensor]], encoded)
        for comp_idx, latents in results.items():
            current_batch_latents[comp_idx] = (latents[0].detach(), latents[1].detach().to(torch.int32))

    total_batches = math.ceil(len(top_ctx_sequence_ids) / runtime.loader.batch_size)
    for batch_ids, batch_tokens in tqdm(
        runtime.loader.get_batches_by_ids(top_ctx_sequence_ids),
        total=total_batches,
        desc="Top Co-activation Dump",
    ):
        current_batch_latents.clear()
        pending_coact.clear()
        tokens = cast(torch.Tensor, batch_tokens)

        runtime.model.forward(
            tokens,
            num_gen=1,
            tokenize_final=False,
            activations_callback=lambda layer_idx, acts: coact_callback(layer_idx, acts),
            return_activations=False,
        )

        if runtime.multi_gpu:
            with torch.no_grad():
                for pending in pending_coact:
                    results = pending.synchronize()
                    for comp_idx, latents in results.items():
                        current_batch_latents[comp_idx] = (
                            latents[0].detach(),
                            latents[1].detach().to(torch.int32),
                        )

        top_coactivation.update_batch(batch_ids, current_batch_latents)

    if not runtime.fast:
        print("Freeing model and SAE bank for reduction...")
        runtime.model = None
        runtime.bank = None
        torch.cuda.empty_cache()

    print("Running top co-activation reduction...")
    top_coactivation.reduce(seq_offsets, seq_targets_global)
    top_coactivation.save("outputs/top_coactivation.pt")
    print("  ✓ top_coactivation saved")
    print("")
