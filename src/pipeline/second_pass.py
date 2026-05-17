import math
import time
from typing import Dict, Tuple, cast

import torch
from tqdm import tqdm

from .runtime import get_runtime
from .encoding import encode_layer_components
from config import config
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
    
    if top_coactivation.mode == "freq_weighted":
        top_coactivation.set_frequency_factors(latent_stats.active_count.to(runtime.device))
    
    print(f"Co-activation mode: {top_coactivation.mode}")

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
    dump_t0 = time.perf_counter()
    dump_row_start = 0
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

        top_coactivation.update_batch(batch_ids, current_batch_latents, dump_row_start=dump_row_start)
        dump_row_start += int(batch_ids.shape[0])
    print(f"  [timing] top_coactivation dump: {time.perf_counter() - dump_t0:.2f} s")
    if bool(getattr(config.latents.top_coactivation, "dump_profile", True)):
        print(top_coactivation.dump_timing_summary())

    if not runtime.fast:
        print("Freeing model and SAE bank for reduction...")
        runtime.model = None
        runtime.bank = None
        torch.cuda.empty_cache()

    print("Running top co-activation reduction...")
    reduce_t0 = time.perf_counter()
    # seq_len is tokens.shape[1] from the last batch
    top_coactivation.reduce(
        seq_offsets, 
        seq_targets_global, 
        seq_len=tokens.shape[1], 
        active_count=latent_stats.active_count
    )
    print(f"  [timing] top_coactivation reduce+postprocess: {time.perf_counter() - reduce_t0:.2f} s")
    save_t0 = time.perf_counter()
    top_coactivation.save("outputs/top_coactivation.pt")
    print(f"  top_coactivation saved ({time.perf_counter() - save_t0:.2f} s)")
    print("")
