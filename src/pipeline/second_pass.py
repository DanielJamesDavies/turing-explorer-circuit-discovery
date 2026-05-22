import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, cast

import torch
from tqdm import tqdm

from .distributed.interfaces import build_output_paths
from .runtime import get_runtime
from .encoding import encode_layer_components
from config import config
from sae.async_encode import PendingEncode
from store.context import top_ctx
from store.latent_stats import latent_stats
from store.top_coactivation import top_coactivation


@dataclass(frozen=True)
class SecondPassDumpResult:
    sequence_count: int
    batch_count: int
    seq_len: int
    elapsed_s: float
    model_forward_s: float = 0.0
    sae_encode_s: float = 0.0
    update_dump_s: float = 0.0


def run_second_pass(output_root: str = "outputs") -> None:
    runtime = get_runtime()
    output_paths = build_output_paths(output_root)
    print("--- Second Pass: Top Co-Activation ---")
    dump_result = run_second_pass_dump()

    seq_offsets, seq_targets_global = top_ctx.get_sequence_to_latents_csr(device=runtime.cpu_device)
    if dump_result.seq_len <= 0:
        raise ValueError("second pass cannot reduce an empty replay sequence list")

    if not runtime.fast:
        print("Freeing model and SAE bank for reduction...")
        runtime.model = None
        runtime.bank = None
        torch.cuda.empty_cache()

    print("Running top co-activation reduction...")
    reduce_t0 = time.perf_counter()
    top_coactivation.reduce(
        seq_offsets,
        seq_targets_global,
        seq_len=dump_result.seq_len,
        active_count=latent_stats.active_count,
    )
    print(f"  [timing] top_coactivation reduce+postprocess: {time.perf_counter() - reduce_t0:.2f} s")
    save_t0 = time.perf_counter()
    output_paths.run_root.mkdir(parents=True, exist_ok=True)
    top_coactivation.save(str(output_paths.top_coactivation))
    print(f"  top_coactivation saved ({time.perf_counter() - save_t0:.2f} s)")
    print("")


def run_second_pass_dump(
    sequence_ids: list[int] | None = None,
) -> SecondPassDumpResult:
    """Run only the top-coactivation candidate dump phase."""

    runtime = get_runtime()
    assert runtime.loader is not None
    assert runtime.model is not None
    assert runtime.bank is not None

    top_coactivation.set_device(runtime.device)
    
    if top_coactivation.mode == "freq_weighted":
        top_coactivation.set_frequency_factors(latent_stats.active_count.to(runtime.device))
    
    print(f"Co-activation mode: {top_coactivation.mode}")

    top_ctx_sequence_ids = (
        list(sequence_ids)
        if sequence_ids is not None
        else top_ctx.get_all_sequence_ids()
    )
    top_coactivation.prepare_dump(top_ctx_sequence_ids)

    current_batch_latents: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    pending_coact: list[PendingEncode] = []
    model_forward_s = 0.0
    sae_encode_s = 0.0
    update_dump_s = 0.0

    def coact_callback(layer_idx: int, activations: Tuple[torch.Tensor, ...]) -> None:
        nonlocal sae_encode_s
        assert runtime.bank is not None
        encode_t0 = time.perf_counter()
        encoded = encode_layer_components(
            runtime.bank,
            layer_idx,
            activations,
            primary_device=runtime.device,
            multi_gpu=runtime.multi_gpu,
        )
        sae_encode_s += time.perf_counter() - encode_t0
        if runtime.multi_gpu:
            pending_coact.append(cast(PendingEncode, encoded))
            return

        results = cast(Dict[int, Tuple[torch.Tensor, torch.Tensor]], encoded)
        for comp_idx, latents in results.items():
            current_batch_latents[comp_idx] = (latents[0].detach(), latents[1].detach().to(torch.int32))

    total_batches = math.ceil(len(top_ctx_sequence_ids) / runtime.loader.batch_size)
    dump_t0 = time.perf_counter()
    dump_row_start = 0
    batch_count = 0
    seq_len = 0
    for batch_ids, batch_tokens in tqdm(
        runtime.loader.get_batches_by_ids(top_ctx_sequence_ids),
        total=total_batches,
        desc="Top Co-activation Dump",
    ):
        batch_count += 1
        current_batch_latents.clear()
        pending_coact.clear()
        tokens = cast(torch.Tensor, batch_tokens)
        seq_len = int(tokens.shape[1])

        model_t0 = time.perf_counter()
        runtime.model.forward(
            tokens,
            num_gen=1,
            tokenize_final=False,
            activations_callback=lambda layer_idx, acts: coact_callback(layer_idx, acts),
            return_activations=False,
        )
        model_forward_s += time.perf_counter() - model_t0

        if runtime.multi_gpu:
            with torch.no_grad():
                for pending in pending_coact:
                    encode_t0 = time.perf_counter()
                    results = pending.synchronize()
                    sae_encode_s += time.perf_counter() - encode_t0
                    for comp_idx, latents in results.items():
                        current_batch_latents[comp_idx] = (
                            latents[0].detach(),
                            latents[1].detach().to(torch.int32),
                        )

        update_t0 = time.perf_counter()
        top_coactivation.update_batch(batch_ids, current_batch_latents, dump_row_start=dump_row_start)
        update_dump_s += time.perf_counter() - update_t0
        dump_row_start += int(batch_ids.shape[0])
    elapsed_s = time.perf_counter() - dump_t0
    print(f"  [timing] top_coactivation dump: {elapsed_s:.2f} s")
    if bool(getattr(config.latents.top_coactivation, "dump_profile", True)):
        print(top_coactivation.dump_timing_summary())
    return SecondPassDumpResult(
        sequence_count=len(top_ctx_sequence_ids),
        batch_count=batch_count,
        seq_len=seq_len,
        elapsed_s=elapsed_s,
        model_forward_s=model_forward_s,
        sae_encode_s=sae_encode_s,
        update_dump_s=update_dump_s,
    )
