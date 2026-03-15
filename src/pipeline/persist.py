import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

import torch

from .runtime import get_runtime
from store.context import mid_ctx, neg_ctx, top_ctx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store.search_cache import generate_search_cache
from store.seq_repr import SeqRepr
from config import config


def offload_to_cpu() -> None:
    """Move stores to CPU memory to free VRAM for subsequent stages."""
    runtime = get_runtime()
    if runtime.fast:
        return

    print("Offloading stores to CPU...")
    latent_stats.set_device(runtime.cpu_device)
    top_ctx.set_device(runtime.cpu_device)
    mid_ctx.set_device(runtime.cpu_device)
    neg_ctx.set_device(runtime.cpu_device)
    logit_ctx.set_device(runtime.cpu_device)


def offload_model_and_sae() -> None:
    """Release model/SAE GPU memory before ANN-heavy negative-context build."""
    runtime = get_runtime()
    if runtime.model is None and runtime.bank is None:
        return

    print("Offloading model and SAE bank...")
    runtime.model = None
    runtime.bank = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reload_model_and_sae() -> None:
    """Recreate model/SAE resources after the ANN step completes."""
    runtime = get_runtime()
    if runtime.model is not None and runtime.bank is not None:
        return

    print("Reloading model and SAE bank...")
    from model.inference import Inference
    from sae.bank import SAEBank

    runtime.model = Inference(device=runtime.device, compile=runtime.compile)
    runtime.bank = SAEBank(devices=runtime.devices, load_decoders=runtime.fast, compile=runtime.compile)


def save_results() -> None:
    runtime = get_runtime()
    os.makedirs("outputs", exist_ok=True)
    assert runtime.seq_repr is not None
    assert runtime.bank is not None
    assert runtime.loader is not None

    # Use config to limit peak memory during save
    save_workers = int(config.persist.save_workers or 1)
    print(f"Saving outputs (workers={save_workers})...")
    
    tasks = {
        "latent_stats": lambda: latent_stats.save("outputs/latent_stats.pt"),
        "top_ctx": lambda: top_ctx.save("outputs/top_ctx.pt"),
        "mid_ctx": lambda: mid_ctx.save("outputs/mid_ctx.pt"),
        "seq_repr": lambda: cast(SeqRepr, runtime.seq_repr).save("outputs/seq_repr.pt"),
        "logit_ctx": lambda: logit_ctx.save("outputs/logit_ctx.pt"),
    }

    with ThreadPoolExecutor(max_workers=save_workers) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"  ✓ {name} saved")
            except Exception as error:
                print(f"  ✗ {name} failed: {error}")
                raise
    
    # Phase 2: Build search cache after heavy tensor saves are finished
    gc.collect()
    
    search_cache_enabled = config.persist.search_cache_enabled
    if search_cache_enabled is not False:
        print("Building search cache...")
        try:
            generate_search_cache(
                top_ctx,
                runtime.bank,
                runtime.loader,
                output_path="outputs/search_cache.parquet"
            )
        except Exception as error:
            print(f"  ✗ search_cache failed: {error}")
            # We don't raise here to allow the pipeline to continue even if cache build fails
    else:
        print("  [search_cache] skipped (disabled in config)")
