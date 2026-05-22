import os
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, cast

import torch

from .runtime import get_runtime
from .distributed.interfaces import build_output_paths
from store.context import mid_ctx, neg_ctx, top_ctx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store.search_cache import generate_search_cache
from store.seq_repr import SeqRepr
from config import config
from observability.timing import format_duration


def _atomic_save_path(path: str) -> str:
    return f"{path}.tmp"


def _save_artifact(path: str, save_fn: Callable[[str], None]) -> None:
    """Save an artifact, optionally through a temporary path and atomic rename."""
    if bool(config.persist.atomic_saves):
        tmp_path = _atomic_save_path(path)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        save_fn(tmp_path)
        if os.path.exists(tmp_path):
            os.replace(tmp_path, path)
        return
    save_fn(path)


def search_cache_build_mode() -> str:
    if config.persist.search_cache_enabled is False:
        return "disabled"
    if config.persist.build_search_cache_after_pipeline is False:
        return "deferred"
    return "pipeline_end"


def build_search_cache_artifact(
    top_ctx_store,
    bank,
    loader,
    output_path: str = "outputs/search_cache.parquet",
) -> None:
    search_t0 = time.perf_counter()
    generate_search_cache(
        top_ctx_store,
        bank,
        loader,
        output_path=output_path,
    )
    print(f"  [timing] search_cache build: {format_duration(time.perf_counter() - search_t0)}")


def build_search_cache_if_enabled(
    top_ctx_store,
    bank,
    loader,
    output_path: str = "outputs/search_cache.parquet",
) -> bool:
    mode = search_cache_build_mode()
    if mode == "disabled":
        print("  [search_cache] skipped (disabled in config)")
        return False
    if mode == "deferred":
        print("  [search_cache] deferred; run scripts/build_search_cache.sh after the pipeline")
        return False

    print("Building search cache...")
    try:
        build_search_cache_artifact(top_ctx_store, bank, loader, output_path=output_path)
        return True
    except Exception as error:
        print(f"  ✗ search_cache failed: {error}")
        # We don't raise here to allow the pipeline to continue even if cache build fails
        return False


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
    if bool(config.hardware.keep_model_loaded_for_neg_ctx):
        print("Keeping model and SAE bank loaded for neg_ctx (hardware.keep_model_loaded_for_neg_ctx=true).")
        return
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


def save_results(output_root: str = "outputs") -> None:
    runtime = get_runtime()
    paths = build_output_paths(output_root)
    os.makedirs(paths.run_root, exist_ok=True)
    assert runtime.seq_repr is not None
    assert runtime.bank is not None
    assert runtime.loader is not None

    # Use config to limit peak memory during save
    save_workers = int(config.persist.save_workers or 1)
    print(f"Saving outputs (workers={save_workers})...")
    
    tasks = {
        "latent_stats": lambda: _save_artifact(str(paths.latent_stats), latent_stats.save),
        "top_ctx": lambda: _save_artifact(str(paths.top_ctx), top_ctx.save),
        "mid_ctx": lambda: _save_artifact(str(paths.mid_ctx), mid_ctx.save),
        "seq_repr": lambda: _save_artifact(str(paths.seq_repr), cast(SeqRepr, runtime.seq_repr).save),
        "logit_ctx": lambda: _save_artifact(str(paths.logit_ctx), logit_ctx.save),
    }

    def timed_save(fn):
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=save_workers) as executor:
        futures = {executor.submit(timed_save, fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                elapsed = future.result()
                print(f"  [timing] {name} save: {format_duration(elapsed)}")
                print(f"  ✓ {name} saved")
            except Exception as error:
                print(f"  ✗ {name} failed: {error}")
                raise
    
    # Phase 2: Build search cache after heavy tensor saves are finished
    gc.collect()
    
    if config.latents.seq_latent_index.enabled:
        print(f"  ✓ seq_latent_index shards written to {paths.seq_latent_index_dir}/")

    mode = search_cache_build_mode()
    if mode == "disabled":
        print("  [search_cache] skipped (disabled in config)")
    elif mode == "deferred":
        print("  [search_cache] deferred; run scripts/build_search_cache.sh after the pipeline")
    else:
        print("  [search_cache] scheduled for pipeline end")
