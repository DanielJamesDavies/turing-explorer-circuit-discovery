"""Full pipeline profiler.

Profiles one real batch end-to-end — LLM forward pass, SAE encode, latent
stats update, and top-context update — using the same initialisation and
callback structure as pipeline.py.

Usage (from the project root or src/):
    python -m debug.profile_pipeline
    # or: python src/debug/profile_pipeline.py

Output:
  - CUDA-time-sorted summary table printed to stdout
  - Chrome trace written to profile_trace/full_pipeline.json
    (open in chrome://tracing or https://ui.perfetto.dev)
"""

import os
import sys
import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import component_idx
from sae.bank import SAEBank
from store.latent_stats import latent_stats
from store.context import top_ctx

WARMUP_BATCHES  = 2   # batches run before profiling starts
PROFILE_BATCHES = 1   # batches profiled  (1 is enough for a clear picture)
OUTPUT_DIR      = "profile_trace"
TRACE_FILE      = "full_pipeline.json"
ROW_LIMIT       = 30


def _make_callback(bank):
    """Returns an SAE callback that annotates each section for the profiler."""
    n_kinds = len(bank.kinds)

    def callback(layer_idx, sequence_ids, activations):
        with torch.no_grad():
            if bank.parallel_kinds:
                with record_function("sae_encode"):
                    latents_list = bank.encode_layer_kinds_parallel(activations, layer_idx)
                for kind_idx, latents in enumerate(latents_list):
                    comp_idx = component_idx(layer_idx, kind_idx, n_kinds)
                    with record_function("latent_stats_update"):
                        latent_stats.update_component(comp_idx, latents)
                    with record_function("top_ctx_update"):
                        top_ctx.update_component(comp_idx, sequence_ids, latents)
            else:
                for kind_idx, kind in enumerate(bank.kinds):
                    comp_idx = component_idx(layer_idx, kind_idx, n_kinds)
                    with record_function("sae_encode"):
                        latents = bank.encode(activations[kind_idx], kind, layer_idx)
                    with record_function("latent_stats_update"):
                        latent_stats.update_component(comp_idx, latents)
                    with record_function("top_ctx_update"):
                        top_ctx.update_component(comp_idx, sequence_ids, latents)

    return callback


def _run_batch(model, batch_tokens, batch_ids, callback):
    model.forward(
        batch_tokens,
        num_gen=1,
        tokenize_final=False,
        activations_callback=lambda l, a: callback(l, batch_ids, a),
        return_activations=False,
    )


def main():
    print("")
    fast    = is_fast_memory()
    compile = should_compile()
    devices = detect_devices()
    device  = devices[0]

    print(f"Flash SDP enabled:         {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"Mem-efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"Math SDP enabled:          {torch.backends.cuda.math_sdp_enabled()}")
    print("")

    print("Initializing DataLoader...")
    loader = DataLoader(device=device, pin_memory=fast)

    print("Initializing Model...")
    model = Inference(device=device, compile=compile)

    print(f"Initializing SAE Bank ({len(devices)} device{'s' if len(devices) > 1 else ''})...")
    bank = SAEBank(devices=devices, load_decoders=fast, compile=compile)

    callback = _make_callback(bank)
    batch_iter = loader.get_batches()

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"\nWarming up ({WARMUP_BATCHES} batch{'es' if WARMUP_BATCHES > 1 else ''})...")
    for _ in range(WARMUP_BATCHES):
        batch_ids, batch_tokens = next(batch_iter)
        with torch.no_grad():
            _run_batch(model, batch_tokens, batch_ids, callback)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # ── Profile ───────────────────────────────────────────────────────────────
    print(f"Profiling {PROFILE_BATCHES} batch{'es' if PROFILE_BATCHES > 1 else ''}...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_flops=False,
    ) as prof:
        for _ in range(PROFILE_BATCHES):
            batch_ids, batch_tokens = next(batch_iter)
            with torch.no_grad(), record_function("pipeline_batch"):
                _run_batch(model, batch_tokens, batch_ids, callback)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=ROW_LIMIT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trace_path = os.path.join(OUTPUT_DIR, TRACE_FILE)
    prof.export_chrome_trace(trace_path)
    print(f"Trace saved → {trace_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
