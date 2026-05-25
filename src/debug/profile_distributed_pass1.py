"""Profile the distributed pass-1 worker hot path.

This mirrors the pass-1 worker runtime and first-pass callback, but it profiles a
small number of batches and intentionally skips final partial artifact saves.

Example:
    python -m debug.profile_distributed_pass1 \
      --manifest /outputs/<run_id>/distributed/manifest.json \
      --worker-id 0 \
      --warmup-batches 1 \
      --profile-batches 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, cast

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from config import config
from pipeline.component_index import component_idx
from pipeline.distributed.interfaces import get_worker_shard_ids
from pipeline.distributed.manifest import load_manifest
from pipeline.distributed.pass1.worker import initialize_pass1_worker_resources
from pipeline.runtime import clear_runtime, get_runtime
from pipeline.seq_latent_index import SeqLatentIndexAccumulator
from sae.async_encode import PendingEncode
from store.context import mid_ctx, top_ctx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx


Latents = Tuple[torch.Tensor, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to distributed manifest.json")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--profile-batches", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument("--output-dir", default="profile_trace")
    parser.add_argument(
        "--seq-latent-index-dir",
        default="/tmp/turing-profile-seq-latent-index",
        help="Temporary directory for seq_latent_index writes during profiling.",
    )
    return parser.parse_args()


def _update_stores(
    mid_ctx_warmup: int,
    current_batch_last_latents: Dict[int, torch.Tensor],
    comp_idx: int,
    sequence_ids: torch.Tensor,
    latents: Latents,
    accumulator: Optional[SeqLatentIndexAccumulator],
) -> None:
    with record_function("latent_stats_update"):
        latent_stats.update_component(comp_idx, latents)

    with record_function("top_ctx_update"):
        top_ctx.update_component(comp_idx, sequence_ids, latents)

    if latent_stats.component_steps[comp_idx] >= mid_ctx_warmup:
        with record_function("mid_ctx_update"):
            mid_ctx.update_component(
                comp_idx,
                sequence_ids,
                latents,
                latent_stats.mean_seq[comp_idx],
                latent_stats.std_seq(comp_idx),
            )

    current_batch_last_latents[comp_idx] = latents[1][:, -1, :].detach()

    if accumulator is not None:
        with record_function("seq_latent_index_update"):
            accumulator.update(comp_idx, sequence_ids, latents)


def _run_profiled_batches(
    assigned_shard_ids: Sequence[int],
    *,
    warmup_batches: int,
    profile_batches: int,
    seq_latent_index_dir: str,
    output_dir: str,
    row_limit: int,
) -> None:
    runtime = get_runtime()
    assert runtime.bank is not None
    assert runtime.loader is not None
    assert runtime.model is not None
    assert runtime.seq_repr is not None

    last_layer_idx = runtime.bank.n_layer - 1
    resid_kind_idx = runtime.bank.kinds.index("resid")
    n_kinds = len(runtime.bank.kinds)
    pending_encodes: list[tuple[PendingEncode, torch.Tensor]] = []
    current_batch_last_latents: Dict[int, torch.Tensor] = {}

    seq_idx_cfg = config.latents.seq_latent_index
    accumulator: Optional[SeqLatentIndexAccumulator] = None
    if seq_idx_cfg.enabled:
        accumulator = SeqLatentIndexAccumulator(
            shard_id_ranges=runtime.loader.shard_id_ranges,
            top_k_per_component=seq_idx_cfg.top_k_per_component,
            output_dir=seq_latent_index_dir,
        )

    def activations_callback(
        layer_idx: int,
        sequence_ids: torch.Tensor,
        activations: Tuple[torch.Tensor, ...],
    ) -> None:
        assert runtime.bank is not None
        assert runtime.seq_repr is not None

        if layer_idx == last_layer_idx:
            with record_function("seq_repr_update"):
                runtime.seq_repr.update(sequence_ids, activations[resid_kind_idx])

        if runtime.multi_gpu:
            raise RuntimeError("profile_distributed_pass1 expects one worker-local GPU")

        if runtime.bank.parallel_kinds:
            with record_function("sae_encode_parallel_all_kinds"):
                latents_list = runtime.bank.encode_layer_kinds_parallel(activations, layer_idx)
            for kind_idx, latents in enumerate(latents_list):
                comp_idx = component_idx(layer_idx, kind_idx, n_kinds)
                _update_stores(
                    runtime.mid_ctx_warmup,
                    current_batch_last_latents,
                    comp_idx,
                    sequence_ids,
                    latents,
                    accumulator,
                )
            return

        for kind_idx, kind in enumerate(runtime.bank.kinds):
            comp_idx = component_idx(layer_idx, kind_idx, n_kinds)
            with record_function(f"sae_encode_{kind}"):
                latents = runtime.bank.encode(activations[kind_idx], kind, layer_idx)
            _update_stores(
                runtime.mid_ctx_warmup,
                current_batch_last_latents,
                comp_idx,
                sequence_ids,
                latents,
                accumulator,
            )

    def run_one_batch(batch_ids: torch.Tensor, batch_tokens: torch.Tensor) -> None:
        current_batch_last_latents.clear()
        pending_encodes.clear()
        with record_function("model_forward_with_callbacks"):
            _tokens, last_logits, _ = runtime.model.forward(
                batch_tokens,
                num_gen=1,
                tokenize_final=False,
                activations_callback=lambda layer_idx, acts: activations_callback(
                    layer_idx,
                    batch_ids,
                    acts,
                ),
                return_activations=False,
            )

        if last_logits is not None:
            with record_function("logit_softmax"):
                probs = torch.softmax(last_logits[:, -1, :], dim=-1)
            with record_function("logit_ctx_update"):
                logit_ctx.update(current_batch_last_latents, probs)

        if accumulator is not None:
            with record_function("seq_latent_index_on_batch_complete"):
                accumulator.on_batch_complete(int(batch_ids.max().item()))

    batch_iter = runtime.loader.get_batches_for_shards(list(assigned_shard_ids))
    total_batches = runtime.loader.num_batches_for_shards(list(assigned_shard_ids))
    print(f"Profiling worker shards={list(assigned_shard_ids)} total_batches={total_batches}")
    print(
        "Config: "
        f"batch_size={config.data.batch_size} "
        f"memory={config.hardware.memory} "
        f"parallel_kinds={config.hardware.parallel_kinds} "
        f"encode_backend={config.sae.encode_backend} "
        f"topk_backend={config.sae.topk_backend}"
    )

    with torch.no_grad():
        for i in range(warmup_batches):
            batch_ids, batch_tokens = next(batch_iter)
            print(f"Warmup batch {i + 1}/{warmup_batches}")
            run_one_batch(cast(torch.Tensor, batch_ids), cast(torch.Tensor, batch_tokens))

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
        torch.cuda.reset_peak_memory_stats(runtime.device)

    activities = [ProfilerActivity.CPU]
    if runtime.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        with_flops=False,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            for i in range(profile_batches):
                batch_ids, batch_tokens = next(batch_iter)
                print(f"Profile batch {i + 1}/{profile_batches}")
                with record_function("profiled_pass1_batch"):
                    run_one_batch(cast(torch.Tensor, batch_ids), cast(torch.Tensor, batch_tokens))

        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)

    print("\n" + "=" * 120)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))
    print("=" * 120)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trace_path = output_path / f"distributed_pass1_worker{runtime.device.index or 0}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"Trace saved to {trace_path}")

    if runtime.device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(runtime.device) / (1024**3)
        print(f"Peak CUDA allocated during profiled window: {peak_gib:.2f} GiB")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    shard_ids = get_worker_shard_ids(manifest, args.worker_id)

    try:
        initialize_pass1_worker_resources(manifest, args.worker_id)
        _run_profiled_batches(
            shard_ids,
            warmup_batches=args.warmup_batches,
            profile_batches=args.profile_batches,
            seq_latent_index_dir=args.seq_latent_index_dir,
            output_dir=args.output_dir,
            row_limit=args.row_limit,
        )
    finally:
        clear_runtime()


if __name__ == "__main__":
    main()
