"""Integrated stage helpers for distributed controller workflows."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .controller_contracts import DistributedParts1To3Result
from .manifest import DistributedRunManifest


def run_parts_1_to_3(
    manifest: DistributedRunManifest,
    *,
    worker_runner: Optional[Callable[[DistributedRunManifest, int], Dict[str, str]]] = None,
    merge_runner: Optional[Callable[..., Dict[str, object]]] = None,
    neg_ctx_runner: Optional[Callable[..., object]] = None,
    seq_latent_index_enabled: bool = True,
    vocab_size: int | None = None,
    resume_neg_ctx: bool = True,
) -> DistributedParts1To3Result:
    """
    Execute the current integrated distributed path: pass-1 workers, pass-1 merge,
    then standalone negative context over merged canonical artifacts.
    """

    if worker_runner is None:
        from .worker import run_pass1_worker

        worker_runner = run_pass1_worker
    if merge_runner is None:
        from .pass1_merge import merge_pass1_worker_outputs

        merge_runner = merge_pass1_worker_outputs
    if neg_ctx_runner is None:
        from pipeline.negative_context import run_negative_context_stage

        neg_ctx_runner = run_negative_context_stage

    worker_artifacts = {
        worker_id: worker_runner(manifest, worker_id)
        for worker_id in range(manifest.worker_count)
    }
    pass1_merge = merge_runner(
        manifest,
        seq_latent_index_enabled=seq_latent_index_enabled,
        vocab_size=vocab_size,
    )
    negative_context = neg_ctx_runner(
        manifest.output_root,
        manifest_path=manifest.manifest_path,
        resume=resume_neg_ctx,
    )
    return DistributedParts1To3Result(
        worker_artifacts=worker_artifacts,
        pass1_merge=pass1_merge,
        negative_context=negative_context,
    )


__all__ = [
    "run_parts_1_to_3",
]
