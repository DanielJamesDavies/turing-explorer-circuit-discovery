"""Runtime wrapper for in-pipeline negative-context construction."""

from __future__ import annotations

from pathlib import Path

import torch

from store.context import build_global_sequence_ids_tensor
from store.neg_context import NegCtxStats


def build_negative_contexts(output_root: str | Path = "outputs") -> None:
    compat = _compat_module()
    runtime = compat.get_runtime()
    output_paths = compat.build_output_paths(output_root)
    print("--- ANN Step: Building Negative Contexts ---")
    assert runtime.seq_repr is not None

    try:
        neg_stats: NegCtxStats = compat.build_neg_ctx(
            runtime.seq_repr,
            compat.top_ctx,
            compat.mid_ctx,
            compat.neg_ctx,
        )
        output_paths.run_root.mkdir(parents=True, exist_ok=True)
        compat.neg_ctx.save(str(output_paths.neg_ctx))
        torch.save(build_global_sequence_ids_tensor(compat.neg_ctx.ctx_seq_idx), output_paths.global_negctx_ids)
        neg_stats.save(str(output_paths.run_root / "neg_ctx_stats.json"))
        neg_stats.print_summary(compat.neg_ctx.num_ctx_sequences)
        print(f"  ✓ neg_ctx built and saved to {output_paths.neg_ctx}")
    except ImportError as error:
        print(f"  ✗ neg_ctx skipped: {error}")
    print("")


def _compat_module():
    import pipeline.negative_context as compat

    return compat


__all__ = ["build_negative_contexts"]
