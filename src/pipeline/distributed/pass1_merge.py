"""Compatibility facade for distributed pass-1 merge helpers."""

from __future__ import annotations

from .pass1.contracts import (
    LatentStatsPartial,
    LogitCtxPartial,
    MID_CTX_CANDIDATE_POOL_DEFAULTS,
    MidCtxCandidatesPartial,
    PASS1_PARTIAL_FILENAMES,
    SeqReprPartial,
    TopCtxPartial,
)
from .pass1.cli import build_arg_parser, main
from .pass1.context_merge import (
    MID_CTX_WEIGHTED_RESERVOIR_HASH_VERSION,
    load_and_merge_mid_ctx_candidate_partials,
    load_and_merge_mid_ctx_reservoir_partials,
    load_and_merge_top_ctx_partials,
    merge_mid_ctx_candidate_partials,
    merge_mid_ctx_reservoir_partials,
    merge_mid_ctx_reservoir_row,
    merge_top_ctx_partials,
)
from .pass1.latent_stats_merge import (
    load_and_merge_latent_stats_partials,
    merge_latent_stats_partials,
)
from .pass1.logit_ctx_merge import (
    load_and_merge_logit_ctx_partials,
    merge_logit_ctx_partials,
)
from .pass1.seq_repr_merge import (
    load_and_merge_seq_repr_partials,
    merge_seq_repr_partials,
)
from .pass1.seq_latent_index_merge import merge_seq_latent_index_shards
from .pass1.reports import build_pass1_sanity_report
from .pass1.writer import merge_pass1_worker_outputs

__all__ = [
    "LatentStatsPartial",
    "LogitCtxPartial",
    "MID_CTX_CANDIDATE_POOL_DEFAULTS",
    "MID_CTX_WEIGHTED_RESERVOIR_HASH_VERSION",
    "MidCtxCandidatesPartial",
    "PASS1_PARTIAL_FILENAMES",
    "SeqReprPartial",
    "TopCtxPartial",
    "build_arg_parser",
    "build_pass1_sanity_report",
    "load_and_merge_latent_stats_partials",
    "load_and_merge_logit_ctx_partials",
    "load_and_merge_mid_ctx_candidate_partials",
    "load_and_merge_mid_ctx_reservoir_partials",
    "load_and_merge_seq_repr_partials",
    "load_and_merge_top_ctx_partials",
    "main",
    "merge_latent_stats_partials",
    "merge_logit_ctx_partials",
    "merge_mid_ctx_candidate_partials",
    "merge_mid_ctx_reservoir_partials",
    "merge_mid_ctx_reservoir_row",
    "merge_pass1_worker_outputs",
    "merge_seq_latent_index_shards",
    "merge_seq_repr_partials",
    "merge_top_ctx_partials",
]


if __name__ == "__main__":
    main()
