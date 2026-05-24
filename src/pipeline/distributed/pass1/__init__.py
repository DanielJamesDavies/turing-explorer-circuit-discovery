"""Distributed pass-1 merge package exports."""

from .cli import build_arg_parser, main
from .contracts import (
    LatentStatsPartial,
    LogitCtxPartial,
    MID_CTX_CANDIDATE_POOL_DEFAULTS,
    MidCtxCandidatesPartial,
    PASS1_PARTIAL_FILENAMES,
    SeqReprPartial,
    TopCtxPartial,
)
from .context_merge import (
    load_and_merge_mid_ctx_candidate_partials,
    load_and_merge_top_ctx_partials,
    merge_mid_ctx_candidate_partials,
    merge_top_ctx_partials,
)
from .latent_stats_merge import (
    load_and_merge_latent_stats_partials,
    merge_latent_stats_partials,
)
from .logit_ctx_merge import (
    load_and_merge_logit_ctx_partials,
    merge_logit_ctx_partials,
)
from .reports import build_pass1_sanity_report
from .seq_repr_merge import (
    load_and_merge_seq_repr_partials,
    merge_seq_repr_partials,
)
from .seq_latent_index_merge import merge_seq_latent_index_shards
from .writer import merge_pass1_worker_outputs
from .worker import (
    configure_mid_ctx_candidate_pool,
    initialize_pass1_worker_resources,
    run_pass1_worker,
    save_pass1_partials,
    validate_pass1_worker_inputs,
)

__all__ = [
    "LatentStatsPartial",
    "LogitCtxPartial",
    "MID_CTX_CANDIDATE_POOL_DEFAULTS",
    "MidCtxCandidatesPartial",
    "PASS1_PARTIAL_FILENAMES",
    "SeqReprPartial",
    "TopCtxPartial",
    "build_arg_parser",
    "build_pass1_sanity_report",
    "load_and_merge_latent_stats_partials",
    "load_and_merge_logit_ctx_partials",
    "load_and_merge_mid_ctx_candidate_partials",
    "load_and_merge_seq_repr_partials",
    "load_and_merge_top_ctx_partials",
    "main",
    "merge_latent_stats_partials",
    "merge_logit_ctx_partials",
    "merge_mid_ctx_candidate_partials",
    "merge_pass1_worker_outputs",
    "merge_seq_repr_partials",
    "merge_seq_latent_index_shards",
    "merge_top_ctx_partials",
    "configure_mid_ctx_candidate_pool",
    "initialize_pass1_worker_resources",
    "run_pass1_worker",
    "save_pass1_partials",
    "validate_pass1_worker_inputs",
]
