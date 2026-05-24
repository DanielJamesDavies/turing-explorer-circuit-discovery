"""Shared contracts for distributed pass-1 merge helpers."""

from __future__ import annotations

from typing import Dict

from ..pass1_partials import Pass1PartialMetadata


LatentStatsPartial = tuple[Pass1PartialMetadata, Dict[str, object]]
TopCtxPartial = tuple[Pass1PartialMetadata, Dict[str, object]]
MidCtxCandidatesPartial = tuple[Pass1PartialMetadata, Dict[str, object]]
SeqReprPartial = tuple[Pass1PartialMetadata, Dict[str, object]]
LogitCtxPartial = tuple[Pass1PartialMetadata, Dict[str, object]]

MID_CTX_CANDIDATE_POOL_DEFAULTS = {
    "enabled": True,
    "band_margin_sigma": 1.0,
    "on_truncation": "replay_fallback",
}

PASS1_PARTIAL_FILENAMES = {
    "latent_stats": "latent_stats.partial.pt",
    "top_ctx": "top_ctx.partial.pt",
    "mid_ctx_candidates": "mid_ctx_candidates.partial.pt",
    "seq_repr": "seq_repr.partial.pt",
    "logit_ctx": "logit_ctx.partial.pt",
}

__all__ = [
    "LatentStatsPartial",
    "LogitCtxPartial",
    "MID_CTX_CANDIDATE_POOL_DEFAULTS",
    "MidCtxCandidatesPartial",
    "PASS1_PARTIAL_FILENAMES",
    "SeqReprPartial",
    "TopCtxPartial",
]
