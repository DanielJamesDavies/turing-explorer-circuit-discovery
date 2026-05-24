"""Shared contracts for distributed pass-2 reduce helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..pass2_partials import CandidateDumpMetadata, CandidatePreAggregationMetadata
from ..pass2_replay import Pass2ReplayList


@dataclass(frozen=True)
class CandidateDumpReducerEntry:
    metadata: CandidateDumpMetadata
    payload: Dict[str, Any]


@dataclass(frozen=True)
class CandidateDumpReducerInputs:
    entries: tuple[CandidateDumpReducerEntry, ...]
    mode: str
    m: int
    n_candidates_per_component: int
    n_latents_per_latent: int
    num_components: int
    d_sae: int
    total_sequence_count: int
    total_token_count: int


@dataclass(frozen=True)
class GlobalTopCtxTargetMapping:
    replay: Pass2ReplayList
    seq_offsets: torch.Tensor
    seq_targets_global: torch.Tensor
    sequence_ids: tuple[int, ...]
    sid_to_row: Dict[int, int]
    sid_to_row_tensor: torch.Tensor


@dataclass(frozen=True)
class SimpleExactCandidateDump:
    sequence_ids: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_vals: torch.Tensor
    sid_to_row: Dict[int, int]
    sid_to_row_tensor: torch.Tensor
    mode: str
    m: int
    n_candidates_per_component: int
    n_latents_per_latent: int
    num_components: int
    d_sae: int
    seq_len: int
    total_token_count: int


@dataclass(frozen=True)
class PmiReduceInputs:
    active_count: torch.Tensor
    total_replay_tokens: int
    total_worker_tokens: int


@dataclass(frozen=True)
class CandidatePreAggregationReducerEntry:
    metadata: CandidatePreAggregationMetadata
    payload: Dict[str, Any]


@dataclass(frozen=True)
class CandidatePreAggregationReducerInputs:
    entries: tuple[CandidatePreAggregationReducerEntry, ...]
    mode: str
    m: int
    num_components: int
    d_sae: int
    target_start_id: int
    target_end_id: int
    total_sequence_count: int
    total_contribution_count: int


@dataclass(frozen=True)
class TargetRange:
    reducer_id: int
    target_start_id: int
    target_end_id: int


@dataclass(frozen=True)
class MapReduceTargetShardResult:
    target_range: TargetRange
    top_indices: torch.Tensor
    top_values: torch.Tensor
    summed_target_ids: torch.Tensor
    summed_candidate_ids: torch.Tensor
    summed_values: torch.Tensor


@dataclass(frozen=True)
class MapReduceShardMemoryEstimate:
    shard_count: int
    contribution_count: int
    tensor_bytes: int
    file_bytes: int
    guardrail_bytes: Optional[int] = None
    exceeds_guardrail: bool = False


@dataclass(frozen=True)
class Pass2ReduceSchedulerConfig:
    reducer_mode: str = "mapreduce_target_ranges"
    reducer_count: int = 1
    execution_mode: str = "sequential"
    backend: str = "cpu"
    resume: bool = False
    cleanup: bool = False
    memory_guardrail_bytes: Optional[int] = None
    chunk_size: Optional[int] = None


@dataclass(frozen=True)
class MapReduceTargetShardArtifact:
    path: Path
    target_range: TargetRange
    top_indices: torch.Tensor
    top_values: torch.Tensor
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class MapReduceReduceResult:
    artifact_path: Path
    report_path: Path
    shard_paths: tuple[Path, ...]
    report: Dict[str, Any]


@dataclass(frozen=True)
class SimpleExactReduceResult:
    artifact_path: Path
    report_path: Path
    report: Dict[str, Any]


__all__ = [
    "CandidateDumpReducerEntry",
    "CandidateDumpReducerInputs",
    "CandidatePreAggregationReducerEntry",
    "CandidatePreAggregationReducerInputs",
    "GlobalTopCtxTargetMapping",
    "MapReduceReduceResult",
    "MapReduceShardMemoryEstimate",
    "MapReduceTargetShardArtifact",
    "MapReduceTargetShardResult",
    "Pass2ReduceSchedulerConfig",
    "PmiReduceInputs",
    "SimpleExactCandidateDump",
    "SimpleExactReduceResult",
    "TargetRange",
]
