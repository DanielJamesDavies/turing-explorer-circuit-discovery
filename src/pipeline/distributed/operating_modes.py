"""Documented runtime mode taxonomy for run-root pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .manifest import RunMode, generate_run_id


@dataclass(frozen=True)
class OperatingModeDefinition:
    """Static contract for one supported pipeline operating mode."""

    name: str
    description: str
    required_parts: tuple[str, ...]
    allowed_entrypoints: tuple[str, ...]
    dry_run_behavior: str
    exactness_status: str
    canonical_run_root_required: bool = True
    requires_explicit_acknowledgement: bool = False


SINGLE_PROCESS_ENTRYPOINT = "python src/main.py"
DISTRIBUTED_CONTROLLER_ENTRYPOINT = "pipeline.distributed.controller.plan_distributed_run"
DISTRIBUTED_WORKER_ENTRYPOINT = "python -m pipeline.distributed.worker"
NEGATIVE_CONTEXT_ENTRYPOINT = "python -m pipeline.negative_context"
CANDIDATE_SELECTION_ENTRYPOINT = "python -m pipeline.candidate_selection"


OPERATING_MODE_DEFINITIONS: Mapping[RunMode, OperatingModeDefinition] = {
    RunMode.SINGLE_PROCESS: OperatingModeDefinition(
        name=RunMode.SINGLE_PROCESS.value,
        description=(
            "Current single-process pipeline and permanent correctness oracle for "
            "distributed equivalence checks."
        ),
        required_parts=("single_process_pipeline",),
        allowed_entrypoints=(SINGLE_PROCESS_ENTRYPOINT,),
        dry_run_behavior=(
            "No distributed worker dry-run; planned run-root support should resolve "
            "the canonical output paths without launching workers."
        ),
        exactness_status="single_process_oracle",
    ),
    RunMode.DISTRIBUTED_SIMPLE_EXACT: OperatingModeDefinition(
        name=RunMode.DISTRIBUTED_SIMPLE_EXACT.value,
        description=(
            "Exact distributed pass 1, pass 2, and discovery using worker-local "
            "candidate dumps plus centralized simple exact pass-2 reduction."
        ),
        required_parts=("part1", "part2", "part3", "part4", "part5_mode_a", "part6"),
        allowed_entrypoints=(
            DISTRIBUTED_CONTROLLER_ENTRYPOINT,
            DISTRIBUTED_WORKER_ENTRYPOINT,
            NEGATIVE_CONTEXT_ENTRYPOINT,
            CANDIDATE_SELECTION_ENTRYPOINT,
        ),
        dry_run_behavior=(
            "Controller dry-run must build the manifest, output layout, worker "
            "assignments, artifact paths, and per-worker commands without model or SAE loading."
        ),
        exactness_status="exact_equivalent_after_oracle_checks",
    ),
    RunMode.DISTRIBUTED_MAPREDUCE_EXACT: OperatingModeDefinition(
        name=RunMode.DISTRIBUTED_MAPREDUCE_EXACT.value,
        description=(
            "Exact distributed pass 1, pass 2, and discovery using pass-2 partial-sum "
            "shuffle and target-range MapReduce reducers."
        ),
        required_parts=(
            "part1",
            "part2",
            "part3",
            "part4",
            "part5_mode_a",
            "part5_mode_b",
            "part6",
        ),
        allowed_entrypoints=(
            DISTRIBUTED_CONTROLLER_ENTRYPOINT,
            DISTRIBUTED_WORKER_ENTRYPOINT,
            NEGATIVE_CONTEXT_ENTRYPOINT,
            CANDIDATE_SELECTION_ENTRYPOINT,
        ),
        dry_run_behavior=(
            "Controller dry-run must expose worker assignments and reducer target ranges "
            "without model, SAE, or reducer execution."
        ),
        exactness_status="exact_mapreduce_equivalent_after_simple_exact_checks",
    ),
    RunMode.DISTRIBUTED_EXPERIMENTAL_FAST: OperatingModeDefinition(
        name=RunMode.DISTRIBUTED_EXPERIMENTAL_FAST.value,
        description=(
            "Opt-in approximate or quality-changing distributed mode for experiments "
            "only after exact baseline artifacts have been produced."
        ),
        required_parts=("part1", "exact_baseline_artifacts", "experimental_policy"),
        allowed_entrypoints=(
            DISTRIBUTED_CONTROLLER_ENTRYPOINT,
            DISTRIBUTED_WORKER_ENTRYPOINT,
        ),
        dry_run_behavior=(
            "Controller dry-run must print every quality-changing toggle and refuse "
            "execution unless explicit experimental acknowledgement is present."
        ),
        exactness_status="experimental_non_exact",
        requires_explicit_acknowledgement=True,
    ),
}


def operating_mode_definition(mode: RunMode | str) -> OperatingModeDefinition:
    """Return the documented contract for a supported operating mode."""

    parsed_mode = mode if isinstance(mode, RunMode) else RunMode(mode)
    return OPERATING_MODE_DEFINITIONS[parsed_mode]


def all_operating_mode_definitions() -> tuple[OperatingModeDefinition, ...]:
    """Return mode definitions in RunMode declaration order."""

    return tuple(OPERATING_MODE_DEFINITIONS[mode] for mode in RunMode)


def canonical_run_root(output_base: str | Path, run_id: str) -> Path:
    """Resolve the canonical `outputs/<run_id>/` root for every mode."""

    return Path(output_base) / run_id


def default_run_root(
    output_base: str | Path,
    normalized_config_hash: str,
    *,
    run_id: str | None = None,
) -> Path:
    """Resolve a caller-provided run ID or generate one from the config hash."""

    return canonical_run_root(output_base, run_id or generate_run_id(normalized_config_hash))
