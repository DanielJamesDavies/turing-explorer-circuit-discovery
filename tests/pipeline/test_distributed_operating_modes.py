from datetime import datetime, timezone
from pathlib import Path

from pipeline.distributed.manifest import RunMode, generate_run_id
from pipeline.distributed.operating_modes import (
    CANDIDATE_SELECTION_ENTRYPOINT,
    DISTRIBUTED_CONTROLLER_ENTRYPOINT,
    DISTRIBUTED_WORKER_ENTRYPOINT,
    NEGATIVE_CONTEXT_ENTRYPOINT,
    OPERATING_MODE_DEFINITIONS,
    SINGLE_PROCESS_ENTRYPOINT,
    all_operating_mode_definitions,
    canonical_run_root,
    default_run_root,
    operating_mode_definition,
)


def test_every_run_mode_has_documented_taxonomy():
    assert set(OPERATING_MODE_DEFINITIONS) == set(RunMode)

    for mode in RunMode:
        definition = operating_mode_definition(mode)
        assert definition.name == mode.value
        assert definition.description
        assert definition.required_parts
        assert definition.allowed_entrypoints
        assert definition.dry_run_behavior
        assert definition.exactness_status
        assert definition.canonical_run_root_required is True


def test_single_process_is_documented_as_oracle_with_main_entrypoint():
    definition = operating_mode_definition(RunMode.SINGLE_PROCESS)

    assert "correctness oracle" in definition.description
    assert definition.required_parts == ("single_process_pipeline",)
    assert definition.allowed_entrypoints == (SINGLE_PROCESS_ENTRYPOINT,)
    assert definition.exactness_status == "single_process_oracle"
    assert "No distributed worker dry-run" in definition.dry_run_behavior


def test_distributed_simple_exact_documents_parts_and_dry_run_contract():
    definition = operating_mode_definition("distributed_simple_exact")

    assert definition.required_parts == (
        "part1",
        "part2",
        "part3",
        "part4",
        "part5_mode_a",
        "part6",
    )
    assert DISTRIBUTED_CONTROLLER_ENTRYPOINT in definition.allowed_entrypoints
    assert DISTRIBUTED_WORKER_ENTRYPOINT in definition.allowed_entrypoints
    assert NEGATIVE_CONTEXT_ENTRYPOINT in definition.allowed_entrypoints
    assert CANDIDATE_SELECTION_ENTRYPOINT in definition.allowed_entrypoints
    assert "without model or SAE loading" in definition.dry_run_behavior
    assert definition.requires_explicit_acknowledgement is False


def test_mapreduce_exact_requires_mode_b_and_simple_exact_checks():
    definition = operating_mode_definition(RunMode.DISTRIBUTED_MAPREDUCE_EXACT)

    assert "part5_mode_b" in definition.required_parts
    assert "pass-2 partial-sum shuffle" in definition.description
    assert definition.exactness_status == "exact_mapreduce_equivalent_after_simple_exact_checks"
    assert "reducer target ranges" in definition.dry_run_behavior


def test_experimental_fast_requires_acknowledgement_and_marks_non_exact():
    definition = operating_mode_definition(RunMode.DISTRIBUTED_EXPERIMENTAL_FAST)

    assert definition.requires_explicit_acknowledgement is True
    assert definition.exactness_status == "experimental_non_exact"
    assert "Opt-in approximate" in definition.description
    assert "quality-changing toggle" in definition.dry_run_behavior
    assert "exact_baseline_artifacts" in definition.required_parts


def test_mode_definitions_are_returned_in_run_mode_order():
    definitions = all_operating_mode_definitions()

    assert [definition.name for definition in definitions] == [mode.value for mode in RunMode]


def test_canonical_run_root_policy_and_default_run_id_shape():
    run_id = "20260517-002500-abcdef12"

    assert canonical_run_root(Path("outputs"), run_id) == Path("outputs") / run_id
    assert default_run_root(Path("outputs"), "abcdef1234567890", run_id=run_id) == (
        Path("outputs") / run_id
    )
    assert generate_run_id(
        "ABCDEF1234567890",
        timestamp=datetime(2026, 5, 17, 0, 25, 0, tzinfo=timezone.utc),
    ) == run_id
