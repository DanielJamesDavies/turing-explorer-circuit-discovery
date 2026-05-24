# Plan: Negative Context Pipeline Refactor

> **Goal:** Split `src/pipeline/negative_context.py` into a focused pipeline-stage package while preserving existing CLI behavior, run-root artifact contracts, resume semantics, reports, and integration with `store.neg_context`.
>
> **Created:** 2026-05-23

---

## Phase 1 - Establish Compatibility Baseline

- [x] Record the current public API from `src/pipeline/negative_context.py`: loaded artifact dataclasses, stage plan/result dataclasses, input loading, stage execution, resume classification, metadata/report builders, backend comparison, in-pipeline `build_negative_contexts`, and CLI `main`.
- [x] Search repository imports of `pipeline.negative_context` and direct function/class imports to identify compatibility requirements.
- [x] Confirm this plan is separate from `agent-planning/refactoring/neg-context-store-refactor.md`: pipeline code may depend on `store.neg_context`, but store-level backend code must not depend on pipeline manifests or run-root layout.
- [x] Run the focused baseline command before refactoring: `python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py tests/pipeline/test_distributed_manifest.py -q`.
- [x] Note any pre-existing failures, environment issues, or skipped tests in this plan before implementation begins.

### Phase 1 Notes

Current public and compatibility-sensitive names in `src/pipeline/negative_context.py`:

- Input contracts and loaded artifact types: `SeqReprLike`, `LoadedContext`, `LoadedSeqRepr`, `NegativeContextInputs`, and `BuildNegCtxFn`.
- Stage result, comparison, and planning dataclasses: `NegativeContextRunResult`, `NegativeContextComparisonResult`, `NegativeContextStagePlan`, and `NegativeContextStageClassification`.
- Public stage helpers: `load_negative_context_inputs`, `run_negative_context_stage`, `plan_negative_context_stage`, `classify_negative_context_stage`, `build_negative_context_stage_metadata`, `build_negative_context_sanity_report`, and `print_negative_context_sanity_summary`.
- Backend comparison helpers: `compare_negative_context_backends` and `build_negative_context_comparison_report`.
- Runtime and CLI helpers: `build_negative_contexts`, `main`, and `configured_neg_ctx_sequences`.
- Compatibility-sensitive private helpers currently owned by the large module and planned for mechanical extraction: `_require_artifacts`, `_load_torch_payload`, `_validate_config_hash_if_present`, `_context_from_payload`, `_seq_repr_from_payload`, `_validate_seq_repr_cap_mapping`, `_validate_negative_context_inputs`, `_empty_neg_context_like`, `_manifest_neg_ctx_devices`, `_manifest_neg_ctx_devices_from_manifest`, `_manifest_neg_ctx_config`, `_neg_ctx_part_dir`, `_artifact_metadata`, `_write_part_marker`, `_populated_row_count`, `_fill_summary`, `_neg_ctx_validation_summary`, `_sample_row_comparisons`, `_stats_timing_ms`, and `_atomic_write_json`.

Observed compatibility call sites:

- `src/pipeline/run_pipeline.py` imports `build_negative_contexts` from `pipeline.negative_context`.
- `src/pipeline/distributed/controller.py` imports `run_negative_context_stage` lazily inside `run_parts_1_to_3`.
- `src/pipeline/distributed/operating_modes.py` documents the standalone entrypoint as `python -m pipeline.negative_context`.
- `tests/pipeline/test_negative_context_stage.py` imports `LoadedContext`, `load_negative_context_inputs`, `run_negative_context_stage`, `plan_negative_context_stage`, `compare_negative_context_backends`, and `build_negative_context_comparison_report` directly from `pipeline.negative_context`.
- `tests/pipeline/test_negative_context_stage.py` also monkeypatches module globals under `pipeline.negative_context`, including `neg_ctx.num_ctx_sequences`, `config.hardware.ann_device`, `config.latents.neg_ctx.*`, and `build_neg_ctx`.
- `tests/pipeline/test_distributed_pass1_merge.py` imports `run_negative_context_stage` from `pipeline.negative_context`.

Boundary decision:

- Keep this pipeline-stage refactor separate from the completed `store.neg_context` backend refactor.
- `src/pipeline/negative_context.py` may continue to import backend entrypoints and validation from `store.neg_context`.
- `src/store/neg_context.py` and files under `src/store/neg_ctx/` currently have no imports of `pipeline.negative_context`, distributed manifests, negative-context run configs, run-root layout helpers, or distributed part metadata, so the dependency direction is currently clean.

Baseline verification:

- Command: `python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py tests/pipeline/test_distributed_manifest.py -q`
- Result on 2026-05-24: `50 passed in 1.39s`.
- Pre-existing failures/blockers: none observed.

## Phase 2 - Create The Pipeline Package Shell

- [x] Create `src/pipeline/negative_context_stage/__init__.py` as the new package entrypoint.
- [x] Use `negative_context_stage` rather than `negative_context/` initially to avoid a Python import conflict with the existing `src/pipeline/negative_context.py` module.
- [x] Keep `src/pipeline/negative_context.py` as the compatibility facade and CLI module during extraction.
- [x] Re-export moved public names from `src/pipeline/negative_context.py`.
- [x] Verify imports with `python -m pytest tests/pipeline/test_negative_context_stage.py -q`.

### Phase 2 Notes

- Added `src/pipeline/negative_context_stage/__init__.py` as the new package entrypoint.
- Used lazy package exports through `__getattr__` so `pipeline.negative_context_stage` can expose the intended public API while implementation still lives in `pipeline.negative_context`.
- Left `src/pipeline/negative_context.py` unchanged in this phase so it remains the source-of-truth module and keeps `python -m pipeline.negative_context` working during extraction.
- The package `__all__` mirrors the planned public stage API: input contracts, loaded artifact dataclasses, stage planning/result dataclasses, stage execution, resume classification, metadata/report helpers, backend comparison, runtime wrapper, CLI `main`, and `configured_neg_ctx_sequences`.
- Import smoke check on 2026-05-24 with `PYTHONPATH=src`: `LoadedContext` and `run_negative_context_stage` resolve through `pipeline.negative_context_stage`.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_negative_context_stage.py -q` -> `16 passed in 0.34s`.
- Lints/diagnostics reported no errors for `src/pipeline/negative_context_stage/__init__.py`.

## Phase 3 - Extract Artifact Input Contracts And Loaders

- [x] Create `src/pipeline/negative_context_stage/inputs.py`.
- [x] Move `SeqReprLike`, `LoadedContext`, `LoadedSeqRepr`, `NegativeContextInputs`, `BuildNegCtxFn`, and `load_negative_context_inputs`.
- [x] Move input-only helpers: `_require_artifacts`, `_load_torch_payload`, `_validate_config_hash_if_present`, `_context_from_payload`, `_seq_repr_from_payload`, `_validate_seq_repr_cap_mapping`, `_validate_negative_context_inputs`, and `_empty_neg_context_like`.
- [x] Preserve run-root artifact names and tensor payload validation exactly.
- [x] Re-export public input contracts from `src/pipeline/negative_context.py` and `src/pipeline/negative_context_stage/__init__.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_negative_context_stage.py -q`.

### Phase 3 Notes

- Added `src/pipeline/negative_context_stage/inputs.py` for the negative-context stage input contracts and pass-1 artifact loading path.
- Moved `SeqReprLike`, `LoadedContext`, `LoadedSeqRepr`, `NegativeContextInputs`, `BuildNegCtxFn`, and `load_negative_context_inputs` into the new inputs module.
- Moved input-only helper ownership for required artifact checks, torch payload loading, config-hash validation, context payload conversion, sequence-representation payload conversion, capped `seq_repr` mapping validation, input compatibility validation, and empty `neg_ctx` output construction.
- Updated `src/pipeline/negative_context.py` to import and re-export the moved public names and compatibility-sensitive private helpers, so existing imports from `pipeline.negative_context` still resolve.
- Updated `src/pipeline/negative_context_stage/__init__.py` to export the public input contracts directly from `inputs.py` while leaving later-stage names lazily resolved from `pipeline.negative_context`.
- Import smoke check on 2026-05-24 with `PYTHONPATH=src`: old `pipeline.negative_context`, new `pipeline.negative_context_stage`, and `pipeline.negative_context_stage.inputs` resolve `LoadedContext` and `load_negative_context_inputs` to the same objects; `_empty_neg_context_like` also remains available through the old module.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_negative_context_stage.py -q` -> `16 passed in 0.38s`.
- Lints/diagnostics reported no errors for `src/pipeline/negative_context.py`, `src/pipeline/negative_context_stage/__init__.py`, or `src/pipeline/negative_context_stage/inputs.py`.

## Phase 4 - Extract Stage Planning And Resume Classification

- [x] Create `src/pipeline/negative_context_stage/planning.py`.
- [x] Move `NegativeContextStagePlan`, `NegativeContextStageClassification`, `plan_negative_context_stage`, and `classify_negative_context_stage`.
- [x] Move planning/resume helpers: `_neg_ctx_part_dir`, `_artifact_metadata`, `_manifest_neg_ctx_devices`, `_manifest_neg_ctx_devices_from_manifest`, and `_manifest_neg_ctx_config` if they are not better owned by stage execution.
- [x] Preserve completed/missing/failed/stale classification behavior and metadata equality checks.
- [x] Re-export public planning functions/classes from `src/pipeline/negative_context.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_negative_context_stage.py tests/pipeline/test_distributed_manifest.py -q`.

### Phase 4 Notes

- Added `src/pipeline/negative_context_stage/planning.py` for stage planning, resume classification, negative-context metadata construction, manifest-derived device selection, manifest `neg_ctx` config construction, part-directory resolution, and artifact metadata snapshots.
- Moved `NegativeContextStagePlan`, `NegativeContextStageClassification`, `plan_negative_context_stage`, and `classify_negative_context_stage` into the new planning module.
- Moved `_neg_ctx_part_dir`, `_artifact_metadata`, `_manifest_neg_ctx_devices`, `_manifest_neg_ctx_devices_from_manifest`, and `_manifest_neg_ctx_config` into the planning module while re-exporting them through `pipeline.negative_context` for compatibility.
- Moved `build_negative_context_stage_metadata` in this phase as well because `plan_negative_context_stage` depends on it; keeping the builder in `pipeline.negative_context` would create a circular dependency between the facade and the extracted planning module.
- Updated `src/pipeline/negative_context.py` to import and re-export the moved planning names, preserving existing imports from `pipeline.negative_context`.
- Updated `src/pipeline/negative_context_stage/__init__.py` to expose public planning names directly from `planning.py`.
- Import smoke check on 2026-05-24 with `PYTHONPATH=src`: old `pipeline.negative_context`, new `pipeline.negative_context_stage`, and `pipeline.negative_context_stage.planning` resolve `NegativeContextStagePlan`, `plan_negative_context_stage`, and `build_negative_context_stage_metadata` to the same objects; `_neg_ctx_part_dir` remains available through the old module.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_negative_context_stage.py tests/pipeline/test_distributed_manifest.py -q` -> `31 passed in 0.42s`.
- Lints/diagnostics reported no errors for `src/pipeline/negative_context.py`, `src/pipeline/negative_context_stage/__init__.py`, or `src/pipeline/negative_context_stage/planning.py`.

## Phase 5 - Extract Stage Execution

- [x] Create `src/pipeline/negative_context_stage/stage.py`.
- [x] Move `NegativeContextRunResult` and `run_negative_context_stage`; `build_negative_context_stage_metadata` moved in Phase 4 with planning to avoid a circular dependency.
- [x] Keep store backend calls through `store.neg_context.build_neg_ctx` and `store.neg_context.validate_neg_ctx_output`.
- [x] Preserve marker writes for `started.json`, `failed.json`, and `completed.json`.
- [x] Preserve manifest update behavior for `NegativeContextRunConfig`.
- [x] Preserve dry-run and resume behavior exactly.
- [x] Re-export `run_negative_context_stage`, `NegativeContextRunResult`, and `build_negative_context_stage_metadata` from `src/pipeline/negative_context.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_negative_context_stage.py -q`.

### Phase 5 Notes

- Added `src/pipeline/negative_context_stage/stage.py` for negative-context stage execution.
- Moved `NegativeContextRunResult`, `run_negative_context_stage`, and the stage-owned `_write_part_marker` helper into the new stage module.
- Kept `src/pipeline/negative_context.py` as the compatibility surface by importing and re-exporting the moved stage names.
- Preserved existing monkeypatch compatibility for tests and callers that patch `pipeline.negative_context.build_neg_ctx` or `pipeline.negative_context.validate_neg_ctx_output`: the extracted stage runner resolves backend, validation, report, and atomic-write helpers through the old facade at runtime.
- Preserved marker behavior for `started.json`, `failed.json`, and `completed.json`, including status fields, metadata payloads, error payloads, and completed artifact paths.
- Preserved manifest update behavior by continuing to write `NegativeContextRunConfig` through `_manifest_neg_ctx_config`.
- Updated `src/pipeline/negative_context_stage/__init__.py` to expose `NegativeContextRunResult` and `run_negative_context_stage` directly from `stage.py`.
- Import smoke check on 2026-05-24 with `PYTHONPATH=src`: old `pipeline.negative_context`, new `pipeline.negative_context_stage`, and `pipeline.negative_context_stage.stage` resolve `NegativeContextRunResult` and `run_negative_context_stage` to the same objects; `_write_part_marker` remains available through the old module.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_negative_context_stage.py -q` -> `16 passed in 0.33s`.
- Lints/diagnostics reported no errors for `src/pipeline/negative_context.py`, `src/pipeline/negative_context_stage/__init__.py`, or `src/pipeline/negative_context_stage/stage.py`.

## Phase 6 - Extract Reports And Sanity Summary

- [x] Create `src/pipeline/negative_context_stage/reports.py`.
- [x] Move `build_negative_context_sanity_report` and `print_negative_context_sanity_summary`.
- [x] Move report helpers: `_populated_row_count`, `_fill_summary`, `_neg_ctx_validation_summary`, `_stats_timing_ms`, and `_atomic_write_json` if still shared.
- [x] Preserve sanity report schema, metadata shape, timing keys, memory keys, fill distribution, and console summary format.
- [x] Re-export public report functions from `src/pipeline/negative_context.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_negative_context_stage.py -q`.

### Phase 6 Notes

- Added `src/pipeline/negative_context_stage/reports.py` for sanity report construction, console summary formatting, validation summaries, fill/populated-row summaries, stats timing summaries, and atomic JSON writes.
- Moved `build_negative_context_sanity_report`, `print_negative_context_sanity_summary`, `_populated_row_count`, `_fill_summary`, `_neg_ctx_validation_summary`, `_stats_timing_ms`, and `_atomic_write_json` into the reports module.
- Updated `src/pipeline/negative_context.py` to import and re-export the moved public report functions and compatibility-sensitive private helpers, so existing old-module imports still resolve.
- Updated `src/pipeline/negative_context_stage/__init__.py` to expose public report functions directly from `reports.py`.
- The backend comparison path still uses `_fill_summary`, `_populated_row_count`, `_stats_timing_ms`, and `_atomic_write_json` through the old facade; those names now resolve to the reports module.
- Import smoke check on 2026-05-24 with `PYTHONPATH=src`: old `pipeline.negative_context`, new `pipeline.negative_context_stage`, and `pipeline.negative_context_stage.reports` resolve `build_negative_context_sanity_report` and `print_negative_context_sanity_summary` to the same objects; `_atomic_write_json` and `_fill_summary` remain available through the old module.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_negative_context_stage.py -q` -> `16 passed in 0.36s`.
- Lints/diagnostics reported no errors for `src/pipeline/negative_context.py`, `src/pipeline/negative_context_stage/__init__.py`, or `src/pipeline/negative_context_stage/reports.py`.

## Phase 7 - Extract Backend Comparison

- [x] Create `src/pipeline/negative_context_stage/comparison.py`.
- [x] Move `NegativeContextComparisonResult`, `compare_negative_context_backends`, and `build_negative_context_comparison_report`.
- [x] Move `_sample_row_comparisons` if it is only used by comparison reporting.
- [x] Preserve equivalence report schema, tolerance handling, sample row content, ordering/tie note, and output path `neg_ctx_equivalence_report.json`.
- [x] Re-export comparison functions/classes from `src/pipeline/negative_context.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py -q`.

### Phase 7 Notes

- Added `src/pipeline/negative_context_stage/comparison.py` for negative-context backend equivalence checks.
- Moved `NegativeContextComparisonResult`, `compare_negative_context_backends`, `build_negative_context_comparison_report`, and `_sample_row_comparisons` into the comparison module.
- Kept the equivalence report path `neg_ctx_equivalence_report.json`, schema fields, tolerance payload, sample row payloads, fill/populated-row summaries, timing summaries, and ordering/tie note unchanged.
- Updated `src/pipeline/negative_context.py` to import and re-export the moved comparison names and `_sample_row_comparisons`, preserving old import compatibility.
- Updated `src/pipeline/negative_context_stage/__init__.py` to expose public comparison names directly from `comparison.py`.
- Import smoke check on 2026-05-24 with `PYTHONPATH=src`: old `pipeline.negative_context`, new `pipeline.negative_context_stage`, and `pipeline.negative_context_stage.comparison` resolve `NegativeContextComparisonResult`, `compare_negative_context_backends`, and `build_negative_context_comparison_report` to the same objects; `_sample_row_comparisons` remains available through the old module.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py -q` -> `35 passed in 0.37s`.
- Lints/diagnostics reported no errors for `src/pipeline/negative_context.py`, `src/pipeline/negative_context_stage/__init__.py`, or `src/pipeline/negative_context_stage/comparison.py`.

## Phase 8 - Extract In-Pipeline Runtime Wrapper And CLI

- [x] Create `src/pipeline/negative_context_stage/runtime.py`.
- [x] Move `build_negative_contexts`, keeping its interaction with initialized runtime stores unchanged.
- [x] Create `src/pipeline/negative_context_stage/cli.py`.
- [x] Move `main`, CLI parser construction, dry-run handling, compare-backends handling, and normal stage invocation.
- [x] Keep `python -m pipeline.negative_context` working by delegating from `src/pipeline/negative_context.py`.
- [x] Move `configured_neg_ctx_sequences` to `cli.py` or `reports.py` depending on actual usage.
- [x] Verify CLI behavior with `python -m pytest tests/pipeline/test_negative_context_stage.py tests/pipeline/test_distributed_controller.py -q`.

### Phase 8 Notes

- Added `src/pipeline/negative_context_stage/runtime.py` for the in-pipeline `build_negative_contexts` wrapper.
- Moved `build_negative_contexts` into the runtime module while preserving compatibility with old-module globals by resolving runtime stores and `build_neg_ctx` through `pipeline.negative_context` at call time.
- Added `src/pipeline/negative_context_stage/cli.py` for `main`, CLI parser construction, dry-run handling, compare-backends handling, normal stage invocation, and `configured_neg_ctx_sequences`.
- Updated `src/pipeline/negative_context.py` to import and re-export `build_negative_contexts`, `main`, and `configured_neg_ctx_sequences`; `python -m pipeline.negative_context` still delegates to `main()`.
- Updated `src/pipeline/negative_context_stage/__init__.py` to expose the runtime and CLI public names directly.
- Import smoke check on 2026-05-24 with `PYTHONPATH=src`: old `pipeline.negative_context`, new `pipeline.negative_context_stage`, `pipeline.negative_context_stage.runtime`, and `pipeline.negative_context_stage.cli` resolve `build_negative_contexts`, `main`, and `configured_neg_ctx_sequences` to the same objects.
- CLI smoke check on 2026-05-24: `PYTHONPATH=src python -m pipeline.negative_context --help` exits successfully and shows the existing flags.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_negative_context_stage.py tests/pipeline/test_distributed_controller.py -q` -> `42 passed in 0.82s`.
- Lints/diagnostics reported no errors for `src/pipeline/negative_context.py`, `src/pipeline/negative_context_stage/__init__.py`, `src/pipeline/negative_context_stage/runtime.py`, or `src/pipeline/negative_context_stage/cli.py`.

## Phase 9 - Reduce `negative_context.py` To A Compatibility Facade

- [x] Replace `src/pipeline/negative_context.py` with a small facade that imports public names from `src/pipeline/negative_context_stage/`.
- [x] Define `__all__` in `src/pipeline/negative_context_stage/__init__.py` or `src/pipeline/negative_context.py` so the stable public API is explicit.
- [x] Confirm all existing imports from `pipeline.negative_context` still resolve without caller changes.
- [x] Avoid broad test rewrites; only adjust tests if they intentionally inspect private helper ownership.
- [x] Confirm `src/pipeline/negative_context.py` is reduced to a small facade, ideally under 120 LOC.

### Phase 9 Notes

- Reduced `src/pipeline/negative_context.py` to an 85-line compatibility facade.
- Updated the facade to import the stable public API from `src/pipeline/negative_context_stage/` and define an explicit `__all__` matching the package entrypoint.
- Kept compatibility-sensitive module globals and private helper re-exports on `pipeline.negative_context`, including `config`, `neg_ctx`, `build_neg_ctx`, `validate_neg_ctx_output`, `_empty_neg_context_like`, `_atomic_write_json`, `_write_part_marker`, and related private helpers used by existing tests or old callers.
- Confirmed the facade no longer owns any class or function definitions except the `python -m pipeline.negative_context` module-execution delegation.
- Import smoke check on 2026-05-24 with `PYTHONPATH=src`: `pipeline.negative_context.__all__ == pipeline.negative_context_stage.__all__`, public objects such as `run_negative_context_stage` and `build_negative_contexts` resolve to the same objects through old and new paths, and compatibility globals/private helpers remain present on the old module.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_negative_context_stage.py -q` -> `16 passed in 0.35s`.
- Lints/diagnostics reported no errors for `src/pipeline/negative_context.py` or `src/pipeline/negative_context_stage/__init__.py`.

## Phase 10 - Testing And Verification

- [x] Run focused stage tests: `python -m pytest tests/pipeline/test_negative_context_stage.py -q`.
- [x] Run store backend tests: `python -m pytest tests/store/test_neg_context_backend.py -q`.
- [x] Run distributed integration-adjacent tests: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_manifest.py -q`.
- [x] Run lints/diagnostics on all newly created and edited files.
- [x] Confirm no run-root artifact path, marker schema, sanity report schema, equivalence report schema, manifest update behavior, or CLI behavior changed.
- [x] Confirm no generated artifacts, native binaries, or unrelated local outputs were touched by the refactor.

### Phase 10 Notes

- Focused stage verification on 2026-05-24: `python -m pytest tests/pipeline/test_negative_context_stage.py -q` -> `16 passed in 0.36s`.
- Store backend verification on 2026-05-24: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `19 passed in 0.12s`.
- Distributed integration-adjacent verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_manifest.py -q` -> `80 passed in 1.07s`.
- Lints/diagnostics reported no errors for `src/pipeline/negative_context.py`, files under `src/pipeline/negative_context_stage/`, or this plan file.
- Import/API smoke check on 2026-05-24 confirmed `pipeline.negative_context.__all__ == pipeline.negative_context_stage.__all__`, old and new public imports resolve to the same objects, and compatibility globals/private helpers such as `config`, `neg_ctx`, `build_neg_ctx`, and `_write_part_marker` remain available from the old module.
- Confirmed `src/pipeline/negative_context.py` no longer owns class or function definitions; implementation ownership now lives under `src/pipeline/negative_context_stage/`.
- The focused and integration tests above cover run-root artifact paths, started/failed/completed marker schemas, sanity report schema, equivalence report schema, manifest `neg_ctx` update behavior, and CLI/controller integration.
- `git status --short` shows this refactor's negative-context source/planning changes plus pre-existing unrelated refactor and documentation changes; no generated artifacts, native binaries, or output-run files were touched by this phase.

---

## Open Questions

- Should the new package be named `negative_context_stage` permanently, or should a later migration rename `negative_context.py` into a true `negative_context/` package?
- Should `_atomic_write_json` stay local to this package or move to a shared pipeline utility module used by other distributed stages?
- Should planning and stage execution share a small metadata builder module, or is keeping metadata construction in `stage.py` clearer?
- Should `configured_neg_ctx_sequences` remain public if no callers outside tests use it?

## Risks / Assumptions

- The store-level `neg_ctx` backend refactor and this pipeline-stage refactor should remain separate; pipeline depends on store, not the other way around.
- Resume classification relies on exact metadata equality; extraction must not change metadata key order/content or report payload shape.
- CLI behavior and command names are part of the distributed rollout workflow and should remain stable.
- This refactor should not change backend algorithms, canonical artifact names, sanity/equivalence report schemas, or public import paths.
