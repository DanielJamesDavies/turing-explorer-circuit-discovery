# Plan: Distributed Controller Refactor

> **Goal:** Split `src/pipeline/distributed/controller.py` into focused controller modules while preserving the existing controller CLI, public imports, dry-run output, manifest planning behavior, and worker command contract.
>
> **Created:** 2026-05-23

---

## Phase 1 - Establish Compatibility Baseline

- [x] Record the current public API from `src/pipeline/distributed/controller.py`: controller dataclasses, CLI helpers, `plan_distributed_run`, config/preflight helpers, worker command helpers, dry-run/report helpers, resume classification, and `run_parts_1_to_3`.
- [x] Search repository imports of `pipeline.distributed.controller` and direct function/class imports to identify compatibility requirements.
- [x] Confirm the first refactor should use sibling modules such as `controller_planning.py` rather than a `controller/` package to avoid a Python import conflict with `controller.py`.
- [x] Run the focused baseline command before refactoring: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_layout.py -q`.
- [x] Note any pre-existing failures, environment issues, or skipped tests in this plan before implementation begins.

### Phase 1 Notes

Current public and compatibility-sensitive names in `src/pipeline/distributed/controller.py`:

- Controller/result dataclasses: `PreflightReport`, `WorkerCommand`, `DiscoveryDryRunEstimate`, `LocalCompatibilityReport`, `H100ExactModeReport`, `ControllerPlan`, and `DistributedParts1To3Result`.
- CLI and planning entrypoints: `build_arg_parser`, `main`, `plan_distributed_run_from_args`, and `plan_distributed_run`.
- Config and preflight helpers: `load_and_hash_config`, `run_preflight_checks`, `native_extension_availability`, `REQUIRED_NATIVE_EXTENSIONS`, `_validate_config_strict`, `_normalize_for_hash`, `_root_config_dump`, `_resolve_config_path`, `_check_output_writable`, `_candidate_dump_m_from_config`, `_distributed_cli_defaults`, `_parse_physical_ids`, and `_visible_cuda_device_count`.
- Worker command and launch helpers: `build_worker_commands`, `launch_worker_processes`, and `_worker_pythonpath`.
- Dry-run and report helpers: `format_dry_run`, `build_discovery_dry_run_estimate`, `build_local_compatibility_report`, `build_h100_exact_mode_report`, `_format_discovery_dry_run_estimate`, `_format_local_compatibility_report`, and `_format_h100_exact_mode_report`.
- Resume and integrated-stage helpers: `classify_resume_workers`, `_validate_marker_identity`, and `run_parts_1_to_3`.

Observed compatibility call sites:

- `src/pipeline/distributed/__init__.py` re-exports the stable controller surface from `pipeline.distributed.controller`, including the dataclasses, planning, preflight, command, report, resume, and integrated-stage helpers.
- `src/pipeline/distributed/operating_modes.py` documents `pipeline.distributed.controller.plan_distributed_run` as the distributed controller entrypoint.
- `tests/pipeline/test_distributed_controller.py` imports `pipeline.distributed.controller` as a module and directly imports `build_arg_parser`, `build_worker_commands`, `build_discovery_dry_run_estimate`, `classify_resume_workers`, `launch_worker_processes`, `load_and_hash_config`, `main`, `plan_distributed_run`, `plan_distributed_run_from_args`, and `run_parts_1_to_3`.
- Repository search did not find production callers outside `src/pipeline/distributed/__init__.py` and the operating-mode entrypoint string, but the facade must preserve old imports because tests and external scheduler/manual workflows use `python -m pipeline.distributed.controller`.

Extraction shape decision:

- Use sibling modules such as `controller_contracts.py`, `controller_config.py`, and `controller_planning.py`, not a `controller/` package, because `src/pipeline/distributed/controller.py` must remain importable as the compatibility facade during this refactor.

Baseline verification:

- Command: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_layout.py -q`
- Result on 2026-05-24: `56 passed in 0.61s`.
- Pre-existing failures/blockers: none observed.

## Phase 2 - Extract Controller Contracts

- [x] Create `src/pipeline/distributed/controller_contracts.py`.
- [x] Move `PreflightReport`, `WorkerCommand`, `DiscoveryDryRunEstimate`, `LocalCompatibilityReport`, `H100ExactModeReport`, `ControllerPlan`, and `DistributedParts1To3Result`.
- [x] Update `controller.py` to import and re-export these dataclasses.
- [x] Keep dataclass field names and types unchanged.
- [x] Verify with `python -m pytest tests/pipeline/test_distributed_controller.py -q`.

### Phase 2 Notes

- Added `src/pipeline/distributed/controller_contracts.py` for the shared controller dataclasses and an explicit `__all__`.
- Moved `PreflightReport`, `WorkerCommand`, `DiscoveryDryRunEstimate`, `LocalCompatibilityReport`, `H100ExactModeReport`, `ControllerPlan`, and `DistributedParts1To3Result` without changing field names, field types, defaults, or frozen dataclass behavior.
- Updated `src/pipeline/distributed/controller.py` to import the dataclasses from `controller_contracts.py`, so existing imports from `pipeline.distributed.controller` and `pipeline.distributed` continue to resolve.
- Removed the now-unused `dataclass` and `RunLayout` imports from `controller.py`; `RunLayout` remains owned by the contract module for `ControllerPlan`.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_controller.py -q` -> `26 passed in 0.50s`.
- Lints/diagnostics reported no errors for `controller.py` or `controller_contracts.py`.

## Phase 3 - Extract Config Loading And CLI Defaults

- [x] Create `src/pipeline/distributed/controller_config.py`.
- [x] Move `load_and_hash_config`, `_validate_config_strict`, `_normalize_for_hash`, `_root_config_dump`, `_resolve_config_path`, `_candidate_dump_m_from_config`, `_distributed_cli_defaults`, and `_parse_physical_ids`.
- [x] Preserve strict Pydantic validation, normalized config hash behavior, and CLI default extraction behavior exactly.
- [x] Update `controller.py` to import and re-export public config helpers.
- [x] Verify config behavior with `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_config.py -q`.

### Phase 3 Notes

- Added `src/pipeline/distributed/controller_config.py` for strict config loading, normalized hash construction, root config dumping, model/SAE path resolution, candidate dump sizing, distributed CLI defaults, and physical device parsing.
- Moved `load_and_hash_config`, `_validate_config_strict`, `_normalize_for_hash`, `_root_config_dump`, `_resolve_config_path`, `_candidate_dump_m_from_config`, `_distributed_cli_defaults`, and `_parse_physical_ids` without changing parsing, hashing, default extraction, or fallback behavior.
- Updated `src/pipeline/distributed/controller.py` to import the moved helpers from `controller_config.py`, preserving `pipeline.distributed.controller.load_and_hash_config` and compatibility-sensitive private helper access through the old module.
- Removed the now-unused `hashlib`, `json`, and `yaml` imports from `controller.py`; those dependencies now live with the config helper implementation.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_config.py -q` -> `38 passed in 0.49s`.
- Lints/diagnostics reported no errors for `controller.py` or `controller_config.py`.

## Phase 4 - Extract Preflight Checks

- [x] Create `src/pipeline/distributed/controller_preflight.py`.
- [x] Move `REQUIRED_NATIVE_EXTENSIONS`, `run_preflight_checks`, `native_extension_availability`, `_estimate_preflight_disk_bytes`, `_check_output_writable`, and `_visible_cuda_device_count`.
- [x] Preserve output-root collision handling, shard-table construction checks, device visibility checks, native extension gates, and disk-space estimate behavior.
- [x] Re-export `run_preflight_checks` and `native_extension_availability` from `controller.py`.
- [x] Verify preflight behavior with `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_shard_table.py -q`.

### Phase 4 Notes

- Added `src/pipeline/distributed/controller_preflight.py` for native-extension requirements, preflight validation, conservative disk estimates, output-writability checks, and CUDA visibility probing.
- Moved `REQUIRED_NATIVE_EXTENSIONS`, `run_preflight_checks`, `native_extension_availability`, `_estimate_preflight_disk_bytes`, `_check_output_writable`, and `_visible_cuda_device_count` into the preflight module without changing output-root collision handling, shard-table construction checks, device assignment validation, native extension gates, disk-space estimates, or preflight report fields.
- Updated `src/pipeline/distributed/controller.py` to expose the moved names from the old module. `run_preflight_checks` remains as a compatibility wrapper that syncs old-module monkeypatches for `_check_output_writable` and `_visible_cuda_device_count` before delegating to `controller_preflight.run_preflight_checks`.
- Kept `importlib` and `shutil` available on `pipeline.distributed.controller` because existing tests monkeypatch `controller.importlib.util.find_spec` and `controller.shutil.disk_usage`; those module monkeypatches continue to affect the extracted preflight implementation.
- Initial verification exposed the old-module monkeypatch compatibility gap; after adding the wrapper, verification on 2026-05-24 passed: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_shard_table.py -q` -> `42 passed in 0.62s`.
- Lints/diagnostics reported no errors for `controller.py` or `controller_preflight.py`.

## Phase 5 - Extract Worker Commands And Launching

- [x] Create `src/pipeline/distributed/controller_commands.py`.
- [x] Move `build_worker_commands`, `launch_worker_processes`, and `_worker_pythonpath`.
- [x] Preserve command shape: `python -m pipeline.distributed.worker --manifest ... --phase ... --worker-id ...`.
- [x] Preserve per-worker environment behavior, including `CUDA_VISIBLE_DEVICES` and `PYTHONPATH`.
- [x] Re-export `WorkerCommand`, `build_worker_commands`, and `launch_worker_processes` from `controller.py`.
- [x] Verify command behavior with `python -m pytest tests/pipeline/test_distributed_controller.py -q`.

### Phase 5 Notes

- Added `src/pipeline/distributed/controller_commands.py` for worker command construction, subprocess launching, and controller-specific `PYTHONPATH` construction.
- Moved `build_worker_commands`, `launch_worker_processes`, and `_worker_pythonpath` while preserving the worker command shape: `python -m pipeline.distributed.worker --manifest ... --phase pass1|pass2|discovery --worker-id ...`.
- Preserved per-worker environment construction through `worker_environment`, including `CUDA_VISIBLE_DEVICES`, and kept prepending `<project-root>/src` to `PYTHONPATH`.
- Updated `src/pipeline/distributed/controller.py` to expose compatibility wrappers for `build_worker_commands` and `launch_worker_processes`. The launch wrapper syncs `controller.subprocess` into the extracted module so existing monkeypatches of `pipeline.distributed.controller.subprocess.Popen` still affect launches.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_controller.py -q` -> `26 passed in 0.52s`.
- Lints/diagnostics reported no errors for `controller.py` or `controller_commands.py`.

## Phase 6 - Extract Report Builders And Dry-Run Formatting

- [x] Create `src/pipeline/distributed/controller_reports.py`.
- [x] Move `build_discovery_dry_run_estimate`, `build_local_compatibility_report`, `build_h100_exact_mode_report`, `_format_discovery_dry_run_estimate`, `_format_local_compatibility_report`, and `_format_h100_exact_mode_report`.
- [x] Create `src/pipeline/distributed/controller_dry_run.py`.
- [x] Move `format_dry_run`.
- [x] Preserve dry-run text content and ordering unless tests explicitly permit updates.
- [x] Re-export public report builders and `format_dry_run` from `controller.py`.
- [x] Verify dry-run/report behavior with `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_operating_modes.py -q`.

### Phase 6 Notes

- Added `src/pipeline/distributed/controller_reports.py` for discovery dry-run estimates, local compatibility reports, H100 exact-mode reports, and their text-formatting helpers.
- Added `src/pipeline/distributed/controller_dry_run.py` for `format_dry_run`, including base manifest/preflight lines, worker command rendering, optional pass-2 benchmark rendering, and report-section ordering.
- Updated `src/pipeline/distributed/controller.py` to import and re-export public report builders, private report formatters, and `format_dry_run`, preserving old imports from `pipeline.distributed.controller`.
- Kept dry-run text content and ordering unchanged: base run metadata, workers, optional pass-2 benchmark, discovery estimate, local compatibility, and H100 exact-mode sections.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_operating_modes.py -q` -> `33 passed in 0.54s`.
- Lints/diagnostics reported no errors for `controller.py`, `controller_reports.py`, or `controller_dry_run.py`.

## Phase 7 - Extract Resume And Stage Helpers

- [x] Create `src/pipeline/distributed/controller_resume.py`.
- [x] Move `classify_resume_workers` and `_validate_marker_identity`.
- [x] Preserve pending/completed/failed/stale classification behavior exactly.
- [x] Create `src/pipeline/distributed/controller_stages.py`.
- [x] Move `run_parts_1_to_3`.
- [x] Keep `run_parts_1_to_3` as a mechanical extraction only; do not expand it into a full pipeline runner in this refactor.
- [x] Re-export `classify_resume_workers` and `run_parts_1_to_3` from `controller.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_negative_context_stage.py -q`.

### Phase 7 Notes

- Added `src/pipeline/distributed/controller_resume.py` for `classify_resume_workers` and `_validate_marker_identity`, preserving config-hash stale classification, failed/completed marker priority, pending fallback, and stale-on-marker-error behavior.
- Added `src/pipeline/distributed/controller_stages.py` for `run_parts_1_to_3`, keeping it as a mechanical integrated helper for pass-1 workers, pass-1 merge, and standalone negative-context execution.
- Updated `src/pipeline/distributed/controller.py` to import and re-export `classify_resume_workers`, `_validate_marker_identity`, and `run_parts_1_to_3`, preserving old imports from `pipeline.distributed.controller`.
- Kept `run_parts_1_to_3` runner injection semantics unchanged for `worker_runner`, `merge_runner`, and `neg_ctx_runner`; default imports remain lazy inside the extracted helper.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_negative_context_stage.py -q` -> `42 passed in 0.74s`.
- Lints/diagnostics reported no errors for `controller.py`, `controller_resume.py`, or `controller_stages.py`.

## Phase 8 - Extract Core Planning

- [x] Create `src/pipeline/distributed/controller_planning.py`.
- [x] Move `plan_distributed_run`.
- [x] Keep manifest construction, run ID generation, output-root selection, layout creation, work assignment construction, device assignment construction, and report construction behavior unchanged.
- [x] Update imports to use extracted config, preflight, command, and report modules.
- [x] Re-export `plan_distributed_run` from `controller.py`.
- [x] Verify planning behavior with `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_assignments.py -q`.

### Phase 8 Notes

- Added `src/pipeline/distributed/controller_planning.py` for the core `plan_distributed_run` implementation.
- Moved manifest construction, run ID generation, output-root selection, layout creation, manifest saving, work assignment construction, device assignment construction, worker command generation, report construction, and dry-run text construction without changing behavior.
- Updated the extracted planner to depend on the previously extracted config, preflight, command, report, and dry-run modules.
- Kept `src/pipeline/distributed/controller.py` as the compatibility surface by adding a thin `plan_distributed_run` wrapper. The wrapper syncs old-module hooks for `run_preflight_checks`, `_visible_cuda_device_count`, and `build_worker_commands` into `controller_planning.py` before delegating, preserving existing monkeypatch-based tests and old caller behavior.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_assignments.py -q` -> `55 passed in 0.57s`.
- Lints/diagnostics reported no errors for `controller.py` or `controller_planning.py`.

## Phase 9 - Extract CLI Entrypoint

- [x] Create `src/pipeline/distributed/controller_cli.py`.
- [x] Move `build_arg_parser`, `plan_distributed_run_from_args`, and `main`.
- [x] Keep `python -m pipeline.distributed.controller` working by delegating from `controller.py`.
- [x] Preserve CLI flags, defaults, phase-specific command regeneration, dry-run printing, and `--launch` behavior.
- [x] Re-export `build_arg_parser` and `plan_distributed_run_from_args` from `controller.py`.
- [x] Verify CLI behavior with `python -m pytest tests/pipeline/test_distributed_controller.py -q`.

### Phase 9 Notes

- Added `src/pipeline/distributed/controller_cli.py` for `build_arg_parser`, `plan_distributed_run_from_args`, and `main`.
- Moved CLI flag definitions, distributed config default extraction, physical device parsing, phase-specific worker-command regeneration, dry-run printing, and optional `--launch` behavior without changing command-line surface.
- Kept `src/pipeline/distributed/controller.py` as the module-execution and compatibility surface. `build_arg_parser`, `plan_distributed_run_from_args`, and `main` now delegate to `controller_cli.py`; `if __name__ == "__main__": main()` remains in `controller.py`, so `python -m pipeline.distributed.controller` still works.
- Added a CLI compatibility sync in `controller.py` so old-module hooks for `plan_distributed_run`, `build_worker_commands`, `launch_worker_processes`, and `format_dry_run` are copied into `controller_cli.py` before delegating.
- Verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_controller.py -q` -> `26 passed in 0.51s`.
- Lints/diagnostics reported no errors for `controller.py` or `controller_cli.py`.

## Phase 10 - Reduce `controller.py` To A Compatibility Facade

- [x] Replace `src/pipeline/distributed/controller.py` with a small facade that imports public names from the extracted controller modules.
- [x] Keep a direct `main()` delegation so module execution remains stable.
- [x] Define `__all__` if helpful for the stable public controller API.
- [x] Confirm all existing imports from `pipeline.distributed.controller` still resolve without caller changes.
- [x] Avoid broad test rewrites; only adjust tests if they intentionally inspect private helper ownership.
- [x] Confirm `src/pipeline/distributed/controller.py` is reduced to a small facade, ideally under 150 LOC.

### Phase 10 Notes

- Reduced `src/pipeline/distributed/controller.py` to a 147-line compatibility facade that imports the stable controller API from the extracted contract, config, preflight, command, report, dry-run, resume, stage, planning, and CLI modules.
- Kept direct module execution stable through `if __name__ == "__main__": main()`, with `main` delegating to `controller_cli.py`.
- Added explicit `__all__` for the stable public controller API re-exported by `pipeline.distributed.controller` and `pipeline.distributed`.
- Preserved compatibility-sensitive old-module globals and hooks used by tests or old callers, including `RunMode`, `DistributedRunManifest`, `importlib`, `shutil`, `subprocess`, `_visible_cuda_device_count`, `_check_output_writable`, `_worker_pythonpath`, and the private config/report helper imports.
- Import smoke check on 2026-05-24 with `PYTHONPATH=src` confirmed every name in `controller.__all__` resolves and the old direct controller imports still work.
- Focused verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_controller.py -q` -> `26 passed in 0.48s`.
- Lints/diagnostics reported no errors for `controller.py`.

## Phase 11 - Testing And Verification

- [x] Run focused controller tests: `python -m pytest tests/pipeline/test_distributed_controller.py -q`.
- [x] Run controller-adjacent tests: `python -m pytest tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_assignments.py -q`.
- [x] Run worker and stage integration-adjacent tests: `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_negative_context_stage.py tests/pipeline/test_distributed_operating_modes.py -q`.
- [x] Run lints/diagnostics on all newly created and edited files.
- [x] Confirm no manifest schema, config hash, dry-run text, worker command shape, preflight behavior, resume classification, or public import path changed.
- [x] Confirm no generated artifacts, native binaries, or unrelated local outputs were touched by the refactor.

### Phase 11 Notes

- Focused controller verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_controller.py -q` -> `26 passed in 0.47s`.
- Controller-adjacent verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_assignments.py -q` -> `59 passed in 0.25s`.
- Worker and stage integration-adjacent verification on 2026-05-24: `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_negative_context_stage.py tests/pipeline/test_distributed_operating_modes.py -q` -> `50 passed in 0.66s`.
- Lints/diagnostics reported no errors for `controller.py`, all extracted `controller_*.py` modules, or this plan file.
- Import smoke check with `PYTHONPATH=src` confirmed every name in `pipeline.distributed.controller.__all__` resolves through the compatibility facade.
- The focused and adjacent suites above cover manifest schema, normalized config hash behavior, dry-run text sections, worker command shape, preflight gates, resume classification, CLI/default behavior, and public imports from both `pipeline.distributed.controller` and `pipeline.distributed`.
- `git status --short -- "src/pipeline/distributed/controller.py" "src/pipeline/distributed/controller_*.py" "agent-planning/refactoring/distributed-controller-refactor.md"` shows only this refactor's controller source files and planning file; no generated artifacts, native binaries, output runs, or unrelated local outputs were touched by this refactor.

---

## Open Questions

- Should this eventually become a true `src/pipeline/distributed/controller/` package after the compatibility facade has stabilized?
- Should `run_parts_1_to_3` stay under controller ownership, or move to a future orchestration module once a full distributed runner exists?
- Should dry-run formatting remain text-first, or eventually produce structured report objects that render to text separately?
- Should private config helpers remain importable through `controller.py`, or should only documented public helpers be re-exported?

## Risks / Assumptions

- `plan_distributed_run` is the integration hotspot and should be extracted late after config, preflight, commands, and reports are stable.
- Dry-run output is user-facing and test-covered; formatting drift should be avoided unless intentionally updated.
- Worker command shape is part of the scheduler/manual-launch contract and must remain stable.
- This refactor should not change manifest planning, output layout creation, config hashing, preflight gates, or launch behavior.
