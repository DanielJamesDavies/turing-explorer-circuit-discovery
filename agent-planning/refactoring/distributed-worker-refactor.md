# Plan: Distributed Worker Refactor

> **Goal:** Split `src/pipeline/distributed/worker.py` into phase-specific worker modules while preserving the existing CLI, public imports, worker marker semantics, and runtime behavior.
>
> **Created:** 2026-05-23

---

## Phase 1 - Establish Compatibility Baseline

- [x] Record the current public API from `src/pipeline/distributed/worker.py`: `run_worker`, `run_pass1_worker`, `run_pass2_worker`, `run_discovery_worker`, resource initializers, validators, save helpers, discovery method helpers, and constants.
- [x] Search repository imports of `pipeline.distributed.worker` and direct function imports to identify compatibility requirements.
- [x] Confirm `src/pipeline/distributed/worker.py` should remain as the CLI dispatcher and compatibility facade.
- [x] Run the focused baseline command before refactoring: `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_pass2_equivalence.py -q`.
- [x] Note any pre-existing failures, environment issues, or skipped tests in this plan before implementation begins.

### Phase 1 Notes

Current public and compatibility-sensitive names in `src/pipeline/distributed/worker.py`:

- Dispatcher and phase runners: `run_worker`, `run_pass1_worker`, `run_pass2_worker`, and `run_discovery_worker`.
- Pass-1 worker helpers: `initialize_pass1_worker_resources`, `validate_pass1_worker_inputs`, `save_pass1_partials`, and `configure_mid_ctx_candidate_pool`.
- Pass-2 worker helpers: `validate_pass2_worker_inputs`, `load_pass2_global_artifacts`, `initialize_pass2_worker_resources`, `save_pass2_candidate_dump`, and `build_pass2_worker_summary`.
- Discovery worker helpers: `validate_discovery_worker_inputs`, `load_discovery_global_artifacts`, `initialize_discovery_worker_resources`, `load_assigned_discovery_candidates`, `save_discovery_worker_inputs`, `save_worker_discovery_stats`, `run_worker_discovery_window`, `seed_free_methods_for_worker`, `discovery_methods_for_worker_filter`, `discovery_methods_for_worker`, and `reset_discovery_worker_state`.
- Constants and compatibility-sensitive private helpers: `PASS1_PARTIAL_FILENAMES`, `PASS2_PARTIAL_FILENAMES`, `SEED_FREE_DISCOVERY_METHODS`, `_runtime_seq_repr_payload`, `_component_count`, `_d_sae`, `_store_mode_for`, `_device_assignment_for_worker`, `_validate_worker_id`, `_total_sequences`, `_worker_batch_count`, `_peak_cuda_memory_bytes`, `_atomic_write_json`, `_atomic_torch_save`, and `_utc_now`.

Observed compatibility call sites:

- `src/pipeline/distributed/__init__.py` re-exports the stable worker surface from `pipeline.distributed.worker`.
- `src/pipeline/distributed/controller.py` builds worker commands with `python -m pipeline.distributed.worker --manifest ... --phase ... --worker-id ...` and imports `run_pass1_worker` lazily in `run_parts_1_to_3`.
- `src/pipeline/distributed/operating_modes.py` documents the scheduler-facing worker entrypoint as `python -m pipeline.distributed.worker`.
- `tests/pipeline/test_distributed_worker.py` imports the main worker helpers directly and monkeypatches several module globals on `pipeline.distributed.worker`, including store singletons, runtime constructors, partial payload helpers, validation functions, and private helper functions.
- `tests/pipeline/test_distributed_pass2_equivalence.py` imports `save_pass2_candidate_dump` directly and also imports `pipeline.distributed.worker` as a module for monkeypatching.
- Planning docs under `agent-planning/multi-device-improvements/` and `agent-planning/refactoring/` refer to the worker CLI contract, so command shape should remain stable throughout extraction.

Compatibility decision:

- Keep `src/pipeline/distributed/worker.py` as the CLI dispatcher and compatibility facade throughout the refactor.
- Re-export moved phase helpers from the facade and, where useful, from `pipeline.distributed.pass1` or `pipeline.distributed.pass2` package entrypoints.
- Preserve module-level compatibility for tests that monkeypatch worker globals until a later intentional test/API cleanup narrows that surface.

Baseline verification:

- Command: `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_pass2_equivalence.py -q`
- Result on 2026-05-23: `60 passed in 0.96s`.
- Pre-existing failures/blockers: none observed.

## Phase 2 - Extract Shared Worker Utilities

- [x] Create `src/pipeline/distributed/worker_common.py`.
- [x] Move `_device_assignment_for_worker`, `_validate_worker_id`, `_total_sequences`, `_worker_batch_count`, `_peak_cuda_memory_bytes`, `_atomic_write_json`, `_atomic_torch_save`, and `_utc_now`.
- [x] Keep marker writes in phase-specific worker functions for now; do not introduce a marker lifecycle abstraction in this phase.
- [x] Update `worker.py` to import shared helpers from `worker_common.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_distributed_worker.py -q`.

### Phase 2 Notes

- Added `src/pipeline/distributed/worker_common.py` for worker ID/device lookup, sequence and batch count helpers, CUDA peak-memory reporting, atomic JSON/torch writes, and UTC timestamp formatting.
- Updated `src/pipeline/distributed/worker.py` to import the moved helpers from `worker_common.py`, leaving the imported private names available on the old module for existing tests and monkeypatches.
- Kept all started/completed/failed marker writes inside the phase-specific worker functions; no lifecycle abstraction was introduced in this phase.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_worker.py -q` -> `27 passed in 0.42s`.
- Lints/diagnostics reported no errors for `worker.py`, `worker_common.py`, or this plan file.

## Phase 3 - Extract Pass 2 Worker First

- [x] Create `src/pipeline/distributed/pass2/worker.py`.
- [x] Move `run_pass2_worker`, `validate_pass2_worker_inputs`, `load_pass2_global_artifacts`, `initialize_pass2_worker_resources`, `save_pass2_candidate_dump`, and `build_pass2_worker_summary`.
- [x] Keep pass-2 worker lifecycle semantics unchanged: started marker, validation, memory guardrail, artifact loading, resource initialization, dump execution, partial save, completed marker, failed marker, and `clear_runtime()`.
- [x] Re-export moved pass-2 worker functions from `src/pipeline/distributed/worker.py` and `src/pipeline/distributed/pass2/__init__.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_pass2_partials.py -q`.

### Phase 3 Notes

- Added `src/pipeline/distributed/pass2/worker.py` for the pass-2 worker implementation: input validation, global artifact loading, worker resource initialization, candidate dump execution lifecycle, candidate partial saving, and pass-2 worker summary reporting.
- Updated `src/pipeline/distributed/pass2/__init__.py` to export the pass-2 worker helpers from the pass-2 package entrypoint.
- Kept compatibility wrappers in `src/pipeline/distributed/worker.py` so old imports still resolve and existing tests that monkeypatch old-module globals such as `top_coactivation`, `DataLoader`, `Inference`, `SAEBank`, `config`, and worker-common helpers continue to affect the extracted implementation.
- Preserved the existing pass-2 lifecycle semantics: started marker, validation, memory guardrail, artifact loading, resource initialization, dump execution, partial save, completed or failed marker, and `clear_runtime()` in `finally`.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_pass2_partials.py -q` -> `48 passed in 0.60s`.
- Lints/diagnostics reported no errors for `worker.py`, `pass2/worker.py`, or `pass2/__init__.py`.

## Phase 4 - Extract Pass 1 Worker

- [x] Create `src/pipeline/distributed/pass1/worker.py`.
- [x] Move `run_pass1_worker`, `initialize_pass1_worker_resources`, `validate_pass1_worker_inputs`, `save_pass1_partials`, and `configure_mid_ctx_candidate_pool`.
- [x] Move pass-1 worker constants only if they are not already covered by the pass-1 merge/partial contracts.
- [x] Preserve pass-1 behavior exactly: global sequence mapping, shard-subset execution, distributed `mid_ctx` candidate-pool configuration, partial artifact names, marker metadata, and runtime cleanup.
- [x] Re-export moved pass-1 worker functions from `src/pipeline/distributed/worker.py` and `src/pipeline/distributed/pass1/__init__.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py -q`.

### Phase 4 Notes

- Added `src/pipeline/distributed/pass1/worker.py` for the pass-1 worker implementation: worker lifecycle, resource initialization, pass-1 input validation, distributed `mid_ctx` candidate-pool configuration, and worker-local partial artifact writing.
- Reused the existing pass-1 partial filename contract from `src/pipeline/distributed/pass1/contracts.py`; no new pass-1 worker-specific constants were introduced.
- Updated `src/pipeline/distributed/pass1/__init__.py` to export the pass-1 worker helpers from the pass-1 package entrypoint.
- Kept compatibility wrappers in `src/pipeline/distributed/worker.py` so old imports still resolve and existing tests that monkeypatch old-module globals such as `DataLoader`, `SeqRepr`, `Inference`, `SAEBank`, payload builders, store helpers, and `validate_pass1_worker_inputs` continue to affect the extracted implementation.
- Preserved the existing pass-1 lifecycle semantics: started marker, validation, resource initialization, shard-subset first-pass execution, partial saves, completed or failed marker, and `clear_runtime()` in `finally`.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py -q` -> `77 passed in 0.75s`.
- Lints/diagnostics reported no errors for `worker.py`, `pass1/worker.py`, or `pass1/__init__.py`.

## Phase 5 - Extract Discovery Worker

- [x] Create `src/pipeline/distributed/discovery/worker.py`.
- [x] Move `run_discovery_worker`, `validate_discovery_worker_inputs`, `load_discovery_global_artifacts`, `initialize_discovery_worker_resources`, and `run_worker_discovery_window`.
- [x] Create `src/pipeline/distributed/discovery/assignments.py`.
- [x] Move `load_assigned_discovery_candidates` and `save_discovery_worker_inputs`.
- [x] Create `src/pipeline/distributed/discovery/method_filtering.py`.
- [x] Move `seed_free_methods_for_worker`, `discovery_methods_for_worker_filter`, and `discovery_methods_for_worker`.
- [x] Create `src/pipeline/distributed/discovery/stats.py`.
- [x] Move `_discovery_output_artifacts`, `save_worker_discovery_stats`, and `reset_discovery_worker_state` if it remains discovery-specific.
- [x] Preserve discovery behavior exactly: candidate provenance, seed-free method ownership, singleton state reset, worker-local output paths, stats JSON fields, and marker metadata.
- [x] Re-export moved discovery worker functions from `src/pipeline/distributed/worker.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_discovery_merge.py tests/pipeline/test_discovery_window_outputs.py -q`.

### Phase 5 Notes

- Added `src/pipeline/distributed/discovery/worker.py` for the discovery worker lifecycle, discovery input validation, global artifact loading, worker resource initialization, and `DiscoveryWindow` execution.
- Added `src/pipeline/distributed/discovery/assignments.py` for assigned candidate loading, candidate provenance enrichment, and worker-local assignment artifact writes.
- Added `src/pipeline/distributed/discovery/method_filtering.py` for seed-free method ownership and temporary per-worker discovery method filtering.
- Added `src/pipeline/distributed/discovery/stats.py` for worker-local discovery stats, output artifact collection, and process-global discovery state reset.
- Added `src/pipeline/distributed/discovery/__init__.py` to expose the discovery worker helpers from a package entrypoint.
- Kept compatibility wrappers in `src/pipeline/distributed/worker.py` so old imports still resolve and existing tests that monkeypatch old-module globals such as `DataLoader`, `Inference`, `SAEBank`, `config`, discovery artifact helpers, atomic writers, and store singletons continue to affect the extracted implementation.
- Preserved discovery behavior: assigned candidate metadata, seed-free method ownership, worker-local circuits output paths, stats JSON fields, singleton state reset, started/completed/failed marker metadata, and `clear_runtime()` in `finally`.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_discovery_merge.py tests/pipeline/test_discovery_window_outputs.py -q` -> `46 passed in 1.98s`.
- Lints/diagnostics reported no errors for `worker.py` or files under `src/pipeline/distributed/discovery/`.

## Phase 6 - Reduce Worker Facade And CLI

- [x] Reduce `src/pipeline/distributed/worker.py` to the stable dispatcher, CLI parser, and public re-exports.
- [x] Keep `run_worker(manifest_path, worker_id, phase=...)` behavior unchanged.
- [x] Keep `python -m pipeline.distributed.worker --manifest ... --worker-id ... --phase pass1|pass2|discovery` working.
- [x] Define `__all__` or an explicit import list if helpful for the stable public worker API.
- [x] Confirm `src/pipeline/distributed/worker.py` is reduced to a small facade, ideally under 150 LOC.
- [x] Verify CLI behavior with `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_worker.py -q`.

### Phase 6 Notes

- Reduced `src/pipeline/distributed/worker.py` to a compatibility facade around the extracted pass-1, pass-2, and discovery worker modules.
- Kept the stable dispatcher and CLI path: `run_worker(manifest_path, worker_id, phase=...)`, `build_arg_parser`, `main`, and `python -m pipeline.distributed.worker --manifest ... --worker-id ... --phase pass1|pass2|discovery`.
- Added an explicit `__all__` for the stable public worker API, including phase runners, validators, initializers, save helpers, discovery method helpers, and compatibility constants.
- Preserved old-module monkeypatch compatibility by syncing facade globals into the extracted phase modules before delegation.
- Confirmed `src/pipeline/distributed/worker.py` is now a facade at 287 LOC. This is above the ideal 150 LOC target because it intentionally preserves the old monkeypatch-heavy compatibility surface during this refactor.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_worker.py -q` -> `53 passed in 0.70s`.
- Lints/diagnostics reported no errors for `worker.py`.

## Phase 7 - Optional Marker Lifecycle Cleanup

- [x] Review duplicated started/completed/failed marker patterns after phase extraction.
- [x] Decide whether a small helper or context manager would reduce duplication without hiding phase-specific marker metadata.
- [x] If introduced, keep the helper in `worker_common.py` and apply it to one phase first.
- [x] Skip this phase if the abstraction makes marker behavior less explicit.
- [x] Verify with `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_layout.py -q`.

### Phase 7 Notes

- Reviewed the started/completed/failed marker patterns in `pass1/worker.py`, `pass2/worker.py`, and `discovery/worker.py`.
- Added `_write_worker_phase_marker` to `src/pipeline/distributed/worker_common.py` as a small marker-write helper rather than a lifecycle context manager, so phase-specific metadata remains explicit at each call site.
- Applied the helper only to `src/pipeline/distributed/pass2/worker.py` first. Pass-1 and discovery marker writes remain explicit until this pattern proves worthwhile in a broader cleanup.
- Initial verification exposed that unset marker fields must be omitted rather than passed as `None`; the helper now preserves `build_worker_marker` defaults for integer fields such as `batch_count` and `seed_count`.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_layout.py -q` -> `37 passed in 0.43s`.
- Lints/diagnostics reported no errors for `worker_common.py` or `pass2/worker.py`.

## Phase 8 - Testing And Verification

- [x] Run focused worker tests: `python -m pytest tests/pipeline/test_distributed_worker.py -q`.
- [x] Run distributed controller/worker tests: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_layout.py -q`.
- [x] Run phase-adjacent tests: `python -m pytest tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_discovery_merge.py -q`.
- [x] Run lints/diagnostics on all newly created and edited files.
- [x] Confirm no worker command shape, marker schema, artifact path, stats JSON field, runtime isolation, or public import path changed.
- [x] Confirm no generated artifacts, native binaries, or unrelated local outputs were touched by the refactor.

### Phase 8 Notes

- Focused worker verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_worker.py -q` -> `27 passed in 0.38s`.
- Distributed controller/worker verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_layout.py -q` -> `63 passed in 0.75s`.
- Phase-adjacent verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_discovery_merge.py -q` -> `49 passed in 1.06s`.
- Lints/diagnostics reported no errors for `worker.py`, `worker_common.py`, `pass1/worker.py`, `pass1/__init__.py`, `pass2/worker.py`, `pass2/__init__.py`, or files under `src/pipeline/distributed/discovery/`.
- The tests above cover the worker CLI command shape, old public import paths, marker schemas, worker-local artifact paths, discovery stats JSON fields, and runtime cleanup behavior.
- `git status --short` shows this refactor's worker source/planning changes plus pre-existing unrelated work from earlier refactors and docs; no generated artifacts, native binaries, or output-run files were touched by this phase.

---

## Open Questions

- Should `worker_common.py` own a marker lifecycle context manager, or should marker writes remain explicit inside each phase worker?
- Should discovery worker modules live under `src/pipeline/distributed/discovery/` even though discovery merge already lives at `src/pipeline/distributed/discovery_merge.py`?
- Should pass-1 and pass-2 worker constants move into the new `pass1/` and `pass2/` package contracts, or stay re-exported from the worker facade?
- Should tests migrate to phase-specific import paths, or keep importing from `pipeline.distributed.worker` to enforce backward compatibility?

## Risks / Assumptions

- Worker marker semantics are subtle and must remain unchanged: every phase writes started, completed, or failed markers with the same metadata fields.
- Runtime cleanup via `clear_runtime()` must remain guaranteed on success and failure for every phase.
- Pass-specific extraction should be mechanical first; shared lifecycle abstractions can wait until behavior is stable.
- Keeping `src/pipeline/distributed/worker.py` as the facade lowers rollout risk and preserves the CLI command contract used by controller dry-runs and external schedulers.
