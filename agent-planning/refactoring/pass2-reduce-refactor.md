# Plan: Pass 2 Reduce Refactor

> **Goal:** Split `src/pipeline/distributed/pass2_reduce.py` into a focused `src/pipeline/distributed/pass2/` package while preserving the existing public API, reducer behavior, and test coverage.
>
> **Created:** 2026-05-23

---

## Phase 1 — Establish Compatibility Baseline

- [x] Record the current exported names from `src/pipeline/distributed/pass2_reduce.py` and `src/pipeline/distributed/__init__.py` that tests or callers rely on.
- [x] Identify import cycles that could appear when moving dataclasses, simple reducer logic, MapReduce logic, and report helpers into separate files.
- [x] Run the focused baseline command before refactoring: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q`.
- [x] Save any pre-existing test failures or environment blockers in this plan before making code changes.

### Phase 1 Notes

Current public names used directly from `pipeline.distributed.pass2_reduce` or re-exported through `pipeline.distributed.__init__`:

- Contract/result dataclasses: `CandidateDumpReducerEntry`, `CandidateDumpReducerInputs`, `GlobalTopCtxTargetMapping`, `SimpleExactCandidateDump`, `PmiReduceInputs`, `CandidatePreAggregationReducerEntry`, `CandidatePreAggregationReducerInputs`, `TargetRange`, `MapReduceTargetShardResult`, `MapReduceShardMemoryEstimate`, `Pass2ReduceSchedulerConfig`, `MapReduceTargetShardArtifact`, `MapReduceReduceResult`, and `SimpleExactReduceResult`.
- CLI/stage entrypoints: `build_arg_parser`, `main`, and `run_simple_exact_reduce_stage`.
- Reducer input/global mapping helpers: `load_candidate_dump_reducer_inputs`, `validate_candidate_dump_reducer_inputs`, `load_candidate_preaggregation_reducer_inputs`, `load_global_top_ctx_target_mapping`, `load_global_active_count`, `validate_global_active_count`, `build_global_top_ctx_target_mapping`, `validate_candidate_dump_sequence_coverage`, and `validate_candidate_preaggregation_reducer_inputs`.
- Simple exact reduce helpers: `build_simple_exact_candidate_dump`, `attach_simple_exact_dump_to_store`, `reduce_simple_exact_candidate_dump`, `run_simple_exact_reduce_and_write`, `build_simple_exact_reduce_report`, `validate_saved_top_coactivation_artifact`, `validate_pmi_reduce_inputs`, and `validate_top_coactivation_reduce_output`.
- MapReduce helpers: `partition_target_ranges`, `shard_preaggregation_by_target_range`, `reduce_mapreduce_target_range`, `sorted_coo_preaggregation_payload`, `build_mapreduce_storage_metadata`, `save_mapreduce_partial_sum_shard`, `load_mapreduce_partial_sum_shard`, `validate_mapreduce_partial_sum_shard`, `load_mapreduce_reducer_shards`, `estimate_mapreduce_shard_tensor_bytes`, `estimate_mapreduce_reducer_input_bytes`, `checksum_coo_payload`, `validate_pass2_reduce_scheduler_config`, `mapreduce_target_shard_path`, `cleanup_mapreduce_target_shards`, `save_mapreduce_target_shard_result`, `load_mapreduce_target_shard_result`, `validate_mapreduce_target_shard_artifact`, `validate_mapreduce_target_shard_result`, `run_mapreduce_reduce_and_write`, `stitch_mapreduce_target_shards`, `apply_pmi_postprocess_to_topk`, and `compute_total_tokens_per_target`.
- Reporting helpers: `build_pass2_reduce_manifest_metrics` and `format_pass2_reduce_benchmark_report`.

Observed compatibility call sites:

- `tests/pipeline/test_distributed_pass2_reduce.py` imports many public helpers directly from `pipeline.distributed.pass2_reduce`.
- `tests/pipeline/test_distributed_controller.py` imports `pass2_reduce` as a module through `pipeline.distributed`.
- `src/pipeline/distributed/__init__.py` re-exports the pass-2 reduce public surface, so the facade must keep those names available.
- Runtime controller/preflight code refers to the `pass2_reduce` part name for native-extension checks and reporting, but does not directly call reducer functions.

Import-cycle risks to avoid during extraction:

- Keep `contracts.py` dependency-light. Dataclasses should not import simple, MapReduce, reports, or CLI modules.
- `inputs.py` can depend on `contracts.py` and `pass2_partials.py`, but simple and MapReduce modules should import input helpers rather than the reverse.
- `simple.py` and `mapreduce.py` should share report helpers through `reports.py`; `reports.py` should not import execution modules.
- `mapreduce.py` should depend on `mapreduce_io.py`, but `mapreduce_io.py` should stay limited to contracts, tensor/JSON persistence, checksums, and validation.
- `cli.py` should import execution entrypoints late and should be invoked from `pass2_reduce.py`, not imported by lower-level modules.

Baseline verification:

- Command: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q`
- Result on 2026-05-23: `84 passed in 0.85s`.
- Pre-existing failures/blockers: none observed.

## Phase 2 — Extract Shared Contracts

- [x] Create `src/pipeline/distributed/pass2/__init__.py` as the new package entrypoint for pass-2 reduce helpers.
- [x] Create `src/pipeline/distributed/pass2/contracts.py` for reducer dataclasses and shared type contracts.
- [x] Move `CandidateDumpReducerEntry`, `CandidateDumpReducerInputs`, `GlobalTopCtxTargetMapping`, `SimpleExactCandidateDump`, `PmiReduceInputs`, `CandidatePreAggregationReducerEntry`, `CandidatePreAggregationReducerInputs`, `TargetRange`, `MapReduceTargetShardResult`, `MapReduceShardMemoryEstimate`, `Pass2ReduceSchedulerConfig`, `MapReduceTargetShardArtifact`, `MapReduceReduceResult`, and `SimpleExactReduceResult`.
- [x] Update `pass2_reduce.py` to import these contracts and re-export them for backward compatibility.
- [x] Verify no behavior changed by running `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py -q`.

### Phase 2 Notes

- Added `src/pipeline/distributed/pass2/contracts.py` with the shared reducer dataclasses and a narrow `__all__`.
- Added `src/pipeline/distributed/pass2/__init__.py` to expose the same contract names from the new package entrypoint.
- Updated `src/pipeline/distributed/pass2_reduce.py` to import those names from `pass2.contracts`; existing callers can still import the dataclasses from `pipeline.distributed.pass2_reduce`.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py -q` -> `54 passed in 0.49s`.

## Phase 3 — Extract Reducer Input Loading And Global Mapping

- [x] Create `src/pipeline/distributed/pass2/inputs.py` for reducer input loading, cross-worker validation, global `top_ctx` mapping, active-count loading, and sequence coverage validation.
- [x] Move `load_candidate_dump_reducer_inputs`, `validate_candidate_dump_reducer_inputs`, `load_candidate_preaggregation_reducer_inputs`, `load_global_top_ctx_target_mapping`, `load_global_active_count`, `validate_global_active_count`, `build_global_top_ctx_target_mapping`, `validate_candidate_dump_sequence_coverage`, `validate_candidate_preaggregation_reducer_inputs`, `_normalize_candidate_dump_entries`, `_normalize_preaggregation_entries`, `_required_top_ctx_tensor`, `_build_sequence_to_targets_csr`, and `_build_sid_to_row_tensor`.
- [x] Keep `pass2_reduce.py` as a compatibility facade that imports and re-exports the moved functions.
- [x] Verify with `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py -q`.

### Phase 3 Notes

- Added `src/pipeline/distributed/pass2/inputs.py` for reducer input loading, cross-worker validation, global `top_ctx` replay mapping, global `active_count` loading, sequence coverage validation, and input-only private helpers.
- Updated `src/pipeline/distributed/pass2/__init__.py` to expose the public input helpers from the new package.
- Updated `src/pipeline/distributed/pass2_reduce.py` to import and re-export the moved helpers for backward compatibility.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py -q` -> `68 passed in 0.50s`.

## Phase 4 — Extract Simple Exact Reduce Path

- [x] Create `src/pipeline/distributed/pass2/simple.py` for the simple exact reducer path.
- [x] Move `run_simple_exact_reduce_stage`, `build_simple_exact_candidate_dump`, `attach_simple_exact_dump_to_store`, `reduce_simple_exact_candidate_dump`, `run_simple_exact_reduce_and_write`, `build_simple_exact_reduce_report`, `validate_saved_top_coactivation_artifact`, `validate_pmi_reduce_inputs`, `validate_top_coactivation_reduce_output`, `_validate_simple_dump_reduce_dimensions`, and `_atomic_store_save`.
- [x] Keep imports arranged so the existing `store.top_coactivation` dependency remains local to execution paths where practical.
- [x] Verify simple exact behavior with `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_equivalence.py tests/store/test_top_coactivation_modes.py -q`.

### Phase 4 Notes

- Added `src/pipeline/distributed/pass2/simple.py` for the simple exact reducer path.
- Kept the `store.top_coactivation` import local inside `run_simple_exact_reduce_stage`.
- Updated `src/pipeline/distributed/pass2/__init__.py` and `src/pipeline/distributed/pass2_reduce.py` so existing simple reducer imports continue to resolve.
- Kept simple-path private helpers for JSON writing, byte counting, and memory tracing local to `simple.py` for now; these can be consolidated during the reporting/CLI extraction phase.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_equivalence.py tests/store/test_top_coactivation_modes.py -q` -> `77 passed in 0.80s`.

## Phase 5 — Extract MapReduce Core And Storage

- [x] Create `src/pipeline/distributed/pass2/mapreduce.py` for MapReduce reducer semantics and orchestration.
- [x] Move `partition_target_ranges`, `shard_preaggregation_by_target_range`, `reduce_mapreduce_target_range`, `validate_pass2_reduce_scheduler_config`, `mapreduce_target_shard_path`, `cleanup_mapreduce_target_shards`, `run_mapreduce_reduce_and_write`, `stitch_mapreduce_target_shards`, `apply_pmi_postprocess_to_topk`, `compute_total_tokens_per_target`, `_write_mapreduce_top_coactivation_artifact`, and related MapReduce output validation.
- [x] Create `src/pipeline/distributed/pass2/mapreduce_io.py` for sorted COO shard persistence and checksums.
- [x] Move `sorted_coo_preaggregation_payload`, `build_mapreduce_storage_metadata`, `save_mapreduce_partial_sum_shard`, `load_mapreduce_partial_sum_shard`, `validate_mapreduce_partial_sum_shard`, `load_mapreduce_reducer_shards`, `estimate_mapreduce_shard_tensor_bytes`, `estimate_mapreduce_reducer_input_bytes`, `checksum_coo_payload`, `save_mapreduce_target_shard_result`, `load_mapreduce_target_shard_result`, `validate_mapreduce_target_shard_artifact`, `validate_mapreduce_target_shard_result`, and `_validate_sorted_coo_payload`.
- [x] Verify MapReduce behavior with `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py -q`.

### Phase 5 Notes

- Added `src/pipeline/distributed/pass2/mapreduce.py` for target-range partitioning, reducer orchestration, target-shard stitching, PMI postprocess, scheduler config validation, and canonical MapReduce artifact writing.
- Added `src/pipeline/distributed/pass2/mapreduce_io.py` for sorted COO partial-sum persistence, checksums, memory estimates, reducer shard loading, and target-shard artifact validation.
- Updated `src/pipeline/distributed/pass2/__init__.py` and `src/pipeline/distributed/pass2_reduce.py` so existing MapReduce imports continue to resolve.
- `src/pipeline/distributed/pass2_reduce.py` is now a compatibility facade plus the current CLI/report helpers.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py -q` -> `54 passed in 0.54s`.

## Phase 6 — Extract Reporting And CLI

- [x] Create `src/pipeline/distributed/pass2/reports.py` for reducer report helpers.
- [x] Move `build_pass2_reduce_manifest_metrics`, `format_pass2_reduce_benchmark_report`, `_candidate_dump_entry_tensor_bytes`, `_file_size_or_zero`, `_start_memory_trace`, `_stop_memory_trace`, `_atomic_torch_save`, and `_atomic_write_json` if they remain shared by multiple modules.
- [x] Create `src/pipeline/distributed/pass2/cli.py` for `build_arg_parser` and `main`.
- [x] Keep `python -m pipeline.distributed.pass2_reduce` working by delegating from `pass2_reduce.py` to the new CLI module.
- [x] Verify CLI/parser behavior with `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_pass2_reduce.py -q`.

### Phase 6 Notes

- Added `src/pipeline/distributed/pass2/reports.py` for reducer manifest metrics, benchmark formatting, tensor byte accounting, atomic JSON/torch writes, file-size checks, and memory tracing helpers.
- Added `src/pipeline/distributed/pass2/cli.py` for `build_arg_parser` and `main`.
- Updated `simple.py`, `mapreduce.py`, and `mapreduce_io.py` to use the shared report/IO helpers where applicable.
- Updated `src/pipeline/distributed/pass2_reduce.py` to delegate CLI and reporting to the extracted modules while preserving the old module execution path.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_pass2_reduce.py -q` -> `80 passed in 0.95s`.

## Phase 7 — Preserve Public API And Package Exports

- [x] Update `src/pipeline/distributed/pass2_reduce.py` to be a thin compatibility module that imports public names from the new modules.
- [x] Update `src/pipeline/distributed/pass2/__init__.py` to expose the intended pass-2 reduce API.
- [x] Update `src/pipeline/distributed/__init__.py` only if needed, preserving existing exported names.
- [x] Search for repository imports of `pipeline.distributed.pass2_reduce` and confirm they still resolve.
- [x] Avoid broad test rewrites; update tests only where they intentionally inspect module ownership rather than behavior.
- [x] Verify backward compatibility with the full distributed-focused command documented in Part 8.

### Phase 7 Notes

- Confirmed `src/pipeline/distributed/pass2_reduce.py` is a 141-line compatibility facade with explicit `__all__`.
- Confirmed `src/pipeline/distributed/pass2/__init__.py` exposes the intended pass-2 package API, including contracts, input helpers, simple exact helpers, MapReduce helpers, report helpers, and CLI helpers.
- No update was needed in `src/pipeline/distributed/__init__.py`; it continues importing the preserved names from `pass2_reduce.py`.
- Repository import search found old-facade imports in `src/pipeline/distributed/__init__.py` and `tests/pipeline/test_distributed_pass2_reduce.py`; both resolve through the facade.
- Import smoke check on 2026-05-23: old `pipeline.distributed.pass2_reduce` and new `pipeline.distributed.pass2` both expose the 58 expected public names.
- Distributed import/API verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_operating_modes.py -q` -> `60 passed in 0.82s`.
- Pass-2 contract verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_pass2_benchmark.py tests/store/test_top_coactivation_modes.py -q` -> `104 passed in 1.04s`.
- Lints/diagnostics reported no errors for the edited pass-2 facade/package files.

## Phase 8 — Testing And Verification

- [x] Run focused reducer tests: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py -q`.
- [x] Run pass-2 contract tests: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_pass2_benchmark.py tests/store/test_top_coactivation_modes.py -q`.
- [x] Run distributed import/API tests: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_operating_modes.py -q`.
- [x] Run lints/diagnostics on all newly created and edited files.
- [x] Confirm `src/pipeline/distributed/pass2_reduce.py` is reduced to a small compatibility/CLI facade, ideally under 200 LOC.
- [x] Confirm no generated artifacts, native binaries, or unrelated local outputs were touched by the refactor.

### Phase 8 Notes

- Focused reducer verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py -q` -> `54 passed in 0.50s`.
- Pass-2 contract verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_pass2_benchmark.py tests/store/test_top_coactivation_modes.py -q` -> `104 passed in 1.01s`.
- Distributed import/API verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_operating_modes.py -q` -> `60 passed in 0.83s`.
- Lints/diagnostics reported no errors for `pass2_reduce.py` and all files under `src/pipeline/distributed/pass2/`.
- `src/pipeline/distributed/pass2_reduce.py` is 141 LOC and now serves as a compatibility facade plus module execution delegator.
- `git status --short` shows this refactor's source/planning changes and an unrelated existing change in `agent-planning/multi-device-improvements/part-8-testing-and-benchmarks.md`; no generated artifacts, native binaries, or output-run files were touched by this refactor.

---

## Open Questions

- Should `pass2_reduce.py` remain the CLI module permanently, or should it become only a compatibility facade with CLI in `pass2_reduce_cli.py`?
- Should atomic JSON/torch save helpers live in a shared distributed utility module if other distributed files need them too?
- Should MapReduce report construction stay with MapReduce orchestration or move fully into `src/pipeline/distributed/pass2/reports.py`?
- Should tests gradually import from the new modules, or keep importing from `pass2_reduce.py` to enforce backward compatibility?

## Risks / Assumptions

- This refactor should not change reducer algorithms, artifact schemas, report shapes, or public import names.
- Import cycles are the main implementation risk because the current file relies on shared dataclasses and helpers throughout.
- Keeping `pass2_reduce.py` as a compatibility facade lowers rollout risk and lets downstream code migrate gradually.
- The first pass should optimize for mechanical extraction and test stability, not deeper algorithmic cleanup.
