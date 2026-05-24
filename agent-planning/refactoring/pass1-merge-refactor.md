# Plan: Pass 1 Merge Refactor

> **Goal:** Split `src/pipeline/distributed/pass1_merge.py` into a focused `src/pipeline/distributed/pass1/` package while preserving existing merge behavior, CLI compatibility, and public imports.
>
> **Created:** 2026-05-23

---

## Phase 1 — Establish Compatibility Baseline

- [x] Record the current public names exported from `src/pipeline/distributed/pass1_merge.py` and re-exported from `src/pipeline/distributed/__init__.py`.
- [x] Search repository imports of `pipeline.distributed.pass1_merge` and direct function imports to identify compatibility requirements.
- [x] Run the focused baseline command before refactoring: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_worker.py tests/store/test_mid_ctx_modes.py -q`.
- [x] Note any pre-existing failures, environment issues, or skipped tests in this plan before implementation begins.

### Phase 1 Notes

Current public names used directly from `pipeline.distributed.pass1_merge` or re-exported through `pipeline.distributed.__init__`:

- Type aliases and constants: `LatentStatsPartial`, `TopCtxPartial`, `MidCtxCandidatesPartial`, `SeqReprPartial`, `LogitCtxPartial`, `MID_CTX_CANDIDATE_POOL_DEFAULTS`, and `PASS1_PARTIAL_FILENAMES`.
- CLI entrypoints: `build_arg_parser` and `main`.
- Artifact-specific load/merge helpers: `load_and_merge_latent_stats_partials`, `merge_latent_stats_partials`, `load_and_merge_top_ctx_partials`, `merge_top_ctx_partials`, `load_and_merge_mid_ctx_candidate_partials`, `merge_mid_ctx_candidate_partials`, `load_and_merge_seq_repr_partials`, `merge_seq_repr_partials`, `load_and_merge_logit_ctx_partials`, and `merge_logit_ctx_partials`.
- Writer, index, and reporting helpers: `merge_seq_latent_index_shards`, `merge_pass1_worker_outputs`, and `build_pass1_sanity_report`.

Observed compatibility call sites:

- `tests/pipeline/test_distributed_pass1_merge.py` imports the load/merge helpers and writer/index helpers directly from `pipeline.distributed.pass1_merge`.
- `tests/pipeline/test_distributed_controller.py` imports `pass1_merge` as a module through `pipeline.distributed` and calls `pass1_merge.build_arg_parser()`.
- `src/pipeline/distributed/__init__.py` re-exports the stable pass-1 merge public surface from `pass1_merge.py`.
- `src/pipeline/distributed/controller.py` imports `merge_pass1_worker_outputs` from `.pass1_merge` inside `run_parts_1_to_3`.

Import-cycle risks to avoid during extraction:

- Keep pass-1 contracts dependency-light; type aliases and constants should not import merge, writer, report, or CLI modules.
- The writer should be extracted after artifact-specific merge modules because `merge_pass1_worker_outputs` calls every load/merge path.
- `context_merge.py` will likely depend on latent-stats semantics for `mid_ctx` selection, but latent-stats merge code should not depend on context merge code.
- `cli.py` should import writer execution late and should be invoked from `pass1_merge.py`, not imported by lower-level modules.

Baseline verification:

- Command: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_worker.py tests/store/test_mid_ctx_modes.py -q`
- Result on 2026-05-23: `80 passed in 1.67s`.
- Pre-existing failures/blockers: none observed.

## Phase 2 — Create The Pass 1 Package Shell

- [x] Create `src/pipeline/distributed/pass1/__init__.py` as the new package entrypoint for pass-1 merge helpers.
- [x] Create `src/pipeline/distributed/pass1/contracts.py` for shared type aliases and constants.
- [x] Move `LatentStatsPartial`, `TopCtxPartial`, `MidCtxCandidatesPartial`, `SeqReprPartial`, `LogitCtxPartial`, `MID_CTX_CANDIDATE_POOL_DEFAULTS`, and `PASS1_PARTIAL_FILENAMES` into `contracts.py`.
- [x] Update `pass1_merge.py` to import these contracts while still exposing the same public names.
- [x] Verify the package shell with `python -m pytest tests/pipeline/test_distributed_pass1_merge.py -q`.

### Phase 2 Notes

- Added `src/pipeline/distributed/pass1/contracts.py` with the pass-1 merge type aliases, candidate-pool defaults, partial artifact filenames, and a narrow `__all__`.
- Added `src/pipeline/distributed/pass1/__init__.py` to expose the same contract names from the new package entrypoint.
- Updated `src/pipeline/distributed/pass1_merge.py` to import those names from `pass1.contracts`; existing callers can still import them from `pipeline.distributed.pass1_merge`.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py -q` -> `39 passed in 0.52s`.
- Lints/diagnostics reported no errors for `pass1_merge.py`, `pass1/contracts.py`, or `pass1/__init__.py`.

## Phase 3 — Extract Latent Stats Merge

- [x] Create `src/pipeline/distributed/pass1/latent_stats_merge.py`.
- [x] Move `load_and_merge_latent_stats_partials`, `merge_latent_stats_partials`, `_merge_welford_state`, `_clamp_small_negative_variance_state`, `_validate_latent_stats_partial_set`, and `_validate_merged_latent_stats`.
- [x] Keep the Welford merge implementation unchanged except for import paths.
- [x] Re-export the moved public functions from `pass1_merge.py` and `pass1/__init__.py`.
- [x] Verify latent-stats behavior with `python -m pytest tests/pipeline/test_distributed_pass1_merge.py -q`.

### Phase 3 Notes

- Added `src/pipeline/distributed/pass1/latent_stats_merge.py` for latent-stats partial loading, parallel Welford merge semantics, tiny negative variance clamping, partial-set validation, and merged payload validation.
- Updated `src/pipeline/distributed/pass1/__init__.py` to expose `load_and_merge_latent_stats_partials` and `merge_latent_stats_partials` from the new package.
- Updated `src/pipeline/distributed/pass1_merge.py` to import and re-export the moved public latent-stats helpers for backward compatibility.
- Removed the latent-stats-only private helpers from `pass1_merge.py`; context, seq-repr, logit-context, writer, report, and CLI behavior remain in the facade for later phases.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py -q` -> `39 passed in 0.56s`.
- Lints/diagnostics reported no errors for `pass1_merge.py`, `pass1/__init__.py`, or `pass1/latent_stats_merge.py`.

## Phase 4 — Extract Context Merges

- [x] Create `src/pipeline/distributed/pass1/context_merge.py`.
- [x] Move `load_and_merge_top_ctx_partials`, `merge_top_ctx_partials`, `_validate_top_ctx_partial_set`, `_validate_top_ctx_sequence_range`, and `_validate_merged_top_ctx`.
- [x] Move `load_and_merge_mid_ctx_candidate_partials`, `merge_mid_ctx_candidate_partials`, `_validate_mid_ctx_candidate_partial_set`, `_validate_latent_stats_for_mid_ctx`, `_std_seq_from_latent_stats`, `_concatenate_mid_ctx_candidates`, `_select_mid_ctx_candidates`, and `_validate_merged_mid_ctx`.
- [x] Preserve current distributed priority-reservoir and candidate-pool truncation semantics exactly.
- [x] Re-export the moved public functions from `pass1_merge.py` and `pass1/__init__.py`.
- [x] Verify context behavior with `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/store/test_mid_ctx_modes.py -q`.

### Phase 4 Notes

- Added `src/pipeline/distributed/pass1/context_merge.py` for `top_ctx` and `mid_ctx_candidates` partial loading, context merge semantics, sequence-range validation, stats-aware mid-context filtering, deterministic priority-reservoir selection, truncation policy handling, and merged payload validation.
- Updated `src/pipeline/distributed/pass1/__init__.py` to expose the top/mid context load and merge helpers from the new package.
- Updated `src/pipeline/distributed/pass1_merge.py` to import and re-export the moved context helpers while keeping the Phase 1 contract constants and type aliases available from the old module.
- Verification on 2026-05-23: initial run exposed a missing facade re-export for `MID_CTX_CANDIDATE_POOL_DEFAULTS`; after restoring the contract re-exports, `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/store/test_mid_ctx_modes.py -q` -> `42 passed in 0.53s`.
- Lints/diagnostics reported no errors for `pass1_merge.py`, `pass1/__init__.py`, or `pass1/context_merge.py`.

## Phase 5 — Extract Seq Repr And Logit Context Merges

- [x] Create `src/pipeline/distributed/pass1/seq_repr_merge.py`.
- [x] Move `load_and_merge_seq_repr_partials`, `merge_seq_repr_partials`, `_validate_seq_repr_partial_set`, `_mapping_from_seq_repr_payload`, `_validate_seq_repr_mapping_compatibility`, and `_validate_merged_seq_repr`.
- [x] Create `src/pipeline/distributed/pass1/logit_ctx_merge.py`.
- [x] Move `load_and_merge_logit_ctx_partials`, `merge_logit_ctx_partials`, `_validate_logit_ctx_partial_set`, `_select_logit_ctx_events`, `_validate_logit_ctx_token_range`, and `_validate_merged_logit_ctx`.
- [x] Preserve deterministic tie-breaking for logit event top-K merges.
- [x] Re-export the moved public functions from `pass1_merge.py` and `pass1/__init__.py`.
- [x] Verify with `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py -q`.

### Phase 5 Notes

- Added `src/pipeline/distributed/pass1/seq_repr_merge.py` for capped/uncapped `seq_repr` partial loading, global slot mapping, mapping compatibility validation, and merged payload validation.
- Added `src/pipeline/distributed/pass1/logit_ctx_merge.py` for `logit_ctx` partial loading, exact event top-K merge semantics, token range validation, deterministic tie-breaking, and merged payload validation.
- Updated `src/pipeline/distributed/pass1/__init__.py` to expose the sequence-representation and logit-context load/merge helpers from the new package.
- Updated `src/pipeline/distributed/pass1_merge.py` to import and re-export the moved public helpers for backward compatibility; private seq-repr and logit-context helpers now live with their owning modules.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py -q` -> `50 passed in 0.60s`.
- Lints/diagnostics reported no errors for `pass1_merge.py`, `pass1/__init__.py`, `pass1/seq_repr_merge.py`, or `pass1/logit_ctx_merge.py`.

## Phase 6 — Extract Seq Latent Index Merge

- [x] Create `src/pipeline/distributed/pass1/seq_latent_index_merge.py`.
- [x] Move `merge_seq_latent_index_shards`, `_parse_seq_latent_index_shard_id`, `_validate_seq_latent_index_shard_file`, `_seq_latent_index_files_equivalent`, and `_copy_file_atomic`.
- [x] Preserve duplicate-shard behavior: identical duplicates are accepted, differing duplicates fail.
- [x] Re-export `merge_seq_latent_index_shards` from `pass1_merge.py` and `pass1/__init__.py`.
- [x] Verify index-shard behavior with `python -m pytest tests/pipeline/test_distributed_pass1_merge.py -q`.

### Phase 6 Notes

- Added `src/pipeline/distributed/pass1/seq_latent_index_merge.py` for canonical `seq_latent_index` shard copying, shard filename parsing, shard payload validation, duplicate-identical acceptance, duplicate-different rejection, and atomic file copying.
- Updated `src/pipeline/distributed/pass1/__init__.py` to expose `merge_seq_latent_index_shards` from the new package.
- Updated `src/pipeline/distributed/pass1_merge.py` to import and re-export `merge_seq_latent_index_shards` for backward compatibility; the old module no longer owns the private index-shard helpers.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py -q` -> `39 passed in 0.50s`.
- Lints/diagnostics reported no errors for `pass1_merge.py`, `pass1/__init__.py`, or `pass1/seq_latent_index_merge.py`.

## Phase 7 — Extract Writer, Reports, And CLI

- [x] Create `src/pipeline/distributed/pass1/writer.py`.
- [x] Move `merge_pass1_worker_outputs`, `_with_canonical_metadata`, `_pass1_partial_paths`, `_atomic_torch_save`, `_atomic_write_json`, and `_validate_written_artifacts`.
- [x] Create `src/pipeline/distributed/pass1/reports.py`.
- [x] Move `build_pass1_sanity_report`, `_tensor_summary`, `_sequence_id_range`, `_context_fill_rate`, `_seq_repr_fill`, and `_logit_ctx_count_summary`.
- [x] Create `src/pipeline/distributed/pass1/cli.py`.
- [x] Move `build_arg_parser` and `main`.
- [x] Keep `python -m pipeline.distributed.pass1_merge` working by delegating from `pass1_merge.py` to `pass1.cli`.
- [x] Verify CLI and writer behavior with `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_pass1_merge.py -q`.

### Phase 7 Notes

- Added `src/pipeline/distributed/pass1/reports.py` for pass-1 sanity report construction and report-only tensor/fill/count summaries.
- Added `src/pipeline/distributed/pass1/writer.py` for `merge_pass1_worker_outputs`, canonical metadata injection, pass-1 partial path discovery, atomic artifact/report writes, written-artifact validation, manifest update, and pass-2 replay assignment.
- Added `src/pipeline/distributed/pass1/cli.py` for `build_arg_parser` and `main`.
- Updated `src/pipeline/distributed/pass1_merge.py` to a 64-line compatibility facade with explicit `__all__`, preserving old imports and `python -m pipeline.distributed.pass1_merge`.
- Updated `src/pipeline/distributed/pass1/__init__.py` to expose writer, report, and CLI helpers from the package entrypoint.
- Verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_pass1_merge.py -q` -> `65 passed in 0.86s`.
- Lints/diagnostics reported no errors for `pass1_merge.py`, `pass1/__init__.py`, `pass1/cli.py`, `pass1/writer.py`, or `pass1/reports.py`.

## Phase 8 — Compatibility Facade And Package Exports

- [x] Reduce `src/pipeline/distributed/pass1_merge.py` to a small compatibility module that imports public names from `src/pipeline/distributed/pass1/`.
- [x] Update `src/pipeline/distributed/pass1/__init__.py` to expose the intended pass-1 merge API.
- [x] Update `src/pipeline/distributed/__init__.py` only where necessary, preserving existing exported names and import behavior.
- [x] Search for imports of moved helpers and confirm every old import path still resolves.
- [x] Avoid broad test rewrites; keep tests behavior-focused unless a test explicitly checks module ownership.
- [x] Confirm `src/pipeline/distributed/pass1_merge.py` is reduced to a small facade, ideally under 150 LOC.

### Phase 8 Notes

- Confirmed `src/pipeline/distributed/pass1_merge.py` is a 64-line compatibility facade with explicit `__all__`.
- Confirmed `src/pipeline/distributed/pass1/__init__.py` exposes the intended pass-1 merge API, including contracts, artifact-specific load/merge helpers, index merge, report helper, writer entrypoint, and CLI helpers.
- No update was needed in `src/pipeline/distributed/__init__.py`; it continues importing the preserved pass-1 public names from `pass1_merge.py`.
- Repository import search found old-facade imports in `src/pipeline/distributed/controller.py`, `tests/pipeline/test_distributed_pass1_merge.py`, and `tests/pipeline/test_distributed_controller.py`; all remain supported by the facade.
- Import smoke check on 2026-05-23: old `pipeline.distributed.pass1_merge`, new `pipeline.distributed.pass1`, and existing `pipeline.distributed` re-exports all expose the expected pass-1 merge public names.
- No test rewrites were needed for this compatibility phase.

## Phase 9 — Testing And Verification

- [x] Run focused pass-1 merge tests: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py -q`.
- [x] Run pass-1 contract tests: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_worker.py tests/store/test_mid_ctx_modes.py -q`.
- [x] Run distributed integration-adjacent tests: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_negative_context_stage.py -q`.
- [x] Run lints/diagnostics on all newly created and edited files.
- [x] Confirm no artifact schema, tensor field, sanity report, manifest update, or pass-2 replay assignment behavior changed.
- [x] Confirm no generated artifacts, native binaries, or unrelated local outputs were touched by the refactor.

### Phase 9 Notes

- Focused pass-1 merge verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py -q` -> `39 passed in 0.49s`.
- Pass-1 contract verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_worker.py tests/store/test_mid_ctx_modes.py -q` -> `80 passed in 0.81s`.
- Distributed integration-adjacent verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_negative_context_stage.py -q` -> `50 passed in 22.90s`.
- Lints/diagnostics reported no errors for `pass1_merge.py`, all files under `src/pipeline/distributed/pass1/`, or this plan file.
- Artifact schemas, tensor fields, sanity-report shape, manifest update behavior, and pass-2 replay assignment behavior are covered by the pass-1 merge, worker, controller, interface, and negative-context stage tests above; no behavior-focused test rewrites were needed.
- `git status --short` shows this refactor's source/planning changes plus pre-existing unrelated work such as `agent-planning/multi-device-improvements/part-8-testing-and-benchmarks.md` and the earlier pass-2 refactor files; no generated artifacts, native binaries, or output-run files were touched by this refactor.

---

## Open Questions

- Should `pass1_merge.py` remain the permanent CLI module, or should all CLI logic live only in `src/pipeline/distributed/pass1/cli.py`?
- Should atomic save helpers eventually move to a shared distributed utility module used by both pass-1 and pass-2 reducers?
- Should `top_ctx` and `mid_ctx` live in one `context_merge.py` module, or should `mid_ctx` get its own module because it carries more policy and exactness semantics?
- Should tests import from the new `pipeline.distributed.pass1` package after extraction, or keep old imports to enforce backward compatibility?

## Risks / Assumptions

- This refactor should not change pass-1 merge algorithms, artifact schemas, sanity-report shape, manifest update behavior, or public import names.
- `mid_ctx` is the highest-risk extraction because candidate-pool filtering, replay fallback signaling, deterministic priority selection, and exactness policy are tightly coupled.
- The writer depends on many merge helpers, so it should be extracted after artifact-specific modules are stable.
- Keeping `pass1_merge.py` as a facade lowers rollout risk and allows downstream callers to migrate gradually.
