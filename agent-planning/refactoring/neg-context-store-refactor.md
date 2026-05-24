# Plan: Neg Context Store Refactor

> **Goal:** Split `src/store/neg_context.py` into a focused `src/store/neg_ctx/` package while preserving existing public imports, exact retrieval behavior, artifact semantics, and test coverage.
>
> **Created:** 2026-05-23

---

## Phase 1 - Establish Compatibility Baseline

- [x] Record the current public API from `src/store/neg_context.py`: `NegCtxStats`, `TorchANNIndex`, memory guardrail helpers, sharded ANN helpers, device helpers, `validate_neg_ctx_output`, and all `build_neg_ctx*` entrypoints.
- [x] Search repository imports of `store.neg_context` and direct function/class imports to identify compatibility requirements.
- [x] Confirm `src/store/neg_context.py` should remain as a facade so existing imports such as `from store.neg_context import build_neg_ctx` keep working.
- [x] Run the focused baseline command before refactoring: `python -m pytest tests/store/test_neg_context_backend.py tests/pipeline/test_negative_context_stage.py -q`.
- [x] Note any pre-existing failures, environment issues, or skipped tests in this plan before implementation begins.

### Phase 1 Notes

Current public and compatibility-sensitive names in `src/store/neg_context.py`:

- Stats/reporting: `NegCtxStats`.
- ANN and memory guardrails: `TorchANNIndex`, `estimate_neg_ctx_ann_memory`, `estimate_neg_ctx_ann_memory_for_shape`, `check_neg_ctx_memory_guardrail`, and `_record_ann_memory_estimate`.
- Sharded ANN support: `ANNIndexShard`, `ShardedANNIndex`, `partition_index_slots`, and `merge_shard_search_results`.
- Device and partition policy: `_ann_device`, `parse_neg_ctx_devices`, `partition_components`, and `_validate_cuda_devices`.
- Output validation and component processing: `validate_neg_ctx_output`, `_process_component`, and `_process_component_sharded`.
- Backend entrypoints: `build_neg_ctx`, `build_neg_ctx_single_gpu_exact`, `build_neg_ctx_multi_gpu`, and `build_neg_ctx_index_sharded`.

Observed compatibility call sites:

- `src/pipeline/negative_context.py` imports `NegCtxStats`, `build_neg_ctx`, `build_neg_ctx_multi_gpu`, `build_neg_ctx_single_gpu_exact`, and `validate_neg_ctx_output` from `store.neg_context`.
- `tests/store/test_neg_context_backend.py` imports `NegCtxStats`, `TorchANNIndex`, `ShardedANNIndex`, `_ann_device`, `_validate_cuda_devices`, `check_neg_ctx_memory_guardrail`, `estimate_neg_ctx_ann_memory`, `merge_shard_search_results`, `_process_component`, `parse_neg_ctx_devices`, `partition_index_slots`, `partition_components`, and `validate_neg_ctx_output`.
- `tests/pipeline/test_negative_context_stage.py` imports `NegCtxStats`.
- `tests/pipeline/test_distributed_pass1_merge.py` imports `NegCtxStats`.
- `tests/store/test_neg_context_backend.py` also monkeypatches module globals under `store.neg_context`, including `torch.cuda.is_available`, `torch.cuda.device_count`, and `config.hardware.ann_device`.

Compatibility decision:

- Keep `src/store/neg_context.py` as the compatibility facade throughout the refactor.
- Re-export private helpers that existing tests currently import or monkeypatch until a later intentional cleanup narrows the public surface.
- Preserve module-level `torch` and `config` access through the facade during extraction or update tests deliberately in a separate behavior-neutral step.

Baseline verification:

- Command: `python -m pytest tests/store/test_neg_context_backend.py tests/pipeline/test_negative_context_stage.py -q`
- Result on 2026-05-23: `35 passed in 0.42s`.
- Pre-existing failures/blockers: none observed.

## Phase 2 - Create The `neg_ctx` Package Shell

- [x] Create `src/store/neg_ctx/__init__.py` as the new package entrypoint.
- [x] Add package-level exports that mirror the intended public API from `store.neg_context`.
- [x] Update `src/store/neg_context.py` only enough to import from the new package once extracted modules exist.
- [x] Verify the package can be imported without changing behavior using `python -m pytest tests/store/test_neg_context_backend.py -q`.

### Phase 2 Notes

- Added `src/store/neg_ctx/__init__.py` as the package entrypoint.
- The package shell currently re-exports the compatibility-sensitive surface from `store.neg_context`, including private helpers that tests import directly.
- Left `src/store/neg_context.py` unchanged in this phase to avoid introducing a circular import before any implementation modules have moved into `store.neg_ctx`.
- Import smoke check on 2026-05-23: `store.neg_ctx.NegCtxStats` resolves to the same object as `store.neg_context.NegCtxStats`.
- Verification on 2026-05-23: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `19 passed in 0.12s`.

## Phase 3 - Extract Stats And Reporting

- [x] Create `src/store/neg_ctx/stats.py`.
- [x] Move `NegCtxStats` into `stats.py`.
- [x] Preserve JSON report field names, timing fields, fill-rate calculations, `record_seq_repr`, `print_summary`, `save`, and `merge_from` behavior exactly.
- [x] Re-export `NegCtxStats` from `src/store/neg_ctx/__init__.py` and `src/store/neg_context.py`.
- [x] Verify with `python -m pytest tests/store/test_neg_context_backend.py -q`.

### Phase 3 Notes

- Added `src/store/neg_ctx/stats.py` for `NegCtxStats`, including fill-rate properties, sequence-representation recording, console summary printing, JSON stats saving, and stats merging.
- Updated `src/store/neg_context.py` to import `NegCtxStats` from `store.neg_ctx.stats`, preserving existing imports from `store.neg_context`.
- Updated `src/store/neg_ctx/__init__.py` so `NegCtxStats` is exported directly from the new package while remaining compatibility exports are lazily resolved from `store.neg_context`.
- The lazy package exports avoid a circular import now that `store.neg_context` imports the extracted stats module.
- Import smoke check on 2026-05-23: `store.neg_ctx.stats.NegCtxStats`, `store.neg_ctx.NegCtxStats`, and `store.neg_context.NegCtxStats` all resolve to the same class.
- Verification on 2026-05-23: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `19 passed in 0.13s`.

## Phase 4 - Extract ANN Index And Memory Guardrails

- [x] Create `src/store/neg_ctx/ann.py`.
- [x] Move `TorchANNIndex`, `estimate_neg_ctx_ann_memory`, `estimate_neg_ctx_ann_memory_for_shape`, `check_neg_ctx_memory_guardrail`, and `_record_ann_memory_estimate`.
- [x] Preserve exact cosine similarity search semantics, chunk sizing behavior, result device behavior, and guardrail failure/warning behavior.
- [x] Re-export public ANN helpers from `src/store/neg_ctx/__init__.py` and `src/store/neg_context.py`.
- [x] Verify with `python -m pytest tests/store/test_neg_context_backend.py -q`.

### Phase 4 Notes

- Added `src/store/neg_ctx/ann.py` for the exact PyTorch cosine ANN index, ANN memory estimation helpers, memory guardrail checks, and stats memory-estimate recording.
- Updated `src/store/neg_context.py` to import `TorchANNIndex`, `estimate_neg_ctx_ann_memory`, `estimate_neg_ctx_ann_memory_for_shape`, `check_neg_ctx_memory_guardrail`, and `_record_ann_memory_estimate` from `store.neg_ctx.ann`.
- Updated `src/store/neg_ctx/__init__.py` so the ANN helpers are exported directly from the package while not-yet-extracted names remain lazy compatibility exports from `store.neg_context`.
- Import smoke check on 2026-05-23: `store.neg_ctx.ann.TorchANNIndex`, `store.neg_ctx.TorchANNIndex`, and `store.neg_context.TorchANNIndex` all resolve to the same class.
- Verification on 2026-05-23: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `19 passed in 0.13s`.

## Phase 5 - Extract Sharded ANN Support

- [x] Create `src/store/neg_ctx/sharded_ann.py`.
- [x] Move `ANNIndexShard`, `ShardedANNIndex`, `partition_index_slots`, and `merge_shard_search_results`.
- [x] Preserve contiguous slot partitioning, global slot ID restoration, and top-K merge semantics.
- [x] Keep imports from `ann.py` explicit so sharded search depends on `TorchANNIndex` without circular imports.
- [x] Verify sharded backend behavior with `python -m pytest tests/store/test_neg_context_backend.py -q`.

### Phase 5 Notes

- Added `src/store/neg_ctx/sharded_ann.py` for `ANNIndexShard`, contiguous ANN slot partitioning, shard-result top-K merging, and `ShardedANNIndex`.
- Kept the dependency on `TorchANNIndex` explicit via `from .ann import TorchANNIndex`.
- Updated `src/store/neg_context.py` to import and re-export the sharded ANN names, preserving existing imports from `store.neg_context`.
- Updated `src/store/neg_ctx/__init__.py` so sharded ANN helpers are exported directly from the package while not-yet-extracted names remain lazy compatibility exports from `store.neg_context`.
- Import smoke check on 2026-05-23: `store.neg_ctx.sharded_ann.ShardedANNIndex`, `store.neg_ctx.ShardedANNIndex`, and `store.neg_context.ShardedANNIndex` all resolve to the same class.
- Verification on 2026-05-23: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `19 passed in 0.13s`.

## Phase 6 - Extract Device And Partition Policy

- [x] Create `src/store/neg_ctx/devices.py`.
- [x] Move `_ann_device`, `parse_neg_ctx_devices`, `partition_components`, and `_validate_cuda_devices`.
- [x] Preserve accepted config values for `hardware.ann_device`: `auto`, `cpu`, `gpu`, `cuda`, and `cuda:N`.
- [x] Preserve duplicate device removal and CUDA range validation behavior.
- [x] Verify device parsing and partitioning with `python -m pytest tests/store/test_neg_context_backend.py -q`.

### Phase 6 Notes

- Added `src/store/neg_ctx/devices.py` for ANN device selection, configured CUDA device parsing, component partitioning, and CUDA visibility validation.
- Updated `src/store/neg_context.py` to import and re-export `_ann_device`, `parse_neg_ctx_devices`, `partition_components`, and `_validate_cuda_devices`, preserving existing tests and callers that import from `store.neg_context`.
- Updated `src/store/neg_ctx/__init__.py` so device helpers are exported directly from the package while not-yet-extracted names remain lazy compatibility exports from `store.neg_context`.
- Existing monkeypatch tests remain compatible because the moved helpers use the shared `torch` module and shared `config` object.
- Import smoke check on 2026-05-23: `store.neg_ctx.devices._ann_device`, `store.neg_ctx._ann_device`, and `store.neg_context._ann_device` all resolve to the same function.
- Verification on 2026-05-23: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `19 passed in 0.14s`.

## Phase 7 - Extract Output Validation

- [x] Create `src/store/neg_ctx/validation.py`.
- [x] Move `validate_neg_ctx_output`.
- [x] Preserve all validation checks for tensor rank, shape, sequence ID bounds, finite similarities, non-negative similarities, and value/ID consistency.
- [x] Re-export `validate_neg_ctx_output` from `src/store/neg_ctx/__init__.py` and `src/store/neg_context.py`.
- [x] Verify validation behavior with `python -m pytest tests/store/test_neg_context_backend.py tests/pipeline/test_negative_context_stage.py -q`.

### Phase 7 Notes

- Added `src/store/neg_ctx/validation.py` for `validate_neg_ctx_output`.
- Preserved rank, shape, configured row-count, non-negative sequence ID, finite similarity, non-negative similarity, sequence-bound, and positive-value/ID consistency checks.
- Updated `src/store/neg_context.py` to import and re-export `validate_neg_ctx_output`, preserving existing callers such as `pipeline.negative_context`.
- Updated `src/store/neg_ctx/__init__.py` so `validate_neg_ctx_output` is exported directly from the new package.
- Import smoke check on 2026-05-23: `store.neg_ctx.validation.validate_neg_ctx_output`, `store.neg_ctx.validate_neg_ctx_output`, and `store.neg_context.validate_neg_ctx_output` all resolve to the same function.
- Verification on 2026-05-23: `python -m pytest tests/store/test_neg_context_backend.py tests/pipeline/test_negative_context_stage.py -q` -> `35 passed in 0.41s`.

## Phase 8 - Extract Component Processing

- [x] Create `src/store/neg_ctx/component.py`.
- [x] Move `_PAIR_CHUNK`, `_process_component`, and `_process_component_sharded`.
- [x] Preserve capped and uncapped `seq_repr` mapping behavior exactly.
- [x] Preserve positive-context filtering, query matrix construction, ANN query, positive membership filtering, fill-count updates, and write layout exactly.
- [x] Avoid tensor-logic cleanup in this phase; extraction should be mechanical to reduce performance and correctness risk.
- [x] Verify with `python -m pytest tests/store/test_neg_context_backend.py -q`.

### Phase 8 Notes

- Added `src/store/neg_ctx/component.py` for `_PAIR_CHUNK`, `_process_component`, and `_process_component_sharded`.
- Kept the extraction mechanical: positive-context pair collection, capped/uncapped slot mapping, scatter-mean query construction, ANN querying, positive membership filtering, fill-count updates, and CPU store writes are preserved.
- Updated `src/store/neg_context.py` to import and re-export the component helpers, preserving tests that import `_process_component` from `store.neg_context`.
- Updated `src/store/neg_ctx/__init__.py` so component helpers are exported directly from the package.
- Import smoke check on 2026-05-23: `store.neg_ctx.component._process_component`, `store.neg_ctx._process_component`, and `store.neg_context._process_component` all resolve to the same function.
- Verification on 2026-05-23: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `19 passed in 0.15s`.

## Phase 9 - Extract Backend Orchestration

- [x] Create `src/store/neg_ctx/backends.py`.
- [x] Move `build_neg_ctx`, `build_neg_ctx_single_gpu_exact`, `build_neg_ctx_multi_gpu`, and `build_neg_ctx_index_sharded`.
- [x] Keep backend selection driven by `config.latents.neg_ctx.backend`.
- [x] Preserve stats aggregation, per-device timings, component assignments, memory estimates, output validation, and progress output.
- [x] Re-export all backend entrypoints from `src/store/neg_ctx/__init__.py` and `src/store/neg_context.py`.
- [x] Verify with `python -m pytest tests/store/test_neg_context_backend.py tests/pipeline/test_negative_context_stage.py -q`.

### Phase 9 Notes

- Added `src/store/neg_ctx/backends.py` for `build_neg_ctx`, `build_neg_ctx_single_gpu_exact`, `build_neg_ctx_multi_gpu`, and `build_neg_ctx_index_sharded`.
- Preserved backend selection through `config.latents.neg_ctx.backend`, including `single_gpu_exact`, `multi_gpu_exact`, and `multi_gpu_index_sharded_exact`.
- Preserved stats aggregation, per-device timing summaries, component assignments, index-shard assignments, memory guardrail recording, output validation, and tqdm/progress output.
- Updated `src/store/neg_ctx/__init__.py` so backend entrypoints are exported directly from the package.
- Reduced `src/store/neg_context.py` to a compatibility facade that re-exports the extracted stats, ANN, sharded ANN, device, validation, component, and backend helpers.
- Import smoke check on 2026-05-23: `store.neg_ctx.backends.build_neg_ctx`, `store.neg_ctx.build_neg_ctx`, and `store.neg_context.build_neg_ctx` all resolve to the same function.
- Verification on 2026-05-23: `python -m pytest tests/store/test_neg_context_backend.py tests/pipeline/test_negative_context_stage.py -q` -> `35 passed in 0.40s`.

## Phase 10 - Reduce `neg_context.py` To A Compatibility Facade

- [x] Replace `src/store/neg_context.py` with a small facade that imports public names from `src/store/neg_ctx/`.
- [x] Define `__all__` in either `src/store/neg_ctx/__init__.py` or `src/store/neg_context.py` so the stable public API is explicit.
- [x] Confirm all existing imports from `store.neg_context` still resolve without caller changes.
- [x] Avoid broad test rewrites; only adjust tests if they intentionally inspect private helper ownership.
- [x] Confirm `src/store/neg_context.py` is reduced to a small facade, ideally under 100 LOC.

### Phase 10 Notes

- Confirmed `src/store/neg_context.py` is now a 62-line compatibility facade.
- Added explicit `__all__` to `src/store/neg_context.py`, matching the package-level `src/store/neg_ctx/__init__.py` export list.
- Preserved compatibility-sensitive module globals `torch` and `config` on `store.neg_context` for existing monkeypatch tests.
- Repository import search found old-facade imports in `src/pipeline/negative_context.py`, `tests/store/test_neg_context_backend.py`, `tests/pipeline/test_negative_context_stage.py`, and `tests/pipeline/test_distributed_pass1_merge.py`; no caller changes were needed.
- Import smoke check on 2026-05-23: every name in `store.neg_context.__all__` resolves, `store.neg_context.__all__ == store.neg_ctx.__all__`, and `torch`/`config` remain present on the facade.
- Verification on 2026-05-23: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `19 passed in 0.13s`.

## Phase 11 - Testing And Verification

- [x] Run focused backend tests: `python -m pytest tests/store/test_neg_context_backend.py -q`.
- [x] Run negative-context stage tests: `python -m pytest tests/pipeline/test_negative_context_stage.py -q`.
- [x] Run related distributed pass-1 tests that feed negative context: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/store/test_mid_ctx_modes.py -q`.
- [x] Run lints/diagnostics on all newly created and edited files.
- [x] Confirm no backend output semantics, stats JSON fields, memory guardrail behavior, or `neg_ctx.pt` tensor layout changed.
- [x] Confirm no generated artifacts, native binaries, or unrelated local outputs were touched by the refactor.

### Phase 11 Notes

- Focused backend verification on 2026-05-23: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `19 passed in 0.13s`.
- Negative-context stage verification on 2026-05-23: `python -m pytest tests/pipeline/test_negative_context_stage.py -q` -> `16 passed in 0.41s`.
- Related distributed/pass-1 verification on 2026-05-23: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/store/test_mid_ctx_modes.py -q` -> `69 passed in 1.01s`.
- WSL local pipeline smoke on 2026-05-23: `source .venv/bin/activate && bash scripts/run.sh` -> completed successfully in 11m 52s.
- WSL distributed controller dry-run on 2026-05-23: `source .venv/bin/activate && PYTHONPATH=src python -m pipeline.distributed.controller --project-root . --config config.yaml --output-base outputs --use-cpu --worker-count 1 --dry-run` -> completed successfully.
- Lints/diagnostics reported no errors for `src/store/neg_context.py`, all files under `src/store/neg_ctx/`, or this plan file.
- Backend output semantics, stats JSON fields, memory guardrail behavior, and `neg_ctx.pt` tensor layout remain covered by the backend and stage tests above; no behavior-focused test rewrites were needed.
- `git status --short` shows this refactor's store source/planning changes plus pre-existing unrelated work such as `.gitignore`, `README.md`, multi-device planning, and earlier pass-1/pass-2/attribution refactor files; no generated artifacts, native binaries, or output-run files were touched by this phase.

---

## Open Questions

- Should private helpers such as `_ann_device`, `_validate_cuda_devices`, and `_process_component` remain importable through `store.neg_context`, or should only documented public entrypoints be re-exported?
- Should shared atomic/stat/report utilities eventually move to a broader `store` or `pipeline.distributed` utility module?
- Should `stats.py` keep printing behavior on `NegCtxStats`, or should console/report formatting eventually move to a separate reporter module?
- Should `component.py` expose a small typed result object for timing instead of returning plain dictionaries in a later cleanup?

## Risks / Assumptions

- The highest-risk code is `_process_component` and `_process_component_sharded`; these are performance-sensitive and should be moved mechanically before any cleanup.
- Backend output semantics must remain identical: final `neg_ctx.pt` rows contain global sequence IDs and cosine similarities regardless of backend.
- Keeping `src/store/neg_context.py` as a facade lowers rollout risk and preserves all existing call sites.
- This refactor should not change ANN search algorithms, positive filtering, memory guardrail policy, stats JSON shape, or public import names.
