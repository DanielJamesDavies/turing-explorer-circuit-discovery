# Plan: Part 3 - Negative Context

> **Goal:** Build `neg_ctx` from merged global pass-1 artifacts using exact single- or multi-GPU retrieval, preserving the canonical `neg_ctx.pt` contract for downstream pass 2, candidate selection, and discovery.
>
> **Created:** 2026-05-16

---

## Scope

This part starts after Part 2 has produced validated global first-pass artifacts:

- `outputs/<run_id>/top_ctx.pt`
- `outputs/<run_id>/mid_ctx.pt`
- `outputs/<run_id>/seq_repr.pt`

It builds:

- `outputs/<run_id>/neg_ctx.pt`
- `outputs/<run_id>/neg_ctx_stats.json`

It does not rerun model forwards, modify pass-1 merge logic, run pass 2, or run discovery.

The initial target is exactness and artifact compatibility. The current `single_gpu_exact` and `multi_gpu_exact` semantics remain the baseline.

---

## Phase 1 - Input Contract And Loader

- [x] Add a negative-context runner that loads merged global `top_ctx`, `mid_ctx`, and `seq_repr` artifacts without requiring model or SAE initialization.
- [x] Validate that required pass-1 artifacts exist before starting ANN work.
- [x] Validate artifact compatibility: component count, SAE width, context counts, sequence ID ranges, `seq_repr` cap metadata, and config hash when available.
- [x] Support the current in-pipeline path and a standalone distributed-stage path that can be launched after Part 2 merge.
- [x] Keep the canonical output paths `outputs/<run_id>/neg_ctx.pt` and `outputs/<run_id>/neg_ctx_stats.json` for downstream code in run-root mode.
- [x] Verification: add a tiny fixture test that loads synthetic pass-1 artifacts and rejects missing or incompatible inputs.

### Phase 1 Notes

- Extended `src/pipeline/negative_context.py` with `load_negative_context_inputs()` for run-root pass-1 artifact loading and validation, covering required artifact presence, context tensor shapes/types, component/SAE-width agreement, sequence ID bounds, optional config-hash checks, and capped `seq_repr` mapping consistency.
- Added `run_negative_context_stage()` plus `python -m pipeline.negative_context --output-root <run_root>` so Part 3 can run after a distributed pass-1 merge without model or SAE initialization in the loader.
- Updated the existing in-pipeline `build_negative_contexts()` path to accept an `output_root` while preserving its default `outputs` behavior.
- Verification: `python -m pytest tests/pipeline/test_negative_context_stage.py -q` -> `5 passed`.

## Phase 2 - Preserve Single-Device Exact Path

- [x] Keep `single_gpu_exact` as the correctness baseline for reduced/local runs.
- [x] Ensure `hardware.ann_device` continues to support `auto`, `cpu`, `gpu`, `cuda`, and explicit `cuda:N`.
- [x] Ensure local RTX 5070 Ti style runs can build `neg_ctx` without the distributed controller.
- [x] Add clearer stats for single-device exact runs: index build time, positive collection time, query time, filter time, write time, fill counts, device, and `seq_repr` cap size.
- [x] Validate that CPU fallback remains available for tests and small synthetic fixtures.
- [x] Verification: extend focused tests around `_ann_device`, `TorchANNIndex`, `_process_component`, and `NegCtxStats` using CPU fixtures.

### Phase 2 Notes

- Preserved the existing `single_gpu_exact` implementation as the default exact backend and kept it usable through both the normal in-pipeline path and the Part 3 run-root stage without requiring the distributed controller.
- Tightened `_ann_device()` so supported values are explicit: `auto`, `cpu`, `gpu`, `cuda`, and `cuda:N`; invalid values fail early and explicit CUDA ordinals are checked against the visible CUDA range.
- Extended `NegCtxStats` with `ann_device`, `seq_repr_n_seqs`, `seq_repr_n_stored`, `seq_repr_repr_dim`, `seq_repr_is_capped`, `seq_repr_cap_percent`, and saved `fill_counts`, while preserving existing timing fields for index build, PosCtx collection, query matrix construction, ANN query, filter, write, and total time.
- Added CPU-focused tests for `_ann_device`, `TorchANNIndex`, `_process_component`, and `NegCtxStats` save metadata.
- Verification: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `9 passed`.

## Phase 3 - Multi-GPU Exact Backend Hardening

- [x] Reuse `multi_gpu_exact` as the initial H100 path: replicate the capped `seq_repr` index on selected devices and split SAE components across devices.
- [x] In distributed runs, default selected devices to the manifest-declared physical devices for the run.
- [x] Keep "all visible devices" as a standalone/single-process convenience only, not the distributed default.
- [x] Validate selected devices from manifest/config and reject unavailable CUDA IDs before any expensive allocation.
- [x] Keep component assignments disjoint and deterministic.
- [x] Ensure each worker thread writes only its assigned `neg_ctx` component slices.
- [x] Record per-device component assignments and timing in `neg_ctx_stats.json`.
- [x] Add a hard validation step after build: every populated row has valid sequence IDs, non-negative finite similarities, and at most `n_sequences` entries.
- [x] Verification: add tests for deterministic component partitioning, duplicate device removal, invalid device rejection, and stats merge behavior.

### Phase 3 Notes

- Kept `multi_gpu_exact` as the H100-oriented exact path: each selected CUDA device builds a replicated exact `TorchANNIndex`, receives a deterministic round-robin component assignment, and writes only disjoint component slices in the shared `neg_ctx` artifact.
- Added manifest-aware device defaulting to `run_negative_context_stage(..., manifest_path=...)`; distributed runs pass manifest physical IDs into `build_neg_ctx(..., selected_devices=...)`, while standalone `multi_gpu_exact` still uses `latents.neg_ctx.devices` or all visible devices when no explicit list is configured.
- Hardened CUDA device validation before index allocation, preserving duplicate device removal and rejecting unavailable or out-of-range CUDA IDs.
- Extended `NegCtxStats` with `component_assignments` and `per_device_timing_ms` so `neg_ctx_stats.json` records which device owned each component slice and each device's index/query/write timing.
- Added `validate_neg_ctx_output()` and call it after both single-device and multi-device builds to reject invalid sequence IDs, non-finite or negative similarities, mismatched tensor shapes, or impossible row widths before saving canonical outputs.
- Verification: `python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py -q` -> `19 passed`.

## Phase 4 - Equivalence And Quality Checks

- [x] Add a reduced-run comparison mode that builds `neg_ctx` with `single_gpu_exact` and `multi_gpu_exact` from the same pass-1 artifacts.
- [x] Compare artifact shapes, dtype, populated row counts, fill-rate distribution, and sampled rows.
- [x] Require exact or near-exact row equality for the same backend settings and deterministic inputs.
- [x] Report any expected ordering/tie differences explicitly.
- [x] Compare total runtime and timing breakdowns to determine whether multi-GPU query compute or CPU write-back dominates.
- [x] Verification: add a synthetic equivalence test where single-device and multi-device component splits produce identical `neg_ctx` rows.

### Phase 4 Notes

- Added `compare_negative_context_backends()` in `src/pipeline/negative_context.py`, with CLI access through `python -m pipeline.negative_context --output-root <run_root> --compare-backends`.
- The comparison mode loads the same merged pass-1 artifacts, builds one `neg_ctx` with `single_gpu_exact` and one with `multi_gpu_exact`, then writes `neg_ctx_equivalence_report.json` under the run root.
- Added `build_negative_context_comparison_report()` to compare shape, dtype, populated row count, fill distribution, exact sequence-ID equality, exact/near-exact similarity equality, max absolute value difference, sampled rows, and timing breakdowns for both backends.
- Reports include an `ordering_or_tie_note` so any row ordering or tie-related differences are visible instead of silently accepted.
- Verification: `python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py -q` -> `22 passed`.

## Phase 5 - Manifest Integration

- [x] Add manifest fields for negative-context backend, selected devices, `n_neighbors`, `n_sequences`, `min_pos_ctx`, `repr_mode`, and `max_repr_seqs`.
- [x] Record whether device selection came from manifest-declared run devices, explicit config override, or standalone all-visible-device discovery.
- [x] Record whether this part used `single_gpu_exact`, `multi_gpu_exact`, or a future index-sharded backend.
- [x] Add part status markers under `outputs/<run_id>/distributed/parts/neg_ctx/`.
- [x] Mark Part 3 completed only after `neg_ctx.pt`, `neg_ctx_stats.json`, and sanity validation all succeed.
- [x] Support resume behavior: skip rebuild only if the config hash, input artifact metadata, backend, devices, and output sanity report match.
- [x] Verification: add dry-run and resume tests for completed, stale, failed, and missing `neg_ctx` outputs.

### Phase 5 Notes

- Added `NegativeContextRunConfig` to the distributed manifest contract with backend, selected devices, device-selection source, `n_neighbors`, `n_sequences`, `min_pos_ctx`, `repr_mode`, and `max_repr_seqs`.
- `run_negative_context_stage()` now records stage metadata, writes `started.json`, `completed.json`, `failed.json`, and `neg_ctx_sanity_report.json` under `outputs/<run_id>/distributed/parts/neg_ctx/`, and updates the manifest `neg_ctx` block after successful completion.
- Added dry-run planning and resume classification via `plan_negative_context_stage()`, plus CLI flags `--dry-run` and `--resume`.
- Resume skips rebuild only when canonical outputs exist and the completed marker plus sanity report metadata match current config hash, backend, devices, and input artifact metadata; stale, failed, and missing outputs are classified explicitly.
- Verification: `python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py tests/pipeline/test_distributed_manifest.py -q` -> `40 passed`.

## Phase 6 - Memory And Scale Guardrails

- [x] Estimate per-device ANN index memory before building the index: `n_stored * repr_dim * 4` plus slot maps and working buffers.
- [x] Warn or fail early when replicated index memory is likely to exceed a configured fraction of available VRAM.
- [x] Log `seq_repr.n_stored`, `seq_repr.n_seqs`, cap percentage, representation dim, and estimated index memory.
- [x] Bound query working memory using the existing chunking strategy and document the expected peak tensor sizes.
- [x] Keep `max_repr_seqs` as the primary guardrail for replicated-index mode.
- [x] Verification: add tests for memory estimate calculations and guardrail behavior without requiring CUDA.

### Phase 6 Notes

- Added `latents.neg_ctx.memory_guardrail_fraction` and `latents.neg_ctx.fail_on_memory_guardrail` to config, and mirrored them in the distributed manifest `NegativeContextRunConfig`.
- Added `estimate_neg_ctx_ann_memory()` to estimate replicated per-device ANN memory from float32 index bytes, capped `slot_to_id`/`id_to_slot` mapping bytes, and peak chunked search working memory.
- Added `check_neg_ctx_memory_guardrail()` before `TorchANNIndex` allocation; CUDA devices fail or warn when estimated memory exceeds the configured fraction of visible VRAM, while CPU paths are explicitly skipped.
- `NegCtxStats`, `neg_ctx_stats.json`, and `neg_ctx_sanity_report.json` now log `seq_repr` cap metadata, estimated index/query/total memory, guardrail fraction, and guardrail byte limit.
- Query working memory is bounded by the existing `TorchANNIndex.chunk_size`: `4096 * n_stored * 4` bytes on CUDA and `512 * n_stored * 4` bytes on CPU, capped by `n_stored`.
- Verification: `python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py tests/pipeline/test_distributed_manifest.py -q` -> `43 passed`.

## Phase 7 - Optional Index-Sharded Design

- [x] Add an index-sharded design only if replicated `seq_repr` is memory-heavy or query time dominates at H100 scale.
- [x] Split `seq_repr` rows across devices instead of replicating the full index.
- [x] For each query row, collect top candidates from every index shard, then merge shard-local top-K results into global top-K before filtering positives.
- [x] Preserve final `neg_ctx.pt` semantics exactly: rows still contain global sequence IDs and cosine similarities.
- [x] Keep index-sharded mode behind a new explicit backend name, for example `multi_gpu_index_sharded_exact`.
- [x] Verification: prove index-sharded exact output matches replicated-index exact output on synthetic and reduced real-data runs.

### Phase 7 Notes

- Added explicit backend `multi_gpu_index_sharded_exact` to runtime config and distributed manifest validation.
- Added `partition_index_slots()` and `ShardedANNIndex` so `seq_repr` rows are split contiguously across selected CUDA devices instead of replicated on every device.
- Added `merge_shard_search_results()` to merge shard-local top-K similarities and global slot IDs into a single global top-K per query before positive filtering.
- Added a sharded component path that keeps the canonical `neg_ctx.pt` contract: output rows still store global sequence IDs and cosine similarities, independent of the backend.
- Added per-device shard assignments and shard memory estimates to `NegCtxStats`/`neg_ctx_stats.json`; the existing memory guardrail is applied to each shard before allocation.
- Verification: `python -m pytest tests/store/test_neg_context_backend.py tests/pipeline/test_negative_context_stage.py tests/pipeline/test_distributed_manifest.py -q` -> `46 passed`.

## Phase 8 - Output Validation And Reporting

- [x] Add a `neg_ctx` sanity report with backend, devices, sequence cap, fill mean/min/max, zero-negative rows, invalid sequence counts, non-finite similarity counts, and timing breakdown.
- [x] Save the report beside `neg_ctx_stats.json` or include it in the stats JSON with a schema version.
- [x] Print a concise summary suitable for H100 benchmark logs.
- [x] Include sampled row comparisons when running equivalence mode.
- [x] Preserve downstream artifact shape and field names so discovery does not need to know which backend produced `neg_ctx`.
- [x] Verification: add tests for sanity report generation and failure on invalid tensors.

### Phase 8 Notes

- Expanded `neg_ctx_sanity_report.json` with explicit `schema_version`, backend, devices, canonical tensor shape/dtypes, sequence-cap metadata, fill distribution, zero-negative rows, validation counters, timing breakdown, and memory estimates.
- Added validation counters for invalid sequence IDs, non-finite similarities, negative similarities, values without sequence IDs, valid entries, configured row width, and total rows.
- `run_negative_context_stage()` now validates the produced `neg_ctx` tensor before saving `neg_ctx.pt`; invalid outputs write `failed.json` and do not write completed markers or canonical outputs.
- Added a concise `[neg_ctx] summary ...` line for benchmark logs with backend, devices, populated rows, fill min/mean/max, validation counters, sequence cap, estimated ANN memory, and total runtime.
- Equivalence reports now include a schema version and continue to include sampled row comparisons for backend drift/tie-order inspection.
- Verification: `python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py tests/pipeline/test_distributed_manifest.py -q` -> `47 passed`.

## Phase 9 - Testing And Verification

- [x] Run focused unit tests for device parsing, component partitioning, ANN search, `_process_component`, stats merging, memory estimates, and sanity validation.
- [x] Run synthetic single-vs-multi equivalence tests.
- [x] Run one-worker/local compatibility tests using CPU or one CUDA device.
- [ ] Run a reduced real-data smoke after Part 2 artifacts exist.
- [ ] Run H100 benchmarks with 1, 2, 4, and 8 devices once available.
- [x] Document exact verification commands and benchmark configs in this file after implementation.

### Phase 9 Notes

- Added an actual CPU backend smoke for `run_negative_context_stage()` using synthetic merged `top_ctx.pt`, `mid_ctx.pt`, and `seq_repr.pt` artifacts. This exercises the real `single_gpu_exact` backend, canonical `neg_ctx.pt` write, stats write, sanity report, and validation path.
- Focused Part 3 verification: `python -m pytest tests/store/test_neg_context_backend.py tests/pipeline/test_negative_context_stage.py tests/pipeline/test_distributed_manifest.py -q` -> `48 passed`.
- Broader pipeline compatibility verification: `python -m pytest tests/pipeline -q` -> `139 passed`.
- Synthetic equivalence coverage is represented by `test_compare_negative_context_backends_writes_equivalence_report`, `test_build_negative_context_comparison_report_detects_equivalent_outputs`, and `test_build_negative_context_comparison_report_reports_differences`; real CUDA multi-backend equivalence still needs a CUDA/H100 run.
- Reduced real-data smoke was not run in this environment because no reduced merged Part 2 artifact set was selected for this verification pass.
- H100 1/2/4/8-device benchmarks were not run locally; benchmark config should use the target run config with `latents.neg_ctx.backend` set to `single_gpu_exact`, `multi_gpu_exact`, and `multi_gpu_index_sharded_exact`, recording `neg_ctx_stats.json`, `distributed/parts/neg_ctx/neg_ctx_sanity_report.json`, and `neg_ctx_equivalence_report.json` where applicable.

---

## Open Questions

- Should Part 3 run inside the distributed controller, or remain a standalone command that consumes merged Part 2 artifacts?
- What fill-row equality threshold is acceptable when CUDA top-K tie ordering differs across devices?
- Is replicated-index memory acceptable with `max_repr_seqs: 200000` on the target 8x H100 node?
- Should `neg_ctx_stats.json` become versioned before adding more backend-specific fields?

## Risks / Assumptions

- `neg_ctx` must be built from global merged `top_ctx`, `mid_ctx`, and `seq_repr`; building worker-local negatives before merge would change semantics.
- Replicated-index `multi_gpu_exact` is simple and exact, but may waste VRAM when `seq_repr` grows.
- CPU write-back can dominate if ANN search is already fast, limiting the benefit of multi-GPU query compute.
- Capped `seq_repr` must use the deterministic global cap from Part 2; per-worker caps would make negatives inconsistent.
- Downstream discovery assumes the canonical `neg_ctx.pt` shape and should not need backend-specific branches.
