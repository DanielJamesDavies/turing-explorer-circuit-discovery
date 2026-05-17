# Plan: H100 Pipeline Scaling And Bottleneck Reduction

> **Goal:** Make the Turing Explorer pipeline faster and more scalable on large single-node GPU machines, especially 8x H100 systems, while preserving current artifact formats and adding new experimental modes behind config toggles.
>
> **Created:** 2026-04-25

---

## Background

The current codebase should run well on a single H100, but it is not yet designed to make full use of an 8x H100 node. The model itself is small enough to fit comfortably on one H100, so the most promising scaling direction is not transformer model parallelism. The better target is pipeline parallelism and map-reduce style work splitting across dataset shards, SAE components, candidate seeds, and negative-context construction.

There are also several non-GPU bottlenecks:

- Sequential data loading still reparses shards instead of fully using the existing `.shard_indices` mmap path.
- Search-cache generation is tokenizer/Pandas/I/O-heavy and currently runs in the main persistence path when enabled.
- Top-coactivation reduction is a single C++/OpenMP monolith, but the local baseline shows the second-pass dump path is even more expensive than the reducer at small scale.
- Negative-context retrieval uses GPU matmul/top-k, but only one device. The local baseline shows compute is already fast at small scale, while write-back dominates.
- Mid-context reservoir sampling is CPU/OpenMP and exact/unbiased, but potentially slow.
- Persistence and artifact generation can create memory spikes and long tail latency at the end of the run.
- Model/SAE reload after negative-context construction is a large hidden fixed cost on local hardware.

The guiding principle for this plan is: preserve existing behavior as the default, then add toggled modes so speed/quality trade-offs can be measured without destabilizing current research results.

---

## Phase 1 — Baseline Profiling And Guardrails

- [x] Add or confirm lightweight timing logs around data loading, first pass, negative context build, second pass dump, top-coactivation reduce, candidate selection, discovery, persistence, and search-cache generation.
- [x] Record a single-GPU baseline on the current local machine or a small H100 test run using a reduced config (`n_shards: 1-4`, modest `n_seeds`) before changing behavior.
- [x] Record peak CPU RAM, peak VRAM, and wall-clock time for each major phase.
- [x] Confirm current full unit suite passes before performance changes: run `pytest -q`.
- [x] Document the baseline numbers in this plan or a sibling benchmark note before starting Phase 2.

### Phase 1 Notes

- Added timing/resource logging in `src/observability/timing.py`, including phase wall-clock duration, best-effort CPU RSS, and CUDA allocation/reservation peaks when CUDA is available.
- Wrapped the main pipeline phases in `phase_timer(...)` from `src/pipeline/run_pipeline.py`.
- Added focused timing logs for DataLoader/model/SAE initialization, candidate scoring/save, first-pass artifact saves, search-cache generation, and second-pass top-coactivation dump/reduce/save.
- Verified tests in WSL with the project venv using `python -m pytest -q`: `509 passed, 14 warnings in 93.39s`.
- The warnings are existing Torch JIT deprecation warnings from `tests/circuit/test_sparse_act.py`.
### Baseline Run — Local Single GPU

Configuration:

- Environment: WSL, project `.venv`
- GPUs detected: `1`
- `data.n_shards: 4`
- `discovery.n_seeds: 16`
- Discovery method: `counterfactual_gradient`
- Seed criterion: `stratified_random`
- Top-coactivation mode: `pmi`

Major phase timings:

| Phase | Wall Time | End CPU RSS | Peak VRAM Alloc | Peak VRAM Reserved |
|---|---:|---:|---:|---:|
| Initialize runtime | 234.6 ms | 830 MB | 0.00 GB | 0.00 GB |
| Initialize resources / data loading | 1m 41.2s | 4,123 MB | 3.61 GB | 3.62 GB |
| First pass: latent stats and contexts | 2m 50.2s | 4,697 MB | 8.38 GB | 10.49 GB |
| Persistence: first-pass outputs | 59.37s | 5,596 MB | 4.32 GB | 10.49 GB |
| Offload stores to CPU | 351.9 ms | 6,496 MB | 4.25 GB | 10.49 GB |
| Offload model and SAE | 249.1 ms | 3,610 MB | 3.30 GB | 10.49 GB |
| Negative context build | 11.12s | 5,105 MB | 2.30 GB | 2.67 GB |
| Reload model and SAE | 1m 11.9s | 8,021 MB | 3.61 GB | 3.65 GB |
| Second pass: top coactivation | 2m 32.7s | 5,918 MB | 7.79 GB | 9.60 GB |
| Candidate selection | 10.7 ms | 5,918 MB | 0.39 GB | 4.23 GB |
| Discovery | 1m 42.7s | 5,980 MB | 13.62 GB | 13.88 GB |

Detailed observations:

- DataLoader initialization itself was fast: `61.7 ms`; model init was `22.50 s`; SAE bank init dominated resource initialization at `1m 18.6s`.
- First pass processed `64` batches in `2m 50.2s`, about `2.66 s/batch`.
- Search cache was a noticeable one-time tail inside persistence: `54.85 s` of the `59.37 s` persistence phase.
- Negative context build was fast overall: `8.1 s` internal total, `11.12 s` phase wall time. The slowest internal step was write-back: `6485.9 ms`.
- Second pass processed `64` top-coactivation dump batches in `130.10 s`, about `2.03 s/batch`.
- Top-coactivation C++ reduction took `14.65 s` internally, with `13.79 s` in parallel reduce using `8` OpenMP threads.
- Top-coactivation save took `3.18 s`.
- Discovery found `11` faithful circuits from `16` seeds, with `258` forward passes and `177.04 s` total forward time reported by the discovery window.
- The highest observed phase peak VRAM allocation was discovery at `13.62 GB`, close to the local 16 GB budget.

## Phase 2 — Indexed Sequential Data Loading

- [x] Update `src/data/loader.py` so `DataLoader.get_batches()` uses the existing per-shard `.shard_indices` mmap path instead of reparsing shards with `np.split`.
- [x] Preserve current global sequence IDs exactly: IDs remain 1-indexed and increase across shards in the same order.
- [x] Preserve current `skip_first_token`, padding, `max_length`, `pin_memory`, and `device` behavior.
- [x] Avoid loading an entire shard into Python lists during normal sequential batching.
- [x] Add unit tests with small synthetic `.npy` shards containing `-1` separators to compare old-style sequence output against indexed sequential output.
- [x] Add tests for shard remainder behavior, empty shards, `skip_first_token=True`, `skip_first_token=False`, `get_sequence()`, and `get_batches_by_ids()`.
- [x] Verify with `pytest tests/` and a small pipeline smoke run.

### Phase 2 Notes

- `DataLoader.get_batches()` uses the same indexed mmap sequence access pattern as `get_batches_by_ids()`: it streams each sequence slice from the per-shard `(start, end)` index and emits global sequence IDs as `start_id + local_idx`.
- `DataLoader.load_shard()` and `get_sequence()` also use the cached index path, so callers no longer need the old `np.split` parse path for normal sequence access.
- Added `tests/test_data_loader.py` with synthetic `.npy` shards covering `-1` separators, shard remainders, empty shards, both `skip_first_token` modes, global ID continuity, `get_sequence()`, and `get_batches_by_ids()`.
- Focused verification: `python -m pytest tests/test_data_loader.py -q` -> `4 passed`.
- Full local verification from PowerShell: `python -m pytest -q` -> `513 passed in 6.15s`.
- Small pipeline smoke verification: the earlier WSL `./scripts/run.sh` completed successfully with this indexed loader path active.

## Phase 3 — Persistence, Search-Cache Decoupling, And Reload Policy

- [x] Keep `persist.search_cache_enabled`, but make the main pipeline clearly skip search-cache work when disabled.
- [x] Add a standalone command/script for building `outputs/search_cache.parquet` from existing saved stores, e.g. `scripts/build_search_cache.sh` and/or `python src/build_search_cache.py`.
- [x] Change the recommended large-run config so search-cache generation happens offline after the main pipeline, not inside `save_results()`.
- [x] Add per-artifact timing logs to `save_results()` so long tail saves are visible.
- [x] Save artifacts atomically where practical: write to a temporary path and rename after success.
- [x] Ensure store saves do not accidentally trigger large GPU allocations or unnecessary GPU-to-CPU round trips.
- [x] Consider adding `persist.atomic_saves: true` and `persist.build_search_cache_after_pipeline: false` config keys, keeping current behavior compatible unless explicitly changed.
- [x] Add a high-VRAM mode/config option that can skip full model/SAE offload and reload around negative-context construction when memory permits.
- [ ] Benchmark `offload/reload` versus `keep_model_loaded` on H100-class VRAM. Local baseline reload cost was `1m 11.9s`.
- [x] Add tests or smoke checks that disabled search cache does not call tokenizer/Pandas generation and that standalone generation still writes a valid Parquet file.
- [x] Verify with focused tests and `pytest -q`.

### Phase 3 Notes

- Added `persist.build_search_cache_after_pipeline`; when `true`, search-cache generation runs as its own phase at pipeline end, and when `false`, the main pipeline preserves `search_cache_enabled: true` but defers cache generation to the standalone script.
- Added `scripts/build_search_cache.sh` and `src/build_search_cache.py` to build `outputs/search_cache.parquet` from `outputs/top_ctx.pt` and the configured token shards after the pipeline.
- Added `persist.atomic_saves`; first-pass artifacts now save to `*.tmp` and are atomically renamed where practical.
- Added `hardware.keep_model_loaded_for_neg_ctx`; when true, the pipeline skips model/SAE offload before negative-context construction and therefore avoids the later reload if memory permits.
- Updated `config.yaml` for the large-run recommendation: `build_search_cache_after_pipeline: false`, `atomic_saves: true`, and `keep_model_loaded_for_neg_ctx: false` by default.
- Focused verification: `python -m pytest tests/test_persist_phase3.py -q` -> `5 passed`.
- Standalone command smoke check: `python src/build_search_cache.py --help` succeeds.
- Full local verification from PowerShell: `python -m pytest -q` -> `518 passed in 9.16s`.
- Remaining H100-specific work: benchmark `hardware.keep_model_loaded_for_neg_ctx: true` on H100-class VRAM before recommending it as a default.

## Phase 4 — GPU-Friendly MidCtx Mode

- [x] Add config schema for mid-context update mode:
  ```yaml
  latents:
    mid_ctx:
      mode: "reservoir_cpu"  # "reservoir_cpu" | "gpu_topk_mid"
  ```
- [x] Keep `reservoir_cpu` as the default and preserve the current C++ reservoir behavior.
- [x] Implement `gpu_topk_mid` as an experimental mode that selects in-band examples closest to the midpoint of the configured sigma band, using GPU-friendly tensor operations.
- [x] Preserve saved `mid_ctx.pt` shape and field names so downstream code can read either mode.
- [x] Store generation metadata in the checkpoint where practical, e.g. `ctx_type`, `mode`, `band_low_sigma`, `band_high_sigma`, and reservoir/topk parameters.
- [x] Add deterministic unit tests for `gpu_topk_mid` on synthetic latents where the expected closest-to-midpoint sequences are obvious.
- [x] Add comparison tests that both modes produce valid `ctx_seq_idx` and `ctx_seq_val` tensors with the same shapes.
- [x] Benchmark `reservoir_cpu` versus `gpu_topk_mid` on a small real shard and record speed plus qualitative differences in selected contexts.

### Phase 4 Notes

- Added `latents.mid_ctx.mode` with allowed values `reservoir_cpu` and `gpu_topk_mid`; `reservoir_cpu` remains the default in `config.yaml`.
- `gpu_topk_mid` keeps the existing CPU artifact/storage shape, but does the score, band mask, midpoint-distance ranking, and top-k selection with tensor operations on the activation device.
- `gpu_topk_mid` stores the in-band sequences closest to the midpoint of `[band_low_sigma, band_high_sigma]`, rather than doing unbiased reservoir sampling across all in-band examples.
- `mid_ctx.pt` now includes metadata fields: `ctx_type`, `mode`, `band_low_sigma`, `band_high_sigma`, and `num_ctx_sequences`, while retaining `ctx_seq_idx`, `ctx_seq_val`, `reservoir_fill`, and `reservoir_n`.
- Focused verification: `python -m pytest tests/store/test_mid_ctx_modes.py -q` -> `3 passed`.
- Full local verification from PowerShell: `python -m pytest -q` -> `521 passed in 8.38s`.
- Benchmark config: local RTX 5070 Ti, `n_shards: 4`, `n_seeds: 16`, `search_cache_enabled: false`, and `warmup_batches: 0` so both modes actually populate `mid_ctx` during the reduced 64-batch run.
- Benchmark artifacts:
  - `runs/phase4_midctx/reservoir_cpu_warmup0.log`
  - `runs/phase4_midctx/gpu_topk_mid_warmup0.log`
  - `runs/phase4_midctx/mid_ctx_reservoir_cpu_warmup0.pt`
  - `runs/phase4_midctx/mid_ctx_gpu_topk_mid_warmup0.pt`
- Local benchmark result:
  - `reservoir_cpu`: first pass `3m 12.4s`, negative context `13.99s`, second pass `2m 50.5s`, discovery `1m 30.8s`, faithful circuits `6`, mid_ctx fill rate `0.7850`.
  - `gpu_topk_mid`: first pass `3m 35.2s`, negative context `13.70s`, second pass `2m 41.3s`, discovery `1m 30.6s`, faithful circuits `6`, mid_ctx fill rate `0.7637`.
- Interpretation: on the local 5070 Ti reduced run, `gpu_topk_mid` is not faster; it adds about `22.8s` to first pass. It may still behave differently on H100-class hardware, but it should remain experimental rather than recommended as the default.
- Config was restored after the benchmark to `mode: "reservoir_cpu"` and `warmup_batches: 100`; `search_cache_enabled` remains `false` per the current testing setup.

## Phase 5 — Negative Context Multi-GPU Backend

- [x] Add config schema for negative-context backend:
  ```yaml
  latents:
    neg_ctx:
      backend: "single_gpu_exact"  # "single_gpu_exact" | "multi_gpu_exact"
      devices: []                  # empty = all visible CUDA devices
  ```
- [x] Keep the current exact single-device implementation as `single_gpu_exact`.
- [x] First optimize write-back in the existing single-device backend, because the local baseline spent `6485.9 ms` of `8062.4 ms` internal negative-context time in write-back.
- [x] Implement `multi_gpu_exact` as a component-parallel mode first: split SAE components across selected GPUs while replicating the ANN index on each device.
- [x] Preserve the current `neg_ctx.pt` artifact shape and semantics.
- [x] Ensure each worker writes only its assigned component slices, then merge or assign the slices into the shared `neg_ctx` store deterministically.
- [x] Add device selection support for explicit CUDA IDs, while keeping `hardware.ann_device` backward-compatible.
- [x] Add logging that reports selected devices, component assignment, per-device timing, and fill-rate summary.
- [x] Add tests for component partitioning, device-list parsing, and merge correctness using CPU/mock devices where possible.
- [ ] Benchmark single-GPU exact versus multi-GPU exact on a larger multi-GPU run and confirm identical or near-identical negatives for the same inputs.
- [x] Treat multi-GPU negative context as lower priority for small/capped runs unless profiling shows the ANN query path, not write-back, dominates.

### Phase 5 Notes

- Added `latents.neg_ctx.backend` with allowed values `single_gpu_exact` and `multi_gpu_exact`; `single_gpu_exact` remains the default in `config.yaml`.
- Added `latents.neg_ctx.devices`; empty means all visible CUDA devices for `multi_gpu_exact`, while explicit entries like `[0, 1]` or `["cuda:0", "cuda:1"]` select a subset.
- Extended `hardware.ann_device` parsing for the single-device backend to accept `cuda` and explicit `cuda:N` values in addition to the existing `auto`, `gpu`, and `cpu` behavior.
- Optimized single-device write-back by transferring and assigning only active latent rows instead of building and copying a full component-sized GPU tensor. The resulting `neg_ctx.pt` shape and row semantics are unchanged.
- Implemented `multi_gpu_exact` as component-parallel exact search: each selected GPU builds its own exact ANN index, owns a deterministic round-robin subset of components, and writes only those component slices into the shared CPU `neg_ctx` store.
- Added `NegCtxStats.backend` and `NegCtxStats.devices` so `outputs/neg_ctx_stats.json` records the backend/device choice.
- Added focused tests in `tests/store/test_neg_context_backend.py` covering device-list parsing, deterministic component partitioning, stats merging, and CPU/mock exact retrieval/write-back semantics.
- Verification: `python -m pytest tests/store/test_neg_context_backend.py -q` -> `4 passed`.
- Full local verification from PowerShell: `python -m pytest -q` -> `525 passed in 8.36s`.
- Remaining Phase 5 work is hardware validation: run `single_gpu_exact` versus `multi_gpu_exact` on the 8x H100 node and compare `neg_ctx.pt` fill stats, sampled rows, and total negative-context timing. Local single-GPU hardware cannot validate the multi-GPU path.

## Phase 6 — Top-Coactivation Dump And Reducer Improvements

- [x] Add clearer timings for second-pass candidate dump internals, inverted-index construction, output allocation, reduce, and PMI postprocess.
- [x] Profile `TopCoactivation.update_batch()` in `src/store/top_coactivation.py`, because the local baseline spent `130.10 s` in the dump path versus `14.65 s` in C++ reduction.
- [x] Optimize second-pass dump before deeper reducer work if profiling shows Python/tensor overhead in `update_batch()` dominates.
- [x] Expose or document OpenMP thread controls such as `OMP_NUM_THREADS` and reducer schedule/chunk size.
- [ ] Improve the C++ inverted-index build path in `src/native/top_coactivation_reduce.cpp` where practical:
  - [x] Parallelize counting if it is a measurable bottleneck.
  - [x] Parallelize or improve write-position construction without introducing races.
  - [ ] Reduce memory churn from large vectors where possible.
  - [x] Keep the Python extension API compatible for the existing reduce path.
- [x] Add native extension tests for reducer determinism and equivalence against a small Python reference implementation.
- [ ] Benchmark before/after dump and reducer timings on representative small and medium runs.
- [ ] Verify native extension build and tests on the target Linux/H100 environment.

### Phase 6 Notes

- Added aggregate `TopCoactivation.update_batch()` dump profiling for dense allocation, scatter/mean, scoring, component top-k, global top-k, CPU transfer, row lookup, and CPU write.
- Added reducer controls under `latents.top_coactivation`: `dump_profile`, `reduce_omp_threads`, and `reduce_schedule_chunk`. `OMP_NUM_THREADS` remains respected when no explicit override is set.
- Extended `top_coactivation_reduce.reduce_topk()` with backward-compatible optional args for OpenMP thread override, dynamic schedule chunk size, and timing prints. Python falls back to the legacy extension signature if an older reducer is still built.
- Split C++ inverted-index timing into target counting, prefix offsets, and row filling; parallelized target counting and row filling with atomic updates/captures.
- Added `TURING_NATIVE_CPU_ONLY=1` to `src/native/setup.py` so CPU extensions can be rebuilt locally when CUDA/PyTorch versions are mismatched.
- Added a native reducer reference-equivalence test against a small Python implementation and gated the previous full-scale reducer benchmark behind `RUN_FULL_SCALE_REDUCE_BENCHMARK=1`.
- Local verification:
  - `python -m pytest tests/store/test_top_coactivation_modes.py -q` -> `6 passed`
  - `TURING_NATIVE_CPU_ONLY=1 python setup.py build_ext --inplace` in WSL -> rebuilt CPU native extensions
  - `python src/native/tests/test_reduce.py` in WSL -> all reducer tests passed
  - `python -m pytest -q` -> `525 passed`
- Real reduced pipeline benchmark with dump profiling:
  - First pass: `2m 56.2s`
  - Negative context: `10.82s`
  - Second pass total: `2m 49.1s`
  - Top-coactivation dump: `144.30s`
  - Top-coactivation update measured total: `142.97s`
  - GPU-to-CPU transfer inside `update_batch()`: `142.71s`
  - Scatter/mean: `0.05s`; component top-k: `0.13s`; CPU write: `0.01s`
  - Reducer + PMI postprocess: `17.21s`
  - Conclusion: current dump bottleneck is the per-batch transfer/synchronization path, not dense allocation, scatter, top-k, or C++ reduction.
- Implemented opt-in GPU-resident candidate dump via `latents.top_coactivation.dump_device`.
  - Current `config.yaml`: `dump_device: "gpu"`
  - Focused verification: `python -m pytest tests/store/test_top_coactivation_modes.py -q` -> `6 passed`; `python -m pytest -q` -> `525 passed`
  - Reduced pipeline benchmark with GPU dump:
    - Candidate dump allocated on `cuda:0`: `67.1 MB`
    - Top-coactivation dump: `141.22s` versus previous CPU-dump `144.30s`
    - Second pass total: `2m 44.6s` versus previous `2m 49.1s`
    - Reported per-batch CPU transfer fell to `0.00s`, but wall-clock sync moved into GPU row lookup/write timing (`139.86s`), so the real bottleneck remains the per-batch CUDA synchronization boundary.
  - Conclusion: GPU-resident dump is compatible and slightly faster, but the bigger Phase 6 opportunity is reducing per-batch synchronous GPU work or moving reduction closer to the GPU path.
- Added direct dump-row writes from the second-pass batch order, avoiding per-batch sequence-id lookup in the hot path.
  - Focused verification: `python -m pytest tests/store/test_top_coactivation_modes.py -q` -> `6 passed`; full suite -> `525 passed`
  - Reduced pipeline benchmark after direct-row write:
    - Top-coactivation dump: `133.14s`
    - Top-coactivation update measured total: `0.21s`
    - Second pass total: `2m 40.4s`
    - Previous GPU-dump benchmark was `141.22s` dump / `2m 44.6s` second pass.
  - Conclusion: the direct-row path is a real small win and confirms most remaining dump wall time is model/SAE forward work, not candidate materialization.
- Latest artifact sanity validation after GPU dump + direct-row write:
  - `latent_stats.pt`, `top_ctx.pt`, `mid_ctx.pt`, `neg_ctx.pt`, `top_coactivation.pt`, `candidates.pt`, and `neg_ctx_stats.json` loaded successfully.
  - Validated expected shapes, dtypes, finite values, valid sequence/latent ID ranges, PMI clamp bounds, candidate count, and negative-context fill stats.
  - Summary: `36` components, `40960` SAE width, `1,474,560` latents, max context seq id `32,768`, `top_coactivation.pt` shape `(36, 40960, 64)`, PMI mode, `94,371,835` nonzero coactivation values, `16` candidates, `1,327,373` populated negative-context rows, fill mean `64.0`.
  - Result: all artifact sanity checks passed.
- Local full CUDA native rebuild was blocked by environment mismatch: detected CUDA `12.0` versus PyTorch CUDA `13.0`. Target H100 rebuild/benchmark remains pending.

## Phase 7 — Shardable Top-Coactivation Reduction

- [x] Design a target-range reducer API that can reduce a subset of global target latent IDs, e.g. `[target_start, target_end)`.
- [x] Add config toggles:
  ```yaml
  latents:
    top_coactivation:
      reduce_backend: "single_process"  # "single_process" | "target_sharded"
      reduce_shards: 1
      reduce_shard_output_dir: null
  ```
- [x] Implement target-sharded reduction without changing final `top_coactivation.pt` shape.
- [x] Allow shard outputs to be written as temporary partial files and concatenated/merged after all shards complete.
- [x] Keep the original single-process reducer as the default.
- [x] Add tests proving sharded reduction equals single-process reduction on synthetic candidate dumps.
- [x] Add failure cleanup behavior for partial shard files.
- [x] Add a repeatable benchmark runner for larger target-sharded runs.
- [ ] Run the larger/H100 target-sharded benchmark and record timings.

### Phase 7 Notes

- Added `latents.top_coactivation.reduce_backend` with allowed values `single_process` and `target_sharded`; `single_process` remains the default in `config.yaml`.
- Added `latents.top_coactivation.reduce_shards`; current default is `1`.
- Added optional `latents.top_coactivation.reduce_shard_output_dir`; when set with `target_sharded`, each target range is written as an atomic `.pt` shard and then merged into the unchanged final `[C, d_sae, K]` tensors.
- Added failure cleanup for file-backed target shards: if a shard reduction or save fails, shard files for the current attempted run plus matching temp files are removed, while unrelated files in the directory are left alone.
- Extended the native `top_coactivation_reduce.reduce_topk()` API with optional `target_start` and `target_end` args. The full-range default remains backward-compatible at the Python call site, while range calls return flattened `[target_end - target_start, K]` shard outputs.
- Added deterministic tie ordering in the native reducer (`value desc`, then candidate ID asc) so full-range and target-range reductions stitch identically when candidate scores tie.
- Implemented sequential target-sharded reduction in `TopCoactivation.reduce()`: split flattened target IDs into ranges, reduce each range, and stitch back to final `[C, d_sae, K]` tensors without changing `top_coactivation.pt` shape.
- Added tests:
  - native `test_target_range_equivalence_small()` proves stitched target ranges equal full reduction on a synthetic dump,
  - store-level `test_target_sharded_reduce_stitches_flat_ranges()` proves Python range dispatch/stitching,
  - store-level `test_target_sharded_reduce_can_write_and_merge_partial_files()` proves partial shard files can be written and merged,
  - store-level `test_target_sharded_reduce_cleans_current_partial_files_on_failure()` proves failed file-backed reductions clean up current-run shard artifacts.
- Added `scripts/benchmark_top_coactivation_shards.sh` to run a single-process baseline plus configurable `target_sharded` shard counts, tee logs under `runs/top_coactivation_shard_bench/`, and restore `config.yaml` on exit.
- Verification:
  - `TURING_NATIVE_CPU_ONLY=1 python setup.py build_ext --inplace` in WSL -> rebuilt CPU native reducer,
  - `python src/native/tests/test_reduce.py` in WSL -> all native reducer tests passed,
  - `python -m pytest tests/store/test_top_coactivation_modes.py -q` -> `9 passed`,
  - `bash -n scripts/benchmark_top_coactivation_shards.sh` in WSL -> passed,
  - `python -m pytest -q` -> `528 passed`.
- Reduced WSL pipeline smoke with `reduce_backend: "target_sharded"` and `reduce_shards: 4` completed successfully:
  - target ranges: `[0, 368640)`, `[368640, 737280)`, `[737280, 1105920)`, `[1105920, 1474560)`,
  - top-coactivation dump: `129.65s`,
  - reduce + PMI postprocess: `18.07s`,
  - second pass total: `2m 37.2s`,
  - final artifact sanity check passed: `top_coactivation.pt` shape `(36, 40960, 64)`, PMI mode, `94,371,835` nonzero coactivation values.
- Remaining Phase 7/8 work is now benchmarking rather than correctness: run the larger/H100 shard benchmark and use the results to choose recommended shard counts and CPU thread settings.

## Phase 8 — H100 Validation And Operating Guide

- [ ] Rebuild and validate native extensions on the target Linux/H100 host before any pipeline benchmark.
- [ ] Run a single-H100 control benchmark with conservative current-runtime settings.
- [ ] Benchmark high-VRAM single-H100 mode with `hardware.keep_model_loaded_for_neg_ctx: true`.
- [ ] Benchmark `latents.neg_ctx.backend: "multi_gpu_exact"` with 2, 4, and 8 visible H100s.
- [ ] Benchmark `latents.top_coactivation.reduce_backend: "target_sharded"` with representative shard counts.
- [ ] Compare artifact shapes, summary stats, sampled rows, and circuit outputs between default and experimental modes before recommending any new default.
- [ ] Add startup warnings when `hardware.multi_gpu: true` is enabled but one or more phases still use only the primary device.
- [ ] Document which phases use the model GPU, ANN GPUs, CPU/OpenMP, GPU dump storage, and file-backed reducer shards.
- [ ] Publish recommended H100 configs for:
  - [ ] single-H100 stable baseline,
  - [ ] single-H100 high-VRAM variant,
  - [ ] 8x H100 current-runtime variant using validated opt-in modes,
  - [ ] conservative research default that favors reproducibility over maximum throughput.
- [ ] Use the H100 benchmark results to decide whether the separate distributed-runtime plan is justified.

### Phase 8 Next Steps

The Phase 8 goal is not to build a new distributed runtime. It is to turn the current opt-in modes into a clear, benchmarked H100 operating guide and make the limits of the current runtime explicit.

1. **Rebuild and validate native extensions on the H100 host.**
   - Build `src/native` against the target machine's PyTorch/CUDA toolchain.
   - Run native top-k and reducer tests before any pipeline benchmark.
   - Confirm Triton/cublasLt/C++ reducer startup messages match the expected accelerated paths.

2. **Run a single-H100 baseline first.**
   - Use the current reduced benchmark config as the control.
   - Record first pass, persistence, negative context, reload, top-coactivation dump, top-coactivation reduce, discovery, total wall time, peak VRAM, and CPU RSS.
   - Keep `search_cache_enabled: false` for pipeline timing unless explicitly benchmarking search cache.

3. **Benchmark high-VRAM single-H100 mode.**
   - Enable `hardware.keep_model_loaded_for_neg_ctx: true`.
   - Compare total wall time and peak VRAM against the baseline.
   - Only recommend this mode if it saves meaningful reload time without crowding the model/SAE/discovery memory budget.

4. **Benchmark multi-GPU negative context.**
   - Test `latents.neg_ctx.backend: "multi_gpu_exact"` with 2, 4, and 8 visible H100s.
   - Track whether the current replicated-index approach is still small enough with the configured `seq_repr.max_repr_seqs`.
   - Record the negative-context timing breakdown separately, because this is where multi-GPU benefit should show most clearly.

5. **Benchmark target-sharded top-coactivation.**
   - Use `scripts/benchmark_top_coactivation_shards.sh`.
   - Sweep shard counts such as `2`, `4`, `8`, and `16`.
   - Record top-coactivation reduce time, postprocess time, shard-file write/merge overhead, total second-pass time, and final artifact sanity checks.

6. **Add startup warnings and device-use documentation.**
   - Make it explicit when `hardware.multi_gpu: true` is enabled but a phase still runs on a single primary device.
   - Document which stages use model GPU, ANN GPUs, CPU/OpenMP, and file-backed shard outputs.
   - Avoid implying the whole pipeline is 8-GPU data-parallel until a distributed runner exists.

7. **Publish recommended H100 configs.**
   - Single-H100 stable baseline.
   - Single-H100 high-VRAM variant.
   - 8x H100 current-runtime variant using multi-GPU negative context plus target-sharded top-coactivation.
   - A conservative research default that favors correctness/reproducibility over maximum throughput.

8. **Decide whether to start the distributed-runtime plan.**
   - Use measured full or medium-scale H100 timings to decide whether first-pass and second-pass model/SAE forward work dominate enough to justify controller/worker orchestration.
   - If justified, continue in `agent-planning/distributed-runtime-additions.md`; otherwise keep the current-runtime operating guide as the recommended H100 path.

## Phase 9 — Testing And Verification

- [x] Run focused unit tests for each changed subsystem after its phase:
  - [x] `tests` covering `DataLoader`,
  - [x] `tests/store` for context and coactivation stores,
  - [x] `tests/native` for native reducer behavior,
  - [x] focused tests for negative-context backend partition/merge logic.
- [x] Run the full suite after each major phase: `pytest -q`.
- [ ] Build native extensions on Linux/H100 after C++ changes:
  ```bash
  cd src/native
  python setup.py build_ext --inplace
  ```
- [ ] Run native tests after rebuild:
  ```bash
  python src/native/tests/test_topk.py
  python src/native/tests/test_reduce.py
  ```
- [x] Run a small pipeline smoke test with default modes to confirm behavior remains backward-compatible.
- [ ] Run a small pipeline smoke test with each new experimental mode enabled individually.
- [ ] Compare artifact shapes and basic stats between default and experimental modes:
  - [ ] `latent_stats.pt`,
  - [ ] `top_ctx.pt`,
  - [ ] `mid_ctx.pt`,
  - [ ] `neg_ctx.pt`,
  - [ ] `top_coactivation.pt`,
  - [ ] `outputs/circuits/summary.json`.
- [ ] Record performance deltas and any quality differences in discovered circuits before recommending a new default.

### Verification Notes

- Focused tests added and run for data loading, persistence/search-cache behavior, mid-context modes, and negative-context backend partition/merge/write-back behavior.
- Full-suite checkpoints passed during implementation:
  - Phase 2: `513 passed`
  - Phase 3: `518 passed`
  - Phase 4: `521 passed`
  - Phase 5: `525 passed`
- Reduced WSL pipeline smoke tests completed successfully before Phase 5, including the Phase 4 `reservoir_cpu` vs `gpu_topk_mid` benchmark runs.
- Post-Phase-6 reduced WSL pipeline smoke tests completed successfully with `latents.top_coactivation.dump_device: "gpu"` and direct dump-row writes.
- Still pending: H100 multi-GPU benchmark, native CUDA rebuild/tests on the target H100 environment, medium/large benchmark coverage, and full artifact/quality comparison before changing broader defaults.

---

## Open Questions

- Should `gpu_topk_mid` use closest-to-band-midpoint examples, random in-band examples, or a hybrid of both?
- Should `multi_gpu_exact` replicate the ANN index on every GPU initially, or should we go straight to an index-sharded design for very large `seq_repr` stores?
- How large are the real production datasets on the planned 8x H100 run, and does `seq_repr.max_repr_seqs` remain capped at `200000`?
- Should search-cache generation remain in `save_results()` when enabled, or should it always be a separate command after this refactor?
- Which `target_sharded` reducer shard counts, OpenMP thread counts, and schedule chunks are best on the target H100 host?
- Do Phase 8 full or medium-scale timings justify starting `agent-planning/distributed-runtime-additions.md`, or are the current-runtime modes sufficient for now?

## Risks / Assumptions

- Existing artifact formats are assumed to be valuable and should remain compatible unless explicitly changed.
- Existing default behavior should remain stable; new faster modes should be opt-in until benchmarked.
- GPU-friendly `mid_ctx` will not be statistically identical to reservoir sampling, so speed improvements must be evaluated alongside downstream circuit quality.
- Replicating the ANN index across all GPUs is simple and likely fine for capped `seq_repr`, but may become memory-heavy if the cap is removed.
- Top-coactivation CUDA reduction is intentionally out of scope for this plan; the plan focuses on C++ improvements and shardability first.
- Data loading and I/O may become less visible on H100 only after GPU compute is accelerated; profiling should be repeated after each major phase.
