# Plan: Part 4 - Distributed Pass 2 Dump

> **Goal:** Replay global `top_ctx` sequences across independent workers and write exact per-sequence top-coactivation candidate dumps that can be reduced by Part 5 into the canonical `top_coactivation.pt`.
>
> **Created:** 2026-05-16

---

## Scope

This part starts after Part 3 has produced:

- `outputs/<run_id>/latent_stats.pt`
- `outputs/<run_id>/top_ctx.pt`
- `outputs/<run_id>/neg_ctx.pt`

Only `latent_stats.pt` and `top_ctx.pt` are required for pass-2 dump itself; `neg_ctx.pt` is part of the full pipeline ordering and should already be available before pass 2.

This part runs model+SAE forwards over the global replay sequence list:

```text
top_ctx_sequence_ids = unique sequences referenced by global top_ctx
```

It writes worker-local candidate dump artifacts. It does **not** run the global top-coactivation reducer, apply PMI postprocess, select candidates, or run discovery.

---

## Phase 1 - Dump Semantics And Refactor Boundary

- [x] Treat the current `TopCoactivation.update_batch()` output as the exact semantic contract for this part.
- [x] Separate "compute candidate profile for a batch" from "write candidate profile into a process-local dump buffer" so distributed workers can reuse the same scoring logic.
- [x] Preserve current scoring modes: `raw`, `freq_weighted`, and `pmi`.
- [x] Preserve current candidate IDs as flattened global latent IDs: `component_idx * d_sae + latent_idx`.
- [x] Preserve `M = min(n_latents_per_latent * 4, num_components * n_candidates_per_component)` unless an explicit schema version changes it later.
- [x] Verification: add unit tests proving the refactored candidate-profile helper matches current `update_batch()` output for `raw`, `freq_weighted`, and `pmi`.

### Phase 1 Notes

- Added `CandidateProfile` and `TopCoactivation.compute_candidate_profile()` in `src/store/top_coactivation.py`.
- `compute_candidate_profile()` now owns the per-batch scoring contract: dense per-sequence aggregation, `raw`/`freq_weighted`/`pmi` scoring, per-component top-N, global top-M, and flattened candidate IDs.
- `TopCoactivation.update_batch()` now calls the helper and only handles PMI token accounting, row lookup, and writing candidate IDs/values into the prepared dump buffers.
- Verification covers `raw`, `freq_weighted`, and `pmi` helper output against the existing `update_batch()` dump tensors.
- Verification: `python -m pytest tests/store/test_top_coactivation_modes.py -q` -> `12 passed`.

## Phase 2 - Global Replay Sequence List

- [x] Build the replay list from merged global `top_ctx.get_all_sequence_ids()`.
- [x] Store the global replay list or a hash/count summary in the manifest so worker sequence assignments are reproducible.
- [x] Preserve deterministic ordering of sequence IDs before partitioning.
- [x] Validate that every replay sequence ID exists in the global `DataLoader` shard ranges.
- [x] Validate that zero/sentinel sequence IDs are excluded.
- [x] Verification: add tests for replay-list construction from synthetic `top_ctx`, including duplicates, zeros, unsorted IDs, and missing IDs.

### Phase 2 Notes

- Added `src/pipeline/distributed/pass2_replay.py` with `build_pass2_replay_list()`, `assign_pass2_replay_sequences()`, and stable SHA-256 replay-list hashing.
- Replay lists can be built from a `top_ctx`-like object with `get_all_sequence_ids()` or directly from a loaded `top_ctx.pt` payload containing `ctx_seq_idx`.
- Replay construction sorts global sequence IDs, de-duplicates repeated references, excludes sentinel `0`, and validates every ID against the manifest shard table.
- `WorkAssignments` now stores `pass2_replay_sequence_count` and `pass2_replay_sequence_hash` so pass-2 worker assignments can be checked against the replay-list contract.
- `assign_pass2_replay_sequences()` partitions replay IDs using the existing contiguous sequence partitioner, preserving deterministic order and no-dropped-remainder behavior for Phase 3.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_assignments.py -q` -> `30 passed`.

## Phase 3 - Sequence Partitioning And Worker Inputs

- [x] Use the Part 1 sequence partitioner to split the global replay list across workers.
- [x] Split the replay list into balanced contiguous chunks by list position, not by physical dataset shard.
- [x] Distribute remainder replay sequences across the first workers deterministically; never drop replay sequences because the count is not divisible by worker count.
- [x] Preserve stable order inside each worker's assigned sequence list.
- [x] Support one-worker mode, where the assigned list is the full replay list.
- [x] Store each worker's sequence count, sequence ID min/max, and replay-list hash in worker metadata.
- [x] Ensure workers can use `DataLoader.get_batches_by_ids()` over their assigned IDs without changing global sequence IDs.
- [x] Verification: add tests that multiple worker sequence assignments are contiguous in replay-list order, disjoint, cover the full replay list exactly once, preserve deterministic ordering, and keep all remainder sequences.

### Phase 3 Notes

- Added `Pass2WorkerInput`, `get_pass2_worker_input()`, and `validate_pass2_replay_assignments()` in `src/pipeline/distributed/pass2_replay.py`.
- Pass-2 assignment validation now checks sorted replay-list order, duplicate-free coverage, shard-table membership, manifest count/hash consistency, and that worker assignments equal deterministic contiguous chunks from the Part 1 sequence partitioner.
- One-worker mode returns the full replay list as worker `0` input while preserving the same metadata contract.
- `WorkerMarker` now records pass-2 sequence metadata: `sequence_count`, `sequence_id_min`, `sequence_id_max`, and `replay_sequence_hash`.
- The assigned sequence list remains plain global sequence IDs, matching `DataLoader.get_batches_by_ids(sequence_ids)` without worker-local renumbering.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_assignments.py -q` -> `45 passed`.

## Phase 4 - Worker Pass-2 Dump Entrypoint

- [x] Add a worker entrypoint that loads the merged global `top_ctx` and `latent_stats` artifacts, then replays only assigned sequence IDs.
- [x] Initialize a full local `Inference` model and full local `SAEBank` for each worker.
- [x] Ensure each pass-2 worker uses one logical compute device and passes a single-device list to `SAEBank`; do not allow a worker to split SAE layers across multiple physical GPUs.
- [x] Use the same `encode_layer_components()` path as current pass 2.
- [x] Set frequency factors from global `latent_stats.active_count` when mode is `freq_weighted`.
- [x] Track per-worker dump timing using the existing dump timing categories where possible.
- [x] Avoid loading or mutating global `top_coactivation.pt`; workers should only write partial dump artifacts.
- [x] Verification: run a one-worker synthetic/small fixture and confirm the worker can emit candidate dump tensors without invoking reduce.

### Phase 4 Notes

- Split the existing second-pass implementation into `run_second_pass_dump()` plus the existing `run_second_pass()` reduce/save wrapper in `src/pipeline/second_pass.py`.
- Added `run_pass2_worker()` and `--phase pass2` support to `src/pipeline/distributed/worker.py`; pass-2 workers validate replay assignments, load global `top_ctx.pt` and `latent_stats.pt`, initialize one-device model/SAE resources, and replay only their assigned global sequence IDs.
- Added `validate_pass2_worker_inputs()`, `load_pass2_global_artifacts()`, and `initialize_pass2_worker_resources()` so required global artifacts are checked before model/SAE initialization.
- Pass-2 workers call the same dump path as single-process pass 2, including `encode_layer_components()`, `top_coactivation.prepare_dump()`, `top_coactivation.update_batch()`, and frequency-factor setup from global `latent_stats.active_count`.
- The worker path stops after candidate dump materialization; it does not call `top_coactivation.reduce()` or write canonical `top_coactivation.pt`. Persisting `candidate_dump.partial.pt` is left to Phase 5.
- Worker completion markers record dump batch count plus the Phase 3 replay metadata.
- Verification: `python -m pytest tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_interfaces.py tests/store/test_top_coactivation_modes.py -q` -> `44 passed`.

## Phase 5 - Simple Exact Dump Artifact

- [x] Define `candidate_dump.partial.pt` with schema version, run ID, worker ID, mode, sequence IDs, candidate IDs, candidate values, `M`, `n_candidates_per_component`, `n_latents_per_latent`, `num_components`, `d_sae`, and token-count metadata.
- [x] Store `candidate_ids` as `int32` and `candidate_vals` as `float32`.
- [x] Store `sequence_ids` in the same row order as `candidate_ids` and `candidate_vals`.
- [x] For PMI mode, store worker token counts needed by Part 5 to compute global PMI consistently.
- [x] Save partial dumps atomically under `outputs/<run_id>/distributed/workers/worker_000/pass2/`.
- [x] Verification: add round-trip schema tests for partial dumps and reject mismatched shape/dtype/mode metadata.

### Phase 5 Notes

- Added `src/pipeline/distributed/pass2_partials.py` with `CandidateDumpMetadata`, schema version `1`, atomic save/load helpers, payload construction from `top_coactivation` dump buffers, and validation for simple exact worker dumps.
- `candidate_dump.partial.pt` saves `{metadata, payload}` where payload contains `sequence_ids`, `candidate_ids`, `candidate_vals`, and `total_tokens_processed`.
- Validation enforces row alignment between `sequence_ids`, `candidate_ids`, and `candidate_vals`; `candidate_ids` must be `int32`, `candidate_vals` must be `float32`, and candidate values must be finite/non-negative.
- Metadata records run/worker/config/device identity, replay sequence bounds/hash, coactivation mode, `M`, candidate/top-K dimensions, component count, SAE width, batch count, sequence length, and worker token count for PMI.
- `run_pass2_worker()` now writes `candidate_dump.partial.pt` under the worker `pass2/` directory after dump materialization and declares it in the completed marker.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass2_replay.py tests/store/test_top_coactivation_modes.py -q` -> `44 passed`.

## Phase 6 - Optional Scalable Pre-Aggregation Hook

- [x] Keep simple exact candidate dumps as the first implementation.
- [x] Define an optional hook that can expand each sequence row into `(target_latent, candidate_latent, value)` contributions for future Part 5 MapReduce mode.
- [x] Do not enable worker-local top-K-per-target merging as an exact default.
- [x] If implemented, write pre-aggregation outputs as a separate schema from simple candidate dumps.
- [x] Ensure simple dumps remain available for equivalence testing even after pre-aggregation exists.
- [x] Verification: add tests that pre-aggregation over one worker's rows produces the same raw contribution multiset as expanding the simple dump during reduce.

### Phase 6 Notes

- Added optional pre-aggregation helpers to `src/pipeline/distributed/pass2_partials.py`; simple `candidate_dump.partial.pt` remains the worker default and equivalence artifact.
- `expand_candidate_dump_to_contributions()` consumes a validated simple dump plus global `top_ctx` CSR tensors and emits raw contribution records: `target_ids`, `candidate_ids`, `values`, and source `sequence_ids`.
- The expansion mirrors native reducer semantics: every target entry in the CSR contributes, duplicate target entries remain duplicated, self-candidates are filtered, and non-positive candidate values are skipped.
- Added a separate `candidate_preaggregation` schema with versioned metadata and atomic save/load helpers for future MapReduce/shuffle work.
- No worker-local top-K-per-target merge is introduced; the hook emits raw records only.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass2_replay.py tests/store/test_top_coactivation_modes.py -q` -> `48 passed`.

## Phase 7 - Local And H100 Memory Modes

- [x] Support local 16GB runs with `dump_device: "cpu"` or a one-worker efficient mode.
- [x] Support H100 runs with `dump_device: "gpu"` inside each worker, followed by final CPU transfer when saving partial dumps.
- [x] Estimate per-worker dump memory before allocation: `assigned_sequences * M * (int32 + float32)`.
- [x] Warn or fail early if a worker's assigned replay list would allocate an unexpectedly large dump.
- [x] Consider chunked partial dump files if a full worker dump is too large for CPU RAM or disk.
- [x] Verification: add memory estimate tests and a small chunked-save fixture if chunking is introduced.

### Phase 7 Notes

- Kept the existing `latents.top_coactivation.dump_device` mode as the local/H100 switch: `cpu` remains the local/low-VRAM default and `gpu` keeps H100 worker dump buffers on the worker-local CUDA device during replay.
- `candidate_dump_payload()` continues to transfer dump buffers to CPU before writing `candidate_dump.partial.pt`, so GPU dump mode does not change the partial artifact contract.
- Added `estimate_candidate_dump_bytes()` and `check_candidate_dump_memory_guardrail()` in `src/pipeline/distributed/pass2_partials.py`; estimates use `assigned_sequences * M * 8` bytes for `int32` IDs plus `float32` values.
- Added optional config guardrails: `latents.top_coactivation.dump_memory_guardrail_bytes` and `latents.top_coactivation.fail_on_dump_memory_guardrail`.
- `run_pass2_worker()` checks the estimate before loading global artifacts or initializing model/SAE resources, so unexpectedly large dumps fail early when configured.
- Chunked partial dumps are not introduced yet; simple full-worker dumps remain the exact artifact. The guardrail is the decision point for whether Phase 7 needs a later chunked-save extension.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass2_replay.py tests/store/test_top_coactivation_modes.py -q` -> `51 passed`.

## Phase 8 - Worker Output Validation And Reporting

- [x] Validate every partial dump row has a matching sequence ID.
- [x] Validate candidate IDs are within `[0, num_components * d_sae)`.
- [x] Validate candidate values are finite and non-negative for current modes.
- [x] Validate no row exceeds `M` candidates.
- [x] Save per-worker timing summaries: model forward time, SAE encode time when available, update/dump time, CPU transfer time, save time, sequence count, and batch count.
- [x] Mark the worker's pass-2 dump complete only after artifact validation succeeds.
- [x] Verification: add tests for invalid candidate IDs, non-finite values, shape mismatches, and missing sequence IDs.

### Phase 8 Notes

- `validate_candidate_dump_partial()` now gates worker completion through a reload-validation pass after `candidate_dump.partial.pt` is atomically written.
- The candidate dump validator checks row alignment with `sequence_ids`, exact `[sequence_count, M]` tensor shapes, candidate ID range, finite/non-negative candidate values, sorted duplicate-free sequence IDs, and token-count consistency.
- `save_pass2_candidate_dump()` now writes `pass2_summary.json` beside `candidate_dump.partial.pt` and returns both artifacts for the completed worker marker.
- `pass2_summary.json` records sequence count/range/hash, batch count, sequence length, dump elapsed time, save elapsed time, artifact size, and existing `top_coactivation.dump_timing` categories such as update and CPU transfer when available.
- Model-forward and SAE-encode timing are marked as not yet separated because the current shared second-pass dump loop does not split those timings internally.
- `run_pass2_worker()` writes `completed.json` only after the partial artifact validates and summary write succeeds; failed validation writes `failed.json` instead.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass2_replay.py tests/store/test_top_coactivation_modes.py -q` -> `53 passed`.

## Phase 9 - Equivalence Testing

- [x] Add a tiny synthetic equivalence test comparing current single-process pass-2 dump to one-worker distributed dump.
- [x] Add a two-worker equivalence test: split replay sequences, write two partial dumps, concatenate them in deterministic order, and compare to a single-process dump.
- [x] Add worker device-isolation tests proving pass-2 workers construct a single-device `SAEBank`.
- [x] Include `raw`, `freq_weighted`, and `pmi` modes in focused tests where practical.
- [x] Confirm `seq_id -> row` reconstruction after concatenating worker dumps matches the single-process dump mapping.
- [x] Confirm worker token counts sum to the single-process token count for PMI mode.
- [x] Verification: run focused pass-2 dump tests before any H100 benchmark.

### Phase 9 Notes

- Added `tests/pipeline/test_distributed_pass2_equivalence.py` with a synthetic runtime that exercises the real `run_second_pass_dump()` loop and `save_pass2_candidate_dump()` artifact path without loading a model or SAE weights.
- One-worker equivalence compares a default single-process replay dump with a one-worker distributed partial for `raw`, `freq_weighted`, and `pmi`.
- Two-worker equivalence splits the replay list, writes two validated `candidate_dump.partial.pt` files, concatenates them in deterministic worker order, and compares the reconstructed `sequence_id -> row` mapping to the single-process dump.
- PMI coverage confirms worker token counts sum back to the single-process dump token count; non-PMI modes remain at zero token count as expected.
- The existing `test_initialize_pass2_worker_resources_uses_single_worker_device()` covers pass-2 worker device isolation and verifies the worker constructs a single-device `SAEBank` with `multi_gpu=False`.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass2_replay.py tests/store/test_top_coactivation_modes.py -q` -> `60 passed`.

## Phase 10 - H100 Benchmark Readiness

- [x] Add benchmark logging for per-worker replay sequence counts, total wall time, average batch time, dump artifact size, peak VRAM, and assignment imbalance.
- [x] Add a dry-run estimate that reports expected worker sequence counts and candidate dump sizes without loading the model.
- [x] Prepare 1-worker versus 2-worker versus 8-worker pass-2 dump benchmark commands for when H100 artifacts are available.
- [x] Separate model+SAE forward time from candidate materialization time so remaining bottlenecks are visible.
- [x] Preserve enough metadata for Part 5 to decide whether simple exact merge is acceptable or MapReduce pre-aggregation is needed.
- [x] Verification: document exact benchmark commands and configs in this file after implementation.

### Phase 10 Notes

- Added `src/pipeline/distributed/pass2_benchmark.py` with `build_pass2_benchmark_estimate()` for model-free dry-run sizing and `build_pass2_benchmark_report()` for aggregating completed worker markers plus `pass2_summary.json`.
- Dry-run estimates report per-worker replay sequence counts, candidate dump bytes, total estimated dump bytes, and assignment imbalance from the manifest's pass-2 assignments.
- Completed-run reports aggregate total wall time (`max(worker duration)`), total worker time, average batch time, dump artifact bytes, peak CUDA memory when available, worker sequence imbalance, and split timing categories.
- `SecondPassDumpResult` now carries `model_forward_s`, `sae_encode_s`, and `update_dump_s`; worker summaries also record save time and peak CUDA memory.
- Controller dry-run text includes a pass-2 candidate dump estimate whenever pass-2 sequence assignments are present in the manifest plan.
- Exact H100 benchmark commands after merged Part 2/3 artifacts and assigned pass-2 replay IDs are available:
  - 1 worker: `python -m pipeline.distributed.worker --manifest <run>/distributed/manifest.json --worker-id 0 --phase pass2`
  - 2 workers: run the same command concurrently for `--worker-id 0` and `--worker-id 1` from a manifest planned with `worker_count=2`.
  - 8 workers: run the same command concurrently for `--worker-id 0` through `--worker-id 7` from a manifest planned with `worker_count=8`.
- Model-free sizing/report snippets:
  - Estimate before launch: `python -c "from pipeline.distributed import load_manifest, build_pass2_benchmark_estimate, format_pass2_benchmark_estimate; m=256; manifest=load_manifest('<run>/distributed/manifest.json'); print(format_pass2_benchmark_estimate(build_pass2_benchmark_estimate(manifest, m=m)))"`
  - Aggregate after completion: `python -c "from pipeline.distributed import load_manifest, build_pass2_benchmark_report, save_pass2_benchmark_report; manifest=load_manifest('<run>/distributed/manifest.json'); report=build_pass2_benchmark_report(manifest); save_pass2_benchmark_report(report, '<run>/distributed/reports/pass2_benchmark_report.json'); print(report.model_dump_json(indent=2))"`
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_benchmark.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_controller.py tests/store/test_top_coactivation_modes.py -q` -> `76 passed`.

---

## Open Questions

- Should simple exact partial dumps be one file per worker or chunked by sequence range from the start?
- Should the replay sequence list be saved as a manifest artifact, or can it be deterministically reconstructed from global `top_ctx` every time?
- Should candidate-profile computation become a pure function outside `TopCoactivation`, or remain a method with a lighter write path?
- Should pass-2 workers force full single-GPU SAE placement even if `hardware.multi_gpu` is true in the base config?
- What dump-size threshold should trigger chunked worker artifacts or direct pre-aggregation?

## Risks / Assumptions

- Exactness depends on every global replay sequence being processed exactly once.
- Candidate dump rows must remain tied to global sequence IDs; worker-local row numbers are not enough.
- PMI mode needs globally consistent token/count metadata in Part 5.
- Worker-local top-K-per-target is not exact and belongs only in an explicitly approximate future mode.
- The first implementation should optimize for single-process equivalence before reducing dump I/O.
