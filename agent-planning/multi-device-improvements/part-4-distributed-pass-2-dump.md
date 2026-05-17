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

- [ ] Treat the current `TopCoactivation.update_batch()` output as the exact semantic contract for this part.
- [ ] Separate "compute candidate profile for a batch" from "write candidate profile into a process-local dump buffer" so distributed workers can reuse the same scoring logic.
- [ ] Preserve current scoring modes: `raw`, `freq_weighted`, and `pmi`.
- [ ] Preserve current candidate IDs as flattened global latent IDs: `component_idx * d_sae + latent_idx`.
- [ ] Preserve `M = min(n_latents_per_latent * 4, num_components * n_candidates_per_component)` unless an explicit schema version changes it later.
- [ ] Verification: add unit tests proving the refactored candidate-profile helper matches current `update_batch()` output for `raw`, `freq_weighted`, and `pmi`.

## Phase 2 - Global Replay Sequence List

- [ ] Build the replay list from merged global `top_ctx.get_all_sequence_ids()`.
- [ ] Store the global replay list or a hash/count summary in the manifest so worker sequence assignments are reproducible.
- [ ] Preserve deterministic ordering of sequence IDs before partitioning.
- [ ] Validate that every replay sequence ID exists in the global `DataLoader` shard ranges.
- [ ] Validate that zero/sentinel sequence IDs are excluded.
- [ ] Verification: add tests for replay-list construction from synthetic `top_ctx`, including duplicates, zeros, unsorted IDs, and missing IDs.

## Phase 3 - Sequence Partitioning And Worker Inputs

- [ ] Use the Part 1 sequence partitioner to split the global replay list across workers.
- [ ] Split the replay list into balanced contiguous chunks by list position, not by physical dataset shard.
- [ ] Distribute remainder replay sequences across the first workers deterministically; never drop replay sequences because the count is not divisible by worker count.
- [ ] Preserve stable order inside each worker's assigned sequence list.
- [ ] Support one-worker mode, where the assigned list is the full replay list.
- [ ] Store each worker's sequence count, sequence ID min/max, and replay-list hash in worker metadata.
- [ ] Ensure workers can use `DataLoader.get_batches_by_ids()` over their assigned IDs without changing global sequence IDs.
- [ ] Verification: add tests that multiple worker sequence assignments are contiguous in replay-list order, disjoint, cover the full replay list exactly once, preserve deterministic ordering, and keep all remainder sequences.

## Phase 4 - Worker Pass-2 Dump Entrypoint

- [ ] Add a worker entrypoint that loads the merged global `top_ctx` and `latent_stats` artifacts, then replays only assigned sequence IDs.
- [ ] Initialize a full local `Inference` model and full local `SAEBank` for each worker.
- [ ] Ensure each pass-2 worker uses one logical compute device and passes a single-device list to `SAEBank`; do not allow a worker to split SAE layers across multiple physical GPUs.
- [ ] Use the same `encode_layer_components()` path as current pass 2.
- [ ] Set frequency factors from global `latent_stats.active_count` when mode is `freq_weighted`.
- [ ] Track per-worker dump timing using the existing dump timing categories where possible.
- [ ] Avoid loading or mutating global `top_coactivation.pt`; workers should only write partial dump artifacts.
- [ ] Verification: run a one-worker synthetic/small fixture and confirm the worker can emit candidate dump tensors without invoking reduce.

## Phase 5 - Simple Exact Dump Artifact

- [ ] Define `candidate_dump.partial.pt` with schema version, run ID, worker ID, mode, sequence IDs, candidate IDs, candidate values, `M`, `n_candidates_per_component`, `n_latents_per_latent`, `num_components`, `d_sae`, and token-count metadata.
- [ ] Store `candidate_ids` as `int32` and `candidate_vals` as `float32`.
- [ ] Store `sequence_ids` in the same row order as `candidate_ids` and `candidate_vals`.
- [ ] For PMI mode, store worker token counts needed by Part 5 to compute global PMI consistently.
- [ ] Save partial dumps atomically under `outputs/<run_id>/distributed/workers/worker_000/pass2/`.
- [ ] Verification: add round-trip schema tests for partial dumps and reject mismatched shape/dtype/mode metadata.

## Phase 6 - Optional Scalable Pre-Aggregation Hook

- [ ] Keep simple exact candidate dumps as the first implementation.
- [ ] Define an optional hook that can expand each sequence row into `(target_latent, candidate_latent, value)` contributions for future Part 5 MapReduce mode.
- [ ] Do not enable worker-local top-K-per-target merging as an exact default.
- [ ] If implemented, write pre-aggregation outputs as a separate schema from simple candidate dumps.
- [ ] Ensure simple dumps remain available for equivalence testing even after pre-aggregation exists.
- [ ] Verification: add tests that pre-aggregation over one worker's rows produces the same raw contribution multiset as expanding the simple dump during reduce.

## Phase 7 - Local And H100 Memory Modes

- [ ] Support local 16GB runs with `dump_device: "cpu"` or a one-worker efficient mode.
- [ ] Support H100 runs with `dump_device: "gpu"` inside each worker, followed by final CPU transfer when saving partial dumps.
- [ ] Estimate per-worker dump memory before allocation: `assigned_sequences * M * (int32 + float32)`.
- [ ] Warn or fail early if a worker's assigned replay list would allocate an unexpectedly large dump.
- [ ] Consider chunked partial dump files if a full worker dump is too large for CPU RAM or disk.
- [ ] Verification: add memory estimate tests and a small chunked-save fixture if chunking is introduced.

## Phase 8 - Worker Output Validation And Reporting

- [ ] Validate every partial dump row has a matching sequence ID.
- [ ] Validate candidate IDs are within `[0, num_components * d_sae)`.
- [ ] Validate candidate values are finite and non-negative for current modes.
- [ ] Validate no row exceeds `M` candidates.
- [ ] Save per-worker timing summaries: model forward time, SAE encode time when available, update/dump time, CPU transfer time, save time, sequence count, and batch count.
- [ ] Mark the worker's pass-2 dump complete only after artifact validation succeeds.
- [ ] Verification: add tests for invalid candidate IDs, non-finite values, shape mismatches, and missing sequence IDs.

## Phase 9 - Equivalence Testing

- [ ] Add a tiny synthetic equivalence test comparing current single-process pass-2 dump to one-worker distributed dump.
- [ ] Add a two-worker equivalence test: split replay sequences, write two partial dumps, concatenate them in deterministic order, and compare to a single-process dump.
- [ ] Add worker device-isolation tests proving pass-2 workers construct a single-device `SAEBank`.
- [ ] Include `raw`, `freq_weighted`, and `pmi` modes in focused tests where practical.
- [ ] Confirm `seq_id -> row` reconstruction after concatenating worker dumps matches the single-process dump mapping.
- [ ] Confirm worker token counts sum to the single-process token count for PMI mode.
- [ ] Verification: run focused pass-2 dump tests before any H100 benchmark.

## Phase 10 - H100 Benchmark Readiness

- [ ] Add benchmark logging for per-worker replay sequence counts, total wall time, average batch time, dump artifact size, peak VRAM, and assignment imbalance.
- [ ] Add a dry-run estimate that reports expected worker sequence counts and candidate dump sizes without loading the model.
- [ ] Benchmark 1-worker versus 2-worker versus 8-worker pass-2 dump after Part 2/3 artifacts are available.
- [ ] Separate model+SAE forward time from candidate materialization time so remaining bottlenecks are visible.
- [ ] Preserve enough metadata for Part 5 to decide whether simple exact merge is acceptable or MapReduce pre-aggregation is needed.
- [ ] Verification: document exact benchmark commands and configs in this file after implementation.

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
