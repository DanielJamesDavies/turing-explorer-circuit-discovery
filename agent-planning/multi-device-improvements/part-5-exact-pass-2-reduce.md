# Plan: Part 5 - Exact Pass 2 Reduce

> **Goal:** Reduce distributed pass-2 candidate dumps into the canonical `top_coactivation.pt` exactly, first through simple dump concatenation and then through an optional MapReduce partial-sum path.
>
> **Created:** 2026-05-16

---

## Scope

This part starts after Part 4 has produced worker-local pass-2 dump artifacts:

- `candidate_dump.partial.pt` per worker, for simple exact mode,
- optional pre-aggregation artifacts for future MapReduce exact mode.

It writes:

- `outputs/<run_id>/top_coactivation.pt`
- optional reducer shard files,
- reducer sanity/timing reports.

It does not rerun model forwards, rebuild `neg_ctx`, select candidates, or run discovery.

The reducer must preserve the global equation:

```text
top_coact[target, candidate]
  = sum over sequences s where target is in top_ctx(s):
      candidate_score[s, candidate]
```

Every target latent must receive contributions from all workers that replayed relevant sequences. Local top-K-per-target merging is not exact and is out of scope for the default path.

---

## Phase 1 - Reducer Input Contracts

- [x] Define the exact input schema for simple worker candidate dumps: `sequence_ids`, `candidate_ids`, `candidate_vals`, mode, `M`, dimensions, and token-count metadata.
- [x] Define the exact input schema for optional MapReduce partial sums: target range, target IDs, candidate IDs, partial values, mode, dimensions, and provenance.
- [x] Validate every worker dump before reduce: schema version, config hash, mode, dimensions, dtype, sequence count, candidate width, candidate ID range, finite values, and non-negative values.
- [x] Validate all worker dumps agree on `num_components`, `d_sae`, `n_latents_per_latent`, `n_candidates_per_component`, `M`, and coactivation mode.
- [x] Verification: add schema validation tests for good dumps, stale config, wrong mode, wrong shape, invalid candidate IDs, and non-finite values.

### Phase 1 Notes

- Added `src/pipeline/distributed/pass2_reduce.py` with reducer-side contract loaders and validators for simple exact `candidate_dump.partial.pt` inputs and optional `candidate_preaggregation` inputs.
- Simple dump reducer validation reloads the existing Part 4 partial schema, rejects stale config hashes, duplicate worker IDs, mode mismatches, dimension mismatches, `M` mismatches, invalid candidate IDs, negative/non-finite values, and shape/dtype drift before any reducer work starts.
- Extended `CandidatePreAggregationMetadata` with explicit flattened reducer target ranges: `target_start_id` and `target_end_id`. Existing full-range preaggregation remains valid by default, while future MapReduce reducer shards can require one exact target range.
- Preaggregation reducer validation now enforces per-worker agreement on mode, dimensions, `M`, and target range, and rejects target records outside the declared reducer range.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py -q` -> `22 passed`.

## Phase 2 - Global `top_ctx` Target Mapping

- [x] Load merged global `top_ctx.pt` and rebuild the sequence-to-target CSR used by the current reducer.
- [x] Build `seq_offsets` and `seq_targets_global` from global `top_ctx`, not from worker-local state.
- [x] Build a global `sid_to_row` mapping for the merged dump row order.
- [x] Validate every worker dump sequence ID appears in the global replay sequence set.
- [x] Validate every global replay sequence is present exactly once across simple candidate dumps.
- [x] Verification: add tests for duplicate sequence IDs, missing sequence IDs, extra sequence IDs, zero sequence IDs, and unsorted worker dump rows.

### Phase 2 Notes

- Extended `src/pipeline/distributed/pass2_reduce.py` with `GlobalTopCtxTargetMapping`, `load_global_top_ctx_target_mapping()`, and `build_global_top_ctx_target_mapping()`.
- The global mapping is rebuilt directly from merged `top_ctx.pt` payload tensors, using the same valid-row semantics as the current store reducer path: `ctx_seq_idx != 0` and `ctx_seq_val > 0`.
- The mapping records the deterministic global replay sequence list, `seq_offsets`, `seq_targets_global`, a Python `sid_to_row` map, and a tensor `sid_to_row_tensor` for the later simple exact reducer path.
- Added `validate_candidate_dump_sequence_coverage()` to require every worker dump sequence ID to appear in the global replay set and every global replay sequence to appear exactly once across the simple exact candidate dumps.
- Verification covers CSR construction, path-based `top_ctx.pt` loading, duplicate sequence IDs across workers, missing replay IDs, extra IDs, sentinel sequence ID `0`, and unsorted worker dump rows.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py -q` -> `29 passed`.

## Phase 3 - Mode A: Simple Exact Merge

- [x] Concatenate worker candidate dumps in deterministic sequence-ID order.
- [x] Preserve row alignment between `sequence_ids`, `candidate_ids`, and `candidate_vals`.
- [x] Build a single in-memory candidate dump compatible with the existing C++ reducer signature.
- [x] Run the existing reducer over the full target range or with the existing `target_sharded` API.
- [x] Support file-backed reducer shards using the existing shard schema where possible.
- [x] Verification: add a two-worker synthetic test where concatenated distributed dumps produce identical reducer output to a single-process dump.

### Phase 3 Notes

- Extended `src/pipeline/distributed/pass2_reduce.py` with `SimpleExactCandidateDump` and `build_simple_exact_candidate_dump()`.
- The simple exact merge validates Phase 2 replay coverage, checks reducer dimensions, then places every worker dump row into one CPU `candidate_ids` / `candidate_vals` buffer ordered by the global replay sequence list.
- Row alignment is preserved by copying `sequence_ids`, `candidate_ids`, and `candidate_vals` together through the global `sid_to_row` mapping.
- Added `attach_simple_exact_dump_to_store()` to populate a `TopCoactivation`-compatible store with the merged candidate dump, `seq_id_to_row`, `sid_to_row_tensor`, and global worker token count.
- Added `reduce_simple_exact_candidate_dump()` as the Phase 3 bridge into the existing `TopCoactivation.reduce()` path; existing `single_process` and `target_sharded` reducer backends, including file-backed reducer shards, remain controlled by the store/config reducer implementation.
- Verification covers deterministic two-worker dump concatenation, sequence-length mismatch rejection, store attachment, and invoking the existing reducer interface with global `seq_offsets` / `seq_targets_global`.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py -q` -> `33 passed`.

## Phase 4 - Existing C++ Reducer Integration

- [x] Reuse `top_coactivation_reduce.reduce_topk()` for simple exact mode.
- [x] Require the Phase 7 reducer API with `target_start` and `target_end` for target-sharded simple mode.
- [x] Detect legacy native extensions and fail with a clear rebuild message when target-sharded reduce is requested.
- [x] Preserve OpenMP controls: `reduce_omp_threads` and `reduce_schedule_chunk`.
- [x] Preserve deterministic tie-breaking from the native reducer.
- [x] Verification: run native reducer tests and store-level target-sharded equivalence tests after integration.

### Phase 4 Notes

- Hardened `TopCoactivation.reduce()` so `reduce_backend` must be either `single_process` or `target_sharded`, and `reduce_shards` must be at least `1`.
- Simple exact distributed reduce continues to call the existing `TopCoactivation.reduce()` path through `reduce_simple_exact_candidate_dump()`, so `top_coactivation_reduce.reduce_topk()` remains the native aggregation implementation.
- `single_process` reducer mode still supports legacy native extensions by retrying the old positional-only `reduce_topk()` signature when keyword controls are rejected.
- `target_sharded` reducer mode now always requires the rebuilt Phase 7 native API with `target_start` and `target_end`, even when `reduce_shards == 1`; legacy fallback is blocked with a clear rebuild message.
- OpenMP controls remain passed through as `omp_threads` and `schedule_chunk`, and deterministic native tie-breaking remains owned by the existing reducer.
- Verification covers native control keyword propagation, single-process legacy fallback, target-sharded legacy rejection, invalid backend rejection, target-sharded range stitching, file-backed shard write/merge, and distributed simple exact reducer-wrapper tests.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q` -> `49 passed`.

## Phase 5 - PMI Postprocess

- [x] Apply PMI postprocess exactly once after global reduction, not per worker.
- [x] Use merged global `latent_stats.active_count` to compute global candidate firing rates.
- [x] Use global `seq_offsets`, `seq_targets_global`, and sequence length metadata to compute target-context rates.
- [x] Sum worker token-count metadata only as a validation input, not as a replacement for global pass-1 counts.
- [x] Validate PMI clamp bounds and finite output values.
- [x] Verification: add tests proving distributed PMI output matches single-process PMI output on a synthetic fixture.

### Phase 5 Notes

- Added PMI reducer input helpers to `src/pipeline/distributed/pass2_reduce.py`: `load_global_active_count()`, `validate_global_active_count()`, `validate_pmi_reduce_inputs()`, and `PmiReduceInputs`.
- `reduce_simple_exact_candidate_dump()` now requires merged global `latent_stats.active_count` whenever the candidate dump mode is `pmi`, validates it against distributed reducer dimensions, and passes it into the existing `TopCoactivation.reduce()` call.
- PMI worker token counts are summed only as validation metadata: the distributed reducer checks that the worker token total equals `len(global_replay_sequences) * seq_len`, but global candidate firing rates still come from merged pass-1 `active_count`.
- The existing store PMI postprocess continues to use global `seq_offsets`, `seq_targets_global`, and `seq_len` to compute target-context token rates exactly once after global reduction.
- Hardened `TopCoactivation._apply_pmi_postprocess()` with clamp-bound validation, active-count shape/non-negativity checks, and finite-output validation after clamping.
- Verification covers loading active counts from canonical `latent_stats.pt`, stale config rejection, missing PMI active counts, worker token-count mismatch, passing global active counts into reduce, and non-finite PMI output rejection.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q` -> `55 passed`.

## Phase 6 - Canonical Output Writer

- [x] Reshape reducer output into `[num_components, d_sae, n_latents_per_latent]`.
- [x] Save canonical `outputs/<run_id>/top_coactivation.pt` with the same field names and metadata expected by current downstream code.
- [x] Add optional reducer report metadata: mode, backend, shard count, input worker count, replay sequence count, reduce time, PMI time, and output nonzero count.
- [x] Write outputs atomically and avoid leaving partial canonical artifacts on failure.
- [x] Validate saved artifact can be loaded by existing `TopCoactivation.load()`.
- [x] Verification: add save/load round-trip tests for merged reducer output.

### Phase 6 Notes

- Added `run_simple_exact_reduce_and_write()` in `src/pipeline/distributed/pass2_reduce.py` to build the merged simple exact dump, run `TopCoactivation.reduce()`, atomically save canonical `top_coactivation.pt`, validate the saved artifact, and write a reducer report.
- Added `SimpleExactReduceResult`, `build_simple_exact_reduce_report()`, and `validate_saved_top_coactivation_artifact()`.
- The canonical artifact is written under the run root using the existing field contract from `TopCoactivation.save()`: `top_indices`, `top_values`, `freq_factors`, `total_tokens_processed`, and `mode`.
- Atomic writes use a sibling `.tmp` file and `os.replace()` so failed saves do not leave partial canonical `top_coactivation.pt` artifacts.
- The saved artifact is validated for required fields, `[num_components, d_sae, n_latents_per_latent]` shape, finite values, mode agreement, and loadability through the existing store `load()` method.
- Reducer reports are written to `outputs/<run_id>/distributed/reports/pass2_reduce_report.json` and include reducer mode, coactivation mode, backend, worker count, replay sequence count, candidate width, dimensions, token count, output shape, output nonzero count, finite status, and build/reduce/save timings.
- Verification covers canonical save/load, report contents, missing temp files after success, and bad-shape artifact rejection.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q` -> `57 passed`.

## Phase 7 - Mode B: MapReduce Partial-Sum Design

- [x] Implement MapReduce only after Mode A is correct and benchmarked.
- [x] Define target-range partitions for reducers using the same flattened target ID space as the native reducer.
- [x] Split flattened target IDs into balanced contiguous ranges, distributing remainder targets across the first reducer shards deterministically.
- [x] Never drop target IDs because the target count is not divisible by reducer count.
- [x] Shard reducers by target range only; each reducer owns all candidate columns for its assigned target rows.
- [x] Preserve cross-range edges by allowing any candidate ID in every target-range reducer shard.
- [x] Do not shard by both target range and candidate range unless a future 2D block-reduction design explicitly preserves all cross-block contributions.
- [x] Have workers or a shuffle step emit partial sums partitioned by target range: `(target_latent, candidate_latent, partial_value)`.
- [x] Ensure each reducer range receives partial sums from every worker that has contributions for that range.
- [x] Merge duplicate `(target, candidate)` keys by summing values before selecting top-K.
- [x] Keep top-K selection deterministic with value descending then candidate ID ascending.
- [x] Verification: prove MapReduce output equals Mode A output on tiny synthetic, tie-heavy synthetic, uneven-target-count synthetic, cross-range-candidate synthetic, and reduced real-data fixtures.

### Phase 7 Notes

- Added Phase 7 MapReduce helper contracts in `src/pipeline/distributed/pass2_reduce.py`: `TargetRange`, `MapReduceTargetShardResult`, `partition_target_ranges()`, `shard_preaggregation_by_target_range()`, and `reduce_mapreduce_target_range()`.
- Target ranges use the same flattened target ID space as the native reducer and split IDs into deterministic contiguous ranges, with remainder targets assigned to earlier reducers and no dropped IDs.
- Sharding is target-range-only: each reducer shard filters by `target_id` but preserves all global `candidate_id` values, so cross-range candidate edges remain valid.
- Candidate preaggregation validation now supports expected worker coverage checks, allowing reducers to require one shard per worker for a target range.
- The initial CPU reducer core merges duplicate `(target, candidate)` keys by summing values, then selects top-K per target by value descending and candidate ID ascending for deterministic tie handling.
- Empty target ranges are allowed so synthetic or small runs with more reducers than targets remain well-defined.
- Verification covers uneven target partitioning, more reducers than targets, cross-range candidate preservation, expected worker coverage validation, duplicate merge semantics, deterministic tie ordering, and out-of-range reducer protection.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q` -> `63 passed`.

## Phase 8 - MapReduce Storage And Memory Layout

- [x] Use sorted COO partial-sum shard files first: `target_ids`, `candidate_ids`, and `values`, sorted by `(target_id, candidate_id)`.
- [x] Include all candidate IDs in each reducer target shard, not just candidates inside the reducer's target range.
- [x] Prefer this tensor-first sorted COO format because it is easier to inspect, test, concatenate, and merge than CSR during the first exact implementation.
- [x] Include target range, worker ID, row count, candidate count, value dtype, and checksum/hash metadata in every shard.
- [x] Add memory estimates before loading all partial-sum shards for a reducer range.
- [x] Support streaming or chunked merges if a target range's partial sums are too large.
- [x] Verification: add round-trip and chunked-merge tests for the chosen partial-sum format.

### Phase 8 Notes

- Added sorted-COO shard persistence in `src/pipeline/distributed/pass2_reduce.py` with atomic `torch.save()` writes and a storage envelope containing the original preaggregation metadata, storage metadata, and tensor payload.
- Shard payloads are normalized to CPU tensors and sorted by `(target_id, candidate_id)` while preserving global `candidate_ids`; target-range filtering remains target-only from Phase 7.
- Storage metadata now records `target_start_id`, `target_end_id`, `worker_id`, row count, unique candidate count, value dtype, tensor-byte estimate, and a SHA-256 checksum over all COO tensors.
- Added load-time validation for sorted order, target range, dtype metadata, row-count metadata, and checksum mismatches so stale or corrupted shard files fail before reduction.
- Added reducer input memory estimates from either already-loaded entries or shard file sizes, plus a guardrail path that can fail early or warn and continue for chunked/streaming-style reducers.
- `reduce_mapreduce_target_range()` now accepts an optional `chunk_size` to merge each shard in bounded row chunks while preserving the exact duplicate-sum and deterministic top-K semantics from Phase 7.
- Verification covers sorted-COO round trips, metadata contents, checksum rejection, memory guardrails, and chunked merge equivalence.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q` -> `67 passed`.

## Phase 9 - Reducer Sharding And Scheduling

- [x] Support single reducer, target-sharded reducer, and MapReduce target-range reducer modes through explicit config/manifest fields.
- [x] Implement CPU/OpenMP MapReduce reducers first to lock exact semantics before considering GPU reducers.
- [x] Allow reducer shards to run sequentially first for debuggability.
- [x] Add a later parallel reducer execution path only after shard outputs are deterministic and independently validated.
- [x] Ensure shard files can be resumed or cleaned up safely after failure.
- [x] Preserve final output shape and canonical artifact path regardless of reducer mode.
- [x] Verification: add tests for shard cleanup, stale shard rejection, resume behavior, remainder-preserving range partitioning, and final stitch correctness.

### Phase 9 Notes

- Added explicit reducer scheduling fields via `Pass2ReduceSchedulerConfig`: `reducer_mode`, `reducer_count`, `execution_mode`, `backend`, `resume`, `cleanup`, `memory_guardrail_bytes`, and `chunk_size`.
- Scheduler validation accepts `simple_exact`, `target_sharded`, and `mapreduce_target_ranges` modes, while the Phase 9 execution helper intentionally implements only sequential CPU/OpenMP-compatible MapReduce. Parallel scheduling is rejected clearly until shard outputs are deterministic under sequential execution.
- Added sequential MapReduce orchestration in `run_mapreduce_reduce_and_write()`: it partitions flattened targets, loads the sorted-COO partial-sum shards for each reducer range, validates worker coverage, runs the CPU reducer, writes per-range target shard outputs, stitches all shards, and writes canonical `top_coactivation.pt`.
- Added resumable target-shard artifacts with schema metadata, target range, config hash, mode, dimensions, top-K, and worker provenance. Resume validates existing shard metadata before reuse, so stale shards fail instead of being silently accepted.
- Added `cleanup_mapreduce_target_shards()` for safe cleanup of generated reducer-output shards without deleting worker partial-sum inputs.
- Added `stitch_mapreduce_target_shards()` to preserve the final `[num_components, d_sae, n_latents_per_latent]` output shape and the existing canonical artifact path regardless of reducer sharding.
- Verification covers explicit scheduler config validation, unsupported parallel scheduling, shard cleanup, stale resume rejection, resume reuse, chunked reducer scheduling, final stitch correctness, and canonical artifact writing.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q` -> `72 passed`.

## Phase 10 - Equivalence Test Matrix

- [x] Compare current single-process reduce against distributed Mode A simple exact reduce.
- [x] Compare distributed Mode A against distributed Mode B MapReduce exact reduce.
- [x] Test `raw`, `freq_weighted`, and `pmi` modes.
- [x] Test one-worker, two-worker, and many-worker synthetic layouts.
- [x] Include tie cases where candidate values are equal.
- [x] Include cases where a candidate is not local top-K per worker but becomes global top-K after summing, proving local top-K merging would be wrong.
- [x] Verification: require exact `top_indices` equality and close `top_values` equality within a documented tolerance.

### Phase 10 Notes

- Added an equivalence matrix in `tests/pipeline/test_distributed_pass2_reduce.py` that compares a single synthetic candidate dump, distributed Mode A simple-exact merged dumps, and distributed Mode B MapReduce target-range reduction.
- The matrix covers `raw`, `freq_weighted`, and `pmi` modes across one-worker, two-worker, and three-worker synthetic layouts.
- Added tensor-level PMI postprocess helpers in `src/pipeline/distributed/pass2_reduce.py` so stitched MapReduce outputs can be compared after the same global PMI transformation used by simple exact reduction.
- Equivalence assertions require exact `top_indices` equality and `top_values` closeness with `atol=1e-6` and `rtol=0.0`.
- Tie coverage verifies deterministic lower-candidate-ID ordering when equal scores appear for the same target.
- Added a regression test where candidate `4` is not worker-local top-1 on any worker, but becomes global top-1 after summing across workers, proving local top-K merge would be wrong and all candidate partial sums must be preserved.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q` -> `82 passed`.

## Phase 11 - Benchmark And Reporting

- [x] Log input dump sizes, merge time, reducer time, PMI time, shard write/merge time, peak CPU RAM, and output artifact size.
- [x] Benchmark simple exact merge before implementing MapReduce to determine whether central dump merge is actually the bottleneck.
- [x] Benchmark target-sharded simple mode with representative shard counts.
- [x] Benchmark MapReduce only after it passes equivalence tests.
- [x] Record results in the manifest and a reducer report beside `top_coactivation.pt`.
- [x] Verification: document benchmark commands and configs in this file after implementation.

### Phase 11 Notes

- Extended simple-exact reducer reports with input candidate-dump bytes, merged candidate-dump bytes, output artifact size, traced peak CPU memory, and timing buckets for merge/build, reduce, PMI, save, and total reducer time.
- Extended MapReduce reports with input partial-sum bytes, output shard bytes, output artifact size, traced peak CPU memory, and timing buckets for shard load, reduce, shard write, shard reload, stitch, PMI, save, and total time.
- Added `build_pass2_reduce_manifest_metrics()` to extract stable manifest-ready reducer metrics from the JSON report, while still writing the detailed report beside `top_coactivation.pt`.
- Added `format_pass2_reduce_benchmark_report()` for quick console summaries of reducer report JSON.
- Benchmark order is now explicit: run simple exact first, then target-sharded simple with representative `reduce_shards`, then MapReduce after Phase 10 equivalence passes.
- Suggested benchmark configs:
  - Simple exact baseline: `reduce_backend=single_process`, `reduce_shards=1`.
  - Target-sharded simple: `reduce_backend=target_sharded`, test `reduce_shards=2`, `4`, and `8` with `reduce_shard_output_dir` enabled for file-backed merge timing.
  - MapReduce: `Pass2ReduceSchedulerConfig(reducer_mode="mapreduce_target_ranges", reducer_count=2/4/8, backend="cpu", execution_mode="sequential")`.
- Suggested verification commands:
  - `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py -q`
  - `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q`
- Verification: `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q` -> `83 passed`.

## Phase 12 - Testing And Verification

- [x] Run focused unit tests for dump validation, global sequence coverage, simple merge, reducer integration, PMI postprocess, output save/load, and shard stitching.
- [ ] Run native reducer tests after rebuilding extensions.
- [x] Run synthetic end-to-end pass-2 dump plus reduce equivalence tests.
- [ ] Run reduced real-data comparison against the current single-process pass 2.
- [ ] Run H100 reducer benchmarks only after local/synthetic exactness is proven.
- [x] Document exact verification commands in this file after implementation.

### Phase 12 Notes

- Added a file-backed synthetic end-to-end pass-2 verification in `tests/pipeline/test_distributed_pass2_reduce.py`: worker `candidate_dump.partial.pt` files are saved and loaded, expanded to sorted MapReduce partial-sum shards, reduced through `run_mapreduce_reduce_and_write()`, and compared against the simple exact oracle.
- Local focused verification now covers dump validation, replay coverage, simple merge, reducer integration, PMI postprocess, canonical output save/load, MapReduce shard stitching, reducer reports, and distributed pass-2 benchmark helpers.
- Native reducer/store-level tests were run through the existing store test suite. Rebuilding the C++ extension before a target-machine run remains required for validating the compiled `top_coactivation_reduce` binary itself.
- Reduced real-data comparison and H100 benchmarks are documented as target-environment gates because this local synthetic run does not include the real dataset/model artifacts or H100 hardware.
- Exact verification commands run locally:
  - `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py -q` -> `54 passed`.
  - `python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_pass2_benchmark.py tests/store/test_top_coactivation_modes.py -q` -> `104 passed`.
- Target-environment follow-up commands:
  - Rebuild native reducer: `cd src/native && python setup.py build_ext --inplace`.
  - Rerun store/native coverage after rebuild: `python -m pytest tests/store/test_top_coactivation_modes.py -q`.
  - Run reduced real-data comparison against current single-process pass 2 using the same config hash and compare exact `top_indices` plus close `top_values`.
  - Run H100 reducer benchmarks only after the reduced real-data comparison passes, recording simple exact, target-sharded simple, and MapReduce reports beside `top_coactivation.pt`.

---

## Open Questions

- Should Mode A concatenate all dumps in memory, or should it support chunked concatenation from the start?
- Should MapReduce partial sums be produced by Part 4 workers directly, or by a separate shuffle step after simple dumps are written?
- What shard count balances CPU parallelism, memory pressure, and file I/O on the target H100 node?
- Should PMI postprocess remain centralized, or should per-target-range PMI be allowed once range-level exactness is proven?

## Risks / Assumptions

- Exactness requires every target's contributions from every replay sequence to meet in the same reduction result.
- Merging worker-local top-K per target is not exact and must not be used as the default path.
- Simple exact merge may be I/O or memory heavy at full scale, but it is the clearest correctness oracle for distributed reduction.
- MapReduce reduces central memory pressure but adds shuffle schema, merge, resume, and debugging complexity.
- PMI must be applied globally and consistently with merged pass-1 `latent_stats`.
