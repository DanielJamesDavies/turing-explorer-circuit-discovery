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

- [ ] Define the exact input schema for simple worker candidate dumps: `sequence_ids`, `candidate_ids`, `candidate_vals`, mode, `M`, dimensions, and token-count metadata.
- [ ] Define the exact input schema for optional MapReduce partial sums: target range, target IDs, candidate IDs, partial values, mode, dimensions, and provenance.
- [ ] Validate every worker dump before reduce: schema version, config hash, mode, dimensions, dtype, sequence count, candidate width, candidate ID range, finite values, and non-negative values.
- [ ] Validate all worker dumps agree on `num_components`, `d_sae`, `n_latents_per_latent`, `n_candidates_per_component`, `M`, and coactivation mode.
- [ ] Verification: add schema validation tests for good dumps, stale config, wrong mode, wrong shape, invalid candidate IDs, and non-finite values.

## Phase 2 - Global `top_ctx` Target Mapping

- [ ] Load merged global `top_ctx.pt` and rebuild the sequence-to-target CSR used by the current reducer.
- [ ] Build `seq_offsets` and `seq_targets_global` from global `top_ctx`, not from worker-local state.
- [ ] Build a global `sid_to_row` mapping for the merged dump row order.
- [ ] Validate every worker dump sequence ID appears in the global replay sequence set.
- [ ] Validate every global replay sequence is present exactly once across simple candidate dumps.
- [ ] Verification: add tests for duplicate sequence IDs, missing sequence IDs, extra sequence IDs, zero sequence IDs, and unsorted worker dump rows.

## Phase 3 - Mode A: Simple Exact Merge

- [ ] Concatenate worker candidate dumps in deterministic sequence-ID order.
- [ ] Preserve row alignment between `sequence_ids`, `candidate_ids`, and `candidate_vals`.
- [ ] Build a single in-memory candidate dump compatible with the existing C++ reducer signature.
- [ ] Run the existing reducer over the full target range or with the existing `target_sharded` API.
- [ ] Support file-backed reducer shards using the existing shard schema where possible.
- [ ] Verification: add a two-worker synthetic test where concatenated distributed dumps produce identical reducer output to a single-process dump.

## Phase 4 - Existing C++ Reducer Integration

- [ ] Reuse `top_coactivation_reduce.reduce_topk()` for simple exact mode.
- [ ] Require the Phase 7 reducer API with `target_start` and `target_end` for target-sharded simple mode.
- [ ] Detect legacy native extensions and fail with a clear rebuild message when target-sharded reduce is requested.
- [ ] Preserve OpenMP controls: `reduce_omp_threads` and `reduce_schedule_chunk`.
- [ ] Preserve deterministic tie-breaking from the native reducer.
- [ ] Verification: run native reducer tests and store-level target-sharded equivalence tests after integration.

## Phase 5 - PMI Postprocess

- [ ] Apply PMI postprocess exactly once after global reduction, not per worker.
- [ ] Use merged global `latent_stats.active_count` to compute global candidate firing rates.
- [ ] Use global `seq_offsets`, `seq_targets_global`, and sequence length metadata to compute target-context rates.
- [ ] Sum worker token-count metadata only as a validation input, not as a replacement for global pass-1 counts.
- [ ] Validate PMI clamp bounds and finite output values.
- [ ] Verification: add tests proving distributed PMI output matches single-process PMI output on a synthetic fixture.

## Phase 6 - Canonical Output Writer

- [ ] Reshape reducer output into `[num_components, d_sae, n_latents_per_latent]`.
- [ ] Save canonical `outputs/<run_id>/top_coactivation.pt` with the same field names and metadata expected by current downstream code.
- [ ] Add optional reducer report metadata: mode, backend, shard count, input worker count, replay sequence count, reduce time, PMI time, and output nonzero count.
- [ ] Write outputs atomically and avoid leaving partial canonical artifacts on failure.
- [ ] Validate saved artifact can be loaded by existing `TopCoactivation.load()`.
- [ ] Verification: add save/load round-trip tests for merged reducer output.

## Phase 7 - Mode B: MapReduce Partial-Sum Design

- [ ] Implement MapReduce only after Mode A is correct and benchmarked.
- [ ] Define target-range partitions for reducers using the same flattened target ID space as the native reducer.
- [ ] Split flattened target IDs into balanced contiguous ranges, distributing remainder targets across the first reducer shards deterministically.
- [ ] Never drop target IDs because the target count is not divisible by reducer count.
- [ ] Shard reducers by target range only; each reducer owns all candidate columns for its assigned target rows.
- [ ] Preserve cross-range edges by allowing any candidate ID in every target-range reducer shard.
- [ ] Do not shard by both target range and candidate range unless a future 2D block-reduction design explicitly preserves all cross-block contributions.
- [ ] Have workers or a shuffle step emit partial sums partitioned by target range: `(target_latent, candidate_latent, partial_value)`.
- [ ] Ensure each reducer range receives partial sums from every worker that has contributions for that range.
- [ ] Merge duplicate `(target, candidate)` keys by summing values before selecting top-K.
- [ ] Keep top-K selection deterministic with value descending then candidate ID ascending.
- [ ] Verification: prove MapReduce output equals Mode A output on tiny synthetic, tie-heavy synthetic, uneven-target-count synthetic, cross-range-candidate synthetic, and reduced real-data fixtures.

## Phase 8 - MapReduce Storage And Memory Layout

- [ ] Use sorted COO partial-sum shard files first: `target_ids`, `candidate_ids`, and `values`, sorted by `(target_id, candidate_id)`.
- [ ] Include all candidate IDs in each reducer target shard, not just candidates inside the reducer's target range.
- [ ] Prefer this tensor-first sorted COO format because it is easier to inspect, test, concatenate, and merge than CSR during the first exact implementation.
- [ ] Include target range, worker ID, row count, candidate count, value dtype, and checksum/hash metadata in every shard.
- [ ] Add memory estimates before loading all partial-sum shards for a reducer range.
- [ ] Support streaming or chunked merges if a target range's partial sums are too large.
- [ ] Verification: add round-trip and chunked-merge tests for the chosen partial-sum format.

## Phase 9 - Reducer Sharding And Scheduling

- [ ] Support single reducer, target-sharded reducer, and MapReduce target-range reducer modes through explicit config/manifest fields.
- [ ] Implement CPU/OpenMP MapReduce reducers first to lock exact semantics before considering GPU reducers.
- [ ] Allow reducer shards to run sequentially first for debuggability.
- [ ] Add a later parallel reducer execution path only after shard outputs are deterministic and independently validated.
- [ ] Ensure shard files can be resumed or cleaned up safely after failure.
- [ ] Preserve final output shape and canonical artifact path regardless of reducer mode.
- [ ] Verification: add tests for shard cleanup, stale shard rejection, resume behavior, remainder-preserving range partitioning, and final stitch correctness.

## Phase 10 - Equivalence Test Matrix

- [ ] Compare current single-process reduce against distributed Mode A simple exact reduce.
- [ ] Compare distributed Mode A against distributed Mode B MapReduce exact reduce.
- [ ] Test `raw`, `freq_weighted`, and `pmi` modes.
- [ ] Test one-worker, two-worker, and many-worker synthetic layouts.
- [ ] Include tie cases where candidate values are equal.
- [ ] Include cases where a candidate is not local top-K per worker but becomes global top-K after summing, proving local top-K merging would be wrong.
- [ ] Verification: require exact `top_indices` equality and close `top_values` equality within a documented tolerance.

## Phase 11 - Benchmark And Reporting

- [ ] Log input dump sizes, merge time, reducer time, PMI time, shard write/merge time, peak CPU RAM, and output artifact size.
- [ ] Benchmark simple exact merge before implementing MapReduce to determine whether central dump merge is actually the bottleneck.
- [ ] Benchmark target-sharded simple mode with representative shard counts.
- [ ] Benchmark MapReduce only after it passes equivalence tests.
- [ ] Record results in the manifest and a reducer report beside `top_coactivation.pt`.
- [ ] Verification: document benchmark commands and configs in this file after implementation.

## Phase 12 - Testing And Verification

- [ ] Run focused unit tests for dump validation, global sequence coverage, simple merge, reducer integration, PMI postprocess, output save/load, and shard stitching.
- [ ] Run native reducer tests after rebuilding extensions.
- [ ] Run synthetic end-to-end pass-2 dump plus reduce equivalence tests.
- [ ] Run reduced real-data comparison against the current single-process pass 2.
- [ ] Run H100 reducer benchmarks only after local/synthetic exactness is proven.
- [ ] Document exact verification commands in this file after implementation.

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
