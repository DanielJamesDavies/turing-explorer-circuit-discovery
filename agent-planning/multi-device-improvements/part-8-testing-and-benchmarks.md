# Plan: Part 8 - Testing And Benchmarks

> **Goal:** Prove the distributed pipeline is correct, reproducible, and worth using by combining unit tests, equivalence tests, smoke runs, H100 benchmarks, and paper-facing artifact validation.
>
> **Created:** 2026-05-16

---

## Scope

This part defines the verification program for Parts 1-7.

It covers unit tests, artifact merge tests, exactness checks against `single_process`, local smoke runs, reduced real-data comparisons, H100 benchmarks, and paper-facing reproducibility reports.

It does not add new distributed algorithms. It decides when the earlier parts are safe to use.

---

## Phase 1 - Test Taxonomy And Gates

- [ ] Define required test categories: unit, schema, merge, equivalence, smoke, benchmark, and reproducibility.
- [ ] Define pass/fail gates for each operating mode: `single_process`, `distributed_simple_exact`, `distributed_mapreduce_exact`, and `distributed_experimental_fast`.
- [ ] Require `distributed_simple_exact` to pass one-worker equivalence before any multi-worker run is trusted.
- [ ] Require `distributed_mapreduce_exact` to match `distributed_simple_exact` before it is recommended.
- [ ] Require experimental fast modes to be compared against exact artifacts from the same config or dataset slice.
- [ ] Verification: add a checklist file or report schema that records which gates have passed for each run.

## Phase 2 - Part 1 Manifest And Runtime Tests

- [ ] Test manifest schema validation, JSON round trips, schema version rejection, stale config rejection, and duplicate worker rejection.
- [ ] Test default run ID generation uses `YYYYMMDD-HHMMSS-<config_hash_8>`.
- [ ] Test schema-version rejection for manifests, partial artifacts, metrics JSONL, sanity reports, and run summaries.
- [ ] Test canonical global sequence ID table construction from shard sequence counts.
- [ ] Test global sequence ID table construction with variable shard lengths, including a shorter final shard.
- [ ] Test stale global sequence table rejection when shard order, shard files, or sequence counts change.
- [ ] Test deterministic shard, sequence, seed, and device assignment helpers.
- [ ] Test pass-1 whole-shard assignment is balanced by actual sequence count, not shard count.
- [ ] Test contiguous sequence/list partitioners distribute remainder items deterministically and never drop them.
- [ ] Test one-device worker isolation and assert distributed workers pass a single-device list to `SAEBank`.
- [ ] Test physical/logical GPU metadata presence, including mocked GPU UUID/name/PCI bus ID when available.
- [ ] Test duplicate physical GPU assignment rejection unless explicit oversubscription/debug mode is selected.
- [ ] Test controller-emitted worker commands include the expected per-worker `CUDA_VISIBLE_DEVICES` values.
- [ ] Test optional subprocess launch planning uses the same command contract as dry-run mode.
- [ ] Test worker output layout creation and marker validation.
- [ ] Test preflight failures for output root not writable, existing run ID without resume, invalid config, stale shard table, unavailable/duplicate devices, insufficient disk estimate, and missing native extensions for selected parts.
- [ ] Test cleanup/retention policy behavior for `keep_all`, `delete_large_partials_on_success`, `delete_all_partials_on_success`, and `manual_cleanup_only`.
- [ ] Test failed-run partials, logs, metrics, and failure markers are preserved by default.
- [ ] Test JSONL metric event schema for controller and worker metrics.
- [ ] Test run-root layout: distributed state lives under `outputs/<run_id>/distributed/`.
- [ ] Test universal run-root layout: both `single_process` and distributed modes write canonical artifacts under `outputs/<run_id>/`.
- [ ] Test dry-run creation for one-worker local and synthetic 8-worker H100 layouts.
- [ ] Test resume classification for pending, completed, failed, stale, partial, and missing workers.
- [ ] Verification: run focused Part 1 tests before any distributed worker implementation is used.

## Phase 3 - Part 2 Pass-1 Merge Tests

- [ ] Test exact Welford merges for `latent_stats` token-level and sequence-level stats.
- [ ] Test `top_ctx` global top-K merge with deterministic tie-breaking.
- [ ] Test deterministic priority-reservoir `mid_ctx` merge equals a single global priority-reservoir pass for any worker split.
- [ ] Test oversampled `mid_ctx` candidate-pool filtering equals a single global priority-reservoir pass when candidate coverage is sufficient.
- [ ] Test distributed `mid_ctx` collection uses merged global stats, not worker-local stats, when deciding mid-band membership.
- [ ] Test deterministic priority-reservoir `mid_ctx` sampling is uniform over valid examples across seeded trials.
- [ ] Test `mid_ctx` candidate-pool truncation/coverage failures are detected.
- [ ] Test default `mid_ctx` candidate-pool config: enabled, `band_margin_sigma: 1.0`, `max_candidates_per_latent: max(256, 4 * num_ctx_sequences)`, and `on_truncation: replay_fallback`.
- [ ] Test `mid_ctx` replay fallback produces the same output as full global priority-reservoir selection.
- [ ] Test `allow_bounded_approx` candidate-pool output cannot be marked paper-ready.
- [ ] Test candidate-pool large partials are deleted only after final `mid_ctx.pt` validation and cleanup policy allows it.
- [ ] Test deterministic `seq_repr` capped and uncapped merges against one global `slot_to_id`/`id_to_slot` mapping.
- [ ] Test `seq_repr` cap determinism under fixed seed and cap changes under changed seed.
- [ ] Test `distributed.sampling_seed` reproducibility for both `seq_repr` and `mid_ctx`.
- [ ] Test changing only `run_id` does not change deterministic `seq_repr` or `mid_ctx` samples for the same config and dataset fingerprint.
- [ ] Test `logit_ctx` event top-K token/prob/count merge semantics.
- [ ] Test `logit_ctx` tie-breaks are stable across worker split and merge order.
- [ ] Test `seq_latent_index` shard merge and duplicate rejection.
- [ ] Verification: run synthetic split-stream equivalence tests comparing merged pass-1 artifacts against single-process pass 1.

## Phase 4 - Part 3 Negative-Context Tests

- [ ] Test `single_gpu_exact` remains the correctness baseline for local/reduced runs.
- [ ] Test `multi_gpu_exact` device parsing, component partitioning, stats merge, and validation logic.
- [ ] Test distributed `neg_ctx` defaults to manifest-declared devices, while standalone mode may use all visible devices.
- [ ] Test memory estimate and guardrail behavior without requiring CUDA.
- [ ] Test synthetic single-vs-multi equivalence where component splits should produce identical rows.
- [ ] Test `neg_ctx` sanity report generation and invalid tensor failure.
- [ ] Verification: compare `single_gpu_exact` and `multi_gpu_exact` on a reduced real-data run before H100 use.

## Phase 5 - Part 4 Pass-2 Dump Tests

- [ ] Test refactored candidate-profile computation against current `TopCoactivation.update_batch()` for `raw`, `freq_weighted`, and `pmi`.
- [ ] Test global replay sequence list construction from `top_ctx`.
- [ ] Test worker sequence partitioning covers every replay sequence exactly once.
- [ ] Test pass-2 replay assignments are contiguous chunks in replay-list order and preserve all remainder sequences.
- [ ] Test pass-2 worker resource construction rejects multi-device `SAEBank` placement.
- [ ] Test partial candidate dump schema validation and atomic writes.
- [ ] Test one-worker and two-worker dump equivalence against current single-process dump.
- [ ] Verification: require worker token counts and row mappings to match single-process expectations for PMI mode.

## Phase 6 - Part 5 Reducer Tests

- [ ] Test simple exact candidate-dump concatenation and global `sid_to_row` construction.
- [ ] Test simple exact distributed reduce against current single-process reduce.
- [ ] Test target-sharded simple reduce and shard stitching.
- [ ] Test PMI postprocess equivalence after distributed reduce.
- [ ] Test MapReduce partial-sum reduce against simple exact reduce.
- [ ] Test target-only reducer sharding preserves cross-range candidate IDs.
- [ ] Test target-range reducer partitioning distributes remainder targets deterministically and never drops target IDs.
- [ ] Test sorted COO partial-sum shard round trips and merge ordering by `(target_id, candidate_id)`.
- [ ] Test CPU/OpenMP MapReduce reducers before any GPU reducer path is considered.
- [ ] Test tie cases and cases where local top-K merging would be wrong.
- [ ] Verification: run native reducer tests after rebuilding extensions whenever reducer APIs change.

## Phase 7 - Part 6 Discovery Tests

- [ ] Test centralized candidate selection over merged global artifacts.
- [ ] Test seed/task partitioning for one-worker, many-worker, and more-workers-than-seeds cases.
- [ ] Test discovery worker resource construction rejects multi-device `SAEBank` placement.
- [ ] Test mocked discovery worker output without loading real model weights.
- [ ] Test circuit-store merge, UUID collision handling, summary merge, and empty worker outputs.
- [ ] Test seed-free method ownership so methods such as `cluster_contrast` do not run once per worker by accident.
- [ ] Verification: run a small discovery/eval smoke before distributed H100 discovery.

## Phase 8 - Part 7 Mode And UX Tests

- [ ] Test config validation for all operating modes.
- [ ] Test `distributed` is the accepted orchestration config namespace.
- [ ] Test strict Pydantic validation rejects unknown or misspelled distributed config keys.
- [ ] Test distributed mode config keeps search-cache generation offline/deferred by default.
- [ ] Test preflight command/report output for local one-worker and synthetic 8-worker modes.
- [ ] Test command parser/help output for distributed entrypoints.
- [ ] Test local one-worker and synthetic 8-worker dry runs.
- [ ] Test output policy: unvalidated distributed partials stay under `outputs/<run_id>/distributed/`, and canonical run artifacts appear at the top of `outputs/<run_id>/` only after checks pass.
- [ ] Test JSONL run/worker metrics schema and report paths.
- [ ] Test mocked device observability sampler output for GPU utilization, VRAM, power, temperature, CPU RAM, disk usage, phase label, worker PID, and physical GPU identity.
- [ ] Test rollout gates reject unsafe mode transitions.
- [ ] Test run reports show exactness status, part statuses, artifact paths, benchmark results, and warnings.
- [ ] Verification: ensure `python src/main.py` remains the `single_process` entrypoint while writing canonical artifacts under `outputs/<run_id>/`.

## Phase 9 - End-To-End Synthetic Equivalence

- [ ] Build a tiny synthetic dataset and mocked/lightweight model/SAE fixture where expected artifacts can be compared cheaply.
- [ ] Run current single-process pipeline or equivalent staged fixture.
- [ ] Confirm the single-process fixture writes canonical outputs under `outputs/<run_id>/`.
- [ ] Run one-worker `distributed_simple_exact` and compare canonical artifacts.
- [ ] Run two-worker `distributed_simple_exact` and compare canonical artifacts.
- [ ] Once implemented, run `distributed_mapreduce_exact` and compare to `distributed_simple_exact`.
- [ ] Verification: require exact tensor equality where deterministic, and documented tolerances where floating-point order differs.

## Phase 10 - Local Real-Data Smoke

- [ ] Run a local one-worker distributed dry run on RTX 5070 Ti style settings.
- [ ] Run a small local distributed smoke with reduced `n_shards`, `n_seeds`, and efficient memory settings.
- [ ] Compare artifact shapes and sanity stats against the current local `single_process` run.
- [ ] Confirm local runs do not require H100-specific configs or multi-GPU hardware.
- [ ] Record wall time and peak memory, but treat correctness as the main local goal.
- [ ] Verification: save local smoke logs and summary reports under a clearly labeled run directory.

## Phase 11 - Reduced Real-Data Multi-Worker Benchmark

- [ ] Run reduced real-data `single_process` baseline.
- [ ] Run reduced real-data one-worker distributed equivalence.
- [ ] Run reduced real-data two-worker distributed equivalence.
- [ ] If available, run reduced real-data eight-worker H100 smoke.
- [ ] Compare `latent_stats.pt`, `top_ctx.pt`, `mid_ctx.pt`, `neg_ctx.pt`, `top_coactivation.pt`, `candidates.pt`, and `circuits/summary.json` under the same run root.
- [ ] Verification: require artifact sanity checks and circuit-output comparison before larger H100 benchmarks.

## Phase 12 - Full H100 Benchmark Protocol

- [ ] Rebuild native extensions on the target H100 host before any benchmark.
- [ ] Run native tests: `src/native/tests/test_topk.py` and `src/native/tests/test_reduce.py`.
- [ ] Run `single_process` or current-runtime H100 baseline first.
- [ ] Run `distributed_simple_exact` with 1, 2, 4, and 8 workers where practical.
- [ ] Benchmark Part 2 pass 1, Part 3 neg_ctx, Part 4 pass-2 dump, Part 5 reduce, and Part 6 discovery separately.
- [ ] Record wall time, GPU utilization, peak VRAM, CPU RAM, disk usage/write throughput, artifact sizes, worker imbalance, cleanup policy effects, and circuit outputs.
- [ ] Record JSONL metrics and device observability samples for each H100 benchmark.
- [ ] Verification: do not recommend H100 distributed defaults until correctness and benchmark reports are complete.

## Phase 13 - MapReduce Decision Gate

- [ ] Use `distributed_simple_exact` H100 benchmarks to determine whether central candidate-dump merge is a real bottleneck.
- [ ] Measure candidate dump size, concatenation time, reducer input memory, reducer time, and shard write/merge overhead.
- [ ] Start or enable `distributed_mapreduce_exact` only if simple exact merge is too slow or memory-heavy.
- [ ] Require MapReduce exact output to match simple exact output on synthetic and reduced real-data fixtures.
- [ ] Benchmark MapReduce only after equivalence is proven.
- [ ] Verification: write a decision note before recommending MapReduce as an operating mode.

## Phase 14 - Paper-Facing Reproducibility Package

- [ ] Save exact config, run manifest, git SHA, native build info, environment summary, and command history for every paper-facing run.
- [ ] Save physical/logical GPU identity metadata and JSONL metrics for every paper-facing run.
- [ ] Save global sequence ID table, `seq_repr` cap mapping, and `mid_ctx` priority-reservoir seed/hash version for every paper-facing run.
- [ ] Save `mid_ctx` candidate-pool settings, coverage/truncation report, and replay-fallback status for every paper-facing run.
- [ ] Save artifact sanity reports for all canonical outputs.
- [ ] Save equivalence reports showing distributed artifacts match the chosen correctness oracle.
- [ ] Save benchmark reports with hardware description and per-part timing.
- [ ] Save circuit summaries and enough provenance to trace each circuit back to seed, worker, method, and config.
- [ ] Verification: define a `paper_ready` report status that requires exact mode, passing sanity checks, and complete provenance.

## Phase 15 - Regression And CI Strategy

- [ ] Keep most distributed tests CPU/synthetic so they can run in normal CI or local development.
- [ ] Mark CUDA/H100 tests explicitly and keep them opt-in.
- [ ] Add focused test commands for each part to avoid always running the full suite.
- [ ] Add a reduced smoke command for pre-H100 validation.
- [ ] Track full-suite checkpoints after major distributed changes.
- [ ] Verification: document which tests are required before local merge, before H100 run, and before paper-facing use.

---

## Open Questions

- What tolerance is acceptable for floating-point differences in merged Welford stats and reducer values?
- Should H100 benchmark reports be plain JSON, Markdown summaries, or both?
- What minimum reduced real-data size is enough before trusting a full H100 run?
- Should paper-facing runs require `distributed_simple_exact`, or can `distributed_mapreduce_exact` be accepted once equivalence is proven?
- Which artifact comparisons are required for circuits: exact circuit UUIDs, exact summaries, or metric-level equivalence?

## Risks / Assumptions

- Passing unit tests is not enough; exactness must be shown against the current single-process oracle.
- H100 speedups can hide semantic drift if artifact comparisons are incomplete.
- Some floating-point differences may be harmless, but tolerances must be explicit and justified.
- Experimental fast modes must not be mixed into paper-facing exact benchmark outputs.
- The benchmark protocol must measure combine/merge overhead, not just GPU worker time.
