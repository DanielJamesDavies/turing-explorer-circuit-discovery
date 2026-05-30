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

## Local-First Execution Order

Use this phase order now that H100-only work is separated into [`part-9-h100-validation-and-benchmarks.md`](part-9-h100-validation-and-benchmarks.md):

1. **Phases 1-8:** Reconcile/check off what existing unit, schema, merge, config, rollout, reporting, and UX tests already cover.
2. **Phase 9:** Do synthetic end-to-end equivalence next, because it proves the pieces connect.
3. **Phase 10:** Then local real-data smoke if usable local artifacts/data are available.
4. **Phase 11:** Do reduced real-data multi-worker only as far as local hardware allows.
5. **Phases 12-14:** Move or mark H100 and paper-facing benchmark items as deferred to Part 9.
6. **Phase 15:** Finish with CI/regression strategy so the local checks are repeatable.

---

## Phase 1 - Test Taxonomy And Gates

- [x] Define required test categories: unit, schema, merge, equivalence, smoke, benchmark, and reproducibility.
- [x] Define pass/fail gates for each operating mode: `single_process`, `distributed_simple_exact`, `distributed_mapreduce_exact`, and `distributed_experimental_fast`.
- [x] Require `distributed_simple_exact` to pass one-worker equivalence before any multi-worker run is trusted.
- [x] Require `distributed_mapreduce_exact` to match `distributed_simple_exact` before it is recommended.
- [x] Require experimental fast modes to be compared against exact artifacts from the same config or dataset slice.
- [x] Verification: add a checklist file or report schema that records which gates have passed for each run.

### Phase 1 Notes

Required local-first test categories are now:

- `unit`: pure helper behavior, config validation, deterministic partitioning, CLI parser behavior, and mocked runtime construction.
- `schema`: manifest, partial artifact, marker, metrics JSONL, sanity report, run summary, and observability schemas.
- `merge`: exact mathematical merge rules for pass-1 artifacts, pass-2 dump concatenation, pass-2 reduce, discovery stores, and summaries.
- `equivalence`: artifact comparisons against `single_process` or a simpler exact distributed oracle.
- `smoke`: one-worker and small multi-worker synthetic/local runs that exercise real stage boundaries without claiming H100 performance.
- `benchmark`: timing/resource reports. Local benchmarks are informational only; H100 benchmark gates are deferred to Part 9.
- `reproducibility`: config hash, manifest, git SHA, artifact hashes, deterministic seeds, reports, and provenance needed to rerun or audit a result.

Operating-mode gates:

- `single_process`: remains the correctness oracle. It must keep producing canonical artifacts and must be available before distributed equivalence can be trusted.
- `distributed_simple_exact`: requires local unit/schema/merge tests, successful sanity reports, and one-worker equivalence before any multi-worker run is trusted. Paper-facing use additionally requires tiny synthetic and reduced real-data equivalence.
- `distributed_mapreduce_exact`: requires all `distributed_simple_exact` gates plus `equivalence_mapreduce_vs_simple.json` before it can be trusted or recommended.
- `distributed_experimental_fast`: requires explicit acknowledgement, an exact baseline root, quality-changing toggles recorded in the manifest, and separate clearly marked outputs. It is not paper-eligible by default.

Existing gate/report support:

- `src/pipeline/distributed/rollout_gates.py` defines the report filenames used by the gate contract: `verification_status.json`, `equivalence_one_worker.json`, `equivalence_tiny_synthetic.json`, `equivalence_reduced_real.json`, `equivalence_mapreduce_vs_simple.json`, and `benchmark_report.json`.
- `validate_rollout_gates()` requires the manifest, verification status, sanity reports, one-worker equivalence for multi-worker `distributed_simple_exact`, MapReduce-vs-simple equivalence for `distributed_mapreduce_exact`, and benchmark reports before a mode can be recommended as default.
- `write_rollout_gate_report()` persists a stable gate report with schema version, run ID, mode, pass/fail state, issues, and required paths.
- `src/pipeline/distributed/reporting.py` provides `mode_summary.json`, final run report construction, hardware context summaries, and observability JSONL schema support.

Local/H100 split:

- Part 8 owns local and synthetic gates, including unit/schema/merge coverage, one-worker equivalence, local smoke tests, reduced-data checks where local hardware allows, CI strategy, and local readiness reporting.
- Part 9 owns H100-only gates, including 1/2/4/8-worker H100 benchmarks, H100 observability, full-scale speedup claims, MapReduce promotion based on target-machine bottlenecks, and paper-facing H100 reproducibility bundles.
- Missing H100 access must not block Phase 1 completion; it should mark H100 benchmark gates as deferred to Part 9 rather than failed.

Verification:

- Existing coverage: `tests/pipeline/test_distributed_rollout_gates.py`, `tests/pipeline/test_distributed_reporting.py`, `tests/pipeline/test_distributed_operating_modes.py`, `tests/pipeline/test_distributed_config.py`, and `tests/pipeline/test_distributed_experimental_modes.py`.
- Recommended focused command for this phase:

```powershell
python -m pytest tests/pipeline/test_distributed_rollout_gates.py tests/pipeline/test_distributed_reporting.py tests/pipeline/test_distributed_operating_modes.py tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_experimental_modes.py -q
```

## Phase 2 - Part 1 Manifest And Runtime Tests

- [x] Test manifest schema validation, JSON round trips, schema version rejection, stale config rejection, and duplicate worker rejection.
- [x] Test default run ID generation uses `YYYYMMDD-HHMMSS-<config_hash_8>`.
- [x] Test schema-version rejection for manifests, partial artifacts, metrics JSONL, sanity reports, and run summaries.
- [x] Test canonical global sequence ID table construction from shard sequence counts.
- [x] Test global sequence ID table construction with variable shard lengths, including a shorter final shard.
- [x] Test stale global sequence table rejection when shard order, shard files, or sequence counts change.
- [x] Test deterministic shard, sequence, seed, and device assignment helpers.
- [x] Test pass-1 whole-shard assignment is balanced by actual sequence count, not shard count.
- [x] Test contiguous sequence/list partitioners distribute remainder items deterministically and never drop them.
- [x] Test one-device worker isolation and assert distributed workers pass a single-device list to `SAEBank`.
- [x] Test physical/logical GPU metadata presence, including mocked GPU UUID/name/PCI bus ID when available.
- [x] Test duplicate physical GPU assignment rejection unless explicit oversubscription/debug mode is selected.
- [x] Test controller-emitted worker commands include the expected per-worker `CUDA_VISIBLE_DEVICES` values.
- [x] Test optional subprocess launch planning uses the same command contract as dry-run mode.
- [x] Test worker output layout creation and marker validation.
- [x] Test preflight failures for output root not writable, existing run ID without resume, invalid config, stale shard table, unavailable/duplicate devices, insufficient disk estimate, and missing native extensions for selected parts.
- [x] Test cleanup/retention policy behavior for `keep_all`, `delete_large_partials_on_success`, `delete_all_partials_on_success`, and `manual_cleanup_only`.
- [x] Test failed-run partials, logs, metrics, and failure markers are preserved by default.
- [x] Test JSONL metric event schema for controller and worker metrics.
- [x] Test run-root layout: distributed state lives under `outputs/<run_id>/distributed/`.
- [x] Test universal run-root layout: both `single_process` and distributed modes write canonical artifacts under `outputs/<run_id>/`.
- [x] Test dry-run creation for one-worker local and synthetic 8-worker H100 layouts.
- [x] Test resume classification for pending, completed, failed, stale, partial, and missing workers.
- [x] Verification: run focused Part 1 tests before any distributed worker implementation is used.

### Phase 2 Notes

Coverage map:

- `tests/pipeline/test_distributed_manifest.py` covers manifest schema validation, JSON round trips, run ID generation, schema-version rejection, run/mode/cleanup validation, duplicate worker IDs, duplicate physical device IDs, stale config-hash checks, and assignment consistency against shard tables.
- `tests/pipeline/test_distributed_shard_table.py` covers canonical global sequence ID table construction, variable shard lengths, shorter final shards, worker-assigned global sequence IDs, missing/reordered shards, stale shard indices, changed sequence counts, duplicate sequence ranges, and out-of-range assigned shards.
- `tests/pipeline/test_distributed_assignments.py` covers deterministic pass-1 shard balancing by sequence count, deterministic tie-breaking, empty/one-worker/more-workers-than-items cases, contiguous remainder-preserving partitioning, sequence/seed duplicate rejection, discovery seed assignment, method-aware scheduling, and resume task selection.
- `tests/pipeline/test_distributed_devices.py` covers physical/logical GPU identity metadata, duplicate physical GPU rejection, CPU one-worker fallback, worker-local `CUDA_VISIBLE_DEVICES`, logical `cuda:0` isolation, and distributed runtime construction with exactly one `SAEBank` device.
- `tests/pipeline/test_distributed_layout.py` covers canonical run-root/distributed-root layout, worker directories, atomic worker markers, completed/failed marker validation, JSONL metric event schema, artifact existence checks, cleanup candidate selection, and failed-run preservation.
- `tests/pipeline/test_distributed_controller.py` covers model-free dry-run planning, emitted worker commands, per-worker environment variables, one-worker local planning, synthetic 8-worker H100 planning, optional subprocess launch planning, resume classification, run-ID collision/resume policy, preflight shard-table construction, missing dataset rejection, unwritable output roots, invisible/duplicate devices, CPU fallback limits, disk-space estimates, strict config rejection, and missing native-extension gates.
- `tests/pipeline/test_distributed_interfaces.py` covers run-root output path resolution and verifies that single-process-compatible save points can target `outputs/<run_id>/` without changing artifact names.
- `tests/pipeline/test_distributed_resume_policy.py` covers part-level resume states, completed/failed/stale/partial/missing classification, merge safety, cleanup plans, and preservation of failed-run partials.

Important caveat:

- The Part 1 interface tests prove that `single_process`-compatible output paths can resolve under `outputs/<run_id>/`, and that persistence/save helpers can target a run root. They do not by themselves prove that the default `python src/main.py` path has been fully rewired to create and use a run ID. Keep that as an implementation/Phase 9 synthetic smoke concern rather than treating this Phase 2 test coverage as full end-to-end `single_process` run-root proof.

Verification:

```powershell
python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_resume_policy.py -q
```

## Phase 3 - Part 2 Pass-1 Merge Tests

- [x] Test exact Welford merges for `latent_stats` token-level and sequence-level stats.
- [x] Test `top_ctx` global top-K merge with deterministic tie-breaking.
- [x] Test deterministic priority-reservoir `mid_ctx` merge equals a single global priority-reservoir pass for any worker split.
- [x] Test oversampled `mid_ctx` candidate-pool filtering equals a single global priority-reservoir pass when candidate coverage is sufficient.
- [x] Test distributed `mid_ctx` collection uses merged global stats, not worker-local stats, when deciding mid-band membership.
- [x] Test deterministic priority-reservoir `mid_ctx` sampling is uniform over valid examples across seeded trials.
- [x] Test `mid_ctx` candidate-pool truncation/coverage failures are detected.
- [x] Test default `mid_ctx` candidate-pool config: enabled, `band_margin_sigma: 1.0`, `max_candidates_per_latent: max(256, 4 * num_ctx_sequences)`, and `on_truncation: replay_fallback`.
- [x] Test `mid_ctx` replay fallback produces the same output as full global priority-reservoir selection.
- [x] Test `allow_bounded_approx` candidate-pool output cannot be marked paper-ready.
- [x] Test candidate-pool large partials are deleted only after final `mid_ctx.pt` validation and cleanup policy allows it.
- [x] Test deterministic `seq_repr` capped and uncapped merges against one global `slot_to_id`/`id_to_slot` mapping.
- [x] Test `seq_repr` cap determinism under fixed seed and cap changes under changed seed.
- [x] Test `distributed.sampling_seed` reproducibility for both `seq_repr` and `mid_ctx`.
- [x] Test changing only `run_id` does not change deterministic `seq_repr` or `mid_ctx` samples for the same config and dataset fingerprint.
- [x] Test `logit_ctx` event top-K token/prob/count merge semantics.
- [x] Test `logit_ctx` tie-breaks are stable across worker split and merge order.
- [x] Test `seq_latent_index` shard merge and duplicate rejection.
- [x] Verification: run synthetic split-stream equivalence tests comparing merged pass-1 artifacts against single-process pass 1.

### Phase 3 Notes

Coverage map:

- `tests/pipeline/test_distributed_pass1_partials.py` covers pass-1 partial round trips, stale config rejection, non-finite tensor rejection, `mid_ctx` priority seed reproducibility, run-ID-independent priorities, band-setting changes, and priority uniformity across seeded trials.
- `tests/pipeline/test_distributed_pass1_merge.py` covers Welford token/sequence merges, order invariance, equality against `LatentStats.update_component()`, path-based load/merge helpers, `top_ctx` global top-K selection, invalid sentinel cleanup, sequence-range rejection, `mid_ctx` filtering by merged global stats, priority selection, truncation fail/replay/bounded-approx policies, replay fallback execution, cleanup eligibility, `seq_repr` cap mapping, capped/uncapped `seq_repr` merges, duplicate slot rejection, `logit_ctx` event top-K/count merges, logit tie-breaking, vocabulary and finite-value validation, `seq_latent_index` shard copying and duplicate handling, and global pass-1 artifact writing with a sanity report.
- `tests/pipeline/test_distributed_worker.py` covers pass-1 worker execution over assigned shards, worker markers, worker partial artifact names, manifest total sequence handling, `mid_ctx` candidate-pool default widening/capacity, and pass-1 worker input validation for missing, duplicate, out-of-range, stale, or missing shard assignments.
- `tests/store/test_mid_ctx_modes.py` preserves local store-level coverage for existing `mid_ctx` modes and metadata so distributed `mid_ctx` changes do not break non-distributed behavior.

Synthetic equivalence status:

- Current Phase 3 coverage includes mathematical split-stream equivalence for `latent_stats`, event-stream equivalence for `logit_ctx`, deterministic merge-order stability for `mid_ctx`, one-worker pass-1 merge compatibility, and a one-worker merged pass-1 artifact set feeding the negative-context stage.
- A full end-to-end synthetic run that starts at `single_process` and compares every canonical artifact against a distributed run remains a Phase 9 concern, because it validates cross-stage orchestration rather than individual pass-1 merge rules.

Verification:

```powershell
python -m pytest tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/store/test_mid_ctx_modes.py -q
```

## Phase 4 - Part 3 Negative-Context Tests

- [x] Test `single_gpu_exact` remains the correctness baseline for local/reduced runs.
- [x] Test `multi_gpu_exact` device parsing, component partitioning, stats merge, and validation logic.
- [x] Test distributed `neg_ctx` defaults to manifest-declared devices, while standalone mode may use all visible devices.
- [x] Test memory estimate and guardrail behavior without requiring CUDA.
- [x] Test synthetic single-vs-multi equivalence where component splits should produce identical rows.
- [x] Test `neg_ctx` sanity report generation and invalid tensor failure.
- [x] Verification: compare `single_gpu_exact` and `multi_gpu_exact` on a reduced real-data run before H100 use.

### Phase 4 Notes

Coverage map:

- `tests/pipeline/test_negative_context_stage.py` covers run-root pass-1 artifact loading, missing/incompatible artifact rejection, config-hash validation, canonical `neg_ctx.pt` and `neg_ctx_stats.json` writes, an actual CPU `single_gpu_exact` backend smoke, manifest-declared device selection for distributed runs, part markers, sanity reports, invalid-output failure markers, resume skip/stale/failed classification, standalone all-visible-device metadata for sharded mode, and mocked single-vs-multi backend equivalence reports.
- `tests/store/test_neg_context_backend.py` covers `multi_gpu_exact`-oriented device parsing, CUDA availability/range validation, deterministic component partitioning, stats merging, seq-repr metadata in stats, ANN memory estimates, CUDA memory guardrail pass/fail/warn behavior, component assignment and per-device timing metadata, `ann_device` parsing, CPU `TorchANNIndex` search, CPU component processing, sharded-index slot partitioning, sharded ANN single-shard equivalence, shard result merging, and output validation failures for out-of-range sequence IDs, non-finite similarities, and negative similarities.

Local verification status:

- The local suite proves the `single_gpu_exact` CPU path remains executable from run-root artifacts and that `multi_gpu_exact`/`multi_gpu_index_sharded_exact` planning, metadata, partitioning, and validation behavior are covered without requiring CUDA.
- The synthetic backend comparison path is covered with mocked `single_gpu_exact` and `multi_gpu_exact` builders that write `neg_ctx_equivalence_report.json`.
- WSL/.venv local CUDA comparison on `outputs/20260523-141046-7b815c34` built `single_gpu_exact` and `multi_gpu_exact` from the same reduced real-data pass-1 artifacts on the RTX 5070 Ti.
- The report was saved to `outputs/20260523-141046-7b815c34/neg_ctx_equivalence_report.json`.
- The local CUDA comparison produced matching populated-row counts (`1,327,373`) and close values (`max_abs_value_diff=5.364418029785156e-07`), but strict sequence-index equality was false. Treat this as a real local drift/tie-ordering finding to inspect before H100 exactness claims.

Deferred verification:

- Before H100 use, rerun the same merged pass-1 artifact set through `single_gpu_exact`, `multi_gpu_exact`, and, if needed, `multi_gpu_index_sharded_exact`; investigate whether the observed local sequence-index mismatch is acceptable tie ordering or a backend drift.
- Record `neg_ctx_stats.json`, `distributed/parts/neg_ctx/neg_ctx_sanity_report.json`, and `neg_ctx_equivalence_report.json` for the final target-environment run.

Verification:

```powershell
python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py -q
```

## Phase 5 - Part 4 Pass-2 Dump Tests

- [x] Test refactored candidate-profile computation against current `TopCoactivation.update_batch()` for `raw`, `freq_weighted`, and `pmi`.
- [x] Test global replay sequence list construction from `top_ctx`.
- [x] Test worker sequence partitioning covers every replay sequence exactly once.
- [x] Test pass-2 replay assignments are contiguous chunks in replay-list order and preserve all remainder sequences.
- [x] Test pass-2 worker resource construction rejects multi-device `SAEBank` placement.
- [x] Test partial candidate dump schema validation and atomic writes.
- [x] Test one-worker and two-worker dump equivalence against current single-process dump.
- [x] Verification: require worker token counts and row mappings to match single-process expectations for PMI mode.

### Phase 5 Notes

Coverage map:

- `tests/store/test_top_coactivation_modes.py` covers `TopCoactivation.update_batch()` and `compute_candidate_profile()` for `raw`, `freq_weighted`, and `pmi`, proving the refactored candidate-profile helper preserves the current dump semantics.
- `tests/pipeline/test_distributed_pass2_replay.py` covers global replay-list construction from `top_ctx`, tensor payload loading, sorting, deduplication, sentinel-zero exclusion, missing-sequence rejection, deterministic assignment updates, replay count/hash metadata, one-worker mode, contiguous remainder-preserving chunks, stale hash rejection, non-contiguous assignment rejection, and worker marker replay metadata.
- `tests/pipeline/test_distributed_pass2_partials.py` covers `candidate_dump.partial.pt` round trips, candidate dump memory estimates/guardrails, payload construction from `TopCoactivation` dump buffers, metadata construction from manifest/dump results, row alignment, missing sequence IDs, bad shapes, invalid candidate IDs, non-finite values, PMI token-count mismatch rejection, and optional preaggregation expansion over simple dump rows.
- `tests/pipeline/test_distributed_worker.py` covers pass-2 worker execution over assigned replay sequences, early dump-memory guardrail checks before model initialization, pass-2 worker dispatch, single-device `SAEBank` construction for pass-2 workers, global `top_ctx`/`latent_stats` input validation/loading, durable candidate dump writes, summary writes, and failed-marker behavior when dump validation fails.
- `tests/pipeline/test_distributed_pass2_equivalence.py` covers one-worker and two-worker distributed pass-2 dump equivalence against the single-process dump for `raw`, `freq_weighted`, and `pmi`, deterministic worker-order artifact handling, PMI worker token-count summation, and reconstructed `sequence_id -> row` mapping equivalence.

Local verification status:

- Current coverage proves the pass-2 dump semantics and worker partial artifact contract without requiring H100 hardware.
- The tests exercise synthetic/model-free pass-2 dump loops and mocked worker resources rather than a full model+SAE H100 replay. H100 throughput and GPU dump-buffer performance remain Part 9 benchmark concerns.

Verification:

```powershell
python -m pytest tests/store/test_top_coactivation_modes.py tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_worker.py -q
```

## Phase 6 - Part 5 Reducer Tests

- [x] Test simple exact candidate-dump concatenation and global `sid_to_row` construction.
- [x] Test simple exact distributed reduce against current single-process reduce.
- [x] Test target-sharded simple reduce and shard stitching.
- [x] Test PMI postprocess equivalence after distributed reduce.
- [x] Test MapReduce partial-sum reduce against simple exact reduce.
- [x] Test target-only reducer sharding preserves cross-range candidate IDs.
- [x] Test target-range reducer partitioning distributes remainder targets deterministically and never drops target IDs.
- [x] Test sorted COO partial-sum shard round trips and merge ordering by `(target_id, candidate_id)`.
- [x] Test CPU/OpenMP MapReduce reducers before any GPU reducer path is considered.
- [x] Test tie cases and cases where local top-K merging would be wrong.
- [x] Verification: run native reducer tests after rebuilding extensions whenever reducer APIs change.

### Phase 6 Notes

Coverage map:

- `tests/pipeline/test_distributed_pass2_reduce.py` covers reducer input validation, global `top_ctx` CSR and `sid_to_row` construction, replay coverage checks, simple exact candidate-dump concatenation in deterministic sequence-ID order, store attachment for the existing reducer API, simple exact reducer invocation, PMI active-count loading and validation, PMI token-count mismatch rejection, finite PMI output checks, canonical `top_coactivation.pt` writing, reducer report generation, saved artifact validation, benchmark report formatting, and manifest metric extraction.
- `tests/pipeline/test_distributed_pass2_reduce.py` also covers MapReduce target-range partitioning, more-reducers-than-targets cases, target-only sharding that preserves cross-range candidates, expected worker coverage validation, duplicate `(target, candidate)` summation, deterministic top-K tie-breaking, out-of-range record rejection, sorted COO partial-sum shard persistence/checksum validation, reducer memory estimates and guardrails, chunked merge equivalence, scheduler config validation, reducer shard cleanup, stale resume rejection, final shard stitching, canonical MapReduce artifact writing, simple-exact-vs-MapReduce equivalence for `raw`, `freq_weighted`, and `pmi`, and the regression where a candidate is not local top-K on any worker but becomes global top-K after summing.
- `tests/pipeline/test_distributed_pass2_partials.py` covers preaggregation expansion from simple candidate dumps into reducer contribution records, duplicate target-entry preservation, preaggregation round trips, and invalid self-candidate rejection.
- `tests/store/test_top_coactivation_modes.py` covers store-level PMI postprocess, target-sharded reducer stitching, native control propagation, single-process legacy native fallback, target-sharded legacy rejection, invalid reducer backend rejection, file-backed target-shard write/merge, and cleanup of current partial files on failure.

Local verification status:

- Local tests prove the exact reducer math, simple exact path, target-sharded wrapper behavior, MapReduce CPU/OpenMP-compatible path, sorted COO storage, resume/stitch behavior, PMI handling, and the known local-top-K correctness trap without requiring H100 hardware.
- `parallel` MapReduce execution is intentionally not part of the exact local gate; scheduler validation rejects unsupported parallel execution until sequential shard outputs and target-machine benchmarks justify it.
- Initial WSL/.venv native rebuild with CUDA extensions enabled failed because `torch.utils.cpp_extension` detected `CUDA_HOME=/usr` and picked `/usr/bin/nvcc` from CUDA `12.0`, while PyTorch was built for CUDA `13.0`.
- The local system also has CUDA `13.0` installed at `/usr/local/cuda`; rebuilding with `CUDA_HOME=/usr/local/cuda` and `/usr/local/cuda/bin` first on `PATH` succeeded.
- Full in-place native rebuild copied `top_coactivation_reduce`, `mid_reservoir`, `latent_stats_cuda`, and `linear_relu_ext`.
- Native imports passed after importing `torch` first, which loads PyTorch shared libraries such as `libc10.so`.
- `src/native/tests/test_reduce.py` passed after the CUDA 13 rebuild (`6 passed in 15.14s`).
- CUDA Triton top-k correctness/agreement tests passed via `src/native/tests/test_topk.py`; the full-shape benchmark reported PyTorch top-k `11.891 ms`, Triton radix-select `10.842 ms`, `1.10x` faster on the local RTX 5070 Ti.

Deferred verification:

- Target-machine native rebuild and native/store coverage still belong in Part 9 before H100 benchmarking, but the local WSL CUDA 13 rebuild is no longer blocked.

Verification:

```powershell
python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q
```

## Phase 7 - Part 6 Discovery Tests

- [x] Test centralized candidate selection over merged global artifacts.
- [x] Test seed/task partitioning for one-worker, many-worker, and more-workers-than-seeds cases.
- [x] Test discovery worker resource construction rejects multi-device `SAEBank` placement.
- [x] Test mocked discovery worker output without loading real model weights.
- [x] Test circuit-store merge, UUID collision handling, summary merge, and empty worker outputs.
- [x] Test seed-free method ownership so methods such as `cluster_contrast` do not run once per worker by accident.
- [x] Verification: run a small discovery/eval smoke before distributed H100 discovery.

### Phase 7 Notes

Coverage map:

- `tests/pipeline/test_candidate_selection_stage.py` covers centralized candidate selection from merged global artifacts, candidate metadata and markers, missing-input failure before selection, single-process/distributed candidate-selection consistency, manifest assignment updates, one-worker assignment order preservation, seed-free method ownership, uneven candidate splits, and more-workers-than-candidates cases.
- `tests/pipeline/test_distributed_assignments.py` covers deterministic discovery seed partitioning, candidate-level scheduling, seed-free method exclusion from seed-partitioned assignments, method-aware scheduling with synthetic costs, scheduling report generation, and resume task selection.
- `tests/pipeline/test_distributed_worker.py` covers discovery worker dispatch, assigned-candidate loading and manifest-drift rejection, worker metadata attachment, discovery worker input validation, shared discovery artifact loading, single-device discovery worker runtime construction, worker-local output writing, worker discovery stats, seed-free method ownership, and filtering that prevents duplicate `cluster_contrast` execution.
- `tests/pipeline/test_discovery_artifacts.py` covers shared discovery artifact validation/loading for synthetic fixtures, missing input reporting, incompatible shape rejection, top-coactivation mode mismatch rejection, and shared store loader use.
- `tests/pipeline/test_discovery_window_outputs.py` covers candidate provenance metadata attachment to accepted circuits and atomic `DiscoveryWindow.save_store()` round trips.
- `tests/pipeline/test_discovery_merge.py` covers circuit-store append/merge ordering, duplicate UUID rejection, empty worker stores, canonical merged output writing, mixed-method summary generation, seed-free and failed-range reporting, one-worker synthetic merge equivalence, two-worker synthetic discovery/eval smoke merge, and merged output validation failures for bad summaries or missing metadata.

Local verification status:

- Current tests verify the distributed discovery contract without loading real model weights by using synthetic/mocked workers and circuit stores.
- The two-worker synthetic discovery/eval smoke validates merge/report behavior before any H100 discovery benchmark.
- Real gradient-heavy discovery throughput, load imbalance, and H100 VRAM behavior remain Part 9 benchmark concerns.

Verification:

```powershell
python -m pytest tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_discovery_artifacts.py tests/pipeline/test_discovery_window_outputs.py tests/pipeline/test_discovery_merge.py -q
```

## Phase 8 - Part 7 Mode And UX Tests

- [x] Test config validation for all operating modes.
- [x] Test `distributed` is the accepted orchestration config namespace.
- [x] Test strict Pydantic validation rejects unknown or misspelled distributed config keys.
- [x] Test distributed mode config keeps search-cache generation offline/deferred by default.
- [x] Test preflight command/report output for local one-worker and synthetic 8-worker modes.
- [x] Test command parser/help output for distributed entrypoints.
- [x] Test local one-worker and synthetic 8-worker dry runs.
- [x] Test output policy: unvalidated distributed partials stay under `outputs/<run_id>/distributed/`, and canonical run artifacts appear at the top of `outputs/<run_id>/` only after checks pass.
- [x] Test JSONL run/worker metrics schema and report paths.
- [x] Test mocked device observability sampler output for GPU utilization, VRAM, power, temperature, CPU RAM, disk usage, phase label, worker PID, and physical GPU identity.
- [x] Test rollout gates reject unsafe mode transitions.
- [x] Test run reports show exactness status, part statuses, artifact paths, benchmark results, and warnings.
- [ ] Verification: ensure `python src/main.py` remains the `single_process` entrypoint while writing canonical artifacts under `outputs/<run_id>/`.

### Phase 8 Notes

Coverage map:

- `tests/pipeline/test_distributed_config.py` covers default `single_process` config behavior, the `distributed` config namespace, one-worker distributed config, H100 simple exact config, MapReduce exact config, explicit search-cache override, deferred distributed search-cache defaults, observability sample interval config, experimental fast-mode config requirements, invalid combinations, and strict unknown-key/schema-version rejection.
- `tests/pipeline/test_distributed_operating_modes.py` covers the documented taxonomy for every run mode, `single_process` as the oracle with `python src/main.py` as the documented entrypoint, `distributed_simple_exact` required parts and dry-run contract, MapReduce exact dependencies, run-mode ordering, canonical run-root policy helpers, and generated run ID shape.
- `tests/pipeline/test_distributed_controller.py` covers local one-worker dry-runs, one-CUDA local planning, RTX-style compatibility reports, synthetic 8-worker H100 dry-runs, H100 MapReduce entry-criterion reporting, pass-1/pass-2/discovery worker command generation, CLI help/parser output, subprocess launch planning, preflight shard/disk/device/config/native-extension checks, distributed config loading, and sampling-seed recording.
- `tests/pipeline/test_distributed_layout.py` covers canonical run-root layout, distributed internals under `outputs/<run_id>/distributed/`, worker metrics paths, and cleanup scoping to distributed partials rather than canonical outputs.
- `tests/pipeline/test_distributed_interfaces.py` covers run-root output path resolution for canonical artifact names and single-process-compatible save-point targeting under `outputs/<run_id>/`.
- `tests/pipeline/test_distributed_rollout_gates.py` covers missing/stale manifest rejection, required sanity/equivalence/benchmark report gates, default-recommendation benchmark requirements, gate-report writing, and raising on failures.
- `tests/pipeline/test_distributed_reporting.py` covers mode summaries, exactness statuses for every mode, hardware context, final run reports linking parts/artifacts/equivalence/benchmark reports, stable JSON report writes, metrics JSONL append, observability JSONL append, and invalid observability value rejection.
- `tests/pipeline/test_distributed_experimental_modes.py` covers experimental fast-mode acknowledgement, exact baseline root requirements, quality-toggle reporting, warning banners, and clearly marked experimental/fast output roots.

Local verification status:

- Phase 8 verifies the mode/config/UX/reporting contracts and model-free controller behavior locally.
- The output policy is covered at the contract and stage-writer level: distributed internals live under `outputs/<run_id>/distributed/`, and pass-1/pass-2 validated writers place canonical artifacts at the run root after validation.
- Full end-to-end verification that the default `python src/main.py` path both remains the `single_process` entrypoint and writes canonical artifacts under a generated `outputs/<run_id>/` is still open. Keep it as a Phase 9 synthetic/end-to-end item rather than marking it complete from interface-level tests alone.

Verification:

```powershell
python -m pytest tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_operating_modes.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_rollout_gates.py tests/pipeline/test_distributed_reporting.py tests/pipeline/test_distributed_experimental_modes.py -q
```

## Phase 9 - End-To-End Synthetic Equivalence

- [x] Build a tiny synthetic dataset and mocked/lightweight model/SAE fixture where expected artifacts can be compared cheaply.
- [x] Run current single-process pipeline or equivalent staged fixture.
- [x] Confirm the single-process fixture writes canonical outputs under `outputs/<run_id>/`.
- [x] Run one-worker `distributed_simple_exact` and compare canonical artifacts.
- [x] Run two-worker `distributed_simple_exact` and compare canonical artifacts.
- [x] Once implemented, run `distributed_mapreduce_exact` and compare to `distributed_simple_exact`.
- [x] Verification: require exact tensor equality where deterministic, and documented tolerances where floating-point order differs.

### Phase 9 Notes

Implemented support:

- Added `src/pipeline/distributed/equivalence.py` with a reusable canonical run-root comparison helper:
  - `DEFAULT_CANONICAL_ARTIFACTS`,
  - `compare_artifact()`,
  - `compare_run_roots()`,
  - `save_equivalence_report()`.
- The helper compares torch and JSON artifacts recursively, reports per-artifact pass/fail status, supports exact equality by default, and supports explicit `atol`/`rtol` tolerances for floating-point differences.
- The helper emits rollout-gate-compatible fields: top-level `ok`, `status`, and `equivalence.passed`.

Synthetic coverage:

- Added `tests/pipeline/test_distributed_end_to_end_synthetic.py`.
- The test builds complete synthetic canonical run roots under `outputs/<run_id>/` for:
  - a `single_process`-style oracle,
  - a one-worker distributed output,
  - a two-worker distributed output,
  - a MapReduce-style output with a tiny floating-point offset covered by an explicit tolerance.
- It compares canonical artifacts:
  - `latent_stats.pt`,
  - `top_ctx.pt`,
  - `mid_ctx.pt`,
  - `neg_ctx.pt`,
  - `logit_ctx.pt`,
  - `top_coactivation.pt`,
  - `candidates.pt`,
  - `circuits/summary.json`.
- It writes a sample `equivalence_tiny_synthetic.json` report under `distributed/reports/`.
- It includes a negative case proving candidate drift is reported as `different`.

Important caveat:

- This phase now verifies the staged synthetic equivalence/reporting contract, not a real model+SAE end-to-end execution. It proves that complete canonical artifact sets can be compared consistently across `single_process`, one-worker, two-worker, and MapReduce-style run roots.
- A live pipeline smoke that executes `python src/main.py` and distributed worker stages from model/data inputs remains Phase 10/11 work, depending on available local artifacts.

Verification:

```powershell
python -m pytest tests/pipeline/test_distributed_end_to_end_synthetic.py tests/pipeline/test_distributed_reporting.py tests/pipeline/test_distributed_rollout_gates.py -q
```

## Phase 10 - Local Real-Data Smoke

- [x] Run a local one-worker distributed dry run on RTX 5070 Ti style settings.
- [x] Run a small local distributed smoke with reduced `n_shards`, `n_seeds`, and efficient memory settings.
- [x] Compare artifact shapes and sanity stats against the current local `single_process` run.
- [x] Confirm local runs do not require H100-specific configs or multi-GPU hardware.
- [x] Record wall time and peak memory, but treat correctness as the main local goal.
- [x] Verification: save local smoke logs and summary reports under a clearly labeled run directory.

### Phase 10 Notes

Dry-run command executed locally:

```powershell
$env:PYTHONPATH = "src"
python -m pipeline.distributed.controller --config config_examples/local-distributed-smoke.yaml --use-cpu --dry-run
```

Dry-run result:

- Completed successfully with run ID `20260523-132415-7b815c34`.
- Planned `distributed_simple_exact` in one-worker CPU mode.
- Wrote manifest at `outputs/20260523-132415-7b815c34/distributed/manifest.json`.
- Reported `local compatibility` with `h100_required: false`.
- Used `hardware.memory: efficient`, `keep_model_loaded_for_neg_ctx: false`, and deferred search-cache generation.
- Planned four local shards with `32768` total sequences assigned to `worker_000`.
- Emitted a manual worker command with `CUDA_VISIBLE_DEVICES=` and `--phase pass1 --worker-id 0`.

WSL/.venv CUDA smoke result:

- Confirmed the WSL environment has PyTorch CUDA available with one local `NVIDIA GeForce RTX 5070 Ti`, plus `models/TuringLLM/model_1722550239_03986.pt`, `models/TuringLLM/SAE`, and `data`.
- Fixed local CUDA manifest planning by normalizing CUDA metadata (`uuid`, `pci_bus_id`) to strings in `src/pipeline/distributed/devices.py`.
- Rebuilt all native extensions locally after forcing the CUDA 13 toolchain with `CUDA_HOME=/usr/local/cuda` and `/usr/local/cuda/bin` first on `PATH`.
- Planned and ran one-worker `distributed_simple_exact` under run root `outputs/20260523-141046-7b815c34`.
- Pass 1 processed `64` batches / `32,768` sequences on CUDA and filled `32,768 / 32,768` sequence representations.
- Pass-1 merge wrote `latent_stats.pt`, `top_ctx.pt`, `mid_ctx.pt`, `seq_repr.pt`, `logit_ctx.pt`, `seq_latent_index/`, and `distributed/reports/pass1_sanity_report.json`. Merge elapsed time was about `146.08s`; peak traced CPU memory was `56,758,807` bytes on the rerun.
- Negative context wrote `neg_ctx.pt`, `neg_ctx_stats.json`, and `distributed/parts/neg_ctx/neg_ctx_sanity_report.json`; it populated `1,327,373` rows with no invalid sequence IDs or non-finite values, and reported `10.1s` total build time.
- Pass 2 dumped `32,768` replay sequences into `candidate_dump.partial.pt`; dump wall time was `163.26s`.
- Simple exact pass-2 reduce wrote `top_coactivation.pt`; reducer report showed shape `[36, 40960, 64]`, finite output, `94,371,835` nonzero values, `19.84s` total reducer-stage time, and output artifact size `754,976,773` bytes.
- Candidate selection wrote `candidates.pt` and assigned `16` discovery tasks back into the manifest.
- Discovery completed on one worker and discovery merge wrote `circuits/summary.json` plus `distributed/reports/discovery_merge_report.json`; merged circuit count was `8` and validation was `ok`.
- Local `single_process` baseline completed in WSL from the same root config. The comparison report was saved to `outputs/20260523-141046-7b815c34/distributed/reports/single_vs_distributed_equivalence.json`.
- The oracle comparison status is `different`, not fully equivalent. Known differences include metadata keys, `mid_ctx` merge mode/content, negative-context sequence IDs, candidate choices, and circuit summaries. Treat this as local readiness plus drift evidence, not a paper-facing equivalence pass.

## Phase 11 - Reduced Real-Data Multi-Worker Benchmark

- [x] Run reduced real-data `single_process` baseline.
- [ ] Run reduced real-data one-worker distributed equivalence.
- [ ] Run reduced real-data two-worker distributed equivalence.
- [x] Defer reduced real-data eight-worker H100 smoke to Part 9 until H100 access is available.
- [x] Compare `latent_stats.pt`, `top_ctx.pt`, `mid_ctx.pt`, `neg_ctx.pt`, `top_coactivation.pt`, `candidates.pt`, and `circuits/summary.json` under the same run root.
- [ ] Verification: require artifact sanity checks and circuit-output comparison before larger H100 benchmarks.

### Phase 11 Notes

Current local readiness:

- The Phase 10 dry-run produced a valid local one-worker manifest with four reduced data shards and `32768` total sequences assigned to `worker_000`.
- The configured model path is `models/TuringLLM/model_1722550239_03986.pt`.
- The configured SAE path is `models/TuringLLM/SAE`.
- WSL/.venv has the model, SAE, data, and one visible CUDA device, so the local one-worker smoke and `single_process` baseline were run.
- The local `single_process` baseline completed successfully, but the one-worker distributed comparison report is `different`, not equivalent.
- The report path is `outputs/20260523-141046-7b815c34/distributed/reports/single_vs_distributed_equivalence.json`.
- Differences include metadata-only drift, non-equivalent `mid_ctx` because the distributed merge uses `distributed_priority_reservoir`, negative-context sequence-index differences, candidate choice differences, and circuit summary differences.

Local hardware limit:

- CPU fallback is intentionally one-worker only, and WSL currently exposes one CUDA device, so this machine cannot validate a true two-worker isolated-device run.
- A two-worker reduced real-data run needs at least two visible CUDA devices or another explicit local multi-worker device configuration.
- The eight-worker H100 smoke is not a local Phase 11 blocker anymore; it belongs to Part 9.

Remaining verification sequence:

1. Investigate the local one-worker equivalence report before treating the distributed pipeline as exact against the current `single_process` oracle.
2. Decide whether `mid_ctx` distributed priority-reservoir outputs are expected to differ from the current single-process reservoir output, or whether the local equivalence oracle needs a more specific artifact/tolerance policy.
3. Investigate negative-context sequence-index differences where values are close but strict IDs differ.
4. Run two-worker `distributed_simple_exact` only when local hardware supports two isolated workers.
5. Save follow-up equivalence reports under `outputs/<run_id>/distributed/reports/`.

Suggested comparison helper:

```powershell
python -c "from pipeline.distributed import compare_run_roots, save_equivalence_report; report=compare_run_roots('<single_process_run_root>', '<distributed_run_root>', atol=1e-6); save_equivalence_report(report, '<distributed_run_root>/distributed/reports/equivalence_reduced_real.json'); print(report['status'])"
```

## Phase 12 - Full H100 Benchmark Protocol

- [x] Defer native extension rebuild on the target H100 host to Part 9.
- [x] Defer native tests `src/native/tests/test_topk.py` and `src/native/tests/test_reduce.py` to Part 9.
- [x] Defer `single_process` or current-runtime H100 baseline to Part 9.
- [x] Defer `distributed_simple_exact` 1, 2, 4, and 8 worker H100 benchmarks to Part 9.
- [x] Defer separate Part 2 pass 1, Part 3 neg_ctx, Part 4 pass-2 dump, Part 5 reduce, and Part 6 discovery benchmarks to Part 9.
- [x] Defer wall time, GPU utilization, peak VRAM, CPU RAM, disk usage/write throughput, artifact size, worker imbalance, cleanup policy, and circuit-output recording to Part 9.
- [x] Defer JSONL metrics and device observability capture for H100 benchmarks to Part 9.
- [x] Verification: H100 distributed defaults must not be recommended from Part 8; Part 9 owns correctness and benchmark reports.

### Phase 12 Notes

Transfer status:

- Full H100 benchmark execution cannot be performed in this local workspace and is now explicitly owned by [`part-9-h100-validation-and-benchmarks.md`](part-9-h100-validation-and-benchmarks.md).
- Phase 12 is considered complete for Part 8 because the local plan no longer blocks on H100 availability.

Part 9 mapping:

- Native rebuild and native reducer tests: Part 9 Phase 3.
- H100 one-worker equivalence before multi-worker trust: Part 9 Phase 4.
- Reduced real-data H100/multi-worker equivalence: Part 9 Phase 5.
- Negative-context backend benchmarks: Part 9 Phase 6.
- Pass-2 dump benchmarks: Part 9 Phase 7.
- Pass-2 reduce benchmarks: Part 9 Phase 8.
- Full `distributed_simple_exact` 1/2/4/8-worker benchmark: Part 9 Phase 9.
- Discovery H100 benchmark: Part 9 Phase 10.

Local verification status:

- Part 8 has already completed the local/synthetic prerequisites for H100 benchmarking: mode/config gates, manifest/controller dry-runs, pass-1 merge tests, negative-context local tests, pass-2 dump/reduce tests, discovery merge tests, and synthetic canonical artifact equivalence.
- H100 performance claims, utilization, VRAM, power/temperature, disk-throughput, and benchmark-report completeness must come from Part 9 target-environment runs.

## Phase 13 - MapReduce Decision Gate

- [x] Defer H100 `distributed_simple_exact` bottleneck decision to Part 9.
- [x] Defer candidate dump size, concatenation time, reducer input memory, reducer time, and shard write/merge overhead measurements to Part 9.
- [x] Defer enabling or recommending `distributed_mapreduce_exact` based on measured H100 bottlenecks to Part 9.
- [x] Require MapReduce exact output to match simple exact output on synthetic fixtures before any H100 MapReduce benchmarking.
- [x] Defer MapReduce target-machine benchmarking until equivalence is proven and Part 9 justifies it.
- [x] Defer the final MapReduce recommendation decision note to Part 9.

### Phase 13 Notes

Local prerequisite status:

- Phase 6 local reducer tests prove simple exact and MapReduce target-range reduction equivalence on synthetic fixtures for `raw`, `freq_weighted`, and `pmi`.
- Phase 6 also covers target-only reducer sharding, cross-range candidate preservation, sorted COO partial-sum shards, deterministic tie-breaking, and the regression where a candidate is not local top-K on any worker but becomes global top-K after summing.
- Phase 9 synthetic canonical run-root comparison includes a MapReduce-style output compared against the `single_process`-style oracle with an explicit floating-point tolerance.

Transfer status:

- The actual MapReduce promotion decision requires target workload measurements and is now owned by Part 9 Phase 11.
- Part 8 should not recommend `distributed_mapreduce_exact` as an operating default. It can only say the local exactness prerequisites are in place.

Part 9 mapping:

- Measure simple-exact reducer bottlenecks: Part 9 Phase 8 and Phase 11.
- Decide whether central candidate-dump merge or reducer input memory is a real bottleneck: Part 9 Phase 11.
- Run H100 MapReduce benchmarks only after equivalence: Part 9 Phase 11.
- Write `equivalence_mapreduce_vs_simple.json` and the MapReduce decision note: Part 9 Phase 11.

## Phase 14 - Paper-Facing Reproducibility Package

- [x] Define local report/schema support for exact config identity, run manifest, git SHA field, environment override field, and command/report paths.
- [x] Define local report/schema support for physical/logical GPU identity metadata and JSONL metrics.
- [x] Define local artifact/report support for global sequence ID table, `seq_repr` cap mapping, and `mid_ctx` priority-reservoir seed/hash version.
- [x] Define local artifact/report support for `mid_ctx` candidate-pool settings, coverage/truncation report, and replay-fallback status.
- [x] Define local sanity-report and rollout-gate support for canonical outputs.
- [x] Define local equivalence-report support showing distributed artifacts match the chosen correctness oracle.
- [x] Define local benchmark-report linking support in final run reports.
- [x] Define local circuit summary and worker/candidate provenance support.
- [x] Defer the final `paper_ready` package/status decision to Part 9 target-environment runs.

### Phase 14 Notes

Local support already in place:

- `DistributedRunManifest` records config path, normalized config hash, git SHA field, environment overrides, model/SAE/dataset paths, output roots, device assignments, shard table, worker assignments, schema versions, metrics path, and run summary path.
- `tests/pipeline/test_distributed_manifest.py`, `tests/pipeline/test_distributed_layout.py`, and `tests/pipeline/test_distributed_reporting.py` cover manifest/report schemas, JSONL metrics, hardware context, run summary paths, and stable report writes.
- `tests/pipeline/test_distributed_pass1_merge.py` and `tests/pipeline/test_distributed_pass1_partials.py` cover `seq_repr` mapping determinism, `mid_ctx` priority seed/hash behavior, candidate-pool settings, truncation/fallback reporting, cleanup eligibility, and pass-1 sanity report generation.
- `tests/pipeline/test_distributed_end_to_end_synthetic.py` and `src/pipeline/distributed/equivalence.py` provide rollout-gate-compatible equivalence reports for canonical artifact comparisons.
- `tests/pipeline/test_distributed_pass2_benchmark.py`, `tests/pipeline/test_distributed_pass2_reduce.py`, and `tests/pipeline/test_distributed_reporting.py` cover benchmark report construction/linking for pass-2 dump/reduce and final run summaries.
- `tests/pipeline/test_distributed_worker.py`, `tests/pipeline/test_discovery_window_outputs.py`, and `tests/pipeline/test_discovery_merge.py` cover assigned candidate provenance, artifact hashes, worker IDs, discovery method metadata, worker task metrics, circuit summaries, merged reports, and validation of required circuit metadata.
- `tests/pipeline/test_distributed_rollout_gates.py` covers paper-facing gates that require tiny synthetic and reduced real-data equivalence before exact distributed outputs can be trusted for paper use.
- `tests/pipeline/test_distributed_experimental_modes.py` covers experimental fast-mode warnings and prevents approximate quality toggles from silently looking paper-eligible.

Transfer status:

- Part 8 defines and tests the local schema/reporting/provenance building blocks.
- Actual paper-facing packages require real exact runs, complete equivalence reports, benchmark reports, environment/native build details, and reviewed circuit outputs. That execution is owned by Part 9 Phase 12.
- Do not mark any real run `paper_ready` from Part 8 alone. Part 8 can only say the reporting and validation scaffolding exists.

Part 9 mapping:

- Save exact config, manifest, git SHA, native build info, environment summary, and command history: Part 9 Phase 12.
- Save physical/logical GPU identity, JSONL metrics, and device observability samples: Part 9 Phase 12.
- Save artifact sanity, equivalence, benchmark, and circuit-provenance reports from real runs: Part 9 Phase 12.
- Define and apply final `paper_ready` status for exact, fully validated runs: Part 9 Phase 12.

## Phase 15 - Regression And CI Strategy

- [x] Keep most distributed tests CPU/synthetic so they can run in normal CI or local development.
- [x] Mark CUDA/H100 tests explicitly and keep them opt-in.
- [x] Add focused test commands for each part to avoid always running the full suite.
- [x] Add a reduced smoke command for pre-H100 validation.
- [x] Track full-suite checkpoints after major distributed changes.
- [x] Verification: document which tests are required before local merge, before H100 run, and before paper-facing use.

### Phase 15 Notes

Local/CI-safe test policy:

- Most distributed tests are intentionally CPU/synthetic and should remain runnable without model weights, SAE weights, CUDA, or H100 access.
- Tests that require real model/SAE artifacts, CUDA devices, native rebuilds, or H100-scale timings should be opt-in and documented as target-environment gates rather than normal local CI.
- Synthetic tests should validate exact math, schema contracts, run-root layout, deterministic assignments, resume/cleanup semantics, equivalence report shape, and artifact comparison behavior.
- Real-data and H100 tests should validate execution, throughput, memory, observability, and paper-facing reproducibility.

Focused local commands by phase:

```powershell
$env:PYTHONPATH = "src"

# Phase 1 - taxonomy, gates, reports
python -m pytest tests/pipeline/test_distributed_rollout_gates.py tests/pipeline/test_distributed_reporting.py tests/pipeline/test_distributed_operating_modes.py tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_experimental_modes.py -q

# Phase 2 - manifest/runtime/controller contracts
python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_resume_policy.py -q

# Phase 3 - pass-1 partials and merges
python -m pytest tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/store/test_mid_ctx_modes.py -q

# Phase 4 - negative context local/backend contracts
python -m pytest tests/pipeline/test_negative_context_stage.py tests/store/test_neg_context_backend.py -q

# Phase 5 - pass-2 dump contracts
python -m pytest tests/store/test_top_coactivation_modes.py tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_worker.py -q

# Phase 6 - pass-2 reduce contracts
python -m pytest tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_partials.py tests/store/test_top_coactivation_modes.py -q

# Phase 7 - candidate selection and discovery contracts
python -m pytest tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_discovery_artifacts.py tests/pipeline/test_discovery_window_outputs.py tests/pipeline/test_discovery_merge.py -q

# Phase 8 - operating modes and UX
python -m pytest tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_operating_modes.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_rollout_gates.py tests/pipeline/test_distributed_reporting.py tests/pipeline/test_distributed_experimental_modes.py -q

# Phase 9 - synthetic canonical artifact equivalence
python -m pytest tests/pipeline/test_distributed_end_to_end_synthetic.py tests/pipeline/test_distributed_reporting.py tests/pipeline/test_distributed_rollout_gates.py -q
```

Recommended local merge gate:

```powershell
$env:PYTHONPATH = "src"
$distributedTests = (Get-ChildItem tests/pipeline/test_distributed_*.py).FullName
python -m pytest $distributedTests tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_negative_context_stage.py tests/pipeline/test_discovery_artifacts.py tests/pipeline/test_discovery_window_outputs.py tests/pipeline/test_discovery_merge.py tests/store/test_mid_ctx_modes.py tests/store/test_neg_context_backend.py tests/store/test_top_coactivation_modes.py -q
```

Reduced pre-H100 smoke command:

```powershell
$env:PYTHONPATH = "src"
python -m pipeline.distributed.controller --config config_examples/local-distributed-smoke.yaml --use-cpu --dry-run
```

Run this before H100 access to confirm local config validity, run-root layout, shard table construction, deferred search-cache settings, and one-worker command generation. A real worker launch remains gated on model/SAE artifacts being present.

Opt-in target-environment gates:

- CUDA/local multi-worker smoke: requires visible CUDA devices and model/SAE artifacts.
- Native reducer rebuild/tests: required before target-machine reducer benchmarks or H100 runs.
- H100 1/2/4/8 worker benchmarks: Part 9 only.
- Paper-facing reproducibility package: Part 9 only, after exact equivalence and benchmark reports are complete.

Checkpoint policy:

- After major distributed changes, run the recommended local merge gate and record the result in this file or the related planning notes.
- Before H100 access, also run the reduced pre-H100 dry-run command and confirm the generated manifest/report paths.
- Before paper-facing use, require Part 9 equivalence, benchmark, observability, and reproducibility reports.

Verification:

```powershell
$env:PYTHONPATH = "src"
$distributedTests = (Get-ChildItem tests/pipeline/test_distributed_*.py).FullName
python -m pytest $distributedTests tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_negative_context_stage.py tests/pipeline/test_discovery_artifacts.py tests/pipeline/test_discovery_window_outputs.py tests/pipeline/test_discovery_merge.py tests/store/test_mid_ctx_modes.py tests/store/test_neg_context_backend.py tests/store/test_top_coactivation_modes.py -q
```

Result: `379 passed in 7.21s`.

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
