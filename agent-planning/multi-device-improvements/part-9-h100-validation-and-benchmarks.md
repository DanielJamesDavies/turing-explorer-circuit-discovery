# Plan: Part 9 - H100 Validation And Benchmarks

> **Goal:** Collect the H100-specific validation, benchmark, rollout, and paper-readiness work from Parts 1-8 so Part 8 can focus on local/synthetic/reduced-data gates that do not require H100 access.
>
> **Created:** 2026-05-23

---

## Scope

This part is intentionally target-environment work. It should run only after the local correctness gates from Part 8 have passed:

- unit/schema/merge tests pass locally,
- one-worker `distributed_simple_exact` is compared against `single_process`,
- reduced real-data smoke runs pass where local hardware allows,
- native reducer APIs are known to match the current code,
- rollout gate reports identify the run as ready for H100 validation.

It covers:

- H100 dry-run inspection,
- native extension rebuild and tests on the H100 host,
- reduced real-data multi-worker equivalence,
- full 1/2/4/8-worker H100 benchmarks,
- negative-context backend comparisons,
- pass-2 dump and reduce benchmarks,
- discovery throughput benchmarks,
- MapReduce decision gates,
- paper-facing reproducibility artifacts.

It does not replace local Part 8 testing. H100 runs should not be used to discover basic merge, schema, or assignment bugs that can be caught locally.

---

## Phase 1 - H100 Access Preconditions

- [ ] Confirm the target machine has the expected 8x H100 CUDA device inventory.
- [ ] Confirm each physical GPU can be isolated by `CUDA_VISIBLE_DEVICES`.
- [ ] Confirm the H100 host has the required model weights, SAE weights, dataset shards, config files, and output storage.
- [ ] Confirm the repo commit, Python environment, PyTorch/CUDA versions, and native build tooling are recorded before benchmarking.
- [ ] Confirm `config_examples/h100-8x-distributed-simple-exact.yaml` still matches the intended target hardware and dataset scale.
- [ ] Confirm `persist.build_search_cache_after_pipeline: false` so search-cache generation stays off the distributed critical path.
- [ ] Verification: write an environment summary report before any expensive H100 run.

## Phase 2 - H100 Controller Dry Run

- [ ] Run the H100 dry run:

```powershell
$env:PYTHONPATH = "src"
python -m pipeline.distributed.controller --config config_examples/h100-8x-distributed-simple-exact.yaml --mode distributed_simple_exact --worker-count 8 --devices 0,1,2,3,4,5,6,7 --dry-run
```

- [ ] Inspect the manifest path, run ID, output root, worker commands, and per-worker `CUDA_VISIBLE_DEVICES` assignments.
- [ ] Confirm there is one worker per physical GPU and each worker-local logical device is `cuda:0`.
- [ ] Confirm pass-1 shard assignments are sequence-balanced and deterministic.
- [ ] Confirm pass-2 replay assignments preserve all remainder sequences and do not drop sequence IDs.
- [ ] Confirm discovery task estimates are balanced enough for the selected methods.
- [ ] Confirm the dry-run report identifies GPU phases, CPU/OpenMP phases, disk-I/O phases, and centralized merge phases.
- [ ] Verification: save the dry-run text, manifest, mode summary, H100 exact-mode report, and preflight report under the run root.

## Phase 3 - Native Extension Rebuild And Tests

- [ ] Rebuild native extensions on the target H100 host before any reducer benchmark:

```powershell
cd src/native
python setup.py build_ext --inplace
```

- [ ] Run native tests after rebuild:

```powershell
python -m pytest src/native/tests/test_topk.py src/native/tests/test_reduce.py -q
```

- [ ] Run store/reducer tests that exercise the rebuilt extension:

```powershell
python -m pytest tests/store/test_top_coactivation_modes.py -q
```

- [ ] Confirm `target_sharded` reducer mode does not fall back to a legacy native signature.
- [ ] Record native build info, compiler details, OpenMP settings, and reducer API compatibility in the benchmark report.

## Phase 4 - One-Worker H100 Equivalence Gate

- [ ] Run `distributed_simple_exact` with one H100 worker before any multi-worker run is trusted.
- [ ] Compare canonical artifacts against a `single_process` oracle for the same config or the same reduced dataset slice.
- [ ] Compare at minimum:
  - `latent_stats.pt`,
  - `top_ctx.pt`,
  - `mid_ctx.pt`,
  - `seq_repr.pt`,
  - `logit_ctx.pt`,
  - `neg_ctx.pt`,
  - `top_coactivation.pt`,
  - `candidates.pt`,
  - `circuits/summary.json`.
- [ ] Require exact tensor equality where deterministic and explicitly documented tolerances where floating-point order can differ.
- [ ] Write `distributed/reports/equivalence_one_worker.json`.
- [ ] Verification: rollout gates must reject multi-worker `distributed_simple_exact` if this report is missing or failing.

## Phase 5 - Reduced Real-Data Multi-Worker Equivalence

- [ ] Run a reduced real-data `single_process` baseline.
- [ ] Run reduced real-data one-worker distributed equivalence.
- [ ] Run reduced real-data two-worker distributed equivalence.
- [ ] If H100 access is available and cost is acceptable, run a reduced real-data eight-worker smoke.
- [ ] Compare canonical artifacts under their run roots using the same artifact list as Phase 4.
- [ ] Compare circuit summaries and circuit provenance enough to catch seed assignment, discovery method, or merge drift.
- [ ] Write `distributed/reports/equivalence_reduced_real.json`.
- [ ] Verification: do not run full-scale H100 benchmarks until reduced real-data sanity checks and artifact comparisons pass.

## Phase 6 - Negative-Context H100 Backend Benchmarks

- [ ] Benchmark `single_gpu_exact`, `multi_gpu_exact`, and `multi_gpu_index_sharded_exact` where applicable.
- [ ] Use the same merged pass-1 artifacts for every backend comparison.
- [ ] Record `neg_ctx_stats.json`, `distributed/parts/neg_ctx/neg_ctx_sanity_report.json`, and `neg_ctx_equivalence_report.json`.
- [ ] Compare artifact shape, dtype, populated rows, fill-rate distribution, exact sequence-ID equality, near-exact similarity equality, and sampled rows.
- [ ] Record selected devices, component assignments, per-device timings, estimated ANN memory, guardrail limits, and actual peak VRAM where available.
- [ ] Decide whether replicated `multi_gpu_exact` is acceptable with the configured `max_repr_seqs`, or whether index-sharded mode is needed.
- [ ] Verification: preserve final `neg_ctx.pt` semantics regardless of backend, so downstream stages do not branch on the backend.

## Phase 7 - Pass-2 Dump H100 Benchmarks

- [ ] Benchmark pass-2 dump with 1, 2, 4, and 8 workers where practical.
- [ ] Use H100 worker mode with `latents.top_coactivation.dump_device: "gpu"` unless testing a local/CPU fallback.
- [ ] Run worker commands from a manifest planned with the matching worker count:

```powershell
python -m pipeline.distributed.worker --manifest <run>/distributed/manifest.json --worker-id 0 --phase pass2
```

- [ ] Run the same command concurrently for all worker IDs in the planned run.
- [ ] Record per-worker replay sequence counts, batch counts, wall time, model-forward time, SAE-encode time, candidate materialization time, save time, artifact size, assignment imbalance, and peak VRAM.
- [ ] Generate `distributed/reports/pass2_benchmark_report.json` after worker completion.
- [ ] Verification: worker token counts and `sequence_id -> row` mapping must still match single-process expectations, especially in PMI mode.

## Phase 8 - Pass-2 Reduce H100 Benchmarks

- [ ] Benchmark simple exact reduce first with `reduce_backend=single_process` and `reduce_shards=1`.
- [ ] Benchmark target-sharded simple reduce with representative `reduce_shards`, for example `2`, `4`, and `8`.
- [ ] Record candidate dump size, concatenation/build time, reducer input memory, reducer time, PMI time, save time, output artifact size, and peak CPU RAM.
- [ ] Run reduced real-data comparison against current single-process pass 2 using the same config hash.
- [ ] Compare exact `top_indices` and close `top_values`.
- [ ] Write `distributed/reports/pass2_reduce_report.json` for every reducer benchmark.
- [ ] Verification: do not benchmark MapReduce as a recommended path until simple exact reduce has passed correctness and produced bottleneck evidence.

## Phase 9 - Full H100 Distributed Simple Exact Benchmark

- [ ] Run the current-runtime or `single_process` H100 baseline first.
- [ ] Run `distributed_simple_exact` with 1, 2, 4, and 8 workers where practical.
- [ ] Benchmark Part 2 pass 1, Part 3 neg_ctx, Part 4 pass-2 dump, Part 5 reduce, Part 6 candidate selection, and Part 6 discovery separately.
- [ ] Record total wall time and per-part wall time.
- [ ] Record GPU utilization, peak VRAM, CPU RAM, disk usage, disk write throughput, artifact sizes, worker imbalance, cleanup policy effects, and circuit outputs.
- [ ] Record JSONL controller/worker metrics and device observability samples for every run.
- [ ] Write `distributed/reports/benchmark_report.json`.
- [ ] Verification: do not recommend H100 distributed defaults until correctness, sanity, and benchmark reports are complete.

## Phase 10 - Discovery H100 Benchmark

- [ ] Run distributed discovery only after candidate selection, worker output schema, circuit-store merge, resume behavior, and seed-free method ownership are validated.
- [ ] Run one discovery worker per H100 with replicated model+SAE resources.
- [ ] Record per-task duration, forward-pass count, accepted circuit count, discovery method, seed-free method ownership, failed task ranges, and peak VRAM where available.
- [ ] Compare merged circuit count against the sum of worker circuit counts plus any designated seed-free method outputs.
- [ ] Validate every accepted circuit has seed metadata, discovery method metadata, eval metadata where expected, and worker/run provenance.
- [ ] Write `distributed/reports/discovery_merge_report.json` and include discovery timing in the final benchmark report.
- [ ] Verification: H100 discovery results are not paper-facing until merged summaries and circuit provenance pass validation.

## Phase 11 - MapReduce Decision Gate

- [ ] Use `distributed_simple_exact` H100 benchmarks to determine whether central candidate-dump merge or reducer input memory is a real bottleneck.
- [ ] Measure candidate dump size, concatenation time, reducer input memory, reducer time, shard write/merge overhead, CPU RAM, and disk I/O.
- [ ] Enable `distributed_mapreduce_exact` only if simple exact reduce is too slow or memory-heavy on the target workload.
- [ ] Require MapReduce exact output to match simple exact output on synthetic and reduced real-data fixtures before H100 MapReduce benchmarking.
- [ ] Benchmark MapReduce target-range reducers only after equivalence is proven.
- [ ] Write `distributed/reports/equivalence_mapreduce_vs_simple.json`.
- [ ] Write a MapReduce decision note explaining whether to keep `distributed_simple_exact` as the recommended H100 mode or promote `distributed_mapreduce_exact`.
- [ ] Verification: rollout gates must reject `distributed_mapreduce_exact` if equivalence or decision reports are missing.

## Phase 12 - Paper-Facing H100 Reproducibility Package

- [ ] Save exact config, manifest, git SHA, native build info, environment summary, and command history.
- [ ] Save physical/logical GPU identity metadata, including UUID/name/PCI bus ID when available.
- [ ] Save JSONL metrics and device observability samples.
- [ ] Save global sequence ID table, `seq_repr` cap mapping, and `mid_ctx` priority-reservoir seed/hash version.
- [ ] Save `mid_ctx` candidate-pool settings, coverage/truncation report, and replay-fallback status.
- [ ] Save artifact sanity reports for every canonical output.
- [ ] Save equivalence reports showing distributed artifacts match the chosen correctness oracle.
- [ ] Save benchmark reports with hardware description and per-part timings.
- [ ] Save circuit summaries and enough provenance to trace each circuit back to seed, worker, method, and config.
- [ ] Define a `paper_ready` report status that requires exact mode, passing sanity checks, passing equivalence, benchmark reports, and complete provenance.
- [ ] Verification: experimental fast outputs, approximate quality toggles, or missing equivalence reports must mark the run as not paper-ready.

## Phase 13 - Cleanup And Retention After H100 Runs

- [ ] Use `keep_all` for validation, equivalence, failed runs, and early H100 profiling.
- [ ] Use `delete_large_partials_on_success` only after exactness, observability, and final reports are trusted.
- [ ] Use `delete_all_partials_on_success` only for mature full-size runs where partials can be regenerated from the manifest.
- [ ] Use `manual_cleanup_only` for paper-facing runs until artifacts have been reviewed and archived.
- [ ] Preserve failed-run partials, logs, metrics, and failure markers by default.
- [ ] Verification: cleanup plans must not delete canonical artifacts, reports, manifests, or unrelated run outputs.

---

## H100 Rollout Order

1. Run the H100 controller dry run and inspect manifest, worker commands, output root, and exact-mode report.
2. Rebuild native extensions and run native/store reducer tests on the target host.
3. Run one-worker `distributed_simple_exact` and compare against `single_process`.
4. Run reduced real-data one-worker and two-worker equivalence.
5. Run reduced real-data eight-worker smoke if practical.
6. Run full `distributed_simple_exact` benchmarks with 1, 2, 4, and 8 workers.
7. Benchmark negative context, pass-2 dump, pass-2 reduce, and discovery separately.
8. Decide whether MapReduce is needed based on measured simple exact bottlenecks.
9. Run MapReduce equivalence and benchmarks only if the decision gate justifies it.
10. Build the paper-facing reproducibility package only from exact, equivalence-gated runs.

---

## Open Questions

- What reduced real-data size is enough before trusting a full H100 run?
- What floating-point tolerances are acceptable for H100 comparisons where operation order differs?
- Should H100 benchmark reports be JSON only, Markdown summaries, or both?
- What worker-count sequence is worth running if H100 allocation time is limited: `1, 2, 4, 8` or only `1, 8`?
- Is replicated-index `multi_gpu_exact` memory acceptable at the target `max_repr_seqs`, or should H100 runs move directly to `multi_gpu_index_sharded_exact` after equivalence?
- What threshold for candidate-dump size, reducer time, or reducer memory should trigger MapReduce promotion?
- Should paper-facing runs require `distributed_simple_exact`, or can `distributed_mapreduce_exact` be accepted once equivalence is proven?

## Risks / Assumptions

- H100 speedups can hide semantic drift if artifact comparisons are incomplete.
- Replicated model+SAE workers assume enough H100 VRAM for one full worker per GPU.
- Native reducer behavior must be validated on the target host, not only on local synthetic tests.
- MapReduce adds shuffle, resume, and debugging complexity; it should not be promoted without measured simple-exact bottlenecks.
- GPU worker time alone is not enough; merge/reduce CPU time, disk I/O, and artifact size must be included in benchmark claims.
- Paper-facing outputs must not mix exact distributed runs with experimental fast-mode artifacts.
