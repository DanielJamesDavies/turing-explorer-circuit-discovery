# Plan: Multi-Device Pipeline Improvements

> **Goal:** Turn an 8x H100 node into true multi-worker throughput by replicating model+SAEs per GPU, splitting forward-heavy work by data/sequence/seed, and merging compact exact artifacts at part barriers.
>
> **Created:** 2026-05-15

---

## Core Direction

The current H100 config is single-process and only partially multi-device: model work is primary-GPU centered, SAE placement uses the first two GPUs, and all eight GPUs are mainly used for negative-context exact ANN.

The desired architecture is one worker process per GPU:

```text
worker 0 / cuda:0: full model + full SAE bank + assigned shards/sequences/seeds
worker 1 / cuda:1: full model + full SAE bank + assigned shards/sequences/seeds
...
worker 7 / cuda:7: full model + full SAE bank + assigned shards/sequences/seeds

controller:
  writes run manifest
  assigns work
  validates partial artifacts
  merges exact part outputs
```

Centralize only compact summaries at part boundaries. Do not stream raw activations or every batch's dense latent state to one device.

Each run gets one canonical run ID and output root:

```text
outputs/<run_id>/
  latent_stats.pt
  top_ctx.pt
  mid_ctx.pt
  seq_repr.pt
  logit_ctx.pt
  top_coactivation.pt
  circuits/
  distributed/
    manifest.json
    workers/
      worker_000/
      worker_001/
    parts/
    reports/
      run_metrics.jsonl
      run_summary.json
```

`run_id` is per whole run, not per device. Workers get `worker_id`s inside that run.
Distributed partials stay under `outputs/<run_id>/distributed/`; validated canonical artifacts for that run live at the top of `outputs/<run_id>/`.
This run-root policy applies to both `single_process` and distributed modes. Legacy top-level `outputs/*.pt` paths may be kept only as compatibility aliases, not as the source of truth.
When the user does not provide a run ID, generate one as `YYYYMMDD-HHMMSS-<config_hash_8>`.

---

## Design Principles

- [ ] Keep the current single-process pipeline as the correctness oracle.
- [ ] Use `outputs/<run_id>/` for every run mode, including `single_process`; do not write new canonical artifacts directly under top-level `outputs/`.
- [ ] Replicate model+SAEs per worker rather than model-parallelizing TuringLLM.
- [ ] Split expensive forward passes across GPUs: pass 1 by dataset shard, pass 2 by replay sequence, discovery by seed.
- [ ] Merge compact artifacts, not raw activations.
- [ ] Prefer exact merges first; add approximate fast paths only after exact equivalence is proven.
- [ ] Preserve canonical output names and final tensor shapes wherever possible.
- [ ] Make all worker assignments deterministic from a manifest.
- [ ] Balance pass-1 whole-shard assignment by actual sequence count, not shard count.
- [ ] Split sequence lists and target ranges with deterministic remainder handling; never drop remainder items.
- [ ] Keep single-GPU/local-PC runs fully supported; distributed execution is an optional runtime mode, not a replacement for `src/main.py`.
- [ ] Build a full-dataset global sequence ID table before any worker assignment; workers must never renumber sequences locally.
- [ ] Isolate worker devices so each distributed worker sees one logical CUDA device and loads a full local model+SAE bank on that device.
- [ ] Record physical GPU identity and JSONL metrics so H100 utilization and failures are debuggable.
- [ ] Require solid unit tests for every mathematical merge or sampling rule before H100 execution.

---

## Local And Small-Compute Compatibility

This plan must preserve the current local development path, including machines such as a single RTX 5070 Ti 16GB.

Local/small-compute mode should continue to use:

- [ ] `single_process` or a one-worker controller path.
- [ ] `hardware.memory: "efficient"` to keep VRAM pressure low.
- [ ] `hardware.keep_model_loaded_for_neg_ctx: false` unless local profiling proves it fits comfortably.
- [ ] Reduced `data.n_shards`, `discovery.n_seeds`, and `discovery.probe_batch_size`.
- [ ] Deferred/offline search-cache generation.

The distributed runtime should wrap the same artifact contracts rather than fork the algorithm. A one-worker distributed run should produce the same canonical artifacts as the current single-process path, while 8x H100 runs use replicated workers and part-barrier merges for throughput.

Do not make replicated model+SAE workers mandatory. That strategy is intended for high-VRAM GPUs; on 16GB local hardware, the current efficient/offload path remains the expected operating mode. Local runs should still use a run ID and write canonical artifacts under `outputs/<run_id>/`.

---

## Operating Mode Workflow

The implemented mode surface is documented in detail in [`part-7-operating-modes.md`](part-7-operating-modes.md). Use these modes deliberately:

- `single_process`: the default local pipeline and correctness oracle. Use this for normal RTX 5070 Ti work and for exact baselines.
- `distributed_simple_exact`: the first H100 target. It uses one isolated worker per GPU, exact pass-barrier merges, and the central exact pass-2 candidate-dump reducer.
- `distributed_mapreduce_exact`: the scalable exact pass-2 path. Use only after `distributed_simple_exact` has passed equivalence and the central reducer is the measured bottleneck.
- `distributed_experimental_fast`: exploratory only. It requires explicit acknowledgement, an exact baseline root, quality toggles, and an output root clearly marked experimental/fast.

Mode-specific config examples:

- [`config_examples/local-one-worker-distributed.yaml`](../../config_examples/local-one-worker-distributed.yaml): local one-worker validation of distributed contracts with efficient memory settings and search-cache generation disabled.
- [`config_examples/h100-8x-distributed-simple-exact.yaml`](../../config_examples/h100-8x-distributed-simple-exact.yaml): future 8x H100 `distributed_simple_exact` run with one worker per physical GPU.

Controller dry-run and worker commands:

```powershell
$env:PYTHONPATH = "src"
python -m pipeline.distributed.controller --config config_examples/local-one-worker-distributed.yaml --dry-run
python -m pipeline.distributed.controller --config config_examples/h100-8x-distributed-simple-exact.yaml --mode distributed_simple_exact --worker-count 8 --devices 0,1,2,3,4,5,6,7 --dry-run
python -m pipeline.distributed.worker --manifest outputs/<run_id>/distributed/manifest.json --worker-id 0 --phase pass1
```

Standalone merge/reduce entrypoints:

```powershell
$env:PYTHONPATH = "src"
python -m pipeline.distributed.pass1_merge --manifest outputs/<run_id>/distributed/manifest.json
python -m pipeline.distributed.pass2_reduce --output-root outputs/<run_id> --candidate-dump outputs/<run_id>/distributed/workers/worker_000/pass2/candidate_dump.partial.pt
```

For H100 rollout, use this order:

1. Run a dry run and inspect the manifest, worker commands, local/H100 report, and output root.
2. Run `distributed_simple_exact` on one worker and compare against `single_process`.
3. Run reduced real-data equivalence, then an 8-worker H100 benchmark.
4. Keep `distributed_mapreduce_exact` disabled until the benchmark shows central pass-2 reduce is the bottleneck.
5. Keep `distributed_experimental_fast` out of paper-facing outputs unless it is explicitly labelled and compared against an exact baseline.

Search-cache generation should stay off the distributed critical path. Distributed configs default `persist.build_search_cache_after_pipeline` to `false`; build the search cache later from validated canonical artifacts under `outputs/<run_id>/`.

Cleanup policy guidance:

- Use `keep_all` for validation, equivalence, failed runs, and early H100 profiling.
- Use `delete_large_partials_on_success` only after the exact pipeline is trusted and final reports/metrics are sufficient for debugging.
- Use `delete_all_partials_on_success` only for mature full-size runs where partials can be regenerated from the manifest.
- Use `manual_cleanup_only` when preserving paper-facing artifacts and deleting by hand after review.

Paper eligibility:

- Paper-eligible: `single_process`, `distributed_simple_exact` after required equivalence gates, and `distributed_mapreduce_exact` after equivalence against `distributed_simple_exact`.
- Exploratory only: `distributed_experimental_fast`, approximate quality toggles, and any run with missing equivalence, benchmark, or final run reports.

---

## Part 1 — Manifest And Worker Runtime

- [ ] Add a run manifest containing config hash, model path, SAE path, dataset path, worker count, CUDA device assignment, output root, shard assignments, sequence assignments, seed assignments, and artifact schema versions.
- [ ] Add schema versions for manifests, partial artifacts, metrics JSONL, sanity reports, and run summaries.
- [ ] Add preflight checks for output writability, run ID collisions, config validity/hash, shard table construction, device availability/uniqueness, disk space, and native extensions.
- [ ] Add a canonical full-dataset sequence ID table: shard index, shard filename, sequence count, global start ID, and global end ID.
- [ ] Add worker device isolation: controller maps physical devices to workers, each worker sees one logical `cuda:0`, and `SAEBank` receives a single-device list.
- [ ] Add deterministic whole-shard partitioning for pass 1, greedily balanced by actual per-shard sequence counts.
- [ ] Add deterministic pass-2 sequence partitioning from the merged global `top_ctx` sequence list, using contiguous chunks with no dropped remainders.
- [ ] Add deterministic seed partitioning for discovery.
- [ ] Record physical/logical GPU identity, including UUID/name/PCI bus ID when available.
- [ ] Add JSONL metrics/report paths and cleanup/retention policy values.
- [ ] Add worker output directories such as `outputs/<run_id>/distributed/workers/worker_000/`.
- [ ] Require each worker to write completion markers and timing/resource summaries.
- [ ] Add validation for missing workers, overlapping assignments, stale config hashes, and incomplete artifacts.

---

## Part 2 — Distributed Pass 1

Each worker owns a disjoint subset of dataset shards and runs the normal first-pass logic with a full local model+SAE bank.

Worker outputs:

- [ ] `latent_stats.partial.pt`
- [ ] `top_ctx.partial.pt`
- [ ] `mid_ctx_candidates.partial.pt`
- [ ] `logit_ctx.partial.pt`
- [ ] `seq_repr.partial.pt`
- [ ] `seq_latent_index/` partial shards, if enabled

Exact merge rules:

- [ ] `latent_stats`: merge Welford/count state exactly across workers.
- [ ] `top_ctx`: for each latent, take top-K from the union of worker top-K rows.
- [ ] `logit_ctx`: exact event top-K merge: sum `latent_counts`, concatenate worker token/prob rows, keep global top tokens by probability, and tie-break by token ID.
- [ ] `seq_repr`: combine by global sequence ID; if capped, use deterministic manifest-level `slot_to_id`/`id_to_slot` tensors from explicit `distributed.sampling_seed`, not `run_id`.
- [ ] `seq_latent_index`: concatenate/validate per-shard files using stable global sequence IDs.
- [ ] `mid_ctx`: collect oversampled deterministic candidate pools during pass 1, merge `latent_stats`, filter candidates with final global stats, then select by priority-reservoir; replay assigned shards only as an exact fallback when candidate-pool coverage is insufficient. Default candidate-pool settings are `enabled: true`, `band_margin_sigma: 1.0`, `max_candidates_per_latent: max(256, 4 * num_ctx_sequences)`, and `on_truncation: replay_fallback` for exact modes.

Verification:

- [ ] Tiny synthetic equivalence test: distributed pass-1 merge equals single-process pass 1.
- [ ] Reduced real-data smoke: compare artifact shapes, finite values, sequence ID ranges, and sampled latent rows.

---

## Part 3 — Negative Context

Build `neg_ctx` after global pass-1 artifacts exist.

Initial exact path:

- [ ] Reuse `multi_gpu_exact`: replicate the capped `seq_repr` index on selected GPUs and split SAE components across devices.
- [ ] Keep each device writing disjoint `neg_ctx` component slices.
- [ ] Validate fill stats, sampled rows, and timing against `single_gpu_exact` on reduced runs.

Later scale path:

- [ ] Add index-sharded ANN only if replicated `seq_repr` becomes memory-heavy or query time dominates.
- [ ] Preserve final `neg_ctx.pt` layout so discovery does not care which backend produced it.

---

## Part 4 — Distributed Pass 2 Dump

After global `top_ctx` is merged, build the global replay sequence list:

```text
top_ctx_sequence_ids = unique sequences referenced by global top_ctx
```

Split this list into contiguous worker chunks with deterministic remainder handling. Each worker replays only its assigned sequences with a full local model+SAE bank and computes the same per-sequence candidate profile currently produced by `TopCoactivation.update_batch()`.

Worker output options:

- [ ] Simple exact mode: write `candidate_ids.partial.pt`, `candidate_vals.partial.pt`, and `sequence_ids.partial.pt`.
- [ ] Scalable exact mode: locally expand sequence rows into partial `(target_latent, candidate_latent, value_sum)` records and write reducer-partitioned files.

The simple exact mode is easiest to validate and should be built first.

---

## Part 5 — Exact Pass 2 Reduce

Pass 2 reduction must preserve the global equation:

```text
top_coact[target, candidate]
  = sum over sequences s where target is in top_ctx(s):
      candidate_score[s, candidate]
```

This means every target latent must receive contributions from every worker that replayed a relevant sequence. Some communication is unavoidable; the goal is to communicate partial sums rather than raw activations.

### Mode A — Simple Exact Merge

- [ ] Concatenate all worker candidate dumps in deterministic sequence order.
- [ ] Build one global `sid_to_row`.
- [ ] Run the existing C++ reducer, optionally with `target_sharded`.
- [ ] Apply PMI postprocess once using global `latent_stats` and global sequence/token counts.
- [ ] Save canonical `outputs/<run_id>/top_coactivation.pt`.

This mode should be the first implementation because it is the simplest exact equivalence target.

### Mode B — Scalable Exact MapReduce

Each worker performs local pre-aggregation:

```text
for seq_id in assigned_sequences:
    candidates = candidate_profile(seq_id)
    targets = global_top_ctx_targets_for_sequence(seq_id)

    for target in targets:
        for candidate, value in candidates:
            if candidate != target:
                local_sum[target, candidate] += value
```

Then each worker shuffles partial sums by target range:

```text
worker_0_to_reducer_0.pt  targets [0, A)
worker_0_to_reducer_1.pt  targets [A, B)
...
worker_7_to_reducer_7.pt
```

Reducers are sharded by target range only; each reducer owns all candidate columns for its targets, so cross-range edges are preserved. Reducers then:

- [ ] Read all worker partial-sum files for their target range.
- [ ] Merge sorted COO partial-sum records: `target_ids`, `candidate_ids`, and `values`, sorted by `(target, candidate)`.
- [ ] Merge duplicate `(target, candidate)` keys by summing values.
- [ ] Keep top-K candidates per target with deterministic tie-breaking.
- [ ] Write reducer shard files using the existing shard schema or a new versioned schema.
- [ ] Stitch reducer shards into final `[36, 40960, K]` tensors.
- [ ] Apply PMI postprocess exactly once globally, or store enough count data for equivalent per-shard PMI.

Do **not** merge worker-local top-K per target as the default exact path. A candidate can be below local top-K on every worker but enter global top-K after summing across workers.

Verification:

- [ ] Tiny synthetic equivalence: single-process reducer, simple exact merge, and MapReduce reducer produce identical `top_indices`/`top_values`.
- [ ] Tie-case test with deterministic candidate ID ordering.
- [ ] Reduced real-data comparison against current single-process pass 2.

---

## Part 6 — Distributed Candidate Selection And Discovery

Candidate selection can remain centralized at first because it is cheap compared with model forwards.

- [ ] Run candidate selection once over merged global artifacts.
- [ ] Split selected seeds across workers.
- [ ] Each worker loads global artifacts plus local full model+SAE bank.
- [ ] Each worker runs discovery/eval for its assigned seeds.
- [ ] Merge circuit stores by appending circuits and validating unique IDs/metadata.
- [ ] Write a global `outputs/<run_id>/circuits/summary.json`.

Later:

- [ ] Add method-level scheduling so expensive methods such as SFC/circuit-tracer get balanced across workers.
- [ ] Add resume support for failed seed ranges.

---

## Part 7 — Operating Modes

- [ ] `single_process`: current pipeline, correctness oracle.
- [ ] `distributed_simple_exact`: distributed pass 1/pass 2/discovery, central exact candidate-dump merge.
- [ ] `distributed_mapreduce_exact`: distributed pass 1/pass 2/discovery, pass-2 partial-sum shuffle and target-range reducers.
- [ ] `distributed_experimental_fast`: optional approximate modes only after exact baselines are benchmarked.
- [ ] Distributed modes keep search-cache generation offline by default.
- [ ] Distributed modes write JSONL controller/worker metrics and optional device observability samples.
- [ ] Distributed modes support cleanup policies such as `keep_all` and `delete_large_partials_on_success`.

Recommended rollout:

1. Build `distributed_simple_exact`.
2. Prove artifact equivalence on tiny and reduced real runs.
3. Benchmark 8x H100 speedup and combine overhead.
4. Build `distributed_mapreduce_exact` only if central candidate-dump merge becomes the bottleneck.

---

## Part 8 — Testing And Benchmarks

- [ ] Unit tests for manifest creation, deterministic assignments, and invalid manifest rejection.
- [ ] Unit tests for sequence-balanced shard assignment, no dropped remainder items, GPU identity metadata, cleanup policies, and JSONL metrics schemas.
- [ ] Merge tests for every pass-1 artifact.
- [ ] Reducer equivalence tests for simple exact and MapReduce exact modes.
- [ ] End-to-end tiny synthetic distributed run compared with single-process outputs.
- [ ] Reduced real-data 1-GPU-vs-2-worker-vs-8-worker comparison.
- [ ] Full H100 benchmark recording wall time, GPU utilization, CPU RAM, VRAM, disk usage, artifact sizes, and circuit outputs.
- [ ] Paper-facing reproducibility run with exact config, manifest, git SHA, native build info, and artifact sanity report.

---

## Open Questions

- [ ] How large are worker candidate dumps at full 256-shard scale, and is simple exact merge already acceptable on the target node?
- [ ] What is the minimum equivalence standard before using distributed artifacts in paper results: exact tensors, exact top-K sets, or statistically indistinguishable circuits?

---

## Risks / Assumptions

- H100 VRAM is assumed sufficient to replicate model+SAE bank per worker.
- Global sequence IDs must remain stable; any drift breaks context, display, neg_ctx, coactivation, and discovery.
- `mid_ctx` merge semantics are the highest pass-1 correctness risk; deterministic priority-reservoir tests must prove unbiased sampling before paper use.
- Pass-2 exactness requires global aggregation by target latent; local top-K merging is not exact.
- The first distributed implementation should optimize for correctness and debuggability before maximum throughput.
