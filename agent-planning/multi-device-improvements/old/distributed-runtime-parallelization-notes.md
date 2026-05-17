# Distributed Runtime Parallelization Notes

> Decision notes for making the Turing Explorer pipeline use a single-node 8x H100 machine efficiently.
> This is not an implementation plan. It records the target architecture, merge contracts,
> operational defaults, observability requirements, and unresolved design questions.

## Summary

Use a dataset-sharded replicated-worker architecture:

- one controller process,
- one worker process per H100,
- one full model and SAE bank per worker,
- disjoint dataset shard assignments,
- local partial artifacts per worker,
- deterministic merge after each distributed stage,
- global downstream stages only after their required artifacts have been merged.

Do not try to make the current single-process runtime fully use 8 GPUs by spreading one model/SAE stack across the node. The model fits on one H100, and the expensive first-pass, second-pass, and discovery work is naturally parallel over sequences or seeds.

The current runtime remains useful as a correctness oracle and single-H100 or current-runtime benchmark. It is not the final 8x H100 design: today the model runs on the primary device, SAE layer splitting only uses the first two devices, and broad GPU parallelism mainly appears in opt-in phases such as multi-GPU negative context.

## Architecture Decisions

### Worker Model

Each H100 worker owns:

- a complete model replica,
- a complete SAE bank,
- a disjoint subset of dataset shards or top-context sequence IDs,
- local partial stores,
- worker-local logs and metrics.

This avoids per-batch synchronization through a central store process. A central-store design would force every worker to stream activations or sparse latent data to one process, making PCIe/NVLink traffic, central store updates, and CPU scheduling likely bottlenecks.

### Controller

Use a Python controller as the canonical launch path for the first single-node distributed runtime.

The controller should:

- create the run manifest,
- inspect shard sequence counts,
- assign work to workers,
- spawn one subprocess per GPU,
- set `CUDA_VISIBLE_DEVICES` per worker,
- record logical and physical GPU IDs,
- monitor worker exit codes,
- own retries and resume behavior,
- run merge and validation phases,
- promote validated outputs to canonical locations.

An external scheduler such as SLURM should launch the controller, not individual workers, in the first version. Multi-node or scheduler-native workers can be added later if needed.

### Work Partitioning

Partition first-pass dataset work by sequence count, not shard count. The controller should greedily assign shards so total expected sequences per worker are balanced. Token count would be better if available, but sequence count is a good first proxy.

Partition second-pass top-coactivation work by the merged global `top_ctx` sequence ID list. Workers should process disjoint sequence ID chunks and emit candidate dump rows keyed by global sequence ID.

Partition discovery by candidate seed after global artifacts and `candidates.pt` are complete. Each discovery worker gets a deterministic subset of candidates.

## Distributed Configuration

Add an explicit distributed config namespace rather than overloading existing runtime keys.

Recommended keys:

- `distributed.enabled`,
- `distributed.run_id`,
- `distributed.output_root`,
- `distributed.scratch_root`,
- `distributed.worker_count`,
- `distributed.launcher`,
- `distributed.resume`,
- `distributed.cleanup_policy`,
- `distributed.assignment_policy`.

Workers should not rely on hand-edited config files for their identity. The controller manifest should provide each worker's `worker_id`, logical device, physical device, assigned shards, sequence ranges, and output paths.

Keep `torch.compile` enabled by default for H100 benchmark and production runs, but log compile and warmup cost separately per worker. Tiny smoke tests may disable compile if compile cost would dominate the test.

## Run Manifest And Artifact Metadata

Distributed execution needs durable metadata before any merge is trusted.

The controller manifest should record:

- `run_id`,
- config hash,
- code revision and dirty status if available,
- model path and optional checksum,
- SAE path and optional checksum or mtime summary,
- dataset path,
- shard sequence counts,
- shard-to-worker assignments,
- global sequence ID mapping,
- global `seq_repr` cap mapping when capped mode is used,
- visible GPU count and physical GPU identities,
- expected artifacts per worker,
- expected merged outputs,
- schema versions for all distributed artifacts.

Every worker artifact should include:

- `schema_version`,
- `run_id`,
- `config_hash`,
- model and SAE identity,
- `worker_id`,
- artifact kind,
- assigned shards or sequence IDs,
- global sequence ID ranges covered,
- creation timestamp,
- completion status or a separate completion marker.

Merges must write to staging outputs first. Canonical `outputs/` artifacts should only be promoted after metadata validation and artifact sanity checks pass. Current single-file atomic saves are not sufficient as a cross-artifact transaction.

## Determinism And Randomness

The distributed runtime should target deterministic, semantically equivalent outputs. It does not need byte-for-byte equality with the current online single-process order where that order is inherently unstable.

Order-sensitive behavior today includes:

- floating point Welford merges,
- `torch.topk` tie ordering,
- reservoir sampling,
- online `mean_seq/std_seq` updates during mid-context collection,
- random candidate selection,
- random counterfactual negatives,
- UUID creation.

Distributed mode should use explicit stable tie-breakers. The default ordering for top-K style merges should be score ascending or descending as appropriate, followed by stable IDs such as `seq_id`, `token_id`, component, latent, method, and candidate index.

Global random decisions should happen on the controller where possible. If workers need randomness, derive per-worker or per-seed RNG streams from `run_seed`, `worker_id`, phase name, and stable candidate identity. UUIDs remain fine for uniqueness but should not be used as equality keys in validation.

## Store Merge Contracts

### `latent_stats`

Merge counts exactly and merge Welford tuples in fixed worker order:

- `active_count`,
- `mean`,
- `m2`,
- `mean_abs`,
- `m2_abs`,
- `seq_count`,
- `mean_seq`,
- `m2_seq`,
- `component_steps`.

`component_steps` should be summed per component or derived from the worker manifests. Distributed `gpu_topk_mid` reduces reliance on online warmup behavior, but the field should still be coherent.

This merge is deterministic and numerically equivalent, but not necessarily byte-identical to one long sequential pass. For stronger reproducibility later, persist worker-side `sum`, `sum_sq`, and `count` and derive means and variances from a fixed-order reduction.

### `top_ctx`

Merge exactly by per-latent top-K:

- gather worker `(seq_id, value)` candidates for each `(component, latent)`,
- sort by `value desc`, then `seq_id asc`,
- keep global K.

The true global top-K is contained in the union of worker top-Ks, so this merge is exact with an explicit tie-breaker.

### `mid_ctx`

Use `gpu_topk_mid` for distributed mode.

`reservoir_cpu` is order/RNG-dependent and should not be the first distributed policy. `gpu_topk_mid` is deterministic and mergeable: for each latent, keep in-band sequences whose score is closest to the configured band midpoint.

Distributed merge ordering:

- distance to midpoint ascending,
- `seq_id` ascending.

Unit tests should compare `gpu_topk_mid` against an independent `torch.topk` reference for non-tie cases and separately verify deterministic tie-breaking.

### `logit_ctx`

Merge similarly to context top-K:

- sum `latent_counts`,
- gather candidate `(token_id, probability)` entries,
- sort by `probability desc`, then `token_id asc`,
- keep global K.

If exact tie provenance is needed later, include `seq_id` in worker payloads and sort by `probability desc`, `token_id asc`, `seq_id asc`.

### `seq_repr`

Merge by stable global sequence ID.

Uncapped mode places each worker representation directly into the canonical `repr_buf` row for that `seq_id`. Capped mode must use one global cap mapping created by the controller:

- choose capped sequence IDs once globally,
- save `slot_to_id` and `id_to_slot` in the manifest,
- give the mapping to every worker,
- never let workers sample their own caps.

### `seq_latent_index`

Workers should write sequence-latent index shards under isolated worker output paths. The merger should copy or stitch by global dataset shard ID.

Validation:

- every expected dataset shard has exactly one index artifact,
- no two workers wrote the same shard,
- shard ID ranges match the manifest,
- tensor shapes and top-K settings match the config.

### `neg_ctx`

Build negative context after global first-pass artifacts are merged:

- merged `seq_repr`,
- merged `top_ctx`,
- merged `mid_ctx`.

Then run the existing `single_gpu_exact` or `multi_gpu_exact` backend against global artifacts. Avoid index-sharded ANN in the first distributed runtime. Revisit it only if global negative context becomes a bottleneck.

### `candidates.pt`

Run candidate selection once on the controller after global stores and top-coactivation are complete. Do not run candidate selection independently on workers.

The controller writes canonical `candidates.pt`, then partitions that list for discovery workers using stable candidate indices or stable `(component, latent, criterion)` identities.

## Top-Coactivation

Use Option A: distributed candidate dump, then global reduce.

### Candidate Dump

Each worker runs the second-pass model and SAE forward for a subset of global top-context sequence IDs, then writes compact per-sequence rows:

- `candidate_ids`: `[num_sequences, M]` int32,
- `candidate_vals`: `[num_sequences, M]` float32,
- row sequence IDs.

With current H100 settings:

- `n_latents_per_latent = 64`,
- `num_components = 36`,
- `n_candidates_per_component = 16`,
- `M = min(64 * 4, 36 * 16) = 256`,
- row size is `256 * 4 + 256 * 4 = 2048` bytes.

The dump is roughly 2 KiB per top-context sequence. A full 256-shard run with about 2.1M top-context rows is roughly 4 GiB total, or about 512 MiB per worker across 8 workers.

Each dump chunk should record:

- row sequence IDs,
- `M`,
- coactivation mode such as `pmi` or `raw`,
- `n_candidates_per_component`,
- `n_latents_per_latent`,
- sequence length or token count if needed by PMI post-processing,
- worker ID,
- chunk index,
- completion marker.

### Merge And Reduce

Merge once after all distributed top-coactivation dump workers complete. Worker chunk flushes are checkpointing, not semantic merges.

Recommended flow:

1. Merge global `top_ctx`.
2. Split the global top-context sequence ID list across workers.
3. Workers write candidate dump chunks keyed by global sequence ID.
4. Merge candidate dump chunks by global sequence ID.
5. Run the existing reducer once globally.
6. Add target-sharded or parallel reducer execution as a follow-on optimization.

Materialize one merged candidate dump for the first implementation. Later, if I/O becomes a bottleneck, consider memory-mapping or streaming chunks directly into the reducer.

### Avoid Local Partial Reductions First

Do not start by having each worker produce a partial `top_coactivation.pt` and then merging those top-K outputs.

Local top-K reduction can discard globally important candidates. A candidate that is second or third on every worker can become first globally once counts or sums are combined. Making local partial reduction exact requires additive accumulators or much larger candidate sets, which is more complex than merging candidate dumps first.

### Dense `max_sid` Lookup Risk

The current GPU dump path can build a dense `seq_id -> row` lookup sized by `max(sequence_id) + 1`. This is fast for dense sequence IDs, but can waste VRAM when a worker processes a sparse subset with large global IDs.

Distributed top-coactivation should prefer:

- direct row offsets when batches are emitted in dump-row order,
- row sequence IDs plus sort/join merge by `seq_id`,
- CPU lookup tensors when GPU memory matters,
- compact mappings if sequence IDs become sparse or very large.

## Discovery

Parallelize discovery by seed/candidate after global artifacts are complete.

Recommended flow:

1. Run candidate selection once on the controller.
2. Split canonical candidates across workers.
3. Each worker loads the model, SAE, and required global artifacts.
4. Each worker runs discovery for assigned candidates.
5. The controller merges circuit shards and recomputes summaries.

Use separate processes, not threads. The code has global store and observability singletons, so threaded discovery would risk shared-state corruption.

Discovery worker outputs should be isolated:

- per-worker circuit shard,
- per-worker discovery logs,
- per-worker accepted/failed metadata,
- no direct writes to canonical `outputs/circuits/summary.json` or `summary.xlsx`.

The merge contract should include:

- candidate identity,
- seed component and latent,
- discovery method,
- accepted or rejected status,
- circuit nodes and edges,
- faithfulness and evaluation metrics,
- deterministic conflict handling if duplicate work appears.

After merge, summaries should be sorted by stable fields such as candidate index, seed component, seed latent, and discovery method. Do not rely on UUID or insertion order for deterministic summaries.

Cluster contrast is separate from seed/candidate discovery. It has a global clustering step and writes under `outputs/cluster_circuits/`. Keep it single-controller initially, or later shard by cluster ID with its own merge contract.

## Runtime, Storage, And Failure Policy

### Directory Layout

Use run-specific directories:

```text
outputs/runs/<run_id>/
  manifest.json
  controller/
  workers/<worker_id>/
  partial/
  merged_staging/
  final/
  logs/
  metrics/
```

Hot intermediates should go to local NVMe scratch when available. Canonical merged artifacts should be promoted to durable storage only after validation.

Compatibility paths such as `outputs/*.pt` can be symlinks, copies, or final promoted artifacts from `outputs/runs/<run_id>/final/`. The run-specific directory should remain the source of truth for distributed execution.

### Scratch And Retention

Workers should write large hot artifacts to `scratch_root` first:

- first-pass partial stores,
- top-coactivation candidate dump chunks,
- reducer shards,
- profiler traces.

Default retention should be conservative:

- keep manifest, metrics, logs, final artifacts, and validation reports,
- keep partial artifacts after failed runs,
- after successful validated runs, delete large worker candidate-dump chunks if cleanup policy allows,
- keep small worker summaries and metadata even if large tensors are cleaned.

Recommended `cleanup_policy` values:

- `keep_all`,
- `delete_large_partials_on_success`,
- `delete_all_partials_on_success`.

Default to `delete_large_partials_on_success` for mature full H100 runs, and `keep_all` while validating the runtime.

### Capacity Checks

Preflight should estimate and check space for:

- first-pass worker partials,
- top-coactivation candidate dumps,
- reducer shards,
- merged staging artifacts,
- final artifacts,
- logs,
- profiler traces.

Check free bytes and, where possible, inode availability. During the run, sample scratch usage and stop cleanly before artifacts are corrupted by disk exhaustion.

### Failure Policy

Classify failures and retry only when retry is meaningful:

- worker crash: retry from the last completed artifact boundary,
- corrupt partial artifact: rerun the owning worker,
- validation failure after merge: preserve staging outputs and do not promote,
- out of memory: abort unless a lower-memory retry policy is explicit,
- out of disk or inode exhaustion: abort and preserve partials,
- config/schema/hash mismatch: abort merge and require clean rerun of mismatched workers,
- missing native extension: fail preflight before expensive work starts.

Retries should be bounded. Do not retry indefinitely on deterministic failures such as OOM, schema mismatch, or missing native extensions.

### Search Cache

Keep search cache generation offline for distributed runs. Build it only after canonical merged artifacts exist, using the final merged `top_ctx.pt` and the controller manifest. Tag the cache with the same `run_id` and config hash.

### CUDA Device Mapping

Record both logical and physical GPU identity. A worker may see its assigned H100 as `cuda:0` because the controller sets `CUDA_VISIBLE_DEVICES`, but the manifest and metrics should also record physical GPU ID, UUID, name, and PCI bus ID when available.

## Observability

The distributed runtime should include structured metrics from the start. The core questions are:

- Did every H100 do useful work?
- Where did wall time go?
- What speedup and parallel efficiency did we get versus the single-H100 baseline?

### Metrics Artifacts

Write structured metrics instead of relying on console logs:

- `outputs/runs/<run_id>/metrics/run_metrics.jsonl`,
- `outputs/runs/<run_id>/metrics/run_summary.json`,
- `outputs/runs/<run_id>/workers/<worker_id>/worker_metrics.jsonl`,
- `outputs/runs/<run_id>/workers/<worker_id>/worker_summary.json`.

Every event should include `run_id`, worker or controller identity, phase name, timestamp, elapsed time when applicable, artifact or chunk ID when applicable, PID, and hostname.

### Phase Metrics

Track at least:

- phase wall time,
- sequences/sec and tokens/sec,
- batches/sec,
- worker batch counts,
- output bytes per artifact,
- merge time per artifact,
- top-coactivation dump size and reduce time,
- discovery seeds/sec,
- forward/backward pass counts.

Phase-specific counters:

- first pass: model forward time, SAE encode time, store update time, `seq_repr` update time.
- first-pass merge: merge time for `latent_stats`, `top_ctx`, `gpu_topk_mid`, `logit_ctx`, `seq_repr`, and `seq_latent_index`.
- negative context: backend, devices, index build time per device, query/filter/write times, fill stats, ANN index size, replicated index memory.
- second pass: top-context sequence count, candidate dump rows and bytes, model forward time, SAE encode time, dump update time.
- top-coactivation reduce: reducer backend, shard count, per-shard reduce time, shard write time, merge time, OpenMP thread count, schedule chunk, final artifact bytes.
- candidate selection: candidates scored, candidates selected, criterion timings if available.
- discovery: candidates assigned/completed, accepted circuits, failed circuits, time per method, eval time, analysis time.
- persistence: artifact save time, artifact size, atomic rename time if measurable.

### NVML And System Sampling

Run a lightweight sampler every 1-5 seconds during controller and worker execution.

Per GPU sample:

- utilization percent,
- memory utilization percent,
- used and total VRAM,
- power draw and power limit,
- temperature,
- SM and memory clocks when available,
- PCIe or NVLink throughput when available.

Host/process sample:

- CPU utilization,
- CPU RSS,
- system RAM usage,
- disk read/write bytes,
- disk write throughput,
- network filesystem latency if outputs live on shared storage.

Each sample should include the active phase label. Current phase resource logging only reports the current CUDA device; distributed runs need all-device snapshots.

### Efficiency Summary

Final summary should report:

- total wall time,
- phase wall-time percentages,
- GPU-hours,
- effective sequences/sec,
- effective tokens/sec,
- average and p95 GPU utilization per device,
- average utilization by phase,
- p50/p95/p99 batch time per worker,
- worker straggler ratio,
- artifact write throughput,
- merge overhead percentage,
- speedup versus single-H100 control,
- parallel efficiency: `speedup / worker_count`,
- idle GPU fraction.

### Logs And Profiling

Keep human-readable logs, but make structured metrics the source of truth.

Recommended logs:

- `logs/controller.log`,
- `logs/merge.log`,
- `workers/<worker_id>/worker.log`,
- `workers/<worker_id>/discovery_logs/...`,
- optional traces under `profiles/<phase>/`.

For one reduced H100 run, collect deeper traces around:

- first-pass model plus SAE encode,
- second-pass model plus SAE encode,
- one `multi_gpu_exact` negative-context worker,
- top-coactivation dump,
- top-coactivation reducer.

Do not enable heavy `torch.profiler`, Nsight Systems, or Nsight Compute by default on full production runs.

### Preflight And End-Of-Run Checks

Preflight should record:

- hostname,
- OS and Python version,
- PyTorch version,
- CUDA runtime and driver versions,
- visible GPU count, names, and VRAM,
- logical and physical GPU IDs,
- native extension availability,
- OpenMP/MKL/PyTorch thread settings,
- config hash,
- git commit and dirty status if available,
- output directory and filesystem type if available.

End-of-run checks should verify:

- all worker summaries exist,
- every worker reached a terminal status,
- all expected artifacts exist,
- artifact metadata matches the manifest,
- structured metrics cover every phase,
- no worker had zero GPU utilization during phases it was expected to run.

## Validation And Acceptance

### Artifact Validation

Before promotion to final outputs, validate:

- every dataset shard assigned exactly once,
- every expected sequence ID range covered exactly once,
- every discovery candidate processed exactly once,
- all partial artifacts share config hash and schema version,
- tensor shapes and dtypes match model/SAE config,
- merged artifact sanity checks pass,
- downstream phases only read validated global artifacts.

### Scientific Validation

Distributed validation should include scientific checks, not only tensor checks.

Compare distributed output against a single-process control on tiny synthetic and reduced real-data runs:

- candidate ranking overlap,
- top-context overlap,
- mid-context overlap under `gpu_topk_mid`,
- top-coactivation neighbor overlap and score correlation,
- negative-context fill-rate and sampled-row agreement,
- discovered circuit count,
- faithfulness distribution,
- top circuit overlap by seed/method.

Define exactness and tolerance per artifact. Deterministic integer IDs and explicit tie-breakers should match exactly. Floating point statistics and scores may tolerate small drift if downstream rankings and scientific conclusions remain stable.

### Test Coverage

Distributed mode needs focused tests beyond current single-process coverage:

- tiny synthetic merge tests for `latent_stats`, `top_ctx`, `gpu_topk_mid`, `logit_ctx`, `seq_repr`, and candidate dumps,
- `gpu_topk_mid` reference tests against independent `torch.topk` for non-tie cases,
- tie-breaker tests for `top_ctx`, `gpu_topk_mid`, `logit_ctx`, and top-coactivation dump merge,
- capped `seq_repr` tests proving all workers use the same global slot mapping,
- `seq_latent_index` merge/collision tests,
- circuit-store shard merge tests,
- candidate partition coverage tests,
- resume/idempotency tests where one worker is rerun,
- small end-to-end equivalence test comparing one-process output to two-worker merged output.

UUIDs and insertion order should not be primary equality keys.

## H100 Rollout Notes

Operational details can dominate H100 runs:

- Build and test native extensions on the target runtime image.
- Be careful with CPU-specific native flags such as `-march=native` if compiling on a different machine from the compute node.
- Preflight should confirm expected CUDA/Triton/cublasLt/native extensions are loaded.
- Set OpenMP, MKL, and PyTorch CPU thread counts deliberately to avoid oversubscription.
- Prefer local NVMe scratch for reducer shards, candidate dumps, traces, and chunk files.
- Account for disk bytes and inode counts before launch.
- Install optional CPU metrics dependencies such as `psutil`, or clearly report degraded metrics.
- Keep README/runbook documentation aligned with actual native extension names and H100 setup commands.

## Pairwise-Discovery Compatibility

Keep the distributed runtime artifact model generic enough for future pairwise-discovery stages.

Pairwise discovery has not been implemented yet. The repository currently contains planning documents for it, but not a `pairwise_discovery` config section, pipeline stage, standalone runner, pair candidate schema, measurement code, atlas reducer, or tests. This section is future-facing: it exists so the distributed runtime does not choose artifact conventions that make pairwise discovery harder later.

The pairwise plan already expects tensor-first, shardable artifacts with manifests. Reuse the same conventions for:

- run IDs,
- schema versions,
- shard metadata,
- completion markers,
- staging directories,
- validation reports,
- cleanup policy.

This should let pairwise discovery plug into the same controller/worker/merge system later.

## Open Design Questions

- Should distributed `latent_stats` persist sums/squares alongside Welford fields for stronger reproducibility?
  Recommendation: start by merging the existing Welford fields in fixed worker order, then add optional persisted sums/squares only if reduced-run equivalence shows unacceptable drift.

- Should `gpu_topk_mid` gain explicit `(distance, seq_id)` tie-breaking before distributed mode relies on it?
  Recommendation: yes. Add explicit tie-breaking before using distributed `gpu_topk_mid` as a correctness-sensitive artifact, and keep a non-tie `torch.topk` reference test.

- What chunk size should worker candidate dumps use for good checkpointing without too many files?
  Recommendation: target chunks around 256-1024 MiB per file, with a config override. This keeps files large enough for efficient I/O while limiting rerun cost after a worker failure.

- Should merged top-coactivation candidate dumps stay materialized, become memory-mapped, or stream directly into the reducer?
  Recommendation: materialize one merged dump for the first implementation because it is easiest to validate. Revisit memory mapping or streaming only if I/O or disk pressure becomes a measured bottleneck.

- Should target-sharded reduction run sequentially first, then use parallel reducer processes later?
  Recommendation: yes. First validate target-sharded reduction sequentially against the merged candidate dump, then add parallel reducer processes once the artifact contract is stable.

- What is the minimum equivalence suite before trusting distributed runs: tiny synthetic, reduced real-data, or both?
  Recommendation: require both. Tiny synthetic tests should prove exact merge semantics, and a reduced real-data run should prove pipeline-level behavior, artifact sanity, and scientific stability.

- How much artifact drift is acceptable for discovery outputs given deterministic but non-byte-identical floating point merges?
  Recommendation: require exact shape/schema/ID equality, tolerate small floating-point score drift, and judge discovery stability by candidate overlap, circuit count, faithfulness distribution, and top-circuit overlap rather than UUID equality.

- Should discovery random negatives be deterministic per seed by default?
  Recommendation: yes. Use deterministic per-seed RNG derived from `run_seed`, discovery method, seed component, and seed latent. Allow nondeterministic mode only as an explicit experimental option.

- Should cluster contrast remain single-controller in the first distributed runtime?
  Recommendation: yes. Keep cluster contrast out of the first distributed worker design because it has a separate global clustering step and output layout. Add cluster-ID sharding later if it becomes important.

- What is the canonical artifact schema strategy for stores that are currently plain `torch.save` dictionaries?
  Recommendation: wrap each distributed artifact in a small versioned payload with `schema_version`, metadata, and tensors. Preserve compatibility by allowing loaders to read legacy monolithic checkpoints where practical.

