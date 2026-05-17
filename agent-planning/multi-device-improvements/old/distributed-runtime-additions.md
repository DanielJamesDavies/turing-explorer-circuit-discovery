# Plan: Distributed Runtime Additions

> **Goal:** Add a controller/worker runtime that can split large Turing Explorer pipeline runs across multiple GPUs or processes, merge partial artifacts deterministically, and prove equivalence against the existing single-process pipeline before using it for full-scale research runs.
>
> **Created:** 2026-04-25

---

## Background

The current H100 scaling work intentionally keeps the existing single-process pipeline intact and adds opt-in acceleration modes for selected phases. That is the right near-term path for Phase 8 of `agent-planning/h100-pipeline-scaling-and-bottlenecks.md`.

This plan is for the larger follow-on runtime that should only begin after Phase 8 benchmarks show that full-scale first-pass and second-pass model/SAE forward work still dominates enough to justify distributed orchestration and mergeable artifact formats.

Target shape:

```text
controller process
  writes run manifest
  assigns dataset shards and target ranges
  monitors worker completion
  validates and merges artifacts

worker 0 on H100: process assigned dataset shards -> local partial stores
worker 1 on H100: process assigned dataset shards -> local partial stores
...
worker 7 on H100: process assigned dataset shards -> local partial stores

merge phase:
  latent_stats
  top_ctx / mid_ctx / logit_ctx
  seq_repr
  top_coactivation candidate or reducer shards
  neg_ctx/discovery inputs
```

Expected benefit:

- Small reduced runs probably do not justify this complexity.
- Full 256-shard runs and repeated research sweeps could plausibly see large wall-clock gains because first-pass and second-pass model/SAE work are data-parallel.
- End-to-end speedup will likely be less than 8x due to initialization, merge, persistence, discovery, and CPU/OpenMP phases, but a `2.5x-6x` improvement on large forward-heavy workloads is a realistic target to test.

---

## Phase 1 — Entry Criteria And Scope Gate

- [ ] Use Phase 8 H100 benchmark results to confirm whether first-pass and second-pass model/SAE forward work dominate full or medium-scale runs.
- [ ] Define minimum expected speedup and maximum acceptable artifact/quality drift before starting implementation.
- [ ] Decide the initial distributed scope: first-pass only, second-pass only, or both first and second pass.
- [ ] Keep transformer model parallelism out of scope unless H100 profiling shows the model forward itself cannot fit or run efficiently on one GPU.
- [ ] Write a short decision note linking the Phase 8 benchmark data to the chosen distributed-runtime scope.

## Phase 2 — Run Manifest And Dataset Partitioning

- [ ] Add a run manifest format containing config hash, model path, SAE path, dataset path, shard assignment, worker count, output directories, artifact schema versions, and completion markers.
- [ ] Add deterministic dataset shard partitioning across workers.
- [ ] Preserve global sequence IDs exactly so display, search, context stores, negative contexts, and discovery all reference the same examples after merging.
- [ ] Add manifest validation for missing shards, overlapping shard assignments, stale config hashes, and incomplete workers.
- [ ] Add unit tests for manifest creation, partition determinism, and invalid manifest rejection.

## Phase 3 — Worker Entrypoints And Local Partial Outputs

- [ ] Add a worker entrypoint that can run one assigned dataset partition on one GPU without touching global in-memory stores.
- [ ] Ensure each worker writes to an isolated output directory with atomic completion markers.
- [ ] Add explicit worker metadata: worker ID, CUDA device, dataset shard IDs, sequence ID range, start/end timestamps, and success/failure status.
- [ ] Keep the current single-process pipeline as the default entrypoint.
- [ ] Add a small local multi-worker smoke test using CPU or one GPU with tiny synthetic shards.

## Phase 4 — Mergeable Store Formats

- [ ] Implement and test `latent_stats` merges using count/sum/max-style reductions.
- [ ] Implement and test `top_ctx` merges as per-latent top-K merges by activation value.
- [ ] Implement and test `logit_ctx` merges as per-latent top-token merges.
- [ ] Define and implement `mid_ctx` merge semantics:
  - [ ] exact or statistically valid reservoir merge for `reservoir_cpu`,
  - [ ] deterministic top-K midpoint merge for `gpu_topk_mid`,
  - [ ] metadata that records which merge policy produced the artifact.
- [ ] Implement and test `seq_repr` concatenation with stable global sequence IDs and capped-storage mappings.
- [ ] Define whether `top_coactivation` merges candidate dumps, reduced target shards, or both.
- [ ] Add small-run equivalence tests comparing merged artifacts against the single-process pipeline.

## Phase 5 — Controller, Resume, And Retry

- [ ] Add a controller entrypoint that writes the manifest, launches or records worker commands, and monitors completion markers.
- [ ] Add resume behavior so completed workers are skipped and failed workers can be rerun without restarting the whole job.
- [ ] Add cleanup rules for incomplete partial artifacts without deleting unrelated run outputs.
- [ ] Add clear logs for worker assignment, worker status, merge start/end, and validation results.
- [ ] Add tests for successful resume, failed-worker retry, and stale partial-output cleanup.

## Phase 6 — Global Merge And Downstream Integration

- [ ] Add a merge command that consumes the manifest and worker outputs, validates all partial artifacts, and writes the canonical global artifacts.
- [ ] Build `neg_ctx` only after global `seq_repr`, `top_ctx`, and `mid_ctx` are available unless a deeper sharded negative-context design is chosen later.
- [ ] Reuse the current `single_gpu_exact` or `multi_gpu_exact` negative-context backends after global merge.
- [ ] Run candidate selection and discovery against the merged global artifacts without changing their public artifact contracts.
- [ ] Add artifact sanity checks for tensor shapes, dtypes, finite values, valid sequence IDs, valid latent IDs, and expected fill statistics.

## Phase 7 — Distributed Top-Coactivation Strategy

- [ ] Decide whether the first distributed implementation should merge second-pass candidate dumps before reduction or produce target-reducer shards directly.
- [ ] Preserve the final `top_coactivation.pt` shape and semantics.
- [ ] Add strict equivalence tests for single-process versus distributed top-coactivation on tiny synthetic data.
- [ ] Add benchmark coverage for dump merge cost, reducer shard cost, shard-file write/merge overhead, and total second-pass wall time.
- [ ] Document when to use current `target_sharded` reduction versus the distributed runtime.

## Phase 8 — Testing And Verification

- [ ] Run focused unit tests for every merge policy and manifest/controller behavior.
- [ ] Run small synthetic end-to-end equivalence tests for single-process versus distributed artifacts.
- [ ] Run a reduced real-data distributed smoke test on one machine before any full H100 job.
- [ ] Compare `latent_stats.pt`, `top_ctx.pt`, `mid_ctx.pt`, `neg_ctx.pt`, `top_coactivation.pt`, `candidates.pt`, and `outputs/circuits/summary.json` against a single-process control.
- [ ] Confirm search-cache generation works from merged artifacts when run as the standalone offline command.
- [ ] Record performance deltas and circuit-quality differences before recommending distributed mode.

## Phase 9 — H100 Rollout Documentation

- [ ] Publish a conservative distributed-runtime config for full-dataset H100 runs.
- [ ] Document which phases use which GPUs or CPU resources.
- [ ] Document expected disk usage, partial artifact layout, and cleanup procedure.
- [ ] Add warnings for modes that are not yet data-parallel or not yet validated at full scale.
- [ ] Add a runbook for launching, resuming, validating, and post-processing distributed runs.

---

## Open Questions

- Should the initial runtime distribute first pass only, second pass only, or both?
- Should workers be launched by a controller subprocess model, a shell/job-launcher script, or an external scheduler?
- Should `top_coactivation` merge candidate dumps, reduced target shards, or support both strategies?
- What exact merge policy should `mid_ctx.reservoir_cpu` use to preserve reservoir sampling semantics across workers?
- Should negative context remain a post-merge global phase, or should it eventually become index-sharded across GPUs?
- Which artifact format changes are acceptable, if any, for faster partial-output persistence?

## Risks / Assumptions

- Existing single-process behavior remains the correctness oracle.
- Existing public artifact names and final tensor shapes should stay compatible unless explicitly changed.
- Distributed runs require strict global sequence-ID discipline; any drift breaks display, search, context, negative-context, and discovery semantics.
- Mergeable `mid_ctx` semantics are the highest correctness risk because reservoir and GPU-topk modes are not statistically identical.
- Controller/worker orchestration adds failure modes around partial files, stale manifests, and retry behavior.
- This runtime should not begin until Phase 8 H100 benchmarks justify the added complexity.
