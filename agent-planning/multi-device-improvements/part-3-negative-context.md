# Plan: Part 3 - Negative Context

> **Goal:** Build `neg_ctx` from merged global pass-1 artifacts using exact single- or multi-GPU retrieval, preserving the canonical `neg_ctx.pt` contract for downstream pass 2, candidate selection, and discovery.
>
> **Created:** 2026-05-16

---

## Scope

This part starts after Part 2 has produced validated global first-pass artifacts:

- `outputs/<run_id>/top_ctx.pt`
- `outputs/<run_id>/mid_ctx.pt`
- `outputs/<run_id>/seq_repr.pt`

It builds:

- `outputs/<run_id>/neg_ctx.pt`
- `outputs/<run_id>/neg_ctx_stats.json`

It does not rerun model forwards, modify pass-1 merge logic, run pass 2, or run discovery.

The initial target is exactness and artifact compatibility. The current `single_gpu_exact` and `multi_gpu_exact` semantics remain the baseline.

---

## Phase 1 - Input Contract And Loader

- [ ] Add a negative-context runner that loads merged global `top_ctx`, `mid_ctx`, and `seq_repr` artifacts without requiring model or SAE initialization.
- [ ] Validate that required pass-1 artifacts exist before starting ANN work.
- [ ] Validate artifact compatibility: component count, SAE width, context counts, sequence ID ranges, `seq_repr` cap metadata, and config hash when available.
- [ ] Support the current in-pipeline path and a standalone distributed-stage path that can be launched after Part 2 merge.
- [ ] Keep the canonical output paths `outputs/<run_id>/neg_ctx.pt` and `outputs/<run_id>/neg_ctx_stats.json` for downstream code in run-root mode.
- [ ] Verification: add a tiny fixture test that loads synthetic pass-1 artifacts and rejects missing or incompatible inputs.

## Phase 2 - Preserve Single-Device Exact Path

- [ ] Keep `single_gpu_exact` as the correctness baseline for reduced/local runs.
- [ ] Ensure `hardware.ann_device` continues to support `auto`, `cpu`, `gpu`, `cuda`, and explicit `cuda:N`.
- [ ] Ensure local RTX 5070 Ti style runs can build `neg_ctx` without the distributed controller.
- [ ] Add clearer stats for single-device exact runs: index build time, positive collection time, query time, filter time, write time, fill counts, device, and `seq_repr` cap size.
- [ ] Validate that CPU fallback remains available for tests and small synthetic fixtures.
- [ ] Verification: extend focused tests around `_ann_device`, `TorchANNIndex`, `_process_component`, and `NegCtxStats` using CPU fixtures.

## Phase 3 - Multi-GPU Exact Backend Hardening

- [ ] Reuse `multi_gpu_exact` as the initial H100 path: replicate the capped `seq_repr` index on selected devices and split SAE components across devices.
- [ ] In distributed runs, default selected devices to the manifest-declared physical devices for the run.
- [ ] Keep "all visible devices" as a standalone/single-process convenience only, not the distributed default.
- [ ] Validate selected devices from manifest/config and reject unavailable CUDA IDs before any expensive allocation.
- [ ] Keep component assignments disjoint and deterministic.
- [ ] Ensure each worker thread writes only its assigned `neg_ctx` component slices.
- [ ] Record per-device component assignments and timing in `neg_ctx_stats.json`.
- [ ] Add a hard validation step after build: every populated row has valid sequence IDs, non-negative finite similarities, and at most `n_sequences` entries.
- [ ] Verification: add tests for deterministic component partitioning, duplicate device removal, invalid device rejection, and stats merge behavior.

## Phase 4 - Equivalence And Quality Checks

- [ ] Add a reduced-run comparison mode that builds `neg_ctx` with `single_gpu_exact` and `multi_gpu_exact` from the same pass-1 artifacts.
- [ ] Compare artifact shapes, dtype, populated row counts, fill-rate distribution, and sampled rows.
- [ ] Require exact or near-exact row equality for the same backend settings and deterministic inputs.
- [ ] Report any expected ordering/tie differences explicitly.
- [ ] Compare total runtime and timing breakdowns to determine whether multi-GPU query compute or CPU write-back dominates.
- [ ] Verification: add a synthetic equivalence test where single-device and multi-device component splits produce identical `neg_ctx` rows.

## Phase 5 - Manifest Integration

- [ ] Add manifest fields for negative-context backend, selected devices, `n_neighbors`, `n_sequences`, `min_pos_ctx`, `repr_mode`, and `max_repr_seqs`.
- [ ] Record whether device selection came from manifest-declared run devices, explicit config override, or standalone all-visible-device discovery.
- [ ] Record whether this part used `single_gpu_exact`, `multi_gpu_exact`, or a future index-sharded backend.
- [ ] Add part status markers under `outputs/<run_id>/distributed/parts/neg_ctx/`.
- [ ] Mark Part 3 completed only after `neg_ctx.pt`, `neg_ctx_stats.json`, and sanity validation all succeed.
- [ ] Support resume behavior: skip rebuild only if the config hash, input artifact metadata, backend, devices, and output sanity report match.
- [ ] Verification: add dry-run and resume tests for completed, stale, failed, and missing `neg_ctx` outputs.

## Phase 6 - Memory And Scale Guardrails

- [ ] Estimate per-device ANN index memory before building the index: `n_stored * repr_dim * 4` plus slot maps and working buffers.
- [ ] Warn or fail early when replicated index memory is likely to exceed a configured fraction of available VRAM.
- [ ] Log `seq_repr.n_stored`, `seq_repr.n_seqs`, cap percentage, representation dim, and estimated index memory.
- [ ] Bound query working memory using the existing chunking strategy and document the expected peak tensor sizes.
- [ ] Keep `max_repr_seqs` as the primary guardrail for replicated-index mode.
- [ ] Verification: add tests for memory estimate calculations and guardrail behavior without requiring CUDA.

## Phase 7 - Optional Index-Sharded Design

- [ ] Add an index-sharded design only if replicated `seq_repr` is memory-heavy or query time dominates at H100 scale.
- [ ] Split `seq_repr` rows across devices instead of replicating the full index.
- [ ] For each query row, collect top candidates from every index shard, then merge shard-local top-K results into global top-K before filtering positives.
- [ ] Preserve final `neg_ctx.pt` semantics exactly: rows still contain global sequence IDs and cosine similarities.
- [ ] Keep index-sharded mode behind a new explicit backend name, for example `multi_gpu_index_sharded_exact`.
- [ ] Verification: prove index-sharded exact output matches replicated-index exact output on synthetic and reduced real-data runs.

## Phase 8 - Output Validation And Reporting

- [ ] Add a `neg_ctx` sanity report with backend, devices, sequence cap, fill mean/min/max, zero-negative rows, invalid sequence counts, non-finite similarity counts, and timing breakdown.
- [ ] Save the report beside `neg_ctx_stats.json` or include it in the stats JSON with a schema version.
- [ ] Print a concise summary suitable for H100 benchmark logs.
- [ ] Include sampled row comparisons when running equivalence mode.
- [ ] Preserve downstream artifact shape and field names so discovery does not need to know which backend produced `neg_ctx`.
- [ ] Verification: add tests for sanity report generation and failure on invalid tensors.

## Phase 9 - Testing And Verification

- [ ] Run focused unit tests for device parsing, component partitioning, ANN search, `_process_component`, stats merging, memory estimates, and sanity validation.
- [ ] Run synthetic single-vs-multi equivalence tests.
- [ ] Run one-worker/local compatibility tests using CPU or one CUDA device.
- [ ] Run a reduced real-data smoke after Part 2 artifacts exist.
- [ ] Run H100 benchmarks with 1, 2, 4, and 8 devices once available.
- [ ] Document exact verification commands and benchmark configs in this file after implementation.

---

## Open Questions

- Should Part 3 run inside the distributed controller, or remain a standalone command that consumes merged Part 2 artifacts?
- What fill-row equality threshold is acceptable when CUDA top-K tie ordering differs across devices?
- Is replicated-index memory acceptable with `max_repr_seqs: 200000` on the target 8x H100 node?
- Should `neg_ctx_stats.json` become versioned before adding more backend-specific fields?

## Risks / Assumptions

- `neg_ctx` must be built from global merged `top_ctx`, `mid_ctx`, and `seq_repr`; building worker-local negatives before merge would change semantics.
- Replicated-index `multi_gpu_exact` is simple and exact, but may waste VRAM when `seq_repr` grows.
- CPU write-back can dominate if ANN search is already fast, limiting the benefit of multi-GPU query compute.
- Capped `seq_repr` must use the deterministic global cap from Part 2; per-worker caps would make negatives inconsistent.
- Downstream discovery assumes the canonical `neg_ctx.pt` shape and should not need backend-specific branches.
