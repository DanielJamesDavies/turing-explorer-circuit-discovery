# Plan: Part 6 - Distributed Candidate Selection And Discovery

> **Goal:** Select seed latents once from merged global artifacts, split discovery work across workers, and merge worker circuit outputs into the canonical `outputs/<run_id>/circuits/` artifacts.
>
> **Created:** 2026-05-16

---

## Scope

This part starts after Part 5 has produced the global discovery inputs:

- `outputs/<run_id>/latent_stats.pt`
- `outputs/<run_id>/top_ctx.pt`
- `outputs/<run_id>/mid_ctx.pt`
- `outputs/<run_id>/neg_ctx.pt`
- `outputs/<run_id>/logit_ctx.pt`
- `outputs/<run_id>/top_coactivation.pt`

It produces:

- `outputs/<run_id>/candidates.pt`
- worker-local circuit stores and summaries,
- merged `outputs/<run_id>/circuits/discovered_circuits.pt`,
- merged `outputs/<run_id>/circuits/summary.json`,
- optional merged `outputs/<run_id>/circuits/summary.xlsx`.

Candidate selection should remain centralized at first because it is cheap compared with gradient-enabled discovery. Discovery/eval should be distributed by seed or by seed-method task.

---

## Phase 1 - Central Candidate Selection

- [ ] Run `CandidateSelector` once over the merged global artifacts.
- [ ] Save canonical `outputs/<run_id>/candidates.pt` before any worker discovery starts.
- [ ] Include candidate-selection metadata: criteria, seed filter, config hash, artifact hashes, selected count, and per-candidate criterion scores.
- [ ] Keep existing candidate selection behavior unchanged for normal `src/main.py` runs.
- [ ] Add a standalone distributed-stage entrypoint that can load merged artifacts and write `outputs/<run_id>/candidates.pt` without launching discovery.
- [ ] Verification: add tests that centralized candidate selection writes the same candidate list in single-process and one-worker distributed modes.

## Phase 2 - Seed And Task Partitioning

- [ ] Use Part 1 assignment helpers to split selected candidates across workers deterministically.
- [ ] Preserve candidate order within each worker assignment.
- [ ] Support one-worker mode where all candidates stay on one worker.
- [ ] Record candidate assignment metadata in the manifest: worker ID, candidate indices, seed `(comp_idx, latent_idx)` pairs, method list, and estimated task count.
- [ ] Add optional method-aware task expansion for later balancing: `(candidate, method)` tasks instead of candidate-only chunks.
- [ ] Verification: add tests for one-worker, uneven candidate counts, more workers than candidates, deterministic order, and no duplicate/missing candidates.

## Phase 3 - Discovery Worker Entrypoint

- [ ] Add a worker entrypoint that loads global artifacts, assigned candidates, local `Inference`, local `SAEBank`, and `DataLoader`.
- [ ] Ensure each discovery worker uses one logical compute device and passes a single-device list to `SAEBank`.
- [ ] Allow each worker to run `DiscoveryWindow` with a worker-specific `output_dir`, for example `outputs/<run_id>/distributed/workers/worker_000/discovery/circuits/`.
- [ ] Ensure each worker saves its assigned candidate subset beside its circuit outputs for traceability.
- [ ] Preserve current discovery method configuration and post-circuit analyses inside each worker.
- [ ] Reset or isolate global singleton state such as `circuit_store` and observability counters per worker process.
- [ ] Verification: add a small mocked worker test that runs discovery over a tiny candidate list and writes a worker-local circuit store without touching canonical `outputs/<run_id>/circuits/`.

## Phase 4 - Artifact Loading And Store Contracts

- [ ] Ensure discovery workers load every required global store: `latent_stats`, `top_ctx`, `mid_ctx`, `neg_ctx`, `logit_ctx`, and `top_coactivation`.
- [ ] Prefer one shared loader utility for discovery-only runs so workers and existing `discover_circuits.py` do not drift.
- [ ] Validate artifact compatibility before model initialization where possible.
- [ ] Fail clearly when required stores are missing or incompatible with the current config.
- [ ] Keep worker output independent from canonical global stores until merge time.
- [ ] Verification: add tests for missing artifact errors, incompatible artifact metadata, and successful store load from synthetic fixtures.

## Phase 5 - Circuit Output Schema

- [ ] Define worker-local circuit artifact names: `discovered_circuits.pt`, `summary.json`, optional `summary.xlsx`, and `worker_discovery_stats.json`.
- [ ] Add worker/run metadata to each accepted circuit: run ID, worker ID, candidate index, seed identifiers, discovery method, config hash, and artifact hashes where practical.
- [ ] Preserve existing circuit object structure and metadata keys used by display/debug tooling.
- [ ] Avoid rewriting circuit UUIDs unless a merge conflict is detected.
- [ ] Save worker circuit stores atomically.
- [ ] Verification: add round-trip tests for worker circuit store save/load and metadata presence.

## Phase 6 - Circuit Store Merge

- [ ] Add a merge command that reads every completed worker circuit store.
- [ ] Append circuits into a fresh global `CircuitStore`.
- [ ] Detect UUID collisions; if any occur, either fail loudly or rewrite UUIDs with a recorded mapping.
- [ ] Preserve circuit metadata, eval metadata, post-analysis metadata, and seed-criteria metadata.
- [ ] Merge worker summaries into canonical `outputs/<run_id>/circuits/summary.json`.
- [ ] Regenerate `summary.xlsx` from the merged store if the existing summary writer is available.
- [ ] Verification: add tests merging multiple worker stores, empty worker stores, duplicate UUIDs, and mixed method outputs.

## Phase 7 - Cluster Contrast And Seed-Free Methods

- [ ] Treat seed-free methods such as `cluster_contrast` separately from seed-partitioned discovery.
- [ ] Decide whether seed-free methods run centrally once, on one designated worker, or in a separate distributed strategy.
- [ ] Prevent accidental duplicate `cluster_contrast` execution across workers.
- [ ] Record seed-free method ownership in the manifest.
- [ ] Keep seed-free method outputs mergeable with normal worker circuit stores.
- [ ] Verification: add tests proving `cluster_contrast` is not launched once per worker by default.

## Phase 8 - Scheduling And Load Balancing

- [ ] Start with candidate-level partitioning for simplicity.
- [ ] Add optional method-aware task partitioning once expensive methods create imbalance.
- [ ] Track per-task duration, forward-pass count, accepted circuit count, and peak VRAM where available.
- [ ] Add resume support for failed workers or failed task ranges.
- [ ] Add a scheduling report showing task distribution by method and worker.
- [ ] Verification: add tests for deterministic candidate-level scheduling and method-aware scheduling on synthetic task costs.

## Phase 9 - Local And H100 Modes

- [ ] Preserve local one-worker discovery for RTX 5070 Ti style runs.
- [ ] Allow local mode to keep current `probe_batch_size`, `neg_ctx_eval_max`, and efficient memory behavior.
- [ ] Allow H100 mode to run one discovery worker per GPU with replicated model+SAE resources.
- [ ] Ensure workers can be run manually from commands emitted by the controller before automatic process launch is required.
- [ ] Add dry-run estimates for candidate count, method count, and expected worker task counts.
- [ ] Verification: add one-worker dry-run tests and synthetic 8-worker assignment tests.

## Phase 10 - Result Validation And Reporting

- [ ] Validate merged circuit count equals the sum of worker circuit counts plus any designated seed-free method outputs.
- [ ] Validate every accepted circuit has seed metadata, discovery method metadata, and eval metadata where expected.
- [ ] Validate summary rows match the merged circuit store.
- [ ] Compare distributed one-worker outputs against existing single-process discovery outputs on a synthetic/mock setup.
- [ ] Add a merged discovery report with worker timings, accepted circuit counts, method counts, eval summary stats, and failed task ranges.
- [ ] Verification: add tests for merged report generation and summary consistency.

## Phase 11 - Testing And Verification

- [ ] Run focused tests for candidate selection, seed assignment, worker output schema, circuit-store merge, summary merge, and seed-free method ownership.
- [ ] Run mocked discovery worker tests without loading real model weights.
- [ ] Run worker resource-construction tests proving discovery workers cannot accidentally use a multi-device `SAEBank`.
- [ ] Run one-worker local smoke once artifacts are available.
- [ ] Run two-worker synthetic discovery/eval smoke before H100 use.
- [ ] Run H100 distributed discovery benchmark only after merge and resume behavior are tested.
- [ ] Document exact verification commands in this file after implementation.

---

## Open Questions

- Should worker discovery split by candidate first, or by `(candidate, method)` task from the start?
- Should `cluster_contrast` be disabled in distributed discovery until it has a dedicated execution plan?
- Should circuit UUID collisions be fatal, or should merge rewrite collisions with a provenance map?
- Should worker summaries be merged by reusing `DiscoveryWindow` summary code, or should a separate summary builder operate directly on `CircuitStore`?
- Which discovery metrics are required before marking a worker task completed?

## Risks / Assumptions

- Candidate selection must run on merged global artifacts; worker-local candidate selection would change the seed distribution.
- Discovery workers use global singleton stores today, so process isolation or explicit reset is important.
- Seed-free methods can accidentally duplicate work if treated like seed-based methods.
- Gradient-heavy methods may produce severe worker imbalance unless method-aware scheduling is added.
- The first implementation should prioritize deterministic assignment and clean circuit-store merge over maximal scheduling sophistication.
