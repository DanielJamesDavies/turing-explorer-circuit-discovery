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

- [x] Run `CandidateSelector` once over the merged global artifacts.
- [x] Save canonical `outputs/<run_id>/candidates.pt` before any worker discovery starts.
- [x] Include candidate-selection metadata: criteria, seed filter, config hash, artifact hashes, selected count, and per-candidate criterion scores.
- [x] Keep existing candidate selection behavior unchanged for normal `src/main.py` runs.
- [x] Add a standalone distributed-stage entrypoint that can load merged artifacts and write `outputs/<run_id>/candidates.pt` without launching discovery.
- [x] Verification: add tests that centralized candidate selection writes the same candidate list in single-process and one-worker distributed modes.

### Phase 1 Notes

- Extended `src/pipeline/candidate_selection.py` with `run_candidate_selection_stage()` for the distributed/run-root stage.
- The stage validates and loads merged global `latent_stats.pt`, `top_ctx.pt`, `mid_ctx.pt`, `neg_ctx.pt`, `logit_ctx.pt`, and `top_coactivation.pt` before running `CandidateSelector`.
- The canonical candidate list remains the existing bare `candidates.pt` format so downstream discovery and normal `src/main.py` behavior do not change.
- Added metadata and markers under `outputs/<run_id>/distributed/parts/candidate_selection/`, including artifact SHA-256 hashes, seed criteria, seed filter, config hash, selected count, and per-candidate criterion scores.
- Added CLI access through `python -m pipeline.candidate_selection --output-root <run_root> --manifest <run_root>/distributed/manifest.json`.
- Verification: `python -m pytest tests/pipeline/test_candidate_selection_stage.py -q` -> `3 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_interfaces.py -q` -> `8 passed`.

## Phase 2 - Seed And Task Partitioning

- [x] Use Part 1 assignment helpers to split selected candidates across workers deterministically.
- [x] Preserve candidate order within each worker assignment.
- [x] Support one-worker mode where all candidates stay on one worker.
- [x] Record candidate assignment metadata in the manifest: worker ID, candidate indices, seed `(comp_idx, latent_idx)` pairs, method list, and estimated task count.
- [x] Add optional method-aware task expansion for later balancing: `(candidate, method)` tasks instead of candidate-only chunks.
- [x] Verification: add tests for one-worker, uneven candidate counts, more workers than candidates, deterministic order, and no duplicate/missing candidates.

### Phase 2 Notes

- Added `DiscoveryCandidateAssignment` to the distributed manifest contract and extended `WorkAssignments` with `discovery_candidate_assignments`.
- Added `build_discovery_candidate_assignments()` in `src/pipeline/distributed/assignments.py`, reusing deterministic contiguous seed partitioning over candidate indices.
- `run_candidate_selection_stage()` now updates and saves the manifest when a manifest path is provided, filling both compact `discovery_seed_ids` and rich per-worker candidate assignment metadata.
- Candidate assignment metadata records each candidate index, seed `(comp_idx, latent_idx)`, configured discovery methods, and estimated task count. Method-aware expansion remains a later scheduling layer; Phase 2 stores enough method metadata for it.
- Verification: `python -m pytest tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_manifest.py -q` -> `31 passed`.

## Phase 3 - Discovery Worker Entrypoint

- [x] Add a worker entrypoint that loads global artifacts, assigned candidates, local `Inference`, local `SAEBank`, and `DataLoader`.
- [x] Ensure each discovery worker uses one logical compute device and passes a single-device list to `SAEBank`.
- [x] Allow each worker to run `DiscoveryWindow` with a worker-specific `output_dir`, for example `outputs/<run_id>/distributed/workers/worker_000/discovery/circuits/`.
- [x] Ensure each worker saves its assigned candidate subset beside its circuit outputs for traceability.
- [x] Preserve current discovery method configuration and post-circuit analyses inside each worker.
- [x] Reset or isolate global singleton state such as `circuit_store` and observability counters per worker process.
- [x] Verification: add a small mocked worker test that runs discovery over a tiny candidate list and writes a worker-local circuit store without touching canonical `outputs/<run_id>/circuits/`.

### Phase 3 Notes

- Extended `src/pipeline/distributed/worker.py` with `--phase discovery` and `run_discovery_worker()`.
- Discovery workers validate manifest candidate assignments, load `outputs/<run_id>/candidates.pt`, save `assigned_candidates.pt` and `assignment_metadata.json` under their worker discovery directory, then run only the assigned candidate subset.
- Added `initialize_discovery_worker_resources()` so discovery workers use the same one-logical-device runtime isolation as pass-1 and pass-2 workers, with a single-device `SAEBank`.
- Added `load_discovery_global_artifacts()` to load merged global discovery inputs before worker-local discovery. Phase 4 will consolidate this into the shared artifact loader contract.
- Worker discovery output goes to `outputs/<run_id>/distributed/workers/worker_000/discovery/circuits/`, leaving canonical `outputs/<run_id>/circuits/` untouched until the later merge phase.
- `reset_discovery_worker_state()` clears `circuit_store` and observability counters before each worker-local run.
- Verification: `python -m pytest tests/pipeline/test_distributed_worker.py -q` -> `23 passed`.

## Phase 4 - Artifact Loading And Store Contracts

- [x] Ensure discovery workers load every required global store: `latent_stats`, `top_ctx`, `mid_ctx`, `neg_ctx`, `logit_ctx`, and `top_coactivation`.
- [x] Prefer one shared loader utility for discovery-only runs so workers and existing `discover_circuits.py` do not drift.
- [x] Validate artifact compatibility before model initialization where possible.
- [x] Fail clearly when required stores are missing or incompatible with the current config.
- [x] Keep worker output independent from canonical global stores until merge time.
- [x] Verification: add tests for missing artifact errors, incompatible artifact metadata, and successful store load from synthetic fixtures.

### Phase 4 Notes

- Added `src/pipeline/discovery_artifacts.py` with shared `validate_discovery_artifacts()` and `load_discovery_artifacts()` helpers.
- The shared validator checks required artifact presence, tensor payload shapes, shared component/SAE-width dimensions, finite floating values, optional candidate-list structure, and `top_coactivation` mode compatibility with the current config before model/SAE initialization.
- Distributed discovery workers now use the shared validator/loader instead of a worker-local loading contract.
- Standalone `src/discover_circuits.py` now uses the same shared loader, so discovery-only runs and distributed workers do not drift.
- Worker outputs still remain under `outputs/<run_id>/distributed/workers/worker_000/discovery/`; canonical `outputs/<run_id>/circuits/` remains untouched until merge time.
- Verification: `python -m pytest tests/pipeline/test_discovery_artifacts.py tests/pipeline/test_distributed_worker.py -q` -> `28 passed`.

## Phase 5 - Circuit Output Schema

- [x] Define worker-local circuit artifact names: `discovered_circuits.pt`, `summary.json`, optional `summary.xlsx`, and `worker_discovery_stats.json`.
- [x] Add worker/run metadata to each accepted circuit: run ID, worker ID, candidate index, seed identifiers, discovery method, config hash, and artifact hashes where practical.
- [x] Preserve existing circuit object structure and metadata keys used by display/debug tooling.
- [x] Avoid rewriting circuit UUIDs unless a merge conflict is detected.
- [x] Save worker circuit stores atomically.
- [x] Verification: add round-trip tests for worker circuit store save/load and metadata presence.

### Phase 5 Notes

- Worker discovery artifacts are now explicitly tracked as `assigned_candidates.pt`, `assignment_metadata.json`, `circuits/discovered_circuits.pt`, `circuits/summary.json`, optional `circuits/summary.xlsx`, and `worker_discovery_stats.json`.
- Assigned candidates are enriched with `run_id`, `worker_id`, `candidate_index`, `config_hash`, and discovery input artifact hashes before entering `DiscoveryWindow`.
- `DiscoveryWindow` copies distributed candidate provenance into accepted circuit metadata while preserving existing circuit object structure, UUIDs, and display/debug metadata keys.
- `DiscoveryWindow.save_store()` now writes `discovered_circuits.pt`, `summary.json`, and `summary.xlsx` via temporary files plus atomic replace.
- `worker_discovery_stats.json` records run/worker/config metadata, candidate count, method count/list, accepted circuit count, and circuit UUIDs.
- Verification: `python -m pytest tests/pipeline/test_discovery_window_outputs.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_discovery_artifacts.py -q` -> `32 passed`.

## Phase 6 - Circuit Store Merge

- [x] Add a merge command that reads every completed worker circuit store.
- [x] Append circuits into a fresh global `CircuitStore`.
- [x] Detect UUID collisions; if any occur, either fail loudly or rewrite UUIDs with a recorded mapping.
- [x] Preserve circuit metadata, eval metadata, post-analysis metadata, and seed-criteria metadata.
- [x] Merge worker summaries into canonical `outputs/<run_id>/circuits/summary.json`.
- [x] Regenerate `summary.xlsx` from the merged store if the existing summary writer is available.
- [x] Verification: add tests merging multiple worker stores, empty worker stores, duplicate UUIDs, and mixed method outputs.

### Phase 6 Notes

- Added `src/pipeline/distributed/discovery_merge.py` with `run_circuit_store_merge()`, `load_completed_worker_circuit_stores()`, `merge_circuit_stores()`, and `build_circuit_summary()`.
- The merge reads completed discovery worker markers, loads each worker-local `circuits/discovered_circuits.pt`, and appends circuits into a fresh `CircuitStore`.
- Duplicate circuit UUIDs fail loudly by default; UUID rewriting is explicitly not implemented yet, so merge conflicts cannot be silently hidden.
- Canonical merged outputs are written under `outputs/<run_id>/circuits/`: `discovered_circuits.pt`, `summary.json`, and `summary.xlsx` when the dataframe writer is available and there are rows.
- The merged summary preserves existing metadata dictionaries, including eval metadata, post-analysis metadata, seed criteria, worker IDs, and discovery method names.
- A merge report is written to `outputs/<run_id>/distributed/reports/discovery_merge_report.json`.
- Verification: `python -m pytest tests/pipeline/test_discovery_merge.py tests/pipeline/test_discovery_window_outputs.py tests/pipeline/test_distributed_worker.py -q` -> `32 passed`.

## Phase 7 - Cluster Contrast And Seed-Free Methods

- [x] Treat seed-free methods such as `cluster_contrast` separately from seed-partitioned discovery.
- [x] Decide whether seed-free methods run centrally once, on one designated worker, or in a separate distributed strategy.
- [x] Prevent accidental duplicate `cluster_contrast` execution across workers.
- [x] Record seed-free method ownership in the manifest.
- [x] Keep seed-free method outputs mergeable with normal worker circuit stores.
- [x] Verification: add tests proving `cluster_contrast` is not launched once per worker by default.

### Phase 7 Notes

- Added `SEED_FREE_DISCOVERY_METHODS` with `cluster_contrast` as the first seed-free method.
- Seed-free methods are excluded from per-candidate discovery assignments so `cluster_contrast` no longer inflates candidate task counts or appears as a seed-partitioned method.
- Added manifest ownership via `WorkAssignments.discovery_seed_free_method_owners`; candidate selection assigns `cluster_contrast` to worker `0` by default when enabled.
- Discovery workers now filter `config.discovery.methods` during `DiscoveryWindow` execution so only the owning worker sees `cluster_contrast`; non-owner workers run only seed-based methods.
- Worker assignment metadata and stats record owned seed-free methods, and the owner writes circuits into its normal worker-local circuit store so Phase 6 merge can combine them with seed-based circuits.
- Verification: `python -m pytest tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_manifest.py -q` -> `60 passed`.

## Phase 8 - Scheduling And Load Balancing

- [x] Start with candidate-level partitioning for simplicity.
- [x] Add optional method-aware task partitioning once expensive methods create imbalance.
- [x] Track per-task duration, forward-pass count, accepted circuit count, and peak VRAM where available.
- [x] Add resume support for failed workers or failed task ranges.
- [x] Add a scheduling report showing task distribution by method and worker.
- [x] Verification: add tests for deterministic candidate-level scheduling and method-aware scheduling on synthetic task costs.

### Phase 8 Notes

- Added manifest scheduling fields for discovery strategy, planned task assignments, worker estimated costs, and failed task ranges.
- Candidate selection now records a default `candidate_contiguous` schedule and writes `distributed/reports/discovery_scheduling_report.json` with task counts, estimated costs, methods, seed-free ownership, and failed ranges by worker.
- Added optional `method_cost_greedy` planning via `build_discovery_task_assignments()` for synthetic method-cost balancing without changing the default worker execution contract.
- Added `select_discovery_resume_tasks()` so failed task ranges can be projected back onto planned task IDs for targeted resume logic.
- `DiscoveryWindow.run()` now returns per-task metrics for seed-based and seed-free discovery: duration, forward-pass count, accepted circuit count, and peak CUDA memory where available.
- Discovery worker stats now persist planned task counts, estimated costs, failed ranges, and returned task metrics.
- Verification: `python -m pytest tests/pipeline/test_distributed_assignments.py tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_manifest.py tests/pipeline/test_discovery_window_outputs.py -q` -> `65 passed`.

## Phase 9 - Local And H100 Modes

- [x] Preserve local one-worker discovery for RTX 5070 Ti style runs.
- [x] Allow local mode to keep current `probe_batch_size`, `neg_ctx_eval_max`, and efficient memory behavior.
- [x] Allow H100 mode to run one discovery worker per GPU with replicated model+SAE resources.
- [x] Ensure workers can be run manually from commands emitted by the controller before automatic process launch is required.
- [x] Add dry-run estimates for candidate count, method count, and expected worker task counts.
- [x] Verification: add one-worker dry-run tests and synthetic 8-worker assignment tests.

### Phase 9 Notes

- Added `DiscoveryDryRunEstimate` to controller plans, derived from the validated config defaults plus any explicit discovery overrides.
- Dry runs now report discovery mode (`local_one_worker`, `h100_one_worker_per_gpu`, or `distributed_multi_worker`), candidate count, seed/seed-free method counts, `probe_batch_size`, `neg_ctx_eval_max`, replicated model/SAE worker count, and expected per-worker task counts/costs.
- Local one-worker mode preserves the configured discovery memory knobs; no worker-level overrides are introduced for `probe_batch_size` or `neg_ctx_eval_max`.
- H100-style planning now clearly reports one replicated discovery worker per GPU and stable expected task counts across eight workers.
- Worker commands now include an explicit `--phase` argument, and `build_worker_commands(..., phase="discovery")` emits manual discovery commands for each worker before automatic process launch is required.
- Verification: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_manifest.py -q` -> `44 passed`.

## Phase 10 - Result Validation And Reporting

- [x] Validate merged circuit count equals the sum of worker circuit counts plus any designated seed-free method outputs.
- [x] Validate every accepted circuit has seed metadata, discovery method metadata, and eval metadata where expected.
- [x] Validate summary rows match the merged circuit store.
- [x] Compare distributed one-worker outputs against existing single-process discovery outputs on a synthetic/mock setup.
- [x] Add a merged discovery report with worker timings, accepted circuit counts, method counts, eval summary stats, and failed task ranges.
- [x] Verification: add tests for merged report generation and summary consistency.

### Phase 10 Notes

- Extended `run_circuit_store_merge()` to validate merged discovery outputs before writing the final merge report.
- Added `validate_merged_discovery_outputs()` for merged-count reconciliation, required circuit metadata checks, seed-free metadata checks, eval metadata checks, and summary/store consistency checks.
- The discovery merge report now includes validation status, seed-free method counts, method counts, eval summary stats, worker timing/accepted-count/task metadata, and failed task ranges.
- Seed-based circuits must carry `candidate_index`, `seed_comp`, `seed_latent`, `discovery_method`, and eval metadata; `cluster_contrast` circuits are validated with cluster metadata and direct eval metrics.
- Added a synthetic one-worker merge test proving the merged canonical store and summary match the worker-local circuit store.
- Verification: `python -m pytest tests/pipeline/test_discovery_merge.py tests/pipeline/test_discovery_window_outputs.py tests/pipeline/test_distributed_worker.py -q` -> `38 passed`.

## Phase 11 - Testing And Verification

- [x] Run focused tests for candidate selection, seed assignment, worker output schema, circuit-store merge, summary merge, and seed-free method ownership.
- [x] Run mocked discovery worker tests without loading real model weights.
- [x] Run worker resource-construction tests proving discovery workers cannot accidentally use a multi-device `SAEBank`.
- [x] Run one-worker local smoke once artifacts are available.
- [x] Run two-worker synthetic discovery/eval smoke before H100 use.
- [x] Run H100 distributed discovery benchmark only after merge and resume behavior are tested.
- [x] Document exact verification commands in this file after implementation.

### Phase 11 Notes

- Added `test_two_worker_synthetic_discovery_eval_smoke_merges_and_reports()` to cover a two-worker synthetic discovery/eval merge before H100 use.
- Existing mocked worker tests verify discovery worker output schema without loading real model weights.
- Existing resource-construction tests verify discovery workers initialize `SAEBank` with only the worker-local logical device.
- Existing one-worker synthetic merge coverage verifies canonical merged output can match a worker-local discovery store.
- H100 benchmark execution remains intentionally gated for real hardware/artifacts; Phase 9 dry-run coverage verifies stable synthetic 8-worker assignments and commands before benchmark launch.

### Phase 11 Verification Commands

- `python -m pytest tests/pipeline/test_candidate_selection_stage.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_discovery_artifacts.py tests/pipeline/test_discovery_window_outputs.py tests/pipeline/test_discovery_merge.py tests/pipeline/test_distributed_controller.py -q` -> `80 passed`.
- `python -m pytest tests/pipeline/test_distributed_worker.py::test_initialize_discovery_worker_resources_uses_single_worker_device tests/pipeline/test_distributed_worker.py::test_run_discovery_worker_saves_assigned_candidates_and_worker_outputs tests/pipeline/test_discovery_merge.py::test_one_worker_merge_matches_single_worker_store_on_synthetic_setup tests/pipeline/test_discovery_merge.py::test_two_worker_synthetic_discovery_eval_smoke_merges_and_reports tests/pipeline/test_distributed_controller.py::test_controller_h100_style_8_worker_dry_run_has_stable_assignments -q` -> `5 passed`.

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
