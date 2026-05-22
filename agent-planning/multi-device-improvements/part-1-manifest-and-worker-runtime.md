# Plan: Part 1 - Manifest And Worker Runtime

> **Goal:** Add the distributed-run control layer needed to describe workers, assign deterministic work, validate partial outputs, and keep the current single-process pipeline as the correctness oracle.
>
> **Created:** 2026-05-16

---

## Scope

This part does not distribute pass 1, pass 2, or discovery yet. It creates the shared contract those later parts will use:

- a versioned run manifest,
- a canonical full-dataset global sequence ID table,
- deterministic assignment helpers,
- one-GPU worker device isolation,
- worker output layout,
- completion markers,
- JSONL metrics/report locations,
- cleanup and retention policy,
- validation utilities,
- one-worker compatibility with local/single-GPU runs.

The current `src/main.py` path remains unchanged and should continue to be the default local path.
However, its canonical artifacts should move to the same run-root convention as distributed runs: `outputs/<run_id>/`.

---

## Phase 1 - Manifest Schema And Contracts

- [x] Define a manifest data model with schema version, run ID, config hash, created timestamp, project/root paths, model path, SAE path, dataset path, output root, worker count, CUDA device assignments, cleanup policy, metrics paths, and artifact schema versions.
- [x] Use manifest schema version `1` for the first implementation and store it as `manifest_schema_version`.
- [x] Store schema versions for every durable distributed contract: manifest, partial artifacts, metrics JSONL, sanity reports, and run summaries.
- [x] Generate a run ID when not provided using timestamp plus normalized config hash: `YYYYMMDD-HHMMSS-<config_hash_8>`, for example `20260517-0025-a1b2c3d4`.
- [x] Include work assignments for pass-1 dataset shards, pass-2 sequence IDs, and discovery seed ranges, even if pass 2 and discovery assignments are initially empty.
- [x] Include run mode values such as `single_process`, `distributed_simple_exact`, `distributed_mapreduce_exact`, and `distributed_experimental_fast`.
- [x] Store enough config identity to detect stale manifests: config path, normalized config hash, relevant environment overrides, and git SHA when available.
- [x] Define manifest status values: `planned`, `running`, `completed`, `failed`, `partial`.
- [x] Define cleanup policy values: `keep_all`, `delete_large_partials_on_success`, `delete_all_partials_on_success`, and `manual_cleanup_only`.
- [x] Preserve failed-run partials, logs, metrics, and failure markers regardless of cleanup policy unless the user explicitly deletes them.
- [x] Use one run ID per whole run, not per device.
- [x] Require every run mode, including `single_process`, to have a run ID and canonical output root.
- [x] Use the run layout `outputs/<run_id>/distributed/manifest.json` for distributed metadata and partials.
- [x] Write validated canonical outputs at the top of the run root, for example `outputs/<run_id>/latent_stats.pt`.
- [x] Keep unvalidated distributed internals under `outputs/<run_id>/distributed/`.
- [x] Verification: add schema round-trip tests that save/load a tiny manifest and reject missing required fields, invalid modes, duplicate workers, and stale schema versions.

### Phase 1 Notes

- Added `src/pipeline/distributed/manifest.py` with strict Pydantic contracts for the run manifest, run modes, manifest statuses, cleanup policies, device assignments, work assignments, schema versions, and run-root path validation.
- Added `generate_run_id()` using `YYYYMMDD-HHMMSS-<config_hash_8>`.
- Added `save_manifest()` and `load_manifest()` JSON helpers.
- Added focused tests in `tests/pipeline/test_distributed_manifest.py`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py -q` -> `9 passed`.

## Phase 2 - Global Sequence ID Table

- [x] Build a canonical full-dataset shard table before any worker assignment.
- [x] Store for every shard: shard index, shard filename, sequence count, global start ID, and global end ID.
- [x] Assign global sequence IDs by prefix sum over actual per-shard sequence counts; do not assume every shard has the same number of sequences.
- [x] Allow the final shard, or any shard, to have fewer sequences than the others; its `[global_start, global_end)` range is simply smaller.
- [x] Derive pass-1 shard assignments, pass-2 sequence assignments, and display/search references from this one table.
- [x] Ensure workers can read only assigned shards while still emitting global sequence IDs from the full-dataset table.
- [x] Reject manifests when shard files, shard order, sequence counts, or cached shard indices differ from the table.
- [x] Verification: add synthetic multi-shard tests proving worker-local reads emit the same global sequence IDs as the single-process loader.
- [x] Verification: add tests where the last shard is shorter than earlier shards and global IDs still form contiguous, non-overlapping ranges.
- [x] Verification: add failure tests for missing shards, reordered shard files, stale shard index files, duplicated sequence IDs, and out-of-range assigned sequence IDs.

### Phase 2 Notes

- Added `src/pipeline/distributed/shard_table.py` with a strict `ShardRecord` model, canonical shard discovery, indexed sequence counting, half-open global ID ranges, table validation, and assigned-shard sequence ID helpers.
- Manifest validation now accepts a `shard_table` and rejects out-of-range pass-1 shard assignments, duplicated pass-2 sequence IDs, and pass-2 sequence IDs outside the canonical table.
- The distributed shard table mirrors `DataLoader` sequence validity rules while keeping the new half-open `[global_start_id, global_end_id)` contract for distributed code.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py -q` -> `20 passed`.

## Phase 3 - Worker Device Isolation

- [x] Assign each distributed worker exactly one physical compute device in the manifest.
- [x] Launch each worker so it sees one logical CUDA device, preferably via `CUDA_VISIBLE_DEVICES=<physical_id>`.
- [x] Record both physical device ID and worker-local logical device ID in worker metadata.
- [x] Record physical GPU identity when available: CUDA ordinal, GPU UUID, device name, PCI bus ID, total VRAM, and hostname.
- [x] Validate no two workers are assigned the same physical GPU unless an explicit oversubscription/debug mode is selected.
- [x] Ensure worker runtime passes a single-device list to `SAEBank`, for example `[torch.device("cuda:0")]` inside the worker process.
- [x] Ensure workers do not inherit root `hardware.multi_gpu: true` as permission to split one worker's SAE bank across multiple physical GPUs.
- [x] Support CPU/one-CUDA-device fallback for local one-worker validation.
- [x] Verification: add tests or dry-run assertions proving an 8-worker H100 manifest maps to eight one-device workers, not one worker using all eight devices.
- [x] Verification: add a unit test around worker resource construction that would fail if `SAEBank` receives more than one device in distributed worker mode.

### Phase 3 Notes

- Added `src/pipeline/distributed/devices.py` with device assignment creation, best-effort GPU identity recording, worker environment isolation, CPU fallback, and worker-local device helpers.
- Added strict `DeviceAssignment` validation so CUDA workers use worker-local `cuda:0` and CPU workers use `cpu`.
- Added `build_distributed_worker_runtime()` in `src/pipeline/runtime.py`; distributed workers are constructed with `multi_gpu=False` and exactly one worker-local device before `SAEBank` is initialized.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py -q` -> `27 passed`.

## Phase 4 - Deterministic Assignment Helpers

- [x] Add a pass-1 shard partitioner that assigns whole dataset shards by actual sequence count, not by simple shard count.
- [x] Use deterministic greedy balancing for pass-1 shards: sort selected shards by descending sequence count, then assign each shard to the worker with the lowest current assigned sequence total.
- [x] Tie-break pass-1 shard balancing deterministically by worker ID and shard ID.
- [x] Store each worker's assigned shard IDs and expected sequence total in the manifest.
- [x] Add a contiguous range/list partitioner for sequence-ID lists that distributes remainder items across the first workers deterministically; never drop remainder items.
- [x] Add a sequence partitioner that maps a global sequence-ID list to worker IDs deterministically while preserving stable order inside each worker.
- [x] Add a seed partitioner that maps selected discovery candidates or candidate indices to worker IDs deterministically.
- [x] Add device assignment helpers that support explicit CUDA IDs, visible-device validation, and a one-worker CPU/CUDA fallback path.
- [x] Ensure all assignment helpers are pure functions with no model loading or global store mutation.
- [x] Verification: add unit tests for sequence-balanced whole-shard assignment, uneven shard sizes, shorter final shards, empty inputs, one-worker inputs, more workers than items, duplicate sequence IDs, no-dropped-remainder partitioning, and invalid device IDs.

### Phase 4 Notes

- Added `src/pipeline/distributed/assignments.py` with pure deterministic helpers for pass-1 shard balancing, contiguous list partitioning, global sequence ID partitioning, discovery seed partitioning, and manifest-ready work assignment construction.
- `WorkAssignments` now stores `pass1_sequence_totals`, and manifest validation rejects mismatches between declared totals and assigned shard counts.
- `build_device_assignments()` now supports explicit visible-device validation via `visible_device_count` while preserving the CPU and one-CUDA-device fallback paths.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py -q` -> `37 passed`.

## Phase 5 - Worker Output Layout And Markers

- [x] Define the canonical run root: `outputs/<run_id>/`.
- [x] Define the distributed metadata/partials root: `outputs/<run_id>/distributed/`.
- [x] Do not create `outputs/latest` initially; keep the first implementation focused on `outputs/<run_id>/` as the only source of truth.
- [x] If any top-level `outputs/*.pt` compatibility path is added later, keep it as an alias/copy only, never as the source of truth.
- [x] Define per-worker directories: `outputs/<run_id>/distributed/workers/worker_000/`.
- [x] Define metrics/report paths: `outputs/<run_id>/distributed/reports/run_metrics.jsonl`, `outputs/<run_id>/distributed/reports/run_summary.json`, and `outputs/<run_id>/distributed/workers/worker_000/metrics.jsonl`.
- [x] Define part-specific subdirectories under each worker, such as `pass1/`, `pass2/`, and `discovery/`.
- [x] Define completion marker files with atomic write semantics, for example `started.json`, `completed.json`, and `failed.json`.
- [x] Include timing/resource summaries in worker metadata: start/end time, duration, device, shard IDs, sequence count, seed count, peak CPU RAM when available, and peak CUDA memory when available.
- [x] Define append-only JSONL metric event shape with run ID, worker ID, phase, timestamp, elapsed time where applicable, physical/logical device IDs, PID, hostname, artifact path/size where applicable, and counters such as sequence count.
- [x] Add validation that a worker cannot be marked completed unless required marker metadata and declared artifact files exist.
- [x] Verification: add tests for marker creation, atomic marker replacement, failed-worker metadata, missing artifact detection, metrics JSONL schema validation, cleanup policy classification, and unrelated-file preservation.

### Phase 5 Notes

- Added `src/pipeline/distributed/layout.py` with canonical run/worker layout builders, directory creation, versioned worker marker schema, versioned JSONL metric event schema, atomic marker writes, marker reads, completion validation, and cleanup candidate classification.
- `create_output_layout()` creates `outputs/<run_id>/distributed/`, `reports/`, `workers/worker_000/`, and per-worker `pass1/`, `pass2/`, and `discovery/` directories without creating `outputs/latest`.
- `validate_worker_completed()` requires completed worker timing metadata, `started.json`, and all declared required artifact files.
- Cleanup candidate selection preserves failed runs, supports `keep_all`, `manual_cleanup_only`, `delete_large_partials_on_success`, and `delete_all_partials_on_success`, and only scopes to distributed partial roots so unrelated/canonical files are preserved.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py -q` -> `47 passed`.

## Phase 6 - Controller Skeleton

- [x] Add a controller entrypoint or module that can create a manifest and output layout without launching real pipeline work.
- [x] Add preflight checks before manifest execution: output root writability, run ID collision unless `--resume`, strict config loading, normalized config hash computation, dataset shard table construction, selected device availability/uniqueness, CPU one-worker fallback viability, rough disk-space estimate, and native extension availability for selected parts.
- [x] Support a dry-run mode that prints planned workers, devices, shard assignments, output paths, and config hash.
- [x] In dry-run mode, print exact worker commands with per-worker `CUDA_VISIBLE_DEVICES=<physical_id>` values for manual debugging.
- [x] Add an optional Python `subprocess.Popen` launcher that runs those same worker commands after the dry-run command path is stable.
- [x] Keep the manifest and command contract scheduler-friendly so SLURM or another external launcher can be added later without changing worker semantics.
- [x] Support a one-worker mode that validates local/small-compute compatibility without requiring multi-GPU hardware.
- [x] Add resume planning logic that can classify workers as pending, completed, failed, or stale from their markers.
- [x] Verification: add tests for dry-run manifest creation, printed worker commands, subprocess launch planning, resume classification, stale config detection, and one-worker output layout.

### Phase 6 Notes

- Added `src/pipeline/distributed/controller.py` with `plan_distributed_run()`, strict config loading/hash generation, preflight checks, manifest/layout creation, scheduler-friendly worker command planning, dry-run text formatting, optional `subprocess.Popen` launching, and resume worker classification.
- Preflight covers output writability, run ID collision handling, device availability/uniqueness, CPU one-worker fallback, rough disk-space reporting, and selected native extension availability.
- Worker commands use the same manifest contract planned for external schedulers: `python -m pipeline.distributed.worker --manifest <manifest> --worker-id <id>` plus per-worker `CUDA_VISIBLE_DEVICES`.
- Resume planning classifies workers as `pending`, `completed`, `failed`, or `stale` from marker files and marks all workers stale when the current config hash differs.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py -q` -> `54 passed`.

## Phase 7 - Integration Boundaries

- [x] Keep existing `pipeline.run()` unchanged for normal `src/main.py` runs.
- [x] Update output-path resolution so existing pipeline code can write to a run root without changing artifact schemas.
- [x] Avoid loading model weights or SAE weights in controller-only code.
- [x] Add clear interfaces that later parts can call to get assigned shard IDs, sequence IDs, seed IDs, and worker output paths.
- [x] Document which later parts will consume each manifest field.
- [x] Ensure distributed paths do not assume H100 hardware; all controller and assignment tests should pass on CPU/local machines.
- [x] Verification: run focused tests for the new manifest/controller modules and confirm existing pipeline tests are not required to change.

### Phase 7 Notes

- Added `src/pipeline/distributed/interfaces.py` with run-root output path resolution, worker assignment accessors, worker output path accessors, and documented manifest field consumers for later parts.
- Kept `pipeline.run()` unchanged; normal `src/main.py` execution still calls the existing zero-argument orchestration path.
- Added optional `output_root` parameters to persistence, second-pass save, and candidate-selection save points. Defaults remain `outputs`, preserving current behavior while allowing later controller/worker code to target `outputs/<run_id>/`.
- Controller-only code remains planning/preflight-oriented and does not instantiate `Inference`, `SAEBank`, model weights, or SAE weights.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/test_persist_phase3.py -q` -> `66 passed`.

## Phase 8 - Testing And Verification

- [x] Add unit tests for manifest schema validation and JSON round trips.
- [x] Add unit tests for generated run IDs using timestamp plus 8-character config hash.
- [x] Add unit tests for manifest, partial artifact, metrics, sanity report, and run summary schema-version rejection.
- [x] Add unit tests for canonical global sequence ID table construction and stale-table rejection.
- [x] Add unit tests for variable per-shard sequence counts, including a shorter final shard.
- [x] Add unit tests for one-device worker isolation and `SAEBank` single-device construction in distributed worker mode.
- [x] Add unit tests for assignment determinism, coverage, sequence-balanced shard assignment, and no dropped remainder items.
- [x] Add unit tests for controller-emitted worker commands and per-worker `CUDA_VISIBLE_DEVICES` values.
- [x] Add unit tests for output directory and marker validation.
- [x] Add unit tests proving `single_process` output paths resolve under `outputs/<run_id>/`.
- [x] Add unit tests for physical/logical GPU metadata presence and duplicate physical GPU rejection.
- [x] Add unit tests for JSONL metrics event schema and cleanup/retention policy behavior.
- [x] Add unit tests for preflight checks: output writability, run ID collision/resume behavior, config validation/hash, shard table construction, device availability/uniqueness, CPU fallback, disk-space estimate, and native-extension availability gates.
- [x] Add a one-worker dry-run test that produces a manifest compatible with the current local RTX 5070 Ti style workflow.
- [x] Add a synthetic 8-worker dry-run test that produces stable H100-style assignments.
- [x] Run focused tests for new modules.
- [x] Run the existing lightweight suite subset most likely to catch import/config regressions.
- [x] Document the exact verification commands in this file after implementation.

### Phase 8 Notes

- Extended manifest tests to cover all durable schema-version rejection paths: manifest, metrics JSONL, sanity report, run summary, and artifact schema versions.
- Extended controller tests for one-worker CUDA-style dry-run planning, synthetic 8-worker H100-style assignment stability, shard-table construction in planning, disk-space reporting, native extension gates, run ID collision/resume behavior, and strict config rejection.
- Extended interface tests to prove `single_process`-style run-root paths resolve under `outputs/<run_id>/` without changing artifact names.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/test_persist_phase3.py tests/test_data_loader.py -q` -> `75 passed`.

---

## Open Questions

- Should config hashing include comments/formatting from `config.yaml`, or a normalized Pydantic dump only?
- Should worker IDs be stable by ordinal only, or should they also encode host/device for future multi-node use?
- Should completion markers be JSON only, or should they include small tensor/artifact summaries for quick validation?

## Risks / Assumptions

- The current single-process pipeline remains the correctness oracle and default local path.
- Global sequence IDs must remain stable across all worker assignments.
- Each distributed worker must be isolated to one logical compute device; accidental multi-GPU `SAEBank` placement inside a worker would invalidate the resource split.
- Manifest/controller code should avoid importing heavy model/SAE modules where possible.
- One-worker distributed mode must be treated as a first-class compatibility target, not an afterthought.
- Later parts depend on this contract, so schema versioning and validation need to be conservative from the start.
