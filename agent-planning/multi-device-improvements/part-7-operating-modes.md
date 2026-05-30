# Plan: Part 7 - Operating Modes

> **Goal:** Define clear runtime modes for local, exact distributed, MapReduce distributed, and experimental fast paths so the same codebase can run safely on both local GPUs and 8x H100 nodes.
>
> **Created:** 2026-05-16

---

## Scope

This part defines how users choose and run the multi-device system once Parts 1-6 exist.

It covers:

- runtime mode names and semantics,
- config/CLI surface,
- compatibility with the existing `src/main.py` path,
- local one-worker behavior,
- H100 exact distributed behavior,
- MapReduce rollout gates,
- experimental fast-mode guardrails,
- mode-specific validation and documentation.

It does not implement the distributed worker internals from Parts 1-6.

---

## Phase 1 - Mode Taxonomy

- [x] Define `single_process` as the current pipeline and permanent correctness oracle.
- [x] Define `distributed_simple_exact` as distributed pass 1/pass 2/discovery with simple exact pass-2 candidate-dump merge.
- [x] Define `distributed_mapreduce_exact` as distributed pass 1/pass 2/discovery with exact pass-2 partial-sum shuffle and target-range reducers.
- [x] Define `distributed_experimental_fast` as opt-in approximate or quality-changing modes only after exact baselines are benchmarked.
- [x] Define `dry_run` behavior for every distributed mode so assignments and artifact paths can be inspected without model loading.
- [x] Use `outputs/<run_id>/` as the canonical artifact root for every mode, including `single_process`.
- [x] Generate default run IDs as `YYYYMMDD-HHMMSS-<config_hash_8>` when the user does not provide one.
- [x] Verification: add tests that every mode has a documented name, description, required parts, and allowed entrypoints.

### Phase 1 Notes

- Added `src/pipeline/distributed/operating_modes.py` with a static `OperatingModeDefinition` taxonomy for all `RunMode` values.
- `single_process` is documented as the `single_process_oracle` with `python src/main.py` as its current entrypoint.
- `distributed_simple_exact` is documented as the exact Parts 1-6 path using the simple exact pass-2 reducer.
- `distributed_mapreduce_exact` is documented as the exact Parts 1-6 path that additionally requires Part 5 Mode B target-range MapReduce reducers and equivalence against simple exact.
- `distributed_experimental_fast` is documented as non-exact, opt-in, and requiring explicit acknowledgement plus exact baseline artifacts.
- Added exported helpers for canonical run-root resolution and default run-root generation via the existing `generate_run_id()` contract.
- Verification: `python -m pytest tests/pipeline/test_distributed_operating_modes.py -q`.

## Phase 2 - Config Surface

- [x] Add a distributed/runtime config section without changing defaults for existing local runs.
- [x] Use `distributed` as the config namespace for orchestration settings.
- [x] Include fields for mode, run ID/output root, worker count, devices, launch strategy, resume policy, cleanup policy, part selection, and strict equivalence checks.
- [x] Include schema-version fields for manifest, partial artifacts, metrics JSONL, sanity reports, and run summaries.
- [x] Use strict Pydantic config validation, for example `extra='forbid'`, so unknown or misspelled distributed config keys fail early.
- [x] Include an explicit `distributed.sampling_seed`; store it in the manifest and derive artifact-specific seeds for `seq_repr` and `mid_ctx`.
- [x] Include `distributed.mid_ctx_candidate_pool` settings: `enabled`, `band_margin_sigma`, `max_candidates_per_latent`, and `on_truncation`.
- [x] Default `distributed.mid_ctx_candidate_pool.enabled: true`, `band_margin_sigma: 1.0`, `max_candidates_per_latent: max(256, 4 * num_ctx_sequences)`, and `on_truncation: replay_fallback` for exact/paper modes.
- [x] Restrict `distributed.mid_ctx_candidate_pool.on_truncation` to `fail`, `replay_fallback`, or `allow_bounded_approx`; paper-eligible modes must use `fail` or `replay_fallback`.
- [x] Keep distributed search-cache generation offline by default, for example `persist.build_search_cache_after_pipeline: false` during distributed benchmark/paper runs.
- [x] Use `outputs/<run_id>/distributed/` for distributed manifests, worker partials, part status, and reports.
- [x] Write validated canonical outputs directly at the top of `outputs/<run_id>/` only after exactness and sanity checks pass.
- [x] Keep `single_process` as the default when no distributed config is present, while still assigning a run ID and writing canonical outputs under `outputs/<run_id>/`.
- [x] Validate that `distributed_mapreduce_exact` cannot be selected until required MapReduce schemas and tests exist.
- [x] Validate that `distributed_experimental_fast` requires explicit acknowledgement in config or CLI.
- [x] Verification: add config validation tests for default local config, one-worker config, H100 simple exact config, MapReduce gated config, and invalid combinations.

### Phase 2 Notes

- Expanded `DistributedConfig` in `src/config.py` with `mode`, `run_id`, `output_base`, `worker_count`, `devices`, `launch_strategy`, `resume_policy`, `cleanup_policy`, `parts`, `strict_equivalence`, `experimental_acknowledgement`, and schema-version fields.
- Kept existing local defaults unchanged: absent `distributed` config still resolves to `single_process`, one worker, `outputs`, manual command launch, `keep_all`, and current search-cache defaults.
- Added strict validation for supported modes, run ID shape, worker/device counts, launch/resume/cleanup policies, unique devices, exact-mode `mid_ctx` truncation policy, and experimental acknowledgement.
- Distributed modes now default `persist.build_search_cache_after_pipeline` to `false` unless the user explicitly sets it, keeping search-cache generation off the critical path for distributed validation and benchmark runs.
- `distributed_mapreduce_exact` is accepted because Part 5 Mode B schemas/tests now exist; it remains an exact mode that must be proven against simple exact before recommendation.
- The canonical run-root and `outputs/<run_id>/distributed/` policy is enforced by the manifest/layout contracts from Parts 1 and 7 Phase 1; Phase 2 exposes the config fields that select those paths.
- Verification: `python -m pytest tests/pipeline/test_distributed_config.py -q` -> `8 passed`.

## Phase 3 - Entrypoints And Commands

- [x] Keep `python src/main.py` mapped to the current `single_process` pipeline.
- [x] Add `--run-id` support, or an equivalent config field, for `single_process` so local and distributed outputs use the same run-root layout.
- [x] Add a distributed controller command for creating manifests and running or printing worker commands.
- [x] Add part-specific commands for running individual stages from existing artifacts, for example pass-1 worker, pass-1 merge, neg_ctx, pass-2 dump, pass-2 reduce, candidate/discovery.
- [x] Support a dry-run command that prints exact per-worker commands, including `CUDA_VISIBLE_DEVICES=<physical_id>`, for manual debugging.
- [x] Support an optional built-in Python `subprocess.Popen` launcher that runs the same worker commands after dry-run command generation is stable.
- [x] Keep external scheduler support as a later integration that consumes the same manifest and worker command contract.
- [x] Support `--dry-run`, `--resume`, and `--run-id` consistently across distributed commands.
- [x] Print a concise mode summary before work starts: mode, workers, devices, output root, selected parts, and exactness guarantees.
- [x] Verification: add CLI/help tests or parser tests that cover every command without loading model weights.

### Phase 3 Notes

- Added `python -m pipeline.distributed.controller` with a model-free CLI for manifest planning, dry-run output, `--run-id`, `--resume`, selected mode/parts, worker count/devices, emitted worker phase commands, and optional `subprocess.Popen` launch via `--launch`.
- The controller CLI reads Phase 2 `distributed` config defaults and lets explicit CLI flags override them.
- Worker commands continue to use the scheduler-friendly contract `python -m pipeline.distributed.worker --manifest <manifest> --phase <pass1|pass2|discovery> --worker-id <id>` with per-worker `CUDA_VISIBLE_DEVICES`.
- Added standalone CLI parsers for `python -m pipeline.distributed.pass1_merge` and `python -m pipeline.distributed.pass2_reduce` so merge/reduce stages can run from existing artifacts without loading model weights.
- Existing `python -m pipeline.negative_context` already supports `--dry-run`, `--resume`, `--manifest`, and run-root `--output-root`; existing `python -m pipeline.candidate_selection` supports run-root candidate selection with an optional manifest.
- `python src/main.py` remains the current `single_process` entrypoint; the Phase 2 `distributed.run_id` field is the run-ID config surface that later phases can wire into the local run-root execution path.
- Verification: `python -m pytest tests/pipeline/test_distributed_controller.py -q` -> `19 passed`.

## Phase 4 - Local Compatibility Mode

- [x] Support one-worker `distributed_simple_exact` on CPU or one CUDA device for local validation.
- [x] Ensure local mode can use `hardware.memory: "efficient"`, `keep_model_loaded_for_neg_ctx: false`, deferred search cache, small `n_shards`, and small `n_seeds`.
- [x] Ensure local mode does not require H100-only configs, multi-GPU devices, or replicated workers.
- [x] Keep search-cache generation deferred/offline in local distributed validation unless the user explicitly enables it.
- [x] Document when to use `single_process` versus one-worker distributed mode locally.
- [x] Do not add `outputs/latest` initially; document `outputs/<run_id>/` as the source of truth for local and distributed runs.
- [x] Add a local example config for one-worker distributed validation if useful.
- [x] Verification: add dry-run tests for RTX 5070 Ti style settings and one-worker distributed assignments.

### Phase 4 Notes

- Added `LocalCompatibilityReport` and `build_local_compatibility_report()` in `src/pipeline/distributed/controller.py` so controller dry-runs explicitly report local one-worker compatibility without model loading.
- One-worker dry-runs now print a `local compatibility:` block with device mode (`cpu`, `single_cuda`, or `auto`), `h100_required`, memory mode, `keep_model_loaded_for_neg_ctx`, search-cache deferral, shard count, seed count, `probe_batch_size`, and `neg_ctx_eval_max`.
- CPU local validation is supported with `--use-cpu`; single-CUDA validation is supported with one declared device, for example `distributed.devices: [0]` or `--devices 0`.
- Added `config_examples/local-distributed-smoke.yaml` as a local one-worker distributed validation config using efficient memory, one device, 4 shards, 16 seeds, deferred search-cache generation, and manual worker commands.
- Existing `single_process` remains the default local path for normal development; use one-worker `distributed_simple_exact` locally when validating manifest, worker, partial-artifact, and merge contracts before multi-worker H100 execution.
- No `outputs/latest` alias was added. The controller continues to create only `outputs/<run_id>/` and distributed internals under `outputs/<run_id>/distributed/`.
- Verification: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_config.py -q` -> `29 passed`.

## Phase 5 - H100 Exact Modes

- [x] Define a recommended H100 `distributed_simple_exact` config: one worker per GPU, replicated model+SAE, global artifact merges, simple exact pass-2 reduce.
- [x] Require H100 worker launch to isolate devices so each worker sees one logical `cuda:0` and loads a full local model+SAE bank.
- [x] Use manifest-declared devices for distributed device-consuming phases by default, including `neg_ctx`; all visible devices may remain a standalone/single-process convenience.
- [x] Define the entry criteria for H100 `distributed_mapreduce_exact`: simple exact merge benchmark shows central dump merge or reducer input memory is a bottleneck.
- [x] Keep current `config_examples/h100-8x.yaml` clearly labeled as current-runtime/single-process, not the future distributed runtime.
- [x] Add a future distributed H100 config only after Parts 1-6 have working exact paths.
- [x] Document which parts use GPUs, CPU/OpenMP, disk I/O, and centralized merges in each exact mode.
- [x] Verification: add dry-run tests for synthetic 8-worker H100 assignments and expected output layout.

### Phase 5 Notes

- Added `H100ExactModeReport` and `build_h100_exact_mode_report()` in `src/pipeline/distributed/controller.py` so 8-worker exact dry-runs print an `h100 exact mode:` block without model loading.
- H100 exact dry-runs now report one worker per physical GPU, worker-local logical device `cuda:0`, manifest-declared physical devices, replicated model+SAE worker count, `neg_ctx` device source, pass-2 reducer strategy, GPU phases, and CPU/I/O phases.
- `distributed_simple_exact` reports `simple_exact_candidate_dump_reduce:<reduce_backend>` and remains the recommended first H100 mode.
- `distributed_mapreduce_exact` reports `mapreduce_target_range_reduce` and is gated by the criterion: enable only after simple exact benchmarks show candidate-dump merge or reducer input memory is a bottleneck.
- Added `config_examples/h100-8x-distributed-simple-exact.yaml` as the future distributed H100 example: `worker_count: 8`, devices `[0..7]`, one logical CUDA device per worker, manifest-declared devices for distributed stages, simple exact pass-2 reduce, and deferred search-cache generation.
- Kept `config_examples/h100-8x.yaml` as the current-runtime/single-process benchmark config; its header already states it is not the future distributed controller/worker runtime.
- Verification: `python -m pytest tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_config.py -q` -> `31 passed`.

## Phase 6 - Rollout Gates

- [x] Require `distributed_simple_exact` to pass one-worker equivalence before two-worker or eight-worker use.
- [x] Require `distributed_simple_exact` to pass tiny synthetic and reduced real-data equivalence before paper-facing runs.
- [x] Require `distributed_mapreduce_exact` to match `distributed_simple_exact` before it can be recommended.
- [x] Require every mode to write a manifest, sanity reports, and verification status.
- [x] Require benchmark results before changing recommended defaults.
- [x] Verification: add mode gate tests that reject unsafe transitions, stale manifests, missing equivalence reports, and missing sanity reports.

### Phase 6 Notes

- Added `src/pipeline/distributed/rollout_gates.py` with `validate_rollout_gates()` and `write_rollout_gate_report()`.
- Rollout gates validate that the manifest exists, loads successfully, matches the requested run/config hash, and is not stale against an optional current config hash.
- Every gated run requires `distributed/reports/verification_status.json` plus sanity/report artifacts. By default the required sanity set is `pass1_sanity_report.json`, `neg_ctx_sanity_report.json`, `pass2_reduce_report.json`, and `discovery_merge_report.json`; callers can override the set for partial-stage validation.
- Multi-worker `distributed_simple_exact` requires `equivalence_one_worker.json` before use.
- Paper-facing `distributed_simple_exact` additionally requires `equivalence_tiny_synthetic.json` and `equivalence_reduced_real.json`.
- `distributed_mapreduce_exact` requires `equivalence_mapreduce_vs_simple.json` before it can be trusted or recommended.
- Any mode being promoted as a recommended default requires `benchmark_report.json`.
- Gate reports accept existing report shapes with `status: passed/completed/ok`, top-level `ok: true`, `validation.ok: true`, `equivalence.passed: true`, or `benchmark.completed: true`.
- Verification: `python -m pytest tests/pipeline/test_distributed_rollout_gates.py -q` -> `7 passed`.

## Phase 6.5 - Preflight Checks

- [x] Run preflight checks before expensive work starts.
- [x] Check output root writability and reject existing `outputs/<run_id>/` unless `--resume` is selected.
- [x] Load config with strict validation and compute the normalized config hash before writing the manifest.
- [x] Build the dataset shard table and fail if shard counts/order are unavailable or stale.
- [x] Validate selected devices exist, are unique, and match manifest-declared device policy.
- [x] Confirm CPU one-worker fallback remains available for local dry-run validation.
- [x] Estimate rough disk space for manifests, logs, metrics, and selected part partials.
- [x] Check native extension availability before selected parts that require native code.
- [x] Verification: add preflight tests for each failure mode using synthetic paths/devices and mocked native-extension checks.

### Phase 6.5 Notes

- Tightened `run_preflight_checks()` so controller planning validates output-root collision/resume policy, output parent writability, config path presence, normalized config hash presence, shard table construction, selected device visibility/uniqueness, CPU one-worker fallback, required native extensions, and rough disk-space availability before manifest/layout creation.
- Preflight now builds the shard table early and reports `shard_count`, `total_shard_bytes`, `rough_required_disk_bytes`, and `free_disk_bytes`; the dry-run output includes the shard and disk estimate fields.
- Disk estimates are intentionally conservative and low-cost: they cover manifests/reports/metrics/worker logs plus rough selected-part partial overhead without trying to predict exact tensor sizes before the run.
- Added synthetic failure-mode tests for missing datasets/shards, unwritable output roots, invisible/duplicate CUDA devices, CPU fallback limits, insufficient disk space, run-ID collisions, strict config rejection, and missing native extensions.
- Verification: `python -m pytest tests/pipeline/test_distributed_controller.py -q` -> `26 passed`.

## Phase 7 - Experimental Fast Modes

- [x] Treat approximate modes as explicitly non-default.
- [x] Require exact baseline artifacts for the same config or dataset slice before fast-mode comparisons.
- [x] Record every quality-changing toggle in the manifest and output reports.
- [x] Prevent experimental fast outputs from silently overwriting canonical exact outputs.
- [x] Add warning banners for approximate local-top-K merges, non-exact `mid_ctx` semantics, or any future non-exact reducer.
- [x] Verification: add tests that experimental modes require acknowledgement and write to separate output roots or clearly marked artifact names.

### Phase 7 Notes

- Extended `DistributedConfig` with `experimental_exact_baseline_root` and `experimental_quality_toggles`.
- `distributed_experimental_fast` now requires `experimental_acknowledgement: true`, a baseline root, at least one quality-changing toggle, and an `output_base` clearly marked with `experimental` or `fast`.
- Added `ExperimentalFastModeRunConfig` to the distributed manifest so experimental acknowledgement, exact baseline root, quality-changing toggles, and warning banner are persisted with the run contract.
- Added `src/pipeline/distributed/experimental_modes.py` with `validate_experimental_fast_mode()` and `write_experimental_fast_report()`.
- Experimental validation requires an existing exact baseline root, non-empty quality toggles, a clearly marked run root such as `outputs/experimental_fast/<run_id>`, the correct experimental run mode, and a warning banner.
- The warning banner is standardized as: `EXPERIMENTAL FAST MODE: outputs are not exact and are not paper-eligible unless compared against exact baseline artifacts.`
- Manifest output-root validation now allows both canonical exact roots such as `outputs/<run_id>` and explicitly marked experimental roots such as `outputs/experimental_fast/<run_id>`.
- Verification: `python -m pytest tests/pipeline/test_distributed_experimental_modes.py tests/pipeline/test_distributed_config.py -q` -> `17 passed`.

## Phase 8 - Resume And Failure Semantics By Mode

- [x] Define which parts are resumable in each mode.
- [x] Define when a part can be skipped because outputs are already valid.
- [x] Define stale-output checks based on config hash, manifest schema, artifact metadata, and part status markers.
- [x] Ensure failed worker outputs are not merged unless explicitly marked valid.
- [x] Add cleanup rules for partial files without deleting unrelated run outputs.
- [x] Support cleanup policies: `keep_all`, `delete_large_partials_on_success`, `delete_all_partials_on_success`, and `manual_cleanup_only`.
- [x] Preserve failed-run partials, logs, metrics, and failure markers by default for debugging.
- [x] Default to `keep_all` while validating the distributed runtime; recommend `delete_large_partials_on_success` only after full-run equivalence and observability are trusted.
- [x] Verification: add tests for resume classification across completed, failed, stale, partial, and missing outputs.
- [x] Verification: add cleanup-policy tests proving successful-run cleanup removes only intended files and failed-run partials are preserved.

### Phase 8 Notes

- Added `src/pipeline/distributed/resume_policy.py` with mode-level resumability, part resume classification, merge safety checks, and cleanup planning.
- `resumable_parts_for_mode()` now documents which parts can be resumed for `single_process`, `distributed_simple_exact`, `distributed_mapreduce_exact`, and `distributed_experimental_fast`.
- `classify_part_resume_state()` reports `completed`, `failed`, `stale`, `partial`, `missing`, and `not_resumable`; parts are skippable only when the completion marker is completed/passed/ok, marker config hash matches, and every required output exists.
- Stale checks cover current config hash mismatch, invalid marker JSON, and marker-level config hash mismatch.
- `completed_worker_ids_for_merge()` blocks merge if any worker is failed, pending, stale, from the wrong phase, or from the wrong run.
- `build_cleanup_plan()` wraps existing cleanup candidate rules without deleting files; failed runs always preserve partials, logs, metrics, and markers.
- Cleanup remains scoped to distributed partial roots. `delete_large_partials_on_success` selects only large partial files, while `delete_all_partials_on_success` selects distributed worker/partial roots; unrelated canonical run outputs are preserved.
- `keep_all` remains the default policy while validating the distributed runtime. `delete_large_partials_on_success` should only be used after equivalence and observability are trusted.
- Verification: `python -m pytest tests/pipeline/test_distributed_resume_policy.py -q` -> `8 passed`.

## Phase 9 - Reporting And UX

- [x] Add a mode summary report for every run.
- [x] Add a final run report that links each part status, artifact path, equivalence result, benchmark result, and known warnings.
- [x] Make exactness status visible: `single_process_oracle`, `exact_equivalent`, `exact_mapreduce_equivalent`, or `experimental_non_exact`.
- [x] Include local/H100 hardware context in reports: device count, device names where available, physical GPU UUIDs, PCI bus IDs, CPU RAM, and CUDA memory summaries.
- [x] Write append-only JSONL metrics for controller and worker events under `outputs/<run_id>/distributed/reports/` and each worker directory.
- [x] Add lightweight device observability sampling during distributed runs: GPU utilization, VRAM used/total, power draw, temperature, CPU RAM, disk usage/write throughput, current phase label, worker PID, and physical GPU identity.
- [x] Keep observability sampling interval configurable and low-overhead, with a default suitable for H100 benchmarks.
- [x] Keep command output concise enough for logs but detailed enough for post-run debugging.
- [x] Verification: add report snapshot tests with stable synthetic manifests.
- [x] Verification: add JSONL metrics schema tests and synthetic observability sampler tests with mocked device/system stats.

### Phase 9 Notes

- Added `src/pipeline/distributed/reporting.py` with `build_mode_summary_report()`, `build_final_run_report()`, stable JSON save helpers, hardware context extraction, and an append-only `ObservabilitySample` JSONL schema.
- Mode summaries now expose the user-facing exactness label for each operating mode: `single_process_oracle`, `exact_equivalent`, `exact_mapreduce_equivalent`, or `experimental_non_exact`.
- Final run reports link part statuses, canonical artifacts, rollout gate state, equivalence report summaries, benchmark report summaries, warnings, output roots, cleanup policy, and manifest-recorded hardware metadata.
- Hardware context is manifest-driven and includes device count, worker-local logical IDs, physical IDs, names, UUIDs, PCI bus IDs, hostnames, and total VRAM where available. CPU RAM and live utilization remain sampled through observability events rather than requiring hardware access during report construction.
- Existing `MetricEvent` JSONL support remains the controller/worker metrics path; Phase 9 adds `ObservabilitySample` JSONL for low-overhead runtime samples containing phase, worker PID, physical GPU identity, GPU utilization, VRAM, power, temperature, CPU RAM, disk usage, and disk write throughput.
- The reporting helpers are deliberately side-effect light: report builders return dictionaries for command/log output, while save helpers write `mode_summary.json`, `run_summary.json`, or append JSONL only when called by controller/worker runtime code.
- Verification: `python -m pytest tests/pipeline/test_distributed_reporting.py -q` -> `7 passed`.

## Phase 10 - Documentation

- [x] Update the high-level multi-device plan with links to mode-specific commands and configs.
- [x] Document local RTX 5070 Ti workflow: use `single_process` or one-worker distributed validation.
- [x] Document H100 workflow: dry run, simple exact run, equivalence checks, benchmark, then optional MapReduce.
- [x] Document distributed search-cache workflow: keep cache generation off the critical path, then build it offline from validated final artifacts.
- [x] Document cleanup policy recommendations for validation runs versus mature full H100 runs.
- [x] Document mode selection guidance in plain language.
- [x] Document which modes are paper-eligible and which are exploratory only.
- [x] Verification: review docs against actual CLI/config names after implementation.

### Phase 10 Notes

- Updated `multi-device-improvements.md` with an `Operating Mode Workflow` section that links back to this part file and the two implemented config examples.
- Documented plain-language mode selection for `single_process`, `distributed_simple_exact`, `distributed_mapreduce_exact`, and `distributed_experimental_fast`.
- Documented the local RTX 5070 Ti path as either `single_process` or `config_examples/local-distributed-smoke.yaml`, keeping efficient memory settings and deferred search-cache generation.
- Documented the H100 path as controller dry run, `distributed_simple_exact`, one-worker/small-run equivalence checks, 8-worker benchmark, then optional MapReduce only if central pass-2 reduce is the bottleneck.
- Documented search-cache generation as offline/post-validation work from canonical `outputs/<run_id>/` artifacts, not part of the distributed critical path.
- Documented cleanup policy recommendations: `keep_all` for validation and failures, `delete_large_partials_on_success` after trust is established, `delete_all_partials_on_success` only for mature reproducible runs, and `manual_cleanup_only` for paper-facing preservation.
- Documented paper eligibility: `single_process`, equivalence-gated `distributed_simple_exact`, and equivalence-gated `distributed_mapreduce_exact`; `distributed_experimental_fast` remains exploratory only.
- Reviewed docs against implemented names: `config_examples/local-distributed-smoke.yaml`, `config_examples/h100-8x-distributed-simple-exact.yaml`, `pipeline.distributed.controller`, `pipeline.distributed.worker`, `pipeline.distributed.pass1_merge`, `pipeline.distributed.pass2_reduce`, `distributed.mode`, `distributed.cleanup_policy`, and `persist.build_search_cache_after_pipeline`.

## Phase 11 - Testing And Verification

- [x] Run config validation tests for all modes.
- [x] Run tests proving `single_process` and distributed modes both write canonical artifacts under `outputs/<run_id>/`.
- [x] Run strict-config tests proving unknown distributed config keys are rejected.
- [x] Run config tests proving distributed modes default to offline/deferred search-cache generation.
- [x] Run command parser/help tests for all distributed entrypoints.
- [x] Run dry-run command-generation tests that assert per-worker `CUDA_VISIBLE_DEVICES` isolation.
- [x] Run dry-run tests for local one-worker and synthetic 8-worker configurations.
- [x] Run tests proving distributed internals stay under `outputs/<run_id>/distributed/` and canonical run artifacts appear at the top of `outputs/<run_id>/` only after validation.
- [x] Run tests proving H100 workers are one-device isolated and cannot inherit multi-device `SAEBank` placement.
- [x] Run cleanup policy, JSONL metrics, and observability sampler tests.
- [x] Run gate/resume/report tests.
- [ ] Run one-worker exact smoke before enabling multi-worker exact mode.
- [x] Document exact verification commands in this file after implementation.

### Phase 11 Notes

- Automated verification passed with the focused distributed suite:

```powershell
python -m pytest tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_operating_modes.py tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass2_replay.py tests/pipeline/test_distributed_pass2_partials.py tests/pipeline/test_distributed_pass2_reduce.py tests/pipeline/test_distributed_pass2_equivalence.py tests/pipeline/test_distributed_pass2_benchmark.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_rollout_gates.py tests/pipeline/test_distributed_experimental_modes.py tests/pipeline/test_distributed_resume_policy.py tests/pipeline/test_distributed_reporting.py -q
```

- Result: `286 passed in 2.90s`.
- Coverage includes mode taxonomy, strict config validation, example configs, deferred search-cache defaults, canonical run-root layout, distributed worker layout, device isolation, worker-local `cuda:0` runtime construction, controller dry-runs, command parser/help output, one-worker local planning, synthetic 8-worker H100 planning, deterministic assignments, pass-1 merge, pass-2 replay, pass-2 partials/reduce/equivalence, pass-2 benchmark summaries, worker orchestration, rollout gates, experimental mode guardrails, resume policy, cleanup policy, JSONL metrics, and reporting/observability schemas.
- `test_distributed_operating_modes.py` verifies the canonical `outputs/<run_id>/` root policy for all modes, including `single_process`; `test_distributed_pass1_merge.py` and `test_distributed_pass2_reduce.py` verify distributed validated artifacts are written at the canonical run root.
- The live one-worker exact smoke was not run in this workspace because no local dataset files are present under `data/`. Keep this as a hard gate before trusting multi-worker exact H100 runs.
- Recommended pre-H100 one-worker smoke sequence:

```powershell
$env:PYTHONPATH = "src"
python -m pipeline.distributed.controller --config config_examples/local-distributed-smoke.yaml --dry-run
python -m pipeline.distributed.controller --config config_examples/local-distributed-smoke.yaml --launch
python -m pipeline.distributed.pass1_merge --manifest outputs/<run_id>/distributed/manifest.json
python -m pipeline.distributed.pass2_reduce --output-root outputs/<run_id> --candidate-dump outputs/<run_id>/distributed/workers/worker_000/pass2/candidate_dump.partial.pt
```

- After the one-worker smoke, run rollout gates and compare canonical artifacts against a `single_process` oracle before enabling two-worker or eight-worker runs.

---

## Open Questions

- Should part-specific commands be separate scripts or subcommands under one controller CLI?
- Should experimental fast modes be allowed in the same output root as exact modes if names are clearly marked?
- What exact report fields are required before a run is considered paper-eligible?

## Risks / Assumptions

- The existing `single_process` path must remain the default and correctness oracle.
- Mode names must not imply full 8-GPU utilization unless the selected mode actually uses it.
- Experimental modes can easily contaminate paper results if artifact names and reports are not explicit.
- `distributed_mapreduce_exact` should not be recommended until it is proven equivalent to `distributed_simple_exact`.
- Good UX matters here: unclear mode selection could lead to invalid benchmarks or misleading hardware claims.
