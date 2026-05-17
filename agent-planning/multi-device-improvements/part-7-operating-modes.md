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

- [ ] Define `single_process` as the current pipeline and permanent correctness oracle.
- [ ] Define `distributed_simple_exact` as distributed pass 1/pass 2/discovery with simple exact pass-2 candidate-dump merge.
- [ ] Define `distributed_mapreduce_exact` as distributed pass 1/pass 2/discovery with exact pass-2 partial-sum shuffle and target-range reducers.
- [ ] Define `distributed_experimental_fast` as opt-in approximate or quality-changing modes only after exact baselines are benchmarked.
- [ ] Define `dry_run` behavior for every distributed mode so assignments and artifact paths can be inspected without model loading.
- [ ] Use `outputs/<run_id>/` as the canonical artifact root for every mode, including `single_process`.
- [ ] Generate default run IDs as `YYYYMMDD-HHMMSS-<config_hash_8>` when the user does not provide one.
- [ ] Verification: add tests that every mode has a documented name, description, required parts, and allowed entrypoints.

## Phase 2 - Config Surface

- [ ] Add a distributed/runtime config section without changing defaults for existing local runs.
- [ ] Use `distributed` as the config namespace for orchestration settings.
- [ ] Include fields for mode, run ID/output root, worker count, devices, launch strategy, resume policy, cleanup policy, part selection, and strict equivalence checks.
- [ ] Include schema-version fields for manifest, partial artifacts, metrics JSONL, sanity reports, and run summaries.
- [ ] Use strict Pydantic config validation, for example `extra='forbid'`, so unknown or misspelled distributed config keys fail early.
- [ ] Include an explicit `distributed.sampling_seed`; store it in the manifest and derive artifact-specific seeds for `seq_repr` and `mid_ctx`.
- [ ] Include `distributed.mid_ctx_candidate_pool` settings: `enabled`, `band_margin_sigma`, `max_candidates_per_latent`, and `on_truncation`.
- [ ] Default `distributed.mid_ctx_candidate_pool.enabled: true`, `band_margin_sigma: 1.0`, `max_candidates_per_latent: max(256, 4 * num_ctx_sequences)`, and `on_truncation: replay_fallback` for exact/paper modes.
- [ ] Restrict `distributed.mid_ctx_candidate_pool.on_truncation` to `fail`, `replay_fallback`, or `allow_bounded_approx`; paper-eligible modes must use `fail` or `replay_fallback`.
- [ ] Keep distributed search-cache generation offline by default, for example `persist.build_search_cache_after_pipeline: false` during distributed benchmark/paper runs.
- [ ] Use `outputs/<run_id>/distributed/` for distributed manifests, worker partials, part status, and reports.
- [ ] Write validated canonical outputs directly at the top of `outputs/<run_id>/` only after exactness and sanity checks pass.
- [ ] Keep `single_process` as the default when no distributed config is present, while still assigning a run ID and writing canonical outputs under `outputs/<run_id>/`.
- [ ] Validate that `distributed_mapreduce_exact` cannot be selected until required MapReduce schemas and tests exist.
- [ ] Validate that `distributed_experimental_fast` requires explicit acknowledgement in config or CLI.
- [ ] Verification: add config validation tests for default local config, one-worker config, H100 simple exact config, MapReduce gated config, and invalid combinations.

## Phase 3 - Entrypoints And Commands

- [ ] Keep `python src/main.py` mapped to the current `single_process` pipeline.
- [ ] Add `--run-id` support, or an equivalent config field, for `single_process` so local and distributed outputs use the same run-root layout.
- [ ] Add a distributed controller command for creating manifests and running or printing worker commands.
- [ ] Add part-specific commands for running individual stages from existing artifacts, for example pass-1 worker, pass-1 merge, neg_ctx, pass-2 dump, pass-2 reduce, candidate/discovery.
- [ ] Support a dry-run command that prints exact per-worker commands, including `CUDA_VISIBLE_DEVICES=<physical_id>`, for manual debugging.
- [ ] Support an optional built-in Python `subprocess.Popen` launcher that runs the same worker commands after dry-run command generation is stable.
- [ ] Keep external scheduler support as a later integration that consumes the same manifest and worker command contract.
- [ ] Support `--dry-run`, `--resume`, and `--run-id` consistently across distributed commands.
- [ ] Print a concise mode summary before work starts: mode, workers, devices, output root, selected parts, and exactness guarantees.
- [ ] Verification: add CLI/help tests or parser tests that cover every command without loading model weights.

## Phase 4 - Local Compatibility Mode

- [ ] Support one-worker `distributed_simple_exact` on CPU or one CUDA device for local validation.
- [ ] Ensure local mode can use `hardware.memory: "efficient"`, `keep_model_loaded_for_neg_ctx: false`, deferred search cache, small `n_shards`, and small `n_seeds`.
- [ ] Ensure local mode does not require H100-only configs, multi-GPU devices, or replicated workers.
- [ ] Keep search-cache generation deferred/offline in local distributed validation unless the user explicitly enables it.
- [ ] Document when to use `single_process` versus one-worker distributed mode locally.
- [ ] Do not add `outputs/latest` initially; document `outputs/<run_id>/` as the source of truth for local and distributed runs.
- [ ] Add a local example config for one-worker distributed validation if useful.
- [ ] Verification: add dry-run tests for RTX 5070 Ti style settings and one-worker distributed assignments.

## Phase 5 - H100 Exact Modes

- [ ] Define a recommended H100 `distributed_simple_exact` config: one worker per GPU, replicated model+SAE, global artifact merges, simple exact pass-2 reduce.
- [ ] Require H100 worker launch to isolate devices so each worker sees one logical `cuda:0` and loads a full local model+SAE bank.
- [ ] Use manifest-declared devices for distributed device-consuming phases by default, including `neg_ctx`; all visible devices may remain a standalone/single-process convenience.
- [ ] Define the entry criteria for H100 `distributed_mapreduce_exact`: simple exact merge benchmark shows central dump merge or reducer input memory is a bottleneck.
- [ ] Keep current `config_examples/h100-8x.yaml` clearly labeled as current-runtime/single-process, not the future distributed runtime.
- [ ] Add a future distributed H100 config only after Parts 1-6 have working exact paths.
- [ ] Document which parts use GPUs, CPU/OpenMP, disk I/O, and centralized merges in each exact mode.
- [ ] Verification: add dry-run tests for synthetic 8-worker H100 assignments and expected output layout.

## Phase 6 - Rollout Gates

- [ ] Require `distributed_simple_exact` to pass one-worker equivalence before two-worker or eight-worker use.
- [ ] Require `distributed_simple_exact` to pass tiny synthetic and reduced real-data equivalence before paper-facing runs.
- [ ] Require `distributed_mapreduce_exact` to match `distributed_simple_exact` before it can be recommended.
- [ ] Require every mode to write a manifest, sanity reports, and verification status.
- [ ] Require benchmark results before changing recommended defaults.
- [ ] Verification: add mode gate tests that reject unsafe transitions, stale manifests, missing equivalence reports, and missing sanity reports.

## Phase 6.5 - Preflight Checks

- [ ] Run preflight checks before expensive work starts.
- [ ] Check output root writability and reject existing `outputs/<run_id>/` unless `--resume` is selected.
- [ ] Load config with strict validation and compute the normalized config hash before writing the manifest.
- [ ] Build the dataset shard table and fail if shard counts/order are unavailable or stale.
- [ ] Validate selected devices exist, are unique, and match manifest-declared device policy.
- [ ] Confirm CPU one-worker fallback remains available for local dry-run validation.
- [ ] Estimate rough disk space for manifests, logs, metrics, and selected part partials.
- [ ] Check native extension availability before selected parts that require native code.
- [ ] Verification: add preflight tests for each failure mode using synthetic paths/devices and mocked native-extension checks.

## Phase 7 - Experimental Fast Modes

- [ ] Treat approximate modes as explicitly non-default.
- [ ] Require exact baseline artifacts for the same config or dataset slice before fast-mode comparisons.
- [ ] Record every quality-changing toggle in the manifest and output reports.
- [ ] Prevent experimental fast outputs from silently overwriting canonical exact outputs.
- [ ] Add warning banners for approximate local-top-K merges, non-exact `mid_ctx` semantics, or any future non-exact reducer.
- [ ] Verification: add tests that experimental modes require acknowledgement and write to separate output roots or clearly marked artifact names.

## Phase 8 - Resume And Failure Semantics By Mode

- [ ] Define which parts are resumable in each mode.
- [ ] Define when a part can be skipped because outputs are already valid.
- [ ] Define stale-output checks based on config hash, manifest schema, artifact metadata, and part status markers.
- [ ] Ensure failed worker outputs are not merged unless explicitly marked valid.
- [ ] Add cleanup rules for partial files without deleting unrelated run outputs.
- [ ] Support cleanup policies: `keep_all`, `delete_large_partials_on_success`, `delete_all_partials_on_success`, and `manual_cleanup_only`.
- [ ] Preserve failed-run partials, logs, metrics, and failure markers by default for debugging.
- [ ] Default to `keep_all` while validating the distributed runtime; recommend `delete_large_partials_on_success` only after full-run equivalence and observability are trusted.
- [ ] Verification: add tests for resume classification across completed, failed, stale, partial, and missing outputs.
- [ ] Verification: add cleanup-policy tests proving successful-run cleanup removes only intended files and failed-run partials are preserved.

## Phase 9 - Reporting And UX

- [ ] Add a mode summary report for every run.
- [ ] Add a final run report that links each part status, artifact path, equivalence result, benchmark result, and known warnings.
- [ ] Make exactness status visible: `single_process_oracle`, `exact_equivalent`, `exact_mapreduce_equivalent`, or `experimental_non_exact`.
- [ ] Include local/H100 hardware context in reports: device count, device names where available, physical GPU UUIDs, PCI bus IDs, CPU RAM, and CUDA memory summaries.
- [ ] Write append-only JSONL metrics for controller and worker events under `outputs/<run_id>/distributed/reports/` and each worker directory.
- [ ] Add lightweight device observability sampling during distributed runs: GPU utilization, VRAM used/total, power draw, temperature, CPU RAM, disk usage/write throughput, current phase label, worker PID, and physical GPU identity.
- [ ] Keep observability sampling interval configurable and low-overhead, with a default suitable for H100 benchmarks.
- [ ] Keep command output concise enough for logs but detailed enough for post-run debugging.
- [ ] Verification: add report snapshot tests with stable synthetic manifests.
- [ ] Verification: add JSONL metrics schema tests and synthetic observability sampler tests with mocked device/system stats.

## Phase 10 - Documentation

- [ ] Update the high-level multi-device plan with links to mode-specific commands and configs.
- [ ] Document local RTX 5070 Ti workflow: use `single_process` or one-worker distributed validation.
- [ ] Document H100 workflow: dry run, simple exact run, equivalence checks, benchmark, then optional MapReduce.
- [ ] Document distributed search-cache workflow: keep cache generation off the critical path, then build it offline from validated final artifacts.
- [ ] Document cleanup policy recommendations for validation runs versus mature full H100 runs.
- [ ] Document mode selection guidance in plain language.
- [ ] Document which modes are paper-eligible and which are exploratory only.
- [ ] Verification: review docs against actual CLI/config names after implementation.

## Phase 11 - Testing And Verification

- [ ] Run config validation tests for all modes.
- [ ] Run tests proving `single_process` and distributed modes both write canonical artifacts under `outputs/<run_id>/`.
- [ ] Run strict-config tests proving unknown distributed config keys are rejected.
- [ ] Run config tests proving distributed modes default to offline/deferred search-cache generation.
- [ ] Run command parser/help tests for all distributed entrypoints.
- [ ] Run dry-run command-generation tests that assert per-worker `CUDA_VISIBLE_DEVICES` isolation.
- [ ] Run dry-run tests for local one-worker and synthetic 8-worker configurations.
- [ ] Run tests proving distributed internals stay under `outputs/<run_id>/distributed/` and canonical run artifacts appear at the top of `outputs/<run_id>/` only after validation.
- [ ] Run tests proving H100 workers are one-device isolated and cannot inherit multi-device `SAEBank` placement.
- [ ] Run cleanup policy, JSONL metrics, and observability sampler tests.
- [ ] Run gate/resume/report tests.
- [ ] Run one-worker exact smoke before enabling multi-worker exact mode.
- [ ] Document exact verification commands in this file after implementation.

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
