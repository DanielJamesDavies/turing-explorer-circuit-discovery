# Plan: Part 2 - Distributed Pass 1

> **Goal:** Run first-pass latent statistics and context collection across multiple independent workers, then merge worker partials into the same canonical artifacts produced by the current single-process pipeline.
>
> **Created:** 2026-05-16

---

## Scope

This part distributes the first forward-heavy stage:

- model forward over dataset shards,
- SAE encoding,
- `latent_stats` updates,
- `top_ctx` updates,
- oversampled `mid_ctx` candidate-pool collection for final filtering after global stats are known,
- `logit_ctx` updates,
- `seq_repr` capture,
- optional `seq_latent_index` writing.

It does not build `neg_ctx`, run pass 2, select candidates, or run discovery.

The output of this part is a validated global first-pass artifact set:

- `outputs/<run_id>/latent_stats.pt`
- `outputs/<run_id>/top_ctx.pt`
- `outputs/<run_id>/mid_ctx.pt`
- `outputs/<run_id>/seq_repr.pt`
- `outputs/<run_id>/logit_ctx.pt`
- `outputs/<run_id>/seq_latent_index/`, if enabled

---

## Phase 1 - Worker First-Pass Entrypoint

- [x] Add a worker entrypoint that can run pass 1 for only the shard IDs assigned in the Part 1 manifest.
- [x] Ensure each worker initializes a local `DataLoader`, full local `Inference` model, full local `SAEBank`, and local stores.
- [x] Ensure distributed workers construct `SAEBank` with exactly one worker-local device; no worker should split SAE layers across multiple physical GPUs.
- [x] Keep the existing `src/main.py` and `pipeline.run()` path unchanged for normal single-process runs.
- [x] Add a way for the worker to write first-pass outputs under `outputs/<run_id>/distributed/workers/worker_000/pass1/`.
- [x] Preserve existing first-pass behavior inside each worker for stores that do not depend on final global statistics: `latent_stats`, `top_ctx`, `seq_repr`, `logit_ctx`, and `seq_latent_index`.
- [ ] During pass 1, write oversampled `mid_ctx` candidate pools instead of final `mid_ctx` rows.
- [ ] After merged global `latent_stats` exist, filter the candidate pools with final global `mean_seq` and `std_seq` to produce final `mid_ctx`.
- [x] Add worker metadata with shard IDs, global sequence ID ranges, batch count, output paths, start/end time, and device.
- [x] Verification: run a one-worker dry pass on synthetic shards and confirm the worker writes the expected partial artifact names and metadata.
- [x] Verification: add a resource-construction unit test that fails if distributed worker mode passes multiple devices to `SAEBank`.

### Phase 1 Notes

- Added `src/pipeline/distributed/worker.py` with a `python -m pipeline.distributed.worker --manifest <manifest> --worker-id <id>` entrypoint for pass-1 workers.
- Added optional shard-subset execution to `DataLoader.get_batches_for_shards()` and `run_first_pass(assigned_shard_ids=...)`, preserving global sequence IDs and leaving the default `src/main.py` path unchanged.
- Pass-1 workers initialize a worker-local runtime from the manifest device assignment, construct `DataLoader`, `Inference`, `SAEBank`, and a full-dataset `SeqRepr`, then save worker-local partials under `outputs/<run_id>/distributed/workers/worker_000/pass1/`.
- Worker markers now cover pass-1 start/completion/failure with assigned shard IDs, sequence totals, timing, device metadata, and saved artifact paths.
- Current `mid_ctx_candidates.partial.pt` writing is a temporary worker-local checkpoint path; the planned oversampled deterministic candidate-pool schema remains unchecked for Phase 6.
- Verification: `python -m pytest tests/pipeline/test_distributed_worker.py tests/test_data_loader.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py -q` -> `36 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py -q` -> `74 passed`.

## Phase 2 - Stable Shard And Sequence IDs

- [x] Consume the canonical full-dataset sequence ID table from Part 1.
- [x] Teach the worker data path to load only assigned shards while preserving the global sequence IDs from the full dataset ordering.
- [x] Avoid renumbering sequences locally inside each worker.
- [x] Store each worker's shard ID ranges in its metadata and validate them against the manifest.
- [x] Ensure `seq_latent_index` partial files preserve canonical shard IDs, not worker-local shard IDs.
- [x] Add validation that worker shard assignments are disjoint and cover the intended `data.n_shards` set exactly.
- [x] Verification: add synthetic multi-shard tests proving that a two-worker split emits the same global sequence IDs as a single-process loader over the same shards.
- [x] Verification: add failure tests for stale global sequence tables, missing assigned shards, duplicated sequence IDs, and out-of-range worker sequence IDs.

### Phase 2 Notes

- Added pass-1 worker input validation that checks the manifest shard table against disk before model/SAE initialization, rejects stale/missing shard files, and requires all pass-1 shard assignments to be disjoint and complete.
- Worker markers now include canonical shard ranges with `shard_index`, `global_start_id`, `global_end_id`, and `sequence_count`, so worker-local outputs can be traced back to the full-dataset sequence ID table.
- `DataLoader.get_batches_for_shards()` is covered by split-worker tests proving assigned shards emit the same global IDs as the single-process loader, without worker-local renumbering.
- `seq_latent_index` worker output stays under the worker pass-1 directory while preserving canonical shard filenames such as `shard_0.pt`.
- Verification: `python -m pytest tests/pipeline/test_distributed_worker.py tests/test_data_loader.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_shard_table.py -q` -> `32 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py -q` -> `80 passed`.

## Phase 3 - Partial Artifact Schema

- [x] Define partial checkpoint schemas for `latent_stats.partial.pt`, `top_ctx.partial.pt`, `mid_ctx_candidates.partial.pt`, `seq_repr.partial.pt`, and `logit_ctx.partial.pt`.
- [x] Include metadata in every partial: schema version, run ID, worker ID, shard IDs, sequence ID min/max, config hash, device, and store-specific mode fields.
- [x] Define a versioned `mid_ctx_candidates.partial.pt` schema with component IDs, latent IDs, sequence IDs, activation values, deterministic priorities, candidate-pool settings, truncation counters, and worker metadata.
- [x] Keep tensor names close to the existing checkpoint names so merge code can reuse current store load/save logic where practical.
- [x] Save partial artifacts atomically to avoid merge reading incomplete files.
- [x] Add partial artifact validation before merge: tensor shapes, dtypes, finite values, expected component count, expected SAE width, and config hash.
- [x] Verification: add round-trip tests for every partial artifact schema using tiny tensors.

### Phase 3 Notes

- Added `src/pipeline/distributed/pass1_partials.py` with `Pass1PartialMetadata`, schema version `1`, atomic `torch.save` writes, load helpers, and validation for all current pass-1 partial artifact types.
- Worker partial files now save as `{metadata, payload}` envelopes, with metadata covering run ID, worker ID, shard IDs, global sequence bounds, config hash, physical/logical device, component count, SAE width, and store-specific mode fields.
- `latent_stats`, `top_ctx`, `seq_repr`, and `logit_ctx` payloads keep their existing tensor names so merge/load code can reuse current store semantics.
- `mid_ctx_candidates.partial.pt` now has versioned candidate-pool fields: `component_ids`, `latent_ids`, `sequence_ids`, `activation_values`, deterministic `priorities`, `candidate_pool_settings`, and `truncation_counters`. It also carries the temporary worker-local `mid_ctx` tensors until the later oversampled collection semantics are implemented.
- Partial validation now rejects wrong artifact names, stale config hashes, invalid shapes/dtypes, out-of-range candidate IDs, and non-finite tensor values before merge.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_worker.py -q` -> `16 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py -q` -> `88 passed`.

## Phase 4 - Merge `latent_stats`

- [x] Implement exact parallel Welford merges for token-level stats: `active_count`, `mean`, `m2`, `mean_abs`, and `m2_abs`.
- [x] Implement exact parallel Welford merges for sequence-level stats: `seq_count`, `mean_seq`, and `m2_seq`.
- [x] Preserve `component_steps` semantics or replace them with explicit merged batch/update counts in metadata if exact equality is not meaningful after worker splits.
- [x] Validate merged counts are equal to the sum of worker counts.
- [x] Validate no NaN or negative variance state is introduced.
- [x] Verification: add tests that split a synthetic latent activation stream across workers and prove merged stats match single-process stats within numerical tolerance.

### Phase 4 Notes

- Added `src/pipeline/distributed/pass1_merge.py` with `load_and_merge_latent_stats_partials()` and `merge_latent_stats_partials()`.
- The merge uses parallel Welford combine semantics for token-level `mean`/`m2`, `mean_abs`/`m2_abs`, and sequence-level `mean_seq`/`m2_seq`, while summing `active_count` and `seq_count` exactly.
- `component_steps` are preserved as merged update-count diagnostics by summing each worker's per-component counts.
- The merged payload validates count sums, finite merged means/M2 tensors, duplicate worker rejection, matching run/config/dimension metadata, and non-negative variance state.
- Verification includes direct synthetic stream equality and a split `LatentStats.update_component()` equivalence test against a single-process store.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py -q` -> `13 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py -q` -> `93 passed`.

## Phase 5 - Merge `top_ctx`

- [x] For each component and latent, concatenate all worker `top_ctx` rows.
- [x] Select the global top-K sequence IDs by activation value using the same K as `config.latents.top_ctx.n_sequences`.
- [x] Preserve deterministic tie-breaking, preferably by value descending then sequence ID ascending.
- [x] Keep invalid sentinel rows zeroed.
- [x] Validate merged sequence IDs are in global range and values are finite/non-negative.
- [x] Verification: add tests where top contexts are split across workers and global top-K requires rows from multiple workers.

### Phase 5 Notes

- Extended `src/pipeline/distributed/pass1_merge.py` with `load_and_merge_top_ctx_partials()` and `merge_top_ctx_partials()`.
- The merge concatenates each worker's `ctx_seq_idx`/`ctx_seq_val` rows and keeps the original top-K width from the partial schema.
- Global selection is deterministic: invalid rows are zeroed, candidates are tie-broken by sequence ID ascending, then selected by activation value descending.
- Validation rejects duplicate workers, mismatched run/config/dimensions, worker-local sequence IDs outside the manifest range, non-finite values, negative values, and non-zero invalid sentinel rows.
- Verification covers global top-K requiring rows from multiple workers, equal-value tie cases, invalid sentinel cleanup, path-based load/merge round trips, and out-of-range worker IDs.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py -q` -> `17 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py -q` -> `97 passed`.

## Phase 6 - Merge `mid_ctx`

- [x] Treat `mid_ctx.mode` as part of the artifact contract.
- [x] Use the long-term distributed `mid_ctx` solution: oversampled deterministic candidate pools collected during pass 1, filtered after final global stats merge.
- [x] Compute/merge global `latent_stats` first, then define the final mid band from global `mean_seq` and `std_seq`.
- [x] During pass 1, workers collect candidate rows from a widened provisional band around the configured final mid band, controlled by `distributed.mid_ctx_candidate_pool.band_margin_sigma`.
- [x] Default `distributed.mid_ctx_candidate_pool.enabled` to `true` for distributed modes.
- [x] Default `distributed.mid_ctx_candidate_pool.band_margin_sigma` to `1.0`.
- [x] Default `distributed.mid_ctx_candidate_pool.max_candidates_per_latent` to `max(256, 4 * num_ctx_sequences)`.
- [x] Default `distributed.mid_ctx_candidate_pool.on_truncation` to `replay_fallback` for exact/paper-eligible modes.
- [x] Allow `allow_bounded_approx` only in explicit experimental modes.
- [x] Store enough candidate data to filter after stats merge: component, latent, global sequence ID, activation value, deterministic priority, and candidate-pool provenance.
- [x] Use an explicit `distributed.sampling_seed`, stored in the manifest, as the root seed for reproducible sampling.
- [x] Derive the `mid_ctx` priority seed by stable hashing over `distributed.sampling_seed`, artifact name, dataset fingerprint, band parameters, component, latent, and sequence ID.
- [x] Do not use `run_id` as the sampling seed source; rerunning the same paper config with a new run ID should produce the same deterministic samples.
- [x] After global stats merge, filter candidate rows by final global mid-band membership and select the best `n_sequences` valid examples per latent by deterministic priority.
- [x] Preserve unbiased sampling over all globally valid mid-band examples while making distributed merge deterministic and reproducible.
- [x] Detect candidate-pool truncation or insufficient coverage per latent and record candidate count, filtered valid count, selected count, truncation flag, and fill rate.
- [x] In paper-eligible exact modes, handle candidate-pool coverage failure with `fail` or `replay_fallback`; do not silently mark bounded-approximate `mid_ctx` as exact.
- [x] Keep a stats-aware replay fallback that replays assigned shards only when candidate-pool validation fails or exact mode explicitly requests guaranteed completeness.
- [x] Classify `mid_ctx` candidate pools as large partial artifacts eligible for deletion only after final `mid_ctx.pt` validation succeeds and cleanup policy allows it.
- [x] Make deterministic priority-reservoir the only paper-eligible distributed `mid_ctx` mode.
- [x] Keep existing single-process `reservoir_cpu` available for non-distributed runs, but do not treat naive worker reservoir concatenation as exact.
- [x] Preserve `reservoir_fill`, `reservoir_n` or replace them with equivalent distributed priority-reservoir metadata: candidate count, valid count, selected count, truncation flag, priority seed/hash version, band bounds, candidate-pool settings, and `num_ctx_sequences`.
- [x] Validate fill counts and sequence ID ranges after merge.
- [x] Verification: add statistical unit tests showing deterministic priority-reservoir selection is uniform over valid examples across many seeded trials.
- [x] Verification: add exactness tests showing candidate-pool filtering equals a single global priority-reservoir pass when no candidate-pool truncation occurs.
- [x] Verification: add tests showing truncation/coverage failures are detected and either fail or trigger replay fallback according to `on_truncation`.
- [x] Verification: add replay-fallback tests showing fallback output equals a full global priority-reservoir pass.
- [x] Verification: add tests proving closest-to-midpoint top-K is not used as the default distributed semantic unless explicitly configured as a biased experimental mode.

### Phase 6 Notes

- Extended `src/pipeline/distributed/pass1_merge.py` with `load_and_merge_mid_ctx_candidate_partials()` and `merge_mid_ctx_candidate_partials()`.
- The merge consumes versioned `mid_ctx_candidates.partial.pt` candidate fields, filters them with merged global `latent_stats.mean_seq`/`m2_seq`, and emits canonical `mid_ctx` tensors with `mode: distributed_priority_reservoir`.
- Deterministic selection uses stored priorities rather than closest-to-midpoint ranking; tests prove lower-priority candidates are selected even when they do not have the highest activation values.
- Merge reports include candidate counts, valid counts, selected counts, truncation counters, fill rates, band settings, pool defaults, and whether replay fallback or bounded approximation is required.
- Truncation policy supports `fail`, `replay_fallback`, and `allow_bounded_approx`; `fail` raises on truncation and `replay_fallback` marks the output as requiring replay.
- Distributed workers now widen the provisional collection band by `band_margin_sigma=1.0` and oversize the worker candidate pool to `max(256, 4 * num_ctx_sequences)` while recording the final band separately for merge filtering.
- `mid_ctx` priorities are derived with `sha256-v1` from `distributed.sampling_seed`, artifact name, dataset fingerprint, final/candidate band parameters, component, latent, and sequence ID. `run_id` is not part of the seed material.
- Candidate-pool metadata records seed provenance, dataset fingerprint, priority hash version, candidate band, final band, final context count, and max candidates per latent.
- Replay fallback now has an explicit stats-aware callback hook for exact fallback execution; without a callback, the merge report marks fallback as required and blocks candidate-pool cleanup eligibility.
- Cleanup eligibility is reported per merge and is false whenever truncation requires fallback or bounded approximation.
- Verification covers stable hash seeding, run-ID-independent reruns, changed seed/fingerprint/band changes, uniform priority distribution across seeded trials, widened worker candidate-pool configuration, fallback execution, bounded-approximation reporting, and cleanup eligibility.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py -q` -> `57 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py tests/store/test_mid_ctx_modes.py tests/store/test_neg_context_backend.py -q` -> `136 passed`.

## Phase 7 - Merge `seq_repr`

- [x] Move `seq_repr` capped sampling from per-worker random initialization to manifest-level deterministic sampling.
- [x] Generate the capped sequence sample once from the global sequence ID table and an explicit `distributed.sampling_seed` stored in the manifest.
- [x] Derive the `seq_repr` cap seed by stable hashing over `distributed.sampling_seed`, artifact name, dataset fingerprint, cap size, and total sequence count.
- [x] Do not use `run_id` as the sampling seed source; new run IDs should not silently change `seq_repr` membership for the same config/data.
- [x] Store `slot_to_id` and `id_to_slot` as tensor artifacts referenced by the manifest; avoid huge JSON mappings.
- [x] Ensure every worker loads the same global mapping before first-pass `seq_repr.update()`.
- [x] For uncapped mode, copy worker representations into rows indexed by global sequence ID.
- [x] For capped mode, accept only sequences selected by the manifest-level cap and copy them into their global slots.
- [x] Validate each selected sequence slot is written at most once and all worker writes agree with global ID bounds.
- [x] Verification: add tests showing distributed capped and uncapped `seq_repr` merges match a single global slot mapping.
- [x] Verification: add tests proving two workers with disjoint shards cannot generate different caps for the same run.
- [x] Verification: add tests for cap determinism under fixed seed and cap changes under changed seed.

### Phase 7 Notes

- Added `src/pipeline/distributed/seq_repr_mapping.py` with deterministic `slot_to_id` / `id_to_slot` mapping helpers, stable shard-table fingerprints, and artifact-specific cap seed derivation.
- Added `sampling_seed` to the distributed manifest contract and updated pass-1 worker initialization so every worker constructs `SeqRepr` with the same manifest-derived slot mapping before `seq_repr.update()`.
- `SeqRepr` now accepts explicit `slot_to_id` / `id_to_slot` tensors while preserving existing random single-process behavior when no mapping is provided.
- Extended `src/pipeline/distributed/pass1_merge.py` with `load_and_merge_seq_repr_partials()` and `merge_seq_repr_partials()` for capped and uncapped payloads.
- The merge copies uncapped rows by global sequence ID and capped rows through the global slot mapping, rejects duplicate selected slot writes, validates finite rows and mapping compatibility, and records written/missing slot counts.
- Verification covers deterministic cap mapping under fixed seed, cap changes under changed seed, uncapped global-row merge, capped global-slot merge, path-based load/merge, and duplicate selected-slot rejection.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py -q` -> `26 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py -q` -> `106 passed`.

## Phase 8 - Merge `logit_ctx`

- [x] Define exact event top-K merge semantics for `top_tokens`, `top_probs`, and `latent_counts`.
- [x] Sum `latent_counts` across workers.
- [x] For each latent, concatenate worker token/prob event rows and retain the configured top tokens by probability.
- [x] Define deterministic tie-breaking for equal probabilities: probability descending, token ID ascending, then worker/candidate row order if still tied.
- [x] Preserve current artifact meaning; do not switch to mean/aggregated per-token probabilities in the first distributed implementation.
- [x] Validate token IDs are in vocabulary range and probabilities are finite.
- [x] Verification: add synthetic tests where different workers contribute different top tokens for the same latent and the global merge keeps the expected event top-K.
- [x] Verification: add tie-case tests proving deterministic ordering across worker split orders.
- [x] Verification: add tests showing `latent_counts` sum exactly and event top-K values match a single-process event stream.

### Phase 8 Notes

- Extended `src/pipeline/distributed/pass1_merge.py` with `load_and_merge_logit_ctx_partials()` and `merge_logit_ctx_partials()`.
- The merge sums `latent_counts` exactly across workers and preserves the current event top-K artifact meaning for `top_tokens` / `top_probs`.
- For each latent, worker event rows are concatenated and selected by probability descending, then token ID ascending, worker ID ascending, and candidate row ascending for deterministic tie handling.
- Validation rejects duplicate worker partials, shape/schema mismatches, non-finite probabilities, negative token IDs, and token IDs outside an optional vocabulary bound.
- Verification covers global event top-K selection across workers, exact count summation, tie determinism with input partials in non-worker order, path-based load/merge, vocabulary-range validation, and non-finite probability rejection.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py -q` -> `31 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py -q` -> `111 passed`.

## Phase 9 - Merge `seq_latent_index`

- [x] Keep worker partial index shards keyed by canonical dataset shard ID.
- [x] Merge by copying or validating shard files into the global `outputs/<run_id>/seq_latent_index/` directory.
- [x] Reject duplicate canonical shard outputs unless their contents are byte/tensor identical.
- [x] Validate every expected shard output exists when `latents.seq_latent_index.enabled` is true.
- [x] Verification: add tests for disjoint shard copy, duplicate shard rejection, and disabled-index no-op behavior.

### Phase 9 Notes

- Extended `src/pipeline/distributed/pass1_merge.py` with `merge_seq_latent_index_shards()` for directory-level merging of worker `seq_latent_index/shard_<id>.pt` outputs.
- Worker index shards remain keyed by canonical dataset shard ID; the merge copies expected shard files into the canonical output directory using atomic copy/replace.
- Duplicate canonical shard outputs are allowed only when the existing and incoming files are byte-identical or tensor-equivalent after loading.
- Validation checks shard filename IDs, expected shard membership, tensor payload shape `[N, 2]`, `int32` dtype, positive sequence IDs, non-negative latent IDs, optional shard sequence bounds, and presence of every expected shard when enabled.
- Disabled `latents.seq_latent_index` mode is a no-op and does not create the output directory.
- Verification covers disjoint shard copy, identical duplicate acceptance, differing duplicate rejection, required expected outputs, disabled-index no-op behavior, and out-of-range sequence IDs.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py -q` -> `37 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py -q` -> `117 passed`.

## Phase 10 - Global Artifact Writer And Validation

- [x] Add a first-pass merge command that reads all worker pass-1 partials and writes canonical global artifacts.
- [x] Write merged outputs atomically.
- [x] Add a first-pass sanity report with tensor shapes, dtypes, finite checks, sequence ID min/max, context fill rates, `seq_repr` fill, and `logit_ctx` counts.
- [x] Record merge timing and peak CPU memory where practical.
- [x] Mark the manifest part status as completed only after every merged artifact passes validation.
- [x] Verification: run a tiny two-worker synthetic merge and confirm canonical artifact files are written in the expected locations.

### Phase 10 Notes

- Added `merge_pass1_worker_outputs()` as the global pass-1 merge writer for worker partials and canonical artifact paths.
- The writer loads all worker pass-1 partials, runs the existing merge helpers, writes `latent_stats.pt`, `top_ctx.pt`, `mid_ctx.pt`, `seq_repr.pt`, and `logit_ctx.pt` with temporary-file atomic saves, and merges `seq_latent_index` shards into the canonical output directory.
- Added `build_pass1_sanity_report()` and `distributed/reports/pass1_sanity_report.json` output with tensor shapes, dtypes, finite checks, sequence ID range, context fill rates, `seq_repr` fill, `logit_ctx` count summaries, index-shard merge report, elapsed time, and peak traced CPU memory.
- The writer reloads and validates written tensor artifacts before writing the sanity report and only then saves the manifest with `status: completed`.
- Verification includes a tiny synthetic two-worker pass-1 merge that writes every worker partial, merges index shards, confirms canonical artifact files, confirms no `.tmp` artifacts remain, validates sanity report contents, and checks persisted manifest status.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py -q` -> `38 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py -q` -> `118 passed`.

## Phase 11 - Testing And Verification

- [x] Add focused unit tests for every merge helper.
- [x] Add synthetic end-to-end pass-1 equivalence tests comparing single-process first-pass artifacts against two-worker merged artifacts.
- [x] Add one-worker compatibility tests proving one-worker distributed pass 1 can produce the same artifact schema as local single-process mode.
- [x] Add mathematically focused tests for Welford associativity, deterministic priority-reservoir uniformity, deterministic `seq_repr` cap mapping, and `logit_ctx` event top-K merging.
- [x] Add seed reproducibility tests proving `distributed.sampling_seed` controls `seq_repr` and `mid_ctx`, while changing only `run_id` does not change sampled rows.
- [x] Add `mid_ctx` candidate-pool tests for sufficient coverage, truncation detection, replay fallback, cleanup eligibility, and bounded-approximation rejection in paper-ready modes.
- [x] Add reduced real-data smoke tests once the synthetic tests pass.
- [x] Run focused tests for data loading, pass-1 worker execution, partial schemas, and merge logic.
- [x] Run the existing relevant store tests after merge implementation.
- [x] Document exact verification commands in this file after implementation.

### Phase 11 Notes

- Added explicit mathematical verification for Welford order invariance across worker partial order.
- Added deterministic priority-reservoir stability coverage proving `mid_ctx` selection is independent of worker partial order for fixed candidate priorities.
- Added seed reproducibility coverage for artifact-specific `seq_repr` cap seed derivation and `mid_ctx` candidate priorities; both paths are controlled by `distributed.sampling_seed` and `seq_repr` explicitly excludes `run_id`.
- Added a `logit_ctx` single event-stream equivalence test showing split-worker event top-K merge keeps the same global tokens/probabilities as a single concatenated stream.
- Added one-worker global pass-1 merge compatibility coverage proving one-worker distributed output keeps the canonical artifact schema and manifest completion behavior.
- Existing reduced real-data smoke coverage validates distributed pass-1 worker inputs against real temporary `.npy` dataset shards and cached shard indices.
- Existing `mid_ctx` candidate-pool tests cover filtering, deterministic priority selection, truncation failure, replay fallback reporting, and bounded-approximation mode reporting through the merge report.
- Verification: `python -m pytest tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py -q` -> `58 passed`.
- Verification: `python -m pytest tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_shard_table.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_assignments.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_interfaces.py tests/pipeline/test_distributed_pass1_partials.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/test_data_loader.py tests/store/test_mid_ctx_modes.py tests/store/test_neg_context_backend.py -q` -> `131 passed`.

---

## Open Questions

- Should `component_steps` be merged, recomputed, or treated as worker-local diagnostic metadata only?
- Should pass-1 workers write full partial stores, or should they stream per-component partials to reduce peak CPU RAM?
- Should worker output paths mirror canonical artifact names or use a versioned partial schema namespace from the start?

## Risks / Assumptions

- Exactness depends on stable global sequence IDs across all workers.
- Distributed `mid_ctx` must use deterministic priority-reservoir semantics; naive reservoir concatenation is not statistically equivalent.
- `seq_repr` capped mode is not reproducible if each worker samples independently.
- Top-K and event-top-K merges require deterministic tie-breaking to avoid small but confusing differences from single-process runs.
- The first implementation should prioritize exact small-run equivalence over maximum H100 throughput.
