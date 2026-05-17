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

- [ ] Add a worker entrypoint that can run pass 1 for only the shard IDs assigned in the Part 1 manifest.
- [ ] Ensure each worker initializes a local `DataLoader`, full local `Inference` model, full local `SAEBank`, and local stores.
- [ ] Ensure distributed workers construct `SAEBank` with exactly one worker-local device; no worker should split SAE layers across multiple physical GPUs.
- [ ] Keep the existing `src/main.py` and `pipeline.run()` path unchanged for normal single-process runs.
- [ ] Add a way for the worker to write first-pass outputs under `outputs/<run_id>/distributed/workers/worker_000/pass1/`.
- [ ] Preserve existing first-pass behavior inside each worker for stores that do not depend on final global statistics: `latent_stats`, `top_ctx`, `seq_repr`, `logit_ctx`, and `seq_latent_index`.
- [ ] During pass 1, write oversampled `mid_ctx` candidate pools instead of final `mid_ctx` rows.
- [ ] After merged global `latent_stats` exist, filter the candidate pools with final global `mean_seq` and `std_seq` to produce final `mid_ctx`.
- [ ] Add worker metadata with shard IDs, global sequence ID ranges, batch count, output paths, start/end time, and device.
- [ ] Verification: run a one-worker dry pass on synthetic shards and confirm the worker writes the expected partial artifact names and metadata.
- [ ] Verification: add a resource-construction unit test that fails if distributed worker mode passes multiple devices to `SAEBank`.

## Phase 2 - Stable Shard And Sequence IDs

- [ ] Consume the canonical full-dataset sequence ID table from Part 1.
- [ ] Teach the worker data path to load only assigned shards while preserving the global sequence IDs from the full dataset ordering.
- [ ] Avoid renumbering sequences locally inside each worker.
- [ ] Store each worker's shard ID ranges in its metadata and validate them against the manifest.
- [ ] Ensure `seq_latent_index` partial files preserve canonical shard IDs, not worker-local shard IDs.
- [ ] Add validation that worker shard assignments are disjoint and cover the intended `data.n_shards` set exactly.
- [ ] Verification: add synthetic multi-shard tests proving that a two-worker split emits the same global sequence IDs as a single-process loader over the same shards.
- [ ] Verification: add failure tests for stale global sequence tables, missing assigned shards, duplicated sequence IDs, and out-of-range worker sequence IDs.

## Phase 3 - Partial Artifact Schema

- [ ] Define partial checkpoint schemas for `latent_stats.partial.pt`, `top_ctx.partial.pt`, `mid_ctx_candidates.partial.pt`, `seq_repr.partial.pt`, and `logit_ctx.partial.pt`.
- [ ] Include metadata in every partial: schema version, run ID, worker ID, shard IDs, sequence ID min/max, config hash, device, and store-specific mode fields.
- [ ] Define a versioned `mid_ctx_candidates.partial.pt` schema with component IDs, latent IDs, sequence IDs, activation values, deterministic priorities, candidate-pool settings, truncation counters, and worker metadata.
- [ ] Keep tensor names close to the existing checkpoint names so merge code can reuse current store load/save logic where practical.
- [ ] Save partial artifacts atomically to avoid merge reading incomplete files.
- [ ] Add partial artifact validation before merge: tensor shapes, dtypes, finite values, expected component count, expected SAE width, and config hash.
- [ ] Verification: add round-trip tests for every partial artifact schema using tiny tensors.

## Phase 4 - Merge `latent_stats`

- [ ] Implement exact parallel Welford merges for token-level stats: `active_count`, `mean`, `m2`, `mean_abs`, and `m2_abs`.
- [ ] Implement exact parallel Welford merges for sequence-level stats: `seq_count`, `mean_seq`, and `m2_seq`.
- [ ] Preserve `component_steps` semantics or replace them with explicit merged batch/update counts in metadata if exact equality is not meaningful after worker splits.
- [ ] Validate merged counts are equal to the sum of worker counts.
- [ ] Validate no NaN or negative variance state is introduced.
- [ ] Verification: add tests that split a synthetic latent activation stream across workers and prove merged stats match single-process stats within numerical tolerance.

## Phase 5 - Merge `top_ctx`

- [ ] For each component and latent, concatenate all worker `top_ctx` rows.
- [ ] Select the global top-K sequence IDs by activation value using the same K as `config.latents.top_ctx.n_sequences`.
- [ ] Preserve deterministic tie-breaking, preferably by value descending then sequence ID ascending.
- [ ] Keep invalid sentinel rows zeroed.
- [ ] Validate merged sequence IDs are in global range and values are finite/non-negative.
- [ ] Verification: add tests where top contexts are split across workers and global top-K requires rows from multiple workers.

## Phase 6 - Merge `mid_ctx`

- [ ] Treat `mid_ctx.mode` as part of the artifact contract.
- [ ] Use the long-term distributed `mid_ctx` solution: oversampled deterministic candidate pools collected during pass 1, filtered after final global stats merge.
- [ ] Compute/merge global `latent_stats` first, then define the final mid band from global `mean_seq` and `std_seq`.
- [ ] During pass 1, workers collect candidate rows from a widened provisional band around the configured final mid band, controlled by `distributed.mid_ctx_candidate_pool.band_margin_sigma`.
- [ ] Default `distributed.mid_ctx_candidate_pool.enabled` to `true` for distributed modes.
- [ ] Default `distributed.mid_ctx_candidate_pool.band_margin_sigma` to `1.0`.
- [ ] Default `distributed.mid_ctx_candidate_pool.max_candidates_per_latent` to `max(256, 4 * num_ctx_sequences)`.
- [ ] Default `distributed.mid_ctx_candidate_pool.on_truncation` to `replay_fallback` for exact/paper-eligible modes.
- [ ] Allow `allow_bounded_approx` only in explicit experimental modes.
- [ ] Store enough candidate data to filter after stats merge: component, latent, global sequence ID, activation value, deterministic priority, and candidate-pool provenance.
- [ ] Use an explicit `distributed.sampling_seed`, stored in the manifest, as the root seed for reproducible sampling.
- [ ] Derive the `mid_ctx` priority seed by stable hashing over `distributed.sampling_seed`, artifact name, dataset fingerprint, band parameters, component, latent, and sequence ID.
- [ ] Do not use `run_id` as the sampling seed source; rerunning the same paper config with a new run ID should produce the same deterministic samples.
- [ ] After global stats merge, filter candidate rows by final global mid-band membership and select the best `n_sequences` valid examples per latent by deterministic priority.
- [ ] Preserve unbiased sampling over all globally valid mid-band examples while making distributed merge deterministic and reproducible.
- [ ] Detect candidate-pool truncation or insufficient coverage per latent and record candidate count, filtered valid count, selected count, truncation flag, and fill rate.
- [ ] In paper-eligible exact modes, handle candidate-pool coverage failure with `fail` or `replay_fallback`; do not silently mark bounded-approximate `mid_ctx` as exact.
- [ ] Keep a stats-aware replay fallback that replays assigned shards only when candidate-pool validation fails or exact mode explicitly requests guaranteed completeness.
- [ ] Classify `mid_ctx` candidate pools as large partial artifacts eligible for deletion only after final `mid_ctx.pt` validation succeeds and cleanup policy allows it.
- [ ] Make deterministic priority-reservoir the only paper-eligible distributed `mid_ctx` mode.
- [ ] Keep existing single-process `reservoir_cpu` available for non-distributed runs, but do not treat naive worker reservoir concatenation as exact.
- [ ] Preserve `reservoir_fill`, `reservoir_n` or replace them with equivalent distributed priority-reservoir metadata: candidate count, valid count, selected count, truncation flag, priority seed/hash version, band bounds, candidate-pool settings, and `num_ctx_sequences`.
- [ ] Validate fill counts and sequence ID ranges after merge.
- [ ] Verification: add statistical unit tests showing deterministic priority-reservoir selection is uniform over valid examples across many seeded trials.
- [ ] Verification: add exactness tests showing candidate-pool filtering equals a single global priority-reservoir pass when no candidate-pool truncation occurs.
- [ ] Verification: add tests showing truncation/coverage failures are detected and either fail or trigger replay fallback according to `on_truncation`.
- [ ] Verification: add replay-fallback tests showing fallback output equals a full global priority-reservoir pass.
- [ ] Verification: add tests proving closest-to-midpoint top-K is not used as the default distributed semantic unless explicitly configured as a biased experimental mode.

## Phase 7 - Merge `seq_repr`

- [ ] Move `seq_repr` capped sampling from per-worker random initialization to manifest-level deterministic sampling.
- [ ] Generate the capped sequence sample once from the global sequence ID table and an explicit `distributed.sampling_seed` stored in the manifest.
- [ ] Derive the `seq_repr` cap seed by stable hashing over `distributed.sampling_seed`, artifact name, dataset fingerprint, cap size, and total sequence count.
- [ ] Do not use `run_id` as the sampling seed source; new run IDs should not silently change `seq_repr` membership for the same config/data.
- [ ] Store `slot_to_id` and `id_to_slot` as tensor artifacts referenced by the manifest; avoid huge JSON mappings.
- [ ] Ensure every worker loads the same global mapping before first-pass `seq_repr.update()`.
- [ ] For uncapped mode, copy worker representations into rows indexed by global sequence ID.
- [ ] For capped mode, accept only sequences selected by the manifest-level cap and copy them into their global slots.
- [ ] Validate each selected sequence slot is written at most once and all worker writes agree with global ID bounds.
- [ ] Verification: add tests showing distributed capped and uncapped `seq_repr` merges match a single global slot mapping.
- [ ] Verification: add tests proving two workers with disjoint shards cannot generate different caps for the same run.
- [ ] Verification: add tests for cap determinism under fixed seed and cap changes under changed seed.

## Phase 8 - Merge `logit_ctx`

- [ ] Define exact event top-K merge semantics for `top_tokens`, `top_probs`, and `latent_counts`.
- [ ] Sum `latent_counts` across workers.
- [ ] For each latent, concatenate worker token/prob event rows and retain the configured top tokens by probability.
- [ ] Define deterministic tie-breaking for equal probabilities: probability descending, token ID ascending, then worker/candidate row order if still tied.
- [ ] Preserve current artifact meaning; do not switch to mean/aggregated per-token probabilities in the first distributed implementation.
- [ ] Validate token IDs are in vocabulary range and probabilities are finite.
- [ ] Verification: add synthetic tests where different workers contribute different top tokens for the same latent and the global merge keeps the expected event top-K.
- [ ] Verification: add tie-case tests proving deterministic ordering across worker split orders.
- [ ] Verification: add tests showing `latent_counts` sum exactly and event top-K values match a single-process event stream.

## Phase 9 - Merge `seq_latent_index`

- [ ] Keep worker partial index shards keyed by canonical dataset shard ID.
- [ ] Merge by copying or validating shard files into the global `outputs/<run_id>/seq_latent_index/` directory.
- [ ] Reject duplicate canonical shard outputs unless their contents are byte/tensor identical.
- [ ] Validate every expected shard output exists when `latents.seq_latent_index.enabled` is true.
- [ ] Verification: add tests for disjoint shard copy, duplicate shard rejection, and disabled-index no-op behavior.

## Phase 10 - Global Artifact Writer And Validation

- [ ] Add a first-pass merge command that reads all worker pass-1 partials and writes canonical global artifacts.
- [ ] Write merged outputs atomically.
- [ ] Add a first-pass sanity report with tensor shapes, dtypes, finite checks, sequence ID min/max, context fill rates, `seq_repr` fill, and `logit_ctx` counts.
- [ ] Record merge timing and peak CPU memory where practical.
- [ ] Mark the manifest part status as completed only after every merged artifact passes validation.
- [ ] Verification: run a tiny two-worker synthetic merge and confirm canonical artifact files are written in the expected locations.

## Phase 11 - Testing And Verification

- [ ] Add focused unit tests for every merge helper.
- [ ] Add synthetic end-to-end pass-1 equivalence tests comparing single-process first-pass artifacts against two-worker merged artifacts.
- [ ] Add one-worker compatibility tests proving one-worker distributed pass 1 can produce the same artifact schema as local single-process mode.
- [ ] Add mathematically focused tests for Welford associativity, deterministic priority-reservoir uniformity, deterministic `seq_repr` cap mapping, and `logit_ctx` event top-K merging.
- [ ] Add seed reproducibility tests proving `distributed.sampling_seed` controls `seq_repr` and `mid_ctx`, while changing only `run_id` does not change sampled rows.
- [ ] Add `mid_ctx` candidate-pool tests for sufficient coverage, truncation detection, replay fallback, cleanup eligibility, and bounded-approximation rejection in paper-ready modes.
- [ ] Add reduced real-data smoke tests once the synthetic tests pass.
- [ ] Run focused tests for data loading, pass-1 worker execution, partial schemas, and merge logic.
- [ ] Run the existing relevant store tests after merge implementation.
- [ ] Document exact verification commands in this file after implementation.

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
