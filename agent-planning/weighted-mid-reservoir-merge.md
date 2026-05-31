# Plan: Weighted Mid-Reservoir Merge

> **Goal:** Replace the oversized distributed `mid_ctx` candidate-pool merge with a compact weighted merge of per-worker `reservoir_cpu` summaries, preserving the scientific target while reducing merge memory, disk, and runtime.
>
> **Created:** 2026-05-31

---

## Phase 1 — Confirm Statistical Contract

- [x] Document the target: for each `(component, latent)`, approximate a uniform sample of `K` sequences from the union of all worker-eligible mid-band candidates.
- [x] Confirm each worker-local `reservoir_cpu` row is a uniform sample without replacement from that worker's eligible stream, with `reservoir_n` recording the stream size.
- [x] Define the merge rule: worker reservoirs contribute according to their `reservoir_n`; workers with more eligible examples should have proportionally higher chance to contribute final samples.
- [x] Decide deterministic merge randomness: use a stable hash/priority key over sampling seed, run/dataset fingerprint, component id, latent id, worker id, sequence id, slot id, and worker `reservoir_n`.
- [x] Verification: write down invariants for output shapes, `reservoir_fill`, `reservoir_n`, deterministic reproducibility, and expected weighted inclusion behavior.

### Phase 1 Findings

Statistical target:

- For each `(component, latent)`, the merged `mid_ctx` row should approximate a uniform sample of up to `K = num_ctx_sequences` sequence IDs from the union of all worker-local mid-band eligible sequence events for that row.
- The target remains the same downstream artifact contract as local `reservoir_cpu`: selected sequence IDs and activation values in `ctx_seq_idx` / `ctx_seq_val`, plus `reservoir_fill` and `reservoir_n` summaries.

Worker-local assumption:

- Each worker-local `reservoir_cpu` row is treated as an unbiased reservoir sample without replacement from that worker's eligible mid-band stream for the same `(component, latent)`.
- `reservoir_n[component, latent]` records the worker's eligible stream size `n_i`, while `reservoir_fill[component, latent]` records the retained row size `m_i <= K`.
- A retained local item therefore represents roughly `n_i / max(m_i, 1)` eligible stream items from that worker.

Weighted merge rule:

- For a fixed `(component, latent)`, collect retained worker reservoir slots where `n_i > 0`, `m_i > 0`, the slot is within `reservoir_fill`, the sequence ID is nonzero, and the activation value is finite.
- Assign each retained item a per-item weight `weight_i = n_i / max(m_i, 1)`.
- Compute an exponential-race key `key = -log(u) / weight_i`, where `u` is a deterministic hash-derived uniform value in `(0, 1]`.
- Select the `K` lowest keys. This gives workers with larger `reservoir_n` proportionally more contribution probability while accounting for the number of retained representatives they exported.

Deterministic randomness:

- Hash material should include the merge hash version, sampling seed, dataset/shard-table fingerprint, component id, latent id, worker id, sequence id, slot id, and worker `reservoir_n`.
- The merge must be independent of input partial ordering by sorting or otherwise canonicalizing worker identity before selection.
- Ties must be broken deterministically by `(key, sequence_id, worker_id, slot_id)`.

Output invariants:

- `ctx_seq_idx` has dtype `torch.int32` and shape `[component_count, d_sae, K]`.
- `ctx_seq_val` has a floating dtype and the same shape as `ctx_seq_idx`; invalid sentinel positions have sequence ID `0` and value `0`.
- `reservoir_fill` has shape `[component_count, d_sae]`, counts selected output slots, and satisfies `0 <= reservoir_fill <= K`.
- `reservoir_n` has dtype `torch.int64`, shape `[component_count, d_sae]`, and equals `sum_i reservoir_n_i` for each row, including rows with fewer than `K` selected samples.
- Rows with no eligible worker samples remain zero-filled with `reservoir_fill == 0` and `reservoir_n == sum_i n_i`.
- For the same inputs and sampling seed, merged output must be byte-stable across worker partial orderings.
- Across repeated deterministic seeds in simulation, expected worker contribution should scale approximately with worker `reservoir_n`, and workers with `reservoir_n == 0` should never contribute selected rows.

## Phase 2 — Artifact Contract

- [x] Add a compact mid-reservoir partial payload, or reuse existing `mid_ctx_candidates` partials in a compact mode containing only `ctx_seq_idx`, `ctx_seq_val`, `reservoir_fill`, and `reservoir_n`.
- [x] Preserve downstream merged `mid_ctx.pt` schema: `ctx_seq_idx`, `ctx_seq_val`, `ctx_type`, `mode`, `band_low_sigma`, `band_high_sigma`, `num_ctx_sequences`, `reservoir_fill`, and `reservoir_n`.
- [x] Add metadata indicating the merge mode, for example `mode: distributed_weighted_reservoir`.
- [x] Keep existing candidate-pool merge code available during transition for tests/backward compatibility.
- [x] Verification: round-trip partial validation catches missing/incorrect reservoir tensors and accepts compact reservoir partials.

### Phase 2 Findings

Artifact contract decision:

- Reuse the existing `mid_ctx_candidates` artifact name and `mid_ctx_candidates.partial.pt` filename for transition compatibility.
- Add a compact `mid_ctx_reservoir_payload()` helper for weighted merge mode. Its payload contains only `ctx_seq_idx`, `ctx_seq_val`, `ctx_type`, `mode`, `merge_source`, `reservoir_fill`, and `reservoir_n`.
- Use `merge_source: worker_local_reservoir` to distinguish compact reservoir partials from legacy candidate-pool partials.
- Keep `mode: reservoir_cpu` on worker-local compact partials; the merged global artifact mode will be set by the Phase 3/4 merge implementation.

Validation contract:

- Both compact reservoir partials and legacy candidate-pool partials must carry `ctx_seq_idx`, `ctx_seq_val`, `reservoir_fill`, and `reservoir_n`.
- Compact partials do not require or export `component_ids`, `latent_ids`, `sequence_ids`, `activation_values`, `priorities`, `candidate_pool_settings`, or `truncation_counters`.
- Candidate-pool validation remains available and still requires the candidate arrays, settings, truncation counters, and deterministic priority keys.
- Reservoir summary validation now checks tensor dtypes/shapes, non-negative `reservoir_n`, `0 <= reservoir_fill <= K`, finite context values, worker sequence ID bounds, and zero-valued invalid sentinels.

Verification added:

- Compact partial save/load round trip under the existing `mid_ctx_candidates` artifact name.
- Missing reservoir tensor rejection.
- Incorrect reservoir tensor shape rejection.
- Existing candidate-pool synthetic payloads were updated to include the reservoir summary tensors already emitted by the real writer.

## Phase 3 — Weighted Merge Algorithm

- [x] Implement a pure function that merges one `(component, latent)` row from multiple worker reservoirs.
- [x] For each worker row, read `n_i = reservoir_n[component, latent]` and selected `m_i = reservoir_fill[component, latent]`.
- [x] Ignore workers with `n_i == 0` or empty selected rows.
- [x] Assign each worker sample a deterministic weighted key so rows from worker `i` are selected with probability proportional to `n_i`.
- [x] Select final `K` rows by lowest weighted key, with deterministic tie-breaking by sequence id and worker id.
- [x] Set merged `reservoir_n` to `sum_i n_i`, and merged `reservoir_fill` to the selected count.
- [x] Verification: unit tests cover equal weights, imbalanced weights, empty workers, partially filled reservoirs, ties, and determinism across worker order.

### Phase 3 Findings

Implemented algorithm:

- Added `merge_mid_ctx_reservoir_row()` in `src/pipeline/distributed/pass1/context_merge.py`.
- The helper is CPU-only and side-effect free: it reads compact worker partial payload rows and returns merged sequence IDs, activation values, selected fill count, and summed `reservoir_n`.
- Worker partials are canonicalized by `metadata.worker_id`, so output is independent of input partial ordering.

Weighted key behavior:

- For each worker row, `worker_n = reservoir_n[component, latent]` is always added to the merged total.
- Rows with `worker_n <= 0` or `reservoir_fill <= 0` do not contribute selected samples.
- Only slots below `reservoir_fill` are considered; trailing tensor slots are ignored even if they contain nonzero values.
- Each retained item uses `weight = worker_n / max(reservoir_fill, 1)` and key `-log(u) / weight`.
- `u` is derived from SHA256 material containing `weighted-reservoir-v1`, sampling seed, dataset fingerprint, component id, latent id, worker id, sequence id, slot id, and worker `reservoir_n`.
- Selection uses the lowest keys with deterministic tie-breaking by `(key, sequence_id, worker_id, slot_id)`.

Verification added:

- Equal worker weights preserve all rows when capacity allows.
- Imbalanced worker weights favor the larger `reservoir_n` stream across deterministic seeded trials.
- Empty worker rows are ignored for selection while preserving summed `reservoir_n`.
- Partial fills ignore non-selected trailing slots.
- Forced key ties break by sequence ID, then worker ID, then slot ID.
- Reversing worker partial order produces identical output.
- A seeded simulation shows contribution ratios scale with `reservoir_n`.

## Phase 4 — Distributed Integration

- [x] Add a merge entry point such as `merge_mid_ctx_reservoir_partials()` beside `merge_mid_ctx_candidate_partials()`.
- [x] Wire Pass 1 writer/merge to choose weighted reservoir merge when configured, avoiding large candidate-pool tensors.
- [x] Add config for merge strategy, for example `distributed.mid_ctx_merge.mode: weighted_reservoir` or reuse `mid_ctx_candidate_pool.enabled: false` as the switch.
- [x] Ensure reports expose `merge_mode`, per-latent total `reservoir_n`, fill rates, and whether any worker reservoirs were empty.
- [x] Verification: distributed Pass 1 tests prove merged output is compact, schema-compatible, and deterministic.

### Phase 4 Findings

Distributed integration:

- Added `distributed.mid_ctx_merge.mode` with allowed values `weighted_reservoir` and `candidate_pool`; default is `weighted_reservoir`.
- Added optional `distributed.mid_ctx_merge.sampling_seed`; when omitted, the writer uses `manifest.sampling_seed`.
- Added `load_and_merge_mid_ctx_reservoir_partials()` and `merge_mid_ctx_reservoir_partials()` beside the existing candidate-pool merge entry points.
- `merge_pass1_worker_outputs()` now dispatches by merge mode: weighted reservoir uses compact summaries, while candidate pool still uses the existing stats-aware candidate-pool path and truncation policy.
- The pass-1 merge CLI accepts `--mid-ctx-merge-mode` and `--mid-ctx-sampling-seed` overrides.

Worker integration:

- `initialize_pass1_worker_resources()` only calls `configure_mid_ctx_candidate_pool()` in `candidate_pool` mode.
- `save_pass1_partials()` writes compact `mid_ctx_reservoir_payload()` partials in `weighted_reservoir` mode and keeps `mid_ctx_candidates_payload()` in `candidate_pool` mode.
- The compatibility facade in `pipeline.distributed.worker` now syncs the compact reservoir payload helper.

Reports and schema:

- The merged weighted artifact preserves the downstream `mid_ctx.pt` schema with `ctx_seq_idx`, `ctx_seq_val`, `ctx_type`, `mode`, `band_low_sigma`, `band_high_sigma`, `num_ctx_sequences`, `reservoir_fill`, and `reservoir_n`.
- Weighted merged artifacts use `mode: distributed_weighted_reservoir` and include a `merge_report` with `merge_mode`, `priority_mode`, hash version, selected counts, total per-row `reservoir_n`, fill rates, and empty-worker row counts.
- The pass-1 sanity report now includes a JSON-safe `mid_ctx_merge` summary with merge mode, total `reservoir_n`, selected count, nonzero rows, and empty-worker indicators.

Verification added:

- Full weighted reservoir partial merge preserves schema and reports.
- Weighted reservoir partial load/merge round trip works under the existing partial artifact name.
- Writer output uses weighted reservoir by default and writes `distributed_weighted_reservoir` merged `mid_ctx`.
- Worker tests prove weighted mode avoids candidate-pool widening and compact partials do not export candidate arrays.
- Candidate-pool mode still calls the widening path and remains available.

## Phase 5 — Statistical Validation

- [ ] Add simulation tests where workers have known eligible stream sizes and synthetic local reservoirs.
- [ ] Compare empirical inclusion rates across many seeds against expected worker weights.
- [ ] Test that a worker with 10x `reservoir_n` contributes about 10x probability mass over repeated seeded merges.
- [ ] Verify no sequence is selected from a worker with `reservoir_n == 0`.
- [ ] Verification: statistical tests use deterministic seeds and loose enough bounds for stable CI while catching major weighting bugs.

## Phase 6 — H100 Smoke Benchmark

- [ ] Run a 64-shard 4xH100 Pass 1 using `reservoir_cpu`, `batch_size: 4096`, and weighted reservoir merge.
- [ ] Measure worker runtime and compact partial sizes; expected partial size should be close to final `mid_ctx` tensors, not multi-GB candidate pools.
- [ ] Run Pass 1 merge and record wall time, peak RSS, and merged artifact sizes.
- [ ] Compare against observed candidate-pool data: pool64 produced ~2.5GiB/worker and merge exceeded 20 minutes on 64 shards.
- [ ] Verification: merge completes quickly, merged `mid_ctx.pt` passes shape/finiteness/fill checks, and no huge `mid_ctx_candidates.partial.pt` files are produced.

## Phase 7 — Full-Run Rollout

- [ ] Update `config_examples/h100-8x-distributed-simple-exact.yaml` to use weighted reservoir merge after smoke validation.
- [ ] Update `docs/h100-8x-full-run-guide.md` with the new merge mode, expected artifact sizes, and monitoring commands.
- [ ] Document scientific status: weighted merge is the paper-eligible distributed approximation/exact-in-distribution merge for `reservoir_cpu` local reservoirs.
- [ ] Run a larger 256-shard smoke before full 3030-shard execution.
- [ ] Verification: compare selected seeds/discovery outputs against a bounded candidate-pool smoke if available.

---

## Developer Handoff Notes

Primary files to inspect or edit:

- `src/pipeline/distributed/pass1_partials.py`: add/validate compact reservoir partial payloads and metadata.
- `src/pipeline/distributed/pass1/context_merge.py`: implement the weighted row merge and distributed merge entry point.
- `src/pipeline/distributed/pass1/writer.py`: choose weighted reservoir merge versus candidate-pool merge and write the merged `mid_ctx.pt`.
- `src/pipeline/distributed/pass1/worker.py`: stop widening `mid_ctx` candidate pools when weighted reservoir merge is selected.
- `src/config.py`: add the merge strategy config and validation.
- `tests/pipeline/test_distributed_pass1_partials.py`: compact partial schema tests.
- `tests/pipeline/test_distributed_pass1_merge.py`: weighted merge and statistical tests.
- `tests/pipeline/test_distributed_worker.py`: worker wiring/config tests.
- `config_examples/h100-8x-distributed-simple-exact.yaml` and `docs/h100-8x-full-run-guide.md`: update only after smoke validation.

Suggested config shape:

```yaml
distributed:
  mid_ctx_merge:
    mode: "weighted_reservoir"  # "weighted_reservoir" | "candidate_pool"
    sampling_seed: 0            # optional; defaults to distributed.sampling_seed
```

Candidate-pool settings can remain for backward compatibility, but `weighted_reservoir` should not call `configure_mid_ctx_candidate_pool()` or export large `component_ids/latent_ids/sequence_ids/activation_values/priorities` arrays.

Preferred partial contract:

```python
{
    "ctx_seq_idx": IntTensor[component_count, d_sae, k],
    "ctx_seq_val": FloatTensor[component_count, d_sae, k],
    "reservoir_fill": IntTensor[component_count, d_sae],
    "reservoir_n": LongTensor[component_count, d_sae],
    "ctx_type": "mid",
    "mode": "reservoir_cpu",
    "merge_source": "worker_local_reservoir",
}
```

Weighted-key implementation guidance:

- Start with a pure-Python/Torch CPU implementation for correctness. The input is compact enough that native code should not be needed initially.
- For each selected worker reservoir item, compute a uniform deterministic `u in (0, 1]` from stable hash material.
- Use an exponential-race style key such as `key = -log(u) / weight`, where `weight = n_i / max(m_i, 1)` for each item from worker `i`. Select the `K` smallest keys.
- The `n_i / m_i` per-item weight accounts for the fact that each retained local reservoir item represents about `n_i / m_i` eligible stream items.
- Tie-break deterministically by `(key, sequence_id, worker_id, slot_id)`.
- Set merged `reservoir_n[component, latent] = sum_i n_i` even if fewer than `K` items are selected.

Acceptance criteria:

- The compact reservoir partial for a 64-shard 4xH100 smoke is small, not multi-GB per worker.
- Pass 1 merge peak RSS is comfortably below the candidate-pool merge observed at ~60-130GB.
- Merge completes on the 64-shard smoke in seconds or a few minutes, not >20 minutes.
- Merged `mid_ctx.pt` schema remains compatible with negative context and display/probe readers.
- Statistical simulation shows worker contribution scales with `reservoir_n`.
- Results are deterministic for the same sampling seed and manifest.

Known data motivating this plan:

- `reservoir_cpu` worker runtime with mid updates active: about `6.4-6.6s/it` on 4xH100 smoke.
- No-mid baseline profile: about `4.62s/batch`.
- `mid_ctx_update` profile cost: about `1.6s/batch`.
- Candidate-pool `max_candidates_per_latent=64` still wrote about `2.5GiB/worker` and `331.6M` total candidates for only 64 shards.
- Pool64 had massive truncation (`~1.26M` affected component-latent groups per worker), so it is not exact enough as a candidate-pool strategy.
- Candidate-pool merge exceeded 20 minutes and reached high RSS even at 64 shards.

## Open Questions

- Should weighted reservoir merge replace candidate-pool merge entirely, or be a separate mode until validated on H100? Recommendation: separate mode first, then promote after 64/256-shard validation.
- What exact weighted-key formula should be used for deterministic weighted sampling without replacement? Recommendation: start with exponential-race keys `-log(u) / (n_i / m_i)`.
- Should duplicate sequence IDs across workers be deduplicated during merge, or can worker shard partitioning guarantee uniqueness?
- Should `reservoir_cpu` use deterministic RNG per candidate in the future to make local reservoirs exactly reproducible?

## Risks / Assumptions

- Worker local reservoirs are assumed to be unbiased summaries of their eligible streams.
- A weighted merge of already sampled reservoirs is exact in distribution only if the weighting algorithm is correct and local reservoirs are representative.
- Current worker reservoirs are RNG-based and may not be exactly reproducible across reruns unless local RNG seeding is controlled.
- Weighted merge solves candidate-pool explosion, but it does not reduce Pass 1 `mid_ctx_update` cost; separate knobs like `warmup_batches` or `update_every_batches` may still be useful.
