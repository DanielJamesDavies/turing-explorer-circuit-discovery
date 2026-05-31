# Plan: GPU Priority Reservoir Mid Context

> **Goal:** Add a scientifically uniform, deterministic, GPU-friendly mid-context sampling mode that preserves the statistical target of `reservoir_cpu` while reducing Pass 1 overhead.
>
> **Created:** 2026-05-31

---

## Outcome

The local `gpu_topk_mid` and `gpu_priority_reservoir` modes were removed after
H100 smoke testing showed they were either scientifically biased or operationally
slower/less reliable than `reservoir_cpu`. The supported local `mid_ctx` path is
now `reservoir_cpu` only. The distributed merge-side deterministic priority
selection remains, because it is part of candidate-pool merging rather than a
local context update mode.

Keep this document as historical context for why the GPU modes were not promoted.

## Phase 1 — Confirm Sampling Contract

- [x] Document the intended statistical contract: uniform sampling over eligible mid-band `(component, latent, sequence)` candidates.
- [x] Confirm how `reservoir_cpu` currently defines eligibility, scores, fill counts, and persisted metadata.
- [x] Confirm how distributed candidate-pool merge already uses deterministic priorities and where that behavior can be reused.
- [x] Decide whether the new mode should be named `gpu_priority_reservoir` and whether it should become the recommended full-run mode after validation.
- [x] Verification: write down the exact invariants the implementation and tests must satisfy before coding.

### Phase 1 Findings

The intended statistical contract is:

- For each `(component, latent)` pair, sample uniformly without replacement from eligible sequence candidates.
- A candidate is exactly one `(component, latent, sequence)` whose per-sequence latent score is inside the configured mid band.
- The selected set should be distributionally equivalent to `reservoir_cpu` for the same eligible candidate set and capacity, but deterministic for a fixed seed/config.
- Selection must not prefer scores near the band midpoint; that behavior belongs only to `gpu_topk_mid`.

Confirmed current `reservoir_cpu` behavior:

- `Context.update_component()` routes all non-`gpu_topk_mid` mid modes to `_update_mid_reservoir`.
- Scores are produced by `compute_seq_scores(top_acts, top_indices, d_sae)`, giving `[d_sae, batch]` mean activation per latent per sequence.
- Eligibility uses strict bounds: `score > mean_seq + band_low_sigma * std_seq` and `score < mean_seq + band_high_sigma * std_seq`, with `std_seq` clamped to `1e-6` in `_mid_band_bounds`.
- The native `mid_reservoir` extension applies Algorithm R per latent on CPU. It increments `reservoir_n` for every eligible in-band candidate, fills up to `num_ctx_sequences`, then replaces a slot with probability `num_ctx_sequences / reservoir_n`.
- `reservoir_fill` is the number of selected slots per `(component, latent)`, capped by `num_ctx_sequences`.
- Persisted `mid_ctx.pt` metadata currently includes `ctx_seq_idx`, `ctx_seq_val`, `ctx_type`, `mode`, `band_low_sigma`, `band_high_sigma`, `num_ctx_sequences`, `reservoir_fill`, and `reservoir_n`.

Confirmed distributed priority behavior:

- Worker partial export already has `mid_ctx_candidates_payload()` with deterministic `_candidate_priorities()`.
- Priority hash material currently includes `MID_CTX_PRIORITY_HASH_VERSION`, `sampling_seed`, artifact name, dataset fingerprint, final/candidate band settings, candidate pool margin, `num_ctx_sequences`, component id, latent id, and sequence id.
- Distributed merge filters exported candidates against global `mean_seq/std_seq` bounds, then selects the lowest priority values per `(component, latent)`.
- Merge tie behavior is deterministic because candidates are first sorted by sequence id, then stably sorted by priority.
- Merge reports already expose candidate counts, valid counts, selected counts, truncation counters, bounded approximation state, and priority metadata.

Phase 1 decisions:

- The new local mode should be named `gpu_priority_reservoir`.
- It should not become the recommended full-run mode until Phase 5 validates performance and distributional behavior on the H100 smoke benchmark.
- The implementation should reuse the distributed priority contract where practical, but with a vectorized hot path for local Pass 1 updates rather than Python per-candidate hashing.

Implementation/test invariants:

- `gpu_priority_reservoir` must preserve the `reservoir_cpu` eligibility rule exactly, including strict inequalities and the `std_seq` floor.
- For a fixed seed/config/candidate identity set, selected candidates must be independent of batch/update order.
- For each `(component, latent)`, `reservoir_n` must equal the total number of eligible candidates observed under the final eligibility rule.
- For each `(component, latent)`, `reservoir_fill` must equal `min(reservoir_n, num_ctx_sequences)` except for empty/invalid sentinel slots.
- `ctx_seq_idx` and `ctx_seq_val` must keep the same shapes, dtypes, sentinel-zero behavior, and persisted schema expected by downstream readers.
- Priority identity must include at least sampling seed, component id, latent id, sequence id, and the relevant band/config fingerprint material used by distributed partial priorities.
- Lower priority wins. Ties must be broken deterministically, preferably by sequence id and then stable component/latent identity.
- Distributed merge must produce the same selected set as a single global priority selection when every worker exports all candidates needed for exact selection.
- Any worker-local candidate pool truncation that can affect exact global priority selection must be visible through counters/report fields and obey the configured truncation policy.

## Phase 2 — Core Implementation

- [x] Extend `MidCtxConfig.validate_mode` to accept `gpu_priority_reservoir`.
- [x] Add a `Context._update_mid_gpu_priority_reservoir` path selected from `update_component`.
- [x] Keep the same mid-band eligibility calculation as `reservoir_cpu`.
- [x] Generate deterministic per-candidate priorities from stable identifiers: sampling seed, component id, latent id, sequence id, and relevant band/config fingerprint material.
- [x] Prefer a vectorized tensor implementation for priority generation and per-latent selection; avoid per-candidate Python loops in the hot Pass 1 update path.
- [x] Keep the `num_ctx_sequences` lowest priorities per latent while preserving `ctx_seq_idx`, `ctx_seq_val`, `reservoir_fill`, and `reservoir_n` semantics.
- [x] Avoid changing downstream artifact shapes or reader assumptions.
- [x] Verification: run focused unit tests for all three mid modes and check lints for edited files.

### Phase 2 Findings

- Implemented `gpu_priority_reservoir` as a local mid-context mode accepted by config validation.
- Added an internal `_priority_val` tensor for the new mode only; it is not persisted in `mid_ctx.pt`, so public artifact schema remains unchanged.
- Priority generation uses a vectorized integer SplitMix-style hash over `distributed.sampling_seed`, band settings, `num_ctx_sequences`, component id, latent id, and sequence id.
- Priority keys are retained as non-negative `int64` values instead of being compressed to `float32`, avoiding avoidable precision loss and tie amplification.
- The update path computes the same strict in-band mask as `reservoir_cpu`, increments `reservoir_n` for new eligible candidates, and keeps the lowest priority keys per latent with `torch.topk(..., largest=False)`.
- Focused verification passed: `python -m pytest tests/store/test_mid_ctx_modes.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass1_partials.py -q`.
- Linter check reported no issues for edited files.

## Phase 3 — Distributed Semantics

- [x] Ensure worker partial payloads correctly record `source_mid_mode: gpu_priority_reservoir`.
- [x] Confirm that candidate-pool merge remains exact for deterministic priorities when workers keep enough candidates.
- [x] Add or update tests showing priority-reservoir partials merge into the same selected set as a single combined priority selection.
- [x] Add a guard or report field that makes truncation visible when worker-local candidate pools are too small for exact global selection.
- [x] Verification: run distributed Pass 1 merge tests and candidate-pool tests.

### Phase 3 Findings

- Worker setup now copies `manifest.sampling_seed` onto `mid_ctx._priority_seed` before Pass 1 allocation, so distributed workers do not depend on import-time config seed state.
- Worker-local `gpu_priority_reservoir` and exported `mid_ctx_candidates` now share the same `splitmix64-v1` deterministic priority material: sampling seed, artifact name, dataset fingerprint, final band settings, candidate band settings, band margin, final `num_ctx_sequences`, component id, latent id, and sequence id.
- Candidate partials continue to record `candidate_pool_settings.source_mid_mode`, so `gpu_priority_reservoir` is visible in worker payload metadata.
- Candidate partial export now reports worker-local truncation as `max(reservoir_n - max_candidates_per_latent, 0)` in `truncation_counters`.
- Added tests for single global priority equivalence, deterministic sequence-id tie-breaking when priority keys collide, source-mode metadata, priority export consistency, truncation visibility, and worker seed propagation.
- Focused verification passed: `python -m pytest tests/store/test_mid_ctx_modes.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass1_partials.py -q`.
- Linter check reported no issues for edited files.

## Phase 4 — Statistical Validation

- [x] Add a deterministic unit test showing `gpu_priority_reservoir` selects the same candidates regardless of batch/update order.
- [x] Add a simulation test comparing priority-reservoir inclusion frequencies against uniform sampling expectations across many seeds.
- [x] Add a test showing `gpu_topk_mid` intentionally differs by midpoint distance, so the new mode is not confused with central-example selection.
- [x] Define acceptance thresholds for empirical uniformity checks before relying on the mode in full runs.
- [x] Verification: run the new statistical tests repeatedly enough to catch seed/order bugs without making CI flaky.

### Phase 4 Findings

- Added a stronger multi-update order-invariance test for `gpu_priority_reservoir` using the same candidate set split across reordered updates.
- Added a deterministic 512-seed inclusion-frequency simulation over 8 equally eligible candidates with 2 selected per seed.
- Acceptance threshold for the unit simulation is max absolute deviation <= 37.5% of the expected inclusion count per candidate. This is intentionally loose enough for stable CI but strict enough to catch obvious seed/priority bias or ordering bugs.
- Added an explicit contrast test showing `gpu_topk_mid` selects band-midpoint examples while `gpu_priority_reservoir` selects by pseudo-random priority for the same eligible candidates.
- Focused verification passed as part of the combined priority-reservoir suite: `python -m pytest tests/store/test_mid_ctx_modes.py tests/pipeline/test_distributed_pass1_merge.py tests/pipeline/test_distributed_worker.py tests/pipeline/test_distributed_pass1_partials.py -q`.
- Linter check reported no issues for edited files.

## Phase 5 — H100 Smoke Benchmark

- [ ] Prepare a 64-shard 4xH100 profile with `batch_size: 4096`, worker thread limits of `4`, local `/root/outputs`, and `latents.mid_ctx.mode: gpu_priority_reservoir`.
- [ ] Run Pass 1 only and compare pre-warmup and post-warmup batch timings against the known `reservoir_cpu` baseline.
- [ ] Merge Pass 1 artifacts and inspect `mid_ctx.pt` for valid shapes, fill rates, nonzero slots, finite values, and expected mode metadata.
- [ ] Compare basic mid-context distribution summaries against a matching `reservoir_cpu` 64-shard run.
- [ ] Verification: record timings, VRAM notes, fill statistics, and any truncation/candidate-pool warnings.

### Phase 5 Runbook

Use a smoke-only warmup value so the 64-shard profile actually exercises `mid_ctx`.
Do not carry this reduced warmup into the full publication run.

```bash
source /root/venvs/turing/bin/activate
cd /workspace/turing
export OUTPUT_BASE=/root/outputs/gpu-priority-smoke
mkdir -p "$OUTPUT_BASE"

cp config_examples/h100-8x-distributed-simple-exact.yaml config-profile-4x-gpu-priority-smoke.yaml

python - <<'PY'
from pathlib import Path
import yaml

p = Path("config-profile-4x-gpu-priority-smoke.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["data"]["n_shards"] = 64
cfg["data"]["batch_size"] = 4096
cfg["distributed"]["worker_count"] = 4
cfg["distributed"]["devices"] = [0, 1, 2, 3]
cfg["distributed"]["mid_ctx_candidate_pool"]["on_truncation"] = "fail"
cfg["latents"]["mid_ctx"]["mode"] = "gpu_priority_reservoir"
cfg["latents"]["mid_ctx"]["warmup_batches"] = 2
cfg["latents"]["neg_ctx"]["devices"] = [0, 1, 2, 3]
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

cp config-profile-4x-gpu-priority-smoke.yaml config.yaml

PYTHONPATH=src:src python - <<'PY'
from config import config
print("n_shards:", config.data.n_shards)
print("batch_size:", config.data.batch_size)
print("workers:", config.distributed.worker_count, config.distributed.devices)
print("mid_ctx:", config.latents.mid_ctx.mode, "warmup", config.latents.mid_ctx.warmup_batches)
print("truncation:", config.distributed.mid_ctx_candidate_pool.on_truncation)
PY

PYTHONPATH=src:src python -m pipeline.distributed.controller \
  --config config-profile-4x-gpu-priority-smoke.yaml \
  --output-base "$OUTPUT_BASE" \
  --mode distributed_simple_exact \
  --worker-count 4 \
  --devices 0,1,2,3 \
  --dry-run | tee /tmp/gpu-priority-smoke-dry-run.txt

export RUN_ROOT=$(awk '/^output_root:/ {print $2}' /tmp/gpu-priority-smoke-dry-run.txt)
export MANIFEST="$RUN_ROOT/distributed/manifest.json"
test -f "$MANIFEST" && echo "manifest ok: $MANIFEST"
```

Run Pass 1 workers:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

for W in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$W PYTHONPATH=src:src python -m pipeline.distributed.worker \
    --manifest "$MANIFEST" \
    --phase pass1 \
    --worker-id "$W" \
    > "$RUN_ROOT/distributed/pass1_worker_$(printf '%03d' "$W")_gpu_priority.log" 2>&1 &
done

wait
```

Inspect timings and merge:

```bash
grep -H "Latent Stats & Ctx: 100%" "$RUN_ROOT"/distributed/pass1_worker_*_gpu_priority.log || true

PYTHONPATH=src:src python -m pipeline.distributed.pass1_merge \
  --manifest "$MANIFEST" | tee "$RUN_ROOT/distributed/pass1_merge.log"
```

Inspect the merged `mid_ctx` artifact:

```bash
python - <<'PY'
import os
import torch
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
mid = torch.load(root / "mid_ctx.pt", map_location="cpu", weights_only=False)
idx = mid["ctx_seq_idx"]
val = mid["ctx_seq_val"]
fill = mid["reservoir_fill"]
n = mid["reservoir_n"]

print("mode:", mid.get("mode"))
print("idx shape:", tuple(idx.shape))
print("val shape:", tuple(val.shape))
print("nonzero slots:", int((idx != 0).sum()))
print("finite values:", bool(torch.isfinite(val).all()))
print("fill mean/max:", float(fill.float().mean()), int(fill.max()))
print("reservoir_n mean/max:", float(n.float().mean()), int(n.max()))
print("full slots:", int((fill == idx.shape[-1]).sum()))
PY
```

## Phase 6 — Full Run Rollout

- [ ] Update the canonical 4x/8x run guidance or config comments only after the smoke benchmark passes.
- [ ] Run the full 4xH100 pipeline using `gpu_priority_reservoir`.
- [ ] Monitor Pass 1 logs through warmup to confirm the post-warmup slowdown is materially reduced.
- [ ] Continue through NegCtx, Pass 2, candidate selection, discovery workers, and merge.
- [ ] Verification: summarize final artifact paths, timings, mid-context validation stats, and downstream discovery differences versus the previous reservoir baseline if available.

---

## Implementation Notes

- Primary files to inspect or edit:
  - `src/config.py` for the allowed `latents.mid_ctx.mode` values and any new priority seed config.
  - `src/store/context.py` for the new update mode and persisted `mid_ctx` metadata.
  - `src/pipeline/distributed/pass1_partials.py` for candidate-priority export and metadata.
  - `src/pipeline/distributed/pass1/context_merge.py` for distributed priority merge semantics and truncation reporting.
  - `tests/store/test_mid_ctx_modes.py` for focused mode behavior tests.
  - `tests/pipeline/test_distributed_pass1_merge.py` and `tests/pipeline/test_distributed_worker.py` for distributed semantics.
- The new mode should preserve the `reservoir_cpu` eligibility rule exactly: compute per-sequence latent scores, apply `mean_seq + band_low_sigma * std_seq < score < mean_seq + band_high_sigma * std_seq`, and sample only those candidates.
- The new mode should differ from `gpu_topk_mid`: selection priority must be pseudo-random uniform, not distance from the band midpoint.
- The persisted `mid_ctx.pt` schema should remain compatible with downstream readers: `ctx_seq_idx`, `ctx_seq_val`, `ctx_type`, `mode`, `band_low_sigma`, `band_high_sigma`, `num_ctx_sequences`, `reservoir_fill`, and `reservoir_n`.
- If priority values must be retained during local updates, add an internal tensor such as `_priority_val` rather than changing the public artifact schema unless downstream consumers need it.
- For exact distributed behavior, the priority used during worker collection and merge must be derived from the same stable candidate identity and seed material.
- Use deterministic tie-breaking when priorities collide, for example by sorting `(priority, sequence_id)` or `(priority, sequence_id, component_id, latent_id)`.
- Acceptance criteria:
  - `gpu_priority_reservoir` is accepted by config validation.
  - It produces valid non-empty mid-context artifacts on a small run.
  - Reordering batches does not change selected candidates for the same seed and candidate set.
  - Inclusion frequencies across many seeds are statistically consistent with uniform sampling.
  - Distributed merge selects the same candidates as a single global priority selection when no truncation occurs.
  - H100 Pass 1 post-warmup timing improves materially versus `reservoir_cpu` while preserving downstream artifact compatibility.

## Open Questions

- Should the priority seed be a new explicit config value, or reuse existing distributed sampling seed material?
- Answered for the current implementation: priorities are stored internally as `_priority_val` during local `gpu_priority_reservoir` updates, but they are not persisted in `mid_ctx.pt`.
- Is exact global priority selection required for non-distributed local runs, or is worker-local exactness sufficient for the current pipeline?
- What bounded truncation threshold is acceptable for worker candidate pools in a full publication run?

## Risks / Assumptions

- Deterministic hash priorities are scientifically valid only if they behave like independent uniform random variables for each eligible candidate.
- Worker-local truncation can bias the global result if too few candidates per latent are retained before merge.
- A pure PyTorch priority implementation may still have overhead if priority generation is CPU/Python-heavy; performance depends on vectorizing this path.
- Downstream code assumes the existing `mid_ctx.pt` schema, so the persisted artifact should remain backward-compatible.
