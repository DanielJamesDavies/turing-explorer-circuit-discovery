# Plan: H100 Full Run Rollout

> **Goal:** Bring up the 8xH100 RunPod environment safely, validate the device/software stack, tune run parameters from measured bottlenecks, execute partial validation runs, and only then launch a full `distributed_simple_exact` run.
>
> **Created:** 2026-05-24

---

## Phase 1 - Pod Bring-Up And Environment Capture

- [ ] Start the RunPod pod with the `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` template, 8x H100 SXM, and about 2.5 TB disk.
- [ ] Add the SSH public key to RunPod and confirm SSH access to the pod.
- [ ] Clone or upload the repo to the pod, then check out the intended commit.
- [ ] Create and activate a clean venv on the pod.
- [ ] Install the CUDA 12.8 hosted profile:

```bash
pip install -r requirements-cu128.txt
```

- [ ] Record the environment before any expensive run:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
nvcc --version
nvidia-smi
df -h
```

- [ ] Save this output under a runbook note or `outputs/h100_environment_summary.txt`.
- [ ] Verification: confirm Torch reports `2.8.0+cu128`, `torch.version.cuda` is `12.8`, `torch.cuda.device_count()` is `8`, and `nvidia-smi` shows 8 H100s.

## Phase 2 - Device Isolation And Native Build Tests

- [ ] Confirm each physical GPU can be isolated and seen as one logical `cuda:0`:

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -c "import torch; print('physical', $i, 'visible', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
done
```

- [ ] Rebuild native extensions on the H100 host:

```bash
cd src/native
python setup.py build_ext --inplace
cd ../..
```

- [ ] Run native and reducer tests:

```bash
python -m pytest src/native/tests/test_topk.py src/native/tests/test_reduce.py -q
python -m pytest tests/store/test_top_coactivation_modes.py -q
```

- [ ] Run the distributed controller/device tests on the pod:

```bash
PYTHONPATH=src python -m pytest tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_controller.py tests/pipeline/test_distributed_devices.py -q
```

- [ ] Verification: no native build mismatch, no fallback to incompatible reducer signatures, and every worker can be isolated to exactly one H100.

## Phase 3 - Data, Model, And Dry-Run Preflight

- [ ] Place the model at `models/TuringLLM/model_1722550239_03986.pt`.
- [ ] Place the SAE bank at `models/TuringLLM/SAE`.
- [ ] Place dataset shards under `data`.
- [ ] Confirm expected storage use before planning:

```bash
du -sh models data . 2>/dev/null
df -h
```

- [ ] Run the 8-worker controller dry run:

```bash
PYTHONPATH=src python -m pipeline.distributed.controller \
  --config config_examples/h100-8x-distributed-simple-exact.yaml \
  --mode distributed_simple_exact \
  --worker-count 8 \
  --devices 0,1,2,3,4,5,6,7 \
  --dry-run
```

- [ ] Inspect the manifest path, output root, worker commands, per-worker `CUDA_VISIBLE_DEVICES`, pass-1 shard balance, pass-2 replay assignment, and discovery task estimates.
- [ ] Verification: dry run produces one worker per physical GPU, each worker-local device is `cuda:0`, and `persist.build_search_cache_after_pipeline` remains false.

## Phase 4 - Conservative Parameter Baseline

- [ ] Start from `config_examples/h100-8x-distributed-simple-exact.yaml` as the exactness-first baseline.
- [ ] Keep these initial safety settings:
  - `distributed.mode: distributed_simple_exact`
  - `distributed.cleanup_policy: keep_all`
  - `distributed.strict_equivalence: true`
  - `hardware.memory: fast`
  - `hardware.compile: true`
  - `hardware.keep_model_loaded_for_neg_ctx: true`
  - `data.batch_size: 512`
  - `latents.neg_ctx.backend: multi_gpu_exact`
  - `latents.neg_ctx.max_repr_seqs: 200000`
  - `latents.top_coactivation.dump_device: gpu`
  - `latents.top_coactivation.reduce_backend: single_process`
  - `discovery.n_seeds: 16`
- [ ] Define the first tuning levers before changing them:
  - Increase or decrease `data.batch_size` based on peak VRAM and throughput.
  - Compare `hardware.keep_model_loaded_for_neg_ctx: true` vs `false` if VRAM pressure appears.
  - Compare `latents.neg_ctx.backend: multi_gpu_exact` vs `multi_gpu_index_sharded_exact` if ANN memory or duplicated index cost is high.
  - Compare pass-2 reducer settings only after simple exact reduce has produced bottleneck evidence.
  - Increase `discovery.n_seeds` only after discovery throughput and circuit quality look healthy.
- [ ] Verification: no tuning change is promoted unless it has a saved run ID, config hash, benchmark numbers, and artifact sanity reports.

## Phase 5 - One-Worker Correctness Gate

- [ ] Create or override a one-worker H100 config using the same data slice and exact mode:
  - `distributed.worker_count: 1`
  - `distributed.devices: [0]`
  - reduce `data.n_shards` if doing a reduced slice first.
- [ ] Run the one-worker distributed path before trusting multi-worker output.
- [ ] Run or reuse a `single_process` oracle for the same reduced slice.
- [ ] Compare canonical artifacts:
  - `latent_stats.pt`
  - `top_ctx.pt`
  - `mid_ctx.pt`
  - `seq_repr.pt`
  - `logit_ctx.pt`
  - `neg_ctx.pt`
  - `top_coactivation.pt`
  - `candidates.pt`
  - `circuits/summary.json`
- [ ] Write `distributed/reports/equivalence_one_worker.json`.
- [ ] Verification: classify any drift as metadata-only, expected floating-point tolerance, expected distributed sampling difference, or a blocker.

## Phase 6 - Reduced Multi-Worker Smoke Runs

- [ ] Run a two-worker reduced real-data smoke with devices `[0, 1]`.
- [ ] Run a four-worker reduced real-data smoke with devices `[0, 1, 2, 3]` if the two-worker run is clean.
- [ ] Run an eight-worker reduced real-data smoke with all devices if the four-worker run is clean.
- [ ] For each run, record:
  - total wall time,
  - per-stage wall time,
  - per-worker shard/replay/task counts,
  - peak VRAM,
  - GPU utilization,
  - CPU RAM,
  - disk used before/after,
  - artifact sizes,
  - failed or stale markers.
- [ ] Compare reduced multi-worker canonical outputs against the one-worker distributed baseline and/or the `single_process` oracle.
- [ ] Write `distributed/reports/equivalence_reduced_real.json`.
- [ ] Verification: do not proceed to full scale until reduced one/two/eight-worker runs have no unexplained semantic drift.

## Phase 7 - Parameter Min-Max Benchmarks

- [ ] Benchmark `data.batch_size` around the baseline, for example `256`, `512`, and `768` if VRAM allows.
- [ ] Choose the largest batch size that keeps peak VRAM comfortably below H100 capacity and improves tokens/sec without causing instability.
- [ ] Benchmark negative-context backends from the same merged pass-1 artifacts:
  - `single_gpu_exact` as a control,
  - `multi_gpu_exact` as the current baseline,
  - `multi_gpu_index_sharded_exact` if replicated index memory or time is high.
- [ ] Benchmark pass-2 dump with 1/2/4/8 workers where practical.
- [ ] Benchmark simple exact reduce first, then target-sharded reduce with `reduce_shards: 2`, `4`, and `8` only if reducer time or memory is material.
- [ ] Keep MapReduce off unless simple exact reduce has clear bottleneck evidence and equivalence can be proven.
- [ ] Benchmark discovery with the intended method list and gradually larger `discovery.n_seeds` values.
- [ ] Verification: produce a short tuning decision note that selects final values for batch size, neg-ctx backend, reducer mode, discovery seed count, and cleanup policy.

## Phase 8 - Full 8xH100 Dress Rehearsal

- [ ] Run the selected final config on a representative but not maximum-cost slice, keeping `cleanup_policy: keep_all`.
- [ ] Confirm all phase markers complete and no worker has failed/stale markers.
- [ ] Confirm all canonical outputs are present at the run root.
- [ ] Validate sanity reports for pass 1, negative context, pass 2, candidate selection, and discovery merge.
- [ ] Check disk headroom remains comfortable:

```bash
df -h
du -sh outputs/* 2>/dev/null
```

- [ ] Verification: the dress rehearsal is accepted only if it completes cleanly, writes benchmark/equivalence reports, and leaves enough disk for a full run plus one failed retry.

## Phase 9 - Full 8xH100 Run

- [ ] Launch the final `distributed_simple_exact` 8-worker run with the selected tuned config.
- [ ] Run pass-1 workers concurrently from the manifest commands.
- [ ] Merge pass-1 outputs and run negative context.
- [ ] Run pass-2 dump workers concurrently.
- [ ] Run pass-2 reduce and candidate selection.
- [ ] Run distributed discovery workers concurrently.
- [ ] Merge discovery outputs.
- [ ] Continuously monitor:
  - `nvidia-smi`,
  - disk usage,
  - worker logs,
  - worker metrics JSONL,
  - failed markers.
- [ ] Verification: full run completes with all canonical artifacts and no unexplained failed/stale worker state.

## Phase 10 - Final Reports, Review, And Retention

- [ ] Save the exact config, config hash, git SHA, command history, environment summary, native build output, and package profile.
- [ ] Save final benchmark reports:
  - total wall time,
  - per-stage wall time,
  - GPU utilization,
  - peak VRAM,
  - CPU RAM,
  - disk throughput or disk deltas,
  - artifact sizes,
  - worker imbalance.
- [ ] Save final equivalence and sanity reports.
- [ ] Review circuit summaries and provenance for accepted circuits.
- [ ] Decide whether the run is:
  - profiling-only,
  - exact but not paper-ready,
  - paper-facing candidate,
  - rejected due to drift or incomplete reports.
- [ ] Keep `cleanup_policy: keep_all` until the run has been reviewed and archived.
- [ ] After review, optionally delete obsolete failed run roots or large partials while preserving canonical artifacts, reports, manifests, and provenance.
- [ ] Verification: no full run is called paper-ready without exact mode, passing sanity checks, passing equivalence, benchmark reports, and complete circuit provenance.

---

## Initial Recommended Parameter Stance

- Use `distributed_simple_exact` first, not MapReduce.
- Use `requirements-cu128.txt` on the RunPod PyTorch 2.8 / CUDA 12.8 image.
- Use `data.batch_size: 512` for the first serious run; tune upward only after peak VRAM is measured.
- Keep `latents.neg_ctx.backend: multi_gpu_exact` initially; test `multi_gpu_index_sharded_exact` if memory or duplicated index time becomes a bottleneck.
- Keep `latents.top_coactivation.reduce_backend: single_process` initially; promote target-sharded or MapReduce only from measured reducer bottlenecks.
- Keep `discovery.n_seeds: 16` during validation; increase only after throughput and circuit output quality look healthy.
- Keep `cleanup_policy: keep_all` until the first full run is reviewed.

## Open Questions

- What reduced real-data slice is large enough to catch drift without burning too much H100 time?
- What exact floating-point tolerances should be accepted for H100 artifact comparisons?
- How large should `discovery.n_seeds` be for the final scientific run after validation?
- Should the first final run prioritize fastest wall time, maximum evidence retention, or minimum risk?
- What disk-use threshold should trigger cleanup before launching another full run?

## Risks / Assumptions

- H100 speed can hide semantic drift unless reduced and full runs produce equivalence reports.
- Local CUDA 13 validation does not replace native extension validation on the CUDA 12.8 provider host.
- `keep_all` is correct for early validation but can consume storage quickly across repeated run roots.
- `multi_gpu_exact` negative context may duplicate enough index data that `multi_gpu_index_sharded_exact` becomes preferable.
- MapReduce adds complexity and should not be introduced unless simple exact reduce is demonstrably too slow or memory-heavy.
