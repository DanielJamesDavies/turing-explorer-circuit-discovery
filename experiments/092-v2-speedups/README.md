# 092: v2 speed-ups (truncate at the anchor, one combined Z/C/A forward) and Z + A + R_C

v2 = 090's `sp_ste` (WCM, A / C trained under the scorer's sparsity-preserving fill, straight-through). Targets: 091's
8 (`experiments/091-v2-resample-small/out/targets.json`, seed 20261025: 4 fail89, 2 deep attention near / far, 2
controls). Reference: 090's `sp_ste` circuits and rows (090 refits are deterministic: 091's CHECK reproduced the same
circuit and the same per-step curve).

src is not edited. `sx_engine.py` patches `src/circuit/instrument/learned_mask.py` in memory on top of 090's
`sp_engine` (tests 1, 2) or 091's combined engine (test 3). Switches (`SPEED`):

- `truncate` (test 1): in `data_loss`, each training micro-batch `tokens[s:s+4]` is cut to `[:, :max(anchor)+1]`.
  Why it is exact: TuringLLM is causal (SDPA `is_causal=True`), uses absolute position embeddings, per-token RMSNorm,
  no padding or attention mask; SAE encode / Top-K / every fill (mean, sparsity-preserving budget) is per position; the
  loss reads only the target pre-activation at each row's anchor and the rank-keep term's other site latents at the
  anchor; the floors / `_rank_clean` are constants precomputed over full windows. No all-position statistic enters the
  v2 training loss (the 079 silent term would, and it is off). Batches are contiguous slices in a fixed order, so
  sorting rows by anchor would change which rows share a batch: the cut is per batch as is.
- `combine` (test 2): the Z, C and A terms (same rows / anchors / targets, three patchers over the same thetas and
  amps) run as ONE forward over the micro-batch repeated 3x; `CombinedPatcher` encodes and decodes once per site and
  applies each row group's own floor and sparsity-preserving fill. Loss = the sum of the three terms exactly as
  `data_loss` builds them (mse at the anchors + rank-keep), one backward. `COMBINE_ROWS` = rows per group per forward
  (memory fallback; chunk means weighted by their share, so the loss is unchanged).
- `drop_floor` + `sp_donor` (test 3): no C term; the 079 donor-resample term with the 48 training contrast contexts as
  donors (091 `setup_terms`, weight 0.25), under a sparsity-preserving DONOR fill: members at alpha x live, then the
  donor's non-member active entries at that position, largest first, up to K - #active members, rest 0 (the training
  analogue of 073's `ResamplePatcher`; tested equal at binary gates). NOTE: 091's `v2R` trains R WITHOUT this fill
  (079-exact) and keeps C, so t3 vs v2R differs in two things.

Files: `sx_engine.py` (patches), `test_sx_engine.py` (CPU tests), `run.py` (fits + scoring), `report.py`
(`out/tables.md`), `queue.sh` (waits for 091 to leave the GPU, then CHECK, then the run), `queue2.sh` (then
`ARMS=t3b_ZAR_rawR`). Logs: `check.log`, `run.log`.

## Commands (WSL, repo root)

    PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/bin/python experiments/092-v2-speedups/test_sx_engine.py
    PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 CHECK=1 .venv/bin/python -u experiments/092-v2-speedups/run.py
    PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/092-v2-speedups/run.py >> experiments/092-v2-speedups/run.log 2>&1
    .venv/bin/python experiments/092-v2-speedups/report.py

The run resumes: (target, arm) rows already in `out/rows.jsonl` are skipped, fitted circuits in `out/circuits/<arm>/`
are reloaded with their `out/fits.jsonl` timing. `ARMS=t3_ZAR` (etc.) runs one arm; `ARMS=t12_both` runs tests 1 + 2
together (not in the default list).

## Results (full tables in `out/tables.md`, rows in `out/rows.jsonl`, fit stats in `out/fits.jsonl`)

**Noise floor first.** 090 refits are bit-deterministic at fixed shapes, but ANY kernel-shape change (bf16 autocast,
different batch / sequence shapes pick different matmul / attention kernels) moves the step-0 data terms by ~10%
(they are tiny relative errors, ~1e-2, at bf16 resolution) and the 400-step fit then wanders: test 2 is mathematically
identical to v2 (same loss, CPU-tested to 1e-7) and still gives membership Jaccard 0.30-0.67 vs 090 (median 0.56)
and flips 2 of 8 rule passes. So "exact" here means "inside that band", and single-target pass flips are not evidence.

**Test 1, truncate at the latest anchor: exact, x1.22 faster.** Step 0 on 11.mlp.16843 (token share 0.27 in that
batch): max |d term| 1.28e-3, |d loss| 1.42e-3 on a data loss of 1.16e-2; the pure batch-shape change (comb) gives
1.46e-3 / 1.44e-3, i.e. the same bf16 kernel noise. 400 steps: Jaccard 0.34-0.61 (median 0.53, inside the noise
band), rule pass 6/8 vs 7/8 (6.attn.4595 flips, as it does under test 2). Fit time 531 s vs 650 s over the 8 (x1.22,
per target x1.05-1.46, best on 11.attn.13198 with token share 0.30); peak memory unchanged (8.7 GB). The gain is far
below the token saving (median share 0.6): on 11.mlp.16843 a 1-step fit already takes 30-34 s (fixed set-up and
post-fit evaluation, untouched by the cut), and 64-token windows leave the per-site work launch-bound.

**Test 2, one combined Z/C/A forward: exact, but SLOWER.** Step 0: |d loss| 1.44e-3, max |d grad theta| 5.8e-5
(scale 5.8e-4), the same as the truncation noise. 400 steps: Jaccard median 0.56, pass 5/8. Fit time 724 s vs 650 s
(x0.90) and peak memory +2-3 GB (11.7 GB vs 8.8 GB at L11; no OOM, all 12 rows in one forward). Not worth adopting.
Combined with truncation (step-0 check only, `comb_trunc`): peak 8.4 GB, same noise.

**Test 3, Z + A + R_C (no C; R under the sparsity-preserving donor fill, weight 0.25).** Rule pass 4/8, resample pass
2/8 (11.mlp.16843, 5.mlp.29712): identical counts to 091 v2R (4/8, 2/8, same two targets); sp_ste 7/8 and 0/8. Fit
time 774 s vs v2R 887 s (x1.15 faster) vs sp_ste 650 s (x1.19 slower); peak ~v2R's (9.5 GB). So it keeps v2R's
resample gain at below v2R's cost, but it does not reach v2's cost (R's donor densify per forward costs more than a
mean-floor pass) and it keeps v2R's rule-pass loss. t3 changes two things vs v2R (no C, and the donor fill);
t3b isolates the dropped C.

**Test 3b (extra), Z + A + R_C with 091's raw R (no donor fill), no C.** Rule pass 4/8, resample pass 1/8
(11.mlp.16843; 5.mlp.29712 misses at rs A 0.79 / rs C 0.75, 9.resid.33488 at rs C 0.797). Fit time 632 s = v2's cost
(650 s; v2R 887 s); peak ~v2R's. So dropping C buys v2's cost back but keeps only part of v2R's resample gain, and
on this sample the donor fill (t3) recovers the second resample pass at +142 s. With the refit noise above, 2/8 vs
1/8 is not a distinguishable difference.

Not run: `t12_both` as 400-step fits (only its step-0 check, `comb_trunc`); given test 2 is slower it was not worth
the GPU time. To run it: `ARMS=t12_both` with the run command above, then `report.py`.
