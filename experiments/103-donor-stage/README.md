# 103: the donor-background (resample) test as a native scorer and a pod stage

Agent donor-stage. Spec: `paper/protocol.md` §12.3 (Daniel). Goal: the pod reports a
donor-background recovery and a `resample_pass` for every circuit, computed by src code, not by an experiment script.

## What was built

| piece | where |
|---|---|
| library + CLI | `src/eval/donor_test.py` |
| CPU tests | `tests/eval/test_donor_test.py` (35) |
| chain stages | `experiments/082-public-runs/chain.sh` (T10, T11, T12; public S6/S9 arms `donor`, `donor_unweighted`) |
| progress | `082/progress.sh` (donor rows, noR rows, process count), `082/progress_tqdm.py` (TL stages T0-T12) |
| equivalence | `run_equiv.sh` (GPU, CLI) + `equiv.py` (compare with 100) -> `out/equiv/` |
| smoke | `run_smoke_chain.sh` -> `smoke_chain/` |
| cost | `cost.py` -> `out/cost.json` |

### The test (`src/eval/donor_test.py`)

The test of 100's `run.py` ("EXACTLY 099 run_baseline"), with the pieces it ran ported into src:
- 073's `ResamplePatcher` (the used path verbatim; the diagnostics-only `frac` / `member_pre` switches dropped),
  `run_full`, `capture_codes`, `at`.
- 100's lazy pins (`BasePins`).
- 099's pooled recovery (`pooled`).

The test:
- **Bases:** the held-out strongest + held-out contrast rows of `eval_probe(rec, "strong")`.
- **Pools:**
  - A = the 48 strongest-train rows;
  - C = the held-out contrast rows;
  - R = the fixed random pool (below).
- **Draws:** `draw_selections` gives 099 / 100's exact generator sequence (one CPU generator seeded
  20261028 + 1000·layer + latent; pools in the order A, C, R; per draw the activating half, then the silent half).
- **Variants:** fit, a1, pinned, and the empty circuit under the same donors.

Several circuits of one target (`donor_target(G, key, rec, {arm: circuit}, pool)`) share the base codes, the pool
codes and the empty draws. The donors depend only on the target, so the rows of one arm are the same whichever other
arms share the launch (tested).

**Row** (`OUT_DONOR/rows[.shard<i>].jsonl`, one per target × arm):
- pooled `recovery[variant][pool]`, plus `recovery_pre` (the pre-activation read);
- `recovery_split[variant][pool]` = {donor_fires, donor_silent}: the (draw, base) pairs split by whether the donor
  fires at the base's anchor, each pooled the same way;
- `donor_fires_frac`;
- `empty_leak[pool]`;
- `empty_means` / `means` per draw;
- `silent_firing`;
- `self_empty` (the identity check);
- the production fields (`prod`), `pass_prod`, and **`resample_pass`**;
- timings;
- `pool_sha256`, `wcm_version`.

Per-base reads and the donors drawn go to `raw[.shard<i>].jsonl`.

**`resample_pass`** = `wcm_eval.passes(production row)` (Z, A, C ∈ BAND = [0.75, 1.5], necessity ≥ 0.9; the pass
rule is unchanged) AND pooled fit recovery A and C each ∈ BAND. An undefined recovery fails. The row carries
`vacuous_tk`, so readers drop vacuous rows from both denominators, as for the production rate.
- *Difference from 099 / 100's `donor_pass`, as the spec asks:* theirs required only Z and necessity on the
  production side; this one requires the full production pass.

**The R pool (deterministic, from local data; no upload, no download):**
- **Draw:** 073's `random_pool` on the run's own corpus loader (`G["M0"].probe_builder.loader`): POOL_SEED 20261016,
  512 draws, ≥ 60 non-zero tokens. This is what 099 used (099 filtered `073/out/random_pool.pt`).
- **Filter:** 099's exact / near rule against every row of every record in the run's CTX_DIR (strongest, mid-band,
  contrast, and `neg_held_v2` if present).
- **Speed-up:** an 8-gram-hash prefilter picks the (pool row, record row) pairs to run `heldout_rule.pair_stats` on.
  It is lossless, because every reject condition implies a shared 8-gram. The CPU test checks it equals the brute-force
  filter on planted exact / shifted-run / 32-run / 8-gram-share rows and on a tiny-vocabulary random set.
- **Build:** the first shard builds it under a file lock and the other shards wait. The file records `records_sha1`,
  and a later run warns if CTX_DIR's record set changed.
- **On the pod:** the TuringLLM corpus (`data/`) is already in TL_ROOT, and each public model's corpus is written by
  its S1 stage. Nothing has to be uploaded.

**Edge case (decision, not blocking):** if a pool is smaller than a base half (a contrast held-out set shorter than
the activating held-out set), it is drawn as consecutive permutations, and the row flags `pool_short`. 100 raised
"donor set too small" there. With full-size records this never triggers, and the draws are 100's.

### CLI / stage

A 062 companion: `launch.sh` runs it as `SCRIPT` with `SHARD=i/k`, and it imports the 062 driver for its `Runner`.

Env:
- `OUT`, `CTX_DIR`, `FAMILY`, `WCM_VERSION`: as the driver.
- `DONOR_TARGETS` (required).
- `DONOR_ARMS` [main]: `main` | a sweep arm name | `label=<path with {key}>`.
- `OUT_DONOR` [OUT/donor].
- `DONOR_POOL` [OUT_DONOR/random_pool.pt].
- `DONOR_POOL_RECORDS` [CTX_DIR].
- `DONOR_DRAWS` [8], `DONOR_VARIANTS` [fit,a1,pinned], `LIMIT`.
- `DONOR_RESCORE=1`: re-score production instead of reading the run's eval rows.
- `REPORT=1`: a summary of the rows.

**Production rows** come from `OUT/main/eval.shard*.jsonl` (held = strong) or `OUT/sweep/eval.shard*.jsonl` (by
arm). They are re-scored with `Runner.score` only when missing.

**Resume:** a (seed, arm) row without `error` is done. A target's rows are written only after every arm of it is
complete.

**Errors:**
- a missing circuit file gives an `error` row, so 082's `job_errors.py` counts it and a rerun retries it;
- a rejected fit (None) gives a `skip` row.

**Chain stages** (each with markers, archive + md5 + verified copy, the `job_errors` early gate, and progress lines
`i/n` read by `progress_tqdm.py`):

| stage | what | switch |
|---|---|---|
| TuringLLM T10 | without-R arm: `WCM_VERSION=v2`, `MODE=sweep`, `SWEEP_ARMS=W_0.001` (= LAM 1e-3) on sweep190, OUT `TL_NOR_OUT` [094/out_v2_noR] (a version-marked OUT cannot hold v2 and v2cr circuits); only under `WCM_VERSION=v2cr` | `TL_NOR` [1], `TL_NOR_ARM` |
| TuringLLM T11 | donor test of the 8k main circuits, 8 chunks (T3's chunk lists), fit lock per chunk -> `TL_V2_OUT/donor/main` | `TL_DONOR` [1], `TL_DONOR_MAIN` [1] |
| TuringLLM T12 | donor test of all 13 sweep arms in one launch (shared work) -> `TL_V2_OUT/donor/sweep`, then the T10 arm -> `TL_NOR_OUT/donor/sweep` | `TL_DONOR`, `TL_DONOR_SWEEP` [1] |
| public S6/S9 `donor` | main list × main circuits -> `out_<model>/donor` | `DONOR` [1], `DONOR_<model>` |
| Pythia S6 `donor_unweighted` | the unweighted list × `U_<adopted lambda>` -> `out_pythia-70m/donor_unweighted` | same |

All three TL stages share one R pool, `TL_DONOR_POOL` [TL_V2_OUT/donor/random_pool.pt], because they share one
CTX_DIR. `DONOR_P` [P] sets the donor processes per GPU. `TL_STOP_AFTER` now defaults to T12.

The stages come after every fit stage of their track (TL: after T8 / T9; public: after the 073 arm), so they delay no
fit.

**Unweighted arms under R:** `Runner.method` applies `WCM_VERSIONS[v2cr]` (`resample_weight` 0.25) to every arm,
weighted or not, and `Runner.fit` sets the donors whenever `resample_weight > 0`. The protocol harness configures the
U arms with the triple floor + "pos" objective that R requires. The smoke confirms it: the `U_0.001` train rows carry
the resample stats; see below.

## Equivalence (GPU, ~9 min; `run_equiv.sh`, `equiv.py`, `out/equiv/compare.json`)

The setup:
- the native CLI on 3 of 100's targets (`targets_equiv.txt`: 2.attn.34021, 6.mlp.209, 2.resid.29259);
- 099's audited records;
- both of 100's lambda-1e-3 arms: `wcm_1e-3` (099's panel18 circuits, which 100 reused) and `v2cr_1e-3` (100's
  fits);
- the R pool **built by the native path** from the local corpus, filtered against the 18 records in CTX_DIR.

**Result: 6 / 6 (target, arm) bit-identical** (max abs difference 0 on every compared number):

| target | arm | n | fit recovery A / C / R (native = 100) | production pass | resample pass |
|---|---|---:|---|---|---|
| 2.attn.34021 | wcm_1e-3 | 255 | 0.4314 / 0.0000 / 0.0051 | no | no |
| 2.attn.34021 | v2cr_1e-3 | 297 | 1.5807 / 0.6595 / 0.5603 | no (Z 0.00) | no |
| 6.mlp.209 | wcm_1e-3 | 495 | 0.0118 / 0.0056 / 0.0000 | yes | no |
| 6.mlp.209 | v2cr_1e-3 | 451 | 0.9402 / 0.9308 / 0.8540 | yes | **yes** |
| 2.resid.29259 | wcm_1e-3 | 160 | 0.5823 / 0.2839 / 0.2147 | yes | no |
| 2.resid.29259 | v2cr_1e-3 | 169 | 1.0814 / 0.8610 / 0.8108 | yes | **yes** |

What was compared:
- the donors drawn: identical index lists for every pool and draw;
- the natural anchor reads;
- the per-draw anchor means of the empty circuit and of the fit / a1 / pinned variants (8 draws × 3 pools);
- every pooled recovery (fit / a1 / pinned × A / C / R).

**The pool:**
- the rebuilt 073 random pool equals `073/out/random_pool.pt`;
- after the filter it equals 099's `panel18/random_pool.pt` (512 of 512 kept, as in 099).

So the R pool construction reproduces from local data.

Not compared bit for bit:
- *Production rows:* re-scored here with `Runner.score` (Z / A / C / necessity equal 100's to the printed
  precision).
- *Timings:* the GPU was shared with arms-102's fits, so this run was ~3.2× slower than 100's idle-GPU times.

## Tests

- `tests/eval/test_donor_test.py`: **35 passed** (Windows CPU and WSL):
  - pooled recovery: pools before dividing; undefined when the denominator is ≤ 1e-9 or a list is empty;
  - `resample_pass`: 0.75 / 1.5 edges inclusive, 0.7499 / 1.5001 / None / NaN fail, R not in the rule, needs the
    production pass, a subset of production passes over 500 random rows;
  - donor draws: equal 100's loop for 3 targets, deterministic, per target, without replacement, the short-pool path,
    an empty pool raises;
  - the donor-fires split;
  - the pool filter: equals the brute-force 099 filter on planted rows (exact, a 34-token shifted run, exactly 32, a
    30-token run, one 8-gram, ≥ 50% 8-grams, padding) and on a tiny-vocabulary random set; `record_rows` includes
    `neg_held_v2`;
  - end to end on the tiny CPU engine (MockModel + EncBank): rows complete and identical whatever other arms share
    the launch; a circuit of every latent recovers 1 (fit, a1, pinned) on every pool and the empty circuit 0; the
    self-donor identity;
  - CLI helpers: arm parsing, production-row lookup (last strongest row, sweep arm match, errors / skips / truncated
    lines ignored), summary counts.
- 082 chain tests + `tests/eval/test_wcm_eval.py` + the donor tests in WSL: **157 passed**.
- `bash -n` on chain.sh / progress.sh; `DRY_RUN=1 TL_TRACK=1 WCM_VERSION=v2cr` prints every new stage with its env,
  archive and error gate (see the smoke for the real run).

## Smoke (~1 h 03 min; `run_smoke_chain.sh` -> `smoke_chain/`, events `smoke_chain/logs/chain.events`)

**Setup.** The 082 chain itself, in its smoke mode: `LIMIT=2`, 20 steps, P = 1, GPU 0, `WCM_VERSION=v2cr`,
`TL_TRACK=1`, `MODELS=pythia-70m`.
- *Inputs read, not written:* the TuringLLM records and lists of the 082 `smoke_v3` run (3.attn.11307, 0.mlp.5196).
- *Pre-marked done (not under test):* T0-T2, T5-T8, Pythia S5, and the alpha1 / nulls / stable-core / 073 arms.
- *Order:* phase `tl` (Pythia stopped at S3), then phase `all`.
- *GPU:* shared with arms-102 (9.7 GB in use at the start).

**Result: chain finished rc 0, 13 jobs, 0 error rows, every stage archived + md5-verified, C4 passed:**

| stage | job rows | archive |
|---|---|---|
| T3 main + alpha1 (v2cr) | 2 + 2 | T3-main 26M |
| T4 sweep W_0.001 / W_0.004 / U_0.001 (v2cr) | 2 / 2 / 2 | T4-sweep 74M |
| **T10** without-R arm (OUT marked `v2`) | 2 | T10-noR 26M |
| **T11** donor test, main | 2 | T11-donor-main 84K (rows, raw, R pool) |
| **T12** donor test, 3 sweep arms in one launch + noR | 6 + 2 | T12-donor-sweep 20K |
| Pythia S6 main / unweighted | 8 / 2 | |
| **Pythia S6 donor / donor_unweighted** | 8 / 2 | 80K / 12K |
| S10 pack (C4) | | results 181 entries, stores 42 |

**Checks.**
- **Unweighted arms run with R under v2cr:** the `U_0.001` train rows carry `resample_weight` 0.25, 48 donors,
  `resample_fill` sparse, and the curve term `resample`, the same as `W_0.004`.
- **Sharing-invariant:** T11's main rows equal T12's `W_0.001` rows (the reused main circuit, scored alongside 2
  other arms) bit for bit: recoveries, per-draw means, empty means.
- **Rows complete:** recovery, the split, leak, self-donor and production fields are present in every row.
  `prod_source` is `eval_rows` (read from the driver's eval rows; nothing re-scored).
- **R pools:**
  - TuringLLM: 512 of 512 kept against 2 records;
  - Pythia: 383 kept on its own corpus.
- **Viewers:** `progress.sh` shows the donor / noR rows; `progress_tqdm.py` shows "after T12".
- **Resume:** rerunning a finished donor job prints "0 with work" and exits before loading the model.
- *Smoke numbers are not results:* the circuits come from 20-step fits (tens to hundreds of thousands of members), so
  the recoveries near 1 mean nothing.

## Pod cost (`cost.py` -> `out/cost.json`)

**Local timing model.** Taken from 100's rows: the same code path, bit-identical results, an idle GPU, 18 targets at
L2 / 5 / 6 / 10, both arms.
- Shared per-target work (base + pool codes, 24 empty draws, the self-donor check): **1.38 + 0.97·L s**.
- Per circuit (fit + a1 + pinned, 72 passes of 32 bases): **1.87 + 2.30·L s**.
- One circuit at the 8k mean layer (5.71) therefore costs **21.9 s** locally (shared 6.9 + circuit 15.0); a sweep
  target with 13 arms costs 200 s.
- This experiment's own runs shared the GPU with arms-102's fits and were 3.2× slower, so they only give the
  contention factor.

**Pod scaling.** A 062 pod process (H100, MPS, P = 6 per GPU) scored ~1.6× faster than this GPU idle (scoring is the
donor test's kind of work: forward only, batch 16). The fits, by contrast, were ~1.6× slower per process. The table
gives both factors, over 48 processes.

| job | local GPU-h (idle) | pod wall h (×0.6) | pod wall h (×1.6) |
|---|---:|---:|---:|
| T11 donor test, 8k main | 48.8 | 0.61 | 1.63 |
| T12 donor test, 190 × 13 sweep arms | 10.6 | 0.13 | 0.35 |
| T12 without-R arm, 190 × 1 | 1.1 | 0.01 | 0.04 |
| GPT-2 small main, 512 (as TuringLLM) | 3.0 | 0.04 | 0.10 |
| Pythia-70m main 512 + U 128 (smoke, ~7.6 s per target de-contended) | 1.4 | 0.02 | 0.05 |
| **total** | **64.8** | **0.81** | **2.16** |

**Plus fixed costs:**
- *Model load per launch:* ~1-2 min on the pod; 4 min here on `/mnt/x`. There are 8 (T11 chunks) + 2 (T12) +
  2-3 (public) launches.
- *The R-pool build:* reads every record once. For 8k records (~1.4 GB) that is a few minutes, on one shard while the
  others wait.
- *The T10 fits:* 190 v2 fits; about 190 × 75 s / 48 ≈ 5 min.

**Overall:** the donor stages add roughly **1-2.5 h of pod wall time**, after the fit stages.

**Cheaper settings, if needed:**
- `DONOR_VARIANTS=fit` (exported to the chain; only `fit` enters `resample_pass`) cuts the per-circuit part by
  about 2/3. The a1 and pinned diagnostics would then be lost.
- `TL_DONOR_SWEEP=0` drops T12.

## Open questions (none blocks the pod; each has a default in place)

1. **Stage order on the TuringLLM track.** The donor stages run last (T10 → T11 → T12, after T8 / T9), so they delay
   no fit. If the pod runs short, T11 (the 8k headline "y% also pass on real backgrounds") comes before T12.
   - Alternative: move T11 right after T3. That makes the headline available earlier, but delays T4-T8 by ~1-1.5 h.
   - Default kept: last.
2. **Variants on the pod.** fit + a1 + pinned costs ~3× fit alone. Only fit enters `resample_pass`; a1 and pinned are
   the §12.3 "reported with it" diagnostics. Default: all three (~1-2 h for the 8k). `DONOR_VARIANTS=fit` cuts T11 to
   ~0.3-0.8 h.
3. **Unweighted arms score a1 = fit.** U circuits have no amplitudes, so a1 repeats fit for 5 of the 13 sweep arms,
   about 13% of T12. Left in for uniform rows.
4. **The `W_0.001` sweep arm is the main circuit**, so T12 repeats T11 for 190 targets (the smoke shows the rows are
   identical). This is cheap, and kept as a cross-stage check.
5. **Short donor pools** (a held-out contrast set smaller than the activating one) use consecutive permutations
   instead of 100's error. This never happens with full-size 094 records. The row flags `pool_short`; report any
   flagged rows separately.
6. **The production side of `resample_pass` is the full rule** (Z, A, C, necessity), as §12.3 says. 099 / 100's
   `donor_pass` used Z and necessity only, so 100's donor-pass counts are not directly comparable. Recomputing 100 with
   the new rule only needs its rows (`prod` + `recovery`).
7. **The R pool is per CTX_DIR.** It is filtered against every record there. On TuringLLM this is the 8k fresh
   records; the sweep and the noR arm share them, and so share the pool. A pod with a different corpus checkout would
   get a different pool; the file records `records_sha1` and its sha256 is in every row.
