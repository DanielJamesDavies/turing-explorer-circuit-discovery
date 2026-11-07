# 108: does WCM with more contexts per optimiser step give better or more stable circuits?

Production v4 (`paper/protocol.md` §0) fits each circuit with **4 training contexts per
AdamW step** (`probe_batch_size` 4, set by `049/amp_eval_pass_v2.setup()` and `062/driver.py:314` for the public
path), 400 steps, cycling deterministically over the 48 training contexts (32 strongest + 16 mid-band). Question
(Daniel, approved design): does a larger effective batch, via gradient accumulation, give better or more stable
circuits? **Nothing in production changes here**: `src/` and `062/driver.py` are untouched, and every override is
applied by the experiment-local wrapper `fit_arm.py`.

## How the engine's accumulation works (verified in code)

`src/circuit/instrument/learned_mask.py`:
- **Effective batch = `batch_size`**, which `ablation_gradient.py:391` passes from the method's `probe_batch_size`.
  The micro-batch is `deep_batch_size` when the fit has more than `deep_site_threshold` upstream sites, else
  `batch_size` (`:1389-1392`). `accum = ceil(batch_size / micro)` (`:1398`). Production sets the threshold to 99
  (`059/protocol_harness.configure`), above TuringLLM's 35 sites, so production runs micro = batch = 4, accum 1.
- **The micro-batch only changes memory.** Per step there is one `zero_grad`. Then for each of the `accum` chunks,
  each loss term (Z, C, A, R) is computed on that chunk and backwarded on its own, as `w · data_loss(chunk) / accum`
  (`:2253-2357`). `data_loss` is a **mean** over the chunk's rows (`:1950`), and the λ penalty is backwarded once
  per step (`:2444`), followed by one `opt.step()` (`:2458`). The step's gradient is therefore the mean over all
  `batch` rows, so the **gradient scale is independent of the batch** (exactly so when the chunks are equal-sized,
  which they are here: 48 rows, chunks of 4). AdamW normalises the step anyway.
- **Which contexts:** chunk j of step t starts at row `s = ((t·accum + j) · micro) mod N` (`:1880-1885`), with
  N = 48 training rows:
  - batch 4: step t takes rows `[4t mod 48, +4)`, a 12-step cycle (production);
  - batch 16: rows `[16t mod 48, +16)`, a 3-step cycle;
  - **batch 48: chunks start at 0, 4, …, 44, so all 48 rows every step (true full batch).**
  - No shuffling at any batch size.
- **Other effects of a larger batch:**
  - The R term redraws one donor row per training row at every forward (`:295-329`), so a larger batch draws more
    donors per step. At batch 48 every row gets its own draw each step.
  - The anneal T = 0.05^(step/(S−1)) and the warm-up round(0.1·S) scale with the step count S
    (`:1630, 2215-2238`), so the 100-step arms anneal 4× faster per step.
- **Peak VRAM.** Every arm runs at micro 4: `deep_site_threshold` 0 and `deep_batch_size` 4. Peak VRAM is therefore
  production's at every depth (~9.9 GB at the deepest target; the card has 16 GB). Accumulation adds no
  activation memory: each chunk's graph is freed before the next chunk.
- **The control is exact.** Setting the threshold to 0 for B4 keeps micro = batch = 4, accum 1, so the control is
  production v4 exactly.

## Design (fixed before any result)

**Substrate.** This is the setup `107-true-baselines/latent_pilot` used for its local v4-equivalent TuringLLM fits:
- the 062 driver, `MODE=main`, `WCM_VERSION=v4`, λ 1e-3, `SKIP_SPEC=1`, `CURVE_STRIDE=1`, `ORDER=list`;
- the context records in `CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx`, reused, so no data pass is
  needed. These are the 15k run's records, **not** the pod's 094 deduplicated records, so the held-out sets carry
  the known duplicate leakage. The comparison is between arms on identical records;
- scoring by the driver's production path: the frozen 049 scorer, held-out strongest (primary, pass rule) and
  held-out mid-band. Induce (`phi_cf_alpha_blind_tk`) is in every row;
- the donor-background test (`src/eval/donor_test.py`, variant `fit`, pool `073-strict-eval/out/random_pool.pt` as
  latent_pilot) when the plan has time. It cost ~15 s per target in latent_pilot; the timing stage measures it.

**Targets.** 16 targets (`targets.txt`, built by `make_targets.py`, `random.Random(20261106)`):
- **Source:** latent_pilot's 48 dev targets, which were never used to choose anything.
- **Eligibility:** arm B with exactly 48 training rows and both held-out sets. This excludes the two thin targets,
  7.attn.38584 and 8.attn.39566.
- **Stratification:** one target per depth band (L1–4, L5–7, L8–11) × site kind cell, then a second per cell, cut at
  16. That gives 6 deep, 5 shallow and 5 mid.
- **Fixed positions:** the deepest, 11.resid.10297 (35 sites), is first; a shallow one, 4.attn.35016, is second.
  The timing stage fits these two.
- **Order:** the list order is the run order, and the plan keeps the first N.

| # | target | # | target | # | target | # | target |
|---|---|---|---|---|---|---|---|
| 1 | 11.resid.10297 | 5 | 3.mlp.32628 | 9 | 5.resid.5995 | 13 | 10.attn.843 |
| 2 | 4.attn.35016 | 6 | 6.mlp.25102 | 10 | 8.resid.20406 | 14 | 1.mlp.37214 |
| 3 | 6.attn.33144 | 7 | 11.mlp.2554 | 11 | 3.attn.1392 | 15 | 7.mlp.33101 |
| 4 | 9.attn.27124 | 8 | 4.resid.25803 | 12 | 6.attn.35949 | 16 | 9.mlp.40169 |

**Arms** (`arms.py`). All arms run at micro-batch 4. They differ from production only in the columns shown.

| arm | effective batch (accum) | steps | peak lr | R donor seed | context passes | role |
|---|---|---|---|---|---|---|
| **B4** | 4 (1) | 400 | 0.1575 | 20261017 (production) | 1,600 | control = production v4 |
| **B48** | 48 (12) | 100 | **0.620595** | production | 4,800 | full batch |
| **B16m** | 16 (4) | 100 | **0.620595** | production | 1,600 | compute-matched to B4 |
| **B16** | 16 (4) | 400 | 0.1575 | production | 6,400 | production steps and lr |
| **B4s2** | 4 (1) | 400 | 0.1575 | **20261108** | 1,600 | seed replicate of B4 |
| **B48s2** | 48 (12) | 100 | 0.620595 | **20261108** | 4,800 | seed replicate of B48 |

**How the 100-step lr is matched.** The 100-step arms match production's **lr budget Σlr** under the v4 schedule:
geometric with `lr_min_frac` 0.05 and warm-up round(0.1·S) steps.
- This is the budget the v4 ×3.15 multiplier was defined by (`driver.py:97-101`; latent_pilot part 2). Σlr sets both
  the pruning walk (Adam moves θ by about lr per step, independent of the gradient scale, so of the batch) and the
  decay budget Σlr·wd.
- Production: Σlr over 400 steps at 0.1575 = **21.40** (the spec's number). Over 100 steps the shape sums to 34.48
  per unit peak, so the matched peak is 21.40 / 34.48 = **0.620595** (×3.940 of production; 10 warm-up steps).
  Σlr·wd = 1.070, as in production.
- B48 uses the same lr as B16m. No better-justified matching exists: with Adam the per-step move is set by lr, not
  by the batch, so equal Σlr is equal pruning and decay budget at any batch.
- **Caveat:** lr / T is about 4× production's at every step (the gate moves 4× further per step over 4× fewer steps).
  If the 100-step arms show early spikes, that is a known side effect of the ×3.15 peak.
- B16 keeps production's steps and lr: 4× the data per step at the same budget.
- The s2 arms change only the R donor-draw seed, which is the fit's only random source (`R_SEED`, `driver.py:460`).
  B4 vs B4s2 is the reference seed overlap.

**Implementation** (`fit_arm.py`, no edits to `src/` or `062/`). It imports the 062 driver and wraps two methods:
- `Runner.method`, after the driver has configured the fit:
  - `method.probe_batch_size = batch`;
  - `learned_mask.deep_site_threshold = 0`, `deep_batch_size = 4`;
  - `learned_mask.lr = 0.620595` for the 100-step arms only;
  - steps and the R seed through the driver's own `STEPS` / `R_SEED` env;
  - it asserts every other v4 setting: geometric, warm-up 0.1, censor 0.05, R 1.0, rank-keep 0.1, λ 1e-3, sparse
    fill, eval_ref means, and the R seed.
- `Runner.fit`:
  - checks the engine's own provenance (`batch_size_used`, `micro_batch`, `accum_chunks`, `steps`) against the arm
    before the circuit is moved into place;
  - saves atomically (temp file and `os.replace`);
  - logs `out/<arm>/b108_fits.jsonl` (wall time, s / step, torch peak allocated / reserved MiB);
  - honours `STOP_EPOCH`: no new fit after it.
- OUT is marked `out/<arm>/BATCH108.json`, and another arm's settings are refused.

**Run** (`run.sh`): one GPU process at a time, resumable.
1. **Timing:** B4 then B16 on the two timing targets, then the donor test of B4 on them.
2. **Plan** (`plan.py`):
   - cost model: t/step = o + accum·u, from B4 vs B16 per target, linear in the upstream site count, × 1.15
     margin, plus measured scoring, start-up and donor costs;
   - choose the largest N in 8–16 such that every arm fits by the deadline. Fewer targets rather than fewer arms;
     the donor test only if it still fits.
3. **Fits:** the arms in the order B4, B48, B16m, B16, B4s2, B48s2, so a partial run still has the control and the
   main contrast.
4. **Donor test.**
5. **Analysis.**

The deadline is min(start + `BUDGET_H` [11] h, `STOP_EPOCH`).

Watchdog, every 5 s:
- GPU memory.used > 15,300 MiB, or host MemAvailable < 4 GB, kills the process group;
- the fit in progress is recorded in `out/<arm>/b108_aborted.txt` and excluded, and the arm restarts on the
  remaining targets;
- the run refuses to start when the card holds > 3,000 MiB.

Estimated cost. latent_pilot's B4-equivalent dev48 fits took 22–162 s (deepest 162 s). Relative to one B4 fit, a
target costs about 1 + 3 + 1 + 4 + 1 + 3 = 13 fits, so 16 targets take roughly 5–6 h plus scoring. That should fit
in 11 h at N = 16; the timing stage decides.

**Known before the run: the pass count is near its ceiling on this panel.** latent_pilot's v4-equivalent fits of
these 16 targets (`dev48_geo_x3.15_warmup_censor`) pass the held-out strongest rule on **16 / 16**, and the B4 control
should reproduce them; `analyze.py` prints the B4 vs those circuits Jaccard as a sanity check of the wrapper.
- Held-out strongest passes can therefore only show losses here.
- The comparison rests on size, seed overlap, the held-out mid-band passes (9 / 13 non-vacuous there) and the donor
  resample pass (12 / 16 there).
- For reference, B4-equivalent vs B4s2-equivalent (latent_pilot's warm-up arm with and without censor, which is
  **not** a seed pair) has a membership Jaccard of median 0.55.

## Pre-stated reading rule (written before any result)

> **A larger batch helps if it raises the held-out pass count, or lowers median size at an equal pass count, or
> raises seed overlap, without lowering the others; differences within ±1 target of 16 are noise.**

- **Pass count:** production rule, held-out strongest, vacuous rows dropped.
- **Size:** the median member count on targets passing in both arms (the ratio to B4).
- **Seed overlap:** the membership Jaccard of B48 / B48s2 against B4 / B4s2.
- **Primary comparisons:** B48 and B16m against B4.
- **Reported, not in the rule:** B16 (4× compute), mid-band passes, induce, donor recovery and resample pass, fit
  time and peak VRAM, the zero term's late rebound in the training curve (stability), and B4 vs B48 overlap.

## How to run

WSL, repo root, when the GPU is free (the queue runs this; `READY` marks it ready):

```
nohup setsid bash experiments/108-batch-size/run.sh > experiments/108-batch-size/out/run.log 2>&1 < /dev/null &
#   env: BUDGET_H (11), STOP_EPOCH (epoch s), DONOR (auto | 1 | 0), GPU_MAX_MIB (15300), RAM_MIN_GB (4), GPU_IDLE_MIB (3000)
DRY_RUN=1 bash experiments/108-batch-size/run.sh          # prints every command; CPU plan from prior timings
CHECK=1 ARM=B48 CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python experiments/108-batch-size/fit_arm.py   # CPU
PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/bin/python experiments/108-batch-size/analyze.py > experiments/108-batch-size/out/analysis.md
```

Outputs, under `out/`:
- `<arm>/` — a driver OUT, plus `b108_fits.jsonl` and `donor/`;
- `stages.tsv` — per-process wall time, return code and peak GPU memory.used;
- `watch.csv`;
- `plan.json` and `targets_run.txt`;
- `analysis.md`;
- `logs/`.

## Files

| File | What |
|---|---|
| `arms.py` | The arm table, the lr-budget match (`lr_at` transcribed from learned_mask; asserts production Σlr 21.40), site counts |
| `make_targets.py` | Builds `targets.txt` / `targets.json` (run once, CPU) |
| `fit_arm.py` | The wrapper around the production driver (`CHECK=1`: CPU settings check) |
| `plan.py` | Sizes N from the timing stage (`PRIOR=1`: dry-run prior) |
| `run.sh` | The queue script (timing → plan → fits → donor → analysis, watchdog, resumable) |
| `analyze.py` | Per target × arm table, per-arm roll-up, Jaccard, the reading-rule numbers |

## Results

(none yet: the GPU run is queued)
