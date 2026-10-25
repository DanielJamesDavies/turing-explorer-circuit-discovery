# 079 — WCM with a fourth training ablation: donor-resample backgrounds

**Status (2026-10-17):** pilot run (27 targets, results below). Silent-position term added afterwards; its rerun is set
up (`run_silent.sh`) but not launched.

## Why

073 (`experiments/073-strict-eval`) scored the 15k run's circuits with everything outside the circuit **resampled
from a real other context** (the causal-scrubbing test) instead of zero or a mean. 0.1% of circuits pass, against 43%
under the production test. The diagnostic (`073/diag.py`, 45 targets) located the failure:

* It is not Top-K censoring of the target or of the members (uncensored reads barely change it).
* With members **pinned** to their natural values, resid targets recover (0.92). So the node set is enough; the
  circuit cannot regenerate its members' values when the rest of the stream is another context (closure).
* With the context's own background, the fitted coefficients overshoot (resid 2.16x): they stand in for the ablated
  rest of the network.

The circuits were trained only against zero / mean backgrounds, so nothing asked them to survive a real one. This
adds that requirement to training.

## What the fourth term is

`rs_engine.py`. A fourth patcher over the same gates and coefficients as the Z / C / A patchers. Its floor (the value
a fully masked latent takes) is a **donor context's natural sparse code, position by position**:

    code = m * alpha * dense + (1 - m) * donor_code

Donors are redrawn at every forward of the term, one per training row, from the target's donor pool. The loss has the
same data, anchors, target and normaliser as the other terms, weighted `w_R / mean(target^2)`.

**src is not edited.** `install()` reads `src/circuit/instrument/learned_mask.py`, applies five exact text edits
(`EDITS`), and execs the result as `circuit.instrument.learned_mask`. Each edit must match exactly once, so a changed
src fails loudly. With weight 0 the patched engine adds nothing: it is the production engine. ablation_gradient
imports `run_learned_mask` at call time, so every fit after `install()` uses it.

**Training differs from the 073 eval in one stated way:** training applies no Top-K re-imposition. That matches the
C / A means ("no Top-K re-imposed in training", protocol v1), so donor entries are not budget-truncated.

## Design

| | |
|---|---|
| Targets | 27 = 3 per band x kind, the first of 073's diag targets (073's stratified order; all have random nulls) |
| Arms | `prod` (the 15k circuit, scored only); `W0` (refit, weight 0: production config on this GPU, the in-run baseline); `RS0.25`, `RS1` (weight 0.25, 1.0) |
| Config | protocol v1: triple floor gamma_C = gamma_A = 0.25, rank-keep 3e-3, lambda 1e-3, 400 steps, train-only means |
| Training donors | `mix`: the 48 train contrast contexts + 48 random corpus contexts from a training pool (seed 20261017), disjoint by corpus id from 073's evaluation pool (asserted) |
| Eval: production scorer | `driver.Runner.score`, held-out strongest: Z / A / C / necessity / induce, the pass rule |
| Eval: resample | `strict_eval.score_target(..., c_held=True)`: C donors = the 16 **held-out** contrast contexts (never a training donor), R donors = 073's pool (never a training donor), A donors = strongest-train (as in 073); plus 074's all-position metrics |

What to read: does `RS` raise the resample pass and resample C / R against `W0` while keeping the production pass?
What does it cost in circuit size and fit time? Do the coefficients come down (alpha median)? Does the Z false-firing
on contrast contexts (074) shrink?

## Files

| File | What |
|---|---|
| `rs_engine.py` | the patched engine (`install`, `EDITS`), `DonorFloors`, the donor patcher (`make_factory`) |
| `fit_score.py` | per target x arm: fit (or load), production scorer, resample + all-position eval; resumable |
| `report.py` | `out/report.md`: per arm and band; paired against W0 |
| `test_rs_engine.py` | CPU tests: edits apply once and compile; weight 0 builds nothing; donor densification; open gate = identity; closed gate = the donor code; donors redrawn per forward |

## Commands for tomorrow

1. CPU tests (done 2026-10-17, 6 passed):

   ```bash
   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/079-resample-training/test_rs_engine.py -q'
   ```

2. **Smoke** (1 target, all arms; ~10-15 min). Check: `train.jsonl` term names include `resample` for RS arms and not
   for W0; `donor_draws` ~ steps x accumulation; no OOM (if OOM: `RS_MICRO=2`); fit time per arm; W0's scores close to
   prod's (RTX vs H100 refit noise).

   ```bash
   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && LIMIT=1 OUT79=experiments/079-resample-training/out_smoke PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -X utf8 -u experiments/079-resample-training/fit_score.py'
   ```

3. **Pilot** (27 targets x 3 refits; time from the smoke, roughly 27 x 3 x fit time):

   ```bash
   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && nohup setsid env PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -X utf8 -u experiments/079-resample-training/fit_score.py > experiments/079-resample-training/run.log 2>&1 < /dev/null & disown; sleep 3'
   ```

4. Report: `REPORT=1` with the same command, or `python experiments/079-resample-training/report.py`.

Knobs: `WEIGHTS` (0.25,1.0), `DONORS` (mix | contrast | random), `ARMS` (W0,RS), `PER_CELL` (3), `STEPS` (400),
`RS_MICRO` (micro-batch for memory).

## Pilot results (2026-10-17, 27 targets, `out_smoke4/`)

Paired against W0: resample pass 0% -> 33% (RS0.25) / 37% (RS1); production pass 37% -> 52%; size 1.15x / 1.48x;
fits ~40% slower. But all-position correlation under the C fill fell 0.76 -> 0.46 / 0.45, with variance explained
strongly negative for attn and mlp targets.

**Trace** (`trace.py`, the four worst targets, `out_smoke4/trace.json`): RS circuits are exact at the anchor (0.93-1.08
of natural) and put 66-90% of their squared error at positions of the SAME activating passage where the target is
naturally silent (W0: 0-23%), mostly a few tokens after the anchor (median +3 to +5; 53-87% after it). One mlp target
also fires on 84% of contrast-context positions. About half of each RS circuit is new members. Reading: to hold the
target up against a real background at the anchor, training recruits passage-level drivers, and nothing in training
looks at the target anywhere but the anchor.

## The silent-position term (added 2026-10-17; not yet run on a GPU)

`rs_engine.py` edit 6 + `silent_penalty` / `make_silent_fn`. With `SILENT["weight"] = w_S > 0`, every data term on the
training activating contexts (every ablation, the donor one included) also pays

    w_S * mean_rows[ mean over silent positions of relu(pre - max(natural pre, 0))^2 ] / mean(natural anchor pre^2)

Silent = the target's natural post-Top-K read is 0 there; anchors excluded. Only a LIFT above the natural value is
charged, so a near-threshold position is not pushed further down. The natural reads come from each batch's own
tokens (computed once, cached by content), so they always match what the engine trained on. It does not yet cover
contrast-context positions (the one target that over-fired there).

Arm names: `W0`, `RS<w>`, `RS<w>_S<v>`, `W0_S<v>`. `REUSE_FROM=<dir>` loads circuits and rows already in that folder
instead of refitting or rescoring.

**Rerun (set up, not launched):** `run_silent.sh`. The pilot's 27 targets: prod / W0 / RS0.25 / RS1 are reused from
`out_smoke4`, and four arms are new: `RS1_S0.25`, `RS1_S1`, `RS0.25_S0.25`, and `W0_S1` (the silent term alone, as
a control). About 2 h. Output in `out_silent/`, with `report.md` written at the end. The report now has `fve (C)`,
`silent mass (C)` and `ctr fire (C)` columns, and paired changes in all-position r and silent mass against W0.

```bash
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && nohup setsid bash experiments/079-resample-training/run_silent.sh > experiments/079-resample-training/silent.log 2>&1 < /dev/null & disown; sleep 3'
```

What to read: does `RS1_S*` keep RS1's resample and production gains while bringing silent mass and all-position r
back to W0's level? Does `W0_S1` alone change anything?

CPU tests: 10 pass (the original 6 + penalty value and masking, zero when not lifted, gradient only where charged,
arm-name parsing).

## Known limits

* Training donors are a mix of contrast and random contexts; activating contexts are not used as donors (their mean
  leaks the target, 073's A leak 0.12).
* One donor per training row per forward: the term is stochastic across steps (as intended; it averages over
  backgrounds during training).
* Attention targets failed 073 even with pinned members, so this term is not expected to fix them; they are in the
  pilot to measure that.
