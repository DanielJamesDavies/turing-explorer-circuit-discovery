# 118 RESUME (gpu-44, DAN-44), written 2026-11-15 14:25 before the 15:05 shutdown

**UPDATE 2026-11-16 03:40: COMPLETE.** Smokes passed; job 3 (54/54, `out/fit_tk/`) and job 4 (216/216,
`out/diag_a1/`) ran 02:03–03:26; GPU lock released 03:26. `analyze.py` and `stability.py` re-run; README.md filled
(all sections); DAN-44 results comment posted. Nothing left to run. The text below is the earlier state, kept for
the record.

Status: GPU jobs 1 and 2 are done. Jobs 3 and 4 (about 70 min of GPU in total) are not run. README.md is a skeleton:
its Results, Caveats and Proposed paper text are placeholders. No DAN-44 results comment is posted yet. Nothing in
paper/ was touched.

Remove or overwrite `.gpu-next`: it still names gpu-44 from 11:53. The shutdown makes it stale.

## Results so far

All numbers below come from `python -I -X utf8 experiments/118-coefficient-diagnostics/analyze.py`, which writes
`out/report.md`.

**Setup.** 216 TuringLLM production circuits, 24 per depth band × kind cell, chosen at random.

- The pass rate is paired and local (this GPU), with the 062 merge pass rule.
- The reduced Z/A/C scorer matches the full production scorer exactly: max |diff| 0 on 216/216 targets, and the pass
  decision agrees on all 216.
- Necessity and vacuity come from the full score; necessity does not depend on α.

### No-refit variants (216 targets, job 1)

Fitted α passes 81.5% [76.4, 86.6] locally. Production on the same targets: 180/216.

| variant | pass % | change | share of the α = 1 → fitted gap recovered |
|---|---|---|---|
| α = 1 | 2.8 | | 0% |
| each member at its circuit median α | 3.7 | | |
| each member at its kind median α | 4.6 | | |
| each member at its site median α | 3.7 | | ~1% |
| α clipped to [0.8, 1.25] | 6.9 | | |
| α clipped to [0.67, 1.5] | 10.6 | | |
| α clipped to [0.5, 2] | 29.6 | −51.9 pp | 34% |
| α clipped to [1/3, 3] | 78.7 | −2.8 pp [−6.0, 0] | 96% |
| α capped at 2 (no lower clip) | 31.9 | | |

Failures sit on the low side: 209 of the α = 1 failures have a score below 0.75, and none have one above 1.5.

### Refits, members fixed (job 2)

- Setup:
  - `out/fit4`, OBJ=pre (pre-activation ratio loss): Adam, 40 steps, learning rate 0.1 with cosine decay, batch of 4
    from the 48 strongest-training rows, starting from α = 1.
  - 70 of the 108 balanced targets were done before the time cap: 24 L0-4, 24 L5-7, 22 L8-11.
- Production α passes 85.7% on these 70.

| arm | pass % [95% CI] | median coefficients | training loss (pre) |
|---|---|---|---|
| one scalar per circuit (fit_global) | 32.9 [21.4, 44.3] | | 0.30 |
| one scalar per site (fit_site) | 37.1 [25.7, 48.6] | 16 | 0.185 |
| one per member, same fitter (fit_member) | 35.7 [24.3, 47.1] | 638 | 0.014 |
| one per member, constrained to [0.5, 2] | 34.3 | | |
| production α | 85.7 | | 0.010 |
| α = 1 | 2.9 | | 2.26 |

- By band, per-site: L0-4 75%, L5-7 33%, L8-11 0%. Production: 88 / 100 / 68.

**Fitter-limited.** With the same fitter, the untied per-member refit passes no better (36%) than the per-site
scalars (37%).

- The per-member refit also reaches a training loss as low as production's (0.014 vs 0.010).
- Yet it fails held-out, mostly through Z (median Z 0.71): the target drops out of the Top-K, and that censored read
  scores 0.
- So this fitter, not the parameterisation, caps all the refit arms near 35%. The run cannot say whether per-site
  scalars could match production, only that they match an untied refit under the same fitter.
- What the production engine has and this fitter lacks:
  - 400 steps;
  - v4's censored-activation term;
  - the R term;
  - gates co-adapted with α.
- **Job 3 fixes the objective:** it uses the post-Top-K read with a censor hinge, and adds a warm-start control
  (per-member, started from production α) to show that the objective keeps production α passing.

### Effective size by contribution (216 targets, job 1)

Contribution = α × natural activation at the anchor × decoder norm. Values are medians over targets.

- **Members needed for 50 / 80 / 90% of the contribution:**
  - At the anchor: 50 / 136 / 199 members, i.e. 7.8% / 21% / 31% of n (median n = 660).
  - Using each member's maximum over positions up to the anchor: 12% / 34% / 49%.
- **Activity:** only 34% of member × context pairs are active at the anchor, and 70% of members are active at ≥ 1
  held-out anchor.
- **Per-context spread:**
  - The per-context N80 is 90, with a coefficient of variation across contexts of 0.09.
  - The Jaccard overlap of the per-context top-80% sets is 0.46.
  - The per-member coefficient of variation of contribution across contexts is 1.25.
- **Where the contribution goes:**
  - Members with α > 1 carry 80% of the contribution; those with α > 2 carry 22% (they are 14% of members).
  - Spearman(α, natural contribution) is ≈ 0.03.

### Off-manifold check (216 targets)

The reference is each member's largest natural value over the target's 64 activating rows.

- `top_ctx.pt` is NOT a maximum: `ctx_seq_val` holds per-sequence means.
- Pairs of an active member and a held-out context above the local natural max (median over targets):

  | values compared | share above the local max | share above 1.5 × local max |
  |---|---|---|
  | natural values, unscaled | 0 | |
  | natural × α | 19% | 7% |
  | edited Z run | 36% | 22% |
  | edited A run | 35% | 21% |

- Against the corpus statistics, z > 6: natural 1.4%, edited Z run 23%.
- **Site reconstruction norm,** edited / natural at the anchor:
  - median 0.91 (Z) and 0.98 (A), p90 1.8–1.9;
  - 12% of (site, context) pairs are above the site's natural p99, against 0% naturally.
- **The α = 1 control (job 4) is NOT run yet.** Without it, the share of the edited-run excess caused by the ablation
  itself rather than by α is unknown.

### Gain vs SAE error (3,782 target × site pairs)

- **Correlation with the site's relative error ‖x − x̂‖/‖x‖:**
  - Spearman with the site median α: 0.08 pooled.
  - Within a circuit: median 0.005 (IQR −0.23 to 0.22).
  - Within the same site across circuits: −0.10.
  - α by error quintile: 1.06, 1.16, 1.19, 1.22, 1.15 (not monotone).
  - Per-site fitted scalars vs the site's error: Spearman 0.14.
- **By site kind:**
  - MLP sites have the largest error (0.35) but not the largest α. Residual sites have the largest α (1.22) at error
    0.23.
- **No sign that α is error compensation.**

### Stability across refits (CPU, T5: 126 targets, 45,985 shared members)

- Membership Jaccard between the two fits: median 0.42.
- **Agreement of α between fit 1 and fit 2:**
  - Spearman 0.645 pooled; per target, median 0.63.
  - Median |log(a1/a2)| is 0.22; 51% of members agree within ×1.25.
- **Same side of 1 in both fits:**
  - 95% of members with α ≥ 2;
  - 89% of members with α in [1.25, 2);
  - 63% of members near 1 (0.8–1.25).
- Shared members have higher α than fit-1-only members: median 1.14 vs 1.03.
- Core refit vs fit 1: Spearman 0.54, median ratio 0.98.

## Remaining GPU step (~70 min), then CPU

Take the lock under the shared rules (no `.gpu-lock`, and `.gpu-next` absent or naming gpu-44), then write gpu-44 into
it. Run from WSL in the repo root.

**Job 3, refits with the tk objective** (54 targets, 6 per cell; ~50 min):

```
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && mkdir -p experiments/118-coefficient-diagnostics/out/fit_tk && HF_HUB_OFFLINE=1 OBJ=tk RUN_OUT=experiments/118-coefficient-diagnostics/out/fit_tk SKIP_DIAG=1 ARMS=fit_site,fit_global,fit_member,fit_member_warm STEPS=40 LR=0.1 FIT_BS=4 MAX_MIN=55 TARGETS=experiments/118-coefficient-diagnostics/targets_fit54.txt PYTHONPATH=src .venv/bin/python -X utf8 experiments/118-coefficient-diagnostics/run118.py > experiments/118-coefficient-diagnostics/out/fit_tk/job3.log 2>&1
```

- Run it with `run_in_background` and poll the log, since `wsl ... &` dies when the shell exits.
- It is resumable: rerunning skips targets already done.
- The OBJ=tk path and the fit_member_warm arm are **untested on GPU**. Smoke-test first with `LIMIT=1` on a scratch
  `RUN_OUT` and check that loss_last < loss_first and that warm stays near production.

**Job 4, the α = 1 off-manifold control** (216 targets; ~15 min):

```
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && HF_HUB_OFFLINE=1 DIAG_A1=1 ARMS= RUN_OUT=experiments/118-coefficient-diagnostics/out/diag_a1 PYTHONPATH=src .venv/bin/python -X utf8 experiments/118-coefficient-diagnostics/run118.py > experiments/118-coefficient-diagnostics/out/diag_a1/job4.log 2>&1
```

Create `out/diag_a1` first. This path is also untested. Release the lock with `delete_path`.

**Then, on the CPU:**

```
python -I -X utf8 experiments/118-coefficient-diagnostics/analyze.py      # sections 1, 1b (fit4), 1c (fit_tk), 2-4 + a1 control
python -I -X utf8 experiments/118-coefficient-diagnostics/stability.py
```

Fill README.md (Results, Caveats, Proposed paper text) from `out/report.md` and this file. Post the DAN-44 comment and
log `kind=done`.

## Caveats to carry into the README

- TuringLLM only. The public models were not run because of time.
- Local GPU. Every comparison is paired on this GPU.
- The refits use a simple fitter: 40 steps, no gates, no R term. See "Fitter-limited".
- Members are fixed and were selected under per-member α, which may favour per-member coefficients.
- The contribution measure is first order and taken at the anchor. Members act at all positions, so the
  position-max variant is also reported.
- The off-manifold reference is local (64 rows) plus corpus z-scores. No corpus maximum is stored.
- `out/fit/` is an aborted batch-8 attempt (memory spill) and is not used.

## Proposed paper text (draft; Daniel decides)

**Body clause (coefficients paragraph).** "...the coefficients are not a per-site gain: replacing each member's α by
its site's median recovers 1% of the α = 1 → fitted gap (3.7% vs 81.5% pass on 216 paired targets), while clipping α
to [1/3, 3] keeps 96% of it; α is uncorrelated with the site's SAE reconstruction error (Spearman 0.08)."

**Appendix.**

- Table: the no-refit variants.
- Effective-size paragraph: about 21% of members carry 80% of the contribution at the anchor, and 34% using the
  maximum over positions.
- Stability paragraph: Spearman 0.65 across refits; 95% of α ≥ 2 members stay above 1.
- The refit result, worded as fitter-limited until job 3 lands.
- The off-manifold paragraph only after the α = 1 control.
