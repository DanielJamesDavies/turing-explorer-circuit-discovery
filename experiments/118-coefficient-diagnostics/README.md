# 118 Coefficient diagnostics (DAN-44): tied-coefficient baselines, band constraints, effective size, off-manifold, SAE error, stability

## Question

Are the fitted coefficients α = softplus(ψ) of the WCM v4 production circuits a per-member description of the
computation, or a coarse gain that one number per site (or per circuit) could supply? Are they error compensation?
Do α-scaled members push the stream off the natural manifold? How many members carry the contribution?

Exp 113 (CPU, archives) already gave the α distribution. This experiment adds the parts that need the model.

## Inputs (read only)

- Production circuits: T3 archive `109-pod-run-analysis/out/archives/pub-turingllm-T3-main/.../094-tl-v2-records/out_v2/main/circuits` (loaded, never written).
- Production context records: T2 `.../082-public-runs/out_turingllm/ctx_fresh` (the same `CTX_DIR` re-score as 115).
- Second fits and core refits: T5 `.../out_v2/stable_core/circuits/{fit2,core}` (126 targets; CPU only).
- Corpus statistics: `outputs/latent_stats.pt` (the store 113 verified).
  - `outputs/top_ctx.pt` was meant as the natural-maximum reference, but its `ctx_seq_val` is a per-sequence MEAN, 10–25× below the target peaks. It is not used. `out/natmax.npy` and the `natmax` / `nat64` arrays in `out/members/*.npz` hold those means and must not be read as maxima.
- No downloads. TuringLLM only; the public models were not run (see Caveats).

## Sample

`select_targets.py` (seed 20261115) draws 24 scored, non-vacuous production targets per depth band × site kind cell
(3 × 3), 216 in all, uniformly at random within each cell.

- 180 of the 216 pass in production.
- On this GPU, 176 pass (81.5%). All comparisons below are paired against this local re-score.

The refits use a balanced subset, `targets_fit.txt`: 12 per cell, interleaved so that a partial run stays balanced.

## Method

**Scorer** (`run118.py`). For each target:

- The production scorer runs once at the fitted α: 062 `driver.Runner.score`, `MODE=main`, `WCM_VERSION=v4`, `CTX_DIR`
  = T2 records, explicit split. This gives necessity, vacuity and the reference Z / A / C.
- A **reduced scorer** then computes Z, A and C (`free0_tk`, `freeM_topk_tk`, `freeN_topk_tk`) for any coefficient
  vector, through 049 `score_circuit`'s own path: `V.read`, the engine `CircuitOnlyPatcher` with `keep_scales`, and the
  same contexts, means and empty baselines.
  - Checked on all 216 targets: max |diff| from the full scorer is 0 exactly, and the pass decision agrees 216 / 216.
- Necessity (`phi_sup_blind`: members set to 0) does not depend on α, so it is taken from the full score.
- Pass rule: 062 `merge` (Z, A, C ∈ [0.75, 1.5], necessity ≥ 0.9). Vacuous rule v2.

**Variants without refit:**

- α = 1.
- The fitted α clipped to [1/c, c], for c ∈ {1.25, 1.5, 2, 3}.
- The fitted α capped above at 2.
- Every member set to its site's, its kind's or its circuit's median fitted α.

**Refits.** Members are fixed and only the coefficients are refitted, starting from α = 1.

- Arms:
  - one scalar per circuit (`fit_global`);
  - one scalar per site (`fit_site`; median 16 per circuit);
  - one per member (`fit_member`: the control, same fitter);
  - one per member constrained to [0.5, 2] via α = 2^tanh(ψ) (`fit_member_c2`);
  - in job 3, one per member started from the production α (`fit_member_warm`: checks that the objective is
    consistent).
- Fitter: Adam on ψ, 40 steps, cosine learning rate from 0.1, 4 of the 48 strongest-training rows per step. The loss is
  summed over the three fills (zero; A and C sparsity-preserving, with the scorer's own training means), using the
  scorer's training empty baselines.
  - Job 2 (`OBJ=pre`): (ratio − 1)² on the target's relu pre-activation.
  - Job 3 (`OBJ=tk`): the post-Top-K ratio the pass rule uses, with a straight-through surrogate, plus a hinge
    (k-th largest site activation − pre) / (a − e). The hinge pushes a censored target back inside the Top-K, which is
    the role of v4's censored-activation term.
- Held-out rows are never seen by a fit.

**Diagnostics** (job 1, all 216 targets):

- A natural run on the 64 activating rows records each member's code at the anchor and its maximum over positions
  ≤ anchor, and the site SAE relative error ‖x − x̂‖ / ‖x‖ at the anchor.
- The fitted-α Z and A runs on the 16 held-out rows record each member's edited value (α × live value) and the edited
  site reconstruction norm.
- Job 4 (`DIAG_A1=1`) repeats the edited capture at α = 1 as the control.

**Contribution** = α × (natural activation at the anchor) × ‖decoder column‖. Also reported with the activation's
maximum over positions ≤ anchor, because the edit acts at every position.

## Runs (local RTX 5070 Ti, under the shared GPU lock)

| job | what | targets | wall | log |
|---|---|---|---|---|
| 1 | full + reduced score, all no-refit variants, diagnostics | 216 | 09:41–10:20 | `out/job1.log`, `out/rows.jsonl`, `out/members/` |
| (2a) | refits, `OBJ=pre`, batch 8: **aborted** after 8 targets (deep targets spilled past 16 GB; 616 s / target) | 8 | 10:21–10:40 | `out/fit/` (not used) |
| 2 | refits, `OBJ=pre`, batch 4: global / site / member / member-c2 | 70 of 108 (time cap) | 10:41–11:51 | `out/fit4/` |
| 3 | refits, `OBJ=tk`: global / site / member / member-warm | see below | | `out/fit_tk/` |
| 4 | α = 1 edited capture (off-manifold control) | 216 | | `out/diag_a1/` |

CPU: `stability.py` (T5 refits), `analyze.py` (writes `out/report.md` and the csvs).

## Results

All numbers come from `analyze.py` (`out/report.md`) and `stability.py`. Intervals are 95% bootstrap (4,000
resamples over targets). The pass rule is the 062 merge rule throughout, and every comparison is paired on this GPU.

### 1. Coefficient variants without refit (n = 216)

Fitted α passes 81.5% [76.4, 86.6] in the local re-score (production on the same targets: 180 / 216 = 83.3%).

| variant | pass % [95% CI] | Δ vs fitted, pp [CI] | share of the α = 1 → fitted gap |
|---|---|---|---|
| fitted α | 81.5 [76.4, 86.6] | | 100% |
| α = 1 | 2.8 [0.9, 5.1] | −78.7 [−84.3, −73.1] | 0% |
| each member at its circuit median α | 3.7 [1.4, 6.5] | −77.8 | 1% |
| each member at its kind median α | 4.6 [2.3, 7.4] | −76.9 | 2% |
| each member at its site median α | 3.7 [1.4, 6.5] | −77.8 [−83.3, −72.2] | 1% |
| α clipped to [0.8, 1.25] | 6.9 [3.7, 10.6] | −74.5 | 5% |
| α clipped to [0.67, 1.5] | 10.6 [6.5, 14.8] | −70.8 | 10% |
| α clipped to [0.5, 2] | 29.6 [23.6, 35.6] | −51.9 [−58.8, −44.9] | 34% |
| α clipped to [1/3, 3] | 78.7 [73.1, 83.8] | −2.8 [−6.0, 0.0] | 96% |
| α capped at 2 (no lower clip) | 31.9 [25.9, 38.0] | −49.5 [−56.5, −42.6] | 37% |

- The failures are on the low side: 209 of the 210 α = 1 failures have a score below 0.75 (or undefined), none above 1.5.
- By band, fitted / α = 1 / site median / clip [0.5, 2]: L0-4 86 / 8 / 10 / 67; L5-7 90 / 0 / 1 / 14; L8-11 68 / 0 / 0
  / 8 (n = 72 each).
- So the members' individual α matter: a per-site constant taken from the same α loses almost the whole effect, and
  the large coefficients (between 2 and 3) carry about two-thirds of it.

JOB3_PLACEHOLDER

### 3. Effective size by contribution (n = 216)

Contribution = α × natural activation at the anchor × decoder norm, on the 16 held-out rows. Medians over targets
(IQR); median circuit size n = 660 (395–960).

| quantity | at the anchor | max over positions ≤ anchor |
|---|---|---|
| members carrying 50% | 50 (32–74), 7.8% of n | 80 (52–117), 12% |
| members carrying 80% | 136 (88–202), 21% | 211 (136–323), 34% |
| members carrying 90% | 199 (129–297), 31% | 306 (191–461), 49% |
| per-context N80 | 90 (52–128), CV across contexts 0.09 | 183 |
| Jaccard of per-context top-80% sets | 0.46 (0.26–0.64) | 0.62 |
| members active (member × context) | 34% | 76% |

- 70% of members are active at the anchor in at least one held-out context.
- Members with α > 1 carry 80% of the contribution; members with α > 2 are 14% of members and carry 22% (30% with the
  position max).
- Spearman(α, natural contribution) ≈ 0.03: α does not just boost the members that already contribute most.
- By band, the 80% share is 22% / 24% / 18% at the anchor (L0-4 / L5-7 / L8-11).

JOB4_PLACEHOLDER

### 5. Gain vs SAE reconstruction error (3,782 target × site pairs, 216 targets)

- Spearman of the site median α with the site's relative error ‖x − x̂‖ / ‖x‖: 0.08 pooled; within a circuit median
  0.005 (IQR −0.23 to 0.22, n = 193 circuits); within the same site across circuits −0.10 (33 sites).
- Site median α by error quintile: 1.06, 1.16, 1.19, 1.22, 1.15 (not monotone).
- MLP sites have the largest error (0.35) but not the largest α (1.14); residual sites have the largest α (1.22) at
  error 0.23.
- The per-site refitted scalars (job 2, 1,177 sites) correlate 0.14 with the site error and 0.35 with the site's median
  production α.
- No sign that α compensates for SAE error.

### 6. Stability across refits (CPU, T5: 126 targets, 45,985 shared members)

- Membership Jaccard between the two fits: median 0.42 (IQR 0.32–0.53).
- α agreement on shared members: Spearman 0.645 pooled, per target median 0.63 (IQR 0.53–0.74); median |log(a1/a2)|
  0.22; 51% agree within ×1.25.
- Same side of 1 in both fits: 95% of members with α ≥ 2 (n = 7,314), 89% with α in [1.25, 2), 63% near 1
  ([0.8, 1.25)), 78% with α < 0.8.
- Shared members have higher α than fit-1-only members (median 1.14 vs 1.03).
- Core refit vs fit 1: Spearman 0.54, median ratio 0.98.

## Caveats

- TuringLLM only; the public models were not run because of time.
- Local GPU. Every comparison is paired against the local re-score, which reproduces production exactly on the reduced
  scorer (216 / 216) but passes 176 vs 180 production targets.
- The refits use a simple fitter: 40 Adam steps, no gates, no R term, members fixed. Members were selected under
  per-member α, which may favour per-member coefficients over tied ones.
- The refit subsets are small (job 2: 70 of 108 planned; job 3: 54), so their intervals are wide.
- The contribution measure is first order and taken at the anchor. Members act at every position, so the
  position-max variant is also reported.
- The off-manifold reference is local (64 activating rows) plus corpus z-scores; no corpus maximum is stored
  (`top_ctx.pt` holds per-sequence means).
- `out/fit/` is an aborted batch-8 attempt (memory spill) and is not used. `out/smoke_*` are one-target smoke runs.

## Proposed paper text

PAPER_PLACEHOLDER
