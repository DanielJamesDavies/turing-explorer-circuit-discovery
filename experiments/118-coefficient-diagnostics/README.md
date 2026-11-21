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
| (smoke) | one target each of `OBJ=tk` (4 arms) and `DIAG_A1=1`, 2026-11-16 | 1 + 1 | 01:55–02:02 | `out/smoke_tk/`, `out/smoke_a1/` |
| 3 | refits, `OBJ=tk`: global / site / member / member-warm (51 before the 55-min cap, 3 in a resumed run) | 54 of 54 | 02:03–03:14 | `out/fit_tk/job3.log`, `job3b.log` |
| 4 | α = 1 edited capture (off-manifold control) | 216 | 03:14–03:26 | `out/diag_a1/` |

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

### 2. Refits with members fixed

**Job 2 (`OBJ=pre`, n = 70 of the 108 planned; 24 / 24 / 22 per band).** Production α passes 85.7% [77.1, 92.9].

| arm | coefficients (median) | pass % [95% CI] | training loss (pre) |
|---|---|---|---|
| one scalar per circuit | 1 | 32.9 [21.4, 44.3] | 0.30 |
| one scalar per site | 16 | 37.1 [25.7, 48.6] | 0.185 |
| one per member | 638 | 35.7 [24.3, 47.1] | 0.014 |
| one per member, in [0.5, 2] | 638 | 34.3 [22.9, 45.7] | 0.032 |
| production α | 638 | 85.7 [77.1, 92.9] | 0.010 |
| α = 1 | 0 | 2.9 [0.0, 7.1] | 2.26 |

The untied refit reached production's training loss (0.014 vs 0.010) but failed held-out, mostly through Z (median
0.71: the target drops out of the Top-K and that read scores 0). That pointed at the objective, so job 3 changed it.

**Job 3 (`OBJ=tk`: the pass rule's post-Top-K read plus the censor hinge; n = 54, 6 per band × kind cell, complete).**

| arm | coefficients (median) | pass % [95% CI] | Δ vs production, pp [CI] | median Z / A / C | training loss (tk) |
|---|---|---|---|---|---|
| production α | 628 | 85.2 [75.9, 94.4] | | 0.92 / 0.99 / 0.96 | 0.010 |
| α = 1 | 0 | 3.7 [0.0, 9.3] | −81.5 | 0.00 / 0.01 / 0.04 | 2.17 |
| each member at its site median α (no refit) | | 5.6 [0.0, 13.0] | −79.6 | 0.01 / 0.18 / 0.25 | |
| one scalar per circuit | 1 | 33.3 [20.4, 46.3] | −51.9 [−64.8, −38.9] | 0.45 / 0.60 / 0.57 | 0.67 |
| one scalar per site | 15 | 37.0 [24.1, 50.0] | −48.1 [−61.1, −35.2] | 0.65 / 0.94 / 0.83 | 0.33 |
| one per member, from α = 1 | 628 | 44.4 [31.5, 57.4] | −40.7 [−53.7, −27.8] | 0.76 / 0.99 / 0.97 | 0.044 |
| one per member, from production α (warm) | 628 | 83.3 [72.2, 92.6] | −1.9 [−14.8, +11.1] | 0.96 / 1.00 / 0.99 | 0.003 |

- **The corrected objective lifts the per-member arm only a little.** Paired on the same 54 targets, tk − pre:
  per member +5.6 pp [−1.9, +14.8] (38.9 → 44.4), per site −1.9 [−9.3, +3.7], per circuit 0.0 [−5.6, +5.6]. The
  per-member arm stays about 41 pp below production.
- **Paired between arms (job 3):** per member − per site +7.4 pp [−3.7, +18.5]; per site − per circuit +3.7
  [−5.6, +14.8]; per member − per circuit +11.1 [−1.9, +22.2]. A refitted per-site scalar beats the per-site median of
  production α by 31.5 pp [18.5, 44.4].
- **By band (production / per site / per member):** L0-4 89 / 72 / 67; L5-7 100 / 39 / 61; L8-11 67 / 0 / 6 (n = 18
  each). Per-site scalars are close to production only in L0-4 and fail every deep target.
- **The warm start is a check on the objective and passes it.** Started from production α, the same 40 steps keep the
  pass rate (83.3 vs 85.2) and lower the training loss (0.003 vs 0.010). There is churn: 39 of the 46 production
  passes still pass, and 6 of the 8 production failures now pass. So the objective is consistent with production α;
  the cold-start fitter just does not find it.
- **The cold per-member fit now under-fits rather than over-fits:** its training loss is 0.044 against production's
  0.010, and its held-out Z is still low (median 0.76), with 22 of the 46 production passes failing.
- Per-site scalars (896 sites): median 1.46 (p10 0.54, p90 2.29); Spearman 0.35 with the site's median production α
  and 0.18 with the site's SAE error.

**Reading.** The refit arms are still fitter-limited: with 40 cold-start steps even the untied parameterisation reaches
only 44%, about half of production's 85%. Under that fitter, one scalar per site (37%) is not distinguishable from one
per member (paired +7.4 pp, CI includes 0), and both are far below production. What this run does show is that
production α is a stable optimum of the pass-rule objective (warm start). Per-site scalars reach 5.6% as site medians of
production α and 37% when refitted, with no deep target passing; a stronger fitter (production's 400 steps with gates
co-adapted) could still raise them, and this run cannot bound how far.

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

### 4. Off-manifold check, with the α = 1 control (n = 216)

The reference is each member's largest natural value over the target's 64 activating rows (positions ≤ anchor); the
corpus z-score uses `latent_stats`. Shares are over active (member, held-out context) pairs, value = max over positions
≤ anchor; median over targets (mean in brackets).

| share of active pairs | natural | natural × α | Z run, fitted α | A run, fitted α | Z run, α = 1 | A run, α = 1 |
|---|---|---|---|---|---|---|
| above the local natural max | 0 | 19% (23%) | 36% (42%) | 35% (41%) | 10% (16%) | 4.8% (7.5%) |
| above 1.5 × local max | 0 | 7.0% (8.7%) | 22% (30%) | 21% (30%) | 4.7% (11%) | 1.2% (4.3%) |
| corpus z > 6 | 1.4% (2.0%) | 9.9% (11%) | 23% (31%) | 23% (30%) | 5.4% (12%) | 3.1% (5.4%) |
| members ever above 1.5 × local max | 0 | 19% | 29% | 29% | 6.8% | 3.4% |

- **Most of the excess is due to α, not to the ablation.** Paired fitted − α = 1 (n = 216), share above the local max:
  Z +25.8 pp [23.1, 28.5], A +33.2 pp [30.3, 36.2]. Measured by the means, the α = 1 ablation alone accounts for 38%
  of the fitted-α excess in the Z run and 18% in the A run; α accounts for the remaining ~62% (Z) and ~82% (A). The
  same split holds for the 1.5 × max and z > 6 measures (Z 37–39%, A 14–18% from the ablation).
- **Site reconstruction norm, edited / natural at the anchor:** at α = 1 the circuit-only run leaves about half the
  natural norm (median 0.53 Z, 0.50 A; p90 0.97 / 0.93), and almost never exceeds the site's natural p99 (0.3% Z, 0% A).
  With fitted α the median comes back to 0.91 (Z) / 0.98 (A), but the p90 is 1.8–1.9 and 12% of (site, context) pairs
  exceed the natural p99.
- So α largely restores the site norm the ablation removes, and in doing so drives about a third of active members
  above anything they reach naturally on these contexts, with a tail of sites over-shooting the natural norm.

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

## Proposed paper text (draft; Daniel decides; paper/ not edited)

**Body clause (coefficients paragraph).** "...the coefficients are not a per-site gain: replacing each member's α by its
site's median recovers 1% of the α = 1 → fitted gap (3.7% vs 81.5% pass on 216 paired targets), while clipping α to
[1/3, 3] keeps 96% of it. Refitting one scalar per site with members fixed reaches 37% (54 targets), against 85% for
production α; under the same short fitter an untied per-member refit reaches 44%, so this comparison is bounded by the
fitter rather than settled. α is uncorrelated with the site's SAE reconstruction error (Spearman 0.08)."

**Appendix.**

- Table: the no-refit variants (§1).
- Refit paragraph: "With members fixed and 40 Adam steps on the pass rule's own post-Top-K read, one scalar per circuit
  passes 33%, one per site 37% and one per member 44% (production 85%; n = 54, paired). Started from production α, the
  same fitter keeps 83%, so the objective is consistent with production; the cold-start fitter does not reach it. Per-site
  scalars fail every L8-11 target (0 / 18)."
- Effective-size paragraph: about 21% of members carry 80% of the contribution at the anchor (34% using the maximum over
  positions); 70% of members are active at ≥ 1 held-out anchor.
- Stability paragraph: Spearman 0.65 between α across refits; 95% of α ≥ 2 members stay above 1.
- Off-manifold paragraph: "In the circuit-only runs, 36% of active member values exceed the member's largest natural
  value on the target's own 64 activating contexts (22% exceed 1.5×). With every α set to 1 the same runs give 10% (Z)
  and 5% (A), so most of the excess comes from α rather than from the ablation: the ablation halves the site's
  reconstruction norm (median 0.53 of natural) and α restores it (0.91), over-shooting the natural p99 in 12% of
  (site, context) pairs."
