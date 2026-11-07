# 100: causal-scrubbing frontier (donor-background robustness vs size, v2CR, failure modes, pod pricing)

Agent scrub-e123; Daniel out, local GPU authorised until 01:00.

**What this is.** Development experiments that decide which donor-background ("causal-scrubbing") stages go into
the 8xH100 pod run. The panel is 099's frozen, availability-based 18-target panel (two targets per depth band x
site kind; 9 development + 9 check), with few firing contexts per target (16 held-out activating bases). It is not
a population sample. The pod run gives the population-scale confirmation; nothing here is a bank-wide rate.

## Design (fixed before any result in this folder was seen)

| | |
|---|---|
| Panel | 099 `panel.json`: `targets_dev9.txt` (development), `targets_check9.txt` (check); copies of 099's lists |
| Records | 099's audited clean records `099/out/records/ctx_v2` (094 machinery), no rebuild; per-record SHA-256 checked against 099's manifest on every run |
| Fits | standard v2 WCM (062 `driver.Runner.method/fit`, `WCM_VERSION=v2`, gamma 0.25, rank-keep 3e-3, 400 steps) at lambda in {1e-3, 2.5e-4, 5e-5, 1e-5}; v2CR = 095's arm (Z + A + C + R_C, R weight 0.25, donors = the 48 training contrast contexts redrawn per forward, sparsity-preserving donor fill, no silent term; 095 `rs2_engine` + 091 `setup_terms`, imported) |
| Reuse | lambda 1e-3 WCM = 099's fitted circuits. 099's manifest source hashes (320 files) match the current tree except `099/run_panel.py` (the documented post-run launcher edit); records match; steps 400. Additionally two 099 circuits (2.attn.34021, 10.resid.8194) were refitted here through the same process and are **bit-identical** (`out/refit_check/fits_check.jsonl`) |
| Random control | per arm: the arm's per-site member counts, drawn (seeded per target and arm) from latents active post-Top-K on the 48 training contexts, **excluding the arm's members**; coefficients fitted with the arm's own objective under 062 `random_null`'s support restriction (lambda 0, gates open, theta_init 40, binarize none). Random vs WCM therefore differs in membership, not gain capacity |
| Production score | `driver.Runner.score`, held-out strongest: post-Top-K rule pass (Z, A_sp, C_sp in [0.8, 1.5], necessity >= 0.9) and protocol 12's pre-activation pass (Z and necessity on the 0 baseline; 095 definitions). Also the circuit at alpha = 1 and the random control |
| Donor test | **exactly 099's** whole-donor background test (`099/run_baseline.py`): bases = held-out activating + held-out silent; donor pools A = training activating, C = held-out silent, R = 099's filtered 512-context random pool; 8 paired draws per pool from 099's generator sequence (the donors are identical to 099's); 073 `ResamplePatcher` (live SAE error, Top-K fill budget). Variants: `fit` (live members, fitted gains), `a1` (gains 1), `pinned` (members at their clean natural value x fitted gains), `rand`, `rand_pinned`, `empty` |
| Self donor | each base as its own donor: `empty` / `a1` are identities; `fit` / `rand` measure own-background gain overshoot |
| Pooled recovery | 099 `final_report.py`: (mean_draws(out) - mean_draws(empty)) / (natural - mean_draws(empty)); undefined (denominator <= 1e-9) stays undefined |
| Donor pass | 099: production free0_tk in [0.8, 1.5], necessity >= 0.9, pooled A and C recovery each in [0.8, 1.5] |
| Verification | `check_vs_099.py`: on 2.resid.7349 (smoke) 100's scorer reproduces 099's donors and per-draw anchor means for empty / fit / a1 / pinned, and the production scores, with max abs difference **0** |

### Pre-stated rule for E2 at a weaker lambda (written before any E1 result)

"Clearly rising" at lambda L means, over the 9 development targets, paired against lambda 1e-3 for the same target:
the median paired change in pooled donor recovery is >= +0.05 for **both** A and C, and A and C each rise on at least
6 of the 9 targets. v2CR is then fitted at the **smallest** lambda in {2.5e-4, 5e-5, 1e-5} satisfying this
("weakest" = weakest sparsity penalty). If no lambda satisfies it, no weaker-lambda v2CR arm is run and the frontier
is reported as flat. The rule uses development results only; check results are never used to pick lambda.

**Outcome of the rule (development):** not satisfied at any lambda (median paired dA / dC: 2.5e-4
-0.011 / -0.001; 5e-5 -0.010 / -0.004; 1e-5 -0.051 / +0.012; A rises on 2-3 of 9 targets). By the rule, **no
weaker-lambda v2CR arm is run** (the brief's priority 5 is dropped by its own condition, not for lack of time).

### Hypotheses for the check targets

Stated in the section "Check hypotheses" below, written after the development results and **before** any check
target is scored in this experiment.

## Check hypotheses (written after the development results of Run A, before any check target is scored here)

The v2 lambda 1e-3 side of the check set is not untouched: 099 already scored those circuits (0/9 donor passes). The
new check evidence is v2CR, the weaker-lambda WCM fits, and their random controls. The extra WCM fits at 2.5e-4 and
5e-5 on the check targets are added (beyond the brief's E1 = development only) so that the flat-frontier reading
also gets a check.

| id | hypothesis (check, n = 9, paired against wcm_1e-3 unless noted) | from dev |
|---|---|---|
| H1 | v2CR raises pooled donor recovery: median paired dA >= +0.3 **and** dC >= +0.3, each positive on >= 7/9 targets | dA +0.76, dC +0.72, 9/9 |
| H2 | v2CR donor passes stay low: <= 3/9 (failures mostly C < 0.8 or production Z out of band) | 1/9 |
| H3 | v2CR does not raise the production pass rate (paired change <= 0) and lowers production Z (paired median dZ < 0) | 5 -> 4, dZ -0.05 |
| H4 | the random control fitted with v2CR's objective stays at median A and C recovery <= 0.05 (the gain is membership, not gain capacity) | -0.004 / 0.000 |
| H5 | v2CR's own-background (self-donor) ratio exceeds v2's (median), i.e. its gains overshoot the clean background | 1.51 vs 1.20 |
| H6 | weaker lambda does not help: the pre-stated 'clearly rising' criterion fails at 2.5e-4 and at 5e-5 | fails at both |
| H7 | pinned recovery rises with size (median pinned A at 5e-5 > at 1e-3) | 0.48 vs 0.19 |
| H6b | (added after the development lambda 1e-5 result, before any check lambda 1e-5 fit) the criterion also fails at 1e-5 on check | fails (dA -0.05, dC +0.01) |
| H8 | v2's C-donor loss is lost encoder signal, not Top-K competition: median normalised pre recovery under C < 0.1 and <= 2/9 targets with normalised pre - post > 0.2 | 0.038, 0/9 |

## Results E1-E3 (full tables `out/main/tables.md`, rows `out/main/rows.jsonl`)

Runs (in this order): A = dev x {wcm 1e-3, 2.5e-4, 5e-5, v2cr 1e-3}, 1 h 37 min; B = dev x wcm 1e-5, 26 min; C = check x {wcm
1e-3, v2cr 1e-3, wcm 2.5e-4, 5e-5}, 1 h 34 min; D = check x wcm 1e-5, 25 min. 0 errors, no rejected fits.
**Checks:** all 18 reused production circuits re-scored here reproduce 099 exactly (same donors; max abs difference 0
in every per-draw anchor mean for empty / fit / a1 / pinned and in the production scores; `check_vs_099.json`); two
refits are bit-identical to 099's circuits. Self-donor identities (empty, a1) hold at the anchors (anchor-mean ratio
0.88-1.01; all-position max abs up to ~4.4 from bf16 round trip / Top-K discontinuities, as 099 reported).
Provenance: `source_check.jsonl` flags `report.py` (Runs B, C) and `run.py` (Run C) as edited during those runs; the
edits were the report and the additive, default-off E4 arm grammar / engine switch (`--engine rs2` is the default and
the only path these runs used; each row carries the `run.py` hash of its own process).

### E1: the size frontier is flat (development, 9 targets; 099's donor test)

| lambda | median size [IQR] | prod pass | pre pass | donor A | donor C | donor R | donor pass | pinned A / C / R | random (fitted gains) A / C / R |
|---|---|---:|---:|---|---|---|---:|---|---|
| 1e-3 (production) | 406 [266, 596] | 5/9 | 5/9 | 0.072 [0.036, 0.195] | 0.008 [0.006, 0.011] | 0.005 [0.000, 0.006] | 0/9 | 0.19 / 0.18 / 0.16 | -0.01 / 0.00 / 0.00 |
| 2.5e-4 | 969 [666, 1503] | 6/9 | 5/9 | 0.026 [0.022, 0.063] | 0.006 [0.000, 0.008] | 0.008 [0.000, 0.011] | 0/9 | 0.33 / 0.14 / 0.19 | -0.00 / 0.00 / 0.00 |
| 5e-5 | 3131 [2270, 6404] | 7/9 | 3/9 | 0.051 [0.005, 0.105] | 0.013 [0.000, 0.050] | 0.007 [0.001, 0.037] | 0/9 | 0.48 / 0.22 / 0.29 | -0.00 / 0.00 / 0.00 |
| 1e-5 | 12701 [7335, 27981] | 7/9 | 7/9 | 0.021 [-0.020, 0.135] | 0.018 [0.007, 0.089] | 0.005 [0.004, 0.085] | 0/9 | 0.51 / 0.43 / 0.47 | -0.00 / 0.00 / 0.00 |
| v2CR 1e-3 (E2) | 437 [398, 755] | 4/9 | 5/9 | 0.873 [0.833, 0.940] | 0.721 [0.659, 0.804] | 0.578 [0.558, 0.672] | 1/9 | 0.33 / 0.28 / 0.29 | -0.00 / 0.00 / 0.00 |

Medians [IQR] across targets of 099's pooled recovery. Figure: `out/main/e1_recovery_vs_size.png` (+ `.pdf`).

- **A 30x larger circuit does not survive real backgrounds any better.** From 406 to 12,701 members the median live
  recovery stays at 0.02-0.07 (A) and 0.01-0.02 (C); 0/9 donor passes at every lambda. The pre-stated "clearly
  rising" rule fails at every lambda (median paired dA / dC vs 1e-3: -0.011 / -0.001, -0.010 / -0.004, -0.051 /
  +0.012; A rises on 2-3 of 9 targets). By that rule no weaker-lambda v2CR arm was run.
- Size helps production (5 -> 7/9 post-Top-K passes) and helps **pinned** recovery (A 0.19 -> 0.51, C 0.18 -> 0.43):
  the bigger supports carry more of the target's input when their members are handed their clean values, but they do
  not regenerate those values on a foreign background.
- **Membership matters.** Equal-size active random circuits with gains fitted by the same objective recover nothing
  (median <= 0.00 on A, C, R at every size, live or pinned) and pass production 0/9.
- **Check confirms** (n = 9; the v2 1e-3 side was already scored by 099): live A / C 0.135 / 0.006 (1e-3), 0.105 /
  0.006 (2.5e-4), 0.040 / 0.011 (5e-5); rule fails (dA -0.031 / -0.115, A rising on 1/9); pinned A 0.45 -> 0.48;
  random 0.00; donor pass 0/9 throughout. Lambda 1e-5 on check (Run D, median 9,485 members): live A / C /
  R 0.040 / 0.034 / 0.011, pinned 0.55 / 0.47 / 0.44, production 8/9, donor pass 0/9.

### E2: training against real backgrounds (v2CR) is what moves donor recovery

| | dev v2 | dev v2CR | check v2 | check v2CR |
|---|---|---|---|---|
| production pass (post-Top-K) | 5/9 | 4/9 | 8/9 | 6/9 |
| pre-activation pass | 5/9 | 5/9 | 8/9 | 7/9 |
| donor pass (099 rule) | 0/9 | 1/9 | 0/9 | 2/9 |
| donor A recovery | 0.072 [0.036, 0.195] | 0.873 [0.833, 0.940] | 0.135 [0.045, 0.454] | 0.834 [0.626, 1.022] |
| donor C recovery | 0.008 [0.006, 0.011] | 0.721 [0.659, 0.804] | 0.006 [0.003, 0.014] | 0.758 [0.664, 0.854] |
| donor R recovery | 0.005 [0.000, 0.006] | 0.578 [0.558, 0.672] | 0.002 [0.000, 0.014] | 0.634 [0.509, 0.788] |
| median paired dA / dC / dR | | +0.76 / +0.72 / +0.58 | | +0.77 / +0.73 / +0.60 |
| random control (v2CR objective) A / C | -0.01 / 0.00 | -0.00 / 0.00 | -0.00 / 0.00 | 0.00 / 0.00 |
| production Z, median paired change | | -0.05 | | -0.03 |
| size (median; paired ratio) | 406 | 437 (x1.16) | 451 | 569 (x1.29) |
| fit time ratio (median; of sums) | | x1.41; x1.41 | | x1.39; x1.39 |

- v2CR lifts every target's donor recovery (dev 9/9 A and C; check 8/9 A, 9/9 C) by ~0.6-0.8, with circuits only
  1.2-1.3x larger, and the random control under the same objective stays at 0: the gain comes from the members v2CR
  picks, not from extra gain capacity. Its median C recovery (0.72 / 0.76) sits just under the pass band, so the
  donor pass rate stays low (1/9, 2/9).
- It costs production passes on this panel (dev 5 -> 4, check 8 -> 6). The dev loss is 6.resid.35910, whose production
  Z collapses to 0.00 under v2CR; Z also collapses on 6.attn.35411 (0.73 -> 0.06), which already failed under
  v2: a circuit trained to hold up on real backgrounds can lose the zero-ablation test. Deep residual 10.resid.38616 (check) barely responds (A 0.25 -> 0.20,
  C 0.01 -> 0.11).
- These agree with 095 (leaky records, 72 targets: resample pass 0 -> 24%, production -7 pp, size x1.18, fit x1.45).

### E3: what fails

- **Gains are necessary, not overshooting on donors.** At alpha = 1 the same members recover nothing on any
  background (dev and check, v2 and v2CR: medians -0.01 to 0.00) and pass production 0-2/9. Fitted gains never push
  donor recovery above 1.5 for v2 (0 targets) and only on one A pool for v2CR. They **do** overshoot the target's
  **own** background: self-donor anchor ratio (fitted circuit, base context as its own donor) median 1.20 (v2 dev),
  1.18 (v2 check), 1.51 / 1.48 (v2CR dev / check); it falls with size (0.87 at 1e-5). Mean-background gains are tuned
  for the mean, not for any real context.
- **Closure / missing dependencies.** Pinning members at their clean values recovers far more than live members for
  v2 (dev A 0.07 -> 0.19, check 0.14 -> 0.45; up to 0.51 at 1e-5), as in 099. For v2CR the order reverses (live 0.87
  > pinned 0.33): v2CR's members are chosen and weighted to be regenerated on donor backgrounds.
- **Lost encoder signal, not lost Top-K competition.** Under C donors v2's normalised pre-activation recovery is
  0.04 (dev) / 0.01 (check), the same as the post-Top-K recovery; 0/9 targets show a pre-post gap > 0.2 at any
  lambda (1/9 for v2CR check). The circuit does not build the target's input on a foreign background; it is not a
  near miss lost at the Top-K cut.
- **What A-donor recovery there is comes from donors that fire themselves.** Splitting (draw, base) pairs by whether
  the donor context fires for the target at the base's anchor (10-16% of A pairs): v2 recovers 0.50 (dev) / 1.04
  (check) on firing donors and 0.02 / 0.02 on silent ones. v2CR recovers 1.58 / 1.25 on firing donors and 0.84 / 0.80
  on silent ones. v2's A score is the donor's own activation leaking through; v2CR genuinely rebuilds the target.

### Cost (measured on the RTX 5070 Ti; projections for the pod)

Per circuit, local, medians by band (donor = 3 pools x 8 draws of 32 bases): shared per-target setup 3.1 / 6.7 / 11.5 s
(L0-4 / L5-7 / L8-11); live donor variant 2.3 / 5.1 / 8.4 s; live + a1 + pinned 7.1 / 16.4 / 26.0 s; production score
1.5 / 2.9 / 4.6 s; v2 fit 28 / 60 / 97 s; v2CR fit x1.40. Donor time grows with depth, barely with size (live: +0.05
s per 1,000 members). Linear fits and the full table are in `out/main/tables.md`.

Projection: an H100 process 2.0-2.5x faster than this GPU, 6 processes per GPU under MPS (062 pod), 8 GPUs; layer mix
from 094's 8k and sweep-190 lists; a rough model from 18 targets at 3 depths, not a benchmark.

| stage | circuits | local hours | pod wall-clock (2.0x / 2.5x) | pod GPU-h |
|---|---:|---:|---|---|
| (a) donor scoring, 8k main, live only | 8,000 | 26.7 | 0.3 / 0.2 h | 2.2 / 1.8 |
| (a) donor scoring, 8k main, live + a1 + pinned | 8,000 | 50.7 | 0.5 / 0.4 h | 4.2 / 3.4 |
| (a) 2k stratified subset, live + a1 + pinned | 2,000 | 12.7 | 0.1 / 0.1 h | 1.1 / 0.8 |
| (a) 1k stratified subset, live + a1 + pinned | 1,000 | 6.3 | 0.1 / 0.1 h | 0.5 / 0.4 |
| (b) donor scoring, 190 x 8 WCM lambda, live + a1 + pinned | 1,520 | 9.7 | 0.1 / 0.1 h | 0.8 / 0.6 |
| (c) v2CR fits, 190 sweep targets, lambda 1e-3 (x1.40 v2) | 190 | 4.3 | < 0.1 h | 0.4 / 0.3 |
| (c) ... plus production + live donor scoring | 190 | 5.1 | 0.1 / < 0.1 h | 0.4 / 0.3 |
| (for scale) v2CR on all 8k main targets (fit only, no random control; mean layer 5.5) | 8,000 | ~177 | ~1.8 / 1.5 h | ~15 / ~12 |

Random controls with fitted gains double any fit stage. Add one model load per process (~1 min) and the pod's own
overheads; the donor stages are cheap next to the fits.

## Check hypotheses, outcome

| id | dev | check | verdict |
|---|---|---|---|
| H1 v2CR dA, dC >= +0.3, >= 7/9 | +0.76 / +0.72, 9/9, 9/9 | +0.77 / +0.73, A 8/9, C 9/9 | confirmed |
| H2 v2CR donor pass <= 3/9 | 1/9 | 2/9 | confirmed |
| H3 production pass not up, Z down | 5 -> 4, -0.05 | 8 -> 6, -0.03 | confirmed |
| H4 random control (v2CR objective) <= 0.05 | -0.00 / 0.00 | 0.00 / 0.00 | confirmed |
| H5 v2CR self-donor ratio > v2's | 1.51 vs 1.20 | 1.48 vs 1.18 | confirmed |
| H6 rule fails at 2.5e-4 and 5e-5 | fails | fails (dA -0.03 / -0.12) | confirmed |
| H6b rule fails at 1e-5 | fails | fails (dA -0.09, A up 2/9; dC +0.013, C up 8/9 but tiny) | confirmed |
| H7 pinned A larger at 5e-5 than 1e-3 | 0.48 vs 0.19 | 0.48 vs 0.45 | confirmed, but small on check |
| H8 C loss = encoder signal (norm pre < 0.1; <= 2/9 gap) | 0.038, 0/9 | 0.009, 0/9 | confirmed |

## Decisions for the pod (E1-E3; see the E4 section for the training variant)

| decision | recommendation | evidence |
|---|---|---|
| (2a) donor-score the main circuits? | **Yes, all 8k**, with live + a1 + pinned and the empty baseline (099's test, 8 draws x A / C / R). Not a subset: ~0.5 h wall on the pod | v2's donor recovery is near 0 on 18/18 clean targets at every size, so the population number is the point, not a by-size sample; cost 4 GPU-h. A 1k-2k stratified subset only if pod time is tighter than ~0.5 h |
| (2b) donor-score the lambda sweep? | **Yes, all 1,520**, live + a1 + pinned | Cheap (~0.1 h wall); turns E1's 9 + 9-target flat frontier into a 190-target statement (does recovery stay flat over a 30x size range? is pinned recovery rising?) |
| (3) v2CR fits on the 190 sweep targets? | **Yes** (which variant: see the updated E4 recommendation below, R weight 1.0 and/or lambda 2.5e-4), with the donor test and production scoring; plus equal-size random controls fitted under v2CR's objective if pod time allows (doubles the fit cost) | +0.7 donor recovery on 18/18 targets, membership-specific (random 0), at 1.2-1.3x size and 1.4x fit time; but production pass falls (13/18 -> 10/18) and donor pass stays low (3/18), so the population trade-off needs the pod. ~0.1 h wall. A weaker-lambda v2CR is not justified by E1's rule |
| protocol 12 donor metric | add the paragraph below (draft, not applied) | keeps the pod comparable with 099 / 100 |

**Draft for `paper/protocol.md` §12 (not applied):**

> **12.3 Donor-background (causal-scrubbing-style) robustness, reported alongside the pass rule.** For every fitted
> circuit, the held-out bases (16 strongest held-out activating + 16 held-out contrast contexts) are run with every
> latent outside the circuit, at every upstream site, taken from a donor context's natural sparse code at the same
> position (073 `ResamplePatcher`: members at alpha x live; the donor's non-member active entries fill the remaining
> Top-K budget, largest first; the current stream's SAE error kept live). Three donor pools: A = the 48
> strongest-train activating contexts, C = the 16 held-out contrast contexts, R = a fixed random-corpus pool filtered
> against every record (exact / near rule). Eight paired draws per pool (one donor per base per draw, seeded per
> target), the empty circuit under the same donors. **Donor recovery** (per pool) = (mean over draws of the circuit's
> mean anchor read - the same for the empty circuit) / (natural mean anchor read - empty), pooled over draws before
> dividing; undefined when the denominator is <= 1e-9, and kept undefined. **Donor pass** = production Z in [0.8, 1.5],
> necessity >= 0.9, and donor recovery A and C each in [0.8, 1.5]. Reported with it, not inside it: R recovery, the
> alpha = 1 and pinned-member variants, the per-pool empty leak, and the split by whether the donor fires at the base
> anchor. The donor test is a robustness diagnostic, stated before any pod result; it is not recursive causal scrubbing
> and does not replace the production pass rule.

## E4 design and selection rule (Daniel's follow-up; written before any E4 fit or result)

Question: which changes to v2 improve donor-background robustness without hurting production? Development on the 9
dev targets; the best 1-2 arms are then confirmed on the 9 check targets. All arms lambda 1e-3, 400 steps unless
named; same donor test, random controls (fitted with the arm's own objective), production / pre-activation scores and
reporting as E1 / E2. Baselines (reused, not refitted): `wcm_1e-3` (v2) and `v2cr_1e-3` (v2CR) from Run A.
Engine: `e4_engine.py` = 095's rs2 engine + four default-off switches (second donor term, drop A, k donors per row per
step, alpha cap); before any E4 arm runs, a v2CR refit through it is compared with Run A's v2CR circuit.

| priority | arm | what changes vs v2CR (Z + A + C + R_C, R weight 0.25, 48 contrast-train donors) |
|---|---|---|
| 1 | `v2cr-w0.1`, `v2cr-w0.5`, `v2cr-w1` | R weight 0.1 / 0.5 / 1.0 |
| 2 | `v2cr-mixCR` | R donors = 48 contrast-train + 48 random-corpus **training** contexts |
| 2 | `v2cr-mixCAR` | R donors = 48 contrast-train + 16 mid-band training activating rows + 48 random training contexts |
| 3 | `rsAC` | all-resample: Z + RA (0.25) + RC (0.25); the A and C mean terms dropped; RA donors = the 16 mid-band training activating rows, RC donors = the 48 contrast-train rows |
| 4 | `v2cr-k4` | the R loss averaged over k = 4 donor draws per row per step (4 donor forwards per step) |
| 5 | `v2cr_2.5e-4` | v2CR at lambda 2.5e-4 |
| 6 | `v2cr-cap2` | v2CR with every amplitude capped at alpha <= 2 |
| 7 | `v2cr-s800` | v2CR at 800 steps |

**Training donors stay disjoint from the evaluation donors, as v2CR's are.** The evaluation donors are A = the 48
strongest-train activating contexts, C = the 16 held-out contrast contexts, R = 099's filtered random pool (plus the 16
held-out activating bases). Hence: activating training donors are the 16 **mid-band** training rows (the strongest-train
rows are the evaluation A pool; all 18 panel targets have mid-band rows); random training donors are 48 new corpus
contexts drawn with 073's procedure under seed 20261030 and filtered with 099's exact / near rule against every panel
record row and against the evaluation random pool (`out/e4_train_random_pool.pt`). Every fit re-checks exact row
overlap between its training donors and every evaluation pool and refuses to fit on any overlap (counts in
`fits.jsonl`, field `e4.overlap_exact`).

**Pre-flight.** A v2CR refit of 2.attn.34021 through `e4_engine` (every E4 switch off) is bit-identical to
Run A's v2CR circuit (`out/e4_check/fits_check.jsonl`). A 20-step smoke of every mechanism (`out/e4_smoke`, scores
meaningless) trained exactly the intended terms: `v2cr-w0.5` / `mixCAR` / `cap2` = zero, floor (C), pos (A),
resample; `rsAC` = zero + two donor terms (240 donor-fill calls = 2 x 120), no floor / pos; `k4` = 4 donor forwards per
step (480 calls, 80 draws). Training-donor overlap with every evaluation pool: 0 exact rows. The random training pool
(48 contexts) is `out/e4_train_random_pool.pt`.

**Selection rule (stated before any E4 result).** Score = median over the 9 dev targets of min(pooled donor A,
pooled donor C) recovery (live, fitted gains). Eligible: production post-Top-K pass count on dev >= 4 (within 1 of
v2's 5/9). Best = the eligible arm with the highest score; arms within 0.02 of the best score tie, and a tie goes to the
smaller fit-time ratio vs v2. Confirmation on check: the best arm, plus the second-best eligible arm if its score is
within 0.05 of the best's; an arm is confirmed on check only if its dev score beats v2CR's dev score (otherwise the
honest outcome is "nothing beats v2CR", and only v2CR's existing check result stands). Cut order if time runs short:
from the bottom of the priority table.

## E4 results (development ~6 h 46 min, then the check confirmation; tables `out/e4/tables_e4.md`)

Dev, 9 targets, every arm complete (0 errors, no rejected fits). Score = median min(donor A, donor C). Fit time =
arm / v2 per target (median). Paired = arm minus v2CR per target (median; targets where the arm is higher).

| arm | size (x v2) | prod pass | pre pass | donor A | donor C | donor R | score | donor pass | random A / C | fit time x v2 | paired vs v2CR |
|---|---|---:|---:|---|---|---|---:|---:|---|---:|---|
| v2 (`wcm_1e-3`) | 406 (x1.00) | 5/9 | 5/9 | 0.072 [0.036, 0.195] | 0.008 [0.006, 0.011] | 0.005 [0.000, 0.006] | 0.008 | 0/9 | -0.01 / 0.00 | 1.00 | -0.68 (0/9) |
| v2CR (`v2cr_1e-3`, R 0.25) | 437 (x1.17) | 4/9 | 5/9 | 0.873 [0.833, 0.940] | 0.721 [0.659, 0.804] | 0.578 [0.558, 0.672] | 0.682 | 1/9 | -0.00 / 0.00 | 1.41 | 0 |
| R 0.1 | 393 (x0.99) | 3/9 | 4/9 | 0.740 [0.610, 0.814] | 0.554 [0.534, 0.630] | 0.450 [0.315, 0.464] | 0.554 | 1/9 | -0.00 / 0.00 | 1.41 | -0.13 (0/9) |
| R 0.5 | 477 (x1.18) | 4/9 | 5/9 | 0.960 [0.782, 0.979] | 0.748 [0.697, 0.888] | 0.589 [0.512, 0.809] | 0.742 | 2/9 | -0.00 / 0.00 | 1.41 | +0.04 (6/9) |
| **R 1.0** | 521 (x1.40) | 4/9 | 6/9 | 0.936 [0.918, 1.008] | 0.785 [0.755, 0.914] | 0.670 [0.299, 0.849] | **0.785** | 2/9 | 0.00 / 0.00 | 1.41 | +0.11 (6/9) |
| donors C + random | 407 (x1.07) | 5/9 | 7/9 | 0.951 [0.824, 0.993] | 0.751 [0.688, 0.763] | 0.685 [0.635, 0.749] | 0.734 | 1/9 | 0.00 / 0.00 | 1.41 | +0.04 (5/9) |
| donors C + mid-A + random | 434 (x1.08) | 3/9 | 4/9 | 0.935 [0.879, 0.979] | 0.765 [0.663, 0.779] | 0.702 [0.558, 0.730] | 0.715 | 1/9 | 0.00 / 0.00 | 1.40 | +0.02 (5/9) |
| all-resample Z + RA + RC | 399 (x0.97) | **2/9** | 3/9 | 0.916 [0.547, 1.006] | 0.667 [0.579, 0.845] | 0.572 [0.346, 0.803] | 0.667 | **3/9** | -0.00 / 0.00 | 1.20 | +0.02 (6/9) |
| k = 4 donors per step | 400 (x1.01) | 4/9 | 5/9 | 0.884 [0.817, 0.961] | 0.779 [0.675, 0.833] | 0.630 [0.567, 0.797] | 0.779 | 1/9 | -0.00 / 0.00 | **2.99** | +0.02 (5/9) |
| **v2CR at lambda 2.5e-4** | 1021 (x2.93) | 5/9 | 5/9 | 0.953 [0.855, 0.989] | 0.766 [0.667, 0.861] | 0.641 [0.513, 0.765] | 0.766 | 3/9 | 0.00 / 0.00 | 1.41 | +0.06 (**8/9**) |
| alpha <= 2 | 420 (x1.17) | 4/9 | 6/9 | 0.841 [0.815, 0.911] | 0.657 [0.646, 0.804] | 0.528 [0.326, 0.627] | 0.657 | 2/9 | 0.00 / 0.00 | 1.42 | -0.03 (2/9) |
| 800 steps | 284 (x0.70) | 5/9 | 5/9 | 0.890 [0.701, 1.014] | 0.673 [0.555, 0.770] | 0.549 [0.355, 0.675] | 0.673 | 1/9 | -0.00 / 0.00 | 2.83 | +0.01 (5/9) |

Figure: `out/e4/e4_recovery_vs_pass.png` (+ `.pdf`): score vs production pass per arm, dev and check panels.

**Selection, as applied (pre-stated rule; `report_e4.py`).** Eligible (dev production pass >= 4/9): R 1.0 (0.785),
k4 (0.779), v2CR at 2.5e-4 (0.766), R 0.5 (0.742), C + random (0.734), v2CR (0.682), 800 steps (0.673), alpha <= 2
(0.657). Not eligible: all-resample (2/9), C + mid-A + random (3/9), R 0.1 (3/9). Best score 0.785; the tie band
(>= 0.765) holds R 1.0, k4 and v2CR at 2.5e-4, and the tie goes to the smallest fit-time ratio: **v2CR at lambda
2.5e-4** (x1.405 vs x1.411 for R 1.0: the tie-break is decided by a 0.4% timing difference, i.e. it is effectively a
coin toss between the two). Second-best within 0.05: **R 1.0**. Both beat v2CR's dev score (0.682), so both run on
check (`e4_check.sh`, started automatically when the dev chain finished).

**Reading (dev only, 9 targets; differences of 0.05 in the score are within the spread across targets).**
- **The R weight is the lever.** Score rises monotonically with it (0.55 / 0.68 / 0.74 / 0.79 at R 0.1 / 0.25 / 0.5 /
  1.0) at no extra fit time; the size grows with it (x0.99 -> x1.40) and production stays at 4/9. R 0.1 loses
  production (3/9).
- **Once R is in training, size does help a little.** v2CR at 2.5e-4 (x2.9 size) is the only arm above v2CR on 8/9
  targets and has the best production (5/9) and donor pass (3/9) of the eligible arms; this contrasts with E1, where
  size without R did nothing.
- **Donor mixes**: adding random training contexts lifts R recovery (0.58 -> 0.69) and production (5/9) but barely the
  score; adding mid-band activating donors costs production (3/9).
- **All-resample (Z + RA + RC)**, the purest causal-scrubbing-style objective, gets the most donor passes (3/9) but
  production collapses (2/9): the mean terms are what hold the production test.
- **k = 4 donors per step** (variance reduction) buys ~0.02 over v2CR at 3x the fit time; **800 steps** gives
  smaller circuits (x0.70) and production 5/9 but no robustness gain at 2.8x the time; **alpha <= 2** lowers the score
  slightly (0.657): robustness uses some large gains but does not depend on them.
- Random controls under every arm's objective stay at 0.00: every gain is membership-specific.

### E4 check confirmation (9 check targets, ~1 h 13 min; v2 and v2CR from Run C)

| arm | size (x v2) | prod pass | pre pass | donor A | donor C | donor R | score | donor pass | random A / C | fit time x v2 | paired vs v2CR |
|---|---|---:|---:|---|---|---|---:|---:|---|---:|---|
| v2 | 451 (x1.00) | 8/9 | 8/9 | 0.135 [0.045, 0.454] | 0.006 [0.003, 0.014] | 0.002 [0.000, 0.014] | 0.006 | 0/9 | -0.00 / 0.00 | 1.00 | -0.66 (0/9) |
| v2CR (R 0.25) | 569 (x1.29) | 6/9 | 7/9 | 0.834 [0.626, 1.022] | 0.758 [0.664, 0.854] | 0.634 [0.509, 0.788] | 0.758 | 2/9 | 0.00 / 0.00 | 1.39 | 0 |
| v2CR at lambda 2.5e-4 (rule-best) | 1217 (x3.01) | **7/9** | 7/9 | 0.912 [0.723, 0.981] | 0.744 [0.720, 0.850] | 0.671 [0.627, 0.802] | 0.729 | 2/9 | -0.00 / 0.00 | 1.42 | -0.01 (4/9) |
| R 1.0 (second) | 656 (x1.70) | 5/9 | 8/9 | 0.970 [0.870, 1.134] | 0.849 [0.715, 0.901] | 0.791 [0.729, 0.911] | **0.849** | **3/9** | -0.00 / 0.00 | 1.40 | **+0.09 (7/9)** |

- **R 1.0 replicates.** Its robustness gain over v2CR holds on check (+0.09, higher on 7/9 targets; dev +0.11, 6/9);
  donor C reaches 0.85 and R 0.79, and it has the most donor passes (3/9). Its cost also replicates: production 5/9
  against v2CR's 6/9 and v2's 8/9 (pre-activation pass 8/9 is unchanged: the losses are post-Top-K and Z: median Z
  0.89). Size x1.7, fit time unchanged vs v2CR.
- **The rule-best arm (v2CR at 2.5e-4) does not replicate its robustness gain** (check score 0.729 vs v2CR 0.758;
  higher on 4/9). It does keep production better (7/9; dev 5/9). Its selection over R 1.0 was a fit-time tie-break
  of 0.4%, i.e. arbitrary; on the robustness criterion R 1.0 is the better arm on both splits.
- Neither arm reaches a donor-pass majority: the median C recovery sits at 0.74-0.85, against a 0.8 threshold.
- Over dev + check (18 targets): v2 13/18 production, 0/18 donor; v2CR 10/18, 3/18; R 1.0 9/18, 5/18; v2CR at 2.5e-4
  12/18, 5/18. There is a real trade-off between the production test and real-background robustness on this panel;
  v2CR at 2.5e-4 is the best compromise on counts (production 12/18 and donor 5/18), R 1.0 the most robust.

### Updated pod recommendation (E4)

| decision | recommendation | evidence | pod cost |
|---|---|---|---|
| (3) the arm to fit on the 190 sweep targets | **v2CR with R weight 1.0 at lambda 1e-3** as the robustness arm, **and v2CR (R 0.25) at lambda 2.5e-4** as the production-preserving arm; both at 400 steps, k = 1, contrast-train donors, no alpha cap; plus v2CR (R 0.25, lambda 1e-3) as the 095-comparable reference if the budget allows | R 1.0: score 0.79 / 0.85 (dev / check), +0.1 over v2CR on 13/18 targets, 5/18 donor passes, but production 9/18 (v2 13/18). v2CR at 2.5e-4: production 12/18, donor 5/18, score no better than v2CR on check. No other variant beat v2CR on both production and robustness: k = 4 and 800 steps cost ~3x fit time for ~0; all-resample and mid-band activating donors lose production; alpha <= 2 and R 0.1 lose robustness | each arm ~0.1 h wall (fit x1.40 of v2 + production + live donor scoring; size x1.7 or x3 barely changes donor time); random controls double the fit part |
| what not to run on the pod | k = 4 donors, 800 steps, all-resample, activating-donor mixes, alpha cap | above | |
| the pass-rule question for the paper | report the production pass and the donor recovery side by side; do not merge them | every R-trained arm trades production passes for donor recovery on this panel | |

These are development choices from 18 hand-picked targets with 16 firing bases each; arm differences of ~0.05 in the
score are within the spread across targets, and one arm (2.5e-4) did not replicate. The pod's population run should
treat both arms as candidates, not as settled.

## Files

| File | What |
|---|---|
| `run.py` | fits + random controls + production / donor / self-donor scoring; resumable per (target, arm) |
| `check_vs_099.py` | 100 vs 099 equivalence on reused circuits |
| `report.py` | tables (E1, E2, E3, cost) and the E1 figure from `out/main` |
| `out/main/rows.jsonl` | one row per (target, arm): sizes, fit / score times, production rows, pooled recoveries, passes |
| `out/main/targets/<key>.<arm>.json` | per-draw raw anchor and pre-activation reads, donor indices, donor-fires flags |
| `out/main/fits.jsonl`, `train.jsonl`, `circuits/` | fit stats, training curves, circuits (`<arm>` and `<arm>_rand`) |
| `out/smoke400/`, `out/refit_check/`, `out/smoke/` | smoke and equivalence runs (smoke = 20 steps, meaningless scores) |
| `e4_engine.py` | E4 engine: 095 rs2 + second donor term, drop A, k donors per step, alpha cap (all default-off) |
| `report_e4.py` | E4 tables, the pre-stated selection rule applied, `out/e4/e4_recovery_vs_pass.png` |
| `e4_pre.sh`, `e4_dev.sh`, `e4_check.sh` | E4 pre-flight, development chain (priority order), check confirmation |
| `out/e4/` | E4 rows / raw / fits / circuits; `out/e4_check/` (engine equivalence), `out/e4_smoke/`, `out/e4_train_random_pool.pt` |
| `logs/` | run logs |
