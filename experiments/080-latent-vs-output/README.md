# 080 — Latent-targeted vs output-targeted circuits

**Question.** Are circuits found for an internal SAE latent a different object from circuits found for
the model's output, and when do they coincide? One method (weighted circuit masking, WCM), two endpoints: only the
thing the circuit must reproduce changes.

This **supersedes the pre-v1, 3-target output-targeted paragraph** in the paper (appendix "Output-Targeted and
Latent-Targeted Circuits", which waits on Linear DAN-48), and **absorbs DAN-137** (prediction circuits: 064 found that
circuit targets rarely drive the next token; this asks what a circuit *for* the next token looks like, and how it
relates to the latent circuit).

**Status.** Code, tests and smoke are done. **Not yet run:** the next-token-effect measurement (stage 0), calibration
and the full run.
- The causal target lists cannot be drawn until stage 0 has run.
- Until then, `targets.txt` / `targets_calib.txt` / `targets_meta.csv` hold the withinkind design (identical to
  `targets_withinkind.*`), so the repo stays runnable.

## Design

### The tier axis: how much a target affects the next token

**Every target is an internal latent.** The tiers do not separate "internal" latents from some other kind. The axis
is how much a latent affects the model's next-token prediction, measured causally in the contexts where it fires.

Three designs are kept. Only the first is the default.

| `SAMPLING` | tier names | tier measure | status |
|---|---|---|---|
| **causal** (default) | **high / low next-token effect** | effect_abs, measured in context by `next_token_effect.py` | default since the second review; needs stage 0 |
| withinkind | direct-path (064) high / mid / low | 064's context-free `boost_peak` | first review; reproducible |
| tierband | direct-path (064) high / mid / low | the same | the original design; reproducible |

**Why the measure changed** (Daniel). 064's `boost_peak` (l_max × peak activation, read through the final
norm at a typical scale) has two problems:
- It is a **direct-path, context-free logit-lens proxy**. It ignores what later layers do with a write, and 064
  measured it as a 2–4× underestimate in L1–8.
- It is **biased toward deep residual latents**.

The causal measure is taken in the target's own contexts, through the whole model downstream. It reads the same
quantity as the output arm's endpoint: log p of the actual next token at the target's anchor.

### Stage 0: the in-context next-token effect (`next_token_effect.py`, `queue_effect.sh`)

Per target, on its **32 strongest training contexts** from the 062 context cache (read only):
- **Contexts:** `rec["D_train"]`, the strongest-train contexts protocol v1 trains on. A thin target without them uses
  the 32 strongest of `strong["train"]` by peak.
- **Anchor t:** the target's peak position (`rec["strong"]["arg"]`), as in production.
- **Next token:** the actual token at t + 1, `rec["strong"]["tgt"][i, t]`. The cache loads 65 tokens (64 of context,
  `tgt` shifted by one), so it exists even for t = 63.
- **Edge cases:** an anchor outside [0, 63] is clamped and counted (`n_clamped`). A context whose next token falls in
  zero padding (next token 0, and only zeros after it: a sequence shorter than 65 tokens) is excluded and counted
  (`n_excluded`).

Two forward passes per context, read at the anchor:
- **Clean:** log p(next token), plus the full next-token distribution.
- **Ablated:** the target latent is removed at its own site at every position, x ← x − a_f(x, t)·d_f (`TargetAblation`).
  - a_f is the post-Top-K activation read exactly as production reads it: `bank.encode` at the target's
    (layer, kind), then `sae.dense.target_latent_activations`, as in `wcm_eval._tap`.
  - The edit is the decoded difference only: x + `bank.decode(delta, add_bias=False)`, with delta = −a_f at latent f
    and 0 elsewhere. This is the single-decode form `learned_mask.LearnedMaskPatcher` uses.
  - So the SAE error is untouched, no other SAE is in the loop, and every later layer recomputes natively. The hook is
    `model.hooks.multi_patch`, the engine's own.
  - A latent that is silent everywhere gives delta = 0. The patch then returns x unchanged, bit for bit (unit-tested).
  - This is the paper's zero ablation.
- **Batching:** the clean and ablated rows of a context run in the same forward (interleaved), and the patcher edits
  only the ablated rows. Thin targets are packed together, up to `BATCH` = 64 rows per forward.
- **Why not "run to the site once, then the rest twice":** `Inference.forward` has no resume-from-layer entry point,
  so doing that would mean bypassing the engine's forward.

Recorded per target (`effect/effect.jsonl`, then `effect/effects.parquet` with 064's `boost_peak` joined):
- **effect_abs** = mean over contexts of |Δ|, where Δ = log p_clean(next) − log p_ablated(next), in nats. **This is
  the sampling measure.**
- **effect_signed** = the mean of Δ. Positive means the latent helps predict the actual next token: removing it lowers
  log p.
- **kl** = the mean KL(clean ‖ ablated) over the full next-token distribution at the anchor.
- n_contexts, n_excluded, n_clamped, mean_act_at_anchor (the clean post-Top-K activation at the anchor),
  lp_clean_mean, frac_act_zero.
- Per-context arrays (Δ, kl, act, lp_clean, next token, anchor) go to `effect/effect_ctx.jsonl.gz`.

**Why effect_abs and not effect_signed.** A latent that *suppresses* the actual next token is as output-relevant as
one that promotes it. A signed mean would let promoting and suppressing contexts cancel, and would rank a strong
suppressor as "low effect". The output arm's endpoint is sensitive to both directions, so the axis is too.

**Pool measured:**
- `POOL=subsample` (default): 300 targets per kind × band group, drawn uniformly with seed 20261018 from
  make_targets' population filters, plus the 2 smoke targets. That is 2,702 targets.
- `POOL=all`: all 14,994. Or `TARGETS=<file>`.
- The run is resumable (append-only jsonl, finished targets skipped, error rows written).
- `BENCH=N` times N targets after a warm-up one and prints sequences/s plus the projected hours for the full
  population and the subsample.

### Targets under SAMPLING=causal (`make_targets.py`; refuses without `effect/effects.parquet`)

- **Population:** every 062 full-run target with a non-vacuous held-out-strongest eval row, a production circuit and
  a cached context record (14,994 of 15,004), restricted to those measured in stage 0 with ≥ 8 usable contexts.
- **Tiers within each of the 9 kind × band groups' measured pool:** rank by effect_abs (ties broken by key). The top
  ⌊n/3⌋ are **high next-token effect** and the bottom ⌊n/3⌋ **low next-token effect**; the middle third is dropped.
  - Tiers are defined within a group, never across groups. So "low" in resid L8-11 and "low" in attn L0-4 are each low
    *for their group*, and every group has both tiers.
  - The report prints the absolute scale per group, in nats.
- **Main list:** one high-vs-low pair per group, 9 per tier: 9 × 2 × 9 = **162 targets**. Fixed seed; cells
  interleaved.
  - The two tiers of a pair are **layer-matched**. Each layer's quota is set by the tier with fewer targets at that
    layer, quotas are dealt round robin over the band's layers in a seeded order, and both tiers are drawn per layer.
  - If a group cannot fill 9 per tier, or cannot layer-match them, the sampler takes what it can, tops up unmatched
    from the rest of the band, and **flags** the group (`THIN`, unmatched counts). This is printed and must be stated
    with the results.
  - With 300 measured per group, each tier has 100 candidates, so thinness is not expected.
- **Calibration list:** one target per group-tier cell, so **18**, disjoint from the main list, layer-matched within
  each pair. A smoke target is forced into its cell when it lands in a causal tier.
- **Smoke list:** unchanged (`1.mlp.35042`, `11.resid.15522`), excluded from the main list.
- **Outputs:** `targets.txt`, `targets_calib.txt`, `targets_meta.csv` (tier, band, kind, pair, cell, effect_abs,
  effect_signed, kl, the legacy `dp_tier` and `boost_peak`, LAT size and LAT pass).

### Legacy designs (reproducible; direct-path tiers)

The tiers use 064's `boost_peak` with the same cut-offs as before, renamed:

| tier | rule | was called |
|---|---|---|
| direct-path (064) high (`dp_high`) | boost_peak ≥ 1.0 nat | writer |
| direct-path (064) mid (`dp_mid`) | in between | weak |
| direct-path (064) low (`dp_low`) | < 0.1 nat | internal |

The renaming changed no sampling: both legacy lists regenerate byte-identical to the earlier ones (checked). Old rows
that carry the old names are mapped by `common.norm_tier`.

- **tierband** (`targets_tierband.txt`, 150 targets: 50 per direct-path tier over the bands).
  - The first review found its tier confounded with site kind: `dp_low` was 48 of 54 attention and
    `dp_high` 51 of 54 residual.
  - The population makes that unavoidable under these tiers:

    | kind | dp_high (L0-4 / L5-7 / L8-11) | dp_mid | dp_low |
    |---|---|---|---|
    | attn | 0 / 1 / 2 | 205 / 413 / 814 | 1,074 / 744 / 841 |
    | mlp | 2 / 4 / 243 | 2,077 / 1,329 / 1,576 | 195 / 30 / 1 |
    | resid | 56 / 593 / 1,489 | 2,175 / 772 / 330 | 38 / 0 / 0 |

- **withinkind** (`targets_withinkind.txt`, 156 targets): six within-kind, within-band pairs of adjacent direct-path
  tiers, 13 per cell, layer-matched exactly (13 of 13 per tier, mean-layer difference 0.00). Calibration is 12 targets.

  | pair (kind, band) | higher / lower tier | LAT size, median | LAT pass | boost_peak, median |
  |---|---|---|---|---|
  | resid L5-7 | dp_high / dp_mid | 503 / 530 | 62% / 62% | 1.38 / 0.70 |
  | resid L8-11 | dp_high / dp_mid | 831 / 670 | 38% / 23% | 2.45 / 0.75 |
  | mlp L8-11 | dp_high / dp_mid | 797 / 705 | 46% / 15% | 1.15 / 0.59 |
  | mlp L0-4 | dp_mid / dp_low | 326 / 304 | 69% / 46% | 0.19 / 0.08 |
  | attn L0-4 | dp_mid / dp_low | 289 / 288 | 69% / 31% | 0.12 / 0.03 |
  | attn L8-11 | dp_mid / dp_low | 649 / 554 | 8% / 0% | 0.17 / 0.05 |

- **Files:** `run.py` and `calibrate.py` read metadata from every `targets_meta*.csv`, so any design's list runs.
  `report.py` takes the tiers from the design's own meta file (`META`, default `targets_meta.csv`).

### Arms

| arm | endpoint (what the fit reproduces) | coefficients | maskable sites | fitted here |
|---|---|---|---|---|
| **LAT** | the target latent's pre-activation (protocol v1) | fitted α | upstream of the target's site | no: the 062 production circuit, loaded from `out_full/out/main/circuits` |
| **OUT_W** | log p(x_{t+1} \| x_≤t) at each context's target anchor t, where x_{t+1} is the actual next token | fitted α | upstream of the target's site | yes |
| **OUT_U** | the same | α = 1 (`free_amplitude=False`) | upstream | yes |
| OUT_W_FULL (optional, off) | the same | fitted α | every site, the target's own included (025's scope) | only with `ARMS=...,OUT_W_FULL` and a config entry |
| OUT_U_FULL (optional, off) | the same | α = 1 | every site | the same; added later, see below |

Everything else is identical to protocol v1 (`paper/protocol.md`):
- the same contexts and split: training on the target's 32 strongest + 16 mid-band training contexts, scoring on its
  16 held-out strongest;
- the same three ablations with the same weights: Z, A mean (γ_A = 0.25) and C mean over the close contrast set
  (γ_C = 0.25), with ablation values from the training contexts;
- free coefficients, the same optimiser (AdamW), 400 steps and batch 4.

What differs for the OUT arms:
- **The endpoint.** The engine's `logit` objective is the squared error of log p(next token) against the full model's
  log-prob. It reproduces; it does not maximise.
- **No rank-keep.** Rank-keep is about the target latent's rank, so it is off.
- **The logit normaliser** (`logit_norm="gap"`). Each data term is divided by the mean over the training contexts of
  (log p_full − log p_empty,Z)², floored at 0.01 nats². That makes each term ≈ (1 − output faithfulness)², the
  analogue of the latent objective's mean(target²) scale. λ then prices a latent against the same relative error for
  every target, instead of against raw nats², whose scale varies by target.
- **λ and lr come from calibration.** Weight decay is rescaled so lr × wd stays at production's 0.05 × 0.05
  (steps × lr × wd = 1.0, the config's calibrated decay product).

### Scope, and why it is the latent circuit's

Only the SAE sites causally upstream of the target's site are maskable (`logit_scope="upstream"`). The target's own
site and every later site run **live**: they are recomputed from the edited stream and never ablated.

So the two circuits are drawn from the same candidate set and differ only in what they must reproduce. Experiment 025
showed that the **ablation scope dominates** the comparison. There, an upstream-only circuit scored 0.06–0.10 output
faithfulness at any size once everything above the target was also ablated. A whole-model OUT circuit would therefore
differ from LAT in scope as well as endpoint, which is exactly the confound this experiment removes. OUT_W_FULL keeps
025's scope available as an optional extra arm, scored on both output scopes (`out_*` native, `upstream_out_*`).

The scope also fixes what "the output" means here. The output metric is the next-token log-prob as far as it depends on
the upstream sites. The upper stack can recompute around any edit.

**The live sites confound the coefficient comparison (found in the calibration).** The number of live sites
depends on the target's layer:
- a `0.resid` target masks 2 of the 36 sites and leaves 34 live;
- an `11.resid` target masks 35 and leaves 1 live.

The coefficients' advantage, measured as the size OUT_U needs over OUT_W to reach output Z 0.8 (per target, the best lr
per arm), grows as fewer sites stay live:

| target layers | OUT_U / OUT_W size (median) |
|---|---|
| L0–4 | 2.5× |
| L5–7 | 5.7× |
| L8–11 | 6.7× |

These ratios are far below the latent endpoint's 27–43×. This fits the live layers rescaling whatever reaches them,
which leaves the coefficients less to do. It is a hypothesis, not yet a test.

So the whole-model pair OUT_W_FULL / OUT_U_FULL is calibrated too (`queue_calib_full.sh` → `calib_full/`). With every
site maskable, its W/U ratio measures the coefficients on the output endpoint with no live layers to help.

**Whole-model calibration (198 fits, 0 errors).**
- **The live-site hypothesis holds.** With every site maskable, the OUT_U_FULL / OUT_W_FULL size ratio at output Z 0.8
  is flat with depth: 6.7× at L0–4, 8.4× at L5–7, 6.9× at L8–11, 7.4× overall (n 18; W wins on every target).
- **But whole-model Z is not comparable with upstream Z.** Zeroing every site wrecks the model:
  - empty-Z log p is −33.9 nats, against −14.9 upstream (full −2.14);
  - so "Z 0.86" (OUT_W_FULL, λ 1e-3, 182 latents) is a raw log p of −6.80, 4.7 nats below the full model;
  - the upstream circuit at "Z 0.81" (OUT_W, λ 1e-3) sits at −4.58.
- **Both mean ablations fail at every swept size.** A and C are 0.04–0.17 in both arms, even OUT_U_FULL at 4,966
  latents. Z, A and C are all in band for 0–6% of fits.
- **A suspected cause, not yet tested.** `logit_norm="gap"` divides every data term by one shared scale, the zero-ablation
  gap². That gap is ~5× larger in nats here than the mean-ablation gaps (31.7 vs ~7), so a nat of A/C error costs the
  same as a nat of Z error even though it is a far larger share of its own gap; with γ = 0.25 it costs a quarter.
  Candidate fix: a per-term normaliser (each ablation's own closed-mask gap).

### Matched size

LAT is one circuit per target. Each OUT arm is fitted on a small λ grid (3–4 values, chosen by calibration so that
median sizes bracket the LAT sizes).

- **Scores.** Every OUT metric (and pass indicator) is interpolated per target, linearly in log10 size, to that
  target's LAT size. This uses 071's `logic.interp_log_n`, reused unchanged. A LAT size outside an arm's size range is
  clamped to the nearest end and flagged; the report counts these cases.
- **Set overlaps** cannot be interpolated. They use the OUT_W λ whose size is nearest the LAT size in log10.

### Calibration (separate stage: `calibrate.py`, `queue_calib.sh`)

On the calibration targets (18 under causal), the stage sweeps, for both OUT_W and OUT_U:
- λ: OUT_W {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, OUT_U {1e-5, 3e-5, 1e-4, 3e-4, 1e-3};
- lr: {0.02, 0.05, 0.1};
- at 400 steps, with the same fit path as the main run.

`REPORT=1` reports, per arm × lr:
- the **failure rate**: fits that errored or were rejected, or whose held-out output Z is missing, negative or ~0
  (< 0.05);
- **monotonicity** in λ, of size and of output Z (tolerance 0.02);
- **size vs λ** against the calibration targets' LAT sizes.

It writes `calib/recommendation.json`:
- **lr:** the one with the fewest failures; ties go to more faithfulness-monotone, then nearest 0.05.
- **λ grid:** at that lr, the 4 consecutive λ whose median sizes best cover the LAT p10–p90.
- **Warnings** when nothing brackets the LAT sizes or failures exceed 25%.

**Nothing is auto-picked.** `run.py` reads `config_main.json` and refuses to start without it. The file is written only
by an explicit `ADOPT=1 calibrate.py`, which refuses a recommendation that carries warnings unless `FORCE=1`, or it is
written by hand.

### Scoring (held-out strongest, activation read, `src/eval/wcm_eval.py`)

Every circuit (LAT, OUT_W, OUT_U) is scored on **both** endpoints, on the same contexts, split, ablation values and
fills.

**Latent:** the production scorer, `score_circuit`.
- Z, A (sparsity-preserving) and C (sparsity-preserving) faithfulness, plus necessity.
- The DAN-8 pass rule: Z, A and C in [0.8, 1.5] and necessity ≥ 0.9.
- LAT rows are re-scored here and must reproduce 062's own eval rows (the smoke checks this).

**Output:** `score_output`, new and additive.
- m = mean log p(actual next token) at the held-out anchors.
- φ_out,π = (m(C) − m(∅)) / (m(full) − m(∅)) for π ∈ {Z, A_sp, C_sp}.
- Upstream latents outside the circuit are set to the fill. Members sit at α × their live value. SAE errors are
  preserved. Sites at or after the target's site run live.
- **Output necessity:** (m(full) − m(members zeroed, everything else live)) / (m(full) − m(∅, Z)).
- Ratios are unclipped, and are None unless m(∅) < m(full) (the latent scorer's guard). Also recorded: Z at α = 1.
- **Vacuous** when |m(full) − m(∅, Z)| < 0.1 nats. The denominator depends only on the target's contexts and scope,
  so such targets are dropped from every output comparison.
- `passes_output` applies the DAN-8 rule to the output metrics.

**Overlap** between LAT and each OUT circuit: Jaccard, |LAT ∩ OUT| / |LAT| and |LAT ∩ OUT| / |OUT|.
- The reference noise floor is the 062 stable-core refit Jaccard (fit 1 vs a reordered refit under the same protocol)
  for the same targets where it exists, otherwise its overall median (~0.54).

**Measurement checks** (`report.py`):
- effect_abs by group in nats (the measured pool's p10 / tertiles / median / p90, and the sampled cells' medians);
- the Spearman correlation between effect_abs and 064's `boost_peak` over the measured pool, overall and per group:
  how far the proxy was off.

### Engine and eval changes (all additive, default off)

- **`src/circuit/instrument/learned_mask.py`:**
  - the `logit` objective may now use the dual / triple floors (before, the combination raised);
  - all three training patchers follow the objective's seed tap;
  - closed-mask diagnostics are measured on the output;
  - there is a new `logit_norm` argument ("none" by default, which multiplies the 025 single-floor term by exactly
    1.0).
- **`src/config.py`:** `learned_mask.mask_objective` ("pos"), `logit_scope` ("upstream") and `logit_norm` ("none"),
  with validators.
- **`src/circuit/discovery/ablation_gradient.py`:**
  - the mask hop passes `objective="logit"`, the probe's next-token targets and the scope when
    `mask_objective="logit"`;
  - the default call is unchanged;
  - `_run_attribution_hop` holds the probe's target tokens for the duration of one hop.
- **`src/eval/wcm_eval.py`:** `score_output`, `passes_output`, `read_logprob`, `output_vacuous`, `next_tokens_at`,
  `scope_sites`, `EditTrackingCircuitOnly`. No existing function changed.
- **Tests:**
  - `tests/circuit/instrument/test_learned_mask.py::TestLogitFloorsAndNorm` (9);
  - `tests/circuit/discovery/test_learned_mask_mode.py::TestAblMaskOutputEndpoint` (6);
  - `tests/eval/test_wcm_eval.py::TestOutputScorer` (10);
  - `experiments/080-latent-vs-output/test_common.py` (41): sampling for all three designs (within-group tertiles,
    causal pairing and layer matching, thin flagging, calibration one per cell, the refusal without the effects file),
    the committed lists, the vocabulary, the P1 statistics, run config, matched size, calibration rule;
  - `experiments/080-latent-vs-output/test_effect.py` (5): the ablation on a tiny synthetic model (a silent latent
    changes nothing, Δ = 0 exactly; the edit is −a_f·d_f only, with no decoder bias and clean rows untouched; the
    interleaved batch equals two separate passes; packing targets changes nothing; context selection and edge cases).

## Registered predictions (written before any run)

Criteria are fixed here, and `report.py` prints each verdict.

**P1. Overlap and cross-faithfulness rise with the target's next-token effect, within site kind and depth band.**

*Re-registered for the third time, still before any run.*
- **Reason:** Daniel. The direct-path proxy (`boost_peak`) is context-free and biased toward deep residual latents.
  The causal in-context measure (effect_abs) reads the same quantity as the output arm's endpoint.
- **Earlier versions:**
  - (1) a Spearman correlation across all targets plus a writer − internal difference. Withdrawn: the tier-kind
    confound.
  - (2) within-kind pairs of adjacent direct-path tiers.
- The pooled Spearman across all targets is still printed, as descriptive only.

The test:
- *Quantities* (three): J(LAT, OUT_W) at the nearest-size λ; OUT_W's latent Z at the LAT size; LAT's output Z
  (output-vacuous targets dropped).
- *(a) Primary: within-group tier difference.*
  - Each of the 9 kind × band groups gives d = median(high next-token effect) − median(low next-token effect) on the
    quantity, over its layer-matched pair.
  - The pooled effect is the **simple mean of the 9 group differences**. Every group gets equal weight, with no
    dependence on noisy per-cell variance estimates of ~9-target medians.
  - 95% interval: bootstrap with targets resampled within each of the 18 cells, independently, B = 2000, seed 20261018.
  - The inverse-variance pooled difference is reported as a sensitivity check, not as the test.
- *(b) Secondary: fixed-effects slope.* OLS of the quantity on log(effect_abs) with one intercept per kind × band
  group, so only within-group variation in the next-token effect identifies the slope. 95% interval from a bootstrap
  within groups, B = 2000.
- *Verdict:*
  - **SUPPORTED** if (a)'s pooled interval is above 0 AND (b)'s slope interval is above 0, for all three quantities.
  - **MIXED** if (a) holds for all three but (b) does not.
  - **NOT SUPPORTED** otherwise.
- The 9 per-group differences are reported with their own intervals; no per-group verdict is registered.

**P2. For low next-token effect targets, the endpoints do not transfer.**
- *Restated with the causal tiers.* Low-effect targets now exist in every kind × band group: 81 targets
  across the 9 groups.
- *Claim:* for these targets, OUT circuits score near 0 on the latent metrics, and LAT circuits score near 0 on output
  faithfulness.
- *Supported if,* over the low next-token effect tier, both OUT_W's latent Z (at the LAT size) and LAT's output Z have
  a median ≤ 0.2 and an upper 95% CI ≤ 0.35.
- Under a legacy design, the report applies the same rule to `dp_low`.

**P3. Coefficients shrink output circuits too.**
- *Claim:* at matched pass rate OUT_U needs more latents, and at matched size it passes less.
- *Supported if both hold:*
  - at the LAT size, the output pass rate of OUT_U − OUT_W has a paired 95% CI below 0;
  - the per-target size needed to reach output Z ≥ 0.8, OUT_U / OUT_W (geometric median over targets where both
    reach it), has a 95% CI above 1.

The rest is reported descriptively: per tier (a tier row pools the 9 groups), per cell, by kind, and pass rates on
both rules.

## How to run

All commands run from the repo root. GPU jobs use WSL's `.venv` (cu128 torch).

```bash
# 0a. benchmark the next-token-effect pass (GPU; ~50 targets; prints sequences/s and projected hours)
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && MODE=bench nohup setsid bash experiments/080-latent-vs-output/queue_effect.sh > experiments/080-latent-vs-output/queue_effect_bench.log 2>&1 < /dev/null & disown'
# 0b. measure the pool (GPU, resumable; POOL=subsample, or POOL=all for every target) -> effect/effects.parquet
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && MODE=run POOL=subsample nohup setsid bash experiments/080-latent-vs-output/queue_effect.sh > experiments/080-latent-vs-output/queue_effect.log 2>&1 < /dev/null & disown'
# 0c. draw the causal lists (CPU; refuses without effect/effects.parquet)
PYTHONPATH=src python experiments/080-latent-vs-output/make_targets.py
#     legacy lists: SAMPLING=withinkind (-> *_withinkind) or SAMPLING=tierband (-> *_tierband)

# 1. smoke (2 targets, 12 / 40 steps, end to end incl. both scorers, report, figure and the runtime estimate)
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && bash experiments/080-latent-vs-output/smoke.sh'

# 2. calibration (GPU, resumable), then read calib/report.txt and calib/recommendation.json
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && nohup setsid bash experiments/080-latent-vs-output/queue_calib.sh > experiments/080-latent-vs-output/queue_calib.log 2>&1 < /dev/null & disown'

# 3. adopt the recommendation explicitly (writes config_main.json; FORCE=1 to adopt despite warnings)
OUT_CAL=experiments/080-latent-vs-output/calib ADOPT=1 PYTHONPATH=src python experiments/080-latent-vs-output/calibrate.py

# 4. full run + report + figure (GPU, resumable)
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && nohup setsid bash experiments/080-latent-vs-output/queue_local.sh > experiments/080-latent-vs-output/queue_local.log 2>&1 < /dev/null & disown'

# 5. report / figure / estimate any time (CPU)
OUT=experiments/080-latent-vs-output/out PYTHONPATH=src python experiments/080-latent-vs-output/report.py
OUT=experiments/080-latent-vs-output/out PAPER=1 PYTHONPATH=src python experiments/080-latent-vs-output/figure.py
PYTHONPATH=src python experiments/080-latent-vs-output/estimate.py

# tests (CPU)
PYTHONPATH=src python -m pytest experiments/080-latent-vs-output/test_common.py experiments/080-latent-vs-output/test_effect.py -q
PYTHONPATH=src python -m pytest -q          # the repo suite
```

Useful variables:
- `next_token_effect.py`: `POOL`, `N_POOL`, `TARGETS`, `N_CTX`, `BATCH`, `BENCH`, `LIMIT`, `DRY=1`, `FINALIZE=1`.
- `make_targets.py`: `SAMPLING`, `SUFFIX`, `EFFECTS_FILE`, `N_PER_CAUSAL`.
- `run.py`: `OUT`, `TARGETS`, `CONFIG`, `ARMS`, `SHARD=i/k`, `LIMIT`.
- `calibrate.py`: `OUT_CAL`, `CAL_ARMS`, `LAMS_OUT_W`, `LAMS_OUT_U`, `LRS`, `STEPS`, `LOGIT_NORM`, `DRY=1`.
- `report.py`: `B`, `SMOKE=1`, `META`, `GROUP`, `EFFECTS_FILE`.
- `figure.py`: `METRIC=Z|A|C`, `PAPER=1`.

## Outputs

**`effect/` (stage 0):** `effect.jsonl` (one row per target), `effect_ctx.jsonl.gz` (per-context arrays),
`effects.parquet`, `bench.json`, `run.log` / `bench.log`.

**`out/` (main) and `calib/`:**

| file | content |
|---|---|
| `eval.shard0.jsonl` | One row per (target, arm, λ) (LAT: `lam="prod"`): the latent scorer's fields, the output scorer's `out_*` fields, size, timings, fit peak memory, and overlap with LAT (`jaccard_lat`, `cont_lat_in_arm`, `cont_arm_in_lat`). Error rows are retried on resume. |
| `members.shard0.jsonl.gz` | Members per circuit. |
| `train.shard0.jsonl` | Fit stats and the per-step training curve. |
| `status.shard0.jsonl` | One row per target. |
| `circuits/<arm>/<λ>/<key>.pt` | Circuits (`circuits/LAT/<key>.pt` is a copy of the production circuit). |
| `ctx/` | Copied context records. |
| `run_config.shard0.json` | The config the run used. |
| `report.txt`, `figures/cross_score_Z.{png,pdf}` | Report and figure. |

**Paper figure:** `PAPER=1 figure.py` writes `paper/figures/latent-vs-output.{png,pdf}` at 5.5 in. It is a 2 × 2
panel: rows are the training endpoint and columns the scoring endpoint, with bars for high vs low next-token effect.
It is descriptive: a tier's bar pools the 9 groups, so read tier effects off the P1 within-group contrasts.

## Caveats

- **Tiers are relative to their group.** "High" in attn L0-4 can be a smaller absolute effect than "low" in resid
  L8-11. That is the design (it removes the kind and depth confound), and the report prints the absolute scale.
- **Tier may still track LAT fit quality within a group**, as it did under the direct-path tiers (LAT pass was lower in
  the lower tier in five of six withinkind pairs). A positive P1 contrast can partly reflect how well the latent circuit
  fits. Report LAT pass by cell alongside P1.
- **effect_abs is measured on training contexts, and the circuits are scored on held-out ones.** It is a property of
  the target used only to sample, never a score. It uses the strongest contexts, so it is the effect where the target
  fires hardest.
- **The ablation is single-latent zero ablation at one site,** in place: no mean ablation and no other SAE in the
  loop. Downstream sites run their natural computation; there is no SAE re-encoding downstream.
- **The output endpoint is read where the target fires.** It is the actual next token at the target's anchor on the
  target's activating contexts, not the model's top-1 and not arbitrary positions. These are the only positions where
  "the circuit for this target" and "the circuit for this prediction" are about the same computation.
- **`logit_norm="gap"` is a design choice made here, not calibrated.** Calibration chooses λ and lr under it. Raw nats²
  (025's) remains available as `LOGIT_NORM=none`.
- The 0.1-nat output-vacuity threshold is absolute: a log-prob has no natural zero.

## Smoke results (RTX 5070 Ti, `smoke.log`)

The smoke ran end to end with **0 error rows**: 10 eval rows over 2 targets (LAT plus 2 λ each of OUT_W and OUT_U),
the timing pass, the calibrate.py code check, report, figure and estimate. The fit settings were 12 steps at lr 0.5,
deliberately uncalibrated, so **the faithfulness numbers below mean nothing** about the science. They only show the
pipeline produces them.

**Re-run after the within-kind redesign** (`smoke_v2.log`, `out_smoke_v2*`). It was needed because `run.py` now reads
metadata through `common.load_meta` and writes a `cell` field. It reproduced every circuit size and score exactly, with
0 errors.

**Not re-run after the causal redesign** (the GPU was in use). `run.py` did not change in that round. The report and
figure were re-run on the CPU against `out_smoke_v2`.

**Circuit sizes and scores** (the targets' direct-path tiers: `1.mlp.35042` dp_low, `11.resid.15522` dp_high):

| target | arm | λ | n | latent Z | output Z | output nec | J(LAT) |
|---|---|---|---|---|---|---|---|
| 1.mlp.35042 | LAT | prod | 221 | 0.95 | 0.33 | 1.05 | – |
| | OUT_W | 1e-3 | 505 | 0.00 | −0.01 | 0.33 | 0.09 |
| | OUT_W | 1e-2 | 0 (rejected) | – | – | – | – |
| | OUT_U | 1e-4 / 1e-3 | 2,649 / 51 | 0.00 / 0.65 | 0.05 / 0.02 | | 0.03 / 0.03 |
| 11.resid.15522 | LAT | prod | 941 | 0.88 | 0.71 | 0.38 | – |
| | OUT_W | 1e-3 / 1e-2 | 1,277 / 987 | 0.00 / 0.00 | 0.38 / −0.04 | | 0.06 / 0.09 |
| | OUT_U | 1e-4 / 1e-3 | 26,155 / 8,828 | 0.00 / 0.00 | 0.70 / −0.05 | | 0.02 / 0.03 |

**Checks:**
- **Output denominators** are healthy: 9.6 nats (L1) and 18.3 nats (L11). The scope check `out_all_sites_ablated`
  passed on both.
- **LAT re-scored vs 062's own eval rows.**
  - L1: within 0.012.
  - L11 resid: a_pos identical and Z within 2e-4, but the sparsity-preserving A and C fills differ (C 0.61 here vs
    0.72 in 062).
  - That fill ranks latents by their mean, so bf16 near-ties across 35 sites can flip between GPUs.
  - Never mix 062's rows with this experiment's: all arms are re-scored on the same machine.

**Timing and memory** (`estimate.py`):
- Fit time is ~1.5–4 s of overhead, plus per step ~0.016–0.039 + 0.0045–0.006 × (masked sites) s.
- At 400 steps that is ~10–20 s for an L1 target and ~85 s for an L11 resid target.
- Peak fit memory: 5.7 GB at L1 and 8.8 GB at L11, so the 16 GB card needs no VRAM guard.
- Scoring a circuit on both endpoints takes 1.6 + 0.17 × sites s.

## Runtime estimate (this GPU)

**Stage 0: the next-token-effect pass. UNMEASURED; arithmetic only.** Run `MODE=bench` first; it prints the measured
numbers.
- **Per target:** 64 sequences of 64 tokens (32 clean + 32 ablated) in one forward, with the edit at one site. The
  nearest timing we have is the smoke scorer: ~1.4 ms per sequence for a forward that edits one or a few sites. That
  gives ~0.09 s of compute.
- **Plus I/O:** loading the target's context record from `/mnt/x` (~20–100 ms over WSL's 9p filesystem) and one host
  copy per pack.
- **Assumed:** 0.12–0.3 s per target (~200–500 sequences/s), plus ~3 min engine start-up.
- **Benchmark** (`BENCH=50`): ~0.5 min of measurement, ~3.5 min with start-up.
- **Subsample** (2,702 targets): ~6–14 min, ~10–17 min with start-up.
- **Full population** (14,994 targets): ~0.5–1.3 h.
- Cheap enough that `POOL=all` is affordable if the benchmark agrees. Tertiles over the whole group pool would then
  replace tertiles over a 300-target subsample.

**Calibration and main run** (from the two smokes' fitted timing models, for the causal composition; fit cost grows
with the number of masked upstream sites, so the deep residual targets, ~85 s per 400-step fit, are the slow ones):
- **Composition:** 162 targets, 54 per band, 18 of them deep resid, mean ~18.8 masked sites (withinkind: 156 targets,
  78 in L8-11, 20.7 sites).
- **Calibration:** 18 targets × 2 arms × 15 fits = 540 fits at 400 steps, output scorer only: **~7.3–8.6 h** (12
  targets, ~5.7–6.5 h, under withinkind). To shorten it, sweep fewer learning rates (`LRS=0.05`) or use `LIMIT`.
- **Main run:** 162 targets × (2 arms × 4 λ) = 1,296 fits, plus 1,458 scorings on both endpoints: **~18–21 h**.
- **Why ranges:** the two smokes' timing passes agree on the deep targets (a 40-step L11 fit takes 11.0–11.7 s) but
  not on the shallow per-step cost. Each range spans the two fits, extrapolated 10× from 12- and 40-step runs. The
  causal list's real layer mix is known only after stage 0, so this assumes targets spread evenly over each band's
  layers.
- Rerun `estimate.py` once the causal `targets.txt` and `config_main.json` exist.
