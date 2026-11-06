# 102: improving production and resample pass rates before the pod (plan, 2026-10-31)

Base algorithm: Z + A + C + R_C at λ 1e-3 (095's "v2CR"), pass floor 0.75. Goal: find changes that raise the
production pass rate and the resample (donor-background) pass rate together, then freeze before the pod.
Status: PLAN, not started. Daniel to pick and order the directions.

## Panels
- **dev**: 095's 72 targets (8 per depth band × site kind), already fitted under v2 / v2CR / v2R, plus 100's 18.
  At 0.75: v2 55/72 production, 0 resample; v2CR 48/72 production, 22/72 resample.
- **check**: a fresh 72 drawn the same way (different seed) from 094's deduplicated records, frozen before any dev
  result is read. Winners are confirmed here, as in 100 E4.
- Selection rule, stated in advance (as in 100 E4): an arm must not lose production pass vs v2CR on dev (paired
  bootstrap CI not entirely below 0), and must raise min(production rate, resample rate). Ties are broken by fit time.

## Direction 1: diagnose first (CPU only, no refits)
Why do circuits fail? Which side the failure falls on decides which fixes can work.
1. **Fit or generalisation?** The train-split vs held-out scores per term (`free0_tr_tk` etc.):
   - fails on train too → optimisation or capacity (directions 3 and 4);
   - passes train but fails held-out → data (direction 2).
2. **Binding term.** Which of Z / A / C / necessity / donor-A / donor-C fails first, by depth band × kind.
3. **Context-set features as predictors of pass/fail** (Daniel's idea):
   - diversity within the 48 activating training contexts: anchor-token entropy, distinct anchor tokens, n-gram
     overlap, source-document count, the residual-stream cosine spread at the anchor;
   - train ↔ held-out similarity: do the 16 held-out contexts resemble the training ones (token overlap, nearest-train
     cosine)?
   - activation profile: the a_pos spread, the strongest/weakest ratio, near-threshold rank, firing density;
   - the contrast set: the close-selector fallback (`ctr_fallback`), contrast activation, contrast ↔ activator
     similarity;
   - target features: layer, kind, decoder norm, SAE error share at the site, the number of active upstream latents;
   - the fit: final loss, the α distribution (share > 2), circuit size.

   Analysis: per-feature pass-rate curves, then a small logistic / GBM with grouped CV (a sanity check that
   prediction is above chance, not a model to ship). Run on dev (72 × 3 arms) and also on v1 15k for the
   features that need no refit (power).
4. **Seed sensitivity.** 081 saw bimodal fit outcomes. Refit the failing dev targets with 2 more seeds (GPU, cheap);
   if failures flip, restarts are a fix.

## Direction 2: what goes in (data)
- **Context selection:** top-48 strongest (now) vs diversity-stratified (cluster the contexts, take the strongest
  per cluster) vs deduplication by anchor n-gram vs a quantile mix. Keep the held-out definition (held-out
  strongest) fixed, so only training changes.
- **Training-set size:** 48 → 96 / 192 contexts (the records may cap this; check 094).
- **Contrast set:** the close selector vs close + random mix; fix the fallback cases.
- **R donor pool:** the contrast-train donors (now) vs a mix with general-corpus contexts; more donors per step
  was tried (k = 4, no gain), so only cheap variants.

## Direction 3: the objective
- **Depth-adaptive λ** (131: deep circuits are capped by one sparsity penalty): λ scaled by band (for example
  1e-3 / 5e-4 / 2.5e-4), not the uniform 2.5e-4 that tripled sizes everywhere.
- **Worst-term objective:** a hinge on whichever of Z/A/C/R is furthest below the band (or a soft-min), not a fixed
  weighted sum.
- **Re-balanced weights:** γ_A, γ_C, γ_R jointly (100 E4 showed R 0.5–1.0 raises robustness; re-test together with
  γ_C).
- **Pre-activation term:** add a small loss on the pre-activation read for near-threshold targets (they fail at
  the Top-K cut).

## Direction 4: optimisation
- **Restarts:** best of 2–3 seeds, chosen on a train-split criterion (never held-out).
- **Initialisation:** gates initialised from an attribution ranking (IG / gradient×activation) instead of uniform.
- **Two-stage:** fit membership, then refit α only, with membership fixed, under all terms (cheap; targets the gain
  overshoot seen in 100 E3).
- **Schedule:** the learning-rate / temperature anneal (800 steps gave nothing; try the lr only).

## Direction 5: repair after fitting
- **Greedy add:** for a failing circuit, add the top-attribution non-members until the train-split criterion is met;
  report the size cost.

## Order (proposal)
1. Direction 1, items 1–3: CPU, ~2–3 h, no GPU; decides which of 2–5 are worth running.
2. Seed restarts on the dev failures: GPU ~1 h.
3. The 3–5 most promising arms from 2–4 on dev: GPU ~1–1.5 h per arm (72 targets × ~1.4× v2 fit time).
4. Confirm the top 1–2 on check, then freeze the algorithm, port it natively (`r-port` work), smoke, and update the
   pod cost.

Integrity: the selection rule above is fixed before any dev arm is read; held-out contexts are never used for
selection inside a fit; the check panel is drawn and frozen before dev results.
