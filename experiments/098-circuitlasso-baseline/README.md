# 098: CircuitLasso as a baseline row in the DAN-159 method comparison

Asked by Daniel: add CircuitLasso (Yin et al. 2026, `yin2026scalable`) as a baseline row in Table 2 and
in the DAN-77 faithfulness-vs-size figure, on the 190 sweep targets. Scope here: implement it, test it, run it on 8
targets, and evaluate.

- The paper's method and evaluations, read from raw text: `PAPER_NOTES.md`.
- Results: `out/table_mini.md`, `out/faithfulness_vs_size.png`, `out/lasso_diagnostics.csv`.

## What runs

For each target, on its training contexts (the same rows every 083 discovery uses), plus the contrast-train rows:

1. **One clean forward pass.**
   - Each (context, position) is one sample.
   - Response y: the target's pre-activation at that position, x · W_enc[i] + b_eff[i]. This is the quantity the 083
     IG / AtP rows attribute.
   - Design X: every upstream site's post-Top-K code at the **same** position. It is sparse, so it is stored as CSR.
2. **The Lasso** (`lasso_core.py`):
   - min_b,β 1/(2n)‖y − b − Xβ‖² + λ‖β‖₁;
   - columns scaled to unit ℓ2 norm, and the coefficients rescaled back afterwards;
   - an unpenalised intercept;
   - FISTA, warm-started along a geometric λ path from λ_max, 20 steps per decade, stopping when the relative
     objective decrease falls below 1e-6. All of this follows C.1 of the paper; the intercept is ours (see below).
3. **The natural λ.** The paper picks λ by held-out prediction accuracy. Here, the λ minimising validation MSE on a
   context-level, stratified 3:1 split of the training rows. The evaluation sets are never touched.
4. **The member sets:**
   - **global top-n at each WCM size:** the top n by |coefficient| at the largest path λ whose support reaches n;
   - **per-site top-c** at the reference circuit's per-site counts, by the same rule within each site;
   - **the support at the natural λ.**
5. **Scoring.** `driver.Runner.score` on held-out strongest, the scorer every 083 row uses. Rows are written into
   OUT083 in the 083 format:
   - `attr/rows.lasso.*.jsonl`: `method=lasso`, `ranking=coef|std|cv`;
   - `fitted/rows.lasso.*.jsonl`: `method=fitted_lasso`, `variant=a1|fit`. The coefficients are fitted with
     `attr_fitted.py`'s DAN-43 fit: support-restricted, λ 0, gates open, the production engine through `Runner.fit`,
     under `WCM_VERSION`.

   `table2.py` picks these rows up. Its change is minimal and tested: label entries plus a per-method natural-size
   label.

| row | 083 analogue |
|---|---|
| Lasso top-n, \|coef\|, α = 1 (`lasso/coef/W_*`) | IG / AtP top-n at each WCM size (the DAN-77 curve points) |
| Lasso per-site counts, α = 1 and fitted (`fitted_lasso/coef/{a1,fit}`) | DAN-43 and activation-ranked rows |
| Lasso natural (`lasso/cv/natural`) | IG / AtP natural size (position union) |
| `std` ranking (\|standardised coefficient\|) | sensitivity only |

## Adaptation choices, and why

| Choice | Paper | Here | Why |
|---|---|---|---|
| regressors | all D features of one earlier location, consecutive layers only | all latents of **every** upstream site jointly | our circuits span every upstream site; one joint Lasso is Eq. 3's block-triangular form restricted to one output column |
| samples | token-**averaged** codes, one observation per prompt | **same position**: every (context, position) of 48 activating-train + 48 contrast-train contexts × 64 positions = 6,144 samples | 96 observations would cap the support at ≤ 96 latents, below the 10²–10³ matched sizes. Our target is a per-position quantity, and the upstream latents at t carry, through the residual stream, what attention brought from earlier positions |
| response | the later feature's code | the target's pre-activation | the same estimand as IG / AtP. A post-Top-K target is 0 at most positions |
| standardisation | columns to unit ℓ2 norm, coefficients rescaled | same | paper (C.1) |
| intercept | none written | unpenalised (exact, by implicit centring; the design stays sparse) | the pre-activation's mean is far from 0 (e.g. −0.20). Without an intercept, always-on latents absorb the offset |
| solver | FISTA, backtracking, warm-started path, relative objective decrease < 1e-6 | FISTA, fixed step 1/L (L by power iteration), function-value restart, warm-started path, same tolerance | paper. The fixed step replaces backtracking; the optimum is the same (tested against sklearn's Lasso to 2e-5) |
| λ | test-accuracy sweep, for the output regression only | natural size: the λ with minimum validation MSE (3:1 context split of the training rows) | the paper's rule adapted to a regression target |
| ranking | \|Â\| (original units, after rescaling) at one λ | **\|coef\| in original units** at the largest λ whose support reaches the needed size | the paper's ranking. The size-matched λ is needed because one λ cannot give every matched size. \|standardised coef\| is kept as a sensitivity row |
| fit data | all prompts of the task | the target's training contexts only (activating + contrast train) | the same rows every 083 method uses. Held-out rows are never seen |

**Data size** (measured on the 8 targets):
- shallow (3.attn.9495, 9 upstream sites): 6,144 samples × 257,945 latents ever active, 7.1M non-zeros;
- deep (10.mlp.2064, 31 sites): 973k latents, 24M non-zeros.

The support reaches several thousand at small λ, so the 10²–10³ matched sizes are reachable; none fell short here.

## Tests

`tests/analysis/test_098_circuitlasso.py` (CPU, 11 tests):
- recovery of a planted sparse dependency (support, coefficients and intercept);
- FISTA equal to sklearn's `Lasso` at three λ;
- fold designs share the full column set;
- the captured top-n is the top-n by |coef| at the first λ whose support reaches n;
- per-site captures hold exactly the requested counts, and unreachable ones are flagged short;
- `table2.py` reads the lasso attr and fitted rows into Table 2 and the DAN-77 points.

The 083 tests still pass.

## Commands

```bash
# 8-target local check (093 v2 circuits as the WCM reference, as-run records); every stage, resumable
wsl: bash experiments/098-circuitlasso-baseline/run_local.sh
# tests (CPU)
PYTHONPATH=src python -m pytest -q tests/analysis/test_098_circuitlasso.py tests/analysis/test_083_v1_comparison.py
```

## Results (8 targets, this RTX 5070 Ti)

**Setup**
- **Targets** (`targets8.txt`, chosen by hand from exp 093's 72): 3 in L0-4, 2 in L5-7, 3 in L8-11, mixing attention,
  MLP and residual sites. Two of them (7.attn.20607, 7.resid.14946) are in the 190-target sweep, the only 093 targets
  that are.
- **References:** 093's v2 circuits (`make_ref093.py` → `out/ref093`, used as REF_RUN), re-scored here.
- **Records:** as-run (062 `out_full/out/ctx`), the same as 093.
- **Protocol:** `WCM_VERSION=v2`.
- **Rows:** in `out/cmp/` (083 layout, so `table2.py` reads them: `out/cmp/table2.md`, `dan77-faithfulness-vs-size.png`).
- **Outputs:** `out/table_mini.md` (summary, per-target cells and diagnostics) and `out/faithfulness_vs_size.png`.

| row (n = WCM size, median 554) | Z | A | C | nec | pass post-Top-K | pass pre-act |
|---|---|---|---|---|---|---|
| WCM v2 (093 circuit), fitted α | 0.96 | 0.95 | 0.89 | 1.00 | **6/8** | 7/8 |
| Lasso top-n, \|coef\| (paper's ranking), α = 1 | 0.00 | 0.00 | 0.00 | 0.47 | 0/8 | 0/8 |
| Lasso per-site counts, \|coef\|, α = 1 / fitted α | 0.00 / 0.00 | 0.00 / 0.22 | 0.00 / 0.00 | 0.46 | 0/8 / 0/8 | 0/8 / 0/8 |
| Lasso per-site counts, \|std coef\|, fitted α | 0.00 | 0.46 | 0.00 | 0.48 | 0/8 | 0/8 |
| Lasso natural (support at CV λ, median 2,246 = 5.0× WCM), α = 1 | 0.00 | 0.00 | 0.00 | 0.51 | 0/8 | 0/8 |
| AtP top-n (max / SFC), α = 1 | 0.00 | 0.03 / 0.00 | 0.04 / 0.00 | 1.00 | 0/8 | 0/8 |
| IG top-n (max / SFC), α = 1 | 0.00 | 0.03 / 0.11 | 0.04 / 0.11 | 1.00 | 0/8 | 0/8 |
| AtP / IG natural (position union, median ~578k = 920× WCM) | 0.99 | 0.99 | 0.99 | 1.00 | 8/8 | 2/8 |
| DAN-43: IG (max) per-site, α = 1 / fitted α | 0.00 / 0.22 | 0.06 / 0.93 | 0.03 / 0.89 | 1.00 | 0/8 / 1/8 | 0/8 / 4/8 |
| activation-ranked (mean) per-site, α = 1 / fitted α | 0.00 / 0.81 | 0.05 / 0.90 | 0.05 / 0.81 | 1.00 | 0/8 / 2/8 | 0/8 / 6/8 |

**Reading.** CircuitLasso is the weakest row in the table.
- **Every matched-size ranking at α = 1 fails.** Lasso, AtP, IG, DAN-43 at α = 1, and activation-ranked all score Z
  ≈ 0. Ranked members without coefficients do not carry these targets at the WCM size.
- **Lasso fails where the others do not.**
  - Its members are not necessary: necessity is about 0.47, whereas every gradient or activation ranking reaches
    about 1.00. Removing Lasso's members leaves half of the target.
  - Fitting coefficients does not rescue it: fitted A is 0.22 (\|std coef\| 0.46), against 0.93 for DAN-43 and 0.90
    for activation-ranked members on the same per-site counts.
  - Its natural support (median 2,246 latents) fails too.
- **The regression fits well in sample.** At the CV λ, median R² is 0.94 over all samples and 0.89 at the activating
  anchors. Yet its top-n shares almost nothing with WCM (median Jaccard 0.024) or with AtP (0.028), while AtP's
  top-n shares 0.25 with WCM.
- **Why, plausibly.** Lasso puts 2–4× more of its members than WCM on the last three upstream sites (e.g. 47% vs 15%
  on 3.attn.9495; 16% vs 4% on 8.resid.4054). It picks the target's proximal *correlates*, latents that co-vary with
  its input on the observational data, not the latents that drive it under ablation. This matches the authors'
  caveat that the coefficients "rank dependencies", not causal effects.
- **Caveat.** 8 targets on as-run records. The direction is unambiguous (0/8 for Lasso on every arm, while the fitted
  DAN-43 and activation rows reach 0.90–0.93 A), but the sizes of the differences are not estimates.

### Ge et al. hierarchical attribution on the same 8 targets (asked by Daniel)

**Run:**
- Command: `REF_RUN=out/ref093 STAGES=ge,table bash run_local.sh` (the `ge` stage was added to this script).
- Method: 083's `ge_hier.py`, unchanged; τ is bisected in log space (7 steps, ±10%) to the W_0.001 size.
- Scoring: every point with n ≤ 20,000 is scored by the same scorer as the rows above.
- ~29 min, 0 error rows.

**Result:** the Ge row is now in `out/cmp/table2.md`. The table is regenerated at the current floor (0.75), so the WCM reference reads 7/8 there, not the 6/8 above (which was at 0.8).

| row (n ≈ WCM size) | n (med) | Z | A | C | φ_ind | nec | pass |
|---|---:|---|---|---|---|---|---|
| WCM v2 (093 circuit), fitted α | 554 | 0.96 | 0.95 | 0.89 | 0.61 | 1.00 | **7/8** |
| Ge hierarchical, τ bisected to the WCM size | 583 (×1.03) | 0.00 | 0.00 | 0.02 | 0.79 | 1.00 | 0/8 |

**Reading:**
- **At matched size it fails like every other rank-and-cut row.** Z is 0 on all 8 targets; A ≤ 0.38 and C ≤ 0.31 on every target.
- **Its members are necessary:** necessity is 1.00, against about 0.47 for Lasso.
- **It does better than the IG / AtP top-n on φ_ind:** 0.79 against 0.59–0.71. This matches 034's finding (its control-side drive is the best of the truncations, but its faithfulness fails).
- **Faithfulness needs a much larger circuit.** Over all bisection points, Z/A/C come into band only at τ = 1e-3, and only on the two shallowest targets:
  - 3.attn.9495: 13,456 latents, 45× WCM, Z 0.76;
  - 2.mlp.9455: 9,454 latents, 33× WCM, Z 0.75.
- **Caveat:** 8 targets on the as-run records. The 190-target version runs on the pod (stage T7).

**Lasso diagnostics** (medians over the 8 targets):
- 6,144 samples; 0.13M–0.97M active upstream latents; 4–24M non-zeros.
- At the CV λ (λ/λ_max 0.007–0.06), 406–3,538 non-zero coefficients (median 2,246). The paper's literal λ = 1e-5
  does not transfer: under its unnormalised objective it is ~10⁻⁶ λ_max here, i.e. essentially unpenalised.
- Every W_0.001 size was reached by the path; no per-site count fell short.
- \|coef\| and \|std coef\| top-n overlap at Jaccard 0.84–0.99.
- The pre-activation read reproduces the post-Top-K value where the target is active (max error 0.004).
- 10.mlp.2064 is the one poor fit (anchor R² −0.23).

**Time per target (Lasso stage):**
- collection + CV path + full path: 10–15 s shallow, 34–48 s deep;
- each fit: 23–37 s shallow, 74–98 s deep;
- each score: 2–5 s.

The 8 targets with both rankings took 24 min, i.e. about 3 min per target (model load excluded).

## Cost for the 190 sweep targets (local, after the pod)

The pod's T7 stage (if `TL_DAN159` is left on) already runs every 083 row on the 190 under v2. Locally only the Lasso
stage is then needed, reading the pod's references and records:

```bash
wsl: REF_RUN=<pod TL_V2_OUT> CTX_DIR=<the 094 v2 records> OUT083=<the pod's dan159 dir (rows land beside the 083 rows)> \
     TARGETS=experiments/094-tl-v2-records/targets_tl8k_sweep190.txt LASSO_RANKINGS=coef FIT_RANKINGS=coef \
     STAGES=lasso,table bash experiments/098-circuitlasso-baseline/run_local.sh
```

| arm set | per target | 190 targets |
|---|---|---|
| Table 2 only (`SIZES=W_0.001`, `coef` only: top-n + natural + per-site a1 + fit) | ~1.5 min shallow, ~2.5 min deep | **~7 h** |
| + the DAN-77 curve (`SIZES=W_all`, 8 WCM prices, `coef` only) | +7 scores (~5–10 s each) and a longer path for the large sizes (+1–2 min) | **~12–14 h** |
| as run here (both rankings, both per-site arms) | ~3 min at W_0.001 only, more with W_all | ~16–20 h |

Notes on the cost:
- **The DAN-77 curve's largest sizes cannot be reached** (the weak-λ arms, up to 37k latents). With 6,144 samples,
  the Lasso support is capped near n (and at `SUPPORT_CAP` 8,192). Those points are scored at the whole support and
  marked `short` in the rows.
- **If the 083 rows are not run on the pod**, add 083's own ~51 local-h (`083/README.md`), or run
  `STAGES=ref,attr,fitted,lasso,table` with the 190 list.
