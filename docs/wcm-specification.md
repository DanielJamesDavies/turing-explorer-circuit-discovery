# WCM specification (production version v4)

This document specifies weighted circuit masking (WCM) exactly as the production code runs it. Every value is taken
from the code or configuration that the pod chain executes, and is cited as `path:line` (paths from the repo root).
Line numbers can drift as the code changes; the cited function or block is the reference. **UNVERIFIED** marks
anything not confirmed in code. Where any other document disagrees with the code, the code is current.

Production = `WCM_VERSION=v4`. `chain.sh` exports it for every track (`experiments/082-public-runs/chain.sh:193`) and
passes it to every launch (`chain.sh:461`, `chain.sh:827`). The driver's own default is `v1`
(`experiments/062-h100-protocol-v1/driver.py:107`), so a run started outside the chain without `WCM_VERSION=v4` is
**not** production.

**Versions.** v1 is the original protocol. v2 trains the A and C terms under the scorer's sparsity-preserving fill,
against the scorer's own means. v3 adds the resample term R_C (weight 1.0) and raises rank-keep to 1e-1. v4 (production)
adds the geometric learning-rate schedule (peak ×3.15, warm-up 0.1) and the censored post-Top-K read. `v4nr` is v4
without R, used only for the without-R comparison arm.

Short names used below:

| Short name | Path |
|---|---|
| `driver` | `experiments/062-h100-protocol-v1/driver.py` |
| `LM` | `src/circuit/instrument/learned_mask.py` (the WCM engine) |
| `AG` | `src/circuit/discovery/ablation_gradient.py` |
| `PH` | `experiments/059-context-pool/protocol_harness.py` |
| `PC` | `src/circuit/protocol_contexts.py` |
| `WE` | `src/eval/wcm_eval.py` (the scorer) |
| `CFG` | `src/config.py` |
| `082/`, `094/`, `107/` | `experiments/082-public-runs/`, `experiments/094-tl-v2-records/`, `experiments/107-true-baselines/` |

Score names follow the README: **Z** = `free0`, **A** = `freeM_topk`, **C** = `freeN_topk` (older names: free0 /
freeM / freeN).

---

## 1. Summary

WCM fits a **weighted circuit** for one target SAE latent (a dictionary latent at layer *L*, site kind *k*). The
circuit is a set of latents at the target's causally upstream SAE sites, plus one coefficient α per member.

- Each upstream latent has two parameters: a gate logit θ, with gate m = σ(θ/T) annealed toward binary, and an
  amplitude ψ, with α = softplus(ψ).
- The parameters are trained with AdamW for 400 steps.
- The loss asks the target's **pre-activation** at its anchor positions on 48 training activating contexts to stay at
  its natural value. This must hold under four ablations of every non-member: zero (Z), the contrast-context mean (C),
  the activating-context mean (A), and a real contrast context's code (R_C). C and A use the scorer's
  sparsity-preserving fill.
- The loss adds a rank-keep hinge, a censored (soft Top-K) read, and a flat L1 price λ per kept gate.
- The output is the set {θ > 0} with each member's final α, rounded to 4 d.p. (`LM:2560-2638`, `LM:2566-2580`,
  `src/circuit/discovery/gradient_base.py:440-476`).
- It is scored on held-out contexts by the Z / A / C faithfulness ratios and by necessity.

## 2. Loss (v4, objective `pos`, mask_floor_source `triple`, free amplitude)

### 2.1 Masked forward (every term)

At every upstream site s, the site's code is re-encoded from the current, already-edited stream. The stream is then
updated as x ← x + W_dec[code − dense], so the SAE error is preserved (`LM:511-604`).

- The upstream sites are all lower layers, plus the earlier kinds in the same layer, in forward order (`AG:291-293`;
  `src/model/families.py:27-40, 102, 109, 116-118`).
- The target's own site is not edited. Its pre-activation `w·x + b` is tapped there (`LM:493-506`).
- Gate: m = σ(θ / T) under `binarize="anneal"` (`LM:532-535`; config default `CFG:903`; nothing overrides it).
- Temperature: T = 0.05^(step/(S−1)), from 1 to 0.05 (`LM:2215-2238`; `anneal_t0` 1.0 and `anneal_reach_frac` 1.0 are
  the engine defaults, `LM:1033, 1039`).
- Code by term:
  - **Z** (`patcher_zero`, floors None): code = m·α·dense (`LM:1770-1776`, `LM:557-565`).
  - **C** (`patcher`): floor = the C mean.
  - **A** (`patcher_pos`): floor = the A mean.
  - C and A (v2 and later: `train_fill="sparse"`, `driver:94-95, 102`) take the scorer's sparsity-preserving fill.
    - Members (θ > 0) are kept at m·α·live. Membership is `m.detach() > 0.5`, i.e. θ > 0 (`LM:472`).
    - Budget per (row, position) = K − #members with α·live ≠ 0 (`LM:179-214`).
    - The fill is the top-`budget` non-members ranked by mean, at their mean. Everything else is 0.
    - The keep mask H is applied straight-through: forward H·code, backward identity (`LM:217-226, 460-479`;
      `sparse_grad="ste"`, `driver:94`).
  - **R_C** (`patcher_donor`): floor = the natural sparse code of a donor context at the same position. One donor row
    per training row is redrawn at every forward from a CPU generator seeded 20261017 (`LM:239, 295-329, 607-615`;
    `CFG:954`).
    - The donor keep mask: members kept; the donor's positive non-member entries fill K − #active members, largest
      first (`LM:243-259`).
    - The donors are the target's **48 contrast-train rows** (`driver:495-498`; `resample_pool="contrast"`,
      `driver:95`). Their codes are captured once, at batch 16 (`LM:240, 263-292, 1789-1797`).
- K = the dictionary's k (`LM:476`, `bank.k`): TuringLLM 128 (`src/sae/topk_sae.py:112-114`), Pythia 16, GPT-2 32
  (`src/model/families.py:110, 119`).

### 2.2 Where the A and C means come from (`train_means_source="eval_ref"`, v2 and later)

- **C mean:** the dense per-latent mean over all (row, position) slots, zeros included, of the 48 contrast-train rows
  (`AG:414-435`; `src/eval/floors.py:141-145`).
- **A mean:** the same over the **48 strongest-train reference rows** (`PC:456-469`; `driver:486-491`). In arm B this
  is not the fit's own training set (32 strongest + 16 mid-band).
- These are bit-identical to the scorer's means (`WE:442-443`).

### 2.3 Data terms

Per term π, from `LM:1878-1994`:

```
D_π = mean_b ( v_π(b) − t(b) )²                                         (LM:1950)
  v_π(b) = target pre-activation w·x+b at training anchor b under ablation π   (LM:1897-1898)
  t(b)   = the same in the unedited model ("reproduce, don't maximise")        (LM:1688-1690)
```

**Censored read** (v4; `censor_temp` 0.05, `driver:102`). Applied to Z, C and A only (the patchers with
`capture_site`, `LM:2126-2135, 1923`); not to R or necessity.

```
v ← v · σ((v − τ_K)/T_b),   t ← t · σ((t − τ_K^clean)/T_b)              (LM:1937-1949)
  τ_K       = the K-th largest OTHER pre-activation at the target's site under that ablation   (LM:1926-1933)
  τ_K^clean = the same in the clean model                                                      (LM:2165-2167)
  both clamped at the latent's JumpReLU threshold when it has one
  T_b = 0.05 × |clean target pre-activation at b|, ≥ 1e-4                                      (LM:2168)
```

**Rank-keep** (`rank_mode="keep"`, `PH:41`; weight 1e-1 under v3 / v4, `driver:105-106, 436-438`; δ 0). Applied inside
the Z, C and A terms, after the MSE (`LM:1967-1993`):

```
+ γ_K · mean_b relu(τ^(r_b) − v_π(b))²
  r_b    = the target's clean rank among the other site latents at anchor b, clamped to [1, k]   (LM:2157-2158)
  τ^(r)  = the r-th largest other pre-activation under the ablation (uncensored v_π)
```

**Necessity term** (off in production: `NEC_W` default 0, `driver:150, 468`). Arena A runs it at weight 0.3, target
0.95 (`driver:151`; arena A's tuning file, see §6).

- A fifth forward: members removed (code = dense·(1 − m)), everything else natural (`LM:540-543, 1803-1808`).
- Its term is a one-sided hinge `mean relu(sgn·(v − [e0 + (1−0.95)(t − e0)]))²` (`LM:1899-1907`). e0 = the target
  under the zero empty circuit (`LM:2172`). The hinge is 0 while the pre-activation necessity is ≥ 0.95.

**Off in production:**
- off-target / specificity (`offtarget_weight` 0, `PH:40`);
- rival term (`RIVAL_W` 0, `driver:139, 454`);
- adaptive λ (`LAM_ADAPT_EPS` None, `driver:141, 455`);
- neg-suppress (`CFG:871`);
- margin (`CFG:875`).

### 2.4 Total loss per step

The engine computes it at `LM:2261-2296, 2375-2444`:

```
L = (1/ν)[ D_Z + 0.25·D_C + 0.25·D_A + 1.0·D_R ]  (+ (0.3/ν)·D_nec, arena A only)
    + λ Σ_{s,i} σ(θ_{s,i})                         # flat L1 on the UN-annealed gate (LM:1827-1838, 2408)
    + λ Σ_{s,i} (1 − σ(θ_{s,i}))·|α_{s,i} − 1|      # leak guard (LM:2422-2437)
ν = mean_b t(b)² over the training anchors (uncensored), ≥ 1e-6                          (LM:2084-2100)
```

- Weights: γ_Z = 1 (`LM:2277`), γ_C = `dual_floor_weight` = 0.25 and γ_A = `triple_floor_weight` = 0.25 (`PH:40`;
  `driver:70`, GAMMA 0.25), γ_R = 1.0 (`driver:95`), k_R = 1 draw (`LM:2288-2293`).
- The rank-keep hinge and the censored read sit inside D_Z, D_C, D_A. Their effective weights are γ_π/ν: rank-keep is
  0.1/ν in Z and 0.025/ν in C and A.
- `amp_l1` = 0, so members' α is free (`CFG:811`).
- `site_lambda_weights` is not passed (`AG:351-404`), so pricing is flat per latent.
- **"Weighted" in WCM refers to the coefficients α (a weighted circuit), not to a weighted sparsity penalty.**
- Each term is backwarded separately, with gradient accumulation; one `opt.step()` per step (`LM:2253-2363, 2457`).
- **Unweighted arms (`U_`):** `free_amplitude=False` (`driver:682`, `PH:38`), so there are no α and no leak guard. The
  same sparse fill and R run at α = 1 exactly (`LM:584-590`). Rank-keep and censor as in WCM.

## 3. Optimisation hyperparameters

| Item | Value (production v4) | Code |
|---|---|---|
| Optimiser | AdamW over θ and ψ, weight decay 0.05. Decoupled decay also pulls ψ toward 0, i.e. α toward softplus(0) = 0.693 | `CFG:748, 756`; `LM:1579, 1666-1670` |
| Base lr | 0.05 × 3.15 = **0.1575 peak** | `CFG:682`; `driver:104, 443-447` |
| Schedule | **geometric** after a **warm-up of 0.1** (round(0.1·400) = 40 steps) | `driver:102`; `LM:1613-1650` |
| Warm-up shape | linear from lr·0.05 = 0.007875 to 0.1575; step 0 = 0.01162; peak at step 39 | `LM:1635-1638`; `CFG:773` (`lr_min_frac` 0.05) |
| Decay shape | lr·0.05^((step−40)/359), last step 0.007875 | `LM:1639-1641` |
| Budgets (computed on CPU from `lr_at`) | Σlr = 21.40 vs 20.00 for constant 0.05 (×1.070); decay budget Σlr·wd = 1.070 (calibrated ≈ 1.0) | |
| Steps | 400, fixed; no early stopping or checkpoint selection | `driver:73`; `PH:41` |
| Batch | 4 training contexts per step (`probe_batch_size` 4), taken in fixed sequential order `[(4t) mod 48, +4)`, no shuffling | `driver:314`; `experiments/049-circuit-graph/amp_eval_pass_v2.py:136`; `src/circuit/discovery/gradient_base.py:178`; `LM:1880-1885` |
| Micro-batch | none: `deep_site_threshold` 99 > max sites (35 TuringLLM, 47 GPT-2), so accum = 1 | `PH:39`; `LM:1389-1398` |
| Engine split | `holdout_frac` 0.25 on the padded probe; the train slice is exactly the 48 training rows | `CFG:724`; `LM:1674-1679`; `PC:374-400, 432-438` |
| Init | θ₀ = 4 (m ≈ 0.982) at every latent of every upstream site (`theta_init_mode` uniform); ψ₀ = softplus⁻¹(1), so α₀ = 1 | `CFG:725`; `LM:1484-1489, 1553-1560` |
| Gate | σ(θ/T), T annealed 1 → 0.05. **Not** hard-concrete: no sampling, deterministic | `LM:532-535, 2215-2238` |
| Membership | σ(θ) > 0.5 ⇔ θ > 0 (`keep_threshold` 0.5; anneal requires 0.5) | `CFG:723`; `LM:1239-1244, 2633` |
| α | softplus(ψ) ≥ 0, unbounded above, no clamp; stored rounded to 4 d.p. Unweighted: α ≡ 1 | `LM:137-140, 2575-2579` |
| Random sources | only the R donor draw (seed 20261017; `R_SEED` unset). Scoring's `free0_perm` uses an RNG seeded by crc32(key). No global `torch.manual_seed` | `driver:145-147`; `LM:310`; `WE:373` |
| Determinism | No deterministic-algorithms flag is set in this path, so GPU bit-reproducibility is **UNVERIFIED**. Empirically, native v3 fits reproduced experiment 102's circuits bit for bit on 3 of 3 targets (experiment 104). That MPS co-tenancy (P = 6, `chain.sh:201`) changes nothing is **UNVERIFIED** in code | |
| Precision, TuringLLM | model and SAEs bf16 on CUDA (`code_dtype` "stream" = bf16 codes; parameters fp32); `set_float32_matmul_precision("high")`; `autocast_bf16` off | `src/model/inference.py:70-74`; `src/sae/topk_sae.py:235-240`; `CFG:760, 1665`; `amp_eval_pass_v2.py:125` |
| Precision, public | model and SAEs float32 (`config.model.dtype` None → float32; the driver does not pass DTYPE, which applies only to the data pass); TF32 matmuls ("high") | `CFG:1964`; `src/model/hf_adapters.py:242-249`; `experiments/069-public-data-pass/common.py:41-57`; `driver:229, 301`; `chain.sh:186, 538` |
| Precision, arena A | float32, TF32 **off** ("highest", `allow_tf32` False) | `107/arena_a/driver_tcz.py:37-49` |

**Contexts per fit** (`CFG:1585-1593`; `PC:281-340`):
- Pools: strongest pool = top store, ≤ 64; mid-band pool ≤ 64. Each pool is split 3:1 (48 train / 16 held out); ranks
  1–2 always train (`keep_top` 2).
- **Training set:** arm B = 32 strongest (a stratified 2/3 of the 48 strongest-train) + 16 mid-band (`PC:319-324,
  365-371`).
- Thin targets (no mid pool, < 8 rows) fall back to arm A = all strongest-train (`PC:360-362`).
- Contrast: 64 close, verified-silent contexts, similarity-stratified, 48 train / 16 held out (`PC:326-336`).
- `min_pool` 8; `min_contrast` 4 (fewer = skipped).
- Window: 64 tokens, `pos = t[:, :64]` from 65-token loads (`PC:84-94`).
- Anchor: argmax of the target's post-Top-K activation per context (`PC:119-137`).

## 4. λ (sparsity price)

**TuringLLM.**
- **λ = 1e-3 fixed** (`driver:70` default; the TuringLLM launches pass no `LAM`, `chain.sh:815-827, 1013`).
- T4 λ sweep on the 190-target list (`chain.sh:240`; `driver:167-183`):
  - **WCM arms (`W_`):** 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3 (8);
  - **unweighted arms (`U_`):** 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 (5).
  - Every arm runs at v4 settings, rank-keep 1e-1 included (the version's rank-keep is the arm default, `driver:121,
    167-168`).
  - `W_0.001` reuses the T3 main circuit (`driver:694-697`).
- T10 without-R arm: `v4nr`, arm `W_0.001` (`chain.sh:248-249`; `driver:103`).

**Public models.**
- **λ pilot** (S5 Pythia / S8 GPT-2) on 30 pilot targets disjoint from the main list, `MODE=sweep`, at v4
  (`chain.sh:621-638`). Arms: Pythia `W_0.0003, W_0.001, W_0.003`; GPT-2 `W_0.001, W_0.003, W_0.01` (`chain.sh:628`).
- **Pre-stated rule**, size only (`082/pilot_lambda.py:11-16, 41-47, 104-117`):
  - keep λ = 1e-3 if its pooled median circuit size (every fitted held-out-strongest circuit of that arm) lies in
    [100, 1000];
  - else take the grid λ whose pooled median is nearest 494 nodes on a log scale, |log10(median/494)|. Ties go to the
    λ nearer 1e-3 in log10, then the larger λ.
  - Pass rates are reported, never used.
- The report is written with `"adopted": false`. The main fit **waits** until a λ is adopted by writing it to
  `STATE/lam_<model>` (manual checkpoint C3, `chain.sh:473-485`). The public production λ is therefore set at run time,
  not in code.
- Pythia's unweighted arm runs at `U_<adopted λ>` on 128 targets (`chain.sh:692-694`).

**Arena A.**
- λ grid `AA_LAMBDAS` 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3 (marked a placeholder in the script), each its own OUT, per scope
  a1 / a2 (`107/arena_a/run_pod.sh:83, 174, 635-640`).

**Arena D** (behavioural WCM).
- 9 λ: 3e-6 … 3e-2 (`107/arena_d/pod_stage_d.sh:56, 98`).

## 5. Evaluation

Scorer:
- Public and arena A: `WE.score_circuit(split="explicit")` (`driver:517-522`).
- TuringLLM: the frozen experiment-049 scorer `amp_eval_pass_v2.score_circuit` (`driver:524-525`), which `WE`
  reproduces (`WE:1-9`).

Every row is scored twice: held-out strongest (**primary**, pass rule) and held-out mid-band (reported)
(`driver:650-655`). The sweep scores held-out strongest only (`driver:704`).

**Contexts.**
- `pt` = the 48 strongest-train rows (the reference), then the held-out set (≤ 16).
- Contrast = 48 train + ≤ 16 held out.
- At most 64 rows of each (`WE:77, 383-392`; `PC:441-453`).

**Reads.**
- `_tk` = the target's post-Top-K activation at the anchor (primary).
- `_pre` = relu / JumpReLU of w·x + b (diagnostic).
- A ratio never mixes the two reads (`WE:193-204, 302-327`).

**Ratios** (`WE:101-110`):
- ratio(x, e) = (x − e)/(a_pos − e), None unless e < a_pos.
- a_pos = the mean natural `_tk` over the held-out rows (`WE:437`).

| Score | Definition (as implemented) | Code |
|---|---|---|
| **Z** `free0_tk` | Circuit-only run on held-out activating rows: members at α×live, every non-member at every upstream site **0**; baseline e0 = the same with no members | `WE:414-426, 446, 506-514` |
| **A** `freeM_topk_tk` | Circuit-only, sparsity-preserving fill with the A mean (48 strongest-train, all positions); baseline eM_topk | `WE:442, 447, 507-511`; `src/eval/ablation_faithfulness.py:267-318` |
| **C** `freeN_topk_tk` | The same with the C mean (48 contrast-train); baseline eN_topk | `WE:443, 449, 509` |
| **Necessity (a)** `phi_sup_blind_tk` | (a_pos − a_{M∖C})/(a_pos − e0). a_{M∖C} = every member SET to 0 in the **live** stream, all else re-encoded, SAE / transcoder errors live | `WE:106-110, 491-497, 541-543` |
| Necessity (b) | Members zeroed with **every upstream error frozen at its clean value**. Only in arena A's `score_arena_a.py` (`nec_b`, primary for the baseline comparisons) and arena D's `score_d.py`; **not** in the production pass rule | `107/arena_a/score_arena_a.py:24-31` |
| Sufficiency to induce `phi_cf_alpha_blind_tk` | §9 | `WE:486-489, 533-535` |
| Recorded | `freeM_dense`, `freeN_dense`, `free0_tr`, `free0_a1`, `free0_perm`, `phi_sup_alpha`, `phi_cf_alpha_role` / `phi_cf_a1_role`, `phi_pin_alpha_{zero,topk,dense}`, `free0_rand` (= the EMPTY circuit, not a random baseline) | `WE:506-583` |

**Pass rule** (`WE:85-90, 1169-1179`): Z, A, C all in **[0.75, 1.5]** AND `phi_sup_blind_tk` **≥ 0.9**, on held-out
strongest, `_tk`. A missing score fails. Verified: `BAND = (0.75, 1.5)`, `NEC_MIN = 0.9`.
- A second pass rate on the `_pre` read, with baseline 0 for Z and necessity, is reported. **UNVERIFIED** where that
  rate is computed for the result tables: it is not in `WE.passes`.

**Vacuous rule.**
- Vacuous ⇔ a_pos_tk ≤ 0, or a_pos − e < 0.05·a_pos for any e in {e0, eM_topk, eN_topk} (`WE:81-82, 118-121, 504,
  1148-1166`).
- Vacuous rows are dropped from every denominator and counted (`WE:1192-1200`).

**Held-out sets.**
- **Public** (`GROUP_SPLIT` 1 default, `driver:212`):
  - group-aware split by duplicate groups (exact, anchor-prefix, near: shared run ≥ 32 tokens or ≥ 50% shared 8-grams),
    then the held-out rule as a final filter (`PC:203-278`; `src/circuit/heldout_rule.py:50-55`). Rows are only
    removed, so held-out sets may hold < 16;
  - the split groups by duplicates only; there is no same-document grouping (the same as TuringLLM);
  - floor 8 (`heldout_rule.py:54`; reported, not dropped);
  - unstable-fire rows are dropped (`PC:259-271`).
- **TuringLLM:** fresh experiment-094 records (`CTX_DIR`, `chain.sh:826`).
  - Strongest pool deduplicated before the split; candidates from the stores plus the corpus scan's top 512
    (experiment 086's `scan.py`).
  - 48 / 16 strongest; mid-band 16 held out when it has ≥ 32 rows.
  - 48 / 16 deduplicated contrast.
  - The same final guard (`094/build_records.py:1-46`).
- **Contrast held-out rows** are deduplicated by whole-window exact copy plus near duplicate only (a ≥ 32-token shared
  run, or ≥ 50% shared 8-grams); there is no anchor-prefix rule for contrast rows. (Stated as a run rule; not
  re-checked line by line in code for this document.)

**Floors** (`src/eval/floors.py`):
- Means: dense post-Top-K code averaged over every (row, position), zeros included (`floors.py:141-145`).
- Pins: the mean code at the anchors (`floors.py:150-158`).
- No BOS skip on these families (`mean_skip_positions` absent; only arena D's Gemma bank sets it, `floors.py:113-130`).

**α = 1 rescore** (`experiments/062-h100-protocol-v1/rescore_alpha1.py`): the main circuits re-scored with every
α = 1 (`chain.sh:685-686, 1014-1016`; `WE:359`).

**Donor-background test** (`src/eval/donor_test.py:1-34, 78-112`):
- Bases: the 16 held-out strongest + 16 held-out contrast rows.
- Ablation: non-members take a donor's natural code at the same position (experiment 073's `ResamplePatcher`): members
  at α×live, the donor's active non-members fill the Top-K budget, error live.
- Pools:
  - **A** = 48 strongest-train;
  - **C** = 16 held-out contrast;
  - **R** = 512 random corpus windows (seed 20261016, ≥ 60 tokens), filtered against every record.
- Draws: 8 paired draws, seed 20261028 + 1000·L + i.
- Recovery = (mean_draw circuit − mean_draw empty)/(natural − empty).
- **Resample pass** = production pass AND recovery A and C (variant "fit") in [0.75, 1.5].
- Variants `fit`, `a1` (α = 1), `pinned` (members at their clean value × α); `DONOR_VARIANTS` selects them and must
  include `fit` (`donor_test.py:692-694`).
- R recovery and `silent_firing` are reported. `silent_firing` = the share of silent-base positions where the target
  fires, a false-positive check (`donor_test.py:349-351`).

**Resample / strict test (experiment 073).**
- `strict_eval.py` runs on the nulls list (`chain.sh:697-698`), with the harder nulls' anchor-circuit pool as
  `RAND_DIR`.
- **UNVERIFIED** in detail: experiment 073's code was not re-read for this document.

**Harder nulls (experiment 084) and stable core.** Run as companions. Not specified here (outside the method).

## 6. Per-model settings

| | TuringLLM | GPT-2 small | Pythia-70m | Arena A (GPT-2 tc+z) | Arena D (Gemma-2-2B) |
|---|---|---|---|---|---|
| Model | TuringLLM, 12 layers, d 1024, Phi-3 tokenizer (`families.py:98-104`) | HF `gpt2`, 12 layers, d 768 (`families.py:113-121`) | HF `EleutherAI/pythia-70m`, 6 layers, d 512, parallel residual (`families.py:105-112`) | HF `gpt2` + runtime family `gpt2-small-tcz` (`107/arena_a/wcm_tcz.py:1-32`) | `google/gemma-2-2b` fp32 (`pod_stage_d.sh:17, 151`) |
| Dictionary | own Top-K SAEs, d_sae 40,960, **k 128** (`topk_sae.py:112-114`) | OpenAI v5 32k Top-K SAEs (jbloom), d_sae 32,768, **k 32**, layer-norm input (`families.py:11-16, 86-94, 119-120`) | EleutherAI sparsify 32k, d_sae 32,768, **k 16**, no input norm (`families.py:7-10, 110-111`) | Kissane z-SAEs (attention, 24,576; L5 / L7 49,152 split z + z2) + Dunefsky transcoders (MLP, 24,576); **ReLU, k = d_sae** (`wcm_tcz.py:6-23`) | D1 PLT transcoders (26 tc sites); D2 Gemma Scope 16k res JumpReLU at L0 / 6 / 12 / 18 / 24; k = d_sae (`107/arena_d/wcm_gemma.py:3-27`) |
| Site kinds | attn, mlp, resid (36 sites; ≤ 35 upstream) | attn, resid_mid, mlp, resid (48; ≤ 47 upstream) | attn, mlp, resid (18) | scope a1 = z + z2 + tc; a2 = tc only (`wcm_tcz.py:30-31`) | tc (D1) / resid at 5 layers (D2) |
| Window | 64 | 64, no BOS (`families.py:15`) | 64, no BOS | 64 + a BOS prefix hidden from every site (`ARENA_A_BOS=1`, `wcm_tcz.py:25-28`; `driver_tcz.py:30-31`) | task prompts; BOS never edited, means skip position 0 (`wcm_gemma.py:20-30`) |
| Corpus | TuringLLM corpus, 24.8M 64-token windows (18.2M unique) (`experiments/085-heldout-audit/README.md:57, 220`) | apollo OWT gpt2-tokenised, parquet files 0–2 @f02886b5, **4,000,000 windows**, exact dedupe (`chain.sh:15-16, 185, 499-506`) | `monology/pile-uncopyrighted` train/00 @3be90335, **4,000,000 windows** (`chain.sh:492-497`) | the first 524,288 windows of OWT file 0 (`run_pod.sh:23, 172`) | SFC / Caples SVA task prompts (`wcm_gemma.py:39-44`) |
| dtype / TF32 | bf16 model + SAE; TF32 "high" | fp32; TF32 "high" | fp32; TF32 "high" | fp32; TF32 off | fp32 (`wcm_gemma.py:852`; `pod_stage_d.sh:436-438`); TF32 **UNVERIFIED** |
| Targets | 8,000 (`094/targets_tl8k.txt`), minus quarantined (`chain.sh:850-855`) | 512, seed 20261024, ≥ 200 firing windows, hub cap 50% (`082/make_targets_public.py:3-18, 57-62`) | 512 (32 per eligible component × 16) | 48 (`run_pod.sh:227`), selected after excluding the 15 tuning dev targets in `107/arena_a/tune_exclude.txt` (`run_pod.sh:107-108` passes it as `EXCLUDE`; excluded keys are never attempted, `107/arena_a/make_targets.py:53-54, 159`) | 4 SVA tasks (`pod_stage_d.sh:57, 101`) |
| Records / split | experiment-094 fresh records, group-aware, deduplicated contrast | `build_record` + group split + held-out rule | same as GPT-2 | `107/arena_a/make_targets.py` records (`run_pod.sh:102`) | task batches (experiment 081's engine) |
| WCM version | v4 | v4 | v4 | **v4 + `NEC_W` 0.3** (from the tuning file `107/arena_a/out/tune/CHOSEN.env`; `run_pod.sh:85, 176, 588-640`) | **v2 + arena C's `V3_LOGIT`** (§6.1) |
| Objective | latent (pos) | latent | latent | latent | **behavioural** (logit / task metric) |
| λ | 1e-3 fixed | pilot → C3 | pilot → C3 | 2.5e-4 … 4e-3 grid | 9-value grid 3e-6 … 3e-2 |
| Steps | 400 | 400 | 400 | 400 (no `STEPS` in the tuning file) | 400 (`pod_stage_d.sh:100`) |
| Approx. GPU-h (8×H100) | T3 main 54.3, T4 sweep 15.0, T11 donor 5.0, … (full table in `docs/reproducing-the-runs.md`) | 26.7 (whole track) | 10.7 (whole track) | up to about 68 GPU-h + 2 h serial (`run_pod.sh:62-64`) | **UNVERIFIED** |

### 6.1 Why arena D runs `WCM_VERSION=v2`

- Arena D's WCM is the **behavioural** (task-output) WCM of arena C. It runs through experiment 081's task engine
  (`wcm_gemma.py:591-608`), not the 062 latent driver.
- 081's `task_common.py` accepts only `v1` / `v2` and exits on anything else
  (`experiments/081-task-circuits/task_common.py:80-86`). The chain exports v4, so `pod_stage_d.sh` pins
  `WCM_VERSION=v2` (`pod_stage_d.sh:147-152`).
- The settings themselves come from arena C's `V3_LOGIT` (`107/arena_c/wcm_marks.py:78-81`): sparse fill + STE,
  R weight 1.0 (k 1), **rank-keep 0**, geometric lr.
- The ×3.15 peak, the warm-up and the censored term are deliberately **not** added: they were validated on latent
  targets only (`wcm_gemma.py:592-593`).
- So arena D's WCM is "behavioural v3-like", not v4. Arena D numbers should be read with that in mind.

## 7. Design properties

1. **Internal endpoint.** The target is a latent's pre-activation, not the model output (`LM:97-99`). The behavioural
   endpoint exists only in arenas C / D.
2. **Held-out generalisation.** Every pass number is on contexts the fit never saw. The engine's train slice is
   exactly the 48 training rows (`PC:374-400`). The A / C means use training rows only (`LM:954-960`; `eval_ref`).
3. **Train what you score.** A and C are trained under the scorer's sparsity-preserving fill with the scorer's own
   means (v2 and later; `LM:165-176`). The censored term (v4) also trains the post-Top-K read the scorer uses
   (`LM:1937-1949`).
4. **Why R.** R makes members hold the target against real other-context backgrounds, not only against means. It is
   the training analogue of the donor test (`LM:229-238`). On experiment 102's 72-target check panel, adding R raised
   the resample pass from 19 to 39 of 72.
5. **Why rank-keep.** On Top-K dictionaries a target can match its pre-activation and still be pushed out of the
   Top-K. The hinge holds the clean rank (`LM:909-913, 1973-1979`).
6. **Censored term, inert on ReLU and k = d_sae dictionaries.** With k = d_sae the K-th "other" pre-activation is the
   seed's own −∞ slot, so both gates are exactly 1 (`LM:1929-1931, 2165`).
   - Arena A's ReLU dictionaries have no threshold (`src/sae/dense.py:53-58`), so the term is inert there. The ReLU
     cut at 0 is not modelled.
   - It is **not** inert on JumpReLU dictionaries: τ is clamped at the threshold (`LM:1932-1933, 2166-2167`). This does
     not affect the production runs, because arena D does not use the censored term.
7. **Rank-keep stays active at k = d_sae.** It holds the clean rank among all latents. The Top-K motivation does not
   apply there (`driver_tcz.py:51-54`), but arena A keeps 1e-1 (the tuning file sets no `AA_RANK_W`).
8. **Flat L1 on σ(θ), not σ(θ/T); the leak guard.** α cannot be inflated behind a closed gate (`LM:2422-2437`).
9. **Geometric lr.** It holds lr/T roughly fixed under the anneal: lr/T ≈ 0.21 at step 40 → 0.1575 at step 399
   (computed on CPU), because T decays over all 400 steps while lr decays over steps 40–399. The ×3.15 peak restores
   Σlr (`driver:97-101, 126-129`; `LM:1607-1612`).
10. **MPS and determinism.** See §3. Small cross-hardware numerical differences are expected. Judge them by how
    widespread they are (there is no code guarantee).

## 8. Notes for readers of the code

1. **Set `WCM_VERSION=v4`.** The driver's default is `v1` (`driver:107`). Only the chain sets v4 for you. The driver
   marks each `OUT` with its version and refuses to mix versions in one `OUT`.
2. **Arena D runs v2 plus arena C's settings**, not v4, because its task engine accepts only v1 / v2 (§6.1).
3. **Weight decay pulls α toward ~0.69, not 1.** AdamW's decoupled weight decay also applies to ψ (`LM:1579, 1670`),
   so it pulls α toward softplus(0) = 0.693.
4. **The driver docstring describes protocol v1** (lr 0.05, rank-keep 3e-3) (`driver:3-10`). Those values are
   overridden at `driver:433-473`.
5. **Some config comments predate v4.** `CFG:680-682` ("schedule shape is irrelevant; constant 0.05") and
   `CFG:761-772` ("decay is the WRONG direction"; the schedule list omits `geometric`). Under v4 the engine accepts
   `geometric` (`LM:1613-1614`).
6. **The censored term covers Z as well.** The driver comment (`driver:100`) says "the A / C terms train the post-Top-K
   activation". The code applies it to Z, C and A (`LM:2133-2135`), not to R or necessity.
7. **The rank-keep weight is not 1e-1 in absolute terms.** It is multiplied by γ_π/ν: 0.1/ν in Z, 0.025/ν in C and A
   (`LM:1991-1992` inside `LM:2354`).
8. **The λ-pilot rule's 494-node anchor is a v1 TuringLLM median** (`pilot_lambda.py:41`). v3 circuits were about
   ×1.33 larger (median 702) and v4 circuits about ×0.89 of v3. The rule was stated before the pilot, so it stands as
   written.
9. **The public-model λ is set at run time** (checkpoint C3), not in code (§4).
10. **UNVERIFIED** (not checked in code):
    - bit-determinism and MPS invariance on H100;
    - where the `_pre`-read pass rate is computed for the result tables;
    - experiment 073's strict-eval internals;
    - arena D's TF32 setting and GPU-hours;
    - TuringLLM BOS handling in its loader (no BOS logic found in `src/model/inference.py`).

## 9. Sufficiency to induce

**It exists, it is computed on every scored row, and it is not in the pass rule.**

- **The read:** `phi_cf_alpha_blind_tk`, named `INDUCE` at `WE:91`.
- **Contexts:** the **16 held-out contrast contexts**, where the target is silent by construction (verified silent at
  every position, `PC:326-336`).
- **Anchors:** the argmax of the target's natural pre-activation on each contrast row (`WE:452-466`).
- **The edit:** every member is SET, at every position, to α × its collapsed pin. The pin is the member's mean code at
  the anchors of the 48 strongest-train rows (`WE:442, 470-471`; `src/eval/floors.py:150-158`). The edit runs in the
  **live** stream through `CounterfactualInterventionPatcher` (`WE:428-432, 533-535`): non-members are re-encoded
  live, not ablated.
- **Score:** (A⁻_int − A⁻)/(a_pos − A⁻), with A⁻ = the natural target at those anchors (`WE:486-489, 467-469`).
- **Variants:** `phi_cf_a1_role_tk` (α = 1) and `phi_cf_alpha_role_tk` (role-blind under `skip_roles`) are recorded
  (`WE:553-558`).
- **TuringLLM path:** the frozen 049 scorer computes the same field (`amp_eval_pass_v2.py:519`).

**What the other reads measure.** None of them induces on a silent context:
- **Z `free0`:** the circuit alone, on held-out **activating** contexts, with every non-member zeroed. Members keep
  their live activating-context values.
- **A `freeM_topk` / C `freeN_topk`:** the same activating contexts with non-members at the sparsity-preserving A / C
  mean fill. C is "sufficiency against a contrast background". The members still carry their activating-context
  values, so it is **not** induction.
- **`freeM_dense` / `freeN_dense`:** the dense-fill versions.
- `eM_topk` / `eN_topk` / `e0` are the empty-circuit baselines of A / C / Z.
- **Donor test:** bases are activating contexts. Its `silent_firing` on the 16 silent bases is a false-positive check
  (does the target fire with members at their live, silent-context values), not induction.
- **Arena A's `score_arena_a.py`:** computes Z / A / C / `nec_a` / `nec_b` and donor recovery only (lines 21-31). The
  induce read for arena A circuits comes only from the 062 driver's own eval rows, which `driver_tcz.py` runs.

**Where it appears.** Every main and sweep scoring row on all three model tracks, and arena A's driver rows, carry it.
`experiments/062-h100-protocol-v1/merge.py` reports its median (`merge.py:104, 115, 156, 196`). It is **not** in
`WE.passes` (`WE:1169-1179`), nor in the donor resample pass.

**Caveats on this read as implemented.**
- The members are injected position-uniformly at their anchor-mean, not position-specifically.
- The rest of the stream is live, so the read measures drive in a natural silent context.
- It does not measure "the circuit alone". A circuit-only induce read (members injected on contrast rows with
  non-members at zero or the C sparse fill) does not exist in the code.
