# WCM protocol: method and evaluation specification

One document for every protocol version, newest first:
- **§0: the current production protocol, `WCM_VERSION=v4`** (what the 8×H100 pod runs). Authoritative.
- **§1–§11: protocol v1**, as executed by the TuringLLM 15k run (the history the later versions build on).
- **§12: the v2 and v3 changes** (sparse-fill training, fresh TuringLLM records, the resample term R, vacuous
  denominators). §0 states the combined result; §12 keeps each decision and its evidence.

---

## 0. Current production protocol: WCM v4

Taken from the code on branch `multi-device`. Every value comes from the code or config the pod executes, cited as
`path:line` (paths from the repo root). Where a document disagrees with the code, the code's value is given and the
disagreement is listed in §0.8. **UNVERIFIED** marks anything not confirmed in code.

Production = `WCM_VERSION=v4`. `chain.sh` exports it for every track (`experiments/082-public-runs/chain.sh:193`) and
passes it to every launch (`chain.sh:461`, `chain.sh:827`). The driver's own default is `v1`
(`experiments/062-h100-protocol-v1/driver.py:107`), so a run started outside the chain without `WCM_VERSION` is
**not** production.

Short names used below: `driver` = `experiments/062-h100-protocol-v1/driver.py`; `LM` =
`src/circuit/instrument/learned_mask.py`; `AG` = `src/circuit/discovery/ablation_gradient.py`; `PH` =
`experiments/059-context-pool/protocol_harness.py`; `PC` = `src/circuit/protocol_contexts.py`; `WE` =
`src/eval/wcm_eval.py`; `CFG` = `src/config.py`.

### 0.1 Summary

WCM fits a **weighted circuit** for one target SAE latent (a dictionary latent at layer *L*, site kind *k*). The
circuit is a set of latents at the target's causally upstream SAE sites, plus one coefficient α per member. Each
latent has two parameters: a gate logit θ, with gate m = σ(θ/T) annealed toward binary, and an amplitude ψ, with
α = softplus(ψ). The parameters are trained with AdamW for 400 steps. The loss asks the target's **pre-activation** at
its anchor positions on 48 training activating contexts to stay at its natural value. This must hold under four
ablations of every non-member: zero (Z), the contrast-context mean (C), the activating-context mean (A), and a real
contrast context's code (R_C). C and A use the scorer's sparsity-preserving fill. The loss adds a rank-keep hinge, a
censored (soft Top-K) read, and a flat L1 price λ per kept gate. The output is the set {θ > 0} with each member's
final α, rounded to 4 d.p. (`LM:2560-2638`, `LM:2566-2580`, `src/circuit/discovery/gradient_base.py:440-476`). It is
scored on held-out contexts by the Z / A / C faithfulness ratios and by necessity.

### 0.2 Loss (v4, objective `pos`, mask_floor_source `triple`, free amplitude)

#### 0.2.1 Masked forward (every term)

At every upstream site s, the site's code is re-encoded from the current, already-edited stream. The stream is then
updated as x ← x + W_dec[code − dense], so the SAE error is preserved (`LM:511-604`).
- The upstream sites are all lower layers, plus the earlier kinds in the same layer, in forward order (`AG:291-293`;
  `src/model/families.py:27-40, 102, 109, 116-118`).
- The target's own site is not edited. Its pre-activation `w·x + b` is tapped there (`LM:493-506`).
- Gate: m = σ(θ / T) under `binarize="anneal"` (`LM:532-535`; config default `CFG:903`; nothing overrides it).
- Temperature: T = 0.05^(step/(S−1)), from 1 to 0.05 (`LM:2215-2238`; `anneal_t0` 1.0, `anneal_reach_frac` 1.0 are
  the engine defaults `LM:1033, 1039`).
- Code by term:
  - **Z** (`patcher_zero`, floors None): code = m·α·dense (`LM:1770-1776`, `LM:557-565`).
  - **C** (`patcher`): floor = the C mean.
  - **A** (`patcher_pos`): floor = the A mean.
  - C and A under v2+ (`train_fill="sparse"`, `driver:94-95, 102`) take the scorer's sparsity-preserving fill.
    - Members (θ > 0) are kept at m·α·live. Membership is `m.detach() > 0.5`, i.e. θ > 0 (`LM:472`).
    - Budget per (row, position) = K − #members with α·live ≠ 0 (`LM:179-214`).
    - The fill is the top-`budget` non-members ranked by mean, at their mean. Everything else is 0.
    - The keep mask H is applied straight-through: forward H·code, backward identity (`LM:217-226, 460-479`,
      `sparse_grad="ste"` `driver:94`).
  - **R_C** (`patcher_donor`): floor = the natural sparse code of a donor context at the same position. One donor row
    per training row is redrawn at every forward from a CPU generator seeded 20261017 (`LM:239, 295-329, 607-615`;
    `CFG:954`).
    - The donor keep mask: members kept; the donor's positive non-member entries fill K − #active members, largest
      first (`LM:243-259`).
    - The donors are the target's **48 contrast-train rows** (`driver:495-498`, `resample_pool="contrast"`
      `driver:95`). Their codes are captured once, at batch 16 (`LM:240, 263-292, 1789-1797`).
- K = the dictionary's k (`LM:476`, `bank.k`): TuringLLM 128 (`src/sae/topk_sae.py:112-114`), Pythia 16, GPT-2 32
  (`src/model/families.py:110, 119`).

#### 0.2.2 Where the A and C means come from (v2+ `train_means_source="eval_ref"`)

- **C mean:** the dense per-latent mean over all (row, position) slots, zeros included, of the 48 contrast-train rows
  (`AG:414-435`; `src/eval/floors.py:141-145`).
- **A mean:** the same over the **48 strongest-train reference rows** (`PC:456-469`; `driver:486-491`). In arm B this
  is not the fit's own training set (32 strongest + 16 mid).
- These are bit-identical to the scorer's means (`WE:442-443`).

#### 0.2.3 Data terms

Per term π, from `LM:1878-1994`:

```
D_π = mean_b ( v_π(b) − t(b) )²                                         (LM:1950)
  v_π(b) = target pre-activation w·x+b at training anchor b under ablation π   (LM:1897-1898)
  t(b)   = the same in the unedited model ("reproduce, don't maximise")        (LM:1688-1690)
```

**Censored read** (v4, `censor_temp` 0.05, `driver:102`). Applied to Z, C and A only (the patchers with
`capture_site`, `LM:2126-2135, 1923`); not to R or necessity.

```
v ← v · σ((v − τ_K)/T_b),   t ← t · σ((t − τ_K^clean)/T_b)              (LM:1937-1949)
  τ_K       = the K-th largest OTHER pre-activation at the target's site under that ablation   (LM:1926-1933)
  τ_K^clean = the same in the clean model                                                      (LM:2165-2167)
  both clamped at the latent's JumpReLU threshold when it has one
  T_b = 0.05 × |clean target pre-activation at b|, ≥ 1e-4                                      (LM:2168)
```

**Rank-keep** (`rank_mode="keep"`, PH:41; weight 1e-1 under v3 / v4, `driver:105-106, 436-438`; δ 0). Applied inside
the Z, C and A terms, after the MSE (`LM:1967-1993`):

```
+ γ_K · mean_b relu(τ^(r_b) − v_π(b))²
  r_b    = the target's clean rank among the other site latents at anchor b, clamped to [1, k]   (LM:2157-2158)
  τ^(r)  = the r-th largest other pre-activation under the ablation (uncensored v_π)
```

**Necessity term** (off in production: `NEC_W` default 0, `driver:150, 468`). Arena A runs it at weight 0.3, target
0.95 (`driver:151`; `107/arena_a/out/tune/CHOSEN.env`).
- A fifth forward: members removed (code = dense·(1 − m)), everything else natural (`LM:540-543, 1803-1808`).
- Its term is a one-sided hinge `mean relu(sgn·(v − [e0 + (1−0.95)(t − e0)]))²` (`LM:1899-1907`). e0 = the target
  under the zero empty circuit (`LM:2172`). The hinge is 0 while the pre-activation necessity is ≥ 0.95.

**Off in production:**
- off-target / specificity (`offtarget_weight` 0, `PH:40`);
- rival term (`RIVAL_W` 0, `driver:139, 454`);
- adaptive λ (`LAM_ADAPT_EPS` None, `driver:141, 455`);
- neg-suppress (`CFG:871`);
- margin (`CFG:875`).

#### 0.2.4 Total loss per step

The engine computes it at `LM:2261-2296, 2375-2444`:

```
L = (1/ν)[ D_Z + 0.25·D_C + 0.25·D_A + 1.0·D_R ]  (+ (0.3/ν)·D_nec, arena A only)
    + λ Σ_{s,i} σ(θ_{s,i})                         # flat L1 on the UN-annealed gate (LM:1827-1838, 2408)
    + λ Σ_{s,i} (1 − σ(θ_{s,i}))·|α_{s,i} − 1|      # leak guard (LM:2422-2437)
ν = mean_b t(b)² over the training anchors (uncensored), ≥ 1e-6                          (LM:2084-2100)
```

- Weights: γ_Z = 1 (`LM:2277`), γ_C = `dual_floor_weight` = 0.25 and γ_A = `triple_floor_weight` = 0.25 (`PH:40`,
  `driver:70` GAMMA 0.25), γ_R = 1.0 (`driver:95`), k_R = 1 draw (`LM:2288-2293`).
- The rank-keep hinge and the censored read sit inside D_Z, D_C, D_A. Their effective weights are γ_π/ν: rank-keep is
  0.1/ν in Z and 0.025/ν in C and A.
- `amp_l1` = 0, so members' α is free (`CFG:811`).
- `site_lambda_weights` is not passed (`AG:351-404`), so pricing is flat per latent.
- **"Weighted" in WCM refers to the coefficients α (a weighted circuit), not to a weighted sparsity penalty.**
- Each term is backwarded separately, with gradient accumulation; one `opt.step()` per step (`LM:2253-2363, 2457`).
- **Unweighted arms (`U_`):** `free_amplitude=False` (`driver:682`, `PH:38`), so there are no α and no leak guard. The
  same sparse fill and R run at α = 1 exactly (`LM:584-590`). Rank-keep and censor as WCM.

### 0.3 Optimisation hyperparameters

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
| Determinism | No deterministic-algorithms flag is set in this path, so GPU bit-reproducibility is **UNVERIFIED** here. Native v3 fits reproduced 102's circuits bit for bit on 3 targets (exp 104, pod runbook). MPS co-tenancy (P=6, `chain.sh:201`) changing nothing is **UNVERIFIED** in code | pod runbook (local notes) |
| Precision, TuringLLM | model and SAEs bf16 on CUDA (`code_dtype` "stream" = bf16 codes; parameters fp32); `set_float32_matmul_precision("high")`; `autocast_bf16` off | `src/model/inference.py:70-74`; `src/sae/topk_sae.py:235-240`; `CFG:760, 1665`; `amp_eval_pass_v2.py:125` |
| Precision, public | model and SAEs float32 (`config.model.dtype` None → float32; the driver does not pass DTYPE, which applies only to the data pass); TF32 matmuls ("high") | `CFG:1964`; `src/model/hf_adapters.py:242-249`; `experiments/069-public-data-pass/common.py:41-57`; `driver:229, 301`; `chain.sh:186, 538` |
| Precision, arena A | float32, TF32 **off** ("highest", `allow_tf32` False) | `107/arena_a/driver_tcz.py:37-49` |

**Contexts per fit** (`CFG:1585-1593`; `PC:281-340`):
- Pools: strongest pool = top store, ≤ 64; mid-band pool ≤ 64. Each pool is split 3:1 (48 train / 16 held out),
  ranks 1–2 always train (`keep_top` 2).
- **Training set:** arm B = 32 strongest (a stratified 2/3 of the 48 strongest-train) + 16 mid-band (`PC:319-324,
  365-371`).
- Thin targets (no mid pool, < 8 rows) fall back to arm A = all strongest-train (`PC:360-362`).
- Contrast: 64 close, verified-silent contexts, similarity-stratified, 48 train / 16 held out (`PC:326-336`).
- `min_pool` 8; `min_contrast` 4 (fewer = skipped).
- Window: 64 tokens, `pos = t[:, :64]` from 65-token loads (`PC:84-94`).
- Anchor: argmax of the target's post-Top-K activation per context (`PC:119-137`).

### 0.4 λ (sparsity price)

**TuringLLM.**
- **λ = 1e-3 fixed** (`driver:70` default; the TL launches pass no `LAM`, `chain.sh:815-827, 1013`).
- T4 λ sweep on the 190-target list (`chain.sh:240`; `driver:167-183`):
  - **W arms:** 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3 (8);
  - **U arms:** 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 (5).
  - Every arm runs at v4 settings, rank-keep 1e-1 included (the version's RANK is the arm default, `driver:121,
    167-168`).
  - W_0.001 reuses the T3 main circuit (`driver:694-697`).
- T10 without-R arm: `v4nr`, arm `W_0.001` (`chain.sh:248-249`; `driver:103`).

**Public models.**
- **λ pilot** (S5 Pythia / S8 GPT-2) on 30 pilot targets disjoint from main, `MODE=sweep`, at v4 (`chain.sh:621-638`).
  Arms: Pythia `W_0.0003, W_0.001, W_0.003`; GPT-2 `W_0.001, W_0.003, W_0.01` (`chain.sh:628`).
- **Pre-stated rule**, size only (`experiments/082-public-runs/pilot_lambda.py:11-16, 41-47, 104-117`):
  - keep λ = 1e-3 if its pooled median circuit size (every fitted held-out-strongest circuit of that arm) lies in
    [100, 1000];
  - else take the grid λ whose pooled median is nearest 494 nodes on a log scale, |log10(median/494)|. Ties go to the
    λ nearer 1e-3 in log10, then the larger λ.
  - Pass rates are reported, never used.
- The report is written with `"adopted": false`. The main fit **waits** until Daniel writes `STATE/lam_<model>` (C3,
  `chain.sh:473-485`). So the public production λ is **not yet known**: it is decided on the pod.
- Pythia's unweighted arm runs at `U_<adopted λ>` on 128 targets (`chain.sh:692-694`).

**Arena A.**
- λ grid `AA_LAMBDAS` 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3 ("placeholder"), each its own OUT, per scope a1 / a2
  (`107/arena_a/run_pod.sh:83, 174, 635-640`).

**Arena D** (behavioural WCM).
- 9 λ: 3e-6 … 3e-2 (`107/arena_d/pod_stage_d.sh:56, 98`).

### 0.5 Evaluation

Scorer:
- Public and arena A: `WE.score_circuit(split="explicit")` (`driver:517-522`).
- TuringLLM: the frozen 049 scorer `amp_eval_pass_v2.score_circuit` (`driver:524-525`), which `WE` reproduces
  (`WE:1-9`).

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
| Necessity (b) | Members zeroed with **every upstream error frozen at its clean value**. Only in arena A's `score_arena_a.py` (`nec_b`, primary for the baseline comparisons) and arena D's `score_d.py`; **not** in the production rule | `107/arena_a/score_arena_a.py:24-31` |
| Sufficiency to induce `phi_cf_alpha_blind_tk` | §0.9 | `WE:486-489, 533-535` |
| Recorded | `freeM_dense`, `freeN_dense`, `free0_tr`, `free0_a1`, `free0_perm`, `phi_sup_alpha`, `phi_cf_alpha_role` / `phi_cf_a1_role`, `phi_pin_alpha_{zero,topk,dense}`, `free0_rand` (= the EMPTY circuit, not a random baseline) | `WE:506-583` |

**Pass rule** (`WE:85-90, 1169-1179`): Z, A, C all in **[0.75, 1.5]** AND `phi_sup_blind_tk` **≥ 0.9**, on held-out
strongest, `_tk`. A missing score fails. Verified: `BAND = (0.75, 1.5)`, `NEC_MIN = 0.9`.
- A second pass rate on the `_pre` read is reported, with baseline 0 for Z and necessity (§12.1).
  **UNVERIFIED** where that rate is computed for the pod tables: it is not in `WE.passes`.

**Vacuous rule (v2, §12.4).**
- Vacuous ⇔ a_pos_tk ≤ 0, or a_pos − e < 0.05·a_pos for any e in {e0, eM_topk, eN_topk} (`WE:81-82, 118-121, 504,
  1148-1166`).
- Vacuous rows are dropped from every denominator and counted (`WE:1192-1200`).

**Held-out sets.**
- **Public** (`GROUP_SPLIT` 1 default, `driver:212`):
  - group-aware split by duplicate groups (exact, anchor-prefix, near: shared run ≥ 32 tokens or ≥ 50% shared 8-grams),
    then the held-out rule as a final filter (`PC:203-278`; `src/circuit/heldout_rule.py:50-55`). Rows are only
    removed, so held-out sets may hold < 16;
  - floor 8 (`heldout_rule.py:54`, reported, not dropped);
  - unstable-fire rows are dropped (`PC:259-271`).
- **TuringLLM:** fresh 094 records (`CTX_DIR`, `chain.sh:826`).
  - Strongest pool deduplicated before the split; candidates from the stores plus the 086 corpus scan's top 512.
  - 48 / 16 strongest; mid-band 16 held out when it has ≥ 32 rows.
  - 48 / 16 deduplicated contrast.
  - The same final guard (`experiments/094-tl-v2-records/build_records.py:1-46`).

**Floors** (`src/eval/floors.py`):
- Means: dense post-Top-K code averaged over every (row, position), zeros included (`floors.py:141-145`).
- Pins: the mean code at the anchors (`floors.py:150-158`).
- No BOS skip on these families (`mean_skip_positions` absent; only arena D's Gemma bank sets it, `floors.py:113-130`).

**α = 1 rescore** (`rescore_alpha1.py`): the main circuits re-scored with every α = 1 (`chain.sh:685-686,
1014-1016`; `WE:359`).

**Donor-background test** (`src/eval/donor_test.py:1-34, 78-112`):
- Bases: the 16 held-out strongest + 16 held-out contrast rows.
- Ablation: non-members take a donor's natural code at the same position (073's `ResamplePatcher`): members at α×live,
  the donor's active non-members fill the Top-K budget, error live.
- Pools:
  - **A** = 48 strongest-train;
  - **C** = 16 held-out contrast;
  - **R** = 512 random corpus windows (seed 20261016, ≥ 60 tokens), filtered against every record.
- Draws: 8 paired draws, seed 20261028 + 1000·L + i.
- Recovery = (mean_draw circuit − mean_draw empty)/(natural − empty).
- **Resample pass** = production pass AND recovery A and C (variant "fit") in [0.75, 1.5].
- Variants `fit`, `a1` (α = 1), `pinned` (members at their clean value × α).
- R recovery and `silent_firing` are reported. `silent_firing` = the share of silent-base positions where the target
  fires, a false-positive check (`donor_test.py:349-351`).

**Resample / strict test (073).**
- `strict_eval.py` runs on the nulls list (`chain.sh:697-698`), with the harder nulls' anchor-circuit pool as
  `RAND_DIR`.
- **UNVERIFIED** in detail here: 073's code was not re-read for this section.

**Harder nulls (084), stable core.** Run as companions. Not specified here (outside the method).

### 0.6 Per-model settings

| | TuringLLM | GPT-2 small | Pythia-70m | Arena A (GPT-2 tc+z) | Arena D (Gemma-2-2B) |
|---|---|---|---|---|---|
| Model | TuringLLM, 12 layers, d 1024, Phi-3 tokenizer (`families.py:98-104`) | HF `gpt2`, 12 layers, d 768 (`families.py:113-121`) | HF `EleutherAI/pythia-70m`, 6 layers, d 512, parallel residual (`families.py:105-112`) | HF `gpt2` + runtime family `gpt2-small-tcz` (`107/arena_a/wcm_tcz.py:1-32`) | `google/gemma-2-2b` fp32 (`pod_stage_d.sh:17, 151`) |
| Dictionary | own Top-K SAEs, d_sae 40,960, **k 128** (`topk_sae.py:112-114`) | OAI v5 32k Top-K SAEs (jbloom), d_sae 32,768, **k 32**, layer-norm input (`families.py:11-16, 86-94, 119-120`) | EleutherAI sparsify 32k, d_sae 32,768, **k 16**, no input norm (`families.py:7-10, 110-111`) | Kissane z-SAEs (attention, 24,576; L5 / L7 49,152 split z + z2) + Dunefsky transcoders (MLP, 24,576); **ReLU, k = d_sae** (`wcm_tcz.py:6-23`) | D1 PLT transcoders (26 tc sites); D2 Gemma Scope 16k res JumpReLU at L0 / 6 / 12 / 18 / 24; k = d_sae (`wcm_gemma.py:3-27`) |
| Site kinds | attn, mlp, resid (36 sites; ≤ 35 upstream) | attn, resid_mid, mlp, resid (48; ≤ 47 upstream) | attn, mlp, resid (18) | scope a1 = z + z2 + tc; a2 = tc only (`wcm_tcz.py:30-31`) | tc (D1) / resid at 5 layers (D2) |
| Window | 64 | 64, no BOS (`families.py:15`) | 64, no BOS | 64 + a BOS prefix hidden from every site (`ARENA_A_BOS=1`, `wcm_tcz.py:25-28`; `driver_tcz.py:30-31`) | task prompts; BOS never edited, means skip position 0 (`wcm_gemma.py:20-30`) |
| Corpus | TuringLLM corpus, 24.8M 64-token windows (18.2M unique) (`experiments/085-heldout-audit/README.md:57, 220`) | apollo OWT gpt2-tokenised, parquet files 0–2 @f02886b5, **4,000,000 windows**, exact dedupe (`chain.sh:15-16, 185, 499-506`) | `monology/pile-uncopyrighted` train/00 @3be90335, **4,000,000 windows** (`chain.sh:492-497`) | the first 524,288 windows of OWT file 0 (`run_pod.sh:23, 172`) | SFC / Caples SVA task prompts (`wcm_gemma.py:39-44`) |
| dtype / TF32 | bf16 model + SAE; TF32 "high" | fp32; TF32 "high" | fp32; TF32 "high" | fp32; TF32 off | fp32 (`wcm_gemma.py:852`; `pod_stage_d.sh:436-438`); TF32 **UNVERIFIED** |
| Targets | 8,000 (`094/targets_tl8k.txt`, 8,000 lines), minus quarantined (`chain.sh:850-855`) | 512, seed 20261024, ≥ 200 firing windows, hub cap 50% (`082/make_targets_public.py:3-18, 57-62`) | 512 (32 per eligible component × 16) | 48 (`run_pod.sh:227`), minus the 9 tuning dev targets (`tune_exclude.txt`; **UNVERIFIED** how they are excluded) | 4 SVA tasks (`pod_stage_d.sh:57, 101`) |
| Records / split | 094 fresh records, group-aware, deduplicated contrast | `build_record` + group split + held-out rule | same as GPT-2 | 107 `make_targets.py` records (`run_pod.sh:102`) | task batches (081 Engine) |
| WCM version | v4 | v4 | v4 | **v4 + NEC_W 0.3** (`CHOSEN.env`; `run_pod.sh:85, 176, 588-640`) | **v2 + arena C's `V3_LOGIT`** (§0.6.1) |
| Objective | latent (pos) | latent | latent | latent | **behavioural** (logit / task metric) |
| λ | 1e-3 fixed | pilot → C3 | pilot → C3 | 2.5e-4 … 4e-3 grid | 9-value grid 3e-6 … 3e-2 |
| Steps | 400 | 400 | 400 | 400 (no STEPS in `CHOSEN.env`) | 400 (`pod_stage_d.sh:100`) |
| GPU-h (pod runbook, v3 basis; v4 fit time ≈ unchanged) | T3 main 54.3, T4 sweep 15.0, T11 donor 5.0, … | 26.7 (whole track) | 10.7 (whole track) | about 68 GPU-h + 2 h serial | **UNVERIFIED** here |

#### 0.6.1 Why arena D runs `WCM_VERSION=v2`

- Arena D's WCM is the **behavioural** (task-output) WCM of arena C. It runs through experiment 081's task engine
  (`wcm_gemma.py:591-608`), not the 062 latent driver.
- 081's `task_common.py` accepts only `v1` / `v2` and exits on anything else (`experiments/081-task-circuits/
  task_common.py:80-86`). The chain exports v4, so `pod_stage_d.sh` pins `WCM_VERSION=v2` (`pod_stage_d.sh:147-152`).
- The settings themselves come from arena C's `V3_LOGIT` (`107/arena_c/wcm_marks.py:78-81`): sparse fill + STE,
  R weight 1.0 (k 1), **rank-keep 0**, geometric lr.
- The ×3.15 peak, the warm-up and the censored term are deliberately **not** added: they were validated on latent
  targets only (`wcm_gemma.py:592-593`).
- So arena D's WCM is "behavioural v3-like", not v4. Say so wherever arena D numbers appear.

### 0.7 Design properties worth stating

1. **Internal endpoint.** The target is a latent's pre-activation, not the model output (`LM:97-99`). The behavioural
   endpoint exists only in arenas C / D.
2. **Held-out generalisation.** Every pass number is on contexts the fit never saw. The engine's train slice is
   exactly the 48 training rows (`PC:374-400`). The A / C means use training rows only (`LM:954-960`; v2 `eval_ref`).
3. **Train what you score.** A and C are trained under the scorer's sparsity-preserving fill with the scorer's own
   means (v2; `LM:165-176`). The censored term (v4) also trains the post-Top-K read the scorer uses (`LM:1937-1949`).
4. **Why R.** R makes members hold the target against real other-context backgrounds, not only against means. It is
   the training analogue of the donor test (`LM:229-238`). On the 102 check panel the resample pass went 19 → 39 of 72
   (§12.3).
5. **Why rank-keep.** On Top-K dictionaries a target can match its pre-activation and still be pushed out of the
   Top-K. The hinge holds the clean rank (`LM:909-913, 1973-1979`).
6. **Censored term, inert on ReLU and k = d_sae dictionaries.** With k = d_sae the K-th "other" pre-activation is the
   seed's own −∞ slot, so both gates are exactly 1 (`LM:1929-1931, 2165`).
   - Arena A's ReLU dictionaries have no threshold (`src/sae/dense.py:53-58`), so the term is inert there. The ReLU
     cut at 0 is not modelled.
   - It is **not** inert on JumpReLU dictionaries: τ is clamped at the threshold (`LM:1932-1933, 2166-2167`). This is
     irrelevant on the pod, because arena D does not use censor.
7. **Rank-keep stays active at k = d_sae.** It holds the clean rank among all latents. The "Top-K motivation does not
   apply" there (`driver_tcz.py:51-54`), but arena A keeps 1e-1 (no `AA_RANK_W` in `CHOSEN.env`).
8. **Flat L1 on σ(θ), not σ(θ/T); the leak guard.** α cannot be inflated behind a closed gate (`LM:2422-2437`).
9. **Geometric lr.** It holds lr/T roughly fixed under the anneal: lr/T ≈ 0.21 at step 40 → 0.1575 at step 399 on
   CPU, because T decays over all 400 steps while lr decays over steps 40–399. The ×3.15 peak restores Σlr
   (`driver:97-101, 126-129`; `LM:1607-1612`).
10. **MPS and determinism.** See §0.3. Small cross-hardware numerical differences are expected. Judge them by how
    widespread they are (no code guarantee).

### 0.8 Open questions and discrepancies (code wins)

1. **v4 in this document.** This section (§0) is the v4 specification. §12.3's T10 arm (`W_0.001_r0.1`, v2) is the
   v3 rule; under v4 the chain runs `v4nr` `W_0.001` (`chain.sh:248-249`).
2. **`paper/main.tex` hyperparameter table (lines 861-869) describes v2**, not v4. It says:
   - rank-keep 3e-3 (code: 1e-1);
   - "AdamW, learning rate 0.05 (constant)" (code: geometric, peak 0.1575, warm-up 40 steps);
   - no R term (code: γ_R = 1.0 on contrast donors);
   - no censored read;
   - ablation methods "Z + C + A" (code: Z + C + A + R_C).
3. **Driver docstring** (`driver:3-10`) still describes protocol v1 (lr 0.05, rank-keep 3e-3). It is overridden at
   `driver:433-473`.
4. **Config comments are stale for v4.**
   - `CFG:680-682` ("schedule shape is irrelevant; constant 0.05") and `CFG:761-772` ("decay is the WRONG direction";
     the schedule list omits `geometric`). Under v4 the engine accepts `geometric` (`LM:1613-1614`).
5. **The censored term covers Z as well.** The driver comment (`driver:100`) says "the A / C terms train the post-Top-K
   activation". The code applies it to Z, C and A (`LM:2133-2135`), not to R or necessity.
6. **The rank-keep weight is not 1e-1 in absolute terms.** It is multiplied by γ_π/ν: 0.1/ν in Z, 0.025/ν in C and A
   (`LM:1991-1992` inside `LM:2354`). §4.2 states this for v1; it holds for v4 too.
7. **The λ-pilot rule's 494 anchor is the v1 TuringLLM median** (`pilot_lambda.py:41`). v3 circuits were ×1.33 larger
   (median 702, §12.3), and v4 about ×0.89 of v3 (pod runbook). The rule is pre-stated, so it stands, but the paper
   should say the anchor is a v1 size.
8. **§4.2 / §5.1 say the A mean is over the fit's training set (arm B).** That is v1. Under v2+ both training and
   scoring use the 48 strongest-train rows (`AG:414-435`). §12.1 records the change; §0.2.2 states the v4 value.
9. **The public λ is undecided until C3 on the pod.** `paper/main.tex:868` already carries `\pending`.
10. **AdamW weight decay also applies to ψ** (`LM:1579, 1670`). It pulls α toward 0.693, not 1. This is not stated
    elsewhere in this document.
11. **UNVERIFIED** (not checked in code):
    - bit-determinism and MPS invariance on H100;
    - the `_pre`-read pass rate's implementation for the pod tables;
    - 073 strict-eval internals;
    - arena A dev-target exclusion mechanics;
    - arena D TF32 setting and GPU-h;
    - TuringLLM BOS handling in its loader (no BOS logic found in `src/model/inference.py`).

### 0.9 Sufficiency to induce

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
- **Old names:** "free0 / freeM / freeN" are older names for Z / A / C. `eM_topk` / `eN_topk` / `e0` are their
  empty-circuit baselines.
- **Donor test:** bases are activating contexts. Its `silent_firing` on the 16 silent bases is a false-positive check
  (does the target fire with members at their live, silent-context values), not induction.
- **Arena A's `score_arena_a.py`:** computes Z / A / C / `nec_a` / `nec_b` and donor recovery only (lines 21-31). The
  induce read for arena A circuits comes only from the 062 driver's own eval rows, which `driver_tcz.py` runs.

**On the pod.** Every main and sweep scoring row on all three 082 tracks and arena A's driver rows carry it. `062
merge.py` reports its median (`experiments/062-h100-protocol-v1/merge.py:104, 115, 156, 196`). It is **not** in
`WE.passes` (`WE:1169-1179`), nor in the donor resample pass.

**Cost to add.**
- **To the pass rule:** no GPU cost. The field is already in every row, so it is a re-analysis (merge / `passes` /
  paper tables). It would be a pass-rule change after the protocol froze, so it needs the usual pre-statement (a
  threshold, e.g. ≥ 0.75), stated before results.
- **Caveats on this read as it stands:**
  - the members are injected position-uniformly at their anchor-mean, not position-specifically;
  - the rest of the stream is live, so the read measures drive in a natural silent context;
  - it does not measure "the circuit alone". A circuit-only induce (members injected on contrast rows with non-members
    at zero or the C sparse fill) does not exist.
- **Adding a circuit-only induce variant:** 1–2 extra no-grad forwards of 16 rows per scored row, against roughly 20
  forwards already in `score_circuit`. That is about +5–10% scoring time, and scoring is a small part of the pod
  (the fits dominate). This is an estimate, **UNVERIFIED**, and it would be a code change to `WE` plus the frozen 049
  scorer for TuringLLM.

---

# Protocol v1 (§1–§11) and the v2 / v3 changes (§12)

This document describes **what the production run executed**: the TuringLLM full run of
15,046 targets in `experiments/062-h100-protocol-v1/` (`out_full/`, finished 2026-09-27; source of every headline
number). Where older plans, the DAN-7 issue text, or `paper/main.tex` disagree with the code, the code is what is
recorded here. The disagreements are listed in §10 (drift) and §11 (paper vs code).

**Code pointers.** Every pointer is `path:line`. `src/` pointers are at commit `0b18c6e` (HEAD on `multi-device`).
The run started after `d5109bc`, and `src/` has not changed since, apart from additive DAN-78 edits that the run did
not use. `src/config.py` has uncommitted DAN-78 insertions after line 1451, so pointers into `DiscoveryConfig` are
to the committed file (`git show HEAD:src/config.py`). The run's config file was `config-h100-triamp.yaml`: the pod
bootstrap copies it over `config.yaml` (`scripts/pod_bootstrap_global.sh:28`).

**The as-run path.** The contexts are built and injected by experiment code, not by the production pipeline:
`experiments/059-context-pool/pool_test.py` and `experiments/059-context-pool/protocol_harness.py`, driven by
`experiments/062-h100-protocol-v1/driver.py`. The run did **not** use the later `src` ports, but both reproduce it:
- **Contexts:** `src/circuit/protocol_contexts.py` (`config.discovery.context_protocol = "v1"`, DAN-78). The 40-target
  rebuild is bit-identical locally.
- **Scorer:** the run used `experiments/049-circuit-graph/amp_eval_pass_v2.py`. The port is `src/eval/wcm_eval.py` +
  `wcm_specificity.py` (DAN-73), identical to the experiment scorer field for field on the same GPU
  (`experiments/067-src-scorer-equivalence/`).

---

## 1. Scope and status

| Item | Status |
|---|---|
| TuringLLM targets, contexts, fit, evaluation, pass rule, diagnostics (§2–§8) | **Frozen**: this is what the full run did |
| Random-circuit baseline, α = 1 re-scoring, unweighted sweep, λ grid (§8) | **Frozen as run**: separate runs, same engine and scorer |
| Pythia-70m and GPT-2 small adaptation (§9) | **OPEN**: DAN-9, DAN-34/35/36, DAN-40/41, DAN-42 |
| Production location of contexts (DAN-78) and scorer (DAN-73) | **Done** (2026-10-07 / 2026-10-09): both in `src/`, verified identical to the as-run code |
| Mid-band evaluation padding defect (§3.6, §10) | **Fixed and re-scored** (2026-10-09): 58 of 14,339 mid-band rows; the reported mid-band numbers are unchanged |
| Exact-duplicate contexts across the train / held-out split (§3.7) | **Measured** (2026-10-09): 25% of held-out strongest rows; effect on the pass rate −0.6 points on a paired 1,800-target re-score. **Superseded 2026-10-23** (exp 085): counting anchor-prefix copies and near-duplicates, 66.8% of held-out strongest rows repeat a training-side row; the 1,800-target re-score was incomplete. Fixed at the source in v2 (§12) |
| **Protocol v2** (§12) | **Decided 2026-10-25** (Daniel): A and C trained under the sparsity-preserving fill; TuringLLM refit on 8k targets from fresh, deduplicated records; public models run under v2. Validation (exp 093) pending |

The run's order of operations per target is contexts, then fit, then eval on held-out strongest, then eval on
held-out mid-band, then specificity (`driver.py:255-308`). Each step resumes from per-target files. Targets are
sharded `targets[i::k]` and interleaved shallow/deep per GPU slot (`driver.py:77-87, 353-361`); this affects
scheduling only.

## 2. Targets and sampling frame

| Step | Count | Code |
|---|---|---|
| Candidates: `stratified_random`, a uniform draw of 455 latents per component (`torch.randperm`, without replacement) over all 36 components (12 layers × {attn, mlp, resid}) | 16,380 | `src/circuit/feature_selection.py:342-358`; `outputs/candidates.pt` (built 2026-08-31; 455 per `comp_idx`, verified) |
| Drop layer-0 attention: it has no upstream SAE site | 15,925 | `config-h100-triamp.yaml:259` (`skip_no_upstream`); `src/eval/ablation_faithfulness.py:369-381` |
| Keep the targets that produced a circuit in the **pre-v1** production run (045/049). It dropped 879 attention seeds: dead latents, or fewer than 8 positives | **15,046** | `experiments/062-h100-protocol-v1/make_targets.py:26-28, 42-43` (reads `049-circuit-graph/tables_full/circuits.parquet`); `049-circuit-graph/README.md` §0 |
| v1 skips: fewer than 8 strongest contexts (`too_few_contexts`, 23), or fewer than 4 verified-silent contrast contexts (`no_contrast`, 19) | 42 skipped | `pool_test.py:127-130, 136-139, 149-152`; `driver.py:272-275` |
| Fitted circuits | **15,004** | `out_full/out/summary.md` |
| Vacuous denominator, excluded from pass rates (§6) | 10 | `merge.py:57-58, 64-65` |

- **Per-cell counts.** Every mlp and resid cell has 455 targets. Attention cells are thinned by the pre-v1 survival
  step: L1 134, L2 299, L3 406, L4 450, L5 357, L6 429, L7 383, L8 346, L9 447, L10 436, L11 439. Computed from
  `targets_full.txt`.
- **Selection.** There is no prioritisation heuristic. The frame is, however, conditioned on surviving the pre-v1
  run, which removed attention latents that had no stored contexts or fewer than 8 positives.
- **Subsets.**
  - Stage 1 (calibration, DAN-8): 40 per cell plus the 16 pilots, 1,416 targets.
  - Sweep: 5 per cell plus the 16 pilots, 191 targets (190 with usable contexts).
  - Code: `make_targets.py:31-39`, with RNG seed 0.
- **Cross-model frame:** OPEN (DAN-9).

## 3. Contexts

All contexts are built once per target by `pool_test.build` (`pool_test.py:75-164`) and cached in `out/ctx/<key>.pt`.
`driver.Runner.contexts` calls the builder with the original selector and probe builder restored
(`driver.py:125-135`).

### 3.1 Activating pools (strongest and mid-band)

| Setting | Value | Code |
|---|---|---|
| Strongest pool | Sequence ids from the top-context store row (the 64 strongest stored activating sequences). Ids ≤ 0 are dropped and ids are deduplicated; the pool is capped at the first 64 | `pool_test.py:118-121, 127` |
| Mid-band pool | Sequence ids from the mid-context store (a reservoir sample in μ + [0.5, 1.5] σ), deduplicated against the strongest pool; the pool is capped at 64 | `pool_test.py:122-127` |
| Thin rule | A pool with **fewer than 8** ids is absent (`None`). No strongest pool means the target is skipped. No mid-band pool means the target is "thin" and uses training arm A | `pool_test.py:128-130`; `driver.py:137-139` |
| Tokens | 64 corpus tokens per context. The shards store each sequence as 65 elements, [source-file index, 64 tokens], between −1 separators; the loader's shard index skips the first element (`skip_first_token=True`, the default, and the only index built: `data/.shard_indices/*_sft1.idx.npy`), so the file index never reaches the model. The ids are read into a 65-wide buffer (`max_length=65`) whose last column is zero padding. The context is columns 0-63, all 64 real tokens. The next-token targets (columns 1-64) are tokens 2-64 plus one padding zero; only the `logit` objective reads them, never the WCM fit (`objective="pos"`) or the scorer | `pool_test.py:85-95`; `src/data/loader.py:65-99, 554-571`; verified on `shard_0.npy` 2026-10-10 |
| Anchor | The argmax over the 64 positions of the target's post-Top-K activation, from a fresh forward pass | `pool_test.py:97-113, 132` |
| Peak | The max of the same quantity. It ranks the contexts for the split | `pool_test.py:104` |

As run, 14,277 targets have full 64/64 pools. 62 arm-B targets have a mid-band pool of 8 to 63 contexts. 478
targets have fewer than 64 strongest contexts, and all of them are thin. These figures are from
`main/contexts.shard*.jsonl`.

### 3.2 Split (stratified, deterministic)

`stratified_split(scores, holdout_frac=0.25, keep_top=2, offset=keep_top)` (`src/circuit/context_split.py:22-47`)
works as follows:

1. Rank the contexts by peak.
2. Cut the ranking into blocks of m = 4.
3. Block b holds out rank `b·4 + (b + 2) mod 4`, except the top 2 ranks, which always train.
4. Trim or top up so that exactly `round(n/4)` contexts are held out, taking or giving from the weak end.

There is no RNG. The rule is applied independently to each pool (`pool_test.py:133`). For a 64-context pool the
result is 48 train and 16 held out.

### 3.3 Training set

| Arm | When | Training contexts | Code |
|---|---|---|---|
| **B** (protocol) | A mid-band pool exists (14,339 targets) | **32 strongest**: a stratified two-thirds of strongest-train (`stratified_split(..., holdout_frac=1/3, keep_top=2)`, blocks of 3). **16 mid-band**: `stratified_subset(mid-train peaks, 16)`, phase 0.5, spread evenly over the mid-train ranking | `pool_test.py:141-145, 234-235`; `context_split.py:57-67` |
| **A** (thin fallback) | No mid-band pool (665 targets) | All of strongest-train (48 when the pool is full: 187 targets; 6 to 47 otherwise: 478 targets) | `pool_test.py:229-230` |

In arm B, 16 strongest-train contexts are neither trained on nor held out. They still enter evaluation, because the
evaluation reference is the full 48 strongest-train (§5.1). As run, 14,502 targets train on 48 contexts, and 502
train on 6 to 47 (24 arm-B targets whose mid-train has fewer than 16 contexts, plus the 478 short-pool arm-A
targets).

### 3.4 Contrast contexts (close selector, verified silent)

The call is `NegContextSelector.select(comp, i, "close", max_sequences=64, batch_size=16, exact=False,
non_activation_threshold=0.0, filter_batch_size=32, load_window_size=256)` (`pool_test.py:147-148`).

- **Candidate pool.** The union of all sequence ids in the `neg_ctx` store
  (`src/utils/neg_context_selector.py:274-301`), minus the target's own top- and mid-store ids (`:321-323, 917-919`).
- **Ranking.** Each candidate gets its max cosine similarity, in mean-pooled final-layer `seq_repr` space, to the
  representations of the target's **top-context store** ids (`:455-510, 1053-1061`). Top ids absent from `seq_repr`
  are forwarded. Mid-band contexts are not in the reference set. Only the 8,192 most similar candidates
  (`max(64·128, 2048)`) are kept, in rank order (`:959-970, 972-`).
- **Silence check.** Candidates are scanned in rank order. A candidate is kept if the target's max post-Top-K
  activation over 64 tokens is ≤ 0, that is, the target is outside its site's Top-K at every position. The scan
  stops at 64 kept (`:743-903`, test at `:842`).
- **Unusable target.** Fewer than 4 kept means the target is skipped (`pool_test.py:149-152`).
- **Order and split.** Kept contexts arrive in similarity order. `stratified_order(-arange(n))` reorders them so that
  the stratified 48 training contexts come first and the 16 held-out contexts last (`pool_test.py:153-157`;
  `context_split.py:50-54`). Every consumer slices `[: n - round(n/4)]`. As run, 14,994 targets have 48/16; 10 have
  fewer contexts.

### 3.5 How contexts reach the engine (injection)

- **Fit.** `P.probe(rec, train, [strong-held, mid-held])` builds a `ProbeDataset` whose first `n_train` rows are the
  training set. It is padded with held-out contexts to the length n at which `n - round(n/4) == n_train`
  (`pool_test.py:247-262`; `driver.py:155-159`). The method's `build_probe_dataset` and `_floor_negatives` are
  replaced by lambdas (`driver.py:158-159`). For a 48-context training set the pad is the 16 held-out strongest
  contexts. The engine uses them only for the reported `holdout_data_loss`.
- **Evaluation and specificity.** `protocol_harness.patch_eval_contexts(G, rec, held)` hands every consumer the
  **48 strongest-train contexts followed by the chosen held-out set** (16 strongest, or 16 mid-band), plus the
  contrast set through a `FixedSelector` (`protocol_harness.py:80-89`; `pool_test.py:318-324`). `CTR_SOURCE=close`
  is the default (`driver.py:112`), so the scorer re-reads this fixed list (`amp_eval_pass_v2.py:372-376`).
- **Manifest.** The corpus ids and anchors of every set are written to `main/contexts.shard*.jsonl`
  (`driver.py:218-252`).

### 3.6 Known defect: mid-band padding (fixed 2026-10-09)

When the mid-band held-out set has fewer contexts than the strongest-train set needs as padding, the probe has fewer
than 64 rows. The scorer's `split_n` then moves part of strongest-train into the "held-out" slice
(`pool_test.py:257-259` against `amp_eval_pass_v2.py:380-381`). As run this affects **58 mid-band rows**: the
held-out slice holds 1 to 11 strongest-train contexts, and the A mean and pins use fewer than 48. A further 53
mid-band rows have fewer than 16 held-out contexts. Held-out strongest rows are unaffected (the pad always matches
by construction).

**Fix.** The scorer takes `EVAL_SPLIT` = `as_run` (default; reproduces the 15k run) or `explicit`. `explicit` takes
the train count from the probe's `metadata["n_train"]`, so the held-out slice is exactly the held-out set. The src
scorer (`src/eval/wcm_eval.py`, DAN-73) defaults to `explicit`. **Every new run uses `explicit`.**

**Re-score.** The 58 rows were re-scored with `explicit` (`062/rescore_mid.py` → `out_rescore_mid/eval_mid.jsonl`; `out_full`
untouched). All 58 now train on 48 and hold out exactly the mid-band held set. One row flips fail → pass. Over all
mid-band rows the pass rate goes 12.09% → 12.10% and the median worst-of-3 stays 0.774, so the reported 12% and 0.77
stand.

### 3.7 Exact-duplicate contexts across the split (measured 2026-10-09)

The corpus repeats boilerplate windows. 26.5% of corpus windows are exact duplicates; in the strongest-context
pools, 22.9% of rows duplicate another row of the same pool (97.8% of pools contain one). The mid-band store is
almost free of them (0.2%). The split does not deduplicate (§3.2), so identical token windows can fall on both
sides (`062/dup_leakage.py`, `out_dup_leakage/leakage.csv`, all 14,994 records):

- 25.1% of held-out strongest rows (59,063 / 235,051) are token-identical to a row the fit trained on; 92.2% of
  targets have at least one (median 4 of 16).
- 35.6% are identical to a row of the 48 strongest-train evaluation reference (A means, pins).

**Effect.** Between targets, those with leaked rows pass less often (44.7% vs 65.5%; −8.3 points within band ×
kind cells), but that compares different latents. The paired test re-scores the same circuits on held-out strongest
with every leaked or duplicated row removed, against the unchanged reference (`062/rescore_dedup.py`,
`out_rescore_dedup/`; 1,799 targets, 200 per band × kind cell, median 11 clean rows of 16). The pass rate goes
41.2% → 40.6% (42 pass → fail, 31 fail → pass); the median worst-of-3 goes 0.363 → 0.370. The "all rows" side is
the stored H100 row and the "clean" side an RTX re-score, so part of the 73 flips is hardware noise (DAN-73/74).

**For new runs:** public-model corpora are deduplicated by window before the stores are built
(`069/prepare_corpus.py`). Whether to also deduplicate each pool before the split, and how the paper words
"held-out", are open decisions (DAN-7).

## 4. Fit: weighted circuit masking, WCM[Z+C+A]

Entry point: `_build_mode_method("ablation_gradient", "mask", ...)` (`driver.py:142-147`;
`src/analysis/circuits/gradient_size_sweep_runner.py:480-506`). This calls
`AblationGradientDiscovery._run_mask_hop` (`src/circuit/discovery/ablation_gradient.py:242-334`), which calls
`run_learned_mask(objective="pos", ...)` (`src/circuit/instrument/learned_mask.py:478-1948`).

### 4.1 Masked forward

At every upstream site s (all sites at lower layers, plus earlier kinds in the same layer in the order
attn → mlp → resid; `ablation_faithfulness.py:369-381`), the site's Top-K code `c` is **re-encoded from the current
(already edited) stream**. The stream is then updated as

`x ← x + W_dec^s [ m·α·c + (1 − m)·v^π − c ]`

so the SAE error is preserved (`learned_mask.py:275-317, 347-348`). The terms are:

- `m = σ(θ/T)`, the annealed gate (`:296-297`).
- `α = softplus(ψ)`, the coefficient (`:137-140, 312`).
- `v^π`, the ablation value: zero for Z, or a **dense** per-latent mean for C and A (`:316-317`). The mean is taken
  over all (sequence, position) slots, zeros included (`src/eval/floors.py:122, 157-166`). No Top-K is re-imposed
  during training.
- The target's site passes through untouched. Its pre-activation `w·x + b` is tapped (`:257-270`).

### 4.2 Objective

Three ablation terms share one set of θ and ψ; each term has its own patcher (`learned_mask.py:1161-1191`). Per
step:

```
L = Σ_π (γ_π / ν) · [ mean_b (pre_π(b) − target(b))²  +  γ_R · mean_b relu(τ_π^(r_b) − pre_π(b))² ]
    + λ Σ_{s,i} σ(θ_{s,i})  +  λ Σ_{s,i} (1 − σ(θ_{s,i})) · |α_{s,i} − 1|
```

| Symbol | Meaning (as run) | Code |
|---|---|---|
| π ∈ {Z, C, A} | Z zero fill; C mean over the **training contrast contexts** (the first 48 of the stratified order); A mean over the **training activating contexts** (the actual training set: 32 strongest + 16 mid in arm B) | `learned_mask.py:788-823` (`floors_train_only`); `protocol_harness.py:36-39` |
| γ_Z, γ_C, γ_A | 1, 0.25, 0.25 (`dual_floor_weight` = C, `triple_floor_weight` = A), at every layer | `learned_mask.py:1501-1513`; `driver.py:47`; `protocol_harness.py:38` |
| target(b) | The natural pre-activation at the anchor of training context b, from an unedited forward pass ("reproduce, don't maximise") | `learned_mask.py:1102-1104, 1114-1116` |
| ν | `mean(target²)` over the training anchors. One shared, bounded normaliser | `learned_mask.py:1338-1356` |
| pre_π(b) | The target's pre-activation at the anchor under the π-masked forward | `learned_mask.py:1258-1259` |
| Rank-keep | `rank_mode="keep"`, γ_R = 3e-3, `rank_delta` 0. r_b = the target's rank among the site's **other** latents by pre-activation at training anchor b in the unmodified model, **clamped to [1, k]**. τ^(r) = the r-th largest other pre-activation under the π-masked forward. It sits inside each term, so it carries γ_π/ν | `learned_mask.py:1286-1312, 1404-1412`; `driver.py:51, 145`; `protocol_harness.py:39` |
| λ | 1e-3 per latent; the penalty is a flat **sum** over latents (not a mean); `site_lambda_weights` is not used | `learned_mask.py:1210-1221, 1583`; `driver.py:47` |
| Leak guard | λ·(1 − σ(θ))·\|α − 1\|, charged so that a closed gate cannot carry an inflated α. It uses σ(θ), **not** σ(θ/T). `amp_l1` = 0, so members' α is unpriced | `learned_mask.py:1597-1612`; `src/config.py:811` |
| Off-target term | Off (`offtarget_weight` 0) | `protocol_harness.py:38` |

### 4.3 Hyperparameters

| Setting | Value | Set at | Default at |
|---|---|---|---|
| `mask_floor_source` | `triple` (Z + C + A) | `protocol_harness.py:36`; `config-h100-triamp.yaml:253` | `src/config.py:790` (`dual`) |
| `free_amplitude` | True (WCM); False (unweighted arms, §8.3) | `protocol_harness.py:36`; `driver.py:262, 318` | `src/config.py:808` |
| `l1_lambda` λ | 1e-3 (main) | `driver.py:47`; `protocol_harness.py:36` | `src/config.py:712` (1e-4) |
| `dual_floor_weight` γ_C, `triple_floor_weight` γ_A | 0.25, 0.25 | `protocol_harness.py:38` | `src/config.py:801, 805` |
| `rank_weight`, `rank_mode`, `rank_delta` | 3e-3, `keep`, 0 | `protocol_harness.py:39`; `driver.py:145` | `src/config.py:827-832` |
| `offtarget_weight` | 0 | `protocol_harness.py:38` | `src/config.py:817` |
| `floors_train_only` | True | `protocol_harness.py:39` | `src/config.py:838` |
| `steps` | 400 | `driver.py:50`; `protocol_harness.py:39` | `src/config.py:679` |
| Optimiser | AdamW over θ and ψ, lr 0.05 constant (`lr_schedule="constant"`), weight decay 0.05 (steps × lr × wd = 1.0) | `learned_mask.py:1078-1086` | `src/config.py:682, 748, 756, 772` |
| Batch | 4 contexts per step (`probe_batch_size=4`, set by the scorer's `setup`); no gradient accumulation (`deep_site_threshold` 99 > 35 sites) | `amp_eval_pass_v2.py:130`; `protocol_harness.py:37`; `learned_mask.py:833-842` | `src/config.py:735` (21) |
| Batch order | Fixed and sequential: step t uses training contexts `[(4t) mod n_train, +4)`; no shuffling; 12 steps per pass over 48 | `learned_mask.py:1240-1247, 1486` | |
| Gate init | θ₀ = 4 (m ≈ 0.98) at every latent of every upstream site (`theta_init_mode="uniform"`) | `learned_mask.py:926-930` | `src/config.py:725` |
| Coefficient init | ψ₀ = softplus⁻¹(1), so α₀ = 1 | `learned_mask.py:997-1001` | |
| Binarisation | `binarize="anneal"`: m = σ(θ/T), T = 0.05^(step/(S−1)), from 1.0 at the first step to 0.05 at the last | `learned_mask.py:1456-1472` | `src/config.py:878`; not overridden by the yaml, `configure()` or `setup()` |
| Keep threshold | 0.5 (nodes = {σ(θ) > 0.5} ⇔ θ > 0, the same set as σ(θ/T) > 0.5) | `learned_mask.py:1792-1797` | `src/config.py:723` |
| Engine split | `holdout_frac` 0.25: train on `[:n − round(n/4)]` (exactly the training set, given the §3.5 padding) | `learned_mask.py:1088-1093` | `src/config.py:724` |
| `probe_sequence_count`, `eval_sequence_count` | 128 (so the padded 64 rows are never truncated) | `protocol_harness.py:41`; `gradient_base.py:282-285` | `HEAD:src/config.py:1484-1487` (16) |
| `floor_negctx_mode` | `close`; inert, because `_floor_negatives` is injected | `protocol_harness.py:40`; `driver.py:159` | `HEAD:src/config.py:1530` |
| Code dtype | `stream` (model dtype); `autocast_bf16` off | `src/config.py:760` | |

### 4.4 Stopping, selection, acceptance

- **Stopping.** A fixed 400 steps. There is no early stopping and no checkpoint selection. `holdout_data_loss` is
  computed on the padded held-out rows and reported only (`learned_mask.py:1455, 1687-1700`).
- **Output.** The output is the set {latent : σ(θ) > 0.5} at the final step, with its converged α stored as node
  metadata `amplitude` (`learned_mask.py:1725-1738, 1792-1797`; `ablation_gradient.py:331`).
- **No post-hoc filtering.** `support_threshold` 0.01 < 0.5 admits every member (`gradient_base.py:532-533`). There
  is no pruning (`pruning_threshold` 0, `config-h100-triamp.yaml:196`; `magnitude_prune` and `recurrence_prune`
  off).
- **No rejection.** Mask modes are never threshold-rejected (`gradient_base.py:615-623`). Only an empty node set is
  rejected; none was in the run.

## 5. Evaluation

The scorer is `score_circuit(c, {}, skip_roles=True)` (`driver.py:174-176`; `amp_eval_pass_v2.py:339-613`). It runs
twice per circuit: held-out strongest (**primary**) and held-out mid-band (**reported**) (`driver.py:286-291`). The
sweep and baselines score held-out strongest only.

### 5.1 Common definitions

| Item | As run | Code |
|---|---|---|
| Activating rows | `pt` = 48 strongest-train followed by the held-out set (≤ 64 rows). The train slice is `[:n − round(n/4)]` (48) and the held-out slice is the rest (16) | `amp_eval_pass_v2.py:366-381`; `protocol_harness.py:86` |
| Contrast rows | The target's 64 close contrast contexts. Train = first 48, held-out = last 16 | `amp_eval_pass_v2.py:372-382` |
| Anchors | Activating rows: the §3.1 argmax. Contrast rows: the argmax of the target's natural pre-activation per sequence | `amp_eval_pass_v2.py:441` |
| Reads | `_tk` = the target's post-Top-K activation at the anchor (**PRIMARY**, every pass/fail number). `_pre` = relu(w·x + b) (**diagnostic**). A ratio never mixes the two reads | `amp_eval_pass_v2.py:138-147, 295-317, 453-458` |
| a_pos | Mean natural `_tk` (`_pre`) over the **held-out** activating rows of the set being scored (strongest or mid-band) | `amp_eval_pass_v2.py:414-416` |
| A mean (`means_tr`) | Dense per-latent mean over all positions of the **48 strongest-train** rows. In arm B this is not the fit's training set | `amp_eval_pass_v2.py:420`; `floors.py:122` |
| C mean (`means_neg`) | Dense per-latent mean over the 48 contrast-train rows; the same contexts as the fit | `amp_eval_pass_v2.py:421` |
| Pins (`pins_tr`) | Per-latent mean dense value **at the anchors** of the 48 strongest-train rows (collapsed) | `amp_eval_pass_v2.py:420`; `floors.py:127-135` |
| Circuit-only run | `CircuitOnlyPatcher`, with every upstream site in scope. Members are kept at α × their **re-encoded live value**; every other latent at every upstream site is set to the fill; the SAE error is preserved | `ablation_faithfulness.py:137-265`; `amp_eval_pass_v2.py:396-406` |
| Fill: zero | All non-members are 0 | `ablation_faithfulness.py:239-243` |
| Fill: dense | All non-members are at their mean | `ablation_faithfulness.py:239-245` |
| Fill: sparsity-preserving (`_topk`) | Per position, members at their values; the non-members with the largest means fill the remaining budget `k − #active members` at their mean; everything else is 0 (k = 128) | `ablation_faithfulness.py:267-297` |
| Empty baselines | Circuit-only runs with no members, on the held-out rows: `e0`, `eM_dense`, `eM_topk`, `eN_dense`, `eN_topk` (all recorded); `e0_tr` on the train rows | `amp_eval_pass_v2.py:424-427` |
| Live-stream edit | `CounterfactualInterventionPatcher`: the named latents are SET to values (or 0) at every position; all other latents are re-encoded live; the error is preserved | `src/eval/counterfactual_faithfulness.py:94-` |

### 5.2 Scores

The ratio form is `ratio(x, e) = (x − e)/(a_pos − e)`. It is `None` when `e ≥ a_pos` under that read
(`amp_eval_pass_v2.py:320-321, 453-458`). All scores are unclipped.

| Score (field) | Definition | Contexts | Baseline | Use |
|---|---|---|---|---|
| **Z** `free0_tk` | ratio(circuit-only, zero fill, fitted α) | held-out activating | `e0` | **pass rule** |
| **A** `freeM_topk_tk` | ratio(circuit-only, sparsity-preserving A-mean fill, fitted α) | held-out activating | `eM_topk` | **pass rule** |
| **C** `freeN_topk_tk` | ratio(circuit-only, sparsity-preserving C-mean fill, fitted α) | held-out activating | `eN_topk` | **pass rule** |
| `freeM_dense`, `freeN_dense` | As A and C with dense fill | held-out activating | `eM_dense`, `eN_dense` | recorded |
| `free0_tr` | Z on the train rows | 48 strongest-train | `e0_tr`, `a_pos_tr` | recorded |
| `free0_a1` | Z with every α = 1 | held-out activating | `e0` | recorded (§8.2) |
| `free0_perm` | Z with α permuted within each site (per-target RNG) | held-out activating | `e0` | recorded |
| `free0_rand` | **The empty circuit**: the live pool is `{}` in the driver, so no random set is drawn. Not a baseline | n/a | `e0` | ignore (`driver.py:176`) |
| **Necessity** `phi_sup_blind_tk` | (a_pos − a_{M∖C}) / (a_pos − e0). a_{M∖C} = the target when **every member is set to 0** in the live stream (all other latents recomputed). Zero ablation only; there is no per-π variant | held-out activating | `e0` | **pass rule** (≥ 0.9) |
| `phi_sup_ctr_tk` | The retired form (a_pos − x)/(a_pos − A⁻) | held-out activating | A⁻ | recorded |
| **Sufficiency to induce** `phi_cf_alpha_blind_tk` (φ_ind^α) | (A⁻_int − A⁻)/(A⁺ − A⁻). Every member is SET to α × its pin at every position; A⁻ = natural target at the contrast anchors; A⁺ = a_pos | held-out contrast (16) | A⁻ (`a_base`) | reported, not required |
| `phi_cf_a1_role_tk` | The same at α = 1. With `skip_roles` every member counts as an activator, so this is role-blind φ_ind | held-out contrast | A⁻ | recorded |
| `phi_sup_alpha` | Members scaled by max(0, 1 − α) in the live stream | held-out activating | `e0` | recorded |
| `phi_pin_alpha_{zero,topk,dense}` | Circuit-only with members **clamped to α × their clean position-wise value**; fills Z, sparsity-preserving A, dense A (no C variant) | held-out activating (≤ 16) | `e0`, `eM_topk`, `eM_dense` | recorded |
| Roles, `phi_sup_role`, `release` | **Not computed**: `skip_roles=True`, so the attribution is all zeros, there are no inhibitors, `phi_sup_role` equals `phi_sup_blind`, and `release` = 0 | n/a | n/a | ignore (DAN-72) |
| `vacuous_tk` | \|a_pos − e0\| < 0.05·a_pos (activation read, Z denominator only) | held-out activating | `e0` | exclusion (§6) |

Every row also records the raw reads (`*_raw_tk`, `*_raw_pre`), the per-block a_pos, the n's, and
`all_sites_ablated` (a scope check; it must be true) (`amp_eval_pass_v2.py:465-478, 599-612`).

## 6. Pass rule, vacuous exclusion, near-threshold stratum

| Rule | As run | Code |
|---|---|---|
| **Pass (DAN-8, adopted 2026-09-27 on stage 1)** | Z, A, C (`free0_tk`, `freeM_topk_tk`, `freeN_topk_tk`) all in **[0.8, 1.5]** AND `phi_sup_blind_tk` **≥ 0.9**, on held-out strongest, activation read. A missing (`None`) score fails | `merge.py:28-30, 50-54`; `pass_rule.py:73-75` |
| Vacuous exclusion | Rows with `vacuous_tk` are dropped from the denominator and counted (10 strongest rows, 23 mid-band rows). An undefined A or C ratio is **not** vacuous: it counts as a fail (**superseded by §12.4**, 2026-11-01: every pass-rule baseline, signed) | `merge.py:57-58, 64-65`; `amp_eval_pass_v2.py:468-469` |
| Graded companion | worst-of-3 = max over π of \|φ_π − 1\|, reported alongside the rule | `merge.py:66, 75-76` |
| Near-threshold stratum | `rank_clean ≥ K_TOPK/2 = 64`. `rank_clean` = the median over the 16 held-out strongest anchors of the target's rank among **all** site latents by pre-activation in the clean model (1 = top; uncapped). It is computed in the specificity pass from the clean model only. The stratum stays in the headline rate and is reported separately on both reads (1,142 targets) | `merge.py:32, 137-148`; `specificity.py:203, 209-210, 230`; `pass_rule.py:79-81` |

## 7. Reported diagnostics

### 7.1 Specificity and the amplifier flag (056)

`specificity.score` is called with the strongest eval injection (`driver.py:179-184`;
`experiments/056-specificity/specificity.py:154-242`).

- **Runs.** Per π ∈ {Z, A, C} there are clean, empty and circuit runs. A and C use the sparsity-preserving fill.
  The fitted α is applied. Everything is read on the 16 held-out strongest anchors (`specificity.py:176-192`).
- **Siblings.** Latents in the clean Top-K at the anchor on ≥ 50% of the **48 strongest-train** contexts
  (`SIB_FRAC` 0.5; `:195-201`).
- **Control.** Other latents active in the clean held-out Top-K (`:203-205`).
- **Per-latent faithfulness.** (relu c − relu e)/(relu n − relu e) on pre-activations, averaged over anchors. NaN
  when the denominator is < 0.05·clean (`MIN_DEN`; `:145-151`).
- **Per-π outputs.** Rank, Top-K share, Jaccard with the clean Top-K, `switched_on` (latents lifted above the clean
  Top-K cut, circuit against empty) (`:213-241`).
- **Amplifier flag** (per π): `target_faith_pre ≥ 0.5` AND `sibling_faith_median ≥ 0.8 × target_faith_pre`.
  `amp_any` = the flag under any π (`merge.py:134-137`). Run result: 29.6% of circuits.

### 7.2 Mid-band transfer

The same scorer runs on the 16 held-out mid-band contexts, with the same 48 strongest-train reference. a_pos is then
the mid-band mean (`driver.py:286-291`). It is reported, not used in the pass rule; run result: 12.1% pass
(unchanged after re-scoring the 58 padded rows, §3.6).

### 7.3 Training curves

Every fresh fit writes `train.shard*.jsonl` (`driver.py:187-211`). Each row holds the fit stats (`loss_initial`,
`loss_final`, `holdout_data_loss`, `n_kept`, `mean_m_kept`, amplitude stats, `n_train_pos`/`n_holdout_pos`) and the
per-step curve: total loss, each weighted term including its rank part, penalty, rank, member count at the keep
threshold, and mean gate (`learned_mask.py:1443-1451, 1643-1678`). Figures: `curves.py`.

## 8. Baselines and controls

### 8.1 Random circuits with fitted coefficients (DAN-17)

`random_null.py`, as quoted in the paper: `OUT_RN=out_random_scale`, 747 targets × **1 draw**, stratified 249 per
site kind over depth band × kind (`random_null.py:96-113`; `out_random_scale/run.log`). An earlier pilot used 63
targets × 2 draws (`out_random/`).

- **Size.** Matched site by site to the production circuit. When a site has too few live latents, all of them are
  taken (`short_sites` is recorded) (`random_null.py:58-66, 150-156`).
- **Members.** Drawn uniformly without replacement from latents with a nonzero post-Top-K activation at any position
  of the **48 strongest-train contexts**. These are not the arm-B training set (`random_null.py:69-89, 148-151`).
- **Fit.** The same `Runner.fit` path and production config, with the engine call wrapped: `support` = the random
  members (others clamped at θ = −12), `l1_lambda` = 0, `binarize="none"`, `theta_init` = 40. Only α moves
  (`random_null.py:44-55`).
- **Scoring.** The production scorer on held-out strongest, with the DAN-8 rule (`random_null.py:164-166, 186`).

### 8.2 α = 1 re-scoring (DAN-20)

`rescore_alpha1.py`: 50 targets per (layer, kind) cell (seed 20261002), 1,750 rescored, 1,746 paired after the
vacuous exclusion (`rescore_alpha1.py:31-42`). The scorer runs with `override_alphas` set to 1 for every member, on
held-out strongest (`:45-54, 79-81`). It is paired with the run's fitted-α rows (`:97-137`).

### 8.3 Unweighted circuit masking (the sweep's `U_` arms)

`MODE=sweep` builds `R.method(γ, λ, weighted=False)`, which gives `free_amplitude=False`: no ψ and no leak guard.
Everything else is identical: contexts, triple floor, γ 0.25, rank-keep 3e-3, anneal, 400 steps (`driver.py:52-53,
311-350`). Nodes carry no amplitude, so the scorer applies α = 1 (`amp_eval_pass_v2.py:362`). Scoring is on held-out
strongest only.

### 8.4 λ grid

| Arm | λ | Targets | Code / output |
|---|---|---|---|
| WCM `W_` | 2.5e-4, 5e-4, **1e-3** (reuses the main circuit), 2e-3, 4e-3 | 190 | `driver.py:48, 330-333`; `out_full/out/sweep/` |
| Unweighted `U_` | 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 | 190 | `driver.py:49`; `out_full/out/sweep/` |
| WCM, weak penalties (paper Fig. 3) | 1e-4, 5e-5, 2.5e-5 | 190 (union of `out_bands` 32, `out_depth` 32, `out_lowlam` 128) | `run_lowlam.sh`; `queue_local.sh`; `SWEEP_ARMS` (`driver.py:58-68`) |
| Depth test | `W_0.001_s800` (800 steps) | 32 | `out_depth/` |

## 9. Per-model adaptation (Pythia-70m, GPT-2 small): decided (§9.1, superseded where they overlap by §9.2)

All of the following must be decided before the public-model runs (DAN-40/41) are reported.

1. **k-dependent settings.** Pythia uses k = 16 and GPT-2 uses k = 32, against 128 on TuringLLM. The following read
   k:
   - the near-threshold cut K/2 (`merge.py:32` hard-codes 128; `pass_rule.py:19`);
   - the rank-keep clamp r ≤ k (`learned_mask.py:1412`);
   - the sparsity-preserving fill budget (`amp_eval_pass_v2.py:404-406` via `G["K"]`; `specificity.py:190`).

   The two hard-coded `K_TOPK = 128` / `K = 128` must become per-model.
2. **λ calibration (DAN-42).** λ = 1e-3 is a per-latent price calibrated on TuringLLM (d_sae 40,960, up to 35
   upstream sites). The public models have d_sae 32,768 and differ in sites and depth. Decide whether to keep 1e-3,
   rescale it, or calibrate it with one probe per target (`src/config.py:683-711`).
3. **Sampling frame (DAN-9).** The TuringLLM frame is 455 per component, conditioned on pre-v1 survival (§2). The
   public-model frame must name the per-component quota, the layer-0 handling, and whether any survival
   conditioning applies, stated once for all three models.
4. **Adapters (DAN-34/35/36).** These cover the model and dictionary adapters. They include GPT-2's four sites per
   layer (attn out, resid-mid, MLP out, resid; note the 053 resid-mid bug) and the layer-normalised SAE input, which
   must reproduce `upstream_sites` ordering and the error-preserving edit.
5. **Context stores per model.** The top and mid stores, the `neg_ctx` candidate pool and `seq_repr` must exist for
   each model (the close selector requires `seq_repr`). Decide the store sizes (64 / 64), the sequence length (64),
   and the `<8` and `<4` rules.
6. **Unchanged unless decided.** γ (0.25 / 0.25), γ_R (3e-3), steps (400), lr, wd, batch, anneal, the split rule,
   the pass rule.
7. **Code location.** The runs should go through `src/circuit/protocol_contexts.py` (DAN-78) and the `src/` scorer
   (DAN-73). Before that, verify that they reproduce 062 on TuringLLM.

### 9.1 Decided (2026-10-07 to 2026-10-10)

| Item | Decision | Issue |
|---|---|---|
| Targets per model | Pythia-70m: 64 per eligible component × 16 = 1,024. Its parallel residual leaves attn(0) and mlp(0) with no upstream site. GPT-2 small: 22 per eligible site × 47 = 1,034 (layer-0 attn-out excluded). No survival conditioning | DAN-9 |
| λ | A one-probe calibration pilot per model (~300 fits) before the main runs | DAN-42 |
| Corpora | `monology/pile-uncopyrighted` (Pythia; its SAEs were trained on the Pile at context 2049) and OpenWebText (GPT-2; its SAEs name `apollo-research/Skylion007-openwebtext-tokenizer-gpt2`, context 64, no BOS); 10M 64-token sequences per model; corpus windows deduplicated | DAN-37 |
| Duplicates | Context pools deduplicated by token window before the train / held-out split (§3.7) | DAN-37, DAN-9 |
| Precision | bf16 for the data pass (it only ranks contexts); float32 for fits and scoring (the precision the adapters and gates were validated in) | DAN-37 |
| Engine | Unchanged from the 15k run, including σ(θ) pricing in the sparsity penalty and leak guard | DAN-36 |
| Contrast contexts | **close, post-Top-K silent**, as in the 15k run, so the model is the only change in the cross-model comparison. The DAN-64 sweep found random or distant contrast sets give more robust circuits (+0.10 pass on distant backgrounds); this is reported as a finding, not adopted | DAN-64 |
| Adapters and scorer | `src/model/hf_adapters.py`, `src/sae/public_bank.py` (DAN-34/35; GPT-2 resid-mid edited inside the attention hook); `src/eval/wcm_eval.py` with per-model K and `split="explicit"` (DAN-73) | DAN-34/35/73 |
| Data release | Corpus sequence ids and anchors only, never corpus text | DAN-63 |

### 9.2 Decided 2026-10-23 to 2026-10-24 (supersedes the matching §9.1 rows)

Fixed before any public-model fit. The run is planned for 2026-10-25; the code is in `experiments/082-public-runs/`
(chain.sh, make_targets_public.py, pilot_lambda.py), `084-harder-nulls/` and `086-heldout-rebuild/`.

| Item | Decision | Issue |
|---|---|---|
| Targets per model | **512**: Pythia-70m 32 per eligible component × 16; GPT-2 small ~11 per eligible site over 47 (layer-0 attn-out excluded). Seeded draw, seed `20261024`; lists md5-frozen before any fit | DAN-9 |
| Eligibility (the frame rule) | Uniform over latents with **≥ 200 distinct firing 64-token windows** in the scanned corpus, and a **hub cap**: latents firing in **> 50% of windows** are excluded. Counts excluded by each rule are reported per component. For comparison, 263 of TuringLLM's 15,046 targets (1.75%) sit above the same cap and 640 (4.25%) have fewer than 200 firing windows | DAN-9 |
| Subsets | Pilot 30 (disjoint from main); nulls 128 (inside main; the first 32 get a 2nd draw); stable core 32; unweighted 128 (Pythia-70m only) | DAN-9 |
| Corpora | Fixed at **4M 64-token windows per model**, pinned revisions: one file of `monology/pile-uncopyrighted` (Pythia-70m); apollo pre-tokenised OpenWebText parquet files 0–2 (GPT-2 small). Exact duplicate windows removed at preparation. Pass 2 (co-activation stores) not built; nothing in the protocol reads it | DAN-37 |
| λ | Pilot of 30 targets × 3 λ (Pythia-70m {3e-4, 1e-3, 3e-3}; GPT-2 small {1e-3, 3e-3, 1e-2}). Pre-stated size-only rule: keep 1e-3 unless the pooled median circuit size leaves [100, 1000], else the grid λ nearest 494 on a log scale. Pass rates are reported, never used to choose. A local preliminary (5 targets) kept 1e-3 for both (median sizes 172 and 333) | DAN-42 |
| Held-out split | Group-aware: exact, anchor-prefix and near-duplicate copies (shared run ≥ 32 tokens, or ≥ 50% shared 8-grams) are grouped, and a group goes wholly to train or to held-out. **No document or source-file rule.** The held-out rule of §3.7 / DAN-157 is then applied as a final check. Contrast rows: exact + near-duplicate only (no anchor) | DAN-157 |
| Random nulls | The DAN-181 harder nulls replace DAN-17: an anchor-active null (WIN = 8, matched per site and by frequency decile, production members included) and a contribution-matched null (production members excluded), each fitted exactly as WCM, plus α = 1 scoring. No DAN-17 refits on the public models | DAN-181 |
| Precision | Fits and scores in float32 with TF32 matrix multiplies (as the DAN-36 gates ran); the data pass in bf16. TuringLLM stays bf16, its validated precision | DAN-37 |
| Unweighted | Pythia-70m only, 128 targets, at the λ adopted for WCM | DAN-40 |

## 10. Drift log

### 10.1 Against the DAN-7 issue text

| DAN-7 says | As run | Why |
|---|---|---|
| "64 per seed" | 64 strongest + up to 64 mid-band activating contexts, and 64 close contrast contexts. Pools of 8–63 are used as they are; fewer than 8 means the pool is absent | DAN-75 spec / 059 arm B |
| "seeded shuffled 48/16 split" | Deterministic stratified rotating split (no RNG); ranks 1–2 always train (`context_split.py`). Training is **32 strongest + 16 mid-band** (arm B), not 48 strongest | 058: the list split was ordered; 059: arm B chosen |
| "negatives verified silent" | Yes: post-Top-K activation is 0 at every position (close selector), not the unverified kNN store | DAN-64: the store gave 38% of targets one shared list |
| "lambda policy per model" | λ = 1e-3 fixed on TuringLLM; per model OPEN (DAN-42) | |
| "triple-floor weights" | γ_C = γ_A = 0.25 at **every** layer. The pre-v1 045 driver and the 029 panel used γ_A 0.10 (≤ L5) / 0.05 deeper | Protocol v1 decision (DAN-7, 2026-09-24) |
| "role-aware phi_sup (+ role-blind for the gap)" | **Role-blind only** (`phi_sup_blind`), with the form (a_pos − a_{M∖C})/(a_pos − e0) | DAN-72: leave-one-out effects are non-additive and no signed subset passes |
| "Roles: one definition (attribution sign)" | **No roles**: `skip_roles=True` | DAN-72 |
| "inhibitor release" | Not computed (no roles); the `release` field is a placeholder | DAN-72 |
| "phi_cf at fitted alpha and at alpha = 1" | Both recorded (`phi_cf_alpha_blind`, `phi_cf_a1_role`, role-blind in effect). Only φ_ind^α is reported | |
| "phi_pin" | Recorded only: α × clean position-wise pins; Z / A-sp / A-dense fills, no C | |
| "F0 at alpha = 1" | `free0_a1` in every row, plus the full α = 1 re-scoring (§8.2) | |
| "Nulls: fitted-amplitude nulls on a fixed ~5% subsample, N draws" | 747 targets (5.0% of 15,004), stratified depth × kind, **1 draw**. `free0_rand` in the eval rows is the empty circuit, not a null | DAN-17 |
| "every empty-circuit baseline recorded" | Yes: `e0`, `eM_dense`, `eM_topk`, `eN_dense`, `eN_topk`, `e0_tr` | |
| (not in DAN-7) | Additions: rank-keep (γ_R 3e-3), means from the training split only (`floors_train_only`, the held-out leak fix), activation-read primacy (DAN-66), every upstream site ablated (DAN-67), pass rule [0.8, 1.5] with necessity ≥ 0.9 (DAN-8), specificity reported, near-threshold stratum | 055/056/DAN-65..67/DAN-8 |

### 10.2 Against older protocols and notes

- **059 docstring.** It says "Protocol v1 takes the 64 strongest", and arm A ("48 strongest") is labelled the v1
  training set (`pool_test.py:3, 9`). Superseded by arm B (`driver.py:4-5`).
- **Illustrative band.** The [0.8, 1.25] band in 059 and the 062 README (`README.md:106`) is superseded by DAN-8's
  [0.8, 1.5] plus necessity.
- **Pre-v1 production 15k (045/049).** It used the store contrast contexts, had no rank-keep, built means from all
  contexts (the leak), used the γ_A depth rule, and ran a different eval pass. None of its circuits are v1; only its
  target list is reused (§2).
- **`main.tex:832`.** The comment there says the table "mirrors protocol-v1.md"; this file did not exist until now.
- **Eval-path changes in v3.** The engine `CircuitOnlyPatcher` ablates every in-scope site (DAN-67), and each ratio
  uses one read (DAN-66) (`amp_eval_pass_v2.py:3-24`). v1/v2 eval rows are not comparable.

### 10.3 Internal inconsistencies in the code as run (to note, not silently fix)

1. **The A reference differs between fit and evaluation for arm B (14,339 targets).**
   - The fit's A ablation value is the dense mean over the actual training set (32 strongest + 16 mid).
   - The scorer's A mean, the φ_ind pins, the specificity siblings and A mean, and the random-null live set all use
     the **48 strongest-train** contexts. 16 of those are never trained on and 16 of the training contexts (the mid
     ones) are absent.
   - This is by design in 059 ("one evaluation reference for all arms"), but the paper says the means are "computed
     on the training contexts".
2. **Fill mismatch.** Training uses a dense mean fill; the pass rule scores the sparsity-preserving fill.
3. **Mid-band padding defect.** See §3.6 (58 rows; fixed behind `EVAL_SPLIT=explicit` and re-scored).
4. **Two "clean rank" definitions.**
   - Rank-keep: rank among the *other* latents, on *training* anchors, clamped to ≤ k.
   - Near-threshold: rank among *all* latents, median over the *held-out* anchors, uncapped.
5. **Penalty temperature.** The penalty and leak guard use σ(θ), while the forward uses σ(θ/T). Selection is
   unaffected (θ > 0 either way).
6. **Two `thin` fields.** The scorer writes `thin` = (n_pos < 64 or n_neg < 64), and the driver then overwrites it
   with "no mid-band pool" (`amp_eval_pass_v2.py:609`; `driver.py:290`). Eval rows carry the driver's meaning.
7. **Misleading field name.** `free0_rand` is the empty circuit (empty live pool), not a random baseline.

## 11. Paper vs code

`main.tex` line numbers are for the working copy of 2026-10-07. 17 disagreements, most important first.

| # | `main.tex` | Paper says | Code as run | Code |
|---|---|---|---|---|
| 1 | 283, 861, 1472 | The ablation means are computed on the (48) training contexts only | True for the **fit**. The **evaluation** A mean (Z/A/C scores), the φ_ind pins and the specificity reference use the 48 **strongest-train** contexts, which for arm B (14,339 targets) is not the training set (32 strongest + 16 mid) | `protocol_harness.py:86`; `amp_eval_pass_v2.py:420`; `pool_test.py:234-235` |
| 2 | 349-351, 873, 324 | Necessity: the circuit's nodes "ablated per π", φ_sup(C; π) | **Zero ablation only**: members set to 0 in the live stream; a_∅ = the zero-ablation empty circuit `e0`. One necessity number, which the pass rule uses | `amp_eval_pass_v2.py:542-553` |
| 3 | 199-200, 856 | Each target has 64 strongest + 64 mid-band contexts, trains on 32 + 16, and is scored on 16 + 16 held out | Pools are capped at 64 and used as they are down to 8. **502** targets train on 6–47 contexts; **472** have fewer than 16 held-out strongest (478 have fewer than 64 stored strongest; six of those with 62–63 still hold out 16; corrected 2026-10-12, DAN-161 audit); 53 mid-band rows have fewer than 16 held out | `pool_test.py:118-130`; `main/contexts.shard*.jsonl` |
| 4 | 313 (also 200) | Every score is read on the 16 held-out contexts | On **58** mid-band rows the "held-out" slice contains 1–11 strongest-**train** contexts (padding defect) | `pool_test.py:257-259`; `amp_eval_pass_v2.py:380-381` |
| 5 | 856 | "Targets with no mid-band store train on 48 strongest (665 of 15,004)" | The rule is "fewer than 8 deduplicated mid-band contexts". Of the 665, **187** train on 48; **478** train on 6–47 | `pool_test.py:128`; `driver.py:137-139` |
| 6 | 206, 853 | 15,046 targets "sampled at random, stratified across sites … every site with an upstream SAE site" | 455 per component (uniform), layer-0 attention dropped (15,925), **then restricted to the 15,046 that produced a circuit in the pre-v1 run** (879 attention targets dropped). Attention cells hold 134–450 rather than 455 | `feature_selection.py:342-358`; `make_targets.py:26-43` |
| 7 | 220, 257-259, 862 (with 364) | The training ablation value is the same object as the evaluation one; sparsity-preserving is the evaluation's | The fit's A and C values are **dense** means (no Top-K re-imposed). The pass rule uses the sparsity-preserving fill. The paper never states that training is dense | `learned_mask.py:330-337`; `floors.py:122` |
| 8 | 201, 857, 1536 | Contrast candidates are ranked by similarity to "any of its activating contexts" / "every activating context is represented" | The reference set is the target's **top-context store** only; mid-band contexts are excluded from the reference (they are still excluded as candidates). Only the 8,192 most similar candidates are scanned | `neg_context_selector.py:317-323, 455-476, 959-970` |
| 9 | 1536 | Candidates are kept "until 64 are selected" | The scan is limited to the top 8,192. 4–63 kept contexts are accepted (10 targets have fewer than 64); fewer than 4 means the target is skipped | `neg_context_selector.py:849-876`; `pool_test.py:149` |
| 10 | 876 | Random baseline: "number of draws pending, DAN-46" | Run and quoted in the body (line 507): **1 draw × 747 targets**, stratified depth × kind; λ 0, θ₀ 40, binarize none, support-restricted. The table row is stale | `random_null.py:35-55`; `out_random_scale/` |
| 11 | 367, 507 | Random members are "drawn from live latents" / "active on its contexts" | They are drawn from latents active on the 48 strongest-train contexts, not the fit's training set (arm B) | `random_null.py:148-151` |
| 12 | 226, 325, 875 | φ_pin: nodes clamped to their clean-run values, "under the same ablation methods" | Clamped to **α ×** clean position-wise values; fills Z, sparsity-preserving A, dense A only (**no C**) | `amp_eval_pass_v2.py:575-597` |
| 13 | 348 (with 416, 1128) | φ_ind sets the nodes to their activating-context means | It sets the nodes to the mean **at the anchors** over the 48 strongest-train contexts, × **α** in the reported score. Every quoted "sufficiency to induce" (0.84, 0.98, 1.19, 0.20) is φ_ind^α | `amp_eval_pass_v2.py:445, 502-503`; `merge.py:78` |
| 14 | 300, 865, 1480 | Closed gates are charged λ(1 − m)\|α − 1\| with m = σ(θ/T) | Charged λ(1 − **σ(θ)**)\|α − 1\|: the temperature is not applied in the penalty. Line 1475's T = 0.05^(step/S), step 1..S, is in code T = 0.05^(step/(S−1)), step 0..S−1 | `learned_mask.py:1465-1467, 1608-1612` |
| 15 | 381, 879 | "targets whose ratio denominator is vacuous are excluded" | Vacuous is defined on the **Z** denominator only (\|a_pos − e0\| < 0.05·a_pos). An undefined A or C ratio counts as a fail (the figure caption at line 424 says this) | `amp_eval_pass_v2.py:468-469`; `merge.py:50-58` |
| 16 | 415, 1252 | Amplifier: siblings restored to ≥ 80% of the target's faithfulness | The flag also requires target faithfulness ≥ 0.5; it is on the **pre-activation** read, under any of Z / A / C | `merge.py:134`; `specificity.py:145-151` |
| 17 | 265-269, 863 | r = the target's rank among its site's other latents | r is clamped to ≤ k = 128: a target outside the clean Top-K at an anchor is asked to beat the k-th rival. Near-threshold uses a different, uncapped all-latent rank on held-out anchors (lines 375, 878) | `learned_mask.py:1412`; `specificity.py:209-210` |

Consistent with the code (checked):

- K = 128 and 40,960 latents.
- γ_Z = 1 and γ_C = γ_A = 0.25; ν = mean(target²); γ_R = 3e-3 inside each term; λ = 1e-3 as a flat sum.
- α₀ = 1; θ₀ = 4; anneal from 1 to 0.05; nodes are m > ½.
- AdamW, lr 0.05 constant, wd 0.05, 400 steps, batch 4.
- Errors preserved, nodes recomputed, every upstream site edited.
- The activation read is primary and the pre-activation read is diagnostic.
- The pass rule: Z, sparsity-preserving A and C in [0.75, 1.5] (0.8 until 2026-10-31, §12), and necessity ≥ 0.9.
- Near-threshold at K/2.
- The run counts: 42 skipped, 10 vacuous, 665 thin, 1,142 near-threshold.
- The sweep grids and 190 targets.
- 1,746 α = 1 targets.
- The split rule: blocks of 4, position (b + 2) mod 4, ranks 1–2 train.

---

## 12. Protocol v2 (decided 2026-10-25, Daniel; frozen before any v2 production fit)

Protocol v2 changes one thing in the method and rebuilds the TuringLLM data. Everything not named here is as in
§2–§9, including §9.2 for the public models.

### 12.1 The method change: A and C are trained under the fill they are scored with

**Problem it fixes.** In v1 the A and C training terms used a **dense** mean fill (§4: every non-member latent at its
mean, no Top-K re-imposed), while the pass rule scores A and C with the **sparsity-preserving** fill. That fill keeps
every member at α·live and fills only the top (K − #active members) non-member slots per position with their means
(`_respect_topk_fill`). A dense code has tens of thousands of non-zero entries, whereas every code the SAE produces has
at most k, so the dense fill is out of distribution for the SAE. Circuits learned to lean on its diffuse background,
which the scorer then removes. Over the 15k:
- the rule pass rate is 46.4% (sparsity-preserving A/C) vs 48.7% (dense A/C);
- deep attention is hit hardest, with A falling from 0.93 (dense) to 0.00 (sparsity-preserving) at L5–7.

**v2.**
- **The fill:** A and C are trained under the sparsity-preserving fill, defined bit-for-bit as the scorer's.
- **Gradients:** straight-through (`sparse_grad = "ste"`).
- **Means:** the training A and C means are computed exactly as the scorer's: the A mean over the 48 strongest-train
  reference rows and the C mean over the 48 contrast-train rows (`train_means_source = "eval_ref"`).
- **Unchanged:** the Z term; γ_Z = 1 and γ_A = γ_C = 0.25; rank-keep γ_R = 3e-3; λ = 1e-3 (re-checked with the
  size-only rule of §9.2); 400 steps; the optimiser; batch; anneal.
- **The switch:** `WCM_VERSION=v2` in `062/driver.py`, mapping to `train_fill = "sparse"`, `sparse_grad = "ste"`,
  `train_means_source = "eval_ref"` (`src/config.py`, default-off: v1 is unchanged bit for bit).
- **Unweighted arms** (Daniel, 2026-10-26): the unweighted (gate-only) comparison arms also run under v2, with every
  coefficient fixed at exactly 1, so they differ from WCM only in the coefficients.

**Evidence before adoption** (exps 089–092; DAN-183):

| Evidence | Result |
|---|---|
| exp 090, 20 targets enriched for failures | rule pass 7/20 (dense) → 15/20 (v2); the 8 earlier failures go 0/8 → 6/8; Z unchanged (paired median ≈ 0); 8/8 passing controls still pass; circuits 0.84–0.86× the size; fit time 1.28× |
| exp 089 | reweighting the dense terms cannot reach the rule's A/C |
| Not included | resample R (DAN-184, appendix only), the silent-position term, and the speed-ups (anchor truncation, batched passes); exps 091–092 |

**Validation (exp 093), stated before it runs.** 72 targets, 8 per depth band × kind, seed 20261026, v1 vs v2 paired
on the same GPU. v2 stays production unless, on this sample:
- its paired pass rate is lower than v1's (the paired bootstrap 95% CI entirely below 0); or
- Z falls systematically (paired median ΔZ < −0.05).

**Pass floor 0.75 (Daniel, 2026-10-31; before any production fit).** The band floor moves from 0.8 to 0.75 for Z, A,
C (and the donor-background pass), on both reads; the 1.5 ceiling and necessity ≥ 0.9 are unchanged. Reason given:
0.75 is the more appropriate tolerance on judgement, not on the numbers. Seen beforehand (sensitivity only): v1 15k
46.4% → 49.5%; exp 093 v2 50 → 55 of 72; exp 095 resample-trained 45 → 48 of 72. Code: `src/eval/wcm_eval.py` BAND,
`062/merge.py`, `083/cmp_common.py`, `paper/tables/common.py`, explorer `stage_circuits.py`. Earlier experiment
reports in this file keep the 0.8 floor they were scored at. The paper states the rule without this history.

**Two pass rates, both recorded and reported (Daniel, 2026-10-26; stated before any v2 production fit).** For every
target, the same rule is scored on two reads, and the two are never mixed inside one rule:
- **Post-Top-K pass rate (headline):** Z, A_sp, C_sp ∈ [0.75, 1.5] and necessity ≥ 0.9 on the activation read (`*_tk`).
- **Pre-activation pass rate:** the same rule on the pre-activation read (`*_pre`). It asks whether the circuit
  reproduces the target's input, separately from whether the target then survives the Top-K competition against
  rival latents.

Both are reported over all targets and for the near-threshold stratum (clean rank ≥ K/2), with a 2×2 split: passes
both / pre-activation only / neither. On the v1 15k, the near-threshold stratum (1,142 targets, 1,134 of them
attention) passes 0.8% post-Top-K and **7.0%** under the pre-activation rule as defined below (baseline 0 for Z and
necessity; `paper/tables/main_result.py`). Only 54.5% of the stratum reach necessity_pre ≥ 0.9: removing the members
must take the target's input itself down by 90%, whereas the post-Top-K read counts a drop below the cut as complete.
- *Correction (2026-10-27):* an earlier version of this paragraph quoted 10.9%. That number came from `062/pass_rule.py`,
  whose pre-activation rule used the empty-circuit Z baseline and the post-Top-K necessity, mixing the reads.
- The rule below is unchanged.

**The pre-activation read's Z and necessity baseline is 0, "latent off"** (Daniel, 2026-10-26). Z_pre = x / a_pos and
necessity_pre = (a_pos − x_removed) / a_pos, both computed from the logged raw values (`free0_raw_pre`, `a_pos_pre`,
`phi_sup_blind_raw_pre`). A and C keep their own mean-ablation baselines.
- **Why:** with every upstream latent zeroed, a residual site's input goes far off its natural range, and the target's
  pre-activation comes out at a median ~190× its clean value. The empty-circuit form (x − e0) / (a_pos − e0) is then
  undefined for 91% of residual targets, whereas the circuit itself stays in range (circuit / clean: median 0.97,
  p10–p90 0.87–1.04; v1 15k, 10 shards).
- **Consistency:** the post-Top-K read already sits on a baseline of 0 in practice (median e0_tk = 0.00).
- **Sensitivity, reported alongside:** for attention and MLP sites, where e0_pre is well defined (median 4–7% of
  a_pos), the pre-activation pass rate is also reported with the empty-circuit baseline.

### 12.2 TuringLLM data: 8k targets from fresh records

- **Targets:** a seeded (20261026) stratified subset of the 15,046 v1 targets, ~229 per layer × kind cell (8,000 in
  all). It includes:
  - the 190 λ-sweep targets;
  - the 126-target null panel;
  - the case-study and random-gallery targets.

  Its precision is about ±1.1 points overall and about ±3 per band × kind cell.
- **Records** (`experiments/094-tl-v2-records/`), built fresh so the held-out leakage of §3.7 cannot occur:
  - strongest pools deduplicated: exact copies, anchor-prefix copies and near-duplicates (shared run ≥ 32 tokens, or
    ≥ 50% shared 8-grams), plus corpus twins;
  - candidates beyond the stores' 64 slots from the exp 086 corpus scan, with the stores' ranking statistic;
  - a group-aware split: no group of copies straddles train and held-out; there is no document or source-file rule;
  - contrast contexts by the same close, silence-checked selector, deduplicated by the same rule (exact + near only,
    since contrast rows have no anchor);
  - sizes as in v1: 32 strongest + 16 mid-band train, 16 + 16 held-out, 48 / 16 contrast.
- **Everything the paper quotes is refit or re-scored under v2 on these records:**
  - the main 8k (mid-band, specificity, α = 1);
  - the 190-target λ sweep (claim 2);
  - the stable core;
  - the harder nulls (DAN-181);
  - the method comparison (DAN-159);
  - the case study and the gallery;
  - the public models (512 each).

  The v1 numbers remain in the record (§1–§11) and are reported only as the v1-vs-v2 comparison.

### 12.3 The resample term R in training, and the donor-background test (Daniel, 2026-10-31; before any production fit)

**Production training adds R** (exps 095, 100; native in `learned_mask.py`, driver `WCM_VERSION=v2cr`; equivalence in
exp 101; final weights 2026-11-01, `WCM_VERSION=v3`, below). The objective is Z + A + C + R_C:
- R_C is a resample ablation whose donors are the target's 48 training contrast contexts, one donor redrawn per
  training row on every forward;
- it uses the sparsity-preserving donor fill, weight 0.25, λ 1e-3, 400 steps;
- unweighted arms (α = 1) take the same R through the same sparse path.

Exp 102 (DAN-187) may still change settings (e.g. rank-keep weight, R weight) before the pod. Any change is recorded
here, with its date, before the pod starts.

**Final production settings (Daniel, 2026-11-01; exp 102; before any production fit): driver `WCM_VERSION=v3`,
the 082 chain's default at the time.** (Superseded for production by **v4**, §0: v3 + the geometric lr schedule
with warm-up and a ×3.15 peak, + the censored read.) Everything above holds, with two changes:
- **R weight γ_R = 1.0** (was 0.25);
- **rank-keep weight 1e-1** (was 3e-3; mode `keep`, δ 0 as before).

Both apply to every arm of a v3 run: the main fit, every WCM and unweighted λ-sweep arm, the λ pilot, and the
companions (α = 1, nulls, stable core). Unweighted arms take R through the sparse path at α = 1. Everything else is
unchanged: Z + A + C + R_C, γ_C = γ_A = 0.25, λ 1e-3, 400 steps, the same donors and donor fill.
- *Evidence* (exp 102, selection rule stated before any result, the frozen 72-target check panel, floor 0.75):
  - production pass 48 → 54 (+8 pp, paired 95% CI [+0, +18]);
  - resample pass 19 → 39 (+28 pp, CI [+17, +39]);
  - circuits ×1.33 larger (median 702 vs 572);
  - fit time ×0.97 of v2cr (so about 1.35× v2).
- *Equivalence:* native v3 fits through the 062 driver reproduce 102's arm `r1.0_rk1e-1` bit for bit on 3 check
  targets, one per depth band (exp 104 part A).
- v1 / v2 / v2cr keep rank-keep 3e-3 and are unchanged.

**Donor-background test, reported for every circuit alongside the pass rule** (099's whole-donor test, as run in exp
100):
- **Bases:** the 16 held-out strongest activating + 16 held-out contrast contexts.
- **The ablation:** every latent outside the circuit, at every upstream site, is taken from a donor context's natural
  sparse code at the same position (073 `ResamplePatcher`):
  - members at α × live;
  - the donor's non-member active entries fill the remaining Top-K budget, largest first;
  - the current stream's SAE error kept live.
- **Three donor pools:**
  - A = the 48 strongest-train activating contexts;
  - C = the 16 held-out contrast contexts (never a training donor);
  - R = a fixed random-corpus pool filtered against every record (exact / near rule).
- **Draws:** eight paired draws per pool, one donor per base per draw, seeded per target; the empty circuit is run
  under the same donors.
- **Donor recovery** (per pool) = (mean over draws of the circuit's mean anchor read − the same for the empty
  circuit) / (natural mean anchor read − empty):
  - pooled over draws before dividing;
  - undefined when the denominator is ≤ 1e-9, and kept undefined (an undefined score fails).
- **Resample pass** = the production pass (§6, floor 0.75) AND donor recovery A and C each in [0.75, 1.5]. It is a
  subset of production passes, so the headline reads "x% pass; y% also pass on real backgrounds".
- **Reported with it, not inside it:**
  - R recovery;
  - the α = 1 and pinned-member variants;
  - the per-pool empty leak;
  - the split by whether the donor fires at the base anchor.
- **Scope:** the test is a robustness diagnostic. It is not recursive causal scrubbing and does not replace the
  production rule.
- **Where it runs on the pod:** the 8k main circuits, the 190-target λ sweep (all arms), the without-R comparison arm
  on the 190 sweep targets (the appendix ablation), and the unweighted arms under R.
- **The without-R comparison arm (Daniel, 2026-11-01):** v2 at v3's rank-keep 1e-1 (chain T10 `TL_NOR_ARM`
  default `W_0.001_r0.1`), so the paired comparison with the production `W_0.001` arm differs in R alone. The
  previous production algorithm (rank-keep 3e-3) is not re-fitted.
- **The proxy used in dev:** exp 095 / 102 selection used 079's 2-draw `strict_eval` version of the same pools. The
  pod and the paper use the 8-draw pooled test above.

### 12.4 Vacuous denominators: every pass-rule baseline, signed (Daniel, 2026-11-01; before any production fit)

**Problem it fixes.** The v1 vacuous flag (§5.2, §6) reads only the Z / necessity denominator and is two-sided
(|a_pos − e0| < 0.05·a_pos), while each pass-rule term is a ratio against its own empty-circuit baseline and is
undefined whenever that baseline reaches a_pos (e ≥ a_pos). Two cases escape the flag and fail in every arm,
whatever the circuit:
- e0 above 1.05·a_pos: zeroing every upstream latent drives a residual or early-MLP target far above its natural
  value (median 5.6×);
- eM_topk or eN_topk ≥ a_pos: the sparsity-preserving mean fill alone drives the target, mostly deep attention,
  above its held-out mean (median 1.17×).

In the v1 15k this is 482 of 15,004 held-out-strongest targets (3.2%; 12% of L8-11 attention), and 3 of 72 on each
exp 102 panel (exp 105).

**Rule.** On the post-Top-K read, a target is **vacuous** when a_pos ≤ 0 or any pass-rule baseline exceeds
0.95·a_pos: a_pos − e < 0.05·a_pos for e = e0 (Z and necessity), eM_topk (A) or eN_topk (C).
- **Population.** Vacuous targets are dropped from the denominator of every pass rate (post-Top-K, pre-activation,
  and the resample pass) and counted. The exclusion set comes from the post-Top-K read and is shared by both reads.
- **Scope.** The rule applies identically to held-out strongest and mid-band, and to every model and arm.
- **Why it is a property of the target.** The baselines are empty-circuit reads on the target's held-out, A-mean
  and C-mean contexts. They do not depend on the circuit and can be computed before any fit (exp 105: the stored
  undefined terms are exactly the rows whose baselines fail e < a_pos; baselines are identical across arms).
- **Relation to v1.** It contains the v1 flag, and replaces the v1 statement (§6, §11 row 15) that an undefined A or
  C ratio counts as a fail. A term still missing on a non-vacuous target fails.
- **Reported with it:**
  - the vacuous count per band × kind and per term (Z/necessity, A, C);
  - the vacuous targets' pass on their defined terms (descriptive);
  - the all-targets pass rate counting vacuous targets as fails (a lower bound).
- **Seen beforehand (sensitivity only):**
  - v1 15k held-out strongest: 49.5% of 14,994 → 51.4% of 14,453 (all-targets 49.5%);
  - mid-band: 14.0% → 16.6%;
  - exp 102: 3 targets per panel excluded in every arm; arm counts, paired differences and the selection-rule
    winner unchanged.
- **Code:** `src/eval/wcm_eval.py` (`vacuous_v2`, stored per row as `vacuous_v2` on the public path; `is_vacuous(row)`
  defaults to this rule and recomputes it from the stored a_pos / e0 / eM_topk / eN_topk reads for rows without the
  field: rows scored earlier, and TuringLLM rows, which come from the frozen 049 scorer; `rule="v1"` gives the old
  flag), with the
  same rule in `062/merge.py:vacuous`, `083/cmp_common.py:vacuous` (also `paper/tables/*`, 098), the donor test's
  summary and the explorer's `stage_circuits.py`. Tests: `tests/eval/test_wcm_eval.py` (cases + cross-implementation
  parity).
