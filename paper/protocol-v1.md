# Protocol v1: frozen method and evaluation specification (DAN-7)

Drafted 2026-10-07. This document describes **what the production run executed**: the TuringLLM full run of
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
`experiments/062-h100-protocol-v1/driver.py`. A port into `src/circuit/protocol_contexts.py`
(`config.discovery.context_protocol = "v1"`) is in progress (DAN-78) and was **not** used by the run. The scorer is
`experiments/049-circuit-graph/amp_eval_pass_v2.py`; moving it into `src/` is DAN-73 (not started).

---

## 1. Scope and status

| Item | Status |
|---|---|
| TuringLLM targets, contexts, fit, evaluation, pass rule, diagnostics (§2–§8) | **Frozen**: this is what the full run did |
| Random-circuit baseline, α = 1 re-scoring, unweighted sweep, λ grid (§8) | **Frozen as run**: separate runs, same engine and scorer |
| Pythia-70m and GPT-2 small adaptation (§9) | **OPEN**: DAN-9, DAN-34/35/36, DAN-40/41, DAN-42 |
| Production location of contexts (DAN-78) and scorer (DAN-73) | **Pending**: the code moves; the definitions must not change |
| Mid-band evaluation padding defect (§3.6, §10) | **Open**: 58 of 14,339 mid-band rows affected; decide whether to rescore |

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
| Tokens | 65 tokens are loaded; the first 64 are the context | `pool_test.py:85-95` |
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

### 3.6 Known defect: mid-band padding

When the mid-band held-out set has fewer contexts than the strongest-train set needs as padding, the probe has fewer
than 64 rows. The scorer's `split_n` then moves part of strongest-train into the "held-out" slice
(`pool_test.py:257-259` against `amp_eval_pass_v2.py:380-381`). As run this affects **58 mid-band rows**: the
held-out slice holds 1 to 11 strongest-train contexts, and the A mean and pins use fewer than 48. A further 53
mid-band rows have fewer than 16 held-out contexts. Held-out strongest rows are unaffected (the pad always matches
by construction).

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
| Vacuous exclusion | Rows with `vacuous_tk` are dropped from the denominator and counted (10 strongest rows, 23 mid-band rows). An undefined A or C ratio is **not** vacuous: it counts as a fail | `merge.py:57-58, 64-65`; `amp_eval_pass_v2.py:468-469` |
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
the mid-band mean (`driver.py:286-291`). It is reported, not used in the pass rule; run result: 12.1% pass. See §3.6
for the 58 affected rows.

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

## 9. Per-model adaptation (Pythia-70m, GPT-2 small): OPEN

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
3. **Mid-band padding defect.** See §3.6 (58 rows).
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
| 3 | 199-200, 856 | Each target has 64 strongest + 64 mid-band contexts, trains on 32 + 16, and is scored on 16 + 16 held out | Pools are capped at 64 and used as they are down to 8. **502** targets train on 6–47 contexts; **478** have fewer than 16 held-out strongest; 53 mid-band rows have fewer than 16 held out | `pool_test.py:118-130`; `main/contexts.shard*.jsonl` |
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
- The pass rule: Z, sparsity-preserving A and C in [0.8, 1.5], and necessity ≥ 0.9.
- Near-threshold at K/2.
- The run counts: 42 skipped, 10 vacuous, 665 thin, 1,142 near-threshold.
- The sweep grids and 190 targets.
- 1,746 α = 1 targets.
- The split rule: blocks of 4, position (b + 2) mod 4, ranks 1–2 train.
