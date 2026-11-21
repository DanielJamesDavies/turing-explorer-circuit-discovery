# 133 — IOI WCM sweeps on GPT-2 small: why 081 v4 collapses, and can any WCM config pass?

Overnight 2026-11-19 (agent ioi-wcm). The companion experiment is **132** (agent ioi-base: substrate ceiling and
baselines).

Setup:
- **Model and dictionaries:** GPT-2 small with the OAI v5 32k Top-K SAEs (48 sites).
- **Task:** 081's IOI task (1,600 training items).
- **Engine:** the 081 v4 engine (`engine.engine_kwargs`), with the overrides listed per row.
- **Scoring:** 081's unchanged scorer (`engine.score` = `eval.wcm_eval.score_task`: coefficients applied, A / C = the
  whole training / contrast pool means), 081's minimal-pair read (`engine.score_cf`, "mp Z"), and the pass rule
  `wcm_eval.passes_task`.
- **Pass rule:** Z, A_sp, C_sp in **[0.75, 1.5]**, necessity ≥ 0.9.
  - **Flag:** src's `wcm_eval.BAND` is (0.75, 1.5). 081's README says [0.8, 1.5]; the scorer uses 0.75.
- src and 081's files are not edited; `engine133.py` patches the engine in memory (131's pattern).

## Verdict

1. **Why 081 v4 IOI collapsed: the v4 learning rate.**
   - At v4's peak lr (0.05 × 3.15 = 0.1575, geometric, warm-up 0.1, absolute wd 0.05), **W does not fit its own
     training stream**: λ 1e-4, 21k latents, Z 0.01.
     - The logged training curves show the normalised errors rising from 0.04 at the start to 0.7–0.8 at the end:
       zero 0.04 → 0.80, C 0.006 → 0.20 (weighted ×0.25, i.e. 0.8), A likewise, R 0.05 → 0.77.
     - Same-name items fail too, Z 0.01, so this is not a generalisation effect.
   - **Not R:** with R off the fit collapses the same way (8.2k latents, Z 0.02, zero-term 0.97).
   - **Peak lr 0.05**, everything else v4: the same λ 1e-4 fit reaches zero-term 0.08 and same-name Z / A / C 0.95 /
     0.95 / 0.94.
   - U at 0.1575 fits fine (training error 0.004–0.025), so the high lr breaks the **free coefficients**.
   - On the attn+mlp scope a peak lr of 0.1 also works (910 latents, same-name pass). 0.1575 was not retested there.
2. **The other three suspects were not the cause.**
   - **(c) objective vs rule:** not an issue for IOI. With single-token answers, the fitted log p(IO) − log p(S) *is*
     the logit difference (`learned_mask._target_logprob` with a contrast token). The mismatch is greater-than's
     only.
   - **(b) padded R donors:** fixed with paired donors (each row's own ABC prompt, same length; `engine133` PAIR). This
     made circuits smaller at the same λ (48-site λ 1e-4: 76k → 59k), with the same same-name Z. It is not why
     IOI failed.
   - **(d) censor / rank-keep / NEC_W:** not tested; src refuses them for the logit objective, and nothing here
     needed them.
3. **The remaining failure is the name split, not the method.**
   - 081's held-out set ("dis") uses Wang et al.'s other 50 names, places and objects, disjoint from training.
     `make_sn.py` builds a **same-name set ("sn")**: 998 fresh items from the training halves of those lists, none of
     them a training item.
   - On "sn", **every fit that trained passes**. For example, 081's own v4 U circuit (132k latents) scores dis Z
     0.18 but sn Z / A / C 0.99 / 0.99 / 0.99, nec 1.03.
   - 132's ceiling explains why the 48-site scope cannot pass on "dis": all 242,770 train-active latents give dis Z
     0.23, because latents for unseen names were never active in training.
   - On the **attn+mlp scope** (resid / resid_mid left live) 132's ceiling is dis Z 0.85, so the remaining sweeps ran
     there.
4. **attn+mlp scope, peak lr 0.05, paired R. These are the best configurations.**
   - **Same names:** W passes from **493 latents** (λ 1e-2: Z 0.79, A 0.78, C 0.84, nec 1.84) up to 117k.
     - Cleaner small passes: 777 latents with R off (λ 3e-3: 0.95 / 1.00 / 0.99, nec 1.54) and 1,361 with R on
       (λ 3e-3: 0.96 / 0.97 / 0.95, nec 1.41).
     - λ 3e-2 collapses (69 latents, Z 0.02).
   - **Random null** (081's RND: random train-active attn+mlp latents, per-site counts matched, coefficients fitted):
     matched to 493 latents, Z 0.04 and 0.01 (2 draws); matched to 1,361, Z 0.01. All fail.
   - **Disjoint names (081's registered held-out rule):** Z rises steadily with size and **first passes at λ 1e-6,
     116,833 latents.**
     - **Confirmed on all 5,000 held-out items:** Z / A / C 0.78 / 0.78 / 0.78, nec 1.18, accuracy 0.95, mp Z 0.78.
     - The **matched random null fails** there: Z / A / C 0.60 / 0.68 / 0.66 (1 draw).
     - λ 3e-6, 91.8k latents: 0.750 / 0.7499 / 0.751 on 500 items, missing the band by 1e-4.
   - **Claim 2 (U vs W), same names, incomplete:**
     - the smallest W pass is 493 latents;
     - U passes at 8,640 (λ 1e-4, sn 0.98 / 0.99 / 1.00) and 28k;
     - U at λ ≥ 1e-3 was not run (time), so the ratio is only bounded by ≤ 17.5×.
     - On disjoint names: U 28k Z 0.55 vs W 34k 0.59; no real separation.
5. **The task off-term does not help, because there is nothing for it to correct.**
   - The term (one-sided hinge on the circuit's own-ABC logit difference exceeding the model's, Z frame, `engine133`
     OFF) ran at every step (400 calls). The circuit-minus-model excess was −0.002 to +0.027 on average, with 50% of
     rows positive.
     - The circuits already output the model's ~0 on their counterfactual, and mp Z equals Z to ±0.02 in every row.
     - 129's failure, circuits that ignore minimal pairs, does not occur on this behavioural endpoint.
   - Results at λ 1e-3 (attn+mlp):
     - no term: 2,563 latents, sn Z 0.95, dis Z 0.23;
     - hinge ×1: 2,592, 0.98, 0.24;
     - hinge ×4: 2,821 (+10%), 1.01, 0.24;
     - squared ×1: 2,577, 0.93, 0.20;
     - W_CF: 4,076 (+59%), 1.07, 0.28.
   - At λ 3e-3, hinge ×4 gave 1,452 latents (+7%), sn Z 0.95 vs 0.96.
   - The off-term and W_CF leave the result unchanged and cost size.

**Best config:**
- **Registered rule (disjoint names):** the attn+mlp scope, W, λ 1e-6, peak lr 0.05 geometric with warm-up 0.1, wd
  0.05, R 1.0 with paired ABC donors, 400 × 4. 116,833 latents (56,195 attn + 60,638 mlp), the only pass.
- **Same names, sparsest:** the same recipe at λ 3e-3 (1,361 latents) or λ 1e-2 (493).

## Job j6: the full 5,000-item check of the disjoint-name pass

| config | n | dis (5,000) Z / A / C | nec | acc | mp Z | pass | sn (998) Z / A / C | sn pass |
|---|---|---|---|---|---|---|---|---|
| `rs5k_AM_W_1e-6` (the W circuit above, rescored) | 116,833 | **0.78 / 0.78 / 0.78** | 1.18 | 0.95 | 0.78 | **yes** | 1.02 / 1.02 / 1.02 | yes |
| `AM_RND_1e-6_d0_5k` (random, matched per site, α fitted; 1 draw) | 116,833 | 0.60 / 0.68 / 0.66 | 1.31 | 0.89 | 0.61 | no | 0.98 / 0.96 / 0.95 | **yes** |

- **On disjoint names (the registered rule), the W circuit passes on all 5,000 items and its matched random null
  fails.** The margin is modest (Z 0.78 vs 0.60), and there is one draw.
- **On the same-name set at this size, the random null passes as well.** About 70% of the train-active attn+mlp
  latents carry the trained names regardless of which ones are picked. So the same-name read only discriminates at
  small sizes: random nulls at 493 and 1,361 latents fail it (Z ≤ 0.04) while W passes.
- The λ 3e-6 5,000-item rescore was dropped, to keep the job under ~45 min.

## Diagnosis details

- **081 v4 training curves** (`inspect_train.py` on `081/out_v4/gpt2-small/train.jsonl`). Per-term values at the start
  and end of the fit, weighted (C and A at γ 0.25):

  | fit | zero | C | A | R | n |
  |---|---|---|---|---|---|
  | v4 W 1e-4 | 0.042 → 0.80 | 0.006 → 0.20 | 0.006 → 0.18 | 0.053 → 0.77 | 21k |
  | v4 U 1e-5 | 0.042 → 0.004 | 0.006 → 0.002 | 0.006 → 0.001 | 0.053 → 0.052 | 75k |
  | v2 W 1e-4 (lr 0.05 constant, no R) | 0.042 → 0.14 | 0.006 → 0.09 | 0.006 → 0.10 | — | 21k |

- **Training forward = scorer.** After every fit, `engine133.soft_eval` recomputes Z on the same-name items with the
  training forward (the patcher), at the final soft gates and at hard-thresholded gates.
  - Hard ≈ scorer to ±0.01 in every row, so the two forward paths agree.
  - Soft ≈ hard (max gap 0.06), so no sub-threshold gate leak inflates training faithfulness.
- **Names:** 24–43% of the 081 W members never fire on the first 256 disjoint-name held-out items (`fire_share.py`).
  Disjoint-name Z rises monotonically with circuit size on both scopes, while same-name Z is saturated from ~1k
  latents.
- **48-site scope, same names** (peak lr 0.05, paired R): W passes at 59k, 32k and 16.7k latents (λ 1e-4, 3e-4,
  1e-3; sn Z 0.97 / 0.93 / 0.89). Disjoint Z is 0.37 / 0.25 / 0.15, under 132's ceiling of 0.23 except at the largest
  size. That sweep was cut after two points, on the coordinator's instruction, in favour of the attn+mlp scope.
- **081 v2 circuits rescored:** W 21k: dis 0.48 / sn 0.72; 9.5k: 0.36 / 0.50; 4.2k: 0.24 / 0.24. These v2 fits were
  also underfitted (v2 lr 0.05 constant, 400 steps).
- **R:** on the attn+mlp scope R costs size. At λ 3e-3: R off 777 latents, R paired 1,361, both passing on same names;
  disjoint Z 0.35 vs 0.21. 800 steps (940 latents) and peak lr 0.1 (910) give smaller circuits at the same λ, both
  passing on same names.

## Results table (`out/table.md`; 500 disjoint-name + 500 same-name items per row unless stated; pass = v4 rule)

Columns: dis = 081's held-out items (names disjoint from training), sn = same-name fresh items.
- **Scopes:** "all" = 48 sites; "attn_mlp" = the 24 attn + mlp sites, with the scorer's ablation scope the same, so
  resid / resid_mid stay live.
- **Fit settings:** "lr" = peak lr; geometric schedule, warm-up 0.1, wd 0.05, 400 steps × 4 unless stated.
- "rescore" rows are 081's own circuits (all sites).

| config | arm | λ | n | dis Z | dis A | dis C | dis nec | dis acc | dis mp Z | dis pass | sn Z | sn A | sn C | sn nec | sn acc | sn mp Z | sn pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rs_v2_W_1e-3 | 081 v2 W | 1e-3 | 4174 | 0.24 | 0.29 | 0.29 | 1.19 | 0.66 | 0.24 | no | 0.24 | 0.28 | 0.27 | 1.06 | 0.66 | 0.25 | no |
| rs_v2_W_3e-4 | 081 v2 W | 3e-4 | 9460 | 0.36 | 0.49 | 0.47 | 1.09 | 0.73 | 0.36 | no | 0.50 | 0.61 | 0.60 | 1.03 | 0.82 | 0.48 | no |
| rs_v2_W_1e-4 | 081 v2 W | 1e-4 | 20961 | 0.48 | 0.53 | 0.51 | 1.04 | 0.83 | 0.46 | no | 0.72 | 0.75 | 0.75 | 1.00 | 0.93 | 0.73 | no |
| rs_v4_W_1e-4 | 081 v4 W | 1e-4 | 21350 | 0.01 | 0.05 | 0.01 | 1.11 | 0.54 | 0.01 | no | 0.01 | -0.01 | 0.01 | 1.02 | 0.53 | 0.02 | no |
| rs_v4_U_1e-6 | 081 v4 U | 1e-6 | 131984 | 0.18 | 0.18 | 0.18 | 1.03 | 0.68 | 0.19 | no | 0.99 | 0.99 | 0.99 | 1.03 | 0.99 | 0.99 | **yes** |
| W_1e-4_R0 | all W lr 0.1575, R off | 1e-4 | 8163 | 0.02 | 0.00 | 0.01 | 1.14 | 0.54 | 0.02 | no | -0.00 | -0.01 | -0.00 | 1.02 | 0.51 | 0.02 | no |
| W_1e-4_lr05 | all W lr 0.05, R random | 1e-4 | 76316 | 0.38 | 0.38 | 0.38 | 1.02 | 0.86 | 0.36 | no | 0.95 | 0.95 | 0.94 | 1.00 | 1.00 | 0.97 | **yes** |
| W_1e-4_lr05_pair | all W lr 0.05, R paired | 1e-4 | 58747 | 0.37 | 0.36 | 0.36 | 1.05 | 0.86 | 0.37 | no | 0.97 | 0.96 | 0.96 | 1.00 | 1.00 | 0.97 | **yes** |
| W_3e-4_lr05_pair | all W lr 0.05, R paired | 3e-4 | 31946 | 0.25 | 0.24 | 0.24 | 1.08 | 0.75 | 0.26 | no | 0.93 | 0.93 | 0.94 | 1.02 | 0.99 | 0.93 | **yes** |
| W_1e-3_lr05_pair | all W lr 0.05, R paired | 1e-3 | 16721 | 0.15 | 0.21 | 0.23 | 1.08 | 0.65 | 0.16 | no | 0.89 | 0.96 | 0.95 | 1.01 | 1.00 | 0.90 | **yes** |
| AM_W_1e-6_lr05_pair | attn_mlp W lr 0.05, R paired | 1e-6 | 116833 | 0.78 | 0.78 | 0.78 | 1.14 | 0.94 | 0.77 | **yes** | 1.03 | 1.03 | 1.03 | 1.06 | 0.99 | 1.03 | **yes** |
| AM_W_3e-6_lr05_pair | " | 3e-6 | 91773 | 0.75 | 0.75 | 0.75 | 1.14 | 0.93 | 0.75 | no (A 0.7499) | 1.03 | 1.03 | 1.03 | 1.07 | 0.99 | 1.03 | **yes** |
| AM_W_1e-5_lr05_pair | " | 1e-5 | 61043 | 0.69 | 0.69 | 0.70 | 1.12 | 0.91 | 0.69 | no | 1.03 | 1.03 | 1.03 | 1.06 | 0.99 | 1.03 | **yes** |
| AM_W_3e-5_lr05_pair | " | 3e-5 | 34117 | 0.59 | 0.59 | 0.59 | 1.20 | 0.89 | 0.58 | no | 1.03 | 1.03 | 1.03 | 1.07 | 0.99 | 1.03 | **yes** |
| AM_W_1e-4_lr05_pair | " | 1e-4 | 13842 | 0.42 | 0.43 | 0.43 | 1.20 | 0.84 | 0.42 | no | 1.02 | 1.02 | 1.02 | 1.06 | 0.99 | 1.02 | **yes** |
| AM_W_3e-4_lr05_pair | " | 3e-4 | 5654 | 0.28 | 0.30 | 0.30 | 1.20 | 0.80 | 0.28 | no | 0.99 | 1.02 | 1.01 | 1.06 | 0.99 | 1.00 | **yes** |
| AM_W_1e-3_lr05_pair | " | 1e-3 | 2563 | 0.23 | 0.27 | 0.28 | 1.27 | 0.75 | 0.23 | no | 0.95 | 1.04 | 1.04 | 1.12 | 1.00 | 0.97 | **yes** |
| AM_W_3e-3_lr05_pair | " | 3e-3 | 1361 | 0.21 | 0.24 | 0.23 | 1.66 | 0.72 | 0.22 | no | 0.96 | 0.97 | 0.95 | 1.41 | 0.99 | 0.99 | **yes** |
| AM_W_5e-3_lr05_pair | " | 5e-3 | 1024 | 0.26 | 0.31 | 0.30 | 1.68 | 0.72 | 0.26 | no | 0.87 | 1.02 | 0.98 | 1.48 | 0.99 | 0.89 | **yes** |
| AM_W_1e-2_lr05_pair | " | 1e-2 | 493 | 0.47 | 0.38 | 0.43 | 1.89 | 0.79 | 0.46 | no | 0.79 | 0.78 | 0.84 | 1.84 | 0.93 | 0.79 | **yes** |
| AM_W_3e-2_lr05_pair | " | 3e-2 | 69 | 0.03 | 0.00 | 0.00 | 1.89 | 0.50 | 0.03 | no | 0.02 | 0.02 | 0.00 | 1.89 | 0.51 | 0.03 | no |
| AM_W_3e-3_lr05_R0 | attn_mlp W lr 0.05, R off | 3e-3 | 777 | 0.35 | 0.36 | 0.37 | 1.60 | 0.78 | 0.34 | no | 0.95 | 1.00 | 0.99 | 1.54 | 0.99 | 0.96 | **yes** |
| AM_W_3e-3_lr10_pair | attn_mlp W lr 0.1, R paired | 3e-3 | 910 | 0.32 | 0.37 | 0.38 | 1.80 | 0.78 | 0.32 | no | 0.88 | 0.96 | 0.96 | 1.55 | 1.00 | 0.89 | **yes** |
| AM_W_3e-3_lr05_pair_800st | attn_mlp W lr 0.05, R paired, 800 steps | 3e-3 | 940 | 0.25 | 0.27 | 0.27 | 1.57 | 0.75 | 0.25 | no | 0.94 | 1.04 | 1.02 | 1.37 | 0.99 | 0.95 | **yes** |
| AM_W_1e-3_..._off1 | + off hinge ×1 | 1e-3 | 2592 | 0.24 | 0.26 | 0.26 | 1.27 | 0.74 | 0.25 | no | 0.98 | 1.06 | 1.06 | 1.12 | 1.00 | 0.99 | **yes** |
| AM_W_1e-3_..._off4 | + off hinge ×4 | 1e-3 | 2821 | 0.24 | 0.27 | 0.27 | 1.28 | 0.73 | 0.24 | no | 1.01 | 1.06 | 1.06 | 1.12 | 0.99 | 1.02 | **yes** |
| AM_W_1e-3_..._sq1 | + off squared ×1 | 1e-3 | 2577 | 0.20 | 0.24 | 0.24 | 1.27 | 0.71 | 0.20 | no | 0.93 | 1.03 | 1.03 | 1.13 | 0.99 | 0.94 | **yes** |
| AM_W_3e-3_..._off4 | + off hinge ×4 | 3e-3 | 1452 | 0.21 | 0.21 | 0.22 | 1.47 | 0.69 | 0.21 | no | 0.95 | 1.04 | 1.02 | 1.28 | 0.99 | 0.97 | **yes** |
| AM_WCF_1e-3_lr05 | attn_mlp W_CF lr 0.05, R random | 1e-3 | 4076 | 0.28 | 0.30 | 0.30 | 1.23 | 0.77 | 0.28 | no | 1.07 | 1.12 | 1.11 | 1.10 | 1.00 | 1.09 | **yes** |
| AM_U_1e-5_lr05_pair | attn_mlp U, R paired | 1e-5 | 28344 | 0.55 | 0.56 | 0.56 | 1.16 | 0.87 | 0.55 | no | 1.00 | 1.00 | 1.00 | 1.06 | 0.99 | 1.00 | **yes** |
| AM_U_1e-4_lr05_pair | attn_mlp U, R paired | 1e-4 | 8640 | 0.33 | 0.34 | 0.34 | 1.22 | 0.78 | 0.33 | no | 0.98 | 0.99 | 1.00 | 1.08 | 1.00 | 0.99 | **yes** |
| AM_RND_1e-2_d0 | RND matched to 493 | — | 493 | 0.04 | -0.03 | -0.02 | 0.78 | 0.52 | 0.01 | no | 0.01 | -0.00 | 0.01 | 0.78 | 0.49 | 0.01 | no |
| AM_RND_1e-2_d1 | RND matched to 493 | — | 493 | 0.01 | -0.01 | -0.01 | 0.20 | 0.49 | 0.00 | no | -0.00 | -0.00 | 0.00 | 0.15 | 0.49 | -0.01 | no |
| AM_RND_3e-3_d0 | RND matched to 1,361 | — | 1361 | 0.01 | -0.02 | -0.01 | 0.83 | 0.49 | 0.01 | no | -0.01 | -0.02 | -0.02 | 0.84 | 0.48 | -0.01 | no |

The j6 rows (the 5,000-item rescore and the 117k random null) are in the j6 section above.

## What was run (commands, WSL, repo root)

```bash
PYTHONPATH=src .venv/bin/python -X utf8 experiments/133-ioi-wcm-sweep/make_sn.py          # CPU: tasks/ioi_sn.json
.venv/bin/python experiments/133-ioi-wcm-sweep/inspect_train.py                           # CPU: 081 per-term curves
.venv/bin/python experiments/133-ioi-wcm-sweep/fire_share.py                              # CPU: dead-on-held-out members
# GPU jobs (one at a time, under REPO/.gpu-lock, watchdog GPU > 15.3 GB / RAM < 3 GB, STOP 07:30, hard 08:10):
EXTRA_ENV='N_DIS=500 N_SN=500 N_TR=300' bash experiments/133-ioi-wcm-sweep/job133.sh experiments/133-ioi-wcm-sweep/configs/j1_diag.json j1_diag
EXTRA_ENV='N_DIS=500 N_SN=500' bash experiments/133-ioi-wcm-sweep/job133.sh experiments/133-ioi-wcm-sweep/configs/j2_lam.json j2_lam   # cut after 2 configs
EXTRA_ENV='N_DIS=500 N_SN=500' bash experiments/133-ioi-wcm-sweep/job133.sh experiments/133-ioi-wcm-sweep/configs/j3_am.json  j3_am
EXTRA_ENV='N_DIS=500 N_SN=500' bash experiments/133-ioi-wcm-sweep/job133.sh experiments/133-ioi-wcm-sweep/configs/j4_off.json j4_off
EXTRA_ENV='N_DIS=500 N_SN=500' bash experiments/133-ioi-wcm-sweep/job133.sh experiments/133-ioi-wcm-sweep/configs/j5_rnd.json j5_rnd
EXTRA_ENV='N_DIS=5000 N_SN=998' bash experiments/133-ioi-wcm-sweep/job133.sh experiments/133-ioi-wcm-sweep/configs/j6_u.json j6_u
.venv/bin/python experiments/133-ioi-wcm-sweep/make_table.py                              # CPU: out/table.md
```

**Files:**
- `run133.py`: the runner; a config is either a fit or a rescore.
- `engine133.py`: the in-memory engine patch:
  - paired R donors;
  - the task off-term (hinge or squared, Z frame);
  - the final-gate stash for `soft_eval`.
- `job133.sh`: the lock and watchdog wrapper.
- `configs/*.json`: the job config lists.
- Outputs:
  - `out/results.jsonl`: one row per config, with the full scorer fields under the `dis_` / `sn_` prefixes, the
    per-term training errors (`terms_last40`), the off-term stats (`p133`), `soft`, `n_by_kind` and timings;
  - `out/circuits/<name>.json`: the circuits;
  - `out/logs/`, `out/watch.csv`.
- Runtime: attn+mlp fits ~130 s (~150 s with the off term); 48-site ~230 s. 1,000 scored items + minimal pairs ~75 s.
  Peak 10.6–11.4 GB.

## Caveats

- **The registered rule passes only at 117k latents** (70% of the 164k train-active attn+mlp latents), near the
  band's lower edge: Z 0.78 against the rule's 0.75, on 5,000 items.
  - The matched random null (0.60) fails, but there is one draw and the margin is 0.18.
  - 132's all-active α = 1 circuit reaches 0.85 on the same scope. So this is a pass near the scope's capacity, not
    a small circuit.
  - At that size the same-name read does not discriminate (random passes it).
- **The attn+mlp scope leaves the residual stream unedited,** so the residual stream carries name identity for free.
  These circuits explain what the attention and MLP blocks write, not the full computation. On the 48-site scope no
  circuit can pass the disjoint-name rule (ceiling 0.23).
- **The same-name read is not the registered rule.** It answers "does the circuit carry the task for the names it
  was fitted on", the generalisation calibration also measured. It does not test transfer to unseen tokens.
- **No replicates:** the presentation order is seeded by the task name, so every config is one fit. Neighbouring λ
  values give a sense of fit noise: monotone in dis Z on attn+mlp, bimodal on 48 sites under v2 (081).
- **Necessity runs high on small circuits** (1.4–1.9 at ≤ 1.5k latents): removing the members overshoots the empty
  circuit. The rule only requires ≥ 0.9.
- **lr:** v4's ×3.15 lr multiplier collapses this behavioural fit at 48 sites; 0.1 works on attn+mlp; 0.1575 was not
  retested on attn+mlp. A per-objective lr calibration is the follow-up for 081 / arena D.
- **Not done (time):**
  - U at λ ≥ 1e-3 (claim 2's ratio);
  - RND at the mid sizes;
  - a structure / head check of the passing circuits;
  - greater-than.
- 500-item reads have a standard error of roughly ±0.01–0.02 on Z.
