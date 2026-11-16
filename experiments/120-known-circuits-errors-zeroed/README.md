# 120 — Known-circuit check on GPT-2 small with the SAE errors zeroed (DAN-215 follow-up)

**Question.** 117 found that WCM v4 circuits for 15 GPT-2 latents on the published IOI and greater-than pathways keep
the published *positions and stages* but not the published *heads* (1 / 66 head-class checks). It named the live SAE
errors as the likely cause. If the same 15 targets are refitted and scored with every upstream SAE error zeroed (116's
regime), do the circuits recover the published heads?

**Answer: no, and the test cannot answer the question on GPT-2. Zeroing the errors removes the published mechanism
from the model itself.**

- **Head classes:** 3 / 66 checks pass (117: 1 / 66). Two of the three are Hanna-head checks (head 8.11 at END) with a
  null of exactly zero, in circuits that fail faithfulness. This is not a recovery.
- **The positive control fails in the error-free model.** With every upstream error zeroed and every latent live:
  - the full model's own attribution ranks a published head first for **3 / 15** targets, and **0 / 15** at that head's
    role position. 117's live full model: 15 / 15, all at their role;
  - its published-head shares are near zero (S-inhibition 0.002-0.003 vs live 0.064-0.071; Hanna heads 0.002 vs
    0.098-0.141);
  - the IO-position share falls from 0.21 to 0.03-0.05.
- **The error-free model does not compute the targets on the task prompts.** The target's pre-activation on the 128
  held-out task prompts turns negative for **14 / 15** targets. Median ratio to the live full model: −0.26 (for example
  9.attn.25821: 18.05 → −5.13). The one exception is 5.attn.27767 (0.31).
- **Positions and stages fall too:** 6 / 32 checks recovered (117: 17 / 32). IO position 3 / 6 (5 / 6), S2 position
  0 / 6 (4 / 6), late MLPs at END 0 / 8 (5 / 8).
- **Faithfulness is worse in its own regime** (fitted and scored with errors zeroed), and near zero once the errors are
  restored:
  - production (corpus) circuits pass on held-out text for **4 / 15** targets (117: 7 / 15) and on 10 / 60 task folds
    (16 / 60);
  - task circuits pass on 20 / 60 folds (117: 42 / 60);
  - 13 / 30 error-zeroed fits score Z = 0 on held-out text (117: 2 / 30).
- **Verdicts** (pre-stated rule): IOI-NM PARTIAL (corpus circuits: IO position) / NOT RECOVERED (task circuits). IOI-SI,
  IOI-IND and GT-MLP are NOT RECOVERED. In 117, every pathway except IND was PARTIAL.

**Reading.** On GPT-2 small with the OAI v5 SAEs, the published IOI and greater-than computation runs largely through
the SAE error terms. Zeroing them at every upstream site removes the targets' task activation and the published-head
attribution, before any circuit is fitted. This supports 117's explanation in a stronger form: the circuits miss the
heads because those heads' contribution is mostly not in the dictionary. It also means the error-free refit is not a
known-answer test on GPT-2. There is no known circuit left to recover in that model.

## What ran (2026-11-15, local RTX 5070 Ti)

- **GPU jobs:** fit jobs 1-4 (job 4 16:00-16:21 after a PC restart; job 3 stopped by hand at 14:54), structure
  16:21-17:12, diagnostic 17:12-17:32. Logs are in `out/logs/`.
- **Fits** (`fit120.py`): 117's `fit117.main` unchanged.
  - Same records (corpus: 117's `out/wcm/ctx`, read-only; task: copies of 117's `task_ctx`), same targets, same
    recipe (WCM v4, λ 1e-3, 400 steps).
  - Training: 116's `install_train_error_ablation`, taken verbatim from `refit_err.py` by AST. Every masked site trains
    on decode(code) alone.
  - Scoring: 116's `errpatch`. Every job is scored with errors zeroed (the primary row) and with errors live (the
    `live` field).
- **Structure** (`structure120.py`): 117's `structure117.main` unchanged.
  - The attribution frame is wrapped (`common120.install_struct_error_zero`) so every patched upstream site outputs
    decode(edited code) with no error.
  - Circuit, 20 random matched nulls, and the **error-free full model** (every latent live, every upstream error zero).
  - 128 held-out task prompts.
- **Diagnostic** (`diag120.py`): 117's original errors-live circuits re-scored with errors zeroed, no refit (116 §1 on
  these targets).
  - The live re-score in this process equals 117's own rows exactly (max |Δ| 0.0000 on Z/A/C/nec), which checks the
    wiring.
- **Report:** `report120.py`.
  - Runs 117's `report117.main` on 120's rows (`out/report.md` / `.json`), then the paired tables (`out/report120.md` /
    `.json`).
  - One guard was added to `117/report117.py`: the descriptive top-heads table skips a frame with zero attention mass,
    which only a collapsed error-zeroed circuit has. 117's own output is unchanged.

## Faithfulness (production scorer, pass = Z, A, C in [0.75, 1.5] and necessity ≥ 0.9)

| circuit | regime | held-out text pass | task folds pass | targets with mean task Z in band | median text Z |
|---|---|---|---|---|---|
| corpus | fitted + scored errors zeroed | 4 / 15 | 10 / 60 | 5 / 15 | 0.50 |
| corpus | fitted errors zeroed, scored live | 1 / 15 | 0 / 60 | 2 / 15 | 0.00 |
| corpus | 117 (fitted + scored live) | 7 / 15 | 16 / 60 | 9 / 15 | 0.76 |
| task | fitted + scored errors zeroed | 1 / 15 | 20 / 60 | 5 / 15 | 0.00 |
| task | fitted errors zeroed, scored live | 0 / 15 | 0 / 60 | 0 / 15 | 0.00 |
| task | 117 | 3 / 15 | 42 / 60 | 12 / 15 | 0.40 |

**Paired by target** (text pass):
- Of 117's 7 passing corpus circuits, 4 also pass error-zeroed: 9.attn.25821, 9.attn.28187, 8.attn.18439, 5.attn.27767.
- One error-zeroed task circuit passes where 117's failed (9.attn.28187).
- No other error-zeroed circuit passes.

**Size.** Median members: corpus 279 vs 574, task 239 vs 336. The MLP targets shrink most: 34-247 members vs 136-1,401.

**Diagnostic (117's circuits, errors zeroed at scoring only):**
- corpus circuits: text pass 7 / 15 → 1 / 15, task folds 16 / 60 → 1 / 60;
- task circuits: text 3 / 15 → 3 / 15, task folds 42 / 60 → 13 / 60.

This matches 116 §1's GPT-2 drop (75% → 33%, and L8-11 50% → 5%).

## Structure (128 held-out task prompts; shares as defined in 117)

| check group | error-zeroed circuits | 117 circuits |
|---|---|---|
| head classes (all) | 3 / 66 | 1 / 66 |
| head classes, corpus / task circuits | 2 / 33, 1 / 33 | 0 / 33, 1 / 33 |
| positions / stages (all) | 6 / 32 | 17 / 32 |
| IO position (NM) | 3 / 6 | 5 / 6 |
| S2 position (SI) | 0 / 6 | 4 / 6 |
| YY position (GT) | 3 / 12 | 3 / 12 |
| late MLPs at END (GT) | 0 / 8 | 5 / 8 |

- **The three head "recoveries":**
  - duplicate-token heads at S2 for the task circuit of 8.attn.18439: 0.122 vs null 0.034, error-free full model
    0.016. This circuit fails faithfulness (text Z 0.00; task Z 0.86 with 4 / 4 folds);
  - Hanna heads at END for the corpus circuits of 8.mlp.11468 (0.37) and 9.mlp.3600 (0.52). Both come almost entirely
    from head 8.11 at END, the top head of 3 / 6 error-zeroed GT corpus circuits. The null mean is exactly 0, and the
    error-free full model's share is 0.002. Both circuits fail on text (Z 0.50, 0.29); 8.mlp.11468 is task-faithful
    (mean task Z 1.13, 0 / 4 folds pass).
  - A circuit share 200× above the full model's, for a target that model no longer computes, is an artefact of fitted
    amplitudes. It is not evidence for the published mechanism.
- **In 53 / 66 head checks the circuit's share is below the error-free full model's.** That full model's shares are
  themselves near zero.

### Error-free full model vs live full model (positive control)

| | error-free full model | live full model (117) |
|---|---|---|
| a published head ranked first | 3 / 15 (all head 0.1 at a non-role position) | 15 / 15 |
| ... at its role position | 0 / 15 | 15 / 15 |
| S-inhibition share at END (NM targets) | 0.002-0.003 | 0.064-0.071 |
| Hanna heads at END (GT) | 0.002-0.003 | 0.098-0.141 |
| IO-position share (NM) | 0.028-0.045 | 0.207-0.214 |
| target pre-activation, ratio to live | median −0.26; negative for 14 / 15 | 1 |

The error-free full model's top head is always 0.1, 0.3 or 0.11, and never a published head at its role position. Target
pre-activations are the `pre` field of the `full` frame in the two `structure.jsonl` files (mean over the 128 prompts).

| target | live pre | error-free pre | | target | live pre | error-free pre |
|---|---|---|---|---|---|---|
| 9.attn.25821 | 18.05 | −5.13 | | 8.mlp.11468 | 1.71 | −1.87 |
| 9.attn.28187 | 18.09 | −4.72 | | 8.mlp.26954 | 2.49 | −1.82 |
| 9.attn.27039 | 17.73 | −5.65 | | 9.mlp.15847 | 1.90 | −0.46 |
| 8.attn.16224 | 5.25 | −1.35 | | 9.mlp.3600 | 2.53 | −2.20 |
| 7.attn.29964 | 2.50 | −1.58 | | 10.mlp.3774 | 1.05 | 0.03 |
| 8.attn.18439 | 2.44 | −1.49 | | 10.mlp.4386 | 2.44 | −0.48 |
| 5.attn.25375 | 5.64 | −1.23 | | | | |
| 5.attn.28128 | 5.16 | −0.18 | | | | |
| 5.attn.27767 | 4.78 | 1.48 | | | | |

## Why this differs from 116's TuringLLM pilot (72% vs 100% pass)

The preliminary report already showed the error-zeroed GPT-2 circuits passing less often than 117's, with many
collapsing to Z ≈ 0. That is the opposite of what 116's TuringLLM pilot suggested. Three cheap checks, all from rows
already logged:

1. **The error-free GPT-2 model does not compute these targets** (table above).
   - In 116's `diag_recon`, TuringLLM attention targets kept a median 60% of their activation with every upstream error
     zeroed, and MLP / residual targets 96-98%.
   - Here, on task prompts, 14 / 15 targets go negative.
   - A circuit inside an error-free model that does not produce the target can only reach Z ≈ 1 through amplitudes,
     and the Z / A / C references remain the live model's activation. That makes Z = 0 collapses (13 / 30) expected.
   - GPT-2's OAI v5 attention and MLP SAEs leave 19-28% of variance unexplained mid-stack (053). The residual-stream
     errors compound across 5-10 upstream layers. The targets are deep (L5-L10) and task-frequent.
   - 116 §1 already showed GPT-2's L8-11 cell falling from 50% to 5% with errors zeroed.
2. **The fits converge on the training rows but generalise worse** (`train.jsonl`, 30 paired fits).
   - The final training loss is close to 117's on every fit (e.g. 18.84 vs 17.20, 10.49 vs 10.19). The initial loss is
     about 5 higher throughout.
   - The held-out data loss is higher than 117's in 21 / 30 fits, often several times higher: 8.attn.16224 corpus 163
     vs 16, 9.attn.27039 task 329 vs 4, 8.mlp.26954 task 24 vs 0.4.
   - So the optimiser is not failing. The error-free circuits fit the 48 training rows and do not transfer to held-out
     rows.
3. **The circuits are much smaller** (median 279 vs 574 corpus), which lowers necessity: removing a small circuit
   from the live model no longer silences the target. As in 116, where 3 of the 5 failures were on necessity alone,
   13 / 30 error-zeroed rows have held-out text necessity < 0.9 (117: 0 / 30; all GT task circuits ≤ 0.41).

The 116 pilot drew 18 random targets from TuringLLM, whose SAEs explain 0.85-0.99 of the variance. These are 15 deep
GPT-2 targets selected for being on circuits that, by this test, run mostly through the errors. The two results do not
conflict.

## Caveats

- 15 targets, one λ, one 400-step budget; L11 MLPs excluded (as 117).
- Errors are zeroed at **every** upstream site at once, residual sites included. This is a stricter regime than SFC,
  where errors are nodes a circuit may keep. An error-*node* variant (errors as optional members) would be the
  informative next test on GPT-2, but it is not run here.
- The training patch is 116's in-process monkeypatch (not reviewed line by line by a second reader; see 116's caveats).
- The attribution is first-order and decomposes a pre-activation; in the error-free frames that pre-activation is
  mostly negative, so shares are shares of the mass of a target that is not active.
- The error-free pre ratios are task-prompt means. They are not measured on the corpus text the corpus circuits were
  fitted on, where the error-zeroed corpus circuits reach a median Z of 0.50.

## PROPOSED paper text (not applied; `paper/` untouched)

**`app:known-circuits`.** Replace the sentence

> A likely cause is that circuits act with the SAE errors live (\cref{app:error-ablation}): GPT-2's attention and MLP
> dictionaries leave $19$--$28\%$ of the variance unexplained mid-stack, so a circuit can reproduce its target without
> the latents that carry the published heads.

with

> A likely cause is that the published computation runs largely through the SAE errors, which every circuit keeps live
> (\cref{app:error-ablation}): GPT-2's attention and MLP dictionaries leave $19$--$28\%$ of the variance unexplained
> mid-stack.
> Zeroing the error at every upstream site removes the published structure from the model itself: on the same
> held-out task prompts the error-free model's pre-activation turns negative for $14$ of the $15$ targets, its own
> attribution ranks a published head first for $3$ of $15$ (none at the head's role position), and its share on the
> IO token falls from $21\%$ to $3$--$5\%$.
> Circuits refitted and scored in that error-free regime do not recover the heads ($3$ of $66$ checks, two of them in
> circuits that fail the pass rule) and keep fewer of the positions ($6$ of $32$ checks, against $17$), so the
> error-free refit is not a known-answer test on GPT-2.

Keep the two sentences after it ("The attribution is first-order ..." and "A weighted circuit is therefore not a
head-level circuit-discovery method; ...").

**Limitations, "A circuit is a fitted intervention"** (`sec:limitations`). The sentence ending "...circuits keep the
published positions and stages but not the published heads, which the full model's own attribution ranks first
(\cref{app:known-circuits})." can stay as is. Optionally append:

> ; with the SAE errors zeroed, the model itself no longer ranks them first.

**`app:error-ablation`, optional.** After "...and pass for none of the targets once the errors are restored." add:

> On $15$ deep GPT-2 latents chosen on published circuits, the error-free refit passes on held-out text for $4$
> (against $7$ with errors live), because the error-free GPT-2 model no longer produces most of these targets
> (\cref{app:known-circuits}).

## Files

| path | what |
|---|---|
| `common120.py` | paths; 116's training error patch (AST-loaded); error-zeroed attribution wrapper |
| `fit120.py` | refit + dual scoring -> `out/wcm/{main,task}/circuits`, `out/wcm/dan215/{eval,status,train}.jsonl` |
| `structure120.py` | error-free structure frames -> `out/wcm/dan215/structure.jsonl` |
| `diag120.py` | 117's circuits scored with errors zeroed -> `out/diag/eval.jsonl` |
| `report120.py` | `out/report.md` / `.json` (117's report on 120 rows), `out/report120.md` / `.json` (paired) |
| `test_120_cpu.py` | CPU tests (3 / 3) |
| `RESUME.md` | resume notes from the PC restart |
| `out/GPU_DONE` | written 17:33 when the GPU work finished |

Run (WSL, repo root): the four commands in `RESUME.md`, steps 1-3 and 5.
