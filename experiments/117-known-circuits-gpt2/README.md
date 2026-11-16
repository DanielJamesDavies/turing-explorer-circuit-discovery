# 117 — Known-circuit check on GPT-2 small (DAN-215)

**Question.** Do WCM v4 circuits for SAE latents that sit on GPT-2 small's published circuits recover those circuits'
known upstream structure? The reference circuits are indirect object identification (IOI; Wang et al. 2022) and
greater-than (Hanna et al. 2023).

**Answer: partly recovered, at the level of positions and stages, not heads.**

- **Positive control.** On the same prompts, the full model's own attribution ranks a published head first for all
  15 targets: 5.5, 8.6, 3.0 or 4.11 for the IOI targets; 6.9 or 9.1 for the greater-than targets.
- **Heads.** The WCM circuits do not keep the published heads: **1 of 66** head-class checks pass the pre-stated rule
  (0 / 33 for the production circuits).
- **Positions and stages.** What they do keep, above a random matched null:
  - the position the information comes from: the IO token for name-mover latents (5 / 6 circuits) and S2 for
    S-inhibition latents (4 / 6);
  - for greater-than, the late-MLP-at-END stage (5 / 8 evaluable).
  - At the published positions, their attention mass sits on early-layer heads (L0-L2) and on local, token-level
    features instead.
- **Verdict per pathway** (pre-stated rule): IOI-NM PARTIAL, IOI-SI PARTIAL, IOI-IND NOT RECOVERED, GT-MLP PARTIAL.
  No pathway is RECOVERED.

## What ran (2026-11-15, local RTX 5070 Ti, three GPU jobs: 03:48-05:05, 06:24-07:26, 08:48-09:34)

- **15 targets**, chosen by the rule below: 3 IOI name-mover (NM) latents, 3 S-inhibition (SI), 3 induction (IND)
  and 6 greater-than MLP latents (GT).
- **Two WCM v4 circuits per target**, both through 062's production `driver.Runner` (lambda 1e-3, 400 steps, v4 =
  Z + A + C + R_C, rank-keep 1e-1, geometric lr x3.15 with warm-up, censored term; `fit117.py`):
  - **corpus** = the production circuit: the target's own record built from the production stores (OpenWebText, 4M
    windows, group-aware split), exactly as the pod run fitted its 512 GPT-2 targets;
  - **task** = the same method on a task record (`lib117.task_record`):
    - 48 strongest + 16 mid-band training-split prompts (arm B: 32 + 16), held-out prompts from the disjoint
      name / noun half;
    - anchors at END (S2 for IND);
    - each prompt followed by EOS + a corpus window so rows are 64-token OpenWebText-like windows;
    - contrast = the task's own counterfactual prompts (IOI ABC, greater-than XX01), verified silent at every
      position, topped up with silent corpus windows where needed (5 targets; see the caveats).
- **Scoring:** the production scorer on held-out text (the record's held-out strongest / mid-band) and on held-out
  task prompts (4 folds of 16 strongest + 1 mid fold).
- **Structure:** `structure117.py` on 128 held-out task prompts; null = 20 random matched draws; plus the full model.
- **Report:** `report117.py` (`out/report.md`, `out/report.json`).

## Targets

| pathway | target | selection stats |
|---|---|---|
| IOI-NM | 9.attn.25821 / 9.attn.28187 / 9.attn.27039 | per-name latents (IO = Scott / Mark / Ryan): fire on 100% of their name's ENDs, 1-4% of other names'; name-mover drive 0.84-0.86 |
| IOI-SI | 8.attn.16224 / 7.attn.29964 / 8.attn.18439 | S-inhibition drive 0.75 / 0.73 / 0.57; fire 1.00 / 0.55 / 0.92 |
| IOI-IND | 5.attn.25375 / 5.attn.28128 / 5.attn.27767 | induction drive 0.83 / 0.86 / 0.81 at S2 |
| GT-MLP | 8.mlp.11468, 8.mlp.26954, 9.mlp.15847, 9.mlp.3600, 10.mlp.3774, 10.mlp.4386 | \|rho(act, YY)\| 0.59-0.70 at END |

Name-mover outputs are name-specific SAE latents. The all-items rule found none, because no L9/L10 attn latent fires
on >= 50% of IOI ENDs with any name-mover drive. This matches 048's Gemma finding of one END latent per name.

## Faithfulness (production scorer; Z / A / C on held-out strongest text; task = mean Z over 4 held-out task folds, folds passing)

| pathway | target | selection | corpus n | corpus text Z/A/C pass | corpus task Z (folds) | task n | task text Z/A/C pass | task task Z (folds) |
|---|---|---|---|---|---|---|---|---|
| IOI-NM | 9.attn.25821 | IO=Scott, share 0.85, fire 1.00 | 497 | 0.83/0.86/0.88 P | 0.69 (0/4) | 323 | 1.09/0.88/0.99 P | 0.92 (2/4) |
| IOI-NM | 9.attn.28187 | IO=Mark, share 0.86, fire 1.00 | 515 | 0.88/0.80/0.77 P | 0.69 (0/4) | 296 | 1.28/0.52/0.46 - | 0.99 (4/4) |
| IOI-NM | 9.attn.27039 | IO=Ryan, share 0.84, fire 1.00 | 862 | 0.51/0.54/0.52 - | 0.42 (0/4) | 442 | 1.05/0.19/0.24 - | 0.98 (3/4) |
| IOI-SI | 8.attn.16224 | share 0.75, fire 1.00 | 578 | 0.95/0.59/0.87 - | 1.48 (2/4) | 450 | 0.15/0.01/0.27 - | 0.85 (3/4) |
| IOI-SI | 7.attn.29964 | share 0.73, fire 0.55 | 945 | 0.74/0.89/0.90 - | 0.91 (0/4) | 515 | 0.75/0.85/0.78 P | 0.94 (4/4) |
| IOI-SI | 8.attn.18439 | share 0.57, fire 0.92 | 574 | 0.91/0.95/0.92 P | 0.69 (0/4) | 355 | 0.98/0.71/0.96 - | 0.90 (4/4) |
| IOI-IND | 5.attn.25375 | share 0.83, fire 1.00 | 475 | 0.80/0.83/0.77 P | 1.08 (4/4) | 336 | 0.36/0.26/0.29 - | 0.93 (4/4) |
| IOI-IND | 5.attn.28128 | share 0.86, fire 0.68 | 687 | 0.70/0.77/0.64 - | 0.68 (0/4) | 297 | 0.86/0.78/0.79 P | 0.93 (4/4) |
| IOI-IND | 5.attn.27767 | share 0.81, fire 0.98 | 478 | 0.86/0.88/0.91 P | 1.11 (4/4) | 219 | 0.40/0.54/0.24 - | 0.87 (4/4) |
| GT-MLP | 8.mlp.11468 | rho 0.65, fire 0.67 | 430 | 0.76/0.85/0.83 P | 0.98 (0/4) | 772 | 0.19/0.15/0.00 - | 0.89 (2/4) |
| GT-MLP | 8.mlp.26954 | rho 0.59, fire 0.95 | 692 | 0.87/0.93/0.75 - | 1.00 (2/4) | 1401 | 0.18/0.91/0.39 - | 0.93 (4/4) |
| GT-MLP | 9.mlp.15847 | rho 0.67, fire 0.70 | 398 | 0.71/0.88/0.77 - | 1.30 (4/4) | 136 | 0.00/-0.03/0.00 - | 0.00 (0/4) |
| GT-MLP | 9.mlp.3600 | rho 0.59, fire 0.79 | 762 | 0.26/0.12/0.22 - | 0.76 (0/4) | 620 | 0.05/0.15/0.09 - | 0.89 (4/4) |
| GT-MLP | 10.mlp.3774 | rho 0.70, fire 0.60 | 737 | 0.76/0.85/0.79 P | 0.84 (0/4) | 171 | 0.00/0.00/0.00 - | 0.00 (0/4) |
| GT-MLP | 10.mlp.4386 | rho -0.65, fire 1.00 | 570 | 0.32/0.88/0.79 - | 0.00 (0/4) | 304 | 0.40/0.24/0.31 - | 0.69 (0/4) |

Necessity is >= 0.86 on every row (in `out/report.md`).

- **Production circuits:**
  - pass on held-out text for **7 / 15** targets (median size 574, range 398-945; median Z 0.76). For comparison,
    the pod run's GPT-2 pass rate on 512 random targets is 82.4%, so these deep, task-frequent targets are harder;
  - mean held-out task Z in [0.75, 1.5] for 9 / 15 targets; 16 / 60 task folds pass.
- **Task circuits** (median 336 members):
  - pass on 42 / 60 task folds, and their mean task Z is in band for 12 / 15;
  - they transfer poorly to text (3 / 15 pass);
  - 2 fits collapsed (9.mlp.15847 and 10.mlp.3774: Z = 0 on both data sets), an optimiser failure at this budget.

## Structure (128 held-out task prompts; head classes = share of attention-site mass, positions = share of all mass)

Medians over targets: the production (corpus) circuit, the task circuit, the full model, and the uniform head base
rate.

| check | targets | corpus circuit | task circuit | full model | uniform base | recovered (corpus + task) |
|---|---|---|---|---|---|---|
| S-inhibition heads at END (NM targets) | 3 | 0.030 | 0.033 | 0.070 | 0.037 | 1 / 6 |
| induction heads at S2 | 6 | 0.000 | 0.000 | 0.054 | 0.039 | 0 / 12 |
| duplicate-token heads at S2 | 9 | 0.003 | 0.003 | 0.027 | 0.031 | 0 / 18 |
| previous-token heads at S1+1 | 9 | 0.000 | 0.000 | 0.007 | 0.021 | 0 / 18 |
| Hanna heads at END (GT) | 6 | 0.005 | 0.007 | 0.114 | 0.065 | 0 / 12 |
| IO position (NM) | 3 | 0.889 | 0.526 | 0.208 | - | 5 / 6 |
| S2 position (SI) | 3 | 0.057 | 0.070 | 0.235 | - | 4 / 6 |
| YY position (GT) | 6 | 0.048 | 0.034 | 0.175 | - | 3 / 12 |
| late MLPs at END, share of MLP mass (GT L9-10) | 4 | 0.018 | 0.053 | 0.108 | - | 5 / 8 |

- **Head-level structure is not recovered.**
  - 1 / 66 head-class checks pass: the task circuit of 9.attn.25821, S-inhibition share 0.105 vs base 0.037 and
    full model 0.070.
  - In 65 / 66 checks the circuit's share of published heads is below the full model's, and usually below the uniform
    base.
  - The positive control works:
    - the full model's attribution puts a published head first for **15 / 15** targets: 5.5@S2, 8.6@END, 3.0@S2,
      4.11@S1+1 for IOI; 6.9@END, 9.1@END for greater-than;
    - its class shares are 1.4-1.9x the base rate for S-inhibition, induction and Hanna's heads;
    - duplicate-token and previous-token heads are at or below the base rate even in the full model.
  - The circuits' top heads are almost all unpublished heads: L0-L2 heads at non-role positions, plus 8.11 / 8.10 at
    END for the NM targets, 7.6 and 3.5. The top head is a published head of the target's own task in 1 / 15 corpus
    and 1 / 15 task circuits (both duplicate-token head 0.1 for 9.attn.27039, and neither at S2, where it acts).
- **Where the circuits do agree with the published account:**
  - name-mover circuits put 0.51-0.95 of their mass at the IO token in 5 / 6 circuits (null 0.07-0.10; full model
    0.21). The exception is 0.07, for the corpus circuit of 9.attn.27039, which fails faithfulness (Z 0.51 text, 0.42
    task). The circuit
    keeps the name the name movers copy and drops the query-side machinery that decides where to attend;
  - S-inhibition circuits are enriched at S2 over the null (0.06-0.13 vs 0.02-0.03), but well below the full model
    (0.20-0.25);
  - greater-than circuits at L9-L10 are enriched in END MLPs of layers 8+ (0.010-0.076 vs null <= 0.003), again below
    the full model (0.08-0.15).
- **Verdicts** (pre-stated rule). Restricting to circuits with mean held-out task Z in [0.75, 1.5] gives the same
  verdicts, except that no production NM circuit is task-faithful (all three have task Z <= 0.69):

  | pathway | production circuits | task circuits |
  |---|---|---|
  | IOI-NM | PARTIAL (IO position) | PARTIAL (IO position) |
  | IOI-SI | PARTIAL (S2 position) | PARTIAL (S2 position) |
  | IOI-IND | NOT RECOVERED | NOT RECOVERED |
  | GT-MLP | PARTIAL (late MLPs) | PARTIAL (late MLPs) |

## Interpretation and caveats

- **SAE errors are kept live in the circuit frame** (the production Z ablation). GPT-2's attention-out and MLP-out SAEs
  have FVU 0.19-0.28 mid-stack (053), so much of each head's output runs through the error term. A circuit can then
  reproduce its target without carrying, as members, the latents that route the published heads' outputs. DAN-214
  (exp 116) found the same dependence on errors: with every upstream error zeroed, the public models' pass rates fall
  from 83% to 50% and from 75% to 33% (paper `sec:limitations`). This is the most likely
  reason the head structure is missing while the content positions remain. **It means the result is a statement
  about WCM circuits as fitted (an intervention that holds with the errors live), not evidence that GPT-2 lacks the
  published circuit.**
- **The attribution is first-order** (d pre / d gain at gain 1, in each frame). It decomposes a pre-activation, not a
  downstream effect, and it is not a head ablation.
  - Head shares split each attn-out slot over heads by |exact head drive|.
  - Shares across site kinds are not additive (residual latents re-decompose the stream), hence the
    attention-mass and MLP-mass denominators.
  - The full model's head shares are themselves modest (1.4-1.9x base), so this is a weak-signal test even for a
    perfect circuit. That is why the full-model column is given beside every check.
- **The random matched null is weak.** Random member sets drive the target's pre-activation negative, so the null's
  attribution mass is small and its shares are noisy. The base-rate condition and the full-model column guard
  against reading too much into "x null".
- **Amendments, all flagged above:**
  - production-frame eligibility (before selection);
  - the per-name NM rule (after the first selection pass, before any fit);
  - the denominator change (after one target's structure rows).
  - The per-name NM targets use IO-substituted prompts (the target's name in the IO slot), so held-out prompts differ
    from training in S name, place, object and template draw, not in the IO name.
- **Records:**
  - task records append EOS + a corpus window to each prompt, so the A / C means include corpus text;
  - for 4 targets the counterfactual prompts were (almost) never silent. 8.attn.16224, 8.mlp.26954 and 10.mlp.4386
    have contrast sets made wholly of 64 silent corpus windows; 5.attn.27767 has 35 counterfactual prompts + 29 corpus
    windows (`task_contrast` in `status.jsonl`). The other 11 use counterfactual prompts only.
- **Scale and budget:**
  - 15 targets, 2 circuits each, one lambda (1e-3), one 400-step budget;
  - 2 / 15 task fits collapsed;
  - L11 MLPs were left out for memory;
  - no IOI backup or negative name movers were tested.
- **Not done:**
  - an error-zeroed (or error-pinned) refit of these targets, which is the decisive follow-up for the error
    explanation (exp 116's machinery);
  - a lambda sweep;
  - the output-endpoint circuits of 081 (they already reported S-inhibition / induction / duplicate-token
    enrichment 3.7-5.3x for an unfaithful IOI logit circuit, with a different metric: pre-activation head drive of
    members, not attribution).

## PROPOSED paper text (not applied; main.tex / tables untouched)

**Body, Limitations** (`sec:limitations`, the "A circuit is a fitted intervention" paragraph). Replace its last
sentence "It inherits any ambiguity in its latents, and is validated by intervention effects, not by a planted ground
truth." with:

> It inherits any ambiguity in its latents, and is validated by intervention effects, not by a planted ground truth: on
> 15 GPT-2 latents at the outputs of the published IOI and greater-than components, circuits keep the published
> positions (the IO name for name-mover latents) and, for greater-than, the late MLPs, but not the published heads
> (1 of 66 head-class checks), although the full model's own attribution ranks a published head first for every target
> (\cref{app:known-circuits}).

**New appendix** (after `app:error-ablation`, which it leans on):

> \section{Known-circuit check on GPT-2 small}\label{app:known-circuits}
> GPT-2 small has published circuits for indirect object identification~\citep{wang2023interpretability} and
> greater-than~\citep{hanna2023how}, which gives a known-answer test of WCM's structure. By a rule fixed before any fit,
> we chose 15 OAI v5 Top-K latents on those pathways:
> three name-mover outputs (per-name latents at L9, as name movers write the IO name),
> three S-inhibition and three induction outputs (attention-out latents whose pre-activation is $\geq 57\%$ driven by
> the published heads, by an exact per-head decomposition), and
> six END-position MLP latents at L8--10 that track the year.
> Each was fitted with the production recipe on its corpus contexts (the production circuit) and, separately, on task
> prompts. The production circuits pass the pass rule on held-out text for 7 of 15 targets, and the task-fitted
> circuits on 42 of 60 held-out task folds.
> We attribute each target's pre-activation to every upstream Top-K slot on 128 held-out task prompts, within the
> circuit (the Z frame) and within the full model, split attention latents over heads, and compare shares with
> 20 random member sets matched per site and with the uniform head rate.
> The full model ranks a published head first for every target. The circuits do not: 1 of 66 head-class checks
> passes (share $\geq 2\times$ the null mean, above its 95th percentile, and $\geq 2\times$ the uniform rate), and
> the circuits' published-head shares lie below the full model's in 65.
> What the circuits keep are positions and stages. Five of six name-mover circuits put 51--95\% of their attribution on
> the IO token (random 7--10\%, full model 21\%), S-inhibition circuits are enriched at S2, and greater-than circuits at L9--10
> are enriched in late MLPs at END. Their attention mass sits on early-layer heads at the content tokens instead of
> on the routing heads.
> By the pre-stated verdict, IOI name-mover, S-inhibition and greater-than structure is partly recovered and induction
> structure is not recovered.
> A likely cause is that circuits act with the SAE errors live (\cref{app:error-ablation}). GPT-2's attention and MLP
> dictionaries leave 19--28\% of the variance unexplained mid-stack, so a circuit can reproduce its target without the latents that
> carry the published heads. A WCM circuit is therefore not a head-level circuit discovery method; it identifies
> which features and positions a latent reads.

Bib entries needed (not in `references.bib`):
- `wang2023interpretability`: Wang, Variengien, Conmy, Shlegeris, Steinhardt, "Interpretability in the Wild: a Circuit
  for Indirect Object Identification in GPT-2 Small", ICLR 2023;
- `hanna2023how`: Hanna, Liu, Variengien, "How does GPT-2 compute greater-than?", NeurIPS 2023.

## Files

| path | what |
|---|---|
| `lib117.py` | env for the production driver, published components, task records, attribution |
| `head_drive_lib.py` | exact per-head decomposition for Top-K slots |
| `select_targets.py` | the selection rule -> `out/targets.json`, `out/targets.txt` (`out/targets_order.txt` = round-robin fit order) |
| `fit117.py` | corpus + task WCM v4 fits and production scoring -> `out/wcm/{main,task}/circuits`, `out/wcm/{ctx,task_ctx}`, `out/wcm/dan215/{eval,status,train}.jsonl` |
| `structure117.py` | attribution in circuit / null / full frames -> `out/wcm/dan215/structure.jsonl` |
| `report117.py` | tables and verdicts -> `out/report.md`, `out/report.json` |
| `smoke_cpu.py`, `smoke_record_cpu.py`, `test_117_cpu.py` | CPU checks |
| `out/logs/` | GPU job logs |
| `out/stores/` | the production data-pass stores, unpacked from the pod tarball (5 GB, regenerable; delete when done) |

`out/` is git-ignored, so the tables above are the record.

Run (WSL, repo root, `.venv`):
`PYTHONPATH=src python -X utf8 experiments/117-known-circuits-gpt2/select_targets.py`,
`TARGETS=experiments/117-known-circuits-gpt2/out/targets_order.txt BUDGET_MIN=70 PYTHONPATH=src python -X utf8 -u experiments/117-known-circuits-gpt2/fit117.py`,
`N_PROMPTS=128 N_DRAWS=20 PYTHONPATH=src python -X utf8 -u experiments/117-known-circuits-gpt2/structure117.py`,
`PYTHONPATH=src:experiments/062-h100-protocol-v1 python -X utf8 experiments/117-known-circuits-gpt2/report117.py`.

## Selection rule (written 2026-11-15 02:50, BEFORE any circuit was fitted)

Targets are OAI v5 32k Top-K SAE latents on GPT-2 small (`jbloom/GPT2-Small-OAI-v5-32k-*`). They are chosen from the
first 512 **training-split** items of 081's task files, using the published components only:

| pathway | site, position | class heads (published) | rule |
|---|---|---|---|
| IOI-NM | attn-out L9, L10 at END | name movers 9.6, 9.9, 10.0 | fires on >= 50% of items AND >= 50% of its pre-activation's head drive comes from the class heads (mean over the items it fires on); top 3 by fire rate x mean activation |
| IOI-SI | attn-out L7, L8 at END | S-inhibition 7.3, 7.9, 8.6, 8.10 | same |
| IOI-IND | attn-out L5, L6 at S2 | induction 5.5, 5.8, 5.9, 6.9 | same |
| GT-MLP | mlp-out L8, L9, L10 at END | (Hanna: MLPs 8-11 compute the comparison) | fires on >= 50% of items; top 2 per layer by \|Spearman rho(activation, YY)\| |

- Head drive is the exact per-head decomposition of the attn-out latent's pre-activation (081 `engine.head_drive`;
  `head_drive_lib.py`, checked exact in `test_117_cpu.py`).
- **Production frame eligibility** (added 02:30, still before any fit or selection run): a candidate must also be
  eligible under the pod run's target frame (082 `make_targets_public`, frame "fired"): it fires in >= 200 of the 4M
  corpus windows and in no more than 50% of them (the hub cap, from `latent_stats.seq_count`). Why: the CPU record
  smoke picked an attn L9 latent that fires on 98% of IOI ENDs and in every corpus window, so no verified-silent
  contrast context exists for it (the production run would have excluded it too).
- **IOI-NM per-name amendment** (added 02:45, after the first selection pass and BEFORE any circuit was fitted; it is
  flagged as an amendment).
  - What happened: the first pass returned NO IOI-NM target. Only 4 L9/L10 attn latents fire on >= 50% of IOI ENDs;
    all 4 are hub latents (> 50% of corpus windows), and none had more than 8% name-mover drive.
  - Why: name movers write the IO *name*, so their SAE outputs are name-specific (as 048 found in Gemma: one END latent
    per name), and an all-items fire rate cannot select them.
  - Amended rule, over all 1,600 training-split items grouped by IO name:
    - a latent qualifies for name n if the group has >= 8 items;
    - it fires at END on >= 50% of them and on <= 10% of the other names' items;
    - >= 50% of its END head drive comes from name movers 9.6 / 9.9 / 10.0 (fallback 0.3);
    - it is frame-eligible;
    - the top 3 by within-name fire rate x mean activation, with distinct names.
  - These targets' task records and structure prompts put the target's name in the IO slot of every prompt (the S name,
    places, objects and, for held-out, the held-out name half are unchanged).
- Fallback, fixed in advance: if a pathway has fewer than 3 eligible latents, the head-share bar drops to 0.3.
- L11 MLP is left out because the deepest GPT-2 targets do not fit the 16 GB card during the fit (107 arena A notes).
- The selection uses the TARGET's own head drive. That makes the target's own heads part of the selection, not of the
  test. The test is always about structure UPSTREAM of the target.

## Pre-stated test (written before any circuit was fitted)

For each circuit, attribution = d(target pre-activation at its anchor) / d(gain) at gain 1, for every active Top-K
slot at every upstream site. It is measured in the circuit's own Z frame (non-members 0, members at alpha x live
value, SAE error kept), on 128 held-out-split task prompts (names / nouns disjoint from training). Attention-out
slots are split over heads by |exact head drive|. Shares are of the total |attribution| mass.

- **Expected upstream classes** (only heads / MLPs upstream of the target are in scope; per pathway, written before any
  fit):
  - **IOI-NM:** S-inhibition at END; induction at S2; duplicate-token 0.1 / 0.10 / 3.0 at S2; previous-token 2.2 /
    4.11 at S1+1; name movers 9.6 / 9.9 at END (L10 targets only); the IO-position share (name movers copy from IO).
  - **IOI-SI:** induction at S2; duplicate-token at S2; previous-token at S1+1; the S2-position share (S-inhibition
    heads attend to S2).
  - **IOI-IND** (target at S2): duplicate-token at S2, previous-token at S1+1, and induction heads below the target's
    layer.
  - **GT-MLP:** Hanna's heads 5.1 5.5 6.1 6.9 7.10 8.8 8.11 9.1 at END; the YY-position share; the late-MLP-at-END
    share (L8-11 MLPs below the target; none for L8 targets).
  - Each check's share is mass(class components at the class position) / total |attribution| mass.
- **Denominator amendment** (06:30, after previewing ONE target's structure rows; flagged). As first written, every
  share was divided by the total |attribution| mass over all sites. That mass is dominated by residual-site latents
  (resid_mid / resid), which re-decompose the same stream that the attention and MLP outputs already decompose. Head
  shares came out at ~0.001 even in the full model, so a near-zero null made "recovered" trivially true. Amended:
  - a head class's share is now taken of the attention-site mass in scope (every head and position), because
    attn-out latents decompose each layer's attention output exactly over heads;
  - the late-MLP check is a share of MLP mass;
  - position checks stay shares of all mass;
  - recovery of a head class also needs share >= 2x the uniform head base rate (class heads / heads in the attention
    layers in scope; 081's T4 base), which removes the zero-null artefact.
  The fit, the selection and the attribution itself are unchanged.
- **Nulls.**
  - (a) **Random matched** (primary): 20 draws, with the same member count per site drawn from the latents active on
    the task's training prompts at that site, the circuit's alphas permuted within each site, scored in the same frame.
  - (b) **Full model**: every latent live, same attribution, for reference.
- **A class counts as recovered for a target** when the circuit's share is >= 2x the random-null mean AND above the
  null's 95th percentile.
- **Verdict per pathway:**
  - RECOVERED if, in a majority of the pathway's targets, a majority of the in-scope expected classes are recovered;
  - PARTIAL if at least one class is recovered in at least half of the targets;
  - NOT RECOVERED otherwise.

## Setup log

- 2026-11-15 02:15. Every input was already cached locally, so nothing was downloaded:
  - GPT-2 small: HF `gpt2`, 526 MB;
  - the four `jbloom/GPT2-Small-OAI-v5-32k-{attn-out,mlp-out,resid-mid,resid-post}-SAEs` repos: 2.3 GB each,
    12 layers each;
  - 081's task files (`experiments/081-task-circuits/tasks/gpt2-small/{ioi,greater_than}.json`);
  - the production data-pass stores: unpacked from the pod tarball
    `Runs/pod-results-2026-11-07/pub-gpt2-small-S10-stores.tgz` into `out/stores/` (data 2.1 GB, stores 2.9 GB;
    `seq_latent_index` skipped because nothing reads it).
- CPU smoke (`smoke_cpu.py`): model + attn-out L9 SAE + one IOI prompt, logit diff IO−S 3.64. PASS.
- CPU tests (`test_117_cpu.py`): exact head drive; full-model frame = natural pre-activation; all-members frame = full
  model; empty frame = zero mass. 3 passed.
