# 121 Error-zeroed refit at scale (DAN-214)

## Questions

116 §1 showed that fixed production circuits, fitted with the SAE errors live, mostly fail once every upstream SAE
error is zeroed (TuringLLM 86% → 35%). Its §1b pilot (18 TuringLLM targets) refitted the circuits with the errors zeroed
during training too: 72% [50, 89] passed in that regime, at about half the size (median 322 vs 584 latents).

1. At scale, and on all three models: when WCM is fitted **and** scored with every upstream SAE error zeroed, how often
   does the circuit pass, and how big is it?
2. How does that compare, paired by target, with the production circuit fitted and scored with errors live?
3. Does a circuit carry over between regimes (the error-zeroed circuit scored with errors live)?

Framing (Daniel's): a circuit is judged in the regime it was fitted in. Error-zeroed circuits that pass in their own
regime are a result (WCM also finds circuits in the error-free SAE model), not a caveat. The gap to errors-live is
reported as measured.

## Method (reuses 116; nothing re-implemented)

- **Fit.** `refit_scale.py` is 116 `refit_err.py` made family-aware. The production WCM v4 fit (062 `driver.Runner`:
  λ 1e-3, 400 steps, the production records and train arm), with 116's `install_train_error_ablation`: the learned-mask
  engine's masked forward subtracts each masked site's SAE error, so training sees decode(code) alone. Every row
  records `train_err_calls > 0` (4,845 per fit on TuringLLM and Pythia; 12,920–16,150 on GPT-2 small, which has four
  site kinds), so the patch was active in every fit.
- **Scoring.** The production scorer on the held-out strongest contexts, with 116 `errpatch` switching the error of
  every upstream site in every circuit-only run (Z / A / C, empty-circuit baselines, pins). Pass = the DAN-8 rule
  (Z, A, C in [0.75, 1.5], nec ≥ 0.9; vacuous = fail) via `109/lib109`. 95% bootstrap CIs over targets (4,000
  resamples); paired differences by `lib109.boot_paired_diff`.
- **Arms, per target.**
  - **(a)** refit with errors zeroed, scored with errors zeroed (the main arm);
  - **(b)** the production circuit (fitted errors live, on the pod) scored with errors live on this GPU = 116's
    `rs_<family>` live row (same scorer, records, code and GPU; not re-run);
  - **(c)** arm (a)'s circuit scored with errors live;
  - **ref** the production circuit scored with errors zeroed, no refit = 116 §1's zero row.
- **Targets** (`select_targets.py`, seed 20261121), drawn from 116's §1 lists, round-robin over depth band × site kind:
  TuringLLM 81 (9 per cell; 18 are the 116 pilot targets, whose zero-fit rows are reused from `116/out/refit_zero`, the
  other 63 new), Pythia-70m 36 (4 per cell × 9), GPT-2 small 36 (3 per cell × 12, four site kinds).
- **Hardware.** Local RTX 5070 Ti (16 GB), one GPU process at a time (`chain.sh`), 2026-11-15 17:33–22:46, plus 23:52–00:13 for the GPT-2 L8-9 leftovers.
  - GPT-2 small layer-10/11 fits do not fit in 16 GB with the train patch: `11.mlp.8451` ran 29 min at full VRAM
    and ~95 W (spilling to host memory; normal fits draw ~200 W) and was killed; `11.resid.5820` showed the same and
    was killed after 5 min. The rest were relaunched as `gpt2c` (log `out/logs/gpt2c.log`): layers 0–7 first, then
    8–9, layers 10–11 skipped. `9.resid.13727` was also memory-bound (773 s) but finished; the job then hit STOP_AT
    22:45. (The `end gpt2c` time in `chain.log` is wrong: the shell expanded it at launch. The real end was 22:46.)
  - Extra session, 2026-11-15 23:52 to 2026-11-16 00:13 (`gpt2d`, log `out/logs/gpt2d.log`), for the leftover L8-9
    targets: `8.mlp.30479` (325 s) and `9.attn.30928` (598 s, memory-bound) finished. `9.resid_mid.24473` started at
    about 00:11, past the 00:10 no-new-target cutoff, and was killed at 00:12 with no row written. The script's
    `STOP_AT` cannot cross midnight, so this job used `MAX_MIN=17`, which let that target start a minute late.
- **TuringLLM error-free-model context.** 116 `diag_recon`: the target's activation when every upstream site is replaced
  by its SAE reconstruction (all latents live), over its clean value. Targets with this ratio in [0.75, 1.5] are those
  the error-free SAE model still computes.

Run: `wsl ... STOP_AT=22:45 bash experiments/121-error-zeroed-refit-scale/chain.sh`, then
`python -X utf8 experiments/121-error-zeroed-refit-scale/report.py` → `out/report.md`, `out/paired_<family>.csv`.
Rows: `out/refit_<family>/rows.jsonl`; circuits `out/refit_<family>/main/circuits/`; logs `out/logs/`.

## Results

Every number comes from `out/report.md`, which `report.py` generates. Pass = the DAN-8 rule. Brackets are 95%
bootstrap CIs over targets; Δ is the paired difference in percentage points. Latents = circuit size.

### Targets completed

| model | planned | done | per cell | not done |
|---|---|---|---|---|
| TuringLLM | 81 | **81** (63 new + 18 pilot) | 9 in every band × kind cell | none |
| Pythia-70m | 36 | **36** | 4 in every cell | none |
| GPT-2 small | 36 | **28** | L0-4 and L5-7: 3 per kind (24); L8-11: 8.attn, 8.mlp, 9.attn, 9.resid | the 7 L10-11 targets (VRAM-bound on 16 GB) and 9.resid_mid.24473 (out of time) |

(GPT-2 was 26 targets before the extra session. Its earlier numbers are kept in the GPT-2 section.)

No row errored. No fit was rejected.

### Headline, per model

| model | n | (a) refit zeroed, scored zeroed | (b) production, scored live | Δ (a) − (b) | ref: production, scored zeroed (no refit) | Δ (a) − ref | (c) refit zeroed, scored live | median latents (a) / (b) | size ratio (a)/(b), median [IQR] |
|---|---|---|---|---|---|---|---|---|---|
| TuringLLM | 81 | **69.1% [59.3, 79.0]** | 81.5% [72.8, 88.9] | −12.3 [−24.7, 0.0] | 32.1% [22.2, 42.0] | +37.0 [24.7, 49.4] | 7.4% [2.5, 13.6] | 343 / 654 | 0.54 [0.40, 0.78] |
| Pythia-70m | 36 | **66.7% [50.0, 80.6]** | 77.8% [63.9, 91.7] | −11.1 [−27.8, 2.8] | 55.6% [38.9, 72.2] | +11.1 [−2.8, 25.0] | 27.8% [13.9, 41.7] | 166 / 227 | 0.71 [0.51, 0.91] |
| GPT-2 small | 28 | **71.4% [53.6, 85.7]** | 82.1% [67.9, 96.4] | −10.7 [−28.6, 7.1] | 35.7% [17.9, 53.6] | +35.7 [14.2, 57.1] | 14.3% [3.6, 28.6] | 280 / 364 | 0.90 [0.64, 1.28] |

Cell-weighted to production cell sizes, (a) / (b): TuringLLM 70.3 / 82.1%, Pythia 62.5 / 75.0%, GPT-2 77.4 / 78.1%.
The GPT-2 weighting is not reliable: its L8-11 band holds 4 targets, which carry that band's full weight.

Median Z / A / C / nec, (a): TuringLLM 0.91 / 0.93 / 0.91 / 1.00, Pythia 0.99 / 1.00 / 1.00 / 1.00, GPT-2 0.83 / 0.90 /
0.87 / 1.00. In arm (c), with the errors restored, the error-zeroed circuits mostly collapse: TuringLLM median
Z / A / C 0.00 / 0.00 / 0.00.

Pass agreement, (b) × (a):

| model | both pass | (b) only | (a) only | both fail |
|---|---|---|---|---|
| TuringLLM | 48 | 18 | 8 | 7 |
| Pythia-70m | 22 | 6 | 2 | 6 |
| GPT-2 small | 18 | 5 | 2 | 3 |

Failure modes:
- **TuringLLM (a):** 56 pass, 13 necessity only (nec 0.66–0.89, median 0.84), 11 faithfulness, 1 vacuous.
  Faithfulness failures are all undershoots (Z < 0.75 in 10, A in 8, C in 10); none overshoot.
- **TuringLLM (b):** 66 pass, 14 faithfulness, 1 necessity only. The error-zeroed circuits are about half the size,
  and necessity-only failures are their characteristic failure (13 vs 1).
- **Pythia (a):** 24 pass, 10 faithfulness (undershoots), 2 necessity only. **(b):** 28 / 7 / 1.
- **GPT-2 (a):** 20 pass, 7 faithfulness (undershoots), 1 necessity only. **(b):** 23 / 3 / 2.

### TuringLLM by depth band and site kind (n = 27 each)

| | (a) refit zeroed / zeroed | (b) production / live | Δ (a) − (b) | ref: production / zeroed | (c) refit zeroed / live | (a) nec-only fails | median latents (a) / (b) |
|---|---|---|---|---|---|---|---|
| L0-4 | 77.8% [63.0, 92.6] | 92.6% [81.5, 100] | −14.8 [−29.6, −3.7] | 55.6% | 22.2% | 5 | 268 / 358 |
| L5-7 | 74.1% [59.3, 88.9] | 88.9% [74.1, 100] | −14.8 [−33.3, 3.7] | 22.2% | 0.0% | 4 | 416 / 802 |
| L8-11 | 55.6% [37.0, 74.1] | 63.0% [44.4, 81.5] | −7.4 [−33.3, 18.5] | 18.5% | 0.0% | 4 | 413 / 1151 |
| attention | 55.6% [37.0, 74.1] | 74.1% [59.3, 88.9] | −18.5 [−40.7, 3.7] | 11.1% | 0.0% | 5 | 416 / 643 |
| MLP | 81.5% [66.7, 96.3] | 74.1% [59.3, 88.9] | +7.4 [−14.8, 29.6] | 55.6% | 7.4% | 4 | 315 / 483 |
| residual | 70.4% [51.9, 85.2] | 96.3% [88.9, 100] | **−25.9 [−44.4, −11.1]** | 29.6% | 14.8% | 4 | 343 / 802 |

Cells, pass (a) / (b) / ref of 9: L0-4 attn 6/8/1, mlp 7/8/7, resid 8/9/7; L5-7 attn 6/8/2, mlp 7/7/3, resid 7/9/1;
L8-11 attn 3/4/0, mlp 8/5/5, resid 4/8/0.

**Error-free SAE model (116 diag_recon).** 50 of the 81 targets keep their activation within [0.75, 1.5] of clean when
every upstream site is replaced by its SAE reconstruction (attention 10/27, MLP 18/27, residual 22/27). On those 50:
(a) **82.0% [70.0, 92.0]**, (b) 92.0% [84.0, 98.0], ref 48.0% [34.0, 62.0]; paired (a) − (b) −10.0 [−22.0, 0.0]. By
kind, within: attention 8/10 vs 10/10, MLP 16/18 vs 15/18, residual 17/22 vs 21/22. On the other 31 targets, which
the error-free model itself no longer computes: (a) 48.4% [32.3, 67.7], (b) 64.5% [48.4, 80.6]. So a good part of the
attention-target gap belongs to the error-free model, not to the fit.

**Against the 18-target pilot (116 §1b).** The pilot gave (a) 72.2% [50.0, 88.9] at median 322 latents, against 100%
at 584 latents for an errors-live refit on the same GPU. At scale (a) is 69.1% [59.3, 79.0] at 343 latents: the
same rate, with an interval half as wide. The pilot's 100% comparator does not scale: it was an errors-live *refit* of 18 targets, while
the scaled comparator (b) is the production circuit scored live, 81.5% here (83.3% on the pilot 18). No errors-live
refits were run beyond the pilot. The pilot's own 18 targets: (a) 72.2%, (b) 83.3%, errors-live refit 100%. The
gap the paper should quote is therefore about −12 points [−25, 0] against production, not −28 against a refit.

### Pythia-70m by band and kind (n = 12 each)

| | (a) | (b) | Δ (a) − (b) | ref | (c) | median latents (a) / (b) |
|---|---|---|---|---|---|---|
| L0-1 | 100% [100, 100] | 91.7% [75.0, 100] | +8.3 [0.0, 25.0] | 83.3% | 66.7% | 88 / 98 |
| L2-3 | 66.7% [41.7, 91.7] | 75.0% [50.0, 100] | −8.3 [−25.0, 0.0] | 66.7% | 16.7% | 166 / 244 |
| L4-5 | **33.3% [8.3, 58.3]** | 66.7% [41.7, 91.7] | −33.3 [−66.7, 0.0] | 16.7% | 0.0% | 433 / 712 |
| attention | 58.3% [33.3, 83.3] | 66.7% [41.7, 91.7] | −8.3 [−33.3, 16.7] | 41.7% | 8.3% | 199 / 326 |
| MLP | 75.0% [50.0, 100] | 100% [100, 100] | −25.0 [−50.0, 0.0] | 83.3% | 41.7% | 118 / 208 |
| residual | 66.7% [41.7, 91.7] | 66.7% [41.7, 91.7] | +0.0 [−25.0, 25.0] | 41.7% | 33.3% | 167 / 224 |

The error-zeroed refit adds little on Pythia (+11 [−3, 25] over the fixed production circuit scored zeroed), because
the fixed circuits already survive error zeroing more often there (55.6%). On Pythia MLP targets the refit is *below*
the fixed circuit scored zeroed (75% vs 83%; 3 vs 2 failures of 12, within noise). The deepest band is where error-zeroed fits
struggle (4 of 12).

### GPT-2 small by band and kind (28 targets; L8-11 = 4)

| | n | (a) | (b) | Δ (a) − (b) | ref | (c) | median latents (a) / (b) |
|---|---|---|---|---|---|---|---|
| L0-4 | 12 | 91.7% [75.0, 100] | 83.3% [58.3, 100] | +8.3 [0.0, 25.0] | 41.7% | 25.0% | 306 / 298 |
| L5-7 | 12 | **50.0% [25.0, 75.0]** | 83.3% [58.3, 100] | **−33.3 [−58.3, −8.3]** | 41.7% | 8.3% | 300 / 377 |
| L8-11 | 4 | 3/4 | 3/4 | 0 | 0/4 | 0/4 | 280 / 708 |
| attention | 8 | 75.0% [37.5, 100] | 100% | −25.0 [−62.5, 0.0] | 12.5% | 0.0% | 476 / 476 |
| MLP | 7 | 71.4% [42.9, 100] | 85.7% | −14.3 [−42.9, 0.0] | 42.9% | 0.0% | 218 / 386 |
| residual | 7 | 57.1% [14.3, 85.7] | 42.9% | +14.3 [−28.6, 57.1] | 28.6% | 42.9% | 199 / 269 |
| resid_mid | 6 | 83.3% [50.0, 100] | 100% | −16.7 [−50.0, 0.0] | 66.7% | 16.7% | 189 / 354 |

The two targets added at 00:13: `8.mlp.30479` passes in both (a) and (b); `9.attn.30928` passes (b) only.

**Earlier GPT-2 numbers (26 targets, before the extra session):** (a) 73.1% [57.7, 88.5], (b) 80.8% [65.4, 96.2],
Δ −7.7 [−26.9, 11.5]; ref 38.5%, (c) 15.4%; median latents 310 / 358, ratio 0.98; cell-weighted 80.3 / 75.9%.

Restricted to layers 0–7 (24 targets, all cells full; unchanged): (a) 17/24 = 70.8%, (b) 20/24 = 83.3%. GPT-2's
error-zeroed circuits are only slightly smaller (ratio 0.90): the TuringLLM halving does not generalise. Exp 120's 15 known-circuit GPT-2
targets point the same way as GPT-2 L5-7 here: error-zeroed circuits passed less often than errors-live ones there
(corpus circuits, text strong: 4/15 vs 7/15; task circuits: 1/15 vs 3/15).

### Reading

- **WCM finds circuits of the error-free SAE model for most targets.** Fitted and scored with every upstream error
  zeroed, circuits pass for 69% of TuringLLM targets, 67% of Pythia-70m and 71% of GPT-2 small (layers 0–9), against
  82%, 78% and 82% for the production circuits in their own regime. On every model the paired gap is about −11 to −12
  points, and its interval reaches 0 or crosses it.
- **The refit recovers most of 116 §1's collapse.** TuringLLM 32% → 69% (+37 [25, 49]); GPT-2 36% → 71% (+36 [14, 57]);
  Pythia 56% → 67% (+11 [−3, 25], less to recover).
- **The regimes do not transfer.** Error-zeroed circuits scored with the errors live pass for 7% / 28% / 14% of
  targets. Each circuit is faithful in the regime it was fitted in.
- **Where the gap is real:** TuringLLM residual-stream targets (70% vs 96%, −26 [−44, −11]), GPT-2 L5-7 (50% vs 83%),
  Pythia L4-5 (33% vs 67%). TuringLLM attention targets (56% vs 74%) are partly limited by the error-free model itself:
  within the targets it still computes, attention passes 8/10.
- **Size.** On TuringLLM the error-zeroed circuits are about half the size (343 vs 654; ratio 0.54). They are smaller on
  Pythia (0.71) and only slightly smaller on GPT-2 small (0.90). "About half the size" is a TuringLLM statement.

## Caveats

- **The train patch** (116 `install_train_error_ablation`, a monkeypatch of `LearnedMaskPatcher.transform`) has not
  been reviewed line by line by a second reader. That the zero-fit circuits pass zero-scoring and fail live-scoring
  is consistent with it doing what it says.
- **The local GPU, not the pod.** Arms (a), (c) and (b) are all scored on this GPU with the same scorer and records;
  (b) is the production circuit (fitted on the pod) re-scored locally (116: local and production pass decisions agree
  on 228/243, 51/54, 59/60 targets). The comparison (a) vs (b) is therefore an errors-zeroed *local fit* against an
  errors-live *pod fit*. The pilot measured the hardware gap for the fit: an errors-live refit locally passed 18/18
  where production passed 15/18, so (b) may be slightly low relative to a local live refit.
- **`diag_recon` covers TuringLLM only.** No error-free-model check exists for Pythia or GPT-2.
- **The public cells are small** (4 per cell for Pythia, 3 for GPT-2). GPT-2's deep band is four targets, and layers
  10–11 have no error-zeroed fit at all (VRAM): the GPT-2 numbers are for layers 0–9 and say nothing about its
  deepest layers.
- **18 of the TuringLLM rows** (arm a and c) come from 116's pilot (`116/out/refit_zero`), run with the same script,
  code and GPU.
- **Selection.** The targets were drawn from 116's §1 lists, which hold scored, non-vacuous production targets only.

## Proposed paper text (for the coordinator; `paper/` was NOT edited)

Anchors are quoted current sentences of `paper/main.tex`. Numbers are rounded as the paper rounds them.

**T1. Limitations, "A circuit is a fitted intervention, not a recovered mechanism."** The comment above it, "Pilot
numbers (18 targets) until experiment 121 lands.", can go. Replace

> It is faithful in the regime it was fitted in, with the SAE errors live; fitted and scored with every upstream
> error zeroed, circuits pass for most targets at about half the size, though a circuit moved from one regime to the
> other is not faithful there (\cref{app:error-ablation}).

with

> It is faithful in the regime it was fitted in, with the SAE errors live. Fitted and scored with every upstream
> error zeroed, WCM also finds circuits for most targets ($69\%$, $67\%$ and $71\%$ on TuringLLM, Pythia-70m and GPT-2
> small, against $82\%$, $78\%$ and $82\%$ with the errors live), smaller on TuringLLM by about half. A circuit
> moved from one regime to the other is not faithful there (\cref{app:error-ablation}).

**T2. `\section{Circuits with the SAE Errors Ablated}` (app:error-ablation).** Replace

> In a pilot of $18$ TuringLLM targets ($2$ per band and kind), refitting with the errors ablated during training as
> well recovers a pass rate of $72\%$ $[50, 89]$, against $100\%$ for the same targets refitted with errors live on
> the same GPU; the error-free circuits are smaller (median $322$ against $584$ latents) and pass for none of the
> targets once the errors are restored.

with

> Refitted with the errors ablated during training as well (the production fit, on $81$ TuringLLM targets, $9$ per
> band and kind, $36$ Pythia-70m and $28$ GPT-2 small targets at layers $0$--$9$), and scored the same way, circuits
> pass in their own regime for $69.1\%$ $[59.3, 79.0]$ of TuringLLM targets, against $81.5\%$ $[72.8, 88.9]$ for the
> production circuits scored with errors live on the same GPU (paired $-12$ points $[-25, 0]$) and $32.1\%$ for the
> production circuits scored with errors zeroed. On Pythia-70m the rates are $66.7\%$ $[50.0, 80.6]$ against
> $77.8\%$ ($-11$ $[-28, 3]$), and on GPT-2 small $71.4\%$ $[53.6, 85.7]$ against $82.1\%$ ($-11$ $[-29, 7]$).
> The gap is largest at TuringLLM residual-stream targets ($70\%$ against $96\%$), GPT-2 layers $5$--$7$ ($50\%$
> against $83\%$) and Pythia's last two layers ($33\%$ against $67\%$); TuringLLM MLP targets pass as often as with
> errors live ($81\%$ against $74\%$). On the $50$ TuringLLM targets that the error-free SAE model still computes
> (activation within $[0.75, 1.5]$ of clean), the error-zeroed circuits pass for $82\%$ against $92\%$.
> The error-zeroed circuits are smaller on TuringLLM (median $343$ against $654$ latents) and Pythia-70m ($166$
> against $227$), slightly smaller on GPT-2 small ($280$ against $364$), and pass for $7\%$, $28\%$ and $14\%$ of
> targets once the errors are restored.

**T3. Same section, closing sentence.** Replace

> The circuits are therefore faithful circuits of the model with its SAE errors, as trained and evaluated, not of an
> error-free SAE model.

with

> Each circuit is therefore faithful in the regime it was fitted in: the production circuits are circuits of the
> model with its SAE errors, and WCM fitted without the errors finds circuits of the error-free SAE model for most
> targets, which in turn do not survive the errors' return.

Optional comment line above T2: `% 2026-11-15 (DAN-214): experiments/121-error-zeroed-refit-scale (out/report.md);
GPT-2 layers 10-11 not refitted (16 GB VRAM).`
