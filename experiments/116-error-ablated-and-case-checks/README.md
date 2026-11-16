# 116 Error-ablated re-score and case-study checks (DAN-214)

## Questions

1. **"The circuit alone" keeps the SAE errors live.** If the SAE errors at every upstream site are zeroed, or replaced
   by their mean, while the fixed production circuit runs alone, do the circuits still pass? (No refit.)
2. **Case studies.** Do the fusion core (10 plasma-confinement latents; 24 physics-group latents) or the antiquity
   story groups (19 latents) suffice on their own, with coefficients refitted on them? What does removing each group
   do in the full, unablated model, against random members of similar contribution rank?

## Inputs (read only)

- Production circuits (WCM v4, 2026-11-07/08 pod run): TuringLLM
  `109-pod-run-analysis/out/archives/pub-turingllm-T3-main/.../094-tl-v2-records/out_v2/main/circuits`; public models
  `.../pub-<model>-S10-results/experiments/082-public-runs/out_<model>/main/circuits`.
- Records: TuringLLM `.../pub-turingllm-T2-records/.../out_turingllm/ctx_fresh`; public `out_<model>/ctx`.
- Case-study groups and contribution ranks: `110-case-studies-v4/out/case/<key>.json` (groups restricted to the
  members present in the v4 circuit, story list) and `110-case-studies-v4/out/members/<key>.csv`.
- Pass rule and vacuous rule: `062/merge.py` via `109/lib109` (imported, not re-implemented). Bootstrap: `lib109.boot_ci`,
  `lib109.boot_paired_diff` (4,000 resamples of targets, seed 20261108).
- Nothing was downloaded. Every model and SAE was already in the local caches.

## Method

**Hardware design (as 115).** Every comparison is PAIRED on this GPU (RTX 5070 Ti): errors live (the production
scoring, re-run locally) vs errors ablated, same circuit, same record, same code, same process. The production rows are
a reference column only. The local scorer is bit-deterministic. The live rows of the 3 smoke targets are bit-identical
to 115's `rs_orig` rows, and the case circuits' full rows equal 110's local re-score exactly (fusion 0.9065 / 1.0920 /
0.9610).

**§1 error switch (`errpatch.py`).** Every circuit-only measurement in both production scorers (049
`score_circuit` for TuringLLM, `src/eval/wcm_eval.score_circuit` for the public models) goes through the engine's
`CircuitOnlyPatcher`. This includes Z / A / C, the empty-circuit baselines and the pins. The patcher already has an
SFC-style error switch (`keep_error_sites` / `error_means`, used in exp 004). `errpatch.install()` wraps its
`__init__`:
- **live**: unchanged.
- **zero**: the SAE error of every in-scope (upstream) site is set to 0, so the stream is decode(patched code) alone.
- **mean**: the error is replaced by its mean over the scorer's own reference rows for that fill. These are the same
  rows the latent means come from: strongest-train rows for A, contrast-train rows for C, and the token-weighted pool of
  the two for Z, which has no latent reference. The mean is over every (row, position), as in
  `collect_site_error_means`.

Live-stream runs keep the error, as runs of the unmodified model must. These are the natural `a_pos`, necessity's
a_{M\C} and sufficiency to induce. Necessity therefore changes only through a_empty. The circuits are not refitted. In
`rescore_err.py` each target is scored three times (live, zero, mean) on the held-out strongest contexts.

**Targets (`select_targets.py`, seed 20261115).** Scored, non-vacuous production targets, drawn uniformly inside each
depth band x site-kind cell. Unlike 115, the draw is not fail-enriched. TuringLLM: 27 per cell, 243 targets (production
pass 86.4% in the sample, 88.0% overall). The "cell-weighted" rates re-weight the cells to their production sizes.

**§2 (`case_checks.py`).** For each tested set S, the steps are:
1. The production scorer with `override_alphas` = S at the fitted α: Z / A / C of S alone, no refit.
2. From the same scorer row, the knock-out in the unmodified model: necessity's live-stream run with only S set to 0.
   Every other latent and every SAE error stays live, and everything downstream is recomputed. drop = 1 − a_KO / a_pos,
   on the held-out strongest contexts, with the activation read.
3. The coefficients refitted on S alone: the production WCM v4 fit restricted to S, with no sparsity penalty and fully
   open gates (`062/random_null.install_support_wrapper`, the stable-core recipe). Then scored like any circuit.

Controls, per S:
- rank-matched random sets (063's recipe: for each member, one non-named, non-generic member within ±15 contribution
  ranks; 8 draws for steps 1–2, 3 of them refitted);
- the |S| largest contributors (steps 1–3).

Refits were done for the claimed sets only: fusion's plasma confinement (10) and all four physics groups (24), and
antiquity's three story groups (19).

## Run

```
python -X utf8 experiments/116-error-ablated-and-case-checks/select_targets.py                     # CPU
wsl: FAMILY=turingllm MAX_MIN=85 PYTHONPATH=src .venv/bin/python -X utf8 experiments/116-.../rescore_err.py   # 35 min
wsl: PYTHONPATH=src .venv/bin/python -X utf8 experiments/116-.../case_checks.py                        # 40 min
wsl: FAMILY=pythia-70m MAX_MIN=25 ... rescore_err.py ; FAMILY=gpt2-small MAX_MIN=30 ... rescore_err.py  # 5 + 25 min
wsl: KEYS=<targets_turingllm.txt, comma-joined> ... diag_recon.py                                   # 10 min
wsl: ERR=zero TARGETS_FILE=<116>/targets_refit.txt ... refit_err.py ; then ERR=live                  # 40 + 28 min
python -X utf8 experiments/116-error-ablated-and-case-checks/report.py                             # CPU -> out/report.md
```
Logs: `out/smoke_tl.log`, `out/rs_<family>.log`, `out/case.log`, `out/diag_recon.log`, `out/refit_{zero,live}.log`. Rows: `out/rs_<family>/rows.jsonl`,
`out/case/rows.jsonl`; paired tables `out/paired_<family>.csv`; full tables `out/report.md`.

## Results

All numbers below come from `out/report.md`, which `report.py` generates. Pass = the DAN-8 rule (Z, A, C in
[0.75, 1.5], necessity ≥ 0.9); a vacuous row counts as a fail. 95% bootstrap CIs are over targets. Δ pass is the paired
mean difference with a bootstrap CI.

### §1 Fixed production circuits, SAE errors live vs ablated (held-out strongest, this GPU)

| model | targets | pass, errors live | pass, errors zeroed | pass, errors mean-ablated | Δ pass (zeroed − live), pp |
|---|---|---|---|---|---|
| TuringLLM | 243 | 86.0% [81.5, 90.1] | 34.6% [28.8, 40.7] | 34.6% [28.8, 40.7] | −51.4 [−58.0, −44.0] |
| Pythia-70m | 54 | 83.3% [72.2, 92.6] | 50.0% [37.0, 63.0] | 46.3% [33.3, 59.3] | −33.3 [−46.3, −22.2] |
| GPT-2 small | 60 | 75.0% [63.3, 85.0] | 33.3% [21.7, 45.0] | 33.3% [21.7, 45.0] | −41.7 [−55.0, −28.3] |

Re-weighting the cells to their production sizes gives similar rates: TuringLLM 85.5 / 37.8 / 37.7%, Pythia
81.2 / 45.8 / 41.7%, GPT-2 74.7 / 34.0 / 34.0%. Flips (live → zeroed), pass → fail / fail → pass: TuringLLM 131 / 6,
Pythia 18 / 0, GPT-2 25 / 0.

**TuringLLM by depth band and site kind** (pass, live → zeroed; median ΔZ / ΔA / ΔC):

| | n | live | zeroed | Δ pp [95% CI] | median ΔZ / ΔA / ΔC |
|---|---|---|---|---|---|
| L0-4 | 81 | 88.9% | 51.9% | −37.0 [−48.1, −25.9] | +0.05 / +0.02 / +0.03 |
| L5-7 | 81 | 92.6% | 30.9% | −61.7 [−72.8, −50.6] | −0.43 / −0.39 / −0.45 |
| L8-11 | 81 | 76.5% | 21.0% | −55.6 [−67.9, −43.2] | −0.52 / −0.42 / −0.38 |
| attention | 81 | 85.2% | 14.8% | −70.4 [−79.0, −60.5] | −0.76 / −0.59 / −0.68 |
| MLP | 81 | 80.2% | 65.4% | −14.8 [−25.9, −3.7] | +0.07 / +0.04 / +0.03 |
| residual | 81 | 92.6% | 23.5% | −69.1 [−79.0, −58.0] | −0.35 / −0.30 / −0.34 |

TuringLLM cells (pass live → zeroed): L0-4 attn 0.96 → 0.22, mlp 0.89 → 0.81, resid 0.81 → 0.52; L5-7 attn
0.93 → 0.19, mlp 0.85 → 0.63, resid 1.00 → 0.11; L8-11 attn 0.67 → 0.04, mlp 0.67 → 0.52, resid 0.96 → 0.07.

Public models (pass live → zeroed):
- **Pythia-70m.** L0-1 94 → 72%, L2-3 78 → 61%, L4-5 78 → 17%. By kind: attn 72 → 39%, mlp 100 → 72%,
  resid 78 → 39%.
- **GPT-2 small.** L0-4 90 → 50%, L5-7 85 → 45%, L8-11 50 → 5%. By kind: attn 87 → 7%, mlp 73 → 33%,
  resid 53 → 40%, resid_mid 87 → 53%.
- The public cells hold 15–20 targets each, so their intervals are wide (`out/report.md`).

What changes:
- **Scores mostly fall; they do not rise.** TuringLLM median Z / A / C go from 0.92 / 0.98 / 0.97 to 0.66 / 0.74 / 0.70.
  Of the live passes that fail with errors zeroed, Z < 0.75 causes 108, C < 0.75 106 and A < 0.75 101. An overshoot
  above 1.5 causes 13–19 (Z, A or C).
- **Many collapse to zero.** In 50 TuringLLM targets (GPT-2: 14; Pythia: 9), the circuit-alone target is censored to 0
  under Z only when the errors are zeroed. The pre-activation reads fall less: median Δ −0.08 (Z), −0.19 (A), −0.20 (C).
- **Necessity is unchanged** (median Δ 0). Its numerator is a live-model run, and the empty-circuit baseline a_∅ is 0
  for almost every target in both modes.
- **Zeroed ≈ mean-ablated.** Pass rates are equal on TuringLLM and GPT-2, and 4 pp apart on Pythia. The error that
  matters is its per-token part, not its mean offset.
- **The α = 1 circuit barely moves** (TuringLLM free0_a1 median Δ 0.000). The fitted coefficients are what rely on the
  live error.

Sanity (reference only, not used in any delta): the local live re-score agrees with the production pass decision on
228 / 243 TuringLLM, 51 / 54 Pythia and 59 / 60 GPT-2 targets. The median max |Δ| on Z/A/C/nec is 0.024 / 0.002 / 0.008,
the same hardware gap 115 measured.

### §1 diagnostic: TuringLLM run on its SAE reconstructions, no circuit (`diag_recon.py`, 243 targets)

Every latent stays live. At every upstream site of the given kinds, the stream is replaced by its SAE reconstruction
(error zeroed) or by reconstruction + mean error. The table gives the target's activation over its clean value on the
held-out strongest contexts, as median [IQR].

| target kind | n | all sites, zeroed | all sites, mean | residual sites only, zeroed | MLP only | attention only |
|---|---|---|---|---|---|---|
| attention | 81 | 0.60 [0.23, 0.90] | 0.69 [0.31, 0.96] | 0.79 [0.51, 0.95] | 0.92 | 0.99 |
| MLP | 81 | 0.98 [0.66, 1.04] | 1.00 [0.73, 1.05] | 1.00 | 1.00 | 1.00 |
| residual | 81 | 0.96 [0.83, 1.05] | 1.00 [0.91, 1.09] | 0.98 | 0.95 | 1.00 |

- **Attention targets lose 40% of their activation (median) even in the error-free model.** Most of this comes from
  the residual-stream SAE errors. So part of the attention collapse in §1 belongs to the SAEs, not to the circuits:
  without its errors, the model no longer computes these targets.
- **Residual targets barely move without errors (0.96),** yet their circuits fall from 93% to 23%. That collapse
  belongs to the fitted circuits.
- Restricting §1 to the 158 targets whose own activation stays within [0.75, 1.5] of clean in the error-free model:
  circuit pass with errors zeroed 47.5%, live 91.8%. For the other 85: 10.6% vs 75.3%.

### §1b Refit pilot: circuits refitted with the errors ablated during training (`refit_err.py`, 18 TuringLLM targets)

There are 2 targets per band x kind cell, drawn from the §1 list (seed 20261116). Each is fitted twice on this GPU with
the production fit (WCM v4, λ 1e-3, 400 steps, same records):
- **live**: unchanged;
- **zero**: the learned-mask engine's masked forward has each masked site's error subtracted, so it trains on
  decode(code) alone, the same thing errpatch's zero mode scores.

Both fits are scored with errors live and zeroed.

| fitted with errors | scored with errors | pass (95% CI) | median Z / A / C / nec | median latents |
|---|---|---|---|---|
| live | live | 100.0% (18/18) | 0.93 / 0.97 / 0.95 / 1.00 | 584 |
| live | zeroed | 38.9% [16.7, 61.1] | 0.68 / 0.65 / 0.44 / 1.00 | 584 |
| zeroed | zeroed | **72.2% [50.0, 88.9]** (13/18) | 0.88 / 0.90 / 0.85 / 1.00 | 322 |
| zeroed | live | 0.0% | 0.00 / 0.00 / 0.00 / 1.00 | 322 |

- **Paired, fitted-and-scored without errors vs with: −27.8 pp [−50.0, −11.1].**
- By kind (zeroed/zeroed): attention 4/6, MLP 5/6, residual 4/6.
- Of the 5 failures, 3 fail only on necessity (0.78–0.87; the error-free circuits are about half the size), and 2 on
  faithfulness (10.attn.18838, 11.resid.33628).
- The live refit reproduces §1 on these targets: production circuits 83.3% live and 38.9% zeroed; local live refit
  100% and 38.9%.
- **Reading.**
  - Most of the §1 collapse is coefficients and membership tuned to the live error. WCM finds error-free circuits for
    most targets when trained for that regime: 72% here against 39% for the fixed circuits.
  - The error-free circuits are a different object. They are half the size, and they fail completely once the errors
    are back. So each circuit is faithful only in the regime it was fitted in.
  - 18 targets is a pilot; the interval is wide.

### §2 Case studies (TuringLLM, production v4 circuits, held-out strongest)

Fusion (9.resid.13945, 1,322 latents): full circuit Z / A / C / nec 0.91 / 1.09 / 0.96 / 1.00, a_pos 33.3.
Antiquity (9.resid.20419, 945 latents): 0.90 / 1.06 / 0.98 / 1.00, a_pos 28.5. The local re-score equals 110's
exactly.

**Knock-outs in the unmodified model** (S set to 0 in the full model, everything else live, drop = 1 − a_KO / a_pos):

| target | set | n | drop, named set | drop, rank-matched random (8 draws: mean, range) | drop, the n largest contributors |
|---|---|---|---|---|---|
| fusion | plasma confinement | 10 | **0.63** | 0.08 (0.05–0.10) | 0.68 |
| fusion | nuclear and power engineering | 6 | 0.20 | 0.05 (0.04–0.07) | 0.66 |
| fusion | stars, cosmos, particles | 3 | 0.12 | 0.02 (0.01–0.03) | 0.50 |
| fusion | quantum and magnetism | 5 | 0.07 | 0.04 (0.03–0.05) | 0.60 |
| fusion | all four physics groups | 24 | **0.77** | 0.13 (0.12–0.15) | 0.80 |
| antiquity | dates / eras | 4 | 0.22 | 0.04 (0.01–0.06) | 0.58 |
| antiquity | early civilisations | 6 | 0.34 | 0.10 (0.07–0.12) | 0.72 |
| antiquity | classical Greece | 9 | 0.61 | 0.09 (0.06–0.12) | 0.75 |
| antiquity | all three story groups | 19 | **0.79** | 0.17 (0.15–0.24) | 0.81 |

**Sufficiency** (the set run alone; Z / A / C):

| target | set | n | production α | coefficients refitted on the set | top-n contributors, refitted | random sets, refitted (3 draws, mean) |
|---|---|---|---|---|---|---|
| fusion | plasma confinement | 10 | 0.00 / 0.19 / 0.00 | 0.00 / 0.64 / 0.00 | 0.00 / 0.78 / 0.03 | 0.00 / −0.08 / 0.00 |
| fusion | all four physics groups | 24 | 0.00 / 0.15 / 0.00 | 0.00 / 0.84 / 0.40 | 0.00 / 0.95 / 0.22 | 0.00 / −0.21 / 0.06 |
| antiquity | all three story groups | 19 | 0.00 / 0.19 / 0.00 | 0.00 / 0.96 / 0.25 | 0.00 / 1.11 / 0.79 | 0.00 / 0.26 / 0.12 |

Reading:
- **The named groups are necessary in the unmodified model,** not only inside the circuit-alone regime:
  - Fusion's 10 plasma latents remove 63% of the target, against 8% for rank-matched random members.
  - Antiquity's 19 story latents remove 79%, against 17%.
  - Classical Greece alone removes 61% (random 9%).
- **They are about as necessary as the circuit's largest contributors** of the same count. The top-10 overlap the plasma
  group in only 2 latents, yet remove 68%.
- **They are not sufficient.** No refitted set passes. Z stays at 0 (under zero ablation of everything else, the target
  never clears Top-K), and C reaches at most 0.40 for the named sets.
- The A read recovers, up to 0.96 for antiquity's story groups. The equally large set of top contributors does as well
  or better: A 1.11, C 0.79 for antiquity.
- So the evidence supports "removing them collapses the target", not "they carry the computation".

## Caveats

- **No refit in §1.** The production coefficients were fitted with the errors live, and the drop measures fixed
  circuits under a regime they were not trained for. §1b (18-target pilot) shows that a refit recovers most of the loss
  (72% vs 100%), but with different, smaller circuits.
- **§1b patches the training engine** by monkeypatching `LearnedMaskPatcher.transform` in-process (no src edit). The
  members-removed "invert" pass and the seed tap are untouched. That this is the intended training regime is
  supported by the numbers (zero-fit circuits pass zero-scoring), but the patch has not been reviewed line by line by a
  second reader.
- **The diagnostic (`diag_recon.py`) reads with the scorer's helpers but its own patcher.** Its clean reference equals
  the scorer's a_pos only up to the same read; it is used for ratios only.
- **Errors are ablated at every upstream site at once,** including the residual sites. The paper's SAEs explain
  0.85–0.99 of variance, and TuringLLM's own error-free model is checked in the diagnostic. Error ablation is a
  stricter regime than SFC, where errors are nodes that a circuit may keep.
- **Sample sizes:** 243 / 54 / 60 targets, 15–27 per cell. The public per-cell intervals are wide.
- **The local GPU is not the pod:** the live pass rate here (86.0%) is the local re-score. Only the paired deltas are
  interpreted.
- **§2 random controls are smaller than the named sets** when the rank window runs out of non-named candidates (fusion
  all-groups 18 vs 24, antiquity 17 vs 19). The top-n control has no such issue.
- **§2 knock-outs are zero-ablations** (the paper's necessity convention), not mean-ablations.
- The §1 mean mode uses the scorer's reference rows (A: strongest-train, C: contrast-train, Z: the pool).

## Proposed paper text (for the coordinator; main.tex and paper/tables were NOT edited)

Anchors are quoted sentences, not line numbers: main.tex is being edited in parallel. Numbers are rounded as the paper
rounds them.

**P1. Intro / method, "faithfulness (the circuit alone reproduces the target)"** (the sentence beginning "We test each
circuit on held-out text for").
- Replace "(the circuit alone reproduces the target)" with "(the circuit, with every other latent ablated and the SAE
  errors live, reproduces the target)".
- Same change wherever "the circuit alone" defines faithfulness. Also the fig:case-study caption "(b,c) ... circuit
  alone under contrast-mean ablation" becomes "circuit with all other latents ablated (SAE errors live) under
  contrast-mean ablation".

**P2. Limitations, "A circuit is a fitted intervention, not a recovered mechanism."** Replace its second sentence
("It acts with the SAE errors kept live, ...") with:

> It acts with the SAE errors kept live, and depends on them: with the errors of every upstream site zeroed (or
> mean-ablated) and the coefficients unchanged, the pass rate falls from $86\%$ to $35\%$ on TuringLLM ($83\%$ to $50\%$
> on Pythia-70m, $75\%$ to $33\%$ on GPT-2 small; paired, \cref{app:error-ablation}), mostly at attention and residual
> targets beyond layer 4, while MLP targets keep most of theirs. Refitted with the errors ablated, circuits pass again for most targets ($72\%$ of an
$18$-target pilot, against $100\%$ for a paired refit with errors live), but they are other circuits: about half the
size, and they fail once the errors are restored. It inherits any ambiguity in its latents
> and is validated by intervention effects, not by a planted ground truth: the corpus is synthetic, but it is not used
> as ground truth for circuits, and a planted-circuit test is left for future work.

**P3. Case study, fusion sentence** ("A second circuit, for magnetic-confinement fusion, has a sharper core inside an
amplifier: removing $10$ of its $1{,}322$ latents collapses the target, ..."). Keep "removing ... collapses the
target" (it holds in the unmodified model too) and add after the parenthesis:

> Removing the same ten latents from the full, unablated model lowers the target by $63\%$, against $8\%$ for random
> members of similar contribution rank, but the ten are not sufficient: refitted on their own, they fail every
> ablation (\cref{app:case-studies}).

If the body still contains "Ten of its 1,322 latents carry the computation" (the issue's main.tex:538), replace it with
"Removing ten of its 1,322 latents collapses the target".

**P4. Case study, antiquity** (after "Removing all $19$ story-group latents, $2\%$ of the circuit, takes the target to
$0.22$, $0.10$, $0.21$ ..."). Add:

> In the unmodified model, removing them lowers the target by $79\%$ ($17\%$ for random members of similar rank), but
> they are not a sufficient core: refitted alone they reach Z, A, C of $0.00$, $0.96$, $0.25$, no better than the $19$
> largest contributors.

**P5. Discussion, stable core** (after "...their shared core alone passes for $44$--$72\%$ of targets
(\cref{app:underdetermination}): ..."). Add the hypothesis sentence the issue asks for:

> One untested explanation is redundancy across sites: a residual SAE re-encodes what earlier sites wrote, so many
> members may be alternative carriers of the same signal, and two fits may pick different ones.

**P6. New appendix paragraph** (in app:protocol after the execution table, or a short `app:error-ablation`):

> \paragraph{SAE errors ablated.} Every faithfulness run keeps each site's SAE error live. To measure how much the
> circuits rely on it, we re-scored fixed production circuits ($243$ TuringLLM targets, $27$ per depth band and site
> kind; $54$ Pythia-70m and $60$ GPT-2 small targets; same records, no refit) with the error of every upstream site
> set to zero, or to its mean over the same reference contexts as the latent fill, in every circuit-only run. Live and
> ablated were scored on the same GPU. Pass rates fall from $86.0\%$ $[81.5, 90.1]$ to $34.6\%$ $[28.8, 40.7]$ on
> TuringLLM ($-51$ points $[-58, -44]$), from $83.3\%$ to $50.0\%$ on Pythia-70m and from $75.0\%$ to $33.3\%$ on GPT-2
> small; mean ablation gives the same rates within $4$ points, so the per-token part of the error matters, not its
> offset. The loss is concentrated at attention ($85\% \to 15\%$ on TuringLLM) and residual targets ($93\% \to 23\%$)
> and at depth ($L5$--$11$: $\approx 85\% \to 26\%$); MLP targets fall from $80\%$ to $65\%$. Scores fall rather than
> overshoot (median Z $0.92 \to 0.66$), and for $50$ TuringLLM targets the circuit-alone target no longer clears the
> Top-K cut. Necessity is unchanged, since its numerator is a run of the unmodified model. Part of the attention loss belongs to the SAEs, not the circuits:
> with every upstream error zeroed and no latent ablated, attention targets keep a median $60\%$ of their activation
> (residual targets $96\%$, MLP targets $98\%$). In a pilot of $18$ TuringLLM targets, refitting with the errors
> ablated during training as well recovers a pass rate of $72\%$ $[50, 89]$, against $100\%$ for the same targets
> refitted with errors live on the same GPU; the error-free circuits are smaller (median $322$ against $584$ latents)
> and pass none of the targets once the errors are restored.
> The circuits are therefore faithful circuits of the model with its SAE errors, as trained and evaluated, not of an
> error-free SAE model.

**P7. App case-studies.** Add both §2 tables (knock-outs in the unmodified model; sufficiency with refitted
coefficients), with this caption: "Knock-outs set the group's latents to zero in the full model (every other latent
and every SAE error live); random = rank-matched members (8 draws), top-n = the n largest contributors; sufficiency =
the group alone, coefficients refitted on it (no sparsity penalty)."
