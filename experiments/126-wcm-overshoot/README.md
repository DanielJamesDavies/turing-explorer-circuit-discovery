# 126: why do shallow-target WCM circuits overshoot held-out?

## Question

Arena A (GPT-2 small, Dunefsky transcoders; `experiments/107-true-baselines/arena_a/`), WCM v4, scope a2
(transcoders only), lambda 1e-3, tuned `CHOSEN.env` (NEC_W 0.3), 400 steps, smoke corpus. Some shallow tc targets pass
on their own training contexts but overshoot on the held-out ones. Example, 3.tc.13057: held-out Z / A / C
2.25 / 2.33 / 1.92, home (training) 1.09 / 1.09 / 1.03. Pass rule: Z, A, C in [0.75, 1.5] and nec_b >= 0.9.

Overshoot here means max(Z, A, C) on ho > 1.5 while home passes (pass_b).

Four hypotheses:
- (A) too many steps (over-training);
- (B) too few training contexts;
- (C) a poor or small corpus. The smoke corpus is 2,048 wikitext windows from 40 articles, and 12-16 of the 16
  held-out rows share a document with training rows;
- (D) the fitted coefficients (alpha p90 ~2.4, max ~5.5).

## Arms

All on the 10 shallow tc targets of `targets_smoke` (L0-4: 0.tc.1257 0.tc.5811 1.tc.23034 1.tc.5306 2.tc.20862
2.tc.5850 3.tc.13057 3.tc.19795 4.tc.10851 4.tc.9014), scope a2, lambda 1e-3, CHOSEN.env.

| arm | settings | how |
|---|---|---|
| A steps | 100, 200, **400** (baseline), 800 | `AA_STEPS` -> driver STEPS; each its own OUT (`out/fit/s<N>`) |
| B training contexts | D_train 8 / 16 (B_mid 4 / 8) vs **32 / 16** | `make_ctx_subsets.py`: a nested, seeded random subset of D_train and B_mid (`out/targets_n8`, `out/targets_n16`, SUBSETS.json). Held-out set, the A-ablation means (strong.train, 48), contrast rows and ho_xdoc reference are unchanged. 1.tc.5306 is a training-arm-A record (no mid-band pool: the fit uses all 48 strong-train rows), so it is left out (9 targets). "home" follows the record: the 8 highest-peak D_train rows. The random subset can drop D_train's top-ranked rows (the protocol keeps ranks 1-2) |
| C corpus | OWT, the first 131,072 windows (a prefix of the pod's 524,288; the first 32k match owt32k byte for byte) | `out/corpus_owt131k`. Data pass (GPU, batch 64, mid warm-up 200 = the pod's 9.8 %), `make_targets.py` (EXCLUDE = tune_exclude.txt), then fit and score on the tc targets in L0-4 that it picks. **These are different targets**, so compare populations, not targets |
| D coefficients | alpha clipped to [1/c, c], c = 1.5, 2, 3, on the s400 circuits | `make_clipped.py` (no refit; `out/clip/c<c>`). Scored as extra `--wcm` labels `c1.5` / `c2` / `c3` in the s400 scoring call. In `score_arena_a.wcm_units` a label that does not start with `l` gets knob_value None (no lambda parse), and the uid carries the label + dir name |

Built on the CPU (2026-11-16):
- `out/corpus_owt131k` (`out/corpus_owt131k.log`);
- `out/targets_n8` and `out/targets_n16`;
- `out/diversity.md`, the smoke records' document / near-duplicate counts. The 16 held-out rows share a document
  with a training row for 12-16 of them (median 12.5 against the fit rows, 15 against all train rows; 3.tc.13057:
  15 / 16). There are no 8-gram near-duplicates (max Jaccard <= 0.15).

## How to run (WSL, repo root)

```
DRY_RUN=1 bash experiments/126-wcm-overshoot/run_night.sh                 # the plan; changes nothing
STOP_EPOCH=$(date -d 'tomorrow 08:00' +%s) nohup bash experiments/126-wcm-overshoot/run_night.sh \
    > experiments/126-wcm-overshoot/out/run_night.out 2>&1 &
```

Stage order:
1. s400, then clip (D), then score s400 with the clipped copies;
2. s200, s100, n8, n16, s800, each fit then scored;
3. analyze;
4. C: data pass, targets, fit, score, diversity;
5. analyze again.

How the runner behaves:
- **Lock:** it takes `.gpu-lock` and refuses to start if the lock exists, `out/ABORTED` exists, or the GPU already
  holds more than `N126_GPU_IDLE_MIB` [1500] MiB.
- **Watchdog:** above 15,300 MiB of GPU memory or below 4 GB of host RAM it kills the job and stops.
- **STOP_EPOCH:** after it, no new stage starts. It is passed to run_pod.sh and to the scorer. The data pass only
  starts if `N126_DP_EST_S` [5400] s fit before STOP_EPOCH.
- **Resuming:** state markers live in `out/state/`, and finished fits and rows are skipped. `N126_STAGES` /
  `N126_SKIP` select stages.
- **Logs:** `out/night.log` (one line per stage start and end), `out/logs/`, `out/watch.csv`.

How fits and scoring run:
- Fits go through `run_pod.sh` with `AA_FROM=P5 STOP_AT=P5`, so P6 / P6b never start (the STOP_AT=P6 -> P6b trap does
  not arise).
- Scoring calls `score_arena_a.py` directly, as p6_worker does, with `--selftest`. It runs without `--donor` (set
  `N126_DONOR=1` to add it) and without Ge recovery.

Dry-run estimate:
- A + B + D: ~3.7 h;
- C: ~2.7 h. The data pass is unmeasured at 131k (~80 min estimated from owt32k) and make_targets on 131k is a guess
  (~45 min);
- **total ~6.4 h.**

Analysis: `analyze.py [--alphas]` -> `out/report.md` (+ .json). It was tested on today's
`arena_a/out/gpu_test/score/a2.s0/rows.jsonl`.

## What each outcome would mean

- **A:** if overshoot falls with fewer steps (s100 / s200) and rises at s800 while home stays in band, the fit is
  over-training. The steps budget also sets the LR schedule length, so s800 is not "s400 plus 400 more".
- **B:** if overshoot rises at n8 / n16, a small training set is the cause and more contexts would help. If it is
  flat, B is ruled out.
- **C:** if the OWT population has far fewer overshooting shallow targets (with xdoc n > 0, i.e. held-out windows
  from unseen documents), the smoke corpus is at fault (few documents, held-out rows sharing training documents).
  The pod corpus (524k) would then be expected to behave like OWT.
- **D:** if clipping brings ho max(Z, A, C) into band while home stays in band, the fitted alphas cause the overshoot
  (amplitudes that do not transfer). If clipped circuits fall below band on home too, the alphas carry real signal
  and the problem is elsewhere.

## Results (night 2026-11-16, local RTX 5070 Ti; full tables in `out/report.md`)

Run notes: the first launch (19:07) fitted 8 of 10 targets in every stage, but the two layer-0 tc targets
(0.tc.1257, 0.tc.5811) error in scope a2 ("does not require grad": no transcoder upstream of layer 0), and
run_pod.sh's every-target-ok check then failed each fit stage, so nothing was scored. `run_night.sh` now drops layer-0
keys (`no0`); the relaunch (23:09) resumed the ok fits and finished at 23:35 with 0 failed stages. 8 smoke targets
(7 for arm B: 1.tc.5306 is arm A) and 8 OWT targets.

| setting | ho pass b | home pass b | overshoot | median ho max(Z,A,C) | median n |
|---|---|---|---|---|---|
| s100 / s200 / s400 / s800 | 5 / 4 / 4 / 4 of 8 | 7 of 8 each | 1 of 8 each | 1.03 / 1.07 / 1.02 / 0.98 | 646 / 413 / 318 / 257 |
| n8 / n16 (fit contexts; base 48) | 1 / 1 of 7 | 6 of 6 / 7 of 7 | 0 / 1 of 7 | 0.41 / 0.79 | 184 / 253 |
| s400 alpha clipped at 1.5 / 2 / 3 | 2 / 4 / 4 of 8 | 7 of 8 each | 1 of 8 each | 0.87 / 0.95 / 1.00 | 318 |
| owt131k (other targets) | 7 of 8 | 6 of 8 | 0 of 8 | 1.01 | 75 |

- **Overshoot is one target, not a pattern.** Only 3.tc.13057 overshoots (ho max 2.1 to 2.6 in every setting).
  Its nec_b is 3.3 on home and 4.6 on held-out, far above 1 even on training, so the target itself behaves unusually
  in scope a2 (its a1 fit passed in the smoke run). The common smoke failure is the opposite: undershoot on
  held-out (2.tc.5850, 2.tc.20862 at 0.7 to 0.87) while home passes.
- **A, steps: no effect** on pass or overshoot from 100 to 800 steps. More steps give a smaller circuit with larger
  alphas (n 646 to 257, alpha p90 1.2 to 2.9) at the same quality.
- **B, contexts: the large effect.** Fewer fit contexts leave home in band (1.05) and drop held-out to 0.79 (16) and
  0.41 (8): classic overfitting, and it shows as undershoot, not overshoot.
- **C, corpus: the fix.** On the 131k OWT corpus 7 of 8 targets pass held-out (median 1.01), with 11 to 16 of the 16
  held-out windows from documents not in training, and those windows score the same (xdoc max 1.00). The smoke corpus
  gave 0 to 2 such windows. Caveat: different targets (picked on the new corpus), so not a paired comparison;
  circuits are also smaller (median 75). The one held-out failure (1.tc.14335) is nec_b 0.79, not Z/A/C.
- **D, alphas: not the cause.** Clipping does not remove 3.tc.13057's overshoot (2.56 at clip 1.5) and pulls the
  other targets below band (ho pass 4 to 2 at clip 1.5): the alphas carry real signal.

### Night 2 (2026-11-17 00:10-03:34): smaller circuits, deep targets

**Larger lambda, owt131k shallow targets (8):** held-out pass 7 / 7 / 4 of 8 at lambda 1e-3 / 4e-3 / 1.6e-2, median
n 75 / 29 / 9 (e.g. 4.tc.6575 368 -> 82 -> 11). At 4e-3 nothing is lost (median ho max 1.01; xdoc ~1.0). At 1.6e-2
the failures are necessity (nec_b 0.71-0.87) with Z/A/C still 0.9-1.07; 1- and 7-latent circuits pass outright.

**Deep owt131k tc targets, WCM lambda 1e-3 vs Dunefsky (their code, prompt 0 = K 1, 5 / 30 / 60 / 150 branches).**
The L9-11 WCM fits exceed this PC's host RAM (watchdog: MemAvailable < 4 GB, then < 2 GB, at the first L8 / L9 fit),
so 8 of 14 deep targets (L5-8) were fitted and scored (`score/owt131k_deep57`, stage `score_deep57`).

| target | WCM n, ho Z / A / C / nec_b, pass | Dunefsky b150 n, ho Z / A / C / nec_b | best Dunefsky ho max(Z,A,C), any budget |
|---|---|---|---|
| 5.tc.1481 | 165, 1.03 / 0.97 / 0.97 / 1.03, pass | 83, 0.16 / 0.11 / 0.10 / 0.98 | 0.16 |
| 5.tc.8088 | 198, 0.97 / 0.92 / 0.92 / 1.00, pass | 80, 0.06 / 0.00 / 0.00 / 0.98 | 0.06 |
| 6.tc.9000 | 415, 0.92 / 0.94 / 0.91 / 1.00, pass | 68, 0.00 / 0.00 / 0.00 / 0.71 | 0.00 |
| 6.tc.24021 | 618, 0.95 / 0.97 / 0.94 / 0.76, fail (nec) | 92, 0.31 / 0.38 / 0.36 / 0.42 | 0.38 |
| 7.tc.3375 | 386, 0.83 / 0.89 / 0.88 / 0.96, pass | 76, 0.26 / 0.12 / 0.10 / 0.79 | 0.26 |
| 7.tc.15758 | 54, 1.00 / 1.00 / 1.00 / 1.00, pass | 53, 0.27 / 0.23 / 0.22 / 1.00 | 0.27 |
| 8.tc.14821 | 312, 0.95 / 0.95 / 0.96 / 0.99, pass | 75, 0.00 / 0.00 / 0.00 / 0.93 | 0.00 |
| 8.tc.22944 | 835, 1.04 / 1.05 / 1.03 / 0.64, fail (nec) | 91, 0.26 / 0.27 / 0.21 / 0.25 | 0.27 |

- WCM passes 6 of 8 on held-out; both failures are necessity, with Z/A/C in band. Dunefsky passes 0 of 32 (8 targets
  x 4 budgets), on held-out and on training rows; its sufficiency is at most 0.38 (held-out) and mostly near 0.
- Size: Dunefsky reaches 53-92 latents at 150 branches, WCM 54-835. The one matched-size pair (7.tc.15758, 53 vs 54)
  is 0.27 vs 1.00. Caveats: their circuits come from one prompt (their method's unit), scored here across contexts
  (our question); 300 branches was not run; 8 targets, one corpus.

Reading: the smoke corpus (few documents per target) is the main reason for the held-out gap; the pod corpus (524k,
of which owt131k is a prefix) should behave like owt131k. Fit-context count matters much more than step count.
