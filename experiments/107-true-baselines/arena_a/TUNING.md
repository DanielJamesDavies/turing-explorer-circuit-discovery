# Arena A: tuning WCM for the tc + z substrate (A13, 2026-11-05, agent `a-tune`)

Goal (Daniel, 2026-11-05 ~13:10): make WCM sensible on arena A's substrate (GPT-2 small + Dunefsky transcoders `tc`
+ Kissane attention SAEs `z` / `z2`) before Saturday's comparison. Only OUR method is tuned. Ge and Dunefsky stay
exactly as published (their own knobs are swept as curves). λ is already a curve on the pod, so this sweep tunes the
other knobs at the pod's λ 1e-3 point.

## 1. Diagnosis (CPU, from the existing panel logs; measured, not guessed)

Sources: `out/panel/wcm_l1e-3*/main/train.shard0.jsonl` (curves at stride 20), the circuits' final gates (node
`attribution_score`), `out/score_smoke/rows.jsonl` (held-out "ho" + home turf = the K = 8 training prompts read at the
anchor), `out/panel/necdiag_summary.md`.

**Did the 400-step fits converge? Yes, in membership.**
- Over the last 100 steps the member count moves by a median of ~1% (at most 6%, on the 35-node 1.z.21179 and the
  NEC_W arm's 8.z.10122).
- The training data terms (zero / floor / pos) end small (≤ 0.02) on 7 of 9 dev fits. Exceptions: 9.z.19003 (zero
  0.66, rising from 0.25 at step 300) and 8.tc.15982 (zero 0.08). The penalty is 85–97% of the final loss.
- The R (donor) term is the largest data term at the end on the deep targets: 1.03 (8.tc.15982), 1.23 (9.z.17708),
  0.79 (9.z.19003), 1.74 (pilot 9.tc.9824).
- The final gates are soft (median kept m 0.60–0.66, min 0.500, binary entropy 0.85–0.92 bits). Production v4 fits
  (`latent_pilot/out/dev48_geo_x3.15_warmup_censor`) look the same (0.56–0.77, 0.65–0.91 bits). So this is how the
  engine behaves everywhere, not an arena A under-convergence.
- **The censored term is inert here by construction.** k = d_sae, so τ_K = the 24,576-th largest *other* latent =
  −inf, and the gate is 1. The logged censored share is 0.000 on every fit. Warm-up / geometric lr are fixed inside
  v4, and nothing in the curves points at them (no late blow-up; arena C's anneal spikes were at constant lr).
- **Rank-keep is active, but it protects nothing here.** The scorer reads post-activation with k = d_sae, so no
  latent is ever censored. Yet v4's γ_R = 0.1 still holds the target's clean rank among the 24,576 pre-activations.
  At the end the rank term is 5.92 on 9.z.19003 (its largest data term), 0.13 on 8.tc.15982 and 0.08 on 4.tc.13440.
  On the pilot it is 0.42 (9.tc.9824) and 4.0 / 1.9 at λ 4e-3, where Z fell.

**Per failing dev target (ho, pass b = Z, A, C in [0.75, 1.5] and nec_b ≥ 0.9; base = v4 λ 1e-3):**

| target | n | ho failure | home (train prompts) | reading → knob |
|---|---|---|---|---|
| 2.tc.17681 | 585 | Z 0.76, nec_b 0.88 (both just at the cut) | passes (Z 1.01, nec_b 0.96) | held-out-only gap at the margins → coverage (λ: pod curve), maybe steps |
| 4.z.22556 | 436 | A 0.74, nec_b 0.71 (nec_b_tc 1.01) | nec_b 0.67: fails in training too | the frozen z error re-supplies the target, a training-side failure → necessity term (NEC_W) |
| 5.tc.22587 | 1906 | nec_b 0.34 (nec_b_tc 0.86) | nec_b 0.60 | same, training-side → NEC_W |
| 8.tc.15982 | 5334 | nec_b 0.67 (nec_b_tc 1.24); Z / A / C fine | Z 0.72, C 0.72 | R is the largest residual (1.03), rank 0.13 → γ_R, steps |
| 9.z.19003 | 2040 | Z 0.35 | Z 0.59 | training-side Z failure: rank term 5.9 against zero 0.66, still growing at the end → γ_R |

Passing under (b): 1.z.21179, 2.z.1360, 4.tc.13440, 9.z.17708 (4 / 9). 2.tc.2019 and 6.z.8399 are vacuous.
11.tc.22949 does not fit on the 16 GB card. NEC_W=1 (the overnight arm, pilot targets) passes under (b) on 1 / 3, the
same as base (8.z.10122). It raises nec_a sharply (3.tc.3752 0.32 → 0.95) but nec_b only slightly (0.52 → 0.69), and
it pushes A out of band (1.54). Hence the smaller 0.3 below.

## 2. Grid (`tune_grid.tsv`; one factor at a time from v4, λ 1e-3, 400 steps; run order = priority)

| config | change | est. GPU-h | why |
|---|---|---|---|
| base | none: the panel circuits, re-scored with the current scorer | 0.05 | the reference (no refit; fits are deterministic) |
| r0 | `AA_RANK_W=0` | 0.74 | rank-keep off (k = d_sae: nothing to protect) |
| nec0.3 | `NEC_W=0.3` | 0.85 | the necessity term, at a weight that should not push A out of band |
| r0.01 | `AA_RANK_W=0.01` | 0.74 | the γ_R dose between r0 and v4 |
| r0_nec0.3 | both | 0.85 | whether the two add (they target different failures) |
| s600 | `STEPS=600` | 1.02 | the untested "more steps" arm (expected null: membership plateaus by ~300) |

**Total ≈ 4.25 GPU-h ≤ 5.** Basis: dev fits at λ 1e-3 sum to 1,983 s, plus eval ~65 s, ~110 s start-up × 4
processes, and ~3 min of scoring. NEC adds ~15% (one more forward per step); STEPS ×1.5.

**Left out, with reasons:**
- λ: the pod sweeps it as a curve.
- The lr multiplier and warm-up: fixed in v4, and the curves give no sign of a problem.
- The censored term: inert at k = d_sae.
- RANK_CUT_K: Top-K only.
- γ (A / C weight) and the R weight: not reachable in MODE=main without touching 062's driver.

## 3. Selection rule (PRE-REGISTERED 2026-11-05, before any tuning run; `tune_summary.py` applies it mechanically)

- Units: the 9 dev targets, held-out strongest (ho).
- A config is **eligible** only when every dev target is either scored or listed in `KILLED`. A fit killed by a
  guard or crashed counts as a **fail**. Any target neither scored nor killed (e.g. cut by STOP_AT) makes the config
  incomplete, and it is not ranked.
- Ranking, in order:
  1. max **P_b** = # dev targets with pass_b on ho (a vacuous target counts as a fail);
  2. max P_a (pass with nec_a);
  3. min median n_latent over the scored dev targets;
  4. fewer knobs changed vs v4 (base = 0, so **v4 wins every full tie**);
  5. grid order.
- The first config is chosen. Nothing else (home-turf counts, per-target sizes, R) enters the choice; those are
  reported only.
- Caveat, stated in advance: with 9 targets a one-target gain is within target-sampling noise. The rule still adopts
  it, and the paper states the dev-set size.

## 4. Dev / production separation

- The dev set (`tune_dev_targets.txt`) is the 9 non-vacuous panel targets.
- `tune_exclude.txt` (15 keys) holds every target that has been looked at: the 12 panel targets (incl. vacuous
  2.tc.2019, 6.z.8399 and 11.tc.22949) plus the 3 pilot targets.
- The pod's targets come from `make_targets.py` on the full data pass. Its frames are seeded shuffles of every
  eligible latent, so overlap is possible. **New option `EXCLUDE=<file>`:** excluded keys are skipped in the
  already-shuffled frame and do not count as a try, so the shuffle and every pick that does not hit an excluded key
  are unchanged.
- Verified on CPU against `out/targets_smoke` (pilot data pass, the 48-target rule):
  - unset: `targets.txt` and `prompts.jsonl` md5-identical to the stored run; `targets.json` differs only in the
    stores path string, which I passed relative;
  - `EXCLUDE=tune_exclude.txt`: picks identical (no dev key falls inside the first two of any frame);
  - excluding the pick 0.tc.1257: it alone is replaced (by 0.tc.330), the other 47 are unchanged.
- With EXCLUDE set, `targets.json` / `MANIFEST.json` gain the exclude fields (file, md5, skipped keys).

## 5. Running it

`bash experiments/107-true-baselines/arena_a/run_tune_a.sh` (WSL, repo root, no arguments). Output goes to
`out/tune/`: `<config>/fit`, `<config>/score`, `<config>/KILLED`, `summary.md`, `summary.json`, `CHOSEN.env`,
`gpu_mem.csv`, `steps.tsv`, `run.log`, `TUNE_DONE`.

How it runs:
- One GPU process at a time. The L1–5 targets share one process, and each L8+ target gets its own.
- Resumable per (config, target).
- Before each launch it waits while card memory.used > 3,000 MiB.
- Guards, checked every 10 s: memory.used > 15,300 MiB, or host MemAvailable < 4 GB → kill the fit and mark the
  target KILLED. STOP_AT (default 08:30) → kill and stop.

Flags: `DRY_RUN=1` prints the commands; `CPU_SMOKE=1` runs the plumbing check (r0, 1.z.21179, 2 steps, CPU,
`out/tune_cpu_smoke`).

## 6. What run_pod.sh needs (A10 owner)

- **P3:** pass `EXCLUDE=$A/tune_exclude.txt` to `make_targets.py`.
- **P5:** source `out/tune/CHOSEN.env` (copy it into the pod bundle) and add its assignments to the driver env:
  - `AA_RANK_W=<w>` is read by `driver_tcz.py`, in any MODE;
  - `NEC_W` is 062's driver env;
  - `STEPS=600` would be the driver's `STEPS`, which run_pod.sh currently labels smoke-only (`AA_STEPS`).
- `CHOSEN.env` for base is empty apart from `AA_TUNE_CONFIG=base`, which means no change.

## Code changes (minimal, listed)

- `make_targets.py`: the `EXCLUDE` option. Unset = the old behaviour, byte for byte.
- `driver_tcz.py`: the `AA_RANK_W` env. It sets `driver.WCM_RANK[version]` and `driver.RANK`, because MODE=main has no
  other route to γ_R. It marks `OUT/AA_RANK_W` and refuses mixed OUTs. Unset = unchanged.
- New: `tune_grid.tsv`, `tune_dev_targets.txt`, `tune_exclude.txt`, `run_tune_a.sh`, `tune_summary.py`.
