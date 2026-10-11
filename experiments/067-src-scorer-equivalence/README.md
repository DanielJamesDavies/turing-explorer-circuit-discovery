# 067 — src scorer equivalence (DAN-73)

2026-10-09. DAN-73 moved the protocol-v1 scorer into src: `src/eval/wcm_eval.py` (`score_circuit`, the pass rule,
vacuity, the graded companion) and `src/eval/wcm_specificity.py` (`specificity_rows`, the amplifier flag, the
near-threshold stratum). The public-model runs (Pythia-70m, GPT-2 small) and any TuringLLM rerun score with them.
This check shows they reproduce the 062 run's scorer on the run's own circuits before anything is quoted from them.

On CPU, `tests/eval/test_wcm_eval.py` already checks field-for-field identity with the experiment code
(`049/amp_eval_pass_v2.score_circuit(c, {}, skip_roles=True)` and `056/specificity.score`) on a tiny mock engine,
in both split modes. This script runs the same comparison on TuringLLM on the GPU.

## What it does

`check_equivalence.py` picks 12 targets stratified over depth band × site kind, plus 4 of the 58 padded mid-band
targets (mid-band eval rows with `n_pos < 64`, protocol-v1.md §3.6). It uses the run's cached contexts
(`062/out_full/out/ctx`) and circuits (`062/out_full/out/main/circuits`). Then:

| | Comparison | Expectation |
|---|---|---|
| (b) | src vs experiment scorer, `as_run` split, held-out strongest and mid-band; specificity src vs 056 | Same GPU, same process: bitwise equal, or float noise if a kernel is nondeterministic |
| (c) | Both vs the stored H100 rows (`eval.shard*.jsonl`, `spec.shard*.jsonl`) | Float-level diffs (H100 vs local GPU); few or no pass-rule verdict flips |
| (d) | `explicit` split on the padded mid-band targets: src vs experiment, and both vs `062/out_rescore_mid/eval_mid.jsonl` | (d) same-GPU equal; vs the re-scored rows equal if they were scored on this GPU |

The report counts, per comparison: fields compared, exact matches, |diff| ≤ 1e-6 / ≤ 1e-3 / larger,
None or type mismatches, missing fields, the largest |diff| with its field, and pass-rule verdict flips (held-out
strongest). Bookkeeping fields (`held`, `arm`, `thin`, `shard`, timings) are not compared.

## How to run

From the repo root (the engine set-up reads `outputs/` relative to it), on a free GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python experiments/067-src-scorer-equivalence/check_equivalence.py
```

- About 72 scorings plus 32 specificity runs; a few minutes after the ~3 min engine start-up.
- Resumable: scored rows go to `out/rows.jsonl` and `out/spec_rows.jsonl`, and a rerun skips them. The selection is
  cached in `out/selection.json`.
- The report goes to `out/report.md`.
- `REPORT=1` re-prints the report without the GPU.
- `KEYS=a,b,c` scores exactly those targets. `N_STRAT` / `N_PAD` change the sample size. `SPEC=0` skips
  specificity.
