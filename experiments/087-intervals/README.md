# 087: every headline number with its n and a 95% interval (DAN-163)

## Question

The paper quotes pass rates, size ratios and paired differences without intervals. What is each number's n and 95%
interval, and which data file does it come from? This is a CPU-only reader: no model, no GPU, no refitting.

## What the code does (`intervals.py`)

It reads the row files the runs write and builds one table of statistics. For each statistic it reports the point
estimate, n, k, and:
- the 95% **Wilson** interval;
- the target-level **bootstrap** interval (resample targets, B = 10,000, seed 20261025 plus a per-statistic crc32
  offset, drawn in fixed blocks so the replicates do not depend on which statistics ran before);
- the exact two-sided **Clopper-Pearson** interval;
- for zero-pass cells, the exact one-sided 95% Clopper-Pearson upper bound (1 − 0.05^(1/n)) and the rule of three
  (3/n);
- the source file(s).

**Which interval is quoted** (the macros; `--quote wilson|bootstrap`):
- a single pass rate quotes the Wilson interval. The bootstrap of a proportion is a resampled Wald interval: it agrees
  with Wilson to about 0.1 pp at n of 10^3 to 10^4, but collapses to [0, 0] at k = 0 and under-covers in small cells;
- medians, size ratios, ratios of pass rates and paired differences quote the percentile bootstrap;
- zero-pass nulls quote the one-sided upper bound.

The pass rule is DAN-8, as in `merge.py`: Z, A and C in [0.8, 1.5] and necessity ≥ 0.9, held-out strongest,
activation read, vacuous rows removed. On the rebuilt held-out sets the headline also needs at least 8 clean held-out
strongest rows (086 `heldout_rule.FLOOR`).

**Sources** (each optional; a missing one is skipped with a note):
- **062, the as-run 15k paper run** (`--run062`, `--e062`): main eval and specificity rows, the 190-target penalty
  sweep (`sweep_rows.csv` + `out_{bands,depth,lowlam}`), α = 1 (`out_alpha1`), the DAN-17 random null
  (`out_random_scale`), and the exact-duplicate re-score (`out_rescore_dedup`).
- **the 082 chain** (`--work WORK`): the 086 re-score on the rebuilt held-out sets, the as-run re-score of the frozen
  1,000-target subset (leakage, paired), the 084 harder nulls, the 083 method comparison (DAN-159 Table 2), and the
  082 public runs (pythia-70m, gpt2-small).

**Outputs**, in `OUTDIR`:
- `intervals.md`: the table;
- `intervals.tex`: one `\newcommand` per statistic. `\X` gives e.g. "46\% [45, 47]", `\XN` the n, `\XFull` both;
- `intervals.json`: every statistic, machine-readable.

## Files and commands

| File | What |
|---|---|
| `intervals.py` | the reader and table builder |
| `test_087_intervals.py` | CPU tests: bootstrap determinism and chunking, Wilson / Clopper-Pearson against published values, the scipy-free beta fallback, paired bootstrap, macro names, the pass rule against `merge.py`; reproduction of the paper's 062 numbers and the 086 smoke leakage reader (skipped when those rows are not on disk) |

```
python experiments/087-intervals/intervals.py                                   # 062 outputs -> out/today
python experiments/087-intervals/intervals.py --no-062 --work <WORK> --name pod # a chain's outputs -> out/pod
python -m pytest experiments/087-intervals/test_087_intervals.py -q
#   options: --B 10000 --seed 20261025 --quote wilson|bootstrap --out DIR --rescore-v2 DIR --rescore-asrun DIR
#            --asrun-targets FILE --nulls084 LABEL=DIR --dan159 DIR --public FAMILY=DIR --lowlam-shards paper|all
```

## Results

`out/` is gitignored, so the outputs are not tracked. Two output sets exist on the author's machine:

- `out/today/`: the 062 as-run numbers. This is the table the paper's current numbers can be checked against.
- `out/smoke_chain/`: a plumbing check on the 082 chain's smoke output (`082-public-runs/smoke_final2`). Its cells
  hold 1–2 targets, so its numbers mean nothing; it only shows that every chain section is read.

Selected rows of `out/today/intervals.md` (Wilson 95% for rates, bootstrap for differences and ratios):

| statistic | value | n | macro |
|---|---|---|---|
| pass rate, all targets | 46.4% [45.6, 47.2] | 14,994 | `\passAll` |
| layers 0-4 / 5-7 / 8-11 | 69.2% / 45.2% / 22.3% | 5,812 / 3,886 / 5,296 | `\passShallow` etc. |
| attn / mlp / resid | 14.3% / 51.8% / 65.0% | 4,093 / 5,449 / 5,452 | `\passAttn` etc. |
| α = 1, all targets | 1.5% [1.0, 2.2] | 1,746 | `\aoneAll` |
| fitted circuits, same targets | 44.7% [42.4, 47.0] | 1,746 | `\aoneFitted` |
| random null (DAN-17) | 0/747, one-sided upper 0.40% | 747 | `\nullAll` |
| sweep, WCM at λ 2.5e-4 / 1e-3 (190 targets) | 57.9% / 41.6% (median 1,102 / 472 latents) | 190 | `\swLam...` |
| smallest passing circuit, unweighted / WCM, layers 8-11 | 10.5x [7.3, 19.0] | 26 | `\swRatioDeep` |
| passes at some penalty, layers 8-11, WCM / unweighted | 80.3% / 40.9% | 66 | `\swEverWcmDeep`, `\swEverUnwDeep` |
| exact-duplicate re-score minus as run (paired) | −0.6 pp [−1.6, +0.3] | 1,799 | `\dedupEffect` |

The table also has every band x kind and per-layer cell, the held-out mid-band rate (12.1%, n = 14,316),
specificity (amplifier flags) and the paired sweep differences.

**In the paper:** `main.tex` does not `\input` `intervals.tex` yet. Comments at lines 105, 159, 170 and 402 name
the 087 macros (e.g. `\passall`) to replace the as-run numbers once the rebuilt-held-out re-score is in, and line 1140
marks the intervals as pending for the v2 run.

## Status and caveats

- The 062 numbers are on the **as-run held-out sets**, which carry the duplicate-context leakage (085 / 086,
  DAN-157). The rebuilt-set and harder-null sections only fill in when pointed at a chain `WORK` dir that has them.
  No full-chain output is in this folder.
- The paper's production protocol has since moved to v2–v4 (`paper/protocol.md` §0, §12). The 062 numbers here are
  protocol v1. Running this on a v4 run needs that run's row files in one of the layouts above.
- The tests were not re-run for this README.
