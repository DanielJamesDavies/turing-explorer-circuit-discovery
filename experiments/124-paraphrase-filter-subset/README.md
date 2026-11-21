# 124 Paraphrase-filtered held-out sets: subset re-score (DAN-217)

## Question

114 found that about 35% of TuringLLM's held-out strongest rows are paraphrases of the same target's training rows.
The corpus holds 11 T5 paraphrases of every Phi-3 original, made chunk by chunk. The DAN-157 rule (exact,
anchor-prefix and 8-gram near copies) does not catch them.

This experiment asks: if those rows are dropped and refilled, do the **fixed** production circuits still pass?

Daniel's design (2026-11-16) replaces 123's document-group rebuild. Same-file or same-topic windows in both splits are
fine; only near-identical sequences matter. Status (2026-11-16 05:43): **DONE.** The GPU build and both re-scores ran
on the local RTX 5070 Ti from 04:28 to 05:43. Refit guard OK on both variants. Results are in **GPU results** below and
in `out/report.md`.

## GPU results (2026-11-16; `out/report.md`, `report.json`, `report_cells.csv`, `report_pairs.csv`)

**Answer.** Dropping and refilling the paraphrased held-out rows leaves the fixed production circuits' pass rate
unchanged within noise. The paired difference is **-0.7 pp [-3.0, +1.7]** (n = 296), and the interval includes 0. With
n = 296, an effect larger than about 3 pp in either direction is ruled out. The flips go both ways (7 pass→fail,
5 fail→pass). That count is about what the local re-score's own disagreement with production would produce (3.3% of
targets, about 10 flips out of 300).

| | pass rate [95% bootstrap] (n) |
|---|---|
| production rows of the subset (H100, reference) | 85.7% [81.7, 89.3] (257/300) |
| orig re-score, local GPU (paired baseline) | 86.0% [81.9, 89.6] (257/299); 1 vacuous; pass call agrees with production on 96.7% |
| para re-score, own population | 85.5% [81.4, 89.2] (253/296); 2 vacuous |
| **paired** (296 non-vacuous on both) | **orig 86.1% [82.1, 89.9] (255/296) → para 85.5% [81.4, 89.2] (253/296)** |
| paired difference | **-0.7 pp [-3.0, +1.7]**; pass→fail 7, fail→pass 5 |
| conservative (orig's 299; para's vacuous / unscorable / missing count as fail) | 86.0% → 84.6% [80.3, 88.6]; difference -1.3 pp [-3.7, +1.0] |
| mid-band (paired 243) | 53.5% → 53.5%; difference 0.0 pp; 0 flips |

By band:
- L0-4: +0.0 pp [-2.5, +2.5] (n 118).
- L5-7: +0.0 pp [-3.8, +3.8] (n 79).
- L8-11: -2.0 pp [-8.1, +4.0] (n 99; 5 p→f, 3 f→p).

By kind:
- resid: -0.9 pp (n 106).
- mlp: +0.0 pp (n 109).
- attn: -1.2 pp (n 81).

No cell's interval excludes 0. The deep band holds the largest point estimate, but the interval there is wide.

**GPU rebuild** (`build_para.jsonl`, bf16, check 300/300, 0 FAIL):
- 1,457 / 4,636 old held rows (31.4%) were dropped as paraphrases, in 276 targets.
- 3,179 old rows were kept; 0 were rejected by the 086 rule or the re-check.
- 1,445 rows were refilled from the T1 scan, in 267 targets; median 4 per target, median scan rank 86, median peak
  new / old 1.000.
- 9 sets are short of the original size and 12 are below floor 8.
- 1 set is unscorable (11.attn.7471).
- Mid-band: 4,478 → 4,439.

Wall clock:
- build: 9 min;
- check + prep: 2 min;
- orig: 31 min;
- para: 31 min.

## Design

**Rule (TuringLLM only).** A held-out row is dropped if it is a paraphrase of any of its target's activating training
rows (strong.train 48, which includes the fit's 32, plus mid.train). Paraphrase means a TF-IDF cosine of 0.3 or more,
using 114's vectoriser and settings exactly:
- hashed token unigrams and bigrams, 2^20 buckets;
- sublinear tf and smooth idf, with L2-normalised rows;
- IDF fitted on 114's own 100k-window sample.

114's code is imported, not re-implemented. The rebuilt IDF reproduces 114's cosine on **all 6,009** of 114's held rows
with a max difference of 0.0 and identical threshold calls (`out/idf_verify.json`). This rule comes **on top of**
DAN-157.

**Refill.** As in 115 / 123, the refill uses `115 build_heldout.build_one`, unchanged:
- Candidates are the original held rows in pool order, then the T1 top-512 scan, strongest first.
- The paraphrase test is applied to every candidate first. After that comes 086's rule: firing at the anchor; no
  exact, anchor-prefix or near copy of training or contrast rows or of accepted rows; no mid-pool rows; an in-batch
  firing re-check.
- Fill to the original size (16), with floor 8 (below it, a target is flagged and still scored).

The rule plugs into `build_one` through its group interface. `ParaGroups.of(id)` returns `{"P"}` for a training id or
a paraphrase of a training row, and `{}` otherwise, so the forbidden set is `{"P"}`. Training-side fields are the
production record's, byte for byte, and `STAGE=check` verifies this.

**Mid-band.** The same test, drop only (`common123.drop_mid`).

**Contrast held-out sets.** Measured only.

**Subset.** 300 TuringLLM production targets (`targets_subset.txt`, `out/subset.json`):
- drawn from the 7,464 non-vacuous production targets;
- stratified by depth band × site kind with proportional allocation (largest remainder), 25–45 per cell;
- seed 20261116.

The sample's production pass rate is 85.7% (257/300), against 88.0% in the population.

**Re-score.** The fixed production circuits are scored with no refit by the production scorer, `062 driver.py
MODE=main`, run as 115 ran it. The OUT holds the production markers and hard links to the production circuits.
- `orig`: the production records, re-scored on the same local GPU (the paired baseline).
- `para`: the filtered records.
- `verify` is the refit guard.

**Report** (`report124.py`). The statistics are 123's `report.py`, imported, which itself imports 062 merge / 109
lib109 (95% percentile bootstrap, B = 4000).
- paired pass rates with intervals and n;
- the paired difference with a paired bootstrap interval;
- pass→fail and fail→pass transitions;
- the conservative variant (unscorable / vacuous / missing count as fail);
- band × kind cells with margins and transitions;
- mid-band;
- rows dropped and refilled, and vacuous / unscorable targets.

## CPU dry run (2026-11-16; `out/report.md`)

**Paraphrase test alone** (`out/plan_para.jsonl`):

| | subset (300 targets) |
|---|---|
| old held-out strongest rows that are paraphrases (dropped) | 1,457 / 4,636 (31.4%) |
| targets with ≥ 1 dropped / all rows dropped | 276 / 3 |
| median allowed candidates (orig held + scan) | 378.5 |
| certainly short / below floor 8 / unscorable | 4 / 7 / 0 |
| mid-band held rows dropped | 39 / 4,478 (0.9%) |
| contrast held rows that paraphrase a contrast-train row / an activating training row | 65 / 4,800 (1.4%) / 21 / 4,800 (0.4%) |

Dropped share by cell: L0-4 attn 20.0%, mlp 33.2%, resid 39.7%; L5-7 21.0%, 27.0%, 44.2%; L8-11 32.0%, 26.6%,
32.4%.

**Full rebuild with the forward on the CPU** (`out/build_para_cpu.jsonl`, checked by `out/check_para_cpu.jsonl`).
This is the same code as the GPU build, but in fp32 rather than bf16, so a borderline firing call can differ. The
scorer never reads these records.

| | subset (300 targets; 0 errors; check 0 FAIL) |
|---|---|
| old held rows dropped as paraphrases | 1,457 / 4,636 (31.4%), 276 targets |
| old rows kept / rejected by the 086 rule or the stability re-check | 3,176 / 3 |
| rows refilled from the T1 scan | 1,446 (267 targets; median 4 per target; median scan rank 86) |
| strength, median peak new / old | 1.000 |
| short of the original size | 11 |
| below floor 8 (includes sets already below 8 in production) | 12 |
| **unscorable (0 rows)** | **1** (11.attn.7471) |
| mid-band held rows | 4,478 → 4,439 (no set emptied) |

The CPU build took 28 min (about 5 s per target). The GPU build should take about 5 min.

## Local GPU time (`out/time_estimate.json`)

The estimate takes the subset's production `t_eval` (1,067 s on the H100) and multiplies it by the measured local /
production ratio from the 115 pilot (1.63, same circuits on this RTX 5070 Ti). It then adds the measured driver
start-up (210 s per variant) and 115's build cost.

| step | time |
|---|---|
| build (GPU) | ~5 min |
| score orig | ~33 min |
| score para | ~33 min |
| total with the CPU steps | **~1.2 h** |

This is well under the 3 h target, so the subset was not shrunk. About 860 targets would fit in 3 h.

## Public models (measure only; `public_check.py`, `out/public_check.json`)

The public production corpora are not on this machine; `data_public/` is the 069 smoke corpus. So the IDF is fitted on
the de-duplicated rows of each model's own production records (about 86k rows). Everything else is 114's vectoriser.

Control: on the TuringLLM subset, the records-only IDF gives 1,410 hits against 1,457 with the corpus IDF (30.4% vs
31.4%). So the IDF source moves the count by about one point.

| | Pythia-70m (509 targets) | GPT-2 small (512) |
|---|---|---|
| held-out strongest rows with cos ≥ 0.3 to a training row of the same target | 2,147 / 7,476 (28.7%) | 1,009 / 7,695 (13.1%) |
| targets with ≥ 1 | 377 | 276 |
| best match in the same source document (per-row `doc`) | 1,221 (57% of hits) | 586 (58%) |
| against another target's training rows (false-positive baseline) | 12 (0.16%) | 1 (0.01%) |
| mid-band held rows | 773 / 7,889 (9.8%) | 178 / 8,051 (2.2%) |

The public corpora contain no generated paraphrases. A hit there means TF-IDF-similar text: the same document, or
recurring boilerplate in the Pile / OpenWebText. The threshold was calibrated on TuringLLM's T5 paraphrases, so these
counts are "near-identical by the same test", not "paraphrases". No rule was applied.

## Proposed paper text (NOT in paper/; numbers filled from the 2026-11-16 GPU run)

The subset gives 31.4% for {P_DROP}; 114's full-set figure is ~35%. Pick the one the sentence should carry.
Interval-type figures are the paired ones.

**Held-out appendix (`app:heldout`), after the paragraph on the production records:**

> The TuringLLM corpus also contains paraphrases: each Phi-3 document appears with 11 T5 rewrites of it, made chunk by
> chunk, and $86.6\%$ of the interpretability corpus's windows come from these rewrites. The copy rule does not catch a
> paraphrase (a known paraphrase shares a median of $5\%$ of its 8-grams with the original), so we tested for them
> directly. A held-out row counts as a paraphrase of a training row when the cosine similarity of their TF-IDF vectors
> (token unigrams and bigrams) is at least $0.3$. This threshold recovers $75\%$ of known paraphrase pairs, and
> $0.02\%$ of held-out rows pass it against another target's training rows. By this test, $31.4\%$ of TuringLLM's
> held-out strongest contexts paraphrase a training context of the same target. We therefore re-scored the fixed
> production circuits of $300$ targets, stratified by depth band and site kind, on held-out sets with these rows
> removed and refilled from the corpus scan under the same rule ($1{,}445$ rows replaced; $9$ sets below the
> original size, $1$ empty). Both sets were scored on the same GPU. On the $296$ targets non-vacuous under both, the
> pass rate moved from $86.1\%$ [$82.1$, $89.9$] to $85.5\%$ [$81.4$, $89.2$], a paired difference of $-0.7$ points
> [$-3.0$, $+1.7$] ($7$ pass→fail, $5$ fail→pass; \cref{tab:para-rescore}). The same test flags $28.7\%$ of
> Pythia-70m's and $13.1\%$ of GPT-2 small's held-out strongest contexts. Their corpora hold no generated paraphrases,
> and most of these hits share a source document with the training match; no rule was applied there.

(Optional table `tab:para-rescore`: the band × kind cells from `out/report_cells.csv`: n, orig, para, diff,
transitions.)

**Model card (TuringLLM, training-data row), appended:**

> The interpretability corpus is drawn from the same text, so most of its windows have paraphrased near-twins
> elsewhere in it; held-out contexts are filtered against them (\cref{app:heldout}).

## Files

- `para124.py`: the rule (`ParaRule`, `ParaGroups`, `contrast_para`, `plan_one`), 114's vectoriser (imported), the
  portable corpus reader, the IDF sample and cache, the subset draw, and `training_unchanged`.
- `select_subset.py`: `targets_subset.txt` and `out/subset.json`.
- `rebuild124.py`: the stages.
  - `idf`, `verify`: CPU; 114 reproduction.
  - `plan`: CPU.
  - `build`: `DEVICE=cuda` is the real build; `DEVICE=cpu` is the dry run.
  - `check`.
- `rescore124.py`: `prep`, `cmd`, `verify` (the refit guard).
- `report124.py`: `out/report.md`, `report.json`, `report_cells.csv` and `report_pairs.csv`.
- `public_check.py`: the public measure plus the IDF-source control.
- `estimate_time.py`: `out/time_estimate.json`.
- `run_local.sh`: the GPU runner, which checks the GPU lock.
- `test_124.py`: 15 CPU tests, all passing. They cover:
  - the rule: flags, threshold inclusive, equal to 114's formula, the interface, plan, mid, contrast, check;
  - determinism: the IDF sample is 114's first draw, the plan, the subset;
  - resume: plan with a torn line, prep idempotent;
  - report pairing from driver-style rows;
  - 114 reproduction on real data;
  - the corpus reader equals the record rows.

## Open questions (for Daniel)

1. **Subset size.** 300 takes ~1.2 h locally, and ~860 fits in 3 h. A larger n would narrow the paired interval.
2. **Paraphrases among held rows.** The test compares candidates only with training rows. Two held rows that
   paraphrase each other are allowed, as copies are not (DAN-157 rejects them). Should the test also apply within the
   held set?
3. **Contrast.** 1.4% of contrast held rows paraphrase a contrast-training row. Is "measured only" final?
4. **Public.** 29% of Pythia and 13% of GPT-2 rows hit the test, about 42% of them across documents (boilerplate or
   Pile duplicates?). Is report-only the decision, or should the same filter be applied (removal only, as 123's public
   path)?
5. **123's fate.** Should the 123 document-group mode be kept for reference (it is intact and off by default) or
   retired?
