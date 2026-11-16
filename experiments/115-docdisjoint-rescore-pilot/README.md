# 115 Document-disjoint held-out re-score: local pilot (DAN-217)

## Question

114 found that about 35% of each TuringLLM target's held-out strongest rows paraphrase, or come from the same
document as, a training row of the same target. If the held-out strongest set is rebuilt so that it shares **no
source document** with the target's training contexts, do the **fixed** production circuits (WCM v4, T3) still pass?
No refit. This is a pilot on 36 targets on the local RTX 5070 Ti. The full re-score is for a pod.

## Inputs (read only)

- Production circuits and rows: `109-pod-run-analysis/out/archives/pub-turingllm-T3-main/.../094-tl-v2-records/out_v2/main`.
- Production records: `.../pub-turingllm-T2-records/.../out_turingllm/ctx_fresh/<key>.pt`.
- T1 scan (top 512 per target): `.../pub-turingllm-T1-scan/.../out_turingllm/scan/scan_top512.pt`.
- Groups: 085 `corpus_src.npy` / `corpus_hash.npy` and `data/text_paths.txt`, through 114 `check.groups()`.

## Method

**Scorer: the production one, unchanged.** `062/driver.py MODE=main` with `CTX_DIR` set and an `OUT` that holds
copies of the production circuits and the production markers (`WCM_VERSION` v4, `WCM_R`). `Runner.fit` loads an
existing circuit instead of fitting, so nothing is refitted. `SKIP_SPEC=1` (specificity is not in the pass rule).
Pass rule and vacuous rule: `062/merge.py` via 109 `lib109`.

**Targets** (`select_targets.py`, seed 20261115): scored, non-vacuous production targets, 4 per depth band × site
kind cell (3 bands × attn / mlp / resid), 3 passes + 1 fail per cell = 36 (27 pass, 9 fail). The sample is
fail-enriched on purpose, so its pass rate is not the production rate.

**Held-out rebuild** (`build_heldout.py`). This is 086's DAN-157 held-out rule (its own helpers: `forbidden`,
`fire_fn_for`, `stable`, `heldout_rule.select_heldout`) with one added clause:

- Forbidden groups: the group of every activating training row (strong.train 48 + mid.train) and of each row's
  corpus twins (same content hash).
- Candidates: the original held-out rows first (pool order), then the T1 scan rows strongest first. A candidate in a
  forbidden group is dropped. The rest go through the 086 rule unchanged: firing, exact / anchor-prefix / near vs
  training rows and the 64 contrast rows, no mid-pool rows, and the in-batch firing re-check.
- Fill to the original size (16). Only `strong.held` and appended rows change. Training rows, mid-band and contrast
  rows are the record's own.
- Two variants: **doc** (original + its T5 paraphrases) and **topic** (the sibling documents of one leaf topic;
  stricter).

**Sanity check first:** re-score on the original records and compare with the production rows. The comparison
new vs old is paired against this local re-score, so GPU float noise does not enter the deltas.

## Run

```
python -X utf8 experiments/115-docdisjoint-rescore-pilot/select_targets.py      # CPU
python -X utf8 experiments/115-docdisjoint-rescore-pilot/prep_rescore.py        # CPU: out/rs_{orig,doc,topic}
wsl: MODE=main ORDER=list SKIP_SPEC=1 WCM_VERSION=v4 OUT=<115>/out/rs_orig CTX_DIR=<T2 ctx_fresh> \
     TARGETS=<115>/targets_pilot.txt PYTHONPATH=src .venv/bin/python -X utf8 experiments/062-h100-protocol-v1/driver.py
wsl: PYTHONPATH=src .venv/bin/python -X utf8 experiments/115-docdisjoint-rescore-pilot/build_heldout.py
wsl: (driver as above with OUT=out/rs_doc CTX_DIR=out/ctx_doc, then rs_topic / ctx_topic)
python -X utf8 experiments/115-docdisjoint-rescore-pilot/compare.py             # CPU: out/report.md
```

## Results

### Pilot comparison (2026-11-15, paired local design approved by the coordinator)

Both held-out sets were scored on this GPU (bit-deterministic). Production rows are a reference column only. The full
tables are in `out/report.md` and the per-target rows in `out/compare.csv`. Builds were checked independently with
`check_build.py`: 0 new rows share a document / topic with a training row (twins included), the training-side fields
are byte-identical, and every size matches except the one short target below. The sample is fail-enriched (27 pass /
9 fail in production; 28 / 8 in the local re-score), so read the flip rates conditional on the old verdict.

| | doc-disjoint | topic-disjoint |
|---|---|---|
| old held-out rows in a forbidden group | 233 / 536 (43%) | 277 / 536 (52%) |
| candidates allowed (median of 464) | 312 | 250 |
| too few candidates (short of the original size) | 1 (9.attn.18398: 1 row) | 1 (9.attn.18398: 0 rows, unscorable) |
| overlap with the old set (median rows of 16) | 8 | 6 |
| strength, new / old median peak | 0.998 | 0.996 |
| new rows sharing a group with contrast-TRAINING rows | 19 / 521 (10 targets ≥ 1) | 32 / 520 (16 targets) |
| old pass → new fail | 2 / 28 | 4 / 28 (3 / 27 excluding the unscorable target) |
| old fail → new pass | 2 / 8 | 2 / 8 |
| median ΔZ / ΔA / ΔC / Δnec | −0.003 / +0.007 / +0.003 / 0.000 | +0.001 / +0.015 / +0.001 / 0.000 |
| median abs ΔZ / ΔA / ΔC (p90) | 0.025 (0.10) / 0.028 (0.10) / 0.019 (0.09) | 0.030 (0.12) / 0.027 (0.10) / 0.022 (0.09) |

By band (pass→fail / fail→pass), doc: L0-4 0 / 0, L5-7 2 / 1, L8-11 0 / 1. Topic: L0-4 0 / 0, L5-7 2 / 1, L8-11 2 / 1
(one of the L8-11 losses is the unscorable 9.attn.18398).

The flips:
- pass → fail
  - **7.mlp.17228** (both variants): A rises to 1.60 / 1.64, above the 1.5 ceiling. Its A was already 1.20 in
    production vs 0.99 locally, so this target is hardware-sensitive.
  - **7.resid.31432** (both): fails in production; passed locally only through the GPU difference (Z 0.76); the new
    sets put Z at 0.71 / 0.74.
  - **10.resid.4758** (topic only): Z 0.86 → 0.73.
- fail → pass (both variants): **6.mlp.22994** (A 0.72 → 0.76 / 0.78) and **10.resid.23442** (C 0.71 → 0.78 / 0.77).
  Both sat just under the 0.75 floor.

Below floor 8 on the new sets: 4 targets. Three of them (3.attn.24643, 7.attn.27383, 11.attn.25336) already had only
2–4 held-out rows in the production records. The fourth is 9.attn.18398: all 464 of its candidates come from 20
documents that also supply its training rows.

**Pilot reading (36 targets, no stronger claim).** Most deltas are within the noise that separates two GPUs on the
same circuits (median 0.03), even though 43–52% of the old held-out rows were replaced. The flips are almost all
targets that sat within about 0.05 of a pass-rule boundary on the old set. Nothing here rules out an effect of a few
percentage points on the full 7,464. The pod re-score is what can measure that.

### Sanity check (2026-11-15)

The pass decision agrees on 35 of 36 targets and the median gap is 0.03; the details follow.

The local re-score of the 36 production circuits on their ORIGINAL records (`out/rs_orig`, log `out/rs_orig.log`;
fit 0.0 s on every target, so the circuits were loaded and not refitted) **does not reproduce the production rows
within small noise.** `out/sanity.csv` and `out/report.md` hold the numbers.

- Exactly equal on Z / A / C / necessity: 1 of 36. `a_pos_tk` agrees to within 0.004, so the clean activations match.
  The differences are in the ablated reads.
- Max |diff| over Z / A / C / necessity per target: median 0.030, p75 0.047. 7 of 36 targets exceed 0.06 (086's
  measured H100-vs-RTX noise on main).
- Largest differences:

  | target | prod Z / A / C / nec | local Z / A / C / nec |
  |---|---|---|
  | 7.attn.27383 (fail) | 0.46 / 0.49 / 0.87 / 1.00 | 0.70 / 0.66 / 0.43 / 1.00 |
  | 11.attn.25336 (fail) | 0.57 / −1.70 / 0.66 / 1.00 | 0.53 / −1.42 / 0.60 / 1.00 |
  | 7.mlp.17228 (pass) | 0.92 / 1.20 / 0.98 / 0.98 | 0.91 / 0.99 / 0.97 / 0.98 |
  | 6.attn.21701 (pass) | 1.00 / 1.17 / 0.92 / 1.00 | 1.01 / 1.14 / 0.92 / 1.12 |

- The pass decision agrees on 35 of 36. One target flips: **7.resid.31432** goes from fail to pass, because Z moves
  from 0.659 to 0.758, across the 0.75 floor.
- **The local scorer is deterministic.** A second local run of the 6 worst targets (`out/rs_orig_rep`) is bit-identical
  (all diffs 0.0). So the gap is between machines (H100 pod vs RTX 5070 Ti), not run-to-run noise. 086 saw the
  same effect: up to 0.06 on main and 0.47 on a failing sweep arm. It concluded that a paired comparison must re-score
  both sets on the same hardware.

The sanity check therefore fails a strict reproduction test, and the pilot was paused. The coordinator then approved
the paired local design: old set and new set both re-scored locally, both bit-deterministic. The steps that were run:

```
wsl: PYTHONPATH=src .venv/bin/python -X utf8 experiments/115-docdisjoint-rescore-pilot/build_heldout.py
wsl: driver as above with OUT=<115>/out/rs_doc   CTX_DIR=<115>/out/ctx_doc
wsl: driver as above with OUT=<115>/out/rs_topic CTX_DIR=<115>/out/ctx_topic
python -X utf8 experiments/115-docdisjoint-rescore-pilot/compare.py
```

Logs: `out/build_test.log` (2-target test), `out/build.log`, `out/rs_doc.log`, `out/rs_topic.log`. In the topic run,
9.attn.18398 errors (the explicit split needs at least one held-out row); `compare.py` counts it as not passing.
