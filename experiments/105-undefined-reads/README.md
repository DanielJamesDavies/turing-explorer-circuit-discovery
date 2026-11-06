# 105: undefined pass-rule reads that the vacuous flag misses

CPU only; reads stored rows, fits and scores nothing. Script: `undefined_reads.py` (prints everything below; writes
`out/summary.json`, `out/flagged.csv` (every undefined-but-not-vacuous target, all sources), `out/run.log`).

    cd <repo>; PYTHONPATH=src .venv/bin/python experiments/105-undefined-reads/undefined_reads.py

**Status: proposal for Daniel.** `paper/protocol-v1.md` and `src/eval/wcm_eval.py` are NOT edited. The draft §12
text and a patch proposal are at the end.

## 1. The question

In exp 102, 6 of 144 targets (3 of 72 on each panel) have an undefined pass-rule term (`None`) while `vacuous_tk` is
False, so they fail in every arm:

| panel | target | undefined | a_pos_tk | baseline of that term | baseline / a_pos | circuit's raw read / a_pos (base arm) | passes every defined term (base) |
|---|---|---|---:|---|---:|---:|---|
| dev | 11.attn.12206 | A | 3.15 | eM_topk = 4.29 | 1.36 | 1.10 | yes |
| dev | 2.attn.16773 | A | 0.286 | eM_topk = 0.310 | 1.08 | 0.94 | no (Z 0.59) |
| dev | 5.resid.35667 | Z, necessity | 12.4 | e0 = 636 | 51 | 0.92 | yes |
| check | 1.resid.33347 | Z, necessity | 3.83 | e0 = 7.65 | 2.0 | 0.97 | yes |
| check | 3.resid.26238 | Z, necessity | 11.0 | e0 = 131 | 11.9 | 0.79 | yes |
| check | 9.resid.40186 | Z, necessity | 35.1 | e0 = 3296 | 94 | 0.90 | yes |

The same three dev targets are the three undefined ones in exp 095 (the dev panel is 095's 72 targets).

## 2. Cause

Every pass-rule term is a ratio against an **empty-circuit** baseline (`src/eval/wcm_eval.py`):

| term | field | form | undefined (`None`) when |
|---|---|---|---|
| Z | `free0_tk` | (x − e0) / (a_pos − e0) | e0 ≥ a_pos (`ratio`, l. 99-101) |
| A | `freeM_topk_tk` | (x − eM_topk) / (a_pos − eM_topk) | eM_topk ≥ a_pos |
| C | `freeN_topk_tk` | (x − eN_topk) / (a_pos − eN_topk) | eN_topk ≥ a_pos |
| necessity | `phi_sup_blind_tk` | (a_pos − x_removed) / (a_pos − e0) | a_pos − e0 ≤ 1e-9 (`necessity`, l. 104-107) |
| vacuous | `vacuous_tk` | \|a_pos − e0\| < 0.05·\|a_pos\| | (l. 110-113, set at l. 492) |

The vacuous flag misses these targets for two independent reasons:

1. **It is two-sided.** It fires only when e0 lies within ±5% of a_pos. `ratio()` and `necessity()` give up for
   every e0 ≥ a_pos. So an e0 above 1.05·a_pos (all four resid cases, e0 = 2× to 94× a_pos) is undefined but not
   vacuous. The rule is non-monotone in e0: at e0 = 1.04·a_pos the target is excluded, at 1.06·a_pos it is a forced
   fail.
2. **It looks at the Z / necessity denominator only.** It never sees eM_topk or eN_topk, so an A or C baseline at or
   above a_pos (both attention cases) is never flagged. This part is deliberate in v1: protocol §6 and the
   `is_vacuous` docstring say "an undefined A or C ratio is **not** vacuous: it counts as a fail", and so does
   `paper/main.tex` l. 345.

The two mechanisms:
- **Z / necessity (resid and early MLP).** Zeroing every upstream latent puts a residual site's input far off its
  range, and the target's post-Top-K read comes out far above its natural value: in the v1 15k, e0/a_pos has median
  5.6 (p10 1.26, p90 719). The circuit itself stays in range (circuit-only read / a_pos: median 0.98, p10-p90
  0.88-1.02). This is the post-Top-K version of the blow-up that made §12.1 move the pre-activation Z baseline to 0.
- **A (mostly attention; deep MLP at L8-11).** The sparsity-preserving A fill (the mean code of the 48
  strongest-train rows) alone drives the target above its held-out natural mean: eM_topk/a_pos has median 1.17 (p10
  1.03, p90 1.48). The A test then has no gap for the circuit to close. It is not mainly a near-threshold effect: 9.6%
  of the A-undefined targets are in the near-threshold stratum, against 7.6% of all targets.

In every case the denominator is **wrong-signed** (e ≥ a_pos). None of them comes from a_pos ≤ 0 (Top-K-censored
target) or from a denominator that is exactly zero.

**It is a property of the target and its contexts, not of the circuit.** a_pos, e0, eM_topk and eN_topk are
measured with no circuit, from the held-out rows, the 48 strongest-train rows (A mean) and the 48 contrast-train rows
(C mean). Checked on every stored row:
- the stored `None`s are exactly the rows whose baselines fail the scorer's condition: 0 mismatches over 4 terms in
  each of the v1 15k strongest and mid-band sets, the dedup rescore, 095 and both 102 panels;
- every defined term recomputed from the row's raw reads matches the stored value (0 mismatches);
- each target's baselines are identical across all of its arms (095: 3 arms, 102 dev: 4 arms, 102 check: 3 arms).

So, under v1 these targets fail before any circuit is fitted.

## 3. Prevalence (undefined term, `vacuous_tk` False)

v1 15k, held-out strongest (primary; `062/out_full/out/main/eval.*.jsonl`, held = strong, deduplicated as in
`merge.py`): **482 of 15,004 targets (3.2%)**. 301 have A undefined, 180 Z + necessity, and 1 Z + C + necessity.
The v1 flag itself excludes 10.

| band | resid | mlp | attn | all |
|---|---|---|---|---|
| L0-4 | 78 / 2,269 (3.4%; Z/N 78) | 38 / 2,274 (1.7%; Z/N 37, A 1) | 51 / 1,279 (4.0%; A 49, Z/N 2) | 167 / 5,822 |
| L5-7 | 34 / 1,365 (2.5%; Z/N 34) | 2 / 1,363 (0.1%) | 32 / 1,158 (2.8%; A 32) | 68 / 3,886 |
| L8-11 | 28 / 1,819 (1.5%; Z/N 28) | 22 / 1,820 (1.2%; A 22) | **197 / 1,657 (11.9%; A 196)** | 247 / 5,296 |
| all | 140 / 5,453 | 62 / 5,457 | 280 / 4,094 | **482 / 15,004** |

The other sets:

| set | targets | v1-vacuous | undefined, not vacuous | Z / necessity | A (or A + C) |
|---|---:|---:|---:|---:|---:|
| v1 15k held-out strongest | 15,004 | 10 | **482 (3.2%)** | 181 | 301 |
| v1 15k held-out mid-band (reported) | 14,339 | 23 | **2,100 (14.6%)** | 287 | 1,813 (L8-11 attn 684 / 1,416) |
| v1 dedup rescore (`out_rescore_dedup`, 200 per cell) | 1,800 | 1 | 61 (3.4%) | 13 | 48 |
| 095 (= 102 dev targets) | 72 | 0 | 3 | 1 | 2 |
| 102 dev | 72 | 0 | 3 | 1 | 2 |
| 102 check | 72 | 0 | 3 | 3 | 0 |

On mid-band, A is undefined far more often: the A fill is the strongest-train mean, while a_pos is the (lower)
mid-band mean.

## 4. Options and their effect

All four exclusion rules are functions of (a_pos, e0, eM_topk, eN_topk) only, so none of them can be moved by the
fit. "Pass" below is the current rule (Z, A, C in [0.75, 1.5], necessity ≥ 0.9, post-Top-K). No rule changes any
target's pass / fail. The excluded targets all fail under v1, so a rule only changes the denominator.

- **R0, keep as failures (v1 as written).**
- **R1s, signed Z only:** vacuous when a_pos − e0 < 0.05·a_pos. This fixes the resid / MLP cases but not A.
- **R1, undefined → vacuous:** v1 flag OR any term undefined by its baseline (e ≥ a_pos).
- **R2, signed 5% on every pass-rule denominator (recommended):** vacuous when a_pos ≤ 0 or a_pos − e < 0.05·a_pos
  for any e in {e0, eM_topk, eN_topk}, i.e. any baseline above 0.95·a_pos. It contains R0 and R1 (no v1-vacuous
  target leaves).
- **O4, a metric change rather than an exclusion:** Z and necessity on the post-Top-K read with baseline 0 (x /
  a_pos), as §12.1 does for the pre-activation read, and R2 on A / C.

| set | R0 (v1) | R1s | R1 | **R2** | all-in (vacuous = fail) | O4 |
|---|---|---|---|---|---|---|
| v1 15k strongest | 7,426 / 14,994 = 49.5% | 7,426 / 14,813 = 50.1% | 7,426 / 14,512 = 51.2% | **7,426 / 14,453 = 51.4%** | 7,426 / 15,004 = 49.5% | 7,504 / 14,643 = 51.2% (+137 / −59 passes) |
| v1 15k mid-band | 2,002 / 14,316 = 14.0% | 14.3% | 2,002 / 12,216 = 16.4% | **2,002 / 12,092 = 16.6%** | 14.0% | 16.7% |
| v1 dedup rescore | 804 / 1,799 = 44.7% | 45.0% | 46.3% | **804 / 1,728 = 46.5%** | 44.7% | 46.4% |
| 095 v2CR (production arm) | 48 / 72 = 66.7% | 48 / 71 | 48 / 69 | **48 / 69 = 69.6%** | 66.7% | 50 / 70 |
| 102 dev base | 48 / 72 = 66.7% | 48 / 71 | 48 / 69 | **48 / 69 = 69.6%** | 66.7% | 50 / 70 |
| 102 dev r1.0_rk1e-1 | 54 / 72 = 75.0% | 54 / 71 | 54 / 69 | **54 / 69 = 78.3%** | 75.0% | 56 / 70 |
| 102 dev rk1e-1 | 57 / 72 = 79.2% | | | **57 / 69 = 82.6%** | | 59 / 70 |
| 102 dev rk3e-2 | 56 / 72 = 77.8% | | | **56 / 69 = 81.2%** | | 58 / 70 |
| 102 check base | 48 / 72 = 66.7% | 48 / 69 | 48 / 69 | **48 / 69 = 69.6%** | 66.7% | 51 / 72 |
| 102 check r1.0_rk1e-1 | 54 / 72 = 75.0% | 54 / 69 | 54 / 69 | **54 / 69 = 78.3%** | 75.0% | 57 / 72 |
| 102 check rk1e-1 | 55 / 72 = 76.4% | | | **55 / 69 = 79.7%** | | 58 / 72 |

v1 15k strongest by band × kind, R0 → R2 (number excluded by R2 in brackets):

| band | resid | mlp | attn |
|---|---|---|---|
| L0-4 | 85.4% → 88.4% (79) | 78.8% → 80.1% (46) | 35.4% → 37.2% (60) |
| L5-7 | 72.0% → 73.9% (34) | 60.3% → 60.5% (4) | 7.9% → 8.2% (42) |
| L8-11 | 46.4% → 47.1% (28) | 21.7% → 22.0% (27) | 7.1% → 8.2% (**231 of 1,657, 14%**) |

Points that bear on the choice:
- **R2 vs R1 differ by 59 targets on the 15k strongest (124 on mid-band).** All of them are A denominators that are
  positive but below 5% of a_pos, and none of them passes. Under R1 they stay as forced fails: the [0.75, 1.5] band
  is then a window narrower than 3.75% of a_pos on x, which is the noise amplification the existing 5% Z rule
  exists for.
- **The excluded targets are not bad circuits.** 215 of the 541 that R2 adds on the 15k strongest (40%) pass every
  term that IS defined, and 5 of the 6 in exp 102 do. If they were scored on their defined terms only, the 15k
  rate would be (7,426 + 215) / 15,004 = 50.9%. Exclusion gives 51.4%: it flatters the headline by about 0.5 points
  relative to that reading and by 1.9 points relative to counting them as fails.
- **The 102 decision does not change.** In each panel the same 3 targets drop out of every arm, and they fail in
  every arm. So every arm's pass and resample counts, the paired differences, and the selection-rule winner
  (r1.0_rk1e-1 on both panels) are unchanged; only the rates move (+2.9 to +3.4 pp).
- **The pre-activation read.** On the 15k, 365 targets have a pre-read A / C baseline above 0.95·a_pos_pre, and 361
  of them are already in the post-Top-K R2 set. One exclusion set taken from the primary read keeps the 2×2
  (both / pre only / neither) on the same targets.
- **O4** removes the Z / necessity cases by scoring them, not by excluding them, and it aligns the two reads. But it
  changes Z and necessity for every target with e0_tk > 0 (714 on the 15k strongest), so 137 targets gain a pass
  and 59 lose one. It is a metric change decided after seeing data, and it leaves the A cases (60%) to R2 anyway.
  Kept as a sensitivity only.

### Integrity cost of each

| option | cost |
|---|---|
| Keep as failures (R0) | Conservative, but it scores a test that cannot be computed as a circuit failure: the target fails before it is fitted. It is non-monotone in e0, and it penalises deep attention most (12% of L8-11 attn). It is consistent with v1 as stated in advance (§6). |
| Extend vacuity (R2) | Pass rates rise (+1.9 pp on the 15k strongest, +2.6 pp mid-band), and the scored population shrinks non-uniformly (14% of L8-11 attn strongest; 50% of L8-11 attn mid-band). This is acceptable only because the rule is (a) a function of the target and its contexts, verified circuit-independent; (b) one threshold, the existing 5%, applied to every denominator the rule divides by; (c) stated before the v2 production fit; (d) reported with its counts and the all-in lower bound. |
| Own stratum | It is the most transparent, but a stratum's "pass rate" cannot be computed under the full rule (one term is undefined by construction). In practice it is exclusion plus a descriptive line. |
| O4 | It changes a stated metric for ~5% of targets, in both directions, after seeing the data. Its gains are real passes, not a smaller denominator, which looks like tuning. |

## 5. Recommendation

**Adopt R2 for v2, reported as a counted stratum.** A target is vacuous when any baseline of the pass rule reaches 95%
of a_pos (signed). Vacuous targets are excluded from every pass-rate denominator. Their count is reported per band ×
kind and per term, with their pass on the defined terms as a descriptive line. The all-targets rate (vacuous = fail)
is reported alongside as a lower bound.
- It is the minimal principled fix: v1's own 5% rule, made signed and applied to the A and C denominators, which use
  the same ratio form.
- It cannot be gamed by the fit: it is defined on empty-circuit reads and verified circuit-independent on every
  stored row.
- It does not change the exp 102 arm decision.
- It must be written into §12 before the pod starts. It also changes `paper/main.tex` l. 345 ("an undefined A or C
  ratio counts as a fail") and the protocol appendix.

## 6. Draft §12 text (for `paper/protocol-v1.md`, new §12.4; not applied)

> ### 12.4 Vacuous denominators: every pass-rule baseline, signed (Daniel, 2026-11-__; before any production fit)
>
> **Problem it fixes.** The v1 vacuous flag (§5.2, §6) reads only the Z / necessity denominator and is two-sided
> (|a_pos − e0| < 0.05·a_pos), while each pass-rule term is a ratio against its own empty-circuit baseline and is
> undefined whenever that baseline reaches a_pos (e ≥ a_pos). Two cases escape the flag and fail in every arm,
> whatever the circuit:
> - e0 above 1.05·a_pos: zeroing every upstream latent drives a residual or early-MLP target far above its natural
>   value (median 5.6×);
> - eM_topk or eN_topk ≥ a_pos: the sparsity-preserving mean fill alone drives the target, mostly deep attention,
>   above its held-out mean (median 1.17×).
>
> In the v1 15k this is 482 of 15,004 held-out-strongest targets (3.2%; 12% of L8-11 attention), and 3 of 72 on each
> exp 102 panel (exp 105).
>
> **v2 rule.** On the post-Top-K read, a target is **vacuous** when a_pos ≤ 0 or any pass-rule baseline exceeds 0.95·a_pos:
> a_pos − e < 0.05·a_pos for e = e0 (Z and necessity), eM_topk (A) or eN_topk (C).
> - **Population.** Vacuous targets are dropped from the denominator of every pass rate (post-Top-K,
>   pre-activation, and the resample pass) and counted. The exclusion set comes from the post-Top-K read and is
>   shared by both reads.
> - **Scope.** The rule applies identically to held-out strongest and mid-band.
> - **Why it is a property of the target.** The baselines are empty-circuit reads on the target's held-out, A-mean
>   and C-mean contexts. They do not depend on the circuit and can be computed before any fit (exp 105: the stored
>   undefined terms are exactly the rows whose baselines fail e < a_pos; baselines are identical across arms).
> - **Relation to v1.** It contains the v1 flag, and replaces the v1 statement that an undefined A or C ratio counts
>   as a fail. A term still missing on a non-vacuous target fails.
> - **Reported with it:**
>   - the vacuous count per band × kind and per term (Z/necessity, A, C);
>   - the vacuous targets' pass on their defined terms (descriptive);
>   - the all-targets pass rate counting vacuous targets as fails (a lower bound).
> - **Seen beforehand (sensitivity only):**
>   - v1 15k held-out strongest: 49.5% of 14,994 → 51.4% of 14,453 (all-targets 49.5%);
>   - mid-band: 14.0% → 16.6%;
>   - exp 102: 3 targets per panel excluded in every arm; arm counts, paired differences and the selection-rule
>     winner unchanged.
>   - Code: `src/eval/wcm_eval.py` (`vacuous_v2`, `is_vacuous`), with `062/merge.py`, `083/cmp_common.py`,
>     `paper/tables/common.py` and explorer `stage_circuits.py` reading it.

## 7. Patch proposal (not applied)

`src/eval/wcm_eval.py`:

```python
VACUOUS_BASELINES = ("e0", "eM_topk", "eN_topk")      # Z and necessity, A, C


def vacuous_v2(a_pos: float, baselines: Iterable[float], rel: float = VACUOUS_REL) -> bool:
    """Protocol §12.4: vacuous when a_pos <= 0 or any pass-rule baseline e > (1 - rel) a_pos (signed). A function of
    the empty-circuit reads only, so of the target and its contexts, never of the circuit."""
    return bool(a_pos <= 0 or any(a_pos - e < rel * a_pos for e in baselines))

# score_circuit, after the v1 flag (keep vacuous_<read> unchanged for the v1 record):
row["vacuous_v2"] = vacuous_v2(a_pos["tk"], [E[k]["tk"] for k in VACUOUS_BASELINES])


def is_vacuous(row, read="tk", rule="v2"):
    """rule="v2" (§12.4): from the stored post-Top-K baselines, for every read (one exclusion set for both reads), so
    stored v1 rows are re-classified without rescoring; rule="v1": the stored vacuous_<read> flag."""
    if rule == "v1":
        return bool(row.get("vacuous_" + read) or False)
    if row.get("vacuous_v2") is not None:
        return bool(row["vacuous_v2"])
    ap = _num(row.get("a_pos_tk"))
    es = [_num(row.get(k + "_tk")) for k in VACUOUS_BASELINES]
    if ap is None or any(e is None for e in es):
        return bool(row.get("vacuous_tk") or False)
    return vacuous_v2(ap, es)
```

- **Callers that read `vacuous_tk` directly** move to `is_vacuous(row)`: `062/merge.py:vacuous()` (used by
  `paper/tables/*`), `083/cmp_common.py`, `src/export/explorer/stage_circuits.py`, `102/report.py`.
- **Tests:** `tests/eval/test_wcm_eval.py` gains cases for e0 > 1.05·a_pos, eM_topk ≥ a_pos, a positive A
  denominator below 5%, and a_pos = 0.
- **v1 record:** reproducing it means passing `rule="v1"`.

## Files

- `undefined_reads.py`: the analysis (sources: `062/out_full/out/main/eval.*.jsonl`,
  `062/out_rescore_dedup/eval_dedup.jsonl`, `095/out/rows.jsonl`, `102/out/{phase1,check}/rows.jsonl`,
  `062/out_full/out/targets.csv` for the near-threshold share).
- `out/summary.json`, `out/flagged.csv`, `out/run.log`.
