# 086 Held-out rebuild (DAN-157)

New held-out sets for the 15k run's **fixed** circuits, under the rule Daniel approved (final form: **no source-file rule**, the **contrast held-out set deduplicated too**, below-floor targets scored and
flagged). The rebuild follows from the 085 audit: 66.8% of held-out strongest rows repeat a training-side row. There
are no refits, and no training-side field changes. The rule itself lives in `src/circuit/heldout_rule.py`, shared
with the group-aware split for the public models (below).

## Files

| file | what |
|---|---|
| `scan.py` | top-N corpus scan (N = 512) for every 062 target. MODE=scan (sharded over corpus shards, resumable), merge (the exact top N over the shard files), check (scan vs the top store) |
| `rebuild.py` | MODE=build writes `062/out_full/out/ctx_v2/` (the only variant) plus `out/rebuild/build.shard*.jsonl`. MODE=verify is an independent re-check of every record (strongest, mid-band and contrast sets). MODE=report writes `out/rebuild/report.md` |
| `rescore.py` | re-score queue (no refits). Jobs: main (strong + mid + 056 specificity), sweep, alpha1 (default); random (DAN-17) kept but not default. Reads through `driver.Runner` with `CTX_DIR`; writes `out/rescore_<ctx dir>/<job>/`. REPORT=1: headline / worst case / paired leakage effect |
| `subset.py` | the frozen stratified 1,000-target subset for the as-run re-score (`targets_asrun1000.txt` + `.manifest.json`) |
| `passrates.py` | the report's pass-rate bookkeeping (headline, worst case, paired effect); unit-tested |
| `public_smoke.py` | the group-aware split on the local 069 smoke stores (`outputs_public/`, `data_public/`) |
| `targets_smoke40.txt` | 40 smoke targets: 4 at each of L2/6/10 × attn/mlp/resid, plus 4 at L5 resid including `5.resid.37507` |
| `targets_asrun1000.txt` | the frozen as-run subset: seed 20261024, 111 per band × kind cell (112 in L0-4 attn), drawn from the 15,004 targets with a main circuit; md5 in the manifest |

## The rule as implemented

**Strongest** means the top store's ranking statistic: the mean of the target's post-Top-K activation over the 64
positions (`store/context.compute_seq_scores`). That is how the 64-row pool was chosen. The 3:1 split itself ranks by
peak (unchanged).

Per target, for the held-out **strongest** set:

1. **Forbidden set F.** strongest-train (48, including the fit's 32), mid-band train (48), and all 64 contrast rows.
   Their corpus twins are also forbidden: same id, or same content hash (085's hash).
2. **Candidates, in order.**
   - First the original held-out rows, in pool order (= store rank), each at its stored anchor.
   - Then the scan's rows, strongest first. Each anchor is the recomputed argmax.
   - Mid-band pool rows never enter this set, matched by id or by content.
3. **A candidate is accepted iff all of these hold.**
   - It fires: the post-Top-K activation at its anchor is > 0. This is recomputed with the protocol's own forward
     (`protocol_contexts.target_activations`, batch 16).
   - It is not exact, anchor-prefix or near (LCR ≥ 32, or ≥ 50% of its 8-gram positions) against F.
   - It is not exact, anchor-prefix or near against an already-accepted held-out row.
   - There is **no source-file clause** (dropped; `corpus_src.npy` is no longer read).
4. **Fill and floor.** Fill to 16; the floor is 8. Targets below the floor are flagged `below_floor`, **scored and
   flagged, never dropped**, and reported per band × kind. The headline counts targets with ≥ 8 clean rows; the worst
   case counts below-floor targets as fails.
5. **Storage.** Accepted scan rows are appended to `strong.pos/tgt/arg/peak/ids`. `strong.held` = the kept original
   indices (original order), then the appended ones.
6. **Provenance.** `heldout_v2` holds, per row: origin, corpus id, activation at the anchor, peak, mean, scan rank
   and acceptance order. It also holds the rule version and the rejection reasons.

The **mid-band** held-out set (reported) goes through the same rule on its original rows. Rows are only removed.

**Firing is checked twice.** A row near the Top-K cut can fire in one batch composition and not in another. So
besides the selection forward, the final set is forwarded as the scorer sees it: one batch, record order (the last
eval batch after the 48 reference rows). A row that is silent there becomes `unstable_fire`, and the set refilled.

### The contrast held-out set

The scorers split the 64 contrast rows 3:1: `rec["neg"][:48]` train (the C fill's means), `rec["neg"][48:]` held out
(sufficiency to induce, phi_cf), as `split_n(64)` in `eval/wcm_eval.score_circuit` and 049 `amp_eval_pass_v2`. The
rebuild applies the same rule to the held-out 16 with the sides swapped (`select_heldout(..., want="silent")`):

1. **Forbidden.** The 48 contrast-train rows and **every activating row** of the rebuilt record (strongest pool incl.
   the appended scan rows, mid-band pool), by tokens, id and content hash.
2. **Candidates.** First the original contrast held-out rows (record order); then, only when those leave the set
   short, the **close selector's next candidates** (`utils/neg_context_selector`, the run's selector arguments,
   `max_sequences = 64 + NEG_EXTRA` (128); its first 64 are the run's contrast rows, recorded as
   `selector_prefix_ok`), in similarity order, minus the record's own contrast ids / contents.
3. **Accepted iff** silent (the target's post-Top-K activation is 0 at **every** position, recomputed with the
   protocol forward), and neither a **whole-window exact copy** (tokens, id or content hash) nor a **near duplicate**
   (a shared contiguous run of ≥ 32 tokens, or ≥ 50% of its 8-grams shared) of a forbidden row or of an
   already-accepted contrast row. **There is no anchor-prefix rule for contrast rows** (Daniel): they
   never fire, so they have no anchor; `select_heldout(want="silent")` skips the apfx test, and the verifier's apfx
   at the last position is the exact test. A shared prefix shorter than 32 tokens is therefore accepted.
   The final set is forwarded once more as one batch; a row that fires there is excluded (`unstable_silence`).
4. **Fill.** Up to 16; fewer are kept when candidates run out (`contrast_below_floor` = fewer than 8, reported). A
   record with zero contrast held-out rows gets a `no_contrast_heldout` skip row in the re-score.
5. **Storage (the design).** `rec["neg"]` is **not changed**. The rebuilt rows go to `rec["neg_held_v2"]` [h, 64] and
   `rec["neg_held_v2_ids"]`. Only the scorers see them: `protocol_contexts.eval_probe` (and `eval_contexts`, the 049
   injection) use `rec["neg"][:48] + neg_held_v2` with `metadata["n_neg_train"] = 48` when the field is present
   (`eval_contrast`); every scorer takes `n_neg_train` as its contrast-train count (`wcm_eval.contrast_split`, 049,
   056, `wcm_specificity`, 073 / 075 / 076 / 079). A record without the field scores exactly as before (tested).

**Why this design.** The fit consumes all 64 of `rec["neg"]` through `training_probe`: the engine trains on the first
48 and uses the last 16 only for its post-fit `holdout_data_loss` statistic. 083 REFIT and 084 harder nulls fit on
these rebuilt records, so the fit-side contrast must equal the 15k run's byte for byte; leaving `rec["neg"]` untouched
guarantees that (the verifier checks `neg` / `neg_ids` unchanged). Storing a shorter held-out list inside `rec["neg"]`
would also change `split_n(n)` and so the contrast-train slice; the explicit `n_neg_train` keeps the scorer's
contrast-train slice exactly the run's 48 rows whatever the held-out size. (The fit's padded positive filler rows do
follow the rebuilt `strong.held`; like the contrast holdout they reach only `holdout_data_loss`, not the circuit.)

## Smoke results (local, before the final decisions: two variants, no contrast dedupe)

The smoke uses `targets_smoke40.txt` and a scan of one quarter of the corpus (`out/scan_smoke40`; SHARD 0/4, 758
shards). Outputs are in `out/smoke40/` and `out/smoke_repro/`.

**Scan check**
- Every top-store row in the scanned shards is in the scan's list (recall 1.0, at rank ≤ 17).
- The scan's mean matches the store's bf16 value: median relative difference 0.3%, p99 7.5%.

**Rebuild** (then "primary", with the since-dropped source-file clause)
- 38 of 40 targets reach 16 rows. 2 fall below the floor: thin L2 attention latents with 0 and 2 rows, whose scan
  lists ran out.
- Median rows kept from the original held-out set: 2 of 16. Scan rows make up 80.7% of held rows.
- Strength relative to strongest-train: 0.956× by peak, 0.858× by mean score.
- Without the source-file clause (the rule now): the same 38 / 40; median originals kept 5; 0.970× peak.

**Verify:** 80 / 80 records pass.

**Re-score** (main + spec + alpha1 on the 40): 7 s per target, no errors.

**Reproduction on the run's own records** (`CTX_DIR=out_full/out/ctx`, 2 targets, every job)
- The contexts are identical: `a_pos` and `e0` match exactly.
- alpha = 1 and the random null reproduce exactly. Both were scored on this GPU originally.
- main / sweep / spec differ by H100-vs-RTX float noise, up to 0.06 on main Z / A / C and 0.47 on a failing sweep
  arm's A ratio.
- **So the paired comparison must re-score `ctx` (as run) on the same hardware as `ctx_v2`:** the frozen 1,000-target
  subset (`targets_asrun1000.txt`), not the full run.

The final-rule smoke (2 TuringLLM targets, every step) is in `out/smoke_final/`; see the section at the end.

**Cost**

| step | local (RTX 5070 Ti) | 8×H100 |
|---|---|---|
| scan | measured 248 windows/s with all 35 sites → 18.24M unique windows ≈ 20.4 h | 21–50 min, at 1,200–1,800 windows/s/GPU (FLOP scaling; read the real rate from the first scan log lines) |
| rebuild (strongest + mid + contrast) | ≈ 2 s/target plus the contrast refill's selector call when the original 16 do not survive (see the final smoke) | ≈ 20–30 min |
| verify | ≈ 4.5 s/target (≈ 18 h, CPU-bound) | ≈ 45 min with 3 processes per GPU |
| re-score ctx_v2 (main + spec ≈ 29 h, + sweep, alpha1 ≈ 3 h) | ≈ 32 h | ≈ 30 min (the run's eval + spec time was 17 process-h; MPS 6 per GPU) |
| re-score as run, 1,000-target subset (main + spec) | ≈ 2 h | ≈ 2–3 min |

The rebuild's contrast refill needs the TuringLLM stores the close selector ranks with: `outputs/top_ctx.pt`,
`mid_ctx.pt`, `neg_ctx.pt`, `seq_repr.pt` (`STORES_DIR`, default `outputs/`; ≈ 2.6 GB RAM per process).

## Re-score scope and the report

- `rescore.py` on `ctx_v2`: JOBS=main,sweep,alpha1 (default).
- As-run subset: JOBS=main, `CTX_DIR=062/out_full/out/ctx`, `TARGETS=targets_asrun1000.txt`, its own `OUT_RS`, same
  machine as the ctx_v2 re-score.
- REPORT=1 with `OUT_RS` = the ctx_v2 output and `ASRUN_RS` = the as-run subset output: the ctx_v2 headline (targets
  with ≥ 8 clean held-out strongest rows) and worst case (below floor = fail), overall and per band × kind, for main
  (strong, mid), every sweep arm and alpha = 1; the paired leakage effect on the subset (as-run vs ctx_v2 pass rates,
  per band × kind, headline pairs and worst case); reproduction checks against the run's own rows. Written to
  `OUT_RS/report.md`.

## Consumers of held-out data and CTX_DIR

These read the context records through `driver.Runner.contexts`, so `CTX_DIR` applies to all of them. Setting it
forces `EVAL_SPLIT=explicit`, and a missing record is an error (records are never built in a rebuilt directory). Those
that score go through `eval_probe` / `eval_contexts` and so pick up the rebuilt contrast held-out set too.

- `062/driver.py` (main / sweep)
- `rescore_alpha1`, `random_null`, `stable_core`, `rescore_mid`, `rescore_dedup`, `attn_heads`, `freeze_attn`,
  `member_restoration`
- 073 `strict_eval` and `diag`
- 083 `rescore_ref`, `attr_fitted`, `attr_baselines` and `ge_hier`
- 084 `harder_nulls`
- 071 `sweep` and `thirds_rescore`
- 075, 076, 079, 080 `run` / `calibrate`
- 063 `validate` and `case_study`
- 086 `rescore`

056 specificity (via `Runner.spec`) now follows the scorer's `EVAL_SPLIT` and the rebuilt contrast's `n_neg_train`.
`rescore_dedup`, `member_restoration`, 063 `case_study` and 079 `fit_score` read `rec["neg"]` directly: they get the
run's 64 rows (their contrast-train slice is unchanged; any held-out contrast use there is the run's).

These read `out_full/out/ctx` directly and do **not** follow `CTX_DIR`. None of them produces a held-out score in the
paper's tables:

- 063 `survey` / `inspect_circuit` (case-study display)
- 067 `check_equivalence`, 068 `golden_turingllm` (equivalence gates, which must read the as-run records)
- 080 `next_token_effect` (training contexts only)
- 062 `dup_leakage`, 085 `audit` / `peek`

## Public models: group-aware split

`build_record(..., group_split=GroupSplit(...))` is opt-in. The default TuringLLM path is unchanged.
`driver.Runner.contexts` uses it for public families (`GROUP_SPLIT=1`, the default). **Duplicates only, the same
rule as TuringLLM** (Daniel): same-document grouping and the contrast document skip were dropped.
Document ids (`DATA/provenance/shard_<N>.docs.npy`) are still recorded per row (`doc`) as provenance, nothing else.

1. **Joint groups.** Rows of the strongest and mid-band pools that are exact, anchor-prefix (under either row's
   anchor) or near duplicates form one group (connected components).
2. **Split.** The strongest pool is split stratified by whole groups. In the mid-band pool, groups that train in the
   strongest pool must train, and groups held out there may not train: they are held out while they fit, else
   unused.
3. **Contrast.** The run's close selector, unchanged (no document skip).
4. **Final guard.** The rule (exact / anchor-prefix / near vs every training-side row, contrast rows included; must
   fire) runs over both held-out sets. Dropped rows go to `unused`, and `rec["heldout"]` reports the counts.
