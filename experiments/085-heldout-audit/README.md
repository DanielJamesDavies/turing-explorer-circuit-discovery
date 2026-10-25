# 085 Held-out audit: are the 15k run's held-out contexts really held out?

Audit of every stored context record of the 15k protocol-v1 run (`062/out_full/out/ctx/<key>.pt`, 15,046 files;
the sweep, random nulls, α=1, specificity, 073, 084 and the case studies all read these same records). Read-only on
all existing data and code; everything here writes only to `out/`. Run 2026-10-23.

**Short answer: no.** Exact duplicates are the smallest part of it. For the scored read, half of all held-out
strongest rows are the same input as a training-side row. Under a robust rule only a median of **5 of 16** rows per
target are unseen, and **79.5% of targets drop below 8**. The stores hold nothing beyond the 64 strongest, so
topping up needs a new corpus scan. The mid-band held-out set is nearly clean.

## Files

| file | what |
|---|---|
| `corpus_index.py` | one pass over `data/shard_*.npy`. Writes `out/corpus_{src,hash,mult}.npy`: per sequence id, the source-file index, the 069 content hash, and the duplicate multiplicity (~500 MB; regenerable in 90 s) |
| `audit.py` | per-record checks (rules below). Writes `out/audit_all.csv` (one row per target, ~200 columns), `out/examples_all.jsonl` (every flagged held-out row with the row it matches), and `audit_subset.csv` (the 350-target stratified subset: 10 per layer × kind, seed 0) |
| `store_values.py` | what `outputs/top_ctx.pt` and `mid_ctx.pt` hold per target, plus split mechanics for twin rows. Writes `out/store_values.csv` and `out/twin_ranks_subset.csv` |
| `silence_check.py` | GPU, subset only. Re-forwards the stored rows to check contrast silence, how close the target came to Top-K, and that the stored peaks and anchors are reproducible. Writes `out/silence_subset.csv` |
| `report.py` | all tables, as `out/report_all.md` and `out/report_subset.md` |
| `examples.py`, `peek.py` | decoded worst cases (`out/worst_examples.md`) and an ad-hoc decoder |

## 1. The rules as the code implements them

The records come from `src/circuit/protocol_contexts.build_record`, which is identical to 059 `pool_test.build`.
They are scored through `eval_probe` with the 049 scorer.

- **Strongest pool.** The `top_ctx` row for the target: the 64 sequence ids with the highest **mean** activation over
  the 64-token window (`compute_seq_scores`). Ids ≤ 0 are dropped and ids are deduplicated **by id only**, then capped
  at 64.
- **Mid pool.** The `mid_ctx` reservoir row (mean score in μ+[0.5,1.5]σ), minus top-store ids, capped at 64.
- **Thin pools.** A pool with fewer than 8 ids is absent. A target with no strongest pool is skipped (23). A target
  with no mid pool is thin and uses arm A.
- **Peak and anchor.** One fresh forward pass per pool (batch 16). Peak = max of the post-Top-K activation over the
  64 positions; anchor = its argmax. A row where the target never fires has peak 0 and its anchor defaults to position 0.
- **Split.** `stratified_split(peak, 0.25, keep_top=2)`:
  - Stable descending argsort by peak, so tied rows keep their store order.
  - Block b of 4 ranks holds out rank `4b + (b+2) mod 4` (ranks 2, 7, 8, 13, 18, …; 0-based).
  - Ranks 0–1 are never held out. The held-out set is trimmed or topped up to `round(n/4)` from the weak end.
  - With 64 rows this gives 48 / 16.
- **Fit sets.**
  - D = `stratified_split(peak[S_tr], 1/3, keep_top=2)` gives 32 strongest; U = the 16 strongest-train rows not in D.
  - Bm = `stratified_subset(mid-train peaks, 16)`.
  - FIT = D + Bm (arm B), or all of S_tr (arm A).
- **Contrast.** The close selector returns up to 64 verified-silent rows in similarity order. `stratified_order` puts
  48 train first and 16 held last; consumers slice `[:n − round(n/4)]`.
- **Evaluation.** The probe is S_tr (48: the A-mean and pin reference) followed by the held-out set. It is padded to
  `padded_length(48) = 64`, so `probe_from` takes **at most 16** held rows and drops any beyond 16 without warning.
  The 049 scorer and src `wcm_eval` also cap at `N_SEQ = 64`. The C mean comes from contrast-train and induce runs on
  contrast-held.

**Where the code and docs disagree, or the docs are wrong.**

1. *Paper, line 745*, says the pools are "deduplicated". They are deduplicated by corpus id only. 23% of strongest-pool
   rows are token duplicates of another row in the same pool (unique contents per 64: median 49, p10 38).
2. *Protocol §3.7* calls the duplicates "corpus boilerplate". The corpus index shows something else:
   - **52.9% of all 24.8M windows have exactly one exact twin** (multiplicity 2: 13.1M windows; >2: 17.6k).
   - **99.99% of twin pairs share their source-file index**, and the twins sit in unrelated shards.
   - So this is construction-level duplication: about 6.6M windows were written twice. That is the 26.5% figure.
   - On top of that there are 1-token-shifted copies of the same text (LCR 63; e.g. `3.resid.16802`).
   - The source files are synthetic multi-document files that contain **paraphrases of the same passage**. Token-level
     dedup cannot catch those (see `peek.py src 79443`).
3. *The split guarantees the leak.* Twins have identical peaks (100%) and the stable sort puts them at **adjacent
   ranks** (96.2%). Each block of 4 holds out one rank, so a twin pair almost always straddles train and held-out.
4. *`rescore_dedup.py` (the 41.2% → 40.6% paired test) was incomplete.* It removed only rows token-identical to FIT
   rows or to earlier held rows. It kept:
   - rows identical to the 16 unused reference rows U (35.6% − 25.1% = 10.5% of rows);
   - anchor-prefix duplicates;
   - near-duplicates;
   - contrast matches.

   Its "clean" sets therefore still leak, and the protocol §3.7 conclusion should not be cited as settled.
5. Minor:
   - The paper says 472 targets have fewer than 16 held-out strongest rows; the records give **471**. The other counts
     in protocol §3.1/§3.3 reproduce: 14,277 full 64/64 pools, and 478 short pools, all of them thin.
   - The protocol says "64 strongest"; it means highest mean score, while the split ranks by peak.
6. *Specificity* (`056/specificity.py:173`) always slices with `split_n` (as_run). If the held-out set is shorter
   than 16, its "held-out" slice silently takes in reference rows. This is the §3.6 defect again.

## 2. What is wrong (all 15,004 usable records; per-cell tables in `out/report_all.md`)

The criteria are applied to each held-out row against every training-side row: S_tr, M_tr and C_tr (this includes
FIT and the reference) plus all contrast rows.

- **exact**: the 64-token window is identical.
- **anchor-prefix**: tokens[0..anchor] are identical. Every read is at the anchor and the model is causal, so for the
  scored quantity this is the same input.
- **near**: longest common run ≥ 32 at any alignment, or ≥ 50% of the row's 8-grams appear in the other row.
- **same source**: the two rows share a source-file index.

**Held-out strongest (primary)**

| issue | share of rows | targets with ≥ 1 row |
|---|---|---|
| exact dup of a FIT row | 25.1% | 92.2% |
| exact dup of a training-side row | 35.6% | 94.9% |
| anchor-prefix identical | 50.2% | 98.6% |
| near-dup | 55.1% | 99.5% |
| fails R2 (exact, anchor-prefix or near) | 66.8% | 99.8% |
| same source file | 73.0% | 99.8% |
| exact dup inside the held-out set | 5.6% | 58.5% |
| peak ≤ 0 (silent row, anchor forced to 0) | 0.08% (180 rows) | 321 targets, all attn (762 rows across their pools) |

Rows left of 16 under each rule:

| rule | median | < 8 | < 4 | 0 |
|---|---|---|---|---|
| as run | 16 | 2.3% | 1.0% | 0 |
| R0 (exact) | 9 | 33.2% | 6.7% | 0.5% |
| R1 (+ anchor-prefix) | 7 | 59.2% | 18.7% | 3.7% |
| **R2 (+ near-dup)** | **5** | **79.5%** | **29.7%** | **4.6%** |
| R3 (+ same source) | 3 | 93.5% | 60.8% | 12.4% |

- Under R2, fewer than 8 rows survive in 86.8% (L0-4), 90.2% (L5-7) and 93.6% (L8-11) of attention targets, and in
  71–81% of mlp and resid targets.
- Requiring the same anchor for anchor-prefix matches changes nothing (R2sa ≈ R2).
- Pass rate as run, by R2-clean count: 37.8% at 0 clean rows, 37.3% at 1–3, 47.0% at 4–7, 57.6% at 8–11, 59.5% at
  12–16. Leakier targets pass *less* often. That is a comparison between different latents; only the paired re-score
  can settle the effect.

**Other findings**

- **Strength.** Held-out rows match strongest-train in peak (median ratio 1.000), and R2-clean held rows match all held
  rows (0.996). Holding ranks 1–2 in training does not make the held-out set weaker.
- **Anchors.**
  - 12.3% of held-out strongest rows have their anchor at position ≤ 3, where the read sees at most 4 tokens; 2.0% are
    at position 0.
  - 0.8% of targets are position-0 latents: every anchor is at 0. In `0.mlp.1030`, all 64 windows start with the token
    "ol", so every held-out read equals a training read.
- **Mid-band held-out (reported).** Nearly clean: rows exact 0.2%, anchor-prefix 3.2%, near 0.4%; R2-clean median
  16. 19.1% of targets lose at least one row; 2.0% drop below 8, mostly attention.
- **Contrast.**
  - Held-out vs contrast-train: exact in 2.3% of targets, near in 7.5%.
  - One activating/contrast token contradiction: `10.attn.31771`, a row that is silent on re-forward.
  - The forward check (subset: 22,336 rows) confirms every contrast row is silent: 0 rows fire.
  - But they are only just silent: the target's best pre-activation rank reaches 121–129 (K = 128).
  - Silence cannot be verified from the stored data alone, because the records hold no activations.
- **Integrity is fine.**
  - Every stored id hashes to its stored tokens (0 mismatches).
  - No padding, out-of-vocab tokens or <|endoftext|>; no NaN or inf; tgt is pos shifted by one throughout.
  - Peaks reproduce exactly on re-forward, apart from one target (`7.attn.35706`: 4 anchors moved).

**Worst concrete cases** (`out/worst_examples.md`, decoded)

- `9.resid.13846`, `11.attn.30684`, `10.attn.9144`, `5.attn.31575`: all 16 held-out rows are exact copies of
  training-side rows (10–12 of them are FIT rows).
- `5.resid.37507`: all 16 held-out rows are exact copies of FIT rows. The text is a repeated "The text is a matter of
  semantics" window.
- `2.mlp.2039`, `0.mlp.16182`, `2.resid.2139`: 0 exact duplicates, yet all 16 are anchor-prefix identical. An
  exact-only dedup would call these clean.
- `3.resid.16802`, `1.resid.22765`, `4.resid.17195`: 16/16 are 1-token-shifted copies of training rows.
- `3.attn.13573`: 10 strongest-pool rows are silent on re-forward and pinned to anchor 0.

## 3. Can the held-out set be topped up?

**Not from what is stored.**

- `top_ctx` has exactly 64 slots, and the run used all of them.
- `mid_ctx` is a 64-slot reservoir far below the strongest. Its median mean score is 0.11× the top rank-1 context
  and 0.16× rank 64, and its peaks are about 0.45× the strongest peaks. It cannot stand in for strongest contexts.
- `top_coactivation` stores partner latents, `logit_ctx` stores output tokens and `neg_ctx` stores silent
  neighbours; none of them holds activating contexts.
- **In-record only.** If the A-mean and pin reference moved to the fit's own 32 strongest rows, the 16 unused rows U
  would become candidates. R2-clean candidates against FIT + contrast: median 12; ≥ 16 for only 26% of targets; < 8
  for 15.9%. That is not enough, and it changes the reference.
- **Mid-band.** M_tr \ Bm (32 rows the fit never saw, outside the reference) are valid top-ups: median 48 clean
  candidates.

**A new top-N scan (e.g. N = 512) is required**, on the corpus deduplicated by content hash.

- Ranks 65+ are weaker than the stored 64. In the store, the mean score falls from rank 1 to rank 64 to a median
  0.70×. Extrapolating linearly from ranks 48–64 puts rank 128 at about 0.88× (IQR 0.84–0.91) of rank 64. A topped-up
  set would sit at or below the weak end of the original pool, so its strength has to be reported as a covariate.
- **Source-disjoint top-up is unmeasured.** A target's strongest pool spans a median of 29 source files and its fit
  rows 36, out of 95,640. A top-up from files that contribute no training row is probably feasible, but only the scan
  can confirm how strong those contexts are.

## 4. Proposed rebuild rule (re-scoring fixed circuits; no refits)

Per target, built once and stored as a new ctx dir with the same record format: append new rows to `strong.pos/tgt/arg/peak/ids` and
replace `strong.held` (and `mid.held`, `neg` order). All consumers then pick it up unchanged.

1. **Forbidden set F.** Every training-side row: S_tr (48, including the reference), M_tr (48), C_tr, and all 64
   contrast rows. Also every corpus window whose content hash equals one of those rows (this catches the twins
   directly).
2. **Admissible candidate h.** All of the following must hold:
   - peak > 0 (drop silent rows);
   - no exact match, no anchor-prefix match (tokens[0..a_h]) and no near-dup (LCR ≥ 32, or 8-gram containment ≥ 0.5)
     against any row of F;
   - no such match against an already-accepted held row (select greedily, strongest first).
   - Primary: also **source-disjoint**: the candidate's source file contributes no row to FIT or the reference. This
     is the only defence against paraphrase leakage.
   - Sensitivity analysis: the same rule without the source condition.
3. **Fill order.** First the surviving original held rows, then strongest-first from the new scan (or from M_tr \ Bm
   for the mid-band set) until 16. Record each row's provenance (original / top-up) and its rank.
4. **Minimum.** 16 is the target and **8 the floor**. A target below 8 is reported as "no clean held-out set" per
   cell and excluded from the headline. Also give a worst-case rate that counts it as a fail. Never drop it silently.
5. **Contrast held-out.** Apply the same rule against C_tr and the activating rows; replace hits with selector ranks
   65–128, silence-checked (059 `build_neg128` already shows the first 64 are stable).
6. **Scoring.** Use `EVAL_SPLIT=explicit`, keep at most 16 held rows (the 64-row cap), and patch specificity to an
   explicit split before rerunning it.
7. **Public models (069).** `prepare_corpus.py` removes exact duplicate windows only. Anchor-prefix duplicates,
   shifted or near duplicates and same-document windows remain; `provenance/*.docs.npy` would allow a document-
   disjoint split. Those runs have not been fitted yet, so the right fix is a **group-aware split** inside
   `build_record`: cluster pool rows by the rules above plus document id, and assign whole clusters to one side. This
   is better than removing rows after the split.

## 5. Re-score cost (scoring only)

Local RTX 5070 Ti rate, measured: 1,800 held-strong rescores took 4,964 s in `rescore_dedup`, about 2.8 s per target.

| job | rows | local | 8×H100 (MPS 6), estimate |
|---|---|---|---|
| main, held-out strongest | 15,004 | ~11.5 h | ~10–20 min (run total t_eval 14.2 process-h covers both held sets) |
| main, mid-band (only the 2,732 targets whose mid set changes) | 2,732 | ~2.1 h | minutes |
| specificity / rank_clean (056) | 15,004 | ~5 h | t_spec 2.8 process-h |
| sweep arms (10 × 190) | 1,900 | ~1.5 h | 0.9 process-h |
| α = 1 / random null / 073 / 084 / case studies | ~1,750 / 126 / 747 / small | ~2–3 h | minutes |
| contrast rebuild (selector + silence check), if adopted | 15,004 | ~4–8 h | t_ctx 4.4 process-h |
| **top-up corpus scan** (top-N for 15k latents, 18.2M unique windows) | – | ~1 day local (from 2.66 s per 512-batch for the full first pass on the 16 GB card) | a few H100-hours |

**Total:** about 1.5 days on the local GPU including the scan, or under half a day on rented H100s.

The scan is the only expensive step. Its timing is an estimate from a 64-batch profile with every first-pass store
enabled; a top-only pass should be faster.
