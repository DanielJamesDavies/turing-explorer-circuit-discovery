# DAN-219: sources for the four open TODO(DAN-219) flags in paper/main.tex

Checked 2026-11-17 (read-only; no repo files edited). Recompute scripts are in the session scratchpad
(`srchunt/c071.py`, `srchunt/wl.py`), run with `python -I`.

---

## 1. main.tex:2061 (TODO at 2064): "32 targets, paired ... a wash ($50$W/$46$L)"

- **Text source only:** `paper/resources/paper-updates.md:1073-1083` ("At scale (32-seed sweep,
  `analysis-restyled/respect-topk-sweep/`) it does NOT systematically change normalized free-φabl (paired
  m=128: 50W/46L, wash) — it lifts both a_C AND the floor a_empty (0→0.305)"). The sentence has been in
  the paper since the first draft commit `a8e4bd1` with the same `% data:` path.
- **Data: not found.** `analysis-restyled/` exists (gitignored) but holds only `case-study/`,
  `floors-posctx/`, `sae-eval/`. No `*respect*` path in the repo, git history (`git log --all
  --name-only`), `../Runs/` or `../Runs/Backups/`. The sweep would have been
  `src/analysis/circuits/gradient_size_sweep_runner.py --respect-topk` (writes
  `gradient-size-sweep.csv` under a run's `analysis/`); no such output survives in any run root.
- **Recompute attempt:** scanned every `experiments/**/*.jsonl` row carrying both `freeM_dense` and
  `freeM_topk` (`srchunt/wl.py`). No set has 32 targets and 96 paired comparisons, and none gives 50/46.
  (Nearest is exp 006: 32 rows but 4 targets, 14W/18L.) The 32-seed candidate
  `experiments/002-pruned-32seed/` has runners but its result jsonl was never committed.
- **Explaining the count:** "paired m=128" suggests a size sweep at budget m = 128, so 96 = 32 targets x
  3 arms is plausible. That is a guess; nothing on disk confirms it.
- **Verdict: NOT FOUND** (no surviving data; the only source is a notes entry).
- **Suggested TODO replacement / action:** cut the sentence, or keep it labelled pre-v1 with
  `% source: paper/resources/paper-updates.md:1078 (notes only; data analysis-restyled/respect-topk-sweep/
  lost, never committed)`. Note also `paper/resources/v2-number-audit.md:338`: the claim conflicts with the
  v1/v2 evidence of protocol.md §12.1 (rule pass 46.4% sparse vs 48.7% dense; deep attention A 0.93 dense to
  0.00 sparse). Cutting it is the safer choice.

---

## 2. main.tex:2126 and 2129 (TODO at 2131), attributed to experiments/011-binarize-sweep/

| claim | source | value | verdict |
|---|---|---|---|
| "218 per step at layer 10" (STE, node flips) | `011/rows.jsonl`, row `L10-resid` / `mode: ste` / `flips.last50_mean` | 217.9 (last 50 steps); `mean_tail` (2nd half) = 249.6 | **MATCH** (as a last-50-step mean) |
| reach-then-hold "inflates size 21--48%" | `011/anneal_steps.jsonl`, arm `anneal-200-hold` `n` vs arm `anneal-400` `n` | L2 2565/1736 = +47.8%; L8 38101/31557 = +20.7%; L10 49469/40684 = +21.6% | **MATCH** (vs the 400-step reference) |
| hold "leaves churn unchanged" | same file, flips last 25 | hold 8.0/316/310 vs 200-cal 8.3/275/302 | consistent |
| truncate at 200 gives node sets "57--72% oversized" | `011/anneal_truncate.jsonl`, `n` at `stop_at` 200 vs 400 | L2 1991/1736 = **+14.7%**; L8 54368/31557 = +72.3%; L10 63819/40684 = +56.9% | **MISMATCH (partial):** 57–72% holds only for the two deep targets |
| truncate "gate still half-soft" | same, `temperature` | 0.2228 at step 200 | consistent |

Caveat on "per-step node flips grow through training (218 per step ...)": the STE row's second-half mean
(249.6) is higher than its last-50 mean (217.9), so the flip count is not growing at the end of the run.
The 218 is the end-of-run level.

- **Wording fix:** "... yields node sets $15$--$72\%$ oversized ($57$--$72\%$ on the two deep targets) with
  the gate still half-soft". Optionally "($218$ per step over the last 50 steps at layer 10)".
- **Suggested TODO replacement:**
  `% source: experiments/011-binarize-sweep/rows.jsonl (L10-resid ste flips.last50_mean 217.9);
  anneal_steps.jsonl (anneal-200-hold n vs anneal-400 n: +48/+21/+22%); anneal_truncate.jsonl (n stop_at 200
  vs 400: +15/+72/+57%). 3 targets (L2/L8/L10), exploratory, pre-v1.`

---

## 3. main.tex:2150 (TODO at 2148): SFC reimplementation scores $\phiind = \phisup = 0.000$ on three deep targets (L8–10), "even at its most permissive threshold"

- **Source:** `dev-notes/sfc-replication-and-cross-method.md:58-99`, cross-method table, row `sfc-npa`
  (`cf 0.000`, `sup 0.000`; finding 2 says "on all three seeds": L8 25/10628, L9 27/6859, L10 32/3021,
  `node_threshold=1e-6`, in-sample, 2026-07-22). Script:
  `experiments/_archive/002-latent-clustering-search/cross_method_table.py`. Its output `cross_method.jsonl`
  is not in the repo or git history. ($\phiind$ is the former φ_cf, per `paper/math_commands.tex:4`.)
- **The number reproduces the note, but the note is contradicted by a later run on the same three targets.**
  `experiments/_archive/007-free0-cf-32seed/runner.py:17-21` says that SFC nodes carry `role="attributed"`,
  which the cf eval does not recognise, so it "would return a fake 0.0". That runner maps roles before
  scoring (`cf_role_mapped=True`). `cross_method_table.py` has no such mapping. With roles mapped, the same
  `node_threshold=1e-6`, `sfc-npa`, stage `own-threshold`
  (`_archive/007-free0-cf-32seed/rows_s8/s9/s10.jsonl`):

  | target | cf ($\phiind$) | cf_sup ($\phisup$) |
  |---|---|---|
  | L8 25/10628 | 0.391 | 1.000 |
  | L9 27/6859 | 0.0995 | 0.997 |
  | L10 32/3021 | 0.540 | 0.880 |

- **Verdict: MISMATCH.** The 0.000 appears to come from the role-label bug, not from the method. The rerun
  with roles mapped gives $\phiind$ 0.10–0.54 and $\phisup$ 0.88–1.00. (That rerun also differs in
  discovery size, e.g. 43,805 vs 137,021 upstream nodes at L8, so it is not an exact replay.) The
  "output-targeted and latent circuits fail each other's objective" claim is not supported on the latent
  side as stated. In 007 the SFC circuits' zero-ablation sufficiency is low (`free0` 0.04/0.24/0.15), which
  would be a defensible replacement claim.
- **Suggested TODO replacement:** `% source: dev-notes/sfc-replication-and-cross-method.md:79,94 (in-sample,
  pre-v1; raw cross_method.jsonl lost). SUPERSEDED: _archive/007-free0-cf-32seed rows_s8-10 (cf roles mapped)
  give phi_ind 0.39/0.10/0.54, phi_sup 1.00/1.00/0.88 on the same targets; the 0.000 is the unmapped-role
  artefact (007 runner.py:17-21).` Rewrite the sentence, e.g. "scores zero-ablation sufficiency of only
  $0.04$--$0.24$ on each of the three deep target latents ...". Decide whether to keep the $\phiind$/$\phisup$
  claim at all; this paragraph is under the arena C hold, decision 4.

---

## 4. main.tex:2249 (TODO at 2250): "for each source the share of circuits that pass or fail only under C is the same on every background ($0.48$ to $0.51$)"

- **Source:** `experiments/071-contrast-sweep/source_figure.py` (`only_c_fails` + `values`, printed as
  "pass + it"), on `071/out/eval.shard*.jsonl`. Recomputed with `srchunt/c071.py` (no figure written):
  280 targets, matched size n* = 489.5.

  | trained on | pass + fails-only-C (close / random / distant background) |
  |---|---|
  | close | 0.500 / 0.500 / 0.500 |
  | random | 0.482 / 0.482 / 0.482 |
  | distant | 0.508 / 0.508 / 0.508 |
  | fused | 0.478 / 0.478 / 0.478 |

  The same run reproduces the neighbouring numbers: median C 0.905 (close on close) to 0.580 (close on
  distant); pass on distant 0.271 / 0.371 / 0.378.
- **Verdict: MATCH** (0.478–0.508, which rounds to 0.48–0.51 including fused; without fused it is
  0.48–0.51 as well).
- **Suggested TODO replacement:** `% source: experiments/071-contrast-sweep/source_figure.py (only_c_fails,
  "pass + it" column; close 0.500, random 0.482, distant 0.508, fused 0.478; 280 targets, n* 490).`
