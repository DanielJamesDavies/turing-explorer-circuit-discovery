# 123 Document-group-disjoint held-out sets: full rebuild and re-score (DAN-217)

## Question

114 found that 34.6% of TuringLLM held-out strongest rows paraphrase a training row of the same target. The corpus
holds T5 paraphrases of every Phi-3 document. The 36-target pilot (115) re-scored the fixed circuits on
document-disjoint sets and found no large effect, but n = 36 cannot rule out a few points. This experiment does the
same for **every production target of all three models**, with no refit, and reports the pass rates side by side.

Status (2026-11-16 02:15): code, tests and the CPU dry run are done. **The GPU steps have not run.** The coordinator
decides when (see RESUME.md).

## Design

**What changes.** Only the held-out strongest set (the pass rule's set), and the mid-band held-out set, which is
reported. Training rows, D_train, mid.train, B_mid and the contrast rows are the production record's, byte for byte.
`STAGE=check` verifies this for every record. So the production circuits are scored as they are, and nothing is
refitted.

**Group clause** (115's clause, unchanged). A held-out row is allowed only if none of its groups is a group of an
activating training row of the target (strong.train 48, which includes the fit's 32, plus mid.train). Groups are
taken over all of a row's corpus twins (same 085 content hash).

| model | `doc` group | `topic` group (sensitivity) |
|---|---|---|
| TuringLLM | source path with the phi-3-mini / data-augmentation prefix stripped = one Phi-3 original plus its T5 paraphrase file (the paraphrase family; 114 `check.groups()`) | dirname of that path = the up to 6 sibling documents of one leaf topic |
| Pythia-70m, GPT-2 small | the record's own per-row `doc` (069 provenance `shard_<N>.docs.npy`); unknown (−1) is never forbidden | none |

**TuringLLM rebuild** (`rebuild.py STAGE=build`, GPU). This is the production held-out-strongest rule, imported
unchanged from the pilot: `115 build_heldout.build_one`, which is 086's DAN-157 rule plus the group clause.
- Candidates: the original held-out rows (pool order), then the T1 top-512 scan, strongest first.
- Each candidate must fire at its anchor, recomputed with the protocol forward.
- It must not be an exact, anchor-prefix or near copy of any training or contrast row, or of an accepted row.
- Mid-band pool rows never enter.
- The final set gets an in-batch firing re-check.
- Fill to the original size (16), floor 8. A set below 8 is flagged and still scored.

**Public rebuild** (`rebuild.py STAGE=plan`, CPU). Removal only. The public path has no top-up source (its
group-aware split only removes rows), so leaky rows are dropped and not replaced.

**Mid-band** (`MID_RULE=drop`, the default). Leaky mid held rows are removed, with no refill: the mid pool has no
top-up source, as in 086. If every row is leaky, the set is kept as it was and flagged. `MID_RULE=keep` leaves it
untouched.

**Contrast held-out sets**: measured, not rebuilt (see open decisions).

**Re-score** (`rescore.py`; GPU via `062/launch.sh`). The production scorer, unchanged: `062 driver.py MODE=main`
with `CTX_DIR` set to the variant's records and an `OUT` holding the production markers (WCM_VERSION v4, WCM_R) and
hard links to the production circuits.
- `Runner.fit` loads an existing circuit, so nothing is refitted.
- prep lists only targets that have a circuit file. The driver would fit a missing circuit, so prep never lists a
  target without one.
- `STAGE=verify` checks that no status row has t_fit > 0 and that no train row was written.
- The driver resumes per target: status rows are keyed by seed, eval rows by seed + held. `SKIP_SPEC=1`.

**Variants**:
- `orig`: the production records re-scored on the same machine as the new sets. This is the paired baseline. 115
  found that H100 vs RTX scoring disagrees on 1/36 pass decisions (median gap 0.03), so old and new sets must not be
  compared across machines.
- `doc`.
- `topic`: TuringLLM only.

**Report** (`report.py`, CPU). Pass rule, vacuous rule and bootstrap come from 062 merge.py / 109 lib109: 95%
percentile bootstrap over targets, B = 4000, seed 20261108, with n.
- **Per variant:** the variant's own non-vacuous population (the headline definition).
- **Paired:** non-vacuous on both orig and new. Diff with a paired bootstrap interval, and flips in both directions.
- **Conservative:** orig's population, where a new set that is vacuous, unscorable or missing counts as a fail.
- **Cells:** band × site kind with margins.
- **Mid-band:** the same, reported.
- **Headline:** the production rows from the 109 archives are recomputed (88.0 / 81.9 / 82.4%) and checked against the
  orig re-score.

## Dry run (CPU, 2026-11-16; `out/plan.log`, `out/<model>/plan_*.jsonl` / `build_doc.jsonl`)

Every production target was covered: 7,786 TuringLLM, 509 Pythia-70m and 512 GPT-2 small circuits. There were
0 errors. The production headline recomputed from the archives is 88.0% [87.2, 88.7] (6565/7464), 81.9% [78.5, 85.2]
(416/508) and 82.4% [79.1, 85.5] (421/511), as the paper quotes.

**TuringLLM, group clause only.** The firing / duplicate rule and the fill run on the GPU. The CPU counts equal the
pilot's for all 36 × 2 pilot target-groups (a test checks this).

| | doc | topic |
|---|---|---|
| old held-out strongest rows in a forbidden group (dropped) | 54,495 / 120,754 (45.1%) | 65,346 (54.1%) |
| targets with ≥ 1 dropped / all rows dropped | 7,368 / 236 | 7,496 / 515 |
| median allowed candidates (orig held + scan) | 318 | 260 |
| certainly short of the original size | 162 | 270 |
| certainly below floor 8 | 214 (all attention) | 266 |
| **unscorable (0 candidates)** | **10**, all attention | **17** |
| mid-band held rows leaky (removed under MID_RULE=drop) | 5,564 / 118,178 (4.7%) | 11,550 (9.8%) |
| contrast held rows sharing a group with contrast-train / activating-train rows (not rebuilt) | 9.3% / 4.7% | 22.1% / 12.0% |

Unscorable on doc: 10.attn.29674, 10.attn.33695, 11.attn.13375, 11.attn.7486, 2.attn.7698, 7.attn.27151,
7.attn.32218, 7.attn.33961, 9.attn.16671, 9.attn.39704. The dropped share is flat across bands (L0-4 44.7%, L5-7
40.9%, L8-11 46.9%). Short and below-floor counts are upper bounds on what the build will produce. The final counts
come from `build_<group>.jsonl` (`n_old_leaky` = dropped, `n_scan` = replaced, `n_held`, `below_floor`,
`unscorable`).

**Public, document rule (removal, built).** `STAGE=check`: 0 failures on 1,021 records.

| | Pythia-70m | GPT-2 small |
|---|---|---|
| old held-out strongest rows sharing a document with an activating training row (dropped) | 2,738 / 7,476 (36.6%) | 1,585 / 7,695 (20.6%) |
| targets with ≥ 1 dropped | 451 / 509 | 367 / 512 |
| median rows left (of 16) | 10 | 13 |
| below floor 8 / **unscorable (0 rows)** | 179 / **15** | 77 / **3** |
| mid-band held rows leaky | 1,183 / 7,889 (15.0%) | 240 / 8,051 (3.0%) |

So DAN-217's item 4 has an answer. The public corpora have no paraphrases, but their held-out sets share source
documents with training rows: 37% of rows for Pythia (Pile documents are long) and 21% for GPT-2. The current
public split groups by duplicates only.

## GPU cost (the scoring and build steps have NOT been run)

Basis:
- Production T3 `t_eval` (H100, scored under co-tenancy): 7.95 process-hours for all 7,786 TuringLLM targets. GPT-2
  is 0.71 and Pythia 0.29 process-hours.
- The pilot's local RTX 5070 Ti single-process eval took 1.63× the production `t_eval` on the same 36 targets.
  Scoring includes strong + mid.
- The pilot's build took a median of 0.3 s per target per group, plus about 100 s of engine load.

| step | 5× H100 pod (P=6 per GPU, MPS) | local RTX 5070 Ti (1 process) |
|---|---|---|
| plan + check + prep (CPU) | ~5 min | measured: plan 2 min, check 10 s |
| TuringLLM build, doc + topic | ~10 min (BUILD_P=2) | ~1.7 h |
| TuringLLM score, per variant | ~16 min + start-up ≈ 20 min | ~13 h |
| TuringLLM, 3 variants (orig, doc, topic) | **~1 h** | ~40 h (orig + doc only: ~28 h) |
| public, 4 runs (orig + doc × 2 models) | ~25 min (start-up dominated) | not valid locally (smoke data pass) |
| **total** | **~1.5–2 h wall, ~8–10 GPU-h** | TuringLLM only, ~30–42 h |

These are extrapolations, not pod measurements. Per-target co-tenancy on the pod differs from T3's 10 per GPU.

## Run

See RESUME.md for the exact commands. Briefly:
- `run_gpu.sh` runs plan → build → check → prep → score → verify → report.
- Every step resumes.
- `DRY_RUN=1` prints the commands only.

## Files

- `common123.py`: paths (env overrides; `TL_ROOT` / `PUB_ROOT` for a pod), the group labels (`TLGroups`), the pure
  planning functions (`tl_plan`, `leak_counts`, `drop_mid`, `pub_rebuild`, `check_disjoint`) and the jsonl helpers.
  The append helper repairs a torn last line, which the resume test caught.
- `rebuild.py`: `STAGE=plan` (CPU; TuringLLM plan, public build), `build` (GPU, TuringLLM, `SHARD=i/k`), `check`
  (CPU).
- `rescore.py`: `STAGE=prep` (OUT dirs, circuit links, `targets_<variant>.txt`, unscorable exclusions), `cmd`,
  `verify` (counts and the refit guard).
- `report.py`: `out/report.md`, `report_cells.csv`, `report_pairs.csv` and `report.json`.
- `run_gpu.sh`: the end-to-end GPU runner.
- `test_123.py`: 16 CPU tests, all passing. They cover:
  - grouping: no group crosses the split, twins, mid-train groups, public removal;
  - determinism;
  - resume: torn line, lost record, prep idempotent;
  - the report logic: pairing, conservative counting, cell margins, bootstrap;
  - equivalence with the pilot's counts on real data.

  Run: `python -X utf8 -m pytest -q -p no:cacheprovider experiments/123-docgroup-heldout-rescore/test_123.py`.

## Open decisions (for Daniel)

1. **Public refill.** Removal leaves Pythia with 179 / 509 sets below 8 rows and 15 empty. A refill needs a top-N
   corpus scan of the public corpora, the public analogue of T1 (GPU, not built). Alternatively, report removal-only
   with the below-floor stratum shown separately.
2. **Contrast held-out sets** are not rebuilt. 9.3% of TuringLLM contrast held rows share a document with
   contrast-train rows. A rebuild needs the close selector with the stores (086 `rebuild_contrast` plus a group
   clause).
3. **Scope beyond main.** Sweep arms, nulls, α = 1 and the donor test (DAN-217 plan item 2) are not wired. The sweep
   can reuse this design: driver `MODE=sweep` with copies of `sweep/circuits/<arm>/`. The α = 1 rescore and donor
   companions take `CTX_DIR` in the chain.
4. **MID_RULE default** is `drop`. Mid-band is reported, not in the pass rule.
5. **Where to run.** Pod (about 1.5–2 h wall, ~8–10 GPU-h, ideally with T7) or local (TuringLLM only, ~30–42 h).
   The public models need the production data pass (pod, or unpack `pub-*-S10-stores.tgz`).
