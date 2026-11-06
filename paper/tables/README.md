# paper/tables: generators for the DAN-148 tables and the depth-frontier figure

Written 2026-10-27 (DAN-148, results presentation), before the protocol-v2 pod run. Each script reads a driver OUT
folder and writes LaTeX rows (or a figure) into `paper/tables/out/<tag>/`. Nothing here edits `paper/main.tex`: after
the pod, paste the generated rows over the placeholder rows (or `\input` the file) and update the matching
`\pendingcell` / `\pending` markers.

The pass rule, the vacuous exclusion, the depth bands and the row reader are imported from
`experiments/062-h100-protocol-v1/merge.py` (the same module `paper_main.py` and `pass_rule.py` use), never re-coded.
`common.py` holds the shared loaders, the Wilson interval and the pre-activation rule.

Run from the repo root with the project venv: `python -X utf8 paper/tables/<script>.py [options]`.

| Script | Paper target | Default input (protocol v2) | Output |
|---|---|---|---|
| `main_result.py` | `tab:main-result` (sec:main-result): site kind x depth band, n, post-Top-K pass % [95% Wilson], pre-activation pass %, median Z / A / C, necessity, size | `experiments/094-tl-v2-records/out_v2` (`main/eval.shard*.jsonl`, `main/status.shard*.jsonl`, optional `targets.csv` from merge.py for the near-threshold stratum) | `main_result_body.tex`, `main_result.csv`, `main_result_numbers.md` (headline and per-band rates with n and intervals, the "Two pass rates" values of app:production, the caption's skipped / vacuous counts) |
| `matrix_body.py` | `tab:matrix` (Table 2, app:method-comparison), with MIB-style best (bold) / second-best (underline) marking | `experiments/082-public-runs/out_turingllm/dan159/table2.csv` (083 `table2.py` output; chain.sh `OUT083=$TL_WORK/dan159`) | `matrix_body.tex`, `matrix_rows.csv` |
| `depth_frontier.py` | body-figure candidate: pass rate against median circuit size per depth band, WCM and unweighted, across the lambda sweep | `experiments/094-tl-v2-records/out_v2/sweep` (13 arms: W 2.5e-5 .. 4e-3, U 1e-5 .. 1e-3) | `depth-frontier.pdf/.png`, `depth_frontier.csv` |

Public models: `--family pythia-70m` or `--family gpt2-small` (OUT defaults to `experiments/082-public-runs/out_<family>`;
bands and k come from `062/family_table.py`).

## Definitions

- **Population.** Held-out strongest rows of the main run, error and skip rows dropped, last row per (seed, held),
  vacuous denominators excluded and counted (merge.py `vacuous`). Exclusions for the caption: targets with a `skip`
  status row, and vacuous circuits.
- **Headline pass (post-Top-K).** merge.py `passes`: Z, A, C (`free0_tk`, `freeM_topk_tk`, `freeN_topk_tk`) in
  [0.75, 1.5] and necessity (`phi_sup_blind_tk`) >= 0.9; a missing score fails. (Floor 0.8 until 2026-10-31; the
  v1-validation outputs were made at 0.8.)
- **Pre-activation pass** (protocol-v1.md section 12.1, baseline 0): Z_pre = `free0_raw_pre / a_pos_pre`,
  necessity_pre = `(a_pos_pre - phi_sup_blind_raw_pre) / a_pos_pre`, A and C = `freeM_topk_pre`, `freeN_topk_pre`;
  same band and threshold; a non-positive `a_pos_pre` fails. Same population as the headline.
- **Intervals.** 95% Wilson score intervals on the pass counts (z = 1.96).
- **tab:matrix marking.** Per column over every row except the exact-effect block (a subset of targets): Z / A / C
  closest to 1, necessity and pass rates highest, ranked at the printed precision so equal-looking cells share a
  mark; size is not marked; undefined cells are dashes and never marked.

## v1 validation (2026-10-27; NOT paper numbers)

`main_result.py --out experiments/062-h100-protocol-v1/out_full/out --tag v1-validation` reproduces the v1 15k
headline exactly: 46.4% (6,953 / 14,994; Wilson 45.6-47.2), 42 skipped targets, 10 vacuous circuits, by kind
65.0 / 51.8 / 14.3% and by band 69.2 / 45.2 / 22.3%, median size 494 (386 passing), 91.3% inside 10^2-10^3.
The pre-activation pass rate under the section-12.1 rule is 44.4% overall; in the near-threshold stratum (1,142) it
is 7.0% against 0.8% post-Top-K. The protocol's quoted 10.9% for that stratum is a different rule (`pass_rule.py`:
empty-circuit Z_pre and the post-Top-K necessity); applied to the same rows that rule gives 10.9%, so the difference
is the rule, not the data. Under the section-12.1 rule, necessity_pre >= 0.9 holds for only 54.5% of near-threshold
targets (post-Top-K necessity: 99.0%), which is what pulls their pre-activation rate down.

`depth_frontier.py --tag v1-validation --out .../062/out_full/out --extra .../062/out_bands .../062/out_depth
.../062/out_lowlam` reproduces the shared-penalty sweep rates (unweighted 2.1% and 1.5% against WCM 37.5% and 18.2%
at layers 5-7 and 8-11). At layers 0-4 it gives 65.3% for WCM where the paper prints 64%: `paper_sweep.py` keeps
the one vacuous sweep target in the denominator, while these scripts exclude it, as the pass rule says.

`matrix_body.py` was checked for format only, on `experiments/083-v1-comparison/out_smoke/table2.csv` (one target):
the v1 comparison never produced a full `table2.csv`.

Outputs of the validation runs are in `out/v1-validation/` and `out/smoke-083/`; do not copy them into the paper.
