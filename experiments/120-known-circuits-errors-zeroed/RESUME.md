# 120 resume notes (written 2026-11-15 ~14:25 for the 15:05 power-off)

Every stage is resumable: fit120 skips targets whose circuits (`out/wcm/{main,task}/circuits/<key>.pt`) and eval rows
(`out/wcm/dan215/eval.jsonl`, keyed seed/circuit/data/held/fold) exist; structure120 skips rows already in
`out/wcm/dan215/structure.jsonl` (keyed seed/circuit/frame/draw); diag120 skips rows in `out/diag/eval.jsonl`.
A kill mid-target only loses that target's unfinished work.

## Done

- Scripts: `common120.py`, `fit120.py`, `structure120.py`, `diag120.py`, `report120.py`, `test_120_cpu.py`
  (CPU tests 3/3 pass).
- Inputs copied: `out/wcm/task_ctx/*.pt` (117's task records), `out/targets.json` (117's).
- Fits, error-zeroed training + dual scoring: jobs 1-2 finished 12 / 15 targets (`out/logs/fit_job{1,2}.log`).
  Job 3 (`out/logs/fit_job3.log`, started 14:17) is on the last 3 targets (9.mlp.3600, 10.mlp.3774, 10.mlp.4386);
  see the status section at the bottom for where it stopped.

## Left (GPU, in this order; take `.gpu-lock` first per the protocol)

All from WSL at the repo root (`cd '/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2'`):

1. Finish fits (no-op if all 15 are done; re-scans finished targets in ~1 min):
   `BUDGET_MIN=68 PYTHONPATH=src .venv/bin/python -X utf8 -u experiments/120-known-circuits-errors-zeroed/fit120.py > experiments/120-known-circuits-errors-zeroed/out/logs/fit_job4.log 2>&1`
2. Structure test, error-zeroed frames (~50-60 min; rerun the same command if it is cut off):
   `N_PROMPTS=128 N_DRAWS=20 PYTHONPATH=src .venv/bin/python -X utf8 -u experiments/120-known-circuits-errors-zeroed/structure120.py > experiments/120-known-circuits-errors-zeroed/out/logs/struct_job1.log 2>&1`
3. Diagnostic, 117's original circuits scored with errors zeroed (~15-25 min):
   `MAX_MIN=40 PYTHONPATH=src .venv/bin/python -X utf8 -u experiments/120-known-circuits-errors-zeroed/diag120.py > experiments/120-known-circuits-errors-zeroed/out/logs/diag_job1.log 2>&1`
4. Release `.gpu-lock`; write `out/GPU_DONE` (one line: the time) so gpu-refit can start.

## Left (CPU)

5. `CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -X utf8 experiments/120-known-circuits-errors-zeroed/report120.py`
   -> `out/report.md` (117's report117 on the 120 rows) and `out/report120.md` / `.json` (paired with 117, control,
   diagnostic).
6. README.md (questions, method, tables, caveats, proposed paper text anchored on main.tex `app:known-circuits` and
   the Limitations "A circuit is a fitted intervention" paragraph); Linear comment on DAN-215 (status unchanged).

## Status at shutdown

- 14:54: fit job 3 stopped by hand (pkill) so the 15:00 deadline holds. GPU idle, `.gpu-lock` released, NO
  `out/GPU_DONE` yet (gpu-215z still has GPU work left, so gpu-refit must keep waiting).
- Fits complete with eval rows: 13 / 15 targets (all except 10.mlp.3774 and 10.mlp.4386).
- 10.mlp.3774: corpus circuit saved (`out/wcm/main/circuits/10.mlp.3774.pt`); its task fit was cut off mid-fit and
  is redone by step 1. No eval rows for it yet.
- 10.mlp.4386: not started.
- Step 1 therefore takes ~25 min (one task fit + one full target + their scoring), then steps 2-3 (~75-85 min).
- Structure and diag have not been run yet. A partial report (fits only) was checked to work: `out/report120.md`
  reproduces 117's 1/66 head checks and the 15/15 full-model control from 117's rows.
