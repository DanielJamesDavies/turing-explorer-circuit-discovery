# 124 RESUME (DAN-217): state and the exact local GPU launch

**UPDATE 2026-11-16 05:43: the GPU run is DONE** (04:28-05:43). The refit guard is OK and the GPU lock is released.
The results are in README "GPU results" and `out/report.md`: paired -0.7 pp [-3.0, +1.7], n 296. The state below is
the pre-run snapshot.

## State (2026-11-16 ~03:30)

Done, CPU only, local:
- Code and 15/15 CPU tests passing (123's 16/16 still pass; 123 was not changed).
- `targets_subset.txt`: 300 targets, seed 20261116 (`out/subset.json`).
- `out/idf114.npz`: 114's IDF, rebuilt from 114's exact 100k-window sample. It reproduces 114's cosine on all 6,009 of
  114's held rows with a max difference of 0.0 (`out/idf_verify.json`).
- CPU plan (`out/plan_para.jsonl`), the full CPU dry-run build (`out/build_para_cpu.jsonl`, `out/ctx_para_cpu/`) and its
  check (`out/check_para_cpu.jsonl`).
- Public-model check (`out/public_check.json`).
- Time estimate (`out/time_estimate.json`).
- The dry-run report: `out/report.md`.

Not done: everything on the GPU, which is the real build (`out/ctx_para/`) and both re-scores. **No GPU job was
launched.** The CPU dry-run records (`ctx_para_cpu`) are never scored. The scorer reads `ctx_para`, which the GPU
build writes with the protocol forward in bf16, as 115 and 123 do.

## GPU-lock protocol

`.gpu-lock` at the repo root says who owns the GPU. At 02:40 it held `gpu-44`, which belongs to another job.
1. **Before the launch.** Check that `.gpu-lock` is absent. If another job holds it, wait.
2. **Take the lock.** Use the Write tool to create `<repo>/.gpu-lock` with the single line `dan217-124`.
   `run_local.sh` refuses to start, and refuses to start each GPU step, unless the lock holds exactly `$JOB`
   (default `dan217-124`).
3. **After the run** (DONE, or a failure you will not resume tonight), delete the lock with agent-gate:
   `mcp__agent-gate__delete_path` on `X:\Projects\AIs\Turing\Publication\3 Implementation\2\.gpu-lock`, reason
   "124 GPU run finished". The script never deletes it.

## Launch (WSL, one GPU process; ~1.2 h)

```
wsl: cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && \
     bash experiments/124-paraphrase-filter-subset/run_local.sh 2>&1 | tee -a experiments/124-paraphrase-filter-subset/out/run_local.log
```
- Dry run (prints the commands, needs no lock): prefix `DRY_RUN=1`.
- Steps, all resumable: build (GPU, ~5 min) → check → prep → score orig (~33 min) → score para (~33 min) → verify →
  report.
  - A kill mid-run loses nothing. Rerun the same command and done targets are skipped.
  - To run part of it: `STEPS_ONLY=score,verify,report` or `VARIANTS=para`.
- Progress: `tail -f experiments/124-paraphrase-filter-subset/out/rs_orig.log` (then `rs_para.log`).
- Success:
  - `verify` prints `REFIT GUARD ... OK` for orig and para, with missing 0.
  - `out/report.md` has the paired table.

Single steps (repo root; WSL `.venv/bin/python` for the GPU ones, any python for the CPU ones):
```
python -X utf8 experiments/124-paraphrase-filter-subset/select_subset.py                     # CPU, done
STAGE=idf  python -X utf8 experiments/124-paraphrase-filter-subset/rebuild124.py             # CPU, done
STAGE=plan python -X utf8 experiments/124-paraphrase-filter-subset/rebuild124.py             # CPU, done
STAGE=build DEVICE=cpu PYTHONPATH=src python -X utf8 experiments/124-paraphrase-filter-subset/rebuild124.py   # CPU dry run, done (~25 min)
STAGE=build PYTHONPATH=src .venv/bin/python -X utf8 experiments/124-paraphrase-filter-subset/rebuild124.py    # GPU
STAGE=check [DEVICE=cpu] python -X utf8 experiments/124-paraphrase-filter-subset/rebuild124.py
STAGE=prep python -X utf8 experiments/124-paraphrase-filter-subset/rescore124.py
STAGE=cmd  python -X utf8 experiments/124-paraphrase-filter-subset/rescore124.py             # prints the driver lines
STAGE=verify python -X utf8 experiments/124-paraphrase-filter-subset/rescore124.py
python -X utf8 experiments/124-paraphrase-filter-subset/report124.py
python -X utf8 experiments/124-paraphrase-filter-subset/public_check.py                      # CPU, done
python -X utf8 -m pytest -q -p no:cacheprovider experiments/124-paraphrase-filter-subset/test_124.py
```

## Things to know

- `prep` hard-links the production circuits into `out/rs_*/main/circuits/`. Do not rewrite those files in place.
- Changing the rule (threshold, subset) means a fresh build. First delete these via agent-gate: `out/ctx_para/`,
  `out/build_para.jsonl`, `out/check_para.jsonl`, `out/targets_para.txt`, `out/prep_para.json`, `out/rs_para/`. For a
  new subset, also delete `out/rs_orig/` and `out/plan_para.jsonl`.
- The report reads `build_para.jsonl` (GPU) when it exists, else the CPU dry run. Set `BUILD=build_para_cpu.jsonl` to
  force the dry run.
- The subset could grow to ~860 targets inside 3 h (`time_estimate.json` `n_for_3h`). To grow it, set
  `SUBSET_N=<n> python select_subset.py`. This writes a new list, so delete the outputs above first.
