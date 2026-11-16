# 121 RESUME (for a fresh agent after the 15:05 restart)

State at 14:25 on 2026-11-15: all CPU prep is done. **No GPU work has run**, and `out/refit_*` holds only the copied
marker files. The GPU was held by gpu-44 (09:34 to 11:52) and then by gpu-215z (exp 120, from 11:52, in fit job 2 of
at least 3, about 12–14 min per target). The lock was never taken by gpu-refit.

## Goal (the brief, short form)
Scale up 116 §1b, the error-zeroed refit. Arms per target:
- **(a)** WCM v4 refit with every upstream SAE error zeroed in training (λ 1e-3, 400 steps, same records), scored with
  errors zeroed. This is the main arm.
- **(b)** The production circuit scored with errors live on this GPU. This is 116's `rs_<family>` live row, reused and
  not re-run: it is the same scorer, records, code and GPU.
- **(c)** Arm (a) scored with errors live. `refit_scale.py` writes it alongside (a).
- No errors-live refits, except the 18 pilot targets already done in 116 (`116/out/refit_live`).

The report covers pass rates with bootstrap CIs, the paired (a) − (b) difference, failure modes (necessity-only vs
faithfulness), median sizes and the size ratio, everything by band and kind. It adds TuringLLM's error-free-model
context: 116's `diag_recon` ratio in [0.75, 1.5], with arm (a) restricted to those targets.

## Files (all in this folder)
- `select_targets.py`: already run (seed 20261121). It writes `targets_<family>.txt`, the run lists of new fits only,
  in round-robin cell order so a truncated run stays balanced, and `out/targets_<family>.csv`, the full plan.
  - TuringLLM: 81 = 9 per cell. 18 are the 116 pilot targets, whose zero-fit rows are reused from
    `116/out/refit_zero`. The other 63 are new.
  - Pythia-70m: 36 = 4 per cell × 9 cells, drawn from 116's 54.
  - GPT-2 small: 36 = 3 per cell × 12 cells, drawn from 116's 60.
- `refit_scale.py`: a family-aware wrapper around 116's `refit_err.install_train_error_ablation`, `errpatch` and 062
  `driver.Runner`, all imported and not re-implemented.
  - Resumable per target (`out/refit_<family>/rows.jsonl`).
  - Has `MAX_MIN` and `STOP_AT=HH:MM`, after which it starts no new target.
  - `DRY=1` checks the config. Dry runs passed for all three families: λ 0.001, 400 steps, v4, the correct CTX_DIR and
    OUT, and copied production WCM_VERSION / WCM_R markers.
- `chain.sh`: runs the jobs strictly in sequence: tl1 (75 min), tl2 (75), tl3 (40, leftovers), pythia (75), gpt2 (75).
  Each job gets `STOP_AT` (default 22:45). Logs go to `out/logs/<job>.log` and `out/logs/chain.log`, which ends with
  `CHAIN_DONE`.
- `report.py`: CPU only. Writes `out/report.md` and `out/paired_<family>.csv`. Tested on the pilot rows, it
  reproduces 116 §1b exactly: (a) 72.2% [50.0, 88.9], median 322 latents; (b) 83.3%, 548 latents.

## GPU protocol (strict)
1. Wait until `experiments/120-known-circuits-errors-zeroed/out/GPU_DONE` exists AND `.gpu-lock` (repo root) is
   absent. Check with a short blocking loop (about 170 s per Bash call) or a background loop, and don't end the turn
   while waiting.
2. Take the lock by writing `gpu-refit` into `X:\Projects\AIs\Turing\Publication\3 Implementation\2\.gpu-lock` with the
   Write tool. Then log_progress.
3. Launch the chain (one GPU process at a time), with Bash `run_in_background`:
   ```
   wsl bash -lc "cd '/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2' && STOP_AT=22:45 bash experiments/121-error-zeroed-refit-scale/chain.sh"
   ```
   It is resumable: re-running skips the targets that are done. For a single job:
   `wsl bash -lc "cd '/mnt/x/...' && FAMILY=turingllm MAX_MIN=75 STOP_AT=22:45 PYTHONPATH=src .venv/bin/python -X utf8 experiments/121-error-zeroed-refit-scale/refit_scale.py"`.
   Smoke option: add `LIMIT=1` and check that `out/refit_turingllm/rows.jsonl` gets a zero and a live row with
   `train_err_calls > 0`.
4. Hard stop: all GPU work must finish by 23:00. STOP_AT 22:45 stops new targets; a deep TuringLLM target takes about
   4 min.
5. When the chain ends or STOP_AT passes, delete `.gpu-lock` with `mcp__agent-gate__delete_path`, then log_progress.

## Time estimates (local RTX 5070 Ti)
- **TuringLLM** (from 116's zero-fit pilot, fit plus 2 scorings): L0-4 about 50 s, L5-7 about 120 s, L8-11 about
  220 s. That is about 2.2 min per target, roughly 140 min for the 63 new targets, so tl1 + tl2 cover about 64.
- **Pythia-70m**: the pod's median fit was 78 s (shared H100). Locally, expect about 1–1.5 min per target, roughly
  45 min for 36.
- **GPT-2 small**: the pod's median fit was 174 s. Expect about 2.5–3.5 min per target, roughly 100 min for 36.
- **Total** about 4.8 h. If the GPU frees too late, the round-robin order means a truncated run still covers every
  cell. Report whatever has landed.

## After the GPU
1. `python -X utf8 experiments/121-error-zeroed-refit-scale/report.py`, then read `out/report.md`.
2. Write `README.md` here with these sections:
   - questions;
   - method, which reuses 116; arm (b) = 116 rs live rows;
   - results tables;
   - caveats:
     - the train patch has not been reviewed line by line;
     - this is the local GPU, not the pod;
     - `diag_recon` covers TuringLLM only;
     - the public cells are small;
     - the 18 pilot rows come from 116;
   - **proposed paper text**.
3. Anchor the paper text to quoted sentences in `paper/main.tex` (do not edit paper/):
   - the `app:error-ablation` sentence beginning "In a pilot of $18$ TuringLLM targets ($2$ per band and kind),
     refitting with the errors ablated during training as well recovers a pass rate of $72\%$ ...";
   - the closing "The circuits are therefore faithful circuits of the model with its SAE errors, ...";
   - the Limitations sentence "It is faithful in the regime it was fitted in, with the SAE errors live; fitted and
     scored with every upstream error zeroed, circuits pass for most targets at about half the size, ...". The comment
     above it says "Pilot numbers (18 targets) until experiment 121 lands."

   Framing (Daniel's): circuits are faithful in the regime they were fitted in, and WCM also finds circuits in the
   error-free SAE model, as a positive result. Report the gap to errors-live honestly.
4. Post a results comment on Linear DAN-214 and leave its status alone. log_progress with kind=done.

## Rules
- Write only in this folder.
- Edit/Write only; deletes only via agent-gate `delete_path`.
- No downloads and no commits.
