# 107 arena A: Ge et al. (A1) and Dunefsky et al. (A2) on GPT-2 small, plus WCM on the same substrate

Status (2026-11-06 11:20, agent `hardstop`, CPU only): **`run_pod.sh` has a HARD time limit** (Daniel approved). See
"Time limit 2026-11-06" below.

## Time limit 2026-11-06 (agent `hardstop`; `run_pod.sh` header "TIME LIMIT" = the full doc)

- **Soft stop** `STOP_EPOCH` (as before: no new target / arm / stage) and **hard stop** `AA_HARD_GRACE` (3600 s) later.
  At the hard stop a watchdog sends SIGTERM to every process of this run, then SIGKILL after `AA_KILL_GRACE` (30 s).
  It finds them by `AA_RUN_TAG=<run id>`, which `run_cmd` puts in each command's environment and every child inherits:
  `launch.sh` shards and their restart loops, Ge / Dunefsky / scorer workers. No pid files, so nothing is ever deleted.
  Each killed process (its ps line) goes to `out/pod/logs/hard_stop.log`, and a summary goes to `events.log`.
- After the hard stop no command starts except P7's. P7 waits for the kill to finish, then packs whatever finished.
  A missing data pass / corpus / file is left out (`--ignore-failed-read`; tar rc 1 "file changed" is accepted), and a
  failed `--summary` is logged and packing goes on. Exit **4** = stopped by the time limit, after the pack.
- **P7 also runs after a soft stop** (before, `stage` refused it, so nothing was packed). It also runs on every run:
  its marker never skips it.
- **Backstop:** if `run_pod.sh` is still running `AA_PACK_MAX_H` (2 h) after the hard stop (a hung P7), the watchdog
  kills P7's commands and `run_pod.sh` itself. The chain then moves on to arena D and STOP_POD.
- SIGTERM / SIGINT to `run_pod.sh` does the same kill. Before, it left the worker processes running.
- **Default `AA_MAX_H=auto`** (the hard stop, in hours after start) = 1.5 × the §7 / plan §5 wall estimate:
  - 2 h serial (P0–P3, the GPU-0 data pass) + 68 GPU-h / #GPUs (45–55 GPU-h + P4h / P6b 10.5–14.5 GPU-h, upper end);
  - **15.75 h on 8 GPUs** (soft stop at 14.75 h), 28.5 h on 4.
  - Set `AA_MAX_H` (chain `ARENA_A_MAX_H`) or `STOP_EPOCH` (chain `ARENA_A_STOP_EPOCH`) to override; with both set, the
    earlier wins. `AA_MAX_H=0` and no `STOP_EPOCH` = no limit. A12's measured re-cost should replace the 68 GPU-h.
- Fixed in passing: P4 / P4h returned the failed-worker COUNT, so 2 failed workers read as "waiting for approval" and 3
  as STOP_AT. Now they return 1.
- Tests (CPU, `AA_HARDSTOP_TEST=1|hang`: fake P4 process trees of sleeps, one ignoring SIGTERM, then a fake P7):
  - hard stop at 20 s: 12 SIGTERM + 2 SIGKILL, P5 not started, P7 packed, rc 4, 0 processes left;
  - hung P7: the backstop killed it, 0 left;
  - SIGTERM to the script: everything killed, rc 143, 0 left;
  - soft stop only: P7 ran, rc 4;
  - the real P7 on an empty work dir: results tgz only, md5 verified, rc 0.
- Chain dry runs with arenas off are identical to the dryrun agent's 2026-11-06 captures.

Status (2026-11-05 13:00, agent `a10-pod`): **A10 `run_pod.sh` (P0–P7) and A11 (chain `ARENA_A=1`, off by default)
written and dry-run checked; the 2-target GPU smoke is not run yet; A12 re-cost open.** See "A10 / A11" below.

Status (2026-11-03 late, agent `arenaA`, CPU only, no GPU used): **downloads D1–D5 done (approved ~22:00), both
baseline envs built (`pip check` clean), CPU fidelity checks run, WCM adapter written and CPU-tested.** See "Log
2026-11-03" below. Linear: DAN-192 (Ge setup + fidelity), DAN-197 (Ge / WCM runs), DAN-200 (Dunefsky), DAN-195 (WCM
adapter). Specs: `../specs/ge.md`, `../specs/dunefsky.md`, `../specs/evaluation.md`. Structure copied from `../arena_c/`.
Each choice that is ours, not the paper's, is tagged **[ours]**; each one Daniel has to make is tagged **[decide]**.

Goal: pod-ready for the 8×H100 session on Saturday 2026-11-07, with the local smoke done on the Wednesday and Thursday
nights.

## 2026-11-06 (agent `scorer`, CPU): per-target summary + ho_xdoc in `score_arena_a.py`

- **Per target** (`--summary`; `summary.json` `per_target`, `per_method_rollup`): one row per (method, target). Best
  setting, the same rule for every method: max ho pass_b rate over prompts, then pass_a rate, then the smaller median
  n latent (Ge = τ grid index, Dunefsky = (branches, iters), WCM = λ with prompts n/a). At it: pass b / a fractions, medians
  of n, Z / A / C / nec a / b, donor R, home pass b, ho_xdoc; plus "any setting passes b" and the smallest passing n.
  Roll-up: targets passing on a majority of prompts at the best setting, out of the non-vacuous targets.
  **q-fallback** (coordinator 2026-11-06): when no setting passes (b) or (a) on any prompt, the best setting is the
  highest median over prompts of q = min(fold(Z), fold(A), fold(C), nec_b), fold(x) = min(x, 0.75 × 1.5 / x) (= 0.75
  at both band edges, lower outside on either side), then smaller n; such rows are marked "best: q-fallback"
  (`best_rule`). Without it the size tie-break picked n = 0–2 Ge circuits on the smoke.
- **ho_xdoc** [ours]: the ho windows whose document (record `strong["doc"]`, from the data dir's provenance; OWT via
  prepare_corpus `--token-field` = source row) is in none of the circuit's training windows (WCM: all strong-train +
  mid-train rows; Ge / Dunefsky: the prompt row). Computed in `Scorer.unit` from the same per-window reads (no extra
  forward pass), stored as the ho row's `xdoc` dict; `n/a: no provenance` / `empty` notes. Rows scored before it have
  none (summary says "not scored"). Existing columns and sections unchanged (smoke `--summary` diff: additions only).

## A10 / A11 2026-11-05 (agent `a10-pod`, CPU, dry runs only): `run_pod.sh` and the chain stage

- **`run_pod.sh`** (header = the full doc): P0 assets (checks only; missing downloads are printed for approval, exit 2),
  P1 envs (ge107 / tc107 gate; `setup_env.sh` printed for approval), P2 corpus + data pass, P3 `make_targets.py`, P4
  GPU canary then Ge + Dunefsky (one worker per GPU; Ge one process per target), P5 WCM v4 (a1 × 48, a2 × 24 tc,
  × 5 λ, via 062 `launch.sh`, one shard per GPU), P6 scorer (`--selftest --donor`, shared pool) + `--emit-ge-jobs` +
  `ge_recovery_tcz.py`, P7 merge + `--summary` + tgz / md5 on RESULTS. Resumable (markers + the runners' own skips),
  `DRY_RUN=1`, `STOP_AT`, `STOP_EPOCH`, `AA_LIMIT` (smoke). `steps.tsv` + `gpu_mem.csv` for A12.
- Choices [ours, decide]:
  - λ arms `2.5e-4,5e-4,1e-3,2e-3,4e-3` (the 15k sweep's grid).
  - Arena A corpus = the first 524,288 windows of OWT file 0 (`AA_N_SEQ`). The pilot's pass 1 rate (42 s per 2,048
    windows locally) puts the full 4M-window S1 corpus at several H100-hours. `AA_N_SEQ=0` uses S1 as is.
  - Dunefsky extended knob on (K = 2, 100–300 branches).
- `score_arena_a.py`: one change, the env `ARENA_A_CTX_DIRS` (os.pathsep list) overrides the smoke `CTX_DIRS`. Unset
  = unchanged; 12 tests pass.
- **`082/chain.sh`** (A11): `arena_a_stage` after S10 + T12 when `ARENA_A=1`. With it unset, the dry run is identical
  to before (TL on, TL off, `LIMIT=2`; timestamps masked). Runbook entry dated 2026-11-05.
- Not done: the 2-target GPU smoke. `run_ge.py` / `ge_recovery_tcz.py` have never run with `--device cuda`, so the
  P4 canary on the pod is the first check.
- Coordinator 2026-11-05 13:00:
  - λ grid = placeholder, read from `AA_LAMBDAS` (`AA_LAMS` fallback).
  - 524k windows and the extended Dunefsky knob accepted. The extended knob is labelled **"beyond published range"**:
    num_branches > 60 in the units.
- **`run_smoke_local.sh`** (written, dry run passed, `SMOKE_READY`; not run): `run_pod.sh` from P4 on this machine.
  - Inputs: the pilot data pass and `out/targets_smoke` (read only); targets 3.tc.13057 and 3.z.23269.
  - P4: Ge K 2 × 6 τ; Dunefsky K 2 × {5, 10, 20} + 100 on 1 prompt.
  - P5: WCM v4 at λ 1e-3, 400 steps, scopes a1 and a2, `DEEP_BS=1`.
  - P6: score with `--donor --selftest`, then Ge recovery. P7: summary only.
  - One GPU process at a time, `NVIDIA_TF32_OVERRIDE=0`.
  - Watchdog every 30 s: abort above 15,300 MiB GPU or below 4 GB host RAM. It refuses to start if the GPU already
    holds more than 1,500 MiB.
  - Timings go to `out/smoke_pod/smoke_summary.md`.
  - **Ge, Dunefsky and Ge recovery run on the CPU.** The local ge107 / tc107 torch builds are CPU-only, and their
    pinned torch cannot use the sm_120 card.
- Tuning hooks (coordinator 13:20, `TUNING.md`):
  - P3 passes `EXCLUDE=tune_exclude.txt` to `make_targets.py`. A missing file fails P3, and the picks are re-checked
    against it.
  - P5 reads `out/tune/CHOSEN.env` (`AA_TUNE_ENV`). Only the keys AA_RANK_W, NEC_W, STEPS and AA_TUNE_CONFIG are
    accepted, and they go into every WCM fit, production included. STEPS is honoured; `AA_STEPS` overrides it.
  - The choice is recorded in `wcm/TUNE_ENV`, and a later run with a different choice is refused.
  - A missing file fails P5 unless `AA_ALLOW_UNTUNED=1`. The smoke sets it.
  - P5 skips a scope with no targets and refuses an unknown scope.
  - `out/` is not in git, so CHOSEN.env must be copied to the pod, or pointed to with AA_TUNE_ENV.
- Held-out baselines + A8b (Daniel 13:35): new resumable stages.
  - **P4h**: `--emit-heldout-prompts` (K = 8) into `targets_held/`. Then Ge (same τ grid) into `ge_heldout/` and
    Dunefsky (published branch grid; no > 60 knob) into `dunefsky_heldout/`, with the P4 worker scheme.
  - **P6b**: `score_perprompt.py` per GPU on the train and held sets. Scope a1 = WCM a1 + Ge held; a2 = WCM a2 +
    Dunefsky held. The top-k curve is in arm l1e-3; Ge held units use τ stride 2.
  - P6b then merges into `perprompt/all`. The P6 rows serve as the train-prompt baselines (`--baseline-rows`). Its
    Ge recovery runs per shard, then `--summary`.
  - The smoke runs both at 1 held prompt per target.
  - Dunefsky's held-out circuits also get Ge recovery jobs, from the new `ge_jobs_heldout.py`. It reuses
    `score_arena_a`'s `dunefsky_units` + `ge_jobs` unchanged; only the uid / job get the `pp|` prefix, so that
    `score_perprompt.py --summary` joins them to its held-out rows. The jobs are emitted per a2 shard, then run with
    that shard's other jobs.
  - CPU check on the smoke Dunefsky dir: 14 jobs from 14 circuits.
  - Cost ≈ £28–38 (plan §5).
- New `run_pod.sh` knobs (defaults unchanged): `AA_FROM`, `AA_STORES`, `AA_TARGETS_DIR`, `AA_KEYS`, `AA_PACK`,
  `AA_BASELINE_DEVICE`, `AA_GE_NTAU`, `AA_DUNEFSKY_K`, `AA_DUNEFSKY_BRANCHES`. Every command is also timed into
  `cmds.tsv`.

## A8b 2026-11-05 (agent `a-perprompt`, CPU only): `score_perprompt.py`, the per-prompt ("their turf") test

Question: does the part of a WCM circuit that is active on one prompt do as well, on that prompt, as a circuit built
for that prompt (Ge hierarchical / post-hoc, Dunefsky)? `score_arena_a.py` does the other direction (their circuits
on held-out contexts). Sibling script: it imports `score_arena_a` (Scorer, units, helpers); `score_arena_a.py`, `src/`
and the runners are unchanged. Tests: `test_score_perprompt.py`, 9 pass (CPU, no model). Smoke output:
`out/perprompt_smoke/` (`rows.jsonl`, `check.jsonl`, `ge_jobs.pp.jsonl`, `ge_recovery.pp.jsonl`, `summary.md`,
`curves.jsonl` / `curves.csv`, `targets_held/prompts.jsonl`, logs).

**Definitions (all ours):**
- **Units, per (target, λ arm, prompt).** `wcm_full` = the whole circuit on that prompt (= the home scope on prompt q).
  `wcm_fire` = the members with clean code > 0 at some position ≤ anchor, BOS excluded (fire_count's definition; the
  site model hides BOS), α unchanged; latent and (latent, pos) counts recorded. `wcm_fire_top` = **ours
  (adaptation)**: the firing members ordered by α × (max clean code on the prompt), top k for k = 1, 2, 4, … < n_fire.
  Only in one arm (`--topk-arms`, default the first `--wcm` label).
- **Tests.** `Scorer.unit` unchanged, on a one-row eval set: Z / A / C, nec_a, **nec_b (all upstream errors frozen,
  the primary comparison column)**, nec_b_tc (diagnostic), pass a / b. The keep is position-collapsed (as for the
  baselines in `score_arena_a`). Ge recovery: the unit's nodes at their firing positions, as `ge_recovery_tcz.py` jobs.
  Dunefsky coverage is not defined for a WCM unit: it is read from the error nodes of their own graph search.
- **Prompt sets.** `train` = the K = 8 baseline prompts (`prompts_for`; WCM was fitted on them). `held` = the K
  highest-peak rows of the held-out strongest eval set (the first 16 `rec["strong"]["held"]` rows), which WCM never
  saw. `--emit-heldout-prompts DIR` writes `DIR/prompts.jsonl` in `make_targets`' schema. `run_ge.py` and
  `run_dunefsky.py` read it through `--targets DIR` with no change. `--ge-held` / `--dunefsky-held` score those
  circuits on the same rows (prompt asserted); this path is written but not yet run (no held-out baseline circuits exist).
- **Curves.** `curves.jsonl` / `.csv` hold one point per unit, plus the baselines' home rows from `score_arena_a`
  (`--baseline-rows`, every knob point). Comparisons are curves (each method's own sweep against size), not matched
  points.

**CPU check (2026-11-05, 4 threads, peak RSS 4.8 GB; Ge recovery in ge107, 2 threads, 5.6 GB):** 1.z.21179 (35
members) and 2.z.1360 (690, passed) × 2 train + 2 held prompts: 68 units, ~1.6 s per unit on CPU, Ge recovery ~0.8 s
per job.
- Keep-all restriction (every member × position through `restrict()`) against `score_arena_a`'s home scope on
  prompt q: **max |Δ| 0.0** on 4 of 4 (a_pos, e0, eM, eN, all raw reads, Z..nec_b_tc).
- Firing sizes = `fire_count.json` on the same rows, 8 of 8. Clean target read = record peak.
- Ge recovery of `wcm_fire` = `wcm_full` exactly: non-firing members carry no leaf flow. All-kept leaf-sum gap ≤ 2.5e-6.

| target | set | full n | fire n (lat,pos) | full Z / A / C / nec b | fire Z / A / C / nec b | pass b full / fire | rec interp / total |
|---|---|---|---|---|---|---|---|
| 1.z.21179 | train q0, q1 | 35 | 25 (86), 23 (73) | 1.06/0.97/0.95/1.06; 0.94/0.95/0.93/1.00 | 1.06/0.94/0.91/1.06; 0.96/0.96/0.95/1.00 | T T / T T | 18.0 / 0.09; 19.6 / −0.04 |
| 1.z.21179 | held q0, q1 | 35 | 28 (86), 25 (76) | 0.93/0.91/0.89/1.00; 0.81/0.72/0.70/1.00 | 0.93/0.89/0.88/1.00; 0.82/0.71/0.69/1.00 | T F / T F | 16.1 / −0.06; 16.5 / −0.18 |
| 2.z.1360 | train q0, q1 | 690 | 163 (498), 166 (517) | 1.20/1.05/1.06/11.6; 1.37/1.18/1.17/3.04 | 1.06/0.97/0.96/11.6; 1.36/1.11/1.09/3.04 | T T / T T | 20.3 / 0.22; 17.2 / −0.22 |
| 2.z.1360 | held q0, q1 | 690 | 155 (483), 180 (602) | 0.26/0.32/0.46/2.92; 0.52/0.37/0.41/2.13 | 0.19/0.29/0.41/2.92; 0.91/0.51/0.53/2.13 | F F / F F | 16.5 / −0.41; 9.6 / −0.90 |

Reading (2 targets, 8 prompts, not results):
- The firing part (~25–70 % of members at L1–L2) scores like the whole circuit on every prompt. The pass under nec b
  is the same for `wcm_fire` and `wcm_full` on 8 of 8 prompts.
- Top-k: on 2.z.1360 train q1, 16 latents (35 (lat, pos)) still pass b. On the other prompts the pass is lost at
  k ≤ 8–64.
- Per-prompt nec b reaches 3–12 on 2.z.1360, because one prompt's e0 lies close to its a_pos. Single-row ratios are
  noisy, so medians over prompts should be read with that in mind.
- **Baselines here are different targets** (smoke: 3.tc.13057, 7.z2.18144, 9.tc.17800; home rows from
  `out/score_smoke/`). Medians over every knob point: Ge hier n 66, Z / A / C 0.27 / 1.13 / 1.09, nec b 0.97, pass b
  31 of 120 non-vacuous; post-hoc n 78, Z 0.03, pass b 13; Dunefsky n 9, Z / A / C 1.06 / 1.11 / 1.01, nec b 1.30,
  pass b 4 of 28. WCM firing part on train prompts: n 94 (median), Z / A / C 1.06 / 0.96 / 0.96, nec b 2.05, pass b
  4 of 4. Not a comparison until the same targets run.

**Pod cost (48 targets; at £2.65 / GPU-h, plan §5):**
- **WCM side.** 48 × 16 prompts × (5 λ × 2 + ~9 top-k in one arm) ≈ 14.6k one-row units. That is ≈ 2–4 GPU-h at
  0.5–1 s per unit, an estimate: on CPU it is 1.6 s at L1–2, and deeper targets have more sites. Plus 14.6k Ge recovery
  jobs, ≈ 0.5 GPU-h, or ≈ 1 h wall on 8 CPU processes. **≈ 2.5–4.5 GPU-h ≈ £7–12.**
- **Held-out baselines** (optional, a decision). Ge on 48 × 8 held prompts ≈ 0.6 GPU-h. Dunefsky 24 × 8 × 7 = 1,344
  circuits ≈ 5–6 GPU-h. Scoring them per prompt at `--tau-stride 2` is ≈ 12.9k units ≈ 2–3.5 GPU-h. **≈ 8–10 GPU-h ≈
  £21–27.**
- **Both ≈ 10.5–14.5 GPU-h ≈ £28–38.** The Dunefsky held-out units need an a2 process (tc only), like
  `score_arena_a`. The WCM a1 units cannot be scored in a2.

**Open:** (1) whether to run the held-out baselines (≈ £21–27); (2) whether to add a Dunefsky-style coverage for WCM
units (it would need a new attribution pass, so it is left out); (3) a position-specific keep (members only at their
firing positions) would need a new patcher, so the keep stays position-collapsed, as for the baselines; (4) the top-k
score uses the max code over positions (sum or anchor are the alternatives).

  wsl, repo root: `PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= ARENA_A_SCOPE=a1 .venv/bin/python -u experiments/107-true-baselines/arena_a/score_perprompt.py --out <OUT> --wcm l1e-3=<wcm out>[,...] [--keys ...] [--n-prompts 8] [--check]`;
  then `ge_recovery_tcz.py --jobs <OUT>/ge_jobs.pp.jsonl --out <OUT>/ge_recovery.pp.jsonl` (ge107); then
  `score_perprompt.py --summary --out <OUT> --baseline-rows <score_arena_a OUT>/rows.jsonl`.

## A8 2026-11-05 (agent `a8-scorer`): `score_arena_a.py`, the arena A scorer

Scores Ge (hierarchical + post-hoc), Dunefsky and WCM circuits under our tests, plus Ge's leaf-sum recovery and
Dunefsky's coverage. Nothing under `src/` and no existing script changed (it imports `diag_necessity_tcz.Diag` / `run`
and the src scorer pieces). Tests: `test_score_arena_a.py` (8 pass, CPU, no model). Smoke output: `out/score_smoke/`
(`rows.jsonl`, `selftest.jsonl`, `summary.md`, `summary.json`, logs). One CPU debug run (1.z.21179) plus the smoke on
the local GPU 11:04–11:21 (coordinator-approved, one process, fp32, TF32 off): ~1–3 s per unit, 371 + 28 units.

**Definitions (all ours unless noted):**
- **Units / curves.** One unit = one node set: Ge `hier[t]` and post-hoc `S(τ)` (pool attr > τ, strict) at each of the
  30 τ; Dunefsky one file per `num_branches`; WCM one circuit per λ arm. Every point of every sweep is scored; no
  matched-size points. Identical node sets on the same prompt are scored once (flag `reused`).
- **Node mapping [settled].** Ge / Dunefsky positions are BOS-prefixed: pos-0 nodes dropped (`n_bos`), the rest
  shifted to pos − 1. Kissane latents ≥ 24,576 at L5 / L7 → `z2` (latent − 24,576). Nodes at sites not upstream of the
  target are dropped (`n_offsite`; they cannot change the root). Dunefsky: graph rows of type `mlp` with a latent, root
  excluded, and the root's own (layer, latent) at an earlier token excluded (as their read-out). Each row records the
  (latent, pos) count and the latent-level count; the tests use latent-level (position-collapsed) membership.
- **Coefficients.** Baseline members are kept at α = 1 (live value); WCM at its fitted α (the production scorer).
- **Universe.** Ge + WCM in scope a1 (z + z2 + tc). Dunefsky in scope a2 (tc only, attention live): their node universe.
  One scope per process. For the same target, e0 differs between scopes (3.tc.13057: 1.13 in a1, 1.86 in a2).
- **Eval sets.** `ho` = the target's 16 held-out strongest contexts (the production eval set, primary). `home` = the
  circuit's own prompt: a Ge / Dunefsky circuit on its prompt j, WCM on all K = 8 prompts. Each is the record's strong
  row j read at its anchor (by causality, the same as the truncated prompt; asserted against the runners' tokens).
  Baseline reads (a_pos, e0, eM, eN) are taken on the same rows. The panel targets have no `prompts.jsonl`, so their K
  prompts come from a copy of `make_targets.prompts_for`.
- **Z / A / C / nec_a.** `free0` / `freeM_topk` / `freeN_topk` / `phi_sup_blind`. Same src patchers and fills
  (`TapCircuitOnly`, `TapIntervention`, `collect_site_anchors`, `collect_site_means`) and the same arguments as
  `eval.wcm_eval.score_circuit`, with the target-level work shared. nec_a: transcoder / SAE error live.
- **nec_b [settled].** Members zeroed, non-members live, every upstream site's error (tc and z) held at its clean value:
  `Diag(mem="zero", freeze_err={"z","tc"})`. Then (clean − x) / (clean − e0), with e0 the production (live-error) empty
  circuit, as in `necdiag_summary.md` ("g err"). `nec_b_tc` freezes only the tc error ("tc err").
- **Pass.** Z, A, C in [0.75, 1.5] and nec ≥ 0.9, under nec_a and under nec_b. `vacuous` = `wcm_eval.vacuous_v2`
  (§12.4, per eval set): the pass is None there. Under v2, 2.tc.2019 is vacuous (eN within 5% of a_pos); the
  night-log table, which uses the v1 rule, lists it as a fail.
- **Ge recovery [settled].** Three columns from the runner's in-pass leaf parts (`hier[t].parts`; the A11 re-run
  `posthoc[t].parts`): `interp` = embed / a_t (the paper's, primary), `nonerr`, `total` = root of G′ / a_t (bounded
  companion), plus the bias and error shares.
- **Coverage.** 1 − root prune_error / root (dunefsky.md 7.3(a)). It is not bounded by 1 here: the root's bias_error
  is large and negative (3.tc.13057 b10: −5.48 on a root of 6.97; prune_error −2.84; coverage 1.41).

**Checks:**
- Keep-all circuit (every latent at every real upstream site, α = 1): Z = 1 within 1.9e-6 on all 16 (target, scope)
  rows (A and C reads equal a_pos to the same tolerance). The two vacuous targets have no Z, but their reads still
  equal a_pos.
- Empty circuit: free0 = e0 exactly, and e0 = the stored panel `e0_tk` on all 11 WCM targets.
- Regression: column (a) against the 11 stored panel rows (Z, A, C, nec), max |Δ| **0.0** on the GPU, 4e-6 in the CPU
  debug run (criterion 1e-4).
- nec_b and nec_b_tc reproduce the diagnosis' `g_err` / `f_tc_err` to 4 decimals on every panel circuit.

**Smoke medians (non-vacuous units; pass counts are units):**

| method | units (vacuous) | n latent | ho Z / A / C | ho nec a / b | ho pass a / b | home Z / A / C | home nec a / b | home pass a / b | own metric |
|---|---|---|---|---|---|---|---|---|---|
| Ge hier | 180 (60) | 66 | 0.02 / 0.03 / 0.18 | 0.15 / 0.34 | 0 / 0 | 0.27 / 1.13 / 1.09 | 0.22 / 0.97 | 0 / 31 | rec interp 5.32, total 1.00 |
| Ge post-hoc | 180 (60) | 78 | 0.02 / 0.03 / 0.11 | 0.13 / 0.35 | 0 / 0 | 0.03 / 1.12 / 0.82 | 0.22 / 0.96 | 0 / 13 | rec interp 4.76, total 0.68 |
| Dunefsky | 28 (0) | 9 | 0.32 / 0.28 / 0.22 | 0.23 / 0.32 | 8 / 8 | 1.06 / 1.11 / 1.01 | 0.60 / 1.30 | 3 / 4 | coverage 1.40 |
| WCM λ 1e-3 | 11 (2) | 1906 | 0.82 / 0.98 / 0.90 | 0.69 / 1.00 | 3 / 4 | 0.99 / 0.98 / 0.97 | 0.82 / 1.00 | 3 / 5 | – |

Reading (smoke, 2–3 targets per baseline, so not results):
- On held-out, the per-prompt baseline circuits do not carry the target: Ge's Z is ≈ 0.02. Dunefsky passes only on
  3.tc.13057 (8 of its 14 units); on 9.tc.17800, Z ≤ 0.07.
- On home turf, Ge passes only under nec_b (hierarchical 31, post-hoc 13 units). nec_a stays low: the live
  transcoder error, as in the diagnosis.
- 7.z2.18144 is vacuous on both eval sets (e0 > a_pos), like 6.z.8399.

**Added 2026-11-05 11:25–11:35 (coordinator: code + short tests only; the smoke is NOT rescored with these yet):**
- **R, the donor test** (`--donor`; `--donor-variants fit` by default, `--donor-draws 8`, `--donor-pool`,
  `--donor-records`). It is src `eval.donor_test.donor_target`, unchanged, run once per target on that target's
  distinct (members, α) sets. Each set becomes a minimal circuit object (`fake_circuit`: a seed node plus one node per
  member with its amplitude). The prompt does not enter the donor identity: the test runs on the 16 + 16 held-out
  bases only.
  - The R pool comes from `load_or_build_pool` (073's random pool, exact / near filtered against `--donor-records`,
    default `out/targets_smoke/ctx`). In the smoke, the panel targets' records are not in that filter. On the pod,
    pass the single records dir.
  - Output: `donor.jsonl` (`R_A`, `R_C`, `R_R`, plus the full donor row), `donor_raw.jsonl`, `R_pass_a/b` (the pass
    under nec a / b AND donor recovery A and C in band, as `donor_test.resample_pass`). Resumable per unit; a target
    that raises writes error rows, and those are retried.
- **Ge recovery for WCM and Dunefsky circuits.** `score_arena_a.py --emit-ge-jobs` (.venv, no model) writes
  `ge_jobs.<scope>.jsonl`: one job per (unit, prompt), with WCM on each of the K = 8 prompts. The new
  `ge_recovery_tcz.py` (ge107) runs Ge's A11 fixed-mask pass on each job (`ge_core.ungated_pass(masks=...)`, LN
  frozen, their splice unmodified; the root always kept). It writes `ge_recovery.jsonl` with interp / nonerr / total
  plus the bias and error shares. `--summary` averages these per unit over the unit's prompts.
  - Masks [ours]: Dunefsky nodes at their graph positions. Their graph has no z nodes, so attention reaches the root
    only through the detached z error leaves. WCM members at every non-BOS position.
  - `--check` also runs the all-kept mask, which must give leaf sum = a_t.
  - Smoke jobs emitted: 88 (11 WCM × 8 prompts) + 28 (Dunefsky), 8.9 MB. One job each, CPU, 2 threads, to a scratch
    file: Dunefsky 3.tc.13057 p0 b10 interp 0.41 / nonerr −1.12 / total −0.25; WCM 1.z.21179 p0 interp 17.97 /
    nonerr −1.67 / total 0.09. The all-kept leaf-sum gap was 9.4e-7 / 7.1e-7.
- **Coverage (ours, bias-aware)**, next to theirs: 1 − root prune_error / (root − root bias_error), the coverage of
  the part of the root that its encoder bias does not supply. It is not bounded either (prune_error can be < 0).
  Example: 3.tc.13057 b10 gives 1 + 2.84 / 12.45 = 1.23, against 1.41 for theirs. Older rows get it recomputed from
  the Dunefsky file at `--summary`.
- **Necessity (b)**: both variants stay columns (all errors frozen = `nec_b`; tc error only = `nec_b_tc`).
- **Tests:** `test_score_arena_a.py` 12 pass (adds coverage-ours, from/to inverse, job nodes, fake circuit vs
  `circuit_alphas` / `keep_and_scales`, donor identity); `test_ge_recovery_tcz.py` 2 pass in ge107 (node keys, masks
  incl. the root and the all-position rows).
- **To run (after the go):** score with `--donor` per scope; `--emit-ge-jobs` per scope; `ge_recovery_tcz.py --jobs
  <OUT>/ge_jobs.a1.jsonl --jobs <OUT>/ge_jobs.a2.jsonl --out <OUT>/ge_recovery.jsonl --check` (ge107); then
  `--summary`.

**Run 2026-11-05 12:57–13:57 (GPU donor test, then the companion on the CPU):**
- **First donor attempt.** It spilled into shared GPU memory, and the coordinator killed it at 12:53. Cause: src
  `donor_test.capture_codes` stores each code with k = d_sae values plus int64 indices. That is ~2.5 GB per site for
  133 donor rows, × 24 sites.
- **Fix [ours, exact; src unchanged].** `capture_codes_compact` is installed over `donor_test.capture_codes`. It
  keeps only the non-zero entries, padded with value 0. This is exact for `ResamplePatcher`: it uses only entries > 0,
  all of which fit budget K = d_sae, and `scatter_add` is order-free. Unit test: `test_compress_codes_exact`.
- **Memory.** Peak allocated 4.9 GB; `nvidia-smi` peak 7,018 MiB (logged every 30 s in `gpu_mem_donor.csv`, abort
  set at 15,300); `empty_cache` after every target.
- **R pool.** The smoke corpus is small: filtered against the 48 smoke records, 28 of 512 survive, below src's
  minimum of 32. So the pool is filtered against the run's 14 target records instead (`--donor-records run`), which
  keeps 133. The a2 run reuses the same file.
- **Output.** `donor.jsonl`: 399 rows, 0 errors (81 / 109 / 105 distinct node sets for the 3 Ge targets).
- **Companion.** `ge_recovery.jsonl`: 116 of 116 jobs ok, CPU, 3 min. The all-kept leaf-sum gap is ≤ 4.9e-5 on
  every job.
- **Medians, held-out, R recovery (A / C / R pools), and the R-pass under nec a / b:**
  - Ge hier: 0.02 / 0.01 / 0.02, R-pass 0 / 0
  - Ge post-hoc: 0.02 / 0.01 / 0.01, R-pass 0 / 0
  - Dunefsky: 0.02 / 0.00 / 0.02, R-pass 0 / 0
  - WCM: 0.57 / 0.43 / 0.37, R-pass 0 / 1
- **Ge recovery, medians (interp / nonerr / total):** Ge hier 5.32 / −0.49 / 1.00; post-hoc 4.76 / −0.45 / 0.68;
  Dunefsky 0.27 / −0.97 / −0.16; WCM 12.48 / −2.62 / −0.91.

**Still open:** Dunefsky's position-specific keep (dunefsky.md sensitivity) and the greater-than full-replacement
read (7.3(b)) are not computed.

  wsl, repo root (one scope per process; resumable; `--dry-run` lists units; `--summary` is CPU-only):
  `PYTHONPATH=src HF_HUB_OFFLINE=1 ARENA_A_SCOPE=a1 .venv/bin/python -u experiments/107-true-baselines/arena_a/score_arena_a.py --scope a1 --out <OUT> --selftest --ge <ge out> --wcm l1e-3=<wcm out>`;
  the same with `--scope a2 --dunefsky <dunefsky out>` and `ARENA_A_SCOPE=a2`; then `--summary --out <OUT>`.

## Log 2026-11-05 night (agent `arenaA3`; local GPU 01:35–03:31, then CPU): v4 panel + necessity diagnosis

**What ran** (one GPU process at a time; `out/panel/GPU_DONE` written 03:31 when the last one exited):
- `make_panel_targets.py` (new): 12 targets, 2 tc + 2 z per band L1–3 / L4–7 / L8–11, seed 20261105, the pilot's rule
  (full top store, ≥ 48 / 64 mid, ≥ 8 held-out strongest, ≥ 32 contrast), pilot targets excluded, pilot data pass
  reused. `run_panel_a.sh`: v4, λ 1e-3, 400 steps, the pilot's env. Output `out/panel/wcm_l1e-3/`.
- 3 of the 12 fits died at 02:07 (card at 15.96 GB: `CUDA driver error: device not ready`, then allocator asserts).
  Refit one process each (`run_necdiag_a.sh`, `out/panel/wcm_l1e-3_rerun/`): **9.z.19003 and 9.z.17708 ok;
  11.tc.22949 fails again in a fresh process, at 15.9 GB.** L11 tc does not fit on the 16 GB card even at DEEP_BS=1
  (it has the most upstream sites). On an H100 this is not a problem. So the panel has 11 of 12 targets.
- `diag_necessity_tcz.py` (new, forward only) on the 17 base circuits (pilot λ 1e-3 + 4e-3, panel) and 3 arm
  circuits. It reproduces the scorer: a_pos, e0, the necessity read and the free0 read all match the eval rows to
  ≤ 1e-5 (column `chk` in `results_v2.jsonl`). `necdiag_summary.py` (new, CPU) builds the tables below
  (`out/panel/necdiag_summary.md`).
- Three diagnostic arms on the 3 pilot targets. None of them changes a setting: **NEC_W=1** (the existing
  learned-mask necessity term) ok. **λ 2.5e-4** and **STEPS=1200** were killed by the host OOM killer (WSL 30 GB;
  20–23 GB RSS each, on 9.tc.9824, its first target), so they give no result.

### Pass table (held-out strongest; pass = Z, A, C in [0.75, 1.5] and necessity ≥ 0.9)

| fit | target | Z | A | C | nec | n | fit s | pass |
|---|---|---|---|---|---|---|---|---|
| panel | 1.z.21179 | 1.01 | 0.98 | 0.97 | 0.69 | 35 | 80 | fail (nec) |
| panel | 2.tc.17681 | 0.76 | 1.02 | 0.99 | 0.65 | 585 | 120 | fail (nec) |
| panel | 2.tc.2019 | 0.74 | 2.95 | 0.96 | 0.56 | 469 | 105 | fail |
| panel | 2.z.1360 | 0.81 | 0.85 | 0.90 | 1.04 | 690 | 91 | **PASS** |
| panel | 4.tc.13440 | 1.02 | 1.14 | 1.12 | 1.00 | 3000 | 173 | **PASS** |
| panel | 4.z.22556 | 0.77 | 0.74 | 0.76 | 0.67 | 436 | 158 | fail |
| panel | 5.tc.22587 | 0.99 | 1.01 | 0.78 | 0.44 | 1906 | 230 | fail (nec) |
| panel | 6.z.8399 | – | 1.56 | – | – | 1050 | 230 | fail: e0 > a_pos (e0 / clean 1.14): vacuous target |
| panel | 8.tc.15982 | 1.00 | 1.08 | 1.01 | 1.00 | 5334 | 344 | **PASS** |
| panel (rerun) | 9.z.17708 | 0.82 | 0.83 | 0.80 | 0.75 | 2587 | 389 | fail (nec) |
| panel (rerun) | 9.z.19003 | 0.35 | 0.77 | 0.67 | 0.57 | 2040 | 398 | fail |
| panel | 11.tc.22949 | – | – | – | – | – | – | no fit (16 GB card OOM, twice) |
| pilot λ1e-3 | 3.tc.3752 / 8.z.10122 / 9.tc.9824 | 1.25 / 0.93 / 0.71 | 1.79 / 1.02 / 0.85 | 1.70 / 1.00 / 0.79 | 0.32 / 0.60 / 0.32 | 740 / 2334 / 3372 | 138 / 319 / 412 | 0 / 3 |
| arm NEC_W=1 (λ1e-3) | 3.tc.3752 / 8.z.10122 / 9.tc.9824 | 1.17 / 0.83 / 0.57 | 1.54 / 0.93 / 0.80 | 1.48 / 0.90 / 0.77 | **0.95** / 0.74 / 0.63 | 1062 / 3167 / 3626 | 143 / 344 / 532 | 0 / 3 |

**Panel: 3 / 11 pass** (pilot 0 / 6). Of the 8 fails, 4 fail on necessity alone (Z, A and C all in band: 1.z.21179,
2.tc.17681, 5.tc.22587, 9.z.17708), and 1 is a vacuous target. Panel fit time: 80–400 s (0.2–1.0 s / step). Shallow targets take ~1.5–2 min and L8–9 targets
6–7 min. Card memory peaked at 15.9 GB on the deep targets.

### Necessity diagnosis (17 base circuits; the median row leaves out the vacuous 6.z.8399)

What is measured: the target's post-activation read, at the held-out strongest anchors, as the scorer reads it.
At every upstream site the stream is split into members + non-members + bias + error.

- **The error.** At a z site the error is the Kissane SAE error. At a tc site it is `mlp(x_in) − transcoder(x_in)`,
  the part of the real MLP's output that the transcoder does not reconstruct. It is live, so it is recomputed from
  the edited stream.
- **Frozen.** The clean run's value at the same position.
- **Necessity-like fractions** = (clean − x) / (clean − e0), with the members zeroed.

| reading | median | what it is |
|---|---|---|
| e0 / clean | 0.10 | everything zeroed except bias + live error |
| full, every upstream error = 0, / clean | **0.03** | members and non-members live, errors removed: the reconstruction-only model barely activates the target |
| full, every error = its mean, / clean | 0.09 | the same with mean errors |
| (b) necessity, scorer (members 0, rest live) | **0.58** | = phi_sup_blind_tk (reproduced) |
| read surviving removal: zero fill / mean fill of members | 0.54 / 0.53 | how the members are filled hardly matters |
| (e) members 0, non-members at their mean, error live | 0.98 | |
| (f) z sites frozen (non-members + error) | 0.55 | ≈ (b): the z sites do not re-supply the target |
| tc sites frozen (non-members + error) | 0.98 | |
| **tc error only frozen** (tc non-members live) | **1.00** | |
| tc non-members only frozen (tc error live) | 0.72 | |
| z error only frozen | 0.60 | ≈ (b) |
| all errors frozen / all non-members frozen / both | 0.91 / 0.77 / 0.77 | |
| necessity on the 48 train rows | 0.70 | low where the fit was trained too |

Per circuit, the tc-error-frozen reading versus the scorer's: 3.tc.3752 0.88 vs 0.32; 9.tc.9824 0.94 vs 0.32;
5.tc.22587 0.86 vs 0.44; 2.tc.17681 0.62 vs 0.65 (the only one with no gain). Full per-circuit table:
`out/panel/necdiag_summary.md`.

Upstream sites (medians over circuit × site, held-out rows, clean):

| site | L0 / position | members | members active / position | members' share of the decoded norm at the anchor | FVU (centred) |
|---|---|---|---|---|---|
| tc L1–3 / L4–7 / L8–11 | 18 / 49 / 70 | 174 / 161 / 85 | 2.9 / 4.5 / 4.5 | 0.43 / 0.26 / 0.22 | 0.44 / 0.41 / 0.34 |
| z L1–3 / L4–7 / L8–11 | 18 / 16 / 18 | 72 / 32 / 27 | 1.0 / 0.5 / 1.1 | 0.33 / 0.14 / 0.18 | 0.34 / 0.54 / 0.54 |

**Conclusion: mainly (i), and it is specific to transcoders.** What survives member removal is carried by the live
transcoder error, i.e. the real MLP recomputing from the edited stream what the transcoder does not model.
- Evidence for (i):
  - Hold only the tc error at its clean value and removing the members takes the target to e0: median necessity
    1.00, against 0.58 for the scorer's reading.
  - The z (SAE) error does not do this: freezing it gives 0.60.
  - The errors also matter in the clean run: with every error zeroed the target reads 3% of clean.
  - The transcoders' FVU is 0.34–0.44, so the MLP residual is large.
  - On a transcoder substrate the MLP stays live under every edit, so "error live" means a live path through the MLP
    that no latent node represents.
- (ii), redundancy among non-member latents, is a smaller part. Freezing all non-members raises necessity only to
  0.77, and freezing the tc non-members to 0.72. The members carry a minority of each site's decoded norm (0.14–0.43).
  The mean-fill reading (e) shows that with non-members held at their mean the members are necessary (0.98). But that
  stream is off-distribution, and the live tc error responds once the non-members are live.
- (iii), an under-trained fit, is not supported:
  - Over the last 100 of 400 steps membership changes by a median of ~1% (5.4% at most, on the 35-node 1.z.21179),
    while the loss still falls 10–15%, mostly the lr annealing.
  - Train-row necessity is low too (0.70), so the gap is not a generalisation gap.
  - With the necessity term on (NEC_W=1), the same 400 steps reach train necessity 0.93–1.01 and held-out 0.95 / 0.74
    / 0.63. The optimiser reaches necessary circuits within budget when the objective asks for it. v4's objective does
    not, and plain WCM finds circuits that are sufficient but not necessary.
  - The 1200-step arm was OOM-killed on the host, so "more steps" itself is untested.
- λ: at λ 4e-3 (smaller circuits) necessity is lower than at 1e-3 on all 3 pilot targets, as expected from coverage.

### Proposals before Saturday (propose only; v4, BOS hidden, error live and float32 are unchanged)

1. **Report a second necessity reading, alongside the settled one, with the transcoder error held at its clean value
   during member removal** (`f_tc_err` here: the tc error is treated as an error node fixed at its clean value).
   - This is the error-node convention of Dunefsky's graphs (R3: error nodes computed on the clean run). It is the
     fair comparison for A2 / A1.
   - SFC-style error ablation to zero or mean is not usable here: the clean read itself collapses (3% / 9% of
     clean), so the denominators vanish.
   - Cost: one forward per circuit. `diag_necessity_tcz.py` already computes it; for the pod it would be a scorer
     column (needs a src change or an arena-side rescore).
2. **State it as a substrate property in the paper**: on transcoder substrates the live-error necessity measures the
   MLP's untracked computation, not the circuit. Keep the settled pass rule as the headline. Report the pass count
   under both readings.
3. NEC_W=1 is **not** ready as a fix: 0 / 3 pass (A 1.54 on 3.tc; Z 0.57 on 9.tc). Only consider it with its own
   pilot.
4. More steps: no evidence that they help (membership barely moves after step ~300). Untested here because of host OOM.
5. Pod ops:
   - L11 tc targets need more than 16 GB: fine on an H100, but no local L11 tc smoke is possible.
   - Host RAM: one process at λ 2.5e-4 or 1200 steps reached 20–23 GB RSS on 9.tc.9824 (400-step λ 1e-3 fits run
     within it). With 8 processes per pod node, check that host RAM is ≥ 8 × 25 GB before running a low-λ sweep.

**Files (new):** `make_panel_targets.py`, `run_panel_a.sh`, `run_necdiag_a.sh`, `diag_necessity_tcz.py`,
`necdiag_summary.py`. **Outputs:** `out/panel/` (`targets_panel.txt`, `rec/`, `wcm_l1e-3/`, `wcm_l1e-3_rerun/`,
`arm_necw1/`, `diag_nec/results_v2.jsonl`, plus `results.jsonl`, the first pass without the split freezes,
`necdiag_summary.md`, `logs/`, `gpu_mem.csv`, `steps.tsv`, `GPU_DONE`). `out/panel/diag_cputest/` holds the CPU
checks of the diagnostic.

## Log 2026-11-03 (late evening; CPU only)

**Downloaded** (Daniel's approval, ~22:00; `fetch_assets.sh local` + `owt`, pinned revisions; `fetch_assets.sh verify`
against the HF API: 0 missing / mismatched):
- D1 Kissane z-SAEs @ `97e0da0`: 24 files, 2,115,394,805 B (2.12 GB);
- D5 OWT `train-00000-of-00073.parquet` @ `f02886b5`: 242,887,938 B;
- D2–D4 through `setup_env.sh` (CPU torch 2.3.1 190 MB, 2.2.0 187 MB, plus the pinned PyPI wheels).

**Environments** (`TORCH_INDEX=cpu`): `~/venvs/ge107` torch 2.3.1+cpu, TL = the vendored fork, transformers 4.43.1;
`~/venvs/tc107` torch 2.2.0+cpu, transformer-lens 1.11.0, transformers 4.35.2, datasets 2.12.0, hub 0.17.3, wandb
0.16.0. **`pip check`: "No broken requirements found" in both. No pin had to change** (freezes:
`env_freeze_ge.txt`, `env_freeze_tc.txt`). One script fix: the import checks used `python -I`, which ignores
`PYTHONDONTWRITEBYTECODE`, so 17 `__pycache__` folders landed in the two clones; they were removed (agent-gate) and the
checks now run with `-I -B`. Both clones are clean (`git status --ignored`).

**Kissane file format (verified):** each `.pt` is a raw `OrderedDict` `{W_enc [768, d], W_dec [d, 768], b_enc [d],
b_dec [768]}`, float32, saved from `cuda:0` (loads with `weights_only=True, map_location="cpu"`); d = 24,576, and
49,152 at L5 / L7; decoder rows unit-norm ± 0.6%; cfg `act_name blocks.L.attn.hook_z`, `concat_heads`, seq_len 128.
Forward as inferred, `relu((x − b_dec) W_enc + b_enc)`, `f W_dec + b_dec`, **plus one convention not inferred before:
the input is TL's `hook_z` under `fold_value_biases=True`, i.e. without b_V** (centred FVU 0.20–0.49, L0 3–21; with
b_V kept FVU up to 441). Ge's pipeline loads GPT-2 with TL defaults, so it matches; WCM's HF adapter subtracts b_V
(`WCM_ADAPTER.md` section 1). Transcoder pickles: `cuda:0` tensors, cfg L1 8e-5, lr 4e-4, 300M OWT tokens, context 128.

**Fidelity checks (CPU):**

| check | result |
|---|---|
| G0: Ge's own `tests/intergration/test_attributor.py` (ge107) | **3 / 3 pass** |
| conversion parity: lm_saes `SparseAutoEncoder` vs source formulas, real OWT activations | z: max \|Δf\| ≤ 3.8e-6, \|Δx̂\| ≤ 1.7e-6; tc: \|Δf\| ≤ 1.3e-4 (L11; activations up to ~10², so ≤ 1e-6 relative), \|Δout\| ≤ 2.4e-4 |
| G1: Theorem 3.1, gate off, IOI prompt, roots tc8.218, z9.11368, logit " Mary" | **LN frozen: leaf sum = root, relative gap 4.7e-6 / 1.1e-6 / 6.2e-6 (pass ≤ 1e-4).** LN as shipped (scale live): gap 9.1 / 4.0 / 2.0 (×100%), so their code as shipped is **not** linear, confirming ge.md A3: the LN-frozen row is the faithful reading of their paper |
| G3(b): set-separating toy through their `HierachicalAttributor` | **pass**: τ = 2 hierarchical {D: 3}, post-hoc {A: 4, D: 3}; τ = 0.5 drops E (−1); ungated values exact |
| D-F1 (R1) walkthrough paths (tc107, their code unmodified) | **pass**: `print_all_paths` text identical to the saved cell 60; root 7.570007 (saved 7.570009) |
| D-F2 (R2) graph | **pass**: 33 edges and 24 nodes, max rel. 3.1e-6; `keep_latents` = the 11 of dunefsky.md §8.2 |
| D-F2 (R3) error nodes | **pass**: all 53 nodes (24 + 29 error nodes), max rel. 2.2e-4 (on a −0.0011 value) |
| D-F3 (R4, R5) greater-than paths | **pass**: both `print_all_paths` texts identical |
| WCM adapter tests (`test_wcm_tcz.py`, .venv) | **3 / 3 pass**: bitwise splice, edit semantics, parity (`WCM_ADAPTER.md` section 2) |

Notes: G1 runs one plain backward through their splice (`apply_sae`), their `detach_hook` on the attention scores and
their `cache_nodes` forward, not through `HierachicalAttributor.attribute(threshold=-inf)`: that call does the same
backward, but its read-out enumerates every (position, latent) entry of 24 dictionaries into networkx (~8M nodes; the
first attempt was stopped after 10 min). The leaf sum is dominated by cancelling terms (logit root: embeddings +1,363,
dictionary biases −1,715, errors +226), so Ge's interpretable-leaf recovery (A5, A11) is very sensitive to how bias
leaves are counted; report the bias share as planned. Dunefsky R1–R5 took 100 s, 3.8 GB RSS on the CPU; G1 29 s, 5.6 GB.

**Next:** G2, G4 (Fig. 2b on 20 IOI prompts), D-F4 (R6 greater-than metric), a 6-step engine fit and the Z-empty panel
on the CPU; then the GPU timing pilot on the first free night.

## 1. Download inventory (approved 2026-11-03 ~22:00 and fetched; see the log above)

### 1.1 Already on disk (WSL HF cache `/home/daniel/.cache/huggingface/hub`, plus repo files), so nothing is fetched twice

| what | where | revision | size |
|---|---|---|---|
| GPT-2 small (`model.safetensors`, tokenizer, config) | HF cache `models--gpt2` | `607a30d` | 0.55 GB |
| Dunefsky transcoders, all 12 top-level `final_sparse_autoencoder_gpt2-small_blocks.{L}.ln2.hook_normalized_24576.pt` | HF cache `models--pchlenski--gpt2-transcoders` | `4f3ca345` (the spec's pin) | 12 × 151.1 MB = 1.81 GB |
| the same transcoders, LN-folded for HF `fold_ln=False` (circuit-tracer layout, validated 12/12 in 035) | `experiments/035-transcoder-compare/transcoders_ct/layer_{0..11}.safetensors` | from the above | 1.81 GB |
| OpenMOSS code incl. its **vendored TransformerLens fork** (`run_with_ref_cache`, `mount_hooked_modules`) | `third_party/Language-Model-SAEs` | `67bfb0b` (v0.1.0) | in repo |
| Dunefsky code incl. `sae_training/` and `geom_median/` | `third_party/transcoder_circuits` | `8721ff0` | in repo |
| local smoke corpus (wikitext-103, 2,048 × 64 tokens, lead token stored) | `data_public/gpt2-small/` | 069 | 1 MB |
| pip cache: torch 2.8.0+cu128 cp312 and the cu121 nvidia libs (cudnn 8.9.2.26, cublas 12.1.3.1) | `~/.cache/pip` | | (cached) |

Not present: any Kissane attention SAE, any OpenWebText shard (only the apollo dataset's README is cached), the
transcoders' `_log_feature_sparsity.pt` files and `lower_l0/` (both unused, see 1.3).

### 1.2 To download locally (requires Daniel's approval)

Sizes from the HF API (`/api/models/.../tree`) and `curl -sI` on the wheel URLs, 2026-11-03.

| # | source | files | size | why |
|---|---|---|---|---|
| D1 | HF `ckkissane/attn-saes-gpt2-small-all-layers` @ `97e0da0e9182695e9813d15ea32184da83f5c874` | 12 × `gpt2-small_L{L}_Hcat_z_..._v{n}.pt` + 12 × `..._cfg.json` (exact names in `fetch_assets.sh`) | **2,115,396,324 B = 2.12 GB** (L5, L7: 302.2 MB each, 49,152 wide; the other 10: 151.1 MB) | arena A1's attention nodes (Ge + WCM). Site `blocks.L.attn.hook_z`, heads concatenated (768) |
| D2 | `https://download.pytorch.org/whl/cpu/torch-2.3.1%2Bcpu-cp310-cp310-linux_x86_64.whl` | 1 wheel | 190.4 MB | Ge env `ge107` at the authors' own lock (pdm.lock torch 2.3.1), CPU, for the local fidelity checks (§2) |
| D3 | `https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp310-cp310-linux_x86_64.whl` | 1 wheel | 186.7 MB | Dunefsky env `tc107` at their `requirements.txt` pin (torch 2.2.0), CPU, for the local fidelity checks |
| D4 | PyPI, the pinned packages in `setup_env.sh` (transformers 4.43.1 / 4.35.2, tokenizers, datasets 2.19.2 / 2.12.0, pyarrow 16.1.0 / 14.0.2, pandas, numpy 1.26.4 cp310, matplotlib, plotly, wandb 0.17.1 / 0.16.0, jaxtyping, typeguard, networkx 3.3, sentencepiece, pytest, ...) | ≈ 60 wheels across both envs | **≈ 0.2–0.3 GB** (estimate; the largest are pyarrow ≈ 40 MB ×2, plotly 16 MB ×2, numpy 18 MB, transformers 9 MB ×2) | the two pinned envs |
| D5 (optional) | HF dataset `apollo-research/Skylion007-openwebtext-tokenizer-gpt2` @ `f02886b5` | `data/train-00000-of-00073.parquet` | 242.9 MB | an OpenWebText smoke corpus (both dictionary sets were trained on OWT) and Dunefsky's R7 CE check; the wikitext smoke corpus works without it |

**Total required (D1–D4): ≈ 2.7 GB. With D5: ≈ 2.95 GB.** WSL disk has 690 GB free.

Not needed (deliberately not listed for download):
- `pchlenski/gpt2-transcoders` `*_log_feature_sparsity.pt` (12 × 100 KB) and `lower_l0/` (151 MB): no circuit code reads
  them. Their `setup.sh` pattern `*.pt` would fetch both; we point their loader at the cached files instead.
- Ge et al.'s own dictionaries: not public (spec ge.md §0.2). Bloom residual SAEs: not graph nodes (ge.md §2.1, §9).
- No new package for our `.venv` (WCM): the adapter uses the HF stack already there.
- A cp310 CUDA torch for the baselines locally: the local GPU is a Blackwell (sm_120) card, which needs torch ≥ 2.7, so
  their pinned torch cannot use it. Local baseline work is CPU-only by design (§2); GPU timing happens on the pod.

### 1.3 Pod (Saturday): what the pod fetches (it has no HF cache)

| what | size |
|---|---|
| GPT-2 small | 0.55 GB |
| Dunefsky transcoders (12 top-level `.pt` @ `4f3ca345`; `transcoders_ct/` is regenerated by 035's `convert_transcoders.py`, deterministic and asserted, or rsynced: 1.81 GB) | 1.81 GB |
| Kissane attention SAEs (D1) | 2.12 GB |
| OWT corpus: the 082 chain's gpt2-small S1 already fetches apollo files 0-2 (0.73 GB); arena A reuses that corpus | 0 extra |
| `ge107` at torch 2.3.1+cu121 and `tc107` at torch 2.2.0+cu121 (H100 = sm_90, supported) + cu12 nvidia libs: the first env ≈ 2.9 GB, the second ≈ +0.95 GB (pip cache) | ≈ 3.9 GB |

Commands for both are in `fetch_assets.sh` (written, not run). On the pod they are **proposed to Daniel, never run
autonomously** (remote-pod rule).

## 2. Environments

| env | python | key pins | why its own env | used for |
|---|---|---|---|---|
| `.venv` (existing) | 3.12 | torch 2.8 cu128, transformers 5.x | – | WCM (engine, adapter, scorer); GPU locally and on the pod |
| `~/venvs/ge107` (new) | 3.10 (`requires-python = "==3.10.*"`) | torch **2.3.1** + the pdm.lock pins (transformers 4.43.1, hub 0.23.3, numpy 1.26.4, jaxtyping 0.2.29, typeguard 2.13.3, networkx 3.3, ...); `transformer_lens` = **the vendored fork** | Ge's attributor calls fork-only APIs (`run_with_ref_cache`, `mount_hooked_modules`, HookPoint backward hooks in `hooks(fwd, bwd)`); upstream TL lacks them and would shadow the fork | Ge et al. (A1) |
| `~/venvs/tc107` (new) | 3.10 | torch **2.2.0**, transformer_lens **1.11.0**, datasets 2.12.0, einops 0.7.0, hub 0.17.3, wandb 0.16.0, setuptools 67.7.2 (their `requirements.txt`) + era pins we add (transformers 4.35.2, numpy 1.26.4, ...) **[ours, era pins]** | TL 1.11.0 conflicts with the fork; setuptools 67 and datasets 2.12 (dill < 0.3.7) do not import on Python 3.12 | Dunefsky et al. (A2) |

- **No torch deviation.** Unlike arena C (torch 2.8 for the Blackwell card), both baselines run at their exact torch
  pins: CPU wheels locally, cu121 wheels on the pod. The fidelity checks are small enough for the CPU (Ge F1–F4: 20
  IOI prompts of ~16 tokens; Dunefsky R1–R6: ≤ 100 prompts of ≤ 10 tokens). Fallback if the pod needs a newer torch:
  the cached torch 2.8 cp312 wheel, labelled as in arena C.
- **No install into `third_party/`.** `pip install -e` of either clone would write `*.egg-info` into the clone (the
  OpenMOSS `pyproject.toml` has no `[build-system]`, so pip falls back to setuptools). Both are used via `PYTHONPATH`
  instead (`ge_env.sh` / `tc_env.sh`, written by `setup_env.sh` into the env's `bin/`). `importlib.metadata.version
  ("lm-saes")` is only called when saving an SAE (`sae.py:737,740`), which we never do.
- **Their loader needs CUDA as shipped:** the transcoder pickles hold `cuda:0` tensors and
  `SparseAutoencoder.load_from_pretrained` calls `torch.load(path)` without `map_location` (`sparse_autoencoder.py:624`).
  On the CPU our glue builds `SparseAutoencoder(cfg)` itself and loads the pickle's `state_dict` with
  `map_location="cpu"` **[ours, loader only; the algorithm code is untouched]**; on the pod their loader runs as
  shipped. (The pickles also record the training run: L1 8e-5, lr 4e-4, 300M tokens, OWT, context 128.)
- `pip check` runs at the end of each env; `pip freeze` goes to `env_freeze_{ge,tc}.txt`.

## 3. A1: Ge et al., the authors' library code (`HierachicalAttributor`, v0.1.0)

**Entry points (unmodified):**
- `lm_saes.circuit.transformer.hierarchical_attribute_transformer_with_saes(model, saes, input, target, candidates,
  threshold)` (`transformer.py:43-74`): `apply_sae` (`context.py:8-41`; `recon + (x - recon).detach()`), detaches
  `blocks.{i}.attn.hook_attn_scores` for i < 12, then `HierachicalAttributor.attribute` (`attributors.py:127-187`):
  backward hook `where(v * grad > τ, grad, 0)` on every candidate, one `target.backward()`.
- Post-hoc comparator (Fig. 2b): the same call with no gate (τ = −∞) and a cut after the pass (spec §3.6); the A11
  recovery re-run is ours (a fixed-mask backward through their hooks).
- `DirectAttributor` (`attributors.py:76-124`) for the F5–F7 direct-contribution checks.
- Their unit tests `tests/intergration/test_attributor.py` (3 tests, toy graph).

**Dictionaries into their `SparseAutoEncoder`** (in memory, from the cached files; their class unchanged)
**[ours: conversion glue]**:

| dictionary | `hook_point_in` → `hook_point_out` | weights | bias handling |
|---|---|---|---|
| Dunefsky transcoder L | `blocks.L.ln2.hook_normalized` → `blocks.L.hook_mlp_out` (model loaded with TL defaults, `fold_ln=True`, the transcoders' training convention, 035) | `encoder.weight = W_enc.T`, `decoder.weight = W_dec.T` | `encoder.bias = b_enc − b_dec · W_enc` (their input centring folded, exact, as 035); `decoder.bias = b_dec_out`; `apply_decoder_bias_to_pre_encoder=False`; `norm_activation="none"` (factor 1, so ge.md A4 is moot); `feature_act_mask = feature_act_scale = 1` |
| Kissane z-SAE L | `blocks.L.attn.hook_z` → same (the same-hook path, `context.py:18-22`) | `encoder.weight = W_enc.T`, `decoder.weight = W_dec.T` | `decoder.bias = b_dec`, `apply_decoder_bias_to_pre_encoder=True` (Kissane: `relu((x − b_dec) W_enc + b_enc)`, `W_dec f + b_dec`); `norm_activation="none"` |

- **hook_z shape glue [ours]:** `hook_z` is `[batch, pos, 12, 64]` and their `SparseAutoEncoder.forward` acts on the last
  dimension. A subclass `ZSparseAutoEncoder` overrides only `forward` (flatten the head axes, call the parent, reshape
  back); `hook_feature_acts` and the gate are untouched. The z error then passes through W_O, as on WCM's side (ge.md
  §9 G2).
- **Parity tests (CPU):** each converted dictionary vs its source forward (`sae_training.SparseAutoencoder` for the
  transcoders, the Kissane formula for the z-SAEs) on real activations, max |Δf| ≤ 1e-5.

**Candidates:** feature acts only, `Node("blocks.L.hook_mlp_out.sae.hook_feature_acts")` and
`Node("blocks.L.attn.hook_z.sae.hook_feature_acts")`, passed explicitly (their API takes `candidates`). The v0.1.0
default also adds `hook_attn_scores` (QK nodes), which the arena's OV-only scope excludes (ge.md §2.3). No code change.

**LayerNorm (ge.md A3), both rows:** (i) as shipped (LN scale live, their code); (ii) LN scale frozen by adding their own
`detach_hook` at `blocks.L.ln1.hook_scale`, `blocks.L.ln2.hook_scale`, `ln_final.hook_scale` through `model.hooks`
**[ours, labelled]**. Theorem 3.1 (F1) only holds for (ii); for (i) the leaf-sum gap is reported.

**Root:** `Node("blocks.L.<site>.sae.hook_feature_acts", "0.{pos}.{feat}")`: one latent, one prompt (batch 0), one
position (the target's max-activating position on that prompt, truncated after it, BOS at 0) **[ours: which prompts and
which position]**. τ: 30 values, log-spaced in τ / a_t (A7), the same grid for hierarchical and post-hoc.

**Fidelity checks (DAN-192; all before any comparison row):**

| check | what it reproduces | pass |
|---|---|---|
| G0 | their `test_attributor.py` (3 tests) in `ge107` | all pass |
| G1 (ge.md F1) | Theorem 3.1 with the gate off (τ = −∞, not 0: the gate is signed), LN frozen | Σ leaf attributions (features + errors + biases + embeddings) = a_t, \|Δ\| / \|a_t\| ≤ 1e-4, every prompt and root |
| G2 (F2) | Theorem 3.1 on a gated G' | the linearised forward with only N' kept equals the in-pass leaf sum |
| G3 (F3) | in-pass vs post-hoc: (a) their toy (root 16; τ = 5 → {h1[1]: 6, h2[0]: 8}); (b) our set-separating toy (post-hoc {A, D}, hierarchical {D}, both recovery 3/7) | exact |
| G4 (F4) | **their Fig. 2b**: hierarchical ≥ post-hoc recovery at matched node counts, on 20 IOI prompts (A12, ours: Wang et al. templates, GPT-2 top-1 = IO name, ABBA/BABA balanced), root = IO logit (A10) | mean curve dominates over the whole range, and on most prompts. Their only quantitative claim; no numbers are published, so the ordering is the check |
| G5–G7 (F5–F7) | bracket L1.H1, induction L5.H1 / L2.H2, IOI heads | qualitative, not gates |

## 4. A2: Dunefsky et al., official code (`transcoder_circuits@8721ff0`)

**Entry points (unmodified),** exactly dunefsky.md §6: `HookedTransformer.from_pretrained('gpt2')` (TL 1.11.0
defaults) → `SparseAutoencoder.load_from_pretrained(path)` × 12 (the cached `.pt` files) → `model.run_with_cache(toks[:,
:pos+1])` → `make_sae_feature_vector(tc[L], i, token=-1)` → `greedy_get_top_paths(model, tcs, cache, fv, num_iters,
num_branches, do_raw_attribution=True)` → `paths_to_graph` → `add_error_nodes_to_graph` (own-eval column only). Read-out
glue (ours, no change to their code): distinct `(layer, latent)` transcoder nodes, root excluded; heads, embeddings and
errors in separate columns.

- `do_raw_attribution=True`: the default cap mode hits the `A or B and C` precedence bug (`circuit_analysis.py:426`)
  and `paths_to_graph` asserts RAW (`:658`), so graphs need True (spec A1, A2).
- Knob: `num_branches` ∈ {5, 10, 15, 20, 30, 40, 60} at `num_iters` = 5 (primary); {100, 150, 200, 300} extended
  (labelled); `num_iters` 3 and 6 secondary. No threshold exists to sweep.
- Node universe = transcoder latents only, attention live through raw heads, so WCM is fitted a second time on
  transcoder sites only (arena A-TC, dunefsky.md §7.1). Roots: transcoder latents only.

**Fidelity checks (published numbers from their saved notebook outputs):**

| check | reproduces | pass |
|---|---|---|
| D-F1 (R1) | walkthrough mlp8tc[89], "Oh, that rifle model is a 6M", 3 × 15, raw | root 7.5700; level-1 [0][0] mlp0tc[8829]@9 0.8468, [0][1] mlp3tc[14171]@9 0.5593, [0][3] attn2[2]@8 0.3468; all 15 level-1 values ≤ 1e-3 rel.; beam-2/3 strings exact |
| D-F2 (R2, R3) | the graph | 24 non-error nodes, 33 edges, mlp0tc[8829]@9 1.40737, attn0[5]@9 1.09723, embed0@9 0.86215; root tc_error 1.49699, bias_error −7.15799, prune_error 2.54064; `keep_latents` = the 11 listed in dunefsky.md §8.2 |
| D-F3 (R4, R5) | greater-than paths | top-3 level-1 nodes and values to 2 s.f. |
| D-F4 (R6) | greater-than metric | 0.8829 / 0.7978 / 0.8453 (± 1e-3); tc10[5315] at variance rank 22 |
| D-F5 (R7 + 035) | transcoder fidelity | CE with 11 transcoders within ±0.05 of 4.234 (needs D5 or the pod corpus); our adapter's activations = their `SparseAutoencoder` ≤ 1e-5 |
| D-F6–F8 | input×gradient unit test; read-out glue; scoring identity (keep-all = clean root exactly) | exact |

## 5. WCM adapter (DAN-195): family `gpt2-small-tcz`

New module `wcm_tcz.py` here, registered at runtime like arena C's `wcm_marks.py` (no change under `src/`): a
`ModelFamily`, a site model in `hf_adapters._SITE_MODELS`, and a `PublicSAEBank` subclass (`patch_bank_dispatch` for
062's driver). Reuses `src` `HFSiteModel`, `sae.topk_sae.SAE` (as `MarksSAE` does: k = d_sae, whole ReLU code, an
expanded arange as indices), `NullSAE`, and 035's LN-folded transcoder weights (`transcoders_ct/`, validated 12/12;
035's `ours_gpt2.py` is the old tri-amp code, so only its loading idea is reused).

| site (layer, kind) | encode reads | edit (`x ← x + (ĉ − c) W_D`, decode of the code difference, no bias) | dictionary |
|---|---|---|---|
| (L, `z`) | the input of `h[L].attn.c_proj`, i.e. merged heads `[b, T, 768]` = TL `hook_z` flattened (head-major) | the same tensor, by a forward pre-hook on `c_proj`: the z error passes through W_O | Kissane L (24,576; L5, L7 49,152) |
| (L, `tc`) | the input of `h[L].mlp` (full `ln_2` output, HF `fold_ln=False`; hence the LN-folded `transcoders_ct` weights) | the **MLP output**: the hook calls `on_site(L, "tc", ln2_out)` and adds `(y − ln2_out)` to `mlp(ln2_out)`; that is 035's `mlp_out ← mlp_out + (ĉ − c) W_dec`, and the transcoder error passes through | Dunefsky L (24,576) |

- `kinds = ("z", "tc")` (plus `z2`, below), `same_layer_upstream = {"z": (), "tc": ("z",)}`, 24 real sites. The A-TC
  universe (A2) is the same family with every `z` slot a `NullSAE` (attention live and undecomposed), as arena C does
  for embed.
- **d_sae differs by site** (49,152 at z L5 / L7). `src` assumes one `d_sae` per family (223 uses in 82 files,
  e.g. 20 in `learned_mask.py`, the stores), so per-site widths are not an option without touching `src`. Two exact
  workarounds **[ours, no numeric effect; decide]**:
  - **split (recommended):** kinds `("z", "z2", "tc")`; at L5 / L7 the 49,152-wide dictionary is cut into two
    24,576-latent halves, `z` = latents 0..24,575 and `z2` = 24,576..49,151, both encoded from the same clean `hook_z`
    (ReLU is per latent and the decode is linear, so the two halves' edits add exactly: the hook returns
    `x + (y_z − x) + (y_z2 − x)`). At the other 10 layers `z2` is a `NullSAE`. Member ids map back to the original
    index. Cost ≈ the unpadded one.
  - **pad:** every 24,576 dictionary padded to 49,152 with dead latents (zero encoder row, bias −1, zero decoder). Simple
    but ≈ 2× the dictionary compute and memory for 22 of 24 sites.
- **BOS [decide]:** both dictionary sets were trained with BOS at position 0 (TL default); `SAESource.prepend_bos` is
  read by nothing in `src`, and the 069 corpus loader drops the stored lead token. Proposal: the site model prepends BOS
  inside `forward` and hides position 0 from every site (on_site sees `x[:, 1:]`; the BOS row stays clean; logits
  returned for positions 1..T). Then BOS is a frozen live prefix: never a member, never ablated, never a root, which is
  also the Ge (A14) and Dunefsky (A15) convention. Without it, position 0 is a content token used as the attention sink,
  off the dictionaries' distribution.
- **Precision:** float32, no TF32 (arena C).
- **WCM version:** production latent objective as arena C §5: 062's `driver.py` with the new family, `WCM_VERSION=v3`
  (Z + C + A, R, rank-keep 1e-1 per the current-WCM memory), k = d_sae (sparse fill = plain mean fill; rank-keep's Top-K
  motivation does not apply, stated). Random null and anchor-support rate reported (ReLU dictionaries: weak null, 035 /
  052).
- **Tests (CPU, `test_wcm_tcz.py`):** splice exactness (unpatched and open-gate patcher over all 24 sites = raw HF model,
  bitwise); encode parity vs Dunefsky's `SparseAutoencoder` and vs the Kissane formula ≤ 1e-5; HF z vs TL `hook_z` (from
  a tensor saved in `ge107`) ≤ 1e-5; HF tc input vs TL `ln2.hook_normalized` mapped through the LN fold; tiny 6-step fit
  writes a circuit.
- **Z-empty explosion check** (arena C Finding 2): forward-only diagnostic on a 24-target panel before the run. Lower
  risk here (no residual SAEs, which caused it), but checked.

**Scoring the baselines under our tests:** keep set = the method's (site, latent) members kept at every position
(position-collapsed); every other latent at layers below the target gets the test's fill; Z, A, C, necessity, R on the
circuit's own prompt (home turf) and on held-out prompts (labelled ours). Plus each paper's own evaluation: Ge leaf-sum
recovery (A11) for all three methods; Dunefsky "attribution coverage" and the greater-than-style full-replacement read
(ours, dunefsky.md §7.3).

## 6. Targets and prompts (DAN-197)

- **48 latents**: 2 per layer × {tc, z} × 12 layers, seeded, from latents with full top + mid stores after the data
  pass **[ours]**. Dunefsky covers the 24 tc targets; Ge and WCM all 48.
- **Per target:** K = 8 activating prompts **[decide; ours]**, each truncated after its max-activating position; one Ge
  and one Dunefsky circuit per prompt; one WCM fit on the same K prompts per λ (5 λ). Held-out: the protocol's
  held-out contexts for that target.
- **Corpus:** local smoke on `data_public/gpt2-small` (wikitext); pod on the 082 gpt2-small S1 OWT corpus. The data pass
  is 069's `run_data_pass.py` with the family added (as `../arena_c/data_pass_marks.py`): pass 1 + neg_ctx only (PMI pass
  2 assumes Top-K).

## 7. Local smoke plan and timing

| when | where | step |
|---|---|---|
| Wed day | CPU | after approval: `fetch_assets.sh local`; `setup_env.sh` (both envs); G0, G3; conversion + parity tests; D-F1–D-F4 on CPU in `tc107`; G1, G2, G4 on CPU in `ge107` |
| Wed day | CPU | `wcm_tcz.py` + `test_wcm_tcz.py` (splice, parity, 6-step fit) |
| Wed night (GPU, ≤ 8 h, stop 07:30) | 5070 Ti, `.venv` | data pass on the smoke corpus; Z-empty diagnostic; WCM smoke on 3 targets (2 tc, 1 z) × 2 λ; **timing pilot** (fit s/step, peak memory) |
| Thu day | CPU | Ge + Dunefsky circuits for the 3 smoke targets (CPU, their envs); score all three methods under our tests and their own; tables |
| Thu night | GPU | rerun anything that failed; a 48-target data-pass dry run on the smoke corpus |
| Fri | – | `run_pod.sh` dry run (`DRY_RUN=1`), pod command list for Daniel |

**Timing (estimates, not measurements; the pilot replaces them):**

| item | estimate | basis |
|---|---|---|
| WCM fit, 24 sites, 400 steps, 64-token contexts | ≈ 10–13 min local, ≈ 3–5 min H100 | arena C latent path 1 s / step on 19 × 32k Pythia sites; GPT-2 dictionaries ≈ 1.4× the encode / decode FLOPs, model ≈ 4.5× |
| WCM A1: 48 targets × 5 λ = 240 fits; A2: 24 × 5 = 120 fits | ≈ 18 + 6 GPU-h H100 | above |
| Ge: 48 × 8 prompts × (30 τ hierarchical + 30 post-hoc + 30 A11 re-runs + 1) passes | ≈ 0.6 GPU-h | ≈ 60 ms per forward + backward with 24 dictionaries at T ≤ 64 |
| Dunefsky primary: 24 × 8 × 7 `num_branches` = 1,344 circuits | ≈ 5–6 GPU-h | dunefsky.md §6: 3–15 s at 3 × 15, linear in branches |
| Dunefsky extended (100–300 branches) | ≈ 40 GPU-h at K = 8; **≈ 10 GPU-h at K = 2 [decide]** | linear in branches |
| scoring (our tests, own + held-out) of ≈ 5k Ge (13 grid readings / prompt), 1.3k Dunefsky, 360 WCM circuits | ≈ 9 GPU-h | ≈ 5 s per scored circuit |
| data pass on the OWT corpus (24 dense ReLU dictionaries) | ≈ 1–2 GPU-h | arena C pass 1: 54 s for 4,096 × 64 tokens at 19 sites |
| **total** | **≈ 45–55 GPU-h ≈ 6–7 h wall on 8 × H100 ≈ $160–190** at the runbook's ≈ $3.45 / GPU-h, on top of the 082 budget **[decide]** | |

## 8. Pod stage (`run_pod.sh`, to be written Thursday; each stage resumable per circuit, `STOP_AT`)

| stage | calls | GPUs |
|---|---|---|
| P0 assets | `fetch_assets.sh pod` (approved list), sha / size verify; `convert_transcoders.py` (035) | – |
| P1 envs | `setup_env.sh` with `TORCH_INDEX=cu121`; `pip check`; G0 + D-F1 as the env gate | 1 |
| P2 data pass | `data_pass_tcz.py` on the 082 S1 OWT corpus, sharded | 8 |
| P3 targets | `make_targets.py`: 48 targets, K prompts each, held-out sets | CPU |
| P4 baselines | `run_ge.py` (ge107) and `run_dunefsky.py` (tc107), one process per GPU, sharded by target | 8 |
| P5 WCM | `driver_tcz.py` (062's driver on the family) A1 (24 sites) and A2 (tc only) × 5 λ | 8 (MPS as 062) |
| P6 scoring | `score_arena_a.py` (.venv): our tests for all three methods + Ge leaf-sum for all + Dunefsky own-eval | 8 |
| P7 pack | curves (`../specs/evaluation.md`), tables, tgz to the global volume | CPU |

## 9. Risks to being pod-ready by Friday (largest first)

1. **Per-site d_sae** (z L5 / L7 are 49,152, everything else 24,576): `src` takes one `d_sae` per family everywhere.
   Implemented as the `z2` split (`wcm_tcz.py`, CPU-tested exact); still to confirm that the data pass, stores and
   scorer accept a site that shares its input tensor with another.
2. **The transcoder site in the engine:** the fit path is audited and compatible (`WCM_ADAPTER.md` section 1); the
   scorer / fills (`eval/wcm_eval.py`) are not yet audited for `x − decode(c)` uses at a site (P5).
3. ~~Kissane format inferred~~: verified 2026-11-03 (log above); one extra convention found (no b_V), handled.
4. **Old pins on Python 3.10:** `tc107`'s resolver (TL 1.11.0 with hub 0.17.3, datasets 2.12.0, setuptools 67) may need
   one or two more era pins; `pip check` will say. The vendored TL fork with torch 2.3.1 is the authors' own lock, so low
   risk.
5. **BOS convention** (§5, [decide]) and K / extended-knob scope ([decide]) need Daniel before the targets are built.
6. **Time:** two local nights only. The GPU pilot is the critical path; if Wednesday night fails, Thursday night is
   the only retry, and the pod numbers in §7 stay estimates.
7. **Pod budget** grows by ≈ $160–190 for arena A (§7).

## Files

- `README.md` (this file), `setup_env.sh` (both envs; written, not run), `fetch_assets.sh` (the download list as
  commands; written, not run).
- Written 2026-11-03: `ge_dicts.py` (conversion + `ZSparseAutoEncoder`), `ge_checks.py` (format, G1, G3b),
  `dunefsky_checks.py` (R1–R5), `wcm_tcz.py`, `test_wcm_tcz.py`, `WCM_ADAPTER.md` (design + proposed src changes),
  `env_freeze_ge.txt`, `env_freeze_tc.txt`; results in `out/ge_checks/`, `out/dunefsky_checks/`.
- To write: G2 / G4 in `ge_checks.py`, R6 in `dunefsky_checks.py`, `data_pass_tcz.py`, `driver_tcz.py`,
  `make_targets.py`, `run_ge.py`, `run_dunefsky.py`, `score_arena_a.py`, `run_pod.sh`.
