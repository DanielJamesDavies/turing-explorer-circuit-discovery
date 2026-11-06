# Arena A: implementation plan to pod-ready (written 2026-11-04 morning)

Goal: arena A (Ge et al. and Dunefsky et al. on GPT-2 small, with WCM on the same substrate) runs as a separate stage
after the main 082 chain on the 8×H100 pod on **Saturday 2026-11-07**. If it isn't ready by Friday evening, the pod
runs without it and arena A runs afterwards (DAN-206 rule). Status and method details: `README.md`, `WCM_ADAPTER.md`,
`../specs/ge.md`, `../specs/dunefsky.md`. Linear: DAN-192 (Ge), DAN-197 (Ge / WCM runs), DAN-200 (Dunefsky), DAN-195
(WCM adapter), parent DAN-188.

## 0. Settled (don't reopen)

| item | setting | source |
|---|---|---|
| WCM for latent targets | `WCM_VERSION=v4` (v3 + geometric lr ×3.15, warm-up 0.1, censored term), 062's driver on the `gpt2-small-tcz` family | Daniel 2026-11-04 |
| BOS | `ARENA_A_BOS=1`: the site model prepends BOS and hides position 0 from every site (frozen live prefix; Ge A14, Dunefsky A15) | Daniel 2026-11-03 ("do what you think is correct") |
| SAE error at a site | kept live, as circuit-tracer and Dunefsky do. Audit 2026-11-04: every engine and scorer path writes x + decode(code difference), so the transcoder difference form is exact | Daniel 2026-11-03; audit below |
| Per-site width | the `z2` split at L5 / L7 (exact, CPU-tested) | README §5 |
| Activation function | Kissane z-SAEs and Dunefsky transcoders are ReLU: no `threshold` buffer, so `sae.dense.seed_threshold` returns None (unchanged behaviour) | engine change 1, 2026-11-04 |
| Means | **do not** set `mean_skip_positions` on the arena A bank: the site model already hides BOS, so the sites' activations start at position 1 (setting it would drop the first real token) | engine change 3, 2026-11-04 |
| Precision | float32, no TF32 | README §5 |
| Downloads | D1–D5 done (approved 2026-11-03) | README log |

**Audit result (engine change 2, 2026-11-04).** Paths checked: `learned_mask.LearnedMaskPatcher.transform` (every
branch returns `x + decode(delta_code, add_bias=False)`, including the inverted necessity path);
`eval/ablation_faithfulness.CircuitOnlyPatcher` (`decode(patched) + error`, error = x − decode(natural); its optional
`keep_error_sites` SFC mode is never used by WCM or the scorer); `eval/wcm_eval.LiveScalePatcher`,
`eval/counterfactual_faithfulness`, `eval/donor_test`, `eval/node_presence`, `eval/cluster_faithfulness`,
`081-task-circuits/engine.py` (all `decode(patched) + error`). `collect_site_error_means` is only for Caples' substrate
(arena D, deliberate). No path writes anything other than x plus a decoded code difference, so a transcoder site's
`y − x_in` added to the MLP output is exactly the replacement change, with the real MLP's error live.

## 1. Decisions needed from Daniel (recommendation first)

1. **K prompts per target:** 8 (recommended; README §6). Fewer = cheaper Ge / Dunefsky but noisier per-target curves.
2. **Dunefsky's extended knob** (100–300 branches): K = 2 prompts only, ~10 GPU-h (recommended), or drop it.
3. **λ count for WCM:** 5 (recommended), which matches the curve plan in `../specs/evaluation.md`.

These only change the pod cost (§5), not the code, so work proceeds on the recommended values.

## 2. Work items, in order

Each item: what, where, acceptance test. CPU unless marked GPU. "Agent" = a briefed subagent (PM mode); I review every
result before it counts.

| # | item | files | acceptance | when |
|---|---|---|---|---|
| A1 | remaining Ge fidelity: G2 (Theorem 3.1 on a gated G'), G4 (Fig. 2b, 20 IOI prompts, hierarchical ≥ post-hoc recovery at matched node counts) | `ge_checks.py`, `out/ge_checks/` | G2 relative gap ≤ 1e-4; G4 mean curve dominates over the whole range | Wed day (agent, `ge107`) |
| A2 | Dunefsky D-F4 (R6 greater-than metric) | `dunefsky_checks.py` | 0.8829 / 0.7978 / 0.8453 ± 1e-3; tc10[5315] at variance rank 22 | Wed day (agent, `tc107`) |
| A3 | `driver_tcz.py`: 062's driver on the family (registers `wcm_tcz` before importing the driver, as arena C's `driver_marks.py`), `WCM_VERSION=v4` | `driver_tcz.py` | 6-step CPU fit on 1 tc and 1 z target writes a circuit and a status row with `wcm_version: v4` | Wed day |
| A4 | Z-empty diagnostic panel (arena C Finding 2), forward only | `diag_zero_empty_tcz.py` (copy of arena C's) | no explosion (empty-circuit read within the panel's normal range) on 24 targets; report the table | Wed day (CPU) / Wed night (GPU) |
| A5 | data pass: 069 `run_data_pass.py` with the family (pass 1 + neg_ctx; PMI pass 2 assumes Top-K) | `data_pass_tcz.py` (as `../arena_c/data_pass_marks.py`) | stores written for all 36 sites incl. `z2`; confirms the stores and scorer accept a site that shares its input with another (README risk 1) | Wed night (GPU smoke corpus) |
| A6 | targets + prompts: 48 latents (2 per layer × {tc, z} × 12), K prompts each truncated after the max-activating position, held-out sets as the protocol | `make_targets.py` | lists + manifest (md5), seeded, deterministic re-run | Thu day |
| A7 | baseline runners, resumable per circuit: `run_ge.py` (30 τ hierarchical + post-hoc + A11 re-runs), `run_dunefsky.py` (7 `num_branches` at `num_iters` 5) | `run_ge.py`, `run_dunefsky.py` | on 3 smoke targets: circuits written, a rerun skips them, G1 / D-F1 still pass inside the runner | Thu day (agent, CPU) |
| A8 | scorer: our tests (Z, A, C, necessity, R; home turf + held-out, labelled) for all three methods, Ge leaf-sum recovery for all three, Dunefsky's own coverage | `score_arena_a.py` | keep-all circuit = clean read exactly; scores for the 3 smoke targets × 3 methods; table | **done 2026-11-05** (README "A8"; R and the ge107 recovery companion wired + unit-tested, smoke not yet rescored with them) |
| A8b | per-prompt ("their turf") test: the WCM members that fire on one prompt (fire_count's definition) vs the circuits Ge / Dunefsky build for that prompt, on that prompt; + top-k adaptation curve, held-out-prompt variant | `score_perprompt.py` (sibling, imports `score_arena_a`), `test_score_perprompt.py` | keep-all restriction = home scope on prompt q exactly; 2 panel circuits × 2 prompts on CPU; pod cost | **done 2026-11-05 CPU** (README "A8b"; GPU and the held-out baseline runs not done) |
| A9 | **GPU timing pilot** (one GPU process at a time; +3 / +10 min checks; stop 08:30): data pass on the smoke corpus, Z-empty panel, WCM smoke 3 targets (2 tc, 1 z) × 2 λ at v4; fit s / step and peak memory | `run_pilot_a.sh` | 0 errors, no shared-memory spill (≤ 15.6 GB), timings recorded in README §7 | **Wed night** (first in the queue) |
| A10 | pod stage `run_pod.sh` P0–P7 (README §8), `DRY_RUN=1`, resumable, `STOP_AT`; downloads and env builds listed as commands for Daniel to approve on the pod | `run_pod.sh` | dry run prints every command; a 2-target smoke through P2–P7 locally | **done 2026-11-05 (CPU, dry runs only)** (README "A10 / A11"): dry run prints every command (rc 0); STOP_AT=P0 / P3 sandbox runs ok; **the 2-target GPU smoke is NOT run yet**: `run_smoke_local.sh` written, dry run passed (`SMOKE_READY`), for the coordinator to chain |
| A11 | chain integration: a post-main stage in `082/chain.sh` (`ARENA_A=1`, off by default) that calls `run_pod.sh` after the main tracks finish, plus the runbook entry | `082/chain.sh`, runbook | `DRY_RUN=1` chain shows the stage after T12 / the public tracks; off = unchanged dry-run output | **done 2026-11-05**: off = identical dry run (TL on / off / `LIMIT=2`, timestamps masked); on = the stage after S10 + T12; runbook entry 2026-11-05 |
| A12 | re-cost in £ from A9's timings; fit inside £600 together with arena D | runbook, DAN-206 | a line per stage, total ≤ £600 | Thu |
| A13 | WCM tuning for the tc + z substrate on a 9-target DEV set (panel), pre-registered rule, 6 configs (base, γ_R 0 / 1e-2, NEC_W 0.3, both, 600 steps) ≈ 4.25 GPU-h; dev + pilot keys excluded from production (`make_targets.py` EXCLUDE) | `TUNING.md`, `tune_grid.tsv`, `run_tune_a.sh`, `tune_summary.py`, `tune_exclude.txt` | `out/tune/summary.md` + `CHOSEN.env` (rule applied mechanically); run_pod.sh P3 passes EXCLUDE, P5 sources CHOSEN.env | Thu night (GPU, after arena D smoke) |

## 3. GPU schedule (local card, one process at a time)

- **Wed night (from ~18:30):** A9 first (critical path for the cost). Then the leftover v4 work: the 7 unfinished
  check-panel targets (`check_geo_x3.15_warmup_censor`) and the check-panel donor tests for the warm-up arms.
- **Thu night:** arena D's GPU smoke (its own plan) has priority; arena A gets any rerun A9 needs.

## 4. Cut lines if time runs short (in this order)

1. Drop Dunefsky's extended knob.
2. K = 4 prompts per target.
3. 3 λ instead of 5.
4. Ge + WCM only on the 24 tc targets (A-TC universe); z targets later.
5. Arena A after the pod run, on a cheaper later session.

## 5. Pod cost (to replace with A9's numbers)

README §7 estimate: 45–55 GPU-h ≈ £125–145 at ~£2.65 / GPU-h, mostly WCM fits (24 GPU-h at 5 λ), Dunefsky (5–6
GPU-h, plus ~10 if extended at K = 2) and scoring (~9 GPU-h).
**Added 2026-11-05 (Daniel 13:35):**
- P4h, the held-out baselines: Ge 48 × 8 held prompts ≈ 0.6 GPU-h; Dunefsky 24 × 8 × 7 published branch values =
  1,344 circuits ≈ 5–6 GPU-h; no > 60 knob on held-out.
- P6b, the A8b per-prompt scoring: ≈ 14.6k WCM one-row units + ≈ 12.9k held baseline units at τ stride 2, plus their
  Ge recovery, ≈ 4.5–8.5 GPU-h.
- Together **≈ 10.5–14.5 GPU-h ≈ £28–38**: the A8b agent's estimate, kept because no better numbers exist yet. The
  smoke's `cmds.tsv` gives per-unit seconds for A12. The WCM fit time per step is the biggest unknown;
v4's fit time is the same as v3's (measured 2026-11-04).

## 6. Risks

1. The GPU pilot is the critical path: if Wednesday night fails, Thursday night is the only retry.
2. Stores / scorer with the `z2` shared-input site (A5 confirms).
3. Old pins on Python 3.10 on the pod (cu121 wheels): `pip check` at P1; fallback = the cached torch 2.8 wheel, labelled.
4. Pod remote rule: every pod command is proposed to Daniel first; the stage never fetches anything unapproved.
