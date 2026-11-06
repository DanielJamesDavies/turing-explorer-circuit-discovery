# Arena D: implementation plan to pod-ready (written 2026-11-04 morning)

Goal: arena D (Caples et al.'s masks and Anthropic-style attribution graphs on Gemma-2-2B, each with WCM on its own
substrate) runs as a separate stage after the main 082 chain on the 8×H100 pod on **Saturday 2026-11-07**, as far as it
is ready. If nothing is ready by Friday evening, it runs after the pod (DAN-206 rule). Status and method details:
`README.md`, `../specs/caples.md`, `../specs/attribution-graphs.md`. Linear: DAN-201 (D1), DAN-202 (D2), DAN-203 (WCM
adapter), parent DAN-188.

## 0. Settled (don't reopen)

| item | setting | source |
|---|---|---|
| Scope for Saturday | **tasks only**: D2 (Caples grids A / B / C + a dense grid; D2-WCM on rc, both modes), then D1 tasks (AG + D1-WCM on the 4 SVA tasks). Latent-rooted rows and the CLT are out (later) | 2026-11-03 costing (tasks only ~£17–29) |
| Activation function | Gemma Scope and PLT dictionaries are JumpReLU. The engine reads the `threshold` buffer of `wcm_gemma.JumpReLUSAE` (`sae.dense.seed_threshold`): the scorer's diagnostic reads use pre · (pre > θ), and the censored term's cut is never below θ. The primary post-Top-K read was already the bank's own JumpReLU encode | Daniel 2026-11-03; engine change 1, 2026-11-04 |
| SAE error at a site | native mode: error live (circuit-tracer practice); Caples mode: their per-position mean error, deliberately (their substrate). Audit 2026-11-04: every engine path writes x + decode(code difference), so the PLT difference form is exact | Daniel 2026-11-03; audit in `../arena_a/IMPLEMENTATION_PLAN.md` §0 |
| BOS | excluded from pooled means: the Gemma banks set `mean_skip_positions = 1` (their sites see position 0; the site model restores it after every edit) | Daniel 2026-11-03; engine change 3, 2026-11-04 |
| Behavioural WCM (tasks) | arena C's adopted behavioural setting (Z + A + C + R, no rank-keep, geometric lr), unchanged. The ×3.15 peak / warm-up / censored term were validated on latent targets only (the censored term needs `objective='pos'`), so they are not added untested | arena C §8; v4 evidence is latent-only |
| Gated weights | access confirmed. `HF_TOKEN` only from the shell environment on the pod, never in a file or log; the fetch is proposed to Daniel first | Daniel 2026-11-03 |
| Never run their script in `third_party/` | `run_caples.sh` works on a `git clone --local` copy (their anchors live in `masks/` there) | README §0.4 |

## 1. Decisions needed from Daniel (recommendation first)

1. **D1 task set:** the 4 SVA tasks, K = 40 prompts each (SFC's evaluation set), rc first (recommended).
2. **Weights for D1 on the pod:** official `google/gemma-2-2b` fp32 (recommended), with a tensor-by-tensor check
   against the unsloth bf16 mirror (tells whether local bf16 results carry over).

## 2. Work items, in order

"Agent" = a briefed subagent (PM mode); I review every result before it counts.

| # | item | files | acceptance | when |
|---|---|---|---|---|
| ✓ D1 (2026-11-05) | wire the 2026-11-04 engine changes into the adapter: `GemmaSAEBank.mean_skip_positions = 1` (both families); tests that `seed_threshold` reads the JumpReLU buffer and that pooled means skip BOS | `wcm_gemma.py`, `test_wcm_gemma.py` | 9 existing + 2 new tests pass (CPU) | Wed day |
| ✓ D2 (2026-11-05) | task runner: arena C's `wcm_marks.run` / 081-engine path for both families (`--means pooled | per_position`, behavioural setting above), resumable per (task, λ, means, mode) | `wcm_gemma.py run` | 6-step CPU fit on a tiny random Gemma-2 for each family writes a circuit; rerun skips it | Wed day (agent) |
| ✓ D3 (2026-11-05; tiny model; the real-model version is a GPU test for D7) | Caples fill: their `get_sae_means` / `get_sae_error_means` run in `caples107` and saved as `.pt` (per-position code means and ε̄, 640 corrupted prompts, running-mean bug kept); loaded by the `.venv` side as the C fill and the Caples-mode error. Our exact means computed too; both reported | `caples_means.py` (caples107), loader in `wcm_gemma.py` | the open-gate Caples-mode LD equals their `run_sae_hook_fn(use_mean_error=True)` LD (README §4.3 test 4) | Wed day–Thu (agent) |
| ✓ D4 (2026-11-05; tiny model, CPU) | scorer path: per-position membership (their `sae_mask_{i}.pt`, θ > 0) and position-collapsed WCM circuits under our tests (Z, A, C, necessity (a) error live + (b) error clean) in both modes; their own evaluation through their `SAEMasks` in `caples107`, plus the SFC-normalised version with m(∅) | `score_d.py`, `test_score_d.py` | keep-all = clean LD exactly; one Caples mask and one WCM circuit scored both ways on CPU | Thu day |
| ✓ D5 (2026-11-05; CPU dry runs; F3 runs in D7) | fidelity scripts: `run_caples.sh` (grids A / B / C on a clone copy), `caples_fidelity.py` (model LD 3.21 ± 0.02, sizes ± 30 %, f ± 0.05 of the anchors), `ag_fidelity.py` (F1 provenance, F3 Dallas in bf16, F5, F6), `ag_tasks.py` (resumable per prompt, transcoders pinned `@3412e19b…`) | as named | dry runs parse; `ag_fidelity.py` F3 runs on the GPU night | Wed–Thu (agent) |
| D6 (written 2026-11-05, `test_real_open_gate_identity_gpu`; runs in D7) | real-model identity test: the open-gate PLT and gsres site models on the unsloth bf16 2B, max \|Δlogit\| = 0 vs the raw HF model | `test_wcm_gemma.py -m gpu` | bitwise identity on 8 SVA prompts | **Thu night** (GPU) |
| D7 (script ready 2026-11-05, `DRY_RUN=1` clean) | **GPU smoke** (one GPU process at a time; +3 / +10 min checks; stop 08:30): D6; `caples_smoke.py` (bf16, N 64, one λ); WCM tiny fits on both families; F3 Dallas; timings per family (batch 4 × 7 tokens) | `run_pilot_d.sh` | 0 errors, no shared-memory spill, timings in README §8 | **Thu night** |
| ✓ D8 (2026-11-05; CPU dry runs + a CPU queue test) | pod stage `pod_stage_d.sh` (README §9): gate on GPU 0, then one job per GPU, resumable, `DRY_RUN=1`; venv builds and weight fetches listed as commands for Daniel to approve | `pod_stage_d.sh`, `pod_checks_d.py`, `pod_summary_d.py` | dry run prints every command | Thu–Fri |
| ✓ D9 (2026-11-05) | chain integration: a post-main stage in `082/chain.sh` (`ARENA_D=1`, off by default) after arena A's, plus the runbook entry | `082/chain.sh`, runbook | `DRY_RUN=1` shows it last; off = unchanged dry-run output (TL on / off, `LIMIT=2`, `ARENA_A=1`: identical, timestamps masked); STOP_POD no longer needs the arenas to succeed | Fri |
| D10 | re-cost in £ from D7's timings, together with arena A, against the £600 cap | runbook, DAN-206 | total ≤ £600 | Fri |

## 3. Cut lines if time runs short (in this order)

1. D1 (AG) after the pod; Saturday runs **D2 only** (Caples grids + D2-WCM on rc, both modes) (README §11 risk 2).
2. D2-WCM in Caples mode only (the like-for-like row); native mode later.
3. Arena D entirely after the pod.

## 4. Pod cost (to replace with D7's numbers)

README §8: tasks only ≈ 7–12 GPU-h, ~2–3 h wall on 8 GPUs after a ~1 h setup + gate, ≈ £17–29 (setup and gate
time included at the pod's rate). Disk: provision ≥ 150 GB for the weights, the two venvs and the outputs.

## 5. Risks

1. Build time: D2–D4 are the bulk. The D2-only cut line keeps Saturday feasible.
2. `HF_TOKEN` must reach the pod shell (environment only). Without it D2 as published can't run; D1 falls back to the
   unsloth bf16 mirror, labelled.
3. Anchor noise: no seed in their script; accept ± 30 % in size.
4. Precision drift (bf16 locally, fp32 on the pod), checked by the tensor comparison in §1.2.
