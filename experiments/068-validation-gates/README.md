# 068 — validation gates for the public-model and TuringLLM runs (DAN-36, DAN-74)

Two gates before the H100 runs (Pythia-70m and GPT-2 small, 1,024 targets each, through the 070
adapters):

- **DAN-36**: the production engine on a public model agrees with code already verified (033's standalone Pythia
  implementation).
- **DAN-74**: HEAD still reproduces the TuringLLM 15k-circuit run (062 `out_full`) after the latest src changes.

| File | Gate | Runs on |
|---|---|---|
| `ref033.py` | DAN-36: 033's fit, scorer and null, transcribed device-agnostic (the standalone reference) | (library) |
| `pythia_033_panel.py` | DAN-36: 033's panel re-fitted through the engine, scored by 033's scorer, plus nulls | CPU or GPU |
| `native_onoff.py` | DAN-36: native kernels on vs off | GPU |
| `golden_turingllm.py` | DAN-74: 32-target golden set, two local refits, vs the H100 circuits and eval rows | GPU |

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| Identity edit is exactly zero at every site | **passed** (070) | `tests/model/test_public_adapters.py::test_zero_decoded_delta_is_identity_at_every_site`, `::test_engine_mask_patcher_natural_pass_is_untouched`, `::test_unedited_engine_forward_matches_hf_logits_bitwise`; 070 `recon_audit.py`: `ident` = 0.0 at every site of both models |
| L0 = k at every site | **passed** (070) | `::test_topk_slots_equal_k_at_every_site`, `::test_preact_topk_reproduces_encode_at_every_site`; 070 recon audit: Pythia 16.00 non-zeros at every token of every site, GPT-2 31.69–32.00 (SAELens ReLU after top-k) |
| Engine vs standalone circuit, same seed and contexts: comparable size and scores | **passed on a 4-seed CPU subset**: verdicts 4/4, F0 / FM within ~0.1, n 1.06–1.69× (explained: gate pricing, below) | below |
| 033 panel reproduces its headline (22/30, nulls 0/146) through the engine | **pending GPU** (subset: engine 2/4 = 033 2/4, nulls 0/4) | below |
| Native kernels on vs off match | **pending GPU** (the kernels are CUDA-only) | below |
| DAN-74 golden set | **pending GPU** (written, selection done, not run) | below |

## DAN-36: the 033 panel through the engine (`pythia_033_panel.py`)

### What 033 is (the reference)

`033-cross-sae-topk/crosssae_topk.py`, arm `triamp400`: triple floor (zero 1.0, negctx-mean 0.25, posctx-mean
**0.10**, each over mean(target²)), free amplitudes α = softplus(ψ), λ = 1e-3 per latent, AdamW lr 0.05 / wd 0.05,
θ₀ = 2.0, gate σ(θ/T) with T annealed 1 → 0.05, 400 steps, batch 4 (contexts `(4·step) mod 48`), keep θ > 0, fp32
with TF32 on GPU.

- **Contexts.** wikitext-103 windows of 64 tokens (`scan.pt` / `scan_panel.pt`, the same 20,000-window table). 64
  strongest windows (first 48 train, last 16 held out) and 64 verified-silent windows (48 train for the floor means,
  16 held out). Anchor = argmax over positions of the natural seed pre-activation.
- **Panel = 30 seeds.** `rows.jsonl` has 32 `triamp400` rows. Six resid L2 / L4 seeds came from `scan.pt`; the
  panel run skipped the four it shared with them, so those keep `scan.pt`'s negatives. The panel is the 32 minus
  two with held-out a_pos < 1.0 (mlp L2 #12294, attn L4 #2654): 6 from `scan.pt` + 24 from `scan_panel.pt`.
- **Pass rule (ALL-PASS, 029 / 031):** ampF0 and ampFM both in [0.8, 1.25] and sup > 0.9, on the held-out 16.
- **Headline, recomputed from rows.jsonl:** 22/30 pass, median n 51; nulls 0/146 (28 seeds × 5 draws + the two
  `scan.pt`-only seeds × 3).
- **Nulls.** Random members from the live pool (latents firing anywhere on the training contexts) at the circuit's
  n, `random.Random(1000 + latent)`. Gates fixed (θ = ±40), only α fitted, λ = 0.
- Do-not-quote for the paper (DAN-38: unshuffled held-out split), but it is the reference implementation.

### How the comparison is made

- `ref033.py` transcribes 033's fit, scorer and null device-agnostically (`ERROR_MODE=current`, no `BOUND`, the
  configuration of `rows.jsonl`).
- `pythia_033_panel.py` fits each seed with 033's contexts and anchors through
  `_build_mode_method("ablation_gradient", "mask") → discover() → run_learned_mask`. Contexts are injected as 062's
  driver does, with 033's settings in `config.discovery.learned_mask` (`configure_033`).
- Every circuit, engine or standalone, is scored by **033's own scorer** on 033's held-out 16.
- Arms per seed:
  - `engine`.
  - `standalone`: 033's own fit on the same machine, which separates "engine vs standalone" from "CPU vs 033's
    GPU".
  - `null0`: 033's null through the engine core. `run_learned_mask` gets a fixed support, gates frozen on
    (θ₀ = 40), λ = 0, members drawn at the engine circuit's n.
  - `standalone_notemp` (diagnostic): 033's fit with the gate priced the engine's way (below).

### Settings that cannot match (engine vs 033)

| # | 033 | engine | effect |
|---|---|---|---|
| 1 | L1 and the α leak guard price the **annealed** gate: λ·Σσ(θ/T) and λ·Σ(1−σ(θ/T))·\|α−1\| | price **σ(θ)** (`learned_mask.py` `mask_sum` ~l.1221, `off_member` ~l.1608); the forward still uses σ(θ/T); no config knob | late in training 033's price concentrates on boundary latents (gradient ×1/T up to 20×), so the engine prunes less: **bigger circuits** (attn/1/32511: 59 vs 35). Confirmed by the `standalone_notemp` arm: 033's code with σ(θ) pricing gives n = 61, F0 0.702, FM 0.759 ≈ the engine's 59 / 0.693 / 0.747. **The one real difference** |
| 2 | mlp(L) seeds also mask attn(L) | Pythia's parallel residual: attn(L) does not feed mlp(L) (070); not an upstream site | inert in 033 (zero data gradient, pruned by L1 within ~40 steps); nulls here draw only from engine sites |
| 3 | null gates fixed at ±40, not trained | support members θ₀ = 40 (σ(40/T) = 1, gradient 0; AdamW decay leaves ~15 at step 400), non-support clamped at −12 | equivalent up to σ(−12/T) ≤ 6e-6 leak at the first steps |
| 4 | α kept at full precision | α stored rounded to 4 decimals (`learned_mask.py:1738`) | ≤ 5e-5 per α |
| 5 | seed read (x − b_dec)·w + b_enc | x·w + (b_enc − W_enc·b_dec) | algebraically equal (070 test: bitwise on the reference stream) |
| 6 | GPU, TF32 | this run: CPU fp32 | float-level. The `standalone` arm measures it: 033's fit on CPU reproduces 033's GPU rows to ≤ 0.024 in F0 / FM with the same n, on both seeds it ran on |
| 7 | null at 033's circuit n | null at the engine circuit's n | by design (the engine's null) |

Everything else is set to 033's value: triple floor; γ_C 0.25, γ_A 0.10; λ 1e-3; 400 steps; lr 0.05; AdamW wd 0.05;
θ₀ 2.0; anneal; keep 0.5; batch 4; 48 / 16 split; train-only means; no rank-keep; no off-target term; amp_l1 0;
fp32.

### Results: CPU subset (4 of 30 seeds; `out_panel/report.md`, rows `out_panel/rows.*.jsonl`)

The full panel on CPU would take ~25 h (cost grows with upstream sites; the 30 seeds total ~145 site-units at
~10 min each for engine + standalone + one null on an idle CPU), so this run is a subset. It covers L1, every kind,
two 033 passes and two 033 fails. During this run the CPU was shared with another agent's 069 data pass (load ~30 on
16 threads), so fits ran ~3× slower than idle and the planned six seeds were cut to four. The standalone arm was
dropped for the last two seeds.

| seed | 033 (GPU) n / F0 / FM / sup / pass | engine n / F0 / FM / sup / pass | standalone (CPU) n / F0 / FM / pass | 033 code with engine pricing | engine null n / F0 / FM |
|---|---|---|---|---|---|
| attn/1/31694 | 18 / 1.022 / 1.013 / 0.998 / PASS | 19 / 1.021 / 0.985 / 0.997 / PASS | 18 / 1.022 / 1.015 / PASS | | 19 / 0.000 / 0.005 |
| attn/1/32511 | 35 / 0.782 / 0.742 / 0.996 / fail | 59 / 0.693 / 0.747 / 0.997 / fail | 35 / 0.759 / 0.758 / fail | 61 / 0.702 / 0.759 / fail | 59 / 0.000 / 0.000 |
| mlp/1/26871 | 23 / 0.729 / 0.943 / 0.995 / fail | 30 / 0.679 / 0.827 / 1.000 / fail | | | 30 / 0.000 / 0.000 |
| resid/1/6626 | 11 / 1.051 / 0.950 / 1.000 / PASS | 13 / 1.013 / 0.976 / 1.000 / PASS | | | 13 / 0.000 / 0.000 |

- **Verdicts:** engine vs 033 agree on 4/4 seeds (pass 2/4 both). The standalone (CPU) arm agrees with 033 on 2/2.
- **Size:** n_engine / n_033 median 1.24 (1.06–1.69). The engine is never smaller.
- **Scores:** |F0 engine − 033| median 0.044 (max 0.090); |FM| median 0.027 (max 0.117, mlp/1/26871, a fail
  either way).
- **Nulls:** 0/4 pass, max F0 0.000. 033 on these seeds: 0/20.
- **033 on CPU vs 033 on GPU** (`standalone` vs `rows.jsonl`): same n on both seeds, F0 / FM within 0.024. The
  transcription is faithful, and machine drift is small next to the engine difference.
- **Membership** (same machine, same contexts):
  - engine vs standalone Jaccard 0.85 (attn/1/31694) and 0.59 (attn/1/32511);
  - engine vs 033's code with the engine's pricing: **0.90** (59 vs 61 members).
  - So once the objective matches, the engine and the standalone code find nearly the same circuit. The remaining
    gap is the pricing (difference #1), not the adapter or the engine plumbing.

**Status:**

- Engine vs standalone (same seed, same contexts, comparable size and scores): **passed on the subset**. Functions
  agree: same verdicts, F0 / FM within ~0.1, nulls dead. Size runs 6–69% larger because of the σ(θ) pricing.
- 22/30 headline: **pending GPU** (full panel).

The CPU subset was run as several processes sharing `out_panel/` (`PART=a..e`, one rows file each), e.g.:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TURING_NATIVE_CPU_ONLY=1 THREADS=8 PART=a SEEDS=attn/1/31694 \
  .venv/bin/python experiments/068-validation-gates/pythia_033_panel.py          # REPORT=1 rebuilds out_panel/report.md
```

Full panel, on the GPU:

```bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src HF_HUB_OFFLINE=1 DEVICE=cuda SEEDS=all OUT=experiments/068-validation-gates/out_panel_gpu \
  .venv/bin/python experiments/068-validation-gates/pythia_033_panel.py
```

- Arms: engine + standalone + 5 nulls per seed (N_NULL defaults to 5 on CUDA), all 30 seeds.
- Time: 033 fitted in 10–50 s per fit on GPU, and the engine is ~1.6× slower than the standalone code on CPU. That
  makes ~7 fits × ~40 s per seed, **~2.5 h**. `ARMS=engine,null` saves ~20 min.
- Use a separate OUT: resume keys do not include the device, so a shared OUT would skip seeds already done on CPU.
- Add `ARMS=engine,standalone,null,standalone_notemp` to repeat the pricing diagnostic on every seed (+~20 min).
- **Proposed gate** (Daniel to confirm):
  - engine pass count within ±3 of 22 on the same 30 seeds (033's own clean-error re-run flipped 5 of 28 verdicts, net −3);
  - per-seed verdict agreement ≥ 80%;
  - median n_engine / n_033 within [0.5, 2];
  - 0 null passes.

## DAN-36: native kernels on vs off (`native_onoff.py`, pending GPU)

The native kernels on the fit and scoring path all sit in the **no-grad** SAE encode (`src/sae/topk_sae.SAE.encode`):
cublasLt `linear_relu` (loaded whenever `src/native/linear_relu_ext*.so` exists, used only for CUDA **bfloat16**
x / W / b), the Triton top-k (`TURINGLLM_TOPK_IMPL=triton`, CUDA bf16; config default `pytorch`), and the fused exact
top-k (`TURINGLLM_SAE_ENCODE_IMPL=fused_exact_topk`, native .so with `TURINGLLM_FUSED_EXACT_TOPK_USE_NATIVE=1`). The
mask's training forward runs with grad enabled and always takes the plain PyTorch branch, so the kernels reach a fit
only through its no-grad passes (floors, held-out loss) and reach every scorer pass.

**Consequence for the public runs:** at the public-family default `model.dtype = None` (float32, what 033 / 053 ran
at), no native kernel is ever called. "On vs off" is then bitwise identical by construction, which is a vacuous pass.
The kernels only matter if the H100 runs use `model.dtype = bfloat16` (or on TuringLLM, whose SAEs are bf16 on GPU).

The script runs each mode in its own subprocess (`off`, `off2` = `off` again for the GPU's run-to-run floor, `on` =
cublasLt + Triton, `on_exact` = cublasLt + native fused exact top-k), for float32 and bfloat16, on two 033 seeds,
with the protocol-v1 fit (062 settings) and the src scorer (`wcm_eval.score_circuit`). It **counts native calls** in
the fit and in the scorer, so a vacuous match is labelled as such. Gate: `on` / `on_exact` within the `off`-vs-`off2`
floor (Jaccard ≥ min(floor, 0.98), |dα| ≤ max(floor, 1e-2)) and the same DAN-8 verdict.

CPU dry run (5 steps, float32, attn/1/31694, `off` / `off2` / `on`): all three bitwise identical (130 / 130 row
fields exact), and **0 native calls** in fit and scorer (64 no-grad + 45 grad encodes in the fit, 143 no-grad in the
scorer). This checks the script, not the kernels.

```bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python experiments/068-validation-gates/native_onoff.py
```

16 child runs (2 dtypes × 2 seeds × 4 modes), each ~1 min start-up + fit (~30 s on GPU) + score: **~30–40 min**.
Report: `out_native/report.md`; `REPORT=1` re-prints it.

## DAN-74: TuringLLM golden regression set (`golden_turingllm.py`, pending GPU)

**Pinned commit.** The 15k run (062 `out_full`, 8×H100) pulled `multi-device` at **538cbed**. Stage 1's
1,416 targets ran earlier (d5109bc / 564b68f) and were skipped by the full run's resume; between those commits only
`062/driver.py` changed, `src/` is identical. Since then: committed HEAD (76e1531) changed `src/circuit/discovery/
base.py`, `gradient_base.py`, `probe_dataset.py`, `config.py` and added `protocol_contexts.py` (all behind
`context_protocol == "v1"`, default `legacy`, and bypassed by the driver's context injection); the uncommitted
working tree (the new adapters) changes `ablation_gradient.py` (site sort by `bank.kinds`, identical to the
alphabetical sort for attn / mlp / resid), `ablation_faithfulness.upstream_sites` (TuringLLM keeps its rule),
`hooks.py` / `inference.py` / `bank.py` (family dispatch), `neg_context_selector.py`, `config.py`, and 049's
`amp_eval_pass_v2.py` (`EVAL_SPLIT`, default `as_run` = the run). None should move a TuringLLM fit; this gate checks.

**Golden set** (`out_golden/selection.json`, done on CPU): 32 targets, `random.Random(74)`, fitted and non-vacuous
(held-out strongest row), non-thin, one per (layer, kind) cell of the 36 minus dropped cells. `0.attn` has no
circuits (TuringLLM attn L0 has no upstream sites), so one dropped cell was refilled: attn 9 / mlp 11 / resid 12,
every layer ≥ 2, dropped `5.attn`, `8.mlp`, `9.attn`. H100 circuit sizes 24–1,179 members (median ~470).

**What it does.** Imports 062's `driver.Runner` with `OUT` pointed at `out_golden/driver_out` (it refuses any OUT
inside `out_full`), reads each target's context record from `out_full/out/ctx` (read-only), and:

1. **fit**: `Runner.method(GAMMA, LAM, True)` + `Runner.fit` twice per target (`r0`, `r1`) → `out_golden/fits/<rep>/`.
2. **score**: `Runner.score` (049 v2, `EVAL_SPLIT=as_run`, as the run) on held-out strongest and mid-band for `r0`,
   `r1`, and the run's own H100 circuit scored twice (`h100a`, `h100b`).
3. **report** (CPU): `out_golden/report.md` and dict stores `out_golden/stores/{r0,r1,h100}.pt` for 046's
   `compare_stores.py`.

**Gates.**

| Check | Rule |
|---|---|
| Local noise floor | `r0` vs `r1` per target: Jaccard, max \|dα\| (046 semantics: intersection) |
| Memberships / α, HEAD vs H100 | per target EXACT, else EQUIV (Jaccard ≥ 0.98 and \|dα\| ≤ 1e-2), else FLOOR (both refits no farther from the H100 circuit than from each other: J ≥ J(r0, r1) − 0.02 and \|dα\| ≤ max(1e-2, 2·dα(r0, r1))), else DIFFERENT. **Pass: no DIFFERENT.** |
| `_tk` eval rows | `h100a` vs `h100b` must be identical (same circuit, same GPU). `h100a` vs stored rows (scorer RTX vs H100), `r0` vs stored, `r0` vs `r1`: field counts exact / ≤1e-6 / ≤1e-3 / larger and DAN-8 verdict flips |

Bitwise equality with the H100 circuits is not expected: a local RTX fit differs from an H100 fit at float level, and
the mask's member set is sensitive to that (memory: ~16% membership non-uniqueness under reruns). The local
`r0`-vs-`r1` floor is the reference; the H100 rows are the looser one. Stored α are rounded to 4 decimals
(`learned_mask.py:1738`), so "exact" α means equal at 1e-4.

```bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python experiments/068-validation-gates/golden_turingllm.py
# then, optionally, 046's own verdict line:
.venv/bin/python experiments/046-fit-profiling/compare_stores.py experiments/068-validation-gates/out_golden/stores/r0.pt experiments/068-validation-gates/out_golden/stores/h100.pt
.venv/bin/python experiments/046-fit-profiling/compare_stores.py experiments/068-validation-gates/out_golden/stores/r0.pt experiments/068-validation-gates/out_golden/stores/r1.pt
```

Runtime on the RTX 5070 Ti (one process; the GPU must be free, 062's `stable_core.py` was using it at the time): ~3 min
start-up, 64 fits at ~40–60 s (062's local timings: fit 23–43 s, deep targets slower), 256 scorings at ~3–6 s →
**~1.3–1.5 h**. Resumable (`PHASES=fit,score,report` select parts; `LIMIT=2` for a smoke test, ~6 min).
`PHASES=select,report` were dry-run on CPU (selection written; the report builds, compare_stores reads the stores).

## Open decisions

1. **Gate pricing (engine vs 033).** The engine prices σ(θ); 033 priced the annealed gate σ(θ/T).
   - The engine is the objective the 15k run and protocol v1 used, so changing it would break comparability with
     those.
   - Recommendation: keep the engine as is, record the difference, and judge DAN-36 on functions (verdicts,
     scores, nulls), not on 033's sizes.
   - If σ(θ/T) pricing is wanted, it is a src change in `learned_mask.py` (`mask_sum`, `off_member`) plus a
     λ re-calibration. It is not a bug fix.
2. **Panel gate thresholds** (proposed above): ±3 passes of 22, ≥ 80% verdict agreement, median size ratio within
   [0.5, 2], 0 null passes.
3. **Native kernels and public-model precision.** At float32, no native kernel runs, so "on vs off" is vacuous.
   - If the H100 public runs stay float32 (033 / 053 precision), that gate is moot.
   - If they move to bfloat16 for speed, run `native_onoff.py` first. Its bfloat16 rows are the real test.

