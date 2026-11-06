# 106: speed profile of the v3 WCM fit, and prototyped wins (2026-11-01, agent profiler)

Goal: find measured, easy wins on the ~165 GPU-h pod run (8×H100, ~22 h, ~$560) before launch. Fitting dominates.
Nothing in `src/`, the driver, `chain.sh` or the protocol was changed. Every prototype is an in-memory patch in this
folder (`fastlm.py`: exact string replacements of `learned_mask.py`'s text, each one a proposed adoption hunk).

STATUS (09:45): the GPU came free at 09:19 (sibling agent prepod's queue), so only a minimal GPU session ran before
the 10:00 stop. **Measured:** base vs the "exact" bundle on the 3 panel targets (below) and a flag attribution on the
shallow target (§4). **Not run:** the torch.profiler breakdown (`profile_fit.py`, ready), batch4 / compile_sp /
triton_topk / no_cfeval benches (ready: `run_quick.sh`).

### Measured (RTX 5070 Ti, 400-step v3 fits, 104 panel; `out/bench/compare.json`)

| target | base t_fit | base == 104 circuit | bundle t_fit | speed-up | bundle n (base n) | Jaccard vs base |
|---|---:|---|---:|---:|---|---:|
| 2.attn.11852 | 34.1 s | **identical** | 27.1 s | ×1.26 | 332 (320) | 0.663 |
| 6.mlp.36361 | 89.7 s | **identical** | 84.2 s | ×1.07 | 774 (702) | 0.629 |
| 9.resid.22205 | 132.9 s | **identical** | 127.4 s | ×1.04 | 1440 (1452) | 0.654 |

Bundle = early_exit + donor_gpu + rank_nosync. **It is NOT result-preserving**: membership changes on all three
targets (Jaccard ~0.65, max |Δθ| 3.4-6.1, loss curves differ). That is a bug in one of the prototype patches (they were
meant to be arithmetic-neutral; the CPU engine tests do not see it because their stub has no donor RNG timing / GPU
copies), not a property of the idea. §4 has the attribution. Base itself reproduces 104's circuits exactly, so the
harness is faithful.

### Flag attribution (2.attn.11852, 400 steps, `logs/bench_attr.log`)

| variant | t_fit | speed-up | circuit vs base / vs 104 | raw θ, ψ | loss curve |
|---|---:|---:|---|---|---|
| base | 34.1 s | — | — / identical | — | — |
| **early_exit** | 28.6 s | **×1.19** | **identical / identical** | max \|Δ\| 0 | identical |
| **donor_gpu** | 31.6 s | **×1.08** | **identical / identical** | max \|Δ\| 0 | identical |
| rank_nosync | 33.6 s | ×1.01 | **changed** (n 332 vs 320, Jaccard 0.66) | max \|Δθ\| 3.4 | differs |

So the change comes entirely from `rank_nosync` (the topk-with-global-max-k replacement; I expected identical sorted
values and have not found why it differs — not debugged, out of time). It also buys almost nothing: **drop it**.
early_exit and donor_gpu are bit-identical on this target. Their gains on the mid / deep targets were not measured
separately; the bundle's (with rank_nosync's ~1%) were ×1.07 (L6) and ×1.04 (L9), consistent with early exit saving
the layers after the target (more of them for shallow targets).

### Confirmation run (09:45-09:53, `VARIANTS="base;early_exit,donor_gpu"`, all 3 targets; `logs/bench_confirm.log`,
`out/bench/compare.json` now holds this run's comparison)

| target | base t_fit | early_exit + donor_gpu | speed-up | members / α (4 dp) vs base | raw θ / ψ vs base | loss curve | vs 104 circuit |
|---|---:|---:|---:|---|---|---|---|
| 2.attn.11852 | 34.1 s | 27.7 s | **×1.23** | identical (320 = 320, Δα 0) | 12 / 12 tensors exactly equal | identical | identical |
| 6.mlp.36361 | 89.7 s | 84.8 s | **×1.06** | identical (702 = 702, Δα 0) | 38 / 38 exactly equal | identical | identical |
| 9.resid.22205 | 132.9 s | 127.9 s | **×1.04** | identical (1452 = 1452, Δα 0) | 58 / 58 exactly equal | identical | identical |

**Bit-identical on all three targets.** Sum of the 3 fits: 256.7 → 240.4 s (×1.07). Weighting the measured
speed-ups over the 8k's uniform layer mix (shallow ×1.23, mid ×1.06, deep ×1.04, per-layer fit times as §1) gives
~×1.06-1.07 on TL fit time (E): ~4-5 of ~72 TL fit GPU-h. Public models not measured.

## 4. Recommendation

- **Adopt before the pod (bit-identical, small hunks):** W1 early_exit and W2 donor_gpu (`fastlm.py` hunks
  `early_exit.*`, `donor_gpu.*`). **Confirmed bit-identical on all 3 targets** (members, α, raw θ / ψ, loss curve,
  vs base and vs 104); measured ×1.23 (L2), ×1.06 (L6), ×1.04 (L9) on one local process. Estimate (E): ~×1.06-1.07 on
  TL fit time → ~4-5 GPU-h; if the public adapters gain alike (not measured) ~2-3 GPU-h more → **~6-8 GPU-h,
  ~0.8-1 h wall, ~$20-25** (lower than the 09:45 estimate, which predated the mid / deep confirmation). Under MPS P=6
  the donor_gpu part (sync removal) may gain less than locally, since other processes fill the idle GPU (E).
- **Adopt (CPU, bit-identical):** W8 memoised 8-gram predicate in the T2 verify: 2.31 → 0.30 s per record on the
  CPU predicate part (83,820 calls exactly equal).
- **Do not adopt:** rank_nosync (changes results, ~1%).
- **Not measured, scripts ready (`run_quick.sh`, ~45 min):** batch4 / batch2 (W4, float-noise + memory risk),
  compile_sp (W5), triton_topk (W6), no_cfeval (W7), and the torch.profiler breakdown (`profile_fit.py`). I would
  not adopt any of these without that run; W4 is the one with the largest plausible gain (one forward instead of four),
  but it changes numerics and multiplies the live graph.

## Files

| file | what |
|---|---|
| `h106.py` | builds the 062 driver's `Runner` exactly as production (WCM v3, `CTX_DIR` = 062 records, explicit split); `fit_one` = `Runner.fit` + timing / peak memory / final raw parameters |
| `fastlm.py` | prototype engine: patched copy of `src/circuit/instrument/learned_mask.py` (flags below), installed over `run_learned_mask` |
| `profile_fit.py` | per target: a timed short fit (sync per step) + a torch.profiler window (CPU+CUDA, record_shapes, with_stack) with labels on every engine piece; kernel time by term / piece / kernel category, syncs and their call sites |
| `bench.py` | full 400-step fits of the 104 panel per variant; `compare.json`: speed vs base, circuit / parameter / loss-curve identity vs base and vs 104's bit-identity circuits |
| `run_engine_tests_cpu.py` | src's 173 learned-mask engine tests run against the patched engine |
| `verify_fast.py` | T2 record verify (094) with a memoised 8-gram predicate; per-call exactness check |
| `run_all.sh` | the GPU session (profile base → bench → profile variants), resumable |

## 1. Where the time goes (measured without a GPU of my own: existing clean logs)

Per TuringLLM target, the T3 main job (prepod's clean local timing panel, 104 `out/timing_tl`, 12 targets, one per
layer, RTX 5070 Ti, nothing else running):

| version | fit s (sum of 12) | eval s (strong + mid) | spec s | fit share |
|---|---:|---:|---:|---:|
| v1 | 514.9 | 64.0 | 12.1 | 87% |
| v2 | 662.9 | 63.7 | 12.2 | 90% |
| **v3** | **986.2** | 64.1 | 12.2 | **92%** |

v3 fit by layer (s): L0 16, L1 26, L2 41, L3 46, L4 63, L5 80, L6 77, L7 101, L8 118, L9 123, L10 140, L11 156: about
13 s per layer of depth plus ~10 s. Public models, 104 part B (24 targets each, local): Pythia-70m v3 fit 44.9 s mean
(v2 31.6), GPT-2 small v3 **162.3 s** mean (v2 108.2), eval 1.4 / 4.4 s. The 062 15k run on the pod (v1, H100 MPS P=6):
fit 74.3 s, eval 3.4 s, spec 0.66 s per target per process, the same depth slope (L0 20 s → L11 122 s).
So the fit is >90% of every fitting stage, and everything below is about the fit.

### 1.1 Structure of one v3 fit (code reading, `learned_mask._run_learned_mask_impl`)

- **400 steps × 4 data terms** (Z, C, A, R), each its own grad forward + backward on the same 4 training rows ×
  64 tokens (`micro_bs` 4, `deep_site_threshold` 99, so no accumulation).
- Each term forward runs the **whole 12-layer model** (`inference.forward` → `model(tokens)`, logits of the last
  position) although the loss reads only the seed pre-activation at the seed's site: for a layer-l target the
  layers after l are pure waste (no masked site there, nothing reads them).
- At every masked site (all upstream sites, up to 35): SAE encode (bf16 gemm 256×1024×40960, relu, **torch.topk
  k=128 over 40960** in grad mode), densify ([4, 64, 40960] bf16), the code (C / A: `sparse_keep_mask` = argsort of
  the 40960 means + ~8 full-size bool passes; code + straight-through = ~7 full-size passes; R: `DonorPool.get` +
  `donor_keep_mask` = an fp32 densify + **a second topk over [256, 40960] fp32**), decode (gemm 256×40960×1024).
  The backward repeats the two gemms (grad wrt input only; the backbone is frozen) and ~10 full-size passes.
- Host syncs inside the loop (each stalls the GPU queue): `DonorPool.get` indexes CPU tensors and copies 2 pageable
  tensors H2D **per site per R forward** (≈70 per step at depth); rank-keep "keep" mode calls `int(_r.max())`
  (D2H) and `torch.tensor([_sd], device=...)` (H2D) **per Z / C / A term per step** (6 per step).
- Once per fit (outside the loop): natural forward, the scorer's A / C means (2 forwards of 48), the closed-mask
  diagnostics (2), the rank-keep clean references (9 small forwards), the donor capture (3 forwards), the held-out
  forward, and in `discover()` the post-fit counterfactual-faithfulness eval (mask modes never reject on it; it fills
  two metadata fields).

## 2. Candidate wins

Each with: measured (M) or estimated (E) effect, effect on results, effort, risk. Prototype = flag in `fastlm.py`
(bench name in brackets).

| # | win | expected effect on results | status |
|---|---|---|---|
| W1 | **early exit at the seed's site** [`early_exit`]: the seed tap raises once `seed_pre` is captured; `_forward_preact` catches it. Saves the forward of every layer after the target's (all 4 terms, all 400 steps, plus every setup forward through `_forward_preact`) | none expected: the same kernels on the same inputs up to the seed; nothing after it is read | prototyped; CPU engine tests 173/173 pass; GPU bench pending |
| W2 | **donor codes on the GPU** [`donor_gpu`]: `DonorPool` keeps the 48 donors' codes on the device (~5 MB per site) and moves the drawn row indices once per draw (pinned, non-blocking) | none expected: the same values gathered on the device instead of the host | prototyped; tests pass; GPU pending |
| W3 | **no per-term syncs in rank-keep** [`rank_nosync`]: `topk(k = max clean rank over all anchors)` computed once instead of `int(_r.max())` per term; the seed index tensor built once | none expected: topk values are sorted, so the r-th value is the same for any k ≥ r | prototyped; tests pass; GPU pending |
| W4 | **batch the 4 terms** [`batch4`, `batch2`]: one model forward / SAE encode / decode per site for the stacked rows [Z | C | A | R], one backward | float-noise level: gemms at 4× the rows (other cuBLAS kernels), theta's gradient summed inside one backward; membership may churn at the margin. Memory: ~4× (2×) one term's graph; deep targets may not fit P=6 on 80 GB | prototyped; GPU pending |
| W5 | **compile the sparse-fill chain** [`compile_sp`]: inductor (emulate_precision_casts) fuses the keep mask (after the eager argsort; integer / bool ops, exact) and the code + straight-through tail | forward rounding emulated; backward reductions may sum in another order: float-noise level | prototyped; GPU pending |
| W6 | **Triton top-k in grad mode** [`triton_topk`]: indices from the existing radix-select kernel, values gathered (differentiable) | identical unless the 128th value ties in bf16 (then another tied latent may be kept) | prototyped; GPU pending |
| W7 | **skip discover()'s post-fit cf eval for mask modes** [`no_cfeval`] | membership / α untouched by construction (mask modes never reject); circuit.metadata loses `counterfactual_faithfulness` / `posctx_suppression_score` (read only by old analysis tools, not by 062 / 083 / 086 / 094 / 104) | prototyped; GPU pending |
| W8 | **T2 verify: memoised 8-gram predicate** (`verify_fast.py`) | identical (every one of 83,820 `_c8_vs` calls equal; failure lists equal) | **M: CPU predicate part 2.31 → 0.30 s per record (×7.8)** on the 10 smoke records |
| W9 | sparse custom backward for the masked sites (decode / encode backward only need the column sums and the k active entries per row, not dense [256, 40960] gradients) | float-noise level (summation order) | not prototyped: large effort (custom autograd + kernel), listed for after the pod |
| W10 | CUDA graphs / whole-step torch.compile | — | not prototyped: per-fit Python patcher closures, the CPU donor RNG and the step's data-dependent shapes (keep-mask budgets) make capture a rewrite, not an easy win |
| — | dtype / TF32 | — | nothing to gain: the model, SAEs and codes already run in bf16; `set_float32_matmul_precision("high")` is set; the only fp32 tensors are the parameters ([d_sae] vectors) and the R term's donor densify (W2's neighbour; part of `donor_keep_mask`) |
| — | data loading | — | records are ~170 KB `.pt` per target, loaded once per target (t_ctx 0.0 s in every status row) |

## 3. What the pod cost becomes

Base: prepod's measured v3 budget (104 `out/cost_v3.json`, scenario r = 1.5): **164.8 GPU-h** full. Its fitting part
(the rows a fit speed-up scales), from the same file:

| rows | GPU-h | fit part |
|---|---:|---:|
| T3 main 8,000 | 54.3 | 52.4 |
| T4 sweep 190 × 12 | 15.0 | ~14.5 |
| T5 stable, T6 nulls, T10 noR | 5.6 | ~4.9 |
| public: Pythia (r 1.5) | 14.3 | ~12.4 (local fit-s share 87%) |
| public: GPT-2 (r 1.5) | 35.8 | ~29.0 (local fit-s share 81%) |
| **fit total** | | **~113 of 164.8** |
| not scaled: T1 scan 3.5, T2 records 10.6, α = 1 1.3, T7 DAN-159 15.9 (other methods; its 110 s v3 support fits per target are inside), T11 / T12 donor 6.2, data passes | ~52 | 0 |

Rule: a fit speed-up s saves 113 × (1 − 1/s) GPU-h. Per 10% (s = 1.11): **~11 GPU-h, ~1.4 h of 8-GPU wall, ~$35** at
$3.1 / GPU-h. W8 (verify) acts on T2 only: verify is 4.8 of T2's 10.6 GPU-h in the plan; its CPU predicate is the part
measured here (§2).
