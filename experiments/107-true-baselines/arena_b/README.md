# 107 arena B: CircuitLasso (re-implementation) vs WCM on GPT-2 small + OAI v5 32k resid-post SAEs

Linear: DAN-193 (setup), parent DAN-188. Spec: `../specs/circuitlasso.md` (the method, every ambiguity A1-A19 and the
fidelity checks FC1-FC7; this README does not repeat it). Status 2026-11-03: **plan only, nothing built or run.**
Target: pod-ready by Fri 2026-11-06 for the Sat 2026-11-07 8×H100 pod, as a separate stage after the GPT-2 main chain
(DAN-206).

## 1. The method and where it lives

- **Paper:** Yin, Wei, Gao, Dhurandhar, Natesan Ramamurthy, Yu, "Scalable Circuit Learning for Interpreting Large
  Language Models", arXiv 2606.16939. v2 (2026-10-09) = EMNLP 2026 main; v1 = ICML 2026 MechInterp workshop.
  Re-checked 2026-11-03: arXiv API still lists v2 as the latest version.
- **Code: none released.** Re-checked 2026-11-03: `gh search repos circuitlasso` / `"circuit lasso"` (0 hits),
  `gh search code CircuitLasso` (only third-party notes: jancompute/library, zhaoyang97/Paper-Notes), first author
  `naiyuyin`'s repos (d3m-primitives, ICDH, homepage, rpi_d3m_primitives: none relevant), web search (paper pages
  only). The spec's 2026-11-01 check also covered the IBM org, IBM Research page, ICML page and HF papers.
  **So the row is "CircuitLasso, re-implementation (ours)".**
- **Published setting used here** (spec §2-§4, C.4 of the paper):
  - GPT-2 small; OAI `v5_32k` resid-post SAEs (TopK 32, LN on the input, `b_dec` subtracted) at all 12 block outputs.
    The paper's D.1 says "ReLU / L1"; that is wrong, the Table 5 / 8 codes reproduce exactly on v5_32k (FC1, done
    2026-11-01). Paper layer ℓ = our `blocks.(ℓ−1).hook_resid_post`.
  - Data: CoLA train, 8,551 sentences, raw, `<|endoftext|>` prepended.
  - **"Token-averaged":** for each sentence, each layer's 32,768-dim code is averaged over all T positions *including
    the BOS* (denominator n_tokens + 1); one row per sentence. So Z_ℓ is 8,551 × 32,768, one observation per prompt,
    no positions anywhere.
  - **"Consecutive layers":** Eq. 5 is fitted only for pairs (ℓ, ℓ+1), ℓ = 1..11: every layer-(ℓ+1) feature is
    Lasso-regressed on all 32,768 layer-ℓ features (11 × 32,768 single-output Lassos; separable, so per-column = the
    multi-output optimum). Nothing skips a layer; attn / MLP / resid_mid are not nodes.
  - Lasso: ½‖y − X̃β‖² + λ‖β‖₁ (sum, not mean), predictor columns scaled to unit ℓ2 norm (no centring), coefficients
    rescaled back, no intercept, response raw; FISTA, tol 1e-6 relative objective decrease, warm-started.
  - Eq. 6 (feature → label, ℓ1-logistic on Z_12) with λ_y = 1e-5; used only for the fidelity checks FC3 / FC4 / FC6.
  - λ_F for Eq. 5 is **not reported** (A1); the circuit read-out is **hand-picked** (A7). Both gaps are filled by our
    labelled rules (§4).
- **DAN-193's description is out of date** on two points (the spec corrects them): the data are CoLA sentences, not
  "thousands of corpus windows"; and the columns are unit-ℓ2 scaled, uncentred ("standardised" would imply centring).
  Suggest Daniel updates the issue text.

## 2. Download inventory

Everything except CoLA is already on disk (checked 2026-11-03):

| item | where | status |
|---|---|---|
| GPT-2 small (`gpt2`: model.safetensors, tokenizer, configs) | WSL `~/.cache/huggingface/hub/models--gpt2` (526 MB) | **on disk** |
| `jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs` @ `21df74db`, layers 0-11 (cfg.json + sae_weights.safetensors each) | WSL HF cache (2.3 GB) | **on disk**, all 12 layers |
| the other three GPT-2 kinds (attn-out, mlp-out, resid-mid), needed only by the WCM side | WSL HF cache (2.3 GB each) | **on disk** |
| GPT-2 local data-pass stores (top / mid / neg ctx, latent_stats, seq_repr) for the local smoke's records | `outputs_public/gpt2-small` (3.4 GB) | **on disk** |
| solver | `experiments/098-circuitlasso-baseline/lasso_core.py` (FISTA, sklearn-checked) | **on disk**; needs two modes (§3) |
| Python stack | project WSL `.venv`: torch 2.8.0+cu128, transformers 5.1.0, sklearn 1.9.0, scipy 1.18.1, pyarrow 23.0.0 | **on disk**; `datasets` is not installed and is not needed (pyarrow reads the parquet) |
| **CoLA (GLUE)**, HF `nyu-mll/glue` @ `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` | `cola/train-00000-of-00001.parquet` 251,124 B; `cola/validation-00000-of-00001.parquet` 37,551 B; (`cola/test-…parquet` 37,719 B, labels hidden: not needed) | **to download, 288,675 B (≈ 0.3 MB)**, needs Daniel's yes |

- No new environment, no third-party clone and no new weights. The pod already has GPT-2 and its SAEs (the main
  chain's GPT-2 track); the CoLA parquet files (0.3 MB) are copied with the arena B folder.
- FC2 is already confirmed from the HF datasets-server metadata (no download): train 8,551 rows, 6,023 acceptable
  (70.44%); validation 1,043 rows, 721 acceptable (69.13%). Both match the Table 9 majority rates (spec A5).
  Train sentences: 40.7 characters mean, max 231, so roughly 10 tokens + BOS per row (~95k positions in all).
- **Substrate sharing:** the substrate is exactly our production GPT-2 family (`src/model/families.py` `gpt2-small`,
  jbloom v5 32k), so arena B needs no WCM adapter: the WCM side is the pod's own GPT-2 run (the 070 adapter). It does
  NOT share arena C's Pythia / Marks substrate (the paper's Pythia use is the one-layer BiB debiasing demo, out of
  scope, spec §6.4).

## 3. Build: entry points (all new, in this folder; nothing in src/, 062, 082 or third_party changes)

| file | what | GPU |
|---|---|---|
| `cola_data.py` | read the CoLA parquet (pyarrow), tokenise with BOS, run GPT-2 + the 12 resid-post SAE encoders through the src family loader (`model.families` / `sae.public_bank`), average codes over the true length (padding excluded, A16) → `out/data/Z_{0..11}.pt` sparse CSR [M, 32768] fp32, plus `y.pt`, `split.json`. Both train (fit) and validation (their-eval held-out). | optional; CPU ≈ 60 TFLOP of SAE encodes, est. 10-20 min |
| `cl_core.py` | thin layer over `098/lasso_core.py`, imported unchanged: (a) **no-intercept mode** (uncentred design, raw y, m = 0; implemented here by passing a centring-off design, or as a small subclass); (b) **ℓ1-logistic FISTA** for Eq. 6 (mean / sum CE × intercept on / off, the FC3 variants); (c) a **blocked multi-output solver** on the Gram matrix G = X̃ᵀX̃ over the active columns (p ≈ 15-25k per layer, fp32, 1-2.5 GB) for the pod, columns in blocks of 2-4k, warm-started along the λ_F grid; per-column stopping at tol 1e-6. | CPU (tests, smoke columns), GPU (full fit) |
| `cl_fit.py` | fit A_{ℓ,ℓ+1} for ℓ = 0..10 (our block indices) over the λ_F grid; resumable per (pair, column block, λ); writes sparse Â per λ to `out/fit/A_{l}_{l+1}.lam{λ}.pt`; `--columns` restricts to the columns a root set's closure needs (exact by separability), for the local smoke. Also Eq. 6 (`--eq6`, FC3). | GPU on the pod |
| `cl_readout.py` | latent-rooted circuits (labelled ours, spec A7): **R1** = backward ancestor closure of the root over the non-zero pattern of {Â} at each λ_F, down to block 0; **R2** (sensitivity) = top-k parents per node by \|Â\|, recursively, k ∈ {1, 2, 3, 5, 10, 20}. Members = distinct (resid site, latent); native count = the same (positions never exist). | CPU |
| `cl_checks.py` | FC1 (unit test on the Tab. 5 / 8 codes, BOS on passes, BOS off fails), FC2, FC3 (Eq. 6 variants vs Tab. 9 / Fig. 5), FC4 (the 8 named edges, ≥ 6 in their child's top 10), FC5 (no-intercept FISTA = sklearn `Lasso(fit_intercept=False, alpha=λ/M)` to 1e-4; block = per-column), FC6 (optional Fig. 4 shapes). Writes `out/checks.json`. | CPU (FC1 needs one forward of a few sentences) |
| `score_cl_ours.py` | CL circuits under **our** tests: 083 `cmp_common.make_circuit` + `driver.Runner.score(…, "strong")` on the GPT-2 records, α = 1, exactly as 098's `lasso_rows.py` scores its member sets; both scopes (§4.3). | GPU |
| `eval_theirs.py` | **their** evaluation, generalised to a latent root (labelled our adaptation): (i) the surrogate edge ablation (zero an edge's coefficient in the linear chain, D.5 / A10); (ii) Marks-style node faithfulness / completeness in the real model on CoLA, m = the root's token-averaged code, non-members at the 11 resid sites below the root mean-ablated to their CoLA means (pooled over tokens; CoLA has no fixed template), SAE errors kept. Scores CL and WCM circuits alike, on CoLA train (as published) and validation (ours). | GPU (light) |
| `curves.py` | arena C's `curves.py` pattern: both methods × both evaluations, MIB-style nAUC on the 1-2-5 log grid, ≤ k reading, paired bootstrap over targets (`specs/evaluation.md`). | CPU |
| `test_cl.py` | CPU tests: planted sparse chain recovered by no-intercept mode; FC5 equivalence; block = per-column; closure on a toy 3-layer Â; R2 top-k; BOS-inclusive averaging with padding; the FC1 values (skips if weights absent). | CPU |
| `run_arena_b.sh` | stages `data → checks → fit → readout → score_ours → eval_theirs → curves`, resumable markers, `STOP_AT` for local runs, `CUDA_VISIBLE_DEVICES` / `SHARD` for the pod; arena C's `run_sfc.sh` as the template. | — |

## 4. Design decisions (labelled; open ones flagged for Daniel)

### 4.1 Targets

- CircuitLasso's node universe is the 12 resid-post sites, so **only resid targets** apply. A block-0 target has no
  parent layer (the embedding is not a node), so: the pod's GPT-2 main targets at `resid` blocks 1-11, about
  11 × 10.9 ≈ 117 of the 512 (the frame draws 10-11 per site, `082/make_targets_public.py`). attn / mlp / resid_mid /
  resid-0 targets are "not applicable" for this row.
- Report the rate of targets that **never fire on CoLA** (empty response column → empty circuit). These are a
  property of the published data choice, not dropped. Optional (ours, labelled): the same fit on our OWT windows.

### 4.2 The λ_F grid and the paper-like point

- One absolute λ_F per fit, shared by every column (the paper uses one λ). Grid: geometric, ~3 per decade, chosen in
  the local smoke so that R1 closures span ~10 to 10⁵ latents on the smoke roots, then **frozen before the pod**
  (training data only). Expected range from the scales: ‖y‖ ≈ √M · 0.3 ≈ 30, so λ_max ≲ 30 down to ~1e-3.
- Mark the "paper-like" point λ_F = 1e-5 under the convention FC3 calibrates (if Eq. 6 is a mean CE, that is
  1e-5 · M ≈ 0.086 in Eq. 5's sum convention). Spec A1.

### 4.3 Scope under our tests (**open, for Daniel; DAN-193's "decide"**)

Our scorer ablates every non-member at every upstream site of the target (`upstream_sites`, all 4 kinds). A CL
circuit has resid members only, so under the native 4-kind scope its attn / mlp / resid_mid codes are all ablated:
a large, structural penalty. Options, both cheap:
- **(a) native scope** (WCM's published scope; CL's non-resid sites ablated). Always reported.
- **(b) resid-only scope** for both methods (matched node universe): our scorer with the non-resid sites left at
  clean values, via a runtime wrapper of `upstream_sites` in `score_cl_ours.py` (no src change, arena C's
  `register()` pattern), plus a **WCM arm fitted with resid-only sites** (same wrapper around the driver,
  `driver_resid.py`). Labelled ours.
- Recommendation: report (a) and (b); (b) needs the WCM resid-only arm on the ~117 targets (cost §6).

### 4.4 Other labelled choices

- BOS: CL's data use BOS (faithful); our records use `prepend_bos=False` 64-token OWT windows. Each method is fitted
  on its own data and scored on the same records. Stated.
- Size = distinct (site, latent); for CL the native count is identical (no positions). Both columns still written.
- α = 1 only (CL places no coefficients in the model). The DAN-43 "fitted α" variant would be ours and is not run.
- Held-out: CL never sees our records, so our tests are held-out by construction. Their eval on CoLA train (as
  published) and CoLA validation (ours, labelled).
- The WCM curve: rule 3 needs a WCM λ sweep, not one point. The main chain fits these targets at the adopted λ only;
  the arena B stage adds the resid targets at λ ∈ {3e-4, 3e-3, 1e-2, 3e-2} (plus the main λ), production version.

## 5. Fidelity gate (before the pod)

| check | pass rule | can run on CPU tonight-style |
|---|---|---|
| FC1 data pipeline | Tab. 5 / 8 to 4 dp with BOS, fails without (already passed by hand 2026-11-01; becomes a test) | yes (few sentences) |
| FC2 split / label rates | 8,551 / 70.44%; 1,043 / 69.13% | **already confirmed from HF metadata** |
| FC5 solver | no-intercept FISTA = sklearn to 1e-4; block = per-column | yes |
| FC3 Eq. 6 calibration | Tab. 9 accuracies ± 1.5 pt, ‖A‖₁ at 1e-5 ± 15% of 9,257.6, ≥ 4 of Fig. 5's top 5 | yes (one 8,551 × ~20k sparse logistic path; minutes) |
| FC4 named edges | ≥ 6 of 8 named parents in their child's top 10 at the paper-like λ_F | yes, if fitted on just those 6 child columns (exact) |
| FC6 Fig. 4 shapes | optional; report, never tune toward it | GPU (light) |

If FC3 fails for every variant, the row is frozen as "re-implementation, Eq. 6 not reproduced" (spec FC3). FC4 is
soft by design (hand-picked parents, unknown λ_F).

## 6. Local smoke and pod stage

**Local smoke (CPU first; GPU only in an approved overnight slot):**
1. `cola_data.py` on CPU (all 8,551 + 1,043 sentences, ~10-20 min est.), FC1/FC2/FC5 tests.
2. FC3 on CPU; FC4 on the 6 child columns.
3. Closure growth: 3-5 resid roots from the 104 GPT-2 cross-model panel, column-restricted fits (`--columns`) layer by
   layer down from the root, at 4-5 λ_F values; record closure sizes per λ and per-column FISTA time → freezes the
   grid and the pod timing.
4. Score those circuits under our tests on the local GPT-2 stores (`outputs_public/gpt2-small` records) — this needs
   the GPU, so in an approved overnight slot (≤ 15 min), not while tonight's job runs.
5. Dry run of `run_arena_b.sh` with `LIMIT=2`.

**Pod stage** (separate, after the GPT-2 main S9; one process per GPU):
- `fit`: 11 pairs × ~5 column blocks over 8 GPUs. Estimate (not measured): Gram G·B per FISTA iteration ≈
  2·p²·c ≈ 3e12 FLOP for p = 20k, c = 4k → ~0.05-0.1 s fp32 on an H100; ~10⁴ iterations over the warm-started grid →
  ~10-15 min per block, **~8-15 GPU-h total, ~1-2 h wall** (×3 uncertainty until step 3 above measures it).
- `readout` (CPU, minutes), then `score_ours`: ~117 targets × ~10 λ × (R1 + 6 R2 points) ≈ 2,000 scorings × 2
  scopes at ~5 s (the 062 pod t_eval 1.3-5.4 s by layer) → **~5 GPU-h**.
- WCM λ sweep on the resid targets: 117 × 4 λ ≈ 470 fits at ~45 s on an H100 (local GPT-2 ~84 s per v2 fit, v3 ×1.5,
  H100 ÷2.7) → **~6 GPU-h**; resid-only arm (if 4.3 (b) is chosen) another ~6 GPU-h.
- `eval_theirs` + `curves`: < 1 GPU-h.
- **Arena B total ≈ 20-30 GPU-h, ≈ 3-4 h wall on 8 GPUs, ≈ $60-100**, to be re-projected from smoke step 3.
- Inputs it needs from the main chain: the GPT-2 track's records (`OUT/ctx`) and S9 circuits; output tar + md5 to
  `RESULTS` like every chain stage.

## 7. Risks to "pod-ready by Friday"

1. **Pair-fit cost is unmeasured** (biggest unknown): if FISTA needs far more iterations at small λ_F (near-collinear
   token-averaged codes, BOS-carried near-constant columns), the full 11-pair fit could be several times the estimate.
   Mitigation: column-restricted fits (exact) for only the closures of the ~117 roots, measured in smoke step 3.
2. **Closure blow-up**: at small λ_F, R1 can cover most of each layer within a few hops, so the curve may jump from
   ~10² straight to ~10⁵ with few points between. R2 (top-k) is the planned fallback reading.
3. **Many GPT-2 resid targets may be silent on CoLA** (OWT-drawn latents, short grammatical sentences): empty
   circuits. Reported as a rate; it is the published data choice.
4. **The scope decision (4.3) is Daniel's**; option (b) needs a small runtime wrapper and a WCM arm, ~1 day of code +
   tests. Only (a) is needed for a minimal row.
5. **Calendar:** ~1 day of build (as the 107 README estimated) + one overnight GPU slot for smoke step 4, competing
   with the arena A builds and the Wed decision on the main fit additions. FC3 may not reproduce (logged, not blocking).
6. CoLA download needs Daniel's yes (0.3 MB).

## 8. Sources

- Paper: https://arxiv.org/abs/2606.16939 (v2); HTML https://arxiv.org/html/2606.16939
- ICML workshop page: https://icml.cc/virtual/2026/79543; IBM: https://research.ibm.com/publications/scalable-circuit-learning-for-interpreting-large-language-models
- CoLA: https://huggingface.co/datasets/nyu-mll/glue (tree `cola/`, revision `bcdcba7`)
- SAEs: https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs
