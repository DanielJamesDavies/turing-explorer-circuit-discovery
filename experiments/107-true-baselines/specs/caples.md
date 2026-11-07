# Spec: Caples et al. 2025, learned binary masks over SAE latents ("Scaling Sparse Feature Circuit Finding to Gemma 9B")

Written by `spec-caples` for 107 (proposed arena D, Gemma-2). This file says what the post and the authors'
code do. Anything we would change is labelled **our adaptation**. Nothing was cloned or downloaded: the code was read as
raw text through the GitHub API, and the post through WebFetch.

## 0. Sources, versions, tags

**The work.**
- Diego Caples, Jatin Nainani, Callum McDougall, "rrenaud". "Scaling Sparse Feature Circuit Finding to Gemma 9B".
- An **interim report** of a MATS Winter 7 sprint (Neel Nanda's stream), posted on LessWrong / Alignment Forum on
  **2025-01-10**, post id `PkeB4TLxgaNnSmddg`.
- **No arXiv, OpenReview or workshop version exists.** I checked:
  - the arXiv API for authors Nainani and Caples (16 hits, none on this topic);
  - web searches.
- There is one version of the post. LessWrong keeps no visible revision list. Our bib entry `caples2025scaling`
  matches (paper/references.bib:441-446).
- The post's figures are images. Their plotted values are not in the text. The figure values below come from the
  authors' saved mask files in the repo (§6.3).

**Code: `github.com/NainaniJatinZ/ScalableSAECircuits`.** The post and the README link it. The repo has 4 stars, 2
forks and **no licence**.
- **HEAD = `f54a4efce908432b8771ea671a248ee30ec2b3af`** (2025-01-27, "sae gap code"). 25 commits in total, from
  2024-12-23 to 2025-01-27.
- **Post-date commit = `00998a72803f63291befe2821151bf0dbd1a97a0`** (2025-01-10, "added lesswrong link").
- The main notebook is `ScalableSAECircuits_Colab.ipynb`, the "Main Masking notebook" in the README. It is identical
  at the post-date commit and at `3d2136c` (2025-01-10, "Resetting plots wrt full model perf").
- **The per-task notebooks were renamed on 2025-01-22** (`6f6c910`) from the root into `task_specific/`. Their blob SHAs
  are byte-identical at the post-date commit and at HEAD:
  - `sva.ipynb` `3163996`;
  - `error_pred.ipynb` `d2dd4a6`;
  - `ioi_run.ipynb` `b54f09e`.
  `GenericNotebook.ipynb` changed after the post.
- **Files read** (every code cell, plus the printed outputs of the notebooks):
  - `ScalableSAECircuits_Colab.ipynb`;
  - `task_specific/{sva, error_pred, ioi_run, GenericNotebook, binary_mask_code}.ipynb` (the last two by grep only);
  - `helpers/utils.py` (at HEAD, and its training lines at `11f2817`, `a53824b` and `29e278b`);
  - `task_specific/circ_disc.py`, `interp_bottlenecks.py`, `scripts/sae_gap.sh`, `README.md`, `.gitignore`;
  - the scalar fields of every `masks/**/_run.json` under 200 kB.
- **Things added after the post** (2025-01-16 onwards): `helpers/utils.py` as a shared module, an optional mean / true
  error term, a Gemma-2-2B path, and the SAE-gap sweep (`interp_bottlenecks.py`, `sae_gap.sh`). None of this is in the
  post. When used, it is labelled **post-publication code**.

**Tags.**
- [post §x] = the LessWrong text, by its section numbers.
- [code file:line @sha] = a line in a .py file. Line numbers are from `cat -n` of the raw file.
- [code nb cell N @sha] = a notebook cell, by index from 0 over all cells, as read with `json.load`.
- @HEAD = `f54a4ef`; @post = `00998a7`.
- [inferred] = my reading where the sources are silent.

**Local context read.**
- 107 `README.md`, `specs/evaluation.md`, `specs/sfc.md`.
- `dev-notes/2026-09-28-baseline-audit.md` §4.
- 048 and 050 READMEs, and `048-gemma-tasks/fit_gemma_task.py`.
- 061 README.
- `src/config.py:650-890` and `src/circuit/instrument/learned_mask.py`, both on the unweighted-arm side.
- `paper/resources/related-reading/` holds no Caples notes.

---

## 1. Summary and claims

- **Two proposals** [post TL;DR, §3]:
  1. Put **only a few residual-stream SAEs**, about every 10 layers, instead of SAEs at every layer and site.
  2. Find circuits by **optimising a binary mask over SAE latents with continuous sparsification** (Savarese et al.
     2020), instead of thresholding attribution scores.
- **Error terms:** none are used. The SAE reconstruction replaces the residual stream at each SAE site [post §3.3].
- **Ablation:** a masked latent takes its **per-token (position-specific) mean**.
- **Objective:** the circuit's logit difference is optimised to **match the model's** [post §4.1].
- **Masks are per token position**, because the data is templatic [post §4.1].
- **Claims:**
  - **C1.** Masking recovers more performance with fewer latents than integrated gradients (IG) on all tasks shown.
    On dictionary-key, IG does not reach 50% [post §4.2].
  - **C2.** One circuit reaches about 95% faithfulness with fewer than 20 latents [post TL;DR]. In the repo this is the
    SVA-rc run (§6.3).
  - **C3.** Completeness under SFC-style complement ablation is near 0 for both masking and IG. IG is sometimes slightly
    closer to 0 [post §4.3].
  - **C4.** Masks at different sparsity multipliers are **nested**: latents of smaller circuits nearly always appear in
    larger ones. Ten multipliers were tested, and no numbers are given [post §4.4].
  - **C5.** A case study: Gemma-2-9B's dictionary-key check uses generic duplicate-token latents, and an adversarial
    key/value swap exploits it [post §5].
- **The authors' caveats:**
  - residual-only circuits say what is represented, not how or when it was computed [post §3.1, §6];
  - IOI needs a name latent in every SAE [post §4.2.3];
  - the best number and placement of SAEs is open [post §6].

---

## 2. Substrate

### 2.1 Model

- **Model:** `google/gemma-2-9b` (base, pretrained). It is loaded with sae_lens's `HookedSAETransformer.from_pretrained`,
  a TransformerLens model [code Colab cell 3 @post; code task_specific/circ_disc.py:43 @HEAD].
  - The post says "Gemma 9B" throughout. 2B appears only as a remark: the code tasks are said to need models above 2B
    [post §4.2.1].
  - **The brief's "Gemma-2-2B" is not the published substrate.**
- **dtype:** no dtype argument is passed, so the TransformerLens default **float32** applies [inferred from code Colab
  cell 3].
  - The SVA notebook's `nvidia-smi` shows **45,018 MiB** in use [code task_specific/sva.ipynb cell 3 output].
  - The SLURM script asks for an A100 [code scripts/sae_gap.sh:18 @HEAD].
  - So the published runs were on a ≥48 GB GPU, most likely an A100-80GB [inferred].
- **Load-time notes:** TransformerLens warns that `center_unembed` is disabled (logit softcap) and that writing weights
  can't be centred (no LayerNorm) [code Colab cell 3 outputs]. The model parameters are frozen
  (`requires_grad_(False)`).
- **Logits:** TransformerLens Gemma-2 logits include the final softcap [inferred].

### 2.2 SAEs (published runs)

| | value | source |
|---|---|---|
| release | `gemma-scope-9b-pt-res` (Gemma Scope, JumpReLU) | [code Colab cell 4 @post] |
| layers | **7, 14, 21, 40** (of 42) | same |
| width | 16k (`d_sae` = 16,384) | same |
| L0 variant | `average_l0_` **92, 67, 129, 125** respectively | same |
| hook | `blocks.{L}.hook_resid_post` | [code Colab cell 43 (mask_dict keys)] |
| encode / decode | `sae.encode` / `sae.decode` of sae_lens; JumpReLU threshold inside `encode` | [code helpers/utils.py:263,304] |

- **The same four SAEs are used in every task notebook:** sva, error_pred, ioi_run, Generic and binary_mask_code
  [code grep of all notebooks @HEAD].
- **The post omits the layers, widths and L0s.** It says only "four residual SAEs, about every 10 layers"
  [post §4.1].
- **Post-publication code:**
  - `interp_bottlenecks.py` picks layers `range(0, n_layers, sae_gap)` and, per layer, the 16k SAE whose L0 is closest
    to 100 [code interp_bottlenecks.py:146-172 @HEAD];
  - it supports `google/gemma-2-2b` with `gemma-scope-2b-pt-res` [code interp_bottlenecks.py:141-144];
  - `sae_gap.sh` runs 2B at gaps 6 and 3 [code scripts/sae_gap.sh:4,25-26];
  - `circ_disc.py` uses gap 13 with L0 closest to 32 on 9B (the comment says 100) [code task_specific/circ_disc.py:80,87].

### 2.3 How an SAE site is spliced in (every forward)

From [code Colab cell 7 @post; helpers/utils.py:258-315 @HEAD]:

```
pos_mask[b,t] = (token[b,t] != BOS)                    # BOS positions are never edited
c   = sae.encode(resid_post) * pos_mask[...,None]      # codes, zeroed at BOS
c'  = <mask / ablation transform of c>                 # §3
out = sae.decode(c')
resid_post <- where(pos_mask, out, resid_post)         # reconstruction REPLACES the stream; no error term
```

- **Error handling (published):** the SAE error is **dropped** at all four sites, in every forward: training, the
  "model + SAE" reference, and evaluation [post §3.3; code Colab cell 7].
- **Post-publication options:** `use_error` (true error added back) and `use_mean_error` (an error term at its mean
  over corrupted prompts) [code helpers/utils.py:308-315, 380-402 @HEAD]. They appear only in 2B / gap runs saved
  under `masks/sva/rc_train*_mean_error/` and `rc_trainw_error/`.
- **Positions:** all prompts in a task have exactly the same tokenised length. The data is filtered to
  `len == example_length` [code Colab cell 10]. So position t means the same template slot in every prompt.

---

## 3. The algorithm, exactly

### 3.1 Pseudocode (Colab / error_pred / ioi_run form; variants in §3.2)

```
# setup -----------------------------------------------------------------------------------------------
data  = filter(rows, len(tok(clean_prefix)) == len(tok(patch_prefix)) == L)[:N]      # N = 3000 (Colab)
clean, corr      = to_tokens(clean_prefix), to_tokens(patch_prefix)                   # BOS prepended
y_clean, y_corr  = first token of clean_answer, of patch_answer
batches          = reshape to [n_b, 16, L], drop remainder                            # n_b = N // 16
LD(logits)       = mean_b ( logits[b,-1,y_clean[b]] - logits[b,-1,y_corr[b]] )       # batch-MEAN logit diff

# ablation values: per-position mean of SAE codes over CORRUPTED prompts
for each SAE s: mu_s = zeros(d_sae)
for i in 0..39, j in 0..15:                                  # 40 x 16 = 640 corr prompts (Colab)
    c = codes of corr[i, j] (single sequence -> shape [1, L, d_sae], 0 at BOS)
    mu_s = mu_s + (c - mu_s) / (i + 1)                       # NOTE: divides by the BATCH index (bug, §10)
# mu_s ends with shape [1, L, d_sae]: a position-specific mean

# mask: one logit per (position, latent) per SAE
theta_s = ones(L, d_sae)                                     # init 1.0, a Parameter
T(r)    = 200 ** r                                           # "max_temp" = 200 (class default 1000, overwritten)
gate_s  = sigmoid(theta_s * T(r))                            # T MULTIPLIES theta; T grows during training
c'      = mu_s + gate_s * (c - mu_s)                         # gate 1 keeps c, gate 0 gives mu

# training ------------------------------------------------------------------------------------------------
S   = n_b * portion_of_data                                  # "total_steps"; portion 0.3 (Colab)
opt = Adam(all theta_s, lr = 1.0)                            # plain Adam, no weight decay, constant lr
for i, (x, y, z) in enumerate(batches):                      # one pass, batches in order 0, 1, 2, ...
    r     = (i / S) * 1.1                                    # Python precedence: i/S*1.1
    LD_c  = LD( model with SAEs, soft gate, mean ablation )(x)
    LD_m  = LD( model with SAEs, no mask )(x)                # SAE-spliced model, NO error term
    task  = | LD_m - LD_c |                                  # L1 on the batch-mean difference
    spars = (1/n_SAE) * sum_s sum_{t,k} gate_s[t,k]          # l1 = 1.0; the TEMPERED gate is penalised
    loss  = task + lambda * spars                            # lambda = "sparsity_multiplier"
    loss.backward(); opt.step()
    if i >= 1.1 * S: break                                   # break after processing i, so the last r ~ 1.21

# read-out and evaluation ----------------------------------------------------------------------------------
member_s[t,k] = theta_s[t,k] > 0                             # equivalent to gate > 1/2 at any T > 0
size          = sum_s sum_{t,k} member_s[t,k]                # "total_density": (position, latent) PAIRS
c'            = mu_s + member_s * (c - mu_s)                 # binarised
LD_circ       = mean over batches [0, -1, -2, ..., -9] of LD (circuit)     # batch[-i], i = 0..9
faithfulness  = LD_circ / avg_LD                             # avg_LD = SAE-spliced LD over batches 0..9
```

**Sources for each line.**

- **Mask class:** `SparseMask` [code Colab cell 6 @post; helpers/utils.py:142-181 @HEAD].
  - Parameter init `torch.ones` (:146/:148).
  - `temperature = max_temp ** ratio_trained` (:169).
  - `sigmoid(mask * temperature)` (:170).
  - `sparsity_loss = abs(mask).sum() * l1` (:172).
  - Mean interpolation `diff*mask + mean` (:180-181).
  - Binary read-out `(self.mask > 0)` (:161).
- **Training loop:** [code Colab cell 16 @post; helpers/utils.py:441-552 @HEAD].
  - `max_temp = 200` (:496).
  - `Adam` (:499).
  - `ratio_trained = i / total_steps*1.1` (:508).
  - `loss = abs(model_logit_diff - fwd_logit_diff)` (:463).
  - `sparsity_loss / len(saes)` (:476).
  - `total_loss = loss + sparsity_loss*multiplier` (:522-523).
  - The break at `i >= total_steps*1.1` (:537).
- **The reference forward:** `logitfn(tokens, False, False)` runs the SAE hooks with no mask, i.e. the SAE-spliced
  model with no error [code Colab cell 16; helpers/utils.py:456].
- **Read-out and evaluation:** `eval_logit_diff` with `batch[-i]` [code Colab cell 16; helpers/utils.py:574-589];
  `faithfulness = logit_diff / avg_logit_diff` [code Colab cell 16].
  - `avg_logit_diff` is the SAE-spliced model over batches 0..9 [code Colab cell 12].
  - In `helpers/utils.py` @HEAD, the denominator is instead the **true** model on the same eval batches (:582-584,
    :620).
- **Means:** `get_sae_means(corr_tokens, 40, 16)` [code Colab cell 13]. The single-sequence forward gives a `[1, L,
  d_sae]` mean [inferred from `mean_tokens[i, j]` indexing].

**Nothing else is learned.** There are no per-latent scales, no edges and no thresholds. The circuit is a set of
(SAE, position, latent) triples [code; post §3.2].

**Implementation side effects.**
- The SAE parameters are never set `requires_grad=False`, so their gradients are computed and later cleared
  [code helpers/utils.py:543-545 clears them].
- The reference forward also builds a graph. Neither changes the maths. Both cost memory (§8) [inferred].

### 3.2 Per-task variants (all pre-post; they were used for the post's numbers)

| | Colab (main) | error_pred (code tasks) | ioi_run (IOI) | **sva (SVA)** |
|---|---|---|---|---|
| file @sha | Colab cells 13-18 @post | task_specific/error_pred.ipynb cells 18-26, 60-67 | ioi_run.ipynb cells 18-31 | sva.ipynb cells 21, 30-39 |
| lr (Adam) | **1.0** | **1.0** | **1.0** | **0.05** |
| `r` schedule | (i/S)·1.1, stop at i ≥ 1.1·S | same | same | **i/S, stop at i ≥ 1.3·S** |
| S | 0.3·n_b | 0.3·n_b (0.4 in a one-off probe run) | **0.8·n_b** | **0.5·n_b** |
| task loss target | per-batch SAE-model LD | same | same | **fixed scalar** `avg_logit_diff/10` = 3.514 (SAE-model mean over batches 0..9) |
| sparsity term | (1/n_SAE)·Σ gate | same | same | (1/n_SAE)·Σ gate **/ L** (divided by `example_length`) |
| N prompts | 3000 | 3000 | 2900 | **10,000** |
| mean set | 40×16 corr | 25×16 corr | 25×16 corr | **10×16 corr** |
| λ sweep | 5 values {0.01, 0.0205, 0.0447, 0.0827, 0.1345} | same (+0.1, 0.5, 1 one-offs) | 10 values, 0.0001 to 0.07 | 10 values 0.05 to 0.905 (step 0.095) ∪ {1, 2.05, 4.47, 8.27, 13.45} (+15) |
| faithfulness denominator | SAE-model LD (batches 0..9) | same | same | same (3.514) |

**Schedule ends.** The last step is the first integer i ≥ c·S (c = 1.1 or 1.3). So the final r is about 1.1·1.1 =
1.21 in the Colab form (r = 1.1·i/S) and about 1.3 in the SVA form (r = i/S). I computed them [inferred, arithmetic]:
- **Colab / code tasks at N = 3000:** n_b = 187, S = 56.1, 63 steps, final r = 1.216, **T_end ≈ 627**.
- **IOI:** about 161 steps, T_end ≈ 626.
- **SVA:** n_b = 625, S = 312.5, 408 steps, final r = 1.302, **T_end ≈ 993**.
  - The notebook prints "408it … temperature=tensor(992.7993)" [code sva.ipynb cell 36 output].
  - **This confirms the loop arithmetic.**

**The λ grids** come from `mf(x) = (x² − a²)/(b² − a²)·(b − a) + a` at `x = a + i(b − a)/n`. This is a quadratic
spacing [code Colab cell 17; sva.ipynb cell 38].

**`helpers/utils.py` before the post** (`11f2817`, `a53824b`) used lr 0.05, `r = i/S` and a stop at 1.3·S, the same as
SVA [code helpers/utils.py:491,514,542 @11f2817].
- In that version the `l1` / `distinct_l1` arguments are swapped (:172, :174). This is harmless because both are 1.0.
- @HEAD the module has lr 0.05 with the 1.1 schedule (:485, :508, :537). That is the post-publication 2B
  configuration.

### 3.3 How circuit size is controlled

- **The only knob is λ** ("sparsity multiplier"). One full training run per λ [post §4.4; code Colab cell 18].
- **Size** is counted in (SAE, position, latent) pairs, summed over SAEs [code Colab cell 16 `total_density`].
  - The JSON stores, per SAE, the latent indices of the pairs, with the positions dropped (`torch.where(...)[1]`). So
    the same latent at two positions appears twice.
  - The post's x-axis is "Number of Nodes / Latents" [code Colab cell 32].
- **BOS row:** the BOS position's θ receives only the sparsity gradient (its codes are zeroed and the stream is
  restored), so it is driven negative and never counted [inferred].

---

## 4. Every hyperparameter: post vs code

"—" means the post is silent. Code values are the published-run notebooks @post; HEAD-only values are marked.

| hyperparameter | post | code |
|---|---|---|
| model | "Gemma 9B" [§TL;DR] | `google/gemma-2-9b`, TransformerLens fp32 (default) |
| SAE sites | 4 residual SAEs, about every 10 layers [§4.1] | resid_post L7/14/21/40 |
| SAE release / width / L0 | — | gemma-scope-9b-pt-res, 16k, L0 92/67/129/125 |
| error nodes | none [§3.3] | none (post-pub: optional mean / true error) |
| BOS | — | never edited; codes zeroed at BOS |
| nodes | SAE latents, per token [§4.1] | (SAE, position, latent) |
| mask parametrisation | binary mask by continuous sparsification [§3.2] | θ ∈ ℝ^{L×16384} per SAE; gate σ(T·θ) |
| θ init | — | 1.0 (gate σ(1) = 0.73 at T = 1) |
| temperature | — | T = 200^r, geometric in step; r from 0 to about 1.21 (T_end ≈ 627) or to about 1.30 (SVA, T_end ≈ 993) |
| ablation of masked latents | per-token mean ablation [§4.1] | position-specific mean of codes over **corrupted** prompts (640 / 400 / 160), buggy running mean |
| objective | match the circuit's logit diff to the model's [§4.1] | \|LD_SAE-model − LD_circuit\|, batch means, L1; SVA uses a fixed scalar target |
| metric | logit diff, correct − incorrect [§4.1] | raw logits at the last position, clean answer − patch answer |
| sparsity penalty | "sparsity multiplier" [§4.4] | λ·mean_SAE Σ σ(T·θ) (penalty on the tempered gate); SVA also ÷ L |
| λ grid | 10 masks [§4.4] | 5 (code tasks), 10 (IOI), 15 (SVA) |
| optimiser | — | Adam, constant lr, no weight decay |
| lr | — | 1.0 (code, IOI, Colab); 0.05 (SVA; utils.py @HEAD) |
| batch | — | 16 |
| steps | — | 63 (code), about 161 (IOI), 408 (SVA): one pass over the first 33% / 88% / 65% of batches |
| read-out | binary mask | θ > 0 |
| eval batches | — | batches [0, −1, …, −9], 160 prompts; batch 0 is a training batch |
| faithfulness | "ratio of circuit performance to model performance", citing Wang et al. [§4.2] | LD_circ / LD_SAE-model. The Colab plots rescale it to LD_circ / LD_full-model, with the SAE-model as a dotted ceiling [code Colab cells 28, 32] |
| seeds | — | none set, one run per λ |
| libraries | — | `sae-lens`, `transformer-lens`, unpinned (`pip install`) [code Colab cell 1]; `SAE.from_pretrained(...)[0]` and `cfg.hook_layer` imply sae-lens < 6 [inferred] |
| logging | — | wandb, project "sae circuits" |

---

## 5. Data and tasks

All tasks are templatic clean/corrupt pairs with `clean_prefix`, `patch_prefix`, `clean_answer` and `patch_answer`, all
filtered to one tokenised length L [code Colab cell 10].

| task | file (repo) | L (incl. BOS) | prompts used | train batches | eval batches | source |
|---|---|---|---|---|---|---|
| **SVA across a relative clause** ("The friends that the dancer visits" → " go" / " goes") | `data/sva/rc_train.json` | 7 | 10,000 (625 batches) | 0..407 | 0, 616..624 | sva.ipynb cells 13-18 |
| **Dictionary-key error** (Python code-output prediction) | `data/codereason/key/data_len5_digit1.json` | 41 | 3,000 of 20,000 passing (187) | 0..62 | 0, 178..186 | error_pred cells 53-67; Colab cells 9-10 |
| **List-index error** | `data/codereason/index/data_len6_digit1.json` | 29 | 3,000 (187) | 0..62 | 0, 178..186 | error_pred cells 10-26 |
| **IOI**, BABA, 2 templates | `data/ioi/ioi_train_21.json` (generated with `ioi_dataset.py`) | 20 | 2,900 | about 0..160 | 0, last 9 | ioi_run cells 11-31, 85 |

- **The SVA data are exactly SFC's.** Their `rc_train.json` blob is `83ca7ca`, identical to our
  `data/external/sfc_feature_circuits/data/rc_train.json` (git hash-object). Likewise rc_test `a105e36`, simple_train,
  nounpp_train and within_rc_train.
  - SFC's **test** split ships in their repo but is **unused**. Evaluation draws from the tail of the train file.
- **The Colab dropdown mislabels the code tasks.** Its "codereason/key" entry points at the index file (L = 24), and
  "codereason/index" points at the key file (L = 41) [code Colab cell 9]. The task notebooks are consistent; use those.
- **Mean set overlap:** the mean set is the corrupted counterparts of batches 0..39 (or 0..24, or 0..9), i.e. of
  **training** prompts.
- **Split:** there is no separate validation set and no shuffling. Batches come in file order.

---

## 6. Their evaluation, and numbers usable as fidelity checks

### 6.1 Faithfulness

- **Code:** `f = LD(C) / LD(ref)` with batch-mean logit differences. There is **no empty-circuit subtraction** [code
  Colab cell 16].
  - `ref` is the SAE-spliced model (the JSON value).
  - The post's figures use the full model [post §4.2; code Colab cell 28: `normalized_faithfulness =
    f·avg_logit_diff/avg_model_diff`].
  - **Numerator and denominator use different batches:** {0, −1..−9} vs 0..9 [code Colab cells 12, 16].
- **Empty circuit (everything at its mean):**
  - LD = 0.000 on 3 batches, for both code tasks [code error_pred cells 19, 61 outputs];
  - −0.071 on SVA batch 0 [code sva.ipynb cell 22 output];
  - −0.346 on IOI [code ioi_run cell 21 output];
  - so f ≈ the SFC-normalised `(m(C) − m(∅))/(m(M) − m(∅))` within about 2% on SVA and 10% on IOI [inferred].
- **Reference LDs** (model + SAEs, no error, batches 0..9):
  - SVA **3.514** [sva.ipynb cell 19];
  - list-index **4.165** [error_pred cell 16];
  - dictionary-key **4.840** [error_pred cell 59; Colab cell 12];
  - IOI **3.670** [ioi_run cell 16].
  - **The full-model LDs were not printed in any saved output.**

### 6.2 Completeness and stability

- **"SFC-like" completeness** [code Colab cells 41-49]:
  - load a circuit;
  - set θ to −1.9455 (gate 0) for every latent index in the circuit **at all positions**, keeping the rest live;
  - score LD on **one** batch, `clean_tokens[-batch_idx]` with `batch_idx = −1`, i.e. batch 1, a training batch;
  - completeness = LD / full-model LD.
  - The IG complement is computed analogously.
  - The post reports both near 0, with IG sometimes closer [post §4.3].
- **"Wang-style" completeness** [code Colab cells 51-55]:
  - random subsets K of a circuit (λ = 0.0447);
  - F(C∖K) vs F(M∖K) on 3 batches (0, −1, −2), against y = x.
  - The code draws **7 sets of random size**. The post describes **five 14-node subsets of a 55-node SVA circuit**
    [post §4.3].
- **Stability:** nesting across λ [post §4.4]. No code cell computes it, so it was visual [inferred].

### 6.3 The IG baseline (theirs, "the method Marks et al. used")

- **Per SAE, separately,** the codes at that SAE are interpolated from clean toward corrupted:
  - `a = (1 − α)·c_clean + α·c_corr`, with α ∈ {0, 0.1, …, 0.9} (10 steps, left Riemann);
  - every SAE stays spliced (no error);
  - the metric is the batch **sum** of raw-logit LD;
  - effect = mean over the batch of `grad ⊙ (c_clean − c_corr)`, averaged over the 10 steps, then over **25 batches**.
  - Source: [code Colab cell 20; cell 21 `get_all_sae_ig_effects(25, …)`].
- **Circuit:** the (position, latent) pairs with |IE| > τ, the rest mean-ablated.
- **Sweep:** τ quadratic-spaced over [5e-5, 0.05] in 10 steps [code Colab cells 24-25].
- **This is their re-implementation, not SFC's code.** It differs from SFC in several ways: no error nodes, a
  per-SAE interpolation with the other SAEs live, raw logits rather than log-probs, and no edges [inferred from
  sfc.md §1].

### 6.4 Published-run numbers (saved mask JSONs @HEAD; f is against the SAE-model)

These are the curves behind the post's figures [code masks/**/_run.json]. Size = (position, latent) pairs over 4 SAEs.

| task | λ → (size, f) |
|---|---|
| **SVA-rc** (`masks/sva/rc/logit_diff_*`) | 0.05 → (141, 0.966); 0.145 → (42, 0.948); 0.24 → (30, 0.983); 0.335 → (25, 0.977); 0.43 → (22, 0.964); 0.525 → (21, 0.966); 0.715 → (19, 0.945); 1.0 → (18, 0.940); 2.05 → (15, 0.882); 4.47 → (7, 0.636); 8.27 → (6, 0.552); 13.45 → (5, 0.453); 15 → (2, 0.049) |
| **list-index** (`codereason/index/len6_digit1`) | 0.01 → (399, 0.993); 0.0205 → (275, 1.079); 0.0447 → (174, 0.932); 0.0827 → (104, 0.692); 0.1 → (71, 0.552); 0.1345 → (42, 0.266); 1 → (1, 0) |
| **dictionary-key** (`codereason/key/len5_digit1`) | 0.01 → (485, 0.991); 0.0205 → (295, 1.034); 0.0447 → (183, 0.881); 0.0827 → (79, 0.484); 0.1345 → (44, 0.192); 0.5 → (2, 0) |
| **IOI** (`ioi/baba21`) | 1e-4 → (5,987, 1.012); 8.2e-4 → (1,927, 0.948); 1e-3 → (1,695, 0.946); 2.9e-3 → (822, 0.877); 6.4e-3 → (327, 0.443); 1.1e-2 → (150, 0.286); 1.8e-2 → (96, 0.254) |

- **C2 matches SVA:** 19-21 latents give f = 0.945-0.966 against the model + SAE.
- **The SVA sizes shrink by a few per step,** which the per-λ run outputs confirm [code sva.ipynb cells 36, 39
  outputs].
- **Files without the `logit_diff_` prefix** under `masks/sva/rc/` (e.g. `0.05_run`: 148 latents, f = 1.086) come from
  an earlier objective. Don't use them [inferred from the naming; the λ grid is the same].
- **Post-publication Gemma-2-2B** (`masks/sva/rc_train_saegap6_mean_error/`, from `sae_gap.sh`: SAEs at L0/6/12/18/24,
  L0 ≈ 100, mean error on, lr 0.05, portion 0.4):
  - 47 → 0.968, 87 → 0.901, 157 → 0.963, 370 → 0.972, 790 → 0.965;
  - f is against the **true** model here [code helpers/utils.py:620].
  - The gap-3 runs are flat at about 0.5.
  - **Not published.** Use them only as a sanity anchor for a 2B run.

---

## 7. Rooting

- **Root:** the **task metric only**, the logit difference at the last position [post §4.1; code everywhere].
- **No internal latent is ever a root.**
- **Consequences:**
  - A latent-rooted Caples row (the same mask machinery with a latent read as the objective) is **our adaptation**.
    It must be labelled so (README rule 1).
  - Our former unweighted arm *is* that adaptation (§11).

---

## 8. Cost on a 16 GB GPU (estimates from shapes, labelled [inferred])

**Shapes:**

| | layers | d_model | d_mlp | heads (q / kv × 256) | params |
|---|---|---|---|---|---|
| Gemma-2-9B | 42 | 3,584 | 14,336 | 16 / 8 | 9.24B |
| Gemma-2-2B | 26 | 2,304 | 9,216 | 8 / 4 | 2.61B |

- Vocabulary is 256,000 for both.
- A 16k residual SAE has 2·d_model·16,384 weights: 9B 117M (470 MB fp32, 235 MB bf16); 2B 75M (302 / 151 MB, which
  matches 050's 151 MB).
- Masks are L × 16,384 per SAE: at most 2.7 MB, negligible.

**Activations kept for the backward.**
- Assumption: about 5·d_model + 3·(heads·256) + 3·d_mlp floats per token per layer, from the first SAE layer up.
- Logits: 16·L·256k floats, about 3× for the softcap, per forward.
- In the authors' code the reference forward also builds a graph, because the SAE params require grad (§3.1).

| configuration | weights | SAEs (+ their grads) | activations + logits (both forwards) | total | 16 GB? |
|---|---|---|---|---|---|
| **9B fp32, as published**, SVA (L = 7) | 37.0 GB | 1.9 (+1.9) | about 2.5 | **about 43 GB**; the measured snapshot was 45.0 GB, so the estimate agrees | **no** |
| 9B fp32, dictionary-key (L = 41) | 37.0 | 1.9 (+1.9) | about 17 | about 58 GB | no (A100/H100-80GB) |
| 9B bf16, any task | 18.5 | 0.9 | — | > 18.5 GB on weights alone | **no** |
| 2B fp32 (TL default), 5 SAEs (gap 6), SVA | 10.5 | 1.5 (+1.5) | about 1.8 | about 15.3 GB | at the limit; no headroom |
| 2B fp32, L = 41 | 10.5 | 1.5 (+1.5) | about 10 | about 24 GB | no |
| 2B bf16, SVA / L = 41 | 5.2 | 0.75 (+0.75) | about 1 / about 5 | about 8 / about 12 GB | yes / yes, tight |

**Conclusions.**
- **The published configuration (9B) cannot run on the 16 GB card** in any dtype without quantising the model, and
  quantising changes the model, so it is not faithful.
- It needs a pod (A100/H100-80GB, fp32 as published). Per standing rule, any pod use is proposed to Daniel first.
- **Time:** an A100 ran SVA at 2.6 it/s (408 steps in 2:35 per λ) [code sva.ipynb cell 36 output]. The 640-sequence
  mean pass took 3:36 at L = 41 [code Colab cell 13 output].
- **On the 16 GB card, only Gemma-2-2B fits.** That is the authors' **post-publication** code path
  (`interp_bottlenecks.py`, `gemma-scope-2b-pt-res`). Run it in bf16, or in fp32 with two memory-only changes that
  leave the maths the same:
  - SAE params frozen;
  - the reference forward under `no_grad`.
  Each change is labelled; bf16 is a numeric change.

---

## 9. WCM on the same substrate

### 9.1 What exists

- **050 / 048 ran WCM-family fits on Gemma-2-2B (`unsloth/gemma-2-2b`, bf16) with Gemma Scope 1 16k JumpReLU SAEs**
  [048 README:7-19; fit_gemma_task.py:87-141]:
  - at "TIER 2" L0 (the third rung of each layer's L0 ladder);
  - at three site kinds: mlp = `post_feedforward_layernorm` output, att = `o_proj` input, res = decoder-layer output
    (resid_post);
  - with the intervention `x ← x + (ĉ − c)·W_dec` (error preserved; BOS never edited);
  - with zero / mean / dual floors and free amplitudes;
  - with a logit-contrast objective of log p(target) − log p(contrast), squared error, λ 3e-3, 400 steps, lr 0.05;
  - with a 75/25 split.
- **Its res sites:** 050's three-kind fits used `LAYER_STRIDE = 2` to fit the card [050 README:52-58].
- **Neither is 9B, and neither uses Caples' 4-site, no-error, per-position-mean set-up.**
- `fit_gemma_task.py` is a **standalone port** (tri-amp era), not `src/circuit/instrument/learned_mask.py`. The
  implementation audit found bugs in the standalone ports, so reuse it only after a parity test against the
  engine [inferred, memory `implementation-audit-2026-09-22`].

### 9.2 What WCM needs for arena D (build list)

1. **Model loader:** `google/gemma-2-9b` in the HF stack. It runs on the pod only (§8). A 2B row runs locally.
2. **SAE loader:** the `gemma-scope-9b-pt-res` npz files for the exact four SAEs (`layer_7/width_16k/average_l0_92`,
   `14/…_67`, `21/…_129`, `40/…_125`), with an **explicit L0 per layer**, not 048's TIER ladder.
   - Check that JumpReLU encode is `pre·(pre > threshold)` with no `b_dec` subtraction, as in sae_lens's Gemma Scope
     loading (048's `SUB_BDEC = 0`).
   - Verify by reconstruction FVU / L0 against the configured average L0 [inferred, to verify].
3. **Sites:** resid_post of those 4 layers only, BOS never edited (matches §2.3).
4. **An error-dropped mode (new).** Caples' substrate replaces the stream with the reconstruction at every site, in
   circuit and reference alike. The engine preserves the error (learned_mask.py:317 "error term preserved"). Two WCM
   rows follow:
   - **WCM on Caples' substrate** (error dropped; the reference is the SAE-spliced model). This is the like-for-like
     row.
   - **WCM native** (error kept). Labelled; it is not the same substrate, so it is reported separately (README rule 2).
5. **A per-position mean floor (new).** The engine's precomputed floors are `{site: [d_sae]}`, position-agnostic
   (learned_mask.py:928-934). Caples' ablation is `[L, d_sae]` over corrupted prompts. It is needed at least for
   scoring both methods in Caples' frame; optionally also as a WCM training fill (labelled ours).
6. **Data plumbing:**
   - their filtered, length-matched prompts, batches in file order, the same training batches and the same eval batches
     {0, −1..−9};
   - targets `clean_answer` / `patch_answer` as the token pair.

### 9.3 How the two objectives line up on one task metric

- **Same metric.**
  - WCM's logit objective reproduces `log p(target) − log p(contrast)` per prompt, with squared error
    (learned_mask.py:1715-1725).
  - For a token pair, the log-softmax normaliser cancels: `log p(a) − log p(b) = logit_a − logit_b`, with the softcap
    applied before both.
  - So **WCM's endpoint is exactly Caples' logit difference** [inferred, algebra].
  - Score both on mean LD at the last position.
- **Differences that remain, by design:**

| | Caples | WCM (logit) |
|---|---|---|
| loss on the endpoint | \|mean_batch LD_ref − mean_batch LD_C\| (L1 of batch means: per-prompt errors can cancel) | mean over prompts of (LD_C − LD_ref)² |
| reference | SAE-spliced model (no error) | full model (errors kept), or the SAE model in error-dropped mode |
| per-latent strength | none (kept = natural value) | learned amplitude α per kept latent |
| ablations trained against | one: the per-position corrupted mean | three at once: Z (zero), A (sparsity-preserving mean), C (contrast fill) |
| positions | one mask per position | one mask shared across positions |

- **One score set for both rows:**
  - **(a) Their evaluation:** binary circuit, per-position corrupted-mean ablation, `LD_C / LD_SAE-model` and
    `LD_C / LD_full` on their eval batches. Add the SFC-normalised `(m(C) − m(∅))/(m(M) − m(∅))` with m(∅) reported
    (evaluation.md §1.2).
  - **(b) Ours:** Z / A / C / R and necessity on held-out prompts. Use the SFC test split (`rc_test.json`) for SVA,
    since it exists and they did not use it.
- **Size axis (evaluation.md §6.1):** distinct (site, latent), position-collapsed, as the primary axis.
  - Caples' native count is (site, position, latent) pairs, in a second column.
  - A shared WCM mask has no per-position count. If one is wanted, count `n_latents × (L − 1)` and say so; it is
    unfavourable to WCM.
  - Optional extra row: WCM with per-position masks (our adaptation), to match units exactly.

**Attribution graphs do not share the substrate.** The brief's "arena D shared with attribution graphs" shares only the
model. Attribution graphs use transcoders (CLT / PLT, 038), not Gemma Scope residual SAEs. By README rule 2, Caples vs
WCM and AG vs WCM are separate comparisons. Their only overlap is Gemma-2-2B, which is **not** Caples' published
model. The 107 README lists arenas A-C only; arena D still has to be added.

---

## 10. Ambiguities, bugs, and fidelity checks

### 10.1 Bugs and quirks in the official code (keep them for the "as published" row)

- **B1. Running-mean bug** (reported by commenter Matt Levinson under the post) [code helpers/utils.py:356-378;
  Colab cell 13].
  - **The bug:** every sample in batch i updates `m ← m + (x − m)/(i+1)`.
  - **Effect:** the weights still sum to 1, so the estimate is unbiased for i.i.d. data, but it is dominated by late
    samples.
  - **Effective sample size** (computed): about **77** of 640 (Colab), **48** of 400 (code / IOI) and **19** of 160
    (SVA) [inferred, arithmetic].
  - **Recommendation:** "as published" keeps it. A fixed-mean row is ours.
- **B2. Eval overlaps training.** `batch[-i]` for i = 0..9 includes batch 0, a training batch: 16 of 160 eval prompts.
  "SFC-like" completeness scores batch 1, also training [code Colab cells 16, 43].
- **B3. Numerator and denominator on different batches** (Colab / error / IOI) [§6.1].
- **B4. The SVA target is one scalar** (3.514) for every batch, not the batch's own SAE-model LD [code sva.ipynb
  cell 30].
- **B5. The Colab task mapping is mislabelled** (§5). The Colab λ files are named after the dropdown key.
- **B6.** The `l1` / `distinct_l1` swap in pre-post `utils.py` is harmless (both 1.0).
- **B7. SAE params require grad** (memory only).
- **B8. The SFC test splits are shipped but unused.**

### 10.2 Ambiguities

- **A1. Which configuration is "the" method?**
  - The post and README point to the Colab notebook (lr 1, the 1.1 schedule).
  - The post's SVA numbers come from `sva.ipynb` (lr 0.05, the 1.3 schedule, ÷L).
  - **Recommendation:** run each task with the notebook that produced its published masks (§3.2 table), and record
    the SHA. Do not unify the configurations; any unification is ours.
- **A2. Faithfulness reference.** The JSON uses the SAE-model and the figures the full model. Report both.
- **A3. The completeness protocol differs between post and code** (5 × 14-node subsets of 55 vs 7 random-size sets).
  Use the post's description for C3 and record the code's.
- **A4. Library versions are unpinned.**
  - `SAE.from_pretrained(...)[0]` and `cfg.hook_layer` need sae-lens < 6. Pin a release from about January 2025 (5.x)
    and a matching transformer-lens 2.x, and record both.
  - TransformerLens vs HF numerics differ slightly. The parity check is F2.
- **A5. The empty circuit and the denominator are never computed together** in their code. We compute m(∅) on the same
  batches.
- **A6. The post says "10 masks"** [§4.4]. The code-task sweeps saved 5 to 7 λ and SVA saved 15. "10" matches
  `sva.ipynb` cell 36's first sweep.

### 10.3 Fidelity checks before any comparison (9B, pod)

- **F1. Reference LDs** of the SAE-spliced model (no error), batches 0..9, should reproduce:
  - SVA 3.51;
  - list-index 4.17;
  - dictionary-key 4.84;
  - IOI 3.67.
  The empty-circuit LDs should reproduce SVA −0.07 (batch 0), code 0.00 and IOI −0.35. Tolerance: about 2%, since
  they used fp32 and the same files [§6.1].
- **F2. Stack parity.** On the same prompts, the full-model LD and the SAE-model LD in our HF stack equal
  TransformerLens's within 1e-2. If not, the WCM row and the Caples row are not on one substrate.
- **F3. Schedule check.** The logged temperature at the final step is 992.8 (SVA, 408 steps) and about 627 (Colab
  form, 63 steps).
- **F4. Curve reproduction.** The (size, f) points of §6.4 should come back:
  - SVA about 19-21 latents at f ≈ 0.95-0.97 for λ 0.5-0.7;
  - dictionary-key f ≈ 0.88 at about 183;
  - IOI f ≈ 0.88 at about 822.
  Sizes are noisy, since no seed is set: accept ±30% in size at matched λ [inferred tolerance].
- **F5. Latent identity.** The latent sets (positions dropped) at matched λ overlap the saved JSON indices (report the
  Jaccard per SAE). Their saved `.pt` masks are git-ignored (`**.pt`), so only indices are available.
- **F6. C1 holds in our run.** Masking beats their IG baseline at every size, and dictionary-key IG stays below 0.5.
- **F7. Nesting (C4).** The fraction of λ_{k+1} members that are also λ_k members is near 1.

A 2B row (local) is checked only against the post-publication saegap6 anchors (§6.4) and is labelled "authors' code,
unpublished configuration".

---

## 11. Caples as published vs our former "unweighted arm"

Sources for our arm:
- `src/config.py:650-890` (house recipe);
- `learned_mask.py:808-828, 1529, 1688, 1986-1999`;
- `dev-notes/2026-09-28-baseline-audit.md` §4;
- 061 README.

"Unweighted arm" = WCM with amplitudes fixed at 1 (`free_amplitude = False`), as run in 061 and protocol v1.

| aspect | Caples as published | our unweighted arm | same? |
|---|---|---|---|
| model / dictionaries | Gemma-2-9B; 4 Gemma Scope 16k residual JumpReLU SAEs | TuringLLM; our SAEs at attn / mlp / resid sites, all layers (061) | no (substrate) |
| SAE error | dropped (reconstruction replaces the stream) | kept (`x + (ĉ − c)W_dec`) | **no** (the audit says "errors kept" as our adaptation; correct) |
| root / objective | task logit difference, \|batch-mean LD_ref − LD_C\| | one internal latent's pre-activation (latent MSE, "pos") | no (our adaptation) |
| reference | SAE-spliced model | full model | no |
| mask granularity | per (position, latent) | one per latent, shared across positions | no (our adaptation) |
| gate | σ(T·θ) | σ(θ/T) | equivalent form (T_ours = 1/T_theirs) |
| temperature schedule | T = 200^r, geometric; ends about 627 (code / IOI) or about 993 (SVA); about 340 only at the nominal step S | 1/T from 1 to 20, geometric over the run (`0.05**prog`) | **no** (the audit's "about 340" is T at i = S, not at the end; the run continues to 1.1·S or 1.3·S) |
| θ init | 1.0 (gate 0.73 at the start) | 4.0 (gate 0.982) | no |
| penalty | λ · mean over SAEs of Σ σ(T·θ) (tempered gate); SVA ÷ L | λ · Σ σ(θ) over all sites (untempered gate) | no (normalisation and tempering both differ) |
| optimiser | Adam, no weight decay, constant lr | AdamW, weight decay 0.05, constant lr | no |
| lr | 1.0 (code / IOI); 0.05 (SVA) | 0.05 | task-dependent |
| steps / batch | 63-408 steps, one pass, batch 16 | 400 steps, micro-batch 4 with accumulation | no |
| ablation of masked latents | one: position-specific mean over corrupted prompts (buggy running mean) | three at once: Z (zero) + A (sparsity-preserving mean) + C (contrast fill), weights 1 / γ / γ, γ = 0.25; train-split means | no (our adaptation) |
| extra terms | none | rank-keep (γ_R 3e-3 in 061) | no |
| read-out | θ > 0 (gate > ½) | gate > 0.5 (`keep_threshold`) | **yes** |
| per-latent strength | none (kept = natural value) | none (α = 1) | **yes** (this is what WCM adds) |
| size knob | λ, one run per value | λ, one run per value | yes |
| eval | LD ratio on 160 prompts incl. one training batch | Z / A / C / R, necessity on held-out contexts | no |

**What WCM adds over Caples, stated precisely.** On top of everything above, WCM learns an amplitude α per kept latent
and fits α jointly with the gate. The penalty prices membership only.

**What our unweighted arm already changed relative to Caples:**
- the root (a latent, not the logit diff);
- the error handling (kept);
- shared vs per-position masks;
- three ablations vs one;
- the schedule, init, optimiser and penalty form.

**Two corrections follow.**
1. **The audit's (c) list** should read:
   - T_end ≈ 627 / 993, not 340;
   - θ₀ 1 vs 4;
   - Adam with no weight decay vs AdamW 0.05;
   - lr 1.0 (code / IOI) or 0.05 (SVA) vs 0.05;
   - the penalty on the tempered gate, a mean over SAEs, vs untempered and summed;
   - plus the loss form (L1 of batch-mean LD vs MSE) and the reference (SAE-spliced vs full model).
2. **The paper's description** (main.tex:1560, "coefficients fixed at 1, objective moved to a single latent, one mask
   shared across positions, and three ablations … the latter is our unweighted circuit masking") **omits the error
   term** (Caples drops it; we keep it) and the schedule / optimiser differences. Suggested addition: "and SAE errors
   kept". The schedule items are better stated as "with our calibrated optimisation recipe".

The audit's fix stands: a robustness run of the unweighted arm with Caples' exact schedule (σ(Tθ), T = 200^r, θ₀ = 1,
Adam lr as published, penalty on the tempered gate). It isolates the recipe from the other adaptations.
