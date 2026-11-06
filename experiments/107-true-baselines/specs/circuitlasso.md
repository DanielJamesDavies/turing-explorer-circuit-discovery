# CircuitLasso: implementation specification (arena B)

Yin, Wei, Gao, Dhurandhar, Natesan Ramamurthy, Yu, "Scalable Circuit Learning for Interpreting Large Language Models",
arXiv 2606.16939. Written 2026-11-01 for exp 107, arena B.

## 0. Sources, versions and tags

**Paper versions**
- v1 (15 Jun 2026): arXiv comment "Accepted to the Mechanistic Interpretability Workshop at ICML 2026". The ICML
  virtual page (icml.cc/virtual/2026/79543) lists it as a workshop poster, OpenReview id `CWHBI6xwcE`.
- v2 (9 Oct 2026), **the version specified here**: arXiv comment "Accepted to the Main Conference of EMNLP 2026",
  ACL format, 20 pages. The first author's homepage also lists it as EMNLP 2026.
- **So the paper is not an ICML 2026 main-conference paper.** Cite it as EMNLP 2026; the ICML 2026 venue is a workshop.
- Spot check of v1's appendix: the same λ sweep and C.1 / C.3 (= v2 C.4) wording. Nothing in v1 contradicts v2.

**How it was read**
- The v2 HTML was read through WebFetch.
- WebFetch's tool cache also saved the v2 PDF. I read all 20 pages of it as page images and rendered Figures 3 and 4
  at high resolution to read their axes and edges.
- No clone, and nothing downloaded on purpose.

**Code: none released.** Checked on 2026-11-01:
- the paper (no footnote, URL or availability statement);
- the arXiv abstract pages for v1 and v2;
- `gh search repos circuitlasso` / `"circuit lasso"` (0 results);
- `gh search code CircuitLasso` (only third-party notes);
- the repos of the GitHub accounts `naiyuyin` (first author: homepage, ICDH, D3M) and `dennislwei` (Dennis Wei);
- the IBM org (`circuit`, `lasso`);
- the IBM Research publication page;
- the ICML virtual page;
- the first author's homepage;
- Hugging Face papers (404).

OpenReview blocks automated reads, so its reviews were **not** read.

**Tags**
- `[paper §x / Eq. y / App. z / Tab. n / Fig. n]`: v2.
- `[code repo file:line @SHA]`: code read on GitHub.
- `[verified locally]`: computed on this machine on 2026-11-01. HF GPT-2 and the cached
  `jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs` were loaded offline, via the project's WSL `.venv`; nothing was
  written to disk.
- `[Neuronpedia]`: the public feature API.
- `[inferred]`: my reading; the reasoning is given.

---

## 1. Summary

**The framing: a linear structural-equation surrogate.**
- N components (neurons, heads or SAE features) are observed over M inputs as X ∈ R^{N×M}. They are modelled as
  `X = AᵀX + ε` with a weighted adjacency A ∈ R^{N×N}, and `A[i,j] ≠ 0` means "x_i → x_j" [paper §3.1, Eq. 1].
- A is learned by `min ‖X − AᵀX‖²_F + λ‖A‖₁` subject to the graph being acyclic [Eq. 2].
- **Acyclicity is not learned.** It is a fixed block-upper-triangular mask in the model's computation order: layer,
  then attention before MLP. So the problem becomes ordinary Lasso regressions [§3.2, Eq. 3; App. B "On Acyclicity"].
- **ε is linearisation error, not noise.** The authors disclaim identifiability: the target is the dependency
  *skeleton*, and the weights "rank dependencies for selection, not ... quantify them" [§3.1; App. B].

**A circuit** is the skeleton of feature→feature edges, meaning the non-zero pattern of Â between consecutive
locations, plus a feature→target vector Â_{L,y}. The paper turns it into a displayed circuit only by
- ranking: features by |Â_{L,y}|, parents by |Â_{i−1,i}|;
- hand-picking roots and "most influential parents".

No threshold and no cap are applied [Fig. 3 caption; §4.2.1; App. C.4].

**One circuit per dataset, not per prompt.** CircuitLasso returns "one population-level circuit per task" [§2]: the
matrices {Â} are fitted once on the whole dataset. Any root's circuit is read off the same matrices, so the fit
does not depend on the root [inferred from §4.2.1 and C.4].

---

## 2. Substrate, per experiment

| Experiment | Model | Components / dictionaries | Sizes | Data |
|---|---|---|---|---|
| **InterpBench** (§4.1, C.2, C.3) | InterpBench semi-synthetic transformers (Gupta et al. 2024): the 16 "main synthetic" cases + the real IOI case | **raw neurons**: outputs of attention and MLP modules at L locations of width d; attention before MLP within a layer [§3.2] | per-case L and d not reported | the baselines' own clean and corrupted prompt sets, M prompts × n_token each, concatenated to 2M [C.2]; M and n_token not reported |
| **CoLA case study + faithfulness** (§4.2.1, C.4, D.5) | GPT-2 small | OpenAI SAEs (Gao et al. 2025) on the **final output of each of the 12 blocks** [C.4] | L = 12, d = 768, D = 32,768 [C.4] | CoLA training split, M = 8,551 sentences [§4.2.1, C.4] |
| **Bias in Bios** (§4.2.2, C.5, D.6) | Pythia-70M; Gemma-2-2B; Gemma-2-9B | Marks et al. SAEs (Pythia, D = 64·d); Gemma Scope (JumpReLU, D = 8·d) [D.1]; **one location only**, e.g. the layer-22 output of Gemma-2-2B [C.5] | — | BiB ambiguous (train) / balanced (test) sets [D.6] |

### 2.1 Pinning the GPT-2 dictionary exactly

**What the paper says, and why it is wrong.** D.1 describes the Gao et al. GPT-2 SAEs as "a ReLU-linear encoder with
D=32768 and L1 sparsity regularization" [paper D.1]. That cannot be the dictionary used:
- OpenAI's ReLU/L1 releases (`v1`, `v4`) exist only at `mlp_post_act` and `resid_delta_mlp`
  [code openai/sparse_autoencoder sparse_autoencoder/paths.py:2-24 @4965b94];
- the only D = 32,768 release at block outputs (`resid_post_mlp` = TL `blocks.{l}.hook_resid_post`) is `v5_32k`,
  which is TopK(32), with no L1 and layer-normed inputs [paths.py:26-38 @4965b94; repo README hook table].

**Verified locally: the paper's own numbers reproduce exactly on `v5_32k` resid-post with a prepended BOS.** I
recomputed the per-token codes in Tables 5 and 8 with:
- HF `gpt2`;
- `<|endoftext|>` (id 50256) prepended;
- the hook at `blocks.11` output;
- the jbloom re-upload of `v5_32k` layer 11: cfg `hook_name blocks.11.hook_resid_post`, `topk k=32`,
  `normalize_activations layer_norm`, `apply_b_dec_to_input true`;
- the encoder `TopK₃₂(ReLU)((LN(x) − b_dec) W_enc + b_enc)`.

| quantity [paper Tab.] | paper | with BOS | without BOS |
|---|---|---|---|
| "Kiss himself." z₁₂,₂₀₇₂₆ per token / mean [Tab. 5] | (0,0,0,1.5254,0) / 0.3051 | (0,0,0,1.5254,0) / 0.3051 | (0,0,1.7773,0) / 0.4443 |
| "He said that himself was hungry." z₁₂,₂₀₇₂₆ / mean [Tab. 5] | 1.3509 at the 5th token / 0.1689 | 1.3509 at index 4 / 0.1689 | 1.5747 / 0.2250 |
| same sentence, z₁₂,₃₀₉₂ at "." [Tab. 8] | 6.0688 | 6.0688 | 6.5737 |
| "...hungry?" / "...hungry!" / "...hungryyet" z₁₂,₃₀₉₂ [Tab. 8] | 3.4722 / 3.7391 / 0 | 3.4722 / 3.7391 / 0 | 3.4164 / 4.0106 / 0 |
| "This movie just watches itself." mean [Tab. 5] | 0.1861 | 0.1861 | 0.2055 |
| "Myself saw me." [Tab. 5] | 0.8009 at index 2 / mean 0.1602 | 0.8009 at index 2 / mean **0.1335** | 0.9346 / 0.1869 |

**What follows from this check:**
- **Layer indexing:** the paper's "layer ℓ" (1-based) is `blocks.(ℓ−1).hook_resid_post`.
  - Confirmed independently on Neuronpedia: `gpt2-small/11-res_post_32k-oai/20726` is a reflexive "-self" feature,
    `11-res_post_32k-oai/3092` fires on sentence-final ".", and `7-res_post_32k-oai/15186` is a reflexive feature
    (the paper's z₈,₁₅₁₈₆ "-self").
- **The dictionary** is OAI `v5_32k` resid_post: TopK(32) with LayerNorm on the input. The jbloom re-upload, which is
  our arena B substrate, gives bit-identical codes to 4 dp.
- **A BOS (`<|endoftext|>`) is prepended**, and the token average includes the BOS position (the denominator is
  n_tokens + 1).
  - Every per-token value checked, and every mean except "Myself saw me." (below), matches with BOS. None match
    without it.
  - In the rows I counted against the GPT-2 tokenizer, the paper's listed vector lengths equal n_tokens + 1, except
    a few rows that drop a trailing 0 ("He said ...", "Genie ...", "Myself ...").
- **"Myself saw me." has an arithmetic slip in the paper:** 0.8009/5 instead of /6. It is the only inconsistent entry.
- [inferred] BOS is presumably TransformerLens's default `prepend_bos=True`; the README of OpenAI's repo uses
  `HookedTransformer`.

---

## 3. The algorithm

### 3.1 Notation

- Locations ℓ = 1..12 (the block outputs).
- z_{ℓ,t} ∈ R^D is the SAE code at token t.
- Z_ℓ ∈ R^{D×M} holds the token-averaged codes, one column per prompt [C.4].
- Â_{ℓ,ℓ+1} ∈ R^{D×D}. Entry [a, b] is the edge from feature a at ℓ to feature b at ℓ+1 [Eq. 5: Z_j ≈ Â_{i,j}ᵀ Z_i].
- Â_{12,y} ∈ R^D.

### 3.2 Pseudocode (SAE-feature version: the GPT-2 / CoLA pipeline)

```
INPUT  GPT-2 small; OAI v5_32k resid_post SAEs E_0..E_11; sentences S_1..S_M with labels y_m ∈ {0,1}
       (CoLA train, M = 8,551); λ_F (feature-feature), λ_y (feature-target)

# ---- 1. data matrix: one observation per prompt -----------------------------------------------
for m in 1..M:                                                        [C.4]
    ids   = [<|endoftext|>] + gpt2_tokenize(S_m)                       [verified locally, §2.1]
    run GPT-2 once (no backward pass)                                  [§2, App. A.2]
    for b in 0..11:                                                    paper layer ℓ = b + 1
        x_t   = resid_post_b[t]          for t = 0..T−1 (T includes BOS)
        z_t   = TopK_32( ReLU( (LN(x_t) − b_dec) W_enc + b_enc ) )     the SAE's own encoder
        Z_ℓ[:, m] = (1/T) Σ_t z_t                                      "average across tokens" [C.4]
# (padding positions, if batched, must be excluded from the average  [inferred])

# ---- 2. feature→feature regressions, consecutive layers only -----------------------------------
for ℓ in 1..11:                                                       {A_{i,i+1}}, i = 1..L−1   [C.4]
    X  = Z_ℓᵀ                     (M × D design: rows = prompts, columns = layer-ℓ features)
    s  = column ℓ2 norms of X;  X̃ = X / s     no centring, no intercept      [C.1; Eq. 5]
          (columns with s = 0, i.e. features never active on the data, get coefficient 0     [inferred])
    for each layer-(ℓ+1) feature b:                    one Lasso per downstream feature
        y_b = Z_{ℓ+1}[b, :]ᵀ  (M-vector; raw, not standardised        [inferred])
        β̃  = argmin_β  ½‖y_b − X̃β‖²₂ + λ_F ‖β‖₁                        [C.1 form; Eq. 5]
        Â_{ℓ,ℓ+1}[:, b] = β̃ / s                     "coefficients are rescaled afterward"  [C.1]
# 11 × 32,768 = 360,448 single-output Lassos if every column is fitted.
# The Frobenius loss and the elementwise ℓ1 penalty separate exactly over output columns, so the
# multi-output solve and per-column solves have the same optimum. Fitting only the columns a
# root's backward trace needs is EXACT, not an approximation.              [inferred, math]

# ---- 3. feature→target regression (Eq. 6), CoLA only -------------------------------------------
X̃ = Z_12ᵀ / s
β̃_y = argmin_β  L_CE( y, σ(X̃β) ) + λ_y ‖β‖₁ ,  λ_y = 1e-5                [Eq. 6, fn. 2; D.2]
Â_{12,y} = β̃_y / s
# Classification, so cross-entropy [fn. 2; Tab. 9 reports accuracy]. Whether the CE is a mean or
# a sum, and whether there is an intercept: not stated (see A2, A3).

# ---- 4. solver (every subproblem) --------------------------------------------------------------
FISTA (Beck & Teboulle 2009) with backtracking line search;                [C.1]
warm-started "across both the neuron activations dimension and the regularization path";
stop when the relative objective decrease < 1e-6; PyTorch; one A100.

# ---- 5. circuit read-out --------------------------------------------------------------------------
dataset-level (all main-paper circuits):                                   [§4.2.1; C.4; Fig. 3]
    roots    : rank layer-12 features by |Â_{12,y}|; pick interpretable top features BY HAND
               (Fig. 3: 20726, 776, 19322, 3092; Fig. 6: 3092)
    parents  : for a node b at layer ℓ, rank layer-(ℓ−1) features a by |Â_{ℓ−1,ℓ}[a, b]|;
               keep the "most influential parents", chosen BY HAND (no cap, no threshold)
    recurse  : layer by layer down to layer 8 in the figures (5 layers); result = a tree-shaped path set
prompt-specific (appendix only):                                           [§4.2.1; C.4]
    root score   s = |Â_{12,y}| ⊙ |z_12|      (z_12 = that prompt's codes)
    parent score s = |∂z_{L,k}/∂z_{L−1}| ⊙ |z_{L−1}|   (see A8 for what the derivative is)
```

### 3.3 The neuron version (InterpBench), for completeness

- **Data** [C.2]:
  - clean and corrupted prompt arrays (M × n_token each) are concatenated to 2M × n_token;
  - at each location, d × 2M × n_token activations are averaged over tokens to H_ℓ ∈ R^{d×2M};
  - the locations are stacked in computation order to H̃ ∈ R^{Ld×2M}.
- **Regression:** Eq. 3 with a block-upper-triangular A. The lower blocks are initialised at zero and held fixed.
- **Each location regresses on all preceding locations, not only the previous one:** the ℓ-th subproblem predicts
  block ℓ from the p_ℓ = (ℓ−1)d coordinates of all earlier blocks [App. A.1]. This differs from the SAE case, which
  uses consecutive pairs.
- **Edges:** "Finally, we infer the causal circuit" [C.2]. How neuron-level coefficients become the component-level
  edge set that SHD is scored on is not stated (A18).

### 3.4 What is NOT in the method

- **No pre-screening.** No dead- or rare-feature filter is mentioned.
- **No intercept and no centring.**
- **No cross-validation for A_{i,i+1}.**
- **No per-target refit.**
- **No use of positions.** Everything is token-averaged.
- **No intervention or backward pass during discovery** [§2; App. A.2].

### 3.5 Computational cost claims

- **Neurons:** O(M·L(L−1)d² / (2√ε)) [Prop. 3.1, App. A.1].
- **Against EAP-ig:** CircuitLasso is provably cheaper if
  `L(L−1)/(24 S (2+f) √ε) < n_token ≪ d`, or `n_token ≫ d` and `n_token > √(L(L−1)d / (24 S √ε))` [Prop. 3.2, App. A.2].
- **SAE pair:** O(M D² / √ε) per pair (Eq. 5); O(M D / √ε) for Eq. 6 [§3.3.2].
- **Reported runtimes:**
  - InterpBench: 16.3 s per case, against 49.1 s for EAP-ig and 33.7 s for EAP; the nonlinear variant is ×3.7 [§4.1].
  - BiB: 36.5 / 47.2 / 107.4 s, against SHIFT's 257.6 / 371.2 / 908.4 s [Tab. 1].
  - **No runtime is reported for the GPT-2 CoLA fit.**
- [inferred, estimate] The SAE design is sparse. A CoLA sentence is ~5–15 tokens plus BOS, with 32 active codes per
  token, so ≤ ~500 non-zeros per row and ~3–4M per layer. The dense multi-output A is 32,768² fp32 = 4.3 GB. Per-column
  solves on sparse CSR are cheap; this is an estimate, not a measurement.

---

## 4. Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| model | GPT-2 small | §4.2.1 |
| locations | the 12 block outputs (`hook_resid_post`, blocks 0–11) | C.4; verified locally |
| SAE | OAI `v5_32k` resid_post_mlp, D = 32,768, TopK k = 32, LayerNorm on the input, b_dec subtracted | verified locally (the paper's D.1 text says ReLU/L1, which is wrong) |
| BOS | `<|endoftext|>` prepended, included in the average | verified locally (Tab. 5/8) |
| aggregation | mean over all T positions of a prompt (BOS included) | C.4; verified locally |
| M | 8,551 (CoLA train) | §4.2.1, C.4 |
| pairs | consecutive layers only: A_{i,i+1}, i = 1..11, plus A_{12,y} | C.4 |
| loss, Eq. 5 | ½‖·‖²_F (sum of squares) | C.1 |
| loss, Eq. 6 | cross-entropy (classification) | fn. 2 |
| intercept / centring | none written | Eq. 5/6, C.1 |
| standardisation | predictor columns to unit ℓ2 norm; coefficients rescaled afterwards | C.1 |
| solver | FISTA with backtracking; warm start along the output dimension and the λ path | C.1 |
| tolerance | relative objective decrease < 1e-6 | C.1 |
| backtracking η, initial L, max iterations, dtype | **not reported** | — |
| λ_y (Eq. 6) | **1e-5**, chosen by best test accuracy over {0, 1e-5, 5e-5, 1e-4, 5e-4} | D.2, Tab. 9 |
| λ_F (Eq. 5, A_{i,i+1}) | **not reported** | — (A1) |
| circuit cap / threshold | none ("We do not impose a fixed cap ... or a global coefficient threshold") | Fig. 3 caption |
| roots shown | z₁₂,₂₀₇₂₆, z₁₂,₇₇₆, z₁₂,₁₉₃₂₂, z₁₂,₃₀₉₂ (Fig. 3); z₁₂,₃₀₉₂ (Fig. 6) | Fig. 3, 6 |
| hardware | 1 × A100, PyTorch | C.1 |
| InterpBench | 16 main synthetic cases + IOI; 3 trials; the baselines' prompt sets | §4.1, C.2 |
| BiB location | e.g. the layer-22 output of Gemma-2-2B; one layer per model | C.5 |

---

## 5. Data

**CoLA (Warstadt et al. 2019, via GLUE).**
- Public release: 10,657 sentences from 23 linguistics publications.
- The case study uses "the 8,551 training sentences in the public release" [§4.2.1].
- y = acceptability (1 = acceptable) [C.4].

**The "test set" of Table 9 is not named. [inferred]: GLUE CoLA validation (1,043).**
- The strongest-λ row has train accuracy 70.44% and test accuracy 69.14%. These equal the majority-class rates of
  CoLA train (6,023 / 8,551 = 70.44%) and of GLUE validation (721 / 1,043 ≈ 69.1%), the signature of an all-
  "acceptable" classifier.
- The label counts are from my knowledge of GLUE. Verify them on the dataset (FC2).

**Prompt formatting:**
- **Sentences are used raw.** The examples in Tables 5–8 are bare sentences with only BOS in front, and they reproduce
  exactly (§2.1).
- **No truncation is mentioned.** CoLA sentences are far below GPT-2's 1,024 and the SAEs' 64-token training context
  [inferred].

**Other data:**
- **No corpus windows are used.** The 107 README's "thousands of corpus windows, as they use" is not what the paper
  does; the paper uses the task's own sentences.
- **InterpBench:** the baselines' (EAP / EAP-ig, Gupta et al. protocol) clean and corrupted prompts [C.2]. Counts not
  reported.
- **BiB:** the Marks et al. ambiguous and balanced sets [D.6].

---

## 6. Their evaluation

### 6.1 InterpBench structural accuracy (§4.1, C.3; Tab. 3–4; Fig. 2)

**Metric:** SHD between the recovered and ground-truth edge sets, = FP + FN [Tab. 4]. Mean ± sd over 3 trials.

**Per-case results** (|E| = ground-truth edges) [Tab. 3]:

| case | 3 | 4 | 8 | 11 | 13 | 18 | 19 | 20 | 21 | 26 | 29 | 33 | 34 | 35 | 36 | 37 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| \|E\| | 4 | 8 | 2 | 3 | 3 | 6 | 3 | 2 | 8 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| CL-linear | 1.00 | 2.33 | 1.00 | 1.00 | 5.00 | 3.00 | 3.33 | 1.00 | 11.00 | 1.00 | 2.00 | 1.00 | 1.00 | 2.00 | 1.00 | 1.00 |
| EAP-ig | 1.00 | 2.00 | 1.00 | 1.00 | 5.00 | 3.00 | 3.00 | 1.00 | 11.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| EAP | 1.00 | 4.00 | 1.00 | 2.33 | 5.00 | 5.00 | 5.00 | 2.00 | 13.00 | 1.00 | 2.00 | 1.00 | 2.00 | 3.00 | 1.00 | 1.00 |
| CL-nonlinear | 1.00 | 2.00 | 1.00 | 1.00 | 4.00 | 3.00 | 3.00 | 1.00 | 10.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Means over all 17 tasks** [§4.1]: CL-linear 3.16, EAP-ig 2.98, EAP 3.61, CL-nonlinear 2.84.
- IOI is shown only in Fig. 2.
- [inferred, arithmetic] Backed out from the means, IOI's SHD is ≈ 16.1 (CL-linear), 15.7 (EAP-ig), 12.0 (EAP) and
  15.3 (nonlinear). IOI dominates every mean.

**Error breakdown over the 16 cases** [Tab. 4]:

| method | FP | FN | precision | recall |
|---|---|---|---|---|
| CL-linear | 0.44 | 1.90 | 0.705 | 0.398 |
| EAP-ig | 0.54 | 1.65 | 0.852 | 0.491 |
| EAP | 0.85 | 2.23 | 0.514 | 0.299 |

**Not specified:**
- the binarisation rule that turns Â into edges (A18);
- the nonlinear variant's model: "kernel functions or multi-layer perceptrons" [App. B], with no detail.

### 6.2 CoLA faithfulness and completeness (§4.2.1, D.5, Fig. 4)

**Their definitions** (taken from Marks et al.; this paper does not restate them):
- **metric:** m = p(Y = grammatically correct) − p(Y = grammatically wrong) [D.5];
- **faithfulness:** F(C) = (m(C) − m(∅)) / (m(M) − m(∅));
- **completeness:** F(M∖C);
- **ideal values:** F = 1, completeness = 0 [Fig. 4 caption].

**What Marks et al. do, for reference** (Marks et al. §3.2, read 2026-11-01 via WebFetch; that paper, not this one):
- nodes outside C are mean-ablated to their position-specific means;
- scores are on a held-out split.

**Node ablation:**
- The protocol is Marks et al.'s.
- SHIFT is run without SAE error nodes and without attention / MLP SAEs, "to ensure fairness".
- For CircuitLasso, "we focus only on SAE features within the learned circuit and still ablate features in the
  original LLM" [D.5].

**Edge ablation (their addition):**
- An edge's coefficient is set to zero inside the linear surrogate m̂ = Â_{L,y}ᵀ z_L, not in the model [D.5].
- SHIFT has no edges, so it is excluded from this test [fn. 4].

**Data:** the CoLA *training* set [Fig. 4 caption].

**Fig. 4 read from the rendered figure** (values ±0.02 by eye):

| panel | x-axis | CircuitLasso | SHIFT |
|---|---|---|---|
| node faithfulness | Nodes 0–1,700 | 0 → 0.20 @50 → 0.29 @100 → 0.44 @200 → 0.60 @500 → 0.67 @1,000 → 0.72 @1,500 → 0.74 @1,700 | 0.20 / 0.30 / 0.45 / 0.62 / 0.70 / 0.75 / 0.78 at the same points |
| node completeness | Nodes 0–300 | 1.00 @0 → 0.45 @10 → 0.11 @25 → 0.02 @50 → ~0.01 @100–300 | 0.40 @10, 0.10 @25, ~0 after |
| edge faithfulness | Edges 0–3,500 | 0 → 0.65 @500 → 0.80 @1,000 → 0.88 @1,500 → 0.90 @2,000 → 0.92 @2,500 → 0.95 @3,000 → 0.98 @3,500 | — |
| edge completeness | Edges 0–3,000 | 1.00 → 0.15 @100 → 0.08 @250 → 0.05 @500 → 0.04 @1,000 → 0.03 @2,000 → 0.02 @3,000 | — |

The text says "per-quartile results" are in D.5. **They are not there** (in v2 or v1). Only Fig. 4 exists.

**Not specified:**
- how a CircuitLasso node set of size N is formed;
- the ablation value;
- where p(·) comes from (a probe? the Eq. 6 classifier?);
- which layers' features count as nodes;
- whether edges are only the A_{L,y} entries.

See A9 and A10.

### 6.3 Coefficient statistics (D.2–D.3; Tab. 9; Fig. 5)

**Table 9, Eq. 6 on CoLA, rows in printed order:**

| λ | train CE | train ‖A‖₁ | train acc | test CE | test acc |
|---|---|---|---|---|---|
| 0 | 0.0612 | – | 99.36 | 0.9977 | 71.10 |
| 1e-5 | 0.1741 | 9257.6 | 95.70 | 0.6684 | **72.77** |
| 5e-5 | 0.3498 | 1898.7 | 85.44 | 0.5566 | 72.48 |
| "5e-4" | 0.5535 | 45.3 | 72.14 | 0.5566 | 70.28 |
| "1e-4" | 0.6143 | 0.5 | 70.44 | 0.6283 | 69.14 |

**The last two rows' λ labels are almost certainly swapped** (A4).

**At λ = 1e-5** [D.3, Fig. 5]:
- 82.2% of |Â_{12,y}| are below 0.01, 83.3% below 0.10, 86% below 0.41;
- the histogram's x-axis runs to ~12;
- the top 5 by |Â_{12,y}|: 20726 (12.3138), 3908 (10.8914), 3261 (10.6946), 4628 (10.5880), 1168 (10.4870).
- Three of Fig. 3's four roots (776, 19322, 3092) are not in this top 5: the roots were picked for interpretability,
  not by rank.

### 6.4 Bias in Bios (§4.2.2, Tab. 1–2)

- **Method:** Eq. 6 at one layer; gender features are identified by hand and zeroed (zero ablation), and the result is
  fed to the trained or a retrained classifier.
- **Profession / gender / worst-group accuracy (%):**

| method | Pythia-70M | Gemma-2-2B | Gemma-2-9B |
|---|---|---|---|
| CircuitLasso | 90.5 / 50.1 / 75.8 | 77.5 / 50.7 / 50.5 | 81.5 / 50.3 / 69.8 |
| CircuitLasso-retrain | 94.2 / 50.6 / 88.7 | 95.1 / 52.8 / 92.9 | 96.9 / 50.5 / 91.5 |
| SHIFT | 88.5 / 54.0 / 76.0 | 72.8 / 51.6 / 43.7 | 77.1 / 52.8 / 67.9 |
| SHIFT-retrain | 93.1 / 52.0 / 89.0 | 94.2 / 52.4 / 92.4 | 96.0 / 51.3 / 90.3 |

- **Features selected:** 41 / 55 / 59 for CircuitLasso, against 49 / 65 / 71 for SHIFT [Tab. 1].
- **The authors' framing:** a "utility demonstration", not circuit learning [§4.2.2]. It is out of scope for arena B.

### 6.5 What they never evaluate

- held-out faithfulness;
- interventions in the model using the coefficients;
- per-feature (latent-rooted) circuits scored by any metric;
- circuit size against depth.

---

## 7. Ambiguities, with the most faithful reading

| # | Point | Most faithful reading | Why |
|---|---|---|---|
| A1 | **λ_F for A_{i,i+1}** is never given | Treat λ_F as the method's knob and **sweep it**. Mark the point "λ_F = λ_y = 1e-5 under the convention calibrated in FC3" as the paper-like point. | The paper uses one symbol λ throughout and selects "λ = 1e-5 as the optimal setting" only for A_{L,y} [D.2]. The losses differ (sum of squares vs CE), so the same number does not mean the same sparsity. 107 rule 3 asks for a curve over λ anyway. |
| A2 | **Loss normalisation and penalty space.** C.1 writes ½‖X − AᵀX‖²_F (a sum). For the CE, mean or sum is unknown. | Eq. 5: ½·sum, with the penalty on the standardised coefficients ("standardized ... before solving"). Eq. 6: pick the variant that reproduces Tab. 9 + Fig. 5 (FC3). | The text gives the form for Eq. 5. For Eq. 6, only the published numbers can decide. |
| A3 | **Intercept / centring**: none written | **None** for Eq. 5. For Eq. 6, decide by FC3 (with vs without an unpenalised bias). | Eqs. 5–6 and C.1 mention only scaling. The BOS position's codes appear in every prompt (scaled by 1/T), so BOS-carried features already act as a near-constant column [inferred]. The strongest-λ row collapsing exactly to the majority rate hints at bias capacity of some kind. |
| A4 | **Table 9's last two λ labels** | The rows are λ = 1e-4 (‖A‖₁ 45.3) and 5e-4 (‖A‖₁ 0.5), mislabelled. The selected λ = 1e-5 is unaffected. | ‖A‖₁ and train accuracy must fall monotonically in λ. v1 prints the same labels. |
| A5 | **"Test set" of Table 9** | GLUE CoLA validation (1,043) | The majority-class rates match (§5). |
| A6 | **SAE type**: the paper's text says ReLU/L1 | TopK(32) `v5_32k`, LayerNorm input, b_dec subtracted | The paper's numbers reproduce exactly (§2.1). No ReLU/L1 release exists at block outputs. |
| A7 | **Circuit read-out**: no rule; roots and parents are hand-picked | For an algorithmic baseline (our reading, labelled): **R1** = the backward ancestor closure of the root over the non-zero pattern of {Â_{ℓ−1,ℓ}} at λ_F, swept over λ_F. **R2** (sensitivity) = the top-k parents per node by \|Â\|, recursively, swept over k. | R1 uses only the paper's own definition, "A[i,j] ≠ 0 indicates the directed dependency" [§3.1], and the skeleton claim, so its only knob is λ. R2 mimics "most influential parents" [C.4] but adds a k the paper does not have. |
| A8 | **Prompt-specific parent score** \|∂z_{L,k}/∂z_{L−1}\| | The derivative of the linear surrogate, = Â_{L−1,L}[:, k] | "as in the final layer" [C.4]. The method has no backward pass [§2]. Not needed for dataset-level circuits. |
| A9 | **Fig. 4 node ablation**: how N-node sets are built; the ablation value; the source of p(·); which layers | Not reconstructable. Best guess: p(·) = the Eq. 6 classifier σ(Â_{12,y}ᵀ z̄_12) on the re-run model; mean ablation (Marks); node sets ranked by coefficient magnitude. | D.5 defers everything to Marks et al. GPT-2 has no native acceptability head. |
| A10 | **Edge ablation scope**: only A_{12,y} entries, or the chain through {Â_{i,i+1}} | Literal: m̂ = Â_{12,y}ᵀ z_12, so the edges are the A_{12,y} entries (≤ 32,768). The x-axis reaching 3,500 is compatible. | This is the only formula given [D.5]. |
| A11 | **Pairs**: "every pair (i, j) where i precedes j" [§3.3.2] vs consecutive only [C.4] | Consecutive only, for GPT-2 | C.4 describes the actual experiment; Fig. 3 traces only through A_{i−1,i}. |
| A12 | **"Warm-started across the neuron activations dimension"** | Output columns solved in sequence, each initialised from the previous solution | It affects speed only; each column's optimum is unique for a generic design. |
| A13 | **Backtracking constants, max iterations, dtype** | Any FISTA variant that reaches the same optimum at tolerance 1e-6 | The optimum is defined by the convex problem. |
| A14 | **Dead or never-active features** (zero columns) | Coefficient 0, columns dropped | A zero column cannot be scaled to unit norm, and its gradient is 0. |
| A15 | **The "Myself saw me." mean in Tab. 5** | A typo (÷5 instead of ÷6) | The value itself matches BOS exactly. |
| A16 | **Batch padding** | Exclude padding from the mean | The per-sentence means in Tab. 5 use the true length + 1. |
| A17 | **Fig. 3 roots**: the caption names 4 roots, but only z₁₂,₂₀₇₂₆'s tree is drawn | Treat the drawn edges as illustrations of a hand-picked traversal | Fig. 3 caption. |
| A18 | **InterpBench**: neuron→component aggregation and binarisation for SHD | Unknown. A plausible reading is any non-zero coefficient between a component's neurons. Not used in arena B. | C.2 ends at "we infer the causal circuit". |
| A19 | **Response scaling in Eq. 5** | The response is not standardised; only the "predictor columns" are | C.1 wording. |

---

## 8. Fidelity checks for accepting a re-implementation

The tolerances are ours, not the paper's.

**FC1. Data pipeline. DONE 2026-11-01, passes.**
- Reproduce the Tab. 5 and Tab. 8 values to 4 dp (§2.1).
- Keep this as a unit test: "Kiss himself." → (0, 0, 0, 1.5254, 0), mean 0.3051, on z₁₂,₂₀₇₂₆; "He said that himself was
  hungry." → 6.0688 at the final "." on z₁₂,₃₀₉₂.
- Without BOS, the test must fail.

**FC2. CoLA split and label rates.**
- Train: 8,551 sentences, 70.44% acceptable.
- The evaluation split's majority rate should be ≈ 69.1%.

**FC3. Eq. 6 calibration** (this resolves A2/A3 and fixes the λ scale).
- **Variants to fit** on CoLA train, over λ_y ∈ {0, 1e-5, 5e-5, 1e-4, 5e-4}: {mean CE, sum CE} × {no intercept,
  unpenalised intercept}.
- **Accept the variant that reproduces:**
  - Tab. 9 accuracies within ±1.5 pts (train 99.4 / 95.7 / 85.4 / 72.1 / 70.4; test 71.1 / 72.8 / 72.5 / 70.3 / 69.1,
    with A4's relabelling);
  - ‖A‖₁ at 1e-5 within ±15% of 9,257.6 (original units);
  - Fig. 5's top-5 set {20726, 3908, 3261, 4628, 1168}, at least 4 of 5, with magnitudes in [10, 13];
  - the |A| shares <0.01 / <0.10 / <0.41 = 82.2 / 83.3 / 86 within ±3 pts.
- **If no variant passes**, report that, and freeze the closest one, labelled "re-implementation, Eq. 6 not reproduced".

**FC4. Eq. 5 edges named in the text** [§4.2.1].
- **The edges.** For each pair below, the parent should rank in the top 10 by |Â_{ℓ−1,ℓ}[·, child]| of its child's
  column, at the λ_F of the paper-like point (A1). Paper layer ℓ = resid_post block ℓ−1.
  - z₈,₁₅₁₈₆ → z₉,₁₀₆₀₉ and z₈,₂₇₁₇₀ → z₉,₁₀₆₀₉ (merging)
  - z₁₀,₂₉₈₅ → z₁₁,₆₃₆₈ (dropping)
  - z₉,₂₂₃₁₄ → z₁₀,₅₅₇₄ (cause–effect)
  - z₈,₁₉₈₈ → z₉,₂₂₃₁₄ (anti-causal)
  - z₁₁,₂₉₀₄₁ → z₁₂,₂₀₇₂₆ (spurious)
  - the "-self" persistence chain z₈,₁₅₁₈₆ → z₉,₉₅₉₂ → z₁₀,₂₉₈₅ → z₁₁,{6368 or 29778} → z₁₂,₂₀₇₂₆
  - Fig. 3 also draws z₁₁,{6368, 29778, 29041, 21518} → z₁₂,₂₀₇₂₆.
- **Pass rule:** at least 6 of the 8 named edges.
- **Why the pass rule is soft:** the paper's parents were hand-picked, and λ_F is unknown.

**FC5. Solver equivalence.**
- On 3 CoLA columns, the no-intercept FISTA must equal sklearn `Lasso(fit_intercept=False)` on the scaled design,
  with α = λ/M for the 1/(2M) convention, to 1e-4 relative in coefficients.
- Multi-output vs per-column solutions must be identical.

**FC6 (optional, weak). Fig. 4 shapes, under the A9 best guess.**
- Node completeness < 0.05 by 50 nodes.
- Node faithfulness 0.55–0.65 at 500 nodes.
- If this does not reproduce, report it; do not tune toward it.

**FC7 (not planned). Reproduce Tab. 3.** InterpBench needs the unspecified binarisation (A18).

---

## 9. Mapping onto arena B

**Arena B substrate:** GPT-2 small, OAI v5 Top-K 32k SAEs, latent targets.

### 9.1 What is native

- **Model and dictionary: identical.**
  - Our resid sites are `jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs` [src/model/families.py:86-94, 113-121], with
    the same LN-in, `k = 32`, `b_dec` subtraction [src/model/hf_adapters.py:65-73].
  - FC1 shows codes identical to the paper's.
- **Node universe:** resid_post latents of blocks 0–11 (12 × 32,768).
- **A latent root is native for a resid_post target.** Eq. 5 is feature→feature, and the paper's own circuits start
  at latents (z₁₂,₂₀₇₂₆, z₁₂,₃₀₉₂). A root at block b < 11 uses the same matrices (A_{b+1, b+2}'s column for the
  root's layer).
- **Data, as published:**
  - CoLA train (8,551 sentences, BOS prepended), token-averaged, one row per sentence;
  - **one fit shared by all targets**;
  - the response is the root's token-averaged post-TopK code (not a pre-activation).
- **Circuit:**
  - backward trace through {Â_{ℓ−1,ℓ}} only, consecutive layers;
  - R1 (support closure) swept over λ_F as the method's own knob (rule 3).
- **Their evaluations, re-computable for a latent root:**
  - the surrogate edge ablation, generalised from y to the root (needs the A10 decision; labelled as such);
  - Marks-style F / completeness with m = the root's activation, labelled "our adaptation", as DAN-48 is for SFC.

### 9.2 Gaps (anything we do here is labelled "our adaptation")

| Gap | Consequence | What we report |
|---|---|---|
| **Targets at attn / mlp / resid_mid sites** | CircuitLasso has no node universe there (C.4 uses block outputs only) | Run CircuitLasso only on resid targets; other kinds are "not applicable". Open decision 3 in the README: WCM on resid-only as well, for a matched node universe. |
| **Targets that never fire on CoLA** | The root's response column is all zero, so the circuit is empty | Measure and report the rate. Optionally, our adaptation (labelled): the same fit on our corpus windows. |
| **Per-target training contexts** (WCM / 083 rows) | CircuitLasso has none: one dataset-level fit | Do not refit per target for the faithful row. A per-target-context fit is our adaptation. |
| **Positions** | Our scorers (Z, A, C, necessity, R) are per position at activating anchors; CircuitLasso circuits are position-free | Score the member set unchanged. Note that the method never sees positions. |
| **BOS** | Our pod run uses `prepend_bos=False` and 64-token windows [families.py:119-120]; CircuitLasso's data uses BOS | Fit with BOS (faithful); score on our records as for every row. Note the mismatch. |
| **Read-out rule** | The paper has none (hand-picked) | R1 primary, R2 sensitivity, both labelled "our reading of an unspecified step". No size matching (rule 3). |
| **λ_F** | Unknown | Sweep it; mark the FC3-calibrated λ = 1e-5 point. |
| **Train vs held-out** | They score on train only | Both (README). |
| **Eq. 6** | No task label exists for a latent target | Not used for latent roots. Needed only for FC3 / FC4 / FC6. |

### 9.3 `098/lasso_core.py` against the paper

| Item | lasso_core | Paper | Action |
|---|---|---|---|
| intercept / centring | unpenalised intercept via implicit centring [lasso_core.py:9-10, 56-62, 91-94] | none (A3) | needs a no-intercept mode: uncentred design, raw y, m = 0 |
| column scaling | uncentred unit-ℓ2 norm, penalty in scaled space, rescaled back [86-93, 155-160] | same [C.1] | matches |
| loss scaling | 1/(2n)·‖·‖² [5, 121-123] | ½‖·‖² [C.1] | λ_paper = n·λ_ours. Rescale before quoting λ. |
| step size | fixed 1/L (power iteration, +5%) with function-value restart [11, 102-152] | backtracking line search | same optimum; tolerance-level differences only. Document it, no change needed (FC5). |
| stopping | relative objective decrease < tol = 1e-6 [126, 147-150] | same | matches |
| path | geometric grid from λ_max, 20 per decade, warm-started [176-198] | warm-started path, grid unknown | fine |
| output | single column | multi-output | exact by separability; per-column solving is correct |
| loss type | squared error only | Eq. 6 needs ℓ1-logistic (CE) | missing; needed only for FC3 |
| design | COO→CSR over the active columns [65-94] | dense or unknown | fine (drops zero columns = A14) |

---

## 10. Corrections to `098-circuitlasso-baseline/PAPER_NOTES.md`

1. **Venue:** "ICML 2026" is wrong. v1 was an ICML 2026 *workshop* paper (MechInterp); v2, the version read, is
   *EMNLP 2026 main* (§0). The 107 README's arena-B row inherits this.
2. **SAE type:** the notes do not mention that the paper calls the GPT-2 SAEs "ReLU ... L1" [D.1]. They are TopK(32)
   `v5_32k` (§2.1). The notes' "block outputs (residual stream after each block)" is correct, and is now verified
   numerically and on Neuronpedia.
3. **BOS:** the notes say every token enters the average but miss that a BOS is prepended and averaged in (§2.1).
4. **Fig. 4 is not unknown.** The rendered PDF gives the axes (Nodes 0–1,700 / 0–300; Edges 0–3,500 / 0–3,000) and
   readable values (§6.2). The "per-quartile results" the text cites do not exist in either version.
5. **Table 9:** the notes list the sweep correctly. They miss that the last two rows' labels are inconsistent (A4),
   and that the "test" set is unnamed (A5).
6. **Neurons regress on *all* preceding locations [App. A.1], not consecutive ones.** The notes' "the neuron
   experiments do the same" is right on token averaging but should not be read as consecutive pairs.
7. **Warm start:** the paper says "across both the neuron activations dimension and the regularization path"; the
   notes' "across the output dimension" is an interpretation.
8. **Fig. 3:** the notes are correct that roots and parents are hand-picked. Add:
   - only the z₁₂,₂₀₇₂₆ tree is drawn, although four roots are named;
   - three of the four named roots are not in Fig. 5's top 5.
9. **Marks's protocol uses a held-out split.** The notes attribute mean ablation to Marks correctly. CircuitLasso
   departs from Marks by scoring on train.
10. **Correct as written:**
    - Eq. 5, consecutive layers (C.4), M = 8,551, token averaging;
    - the standardisation wording, the solver, the tolerance;
    - λ = 1e-5 (72.77%) and the D.3 shares;
    - the InterpBench numbers;
    - BiB Tables 1–2;
    - edge ablation in the surrogate.

Also, outside PAPER_NOTES: the 107 README's "thousands of corpus windows, as they use" is wrong. The paper uses the
task's own sentences (CoLA, 8,551).
