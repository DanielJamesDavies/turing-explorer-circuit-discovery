# CircuitLasso (Yin et al. 2026, "Scalable Circuit Learning for Interpreting Large Language Models"): reading notes

arXiv 2606.16939 **v2** (9 Oct 2026), ICML 2026, bib key `yin2026scalable`. Read 2026-10-27 for exp 098 (DAN-159 baseline row).

## How it was read

- **Raw text, whole paper.** The arXiv HTML (`/html/2606.16939`) was downloaded and converted to plain text with a local
  tag-stripper (math kept as its LaTeX `alttext`), and the PDF (`/pdf/2606.16939`) was converted with `pdftotext -layout`.
  Both were read directly, not through a summariser: §1–5, Appendix A (proofs), B (assumptions), C.1–C.5 (experiment
  details), D.1–D.6.
- **Not readable as text:** Figures 2, 4 and 5 are images. Figure 4 (faithfulness / completeness curves and the
  "per-quartile results" D.5 refers to) has no text content; its x-axis and the circuit sizes it uses are **unknown**.
  Table 3 (per-case SHD) was skimmed, not transcribed.
- **Our earlier note** (`paper/resources/related-reading/C-repair-scale.md` §8) came from a summarising fetch. It is
  accurate on what it states, but it misses the token averaging, the consecutive-layer restriction, the column
  standardisation and the λ rule (below). Its "evaluated on the training set" is confirmed (Fig. 4 caption: "on the
  CoLA training set").

## The method

**General form (Eq. 1–2).** A linear SEM surrogate `X = AᵀX + ε` over N components, `min ‖X − AᵀX‖²_F + λ‖A‖₁`
subject to acyclicity. ε is read as linearisation error, not noise. The authors say the goal is the "dependency
skeleton", not per-edge causal effects (§3.1, App. B: edge weights "rank dependencies for selection, not ... quantify
them").

**Neurons (Eq. 3).** Acyclicity is replaced by a block upper-triangular A in computation order (layer, then attention
before MLP); the problem decouples into one multi-output Lasso per location.

**SAE features (Eq. 5), exactly.**

- `Â_{i,j} = argmin ‖Z_j − A_{i,j}ᵀ Z_i‖²_F + λ‖A_{i,j}‖₁`, with `Z_i, Z_j ∈ R^{D×M}` the SAE codes at locations i and
  j over M observations and `A_{i,j} ∈ R^{D×D}`. Every feature at the later location j is regressed on **all D features
  of ONE earlier location i** (a multi-output Lasso; one D-vector regression per later feature).
- "Repeated for every pair (i, j) where i precedes j". In the GPT-2 experiment they fit only **consecutive layers**,
  `{A_{i,i+1}}`, on the **block outputs (residual stream after each block)**, L = 12, d = 768, D = 32,768 (C.4).
- **Positions: pooled by averaging, one observation per prompt.** C.4: SAE features of shape D × n_token per prompt are
  "average[d] across tokens to produce Z_i ∈ R^{D×M}". M = 8,551 CoLA training sentences, so M observations, not
  M × n_token. The neuron experiments do the same (C.2: activations averaged over tokens, clean and corrupted prompts
  stacked into 2M observations). There is no same-position or per-token variant in the paper.
- **Output regression (Eq. 6).** `Â_{i,y} = argmin L_pred(y, Aᵀ_{i,y} Z_i) + λ‖A_{i,y}‖₁`, MSE for regression and
  cross-entropy for classification. On CoLA, y = the acceptability label (not a model logit), regressed on the
  last-layer features.
- **Intercept:** none in Eq. 5 or 6 as written; not discussed.
- **Standardisation (C.1):** "Predictor columns are standardized to unit ℓ2 norm before solving, and coefficients are
  rescaled afterward." Centring is not mentioned.
- **Solver (C.1):** every instance decomposes into multi-output Lasso subproblems `min ½‖X − AᵀX‖²_F + λ‖A‖₁`, solved
  with FISTA with backtracking line search, **warm-started across the output dimension and along the regularisation
  path**, terminated when the relative objective decrease is below 1e-6; PyTorch, one A100.
- **Data:** all M training prompts of the task (CoLA training split; BiB training set). Not task-token positions: every
  token of every prompt enters the average.

**How λ is chosen.** Only for the output regression `A_{L,y}` (D.2, Table 9): a sweep λ ∈ {0, 1e-5, 5e-5, 1e-4, 5e-4};
**λ = 1e-5 is selected because it gives the best test-set prediction accuracy** (72.77%). At that λ the coefficient
distribution is heavy-tailed: 82.2% of |A_{L,y}| below 0.01, 86% below 0.41 (D.3). **No λ is reported for the
feature-to-feature fits A_{i,i+1}**, and no rule is stated for them.

**How edges become a circuit.** There is no threshold rule and no top-k rule.

- **Dataset-level circuit (Fig. 3, D.4):** rank last-layer features by |A_{L,y}|; hand-pick interpretable top features
  as roots; recurse backward through A_{i−1,i}, at each layer taking each node's "most influential parents" by
  |coefficient|. Fig. 3 caption: "We do not impose a fixed cap on the number of features per layer or a global
  coefficient threshold"; roots and parents "are chosen for human-interpretable illustration". The result is a
  tree-shaped path through consecutive layers.
- **Per-prompt variant:** score `s = |A_{L,y}| ⊙ |z_L|` (coefficient × the prompt's activation), and for parents
  `|∂z_{L,k}/∂z_{L−1}| ⊙ |z_{L−1}|`.
- **Bias-in-Bios:** rank one layer's features by |A_{s,y}|, then **manually** identify gender features and zero them.

## Evaluations, with definitions

1. **InterpBench (neurons, not SAE features; §4.1, C.2–C.3).** Gupta et al. 2024's semi-synthetic transformers with
   known ground-truth circuits: the 16 main synthetic cases plus the IOI case.
   - **Metric:** structural Hamming distance (SHD) between the recovered and ground-truth edge sets, i.e. false
     positives + false negatives (Table 4). Mean over three trials.
   - **Results:** CircuitLasso-linear 3.16, EAP-ig 2.98, EAP 3.61, nonlinear variant 2.84. Runtime 16.3 s per case vs
     49.1 s (EAP-ig) and 33.7 s (EAP), on an A100. About 81% of CircuitLasso's errors are missed edges.
   - **Not stated in the text:** how CircuitLasso's coefficients are binarised into edges for SHD (C.2 ends "Finally,
     we infer the causal circuit").
2. **CoLA, GPT-2 small, OpenAI SAEs (Gao et al. 2025, D = 32,768, block outputs; §4.2.1, C.4, D.5).** 8,551 training
   sentences.
   - **Qualitative:** interpretable feature paths (persistence, merging, dropping, cause-effect, spurious), labelled
     by a multi-prompt procedure (top-activating tokens) and a single-prompt check (perturbing a token switches the
     feature off).
   - **Faithfulness and completeness** "following the standard protocol of Marks et al. (2025)". The paper only adds
     the metric m = p(grammatical) − p(ungrammatical). The definitions below are Marks et al.'s, which this paper does
     not restate:
     - faithfulness F(C) = (m(C) − m(∅)) / (m(M) − m(∅)), where C keeps only the circuit's features and everything
       else is mean-ablated (ideal 1);
     - completeness = F(M \ C), the complement circuit (ideal 0).
   - **Node ablation:** compared with SHIFT's feature circuit, with SAE errors and attention / MLP SAEs excluded from
     SHIFT "to ensure fairness"; the circuit's features are ablated "in the original LLM". "Comparable" to SHIFT.
   - **Edge ablation (their novelty):** set coefficients to zero **inside the linear surrogate** `m̂ = Â_{L,y}ᵀ z_L`,
     not in the model.
   - **Evaluated on the CoLA training set** (Fig. 4 caption). The circuit sizes and the per-quartile split are in
     Fig. 4 only (an image), so they are unknown here.
3. **Bias-in-Bios (one layer, Eq. 6 only; §4.2.2, C.5, D.6).**
   - **Setup:** Pythia-70M (Marks et al. SAEs), Gemma-2-2B and Gemma-2-9B (Gemma Scope); a profession classifier
     trained on a gender-confounded set and tested on a balanced set.
   - **Metrics:**
     - Profession accuracy on the balanced test set;
     - Gender: how predictable gender is from the predictor's representation (50% is best);
     - Worst-group accuracy.
   - **Method:** CircuitLasso = SAE features token-averaged, a sparse linear classifier, the manually identified gender
     features zeroed, with or without retraining.
   - **Results:** about SHIFT-level accuracy (e.g. Pythia 90.5 vs 88.5; retrain 94.2 vs 93.1), at 7–8× less runtime
     (Table 1: 36.5 / 47.2 / 107.4 s vs 257.6 / 371.2 / 908.4 s) and with fewer features (41 / 55 / 59 vs 49 / 65 /
     71).
   - **Authors' framing:** "a utility demonstration", "not ... a multi-layer circuit learning experiment".
4. **Runtime (Props. 3.1–3.2).** Cost O(M L(L−1) d² / 2√ε) for neurons and O(M D² / √ε) per SAE layer pair, against
   EAP-ig's extra forward and backward per observation.
5. **Not evaluated:** held-out faithfulness, any intervention on the model with the coefficients themselves (edge
   ablation is in the surrogate), latent targets inside the model, and circuit size against depth. The authors, §2:
   "Masking maximizes faithfulness directly, so it offers guarantees we do not".

## Consequences for the 098 port (settled in README.md)

| Paper | Our target-latent port | Why |
|---|---|---|
| One later feature regressed on all features of ONE earlier location; consecutive layers | the target's pre-activation regressed **jointly on every upstream site's latents** (all layers and kinds before the target) | our circuits span every upstream site; one Lasso over all of them is Eq. 3's block-triangular form restricted to one output column |
| token-averaged codes, one observation per prompt (M = 8,551) | **same-position samples**: every (training context, position) is one observation | with 96 training contexts, one observation per context caps the Lasso support at ≤ 96 latents, below the 10²–10³ matched sizes; our target is a per-position quantity |
| y = the later feature's code (post-activation) | y = the target's **pre-activation**, the quantity IG / AtP attribute | the same estimand as the other 083 rows; a post-Top-K target is 0 on most positions |
| columns scaled to unit ℓ2 norm, coefficients rescaled back | same | paper |
| no intercept | **unpenalised intercept** (implicit centring) | the pre-activation has a large non-zero mean; without an intercept, always-on latents would absorb the offset |
| FISTA, warm-started along the λ path, relative objective decrease < 1e-6 | same, on GPU with sparse CSR design matrices | paper |
| λ by best test accuracy (Eq. 6 only) | natural size: λ minimising validation MSE on held-out *training* contexts (a context-level 3:1 split) | the paper's rule, adapted to a regression target |
| rank by \|Â\| (rescaled, original units) | **primary ranking = \|coefficient\| in original units** at the largest path λ whose support reaches the needed size | paper's ranking; the standardised \|coefficient\| is kept as a sensitivity row |
