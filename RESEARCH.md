# Research Overview: Sparse Latent Relationships in Language Models

A brief description of the research programme implemented in this repository — what it studies, what is new, how it works, and how results are evaluated.

---

## One-Paragraph Summary

This project studies **circuit discovery over sparse autoencoder (SAE) latents** in TuringLLM, a 12-layer, 254M-parameter transformer trained on a fully documented synthetic corpus. Prior circuit methods (Sparse Feature Circuits, Attribution Graphs) select a *behavioural or logit-level endpoint* — even when that endpoint is automatically discovered — and identify features contributing to it. This project instead uses a **sparse internal feature as the endpoint**: for a given seed SAE latent, which other latents can causally increase, suppress, or sustain its activation? The system discovers these **signed local causal dependency neighbourhoods** at scale, without externally specified behavioural tasks, and validates every candidate by its ability to manipulate the target latent on probe contexts.

## Research Framing

Outputs are filtered summaries of computation, not the whole computation. A model computes many internal variables that never surface cleanly as a single behaviour, so circuit analysis rooted at behavioural or logit endpoints — even automatically discovered ones — only reaches the internal variables that happen to project onto those endpoints. SAE latents — imperfect but empirically meaningful — provide a sample of the model's internal ontology. The central question:

> Instead of only asking which features caused an output, can we map which features cause **other features**, building a dependency graph over the model's own learned variables — and do those graphs survive causal intervention?

## Key Contributions

1. **Latent-as-endpoint circuit discovery.** A pipeline that treats every SAE latent as a first-class explanandum — requiring no externally specified behavioural task — rather than rooting analysis at a behavioural or logit-level endpoint. The claim is context-conditioned, intervention-tested *control* relationships between latents — not a static compatibility matrix or co-occurrence structure.
2. **A causal role taxonomy with matched discovery methods.** Three operational node roles, each discovered by gradient attribution under a different regime:
   - **Counterfactual activator** — absent on negative contexts; would push the seed toward firing if active.
   - **Counterfactual inhibitor** — present on negative contexts; actively suppresses the seed.
   - **Ablation support** — present on positive contexts; necessary to maintain the seed's activation.
3. **Negative-context construction as a first-class variable.** Hard negatives are retrieved by exact cosine k-NN search over mean-pooled final-layer residual representations (semantically similar sequences expected to lack the seed's activation — membership in the stored positive contexts is excluded, but seed inactivity is not verified), and the choice of negative regime (`close` / `random` / `distant`) is studied experimentally. Early finding: semantic closeness and clean causal contrast are not the same thing.
4. **Seed pre-activation attribution (`SeedProjectionInstrument`).** On contrast sequences the seed is typically absent from SAE top-k, so its sparse activation carries no gradient. Attribution instead targets the seed's *encoder pre-activation* computed directly from the SAE input, keeping gradients alive when the seed never fires.
5. **Intervention-based evaluation matched to the latent endpoint** — counterfactual faithfulness and positive-context suppression — rather than borrowing output-level metrics that do not fit the target.
6. **A reproducible laboratory**: a controlled model + SAE bank + known training corpus, a multi-pass data-collection pipeline, 17 discovery methods under one interface, and post-hoc structural analysis (motifs, commonality, supercircuit composition).

## The System

### Substrate

| | |
|---|---|
| Model | TuringLLM (12 layers, d_model 1024, 16 heads, 254M params, 1024 ctx) |
| Training data | ~2B tokens, fully synthetic and documented |
| SAE bank | 36 SAEs (attn / mlp / resid × 12 layers), dictionary 40,960, Top-K = 128 |

### Pipeline (six stages)

1. **Pass 1 — latent statistics & contexts.** Welford activation statistics, top-activating and mid-band reservoir contexts, top predicted tokens, and pooled residual sequence representations, collected across dataset shards.
2. **Negative-context construction.** Pure-PyTorch exact cosine k-NN search (chunked GPU matmul over a per-latent positive-centroid query) finds sequences semantically close to each latent's positive contexts that are not among its stored positives.
3. **Pass 2 — co-activation graph.** Top co-activating latents per latent, scored raw, frequency-weighted, or PMI-clamped.
4. **Candidate selection.** Activity- and frequency-filtered seed latents chosen for discovery.
5. **Discovery.** One or more of 17 methods per seed (below).
6. **Evaluation & post-analysis.** Intervention scoring, minimality pruning, then structural analysis of accepted circuits.

### Discovery Methods (grouped)

| Family | Methods | Evidence used |
|---|---|---|
| Statistical baselines | `coactivation_statistical`, `neighborhood_expansion` | Co-activation thresholds / two-hop structure; no gradients |
| Sparse expansion | 7 `*_top_coact_sparse_expansion` variants, `hard_negative_coact_sparse_expansion` | Variable-depth BFS over the co-activation graph, kind-targeted with passthrough; hard-negative variant adds gradient-validated inhibitors |
| Output attribution | `logit_attribution`, `sfc_attribution_patching`, `top_coact_attr` | activation×gradient to logits; SFC-style integrated-gradient nodes + delta×gradient edges |
| Upstream tracing | `gradient_upstream`, `layerwise_gradient_upstream` | Multi-hop / layerwise backward attribution against the seed's activation |
| Contrastive (core contribution) | `differential_activation`, `counterfactual_gradient`, `ablation_gradient`, `hybrid_gradient` | Pos/neg contrast scans; gradient attribution on negative contexts (activators + inhibitors); ablation-benefit scoring on positive contexts (supports); circuit-level fusion of both with re-evaluation |
| Global baseline | `circuit_tracer_baseline` | SAE-adapted Attribution-Graphs method: direct-effects matrix + Neumann-series influence propagation, fraction-based pruning |

The pluralism is deliberate: no single notion of "relevant feature" is trusted, and simple statistical baselines act as reference points for the gradient methods.

### Evaluation

Output-level (shared by logit-rooted methods):

| Metric | Definition |
|---|---|
| Faithfulness | `1 − MSE(circuit logits, original) / MSE(baseline logits, original)`; baseline is zero- or neg-ctx-mean ablation |
| Kind-local faithfulness | Faithfulness with only target SAE kinds patched (gate for sparse-expansion family) |
| Sufficiency | Target-token probability preserved under circuit-only execution |
| Completeness | `1 − faithfulness(complement)` — does removing the circuit degrade the model? |
| Minimality | Leave-one-out pruning of redundant nodes |

Latent-level (matched to the contrastive family's endpoint):

| Metric | Question it answers |
|---|---|
| Upstream faithfulness | Does the circuit recover the seed's activation (not just the output)? |
| Counterfactual faithfulness | On contrast contexts, does injecting discovered activators and suppressing inhibitors make the seed fire as on positive contexts? |
| Pos-ctx suppression score | On positive contexts, does suppressing activators / injecting inhibitors silence the seed? |

Acceptance gates are per-method: `counterfactual_gradient` gates on counterfactual faithfulness, `ablation_gradient` on suppression, `hybrid_gradient` on a configurable combination.

## Empirical Findings So Far

- A full H100 run produced **7,525 accepted circuits** across seeds and methods, with a complete analysis catalogue (`paper/analysis-catalogue.md`).
- **Co-activation structure alone does not explain circuits.** Latent co-activation profiles are near-orthogonal (PC1+PC2 ≈ 3% variance); overlap between discovered circuits and seeds' co-activation neighbourhoods is low (~3.6%) and uncorrelated with faithfulness. Faithful causal circuits are not local co-activation neighbourhoods.
- **Negative-context regime matters.** Random negatives can outperform semantically-close ANN negatives for counterfactual discovery — evidence that "close" negatives are not verified clean counterfactuals (they may contain partial activators or off-store seed activity).
- **Exact latent reuse across circuits is real but hub-dominated.** ~233 non-seed latents appear in ≥15% of circuits, overwhelmingly early-layer counterfactual inhibitors — motivating hub-corrected, causally validated motif analysis rather than raw commonality counts.
- **Current gradient circuits are one-hop stars** (`upstream latents → seed`), motivating the supercircuit-composition work below.

## Known Limitations & Open Questions

- **Pre-activation is a surrogate for firing.** With Top-K SAEs, a high encoder pre-activation does not guarantee the latent wins a top-K slot and actually activates. `SeedProjectionInstrument` therefore attributes through a continuous surrogate of a discrete firing event. A dedicated validation is planned: does selecting nodes by pre-activation attribution reliably cause *actual top-K entry* of the seed under intervention, compared against direct activation gradients, straight-through estimators, and finite-difference interventions?
- **Star topology.** Accepted circuits are one-hop attribution neighbourhoods; multi-hop causal chains are composed, not yet directly discovered, and require propagation-validated intervention tests.
- **Metric comparability.** MSE-ratio faithfulness is nonstandard; KL-based metrics are planned so results can be compared against SFC / Attribution Graphs conventions.
- **Generic-inhibitor risk.** High-commonality early-layer inhibitors may be objective artifacts (any generic suppressive direction moves the pre-activation loss); per-latent null tests are needed.
- **Single custom model.** The controlled testbed is a deliberate design choice (fully known training distribution), but replication on a public model + public SAEs, or release of the model/SAE bank/dataset, is needed for external comparability.
- **SAE error terms.** Reconstruction residuals are preserved as anchors during instrumentation, but the causal share absorbed by residuals is not yet quantified against circuit claims.

## Direction

1. **Composed supercircuits** — splice circuits whose seeds appear as nodes in other circuits into multi-hop graphs, then validate that interventions at the earliest nodes propagate through intermediates to the final seed (`dev-notes/composed-supercircuit-analysis.md`).
2. **Causally validated motifs** — mine recurring 2–3-node signed subgraphs across circuits, scored by lift over role/layer/kind-preserving null models and by motif-only sufficiency / motif-removal damage (`dev-notes/circuit-motif-and-cohesion-analysis.md`).
3. **Graph-replacement and complement-ablation scores at the latent level** — stronger sufficiency/necessity claims than inject/suppress alone.
4. **Evaluation hardening** — strict held-out context splits (contexts split *before* discovery, threshold selection, and pruning); matched null controls (random latents matched on layer, kind, activation frequency, decoder norm); dose–response and reversed-sign intervention checks for each predicted activator/inhibitor role.
5. **Metric comparability and a public-model replication** — KL-based faithfulness alongside the MSE ratio, and a smaller replication of the counterfactual / ablation / hybrid comparison on a public model with public SAEs.

## Where to Look

| Topic | Location |
|---|---|
| Full system description | `description.md`, `concise_description.md` |
| Method implementations + docs | `src/circuit/discovery/`, `src/circuit/discovery/METHODS.md` |
| Evaluation implementations | `src/eval/` |
| Paper draft | `paper/main.tex` |
| Run analyses & figures | `paper/analysis-catalogue.md` |
| Research notes & plans | `dev-notes/`, `ideas.md`, `agent-planning/` |
