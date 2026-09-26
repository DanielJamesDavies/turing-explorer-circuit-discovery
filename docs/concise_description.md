# Repository Description

This repository is a transformer-interpretability research system focused on **circuit discovery in TuringLLM**. The core idea is to take a trained language model, decompose its internal activations with a bank of **Sparse Autoencoders (SAEs)**, and then search for small groups of sparse latent features that appear to jointly implement a concept, behavior, or prediction. In this codebase, a *circuit* is not just a set of correlated features. It is meant to be a compact, testable sub-network whose activations can be patched back into the model and evaluated for whether they reproduce the original behavior faithfully.

The repository is built around a staged pipeline rather than a single analysis script. It first collects descriptive statistics and context examples for every latent, then builds contrast sets and co-activation structure, then chooses candidate seed latents, then runs multiple discovery algorithms, and finally evaluates and analyzes the discovered circuits. A major theme throughout the project is that there is **no single trusted discovery method**. Instead, the system supports a broad family of complementary methods, ranging from simple statistical neighborhood expansion to gradient-based attribution, contrastive discovery, and SAE-adapted graph-tracing baselines.

## What The Repository Is Modeling

The target model is **TuringLLM**, described in the repository docs as a 12-layer transformer with a 1,024-dimensional residual stream, 16 attention heads, a 4,096-dimensional MLP hidden layer, a 50,304-token vocabulary, and 1,024-token context length. The model is not analyzed directly only in raw activation space. Instead, each layer/component is paired with sparse autoencoders that compress dense activations into sparse interpretable latent codes.

The **SAE bank** contains 36 sparse autoencoders spanning the main transformer component types:

- **attention latents**
- **MLP latents**
- **residual-stream latents**

Each SAE produces a sparse Top-K latent representation, so the working units of analysis are individual latent features rather than full dense vectors. This is what makes the circuit discovery problem tractable: the repository can talk about "feature 512 in layer 3 MLP" as a concrete node in a candidate circuit rather than having to reason about an uninterpretable dense activation basis.

The project also includes a substantial instrumentation layer. The model can be run in standard inference mode, but it can also be run under feature-patching, attribution, and graph-logging hooks. This lets the system ask questions like:

- Which upstream features most influence a seed latent?
- Which neighboring features co-activate with it consistently?
- Which features appear in similar semantic contexts but are absent in contrast cases?
- If only a proposed circuit is left active, how much of the original behavior remains?

## High-Level Pipeline

The full pipeline is best understood as six stages.

### 1. First pass: latent statistics and context collection

The system iterates over tokenized dataset shards and records per-latent activation statistics. This includes mean and variance style statistics, top-activating context sequences, and a "mid-band" reservoir of examples where the latent is active but not necessarily maximally so. It also stores token-level or logit-level summaries so that latents can later be searched by the kinds of predictions or contexts they are associated with.

This stage establishes the descriptive foundation for the rest of the pipeline. Without it, later methods would not know which sequences are the best positive probes for a latent, how frequent or rare it is, or which other features tend to appear near it.

### 2. Negative-context construction

The project puts unusual emphasis on **negative contexts**. Instead of looking only at sequences where a latent strongly activates, it also tries to find sequences that are semantically similar but where the latent does not fire, or does so much less strongly. These negative examples are important because they turn discovery into a contrastive problem rather than pure clustering.

Negative contexts can be built using approximate nearest-neighbor style retrieval over pooled sequence representations. In practical terms, the repository stores per-sequence embeddings and looks for nearby sequences that resemble the positive contexts while lacking the seed feature. These negatives become especially important for methods that try to distinguish:

- features that *cause* the seed to activate
- features that are merely correlated with it
- features that *suppress* or inhibit the seed

### 3. Second pass: co-activation graph construction

Once positive contexts are known, the repository computes a **top co-activation graph**. Each latent gets a list of other latents that tend to appear with it, along with a co-activation score. Depending on configuration, this score may be raw, frequency-adjusted, or PMI-like. This graph is one of the core reusable structures in the project because several discovery methods use it either directly or as a proposal mechanism.

The co-activation graph is especially useful for scalable discovery. Rather than searching the entire latent space at every step, a method can start from a seed and inspect its strongest neighbors first.

### 4. Candidate selection

The system does not usually run expensive discovery on every latent. It first selects **seed latents** that look promising. Candidate selection can use different heuristics, such as random stratified sampling, activation properties, context coherence, graph structure, token prediction behavior, or previously observed discovery yield. The goal is to focus compute on seeds that are likely to produce interesting or interpretable circuits.

### 5. Circuit discovery

This is the heart of the project. For each seed latent, the repository can run one or more discovery methods. These methods differ in what they treat as evidence:

- some trust co-activation structure
- some trust gradients
- some compare positive and negative contexts
- some explicitly trace direct effects through a graph
- some are seed-based and local
- some are cluster-based and seed-free

The system is intentionally pluralistic here because each approach captures a different notion of "relevant feature."

### 6. Evaluation and post-analysis

Every candidate circuit is evaluated rather than accepted at face value. The main question is whether the circuit, when used as an intervention or partial reconstruction, actually reproduces the behavior that motivated its discovery. Circuits can then be pruned for minimality and analyzed for structural properties like layer spread, edge concentration, activity patterns, rarity, and overlap with local co-activation neighborhoods.

## Main Code Areas

The repository is split into a few clear subsystems.

- `src/model` contains TuringLLM, tokenization, inference, and model-hook logic.
- `src/sae` contains the sparse autoencoder bank, accelerated Top-K selection, fused linear-ReLU operations, and other performance-sensitive encoding utilities.
- `src/pipeline` contains runtime setup, the collection passes, negative-context construction, candidate selection, discovery orchestration, and persistence.
- `src/store` contains persistent data structures for latent statistics, positive/mid/negative contexts, sequence representations, token summaries, co-activation graphs, and circuits.
- `src/circuit` contains the circuit domain itself: feature IDs, sparse activation types, instrumentation, patchers, discovery methods, per-seed discovery orchestration, and post-discovery analysis.
- `src/eval` contains the evaluation metrics used to decide whether a circuit is meaningful.
- `src/display` and `src/observability` contain terminal display, logging, progress tracking, and timing utilities.
- `src/native` contains build glue for C++ and CUDA acceleration, including support for heavy reduction and encoding paths.

## Discovery Methods In More Detail

The most distinctive part of the repository is the number of discovery algorithms it supports. They can be grouped into a few broad families.

### 1. Sparse co-activation expansion methods

These methods start from the idea that if a seed latent belongs to a real functional sub-network, then other important members of that sub-network should often appear in its co-activation neighborhood. The repository builds several **top coactivation sparse expansion** variants around this assumption.

At a high level, these methods perform a **variable-depth breadth-first expansion** over the co-activation graph. A parameter such as `coact_depth = [32, 16]` means that at the first hop the method keeps up to 32 strong neighbors, then at the second hop it expands each retained node into up to 16 more neighbors. This creates a candidate graph around the seed without having to search the entire latent space.

The variants differ in which latent types they allow the expansion to include:

- `attn_top_coact_sparse_expansion`: expand only attention features and treat MLP/residual features as passthrough support.
- `mlp_top_coact_sparse_expansion`: expand only MLP features, with attention/residual passthrough.
- `resid_top_coact_sparse_expansion`: expand only residual features, with attention/MLP passthrough.
- `attn_mlp_top_coact_sparse_expansion`: expand over attention and MLP latents while allowing residual passthrough.
- `attn_resid_top_coact_sparse_expansion`: expand over attention and residual latents with MLP passthrough.
- `mlp_resid_top_coact_sparse_expansion`: expand over MLP and residual latents with attention passthrough.
- `all_top_coact_sparse_expansion`: expand over all component kinds, with no passthrough carveout.

The intuition behind the passthrough variants is important. Sometimes the goal is not to discover every active feature in the local region, but to isolate the component family most likely to hold the meaningful mechanism while still letting other component types remain available so that the model can function during evaluation. This makes these methods a compromise between strict sparsity and realistic execution.

These methods are best thought of as **structural priors**. They assume that meaningful circuits should be recoverable from local co-activation structure and causal ordering, even before detailed gradient analysis is applied.

### 2. Hard-negative sparse expansion

`hard_negative_coact_sparse_expansion` extends the sparse expansion family by introducing **inhibitor discovery** from hard negatives. Standard sparse expansion is good at finding activators and partners that co-occur with the seed. It is less naturally suited to finding features that matter specifically because they are present when the seed is *suppressed*.

This method keeps the co-activation-based expansion for the activator side, but then inspects hard-negative contexts to look for latents that are unusually active when the seed ought to fire but does not. Those candidate inhibitors are then filtered or validated using attribution. Conceptually, this is one of the methods that most clearly treats a circuit as containing both "positive drivers" and "negative regulators," not just a cluster of positively correlated features.

### 3. Purely statistical graph baselines

The simplest statistical baseline is `coactivation_statistical`. It uses thresholded co-activation edges to build a circuit directly from the seed's local neighborhood. Compared with the sparse expansion family, this is less of a controlled multi-hop growing strategy and more of a straightforward rule: if a neighbor is sufficiently strong under the chosen co-activation metric, include it.

`neighborhood_expansion` is another graph-centered method. It performs a fixed two-hop neighborhood growth around the seed and tries to respect causal order when wiring edges. This is useful as a lightweight baseline because it avoids gradients entirely and instead asks whether simple graph structure alone already recovers a decent local circuit.

These methods are important even if they are not the most sophisticated, because they provide a clean reference point. If an elaborate gradient method cannot outperform a simple statistical neighborhood, that is useful information.

### 4. Attribution-based seed methods

`logit_attribution` is a classic gradient-style method. It asks which latents and edges most influence the model's output logits, using activation-times-gradient style scoring. The logic is straightforward: a feature is important if it is active and the output is sensitive to it. In practice, this tends to favor latents with direct predictive influence on the final output rather than merely upstream correlation.

`sfc_attribution_patching` is the repository's Sparse Feature Circuits style method, inspired by the literature on feature patching and integrated gradients. It computes node scores using a patch-baseline idea and uses Jacobian- or delta-based edge attribution to score interactions between features. Relative to simple logit attribution, this method tries to be more intervention-aware: instead of only asking what the local gradient says, it asks what changes when a feature state is moved between a baseline and a clean example.

`top_coact_attr` is a legacy hybrid method. It first grows a candidate neighborhood from co-activation structure and then uses multi-hop causal attribution to refine or score that neighborhood. The design reflects an older but still useful idea: propose structure statistically, then validate directionality and importance with gradients.

### 5. Positive/negative contrast methods

`differential_activation` explicitly compares **positive contexts** where the seed fires to **hard negatives** where it does not. It looks for features that are much more active in one set than the other. Features that rise in positive contexts are candidate activators; features that rise in negative contexts may be inhibitors or competitors. The method then validates putative edges with attribution so that it does not remain only a contrastive clustering method.

This family is valuable because it is closer to causal diagnosis than unconditional co-activation. A feature that appears everywhere the seed appears may still be incidental. A feature that sharply distinguishes seed-present from seed-absent contexts is often a better mechanistic clue.

`counterfactual_gradient` goes a step further. Instead of merely comparing activation totals between positive and negative sets, it runs gradient attribution on **contrast sequences** where the seed is inactive. Discovery asks which upstream features would push the seed toward its positive-context activation level; evaluation then tests that claim on positive contexts. The method is designed to discover:

- **absent activators**: upstream features that would likely make the seed fire if they were present
- **present inhibitors**: features that are actively suppressing the seed in the current negative examples

Because the seed is often absent from SAE top-k on contrast sequences, attribution uses the seed's **encoder pre-activation** (via `SeedProjectionInstrument`) rather than sparse top-k activation, so gradients still flow when the seed never fires. The loss pushes contrast pre-activation toward the seed's mean activation on positive contexts.

The repository supports different negative modes for the contrast pass, including close negatives from stored negative contexts, random negatives, and more distant negatives chosen from the corpus by SAE cosine distance from positive contexts. This gives the method flexibility in how "counterfactual" is defined.

`ablation_gradient` complements counterfactual gradient by asking the opposite question on **positive contexts** where the seed already fires: which active upstream features should be ablated to suppress it? It runs a gradient pass on positive probe sequences, scores upstream latents by first-order ablation benefit (`activation × gradient`), and keeps high-scoring features as **ablation supports**. Where counterfactual gradient finds what would activate the seed when it is absent, ablation gradient finds what causally drives it when it is present.

`hybrid_gradient` runs counterfactual and ablation gradient as separate discovery passes, then **fuses** any returned circuits into one candidate by feature ID. Nodes and edges from both sources are merged, with per-method attribution scores and roles preserved. The fused circuit is optionally pruned (leave-one-out or threshold-based), then re-evaluated. Acceptance can require counterfactual faithfulness, suppression score, both, or either, depending on configuration. This method is useful when a mechanism may include both contrast-side activators/inhibitors and positive-context support features.

### 6. Upstream gradient tracing methods

`gradient_upstream` treats discovery as a backward search problem. Starting from a seed latent, it repeatedly looks for upstream latents with large attribution scores and grows the circuit hop by hop. This is a gradient-based analogue of graph BFS: instead of exploring graph neighbors, it explores the most causally influential upstream features according to the current attribution signal.

This method is especially useful when the important mechanism is not limited to the seed's strongest co-activation partners. A causal parent can be relatively rare or diffuse in the co-activation graph yet still have strong gradient influence on the seed.

`layerwise_gradient_upstream` generalizes this idea by sweeping backward **layer by layer** rather than only by local hop structure. Instead of asking "who are the top predecessors of the current node," it asks "across all upstream layers, which features best explain this node?" This can recover broader causal structure, especially when important dependencies jump across layers or mix same-layer and cross-layer effects.

Both upstream methods are more causally ambitious than simple local graph methods, but they are also more computationally expensive and more dependent on the stability of attribution under the chosen probe contexts.

### 7. Attribution-graph and cluster-level methods

`circuit_tracer_baseline` is an SAE-adapted version of the attribution-graph style approach associated with circuit tracing. Instead of relying only on local latent neighborhoods or single-node gradient queries, it constructs a prompt-local direct-effects graph and propagates influence through that graph. The result is a more global picture of which feature nodes and edges explain downstream behavior. In spirit, it is the method in the repository that is closest to a true attribution graph baseline.

`cluster_contrast` is the most different from the rest because it is **seed-free**. Rather than selecting a single latent and building outward from it, it clusters negative-context sequence embeddings, computes cluster-specific output behavior, and then uses gradient-based contrast against other clusters to identify the features that characterize that cluster's behavior. This makes it less about "explain one seed latent" and more about "explain one region of behavioral space."

That method is especially useful when one suspects that the most meaningful units are not individual seed latents but recurring groups of examples with shared model behavior.

## How Discovery Results Are Evaluated

The repository does not treat discovery output as successful merely because a method found a graph. Each graph must earn its place through evaluation.

The central metric is **faithfulness**, which measures how well the circuit reproduces the model's original behavior relative to an ablated baseline. Intuitively, if the original model logits are the target behavior, then a faithful circuit is one whose patched or isolated execution stays close to those logits while the baseline remains much worse.

Several related metrics appear alongside faithfulness:

- **upstream faithfulness** asks whether the circuit reproduces the seed latent's activation rather than only the final output
- **sufficiency** asks whether the circuit alone is enough to recover the desired behavior
- **completeness** asks whether removing the circuit damages the original model
- **minimality** prunes away redundant nodes by leave-one-out testing
- **counterfactual faithfulness** and **posctx suppression score** evaluate the counterfactual-gradient family by intervention on seed activation rather than end-logit reconstruction
- **cluster-specific metrics** evaluate seed-free methods such as `cluster_contrast`

The counterfactual evaluation pair works by patching SAE activations on probe sequences:

- **counterfactual faithfulness** asks whether injecting discovered activators and suppressing inhibitors on contrast contexts can make the seed fire as it does on positive contexts
- **posctx suppression score** asks whether suppressing activators and injecting inhibitors on positive contexts can silence the seed

`counterfactual_gradient` accepts circuits primarily on counterfactual faithfulness. `ablation_gradient` accepts on posctx suppression score. `hybrid_gradient` can require either metric, both, or a configurable combination.

Different methods use slightly different acceptance gates. Gradient-based upstream methods often use upstream faithfulness because their direct target is the seed's activation. Sparse expansion methods often use kind-local faithfulness because they deliberately focus on one subset of component kinds while allowing others to pass through. The counterfactual/ablation/hybrid trio uses seed-intervention scores rather than logit faithfulness.

Once a circuit is accepted, the repository can also run post-analysis methods to characterize it. These include:

- overlap with the seed's top co-activation neighborhood
- distribution of nodes across layers
- concentration or dispersion of edge weights
- average and median node activity
- rarity of included nodes
- token-consistency statistics
- internal co-activation density among the discovered nodes

This helps distinguish, for example, a shallow local motif from a broad distributed circuit.

## Performance And Engineering Choices

Although the repository is research-oriented, it is also engineered for scale. The code supports:

- `torch.compile`
- multi-GPU SAE handling
- Triton Top-K kernels
- CUDA and C++ acceleration for expensive reductions and fused operations
- configurable memory strategies
- asynchronous or staged execution choices for large dataset processing

This matters because the discovery pipeline is expensive. It is not just loading one prompt and computing saliency once; it is collecting statistics across shards, building stores, running many discovery methods, and evaluating each resulting circuit.

## Main Entry Points

- `src/main.py` runs the full end-to-end pipeline.
- `src/discover_circuits.py` runs discovery starting from previously stored outputs.
- `src/display_latents.py` opens an interactive latent inspection flow based on stored contexts.
- `src/search_latents.py` searches latent contexts by keyword and can optionally run patch-style follow-up checks.
- `src/ablation_sensitivity.py` explores how candidate or circuit behavior changes under different discovery settings.

## Summary

This repository is best understood as a **configurable circuit-discovery laboratory for SAE-based transformer analysis**. Its main strength is not a single algorithm, but the combination of:

- rich latent and context stores
- explicit use of positive and negative examples
- a reusable co-activation graph
- several different discovery paradigms
- strong evaluation and pruning steps

In other words, the codebase is designed to let a researcher ask not only "which features correlate with this behavior?" but also "which sparse latent sub-network seems to causally implement it, under which discovery assumptions, and how well does that claim survive evaluation?"
