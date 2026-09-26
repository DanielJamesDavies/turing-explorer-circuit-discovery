# Ideas

## Data Collection

### Top-Token Co-activation Store

The current co-activation graph (`top_coactivation`) averages activations across all token positions per sequence before scoring. This collapses the temporal dimension — two latents firing at opposite ends of the same sequence get the same co-activation score as two firing at the exact same token.

Build an alternative `top_coactivation_peak` store that computes co-activation at the specific token position where the target latent fires most strongly (argmax), rather than averaging across all T positions. This asks the sharper question: "when latent X peaks at position t, what other latents are active at exactly position t?"

The older per-latent data collection codebase (`x:\Projects\AIs\Turing\latent_connections`) collected both variants — sequence-average and top-token — and stored them separately. The current pipeline only does sequence-average.

Store both variants and let discovery methods choose which to use or combine them.

**Pipeline stage:** Pass 2 (`run_second_pass`), inside `top_coactivation.update_batch`. Requires knowing the argmax position per target latent per sequence, which means either pre-computing argmax from Pass 1 top_ctx data or computing it on-the-fly during the dump from the latent activations `[B, T, K]`.

**Resources:** The **dominant VRAM cost** (model weights + SAE bank) is identical — it is the same forward pass. Running both variants in the same pass adds two secondary costs:

- **Intermediate batch tensors** (`[B, M]` cand_vals/cand_ids on GPU): computing both profiles simultaneously doubles these, but they are tiny (a few KB at `B=4`, `M=256`) — negligible.
- **Final result tensors** in `fast` mode: `top_indices` + `top_values` live on GPU until reduce completes. A second set for the peak variant adds roughly 720 MB in `fast` mode (`36 × 40960 × 64 × 8 bytes`). In `efficient` mode these tensors stay on CPU so there is no extra VRAM cost.
- **CPU RAM**: the dump buffers (`candidate_ids`, `candidate_vals`) are already on CPU. A second set doubles that allocation: `S × M × 8 bytes` extra, a few hundred MB depending on top_ctx size.
- Disk: one additional `top_coactivation_peak.pt` file (roughly same size as `top_coactivation.pt`). The C++ reduce would need a second invocation for the peak variant.

---

### Logit Lens Per Layer

The older codebase projects intermediate residual stream representations through `norm_f` + `lm_head` at each layer to see what the model "thinks" at that point in computation. The current `logit_ctx` only captures the final-layer logits.

Add intermediate-layer logit predictions to `logit_ctx`. The model and unembedding head are already in memory during Pass 1; projecting through `norm_f` + `lm_head` at each layer costs one matmul per layer per batch.

This reveals at which layer a concept "crystallizes" and could inform multi-resolution patching discovery.

**Pipeline stage:** Pass 1 (`run_first_pass`), inside `activations_callback`. Currently logit_ctx is only updated from the final `last_logits` returned by the model forward. Would need to run `norm_f` + `lm_head` on each layer's resid activation inside the callback.

**Resources:** VRAM: one `[B, vocab_size]` tensor per layer per batch for the logit projection — `512 × 50304 × 2 bytes × 12 layers ≈ 590 MB` if all held simultaneously, but can be computed and consumed one layer at a time (approx 49 MB peak). CPU/disk: `logit_ctx` storage grows 12× if storing per-layer predictions (currently `[36, d_sae, 32, 32]` for top probs/indices — would need a parallel structure per layer). Compute: 12 additional matmuls of shape `[B, 1024] × [1024, 50304]` per batch, but these are small relative to the main model forward.

---

### ~~Position-Aware PMI Co-activation~~ ✅ Implemented (see `agent-planning/pmi-coactivation-modes.md`)

These two ideas compose naturally into a single metric. Standard PMI uses sequence-level co-occurrence (both fire somewhere in the same sequence). Position-aware PMI uses **token-level co-occurrence** (both fire at the same token position), which is strictly more informative:

```
PMI_position(A, B) = log( P(A & B at same token) / (P(A at any token) * P(B at any token)) )
```

This replaces the ad-hoc `1/log(count+1)^alpha` frequency adjustment with a principled information-theoretic measure, while simultaneously capturing the temporal co-firing signal. High PMI means "these fire at the same positions much more than expected by chance."

One position-aware PMI store subsumes both the positional overlap idea and the frequency adjustment idea.

**Pipeline stage:** Pass 2 (`run_second_pass`), as an alternative scoring function inside `top_coactivation.update_batch`. Instead of (or alongside) the current mean-magnitude + frequency-adjustment scoring, compute token-level co-occurrence counts. Needs `P(A)` per latent (token-level firing rate), which can be derived from `latent_stats.active_count` divided by total tokens processed. `P(A & B)` must be accumulated during the dump — for each sequence, for each token position, identify all active latent pairs at that position and increment their joint count.

**Resources:** VRAM: the expensive part is the per-token pair counting. Cannot enumerate all pairs at each token. Instead, for each target latent's top-K co-active latents at each token, accumulate joint counts in the existing candidate dump tensors. This is roughly the same VRAM as the current dump. CPU RAM: same dump buffer structure. The PMI computation itself is a lightweight post-processing step during reduce (divide counts, take log). Compute: slightly more work per token than the current mean-then-topk approach, since you need per-token topk instead of per-sequence mean topk.

---

### Negative Co-activation Store

During Pass 2, in addition to tracking top co-activating latents, also track top anti-co-activating latents: for each target latent, which other latents are most active when the target is inactive (across the target's neg_ctx sequences)? This pre-computation would make inhibitor discovery much cheaper.

**Pipeline stage:** New sub-step after the ANN step and before or during Pass 2. Requires `neg_ctx` to be built first. Could be structured as a "Pass 2b" that runs the model on neg_ctx sequences (similar to how Pass 2 runs on top_ctx sequences) and records which latents are active. Alternatively, it could piggyback on Pass 2 if neg_ctx sequence IDs are included in the second pass batch loop.

**Resources:** VRAM: same as Pass 2 — model + SAE bank in VRAM for forward + encode. The neg_ctx sequences are a separate set from top_ctx, so this is additional forward passes (not free). CPU RAM: a second set of dump buffers for the negative co-activation candidates — same structure as `top_coactivation` candidate buffers. With `neg_ctx.n_sequences=64` per latent, the total unique neg sequences may be comparable to the top_ctx set (tens of thousands). Disk: one additional `neg_coactivation.pt` file. Compute: roughly doubles the Pass 2 wall time since you're running a second batch of sequences through the model.

---

### Activation Correlation Co-activation (Pearson ρ)

During Pass 2, compute the Pearson correlation between pairs of latent activations across shared active positions. Strong positive correlation suggests co-regulation; strong negative correlation suggests competition/inhibition.

Naive pairwise correlation over 1.5M latents is O(n^2), but this is tractable if scoped to co-activation candidate pairs only. During the second pass dump, for each target latent's top-N co-activation candidates, maintain a running bivariate Welford update (5 scalars per pair: n, mean_x, mean_y, C_xx, C_xy). Memory cost is bounded by `n_latents * n_neighbors * 5 floats` = 1.5M x 64 x 20 bytes, roughly 1.9 GB — fits in RAM. Compute per pair per sequence is approx 10 FLOPs. The single-variable Welford infrastructure already exists in `latent_stats`.

**Pipeline stage:** Pass 2 (`run_second_pass`), extending `top_coactivation.update_batch`. After identifying the top-N co-activation candidates for each target latent in a batch, update the bivariate Welford accumulators for those pairs. Requires a two-pass approach within Pass 2: first run the existing dump to establish which pairs are candidates, then run a second accumulation pass (or accumulate incrementally if the candidate set is known from a prior run). Alternatively, maintain Welford accumulators for all pairs seen during the dump and only retain the top-N after reduce.

**Resources:** CPU RAM: approx 1.9 GB for the bivariate Welford accumulators (`n_components × d_sae × n_neighbors × 5 floats`). This is the dominant cost. VRAM: no additional VRAM beyond the current Pass 2 — the Welford updates happen on CPU after the GPU dump. Compute: approx 10 FLOPs per pair per sequence on CPU, totalling millions of updates per batch — likely adds seconds per batch, not minutes. Disk: correlation values can be stored alongside `top_coactivation.pt` as additional tensors (same shape as `top_values`).

---

## Circuit Discovery Methods

### Implement Gradient-Differential Discovery

Already designed in `docs/gradient-differential-discovery.md` but not yet implemented. Combines gradient scan for activators with differential scan + binary-split ablation for inhibitors. The binary-split ablation is O(log N) forward passes instead of O(N).

**Pipeline stage:** Discovery phase (`run_discovery` → `DiscoveryWindow.run`). New `DiscoveryMethod` subclass registered in `METHOD_REGISTRY`. Runs per-seed after probe dataset construction, same as all other methods.

**Resources:** VRAM: 1 instrumented grad-enabled forward pass (same as `LogitAttribution` / `DifferentialActivation`), plus approx 25 no-grad forward passes for binary-split ablation on neg_ctx (small batch, typically 16 sequences). The instrumented forward is the VRAM peak — requires the full computation graph in memory. With `probe_batch_size=4`, this is manageable. Compute: 20-30 forward passes total per seed (see design doc table). Comparable to `DifferentialActivation`, cheaper than `SFCAttributionPatching`.

---

### Counterfactual Gradient Discovery — see agent-planning/counterfactual-gradient-discovery.md

Discover circuit nodes by running gradient attribution on **negctx sequences** — inputs that are semantically similar to the seed's context but where the seed latent is inactive. This answers a different question from all existing gradient methods: not "what is causing the seed to fire?" but "what is causally different between the contexts where the seed fires and where it doesn't?"

**Two types of nodes discovered:**

1. **Absent activators** — upstream latents with a strong positive gradient `∂(seed_act)/∂(latent_U)` on negctx, but low/zero activation on those sequences. These latents would cause the seed to fire if they were active, but they aren't. Their absence explains why the seed doesn't fire on negctx despite the semantic similarity.
2. **Present inhibitors** — upstream latents with a strong negative gradient `∂(seed_act)/∂(latent_U)` and high activation on negctx. These latents are actively suppressing the seed from firing. Reducing them would allow the seed to activate.

**Algorithm:**

1. Get negctx sequences for seed S (from the `neg_ctx` store, built in Pass 3).
2. Run `SAEGraphInstrument` forward pass on negctx with `grad_enabled=True`.
3. Extract pos_argmax from the graph (position of the seed's highest activation on negctx — likely low but non-zero).
4. Score ALL latents across ALL upstream layers and kinds (layers ≤ seed_layer) — the gradient flows through the full upstream computation graph in one backward pass, so there is no need to restrict to direct predecessors or do BFS hops.
5. For absent activators: compute raw gradient `∂(seed_act)/∂(latent_U)` for all latents in predecessor components (using `compute_feature_gradient` in `attribution.py`, which already exists and returns raw gradient rather than activation × gradient). Select top-K by positive gradient magnitude.
6. For present inhibitors: compute `activation(U) × gradient` (using `compute_latent_upstream_scores` or similar). Select top-K by magnitude where `activation × gradient < 0` (negative contribution = inhibitory).
7. Add both sets to the circuit with roles `"counterfactual_activator"` and `"counterfactual_inhibitor"`. Add directional edges.
8. Evaluate using layer-bounded evals (same as Gradient Upstream Discovery: `upstream_faithfulness` + standard evals restricted to layers ≤ seed_layer).

**What distinguishes this from existing methods:**

- All other gradient methods (`logit_attribution`, `sfc_attribution_patching`, `gradient_upstream`) run on posctx. This is the only gradient method that runs on negctx.
- `hard_negative_coact_sparse_expansion` and `differential_activation` also use negctx but statistically (co-occurrence / mean difference). This method gives *causal* attribution — a gradient-based answer to "what is preventing the seed from firing here?"
- The absent activator signal is unique: `compute_feature_gradient` (raw gradient, unscaled by activation) finds latents that *would* help if active, regardless of whether they currently are. No existing method finds this class of node.

**Reuse of existing infrastructure:** `SAEGraphInstrument`, `get_predecessor_components`, and `compute_feature_gradient` (already in `attribution.py`) are all reusable directly. The main new logic is the negctx context selection and the dual activator/inhibitor scoring pass.

**Pipeline stage:** Discovery phase. New `DiscoveryMethod` subclass. Requires negctx to have been built (Pass 3). Single grad-enabled forward pass per seed on negctx sequences (no multi-hop BFS).

**Resources:** VRAM: one `SAEGraphInstrument` forward pass on negctx batch — same peak VRAM profile as one hop of Gradient Upstream Discovery. Compute: one grad-enabled pass + one backward, lighter than SFC/IG methods. No multi-hop iteration. Negctx sequences are already in RAM after Pass 3.

---

### Gradient Upstream Discovery (Per-Node Context Switching) — see agent-planning/gradient-upstream-discovery.md

Discover circuits by propagating gradient attribution **backwards through the model**, using each latent's own context at each recursive hop rather than always using the seed's context.

**Algorithm:**

1. Start with the seed latent S at (layer L, kind K).
2. Identify S's direct predecessor components using the transformer's residual arithmetic:
  - `attn_L` predecessors: `resid_(L-1)` only
  - `mlp_L` predecessors: `resid_(L-1)` and `attn_L`
  - `resid_L` predecessors: `resid_(L-1)`, `attn_L`, and `mlp_L`
3. Run a grad-enabled forward pass on S's `top_ctx` sequences. Compute `∂activation(S) / ∂activation(U)` for every latent U across the predecessor components, scaled by U's actual activation value: `score(U) = |∂a_S/∂a_U| × a_U`. The gradient automatically captures both direct residual-stream paths and indirect paths simultaneously — no manual path enumeration needed.
4. Select the top-K upstream latents by score. These become the next layer of circuit nodes.
5. For each selected upstream latent U, **switch context**: use U's own `top_ctx` sequences (not the seed's) to compute `∂activation(U) / ∂activation(V)` for V in U's predecessor components. Select top-K again.
6. Repeat for a configurable number of hops (depth), building a backwards BFS tree rooted at the seed.
7. Evaluate the final set of nodes for faithfulness.

**Why per-node context switching matters:** Every existing discovery method runs all gradient attribution on the *seed's* context throughout, even for upstream latents. A latent at layer 4 may be barely active on the seed's layer-8 sequences, making its attribution signal noisy. Switching to that latent's own `top_ctx` grounds each attribution step in the input distribution where the latent is most reliably active.

**What distinguishes this from existing gradient methods:**

- `logit_attribution` and `sfc_attribution_patching` compute `∂(output logit)/∂(latent)` — influence on the model's output. This method computes `∂(latent A)/∂(latent B)` — direct causal influence between latents.
- All existing BFS methods use the seed's ctx for every hop. This method uses each latent's own ctx at its respective hop.
- The resulting circuit is a true causal upstream tree: each edge means "this upstream latent causally drove that downstream latent's activation," grounded in the most informative context for that upstream node.

**Pipeline stage:** Discovery phase. New `DiscoveryMethod` subclass. Runs per-seed. Requires grad-enabled forward passes per hop. Reads `top_ctx` for each selected upstream latent at each recursion level.

**Resources:** VRAM: one grad-enabled forward pass per hop per latent batch (same VRAM profile as `LogitAttribution`). Compute: `depth × top_K × (1 grad forward + 1 backward)` per seed. With depth=3, top_K=16, this is ~48 grad-enabled passes — heavier than the BFS expansion methods but lighter than `SFCAttributionPatching` (which uses `ig_steps × n_submodules` passes). The context-switching means each hop may load different sequences from disk, adding some I/O cost.

---

### Frequency-in-Context Circuit Discovery

Implement the approach from the original `latent_connections` codebase as a new discovery method: given a seed and its probe dataset, count which other latents fire consistently across the probe's positive sequences. Latents firing on >80% of positive sequences are almost certainly in the circuit.

This is the original's simple-but-effective Method 1, scoped to one seed's context rather than a global task dataset. The bet is that a seed's top_ctx is task-specific around that latent's function — we just don't know what the task is.

**Pipeline stage:** Discovery phase. New `DiscoveryMethod` subclass. Runs per-seed. Only needs no-grad forward passes on the probe's positive sequences to collect per-latent firing counts.

**Resources:** VRAM: model + SAE bank for no-grad forward passes on the probe batch. No gradient graph needed — this is the cheapest possible discovery method in terms of VRAM. With `probe_batch_size=4` and 64-token sequences, peak VRAM is just the model forward + SAE encode. CPU: trivial counting. Compute: one no-grad forward pass over all positive sequences (can be batched), plus the frequency counting. Faster than every existing method except `CoactivationStatistical` (which doesn't even need a forward pass).

---

### Probe Dataset Candidate Pre-filters

Three lightweight, no-grad filters that run on the seed's probe dataset to cheaply narrow the candidate latent space before invoking any expensive gradient-based or ablation-heavy discovery method. Each produces a ranked or filtered list of candidate latents from the full 1.5M space.

**Filter 1 — Per-latent Mean Ablation Effect**

For each sequence in the seed's probe dataset and for each candidate latent, replace that latent's activation with its global mean value (from `latent_stats.mean`), then re-run the forward pass and check whether the seed latent's own activation changes. If ablating a latent to its mean has no effect on the seed latent's value, that latent is not in the causal path that produces the seed's activation — discard it.

This is distinct from all existing discovery methods, which ablate latents and measure output logit changes. This filter asks the upstream question: what causes the seed to fire, rather than what does the seed cause? It directly finds the input circuit to the seed.

Practically: for a seed at component `c`, latent `l`, run a patched forward pass setting each candidate's activation to mean and measure `|activation_patched[c, l] - activation_clean[c, l]|`. Candidates with delta below a threshold are discarded.

**Filter 2 — Dataset-Level Statistical Significance**

For each latent, compare its activation values across the seed's probe sequences to its global mean (from `latent_stats`). Compute a z-score: `(probe_mean - global_mean) / global_std`. Latents with a z-score above a threshold are significantly over-active on the seed's context relative to their baseline and are strong candidates for circuit membership. Latents near z=0 are activated no differently than on a random sequence — they are likely bystanders.

This requires no forward passes at all — the probe sequences' latent activations are already computed during probe dataset construction, and the global statistics already exist in `latent_stats.mean` and `latent_stats.std()`. It is the cheapest possible filter: pure arithmetic on existing data.

**Filter 3 — Firing Frequency Over Probe Dataset**

For each latent, count how many sequences in the seed's probe dataset it fires on (activation > 0). Express as a fraction of total probe sequences. Latents firing on a high fraction of probe sequences (e.g., >70%) are structural co-members of the seed's context. Latents firing on very few probe sequences (e.g., <10%) are unlikely to be consistent circuit members.

This is equivalent to the core of Frequency-in-Context Circuit Discovery but used as a pre-filter rather than a standalone method — the resulting frequency-ranked list feeds into any downstream method rather than being the final circuit.

**Combining the filters:** The three filters compose naturally. Apply Filter 3 first (cheapest, no passes needed if activations are cached), Filter 2 next (also free given the data), then Filter 1 for the remaining candidates that passed both thresholds. This cascading structure means the expensive per-latent ablation passes in Filter 1 are only run on a small fraction of the 1.5M latents.

**Pipeline stage:** Discovery phase, immediately after probe dataset construction in `DiscoveryWindow.run`, before the main discovery method's candidate scoring. Could be implemented as a shared `ProbeCandidateFilter` utility that all `DiscoveryMethod` subclasses can optionally invoke to pre-screen candidates.

**Resources:** Filter 2 and 3: VRAM zero, CPU negligible — computed from existing cached data. Filter 1: one no-grad forward pass per latent being tested (or, more efficiently, batch-test multiple latents simultaneously by patching all of them in parallel using a block-diagonal patch). With a probe batch of 4 sequences and testing 1,000 candidate latents (post Filter 2/3 culling), this is 1,000 forward passes — comparable to a single step of Greedy Additive Construction.

---

### Greedy Additive Construction

Directly optimize for faithfulness by greedily building the circuit one node at a time:

1. Start: circuit = {seed}. Evaluate faithfulness.
2. For each candidate in the co-activation neighborhood, temporarily add it and measure Δfaithfulness.
3. Permanently add the candidate with the largest Δfaithfulness.
4. Repeat until faithfulness plateaus or budget exhausted.

This directly optimizes the metric and naturally produces minimal circuits without a separate pruning step.

**Pipeline stage:** Discovery phase. New `DiscoveryMethod` subclass. Runs per-seed.

**Resources:** VRAM: model + SAE bank for no-grad forward passes. Each faithfulness evaluation requires 3 forward passes (original, circuit, baseline). Each greedy step tests K candidates × 3 passes. Compute: O(N × K × 3) forward passes where N = final circuit size and K = candidates per step. With N=20 and K=64, that's approx 3,840 forward passes per seed — expensive but all no-grad on small probe batches. Could be optimized by caching the original and baseline passes (they don't change per candidate). With caching: O(N × K) forward passes (circuit pass only). Wall time: with `probe_batch_size=4` and cached original/baseline, each candidate test is one forward pass (approx 5-10ms), so 64 tests x 20 steps x 10ms, roughly 13 seconds per seed.

---

### Multi-Resolution Patching Discovery

Work top-down from coarse to fine resolution:

1. **Layer scan:** Ablate entire layers, find which matter most.
2. **Kind scan:** Within important layers, ablate each kind.
3. **Latent scan:** Within important (layer, kind) pairs, use gradient attribution or greedy addition to find specific latents.

Dramatically cheaper than scoring all 1.5M latents at once.

**Pipeline stage:** Discovery phase. New `DiscoveryMethod` subclass. Runs per-seed.

**Resources:** VRAM: model + SAE bank for no-grad forward passes during layer/kind scans. The latent scan phase may optionally use grad-enabled passes (if using attribution), which requires more VRAM. Compute: Layer scan = 12 × 3 forward passes. Kind scan = 6-12 x 3 forward passes (only important layers). Latent scan = depends on method chosen for the final phase. Total: 50-100 forward passes for the coarse phases, plus the fine phase cost. Much cheaper than SFC's `ig_steps × n_submodules` passes.

---

### Spectral Community Detection

Treat the co-activation graph as an adjacency matrix:

1. Extract the local subgraph around the seed (2-3 hops, approx 500 nodes).
2. Compute the normalized graph Laplacian.
3. Find the Fiedler vector (2nd smallest eigenvector) to partition the graph.
4. The seed's partition forms the initial circuit.
5. Validate with a faithfulness check.

Fundamentally different from BFS — finds globally coherent communities.

**Pipeline stage:** Discovery phase. New `DiscoveryMethod` subclass. Runs per-seed. Reads from pre-computed `top_coactivation` store (no model forward needed for the spectral part).

**Resources:** VRAM: none for the spectral computation itself (runs on CPU with scipy/numpy). Only needs VRAM for the final faithfulness validation (3 forward passes). CPU RAM: the local subgraph adjacency matrix is 500x500 floats = 1 MB. Eigendecomposition of a 500x500 matrix is instant. Compute: the cheapest discovery method alongside `CoactivationStatistical` — the only forward passes are for evaluation.

---

### Iterative Refinement / Hill-Climbing

Start with a rough circuit from a fast method, then iteratively improve:

1. For each non-seed node, try swapping it with candidates from its co-activation neighborhood.
2. Keep swaps that improve faithfulness.
3. Repeat until no swap improves faithfulness.

**Pipeline stage:** Discovery phase. Could be a standalone `DiscoveryMethod` or a post-processing wrapper around any other method's output. Runs per-seed.

**Resources:** VRAM: model + SAE bank for no-grad forward passes. Compute: each swap test = 3 forward passes (faithfulness eval). Per iteration: `n_nodes × n_candidates × 3` passes. With a 20-node circuit and 16 candidates per node, one iteration = approx 960 forward passes. Typically converges in 2-5 iterations. Total: 2,000-5,000 forward passes per seed. Expensive, but all no-grad on small probe batches. Can be optimized with caching (original and baseline passes are constant).

---

### Random Walk Ensemble

Replace deterministic BFS with stochastic exploration:

1. Run N independent random walks from the seed on the co-activation graph.
2. Transition proportional to co-activation weight.
3. Count visit frequency per node across all walks.
4. Circuit = seed + top-K most frequently visited nodes.
5. Validate edges with one gradient pass.

**Pipeline stage:** Discovery phase. New `DiscoveryMethod` subclass. Runs per-seed. Reads from pre-computed `top_coactivation` store.

**Resources:** VRAM: none for the random walks (pure CPU graph traversal). Only needs VRAM for the final evaluation (3-4 forward passes). CPU RAM: negligible — just visit counters per node. Compute: N random walks of approx 10 steps each on a pre-loaded graph is microseconds. The only real cost is the evaluation forward passes. One of the cheapest methods overall.

---

### Contrastive Pair Discovery

Discover circuits for pairs of related seeds simultaneously:

1. Find two seeds with similar top-ctx centroids but different firing patterns.
2. Discover circuits independently.
3. Compute shared_circuit = intersection, unique_A = A \ B, unique_B = B \ A.
4. Evaluate each sub-circuit for faithfulness on its respective seed.

Tests modularity and composition in the network.

**Pipeline stage:** Post-discovery analysis or a wrapper that runs during the discovery phase. Requires two seeds' circuits to already be discovered (or discovers them inline). The pairing step needs `seq_repr` centroids or `logit_ctx` data.

**Resources:** VRAM: 2× the cost of whichever underlying discovery method is used (one per seed). The intersection/union computation is pure CPU. Additional evaluation forward passes for the sub-circuits (3 passes × 3 sub-circuits × 2 seeds = 18 forward passes). Compute: dominated by the underlying discovery method cost × 2.

---

### Attention-Aware Circuit Discovery

During the probe forward pass, capture raw attention patterns. For latents in attention SAEs, identify which source positions they attend to most strongly. Prioritize co-activation expansion toward latents whose active positions correspond to attended-to positions.

**Pipeline stage:** Discovery phase. Extends the existing probe forward pass to also capture attention weights. Could augment any BFS-based method (e.g., `TopCoactSparseExpansion` family).

**Resources:** VRAM: attention weights for all heads at all layers = `n_layers × n_heads × T × T × B` = `12 × 16 × 64 × 64 × 4 × 4 bytes ≈ 12 MB` — negligible. The main cost is the same as the current probe forward pass. Compute: minor additional processing to extract attention patterns and cross-reference with co-activation candidates.

---

## Config Options to Add

### Ablation Baseline Mode

Add a config option to choose between zero ablation and mean ablation for the non-circuit baseline:

```yaml
discovery:
  ablation_baseline: "zero"  # "zero" | "mean"
```

When `"mean"`, set `avg_acts` from `latent_stats.mean` instead of zeros. The original `latent_connections` codebase used mean ablation and achieved near-perfect faithfulness — the model stays closer to its natural operating regime.

**Pipeline stage:** Discovery phase initialization (`DiscoveryWindow.__init__`). Currently `avg_acts` is hardcoded to zeros. With `"mean"`, load from `latent_stats.mean` (already in memory). One-line change.

**Resources:** No additional resources. `latent_stats.mean` is already allocated (`[36, 40960]` float32, approx 5.6 MB). The tensor is just copied into `avg_acts` instead of zeros.

---

### Probe Dataset Positive Context Source

Add a config option to control which contexts build the positive probe dataset:

```yaml
discovery:
  probe_positive_source: "top_and_mid"  # "top_and_mid" | "top_only"
```

`"top_only"` uses only top_ctx (strongest activations) for positive sequences, giving a tighter, more focused probe. `"top_and_mid"` (current behavior) includes mid-band sequences for more coverage.

**Pipeline stage:** Discovery phase, in `ProbeDatasetBuilder.build_for_latent`. Currently concatenates `top_ctx` and `mid_ctx` sequence IDs. With `"top_only"`, skip the `mid_ctx` concatenation.

**Resources:** No additional resources. Actually reduces resource usage in `"top_only"` mode — fewer sequences loaded and forwarded during probe dataset construction.

---

## Experiments to Run

### Ablate top_ctx with neg_ctx Latent Values

For a given seed latent, take its top_ctx sequences (where it fires strongly) and replace the seed's activation values with the values observed in neg_ctx sequences (where the seed is suppressed/absent). Measure if the logits change significantly.

If the logits shift dramatically, it confirms the seed is genuinely causal for the model's behavior. If they don't change much, the negative contexts aren't different enough in the dimensions that matter.

Can extend this further: replace the entire latent vector at the argmax position with the neg_ctx latent profile to test whether neg_ctx sequences are good counterfactuals.

**Pipeline stage:** Standalone analysis script (like `ablation_sensitivity.py`). Runs after the full pipeline has produced `top_ctx.pt`, `neg_ctx.pt`, and model/SAE weights are available.

**Resources:** VRAM: model + SAE bank for forward passes. Two forward passes per seed tested: one clean, one with patched activations. For a sample of 32 seeds x 2 passes x `probe_batch_size=4` sequences, total is approx 256 forward passes. Compute: minutes, not hours.

---

### Shared Logit Prediction ↔ Co-activation Correlation

Test whether latents predicting the same output token appear in each other's co-activation lists more than chance:

1. For each token in the vocabulary, find all latents whose top logit prediction is that token (from `logit_ctx`).
2. For each such group, measure what fraction of pairs appear in each other's `top_coactivation` top-64.
3. Compare to the base rate (random pairs).

Strong correlation validates that co-activation captures functional circuits (latents contributing to the same prediction). Weak correlation suggests co-activation captures something else (positional co-occurrence, topic similarity) and a different metric is needed.

All data already exists in `logit_ctx` and `top_coactivation` — purely analytical, no new data collection needed.

**Pipeline stage:** Standalone analysis script. Runs after pipeline completion. Loads `logit_ctx.pt` and `top_coactivation.pt` from disk.

**Resources:** VRAM: none (pure CPU analysis). CPU RAM: `logit_ctx` tensors (few hundred MB) + `top_coactivation` tensors (1.5M x 64 x 8 bytes, roughly 730 MB). Compute: iterate over vocab tokens, group latents, check co-activation membership — seconds to minutes depending on implementation.

---

### Co-activation ↔ Logit Prediction Relationship

Investigate whether there is a systematic relationship between a latent's co-activation neighbors and its logit predictions. Specifically:

1. For each latent, get its top co-activation neighbors from `top_coactivation`.
2. For each latent, get its top predicted tokens from `logit_ctx`.
3. For each co-activation neighbor pair (A, B), measure whether the neighbor's logit predictions are related to the target's — do co-activating latents predict related tokens, the same token, or completely unrelated tokens?
4. Compute statistics: what fraction of a latent's top co-activation neighbors share at least one top predicted token? How does co-activation rank correlate with logit prediction overlap?

If co-activation neighbors tend to predict related tokens, it validates that the co-activation graph is capturing functional structure (latents that work together toward the same output). If co-activation neighbors predict unrelated tokens, the co-activation graph may be capturing positional or structural patterns rather than functional ones.

All data already exists in `logit_ctx` and `top_coactivation`.

**Pipeline stage:** Standalone analysis script. Same as above — loads from disk after pipeline completion.

**Resources:** VRAM: none (pure CPU analysis). CPU RAM: same as above (approx 1 GB for both stores). Compute: for each of 1.5M latents, look up 64 neighbors and compare logit prediction overlap — O(1.5M x 64 x 32) comparisons, runs in seconds with vectorized numpy/torch.