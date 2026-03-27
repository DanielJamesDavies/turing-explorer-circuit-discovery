# Circuit Discovery Methods

Each method takes a **seed latent** (identified by layer, kind, and latent index) and tries to build a `Circuit`: a directed graph of SAE features (`CircuitNode`s) connected by weighted edges.

This document is implementation-aligned with the discovery code in this directory.

---

## Quick glossary

- **Seed latent:** The feature we are trying to explain mechanistically.
- **Kind:** SAE component type (`attn`, `mlp`, `resid`).
- **Top co-activation store:** Precomputed nearest neighbors per latent (`top_coactivation.top_indices/top_values`).
- **Probe dataset:** Positive/negative token contexts built for the seed.
- **Hard negative:** Context where seed is expected but absent/suppressed.
- **Kind-local faithfulness:** Faithfulness computed while patching only selected kinds.

---

## Method selection (which one should I run?)

| Method | Needs hard negatives | Uses gradients in discovery | Relative cost | Best use |
|---|---|---|---|---|
| `CoactivationStatistical` | No | No | Very low | Fast baseline / sanity check |
| `NeighborhoodExpansion` | No | No | Low | Controlled two-hop structural expansion |
| `LogitAttribution` | No | Yes | Medium | Direct logit-causal parent/edge discovery |
| `TopCoactivationDiscovery` | No | Yes | Medium-high | Multi-hop upstream + downstream attribution |
| `DifferentialActivation` | Yes | Yes | High | Explicit activator vs inhibitor analysis |
| `SFCAttributionPatching` | Optional (depends on `patch_mode`) | Yes | Very high | Most comprehensive node+edge attribution |
| `TopCoactSparseExpansion` family | `HardNegative...` only | Base family: no, `HardNegative...`: partial | Low to medium | Kind-targeted BFS with optional passthrough |

---

## Shared prerequisites and constraints

- All methods assume a populated co-activation/statistics backend (for neighborhood lookup and activity filters).
- Gradient methods require an instrumented forward graph (`SAEGraphInstrument`) and enough memory for backward/JVP work.
- Methods that use hard negatives (`DifferentialActivation`, `HardNegativeCoactSparseExpansion`) depend on meaningful negative contexts from the probe builder.
- Most methods may return `None` early if probe positives are empty or no candidates survive thresholds/filters.

---

## Root-level methods

### 1. `CoactivationStatistical` - gradient-free baseline

**File:** `coactivation_statistical.py`

The simplest method. Reads precomputed top co-activation neighbors for the seed and includes neighbors that pass:

- `coactivation_threshold` (minimum stored co-activation weight), and
- `min_active_count` (minimum latent activity count).

Edge direction is assigned by causal order (`layer`, then kind order), and edge weights are raw co-activation scores.

**Threshold semantics:** signed weight threshold (`weight >= coactivation_threshold`, not absolute value).

**Key parameters:** `coactivation_threshold`, `max_neighbors`, `min_active_count`, `pruning_threshold`

**Typical failure modes:** no positive probe data; all neighbors filtered out by threshold/activity.

---

### 2. `NeighborhoodExpansion` - two-hop structural expansion

**File:** `neighborhood_expansion.py`

Also gradient-free. Builds a richer structure with explicit branching:

- **Hop 1:** add top co-activation neighbors of the seed.
- **Hop 2:** select top `n_expand` hop-1 nodes and expand each with `m_neighbors` additional neighbors.

Compared with threshold-only inclusion, this gives explicit control over graph density via branching factors.

**Threshold semantics:** structural top-k style expansion + activity filters.

**Key parameters:** `n_expand`, `m_neighbors`, `min_active_count`, `pruning_threshold`

**Typical failure modes:** sparse co-activation region around seed; over-expansion into weak context without pruning.

---

### 3. `LogitAttribution` - two-pass gradient discovery

**File:** `logit_attribution.py`

Uses an `SAEGraphInstrument` for two gradient passes:

- **Pass 1 (logit attribution):** backward from target logit to candidate SAE leaves, score = activation x gradient; candidates above threshold become nodes.
- **Pass 2 (feature-to-feature attribution):** for each included node, backward from that feature to upstream candidates to score directed edges.

**Threshold semantics:** absolute (`abs(score) >= threshold`) for both node and edge inclusion.

**Key parameters:** `logit_threshold`, `edge_threshold`, `max_neighbors`, `pruning_threshold`

**Typical failure modes:** only seed survives threshold; unstable attribution with very small probe batches.

---

### 4. `TopCoactivationDiscovery` - multi-hop co-activation + attribution

**File:** `top_coactivation.py`

Hybrid method:

- **Upstream tracing:** walk upstream for up to `max_hops` via co-activation candidates; keep candidates that pass gradient attribution.
- **Downstream expansion:** gather downstream co-activation candidates from included nodes and validate with attribution before adding child edges/nodes.

**Threshold semantics:** absolute attribution threshold (`abs(score) >= attribution_threshold`).

**Key parameters:** `max_hops`, `max_neighbors`, `attribution_threshold`, `min_active_count`, `pruning_threshold`

**Typical failure modes:** no candidates pass attribution threshold; upstream frontier dies early.

---

### 5. `DifferentialActivation` - pos/neg contrast + causal attribution

**File:** `differential_activation.py`

Requires hard negatives. Three phases:

- **Phase 1 (differential scan):** no-grad scans over positives/negatives; rank latents by `delta = mean(pos) - mean(neg)` to propose activators/inhibitors.
- **Phase 2 (causal validation):** gradient attribution validates upstream/downstream causal links on positive tokens.
- **Phase 3 (optional pruning + eval):** prune and evaluate with both global and kind-local faithfulness.

**Threshold semantics:** absolute attribution magnitude for inclusion; sign still matters conceptually for activator/inhibitor interpretation.

**Key parameters:** `n_activator_candidates`, `n_inhibitor_candidates`, `attribution_threshold`, `pruning_threshold`

**Typical failure modes:** weak positive/negative separation, leading to poor candidate quality.

---

### 6. `SFCAttributionPatching` - Integrated Gradients + typed edge attribution

**File:** `sfc_attribution_patching.py`

Most comprehensive method (inspired by Sparse Feature Circuits work):

- **Node attribution (IG):** interpolate SAE activations between clean and patch baseline for `ig_steps`, average gradients, and score node effects.
- **Edge attribution:** compute edge scores across explicit edge types with stop-gradient isolation logic; keep edges above threshold.

Patch baseline behavior depends on `patch_mode` (for example, zero or negative-driven baselines).

**Threshold semantics:** absolute thresholds for node/edge score magnitude.

**Key parameters:** `node_threshold`, `edge_threshold`, `ig_steps`, `patch_mode`, `min_faithfulness`

**Typical failure modes:** high compute/memory cost; only seed passes node threshold.

---

## `top_coact_expansion/` family

These methods share the same BFS engine in `TopCoactSparseExpansion`. Subclasses differ only in:

- **target kinds:** kinds expanded through BFS co-activation traversal.
- **passthrough kinds:** kinds added from active no-grad instrumentation without BFS.

### Base class: `TopCoactSparseExpansion`

**File:** `top_coact_expansion/top_coact_sparse_expansion.py`

Performs variable-depth BFS over co-activation neighbors, restricted to `target_kinds` at each hop. Optional passthrough stage adds active nodes of `passthrough_kinds`.

This family computes both:

- global faithfulness, and
- kind-local faithfulness (restricted to `target_kinds`),

and uses **kind-local faithfulness** as the acceptance gate.

**Key parameters:** `coact_depth` (for example `[32, 32]`), `min_active_count`, `pruning_threshold`, `probe_batch_size`

### Concrete subclasses

| Class | File | Target kinds | Passthrough kinds |
|---|---|---|---|
| `AllTopCoactSparseExpansion` | `all_top_coact_sparse_expansion.py` | attn, mlp, resid | *(none)* |
| `ResidTopCoactSparseExpansion` | `resid_top_coact_sparse_expansion.py` | resid | attn, mlp |
| `MlpTopCoactSparseExpansion` | `mlp_top_coact_sparse_expansion.py` | mlp | attn, resid |
| `AttnTopCoactSparseExpansion` | `attn_top_coact_sparse_expansion.py` | attn | mlp, resid |
| `AttnMlpTopCoactSparseExpansion` | `attn_mlp_top_coact_sparse_expansion.py` | attn, mlp | resid |
| `AttnResidTopCoactSparseExpansion` | `attn_resid_top_coact_sparse_expansion.py` | attn, resid | mlp |
| `MlpResidTopCoactSparseExpansion` | `mlp_resid_top_coact_sparse_expansion.py` | mlp, resid | attn |

These variants help align expansion with the mechanism you care about (for example, residual-path-focused vs attention-focused analysis).

---

### `HardNegativeCoactSparseExpansion` - BFS activators + gradient-validated inhibitors

**File:** `top_coact_expansion/hard_negative_coact_sparse_expansion.py`

Extends `TopCoactSparseExpansion` with inhibitor discovery:

- **Activator phase:** standard BFS co-activation expansion.
- **Inhibitor phase:** collect top active latents from hard negatives, then keep only those that show suppressive attribution toward the seed on positive tokens.

Like the base family, this method computes global and kind-local faithfulness and gates on kind-local faithfulness.

**Key parameters:** `coact_depth`, `neg_candidate_limit`, `attribution_threshold`, `min_active_count`

---

## Shared evaluation and acceptance gate

All methods evaluate:

- `faithfulness`
- `sufficiency`
- `completeness`

### Metric definitions (implementation-accurate)

| Metric | Definition in code | Interpretation |
|---|---|---|
| **Faithfulness** | `1 - MSE(circuit_logits, original_logits) / MSE(baseline_logits, original_logits)` | Higher means circuit intervention recovers more of the original behavior. |
| **Sufficiency** | `exp(log_prob_circuit(target) - log_prob_original(target))` | Higher means circuit-only model preserves target-token probability better. |
| **Completeness** | `1 - faithfulness(complement)` where complement ablates only circuit nodes | Higher means less explanatory signal remains outside the circuit. |

Acceptance rule:

- Most methods: reject if `faithfulness < min_faithfulness`.
- `TopCoactSparseExpansion` family and `DifferentialActivation`: compute both global and kind-local faithfulness and gate on the kind-local score.

---

## Common metadata emitted on accepted circuits

Most methods write shared metadata fields such as:

- `faithfulness`, `sufficiency`, `completeness`
- seed identifiers (`seed_comp`, `seed_latent`)
- graph size (`n_nodes`, `n_edges`)
- `discovery_method`

Some methods add method-specific metadata (for example hop counts, local faithfulness breakdowns, thresholds, patch mode).
