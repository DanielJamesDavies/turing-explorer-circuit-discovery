# Plan: Layerwise Gradient Upstream Discovery

> **Goal:** Implement `LayerwiseGradientUpstreamDiscovery` — a variant of `GradientUpstreamDiscovery` that replaces per-node BFS with a layer-by-layer iterative sweep and extends gradient attribution to **all** upstream layers rather than only the direct causal predecessors.
>
> **Created:** 2026-04-12

---

## Background & Motivation

### Current algorithm (`GradientUpstreamDiscovery`)

1. BFS starting from the seed node.
2. Each dequeued node runs `_run_hop`, which calls `get_predecessor_components` to obtain only the **direct causal predecessors** (e.g. `resid@L−1`, `attn@L`, `mlp@L`) and passes those as `predecessor_comp_indices` to `compute_latent_upstream_scores`.
3. Newly discovered activator nodes are enqueued for future expansion; depth is tracked with an integer counter.

### Limitations addressed

* Gradient signal can skip layers via the residual stream — a latent at layer L−3 may strongly drive a latent at layer L, but the current hop only ever sees layer L−1.
* Each BFS node independently chooses its own context and runs its own hop, so there is no coordination between nodes at the same (layer, kind).
* Depth counting is a coarse proxy; the natural bound is the layer index itself.

### New algorithm (`LayerwiseGradientUpstreamDiscovery`)

```
1. Initialise work-queue with the seed's (layer, kind).
   Work-queue is a priority queue ordered by layer (highest first).

2. While work-queue is non-empty:
   a. Dequeue the highest-layer (layer, kind) pair.
   b. Collect ALL circuit nodes that live at this (layer, kind).
   c. For each such node:
        - Build its probe-dataset (ctx tokens, up to max_ctx_sequences).
        - Run _run_hop against ALL upstream component indices
          (every (layer', kind') where layer' < layer, across every kind).
        - Add passing nodes to the circuit.
        - For each newly added node's (layer', kind'), if not yet processed
          and layer' ≥ min_layer, enqueue (layer', kind').
   d. Mark (layer, kind) as processed.

3. Evaluate & prune the fully-built circuit (same as current method).
```

**Key differences from current:**

| Dimension | Current (`GradientUpstreamDiscovery`) | New (`LayerwiseGradientUpstreamDiscovery`) |
|---|---|---|
| Expansion unit | Single BFS node | All nodes at a (layer, kind) pair |
| Predecessor scope | Direct causal predecessors only | ALL (layer', kind') with layer' < layer |
| Depth control | Integer hop counter | `max_layers_back` param (or unbounded by layer) |
| Work ordering | FIFO BFS queue | Priority queue, highest layer first (topological) |

---

## Phase 1 — Infrastructure

- [x] **Add `get_all_upstream_components(comp_idx, n_kinds, kinds, min_layer=0)` to `src/pipeline/component_index.py`.**
  - Returns a flat list of every `comp_idx` for all (layer', kind') with `min_layer ≤ layer' < layer` and all kind indices.
  - Different from `get_predecessor_components` which respects causal wiring within the same layer; this version returns *all* components in all strictly-preceding layers.
  - Signature:
    ```python
    def get_all_upstream_components(
        comp_idx: int, n_kinds: int, kinds: Sequence[str], min_layer: int = 0
    ) -> List[int]:
    ```

- [x] **Add `LayerwiseGradientUpstreamConfig` to `src/config.py`.**
  - Fields (all with sensible defaults):

    | Field | Type | Default | Notes |
    |---|---|---|---|
    | `top_k_per_node` | `int` | `8` | Top-K upstream latents per node per pass |
    | `attribution_threshold` | `float` | `0.01` | Min \|score\| to include a latent |
    | `min_active_count` | `int` | `1` | Skip latents below global firing count |
    | `max_ctx_sequences` | `int` | `4` | Ctx sequences per node (across microbatches) |
    | `hop_batch_size` | `int` | `4` | Sequences per microbatch in `_run_node` |
    | `absent_inhibitor_top_k` | `int` | `4` | 0 = disabled |
    | `absent_inhibitor_threshold` | `float` | `0.01` | |
    | `max_layers_back` | `int` | `0` | 0 = go back to layer 0; positive = limit depth |
    | `pruning_threshold` | `float` | `0.0` | |
    | `min_faithfulness` | `float` | `0.2` | |

  - Register it on `DiscoveryConfig` as `layerwise_gradient_upstream: LayerwiseGradientUpstreamConfig = Field(default_factory=LayerwiseGradientUpstreamConfig)`. ✓

---

## Phase 2 — Core Algorithm

- [x] **Create `src/circuit/discovery/layerwise_gradient_upstream.py`** with class `LayerwiseGradientUpstreamDiscovery(DiscoveryMethod)`.
  - `__init__`: mirrors `GradientUpstreamDiscovery.__init__` but reads from `config.discovery.layerwise_gradient_upstream` and exposes `max_layers_back` instead of `depth`.
  - `discover(seed_comp_idx, seed_latent_idx) -> Optional[Circuit]`: top-level entry, wraps `_discover` with `CircuitLogger`.
  - `_discover(...)`: main logic (see below).
  - `_run_node(comp_idx, latent_idx, tokens, all_upstream_comps, logger) -> UpstreamScores`: microbatch gradient accumulator, same structure as `GradientUpstreamDiscovery._run_hop` but receives `all_upstream_comps` instead of calling `get_predecessor_components`.

- [x] **Implement `_discover` layer-by-layer sweep:**
  1. Probe seed → add seed node to circuit.
  2. Push `(seed_layer, seed_kind)` onto a max-heap (negated layer as priority).
  3. Track `expanded: Set[Tuple[int, int]]` (per-node) and `fid_to_uuid` as before.
  4. Loop: pop `(layer, kind)` from heap → collect all unexpanded circuit nodes at this (layer, kind) → for each node run `_run_node` → add new nodes to circuit → enqueue their (layer', kind') if unseen and layer' ≥ `effective_min_layer`.
  5. `effective_min_layer`: `max(0, seed_layer - max_layers_back)` when `max_layers_back > 0`, else `0`.
  6. Within-layer tie-break: `-kind_idx` ordering so resid → mlp → attn within a layer.

- [x] **Handle absent inhibitors** in the layer sweep (same semantics as current: not enqueued, but added to circuit with `role=absent_inhibitor`).

- [x] **Add print logging** consistent with the `[GradUpstream]` format used in current method (use `[LayerwiseGradUpstream]` prefix).

---

## Phase 3 — Evaluation (reuse existing)

- [x] **Reuse evaluation block verbatim** from `GradientUpstreamDiscovery._discover` (minimality pruning, upstream faithfulness, faithfulness, sufficiency, completeness).
- [x] **Set `circuit.metadata["discovery_method"] = "layerwise_gradient_upstream"`** and include `max_layers_back` instead of `depth`.

---

## Phase 4 — Config YAML & Wiring

- [x] **Add `layerwise_gradient_upstream:` block to `config.yaml`** under `discovery:`, mirroring the `gradient_upstream:` block with the new fields. *(done in Phase 1)*
- [x] **Expose the class in `src/circuit/discovery/__init__.py`** (or whichever module exports discovery methods), if such a registry exists. *(no `__init__.py` exists; registered directly in `discovery_window.py`)*
- [x] **Check if the pipeline's `second_pass.py` or `discovery_window.py` needs a new enum/string entry** to select this method; added import + `METHOD_REGISTRY` entry + docstring line in `discovery_window.py`.

---

## Phase 5 — Tests

- [x] **Create `tests/circuit/discovery/test_layerwise_gradient_upstream.py`.**
  - Unit-test `get_all_upstream_components`: verify it returns correct comp indices for a given (layer, kind).
  - Smoke-test `_discover` with a tiny mock SAE bank and inference stub (similar to existing `test_attn_top_coact_sparse_expansion.py` pattern): ensure the circuit grows beyond the seed node and that the work-queue terminates.
  - Verify that nodes from non-adjacent layers can be discovered (key correctness property not covered by existing tests).
  - 27 tests, all passing.

---

## Open Questions

1. **Single vs. batched targets per (layer, kind) group.**  
   Currently the plan processes each node in a (layer, kind) group sequentially (separate backward passes). An alternative would be to concat their ctx tokens and run one joint pass. The joint pass is potentially faster but requires averaging pos_argmax across nodes — may dilute the attribution signal. Recommend separate passes for now; revisit if runtime is a bottleneck.

2. **Same-layer predecessors.**  
   `get_all_upstream_components` will include only strictly-preceding layers (layer' < layer). Should within-layer causal order (attn → mlp → resid at layer L) still be exploited as it is in the current method? Initial recommendation: include within-layer predecessors as well by calling `get_predecessor_components` for the same layer and unioning with `get_all_upstream_components`. This can be made optional via a config flag `include_same_layer_predecessors: bool = True`.

3. **Max layers back vs. no limit.**  
   With all layers available and a large model, the hop scope could be very wide. The `max_layers_back=0` (unbounded) default is intentional for the discovery phase but may need tuning in production runs. Consider surfacing as a prominent config knob.

4. **Interaction with `visited` set.**  
   The current algorithm uses `visited` to prevent re-expanding a node from different ancestors. In the new scheme, a (layer, kind) pair is only processed once, but a given `(comp_idx, latent_idx)` could be discovered by multiple ancestors. The plan allows multiple edges to the same node but only adds it once; this should be preserved.

5. **Config section naming.**  
   `layerwise_gradient_upstream` is verbose but unambiguous. Alternative: `grad_upstream_layerwise`. Either is fine — pick one during implementation.

## Risks / Assumptions

- `SAEGraphInstrument` already records activations for ALL layers in `graph.activations`, so passing a wider `predecessor_comp_indices` to `compute_latent_upstream_scores` should work without changes to the instrumentation layer.
- Memory and runtime will increase proportionally to the number of predecessor layers considered. For a 12-layer model with 3 kinds, a node at layer 6 will now score against 18 upstream components instead of 1–3.
- The `build_probe_dataset` call per node is the dominant latency driver; the extra backward scope adds relatively little.
- Evaluation code (`evaluate_faithfulness`, etc.) is already layer-agnostic and requires no changes.
