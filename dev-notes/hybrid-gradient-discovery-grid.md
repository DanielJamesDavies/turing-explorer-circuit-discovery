# Hybrid Gradient Discovery And Neg-Mode Grid

This note captures the next discovery direction after the balanced 128-seed
comparison between `counterfactual_gradient` and `ablation_gradient`.

## 1. Hybrid Discovery Shape

The hybrid method should run both existing algorithms in their normal form and
then fuse their resulting circuits, rather than assigning fixed roles to each
algorithm up front.

Proposed flow for one seed:

1. Run `counterfactual_gradient` as usual.
2. Run `ablation_gradient` as usual.
3. If both return `None`, reject the seed.
4. If one returns a circuit, keep that circuit as the hybrid candidate.
5. If both return circuits, merge their node and edge sets into one candidate
   circuit.
6. Re-evaluate the merged candidate with the shared counterfactual evaluator.
7. Optionally prune the merged candidate, controlled by config.

The important point is that the hybrid should be a circuit-level fusion:

- Preserve `counterfactual_gradient` nodes with their existing roles:
  `counterfactual_activator` and `counterfactual_inhibitor`.
- Preserve `ablation_gradient` nodes with `ablation_support`.
- Preserve attribution scores and source-method metadata where possible.
- Deduplicate by `FeatureID`, not by node UUID.
- Merge duplicate edges by either max weight or by storing per-source weights.

The hypothesis is that `counterfactual_gradient` contributes better
negative-context activation evidence, while `ablation_gradient` contributes
strong positive-context necessity evidence. Fusing the outputs lets the final
evaluation decide whether the union explains both directions.

## 2. Pruning Toggle

Pruning already exists, but the current config disables it for the relevant
methods:

```yaml
counterfactual_gradient:
  pruning_threshold: 0

ablation_gradient:
  pruning_threshold: 0.0
```

Existing pruning objectives:

- `prune_non_minimal_nodes_cf`: leave-one-out pruning against
  `counterfactual_faithfulness`.
- `prune_non_minimal_nodes_suppression`: leave-one-out pruning against
  `posctx_suppression_score`.

For the hybrid method, pruning should be explicitly toggleable so we can compare
both sides:

```yaml
hybrid_gradient:
  pruning_enabled: false
  pruning_threshold: 0.0
  pruning_objective: "both" # "cf" | "suppression" | "both"
```

Recommended comparison modes:

- `unpruned`: raw fused circuit, best for measuring maximum recoverable signal.
- `cf_pruned`: remove nodes that do not help `counterfactual_faithfulness`.
- `suppression_pruned`: remove nodes that do not help positive-context
  suppression.
- `both_pruned`: remove a node only if it has negligible impact on both scores.

The first implementation can start with `pruning_enabled: false`, matching the
current behavior, then add the pruning variants for the experiment grid.

## 3. 3x3 Algorithm And Neg-Mode Experiment

Run all three discovery algorithms across all three negative-context modes:

- Algorithms:
  - `counterfactual_gradient`
  - `ablation_gradient`
  - `hybrid_gradient`
- Neg modes:
  - `close`: stored hard negatives from `neg_ctx`
  - `random`: uniformly random token sequences
  - `distant`: corpus sequences far from posctx in SAE latent space

This gives a `3 x 3` grid:

```text
counterfactual_gradient × close
counterfactual_gradient × random
counterfactual_gradient × distant
ablation_gradient       × close
ablation_gradient       × random
ablation_gradient       × distant
hybrid_gradient         × close
hybrid_gradient         × random
hybrid_gradient         × distant
```

One implementation detail: `counterfactual_gradient` already has `neg_mode`.
`ablation_gradient` currently uses stored negative tokens for evaluation and
falls back to random tokens only when stored negatives are missing. To make the
grid fair, negative-token selection should be shared across all three methods.

Recommended outputs for the analysis graph:

- Acceptance rate by algorithm and neg mode.
- Mean and median `counterfactual_faithfulness`.
- Mean and median `posctx_suppression_score`.
- Node and edge counts.
- Runtime and peak VRAM.
- Scatter plot of `counterfactual_faithfulness` vs
  `posctx_suppression_score`, colored by algorithm and faceted by neg mode.
- Optional size or alpha encoding by node count, to show whether improvements
  come from larger fused circuits.

Recommended first run:

- Balanced 128-seed sample using round-robin `comp_idx` selection.
- Run unpruned first.
- Then rerun hybrid with pruning enabled after the raw fused baseline is known.

The main question for the grid is not just which method accepts the most
circuits, but which method produces the best tradeoff between:

- high `counterfactual_faithfulness`,
- high `posctx_suppression_score`,
- reasonable circuit size,
- stable behavior across negative-context modes.
