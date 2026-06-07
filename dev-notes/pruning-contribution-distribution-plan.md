# Pruning Contribution Distribution Plan

## Motivation

The first `cf_pruned` pilot showed that aggressive leave-one-out pruning can
shrink a circuit dramatically while changing evaluation behavior too much. One
example went from `34` nodes to `4` nodes:

- `counterfactual_faithfulness`: `0.7672` -> `0.6349`
- `posctx_suppression_score`: `1.0000` -> `0.5476`

That is useful evidence, but not a safe default. Before picking pruning
thresholds, we should inspect the distribution of each latent's contribution to
the scores and look for natural elbows, tails, or role-specific patterns.

## Goal

Build an analysis that answers:

1. Which latents contribute materially to `counterfactual_faithfulness`?
2. Which latents contribute materially to `posctx_suppression_score`?
3. Are there latents that look weak for one score but important for the other?
4. Is there a threshold, percentile, or elbow that removes many nodes while
   keeping score degradation bounded?

The output should be graphs and summary tables that guide pruning thresholds,
not a new pruning policy by itself.

## Proposed Analysis

For each selected circuit, compute leave-one-out contribution scores for every
non-seed node:

```text
cf_drop(node)  = base_counterfactual_faithfulness - cf_without_node
sup_drop(node) = base_posctx_suppression_score    - sup_without_node
```

Interpretation:

- Large positive drop: node is important for that score.
- Near-zero drop: node is probably removable for that score.
- Negative drop: removing the node improves the score, so it may be harmful or
  redundant.

For each node, also record:

- `feature_id`
- layer, kind, latent index
- current role: `counterfactual_activator`, `counterfactual_inhibitor`,
  `ablation_support`, or merged hybrid roles
- source methods: `counterfactual_gradient`, `ablation_gradient`
- stored attribution score
- edge degree
- whether the node is shared by both source methods

## Graphs

Generate one set of plots per pruning pilot subset:

1. **CF Drop Histogram**
   - X-axis: `cf_drop`
   - Y-axis: number of nodes
   - Add vertical guide lines for candidate thresholds, e.g. `0.0`, `0.01`,
     `0.025`, `0.05`, `0.1`.

2. **Suppression Drop Histogram**
   - Same as above, but for `sup_drop`.

3. **CF vs Suppression Scatter**
   - X-axis: `cf_drop`
   - Y-axis: `sup_drop`
   - Color by node role or source method.
   - This is the most important plot for spotting nodes that are weak for CF
     but strong for suppression, which the current `cf_pruned` objective can
     accidentally remove.

4. **Sorted Drop Curves**
   - Sort nodes by increasing `cf_drop`, `sup_drop`, and `max(cf_drop, sup_drop)`.
   - Plot cumulative count removed against the estimated score-risk threshold.
   - This helps find an elbow where many low-impact nodes can be removed before
     the curve steepens.

5. **Circuit Size vs Score Tradeoff**
   - Simulate candidate threshold policies without mutating the circuit:
     - keep nodes where `cf_drop >= threshold`
     - keep nodes where `sup_drop >= threshold`
     - keep nodes where `max(cf_drop, sup_drop) >= threshold`
     - keep nodes where `cf_drop >= threshold OR sup_drop >= threshold`
   - For each policy, report retained node count and estimated worst removed
     contribution.

## Recommended Pruning Rule To Evaluate

The immediate safer rule should be score-preserving across both metrics:

```text
remove node only if cf_drop < threshold AND sup_drop < threshold
```

This is stricter than the current `cf_pruned` run. It should prevent deleting
nodes that matter to suppression even if they appear weak for CF.

For threshold selection, start with a low sweep:

```text
thresholds = [0.0, 0.005, 0.01, 0.025, 0.05]
```

Then choose from the observed distributions. A reasonable default candidate is
the largest threshold before the sorted `max(cf_drop, sup_drop)` curve shows a
clear jump.

## Implementation Plan

1. Add an analysis helper that accepts circuits and their eval context, then
   computes per-node leave-one-out contribution rows without modifying the
   saved circuits.

2. Save rows to:

```text
analysis/<N>/pruning_contributions/node_contributions.csv
```

Each row should include:

```text
variant, candidate_index, layer, kind, latent_idx, node_uuid,
role, source_methods, attribution_score, degree,
base_cf, base_sup, cf_without_node, sup_without_node,
cf_drop, sup_drop
```

3. Save aggregate summaries to:

```text
analysis/<N>/pruning_contributions/summary.json
```

Include percentiles for `cf_drop`, `sup_drop`, and
`max(cf_drop, sup_drop)`.

4. Generate plots to:

```text
analysis/<N>/pruning_contributions/plots/
```

5. Run this first on the same small `hybrid_gradient / random` pruning pilot
   subset, not the full 128-seed grid.

6. Use the resulting plots to pick SFC-style and leave-one-out pruning
   thresholds for the next pilot.

## How This Fits With SFC-Style Pruning

SFC-style pruning is cheap because it thresholds already-computed node and edge
attribution scores. The contribution-distribution analysis is more expensive,
but it only needs to run on a small calibration subset. We can use it to choose
SFC-style thresholds:

- If low `attribution_score` usually means low `cf_drop` and low `sup_drop`,
  SFC-style threshold pruning is a good fast approximation.
- If attribution scores do not predict contribution drops well, use a more
  conservative SFC threshold or keep leave-one-out pruning for final small
  circuits only.

## Success Criteria

This analysis is successful if it gives us:

- A visible distribution of node importance for both scores.
- A clear estimate of how much score degradation each threshold risks.
- Evidence for whether `cf`, `suppression`, or `both` pruning is safest.
- A concrete threshold recommendation for the next `hybrid_gradient / random`
  pruning pilot.
