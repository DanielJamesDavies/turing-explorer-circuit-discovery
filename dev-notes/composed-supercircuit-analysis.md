# Composed Supercircuit Analysis

## Motivation

The current full-run `discovered_circuits.pt` contains `7,525` accepted
circuits, but the saved graph topology is almost entirely seed-centered:

```text
non-seed latent -> seed latent
```

In the checked run, every non-seed latent had shortest directed distance `1`
from the seed, and there were no latent-to-latent edges. This means the current
gradient circuits are useful as local one-hop attribution neighborhoods, but
they do not directly expose longer causal chains.

However, the run contains many circuits whose seeds are themselves latents. This
suggests a compositional analysis:

```text
If circuit(seed = S) contains important latent A,
and circuit(seed = A) exists,
then splice circuit(A) upstream of A.
```

This turns one-hop local circuits into larger composed supercircuits.

## Core Idea

Treat each existing circuit as a reusable local explanation for its seed latent.
Then recursively compose circuits by matching parent circuit nodes to child
circuit seeds.

Example:

```text
Parent circuit:
  A -> S

Child circuit where seed = A:
  B -> A
  C -> A

Composed supercircuit:
  B -> A -> S
  C -> A -> S
```

The composed graph is not a newly discovered causal graph from scratch. It is a
multi-hop approximation built by chaining validated or accepted one-hop local
circuits.

## Starting Point

Start with a small calibration run:

```text
root_seed_count = 64
root selection = circuits whose seed is in the latest layer/kind available
max_depth = 3
top_k_children_per_node = 8
max_total_nodes_per_supercircuit = 2000
```

Starting from late seeds is important because current edges point from upstream
latents into a seed. Late-layer or late-component seeds have the most room for
recursive upstream expansion.

For the usual kind order:

```text
kinds = ("attn", "mlp", "resid")
seed_comp = layer * len(kinds) + kind_idx
```

The first pass should choose root circuits from the largest available
`seed_comp`, then walk backward through matched child circuits.

## Seed Index

Build an index from seed latent identity to circuit:

```text
seed_index[(layer, kind, latent_idx)] = circuit
```

If multiple circuits share the same seed identity, choose one by a stable
priority rule:

1. higher `counterfactual_faithfulness`;
2. higher `posctx_suppression_score`;
3. smaller circuit, to reduce expansion blow-up;
4. stable UUID/name tie-break.

Later, this can become a multi-entry index so competing child explanations can
be compared rather than discarded.

## Expansion Algorithm

For each root circuit:

1. Add the root circuit's seed and selected direct contributors.
2. Rank non-seed nodes by importance.
3. For each important node, look up whether it has a child circuit where that
   node is the seed.
4. If a child circuit exists, merge it into the supercircuit and connect its
   contributors to the matched node.
5. Repeat until `max_depth`, node cap, or no more child circuits.

Pseudo-code:

```text
expand(parent_circuit, parent_seed_key, depth):
    add parent_circuit nodes and edges to supercircuit

    if depth == max_depth:
        return

    important_nodes = top_ranked_non_seed_nodes(parent_circuit)

    for node in important_nodes:
        child = seed_index.get(node.feature_key)
        if child is missing:
            record missing_child
            continue

        merge child into supercircuit
        expand(child, node.feature_key, depth + 1)
```

## Node Importance

Start with edge-local importance:

```text
importance(node) = abs(edge_weight(node -> parent_seed))
```

Then add role-aware and hub-aware variants:

```text
role-aware importance:
  keep top activators and top inhibitors separately

hub-corrected importance:
  importance / log(2 + global_latent_support_count)

composed path weight:
  parent_path_weight * abs(child_edge_weight)
```

The first implementation should keep the scoring simple and deterministic, then
write enough columns to compare ranking strategies.

## Graph Merge Rules

Deduplicate nodes by feature identity, not UUID:

```text
node_key = (layer, kind, latent_idx)
```

Deduplicate edges by feature identity:

```text
edge_key = (source_node_key, target_node_key)
```

When the same node or edge is introduced by multiple child circuits, merge
metadata rather than duplicating graph structure:

```text
source_circuit_uuids = set(...)
min_depth = min(existing_depth, new_depth)
max_abs_weight = max(existing_abs_weight, new_abs_weight)
path_count += 1
roles = union(...)
```

The composed graph should preserve provenance so every node and edge can be
traced back to the circuit that introduced it.

## Metrics

For each supercircuit, compute:

- total nodes and edges;
- number of child circuits expanded;
- number of attempted child expansions;
- child expansion hit rate;
- node counts by depth;
- edge counts by depth;
- max composed depth;
- root faithfulness;
- mean and median child faithfulness;
- fraction of nodes introduced by hubs;
- fraction of nodes with multiple parent paths;
- path-weight concentration;
- role composition by depth.

The most important first question is:

```text
Do composed circuits actually form meaningful depth > 1 structures,
or do most branches terminate immediately because child seed circuits are absent?
```

## Outputs

Add a new analysis suite conceptually named:

```text
circuit_supercircuit_analysis
```

Suggested output directory:

```text
analysis/<N>/circuit-supercircuits/
```

Tables:

```text
tables/supercircuits.csv
tables/supercircuit-node-depths.csv
tables/supercircuit-edges.csv
tables/supercircuit-expansion-events.csv
tables/supercircuit-depth-summary.csv
```

`supercircuits.csv`:

```text
root_uuid, root_name, root_seed_comp, root_seed_latent,
root_seed_layer, root_seed_kind, root_seed_feature,
root_faithfulness, nodes, edges, max_depth,
depth1_nodes, depth2_nodes, depth3_nodes,
expanded_child_circuit_count, attempted_child_count,
missing_child_count, child_hit_rate,
mean_child_faithfulness, median_child_faithfulness,
hub_node_pct, multi_parent_node_pct
```

`supercircuit-node-depths.csv`:

```text
root_uuid, node_key, layer, kind, latent_idx, role,
depth, min_depth, source_circuit_uuid, source_circuit_name,
parent_node_key, incoming_edge_weight, path_weight,
global_seed_available, global_latent_support_count
```

`supercircuit-edges.csv`:

```text
root_uuid, source_node_key, target_node_key,
source_layer, source_kind, source_latent,
target_layer, target_kind, target_latent,
depth, weight, abs_weight, path_weight,
source_circuit_uuid, source_circuit_name
```

`supercircuit-expansion-events.csv`:

```text
root_uuid, depth, parent_circuit_uuid, parent_seed_key,
candidate_node_key, candidate_role, candidate_importance,
child_found, child_circuit_uuid, child_faithfulness,
skipped_reason
```

Figures:

- node depth distribution;
- max depth per root circuit;
- supercircuit size distribution;
- expansion hit rate by depth;
- expansion hit rate by layer/kind;
- root faithfulness vs max depth;
- root faithfulness vs composed node count.

## Caveats

Composed supercircuits are not automatically causally validated chains.

A child circuit for latent `A` explains `A` as a seed under its own positive and
negative contexts. When spliced into a parent circuit, it approximates what
supports `A`, but it may not be the exact mechanism active in the parent seed's
contexts.

Therefore the first-pass output should be described as:

```text
composed local explanations
```

not as fully validated end-to-end causal pathways.

Important risks:

- Expansion can explode through common hub latents.
- A high-faithfulness child circuit may be context-mismatched for the parent.
- The same latent may appear with different roles in different circuits.
- Path weights multiply quickly and may become hard to interpret.
- Missing child circuits may reflect seed selection coverage, not absence of a
  true upstream mechanism.

## Validation Strategy

After the first static composition pass:

1. Select a small set of composed supercircuits with depth `>= 2`.
2. Compare full composed supercircuit interventions against the original
   one-hop parent circuit.
3. Ablate or remove depth-2 and depth-3 nodes and measure score drops.
4. Test whether composed paths preserve seed activation better than direct
   one-hop nodes alone.
5. Check whether high path-weight branches are more causally important than
   low path-weight branches.

The validation pass should be separate and expensive, similar to the planned
motif-only and motif-removal validation.

## Implementation Order

1. Build a seed index from full `discovered_circuits.pt`.

2. Select `64` root circuits from the latest available `seed_comp`.

3. Implement feature-key-based graph merge:

```text
node_key = (layer, kind, latent_idx)
edge_key = (source_key, target_key)
```

4. Expand each root circuit to `max_depth = 3` using top weighted contributors.

5. Write node-depth, edge, expansion-event, and supercircuit summary tables.

6. Plot depth and expansion distributions.

7. Inspect whether depth `>= 2` structures are common enough to justify a larger
   run.

8. Add hub correction and role-aware child selection.

9. Add causal validation on a small selected subset.

## Desired End State

For a late-layer seed, produce a description like:

```text
Root circuit CounterfactualGrad_S31_9528 has 412 direct contributors.
Composed expansion found child circuits for 37 of the top 64 contributors.
The resulting supercircuit has 1,284 nodes, 1,291 edges, and max depth 3.
Most depth-2 nodes come from early attention activator circuits.
Several depth-2 branches route through recurring inhibitor latents, suggesting
generic suppression scaffolding rather than seed-specific machinery.
```

This would complement motif analysis:

- motif analysis finds reusable local shapes;
- supercircuit analysis composes local circuits into approximate multi-hop
  mechanisms;
- causal validation then tests whether those composed paths matter end-to-end.
