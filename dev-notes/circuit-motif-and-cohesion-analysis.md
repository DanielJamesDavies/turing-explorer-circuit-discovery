# Circuit Motif And Cohesion Analysis

## Motivation

The pipeline can now find latent-to-latent circuits. The next problem is
description: how to say what a circuit is, whether circuits share reusable
substructure, and how individual latents inside a circuit relate to each other.

The current analysis stack already gives useful starting evidence:

- `circuit-latent-commonality.json` analyzes `7,525` circuits and finds
  `798,590` unique non-seed latents.
- `233` non-seed latents appear in at least `15%` of analyzed circuits.
- The most common latents appear in around `60%` of circuits.
- Those highest-commonality latents are overwhelmingly
  `counterfactual_inhibitor` nodes, often in early layers.
- Direct seed coactivation overlap is low: mean `coact_overlap_pct` is around
  `3.57%`.
- Internode mutual coactivation density is also low: mean
  `internode_coact_density_pct` is around `0.26%`.
- Existing correlations between coactivation overlap/density and
  counterfactual faithfulness are near zero.
- Exact multi-hop recovery from the seed through the PMI coactivation graph is
  also very small for most circuits.

Interpretation: exact latent reuse is real, but raw reuse appears dominated by
generic recurring inhibitor or hub-like latents. Faithful causal circuits are
not simply local coactivation neighborhoods. The next analysis should move from
"circuits are bags of latents" to "circuits are signed, typed causal graphs."

## Question 1: Do Circuits Contain Shared Mini Circuits?

The answer should not be based on individual latent commonality alone. A shared
mini circuit should mean a small causal subgraph that recurs more than expected
and carries transferable explanatory power.

Represent each circuit as a graph:

```text
node = FeatureID + role + layer + kind
edge = source -> target + attribution weight + sign/type
```

Then search for recurrent subgraphs at several abstraction levels.

### Motif Levels

1. Exact latent motifs
   - Same latent IDs.
   - Same roles.
   - Same directed edges.
   - Useful for literal reused machinery.

2. Typed motifs
   - Same layer/kind/role pattern.
   - Latent IDs may differ.
   - Useful for finding repeated structural shapes, such as
     early-inhibitor-to-seed or activator-chain patterns.

3. Hub-corrected motifs
   - Downweight latents that appear in many circuits.
   - Especially important for high-frequency inhibitors.
   - A motif is interesting only if it has lift over a degree-, layer-, and
     role-preserving null model.

4. Causally validated motifs
   - Motif-only sufficiency should preserve some seed activation or target
     behavior.
   - Motif removal should reduce faithfulness, suppression, or seed activation.
   - Frequent but causally inert motifs should be treated as scaffolding or
     noise.

### Proposed Analysis

Start with small motifs before general subgraph mining:

```text
size-2 motif: A -> seed
size-3 motif: A -> B -> seed
size-3 fan-in: A -> seed <- B
size-3 signed pair: activator + inhibitor around same seed
```

For each motif candidate, record:

- exact node keys: `layer`, `kind`, `latent_idx`, `role`;
- abstract node keys: `layer`, `kind`, `role`;
- edge direction and edge weight summary;
- number and percentage of circuits containing it;
- support by discovery method;
- mean and median faithfulness of containing circuits;
- lift over shuffled null circuits preserving role/layer/kind counts;
- whether the motif is enriched in high-faithfulness circuits;
- motif-only sufficiency score, where feasible;
- motif-removal score drop, where feasible.

The first pass can mine exact 2-node and 3-node motifs from
`discovered_circuits.pt`. If this produces useful candidates, extend to typed
motifs and larger frequent subgraphs.

### Circuit Family Graph

Once motifs are identified, the most natural representation is a weighted
circuit-motif incidence structure, or hypergraph:

```text
node = circuit
hyperedge = validated motif M connecting all circuits that contain M
membership_weight = motif coverage * motif lift * causal score
```

This preserves the fact that a motif can connect many circuits at once. The
pairwise circuit-to-circuit graph is then a projection of this motif hypergraph:

```text
circuit_i -- circuit_j if they share validated motif M
edge_weight = sum(shared motif lift * causal score)
```

Start with hard clustering on the projected graph because it is simple to
inspect. Then add fuzzy clustering on the circuit-motif matrix so circuits can
belong to multiple families with different strengths. This is important because
large circuits may combine generic scaffolding with seed-specific machinery.

Useful fuzzy-family options include:

- non-negative matrix factorization over the circuit-motif matrix, where each
  component is an interpretable family of motifs;
- fuzzy c-means over motif-membership vectors;
- mixed-membership graph models on the projected circuit graph, if a heavier
  probabilistic model becomes useful.

A circuit description can then include:

- its seed latent;
- its dominant roles;
- its strongest motif memberships;
- its hard family/community, if assigned;
- its fuzzy family membership weights;
- which motifs are generic scaffolds versus seed-specific machinery.

This should make it possible to describe a circuit as:

```text
This circuit belongs to family A. It contains a common early-layer inhibitor
scaffold plus a rarer seed-specific activator chain. Its causal weight is
concentrated in a few edges, while its coactivation cohesion is low.
Fuzzy membership: 0.65 early-inhibitor scaffold, 0.25 seed-specific activator
chain, 0.10 mixed signed-control family.
```

## Question 2: How Are Latents Related Inside A Circuit?

Separate coupling from cohesion.

Coupling is about pairwise or directional relationships between latents.
Cohesion is about whether the circuit behaves like a single internally coherent
unit.

### Coupling Metrics

For each pair or edge inside a circuit, measure:

1. Causal edge coupling
   - Use `CircuitEdge.metadata["weight"]`.
   - Track signed and absolute weights.
   - Separate activator, inhibitor, and mixed-role edges.

2. Coactivation coupling
   - Use PMI values from `top_coactivation.pt`.
   - Record directed PMI `A -> B`, directed PMI `B -> A`, and reciprocal
     strength `min(PMI(A -> B), PMI(B -> A))`.
   - This distinguishes causal relationships from natural co-firing.

3. Context coupling
   - Compare firing rates across positive, mid, negative, and top contexts.
   - Record activation correlation or binary co-presence across stored context
     sequences.
   - This is especially useful for deciding whether two latents describe the
     same context pattern or complementary context patterns.

4. Perturbation coupling
   - Ablate or clamp node `A`.
   - Measure the change in node `B`, the seed latent, and the target logit.
   - This is the strongest evidence, but should be run on a smaller selected
     subset because it is expensive.

5. Role compatibility
   - Activator-activator coupling should often show positive co-presence or
     positive causal support.
   - Inhibitor-inhibitor coupling may show common negative-context behavior.
   - Activator-inhibitor coupling may be anti-correlated or context-separated.

### Cohesion Metrics

For each circuit, compute a cohesion profile:

- `internode_coact_density_pct`: current mutual coactivation density.
- `edge_weight_gini`: how concentrated causal weight is across edges.
- node presence on positive contexts:
  `node_presence_pct_activators` and `node_presence_rate_mean`.
- inhibitor absence on positive contexts:
  `node_absence_pct_inhibitors` and `node_inhibitor_rate_mean`.
- circuit-only sufficiency:
  `posctx_circuit_sufficiency`.
- motif density: fraction of nodes/edges covered by recurring validated motifs.
- causal modularity: whether the circuit splits into internally dense modules.
- layer span and layer order: whether the circuit forms a plausible directed
  path or a flat set of direct contributors.
- role purity: whether nodes behave consistently with their assigned role.

The current results suggest that many accepted circuits are causally faithful
but not coactively cohesive. That is not necessarily bad. It may mean the
circuits are distributed causal control sets rather than simple co-firing
communities. Coactivation density should therefore be one descriptive axis, not
the acceptance criterion.

## Proposed New Analysis Pass

Add a new analysis suite, conceptually:

```text
circuit_motif_analysis
```

Inputs:

```text
circuits/discovered_circuits.pt
circuits/summary.json
top_coactivation.pt
top_ctx.pt
mid_ctx.pt
neg_ctx.pt
logit_ctx.pt
```

Outputs:

```text
analysis/<N>/circuit-motifs/tables/motifs.csv
analysis/<N>/circuit-motifs/tables/circuit-motif-membership.csv
analysis/<N>/circuit-motifs/tables/circuit-cohesion.csv
analysis/<N>/circuit-motifs/summaries/circuit-motif-analysis.json
analysis/<N>/circuit-motifs/figures/
```

`motifs.csv` should include:

```text
motif_id, motif_size, motif_kind, exact_signature, abstract_signature,
support_count, support_pct, support_lift, mean_faithfulness,
median_faithfulness, high_faithfulness_enrichment,
mean_edge_weight, motif_only_sufficiency, motif_removal_drop
```

`circuit-motif-membership.csv` should include:

```text
uuid, name, seed_comp, seed_latent, motif_id, motif_role,
motif_node_count, motif_edge_count, motif_coverage_pct
```

`circuit-cohesion.csv` should include:

```text
uuid, name, seed_comp, seed_latent, nodes, edges,
counterfactual_faithfulness, posctx_suppression_score,
internode_coact_density_pct, edge_weight_gini,
node_presence_pct_activators, node_presence_rate_mean,
node_absence_pct_inhibitors, node_inhibitor_rate_mean,
posctx_circuit_sufficiency, motif_coverage_pct,
role_purity_score, causal_modularity, layer_span
```

## Implementation Order

1. Load full circuit objects from `discovered_circuits.pt`.

2. Normalize each circuit into a graph with stable node keys:

```text
exact_key    = (layer, kind, latent_idx, role)
abstract_key = (layer, kind, role)
```

3. Mine exact 2-node and 3-node directed motifs.

4. Compute frequency and hub-corrected lift.

5. Join each motif with existing circuit metrics from `summary.json`.

6. Add per-circuit cohesion rows using existing post-analysis metrics.

7. Select top motifs for expensive causal validation.

8. Run motif-only and motif-removal evaluations on a calibration subset.

9. Build the weighted circuit-motif incidence matrix / motif hypergraph.

10. Project the hypergraph into a circuit-to-circuit similarity graph.

11. Cluster circuits by validated motif memberships, starting with hard graph
    communities and then adding fuzzy family memberships.

12. Use the resulting families to write natural-language circuit descriptions.

## Main Caveats

- Shared individual latents are not enough. High-frequency inhibitors can make
  many unrelated circuits look similar.
- Coactivation is not causality. Low coactivation density does not imply a bad
  circuit if perturbation evidence is strong.
- Faithfulness can exceed `1.0`, so analyses should treat it as a score, not a
  bounded probability.
- Large circuits may contain both generic scaffolding and seed-specific
  machinery. Motif descriptions should separate those two.
- Expensive perturbation validation should be reserved for top candidate motifs,
  not all mined motifs.

## Desired End State

Each circuit should get a compact description like:

```text
Circuit CounterfactualGrad_S5_18145

Family: early-inhibitor scaffold with seed-specific activator support.
Shared motifs: M12, M31.
Cohesion: low coactivation, high causal concentration.
Role behavior: activators partially present on posctx; inhibitors mostly absent.
Interpretation: likely a distributed suppression-gated control circuit rather
than a single co-firing semantic cluster.
```

This would make circuits connectable at three levels:

- shared exact latents;
- shared signed motifs;
- shared higher-level circuit families.
