# Compact Mechanism Target

The `20260531-152059-37117a33` full run produced many accepted
`counterfactual_gradient` circuits, but they look more like large
counterfactual intervention sets than clean, compact mechanisms.

Current shape to improve on:

- Hundreds of nodes per accepted circuit, often around `400`.
- Mostly direct node-to-seed edges rather than sparse multi-step paths.
- Strong counterfactual leverage, but frequent overshoot where
  `counterfactual_faithfulness > 1.0`.
- Low activator-only sufficiency on positive contexts.
- Low coactivation overlap, especially for activators.
- Mixed role purity: activators are not consistently present on positive
  contexts, and inhibitors are often not cleanly absent there.

A better target circuit should look like a small directed mechanism:

```text
token / local-pattern features
        ↓
early attention or MLP features
        ↓
mid-layer abstract feature
        ↓
late residual / MLP feature
        ↓
seed feature
```

Desired properties:

- Tens of nodes, not hundreds.
- Layered causal paths rather than a flat bag of contributors.
- Sparse internal edges with each node having a few meaningful parents or
  children.
- High internal coactivation among circuit nodes and with the seed.
- High sufficiency: keeping only circuit activators should drive the seed on
  positive contexts.
- High necessity or completeness: removing the circuit should meaningfully
  suppress the seed or target behavior.
- Consistent roles: activators should appear on positive contexts; inhibitors
  should be mostly absent on positive contexts and present in suppressive or
  negative contexts.
- Stable semantics across top activating examples and token contexts.

Algorithm direction: keep `counterfactual_gradient` as evidence of causal
leverage, but add stronger compactness, sufficiency, role-consistency, and
internal-structure pressure before treating a discovered set as a mechanistic
circuit.
