# Pairwise Latent Discovery Algorithm

## Purpose

Pairwise latent discovery is an intermediate stage between per-latent statistics and full circuit discovery.

The goal is to learn a sparse, directed evidence map over pairs of latents. Instead of immediately asking which latents form a circuit, the algorithm first asks a smaller question: which upstream latents appear to have meaningful evidence of interaction with which downstream latents?

This produces a pairwise atlas that can later be analyzed directly or used as an optional signal for circuit seed selection.

## Core Idea

The algorithm does not try to measure every possible latent pair. That would be infeasible because the model has roughly 1.47M latents, and the full directed pair space is on the order of trillions of possible pairs.

Instead, the algorithm works in stages:

- Propose a broad but sparse set of plausible directed pair candidates from cheap existing artifacts.
- Add random and matched control pairs so the measured evidence has a baseline.
- Select a smaller budgeted subset of pairs for deeper measurement.
- Measure selected pairs efficiently in target-centered bundles.
- Calibrate measured evidence against controls.
- Reduce everything into a sparse pairwise atlas.

The atlas remains class-free. It stores evidence, confidence, provenance, and calibration information, but it does not initially label relationships as activation, inhibition, gating, redundancy, or synergy.

## Inputs

Pairwise discovery reuses artifacts that already exist earlier in the pipeline:

- Latent statistics, such as activation counts and firing frequencies.
- Top context stores, which identify sequences where each latent is strongly active.
- Mid context stores, which provide less extreme but still relevant activation examples.
- Negative context stores, which provide contrastive examples.
- Top coactivation neighborhoods, which identify latents that often appear together.
- `seq_latent_index`, which records compact sequence-level latent membership.

The stage should not rerun first pass, negative context, or second pass unless explicitly requested by a standalone workflow.

## Directed Pair Schema

Every candidate pair is represented as a directed relationship:

```text
source -> target
```

The source must be structurally upstream of the target. This follows the computation graph of the model rather than treating direction as a learned label.

Each pair should be stored compactly:

- `source_global_id`: integer ID for the source latent.
- `target_global_id`: integer ID for the target latent.
- `proposal_mask`: bitset describing why the pair was proposed.
- `coactivation_score`: cheap score from top coactivation evidence.
- `seq_overlap_score`: cheap score from sequence-level overlap evidence.
- `stratum_id`: compact bucket used for balanced sampling and control matching.

Layer, kind, and local latent IDs can be derived from the global IDs when needed. They do not need to be repeated in every row of a large pair table.

## Proposal Mask

`proposal_mask` records the source or sources that caused a pair to enter the candidate table.

It is a compact bitset rather than a list of strings. For example:

- One bit can mean the pair came from top coactivation.
- One bit can mean the pair came from `seq_latent_index`.
- One bit can mean the pair was selected because the two cheap sources disagreed.
- One bit can mean the pair is a random control.
- One bit can mean the pair is a matched control.

This lets the atlas later compare populations such as coactivation-proposed pairs, sequence-proposed pairs, disagreement pairs, and controls.

## Strata

`stratum_id` describes the type of pair for sampling and comparison.

A stratum can encode properties such as:

- Source kind, such as `attn`, `mlp`, or `resid`.
- Target kind.
- Layer distance.
- Source firing-frequency bucket.
- Target firing-frequency bucket.

Strata prevent unfair comparisons. For example, a frequent nearby `mlp -> resid` pair should not be compared directly to a rare long-distance `attn -> mlp` pair without accounting for those differences.

## Candidate Proposal

Candidate proposal is the cheap first stage.

It reads existing artifacts and creates a sparse table of valid directed pairs. These pairs are not treated as confirmed interactions. They are only candidates worth considering.

The main candidate sources are:

- Top coactivation neighborhoods.
- `seq_latent_index` overlap.
- Disagreement between coactivation and sequence-level evidence.
- Random controls.
- Matched controls.

Top coactivation is useful because it finds latents that appear together. However, coactivation is not directional or causal by itself. The algorithm orients coactivation neighbors according to the model's structural ordering before storing them as directed candidates.

`seq_latent_index` is useful because it provides a cheap sequence-level retrieval axis. It may identify pairs that share sequence membership but do not appear as top coactivation neighbors.

## Controls

Controls are baseline pairs used to interpret measured evidence.

Random controls are valid directed pairs sampled from the broader valid pair space. They estimate the general background distribution.

Matched controls are valid directed pairs that match a proposed pair on important properties, usually through `stratum_id`. For example, if a proposed pair is an `mlp -> resid` pair across five layers with a medium-frequency source and rare target, a matched control should have similar properties.

Controls answer a key question:

```text
Is this measured pair actually unusual compared with similar valid pairs?
```

Without controls, a large raw gradient score may be misleading. It might simply reflect that all pairs of that kind or layer distance tend to score highly.

## Acquisition

Acquisition chooses which candidate pairs receive expensive measurement.

The acquisition policy should not simply take the highest coactivation scores. That would bias the atlas toward relationships already visible through coactivation.

Instead, it should balance several goals:

- Strong cheap signals.
- Disagreement between proposal sources.
- Coverage across layer distances.
- Coverage across kind pairs.
- Coverage across frequency buckets.
- Random controls.
- Matched controls.
- Uncertain or unusual candidates.

The acquisition policy must obey strict budgets:

- Maximum number of measured targets.
- Maximum number of measured pairs.
- Maximum number of sources per target.
- Maximum number of contexts per target.
- Maximum number of controls.
- Maximum table size or shard size.

## Target-Bundled Measurement

Measurement should be organized around targets, not individual pairs.

The inefficient pattern is:

```text
for each pair:
  run a separate forward/backward pass
```

The efficient pattern is:

```text
for each target:
  gather target contexts
  run one or a few grad-enabled passes
  score many upstream source latents for that target
```

This makes the cost scale closer to the number of measured targets and contexts, rather than the raw number of measured pairs.

For example, measuring 1,000 targets with 32 sources per target gives 32,000 measured pairs. If the implementation reuses the target pass across those 32 sources, the run is much more feasible than performing 32,000 separate backward passes.

## Gradient Measurement

The first serious measurement engine should be gradient-based.

For each selected target bundle, the algorithm:

- Chooses target contexts from top, mid, and possibly negative context stores.
- Runs a grad-enabled pass for the target.
- Scores the selected upstream source latents.
- Summarizes the evidence for each `source -> target` pair.

The measurement output should include fields such as:

- `gradient_score_mean`.
- `gradient_score_std`.
- `sign_consistency`.
- `context_count`.
- `observation_count`.
- `measurement_status`.

The output is pair evidence, not a circuit object. Existing circuit discovery methods should not change behavior unless a future config explicitly opts into pairwise-derived signals.

## Measurement Status

The atlas should preserve status information for every relevant pair.

Useful statuses include:

- `proposed_unmeasured`: the pair was proposed but not selected for measurement.
- `measured_supported`: the pair was measured and showed meaningful evidence.
- `measured_null`: the pair was measured but showed weak or null evidence.
- `skipped_missing_context`: the pair could not be measured because usable contexts were missing.
- `random_control`: the pair was included as a random baseline.
- `matched_control`: the pair was included as a matched baseline.

Measured-null pairs are important. They show where cheap proposal signals did not survive deeper measurement.

## Calibration

Calibration compares measured pairs against controls.

The algorithm should compute control-normalized scores when enough controls exist in a comparable stratum. These can include:

- Matched-control percentile.
- Empirical rank within the stratum.
- Z-score against comparable controls.
- Difference from matched-control mean.

Calibration makes the atlas more scientifically useful than raw scores alone. It helps distinguish pairs that are absolutely large from pairs that are unusual relative to their background.

## Confidence

Confidence should be separate from strength.

A pair can have a large measured effect but low confidence if the effect appears in too few contexts, changes sign, or is unstable across runs.

Confidence metrics can include:

- Number of contexts measured.
- Number of observations.
- Sign consistency.
- Run stability.
- Bootstrap uncertainty.
- Agreement between coactivation, sequence overlap, and gradient evidence.

This distinction matters because later analysis may want high-confidence modest effects, high-magnitude uncertain effects, or explicitly uncertain candidates for further intervention testing.

## Atlas Construction

The atlas is the final reduced artifact.

It joins:

- Candidate identity.
- Proposal provenance.
- Cheap evidence scores.
- Acquisition metadata.
- Measurement results.
- Control-normalized scores.
- Confidence metrics.
- Measurement status.

The atlas should be sparse, tensor-first, and shardable. Large runs should write shards plus a manifest rather than one huge Python object.

The atlas should support summary analysis by:

- Proposal source.
- Layer distance.
- Source kind.
- Target kind.
- Stratum.
- Measurement status.
- Control-normalized evidence.
- Confidence.

## Observability

Observability answers a run-health question:

```text
Did this pairwise discovery run behave as expected?
```

It should be produced during every normal run, whether or not a formal eval is being performed.

Good observability should include:

- Candidate counts by `proposal_mask`.
- Candidate and measured-pair counts by `stratum_id`.
- Selected target counts.
- Selected source counts per target.
- Measured, null, skipped, random-control, and matched-control counts.
- Missing-context counts and skipped-pair reasons.
- Control counts and matched-control ratios.
- Budget utilization for candidates, targets, pairs, sources per target, contexts per target, and controls.
- Phase timings for proposal, acquisition, measurement, calibration, atlas reduction, loading, and saving.
- Estimated peak CPU RAM and VRAM.
- Artifact sizes, shard counts, and rows per shard.
- Score min/max/mean/std.
- NaN and inf counts.
- Empty or under-covered strata.
- Calibration availability by stratum.

The output should include readable logs plus a compact machine-readable summary such as `run_health.json`.

Observability is not meant to prove that the atlas is scientifically good. It tells us whether the run was complete, numerically sane, within budget, and interpretable enough to evaluate.

## Algorithmic Evals

Algorithmic evals answer a quality question:

```text
Did the algorithm produce an accurate, useful, explainable, and sufficiently complete atlas?
```

These should be deliberate experiment scripts or notebooks that consume pairwise artifacts. They do not need to run during every normal pipeline execution.

### Control Separation Eval

This eval compares proposed measured pairs against matched controls within comparable `stratum_id` buckets.

Useful metrics include:

- Mean or median score difference versus matched controls.
- Fraction of proposed pairs above the 90th or 95th matched-control percentile.
- Robust effect size per stratum.
- Stratified breakdown by layer distance and kind pair.

This eval asks whether atlas-high pairs are genuinely unusual compared with similar valid pairs.

### Calibration Eval

This eval checks whether control-normalized scores behave sensibly.

Useful metrics include:

- Percentile distribution for random and matched controls.
- Z-score distribution for controls.
- False-positive rate among controls above the strong-pair threshold.
- Calibration curves per stratum.

Controls should mostly look like background. If controls frequently receive extreme atlas scores, the score calibration is not trustworthy.

### Stability And Replicability Eval

This eval reruns pairwise discovery with different random seeds, data shards, or context samples.

Useful metrics include:

- Top-K pair overlap.
- Score correlation for shared measured pairs.
- Sign consistency.
- Jaccard overlap of top neighborhoods per target.
- Stability of layer-distance and kind-pair summaries.

The expected result is not perfect pair-for-pair identity. The goal is that high-confidence pairs and high-level distributional patterns replicate.

### Held-Out Context Prediction Eval

This eval builds the atlas on one context or data split, then measures whether atlas scores predict evidence on held-out contexts.

Useful metrics include:

- Correlation between atlas score and held-out gradient score.
- Correlation between atlas score and held-out coactivation or sequence overlap.
- Held-out support rate for top atlas pairs versus controls.
- Score degradation from build contexts to held-out contexts.

This eval checks that the atlas is not merely memorizing stored contexts.

### Proposal Source Ablation Eval

This eval compares what each proposal source contributes.

The main populations are:

- Top-coactivation-only pairs.
- `seq_latent_index`-only pairs.
- Pairs proposed by both sources.
- Disagreement pairs.
- Random controls.
- Matched controls.

Useful metrics include measurement yield, fraction above matched-control threshold, average confidence, unique high-scoring pairs contributed, and overlap between proposal sources.

This eval tells us whether `seq_latent_index` and disagreement proposals add real value beyond top coactivation.

### Completeness And Coverage Eval

This eval asks whether the atlas covers enough of the intended space to support analysis.

Useful metrics include:

- Proposed and measured pairs per `stratum_id`.
- Coverage by source kind, target kind, layer distance, and frequency bucket.
- Control availability per stratum.
- Fraction of measured targets with enough upstream sources.
- Fraction of strata where calibration is possible.

Completeness does not mean exhaustive coverage. It means the measured sample is broad enough for the intended claims.

### Explainability And Structure Eval

This eval asks whether the atlas reveals understandable organization.

Useful analyses include:

- Layer-distance evidence profiles.
- Kind-pair evidence profiles.
- Incoming and outgoing interaction mass per latent.
- Neighborhood coherence around high-evidence targets.
- Graph or community structure compared with controls.
- Human-readable examples using associated contexts or tokens.

This eval supports interpretability and publication-quality analysis.

### Downstream Circuit Utility Eval

This eval asks whether the atlas helps the larger circuit discovery objective.

Compare baseline seed selection with explicit atlas-informed seed selection under equal compute budgets.

Useful metrics include:

- Circuit yield per seed.
- Mean or max faithfulness.
- Number of circuits above threshold.
- Nodes or edges needed to reach comparable faithfulness.
- Runtime or gradient passes per successful circuit.

This is the practical end-to-end eval: the atlas should either improve circuit discovery or explain why it is useful as an independent scientific artifact.

## Optional Intervention Measurement

Intervention measurement should be a later, separately budgeted validator.

It is more causally direct than gradient measurement, but it is also much more expensive. The first implementation should not depend on it.

A sensible future strategy is to apply interventions only to:

- The strongest gradient-measured pairs.
- High-uncertainty pairs that need clarification.
- Disagreement cases.
- Matched controls for the same strata.

## Optional Seed Selection Signals

Pairwise discovery should not change circuit seed selection by default.

Later, explicit opt-in seed criteria could use atlas-derived features such as:

- Incoming measured interaction mass.
- Upstream diversity.
- Strong directed neighborhoods.
- High-confidence pair evidence.
- High uncertainty.
- Unusual control-normalized evidence.

These should be additional criteria, not implicit replacements for the existing seed selection behavior.

## Expected Scientific Payoff

The algorithm creates a new empirical layer:

```text
latent statistics
directed pairwise latent interactions
multi-latent circuits
```

This makes it possible to study whether latent interactions are sparse, heavy-tailed, clustered, layer-specific, kind-specific, or strongly context-dependent before imposing circuit-level structure.

The most important hypotheses are:

- Coactivation-rich pairs and gradient-sensitive pairs will only partially overlap.
- Sequence-level disagreement cases may reveal conditional relationships missed by coactivation.
- Most valid pairs will be near-null, while a small number carry most interaction mass.
- Different kind pairs and layer distances will have different evidence distributions.
- Good circuit seeds may be identifiable from directed neighborhood structure, not just individual latent strength.

## Efficiency Principles

The implementation should follow a few hard rules:

- Never enumerate all valid latent pairs.
- Keep pair tables compact and tensor-first.
- Use `int32` global latent IDs where possible.
- Use named score columns rather than rich per-row objects.
- Enforce budgets before allocating large tensors.
- Prefer target-bundled gradient measurement.
- Keep controls useful but capped.
- Write large outputs as shards.
- Preserve disabled-by-default behavior.

These constraints are what make the algorithm feasible on real data.
