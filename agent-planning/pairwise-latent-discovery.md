# Plan: Pairwise Latent Discovery

> **Goal:** Add an opt-in `pairwise_discovery` stage that learns sparse directed latent-pair evidence between existing coactivation/statistics and circuit discovery, without changing current passes or circuit discovery behavior by default.
>
> **Created:** 2026-04-25

---

## Phase 1 — Contracts And Configuration

- [ ] Add a disabled-by-default `pairwise_discovery` config section with budgets for candidate proposal, acquisition, measurement, controls, memory/storage caps, outputs, and standalone execution.
- [ ] Define the pipeline contract as `first_pass -> negative_context -> second_pass -> pairwise_discovery -> circuit_seed_selection -> circuit_discovery`.
- [ ] Preserve the existing path when `pairwise_discovery.enabled` is false so current outputs and discovery behavior remain unchanged.
- [ ] Add central validation rules for directed pairs: no same `layer+kind`, no downstream-to-upstream direction, no same-component pair, and source must be structurally upstream of target.
- [ ] Add budget-first config fields such as `candidate_budget`, `measured_pair_budget`, `measured_target_budget`, `sources_per_target`, `contexts_per_target`, `matched_controls_per_pair`, `random_control_budget`, `max_candidate_table_bytes`, `measurement_dtype`, and `store_feature_dtype`.
- [ ] Decide whether the code-facing name remains `candidate_selection` initially while documentation refers to it as `circuit_seed_selection`.
- [ ] Verify by running existing pipeline unit tests that should not depend on pairwise discovery.

## Phase 2 — Pair Schema And Artifacts

- [ ] Create canonical data structures for directed pair identity: source component, source latent, target component, target latent, derived layer/kind metadata, and stable pair keys.
- [ ] Add helper functions for component ordering, flattening/unflattening, sorting, deduping, and validity checks.
- [ ] Define tensor-first artifact formats under `outputs/pairwise_discovery/` for `pair_candidates.pt`, `measurements.pt`, `atlas.pt`, and `summary.json`.
- [ ] Store pair identity compactly using `source_global_id` and `target_global_id` as `int32` values, deriving layer/kind/local latent metadata only when needed.
- [ ] Store candidate provenance using an explicit `proposal_mask` bitset, with bits for sources such as top coactivation, `seq_latent_index`, disagreement proposal, random control, and matched control.
- [ ] Name cheap proposal scores explicitly, starting with `coactivation_score` and `seq_overlap_score`, and leave room for additional named features such as `target_context_overlap`, `rank_agreement`, or `frequency_adjusted_score`.
- [ ] Store `stratum_id` for balanced acquisition and control matching, encoding compact buckets such as source kind, target kind, layer distance, and source/target firing-frequency bins.
- [ ] Implement save/load helpers that tolerate missing artifacts when the stage is disabled or measurement has not yet run.
- [ ] Add unit tests for pair identity, direction validation, deduplication, and artifact round trips.

## Phase 3 — Compute Budgets And Storage Guardrails

- [ ] Treat pairwise discovery as a sparse measurement program, never an all-pairs graph construction.
- [ ] Add documented scale presets such as tiny/debug, research/default, and large/H100 that set candidate, target, pair, context, and control budgets.
- [ ] Enforce candidate-table and measurement-table memory caps before materializing large tensors.
- [ ] Prefer chunked and shardable artifacts for large runs, for example `pair_candidates_shard_000.pt` and `measurements_shard_000.pt`, with a small manifest describing schema, counts, dtypes, and shard paths.
- [ ] Keep candidate proposal CPU/tensor-first and avoid Python object records for each pair.
- [ ] Verify on synthetic large-count fixtures that budget checks fail early with clear messages rather than exhausting RAM or VRAM.

## Phase 4 — Pair Candidate Proposal

- [ ] Build `pair_candidate_proposal` as the cheap candidate stage inside pairwise discovery.
- [ ] Add a top-coactivation proposal source that converts coactivation neighborhoods into valid directed upstream-to-downstream pair candidates while preserving coactivation as an undirected evidence signal.
- [ ] Add a `seq_latent_index` proposal source that proposes pairs from shared sequence membership, target-context overlap, and disagreement with top-coactivation.
- [ ] Add random and matched-control proposal sources from the start, including matching by `stratum_id`, layer distance, kind pair, and firing frequency where available.
- [ ] Store candidate provenance in `proposal_mask` and cheap evidence in named fields such as `coactivation_score` and `seq_overlap_score` without assigning relationship classes.
- [ ] Preserve measured-null and control candidates rather than filtering only to apparently strong pairs.
- [ ] Verify on a tiny fixture that proposed candidates are valid, directional, deduplicated, and include expected provenance.

## Phase 5 — Acquisition And Budgeting

- [ ] Implement an acquisition policy that selects which candidate pairs receive deeper measurement under a configurable budget.
- [ ] Balance acquisition across high cheap signal, source disagreement, uncertainty, layer/kind coverage, layer distance, and control pairs.
- [ ] Make acquisition target-bundled: select target latents or target groups first, then choose a bounded number of upstream sources per target so gradient passes can score many pairs at once.
- [ ] Include random controls and matched controls in every measured stratum where possible, using `stratum_id` to compare proposed pairs against similar valid pairs.
- [ ] Add deterministic seeding so pairwise experiments are reproducible.
- [ ] Record why each pair was selected for measurement.
- [ ] Add tests confirming acquisition respects target, source-per-target, pair, context, and control budgets while covering configured strata.

## Phase 6 — Pairwise Measurement

- [ ] Implement a gradient measurement path that records upstream `source -> target` evidence for selected pairs and writes pair evidence rather than circuits.
- [ ] Reuse existing model, SAE, context, and gradient instrumentation where possible without changing current discovery method behavior.
- [ ] Measure selected pairs in target-centered bundles so each grad-enabled pass can reuse target contexts, activations, and gradients across multiple source latents.
- [ ] Measure raw evidence vectors such as `coactivation_score`, `seq_overlap_score`, conditional target activation, gradient sensitivity, context dependence, run stability, sign consistency, and uncertainty.
- [ ] Store measurement status explicitly, distinguishing proposed/unmeasured, measured/supported, measured/null, skipped/missing-context, and control pairs.
- [ ] Keep intervention measurement optional and separately budgeted for later expensive experiments.
- [ ] Treat intervention measurement as a second-tier validator for a small top slice of gradient-measured pairs plus matched controls.
- [ ] Add tests around measurement filtering, output shape, missing-context handling, and disabled intervention behavior.

## Phase 7 — Calibration, Confidence, And Null Models

- [ ] Add control-normalized scores such as matched-control percentile, empirical rank, and z-score within comparable `stratum_id` buckets when enough controls exist.
- [ ] Track measurement confidence separately from measurement strength, including observation count, context count, sign consistency, run stability, bootstrap uncertainty, and evidence-source agreement.
- [ ] Preserve negative evidence from measured/null pairs so the atlas can represent both supported and unsupported proposed relationships.
- [ ] Compare top-coactivation-proposed, `seq_latent_index`-proposed, disagreement-proposed, random-control, and matched-control populations.
- [ ] Add tests on synthetic distributions to confirm control-normalized scores and confidence metrics behave as expected.

## Phase 8 — Atlas And Analysis

- [ ] Build the pairwise atlas reducer that merges proposed candidates and measurements into a sparse directed interaction evidence table.
- [ ] Keep the atlas class-free: store measured signatures and distribution summaries, not labels like activation, inhibition, or gating.
- [ ] Generate summary statistics for interaction strength distributions, signed evidence distributions, confidence distributions, layer-distance effects, kind-pair effects, source disagreement, and control comparisons.
- [ ] Store raw evidence, provenance, measurement status, confidence metrics, and control-normalized scores together so later analysis can separate effect size from reliability.
- [ ] Add lightweight export options suitable for later plotting or publication analysis.
- [ ] Verify summary generation on synthetic data with known distribution shapes.

## Phase 9 — Pipeline Integration

- [ ] Insert `run_pairwise_discovery()` after `run_second_pass()` and before `run_candidate_selection()` in the full pipeline, guarded by `pairwise_discovery.enabled`.
- [ ] Ensure existing `run_candidate_selection()` and `run_discovery()` do not consume pairwise artifacts unless a future explicit opt-in is added.
- [ ] Add timing labels for pair candidate proposal, acquisition, measurement, calibration, and atlas building.
- [ ] Validate that a disabled run follows the same phase order and writes the same existing artifacts as before.
- [ ] Add an integration test or smoke test for a tiny enabled pairwise run.

## Phase 10 — Standalone Pairwise Runner

- [ ] Add a script analogous to circuit-only discovery, such as `src/discover_pairwise.py`, to run only pairwise discovery from existing artifacts.
- [ ] Load persisted inputs such as latent stats, contexts, top coactivation, `seq_latent_index`, and optional candidates without rerunning first/second passes.
- [ ] Initialize model and SAE resources only when the configured measurement mode requires them.
- [ ] Support config-driven modes first, with CLI flags later for proposal-only, calibration-only, measurement budget, target path, and output directory.
- [ ] Validate required input artifacts and print clear missing-artifact errors.
- [ ] Add a smoke test or documented command for running the standalone pairwise stage.

## Phase 11 — Optional Seed-Selection Signals

- [ ] Add pairwise-derived seed criteria only behind explicit config flags, leaving existing seed selection unchanged by default.
- [ ] Candidate criteria may include incoming measured interaction mass, upstream diversity, strong directed neighborhood evidence, high uncertainty, high confidence, or unusual control-normalized evidence.
- [ ] Attach pairwise seed scores to candidate metadata without changing discovery method semantics.
- [ ] Add ablation-friendly tests showing old seed criteria work unchanged when pairwise criteria are disabled.

## Phase 12 — Observability And Run Health

- [ ] Add run-health logging for candidate counts, selected target counts, selected source counts, measured-pair counts, control counts, shard sizes, and skipped-pair reasons.
- [ ] Log counts by `proposal_mask`, `stratum_id`, source kind, target kind, layer distance, measurement status, and control type.
- [ ] Log budget utilization for `candidate_budget`, `measured_pair_budget`, `measured_target_budget`, `sources_per_target`, `contexts_per_target`, `matched_controls_per_pair`, and `random_control_budget`.
- [ ] Track phase timings for proposal, acquisition, measurement, calibration, atlas reduction, saving, and loading.
- [ ] Track memory and storage observability, including estimated peak CPU RAM, estimated peak VRAM, artifact sizes, shard counts, and rows per shard.
- [ ] Track numerical health for score min/max/mean/std, NaN/inf counts, empty strata, missing contexts, and calibration availability by stratum.
- [ ] Write a lightweight `run_health.json` or equivalent summary beside the pairwise artifacts so every run can be inspected without running formal evals.

## Phase 13 — Algorithmic Evals

- [ ] Add a control separation eval comparing proposed measured pairs against matched controls within comparable `stratum_id` buckets.
- [ ] Add a calibration eval checking whether random and matched controls have sensible normalized score distributions, such as near-uniform percentiles and near-zero z-scores.
- [ ] Add a stability/replicability eval across seeds, data shards, or context resamples, measuring top-pair overlap, score correlation, sign consistency, and neighborhood overlap.
- [ ] Add a held-out context prediction eval that builds the atlas on one split and checks whether atlas scores predict gradient, coactivation, or sequence-overlap evidence on held-out contexts.
- [ ] Add a proposal-source ablation eval comparing top-coactivation-only, `seq_latent_index`-only, both-source, disagreement, random-control, and matched-control populations.
- [ ] Add an atlas completeness/coverage eval reporting whether measured pairs and controls cover enough source kinds, target kinds, layer distances, frequency buckets, and strata for meaningful analysis.
- [ ] Add an explainability/structure eval summarizing layer-distance profiles, kind-pair profiles, incoming/outgoing interaction mass, neighborhood coherence, and graph/community structure relative to controls.
- [ ] Add a downstream circuit utility eval comparing baseline seed selection against explicit atlas-informed seed selection under equal compute budgets.

## Phase 14 — Documentation And Verification

- [ ] Document `pairwise_discovery` as an intermediate evidence layer, not a circuit discovery replacement.
- [ ] Document the artifact contract, standalone script, disabled-by-default behavior, structural direction assumptions, target-bundled measurement strategy, observability outputs, and eval suite.
- [ ] Add config comments explaining candidate proposal, explicit proposal-score fields, target/source/context budgets, controls, calibration, confidence metrics, storage caps, run-health telemetry, eval settings, and class-free atlas outputs.
- [ ] Document success criteria: disabled pipeline unchanged, tiny enabled run writes pair candidates/measurements/atlas/summary, controls exist in measured strata, and atlas distinguishes proposed/unmeasured, measured/null, and measured/supported pairs.
- [ ] Run focused unit tests for new modules.
- [ ] Run existing discovery tests to confirm no regressions.
- [ ] Run one disabled full-pipeline smoke test or equivalent to confirm current behavior is preserved.
- [ ] Run one enabled tiny pairwise-discovery smoke test to confirm artifacts are written.

---

## Details

The central motivation is that jumping directly from individual latents to full circuits is too large a conceptual and technical step. A circuit is a multi-latent structure, but before we can justify grouping latents into circuits we need empirical evidence about how directed pairs of latents relate to each other. Pairwise discovery is intended to be that missing middle layer: it learns a sparse, directed evidence field over latent pairs before circuit seed selection and circuit discovery begin.

The desired pipeline is:

```text
first_pass
negative_context
second_pass
pairwise_discovery
circuit_seed_selection
circuit_discovery
```

This ordering keeps the current evidence flow clean. `first_pass` gathers per-latent statistics and context stores. `negative_context` builds contrast contexts. `second_pass` gathers broad coactivation evidence. `pairwise_discovery` then uses these existing artifacts to propose and measure directed latent-pair evidence. Only after that does the system choose circuit seeds and run circuit discovery.

The pairwise stage should be additive and disabled by default. Existing passes and circuit discovery methods should not change behavior unless pairwise functionality is explicitly enabled. This preserves the current baseline, makes ablations cleaner, and avoids hidden coupling between a new experimental stage and already-working discovery methods. The invariant is: when `pairwise_discovery.enabled` is false, current pipeline outputs and discovery behavior should remain unchanged.

The key scientific choice is to keep pairwise discovery class-free at first. We should not define relationship classes such as activation, inhibition, gating, redundancy, or synergy as fixed categories in the initial algorithm. Those classes may eventually emerge from measured evidence, but defining them too early risks baking in our assumptions. Instead, each directed pair should accumulate a vector of evidence: coactivation, sequence overlap, conditional activation, gradient sensitivity, intervention effects when available, context dependence, run stability, uncertainty, and control-normalized scores. Later analysis can ask whether these evidence vectors form clusters, heavy-tailed distributions, bimodal signed effects, continuous spectra, or layer/kind-specific regimes.

Pairs are directional by construction. A valid pair is always `source -> target`, where the source is structurally upstream of the target. Downstream latents cannot affect upstream latents, and pairs within the same `layer+kind` are excluded. Same-layer cross-kind pairs need a specific decision: either allow only the true forward order such as `attn -> mlp -> resid`, or exclude same-layer pairs initially for simplicity. This direction constraint is not a modeling bias; it reflects the computation graph of the model.

The system cannot measure all latent pairs. Pairwise discovery therefore needs a staged measurement ladder. Cheap sources propose a broad set of candidate pairs. An acquisition policy selects a smaller subset for deeper measurement. Gradient measurement and eventual intervention measurement are applied only to selected pairs. This lets us gather broad coverage without pretending exhaustive pairwise measurement is feasible.

The compute constraint should be explicit in the design. With 12 layers, 3 component kinds, and `d_sae = 40960`, the system has roughly 1.47M latents. The all-pairs directed space is therefore on the order of trillions of pairs, before any contexts, scores, gradients, or controls are considered. Pairwise discovery must be budget-first: it should operate over sparse candidate tables, hard measurement budgets, target/source/context caps, and shardable outputs rather than ever materializing a dense pair graph.

Useful scale presets should make this concrete:

```text
tiny/debug:
  proposed candidates:   10k-100k
  measured pairs:        500-2k
  context sequences:     4-8 per target/group
  storage:               <100 MB

research/default:
  proposed candidates:   1M-10M
  measured pairs:        10k-100k
  context sequences:     8-16
  storage:               ~0.5-5 GB

large/H100:
  proposed candidates:   10M-100M
  measured pairs:        100k-1M
  context sequences:     8-32
  storage:               ~5-50 GB
```

These are starting budgets, not targets that every run should hit. The main tuning knobs are `measured_target_budget`, `sources_per_target`, and `contexts_per_target`, because they determine how many expensive grad-enabled passes are required.

`pair_candidate_proposal` is the cheap candidate stage. The name is intentional: it avoids suggesting that proposed pairs are confirmed interactions. It should read from top coactivation, `seq_latent_index`, latent statistics, contexts, and random/control samplers. Top coactivation is useful because it finds latents that appear together, but it is not directional or causal on its own. `seq_latent_index` is useful because it gives a cheaper sequence-level retrieval axis that may surface structured relationships missed by coactivation. Random and matched controls are essential because without null comparisons the measured distributions will be hard to interpret.

Candidate artifacts should use explicit field names rather than generic score slots. The initial compact schema should include `source_global_id`, `target_global_id`, `proposal_mask`, `coactivation_score`, `seq_overlap_score`, and `stratum_id`. `proposal_mask` is a bitset that records why a pair entered the candidate table, for example top coactivation, `seq_latent_index`, disagreement proposal, random control, or matched control. `stratum_id` is a compact bucket used for balanced sampling and fair comparison, such as source kind, target kind, layer distance, and source/target firing-frequency bins.

Controls are baseline pairs measured to calibrate interpretation. Random controls are valid upstream-to-downstream pairs sampled from the broad pair space. Matched controls are valid pairs sampled to match a proposed pair's stratum, such as kind pair, layer distance, and frequency bins. Their purpose is to answer whether a measured pair is unusual relative to comparable pairs, not merely whether its raw score is large.

The acquisition policy is the bridge between cheap proposal and expensive measurement. It should not simply take the top coactivation pairs, because that would bias the atlas toward coactivation-shaped relationships. It should mix high cheap-signal pairs, disagreement cases, uncertain pairs, diverse layer/kind strata, layer-distance coverage, random controls, and matched controls. This supports both exploitation and exploration while preserving enough null data to estimate background distributions.

Acquisition should be target-bundled for efficiency. The bad pattern is one forward/backward pass per pair. The preferred pattern is to choose target latents or target groups, gather a small context set for each target, then score many candidate upstream sources from the same grad-enabled pass. This makes measurement scale closer to `measured_target_budget * contexts_per_target` than `measured_pair_budget * contexts_per_target`.

The first serious measurement engine should be gradient-based. For selected target latents, the system should gather upstream gradient evidence for selected or proposed source latents and write that evidence to pairwise artifacts, not to circuit objects. This is related to existing gradient discovery methods, but the output contract is different. Existing methods try to grow circuits; pairwise measurement records evidence about `source -> target` pairs.

Intervention measurement should be optional and separately budgeted. It is more causally direct but much more expensive. The plan should leave room for later source-latent interventions, target activation deltas, output deltas, and joint source/target effects, but the first implementation should not require all of that to be useful.

The atlas is the reduced empirical product of pairwise discovery. It merges proposed candidates and measurements into a sparse directed interaction evidence table. It should be thought of as a latent interaction atlas, not as a circuit store. Circuit discovery can later consume atlas information, but the atlas itself should remain a lower-level evidence artifact. The atlas should preserve raw evidence, measurement status, provenance, confidence, and control-normalized scores so later analysis can separate effect magnitude from reliability.

Confidence should be distinct from strength. A pair can have a large apparent effect but low confidence if it appears in too few contexts, changes sign across contexts, or fails to replicate across batches/seeds. Useful confidence fields include `context_count`, `observation_count`, `gradient_score_mean`, `gradient_score_std`, `sign_consistency`, `run_stability`, bootstrap uncertainty, and evidence-source agreement.

The first implementation should also preserve negative evidence. A measured pair with weak or null support is still informative, especially if it was proposed by a strong cheap signal. The atlas should distinguish proposed/unmeasured, measured/supported, measured/null, skipped/missing-context, random-control, and matched-control rows.

Observability and evals should be separate. Observability answers whether a particular run behaved as expected: counts, timings, budget utilization, memory estimates, artifact sizes, skipped pairs, missing contexts, score ranges, and calibration availability. These should be emitted during every normal pairwise run, ideally into a compact `run_health.json` plus readable logs.

Algorithmic evals answer whether the atlas is good. They should be deliberate experiment scripts or notebooks that consume pairwise artifacts and compare atlas behavior against controls, held-out contexts, repeated runs, proposal-source ablations, and downstream circuit discovery. The boundary is: observability checks run health, while evals judge scientific and practical atlas quality.

The first formal eval suite should include control separation, calibration, stability/replicability, held-out context prediction, proposal-source ablation, completeness/coverage, explainability/structure, and downstream circuit utility. Together these evaluate accuracy, explainability, completeness, and whether the atlas improves the larger circuit discovery objective.

The standalone runner is important because pairwise discovery will be experimental and budget-sensitive. We will want to rerun candidate proposal, acquisition, and measurement under different budgets and sources without rerunning `first_pass`, `negative_context`, or `second_pass`. A script such as `src/discover_pairwise.py` should load existing artifacts, initialize model resources only when measurement requires them, run the pairwise stage, and write pairwise artifacts.

Seed selection should only consume pairwise artifacts behind explicit opt-in criteria. Useful future seed signals might include incoming measured interaction mass, upstream diversity, unusually strong directed neighborhoods, uncertainty, or control-normalized pair evidence. These should be additional criteria, not implicit behavior changes.

This design gives three distinct scientific layers:

```text
latent statistics
directed pairwise latent interactions
multi-latent circuits
```

That separation is the main reason for the plan. It lets us study the distribution and geometry of latent interactions before imposing circuit-level explanations.

---

## Hypotheses To Test

- Coactivation-rich pairs and gradient-sensitive pairs will only partially overlap, because coactivation can reflect shared context while gradients are closer to directional influence.
- `seq_latent_index` disagreement cases may reveal conditional, position-specific, or weaker structured relationships that top coactivation misses.
- Pair evidence will likely be sparse and heavy-tailed, with most valid pairs near-null and a small number carrying most measured interaction mass.
- Evidence distributions may differ strongly by kind pair and layer distance, for example `attn -> mlp` versus `mlp -> resid`.
- Strong future circuit seeds may be better identified by incoming measured interaction mass, upstream diversity, or unusual control-normalized neighborhoods than by individual activation strength alone.

---

## High-Level Pseudocode

```text
run_pairwise_discovery(config):
  if not config.pairwise_discovery.enabled:
    return PairwiseDiscoveryResult(disabled=True)

  inputs = load_pairwise_inputs(
    latent_stats,
    top_ctx,
    mid_ctx,
    neg_ctx,
    top_coactivation,
    seq_latent_index,
  )

  pair_schema = build_pair_schema(
    n_layers=12,
    kinds=["attn", "mlp", "resid"],
    d_sae=40960,
    same_layer_policy=config.pairwise_discovery.same_layer_policy,
  )

  budget = resolve_pairwise_budgets(config.pairwise_discovery)
  assert_budget_is_feasible(budget, available_cpu_ram, available_disk, available_vram)

  pair_candidates = propose_pair_candidates(inputs, pair_schema, budget)
  selected_pairs = acquire_pairs_for_measurement(pair_candidates, inputs, budget)

  measurements = []
  if budget.measurement_enabled:
    target_bundles = group_selected_pairs_by_target(selected_pairs, budget)
    measurements = measure_target_bundles(target_bundles, inputs, budget)

  calibrated = calibrate_against_controls(
    selected_pairs,
    measurements,
    group_by="stratum_id",
  )

  atlas = build_pairwise_atlas(
    pair_candidates,
    selected_pairs,
    measurements,
    calibrated,
  )

  save_pairwise_artifacts(pair_candidates, selected_pairs, measurements, atlas)
  return PairwiseDiscoveryResult(disabled=False, atlas=atlas)
```

```text
propose_pair_candidates(inputs, pair_schema, budget):
  candidates = empty_tensor_table(max_rows=budget.candidate_budget)

  for target in valid_target_latents(inputs.latent_stats):
    for neighbor in top_coactivation_neighbors(target):
      source, target = orient_pair_if_valid(neighbor, target, pair_schema)
      if valid_pair(source, target):
        add_or_update_candidate(
          source_global_id=source,
          target_global_id=target,
          proposal_mask=TOP_COACTIVATION,
          coactivation_score=neighbor.score,
          stratum_id=make_stratum(source, target, inputs.latent_stats),
        )

  for seq_group in seq_latent_index_groups(inputs.seq_latent_index):
    for source, target in valid_directed_pairs_from_sequence_group(seq_group, pair_schema):
      add_or_update_candidate(
        source_global_id=source,
        target_global_id=target,
        proposal_mask=SEQ_LATENT_INDEX,
        seq_overlap_score=estimate_sequence_overlap(source, target, seq_group),
        stratum_id=make_stratum(source, target, inputs.latent_stats),
      )

  add_disagreement_candidates(candidates, budget)
  add_random_control_candidates(candidates, pair_schema, budget)
  add_matched_control_candidates(candidates, pair_schema, budget)

  dedupe_sort_and_shard(candidates)
  return candidates
```

```text
acquire_pairs_for_measurement(pair_candidates, inputs, budget):
  selected = []

  for stratum in configured_or_observed_strata(pair_candidates):
    proposed = sample_mixed_proposals(
      pair_candidates,
      stratum=stratum,
      signals=[
        coactivation_score,
        seq_overlap_score,
        source_disagreement,
        uncertainty,
        layer_distance_coverage,
      ],
      budget=budget.per_stratum_pair_budget,
    )

    controls = sample_controls(
      pair_candidates,
      stratum=stratum,
      matched_controls_per_pair=budget.matched_controls_per_pair,
      random_control_budget=budget.random_control_budget,
    )

    selected.extend(proposed)
    selected.extend(controls)

  selected = enforce_global_budgets(
    selected,
    measured_target_budget=budget.measured_target_budget,
    measured_pair_budget=budget.measured_pair_budget,
    sources_per_target=budget.sources_per_target,
  )

  return selected
```

```text
measure_target_bundles(target_bundles, inputs, budget):
  all_measurements = []

  for target, pairs_for_target in target_bundles:
    contexts = choose_target_contexts(
      target,
      top_ctx=inputs.top_ctx,
      mid_ctx=inputs.mid_ctx,
      neg_ctx=inputs.neg_ctx,
      max_contexts=budget.contexts_per_target,
    )

    if contexts_are_missing_or_insufficient(contexts):
      mark_pairs_as_skipped(pairs_for_target, reason="missing_context")
      continue

    source_ids = limit_sources_for_target(
      pairs_for_target.source_global_id,
      max_sources=budget.sources_per_target,
    )

    grad_result = run_grad_enabled_target_pass(
      target=target,
      source_ids=source_ids,
      contexts=contexts,
    )

    for pair in pairs_for_target:
      evidence = summarize_pair_evidence(pair, grad_result)
      all_measurements.append({
        source_global_id: pair.source_global_id,
        target_global_id: pair.target_global_id,
        gradient_score_mean: evidence.mean,
        gradient_score_std: evidence.std,
        sign_consistency: evidence.sign_consistency,
        context_count: evidence.context_count,
        observation_count: evidence.observation_count,
        measurement_status: evidence.status,
      })

  return shard_measurements(all_measurements)
```

```text
build_pairwise_atlas(pair_candidates, selected_pairs, measurements, calibrated):
  atlas = join_on_pair_key(
    pair_candidates=[
      source_global_id,
      target_global_id,
      proposal_mask,
      coactivation_score,
      seq_overlap_score,
      stratum_id,
    ],
    selected_pairs=selected_pairs,
    measurements=measurements,
    calibrated=calibrated,
  )

  atlas = add_status_columns(
    atlas,
    statuses=[
      "proposed_unmeasured",
      "measured_supported",
      "measured_null",
      "skipped_missing_context",
      "random_control",
      "matched_control",
    ],
  )

  summary = summarize_atlas(
    atlas,
    group_by=[
      proposal_mask,
      stratum_id,
      layer_distance,
      source_kind,
      target_kind,
      measurement_status,
    ],
  )

  return atlas, summary
```

---

## Open Questions

- Should component ordering treat same-layer different-kind pairs as valid only in the actual forward order, for example `attn -> mlp -> resid`, or should all same-layer cross-kind pairs be excluded initially?
- Should the first pairwise measurement target all selected circuit seeds, top `n` latents per `layer+kind`, or a hybrid of both?
- What is the minimum useful `seq_latent_index` evidence to store for pair proposal: latent IDs only, activation ranks, activation values, token positions, or sequence-level summaries?
- Should `pair_candidate_proposal` always write `pair_candidates.pt`, or should large runs write only sharded `pair_candidates_shard_*.pt` files plus a manifest?
- Which artifact format is best for large pair tables: Torch tensors first, Parquet summaries later, or both from the beginning?
- What should the default matched-control ratio be: one matched control per measured proposed pair, or a smaller per-stratum control budget?
- How many contexts per target are needed before confidence metrics become meaningful enough for publication analysis?

## Risks / Assumptions

- Pairwise discovery may be expensive unless acquisition budgets and controls are enforced early.
- The pair space is too large for exhaustive enumeration; every implementation path must preserve sparsity and enforce budget caps before allocation.
- Target-bundled measurement is assumed to be the main route to feasible gradient costs; pair-by-pair backward passes would likely be too expensive.
- Coactivation and `seq_latent_index` are proposal signals only; treating either as causal evidence would bias the atlas.
- Controls are necessary for interpretation, but excessive controls can dominate measurement budget unless capped by stratum.
- Large pair tables must be tensor-first and shardable; Python object records per pair will not scale.
- The atlas should remain class-free at first so relationship categories can emerge from measured distributions rather than being predefined.
- Existing circuit discovery must remain reproducible and unchanged when pairwise discovery is disabled.
- The standalone runner depends on reliable persisted artifact loading, so save/load contracts need to be explicit before measurement work expands.
