# WCM: clean confirmation and causal-scrubbing pilot

Date: 28 October 2026. Status: proposed research plan; no new experiment has been run under this plan.

## Recommendation

Finish the clean v2 evidence for the current methods paper, while building a small, independently verified recursive causal-scrubbing pilot. Keep production WCM unchanged during initial evaluation. Expand the pilot only after its hypothesis, implementation, donor coverage, and cost are established.

The two questions are:

1. **Support robustness:** which selected latent sets sustain the target under specified ablations and donor backgrounds, at what size and fitting budget?
2. **Mechanistic explanation:** does a precisely specified interpretation of intermediate computations survive the recursive interventions it implies?

These require different experiments. WCM's selected nodes and coefficients do not alone specify the interpretation graph needed for the second question. Report the two results separately.

## Evidence and scope at the start

- The current manuscript still contains historical v1 headline values and placeholders for clean v2 results. They must not become final claims by removing the placeholders alone.
- Experiment 095's 72-target development panel gives production passes of 50/72 for standard v2 and 45/72 with an added resampling term; corresponding donor-resampling passes are 0/72 and 17/72. Those records have known duplicate leakage. This is motivation for a clean experiment, not its result.
- Experiments 073/074 already implement donor-background scoring, pinning diagnostics, and all-position reads. Experiment 095 implements a resampling-trained arm. Reuse their tested functionality with explicit configuration and provenance.
- Experiments 083/084 supply attribution/fitted-support comparisons and harder nulls. Experiment 098 supplies an **adapted sparse-regression baseline**, with material differences from published CircuitLasso. Preserve that label and its adaptation notes.
- Experiment 094 supplies the fresh TuringLLM record construction and fixed target lists. The inspected local outputs establish smoke readiness, not completion of the 8k confirmation. Remote run status has not been checked for this plan.
- There are substantial pre-existing working-tree changes. Implementation should use a new experiment directory and focused additions, with a source snapshot recorded before runs; do not overwrite earlier evidence.

Existing source material:

- `docs/critical-review-2026-10-27.md`
- `paper/main.tex`, `paper/protocol-v1.md`
- `dev-notes/2026-10-23-pod-plan-public-models.md` (follow its current v2 decisions; older sections describe superseded plans)
- `experiments/083-v1-comparison/README.md`
- `experiments/084-harder-nulls/README.md`
- `experiments/094-tl-v2-records/README.md`
- `experiments/095-v2-resample-appendix/README.md`
- `experiments/098-circuitlasso-baseline/README.md` and `PAPER_NOTES.md`

## Stage 0 — Establish trustworthy inputs and freeze definitions

**Priority:** first; required for every new empirical claim. A clean pilot subset is sufficient to start; the complete production run need not finish first.

1. Resolve and regression-test the context-construction findings in the 27 October audit before launching affected fits:
   - unstable held-out rows must be recorded without expanding integer dictionary keys as keyword arguments;
   - the actual number of training rows consumed by the fitting engine must equal the record's declared count, including records with too little held-out padding;
   - contrast train/held copy filtering must implement the agreed exact/near-copy rule in saved records.
   These are verification/repair tasks, not findings that this plan has newly reproduced. The current source still contains the suspicious merge and padding paths.
2. Independently verify saved pilot records: the agreed duplicate rules, actual train/held consumption, firing at activating anchors, contrast silence, and no evaluation-dependent means or hyperparameter selection. Retain the agreed duplicate-based protocol; do not silently add a source-document restriction.
3. Record the model and SAE identifiers, numerical settings, source revision plus dirty-source hashes, context hashes, target lists, random seeds, fitting settings, and scorer versions. Refuse incompatible cached artifacts.
4. Freeze the current production pass rule as a compatibility metric. Add new continuous diagnostics without retrospectively redefining that rule.

**Output:** an input-validation report and run manifest. **Exit:** every pilot record passes, or is excluded with a declared reason and counted; actual training/evaluation separation is verified.

## Stage 1 — Correct reference implementation of recursive scrubbing

**Priority:** small CPU implementation, independent of the transformer adapter. This can proceed while clean records are prepared.

Suggested location: a new `experiments/099-recursive-causal-scrubbing/` directory, after checking that the identifier is still free at implementation time. Keep experiment orchestration out of production discovery code initially.

Implement the published algorithm over a finite, deterministic computational DAG:

- Explicit low-level graph G, interpretation graph I, correspondence c, and empirical distribution D.
- Validate acyclicity, node/function signatures, correspondence injectivity, edge preservation, and input/output mapping.
- Sample reference inputs from D. For each important parent, sample from D conditioned on equality of that parent's interpretation value and recurse.
- Use a single donor for all unimportant parents of a particular node, matching the published pseudocode. Make alternative grouping a different, named hypothesis.
- Preserve independent recursive draws on separate paths through a shared DAG ancestor. Do not memoize a scrubbed node merely by node ID: that would collapse treeification. Caching deterministic natural activations and matching sets is safe.
- Log the intervention trace, matching-set size, self-donor fraction, path count, natural forwards, runtime, and memory.
- Use exact matching in the reference algorithm. Quantised interpretations are explicitly different hypotheses, with boundaries frozen using development data. Approximate-neighbour sampling is not silently called the published algorithm.

Meaningful correctness tests:

1. Compare the recursive evaluator with an explicitly treeified evaluator on small branching DAGs, including exhaustive output distributions where feasible.
2. Exercise a known correct hypothesis and deliberately incorrect hypotheses with missing dependencies or falsely interchangeable inputs.
3. Check joint donor grouping and separate path draws; verify that changing these produces the expected different distribution.
4. Check the no-swap identity, deterministic reproducibility, singleton matching sets, and reported coverage.
5. Where graph rewrites are used, verify equality with the original computation before any scrubbing.

**Output:** tested reference engine and toy report. **Exit:** recursion/treeification agreement and all declared identities pass. These tests are warranted because implementation errors would invalidate the experiment.

## Stage 2 — Clean support-robustness benchmark

**Purpose:** determine which existing failures survive clean data, and compare methods under a common intervention. This is not yet semantic recursive scrubbing.

### Proposed small panel

- Freeze 27 eligible targets from the fresh-record frame: three per depth band x site kind, where the frame permits.
- Reserve one per cell (9) for development and two per cell (18) for confirmation. Use a fixed seeded draw, before new scores are inspected; do not select only production passes.
- Explain eligibility and any shortage or exclusion. A replacement list must be fixed in advance, not chosen using scores.
- Report results on all evaluable targets, and secondarily conditional on production success. Treat this as a pilot, with wide intervals; it is not a population estimate replacing the full run.
- Keep the main hand-interpreted case studies separate from this sampled panel.

### Minimum comparison arms

1. Production WCM, fitted coefficients.
2. The same WCM nodes with coefficients set to 1 (a coefficient-dependence diagnostic, not a separately optimised competitor).
3. Separately optimised unweighted masking.
4. IG-ranked supports at natural scale and with fitted coefficients, reusing 083's documented latent-target adaptation.
5. Anchor-active/contribution-matched random supports with coefficient fitting, reusing 084 and reporting match quality.
6. The existing adapted sparse-regression baseline, at natural scale and fitted, if its marginal pilot cost is modest. It must be included before a comparative claim against CircuitLasso-style selection.

Use the same targets, training records, evaluation records, donor pools, target read, background conventions, and coefficient-fitting recipe wherever the comparison calls for them. Report method-specific discovery cost and tuning budget. Show achieved size/performance points; do not present unequal sparsity penalties as equal sizes, or coefficient fits as equal capacity to unfitted circuits.

Start at each method's declared operating point and a small prespecified size sweep on development targets. Freeze the sweep before confirmation. The existing 083 frontier design remains authoritative for the main production comparison; this pilot should not replace it with a weaker single-point test.

### Reads and diagnostics

- Existing production criterion and current donor-resampling criterion, clearly named.
- Per-context target-activation error, using a scale fixed from training data; report the denominator and undefined/near-zero cases.
- Activation correlation and pooled error, with constant/undefined cases counted.
- Actual post-Top-K firing, the signed encoder pre-activation or its explicitly named ReLU version, and target rank. Never mix their definitions.
- Silent-position false firing and activating-position recovery on both positive and contrast texts.
- Natural-value pinning versus live recomputation; the original context as its own donor; a small complement-strength sweep.
- Explicit reconstruction-error policy and a small sensitivity diagnostic. Removing error channels changes the intervention; it does not automatically produce a fairer reference.

Use paired target-level estimates and intervals; donor draws are Monte Carlo repeats, not independent targets. Profile 8 draws on development targets, then choose and freeze the confirmation draw count using Monte Carlo uncertainty (for example 32 if needed). Report within-target uncertainty separately from across-target uncertainty.

**Output:** paired curves, diagnostic table, costs, and a clean replication of the existing resampling comparison. **Exit:** shared scorer identities pass and the report identifies the dominant failure patterns, including inconclusive cases.

## Stage 3 — One genuine recursive interpretation test

**Purpose:** test what the intermediates compute, beyond support sufficiency.

Start with `9.resid.20419` (the antiquity case) if its fresh records and refit are available. Select a second, simpler case on development evidence only if it adds a distinct, well-specified hypothesis. Do not make a two-case result stand in for population-wide mechanism recovery.

Before transformer implementation, write a hypothesis card containing:

1. **Explanandum and distribution:** exact target, positions, corpus or controlled prompts, and whether the claim concerns firing strength, firing occurrence, or a semantic response. Use held-out natural text for transfer if controlled prompts are used to obtain matching examples.
2. **Metric:** expected target behaviour, plus per-example fidelity diagnostics. If a clean activation is used as a reference label, include its branch explicitly in the graph and keep its association with the base example intact; never supply reference labels as an unaccounted patched signal.
3. **Complete G, I, and c:** deterministic functions and dependency edges, not only semantic node names. Include positions where a claim is position-specific; an all-position latent membership mask does not imply a token-level causal graph.
4. **Interpretation values:** operational definitions of proposed date/place/context variables. Distinguish external semantic features from activation bins. Matching bins test representational invariance at that resolution; they do not validate the semantic label by themselves.
5. **Background structure:** which inputs are sampled jointly, which independently, and why. Explicitly represent retained dense computations, Top-K competition, positional information, attention, and SAE errors rather than counting only selected features.
6. **Original versus fitted computation:** declare which model the hypothesis explains. The primary mechanism test should concern the original model. A secondary hypothesis may concern the fitted weighted computation, labelled accordingly. A weighted hypothesis about the original model requires an exact rewrite retaining its compensating residual; inserting gains without that rewrite changes G and cannot silently be called an explanation of the original function.
7. **Development/confirmation separation:** freeze semantic definitions, correspondence, grouping, and all sampling rules before confirmatory scoring. Revised hypotheses need fresh confirmation.

Use the tested reference engine to specify adapter behaviour. Verify the adapter's untouched graph reproduces the original forward within measured numerical tolerance. Existing `FeatureGraph` is a gradient/activation container, not an executable causal DAG; a new adapter must implement the actual parent functions and path-specific interventions.

### Donor coverage is part of the result

For canonical scrubbing, use a frozen held-out empirical D for both reference draws and conditional donors; the algorithm permits a self-donor. Report its frequency rather than removing it silently. Require a prespecified useful matching-set size (proposed minimum: 8 distinct examples per tested equivalence class), and report how much of D meets it. Singleton classes provide almost no interchange test and are **untested**, not evidence for a mechanism.

If coverage is inadequate, collect more examples using the frozen property definitions, or explicitly narrow the tested domain. A separate disjoint-recipient/donor transfer experiment is useful but is a named sampling variant. Donor collection, matching rules and activation thresholds must not be tuned against confirmation outcomes.

Compare a small hierarchy of prespecified hypotheses: broad grouped dependencies, the proposed intermediate-feature explanation, and a more specific decomposition only where justified. Do not assume scores must decrease with specificity.

**Output:** hypothesis card, graph, correctness checks, coverage report, scrubbed scores and traces. **Exit:** at least one non-vacuous recursive interpretation experiment with a defensible graph and measured runtime. If the adapter or hypothesis cannot be made precise at a useful cost, report that limit and stop expansion.

### Fairness across discovery methods

Stage 2's support tests apply uniformly to every method. Stage 3 needs an interpretation and correspondence for each tested candidate; the same semantic specification can be used where valid, but a different node set does not automatically implement it. Record which supports can express the fixed hypothesis and which cannot, and keep coverage/cost visible. Do not report a WCM-only interpretation test as a ranking of discovery algorithms.

## Stage 4 — Diagnose, then run one targeted intervention

Keep the first confirmation untouched. Any remedy is a new arm developed separately and tested on new confirmation examples/targets.

| Evidence | Candidate remedy | Trade-off to measure |
|---|---|---|
| Natural pins recover the target; live intermediates fail | Add upstream dependencies, lower sparsity pressure, or constrain intermediate reconstruction | Extra nodes, paths and fitting cost |
| Gains overshoot with the original background restored | Train across complement-strength mixtures including the original context; compare a penalty toward gain 1 and a per-site scalar control | Size versus gain flexibility and robustness |
| Pre-activation is close but Top-K firing is wrong | Represent relevant competitors and improve rank/firing constraints, including silent examples | Target specificity and off-target activation |
| Predictions fail under semantic swaps despite adequate matching | Revise the interpretation or model interacting variables jointly | Less specific explanation; new confirmation required |
| Error-path interventions drive the result | Include/account for those paths and measure their functional contribution | Honest explanation size and scope |
| Confidence intervals are dominated by donor sampling or donor sets are tiny | Increase draws or collect matching data | Evaluation cost; no stronger mechanistic claim until resolved |

The first model-changing candidate, if clean Stage 2 reproduces the existing pattern, is the existing **WCM + resampling** arm. Reuse 095 before inventing recursive training. Compare it at achieved sizes and measure both robustness and production performance. A confidence interval crossing zero is not evidence of noninferiority.

Only train against recursively sampled interventions after Stage 3 establishes that they are valid, informative, and affordable. Keep the interpretation fixed within such an experiment and preserve a separate held-out test. Increasing the circuit size is an acceptable outcome; forcing the original size is not the scientific objective.

## Stage 5 — Expansion decision and paper consequences

| Outcome | Next step |
|---|---|
| WCM is better under shared support tests, with clean confirmation | Complete the existing main-paper comparisons; state the intervention-dependent advantage |
| WCM and baselines all fail detailed interpretation tests | Characterise the common boundary; retain the methods contribution if clean production evidence supports it |
| Weighted supports improve but require gains or substantial retained background computation | Frame the contribution as fitted sparse supports/compression under specified interventions |
| A specific interpretation survives nontrivial recursive swaps and transfers to fresh examples | Include it as a validated case study, with coverage and scope; consider a larger study |
| Gain controls or stronger fitted baselines remove the advantage | Narrow/rewrite the claim before scaling further |
| Matching is vacuous or path cost is prohibitive | Improve specification/data, group computations with an explicitly weaker hypothesis, or stop; no large run by default |

Keep clean v2 population confirmation, public-model replication, fitted baselines, harder nulls, and the number-to-artifact audit as the paper-completion track. Full recursive scrubbing is a bounded extension, not a prerequisite that indefinitely delays every publishable result.

Promote the main resampling boundary into the main text if confirmed. Replace statements equating donor fill with the entire causal-scrubbing algorithm. Regenerate the abstract, figures and conclusion together from the final evidence manifest.

## Execution order and resource gates

1. Repair/verify context handling; freeze manifests and a clean development subset.
2. Implement and test the generic CPU reference engine; write the first hypothesis card.
3. Profile one real target; run the clean support benchmark on the 9 development targets.
4. Freeze settings and run the 18-target support confirmation. In parallel as workstreams, complete the existing paper evidence; this is not an instruction to launch extra agents or jobs.
5. Execute one fully specified recursive case study if the graph and coverage gates pass.
6. Choose at most one remedy for a new paired confirmation before considering a broader campaign.

Before any larger run, record measured seconds/forward, forwards/target, peak memory, path count, donor coverage and total projected GPU-hours. Reuse existing local resources for the pilot where practical. This document does not authorise a new paid cloud campaign or change the existing cloud-run budget. Do not invent a runtime or calendar promise before profiling.

## Literature positioning to preserve

- [Causal scrubbing, main algorithm](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)
- [Causal-scrubbing appendix: graph rewrites, sampling, limitations](https://www.lesswrong.com/posts/kcZZAsEjwrbczxN2i/causal-scrubbing-appendix)
- [ACDC, Appendix N](https://arxiv.org/html/2304.14997): explicitly a restricted class of causal-scrubbing hypotheses.
- [Edge Pruning](https://arxiv.org/html/2406.16778) and [EAP-IG](https://arxiv.org/html/2403.17806): real-input edge replacement and circuit-performance tests.
- [Circuit Tracing](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html): quantitative perturbation validation and declared methodological limits.
- [MIB](https://arxiv.org/html/2504.13151): standardised circuit tests and causal-variable interchange interventions.
- [Comparison of causal scrubbing and causal abstractions](https://www.lesswrong.com/posts/uLMWMeBG3ruoBRhMW): the criteria are not a single ordered ladder of strictness.

Claim additional, explicit validation of WCM and its baselines where performed. Do not claim that other leading work ignores causal validation, or that passing scrubbing proves an interpretation uniquely correct.
