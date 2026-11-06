# Causal-scrubbing pilot: 28 October 2026

Status: initial campaign executed in `experiments/099-recursive-causal-scrubbing/`.
See its `out/summary.md`: 18 clean WCM/IG comparisons, a tested reference engine, and two coarse real-model
recursive hypotheses with numerical, full-state, random-support and full-SAE controls. The plan below records
the original priorities; the semantic antiquity case and all-node recursive validation remain later work.

## Today's question

How much of WCM's target recovery survives specified activation interchanges, and which additional dependencies explain any loss?

Separate two estimands throughout:

1. Whether selected components explain computation in the original model at natural strength.
2. Whether the fitted weighted computation is stable under the specified interventions.

Do not report a single generic "causal-scrubbing score for WCM" without its hypothesis, graph, donor distribution, gain convention, and error-path treatment.

## Existing pieces to reuse

- `experiments/073-strict-eval/strict_eval.py`: whole-donor background resampling. This is not the general recursive algorithm.
- `experiments/073-strict-eval/diag.py`: live versus pinned members, donor dose, self donors, pre-activation and actual firing.
- `experiments/095-v2-resample-appendix/`: production v2 versus resample-trained v2CR, using historical records with known duplicate leakage. Useful for development, not clean confirmation.
- `experiments/077-antiquity-scrub/antiquity_scrub.py`: same-class/cross-class group swaps in the original model. This is a group-intervention experiment, not a recursive scrub. The script exists; inspect its outputs and validity before assuming it has run successfully.
- `experiments/083-v1-comparison/`: attribution-ranked supports and support-restricted coefficient fits.
- `experiments/098-circuitlasso-baseline/`: an adapted Lasso baseline, with changes from the published CircuitLasso method documented in its README.
- `experiments/094-tl-v2-records/`: clean record construction and verification. Inspection on 28 October found local `out/smoke10` and `out/dryrun` directories, not a completed local 8k production run.

`src/circuit/instrument/sae_graph.py` captures attribution tensors and gradients. It does not supply an explicit, independently executable causal-scrubbing graph. `experiments/023-recursive-map/` expands selected upstream drivers; recursive selection is not recursive scrubbing.

## Freeze before evaluating

- Create a run manifest with source/config fingerprints, model/SAE identity, target list, fit-context hashes, evaluation-context hashes, donor pool hashes, gains, and RNG seeds.
- Select 18 exploratory population targets: two per depth band x SAE kind, without selecting on a new scrubbing score. Keep one independently selected target per cell for development and one for a later within-day check. Report unavailable cells and exclusions; do not silently replace failed fits.
- Treat the antiquity case (`9.resid.20419`) as a separately reported, previously selected interpretability case. It is not evidence for a population pass rate.
- Reuse verified clean records if available. Check the actual saved records and split consumed by fitting. Check exact/near-copy filtering for activating and contrast pools and anchor-prefix filtering where prescribed by protocol.
- An old fit cannot be made clean merely by changing its evaluation path. Check its original training contexts against new evaluation contexts. Prefer fresh fits on the frozen clean records for this pilot.
- If clean records or compatible fits are unavailable, explicitly label any historical run as development evidence. Do not substitute it into a clean headline.
- Freeze donors, anchors/position alignment, gain convention, intermediate features, matching rules, and metrics before inspecting their held-out results. Any refinement is developed on the development side, then checked on untouched examples.

## Stage 1: establish the starting point, about 1-2 hours

Start with three targets spanning depth to verify the harness and measure runtime. Expand to the 18-target panel only after those checks pass.

Primary arm: fresh standard v2 WCM with fixed membership and coefficients after fitting. No adaptation to today's scrubbing scores.

Evaluate natural recovery and the existing production Z/A/C interventions, then donor backgrounds from activating, contrast, and random contexts, separately. Reuse the existing scorer but retain raw outputs and denominators.

Use eight donor draws initially, recording draw variability. Add draws only where uncertainty affects the interpretation. Fit means and matching thresholds on training/development examples; evaluation donors must not be fit donors or prohibited copies. Donor sampling should be paired across comparable arms.

Diagnostics on a small subset:

- Natural member strengths versus fitted strengths.
- Recomputed members versus members pinned to natural activations. Pinning is a diagnostic, not a successful recursive explanation.
- Self-donor behavior. At alpha=1, require identity to numerical tolerance; with fitted gains, measure deviation rather than expecting identity.
- SAE error paths preserved versus an explicitly described sensitivity intervention. Changing error treatment changes the hypothesis/reference; report the resulting object clearly.

Secondary arms, after the primary harness works:

- Unweighted masking, with its own fitted support and size recorded; do not treat resetting WCM's gains to one as a substitute for discovering an unweighted circuit.
- One adapted IG-ranked baseline at matched size, at natural strength and with the same coefficient-fitting budget as appropriate.
- Adapted Lasso if artifacts and runtime permit. Keep the documented departures from published CircuitLasso explicit.

Prioritise primary WCM diagnostics over adding every baseline today. Add v2CR later as a modification experiment on the same clean records, rather than silently changing the method under initial evaluation.

## Stage 2: implement the published recursion on tiny graphs, about 1-2 hours

Represent G, I, and their correspondence explicitly. Validate the required mapping and graph structure. Begin with deterministic finite datasets so conditional sampling can be checked exactly.

Implement the original algorithm's semantics:

- At each mapped parent, sample from the dataset conditional on agreement of the interpretation value with the current local reference datum, then recursively recompute that parent.
- Unmapped parents of a particular mapped node use one shared unconditional donor, as in the published pseudocode.
- Different occurrences along different paths have separate recursive state and draws. Do not memoize a stochastic activation globally by graph-node identity.
- Log the actual interventions and conditional donor support.

Meaningful checks:

- Agreement with an explicitly treeified evaluator on the same sampled intervention trace.
- Known correct and deliberately incorrect toy hypotheses.
- A branching/shared-ancestor example that distinguishes proper path-specific recursion from one global donor pass.
- Identity/no-interchange cases; correct sharing of the unimportant-parent donor; reproducible sampling.
- Exact conditional probabilities on a small enumerable dataset.

This verifies implementation semantics. It does not guarantee the algorithm rejects every false hypothesis.

## Stage 3: one real-model recursive hypothesis, remaining time

Use a shallow target for the first adapter, because it limits computational paths. Keep the antiquity semantic case as a second, more ambitious track.

Before a real-model recursive score is accepted:

1. Define a coarse but explicit graph with at least two mapped intermediate stages and branching. Composite computations are allowed, but their functions and all bypass/background paths must be specified.
2. Demonstrate that the unscripted/unintervened graph evaluates to the original target computation within an agreed numerical tolerance. An attribution graph or an invented edge between two selected latents does not establish this.
3. Define the interpretation functions independently of the evaluation result. Activation-state categories may be used for an initial representational-invariance hypothesis; label them accordingly. They do not by themselves validate a semantic story.
4. For semantic hypotheses, specify the variable at each stage, its dependencies, correspondence, and token alignment. Reuse the antiquity prompts/groups only as candidate structure. The existing binary classical/non-classical label is an initial coarse hypothesis and may be inadequate for date/place routes.
5. Report conditional donor support, self-swap rate, and coverage. Exact matching of continuous vectors that only admits self donors is uninformative. If bins are used, freeze them on development data and identify the resulting discretised hypothesis.
6. Execute the full recursion for this chosen graph. A depth-limited run must be labelled partial. Do not describe background replacement alone as the full algorithm.
7. Keep weights and errors explicit. A graph of the fitted weighted model tests that model. A weighted hypothesis about the original model needs an exact decomposition, including any compensating residual, before scrubbing; inserting gains alone changes G.

Start with a few base contexts and measure runtime before increasing sample count. Full recursion can grow exponentially in branching depth. If a faithful real-model adapter is not ready today, deliver the tested reference engine and the completed first-stage data; do not manufacture a real-model result by using an approximate graph without disclosure.

## Metrics and report

For each target and hypothesis, retain:

- Mean target recovery, raw clean/scrubbed target means, and denominators.
- Per-context normalised error, using a training-derived scale with a stated floor, rather than dividing by each context's possibly zero activation.
- Correlation where defined, actual Top-K firing and rank, and false firing on silent contexts.
- Post-Top-K output and signed pre-activation separately; existing ReLU(pre-activation) diagnostics must be named accurately.
- Circuit size, gains, error-path convention, donor coverage, donor-draw uncertainty, forward count, runtime and memory.

Freeze any binary pass rule before evaluation. Existing production pass thresholds may be included for continuity but must not be presented as the original causal-scrubbing paper's universal acceptance rule. Primary pilot interpretation uses continuous paired outcomes. With nine held-back targets, broad population estimates are not credible.

## Decisions after the first results

- Pinned works, live fails: investigate missing upstream dependencies; restore/expand groups chosen on development data, then check on held-back data.
- Self donors plus fitted gains overshoot: test background mixtures and gain regularisation/bounds, including natural-strength and per-site-gain controls.
- Pre-activation works, Top-K firing fails: investigate competition and position alignment before expanding semantic claims.
- Conditional pools collapse to self donors: improve data coverage or choose an explicitly coarser development hypothesis; record the change and use a fresh check.
- All methods fail similarly: evidence for a shared limitation of the tested representations/hypotheses, not WCM-specific inferiority.
- Only larger supports succeed: report the size/robustness trade-off.

Any method or hypothesis changed after seeing evaluation results is exploratory and requires an untouched subsequent check. A passing score supports the stated invariances; it does not certify a uniquely correct mechanism.

## End-of-day deliverable

One report distinguishing:

1. Clean WCM robustness and failure diagnostics on the completed panel.
2. A tested recursive reference implementation.
3. A real-model recursive result if the graph and donor-coverage gates pass; otherwise the exact remaining adapter requirement.
4. One justified next modification, selected from the diagnosed failure rather than a broad new sweep.

Today's goal is a reliable diagnosis and a trustworthy route to recursive evaluation. A large final-protocol comparison is subsequent work.
