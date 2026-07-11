# Next Steps (2026-07-07)

Prioritized queue agreed after the paper restructure ("What Makes a Feature Fire?").

## Priority queue (value per effort)

1. **Pre-gate vs post-gate distributions** — possibly free. The grid runner wrote
   per-run rows to `analysis/.../tables/gradient-method-neg-mode-grid.csv`. If
   rejected runs kept their eval scores, compute pre-gate φ_cf / φ_sup
   distributions per method from existing data. Addresses the acceptance-gate
   circularity flagged by both reviews; converts the Discussion caveat into a
   result. Zero GPU time if the data is there.

2. **Evaluation-hardening experiment package** — one runner, three protocols,
   for the next H100 session:
   - (a) held-out context splits: split posctx before discovery, threshold
     selection, and pruning; final scores on the held-out half;
   - (b) matched null controls: random latents matched on layer, kind, firing
     rate, decoder norm — does a matched random set achieve nontrivial
     φ_cf/φ_sup? (doubles as the generic-inhibitor test);
   - (c) dose–response and reversed-sign checks per role (scale interventions
     up/down/flip; genuine signed relationships should be ~monotonic).
   Both reviews independently said the paper lives or dies here.

3. **Top-K entry validation of the pre-activation surrogate** — the ablation
   table promised in Limitations: does pre-activation attribution predict
   actual Top-K firing under intervention, vs direct activation gradients,
   straight-through estimators, finite differences? Turns
   `SeedProjectionInstrument` into a citable contribution.

4. **Verified-inactivity negatives** — filter retrieved negctx candidates by
   actual seed activation (seq-latent index or a cheap forward pass), then
   re-run the close mode. Upgrades random-vs-close into a controlled 3-way:
   close-unverified vs close-verified vs random.

5. **Supercircuit composition with propagation validation** — stars → chains
   (`dev-notes/composed-supercircuit-analysis.md`). Do after #2 so composed
   graphs are validated under the hardened protocol.

6. **Gemma Scope replication** — once the protocol is frozen, so it runs once.

Paper trivia: `\author{}` is empty; eyeball the two Anthropic Transformer
Circuits bib author lists before submission.

---

# Paper Improvement Ideas (brainstorm, 2026-07-07)

## A. Presentation / main body

- **Worked qualitative example (biggest gap).** The paper is currently 100%
  aggregate statistics — not a single actual circuit is shown. Pick 1–3 seeds
  with clean top contexts (via search cache), show the discovered
  activators/inhibitors/supports with human-readable glosses of their top
  contexts, and the intervention trace (seed activation before/after inject and
  suppress). Mechinterp reviewers expect at least one concrete story; this is
  also the best way to demonstrate the roles taxonomy is real.
- **Concept figure (Figure 1).** Schematic of the latent-endpoint framing:
  seed latent, signed dependency neighbourhood (activators/inhibitors/supports),
  the two intervention directions (inject-on-contrast, suppress-on-positive).
  TikZ or exported diagram. Papers with a strong figure 1 communicate the
  contribution before the reader hits the method.
- **Descriptive statistics of discovered circuits.** Node-count distributions,
  role composition (activator : inhibitor : support ratios) per method,
  layer-gap histograms (seed layer − dependency layer, by role). Basic
  reviewer-expectation data currently absent.
- **Acceptance-rate table** per method × neg mode (grid summary JSON already
  has this). Median scores tell half the story; acceptance rates tell the rest.
- **Compute cost table.** Seconds/seed and peak VRAM per method (timing data
  exists in dev-notes and discovery logs). Practical adoption information.
- **φ_cf vs φ_sup scatter** per circuit, coloured by method, faceted by neg
  mode (planned in the hybrid grid notes; never rendered).

## B. New analyses — cheap (existing artifacts, no GPU)

- **Pre-gate distributions** (queue #1).
- **Method complementarity.** Jaccard overlap between CF and ablation feature
  sets per seed (`hybrid_source_overlap.py` exists; grid CSV has
  post_prune_jaccard columns). Quantifies whether hybrid fusion adds genuinely
  different nodes or re-finds the same ones.
- **Stability across negative modes.** Same seed, three modes: node-set overlap
  of discovered circuits. Robustness evidence reviewers love; if discovery is
  mode-fragile, better we find out first.
- **Seed-property predictors of success.** Correlate acceptance and φ scores
  with seed firing rate, mean activation, layer, kind (candidates.pt +
  latent_stats + grid CSV). Answers "which latents can this method explain?"
  and gives an honest scoping statement.
- **Hub-inhibitor null (partial).** Role-permutation null over stored circuits:
  is the 233-latent recurrence higher than chance given per-circuit role/layer
  counts? Pure post-processing of discovered_circuits.pt.

## C. New analyses — medium (GPU, existing pipeline)

- **Simple-baselines-same-endpoint comparison (scientifically critical).** The
  grid compares the trio only against each other. Add, on the same 128 seeds and
  φ metrics: (i) activation-ranking baseline — top-K most active upstream
  latents on posctx, no gradients; (ii) `differential_activation` (statistical
  contrast); (iii) `gradient_upstream` / `layerwise_gradient_upstream`. If
  activation ranking matches CF gradient, the gradient isn't earning its cost —
  a reviewer will ask exactly this.
- **Role-class ablations.** Evaluate circuits with activators only, inhibitors
  only, supports only, and pairs. Measures the marginal causal contribution of
  each role class — direct evidence the taxonomy carves reality.
- **Size-matched comparisons.** Truncate all methods' circuits to equal K
  before evaluation, to rule out "hybrid wins because it's bigger."
- **Minimality / K-sensitivity curves.** φ_cf vs circuit size as
  top_k_activators/inhibitors sweep — is there a knee? Supports compactness
  claims.
- **Verified-inactivity negatives** (queue #4) and the three-way close
  comparison.

## D. Appendix upgrades

- **Full 3×3 grid table**: median/IQR of both metrics, acceptance rate, mean
  node count, runtime per cell — the numbers behind Figure 1.
- **Contaminated-negative exhibits.** Show 2–3 actual close negatives that
  carry seed activity (tokens highlighted) — turns the contamination
  explanation from plausible to demonstrated.
- **Hub-inhibitor table.** Top recurring inhibitor latents with their top
  contexts, so readers can judge "generic suppressor" vs "shared machinery."
- **Co-activation structure figures** from the analysis catalogue (PMI
  histogram, profile-PCA) backing §5.3's claims.
- **Reproducibility appendix**: exact config for the grid run, hardware,
  wall-clock, seeds.
- **Auto-interp lite.** Keyword/gloss labels for circuit nodes from the search
  cache; report "fraction of accepted-circuit nodes with coherent top
  contexts." Cheap version of interpretability scoring that supports the
  "nodes are readable" premise.

## Suggested order

1. Pre-gate analysis (B) + acceptance-rate table (A) — same CSV, same session.
2. Worked example + concept figure (A) — transforms the read of the paper.
3. Simple-baseline comparison + role-class ablations (C) — next GPU window,
   alongside queue #2's hardened protocol.
4. Remaining descriptive stats and appendix exhibits opportunistically.
