
- 2026-08-09 (night): SEC 5.2/5.5 SENTENCE FISSION (readability priority 2) —
  sec:weighted: opener's triple-result sentence split (weighted pass /
  gate-only fail now separate), null paragraph split with an unbounded-score
  back-reference beside the "median ~21", host-kind sentence split into a
  claim + semicolon list, budget paragraph's six findings now one sentence
  each. sec:underdetermination: all five probe bullets split into
  measurement + verdict sentences; closing organising-claim sentence split.
  NEW note at the phi-free definition in sec:eval-abl: free/pinned are
  unclipped ratios, success = near 1, far-above-1 = overshoot. All numbers
  unchanged. Build green 35pp. Priority 2 CLOSED; next: priority 3
  (catchphrase de-duplication).

- 2026-08-09 (evening, later still): ABSTRACT LENGTH CALIBRATION — measured
  reference abstracts verbatim via arXiv API (SFC 115w/6s, ACDC 151w/10s,
  Cunningham 195w/8s, COLM-faithfulness 196w/9s, Edge Pruning 255w/12s; band
  ~115-200, median ~195). Ours cut ~290w -> ~220w, 12 -> 10 sentences.
  Dropped from the abstract (all retained in the body): the "many internal
  variables never surface" clause, position-locality of the faithfulness cost,
  the "complementary, not reducible" phrasing, and the matched-null mechanism
  detail. Further compression to SFC scale (~115w) would require dropping a
  whole result (underdetermination or the boundary) — Daniel's call. Build
  green 35pp.

- 2026-08-09 (evening, later): ABSTRACT + CONTRIBUTIONS COLD-READER REWRITE —
  abstract now glosses TuringLLM (254M, documented synthetic corpus), opens on
  a concrete output-endpoint list, and replaces coined terms with plain
  language ("indexed by amplitude semantics" -> "depends on the amplitude
  convention"; "size-calibrated families" -> "families of near-equivalent
  sets"; "position-local, depth-scaled" -> "local to token positions and grows
  with the seed's depth"). Contribution bullets 1/2/3/5 split into claim
  sentence + support sentence (4 and 6 already were). All claims unchanged.
  Build green 35pp. Readability priority 1 CLOSED (see
  readability-review-2026-08-09.md ledger); next: priority 2 (sec 5.2/5.5
  fission + unbounded-free-score note).

- 2026-08-09 (evening): EM-DASH PURGE COMPLETE — all ~135 prose em-dash sites
  in main.tex rewritten (sentence splits, comma pairs, parentheses, colons);
  zero claims or numbers changed. Remaining "---" instances are TikZ comment
  rules and the tab:coact-methods no-data cell markers only. Also removed the
  "--- (P1)"/"--- (P2)" dashes from the Fig 1 panel labels. Build green:
  35pp, zero errors / undefined refs / overfull. Readability review + change
  ledger live in paper/readability-review-2026-08-09.md.

- 2026-08-09 (later): CENTREPIECE v2 LANDED — the figure data gate is CLOSED.
  collector2.py: 3 panel-matched seeds/band (sorted-pool selection, matching
  panel-2026-08-09), tail ladder points 16384/65536, resumable ->
  curves2.jsonl (9 seeds). render2.py: median curves + ghosted seed traces +
  TEAL weighted-circuit diamonds from panel triamp400 rows on the same seeds
  (2/band where panel data exists). Legend moved below axes. Caption updated
  (v2: names both amplitude semantics — curves natural, diamonds alpha
  applied, zero fill). Build green 35pp. Remaining figure work: none.
  Remaining run gates: definitive matrix (+tri-amp arms), cross-SAE.

- 2026-08-11 (OVERLAP ANALYSIS, dev-notes only per Daniel): triamp
  membership vs attribution top-n at matched size, 21/22 seeds (22nd =
  vacuous L11-1829, cancelled in WDDM-spill crawl; aggregate locked).
  Findings: median 44% of the weighted circuit inside the attribution
  top-n (56% OUTSIDE, carrying 2.0% of |attr| mass vs 35.8% inside);
  load-bearing test by IN-BAND COUNTS (median is a bimodality trap:
  F0_inside med 0.98 but collapse-or-overshoot per seed): full 20/21 in
  band, inside-only 1/21, outside-only 0/21 — the circuit works only as
  a jointly-calibrated whole; the attribution-invisible members are
  load-bearing. Caveat recorded: halves keep joint-fitted amplitudes
  (claim scoped to the discovered solution, not all subsets).
  Paper-ready sentence in the run README; promotion decision = Daniel's
  (recommended: one sentence in sec 5.3 after "not what faithfulness is
  made of", converting it correlational -> causal). Data:
  rows_overlap.jsonl + overlap.py.

- 2026-08-11 (GE ARC COMPLETE): GE-METRIC RESULTS IN PAPER — attribution-
  mass recovery (their eval, first-order surrogate) medians: topn 0.76,
  ge_hier 0.72, triamp400 0.52 signed (abs 0.52/0.48/0.37); triamp refit
  drift 0.000 med AND max (bit-determinism exact). THE INVERSION: their
  eval ranks the arms opposite to real-model intervention — home-turf at
  the evaluation-framework level + "first-order attribution mass is not
  what faithfulness is made of" (weighted circuit reconstructs with ~1/3
  of the abs mass). New closing paragraph in sec 5.3 (surrogate caveat
  stated). NUANCE kept out of the paper deliberately: hierarchical ~=
  standard on the ungated-mass surrogate (their Fig-2b advantage showed
  on bare drive instead) — README records it; quote only if a reviewer
  asks. Build green 37pp. The Ge comparison Daniel requested is now
  complete end to end: their method under our eval (0/21, best drive) +
  every arm under their eval (inverted ranking) + full fidelity ledger.

- 2026-08-11 (latest): GE REPLICATION COMPLETE + IN PAPER — 22/22 seeds.
  Medians: n=359, F0 0.63 (bimodal 0.00-186, one seed in band), FMd 0.00
  (guarded), cf_bare 0.86 (HIGHEST truncated arm — their gating claim
  survives on control), sup 1.00; PASS 0/21 vs weighted 17/21. tab:matrix
  gains "hierarchical attr. (replication)" row; sec 5.3 prose upgraded
  from recipe-class proxy to direct replication (bimodality + drive
  nuance stated); RW pointer mentions the replication. Build green 37pp.
  Batch history 8->4->depth-aware-2 (WDDM spill; result-identical).
  GE_METRIC RUN LAUNCHED (their eval: attribution-mass recovery for
  ge_hier / topn_attr / triamp400-refit per seed) -> rows_gemetric.jsonl;
  prose on the eval-level home-turf comparison pends those numbers.

- 2026-08-11 (later): GE REPLICATION SMOKE GREEN, FULL RUN LAUNCHED —
  smoke (L2 resid 386): full pipeline end-to-end (discovery 13s at L2,
  bisection deterministic across re-runs, scoring + row written; one
  Path-precedence bug fixed in the members dump). Smoke scores at
  undersized n=28 (3-step bisection): F0 0.40 / FMd 0.00 / cf 0.21 /
  sup 0.68 — direction as expected, NOT quotable. Smoke artifacts
  cleared; full 22-seed run launched (7-step bisection, +-10% size
  tolerance, est 1.5-3h) -> rows.jsonl. WHEN IT LANDS: add
  "Ge-style hierarchical attribution @ n" row to tab:matrix, upgrade
  sec:method-comparison prose from recipe-class proxy to replication,
  add fidelity caveats (README has them + Ge's own reported results for
  the comparison prose). Ge's paper reports NO node counts and NO
  intervention eval (leaf-sum logit recovery inside the linearised
  graph) — the three comparison hooks are in the run README.

- 2026-08-11: GE REPLICATION ARM BUILT AND SMOKE-LAUNCHED (Daniel: "let's
  do an actual replication") — dev-notes/data/ge-replication-2026-08-11:
  hierarchical attribution per Ge et al. 2405.13868 adapted to our stack
  (reverse-causal site sweep with during-backward gating via per-site
  detach masks; stream rewritten decode(code)+detached-error so errors
  are detached as they prescribe; root = seed pre-activation; tau
  bisected per seed to the triamp400 size). Same 22 matrix seeds,
  held-out protocol, scoring verbatim from the panel runner. Fidelity
  caveats in the run README (no transcoder substrate; site-level
  discretisation; probe-averaged). SMOKE running; full run + tab:matrix
  row + prose when green. Method spec fetched from arXiv HTML v2.

- 2026-08-10 (night, later++): GE/CIRCUITLENS COMPARISON WIRED IN (Daniel:
  "definitely use them to compare with our methods") — framing comparison
  using existing matrix data: sec:method-comparison now states the
  truncated top-n attribution rows proxy the prior latent-endpoint recipe
  class (rank-and-cut, no optimisation, no intervention gating = Ge et
  al. / CircuitLens), instantiated with stronger estimators, failing
  0/21 at compact size vs weighted circuits' 17/21; RW latent-endpoint
  paragraph gains the forward pointer with the same numbers. NOT done
  (offered): faithful reimplementation of Ge hierarchical attribution /
  CircuitLens clustering as literal matrix arms — days of work, needs
  Daniel's call on whether a table row is worth it. Build green, now
  37pp (matrix growth + comment-pass additions).

- 2026-08-10 (night, later+): TRI-AMP NOVELTY SWEEP — no precedent found
  for the combination (or for: gains-on-kept-latents, multi-semantics
  training objective, pre-activation mask target, gate-amplitude
  coupling — each individually absent from the literature). Defensive
  citations ADDED (verified at arXiv): stoehr2024activation (EMNLP 2024
  Findings, steering scalars for outputs) + yin2026scalable
  (CircuitLasso, ICML 2026 MI workshop, feature-edge regression
  weights); contrast sentence in the optimisation RW paragraph now
  covers removed-constants / steering-multipliers / edge-weights vs our
  kept-member gains. CAUTION recorded: Li & Janson already use
  matched-budget random comparisons — keep null novelty phrased as
  fitted-amplitudes, not matched-budget (current text OK). Build green
  36pp, zero bib warnings.

- 2026-08-10 (night, later): EMPTY-CELL LITERATURE SWEEP COMPLETE — two
  independent agent sweeps (~35 queries + citation trails through Aug
  2026): NO REFUTATION of the empty-cell claim. Closest partials now
  cited and differentiated in Related Work: golimblevskaia2025circuit
  (CircuitLens, arXiv 2510.14936 — same explanation target, descriptive
  attribution, no intervention; authors VERIFIED at arXiv) added to the
  latent-endpoint paragraph alongside Dunefsky blind-case-studies clause
  and the attribution-graph nuance (their validation intervenes on
  features; their discovery is output-rooted); laptev2025feature
  (Feature Flow, arXiv 2502.03032, VERIFIED) added to the SAE-basis
  static-structure list. Bib entries appended; bibtex re-run (BIBINPUTS
  fix — bibtex must run in build/ with BIBINPUTS pointing at paper/).
  Build green 36pp, zero bib warnings. MANUAL READ still owed by Daniel:
  OpenMOSS Complete Replacement Models post (Feb 2026) — server refuses
  external connections; likely partial overlap (weight-derived global
  feature circuits), not refutation, per secondary sources.

- 2026-08-10 (night): DANIEL'S COMMENT PASS ROUND 1 — responses to
  paper/resources/User Comments.txt written point-by-point in
  paper/resources/comment-responses-2026-08-10.md. Applied (12 edits):
  abstract (filtered compression; contrastive->gradient attribution;
  differentiable clause dropped; individually-negligible glossed;
  boundary co-headlined with "zero measured control ... every seed";
  unsupervised evidence-gathering added); intro (cannot-yet-trace
  softened; superposition-as-hypothesis + polysemanticity named;
  dictionary-learning framing; recruits->excites; story->question;
  overwhelmingly->predominantly; exam->test x2; P1 sentence now
  method-neutral incl. masks; field's-own-criterion -> cited criterion;
  embarrassment/knob rewording; miller2024 citation clarified as design
  response). Build green 36pp. TWO background lit-search agents running
  on the empty-cell claim (direct keywords + adjacent literatures).
  DECISIONS PENDING (Daniel): role-vocab merge (activators/inhibitors
  with test-qualified variants — recommended, ~2h consistent pass);
  16%-in-abstract keep/drop (recommended keep); behavioural/internal
  synonym sentence; central-finding framing (boundary co-headline done;
  full restructure only if Daniel wants boundary as THE finding).
  NOTE for rerun: the paper carries two magnitude families (bisection
  minimal sets 2k-60k vs prefix-curve 10^4-10^5) — depth-stratified
  rerun should pick the headline convention.

- 2026-08-10 (later still): RUNNING EXAMPLE THREADED (readability priority
  6 — ALL REVIEW PRIORITIES NOW CLOSED). New run:
  dev-notes/data/running-example-2026-08-10 (single-seed variant of the
  panel runner; L3-resid 35381; identical held-out protocol; ~7 min).
  Results: triamp400 n=192 F0 1.12 FMd 1.27 sup 1.0 cf_amp 1.42;
  triamp100 n=250 F0 1.13 cf_amp 1.25 (budget calibration reproduces);
  gate400 n=577 F0 0.98 FMd 0.76. Budget-churn: Jaccard 0.60 between the
  two tri-amp memberships. Two sentences added: sec 5.2 (weighted-circuit
  instantiation, FMd honestly flagged as just past the band) and sec 5.5
  (two samples from the family). Sec 5.6 threading SKIPPED (needs an SFC
  replication run). Email footnote CONFIRMED correct by Daniel
  (danieljamesdavies12@ is the work email). Also noted: the seed is
  pre-activation-warm on its close negatives (a_base 4.5 vs a_pos 7.8) —
  a concrete instance of the unverified-negatives caveat, in the run
  README. Build green 36pp.

- 2026-08-10 (later): SCOPED CREF PRUNE — mapped all ~70 body \cref{sec:*}
  against section boundaries; removed exactly 9: sec-4.1 repeat ablmask
  pointers x2 (dual-floor sentence, solvers), sec-4.6 "promised in
  sec:problem" (redundant with the labelcref link) + the "sec:weighted
  returns to this" navigation pointer, sec-4.7 repeat sec:problem
  ("defined in" — the subsection opener already points there), duplicate
  (\cref{sec:eval}) in the grid-table caption, sec-5.1's back-pointer to
  eval-abl, sec-5.2's back-pointer to the just-read drivers-closure,
  sec-5.3's back-pointer to the just-read weighted section. KEPT: all
  evidence pointers, roadmap enumerations, caption self-containment refs,
  Discussion/Limitations back-refs (standalone-reader sections), and the
  deliberately-added sec:eval unbounded-score pointer. Build green 36pp.
  All readability priorities 1-5 now CLOSED; remaining: priority 6
  (running example threading + email check).

- 2026-08-10: MECHANICAL SWEEP (readability priority 5) — optimization/
  factorize -> British everywhere except the verbatim TuringLLM completion
  quote (fig:completions). "A handful." -> "A few hundred." in the sec 5.1
  opener (matches the 10^2-10^3 magnitudes; "handful of positions" kept —
  that one is accurate). Position-local now DEFINED at first use (sec 5.1:
  "no small set of positions carries it; the score accrues position by
  position across the causal prefix"). (P1)/(P2) referencing unified:
  10 appositive \cref{eq:p2(star)} -> \labelcref (renders "(P2)", hyperlink
  kept; verified via pdftotext — zero "Eq. (P2)" remain; the two
  sentence-initial \Cref uses render "Equation (P2)" and stay). Fig 1 +
  centrepiece captions split into sentences. Aggressive cref pruning
  deliberately skipped. Build green 36pp. Priority 5 CLOSED; remaining
  readability items: priority 6 (thread running example through 5.2/5.5/5.6;
  verify correspondence email danieljamesdavies12@ vs danieljdavies8@).

- 2026-08-10: REGISTER PASS (readability priority 4) — sec 4.8 dual-floor
  "Want X? Train Y." Q&A converted to declarative ("The floor is therefore a
  choice of question: ..."); the R5 Q&A beat in sec 5.1 (the centrepiece
  opener) is deliberately KEPT. "a threshold sweep is owed" -> "is required"
  (boundary); Limitations "Protocol debts" -> "Protocol gaps", "what remains
  owed there" -> "still open there are". FACTUAL FIX folded in: dropped
  "and the final cross-method matrix" from the pending-runs clause — the
  definitive matrix landed 2026-08-10, so only the depth-stratified run
  remains pending there. Build green 36pp. Priority 4 CLOSED; next:
  priority 5 (mechanical sweep: optimisation spelling, "a handful",
  position-local definition, cref/caption trims).

- 2026-08-10: CATCHPHRASE DE-DUPLICATION (readability priority 3) — the
  "two criteria the word circuit conflates" formula now appears ONCE at full
  strength (intro); abstract/Fig-1/sec-4.1/discussion instances rephrased.
  "A method wins the metric whose semantics it trains" now stated ONCE (the
  sec:method-comparison home-turf definition, six-demonstrations version);
  sec:weighted forward-references the name, discussion back-references it,
  conclusion varied ("looks best under its own training semantics").
  "Orders of magnitude" 12 -> 7, every survivor quantified (one-to-two /
  one-to-three / eight); vague uses replaced with concrete magnitudes
  (contribution bullet + sec:problem now say 10^2-10^3 vs 10^4-10^5).
  Consistency fixes after the matrix rewrite: discussion's stale "three
  separate ways" dropped; appendix "third home-turf demonstration" ->
  "another". Build green 36pp. Priority 3 CLOSED; next: priority 4
  (register pass — sec 4.6 second person, "queued/owed" voice; note the
  matrix rewrite may already have removed sec 5.3's "queued").

- 2026-08-10: DEFINITIVE MATRIX LANDED — the last method-comparison gate is
  CLOSED. dev-notes/data/matrix-2026-08-10 (runner.py + rows.jsonl + README):
  abl-ig/cf-ig/restoration PA abs-p50 on the 22 panel seeds, held-out, each
  at full size + truncated to the seed's triamp400 n; mask arms joined from
  panel rows. sec:method-comparison rewritten around tab:matrix. Headlines:
  all attribution arms 0.97-0.99 both fills at 10^5-10^6 members (unions
  coincide across methods); at matched size attribution 0/21, gate-only
  3/21, weighted circuit 17/21 (n_med 369); restoration now the CHEAPEST
  attribution arm (33s, rounds-ceiling story resolved); triamp100 cheapest
  arm overall (18s). Home-turf effect now cited as six demonstrations.
  Build green. NOTE: Daniel edited paper prose on disk (em-dash smoothing) —
  main.tex re-read before this edit. Remaining run gate: cross-SAE only.

- 2026-08-10 (late): CROSS-SAE REPLICATION LANDED — the FINAL run gate is
  CLOSED. dev-notes/data/cross-sae-2026-08-10 (crosssae.py standalone
  harness, rows.jsonl 62 rows, README with reporting rules):
  Pythia-70m-deduped + Marks et al. public ReLU dictionaries, wikitext,
  held-out 48/16, VERIFIED-inactive negatives, 6 resid seeds L2/L4, lambda
  1e-3 + 1.0 passes. VERDICT: method/compact-circuits/necessity REPLICATE
  (n=4..~2k at F0/FM 0.80-1.00, sup 0.81-1.0; single-digit fully-faithful
  circuits exist on dense SAEs); the reconstruction null is
  ARCHITECTURE-DEPENDENT (fails 15/15 at compact n; passes reconstruction
  from ~2k up — dense codes nonzero at anchors; Top-K sparsity is
  load-bearing for the null); dense empty floors off-manifold up to 1e8;
  drive eval compact-only there; lambda regime does not transfer (1.0
  needed). Paper: new "Beyond the model and the SAE family" paragraph in
  sec:weighted + Generality limitation rewritten. Build green 35pp.
  ALL RUN GATES NOW CLOSED.

- 2026-08-11: CROSS-SAE REPLICATION REDONE ON GENUINE TOP-K SAEs and written
  into the paper. Daniel's scope call: dense ReLU dictionaries are OUT OF
  DOMAIN (the pre-activation objective exists because Top-K censors), and
  masking a dense SAE to fake sparsity is bad practice — both dropped as
  claims. New: dev-notes/data/cross-sae-topk-2026-08-11 (crosssae_topk.py +
  rows.jsonl 256 rows + README) — EleutherAI/sae-pythia-70m-32k (trained
  Top-K k=16, 32,768 latents, resid/attn/mlp x 6 layers) on
  EleutherAI/pythia-70m, wikitext, held-out 48/16, verified-inactive
  negatives, 30 analysed seeds (2 excluded by a stated a_pos<1.0 rule),
  146 fitted nulls. RESULTS: triamp 22/30 ALL-PASS at n_med 51 vs
  gate-only 2/30 at 175; nulls 0/146 with max ampF0 0.09; anchor support
  0.0021 measured (vs 6.5-37% dense) — the mechanism, measured on both
  architectures. Attention drive weakness recurs (cf 0.07). sec:weighted
  "Beyond the model and the SAE family" now carries these numbers;
  Limitations scopes the method to Top-K and names the dense runs
  exploratory. Build green, zero PENDING markers. NOTE the analytic
  k/d=0.00049 is NOT the right comparison — live-pool sampling biases
  toward frequent latents; quote the measured 0.21%.
  NEXT: two-panel null figure (discovered vs size-matched nulls, Top-K
  vs dense) — proposed to Daniel, not yet built.

- 2026-08-11 (later): CIRCUIT-TRACER COMPARISON — scoped as APPENDIX material
  (Daniel's call), not a body claim and not a gate. Rationale: paper is
  complete without it; body is ~17pp against the ~12.5pp V3 target; a
  method-vs-method comparison on a third model belongs in the appendix.
  Revisit for the body only if the result is striking (e.g. their subgraph
  passes our exact-forward exams cleanly = a notable POSITIVE result for
  attribution graphs). Infra built at dev-notes/data/transcoder-compare-
  2026-08-11 (GPT-2 + Dunefsky/Chlenski PLTs: converted, validated
  end-to-end, FVU 0.003-0.27; isolated venv-ct for circuit-tracer since it
  pins transformers<=4.57.3). NOTE a real bug caught during conversion: the
  transcoders were trained under fold_ln=TRUE (pure normalisation) while
  circuit-tracer loads fold_ln=FALSE (full ln_2 output) — feeding the wrong
  one drops reconstruction from rel_err 0.41 to 0.79 and L0 from 68 to 5,
  silently. Fixed by folding the LN affine into the encoder; asserted
  numerically per layer. RECOMMENDED PIVOT (pending Daniel): EleutherAI/
  skip-transcoder-Llama-3.2-1B-131k — genuine TopK (k=32, 131k latents, 16
  layers, skip connections) AND natively supported by circuit-tracer, so it
  is in-domain for our null and needs no port. Llama-3.2-1B is gated on HF;
  unsloth/Llama-3.2-1B is an ungated mirror.
