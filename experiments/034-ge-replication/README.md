# Ge et al. hierarchical-attribution replication (2026-08-11)

Replication of the discovery method of Ge et al. 2024 (arXiv:2405.13868)
as a first-class comparison arm, rooted at internal seed latents, run on
the SAME 22 matrix seeds / held-out protocol / scoring code as
tab:matrix (029-panel + 031-matrix), so it lands as a
matrix row. Requested by Daniel 2026-08-11 ("actual replication ... real
results to compare").

## Their method (from the paper, arXiv:2405.13868)

- attr(v,t) = a_v * d a_t / d a_v (activation x gradient to target).
- HIERARCHICAL attribution: during the backward pass, gradient flow is
  STOPPED at any node whose attribution falls below threshold tau, so
  upstream scores only flow through surviving nodes. This is their key
  distinctive vs standard post-hoc thresholding.
- SAE reconstruction errors are detached ("cannot be interpreted").
- Local circuits: root = an intermediate SAE feature's activation, on
  prompts where the feature fires.
- tau is swept for sparsity; evaluation is leaf-sum logit recovery
  (Theorem 3.1) — NO intervention validation.
- Substrate: transcoders + frozen attention patterns make the graph
  linear (we do not replicate the substrate — see adaptations).

## Our adaptations (each is a fidelity caveat for the paper)

1. SAE-site granularity: sites processed in reverse causal order
   (downstream -> upstream); each site's gradient is taken with
   already-processed sites' below-tau latents detached — the site-level
   discretisation of their during-backward gating.
2. Stream rewritten at every upstream site as decode(code) + detached
   error: gradients flow only through SAE codes; errors detached exactly
   as they prescribe. (No transcoders on our stack; gradients pass
   through the model's own nonlinearities between sites, where theirs
   pass through a linearised graph.)
3. Root = seed encoder pre-activation at probe anchors (their target is
   feature activation; ours must survive Top-K censoring).
4. Attribution aggregated over the seed's 48 train probes (theirs is
   per-prompt).
5. Gate on |attr| >= tau (keeps negative contributors; their bracket
   case study retains negative contributions).
6. tau bisected per seed (log-space, 7 steps, +-10% tolerance) to land
   membership at the seed's triamp400 size — the matched-size
   comparison. tau=0 reference (all-active membership) logged per seed.
7. No optimisation, no intervention during discovery (faithful to
   them); evaluation is OUR held-out protocol (free0/freeM/cf/sup) —
   that is the comparison point.

## Files

- runner.py — the arm (resumable rows.jsonl keyed (comp_idx, latent,
  arm); SMOKE=1 = 1 seed, 3 bisection steps). Scoring functions verbatim
  from 029-panel/runner.py (scoring_ref.py is the unmodified
  copy for diffing).
- members_ge_hier_<comp>_<latent>.json — discovered memberships.
- smoke.log / runner.log — runs.

## ge_metric.py — their eval, applied to every arm (Daniel's request)

Attribution-mass recovery: with attr = act x grad(seed preact) at the
clean linearisation (their standard-attribution frame, errors detached),
recovery_signed / recovery_abs = kept over total attribution mass. This
is the first-order analogue of their Theorem 3.1 leaf-sum logit
recovery (exact in their linearised graph, surrogate on our stack —
state this wherever quoted). Scored memberships per seed: ge_hier (the
replication), topn_attr (standard attribution at matched size — Ge's
own comparison arm), triamp400 (weighted circuit, membership recovered
by deterministic refit; refit n logged vs panel n as a drift check).
One ungated forward+backward per chunk scores all sites at once.
Run AFTER the replication run frees the GPU -> rows_gemetric.jsonl.
Expected story: high attribution-mass recovery can coexist with 0.00
real-model ablation faithfulness — the analytic-vs-interventional gap
made quantitative (home-turf at the evaluation level).

## RESULTS (full 22-seed run, 2026-08-11)

Medians over 22 seeds (all held-out): n=359 (targets landed 0.86-1.14x),
F0=0.63 (bimodal dead-or-overshoot: per-seed 0.00-186, only ONE seed in
band), FMd=0.00 (guard (-1,3): 20 scorable; excluded 3.22 and -7.18),
cf_bare=0.86, sup=1.00. Discovery median 142s/seed.
**PASS (both fills in band + sup>0.9): 0/21** vs weighted circuit 17/21.
KEY NUANCE: cf_bare 0.86 median is the HIGHEST of the truncated arms
(matrix top-n rows: 0.56-0.66) — Ge's hierarchical-gating claim survives
transfer to real-model intervention on the CONTROL axis; it fails
entirely on faithfulness. Paper updated: tab:matrix row + sec 5.3 prose
+ RW pointer. Batch history: DISC_BS 8 -> 4 (WDDM spill) -> depth-aware
2 for >=25-site seeds (L11 spilled at 4); result-identical (uniform
chunk rescale; tau re-bisected).

## GE-METRIC RESULTS (rows_gemetric.jsonl, 22 seeds x 3 memberships)

Attribution-mass recovery medians (signed / abs):
ge_hier 0.72 / 0.48; topn_attr 0.76 / 0.52; triamp400 0.52 / 0.37.
Triamp refit drift vs panel sizes: 0.000 median AND max — bit-determinism
held exactly. THE DISSOCIATION: under their eval the rank-and-cut arms
lead and the weighted circuit trails; under real-model intervention the
weighted circuit passes 17/21 and they pass 0/21. Home-turf at the
evaluation-framework level. Bonus finding: the weighted circuit holds
only ~1/3 of absolute attribution mass while reconstructing the seed —
first-order attribution mass is not what faithfulness is made of.
NUANCE vs their Fig 2b: on our ungated-mass surrogate, hierarchical ~=
standard attribution (0.72 vs 0.76) — their hierarchical advantage did
NOT show on this surrogate (it showed on bare drive instead); do not
over-claim in either direction (surrogate != their exact leaf-sum in
the pruned linearised graph). Paper: closing paragraph of sec 5.3.

## OVERLAP ANALYSIS (rows_overlap.jsonl; FINAL at 21/22 seeds — the
22nd, L11-resid-1829 (the FMd-vacuous seed), was cancelled mid-refit
after WDDM-spill crawl with the aggregate already locked; NOT in the
paper per Daniel)

Question: is the weighted circuit's advantage carried by members
attribution would never have picked?

1. **A majority of the weighted circuit sits outside the attribution
   top-n at matched size**: median 44% inside (range 13-59%; no strong
   depth trend, shallow 0.42 vs deep 0.49 median).
2. **The outside members are attribution-invisible**: they hold 2.0%
   median share of total |attr| mass (inside members hold 35.8%), with
   median per-member rank around the top 0.5% of ~300k ranked latents —
   i.e., far below the top-n cut in absolute terms.
3. **The load-bearing test — read IN-BAND COUNTS, not medians**:
   F0 in [0.8,1.25]: full 20/21, inside-only 1/21, outside-only 0/21.
   (Median F0_inside 0.98 is a bimodality artifact: per-seed values are
   collapse-or-overshoot; the median accidentally lands near 1.)
   NEITHER half reconstructs alone in 20/21 cases: the circuit works
   only as a jointly-calibrated whole.
4. **Interpretation caveat**: subsets keep their jointly-fitted
   amplitudes; a re-fitted subset could in principle do better. The
   correct claim is "the discovered solution is jointly calibrated and
   its low-attribution members are load-bearing within it", NOT "no
   compact subset could ever work" (the matched-size gate-only and
   truncation rows already bound that separately).

Paper-ready sentence if promoted: "A median 56% of the weighted
circuit's members lie outside the attribution top-n and carry 2% of
its mass, yet removing them breaks held-out reconstruction on 20 of 21
seeds: the members attribution cannot see are load-bearing."

## Status

- 2026-08-11: smoke launched (1 seed). Full 22-seed run pending smoke.
  Estimated full-run cost: per bisection step, one forward+backward per
  upstream site over 48 probes at bs=8; deep seeds ~35 sites -> minutes
  per step, ~2-5h total.

## Their reported results (for the comparison prose; arXiv:2405.13868 v2)

- Model: GPT-2 Small only (12L, 768d). Dictionaries: SAEs on 12 attn
  outputs + 24 resid points, TRANSCODERS per MLP; 24,576 features
  (32x768), ReLU/L1 (8e-5), 1B OpenWebText tokens, L0 ~20-66, var
  explained 69-99%.
- Case studies: bracket (FEATURE-rooted — the cell-adjacent result;
  per-token contribution percentages 104.1/102.6/314.2%, 83.8% through
  L1A.H1), induction (lead feature 35.0%, top-7 aux +33.0%), IOI
  (sJohn/sMary use disjoint SAE features on shared heads).
- NO NODE COUNTS reported anywhere. NO intervention: faithfulness =
  Theorem 3.1 leaf-sum logit recovery, computed analytically INSIDE the
  linearised substitute graph (framed as avoiding backup-behaviour
  confounds). One quantitative exp: hierarchical vs standard
  attribution, 20 IOI samples x 30 sparsity thresholds, curves only, no
  deltas. No comparison to ACDC/patching/other methods. Stated
  limitation: input-specific circuits.
- Comparison hooks for our row: (1) we supply the size accounting they
  never did; (2) their faithfulness is analytic-in-the-linearisation,
  ours intervenes on the real model — running their discovery under our
  evaluation is the test their protocol structurally cannot perform;
  (3) per-prompt vs our reusable membership. Fairness note: they showed
  hierarchical > standard attribution on THEIR metric; if our ge_hier
  row beats plain truncated attribution on ours too, say so.

## Paper hooks (when results land)

- tab:matrix row "Ge-style hierarchical attribution @ n" + prose in
  sec:method-comparison replacing/alongside the recipe-class sentence.
- RW latent-endpoint paragraph: "replicated as a first-class arm".
- Fidelity caveats above stated wherever the row is quoted.
