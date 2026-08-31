# PA structure diagnostics (2026-07-19)

## Test 1 — shared closure library (`shared_library_test.py`, 9 seeds, abl-ig_mean PA)

Hypothesis: closure tails are seed-generic (a shared stream-maintenance
library); heads are seed-specific. Leave-one-out library L(-i) = tail latents
in >= 3 of the other 8 seeds' tails; skeleton = top-K by |attr|.

### Verdict (nuanced)

1. **STRONG form FAILS**: pairwise tail Jaccard ~0.07 (heads ~0.08-0.10) —
   tails are NOT mostly the same latents across seeds. (Rough chance
   baseline for 100k-size sets over ~1M slots is ~0.05, so tails sit barely
   above chance; heads at 512-2048 size are FAR above their tiny chance
   level — strong drivers repeat across seeds more than tails do.)
2. **WEAK form HOLDS and TRANSFERS**: the >=3-of-8 consensus core (~40-70k
   latents) added to a 2k skeleton recovers most mid-band closure from the
   ZERO floor: L3 0.93 (vs full 0.95), L4 0.80, L2 0.83 — leave-one-out, so
   it generalizes to unseen seeds.
3. **Library BEATS mean ablation (the user's question)**: free0(skel+lib)
   > freeM_topk(skel) at every layer L0-L6 (e.g. L3 0.93 vs 0.59, L4 0.80
   vs 0.57, L5 0.36 vs 0.02) — from the HARSHER floor. The background
   COMPUTES; a mean scaffold is not a substitute. The library is a real,
   live object, not mean ablation in disguise.
4. **Deep is seed-specific**: L7-L9, neither library nor mean-fill rescues a
   2k skeleton (both ~0). Deep closure lives in the seed's OWN tail —
   consistent with the depth-scaled irreducible core. Caveat: only ~3 seeds
   contribute deep-site tails, so the deep consensus is undersampled; a
   deep-seed-only library run is the follow-up.

### Compression implication

Mid-band circuits factor as ~2k skeleton + ONE shared ~70k library:
per-seed storage collapses (2k + pointer) when amortized over a catalogue,
though skel+lib does NOT fully reach the seed's own-circuit 0.95 — the
seed-specific tail still carries real closure (library = partial
infrastructure, ~50-90% of the skel->full gap in L2-L6).

Data: `shared_library.jsonl`, log: `run.log`. n=9, one arm, LIB_MIN=3 —
treat as directional; chance baselines estimated not computed.

## Test 2 — recipe rank / fuzzy clusters (`recipe_rank_test.py`, 15 seeds)

NMF on per-position act-grad attributions (split-channel signs, L1-normalized
rows, top-24k columns cov ~0.94), ranks {4,8,16,32} vs a random-column null.

### Verdict: STRONGLY LOW-RANK — the recipes hypothesis passes go/no-go

| rank | R2 real (med) | R2 null (med) |
|---|---|---|
| 4 | **0.749** | 0.010 |
| 8 | 0.812 | 0.018 |
| 16 | 0.837 | 0.034 |
| 32 | 0.862 | 0.062 |

1. **~4 recipes explain ~75% of per-position attribution composition** (null:
   1%); structure score (mean real-null gap) 0.78, uniform across ALL layers
   L0-L11 — depth doesn't destroy the structure.
2. **Factors are position-types, not document memorization**: sequence-usage
   entropy median 0.92 (min observed 0.54) — recipes recur across sequences.
3. **Functional (union-collapsed, phase-1 only)**: magnitude-ranking beats
   recipe-max-weight ranking at matched 8k size (shallow med 0.97 vs 0.82) —
   expected: collapsing recipes to a flat union discards exactly the
   positional structure NMF found; max-factor-weight is a crude membership
   rule. The honest functional test is phase 2's per-position soft keep
   (H*W as [T, d_sae] mask), which the statistical result now justifies
   building. Note deep seeds' capture unions (~17k, free0 med 0.75) are far
   smaller than discovery unions — capture top-128/site/pos is narrower than
   abs_pctl@90 discovery; functional numbers here are internal comparisons
   only, not comparable to discovery-run free0.
4. Factor top-50 latent dumps banked per seed in `factors/` for
   interpretation (are recipe topics nameable? -> case-study follow-up).

## Test 13 — HELD-OUT VALIDATED frontier (the honest numbers; supersedes test 12)

Test 12's frontier was scored on the SAME 20 runs the core extraction optimized
against -> winner's curse. Test 13: 60 runs/site split into two DISJOINT halves;
Louvain (gamma=4) + core extraction on C_A only; cores re-scored on C_B (unseen).
93 sites, 8 seeds.

| bar | coh_A (in) | coh_B (held) | shrink | null | cov_mass IN -> HELD | cores / med size |
|---|---|---|---|---|---|---|
| 0.5 | 0.575 | 0.554 | 0.019 | 0.027 | 69.1% -> **44.7%** | 35 / 32 |
| 0.7 | 0.734 | 0.709 | 0.022 | 0.027 | 54.3% -> **25.0%** | 34 / 22 |
| 0.9 | 0.911 | 0.889 | 0.025 | 0.030 | 32.3% -> **10.6%** | 28 / 14 |

TWO SEPARATE FINDINGS, do not conflate them:
1. **The COHESION is real.** Held-out cohesion barely drops (shrinkage only
   0.019-0.025) and sits ~26x above the size-matched random null (0.027). So
   the extracted cores are genuinely cohesive groups, NOT fitted noise — the
   member selection generalizes to unseen consensus runs.
2. **The COVERAGE was inflated ~2x.** Mass coverage at bar 0.7 falls
   54.3% -> **25.0%** (at 0.9: 32.3% -> 10.6%). The bias is not in each core's
   cohesion but in the PASS/FAIL decision at the bar: cores are extracted to
   land just above the bar, so noise pushes ~half of them back below it when
   re-scored. Classic regression-to-the-bar.

HONEST HEADLINE (supersedes test 12's 55%): **at cohesion >= 0.7, ~25% of site
mass (~30% of latents) is in validated cohesive cores** — ~34 cores/site,
median size 22. Still ~2.5x better than test 11's naive 9.4%/10.6% tight tier,
so resolution + core extraction genuinely helped; just half as much as the
in-sample number claimed.
Recommended reporting: quote HELD-OUT coverage, and note shrinkage + null.

## Tests 11-12 — COHESION is the primary metric; the cohesion-coverage FRONTIER

Test 11 (`cluster_stats_test.py`, 2065 clusters): per-cluster cohesion (mean
within-cluster co-association) vs separation. **Louvain's high Q was driven by
near-zero SEPARATION (0.007), NOT tight clusters**: median cohesion only 0.33,
67.5% of clusters <0.5, mass-weighted 0.30. Bigger/higher-mass clusters are
LOOSER (corr size -0.58, mass -0.34). Tight tier (>0.7) = 409 clusters but only
9.4% of latents / 10.6% of mass. Clusters are NOT positionally focal (median
pos-entropy 0.81; only 7.2% <0.5; median 8 of 64 positions carry 50% of mass).

Test 12 (`cohesion_frontier_test.py`, 93 sites, 8 seeds): make cohesion a
CONSTRAINT, measure coverage. Two levers — Louvain resolution gamma, and
CORE EXTRACTION (iteratively drop the weakest member until the cluster clears
the bar, giving Louvain the "reject a member" ability it lacks).

**% of site MASS in clusters meeting the cohesion bar (core-extracted):**
| bar | g=0.5 | g=1 | g=2 | **g=4** | g=8 |
|---|---|---|---|---|---|
| 0.5 | 37.6 | 49.6 | 64.2 | **69.8** | 67.6 |
| 0.7 | 31.9 | 42.6 | 52.1 | **54.8** | 49.4 |
| 0.9 | 22.3 | 29.2 | 33.8 | 33.3 | 27.6 |

RAW (no extraction) at bar 0.7 peaks at only 14.2% mass — so **core extraction
is worth ~4x** (14% -> 55%), and resolution ~4 is the sweet spot (8 overfits).

VERDICT — the frontier is GOOD, reversing test 11's pessimism:
**at cohesion >= 0.7 we cover ~55% of mass / ~55% of latents** in ~34 cores of
median size 22 per site; at >=0.9, still ~33%. The 9%/11% tight tier from test
11 was an artefact of ONE resolution + forced partitioning, not a data limit.
Operating point recommendation: **gamma=4, bar=0.7** (~34 cores/site, med size
22, 55% mass) — or gamma=2/bar=0.8 for a stricter, larger-core variant.
NEXT: name the cores at that operating point (top members' topctx).

## Tests 8-10 — CONSENSUS + PER-SITE + LOUVAIN: the junk-cluster problem is SOLVED

The junk clusters turned out to be an artefact of three fixable choices, not a
property of the data. Each fix is measured:

| | pooled + CC (t8) | per-site + CC (t9) | **per-site + LOUVAIN (t10)** |
|---|---|---|---|
| communities / site | — (model-wide) | 29 | **15** |
| largest-community frac | 0.92 (one blob) | 0.545 | **0.165** |
| residual frac / mass | .058 / .21 | .094 / .18 | **0 / 0** |
| modularity Q | — | — | **0.585** |
| frac stably homed | 0.80 | 0.674 | (see caveat) |

1. **CONSENSUS fixes restart wobble** (t8): 20-25 bootstrap+restart NMF fits ->
   co-association C[i,j] = fraction of runs i,j co-cluster. **80% of latents are
   stably homed** (strength >0.5) vs the raw single-fit hard-argmax persistence
   of 0.36. So most "junk" was NMF non-uniqueness, not real.
2. **PER-SITE fixes cross-layer conflation** (t9, user's correction): cluster
   within each (layer,kind) SAE so the structure found is purely POSITIONAL.
   Breaks the model-wide blob 0.92 -> 0.545. Discovered ~28 communities/site
   (NOT ~64 = one-per-position: positions share recipes even within a site).
3. **LOUVAIN fixes single-linkage chaining** (t10): modularity beats
   connected-components because CC welds groups via any transitive chain.
   Largest community **0.545 -> 0.165**, ~15 balanced communities/site,
   **Q = 0.585** (>0.3 = significant structure; 0.6 is strong).
   Implementation validated on planted graphs: purity 1.000 (3x30, 5x40,
   8x25); on a chained graph CC finds 1 blob (100%) vs Louvain 4 (25%).

**CAVEAT on "residual 0":** Louvain partitions EVERY node, so zero residual
means it has no noise category — not that every latent is stably homed. The
honest stability number stays the consensus strength (~0.67-0.80 homed); the
~1/3 weak-strength latents now receive a label rather than being flagged.
Also Louvain's modularity resolution limit likely merges fine structure (CC saw
29 communities, Louvain 15) — the resolution knob is a granularity choice,
consistent with "no true K".

RESULT: a per-SAE positional decomposition of the union into ~15 well-separated
communities (Q~0.6), every latent labelled, no blob. This is the interpretable
object the union lacked — descriptive, not functional (tests 3/5/6 stand: it
still can't prune or shrink the circuit). NEXT: name the communities via their
top members' topctx.

## Test 4 — cluster-EXPLAINER evals (`cluster_explainer_test.py`, 15 seeds)

Fuzzy latent clusters (NMF r=16 on positional co-occurrence) as a pure
DESCRIPTION, held to its own standards. Medians:

| metric | value | read |
|---|---|---|
| AUC cluster / marginal / shuffled | 0.768 / 0.732 / 0.497 | clusters beat popularity by only **+0.036** |
| top-cluster persistence (data halves) | **0.359** | latents MOVE (chance 0.0625, but 64% switch) |
| top-cluster persistence (NMF restarts) | 0.436 | half the instability is NMF non-uniqueness |
| membership entropy median / 1-cluster frac | 0.29 / 0.86 | confident memberships... that rotate |
| **within-seed capture Jaccard** | **0.909** | **Test-1 specificity CONTROL PASSES** (vs 0.07 cross-seed) |

Verdict: as currently fit, the explainer FAILS its stability bar and its
predictive lift is small — clusters are real (Test 2) but too soft to narrate.
Fixable-in-principle parts: consensus/multi-restart fitting (restart
instability), more data per fit (32 seqs/half). The durable positive is the
CONTROL: within-seed 0.91 vs cross-seed 0.07 grounds the specificity claim
properly. Stable-cluster member dumps in `clusters/` (3 best-matched per seed).

## Test 3 — DECISIVE soft-mask functional eval (`soft_mask_test.py`, 6 informative seeds)

Per-position free0 (custom multiplicative patcher reproducing production
free0), three keep rules swept over size:

**Matched UNION size (distinct latents = the reported circuit size):**
| union | union-mag | recipe | perpos-mag |
|---|---|---|---|
| 1k | **0.259** | 0.198 | 0.172 |
| 4k | **0.513** | 0.287 | 0.222 |
| 8k | **0.582** | 0.392 | 0.275 |
| 16k | **0.719** | 0.532 | 0.373 |

**Matched EFF size (avg latents kept / position = per-position compute):**
| eff | perpos-mag | recipe | union-mag |
|---|---|---|---|
| 250 | **0.588** | 0.338 | 0.259 |
| 1000 | **0.588** | 0.514 | 0.259 |
| 4000 | 0.588 | **0.626** | 0.513 |

### VERDICT: the recipe-as-closure-circuit line is a NO — union is closure-optimal

1. **On the reported size axis (union membership), plain magnitude union
   DOMINATES** both per-position schemes at every size. Position-awareness
   does NOT yield a smaller closure circuit.
2. **Recipe is Pareto-dominated on BOTH axes**: beaten by union-mag on
   membership, beaten by raw perpos-mag on per-position compute. The low-rank
   structure that is statistically real (Test 2) does NOT translate to a
   functional closure advantage — it wins on neither axis.
3. **Why, and it's the real finding: closure is a MEMBERSHIP property, not a
   positional one.** free0 lets kept latents re-encode live and the model
   resolves per-position values itself — so the "allow, don't force" union
   already captures position-awareness for free, and explicitly gating
   positions (perpos/recipe) only removes reconstruction capacity. **The
   union is not lazy; it is the closure-optimal object.**
4. Caveats: act-grad capture signal (proxy), top-128/site capture
   undersamples deep seeds (L6-L9 all ~0, excluded); shallow/mid result is
   robust across the 6 informative seeds and the mechanism is principled.

### What survives (the union, vindicated three ways)

- Test 1: union finds SEED-SPECIFIC latents (tails ~disjoint) + a real shared
  mid-band library that BEATS mean ablation.
- Test 2: the union has real internal low-rank recipe structure (descriptive
  / interpretability value — nameable "topics of computation").
- Test 3: the union is CLOSURE-OPTIMAL — no positional restructuring beats it.

Together: the big union is specific, structured, AND principled — the honest
object for closure, not a lazy shortcut. The structured-PA program does NOT
replace it; it CHARACTERIZES and DEFENDS it. Recipes stay as an
interpretability decomposition, not a circuit-shrinking method.

---
(historical below — superseded framing)

Combined with Test 1: the structured-PA picture is now empirically live —
fuzzy recipes (low-rank, cross-sequence) for composition + shared library
for mid-band infrastructure + seed-specific residue. Next: (a) per-position
soft-mask eval (the real functional test), (b) cross-seed factor matching
(shared recipe TYPES?), (c) name the factors via topctx decode.
