# Relativity concept circuits on TuringLLM (2026-08-27)

Daniel's question: can we read KNOWLEDGE out of causal circuits, where a
latent stands for a concept? Moved from Gemma to TuringLLM because
Gemma's auto-interp labels proved unreliable (3 of 8 fatally wrong, see
`../041-factual-seeds/README.md`) and because here we own the
training data and can decode it.

## Method: no auto-interp anywhere

1. `find_seqs.py` -- locate the `relativ` stem (and Einstein/gravity/
   spacetime/quantum) in TuringLLM's own training shards. Sequences are
   -1-delimited segments with the first token skipped, exactly as
   `DataLoader._build_shard_index` does; reshaping the raw stream into
   fixed blocks misaligns everything and yields out-of-vocab ids.
   9,223 `relativ` windows from 300 shards.
2. `find_seeds.py` -- run the model over those windows and read EVERY
   latent at EVERY (layer, kind) site AT the anchor token. Two controls:
   * PHYSICS vs DISTRACTOR (the stem also appears in "linguistic
     relativity" / "relatively"), and
   * SPECIFICITY (anchor vs all other positions in the same window).
   Accumulated with `index_add_` on dense [n_comp, d_sae] buffers -- the
   obvious per-element Python loop leaves the GPU at 8% util.
3. `runner.py` -- the 2026-08-09 panel runner with ONE change: seeds
   come from step 2 instead of the discovery candidate pool. Arms,
   triple floor, held-out split, nulls and scoring are untouched, so
   these circuits are comparable to the panel's. Also patched to DUMP
   MEMBERS (`members.jsonl`); the panel runner only ever wrote summary
   rows, which makes a concept circuit unreadable.
   Circuits are fitted on the SEED's own top_ctx/mid_ctx sequences at
   the seed's argmax position (ProbeDatasetBuilder), i.e. the standard
   probe dataset.
4. `read_circuit.py` -- describe each member by ITS OWN stored top_ctx
   (64 sequences per latent, all 36x40960 of them). An earlier version
   sampled random training text instead, which is strictly worse: a
   relativity-specific member is rare in random text so its "top
   contexts" came back as noise, while generic high-frequency members
   looked confident.

## Result 1: the biggest activations are string detectors

Top latents by raw activation on the `relativ` token fire just as hard
on LINGUISTIC relativity (phys/dist 1.2-1.5) -- they detect the string.
The physics-selective latents rank lower but separate 2.7-4.0x. Without
the distractor control we would have circuited a spelling pattern.

## Result 2: two latents group concepts across word forms

* comp 35 / L11 resid / 13633 fires on `relativ` AND `gravit`
  ("detect gravitational fields", "distortions in spacetime,
  manifesting as gravitational frames").
* comp 26 / L8 resid / 455 fires on `relativ` AND `Doppler`
  ("detect stellar motion through Doppler shift").
These are concept groupings, not string matches.

## Result 3: circuits validate, and one reads as knowledge

| seed | n | zero-fill | mean-fill | necessity |
|---|---|---|---|---|
| c35/13633 relativity+gravitation | 355 | 0.851 | 1.218 | 1.000 |
| c23/12639 relativity L7 | 496 | 0.964 | 1.238 | 1.000 |
| c29/4523 relativity L9 #2 | 562 | 1.032 | 1.068 | 1.000 |
| c26/455 relativity+Doppler | 665 | 1.028 | *1.340* | 1.000 |
| c29/3736 relativity-theory | 527 | 1.023 | *1.356* | 1.000 |
| c32/8627 relativity L10 | 494 | 0.915 | *1.349* | 1.004 |

3 of 6 pass both bands; ALL have necessity 1.000 with dead nulls. The
three misses are all the same failure -- mean-fill overshoot ~1.35 --
consistently in one direction, which looks like a floor effect at these
high-activation seeds (a_pos ~23) rather than noise. NOT yet diagnosed.

DETERMINISM: the run was executed twice (once before the member-dump
patch, once after); 37/37 shared arms reproduced bit-identically in n
and faithfulness.

**c29/3736** (seed's own contexts are about quantum-gravity
unification: "bridging the gap between quantum mechanics and general
relativity") decomposes, in its top 12 of 527 members, as:
* COSMOLOGY (5): early-universe rapid expansion (a 4.46); CMB thermal
  radiation / initial expansion from a hot dense state (4.41);
  dark-matter gravitational scaffolding + Big Bang structure (4.28);
  dark-matter candidates beyond neutrinos (3.88); Higgs-boson coupling
  (3.31).
* MATHEMATICAL MACHINERY (3): groups/rings/fields (3.94); partial and
  ordinary differential equations (3.55); continuity, topological and
  metric characterizations (3.35).
* UNINTERPRETABLE (4): precious metals/currency, "Collection:
  Precipitated water", moraines/drumlins (glacial landforms), "Header
  Fields".

i.e. relativity = {the observational domain where GR applies} +
{the formal apparatus of GR}. That is a readable, checkable claim.

## Caveats

* Top 12 of 527 members read; 515 unread. Do not claim the whole
  circuit is interpretable.
* 4 of 12 shown members resist interpretation. Circuits are SUFFICIENT
  not unique ([[l2-crossover-universal-core]]), so an uninterpretable
  member is not evidence of hidden meaning.
* One seed. The cosmology/maths split is an observation, not a result,
  until replicated across the other validated seeds.
* Circuits are corpus-conditioned: fitted on the seed's top-64
  contexts, so the decomposition describes the concept AS USED THERE.

## Edge audit (2026-08-27): from vibes to named causal edges

`edge_audit.py` -- per-member causal weights inside a fitted circuit
(the mask is the SEARCH over 1.5M latents; this is the VERIFICATION of
individual member->seed edges). Necessity = solo knockout, sufficiency
= solo restore, synergy = double-knockout superadditivity. MEAN-FILL
frame through the canonical evaluator (`circuit_only_activation`) with
a new `keep_scales` per-latent amplitude argument; held-out probes.

**METHOD FINDING -- the zero-fill e0 is a censored explosion.** With
every upstream site zero-filled, the seed PRE-activation reads ~1.6e6
(two independent patcher implementations agree); the historical
e0 ~ 0.000 is that out-of-distribution explosion censored by the
post-top-k read. Published F0 numbers stay internally consistent
(numerator, denominator, and nulls share the frame), but "zero-fill =
silent model" is the wrong gloss; mean-fill is the in-distribution
frame and the audit uses it. Needs a methods-appendix paragraph and a
check on the cross-arch arenas.

**RESULT: causal weight is concentrated and READABLE (boson seed).**
c35/13633: top-10 members carry 71% of total |necessity|; only 20/355
members have |nec|>0.01. The named skeleton (contexts from top_ctx):

| nec | site | fires on |
|---|---|---|
| 0.55 | L8 resid 33903 | "the heaviest known elementary particle" (top quark), electroweak interactions |
| 0.53 | L9 resid 1845 | Big Bang / cosmic evolution |
| 0.37 | L10 resid 35170 | stellar mass / nuclear fusion |
| 0.28 | L10 resid 28127 | momentum / collisions |
| 0.23 | L7 resid 19202 | dark matter candidates |
| 0.21 | L10 resid 12332 | "symmetry breaking ... gives particles mass", gauge symmetry, quarks and leptons |

boson <- {top quark, electroweak symmetry breaking, Big Bang, fusion,
collisions, dark matter}: physically correct particle-physics
composition (top quark <-> Higgs coupling is a real physics link), read
off causal weights with no labels anywhere.

**RESULT: the relativity seed is a refinement CHAIN of relativity
latents.** c29/3736 (flat distribution: top-10 = 26%, 160/527 members
weighted): top edges are L6/23753 "relativistic mass scales with
velocity" (0.29), L3/18699 "Einstein's theory of relativity ...
Minkowski / simultaneity" (0.28), and L8/455 -- which is ITSELF one of
our audited seeds (relativity+Doppler): a seed->seed edge, the first
certified link of the concept graph. Daniel's "Einstein -> relativity"
is literally present: an L3 Einstein's-theory latent causally drives
the L9 relativity latent with nec 0.28.

**Synergies are predominantly NEGATIVE (redundancy), sufficiency ~0
everywhere:** members carry overlapping signal and no single edge
reconstructs the seed -- evidence AGGREGATION, not conjunctive binding,
for these two seeds. No strong fact-shaped superadditive pair yet
(best: +0.13 c35). Contrast heavy-tailed boson vs flat relativity:
concept latents differ in how concentrated their causal support is.

**ANOMALY, flagged not hidden:** L0 attn 18179 has top-5 weight in
BOTH circuits but its contexts are junk ("adventure tourism",
"heroes"). A cross-circuit hub with no semantic identity -- candidate
mean-fill artifact or genuine stabilizer role; must be nulled before
any graph claim uses it.

## The relativity concept-formation ladder (2026-08-27, chain audits)

Two more circuits fitted for the chain links (both PASS both bands;
their nulls fail catastrophically -- zero-fill F0 27-78 with FMd=0 and
sup=0, the OOD instability with flipped sign at shallow depth; the
JOINT gate is what discriminates):

  c11/18699 "Einstein's theory of relativity" (L3): n=180,
      F0 1.028 / FMd 1.103 / nec 1.000
  c20/23753 "relativistic mass/corrections" (L6): n=370,
      F0 0.950 / FMd 1.050 / nec 1.000

Edge audits on both close a FOUR-LEVEL certified ladder, every node
context-verified, every edge a held-out knockout measurement:

  L1/17106  "relativistic-" (token stem)
     |  nec 0.170
  L3/18699  "Einstein's theory of relativity" (phrase)
     |  nec 0.127            \  direct skip-edge nec 0.283
  L6/23753  "relativistic corrections/mass"    \
     |  nec 0.288 <- also fed by L4/2777        |
     |     "special relativity" (nec 0.288)     |
  L9/3736   relativity-theory / quantum gravity <-
     ^ also fed by L8/455 relativity+Doppler (nec 0.251),
       itself a validated seed.

Reading: the model REFINES a relativity representation layer by layer
-- token stem -> Einstein-phrase -> special-relativity/corrections ->
deep theory concept -- with skip connections. Concept formation as a
causal ladder, read without labels or logits.

**Shared infrastructure, replicating [[l2-crossover-universal-core]]:**
L0-attn latents 18179 / 9654 / 5663 and L0-mlp 14697 ("Electric...",
physics-units) recur with top weight in EVERY audited circuit.
18179's contexts are semantically junk ("adventure tourism") -- these
are arena-level infrastructure (the universal-core phenomenon), NOT
concept content, and any concept reading must set them aside via
cross-circuit frequency. c20's circuit is FLAT (top-10 = 15%) --
concentration varies by seed here too.

## Knowledge-circuit losses (echo penalty + neg-suppress), first results

Engine: generic `member_penalty` vector (here corr(a_i, a_seed)^2 from
streaming train-probe stats) and a `neg_suppress_weight` data term
(reproduce the seed's natural silence on the stored negctx negatives),
composable with the pos objective and triple floor. v1 neg term was
un-normalised and REWARDED DENSITY (18k members, 40-step smoke); fixed
by dividing by dual_norm like the primary term. Runner: know_runner.py
(arms know400 / echoamp400 / negamp400 -> know_rows/know_members).

All 6 circuits (2 seeds x 3 arms) PASS both bands, sup ~1. Effects on
c29/3736: echo arm dropped 261/527 members (105 new) yet kept every
audited top-necessity edge; known stem/relativity copies 9/9 -> 6/9
(incl. dropping an exact duplicate pair member); token-machinery 3% ->
2%. COSTS: cf_amp runs hot on echo arms (~2.2-2.5); the dropped copies
include 1/mlp/36431, whose contexts carried "Einstein's theory of
relativity in 1905" -- dedup can evict knowledge-adjacent members.
New members trend toward LOW-PEAKINESS academic-discourse accumulators
(field enumerations, empirical-methods prose), not obviously physics
facts. Junk survives at top alpha (Dr.-name-list 5/40837) -- the echo
penalty only prices seed-correlates, not semantic junk.

Analysis tooling per Daniel's spec: `analyse_members.py` renders every
member's OWN top contexts as FULL per-token activation profiles
([[tok=v]] >= 50% peak, [tok=v] >= 15%) + peakiness stat; open
classification after discovery, no member list sought. Profile view is
a decisive upgrade: accumulators vs token detectors are instantly
distinguishable, and secondary firing tokens carry the interpretation.

## Knowledge-loss characterisation (sweep + audits + shallow seed)

ECHO_W frontier on c29/3736 (know400): 5e-3 -> F0 1.02/sup 1.00/6 of 9
stem copies; 2e-2 -> F0 0.87/sup 0.81/1 of 9; 5e-2 -> F0 0.71/sup 0.71.
Hard pressure completes the dedup but degrades NECESSITY (replacements
are substitutable filler); 5e-2 leaves the band. Usable regime: mild.

Edge audits of know400 vs production:
* c35 boson (control): causal skeleton intact -- same physics edges
  (top quark 0.465, Big Bang 0.447, collisions, Higgs, fusion), same
  ordering. No harm on a knowledge-rich seed.
* c29 relativity: freed weight flowed to JUNK -- infrastructure hub
  0/attn/18179 doubled to 0.46 (now #1 edge), Dr-name-list 40837 at
  0.246. Mechanism: junk is UNCORRELATED with the seed, so the echo
  penalty makes it relatively cheaper. The penalty's blind spot.
* c11 Einstein-phrase (shallow): penalty drops the SEMANTICALLY
  CLOSEST members (GPS relativities, spacetime curvature) and keeps L0
  stem duplicates -- at the concept floor, correlation ~ semantic
  proximity. Echo penalty is a DEEP-SEED tool.

NEXT REFINEMENT (designed, not run): price membership by echo-corr^2
PLUS cross-circuit recurrence (18179/9654/5663 + 0/mlp/14697 recur
top-weight in every audited circuit regardless of topic -- the
signature of arena infrastructure, [[l2-crossover-universal-core]]).
Closes both leaks with data we already collect.

## Three new extraction methods (module mining, forward injection,
## differential seeds) -- results

MODULE MINING (`module_mining.py`, CPU over the audit jsonls): a
cross-domain PARTICLE-PHYSICS MODULE serves both boson and relativity
circuits (top quark total 1.12, dark matter 0.76, Higgs 0.40); true
infrastructure (cross-family span) reduces to 18179/34495/13899/35565.
v2 fixes needed: merge know-arm audits as replicates; require
cross-family span for the infrastructure label (4 of 6 audits are
relativity-family, which inflates it).

FORWARD INJECTION (`inject_profile.py`): inject a latent's decoder
direction at natural peak (a_pos_ho -- NOT ctx_seq_val, which stores
sequence means and under-injects ~25x) at one position in neutral
windows; read downstream latent deltas vs matched-norm direction
nulls. Relativity seed: 180 significant directed edges incl. THE BOSON
SEED (35/13633, delta 0.19 vs null 0.00) -- the {boson, relativity}
module confirmed in the forward direction. Einstein-phrase seed: 89
edges including the first OBSERVED INHIBITORY forward edges (L6-L8
deltas to -0.08) and forward drive of 4/3263 + 4/27056, both flagged
independently by module mining. Boson seed: 0 edges is an ARTIFACT --
L11 resid is the bank's last site; nothing downstream to read.

DIFFERENTIAL SEEDS (`diff_runner.py` + seed_vector support in
learned_mask and circuit_only_activation): fit against the VIRTUAL
direction w_A - w_B; shared composition cancels by construction.
c29: 3736-4523 -> n=561, EF=1.088, nulls dead. TOP-10 MEMBERS ALL
NOVEL (in neither plain circuit; 263/561 in neither): the difference
recruited machinery invisible to both plain fits, concentrated at
L8-L9. Named: Higgs-discovery-at-CERN contexts, DARK MATTER HISTORY
(1930s, Fritz Zwicky, Coma Cluster -- the most fact-dense latent seen
yet), event-horizon/quantum-gravity-reconciliation (a plausible true
differentia of 3736). ~half the novel top members are off-topic
accumulators -- non-uniqueness still applies; read, don't trust.
REVERSE direction (4523-3736 on 4523's probes) has a DEGENERATE frame
(natural 0.77, mean-fill 12.4, denom negative): the mirror question
needs a better baseline design before its EF means anything.

SYNTHESIS: three independent methods agree on the same structures
(particle-physics module: backward circuits + mining + forward
injection; 3263/27056: mining + injection). Cross-method agreement is
the evidence standard for the knowledge-graph programme.

## triamp400 vs negamp400 on 5 diverse L9 seeds (arm generalisation)

Seeds: agile-workflow 38310, quantum-measurement 17115, petroleum-eng
40651, medical-imaging 24084, music-brain 23920. Both arms + 1 null
each; profile analyses for both arms per seed.

GATES: 4/5 pass for EACH arm, failing on opposite seeds -- negamp
rescues med-imaging (triamp FMd 1.31 -> 1.05) but breaks music-brain
(1.05 -> 1.27). |FMd-1| favours negamp 3/5, tie 1, triamp 1. The neg
term is a REGULARISER TOWARD THE NEGATIVES' BEHAVIOUR: it helps where
triamp overshoots mean-fill and can overshoot where triamp was calm.

EXPLAINABILITY: no aggregate advantage on diverse seeds -- top-15
peakiness medians identical (0.03-0.05 both arms; music-brain negamp
0.10), token-machinery equal. MEMBERSHIP SHIFT is consistent though:
negamp recruits DEEP ATTENTION blocks (agile: 13 of top-15 exclusive
members are L6-L9 attn; c29 earlier: L5-L9 additions) and sheds
shallow L0-L2 latents. The c29 knowledge-density gain (Feynman, GR-
limits members) does NOT clearly replicate at aggregate level on
non-knowledge seeds.

VERDICT for the big run: knowledge-density is SEED-driven more than
ARM-driven -- supports the many-seeds + post-hoc-classification plan.
Config: triamp400 production + negamp400 as the cheap side arm (same
probes; use as FMd rescue where triamp overshoots, and for its deep-
attn recruitment on knowledge-family seeds). Neither arm's edges
audited on the new seeds yet.

## Completion flow: which latents cause a fact completion?
(`relativity_completions.py`: 29/32 cloze probes correct -- the model
HAS the facts; `completion_flow.py`: knock out each active latent,
measure the answer token's logp drop.)

THREE DIFFERENT CAUSAL ARCHITECTURES:
* "...in the year" -> 1905: a CONCENTRATED LAYER LADDER, L4->L9 resid,
  top knockout dlogp 2.2. Named via top_ctx: EVERY link is a
  "[[year]]"-slot latent (fires on the token 'year' in year-in-history
  prose), several contextualised to physics history (L6/23868 fires on
  Einstein's Annus Mirabilis contexts; L7/32639 and L9/11019 on
  gravitational-lensing / Bose-Einstein year sentences). The chain is
  RELATION machinery (a year is coming); the specific 1905 binding is
  not in any single latent we knocked out.
* "Albert" -> Einstein: OVERDETERMINED -- 192 candidates, max dlogp
  0.104 on a 99%-confident answer. Surface-bigram-like, robust to any
  single-latent knockout.
* "curvature of" -> spacetime: mixed; top single cause is L0-attn 5663
  (the INFRASTRUCTURE stabiliser, again) and the TOP-QUARK module
  latent 33903 contributes (dlogp 0.065) -- a module member causally
  supports a completion.

DISSOCIATION, important: our validated RECOGNITION circuits
(Einstein-phrase 18699, relativity chain) do NOT appear in the
production pathway for "1905". Recognition circuits and production
pathways are different objects in this model -- consistent with the
subject/relation/attribute decomposition in the fact-recall
literature, and measured here at latent level with knockouts.

## The year ladder as circuits: a perfect relay chain

All six ladder latents fitted as seeds (triamp400): 6/6 PASS both
bands (F0 0.87-1.06, FMd 0.98-1.18, sup 1.000 everywhere).

CHAIN TEST -- a perfect upper-triangular matrix: every rung's circuit
contains EVERY lower rung as a member (L9's circuit holds L4-L8; L8
holds L4-L7; L7 holds L4-L6; L6 holds L4; L4/L5 hold none), with alpha
rising monotonically toward the nearest rung (L9: a 1.64 -> 2.05 from
L4 up to L8). The production pathway for "...in the year -> 1905" is
thereby RECONSTRUCTED from seed-rooted circuits: the relation
machinery is a layer-to-layer RELAY, each rung causally constituted by
all rungs below plus fresh evidence. L4 and L5 are independent bases.

Non-ladder membership (profiles): numeral/date-format machinery
("year [[2017]] was a pivotal year" at L1; "[[1]]9[[3]]0s and '40s"
decade attention at L7; digit-quantity latents at L4mlp) plus the
usual infrastructure cast -- 18179 and 40837 recur AGAIN (now in
year circuits), confirming cross-family infrastructure status. The
physics-history flavour of the rungs' own top contexts is NOT
strongly present in their causal composition: the ladder is generic
year machinery whose contexts skew physics because the corpus does.

Refinement of the recognition/production dissociation: the two views
MEET inside the relation machinery -- production-pathway rungs are
connected among themselves by ordinary validated circuits. Circuits
CAN capture production structure; you just have to seed them on
relation latents rather than concept latents.

## ML domain: behavioural floor + a computed completion anatomised

`ml_completions.py`: 20/32 -- concept-level ML knowledge solid
(backprop, gradient descent, overfitting, bias-variance...), NAMED
TERMS missing (ReLU, Adam, discriminator, centroid, cross-entropy,
attention keys). The behavioural knowledge floor mirrors the latent
dedication floor: concepts yes, entities no.

`ml_flow.py` (lens + knockouts) on three well-known facts:
* gradient->descent: semi-lookup (rank 2-4 from L0, collocation).
* maximum->margin: NON-MONOTONIC lens (rank 8 at L2 -> 36 at L6 -> 1
  at L9): the answer is temporarily suppressed mid-stack; late
  binding. Top knockout = infrastructure 5663 AGAIN (dlogp 1.02).
* training set and a -> validation: the most computed (rank 1626 at
  L0, rank 1 only at L10). Causal chain named:
    L7/25475 + L8/25030  "a"-article slot latents (dlogp 1.08/0.47)
    L8/27896  CNN/deep-learning exposition (1.17; ALSO a top cause of
              the margin completion -- an ML-domain workhorse)
    L9/40790  supervised/deep-learning exposition (0.47)
    L10/9551  semi-supervised labeled/unlabeled TRAINING-SET talk
              (0.47) -- the closest content latent to the fact
    L11/8028  ensemble/bagging/bias-variance methods (0.95)
  Pattern: computed completion = syntax-slot state (article machinery)
  x a STACK of ML-domain context latents converging L8-L11. Unlike
  the year fact, there is NO dedicated relay -- the rarer domain
  assembles its completion from general domain latents, later and
  thinner (emergence L10 vs the year's L6). Frequency buys dedicated
  relay hardware; rarer knowledge rides general context.

## Sign census: the fragility hypothesis, tested both ways

`sign_census.py`: solo-remove each of the top-200 active upstream
latents, read the seed pre-act on held-out probes. POS frame (does
anything hold the seed back where it fires?) and NEG frame (CTX=neg:
what suppresses it on near-miss contexts where it is silent?).

POS: relativity 29 act / 4 inh (total inhibition +2.0 = 6% of
natural); boson 32/0 (0%); music-brain 52/0 (0%).
NEG: relativity 27 act / 6 inh (+1.56 = 12% of the near-miss level);
boson 29/5 (+0.62 = 6%).

VERDICT: the fragility hypothesis ("more inhibitors than activators")
is FALSIFIED in both frames for these seeds. Discrete latent brakes
are rare and weak; removing every identified suppressor on near-miss
contexts lifts the seed 6-12%, nowhere near firing it. The seed's
silence is ABSENCE OF DRIVE, not enforced inhibition: evidence-gated,
not inhibition-gated. The posctx raise-mask campaign (gamma >= 1.25)
is uncalibratable and is dropped.

SYNTHESIS with the FMd overshoot: mean-filling non-members overshoots
by ~10-35% on several seeds -- collective, diffuse mild suppression by
the non-member mass, matching the census's 6-12% solo-sum. Suppression
exists as a WEAK DIFFUSE FIELD, not as discrete brake latents. (Two
instruments, same magnitude.)

CAVEATS: solo removals (collective superadditivity untested, but the
summed ceiling caps it low); pool = top-200 by activation; and the
node universe is top-k SAE latents -- inhibition implemented below the
dictionary (raw stream, attention) would be invisible here. Closes
questions #6-#9 of the method list with a measured negative.

## Negative amplification (signed alpha), first run

Engine: signed_amplitude flag (alpha = psi raw; init 1.0; leak guard
and amp_l1 routed through the same transform; softplus path
bit-identical when off). Arm sgnamp400 on c29/3736 + c35/13633.

RESULTS: both circuits near-production (c29 n=492 F0 0.966 FMd 1.280*;
c35 n=341 F0 0.902 FMd 1.207). Negative members are RARE: 6/492 and
7/341 (~1.5%), no pile-up near zero (median alpha 1.07 as before).

CROSS-VALIDATION HIT: boson's strongest negative member (11/mlp/28635,
alpha -4.43) is EXACTLY its census-identified suppressor (musical-
instrument taxonomy). Two independent instruments -- solo-removal
census and signed fitting -- name the same brake.

MIXED ELSEWHERE: relativity's 6 negative members are disjoint from its
census set (its census brakes are each <=1.5% -- likely below the L1
price); 1/mlp/26665 and 1/attn/12218 go NEGATIVE IN BOTH seeds' signed
fits (shared suppressive pair, unnamed yet). CAVEAT: nulls were fitted
unsigned (asymmetric freedom) -- fix before quoting gates.

Method-list #2 closed: tried, works, brakes are rare (~1.5% of
membership), consistent with the weak-diffuse-suppression picture.

## Firing-margin objective (#11): built, run, and it caught something

Engine: margin_topk=k -- the seed tap reports (pre - tau), tau = k-th
largest site pre-activation, so _natural() yields margin targets and
all consumers inherit the frame. Arm mrgamp400 (k=128), 3 seeds.

Value-frame scores run HOT as predicted (F0 1.05-1.25, FMd 1.44-1.63:
the margin objective tolerates hotter absolute drive when the margin
matches). The decisive eval is FIRING FIDELITY (firing_fidelity.py:
per held-out probe, post-top-k read under mean-fill circuit-only
execution; fired iff > 0):

  c29/3736   natural 16/16 | triamp 16/16 (mean 44.6) | margin 16/16
  c35/13633  natural 16/16 | triamp 16/16 (26.4)      | margin 16/16
  c29/23920  natural 16/16 | triamp  0/16 (!!)        | margin 16/16

THE CATCH: the music-brain triamp circuit -- the BEST-calibrated gate
score of the five diverse seeds (F0 0.957, FMd 1.048) -- fires the
seed on ZERO of 16 held-out probes under mean-fill execution. The
mechanism: mean-filling non-members lifts thousands of rivals at the
seed's site, inflating the top-k cutoff; a circuit calibrated to
natural value (0.96x) loses the competition, while the hotter circuits
(1.2-1.35x) clear it incidentally. "Reconstructs the value, loses the
firing" is a REAL, present failure mode that the two-criteria gate
cannot see -- caught on 1 of 3 tested production circuits.

IMPLICATIONS: (1) firing fidelity joins the eval suite as a third
criterion; (2) the margin-fitted arm is robust to fill-induced
competition by construction (trained against tau); (3) the value
gate's calibration ideal (F0 = 1.00 exactly) is, under fill execution,
the RISKIEST operating point -- mild overshoot is what keeps seeds
firing. Methods-appendix material. Frame caveat: zero-fill execution
has its own tau pathology (the censored explosion), so firing checks
belong in the mean-fill frame.
