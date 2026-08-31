# Recursive latent map — does the internal explanation close? (2026-08-02)

Daniel's question: with closure circuits at 10^4-10^5 latents, how can a
human ever read the structure? Proposal: don't read the closure — expand
the seed RECURSIVELY into latents-that-explain-latents and see whether
the graph saturates at readable size.

Method: seed -> top-8 drivers -> each driver's top-8 drivers, to depth 3,
using VALUE-edges (d(node value)/du * (u_nat - u_floor), the D3.3
primitive) — the effect of an upstream latent on ANOTHER LATENT'S VALUE,
never on a behavioural metric. One forward per level, one backward per
node (retain_graph), so cost is linear in nodes expanded (~2-3s per
seed). Every depth's cumulative union is then scored on the full exam
against a size-matched random control. 4 resid seeds, L2 to L11.

## RESULT 1 — the map closes, or nearly, at 10^1-10^2 nodes

Naive depth-3 growth at branch 8 is 8 + 64 + 512 = 584 nodes.

| seed | upstream sites | d1 | d2 | **d3 cumulative** | sharing d2 | sharing d3 | % of naive |
|---|---:|---:|---:|---:|---:|---:|---:|
| L2 resid | 8 | 8 | 27 | **36** | 0.35 | 0.70 | 6.2% |
| L6 resid | 20 | 8 | 30 | **61** | 0.36 | 0.59 | 10.4% |
| L8 resid | 26 | 8 | 39 | **97** | 0.20 | 0.48 | 16.6% |
| L11 resid | 35 | 8 | 32 | **62** | 0.27 | 0.61 | 10.6% |

**Sharing rises with depth on every seed (0.20-0.36 at d2 -> 0.48-0.70
at d3): by the third hop, half to three-quarters of every node's drivers
are latents the graph has ALREADY seen.** The recursive explanation of a
latent folds back on itself rather than fanning out. L2's frontier
turned over outright (new nodes 8 -> 19 -> 9); the deeper seeds are
still growing at d3 but at a decaying rate.

Scale: 36-97 latents vs the 10^4-10^5 closure for the same seeds — two
to three orders of magnitude smaller, and squarely in the range a human
can read.

## RESULT 2 — the recursive union is SUFFICIENT AND NECESSARY

| seed | depth | n | cf_alpha | sup | random cf | random sup |
|---|---:|---:|---:|---:|---:|---:|
| L2 | 1 / 2 / 3 | 8 / 27 / 36 | 1.065 / 1.060 / **1.030** | 0.738 / 0.938 / **1.000** | – | 0.0 |
| L6 | 1 / 2 / 3 | 8 / 30 / 61 | 1.004 / 1.047 / **1.073** | 0.746 / 0.984 / **1.000** | 0.0 | 0.0 |
| L8 | 1 / 2 / 3 | 8 / 39 / 97 | 0.987 / 1.021 / **1.008** | 0.626 / 0.909 / **1.012** | -0.0001 | 0.0 |
| L11 | 1 / 2 / 3 | 8 / 32 / 62 | 0.952 / 0.993 / **1.007** | 0.904 / 1.000 / **0.993** | – | 0.0 |

Two clean regularities, 4/4 seeds:
- **cf_alpha is ~1.0 from depth 1 onward** — drive saturates at the
  first 8 latents (consistent with AMPC; recursion adds no drive).
- **sup climbs monotonically with depth to 1.0** (0.63-0.90 at d1 ->
  0.91-1.00 at d2 -> 0.99-1.01 at d3). The recursion specifically
  recruits what NECESSITY requires.

Size-matched random controls are 0.000 on both columns everywhere
(including at n=97 with alpha pegged at the 8.0 ceiling: cf -0.0001).

**This is the first object in the project that passes BOTH driver gates
at once at readable size.** AMPC is sufficient but was never shown
necessary; the closure mask is neither (cf ~0); the brake mask is
neither. A 36-97 latent recursive map is both.

## RESULT 3 — recursion does NOT reconstruct closure *at branch 8* (NARROWED by RESULT 5a)

free0 stays at 0.000 at every depth on every seed (one transient 0.069
at L2 d2). The transitive closure of DRIVERS is not the CLOSURE object:
(P1) expanded recursively converges on a sufficient-and-necessary drive
core, not on free-running faithfulness. Combined with D3.5 (compressing
(P2) does not converge on (P1) either), the two objects are now shown
NOT to be reachable from each other in EITHER direction.

**Superseded in part.** RESULT 5a shows this held only because branch
was fixed at 8. Widen the branch and the L2 seed's recursive union
reaches free0 0.741 at n=3,242 while keeping cf_alpha and sup at 1.000.
The correct statement is: *driver expansion does not reach closure at
readable size, and does not reach it at any size on deep seeds* — not
that the two objects are mutually unreachable in principle. The D3.5
direction (closure -> drivers) is untouched.

## RESULT 4 — the full exam (`full_matrix.py` -> `full_matrix_v2.jsonl`)

RESULT 3 rested on free0 alone, and free0 has a known false-negative
mode. So the depth-3 unions were re-scored on every floor convention we
have — zero floor (free0), corpus-mean floor dense and top-k
(freeM_dense/freeM_topk), negctx floor dense and top-k
(freeN_dense/freeN_topk), collapsed pins (pin0) — plus the drive and
brake columns, all against the size-matched random control.

| seed (n) | free0 | freeM_d | freeM_k | freeN_d | freeN_k | pin0 | cf_raw | cf_alpha | sup | raise | drop x4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 (36) | 0.000 | 0.000 | 0.000 | 0.113 | 0.031 | 0.000 | 0.177 | **1.030** | **1.000** | -0.970 | -0.473 |
| L6 (61) | 0.000 | 0.000 | 0.000 | 0.075 | 0.048 | 0.000 | 0.581 | **1.073** | **1.000** | -1.000 | -1.530 |
| L8 (97) | 0.000 | 0.000 | 0.000 | 0.527 | 0.270 | 0.000 | 0.383 | **1.008** | **1.012** | -1.000 | -0.842 |
| L11 (62) | 0.000 | 0.000 | 0.000 | 0.312 | 0.148 | 0.000 | 0.776 | **1.007** | **0.993** | -0.993 | -1.841 |
| *random* | 0.000 | 0.000 | 0.000 | 0.008-0.465 | 0.001-0.253 | 0.000 | ~0 | ~0 | 0.000 | 0.000 | ~0.000 |

Three things fall out:

1. **"Not a closure object" now holds under four independent floor
   conventions, not one.** free0, freeM_dense, freeM_topk and pin0 are
   all exactly 0.000 on all 4 seeds. The only free-family column that
   moves is freeN — and random matches it seed-for-seed (L8: 0.527 vs
   0.465; L11: 0.312 vs 0.338, i.e. random *wins*). freeN's signal is
   the negctx floor itself, not the circuit. RESULT 3 was not a free0
   artifact.
2. **The alpha fit is load-bearing, not cosmetic.** At native pin values
   the recursive union reaches only cf_raw 0.18-0.78; alpha (1.28-3.22)
   is what carries it to ~1.0. The map identifies the right *latents*;
   it does not reproduce the right *magnitudes* under counterfactual
   context. Same alpha*(K, kind) law as AMPC.
3. **The recursive map is the exact functional MIRROR of the brake
   circuit.** Silence it and the seed dies (sup ~1.0); amplify it (raise
   / drop x4) and the seed... also moves hard against its natural level
   (-0.97 to -1.84) — the sign convention of the brake columns is
   inverted for a driver, so a driver reads maximally "wrong" on them.
   That is two independent necessity measurements, from opposite
   directions, agreeing. Random is 0.000 on both.

`free0_pre_raw` (uncensored pre-activation) is *not* a usable
discriminator here: random sets read 116 / 1848 / 44288 / 544768 against
a_pos of 6.3 / 14.6 / 15.1 / 36.3. Injecting arbitrary latents into a
zero-floor stream inflates the pre-activation enormously; the recursive
map's 1.2 / 52.8 / 161 / 15040 is *closer* to natural but the metric has
no normalisation that survives this, which is why it stays diagnostic-
only.

## RESULT 5 — the SIZE SWEEP: free evals DO rise, but only on shallow seeds, and one-hop matches recursion (`size_sweep.py` -> `size_sweep.jsonl`)

RESULT 3/4 said "not a closure object" at n=36-97. That was a claim about
one point on a curve. This sweep drives the same algorithm from n=36 to
n=7,266 (branch 8->512, depth 2-3, expansion capped at 260 nodes) and
adds the control that RESULT 1-4 lacked: a size-matched **flat** arm —
the top-n latents by value-edge taken DIRECTLY from the seed, one hop,
no recursion. Same primitive, no compositional structure. 3 resid seeds.

### 5a. Yes — closure IS reachable, on the shallow seed

L2 resid (8 upstream sites), recursive arm:

| n | 36 | 271 | 1,022 | 1,063 | 3,242 |
|---|---:|---:|---:|---:|---:|
| % of dict | 0.011 | 0.083 | 0.312 | 0.324 | 0.989 |
| free0 | 0.000 | 0.151 | 0.500 | 0.508 | **0.741** |
| pin0 | 0.000 | 0.642 | 0.816 | 0.821 | **0.901** |
| cf_alpha | 1.030 | 1.070 | 1.015 | 1.015 | **1.000** |
| sup | 1.000 | 1.000 | 0.987 | 1.000 | **1.000** |

Monotone, and **the drive/necessity columns do not degrade as closure
accumulates**. At n=3,242 this object is simultaneously closure-faithful
(free0 0.741, in the abl-mask's 0.7-1.0 band), drive-sufficient
(cf_alpha 1.000) and necessary (sup 1.000).

**CORRECTED 2026-08-04.** An earlier version of this paragraph claimed
"no other object in the project holds all three: abl-mask has closure
with cf~0". That is **wrong** — it repeated a stale note without
checking. The frozen-recipe panel
(`experiments/015-prelim-matrix/rows_s*.jsonl`) shows abl-mask
at **cf 0.93-1.32 and sup 1.00-1.04** on 8/8 seeds alongside free0
0.91-1.00. The closure mask passes all three gates too. What is
distinctive about the recursive map is doing it at **36-97 nodes** where
abl-mask needs 655-24,157 — a size claim, not a kind claim.

So RESULT 3's "recursion does NOT reconstruct closure" was **size-bound,
not a property of the algorithm** — and is hereby narrowed accordingly.

### 5b. But closure is depth-gated, hard

| seed | sites | closure reached? |
|---|---:|---|
| L2 | 8 | free0 -> 0.741, pin0 -> 0.901 (both) |
| L8 | 26 | pin0 -> 0.718; free0 0.000 to n=7,266 |
| L11 | 35 | both 0.000 to n=5,789 |

**RETRACTED in part by RESULT 6a.** Both deep-seed zeros were ceiling
artifacts of stopping at n<=7,266. Taken to 5% of the dictionary, L8
reaches free0 0.450 (n=53,248) and L11 0.145 (n=71,680). Closure is not
out of reach at depth — it is 10-40x further away than this sweep went.
What survives is the *ordering*: closure cost rises steeply with seed
depth (L2 <0.5% of scope, L8 and L11 5%), while cf_alpha/sup stay ~1.0
at every size on every seed.

Not a scope-size artifact: L8 at n=7,266 is 0.68% of its dictionary and
L2 at n=3,242 is 0.99% — comparable fractions, opposite outcomes. Depth
removes closure from this algorithm's reach, and it does so in stages:
the zero floor goes first, then the pin floor. Under a zero floor a deep
seed needs its whole support stream rebuilt from 26-35 sites of nothing,
and no top-K selection at these sizes can do it.

cf_alpha and sup meanwhile sit at ~1.0 on all 3 seeds at every size.
**The drive/necessity core is depth-invariant; closure is not.**

### 5c. The one-hop flat ranking matches or beats recursion

15 size-matched recursive-vs-flat comparisons. Recursive wins 2 (L2
n=271 free0 0.151 vs 0.054; L8 n=1,810 pin0 0.171 vs 0.137) — and each
reverses at the next size up. Flat wins 4, all at n>=1,022 (L2 n=3,242
free0 0.791 vs 0.741; L8 n=2,243 pin0 0.822 vs 0.544; L8 n=7,266 pin0
0.830 vs 0.718; L2 n=1,022/1,063 free0 0.582 vs 0.500). The other 9 are
ties at the floor. cf_alpha and sup are indistinguishable throughout.

**The causal content is in the first-hop d(seed)/d(latent) value-edge
ranking. The recursive expansion is a more expensive route to a set of
the same quality.** This does not touch the "latents explain latents"
primitive — flat_matched is itself pure latent->latent value-edges, zero
behavioural signal — but it demotes the *recursive/compositional*
framing from load-bearing to convenient.

The one thing recursion still owns is the **small readable object**: at
n=36-97 it produces a self-folding graph with an edge structure (RESULT
1's sharing rates), where flat produces a rank-ordered list with no
internal topology. Flat matches it on the metrics; it does not match it
as a *map*.

### 5d. Branch is the size knob, not depth — and it's seed-dependent

Branch 128, depth 2 -> 3 node growth: L2 +4% (1,022->1,063), L8 +24%
(1,810->2,243), L11 +53% (1,150->1,760). Shallow seeds have already
folded back on themselves after two hops (RESULT 1's mechanism, now
measured at branch 128); deep seeds have not. Depth saturation is
inversely related to seed depth.

### 5e. METRIC VALIDITY — cf_alpha and sup contaminate at large n

The size-matched random control is 0.000 everywhere on free0 and pin0,
at every size to n=7,266. But it is **not** clean on the drive/necessity
columns at scale: L2 n=3,242 random reads sup 0.209; L8 n=7,266 random
reads **cf_alpha 0.398, sup 0.155**.

**cf_alpha ~1.0 at n=7,266 means far less than cf_alpha ~1.0 at n=97.**
Any drive/necessity claim above ~10^3 nodes needs its in-run random
control quoted alongside. Closure metrics (free0, pin0) are size-robust;
drive/necessity metrics are not. This applies retroactively to every
large-n cf/sup number in the project.

Also note alpha: the flat arm's fitted alpha falls to 1.04 at large n,
i.e. essentially no amplification needed. The magnitude problem
identified in RESULT 4 is a small-circuit problem, not a general one.

### Caveats specific to this sweep

- **Expansion was capped at 260 nodes** (MAX_EXPAND), which binds on
  (128,3) and (512,2) for all seeds and on (32,3) for L11. The recursive
  arm at the largest sizes is therefore a *partial* expansion — its
  frontier was truncated where the flat arm's ranking was not. The
  flat-wins-at-large-n conclusion (5c) is consequently the weaker half
  of that result and should be re-run uncapped before it is relied on.
  5a, 5b, 5d and 5e are unaffected.
- 3 seeds, all resid; L2/L8/L11 only.
- b8/d3 reproduces the original RESULT 1-2 numbers exactly on all three
  seeds (n=36/97/62, cf 1.0299/1.0084/1.0069), so the expansion is
  deterministic and the sweep is directly comparable to RESULT 1-4.

## RESULT 6 — the FULL-SCOPE ladder: where every floor turns on (`big_sweep.py` -> `big_sweep.jsonl`)

RESULT 5b reported L8 free0 stuck at 0.000 to n=7,266 and L11 stuck at
0.000 on everything to n=5,789. **Both were ceiling artifacts of where
the sweep stopped.** This run takes the same one-hop value-edge ranking
to 100% of the upstream dictionary — 0.33M / 1.06M / 1.43M latents on
L2 / L8 / L11 — on a 10-rung fractional ladder, with a size-matched
random arm at every rung. Flat rather than recursive, per RESULT 5c
(indistinguishable at matched size) and because recursion costs one
backward per node while flat sorts one full-scope weight vector once.

Drive/necessity columns deliberately omitted: RESULT 5e showed cf_alpha
and sup are size-contaminated in exactly this regime.

### 6a. Every floor turns on. The turn-on point is the finding.

FLAT arm, free0 (zero floor) / freeM_topk (corpus-mean) / pin0 (pins):

| % dict | L2 free0 | L8 free0 | L11 free0 | | L2 pin0 | L8 pin0 | L11 pin0 |
|---:|---:|---:|---:|---|---:|---:|---:|
| 0.5 | **0.624** | 0.000 | 0.000 | | 0.940 | 0.788 | 0.241 |
| 1.0 | 0.791 | 0.000 | 0.000 | | 0.945 | 0.863 | 0.617 |
| 2.0 | 0.836 | 0.000 | 0.000 | | 0.955 | 0.863 | 0.772 |
| 5.0 | 0.866 | **0.450** | **0.145** | | 0.960 | 0.896 | 0.848 |
| 10 | 0.881 | 0.776 | 0.486 | | 0.970 | 0.938 | 0.917 |
| 20 | 0.895 | 0.867 | 0.724 | | 0.975 | 0.979 | 1.014 |
| 50 | 0.935 | 0.925 | 0.890 | | 1.015 | 1.079 | 1.110 |
| 100 | 1.000 | 1.000 | 1.000 | | 1.015 | 1.095 | 1.124 |

**free0 first non-zero: L2 <0.5% (n<1,638) | L8 5% (n=53,248) | L11 5%
(n=71,680).** Deep seeds DO reach closure — they need ~10-40x more
latents than the size sweep's ceiling, which is why RESULT 5b saw only
zeros. "Never leaves 0.000" is retracted; the correct statement is
"needs >=5% of the upstream dictionary".

Floors turn on in strict difficulty order on every seed:
**pin0 < freeM < free0.** L11 is the clean case: pin0 0.241 at 0.5%,
freeM 0.017 at 0.5%, free0 not until 5%. That ordering is a
quantitative statement of how much of the stream each convention makes
the circuit rebuild — pins hand it the whole context, corpus-mean hands
it generic content, zero hands it nothing.

### 6b. free0 is a THRESHOLD test, not a coverage measure

The random arm reads free0 **exactly 0.000** at every rung to 35% of the
dictionary on all three seeds — and on L8 to **75% (n=798,720)**. A
random three-quarters of the entire dictionary, kept at natural values,
produces literally no seed activation. Then at 100% it is 1.000 by
construction.

That 0.000 -> 1.000 step is the whole character of the metric: free0
does not degrade gracefully with coverage. Miss the specific latents
carrying the drive and post-top-k censoring drops the seed below
threshold and the reading collapses to zero regardless of set size.
Every 0.000 in this project's history is that mechanism, not an absence
of structure.

Practical consequence, opposite in sign to RESULT 5e: **free0/freeM/pin0
cannot be size-contaminated below ~50% of scope, so they need no
size-matched control in the regimes we actually work in** — whereas
cf_alpha/sup are contaminated from ~10^3-10^4 nodes and always do.
Random's first breaks: L2 pin0 0.657 at 50%, L8 pin0 0.689 at 75%, L11
pin0 0.459 at 50%; free0 stays 0.000 until 75% (L2 0.450) or 100%.

### 6c. Only ~6-19% of the dictionary is LIVE

Latents carrying non-zero value-edge weight, i.e. that fire anywhere on
the probe batch:

| seed | live / scope | % |
|---|---|---:|
| L2 | 20,914 / 327,680 | **6.4** |
| L8 | 186,090 / 1,064,960 | **17.5** |
| L11 | 267,960 / 1,433,600 | **18.7** |

Everything else is identically zero and can never affect the seed on
these sequences.

**CORRECTED 2026-08-04 — read the qualifier, it is load-bearing.** These
are latents live *on the 8 probe sequences used to build the ranking*
(GRAD_B=8, 512 positions), not an intrinsic property of the dictionary.
Measured over 64 corpus sequences (4,096 positions) at the same 8 L2
upstream sites, **215,741/327,680 = 65.8%** of latents fire at least
once — 10x the 6.4% figure above. The live fraction is a function of how
many sequences you look at.

So an earlier draft of this section claiming "every % of dictionary
figure understates density by 5-16x, the honest denominator is live
latents" was **wrong** and has been removed. There is no single live
denominator to renormalise against; d_sae x n_sites remains the correct
scope for cross-paper size comparisons, and
`circuit-size-normalisation.md` needs no change. What survives is the
narrower, still-useful fact below.

- Past the 8-sequence live cutoff the flat ranking is padding with
  zero-weight latents in arbitrary index order, so rungs above ~20% are
  not really "ranked" — which is exactly where the curve plateaus (L8:
  0.867 at 20%, 0.855 at 35%). This is a statement about the RANKING's
  resolution given 8 probe sequences, not about the model's sparsity.

### 6d. The attribution has a measurable generalisation gap

At 20% of scope L8's ranked set contains *every* non-zero-weight latent
(17.5% cutoff) yet reads free0 0.867, not 1.000. The missing ~13% is
latents live on the 16 eval sequences but silent on the 8 sequences used
to build the ranking. That is a real train/eval generalisation gap in
the value-edge attribution, separately measurable, and it bounds what
any 8-sequence attribution can achieve.

### 6e. Cross-check

L2 flat at 1% (n=3,277) reads free0 0.791; the size sweep's independent
flat arm at n=3,242 read 0.791. Two scripts, same number.

## Why this matters for the thesis

"Latents explain latents" now has a concrete, measured deliverable: for
a given seed, a **36-97 latent graph, built entirely from latent->latent
value-edges, that drives the seed to its natural level and is required
for it to fire**, with every edge causally weighted and the whole object
small enough to read. That is the readable internal map the closure
could never be.

## Caveats

- 4 seeds, all resid. attn/mlp seeds untested and are the historically
  awkward kinds.
- Depth 3 only; L6/L8/L11 frontiers had not fully turned over, so
  "closes" is demonstrated for L2 and *trending* for the rest. Depth 4-5
  is the obvious next run (cheap: ~3s/seed/level).
- ~~Branch fixed at 8; the saturation rate is certainly branch-dependent
  and unswept.~~ Swept in RESULT 5 (branch 8-512): branch, not depth, is
  the size knob, and saturation is inversely related to seed depth.
- Value-edges use 8 probe sequences and a zero floor.
- Edges archived per seed (edges_*.jsonl.gz) for the labelling pass —
  which is the missing step before this is genuinely "understandable"
  rather than merely small.
