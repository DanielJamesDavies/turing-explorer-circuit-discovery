# Is there a universal core? — abl-mask on 32 L2-resid seeds (2026-08-04)

Daniel's question: run the closure mask over many seeds in one layer and
look at the crossover — **are there latents in every seed's circuit?**

All 32 seeds are L2 resid (comp_idx 8, sampled deterministically from
455 candidates, rng 42), so every circuit lives in the identical 8-site
upstream scope (327,680 slots) and membership sets are directly
comparable with no renormalisation. House recipe, abl-mask arm (NPA,
posctx floor, no post-hoc pruning — the mask's membership IS the
object). ~20s per seed after warmup.

## The circuits are real closure objects

| | min | median | max |
|---|---:|---:|---:|
| nodes | 399 | 1,959 | 3,522 |
| free0 | 0.905 | ~0.99 | 1.017 |

32/32 land in the closure band (free0 0.905-1.017, mean ~0.99) across a
9x spread in size. Nothing degenerate in the panel.

## ANSWER: yes — exactly 4 latents, and they are the densest in the model

| k of 32 | latents | cumulative >=k |
|---:|---:|---:|
| **32** | **4** | **4** |
| 31 | 11 | 15 |
| 30 | 19 | 34 |
| >=28 | – | 67 |
| >=16 | – | 698 |
| 2 | 3,536 | 6,824 |
| 1 | 23,680 | 30,504 |

**All 4 universal latents sit at L0/attn.** They are 0.1-1.0% of each
circuit (mean 0.3%) — a tiny core, not a backbone.

And they are not ordinary latents: their median corpus activation
density is **0.3425** (they fire on ~34% of all token positions) against
**0.0012** for the live pool as a whole. **The universal core sits at
the 99.7th percentile of activation density.**

## But most of the crossover is a DENSITY artifact

Two nulls, both at the observed per-seed set sizes, drawn from the live
pool (215,741 latents ever active over 64 corpus sequences / 4,096
positions — 65.8% of scope):

- **N1 uniform** — controls for set size only.
- **N2 density-weighted** — P(latent) proportional to corpus activation
  frequency. Controls for "shared nodes are just the latents that fire
  on everything".

| | universal (32/32) | in >=16/32 | pairwise Jaccard |
|---|---:|---:|---:|
| N1 uniform | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.004 |
| N2 density-weighted | 0.0 +- 0.0 | 570.2 +- 14.0 | 0.079 +- 0.021 |
| **OBSERVED** | **4** | **698** | **0.129 +- 0.026** |

N1 is the easy null and the result clears it by miles — but N1 is not
the fair null. **N2 reproduces 570 of the 698 latents shared by half the
panel (82%) and 61% of the observed pairwise Jaccard from activation
frequency alone.**

**That aggregate hides the structure — see `hist_vs_nulls.json` and the
per-k curves.** The excess is not spread evenly across k; it is confined
to the far tail:

| band | observed | N2 | ratio |
|---|---:|---:|---:|
| k = 13-20 | 504 | 651 | **0.77x** (observed BELOW null) |
| k = 21-23 | 177 | 101 | 1.75x |
| k >= 24 | 209 | 27.5 | **7.6x** |
| k = 32 | 4 | 0.0 | — |

Latents shared by roughly half the panel are not a phenomenon at all:
density alone predicts *more* of them than we observe. Everything real
lives above k~22. Quoting the ">=16" aggregate alone would misdescribe
this in both directions at once.

The honest reading:

- The *existence* of a universal core is real: N2 predicts 0.0 latents
  in all 32 circuits, observed 4. Density alone does not put the same
  latent in every circuit — under N2 even the densest latent enters a
  given circuit ~64% of the time, so 32/32 is ~3e-7.
- The *scale* of the crossover is mostly density. Anyone reporting "our
  circuits share 698 latents" without N2 would be reporting the
  activation histogram.

## Circuits are overwhelmingly seed-specific

Union of all 32 circuits: **30,504 distinct latents** from 60,889
memberships. **23,680 (77.6%) appear in exactly ONE circuit.**

So the picture is not "one shared skeleton plus decoration". It is a
mostly-private circuit per seed, plus a handful of ubiquitous
high-density L0/attn latents that everything routes through.

## Attention sites carry the sharing

Enrichment among latents in >=16/32 circuits, as a fraction of that
site's contribution to the union:

| site | >=16/32 | union | enrich |
|---|---:|---:|---:|
| L1/attn | 46 | 404 | **11.4%** |
| L0/attn | 164 | 3,633 | 4.5% |
| L2/attn | 28 | 629 | 4.5% |
| L0/resid | 162 | 5,563 | 2.9% |
| L0/mlp | 156 | 6,780 | 2.3% |
| L1/resid | 103 | 5,887 | 1.7% |
| L1/mlp | 36 | 4,172 | 0.9% |
| L2/mlp | 3 | 3,436 | **0.1%** |

Attention sites are 2-12x enriched for shared membership; **L2/mlp is
essentially never shared** (3 of 3,436). Whatever is seed-specific is
carried by the MLP immediately below the seed; whatever is shared is
carried by attention, especially at L0-L1.

## What the private periphery IS — two follow-up tests

Daniel: "what does the large private periphery mean? could real circuits
just be this large?" Density profile by sharing band answers the first
half — sharing frequency is almost a pure readout of activation density,
and the private nodes are the RAREST latents in the panel:

| band | latents | median corpus density | never fire in corpus |
|---|---:|---:|---:|
| k=1 | 23,680 | 0.0017 | 2,301 (10%) |
| k=2-5 | 5,142 | 0.0115 | 84 (2%) |
| k=6-15 | 984 | 0.0710 | 0 |
| k=16-23 | 489 | 0.2673 | 0 |
| k=24-31 | 205 | 0.3814 | 0 |
| k=32 | 4 | 0.6685 | 0 |

(live-pool median 0.0012). Note the two denominators: **78% of the UNION
is private, but only 22-48% (median 38%) of any single circuit is** — a
typical circuit is majority-shared. The union is private-dominated only
because 32 seeds each contribute their own batch.

### Test A — the periphery is LOAD-BEARING (`periphery_test.py`)

Every circuit scored five ways on the same probe set:

| arm | min | median | max | zeros |
|---|---:|---:|---:|---:|
| full | 0.905 | **0.994** | 1.017 | 0/32 |
| minus_private (k>=2) | 0.000 | **0.058** | 0.498 | 1/32 |
| private_only (k=1) | 0.000 | **0.000** | 0.000 | **32/32** |
| core_only (k>=24) | 0.000 | 0.000 | 0.074 | 25/32 |
| rand_matched | 0.000 | 0.000 | 0.000 | 32/32 |

Dropping the private 38% takes free0 from 0.994 to **0.058 (5.9%
retention), on 32/32 seeds** — but the private nodes ALONE are 0.000
everywhere, and the shared core alone is ~0.000. **Neither half works;
only the whole does. The circuit is not decomposable into
"core that computes" + "passengers".**

Two escapes are closed by the correlations: retention vs size of the
kept set is **-0.02**, retention vs private share is **0.10**. The
collapse is indifferent to both, so it is not "big remainders survive"
nor "low-private circuits survive". And rand_matched at the same size is
0.000, so minus_private's 0.058 is real signal, just badly degraded.

Caveat: free0 is a threshold test (recursive-map RESULT 6b), so this
measures AGGREGATE contribution, not per-node necessity. It does refute
the passenger hypothesis — passengers by definition would not matter —
but it does not show every individual private latent is required.

### Test B — size is NOT a union over contexts (`probe_scaling.py`)

The hypothesis this run was built to kill: the mask is NPA and trained
over P probe sequences, so its membership might just be a union over
contexts, making "circuit size" a function of how much data it saw
(~31 latents/context x 64 = ~2,000). 6 seeds at P = 8/16/32/64, with
evaluation held FIXED at a 64-sequence probe set.

| P | mean n | free0 on 64 | Jaccard vs P=64 | covers P=64 circuit |
|---:|---:|---:|---:|---:|
| 8 | 1,326 | 0.774 | 0.417 | 48% |
| 16 | 1,383 | 0.844 | 0.469 | 53% |
| 32 | 1,731 | 0.942 | 0.589 | 67% |
| 64 | 2,178 | 0.995 | 1.000 | 100% |

Per-seed n(64)/n(8): 1.45 / 1.97 / 2.08 / 1.67 / 1.35 / 1.36, **mean
1.65 where the union hypothesis predicts 8.0**. Implied scaling
**n ~ P^0.23** — size is nearly saturated by P=8.

**What more contexts buy is membership QUALITY, not volume.** Across the
same 8x of data, n rises 1.65x while free0 rises 0.774 -> 0.995 and
overlap with the reference circuit doubles (48% -> 100%). The mask is
re-deciding which ~2,000 latents to keep, not accreting new ones.

P=64 reproduces the stored reference exactly on all 6 seeds (J=1.000),
so the mask is deterministic and the curve is trustworthy.

### Verdict

**Yes, real circuits are this large.** ~2,000 latents for an L2 seed is
not mask slack (Test A: removing any part collapses it) and not a
data-collection artifact (Test B: size saturates at P^0.23). The drive
is genuinely distributed across ~2,000 mostly-rare latents, and the
handful of universal high-density nodes is the small part of the story.

This also retires the framing that circuit size reflects probe budget —
it does not, at least on this panel.

## Caveats

- One layer, one kind (L2 resid). The universal core being L0/attn may
  be a "bottom of the stack" effect that does not generalise upward.
- Corpus density from 64 sequences / 4,096 positions; the live pool
  (65.8% of scope) would keep growing with more corpus.
- N2 is density-weighted sampling without replacement, which is a
  reasonable but not unique null; it ignores that masks select for
  *causal* relevance, which correlates with density beyond frequency.
- The 4 universal latents are not yet labelled — what they actually
  represent is the obvious next question and the data (`members.jsonl.gz`,
  `universal_nodes.json`) supports it.

## Files

- `runner.py` — 32 x abl-mask + per-seed live sets + corpus density
- `analyse.py` — crossover histogram, N1/N2 nulls, Jaccard
- `rows.jsonl`, `members.jsonl.gz`, `corpus_density.pt`
- `crossover_summary.json`, `universal_nodes.json`
