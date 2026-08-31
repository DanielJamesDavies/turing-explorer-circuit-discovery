# The support-matched null: is our result about selection, or about TopK?

2026-08-12. Llama-3.2-1B + EleutherAI TopK skip-transcoders, the same 6
seeds as the rest of this directory.

## The worry

Every dictionary this method has been validated on is top-k-ish (the
home bank k=128/40,960; the Pythia TopK SAEs; these Llama TopK skip
transcoders). The one ReLU/L1 bank we tried was scoped out of domain
because the null went vacuous there — random size-matched sets scored
~1.0 and the test could not discriminate.

The mechanism is a ratio: latents active AT THE ANCHOR divided by live
pool size. It was 6.5-37% on the ReLU/L1 banks and is 3.5-5.9% here.

That cuts both ways, and the uncomfortable direction is ours. Our PLAIN
null draws from every latent that fires anywhere; at ~4% anchor support
a random draw of 76 latents lands only ~3 that are even active at the
anchor. The rest are exactly zero there, and no fitted amplitude can
rescue a zero (alpha * 0 = 0). So "null dead 0/18" is partly a statement
about the dictionary, not only about our selection — the mirror image of
why the ReLU/L1 null was vacuous. Same confound, opposite sign.

## The test

Draw the null ONLY from latents that fire at the anchor, matched to each
of our members' anchor-firing COUNT (nearest-count sampling, our own
members excluded). This controls for support directly and means the same
thing on any architecture: TopK, JumpReLU or ReLU/L1.

## Result

| arm | n (med) | ampF0 med | ampF0 max | sup med | cf med |
|---|---|---|---|---|---|
| tri-amp (ours) | 76 | 0.847 | 1.425 | 1.00 | 0.872 |
| gate-only (ours) | 126 | 0.917 | 0.957 | 1.00 | 0.634 |
| circuit-tracer (matched) | 76 | 0.000 | 0.000 | 1.00 | 0.118 |
| plain null | 76 | 0.000 | 0.026 | 0.00 | 0.000 |
| **support-matched null** | 75 | **0.000** | **0.054** | **0.22** | **0.038** |

Per seed, ours versus the best of 3 matched draws:

| seed | ours ampF0 | best null | match: ours vs null firings/anchor | distinct draws |
|---|---|---|---|---|
| L4 70641 | 1.425 | 0.000 | 17.11 vs 15.45 | 3/3 |
| L4 100791 | 0.679 | 0.000 | 39.21 vs 48.00 | **1/3** |
| L4 111451 | 0.261 | 0.054 | 39.47 vs 48.00 | 3/3 |
| L6 18986 | 0.940 | 0.000 | 20.02 vs 17.95 | 3/3 |
| L6 35569 | 0.989 | 0.000 | 41.48 vs 48.00 | 3/3 |
| L6 101282 | 0.753 | 0.000 | 38.40 vs 48.00 | 3/3 |

**Ours beats the support-matched null on 6/6 seeds.** Across all draws
faithfulness peaks at 0.054 and drive at 0.207, against our 0.847 median
faithfulness and 0.872 median drive.

The confound is real but does not explain the result: `sup` rises from
0.00 (plain null) to 0.22 (matched), so anchor-active latents do carry
some necessity purely by being active — exactly what needed controlling.
Faithfulness and drive do not follow.

## Honest limits

* **Matching is imperfect on 4 of 6 seeds**, and in the conservative
  direction: the null drew latents firing at 48/48 anchors against our
  members' 38-41. The null therefore had MORE anchor activity than our
  circuit and still lost. The two seeds where matching was tight
  (17.11 vs 15.45, 20.02 vs 17.95) both give 0.000.
* **16 independent draws, not 18.** On L4/100791 the matched candidate
  pool is exhausted at n=70, so all three draws return the same set.
  Quote 16.
* This tests whether OUR result is a support artifact. It does NOT
  establish that the method works on ReLU/L1 or JumpReLU banks — it only
  removes the reason to think our TopK numbers are inflated by sparsity.
  A JumpReLU bank (GemmaScope) remains the right next dictionary, since
  it sits between the two regimes.

## Reproduce

```
SUPPORT_NULL=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. python ours_llama.py run
```

`expandable_segments` matters: without it this arm OOMs under the
MEM_FRAC cap with 12.2 GB allocated but 2.05 GB reserved-and-unallocated
— fragmentation, not real demand.
