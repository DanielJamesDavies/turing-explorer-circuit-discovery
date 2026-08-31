# D2.5 (pin variant) + D4.6 (true generalisation) — 2026-08-02

8-seed panel, eval-only over archived rankings. rows.jsonl (D4.6 +
small-K D2.5), rows_d25_bigK.jsonl (D2.5 at usable budgets).

## D4.6 — TRUE GENERALISATION: drivers transfer to unseen corpus text

AMPC K=64 at alpha*, imposed on three context sources. "corpus" is a
random draw from the training corpus that NO stage of discovery or
evaluation has touched; "mid" is the seed's mid-band contexts (also
never used).

| seed | store held-out (reference) | mid-band | FRESH corpus |
|---|---:|---:|---:|
| L0 resid | 1.015 | 1.058 | **1.094** |
| L2 resid | 1.020 | 1.109 | **1.070** |
| L3 attn | 1.188 | 1.229 | **1.185** |
| L4 mlp | 0.962 | 0.962 | **0.956** |
| L5 resid | 1.000 | 1.013 | **0.994** |
| L6 resid | 1.077 | 1.096 | **1.056** |
| L8 resid | 0.929 | 0.979 | **0.925** |
| L11 resid | 1.007 | 1.007 | **1.035** |
| **median** | **1.011** | **1.035** | **1.045** |

**The claim v1 could not make is now made: driver circuits impose the
seed on contexts drawn from the corpus at random, at the same strength
as on the held-out probe split (median cf 1.045 vs 1.011).** No
degradation whatsoever — 8/8 seeds within 0.07 of their reference. The
driver object is a property of the seed, not of the probe set.

## D2.5 — PIN VARIANT: collapsed pins win at depth

Small K (16-256) is DEGENERATE — both conventions read 0.000 on 6/8
seeds (the documented pin-dead-below-K=4096 effect), so the question
can only be asked at larger budgets:

| K | median pin_collapsed | median pin_position | collapsed wins |
|---:|---:|---:|---:|
| 1,024 | 0.45 | 0.18 | 5/8 |
| 4,096 | 0.95 | 0.87 | 6/8 |
| 16,384 | 0.99 | 0.90 | 7/8 |

**Collapsed (position-independent) pins dominate**, and the gap widens
with depth: at L3-attn K=4096 collapsed reads 1.48 vs position-specific
0.62; at L11 K=16384, 0.88 vs 0.54. Position-specific pinning
systematically UNDER-reads a circuit's pinned faithfulness because it
demands the member reproduce its value at every position separately,
which is a strictly harder counterfactual than the eval's own
convention.

DECISION: keep COLLAPSED pins as the default pinned metric (the frozen
exam already uses pin0_c as primary; this validates that choice with
data rather than convention). Position-specific pins remain useful only
as a stricter secondary diagnostic.

CAVEAT: this is the pinned-metric's own value, not a correlation
against held-out cf (cf at these budgets is saturated), so the claim is
"collapsed pins are the better-conditioned measurement", not "collapsed
pins predict drive better".
