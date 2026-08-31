# D4.3 + D4.4 + D3.4 — CPU analyses over the archives (2026-08-02)

No GPU, no rediscovery: reads the 24-seed R rankings (D2.2), abl-mask
member lists (D3.6), direct-mass weights (D1) and the learned brake sets
(D3.8). Script cpu_analyses.py; rows d43_*.jsonl / d44_*.jsonl /
d34_*.jsonl; tables cpu_analyses_summary.md.

## D4.3 — CONTAINMENT: drivers ARE inside the closure

| driver set | K | median inside closure | base rate | enrichment |
|---|---:|---:|---:|---:|
| direct-mass | 16 | **100%** | 0.94% | 107x |
| direct-mass | 64 | **97%** | 0.94% | 103x |
| direct-mass | 256 | 92% | 0.94% | 93x |
| direct-mass | 1024 | 74% | 0.94% | 57x |
| restoration head | 16 | **100%** | 0.94% | 107x |
| restoration head | 64 | **100%** | 0.94% | 107x |
| restoration head | 1024 | 79% | 0.94% | 62x |

Per-seed at K=64: 81-100% (weakest L9-attn 81%, the identity-failure
seed). **The (P1) driver set is a near-perfect SUBSET of the (P2)
closure membership at driver budgets**, decaying gracefully to ~75-79%
by K=1024 where the driver notion itself is stretched. This is the
containment leg of the two-objects decomposition, measured.

## D4.4 — ANATOMY: drivers are NOT near-seed

| driver | K | mean layer distance | frac within 2 layers | same layer | in direct-top1k | kind mix a/m/r |
|---|---:|---:|---:|---:|---:|---|
| direct-mass | 64 | 3.66 | 0.36 | 0.02 | 1.00 | 0.16/0.19/0.66 |
| direct-mass | 1024 | 3.89 | 0.28 | 0.03 | 1.00 | 0.22/0.23/0.56 |
| restoration | 64 | 3.41 | 0.38 | 0.00 | 0.97 | 0.08/0.17/0.75 |
| restoration | 1024 | 3.41 | 0.38 | 0.03 | 0.48 | 0.19/0.27/0.54 |

Drivers sit a median 3.4-3.9 layers BELOW the seed and only ~1/3 fall
within 2 layers — correcting the intuition (from D0.1's "pinned-drivers
are near-seed") that direct-mass means adjacent. Direct mass is
computed unmediated, but the latents carrying it are distributed in
depth. RESID dominates the mix (54-75%) at every budget. Restoration's
head agrees with direct-mass at K=64 (0.97 overlap with direct-top1k)
but DIVERGES by K=1024 (0.48) — the two notions coincide only at the
head, which is exactly D0.1's split reproduced at panel scale.

## D3.4 — SIGNED-ROLE PAIRING: the closure circuit is ~1/3 inhibitors

abl-mask is role-blind (all members delivered as supports). Stamping
restoration's attribution signs onto its membership:

| seed | closure n | labelled by R | INHIBITOR share |
|---|---:|---:|---:|
| L0 resid | 645 | 94% | 12% |
| L2 resid | 1,894 | 98% | 21% |
| L3 attn | 6,126 | 100% | **49%** |
| L4 mlp | 4,260 | 98% | 38% |
| L5 resid | 4,173 | 96% | 30% |
| L6 resid | 7,264 | 95% | 26% |
| L8 mlp | 33,935 | 88% | 36% |
| L8 resid | 14,598 | 90% | 34% |
| L9 attn | 16,208 | 92% | **51%** |
| L9 resid | 21,986 | 83% | 35% |
| L11 resid | 13,428 | 86% | 39% |
| **median** | | **92%** | **35%** |

83-100% of closure members carry an attribution sign, and a MEDIAN 35%
are inhibitors — rising to ~50% on the two attn seeds. So the closure
circuit is roughly one-third opposition by attribution sign, which
independently corroborates the balance-set reading from the brake work
(D3.8) using a completely different label source. The learned brakes
that sit inside the closure are only 29-80% (median ~47%) R-labelled
inhibitors, re-confirming that the two role definitions are related but
not identical.
