# D4.3 / D4.4 / D3.4 — CPU analyses (2026-08-02)

## D4.3 — Containment: are drivers inside the closure?

| driver set | K | median in-closure | median base rate | median enrichment |
|---|---:|---:|---:|---:|
| C_direct | 16 | 100% | 0.94% | 107x |
| C_direct | 64 | 97% | 0.94% | 103x |
| C_direct | 256 | 92% | 0.94% | 93x |
| C_direct | 1024 | 74% | 0.94% | 57x |
| R_head | 16 | 100% | 0.94% | 107x |
| R_head | 64 | 100% | 0.94% | 107x |
| R_head | 256 | 94% | 0.94% | 100x |
| R_head | 1024 | 79% | 0.94% | 62x |

Per-seed at K=64 (C_direct):

| seed | L/kind | in closure | enrichment |
|---|---|---:|---:|
| 2/19766 | L0 resid | 92% | 117x |
| 8/20333 | L2 resid | 95% | 165x |
| 9/38734 | L3 attn | 98% | 59x |
| 13/30053 | L4 mlp | 97% | 121x |
| 17/38268 | L5 resid | 97% | 162x |
| 20/35678 | L6 resid | 98% | 111x |
| 25/10628 | L8 mlp | 100% | 30x |
| 26/17432 | L8 resid | 100% | 73x |
| 27/6859 | L9 attn | 81% | 55x |
| 29/2753 | L9 resid | 97% | 52x |
| 35/6599 | L11 resid | 97% | 103x |

## D4.4 — Driver anatomy

| driver | K | median dist | median near<=2 | same-layer | direct-top1k | kind mix (a/m/r) |
|---|---:|---:|---:|---:|---:|---|
| C_direct | 64 | 3.66 | 0.36 | 0.02 | 1.00 | 0.16/0.19/0.66 |
| C_direct | 1024 | 3.89 | 0.28 | 0.03 | 1.00 | 0.22/0.23/0.56 |
| R_head | 64 | 3.41 | 0.38 | 0.00 | 0.97 | 0.08/0.17/0.75 |
| R_head | 1024 | 3.41 | 0.38 | 0.03 | 0.48 | 0.19/0.27/0.54 |

## D3.4 — Signed-role pairing for the closure mask

| seed | L/kind | closure n | labelled by R | inhibitor share | learned brakes inside | of those, R-labelled inhib |
|---|---|---:|---:|---:|---:|---:|
| 2/19766 | L0 resid | 645 | 94% | 12% | 10 | 80% |
| 8/20333 | L2 resid | 1894 | 98% | 21% | 15 | 47% |
| 9/38734 | L3 attn | 6126 | 100% | 49% | 37 | 65% |
| 13/30053 | L4 mlp | 4260 | 98% | 38% | 28 | 29% |
| 17/38268 | L5 resid | 4173 | 96% | 30% | None | - |
| 20/35678 | L6 resid | 7264 | 95% | 26% | 20 | 40% |
| 25/10628 | L8 mlp | 33935 | 88% | 36% | None | - |
| 26/17432 | L8 resid | 14598 | 90% | 34% | 54 | 44% |
| 27/6859 | L9 attn | 16208 | 92% | 51% | 32 | 47% |
| 29/2753 | L9 resid | 21986 | 83% | 35% | None | - |
| 35/6599 | L11 resid | 13428 | 86% | 39% | 28 | 57% |

median inhibitor share of labelled closure members: 35%
