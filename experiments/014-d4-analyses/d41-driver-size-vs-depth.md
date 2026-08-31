# D4.1 — Driver size vs depth (2026-08-01)

Computed from D1 rows + D2.3 (+ tiny-K extension rows_tinyk.jsonl) +
D2.2's archived memberships. All on the frozen exam, held-out split.

## The law (the centrepiece contrast)

| seed | layer/kind | min K, amplified cf>=0.8 | min K, flat cf>=0.8 | K* (pinned>=0.8) | full membership | drive:membership |
|---|---|---:|---:|---:|---:|---|
| 2/19766 | L0 resid | 2 | 64 | 231 | 8,731 | 1:4,366 |
| 8/20333 | L2 resid | **1** | 256 | 77 | 138,059 | 1:138,059 |
| 9/38734 | L3 attn | 2 | 256 | 908 | 167,080 | 1:83,540 |
| 13/30053 | L4 mlp | 16 | 4096 | 4,071 | 269,031 | 1:16,814 |
| 17/38268 | L5 resid | **1** | 256 | 4,735 | 315,790 | 1:315,790 |
| 20/35678 | L6 resid | **1** | 256 | 9,178 | 529,799 | 1:529,799 |
| 26/17432 | L8 resid | 4 | 1024 | 7,190 | 567,873 | 1:141,968 |
| 25/10628 | L8 mlp | 64 | 1024 | 3,279 | 543,624 | 1:8,494 |
| 29/2753 | L9 resid | **1** | 64 | 6,321 | 758,023 | 1:758,023 |
| 27/6859 | L9 attn | never | never | ceiling | 546,427 | — |
| 35/6599 | L11 resid | **1** | 64 | 18,296 | 880,522 | 1:880,522 |

**Amplified drive does NOT scale with depth.** On resid seeds the minimal
driver is 1-4 latents at EVERY depth — L11's single latent at gain 5.0
imposes at cf 0.97 (and its 8-latent set needs only gain 1.4). Meanwhile
pinned-driver size grows 77 -> 18,296 and membership grows 8.7k -> 880k
over the same span. Kind is the real axis: mlp minimal drivers are
16-64, attn is bimodal (L3 at 2; L9-attn unreachable at any K/alpha —
the standing identity failure).

Headline sentence candidate: "At every depth we tested, a handful of
latents — often ONE — amplified a few-fold, suffices to switch a seed on
at natural strength; what grows with depth is not the drive but the
closure around it (roughly 10^2 -> 10^5 latents over the same seeds)."

## Caveats
- alpha ceiling 8.0: K=1 cells that fail may succeed at higher gain
  (censored, not refuted); the resid K=1 successes needed 5.0-7.8.
- Sufficiency only (cf-side); necessity/sup and uniqueness (D4.2) are
  separate. Single-latent drivers are unlikely to be unique.
- "Full membership" is the archived R ranking size (all admitted
  members), an upper-bound closure proxy, not a minimal n_eps.
- Single 48/16 split; no resampling error bars yet (D4.2).
