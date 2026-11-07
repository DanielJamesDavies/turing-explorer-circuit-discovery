# 014: CPU analyses of drivers and closure circuits (D4.1, D4.3, D4.4, D3.4)

An early campaign from the driver / closure period, before weighted circuit masking (WCM). It ran no model: it reads
the archives of earlier campaigns on the 11-seed driver panel and asks how the small "driver" sets relate to the much
larger "closure" circuit (the abl-mask membership of `018-maskrefine`).

## Question

- **D4.3 containment:** are the driver sets (direct-mass top-K from `012-driver-bakeoff`, and the head of the
  restoration ranking from `019-roles-drivers`) contained in the closure circuit?
- **D4.4 anatomy:** where do drivers sit relative to the seed (layer distance, same layer, site-kind mix)?
- **D3.4 signed roles:** abl-mask gives every member the same role. If restoration's attribution signs are stamped onto
  its members, what share are inhibitors?
- **D4.1 driver size vs depth:** does the smallest set that switches the seed on grow with the seed's depth?

## What the code does

`cpu_analyses.py` (CPU, no GPU, no rediscovery). For each panel seed it loads:
- the restoration ranking `019-roles-drivers/ranking_R_<comp>_<latent>.jsonl.gz` (first 200,000 rows);
- the abl-mask members `018-maskrefine/members_MF_<comp>_<latent>.jsonl.gz` (the closure);
- the direct-mass weights `012-driver-bakeoff/direct_full_<comp>_<latent>.pt`, ranked to a top 4,096 per site;
- the learned brake set `022-inhibmask/members_v2_lam0.001_<comp>_<latent>.jsonl.gz`, where it exists.

It then computes, at K in {16, 64, 256, 1024}:
- the share of each driver set inside the closure, against the closure's base rate over the dictionary;
- the layer distance, near-seed share, same-layer share and kind mix of the drivers;
- the inhibitor share of the restoration-labelled closure members, and the inhibitor share of learned brakes that fall
  inside the closure.

A seed with a missing archive is skipped.

`d41-driver-size-vs-depth.md` is a hand-written table built from the 012 rows, the 013 amplitude rows (including
`013-amplitude-drivers/rows_tinyk.jsonl`) and the 019 memberships. No script in this folder produces it.

## Files

| File | What |
|---|---|
| `cpu_analyses.py` | The D4.3 / D4.4 / D3.4 pass over the archives |
| `d43_containment.jsonl` | One row per seed × driver set × K: share inside the closure, base rate, enrichment (88 rows) |
| `d44_anatomy.jsonl` | One row per seed × driver set × K: layer distance, near-seed and same-layer shares, kind mix (44 rows) |
| `d34_signed_roles.jsonl` | One row per seed: closure size, labelled share, inhibitor share, brakes inside (11 rows) |
| `cpu_analyses_summary.md` | Tables generated from the three jsonl files, with per-seed detail |
| `README-cpu.md` | The campaign's own write-up of the three analyses, with its reading of each table |
| `d41-driver-size-vs-depth.md` | D4.1: minimal amplified and flat driver sizes, pinned-driver size and closure size by seed |

## Results

From `cpu_analyses_summary.md` (medians over the 11 seeds):
- **Containment.** Direct-mass drivers inside the closure: 100% at K = 16, 97% at K = 64, 92% at K = 256, 74% at
  K = 1024, against a base rate of 0.94% (enrichment 107×, 103×, 93×, 57×). The restoration head: 100%, 100%, 94%, 79%.
  Per seed at K = 64 the direct-mass share is 81–100%, lowest for the L9 attention seed.
- **Anatomy.** Median layer distance from the seed is 3.66 (direct-mass, K = 64) and 3.41 (restoration, K = 64). About
  a third of drivers are within 2 layers (0.36 and 0.38). Residual-stream latents are 54–75% of the mix. The restoration
  head overlaps the direct-mass top 1,000 at 0.97 at K = 64 and at 0.48 at K = 1024.
- **Signed roles.** Closure sizes run from 645 (L0 resid) to 33,935 (L8 mlp). 83–100% of members carry a restoration
  sign, and the median inhibitor share of labelled members is 35% (49% and 51% on the two attention seeds). Learned
  brakes inside the closure are 29–80% restoration-labelled inhibitors.

From `d41-driver-size-vs-depth.md`: on residual-stream seeds an amplified driver of 1–4 latents reaches
counterfactual faithfulness ≥ 0.8 at every depth, while the pinned-driver size grows from 77–231 (layers 0–2) to 18,296
(layer 11) and the archived membership from 8,731 to 880,522 latents. The L9 attention seed is never reached. The file lists its own caveats
(amplitude capped at 8.0, sufficiency only, one 48/16 split).

## Status

Historical. The driver / closure framing these analyses belong to was retired (see the archived driver campaigns in
`experiments/INDEX.md`), and the paper does not cite this folder. Its circuits come from the pre-WCM abl-mask and
attribution methods, so the numbers do not describe WCM circuits. The later role question at set level is
`054-inhibitor-roles`.
