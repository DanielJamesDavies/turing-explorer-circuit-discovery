# 8xH100 production driver: tri-amp + neg-amp at full candidate scale

Built 2026-08-31 at Daniel's request ("prepare for a full run ... on
tri-amp mask and then after with the neg amp mask variant").

## What was promoted into config (src/config.py, learned_mask section)

Previously engine-only kwargs, now first-class config fields with
behaviour-preserving defaults (all off):
`triple_floor_weight` (0.25), `free_amplitude` (False), `amp_l1` (0.0),
`signed_amplitude` (False), `neg_suppress_weight` (0.0),
`margin_topk` (None); `mask_floor_source` now accepts `"triple"`/`"pn"`.
Both discovery classes (ablation_gradient, counterfactual_gradient)
pass them through, so the production pipeline path can express tri-amp
without experiment-runner surgery.

## driver.py

The 029-panel per-seed protocol verbatim (triple floor, 48/16 split,
ampF0/ampFMd/cf_amp/cf_bare/sup, amplitude-fitted nulls) over EVERY
candidate in candidates.pt, plus the scale machinery:

- `SEED_SHARD=i/k` — enumeration index mod k; rows to
  `rows.shard<i>.jsonl`; one shard per GPU via `h100_launch.sh`
  (retry-per-shard, no `set -e`).
- `ARMS` registry: `triamp400` (production), `gate400` (set-only
  reference), `sgnamp400` (neg-amp SIGNED: alpha may go negative),
  `negsup400` (neg-amp SUPPRESS: `NEG_W`=0.5 negctx-read penalty).
  New arms must be added to ARM_SPECS — completeness accounting derives
  from it (harness trap #1).
- Nulls thinned: `N_NULL`=4 fitted draws only where
  `(comp_idx*131071+latent) % NULL_EVERY(=20) == 0`, deterministic
  across shards and resumes.
- CIRCUITS SAVED: every real-arm fit also dumps its membership +
  fitted amplitudes to `members.shard<i>.jsonl` (nulls excluded —
  random draws are not results). ~few GB worst case at 16k seeds.
- Every unfitted seed writes an `arm="skip"` row with a reason; the
  end-of-run report counts rows per arm and lists incomplete seeds,
  exiting nonzero if any (traps #2/#4). Resumable via the shard's own
  rows file.
- `triple_w`: 0.10 for layers <= 5, 0.05 deeper (panel calibration;
  L6 was never measured — it takes the deep value).

Smoke: `SMOKE=1 SEED_SHARD=0/2048 ARMS=<all four>` → 2 seeds, nulls
forced at 1 draw. See smoke.log / rows.shard0.jsonl.

## Decisions still open (Daniel)

1. WHICH neg-amp variant is "the" second pass: `sgnamp400` (signed
   amplitudes — the sign-census/register-discriminator one) or
   `negsup400` (negctx suppression). Both are wired; the launcher runs
   whichever `ARMS` names.
2. λ policy: fixed `LAM`=1e-3 (panel anchor) vs per-seed probe
   calibration (2x cost). Driver currently fixed-λ.
3. Null density: `NULL_EVERY`=20 (5% of seeds, 4 draws) — raise or
   lower with run budget.
4. Whether gate400 rides along (adds ~1/3 to fit cost, buys the
   set-vs-weighted comparison at scale).

## H100 checklist (from PROTOCOL.md, applies verbatim)

Artifacts on LOCAL NVMe (`RUN_ROOT` env); pinned requirements from
042-protocol-freeze; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
measure one shard before extrapolating wall-clock; never trust timings
taken while memory spills.
