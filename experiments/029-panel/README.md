# Depth-stratified tri-amp panel (2026-08-09)

The run that takes the weighted-circuit claims from 3-4 in-sample seeds
at two layers to a stratified, held-out-scored panel. 22 seeds:
L2/L5/L7/L8/L9/L11 resid, L3/L8/L11 mlp, L5 attn. Arms: triamp400
(triple w=0.10 shallow / 0.05 deep, lambda 1e-3), triamp100 (100 steps
at 4x/2x lambda), gate400 (gates only). 124 amplitude-fitted random
nulls (10 draws at L2/L9, 4 elsewhere). HELD-OUT protocol: 48 train /
16 held-out contexts; membership, alphas, floors, pins all from train;
every reported metric read on held-out. Data: rows.jsonl (runner.py).

## Headline results

1. **The weighted-circuit claim survives the panel.** triamp400 passes
   ALL-PASS held-out (both floors in [0.8,1.25] + sup>0.9) on **17/21**
   scorable seeds (one FMd-vacuous seed excluded), at median n=224
   (shallow) / 471 (deep). Median ampF0 0.99 at BOTH depth bands.
2. **Held-out generalisation is essentially free for reconstruction:**
   ampF0 held-out minus in-sample: median -0.000, worst -0.118. The
   within-distribution caveat resolves favourably.
3. **Gate-only collapses under the same exam: 3/21.** Deep medians
   F0 0.46 / FMd 0.09 at 2-3x the node count. The amplitude advantage
   generalises across every layer and host kind tested.
4. **The null is immaculate at scale: 0/124 draws pass anything.**
   ampF0 0.00-48,665 (median ~21); ampFMd never in band jointly; and
   the drive null's July heavy tail VANISHED on held-out probes — max
   cf_amp over 124 draws is 0.042. The 0.895 July outlier was an
   in-sample artifact. Weighted-circuit faithfulness and drive are
   both properties of the selected membership.
5. **Budget-drive replication:** deep cf_amp overshoots at 400 steps
   (median 1.48) and calibrates at 100 steps (1.10 deep, 1.01
   shallow). Seed 1283's closure-without-drive reproduced exactly
   (0.17 at 400 -> 0.83 at 100 steps, n=774). triamp100's cost is
   size (median n 458/1,494) and a lower deep FMd pass rate (12/21
   overall).

## New findings the panel surfaced

- **Host kind is a real axis for DRIVE, not for reconstruction.**
  All kinds reconstruct (F0 0.86-1.11 across the panel), but: resid
  drives broadly (cf 0.86-1.7, deep overshoot); **mlp drives weakly**
  (best 0.88; two seeds negative); **attn is bimodal** (one seed
  drives at bare clean values, one cannot be driven at all —
  injection suppresses, cf -0.37). Echoes the July L9-attn wall.
- **L8 (never run before) behaves like its neighbours** for
  reconstruction; its first seed posts the panel's largest drive
  overshoot (cf_amp 3.44 at bare 1.19 — already-driving seed).
- **ampFMd needs a vacuity guard like free0's**: L11 resid 1829's
  eMd ~= a_pos_ho blew its denominator (FMd 5-11 across arms).
  Guard: exclude FMd outside (-1, 3); one seed affected.

## Reporting rules for the paper

Quote triamp400 as the compact arm (17/21, n~224/471 med), triamp100
as the drive arm; medians per band; the vacuous seed excluded with the
guard stated; all numbers HELD-OUT. The July in-sample numbers remain
quotable as calibration history only.
