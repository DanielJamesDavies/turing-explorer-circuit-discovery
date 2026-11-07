# Adaptive λ: easy targets small, hard targets the smallest size that still passes

Daniel: "take into consideration the strength of targets / topk position / loss curves, so the easiest
targets can reach small sizes with good evals and harder targets reach the smallest sizes they can with good evals";
then "build B and C and run them on some targets"; the criterion throughout: failing → passing may cost size, but
**targets that already pass must not grow**. Every fit here: 062 driver, MODE=main, `WCM_VERSION=v3`, 400 steps,
geometric lr × 3.15 (latent_pilot part 2), full curve; pass = 102's rule on held-out strongest.

## Options

- **A, λ ladder per target** (gold standard; several fits each). Offline on the 15k run's sweep, a ladder that selects
  on the training-context **Z read alone** passes only 0.38 (fixed λ 1e-3: 0.46): Z alone is not enough to select on.
- **C, predict λ from target features** (`c_predict.py`, `c_assign.py`). GBM on static features (layer, kind, clean
  strength, clean Top-K rank) learned on the 15k sweep (189 targets × 5 λ, protocol v1). Offline: pass 0.48 vs fixed
  0.46 at ~30% smaller passing circuits; the oracle (best λ per target) is 0.65. A 100-step probe adds ~3 points.
- **B, adapt λ during the fit** (learned_mask `lam_adapt_*`, default off). Final design (B2/B3): each step, each
  ablation term's training faithfulness f = mean(circuit) / mean(natural) on the batch (Z / A / C; EMA 0.9; R excluded:
  not in the production pass rule, and its error is high on every fit); log λ += rate × clip(d / ε, −1, 1), d = the
  smallest distance of any f to the band [1 − ε, 1 + 2ε] (> 0 inside); adapting from 25% to 70% of the run, then
  frozen; λ within λ₀ × [1/30, up]. The first design (B1: squared error, worst term including R) drove λ to its floor.
- **D, fit generously then prune**: not built.

## Dev: 24 of 093's dev targets (every third, sorted; `targets_dev24.txt`)

`compare_dev24.py`. Reference: fixed λ 1e-3 (18 / 24 pass).

| arm | passes | gained / lost | size on targets passing in both: median (p10-p90) |
|---|---|---|---|
| C (predicted λ, up and down) | 19 | +2 / −1 | 0.63 (0.39-**5.42**): its "lower λ" calls grew passing targets 3-7× |
| **C raise-only** (C's λ where > 1e-3, else fixed; composed from the runs) | 19 | **+1 / −0** | 0.81 (0.39-**1.00**) |
| B2 ε 0.15 | 16 | +1 / −3 | 0.53 (0.32-0.70) |
| B2 ε 0.10 | 18 | +1 / −1 | 0.76 (0.40-1.02) |
| B2 ε 0.15 + censored term | 18 | +2 / −2 | 0.71 (0.42-1.15) |
| **B3 ε 0.10, rise capped ×10** | **19** | **+1 / −0** | **0.72 (0.37-1.02)** |
| B3 + censored term | (20 / 24 at the stop) | +2 / −2 | 0.83 (0.37-1.11) |

## Check: 102's frozen 72-target panel (`compare_check72.py`, runs ended 07:57; 08:30 stop)

Reference for the λ policies: geometric × 3.15 at fixed λ 1e-3 (58 / 72; production v3: 54).

| arm | targets | passes (ref) | gained / lost | size on targets passing in both: median (p10-p90), max |
|---|---|---|---|---|
| geo × 3.15 + censored term (latent_pilot) | 72 | 63 (prod 54) | +10 / −1 vs production | 1.04 (0.77-1.50), 2.70 |
| B3 (ε 0.10, rise ≤ ×10) | 72 | 51 (58) | +1 / −8 | 0.64 (0.46-1.31), 1.49 |
| B3, first 36 only | 36 | 23 (26) | +0 / −3 | 0.60 (0.44-0.89), 1.16 |
| **B4 (ε 0.10, rise ≤ ×3), first 36** | 36 | **25 (26)** | **+0 / −1** | **0.67 (0.59-0.89)**, 1.16 |
| C raise-only (29 of its 33 raised fits; the 4 at λ 2e-3 not run) | 72 | 52 (58) | +0 / −6 | 1.00 (0.38-1.00), 1.00 |

**The dev results did not hold.** On 24 dev targets, B3 and C raise-only both lost nothing; on the check panel both
lose passes (B3 8 of 58, C raise-only 6). Every shrinking policy trades passes for size; the ×3 cap gives the best trade
measured (one pass lost on 36 targets for a third smaller passing circuits).

**Why passes are lost: necessity, and what training cannot see.** Of the losses inspected, the circuit still reproduces
the target (Z / A / C in band) but **necessity** fails once it is small (0.mlp.9455 at 24 latents: 0.77;
3.attn.10409 at 146: 0.81): the rest of the model then carries the behaviour too. Other losses are held-out effects the
training-batch signal misses (3.attn.18614 held-out A 2.06; 4.resid.21103 held-out Z 0.70 while the scorer's
training-context Z read was 0.81, below the band's 0.9 edge: the controller's pre-activation signal is optimistic).
No training term covers necessity, so the fit has no reason to stop shrinking where it breaks.

**Next (proposed):** a necessity term in training (ablate the members, keep the rest; the target must fall to its
empty-circuit value), so shrinking stops where necessity would fail; then B4 again. Until then: the censored term for
pass rate, and λ adaptation only with a tight cap (×3), reported as a size/pass trade.
