# Latent pilot: does the geometric lr schedule change production latent WCM? (2026-11-02)

Daniel, 2026-11-02: "Please can you do the small local pilot to check it". Background: arena C found that the mask
engine's constant lr under the gate anneal (m = sigmoid(θ / T), T 1 → 0.05) makes the gate step lr / T grow 20× over
training. Behavioural fits blew up late in training, and a geometric schedule (lr × 0.05^progress, `lr_schedule=
"geometric"` in learned_mask, default off) fixed it (`../arena_c/WCM_ADAPTER.md` §8). Production latent WCM (062
driver, `WCM_VERSION=v3`) uses the same constant-lr anneal. **Nothing in production changes here**; this is a pilot.

## Design (fixed before any pilot result)

- **Targets** (`targets.txt`, 9 from 102's frozen check panel, the 15k run's records, explicit split):
  - the 6 failing check targets (v3 = 102's arm `r1.0_rk1e-1`, non-vacuous) whose zero term ended highest
    (most late-degraded, the hypothesis' best case): 11.resid.36128, 9.attn.22192, 8.attn.38035, 3.attn.17813,
    3.attn.29509, 6.attn.6889;
  - 3 passing check targets, one per depth band, `random.Random(20261102)`: 0.mlp.9455, 7.attn.17162, 10.mlp.37647.
  - **The panel is enriched for failures**: pass rates here are not population rates.
- **Arms** (`run_pilot.sh`; 062 driver, MODE=main, v3, 400 steps, full curve, SKIP_SPEC=1):
  - `geo_l1e-3`: `LR_SCHEDULE=geometric`, production λ 1e-3;
  - `geo_l3e-3`: geometric, λ 3e-3. Geometric's lr integral is ≈ 0.32× constant's, so at the same λ it prunes less;
    3× λ roughly restores the sparsity budget (sum(lr) × λ);
  - control: 102's check fits of `r1.0_rk1e-1`, which 104 part A showed bit-identical to native v3 (members, α,
    every curve, every score). A driver rerun (`const_l1e-3`) is last in the queue and was expected to be cut by the
    15:45 deadline.
- **Scoring**: the driver's production score, held-out strongest; pass = Z / A / C in [0.75, 1.5] and necessity
  (phi_sup_blind_tk) ≥ 0.9, not vacuous (102's rule). Resample (donor) scores are not run here.
- **Driver change** (default off): `LR_SCHEDULE` env in `062-h100-protocol-v1/driver.py` sets the learned mask's
  schedule; an OUT fitted under it is marked `OUT/LR_SCHEDULE` and refuses another schedule's circuits.
  `tests/test_wcm_version_driver.py`: 10 / 10 pass unset (production); with `LR_SCHEDULE=geometric` exported,
  `test_check_out_version` fails because the override writes its marker into the test's OUT (expected).

`compare.py` prints the table below.

## Results (geo arms 14:49-15:28; `compare_out.md`)

| target | production (102 v3, constant) | geometric λ 1e-3 | geometric λ 3e-3 |
|---|---|---|---|
| 11.resid.36128 | fail, 1,425, Z 0.75 / A 0.92 / C 0.92; zero term 0.011 → **2.08** → 0.24 | PASS, 3,537 | **PASS, 1,884**, 0.84 / 1.00 / 1.02 |
| 9.attn.22192 | fail, 915, 0.73 / 0.88 / 0.84; 0.000 → **0.50** → 0.105 | PASS, 3,362 | **PASS, 1,904**, 0.93 / 0.96 / 0.98 |
| 8.attn.38035 | fail, 830, Z 0.00 | fail, 6,733, Z 0.00 | fail, 1,937, Z 0.00 |
| 3.attn.17813 | fail, 597, Z 0.00 | PASS, 1,531, 0.81 / 0.75 / 0.75 | fail, 618, Z 0.35 |
| 3.attn.29509 | fail, 775, Z 0.09 | fail, 2,692, Z 0.00 | fail, 919, Z 0.11 |
| 6.attn.6889 | fail, 697, C 0.72 | PASS, 2,782, 0.78 / 0.95 / 0.77 | fail, 916, Z 0.56 |
| 0.mlp.9455 | PASS, 57 | PASS, 71 | PASS, 35 |
| 7.attn.17162 | PASS, 859 | PASS, 2,923 | PASS, 1,081 |
| 10.mlp.37647 | PASS, 982 | PASS, 2,297 | PASS, 1,405 |
| **passes** | **3 / 9** | **7 / 9** (circuits 1.2-8× larger, most 2-4×) | **5 / 9** (median size ≈ 1.3×) |

Necessity is 0.95-1.00 everywhere. Reading:

- **At roughly matched size (λ 3e-3) geometric rescues exactly the two targets whose production fit blew up late**
  (zero term back up to 2.08 and 0.50 after its minimum), and loses none of the 3 passes.
- The other four failures are attention targets with Z ≈ 0 under every schedule: the known attention failure
  (depth-hypothesis 131: source-position content), not this instability. At λ 1e-3 geometric "rescues" two of them
  only by keeping 2-4× more latents.
- So the instability is present in production latent fits, but it flips borderline targets rather than most
  failures. 9 targets enriched for failures cannot give the population effect.

**Proposed next (Daniel's decision; nothing in production changed):** the full 72-target check panel under geometric
λ 3e-3 (~3.5 h local, overnight), against 102's production numbers (54 / 72 production pass), before choosing the
pod's schedule.

## Part 2: why geometric circuits are larger, and the fix (16:33-16:51)

Daniel: the circuits are larger, passing ones too? Yes: at λ 1e-3 the passing targets grew 1.2-3.4× as well. Cause:
the lr budget. A gate starts at θ_init 4 and is pruned once θ crosses 0; Adam moves θ ~lr per step, so the distance a
gate can travel is sum(lr): constant 0.05 × 400 = 20, geometric 0.05 × 400 × 0.317 ≈ 6.3. Latents with a weak or
inconsistent push no longer finish the walk. (learned_mask's own lr-schedule comment records the same: decay gives
bigger circuits, because late lr does the pruning; it is also what blows the fits up.)

**Fix: geometric with the peak lr × 3.15** (`run_pilot2.sh`, driver `LR_PEAK_MULT`, default off, recorded in the OUT
marker). sum(lr) = 20 again, so the sparsity and weight-decay budgets match constant's; lr / T is fixed at 3.15×
constant's starting value (constant ends at 20×).

| target | production | geometric × 3.15, λ 1e-3 | size ratio |
|---|---|---|---|
| 11.resid.36128 | fail, 1,425 | **PASS, 1,337**, 0.91 / 1.01 / 0.95 | 0.94 |
| 9.attn.22192 | fail, 915 | **PASS, 910**, 0.92 / 0.94 / 0.96 | 0.99 |
| 8.attn.38035 | fail, 830 | fail, 1,150, Z 0.00 | 1.39 |
| 3.attn.17813 | fail, 597 | fail, 675, 0.55 / 0.97 / 0.96 | 1.13 |
| 3.attn.29509 | fail, 775 | fail, 804, 0.00 / 0.92 / 0.96 | 1.04 |
| 6.attn.6889 | fail, 697 | fail, 793, 0.69 / 0.80 / 0.91 | 1.14 |
| 0.mlp.9455 | PASS, 57 | PASS, 50 | 0.88 |
| 7.attn.17162 | PASS, 859 | PASS, 1,003 | 1.17 |
| 10.mlp.37647 | PASS, 982 | PASS, 956 | 0.97 |
| **passes** | **3 / 9** | **5 / 9**, median size ratio ≈ 1.04 | |

The size cost is gone; the two late-blow-up targets pass at production size; no pass is lost; the attention failures
improve on A / C but still fail Z (the separate attention problem). Caveat: the higher starting lr / T brings early
spikes back on some fits (3.attn.29509: 5.1 mid-run, recovered); a short lr warm-up would address it.

**Proposed (Daniel's decision):** the 72-target check panel under geometric × 3.15, λ 1e-3 (~3.5 h local), against
production's 54 / 72, before choosing the pod's schedule; optionally a warm-up arm.

**Control (`const_l1e-3`, the driver rerun of production, 15:27-15:46): identical to 102 on all 9 targets** (same
sizes, same Z / A / C / necessity to the printed precision, 3 / 9 pass), so the geometric arms differ from production
only in the schedule. Its every-step curves show what 102's stride-20 log hid: spikes in production fits are more
common than the stored curves suggest (worst zero term after its minimum, every step vs stride 20: 10.mlp.37647 0.64 vs
0.01, 6.attn.6889 0.41 vs 0.14, 3.attn.29509 0.34 vs 0.12), including on targets that pass.
