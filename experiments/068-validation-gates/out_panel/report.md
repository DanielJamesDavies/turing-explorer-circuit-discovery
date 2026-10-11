# DAN-36: 033 Pythia panel through the engine (2026-10-09 16:42)

steps 400. Pass = ampF0 and ampFM in [0.8, 1.25] and sup > 0.9 (held-out 16). 033 = rows.jsonl (GPU, TF32); engine / standalone = this run, scored by 033's scorer (ref033.Seed.score).

| seed | 033 n / F0 / FM / sup / cf / pass | engine n / F0 / FM / sup / cf / pin0 / pass | standalone n / F0 / FM / sup / cf / pass | standalone, engine pricing n / F0 / FM / pass | engine fit s | nulls pass (max F0) |
|---|---|---|---|---|---|---|
| attn/1/31694 | 18 / 1.022 / 1.013 / 0.998 / -0.009 / PASS | 19 / 1.021 / 0.985 / 0.997 / -0.008 / 0.903 / PASS | 18 / 1.022 / 1.015 / 0.998 / -0.009 / PASS |  | 2042 | 0/1 (0.000) |
| attn/1/32511 | 35 / 0.782 / 0.742 / 0.996 / 0.315 / - | 59 / 0.693 / 0.747 / 0.997 / 0.296 / 0.344 / - | 35 / 0.759 / 0.758 / 0.996 / 0.317 / - | 61 / 0.702 / 0.759 / - | 2230 | 0/1 (0.000) |
| mlp/1/26871 | 23 / 0.729 / 0.943 / 0.995 / 2.483 / - | 30 / 0.679 / 0.827 / 1.000 / 2.519 / 1.071 / - |  |  | 1153 | 0/1 (0.000) |
| resid/1/6626 | 11 / 1.051 / 0.950 / 1.000 / 1.085 / PASS | 13 / 1.013 / 0.976 / 1.000 / 1.109 / 0.815 / PASS |  |  | 1621 | 0/1 (0.000) |

## Summary (4 seeds)

- pass: 033 2 / engine 2 / standalone 1 (of 2 with a standalone row)
- per-seed verdict agreement: engine vs 033 4/4; standalone vs 033 2/2; engine vs standalone 2/2
- size ratio n_engine / n_033: median 1.24 (min 1.06, max 1.69); n_engine / n_standalone median 1.37
- |ampF0 engine - 033| median 0.044 (max 0.090); |ampFM| median 0.027 (max 0.117)
- engine nulls: 0 / 4 pass, max ampF0 0.000 (033 on these seeds: 0 / 20 pass, max ampF0 0.000)
- membership Jaccard engine vs standalone (same machine, same contexts; expected to differ numerically): attn/1/31694 0.85, attn/1/32511 0.59