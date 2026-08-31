# Comparison statistics

## gemma-tc

| method | matched-size f0 med [95%CI] | wins-losses vs tri-amp (|f0-1|) | nodes-to-band med | ntb / tri-amp size |
|---|---|---|---|---|
| ct-direct | 0.22 [0.07, 0.75] | 0-6 of 6 | 20000 (3/6 never) | 465.1x |
| ct-published | 0.01 [0.00, 0.10] | 0-6 of 6 | 6346 (3/6 never) | 147.6x |
| ct-rooted | 0.27 [0.05, 0.52] | 0-6 of 6 | 276 (5/6 never) | 13.1x |
| ct-rooted+amp | 0.87 [0.78, 1.27] | 3-3 of 6 | 43 (4/6 never) | 1.0x |
| null | 0.01 [0.00, 0.04] | 0-6 of 6 | never (6/6 never) | - |
| sfc | 0.14 [0.02, 0.67] | 0-6 of 6 | 20000 (1/6 never) | 465.1x |
| sfc+amp | 0.91 [0.54, 1.32] | 1-5 of 6 | 140 (4/6 never) | 1.0x |
| tri-amp | 1.14 [0.79, 1.22] | (ref) | 10 | 1.0x |
| tri-mask | 0.92 [0.58, 1.22] | 2-4 of 6 | 17 (3/6 never) | 1.7x |

## llama-tc

| method | matched-size f0 med [95%CI] | wins-losses vs tri-amp (|f0-1|) | nodes-to-band med | ntb / tri-amp size |
|---|---|---|---|---|
| coact | 0.00 [0.00, 0.00] | 0-6 of 6 | never (6/6 never) | - |
| coact+amp | 0.82 [0.71, 1.09] | 4-2 of 6 | 71 (4/6 never) | 1.0x |
| ct-direct | 0.00 [0.00, 0.00] | 0-6 of 6 | never (6/6 never) | - |
| ct-rooted | - [-, -] | 0-0 of 6 | never (6/6 never) | - |
| ct-rooted+amp | - [-, -] | 0-0 of 6 | never (6/6 never) | - |
| null | 0.00 [0.00, 0.00] | 0-6 of 6 | never (6/6 never) | - |
| sfc | 0.00 [0.00, 0.09] | 0-6 of 6 | never (6/6 never) | - |
| sfc+amp | 0.92 [0.90, 1.25] | 4-2 of 6 | 81 (2/6 never) | 1.0x |
| support-null | 0.00 [0.00, 0.05] | 0-6 of 6 | never (6/6 never) | - |
| tri-amp | 0.94 [0.68, 1.43] | (ref) | 71 (1/6 never) | 0.8x |
| tri-mask | 0.55 [0.12, 0.96] | 2-4 of 6 | 150 (2/6 never) | 2.2x |

## turingllm

| method | matched-size f0 med [95%CI] | wins-losses vs tri-amp (|f0-1|) | nodes-to-band med | ntb / tri-amp size |
|---|---|---|---|---|
| abl-gradient | 0.75 [0.19, 2.67] | 1-21 of 22 | 589152 (2/22 never) | 465.4x |
| cf-gradient | 1.10 [0.16, 5.35] | 0-22 of 22 | 589225 (2/22 never) | 465.4x |
| ge-hier | 0.81 [0.15, 7.46] | 0-13 of 22 | never (22/22 never) | - |
| null | 20.90 [0.16, 580.62] | 0-13 of 22 | never (22/22 never) | - |
| restoration | 5.23 [0.24, 56.24] | 1-21 of 22 | 558020 (2/22 never) | 436.5x |
| tri-amp | 0.96 [0.90, 1.02] | (ref) | 394 (4/22 never) | 0.4x |
| tri-mask | 0.66 [0.48, 0.86] | 3-18 of 22 | 370 (19/22 never) | 1.3x |
