# DAN-74 golden regression set (2026-10-09 19:24)

32 targets (selection.json, seed 74); the 15k run = /mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2/experiments/062-h100-protocol-v1/out_full/out (pinned commit 538cbed). Local refits: r0, r1.

## Memberships and α (046 compare_stores semantics)

| target | n H100 | n r0 / r1 | J(r0,r1) | dα(r0,r1) | J(r0,H100) | dα(r0,H100) | J(r1,H100) | dα(r1,H100) | status |
|---|---|---|---|---|---|---|---|---|---|
| 0.mlp.7357 | 24 | 25 / 25 | 1.0000 | 0 | 0.9600 | 0.06 | 0.9600 | 0.06 | DIFFERENT |
| 0.resid.23442 | 125 | 138 / 138 | 1.0000 | 0 | 0.8652 | 0.34 | 0.8652 | 0.34 | DIFFERENT |
| 1.attn.18303 | 111 | 109 / 109 | 1.0000 | 0 | 0.8487 | 0.84 | 0.8487 | 0.84 | DIFFERENT |
| 1.mlp.9498 | 160 | 176 / 176 | 1.0000 | 0 | 0.7592 | 1.4 | 0.7592 | 1.4 | DIFFERENT |
| 2.attn.6976 | 160 | 162 / 162 | 1.0000 | 0 | 0.8613 | 1.6 | 0.8613 | 1.6 | DIFFERENT |
| 2.mlp.18100 | 276 | 284 / 284 | 1.0000 | 0 | 0.8006 | 1.4 | 0.8006 | 1.4 | DIFFERENT |
| 2.resid.2330 | 274 | 233 / 233 | 1.0000 | 0 | 0.6095 | 1.9 | 0.6095 | 1.9 | DIFFERENT |
| 3.attn.18125 | 310 | 321 / 321 | 1.0000 | 0 | 0.6605 | 0.98 | 0.6605 | 0.98 | DIFFERENT |
| 3.mlp.27278 | 291 | 277 / 277 | 1.0000 | 0 | 0.6706 | 1.1 | 0.6706 | 1.1 | DIFFERENT |
| 3.resid.654 | 379 | 376 / 376 | 1.0000 | 0 | 0.7807 | 2.2 | 0.7807 | 2.2 | DIFFERENT |
| 4.attn.32494 | 336 | 347 / 347 | 1.0000 | 0 | 0.5523 | 2.7 | 0.5523 | 2.7 | DIFFERENT |
| 4.mlp.32078 | 442 | 462 / 462 | 1.0000 | 0 | 0.5613 | 1.7 | 0.5613 | 1.7 | DIFFERENT |
| 4.resid.21395 | 430 | 450 / 450 | 1.0000 | 0 | 0.6088 | 2.9 | 0.6088 | 2.9 | DIFFERENT |
| 5.mlp.27573 | 598 | 614 / 614 | 1.0000 | 0 | 0.5679 | 1.7 | 0.5679 | 1.7 | DIFFERENT |
| 5.resid.18594 | 404 | 389 / 389 | 1.0000 | 0 | 0.7127 | 1.8 | 0.7127 | 1.8 | DIFFERENT |
| 6.attn.2557 | 489 | 501 / 501 | 1.0000 | 0 | 0.5091 | 1.5 | 0.5091 | 1.5 | DIFFERENT |
| 6.mlp.20793 | 702 | 726 / 726 | 1.0000 | 0 | 0.5955 | 2.9 | 0.5955 | 2.9 | DIFFERENT |
| 6.resid.30675 | 476 | 459 / 459 | 1.0000 | 0 | 0.7642 | 2.5 | 0.7642 | 2.5 | DIFFERENT |
| 7.attn.10250 | 715 | 677 / 677 | 1.0000 | 0 | 0.5049 | 2.2 | 0.5049 | 2.2 | DIFFERENT |
| 7.mlp.38546 | 590 | 568 / 568 | 1.0000 | 0 | 0.4122 | 3 | 0.4122 | 3 | DIFFERENT |
| 7.resid.31336 | 385 | 374 / 374 | 1.0000 | 0 | 0.5458 | 1.8 | 0.5458 | 1.8 | DIFFERENT |
| 8.attn.11721 | 468 | 445 / 445 | 1.0000 | 0 | 0.4894 | 1.8 | 0.4894 | 1.8 | DIFFERENT |
| 8.resid.15479 | 997 | 1100 / 1100 | 1.0000 | 0 | 0.6643 | 2.6 | 0.6643 | 2.6 | DIFFERENT |
| 9.mlp.7789 | 760 | 765 / 765 | 1.0000 | 0 | 0.5545 | 4.4 | 0.5545 | 4.4 | DIFFERENT |
| 9.resid.1841 | 850 | 779 / 779 | 1.0000 | 0 | 0.5924 | 1.8 | 0.5924 | 1.8 | DIFFERENT |
| 10.attn.33695 | 666 | 584 / 584 | 1.0000 | 0 | 0.5024 | 2.2 | 0.5024 | 2.2 | DIFFERENT |
| 10.mlp.1713 | 723 | 803 / 803 | 1.0000 | 0 | 0.4830 | 2.3 | 0.4830 | 2.3 | DIFFERENT |
| 10.resid.34222 | 944 | 790 / 790 | 1.0000 | 0 | 0.6100 | 2.5 | 0.6100 | 2.5 | DIFFERENT |
| 11.attn.2940 | 1179 | 906 / 906 | 1.0000 | 0 | 0.3180 | 1.1 | 0.3180 | 1.1 | DIFFERENT |
| 11.mlp.20886 | 699 | 622 / 622 | 1.0000 | 0 | 0.3949 | 2.1 | 0.3949 | 2.1 | DIFFERENT |
| 11.resid.20555 | 969 | 1458 / 1458 | 1.0000 | 0 | 0.4127 | 3.5 | 0.4127 | 3.5 | DIFFERENT |
| 1.resid.210 | 123 | 116 / 116 | 1.0000 | 0 | 0.7970 | 0.86 | 0.7970 | 0.86 | DIFFERENT |

Status counts: {'DIFFERENT': 32}. Local noise floor J(r0, r1): min 1.0000, median 1.0000; dα(r0, r1) max 0. **Membership gate: FAIL**

## Eval rows (049 v2 scorer, EVAL_SPLIT=as_run)

| comparison | held | rows | identical rows | fields | exact | <=1e-6 | <=1e-3 | >1e-3 | None/type | missing | max abs diff (at) | pass flips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h100a vs h100b (same circuit, same GPU: must be identical) | strong | 32 | 32 | 1728 | 1728 | 0 | 0 | 0 | 0 | 0 | 0 (-) | 0 |
| h100a vs h100b (same circuit, same GPU: must be identical) | mid | 32 | 32 | 1728 | 1728 | 0 | 0 | 0 | 0 | 0 | 0 (-) | 0 |
| h100a vs stored H100 row (scorer, RTX vs H100) | strong | 32 | 0 | 1728 | 934 | 0 | 120 | 674 | 0 | 0 | 7.58 (6.resid.30675 phi_pin_alpha_zero_raw_tk) | 1 |
| h100a vs stored H100 row (scorer, RTX vs H100) | mid | 32 | 0 | 1728 | 917 | 0 | 100 | 711 | 0 | 0 | 5.98 (7.resid.31336 freeM_dense_tk) | 0 |
| r0 vs r1 (fit noise floor) | strong | 32 | 32 | 1728 | 1728 | 0 | 0 | 0 | 0 | 0 | 0 (-) | 0 |
| r0 vs r1 (fit noise floor) | mid | 32 | 32 | 1728 | 1728 | 0 | 0 | 0 | 0 | 0 | 0 (-) | 0 |
| r0 vs stored H100 row (HEAD refit vs the run) | strong | 32 | 0 | 1728 | 821 | 0 | 20 | 887 | 0 | 0 | 9.81 (9.resid.1841 freeM_topk_raw_tk) | 4 |
| r0 vs stored H100 row (HEAD refit vs the run) | mid | 32 | 0 | 1728 | 823 | 0 | 20 | 885 | 0 | 0 | 6.03 (9.mlp.7789 freeM_topk_tk) | 0 |
| r1 vs stored H100 row | strong | 32 | 0 | 1728 | 821 | 0 | 20 | 887 | 0 | 0 | 9.81 (9.resid.1841 freeM_topk_raw_tk) | 4 |
| r1 vs stored H100 row | mid | 32 | 0 | 1728 | 823 | 0 | 20 | 885 | 0 | 0 | 6.03 (9.mlp.7789 freeM_topk_tk) | 0 |

Scorer determinism (h100a vs h100b identical _tk rows): **PASS**